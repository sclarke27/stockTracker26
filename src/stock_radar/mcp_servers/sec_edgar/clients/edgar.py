"""SEC EDGAR API client for filings, insider transactions, and full-text search."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from io import StringIO

import httpx
from loguru import logger

from stock_radar.mcp_servers.sec_edgar.config import (
    DEFAULT_INSIDER_FILING_LIMIT,
    DEFAULT_MAX_FILING_TEXT_LENGTH,
    SEC_ARCHIVES_URL,
    SEC_EFTS_URL,
    SEC_FILING_INDEX_URL,
    SEC_SUBMISSIONS_URL,
    SEC_TICKERS_URL,
    SERVER_NAME,
)
from stock_radar.mcp_servers.sec_edgar.exceptions import (
    ApiError,
    CikNotFoundError,
    FilingNotFoundError,
)
from stock_radar.models.sec_edgar import (
    Filing,
    FilingSearchHit,
    FilingSearchResponse,
    FilingsResponse,
    FilingTextResponse,
    InsiderTransaction,
    InsiderTransactionsResponse,
)
from stock_radar.utils.rate_limiter import RateLimiter


class _HTMLTextExtractor(HTMLParser):
    """Simple HTML-to-text converter using stdlib HTMLParser."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []

    def handle_data(self, data: str) -> None:
        self._pieces.append(data)

    def get_text(self) -> str:
        return "".join(self._pieces)


class EdgarClient:
    """Async client for the SEC EDGAR free APIs.

    Wraps the Submissions API, filing archives, company tickers lookup,
    and the EFTS full-text search endpoint. All requests are rate-limited
    to comply with SEC's 10 req/sec policy.

    Args:
        http_client: Shared httpx async client.
        rate_limiter: Rate limiter enforcing SEC's request limits.
        user_agent: User-Agent header value (SEC requires contact info).
    """

    # Form types that represent insider transactions.
    _INSIDER_FORMS = {"3", "4", "5"}

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        rate_limiter: RateLimiter,
        user_agent: str,
    ) -> None:
        self._http = http_client
        self._limiter = rate_limiter
        self._user_agent = user_agent
        self._ticker_to_cik: dict[str, tuple[str, str]] | None = None

    async def resolve_cik(self, ticker: str) -> tuple[str, str]:
        """Resolve a ticker symbol to a CIK number and company name.

        Args:
            ticker: Stock ticker symbol (case-insensitive).

        Returns:
            Tuple of (cik_10digit, company_name).

        Raises:
            CikNotFoundError: If the ticker is not found.
        """
        if self._ticker_to_cik is None:
            await self._load_ticker_map()

        assert self._ticker_to_cik is not None
        key = ticker.upper()
        if key not in self._ticker_to_cik:
            raise CikNotFoundError(f"Ticker '{ticker}' not found in SEC database.")

        return self._ticker_to_cik[key]

    async def get_filings(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        limit: int = 50,
    ) -> FilingsResponse:
        """Get recent SEC filings for a company.

        Args:
            ticker: Stock ticker symbol.
            form_types: Filter by form types (e.g. ["8-K", "10-K"]).
            limit: Maximum number of filings to return.

        Returns:
            FilingsResponse with company info and filing list.
        """
        cik, company_name = await self.resolve_cik(ticker)
        submissions = await self._get_submissions(cik)

        recent = submissions["filings"]["recent"]
        filings: list[Filing] = []

        for i in range(len(recent["accessionNumber"])):
            form = recent["form"][i]
            if form_types and form not in form_types:
                continue

            filings.append(
                Filing(
                    accession_number=recent["accessionNumber"][i],
                    form_type=form,
                    filing_date=recent["filingDate"][i],
                    primary_document=recent["primaryDocument"][i],
                    description=recent["primaryDocDescription"][i],
                )
            )

            if len(filings) >= limit:
                break

        return FilingsResponse(
            ticker=ticker.upper(),
            cik=cik,
            company_name=company_name,
            filings=filings,
        )

    async def get_filing_text(
        self,
        ticker: str,
        accession_number: str,
        max_length: int = DEFAULT_MAX_FILING_TEXT_LENGTH,
    ) -> FilingTextResponse:
        """Get the full text content of a specific filing.

        Args:
            ticker: Stock ticker symbol.
            accession_number: SEC accession number (with dashes).
            max_length: Maximum characters to return.

        Returns:
            FilingTextResponse with filing text (HTML stripped).

        Raises:
            FilingNotFoundError: If the accession number is not found.
        """
        cik, _ = await self.resolve_cik(ticker)
        submissions = await self._get_submissions(cik)

        recent = submissions["filings"]["recent"]
        filing_idx = None
        for i, acc in enumerate(recent["accessionNumber"]):
            if acc == accession_number:
                filing_idx = i
                break

        if filing_idx is None:
            raise FilingNotFoundError(f"Filing {accession_number} not found for {ticker}.")

        form_type = recent["form"][filing_idx]
        filing_date = recent["filingDate"][filing_idx]
        primary_doc = recent["primaryDocument"][filing_idx]

        # Build archive URL (accession number without dashes in path).
        accession_no_dashes = accession_number.replace("-", "")
        url = SEC_ARCHIVES_URL.format(
            cik=cik.lstrip("0") or "0",
            accession=accession_no_dashes,
            document=primary_doc,
        )

        response = await self._request(url)
        raw_text = response.text

        # Strip HTML if content looks like HTML.
        if "<html" in raw_text.lower() or "<body" in raw_text.lower():
            content = self.strip_html(raw_text)
        else:
            content = raw_text

        truncated = len(content) > max_length
        if truncated:
            content = content[:max_length]

        return FilingTextResponse(
            ticker=ticker.upper(),
            accession_number=accession_number,
            form_type=form_type,
            filing_date=filing_date,
            content=content,
            truncated=truncated,
        )

    async def get_insider_transactions(
        self,
        ticker: str,
        limit: int = DEFAULT_INSIDER_FILING_LIMIT,
    ) -> InsiderTransactionsResponse:
        """Get insider transactions for a company from Form 3/4/5 filings.

        Args:
            ticker: Stock ticker symbol.
            limit: Maximum number of insider filings to parse.

        Returns:
            InsiderTransactionsResponse with aggregated transactions.
        """
        cik, company_name = await self.resolve_cik(ticker)
        submissions = await self._get_submissions(cik)

        recent = submissions["filings"]["recent"]
        transactions: list[InsiderTransaction] = []
        parsed_count = 0

        for i in range(len(recent["accessionNumber"])):
            if recent["form"][i] not in self._INSIDER_FORMS:
                continue
            if parsed_count >= limit:
                break

            accession_number = recent["accessionNumber"][i]
            primary_doc = recent["primaryDocument"][i]
            accession_no_dashes = accession_number.replace("-", "")
            stripped_cik = cik.lstrip("0") or "0"

            # Resolve XML document: use primaryDocument if XML, otherwise
            # fetch the filing index to find the XML source.
            xml_doc = primary_doc if primary_doc.lower().endswith(".xml") else None
            if xml_doc is None:
                xml_doc = await self._find_xml_document(
                    stripped_cik, accession_no_dashes, accession_number,
                )
            if xml_doc is None:
                continue

            url = SEC_ARCHIVES_URL.format(
                cik=stripped_cik,
                accession=accession_no_dashes,
                document=xml_doc,
            )

            try:
                response = await self._request(url)
                parsed = self.parse_form4_xml(response.text)
                transactions.extend(parsed)
            except (ApiError, ET.ParseError):
                logger.warning(
                    "Failed to parse insider filing",
                    accession=accession_number,
                    server=SERVER_NAME,
                )

            parsed_count += 1

        # Sort by transaction date, most recent first.
        transactions.sort(key=lambda t: t.transaction_date, reverse=True)

        return InsiderTransactionsResponse(
            ticker=ticker.upper(),
            cik=cik,
            company_name=company_name,
            transactions=transactions,
        )

    async def search_filings(
        self,
        query: str,
        form_types: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> FilingSearchResponse:
        """Search EDGAR filings using full-text search (EFTS).

        Args:
            query: Search query text.
            form_types: Filter by form types (e.g. ["8-K", "10-K"]).
            date_from: Start date (YYYY-MM-DD).
            date_to: End date (YYYY-MM-DD).

        Returns:
            FilingSearchResponse with search results.
        """
        body: dict = {"q": query}
        if form_types:
            body["forms"] = form_types
        if date_from or date_to:
            body["dateRange"] = "custom"
            if date_from:
                body["startdt"] = date_from
            if date_to:
                body["enddt"] = date_to

        response = await self._request(SEC_EFTS_URL, method="POST", json_body=body)
        data = response.json()

        hits_data = data.get("hits", {})
        total = hits_data.get("total", {}).get("value", 0)

        search_hits: list[FilingSearchHit] = []
        for hit in hits_data.get("hits", []):
            source = hit.get("_source", {})
            display_names = source.get("display_names", [""])
            company_name = display_names[0] if display_names else ""
            ciks = source.get("ciks", [""])
            cik = ciks[0] if ciks else ""

            search_hits.append(
                FilingSearchHit(
                    accession_number=source.get("adsh", ""),
                    form_type=source.get("root_form", ""),
                    filing_date=source.get("file_date", ""),
                    company_name=company_name,
                    cik=cik,
                    description=source.get("root_form", ""),
                )
            )

        return FilingSearchResponse(
            query=query,
            total_hits=total,
            hits=search_hits,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _find_xml_document(
        self,
        cik: str,
        accession_no_dashes: str,
        accession_number: str,
    ) -> str | None:
        """Find the XML document for a filing by fetching its index.

        When a Form 4's primaryDocument is HTML, the actual XML source is
        still available in the filing directory. This method fetches the
        filing index JSON and looks for an ``.xml`` file.

        Args:
            cik: CIK number (stripped of leading zeros).
            accession_no_dashes: Accession number without dashes.
            accession_number: Original accession number (for logging).

        Returns:
            The XML document filename, or ``None`` if not found.
        """
        index_url = SEC_FILING_INDEX_URL.format(
            cik=cik,
            accession=accession_no_dashes,
        )
        try:
            response = await self._request(index_url)
            index_data = response.json()
        except (ApiError, ValueError):
            logger.warning(
                "Failed to fetch filing index",
                accession=accession_number,
                server=SERVER_NAME,
            )
            return None

        for item in index_data.get("directory", {}).get("item", []):
            name = item.get("name", "")
            if name.lower().endswith(".xml"):
                return name

        logger.debug(
            "No XML document in filing index",
            accession=accession_number,
            server=SERVER_NAME,
        )
        return None

    async def _request(
        self,
        url: str,
        method: str = "GET",
        json_body: dict | None = None,
    ) -> httpx.Response:
        """Make a rate-limited HTTP request with proper User-Agent.

        Args:
            url: Request URL.
            method: HTTP method (GET or POST).
            json_body: JSON body for POST requests.

        Returns:
            The httpx Response.

        Raises:
            ApiError: On HTTP errors.
        """
        await self._limiter.acquire()

        headers = {"User-Agent": self._user_agent}

        logger.debug(
            "SEC EDGAR request",
            method=method,
            url=url,
            server=SERVER_NAME,
        )

        try:
            if method == "POST":
                response = await self._http.post(url, json=json_body, headers=headers)
            else:
                response = await self._http.get(url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ApiError(
                f"SEC EDGAR HTTP {exc.response.status_code}: " f"{exc.response.text[:200]}"
            ) from exc

        return response

    async def _get_submissions(self, cik: str) -> dict:
        """Fetch the submissions JSON for a CIK.

        Args:
            cik: 10-digit zero-padded CIK.

        Returns:
            Parsed submissions JSON.
        """
        url = SEC_SUBMISSIONS_URL.format(cik=cik)
        response = await self._request(url)
        return response.json()

    async def _load_ticker_map(self) -> None:
        """Fetch and parse the SEC company_tickers.json file.

        Builds an in-memory dict mapping ticker -> (cik_10digit, company_name).
        """
        response = await self._request(SEC_TICKERS_URL)
        data = response.json()

        ticker_map: dict[str, tuple[str, str]] = {}
        for entry in data.values():
            ticker = entry["ticker"].upper()
            cik = str(entry["cik_str"]).zfill(10)
            name = entry["title"]
            ticker_map[ticker] = (cik, name)

        self._ticker_to_cik = ticker_map
        logger.info(
            "Loaded ticker-to-CIK map",
            count=len(ticker_map),
            server=SERVER_NAME,
        )

    @staticmethod
    def parse_form4_xml(xml_content: str) -> list[InsiderTransaction]:
        """Parse a Form 4 XML document into InsiderTransaction objects.

        Args:
            xml_content: Raw XML string of the Form 4 filing.

        Returns:
            List of InsiderTransaction objects found in the filing.
        """
        root = ET.parse(StringIO(xml_content)).getroot()

        # Extract owner info.
        owner_el = root.find(".//reportingOwner")
        owner_name = ""
        owner_title = ""
        if owner_el is not None:
            name_el = owner_el.find(".//rptOwnerName")
            if name_el is not None and name_el.text:
                owner_name = name_el.text.strip()

            title_el = owner_el.find(".//officerTitle")
            if title_el is not None and title_el.text:
                owner_title = title_el.text.strip()
            elif owner_el.find(".//isDirector") is not None:
                is_director = owner_el.find(".//isDirector")
                if (
                    is_director is not None
                    and is_director.text
                    and is_director.text.strip().lower() in ("true", "1")
                ):
                    owner_title = "Director"

        # Extract non-derivative transactions.
        transactions: list[InsiderTransaction] = []
        for txn_el in root.findall(".//nonDerivativeTransaction"):
            date_el = txn_el.find(".//transactionDate/value")
            code_el = txn_el.find(".//transactionCoding/transactionCode")
            shares_el = txn_el.find(".//transactionShares/value")
            price_el = txn_el.find(".//transactionPricePerShare/value")
            post_shares_el = txn_el.find(".//sharesOwnedFollowingTransaction/value")

            if (
                date_el is None
                or code_el is None
                or shares_el is None
                or date_el.text is None
                or code_el.text is None
                or shares_el.text is None
            ):
                continue

            price: float | None = None
            if price_el is not None and price_el.text:
                try:
                    price = float(price_el.text.strip())
                except ValueError:
                    price = None

            post_shares = 0.0
            if post_shares_el is not None and post_shares_el.text:
                try:
                    post_shares = float(post_shares_el.text.strip())
                except ValueError:
                    post_shares = 0.0

            transactions.append(
                InsiderTransaction(
                    owner_name=owner_name,
                    owner_title=owner_title,
                    transaction_date=date_el.text.strip(),
                    transaction_code=code_el.text.strip(),
                    shares=float(shares_el.text.strip()),
                    price_per_share=price,
                    shares_owned_after=post_shares,
                )
            )

        return transactions

    @staticmethod
    def strip_html(text: str) -> str:
        """Remove HTML tags and collapse excessive whitespace.

        Args:
            text: Raw HTML or plain text.

        Returns:
            Clean text with tags removed and whitespace normalized.
        """
        extractor = _HTMLTextExtractor()
        extractor.feed(text)
        result = extractor.get_text()
        # Collapse runs of 3+ newlines into 2.
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()
