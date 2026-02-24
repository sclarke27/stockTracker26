"""Tests for the SEC EDGAR API client."""

from __future__ import annotations

import httpx
import pytest
import respx

from stock_radar.mcp_servers.sec_edgar.clients.edgar import EdgarClient
from stock_radar.mcp_servers.sec_edgar.config import (
    SEC_EFTS_URL,
    SEC_TICKERS_URL,
)
from stock_radar.mcp_servers.sec_edgar.exceptions import (
    ApiError,
    CikNotFoundError,
    FilingNotFoundError,
)
from stock_radar.utils.rate_limiter import RateLimiter

# === Sample responses used by both client and server tests ===

SAMPLE_COMPANY_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
}

SAMPLE_SUBMISSIONS = {
    "cik": "320193",
    "entityType": "operating",
    "sic": "3571",
    "sicDescription": "Electronic Computers",
    "name": "Apple Inc.",
    "tickers": ["AAPL"],
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-25-000001",
                "0000320193-25-000002",
                "0000320193-24-000003",
                "0000320193-24-000004",
            ],
            "filingDate": ["2025-01-15", "2025-01-10", "2024-12-15", "2024-11-20"],
            "form": ["8-K", "4", "10-K", "4"],
            "primaryDocument": [
                "doc1.htm",
                "xslF345X05/wk-form4_0002.xml",
                "doc3.htm",
                "xslF345X05/wk-form4_0004.xml",
            ],
            "primaryDocDescription": [
                "Current Report",
                "Statement of Changes",
                "Annual Report",
                "Statement of Changes",
            ],
        },
        "files": [],
    },
}

SAMPLE_FORM4_XML = """\
<?xml version="1.0"?>
<ownershipDocument>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001234567</rptOwnerCik>
            <rptOwnerName>Cook Timothy D</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>true</isDirector>
            <isOfficer>true</isOfficer>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2025-01-10</value></transactionDate>
            <transactionCoding>
                <transactionCode>S</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>50000</value></transactionShares>
                <transactionPricePerShare><value>185.50</value></transactionPricePerShare>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction>\
<value>3000000</value>\
</sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""

SAMPLE_EFTS_RESPONSE = {
    "took": 42,
    "timed_out": False,
    "hits": {
        "total": {"value": 2, "relation": "eq"},
        "max_score": 18.5,
        "hits": [
            {
                "_id": "0000320193-25-000001:doc1.htm",
                "_source": {
                    "ciks": ["0000320193"],
                    "adsh": "0000320193-25-000001",
                    "root_form": "8-K",
                    "file_date": "2025-01-15",
                    "display_names": ["Apple Inc. (AAPL) (CIK 0000320193)"],
                },
            },
            {
                "_id": "0000789019-25-000010:report.htm",
                "_source": {
                    "ciks": ["0000789019"],
                    "adsh": "0000789019-25-000010",
                    "root_form": "10-Q",
                    "file_date": "2025-01-12",
                    "display_names": ["MICROSOFT CORP (MSFT) (CIK 0000789019)"],
                },
            },
        ],
    },
}

SAMPLE_FILING_HTML = """\
<html><body>
<h1>CURRENT REPORT</h1>
<p>Pursuant to Section 13 or 15(d) of the Securities Exchange Act of 1934</p>
<p>Date of Report: January 15, 2025</p>
<div>Apple Inc. has entered into a definitive agreement.</div>
</body></html>"""


# === Fixtures ===


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    """Provide a generous rate limiter for tests."""
    return RateLimiter(
        requests_per_minute=600,
        requests_per_day=50_000,
        requests_per_second=10,
    )


@pytest.fixture()
def edgar_client(rate_limiter: RateLimiter) -> EdgarClient:
    """Provide an EdgarClient with a test httpx client."""
    http_client = httpx.AsyncClient()
    return EdgarClient(
        http_client=http_client,
        rate_limiter=rate_limiter,
        user_agent="StockRadar test@example.com",
    )


# === Tests ===


class TestResolveCik:
    """Tests for ticker-to-CIK resolution."""

    @respx.mock
    async def test_resolves_known_ticker(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        cik, name = await edgar_client.resolve_cik("AAPL")
        assert cik == "0000320193"
        assert name == "Apple Inc."

    @respx.mock
    async def test_case_insensitive(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        cik, name = await edgar_client.resolve_cik("aapl")
        assert cik == "0000320193"

    @respx.mock
    async def test_unknown_ticker_raises(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        with pytest.raises(CikNotFoundError):
            await edgar_client.resolve_cik("ZZZZZ")

    @respx.mock
    async def test_caches_ticker_map(self, edgar_client: EdgarClient) -> None:
        """Second call should not fetch the ticker map again."""
        route = respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        await edgar_client.resolve_cik("AAPL")
        await edgar_client.resolve_cik("MSFT")
        assert route.call_count == 1


class TestGetFilings:
    """Tests for EdgarClient.get_filings()."""

    @respx.mock
    async def test_returns_all_filings(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
        )
        result = await edgar_client.get_filings("AAPL")
        assert result.ticker == "AAPL"
        assert result.cik == "0000320193"
        assert result.company_name == "Apple Inc."
        assert len(result.filings) == 4

    @respx.mock
    async def test_filters_by_form_type(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
        )
        result = await edgar_client.get_filings("AAPL", form_types=["8-K", "10-K"])
        assert len(result.filings) == 2
        form_types = {f.form_type for f in result.filings}
        assert form_types == {"8-K", "10-K"}

    @respx.mock
    async def test_respects_limit(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
        )
        result = await edgar_client.get_filings("AAPL", limit=2)
        assert len(result.filings) == 2

    @respx.mock
    async def test_empty_filings(self, edgar_client: EdgarClient) -> None:
        empty_submissions = {
            **SAMPLE_SUBMISSIONS,
            "filings": {
                "recent": {
                    "accessionNumber": [],
                    "filingDate": [],
                    "form": [],
                    "primaryDocument": [],
                    "primaryDocDescription": [],
                },
                "files": [],
            },
        }
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=empty_submissions)
        )
        result = await edgar_client.get_filings("AAPL")
        assert result.filings == []


class TestGetFilingText:
    """Tests for EdgarClient.get_filing_text()."""

    @respx.mock
    async def test_fetches_and_strips_html(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
        )
        respx.get(url__startswith="https://www.sec.gov/Archives/").mock(
            return_value=httpx.Response(200, text=SAMPLE_FILING_HTML)
        )
        result = await edgar_client.get_filing_text("AAPL", "0000320193-25-000001")
        assert result.ticker == "AAPL"
        assert result.form_type == "8-K"
        assert result.truncated is False
        # HTML tags should be stripped.
        assert "<html>" not in result.content
        assert "definitive agreement" in result.content

    @respx.mock
    async def test_truncates_long_content(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
        )
        long_text = "x" * 200_000
        respx.get(url__startswith="https://www.sec.gov/Archives/").mock(
            return_value=httpx.Response(200, text=long_text)
        )
        result = await edgar_client.get_filing_text("AAPL", "0000320193-25-000001", max_length=100)
        assert result.truncated is True
        assert len(result.content) == 100

    @respx.mock
    async def test_filing_not_found_raises(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
        )
        with pytest.raises(FilingNotFoundError):
            await edgar_client.get_filing_text("AAPL", "9999999999-99-999999")


class TestGetInsiderTransactions:
    """Tests for EdgarClient.get_insider_transactions()."""

    @respx.mock
    async def test_parses_form4_transactions(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=SAMPLE_SUBMISSIONS)
        )
        # Mock the Form 4 XML fetch (there are 2 Form 4s in SAMPLE_SUBMISSIONS).
        respx.get(url__startswith="https://www.sec.gov/Archives/").mock(
            return_value=httpx.Response(200, text=SAMPLE_FORM4_XML)
        )
        result = await edgar_client.get_insider_transactions("AAPL")
        assert result.ticker == "AAPL"
        assert result.company_name == "Apple Inc."
        assert len(result.transactions) >= 1
        txn = result.transactions[0]
        assert txn.owner_name == "Cook Timothy D"
        assert txn.transaction_code == "S"
        assert txn.shares == 50000.0
        assert txn.price_per_share == 185.50

    @respx.mock
    async def test_html_primary_doc_falls_back_to_index(
        self, edgar_client: EdgarClient,
    ) -> None:
        """When primaryDocument is HTML, fetch the filing index to find the XML."""
        html_submissions = {
            **SAMPLE_SUBMISSIONS,
            "filings": {
                "recent": {
                    "accessionNumber": ["0000320193-25-000010"],
                    "filingDate": ["2025-02-01"],
                    "form": ["4"],
                    "primaryDocument": ["xslF345X05/form4.htm"],
                    "primaryDocDescription": ["Statement of Changes"],
                },
                "files": [],
            },
        }
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=html_submissions)
        )
        # Filing index lists the XML document.
        filing_index = {
            "directory": {
                "item": [
                    {"name": "xslF345X05/form4.htm", "type": "text/html"},
                    {"name": "form4.xml", "type": "text/xml"},
                ],
            },
        }
        respx.get(url__regex=r".*/index\.json$").mock(
            return_value=httpx.Response(200, json=filing_index)
        )
        # XML fetch should use the discovered filename.
        respx.get(url__regex=r".*/form4\.xml$").mock(
            return_value=httpx.Response(200, text=SAMPLE_FORM4_XML)
        )
        result = await edgar_client.get_insider_transactions("AAPL")
        assert len(result.transactions) >= 1
        assert result.transactions[0].owner_name == "Cook Timothy D"

    @respx.mock
    async def test_no_insider_filings(self, edgar_client: EdgarClient) -> None:
        no_insider_submissions = {
            **SAMPLE_SUBMISSIONS,
            "filings": {
                "recent": {
                    "accessionNumber": ["0000320193-25-000001"],
                    "filingDate": ["2025-01-15"],
                    "form": ["8-K"],
                    "primaryDocument": ["doc1.htm"],
                    "primaryDocDescription": ["Current Report"],
                },
                "files": [],
            },
        }
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(200, json=no_insider_submissions)
        )
        result = await edgar_client.get_insider_transactions("AAPL")
        assert result.transactions == []


class TestSearchFilings:
    """Tests for EdgarClient.search_filings()."""

    @respx.mock
    async def test_parses_search_results(self, edgar_client: EdgarClient) -> None:
        respx.post(SEC_EFTS_URL).mock(return_value=httpx.Response(200, json=SAMPLE_EFTS_RESPONSE))
        result = await edgar_client.search_filings("artificial intelligence")
        assert result.query == "artificial intelligence"
        assert result.total_hits == 2
        assert len(result.hits) == 2
        assert result.hits[0].accession_number == "0000320193-25-000001"
        assert result.hits[0].form_type == "8-K"

    @respx.mock
    async def test_search_with_filters(self, edgar_client: EdgarClient) -> None:
        respx.post(SEC_EFTS_URL).mock(return_value=httpx.Response(200, json=SAMPLE_EFTS_RESPONSE))
        result = await edgar_client.search_filings(
            "revenue",
            form_types=["10-K"],
            date_from="2025-01-01",
            date_to="2025-01-31",
        )
        assert result.total_hits == 2

    @respx.mock
    async def test_empty_search_results(self, edgar_client: EdgarClient) -> None:
        empty_response = {
            "took": 5,
            "timed_out": False,
            "hits": {"total": {"value": 0}, "hits": []},
        }
        respx.post(SEC_EFTS_URL).mock(return_value=httpx.Response(200, json=empty_response))
        result = await edgar_client.search_filings("xyznonexistent")
        assert result.total_hits == 0
        assert result.hits == []


class TestParseForm4Xml:
    """Tests for Form 4 XML parsing."""

    def test_parses_transaction(self) -> None:
        transactions = EdgarClient.parse_form4_xml(SAMPLE_FORM4_XML)
        assert len(transactions) == 1
        txn = transactions[0]
        assert txn.owner_name == "Cook Timothy D"
        assert txn.owner_title == "Chief Executive Officer"
        assert txn.transaction_date == "2025-01-10"
        assert txn.transaction_code == "S"
        assert txn.shares == 50000.0
        assert txn.price_per_share == 185.50
        assert txn.shares_owned_after == 3000000.0

    def test_empty_xml_returns_empty_list(self) -> None:
        xml = """\
<?xml version="1.0"?>
<ownershipDocument>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerName>Unknown</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship/>
    </reportingOwner>
    <nonDerivativeTable/>
</ownershipDocument>"""
        transactions = EdgarClient.parse_form4_xml(xml)
        assert transactions == []


class TestStripHtml:
    """Tests for HTML stripping."""

    def test_removes_tags(self) -> None:
        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        assert EdgarClient.strip_html(html) == "Hello world"

    def test_plain_text_unchanged(self) -> None:
        text = "Just plain text."
        assert EdgarClient.strip_html(text) == "Just plain text."

    def test_collapses_whitespace(self) -> None:
        html = "<p>Hello</p>\n\n\n<p>World</p>"
        result = EdgarClient.strip_html(html)
        # Should not have excessive blank lines.
        assert "\n\n\n" not in result


class TestErrorHandling:
    """Tests for HTTP error scenarios."""

    @respx.mock
    async def test_http_403_raises_api_error(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(return_value=httpx.Response(403, text="Forbidden"))
        with pytest.raises(ApiError):
            await edgar_client.resolve_cik("AAPL")

    @respx.mock
    async def test_http_500_raises_api_error(self, edgar_client: EdgarClient) -> None:
        respx.get(SEC_TICKERS_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPANY_TICKERS)
        )
        respx.get(url__startswith="https://data.sec.gov/submissions/").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        with pytest.raises(ApiError):
            await edgar_client.get_filings("AAPL")
