"""Finnhub API client for earnings transcript retrieval."""

from __future__ import annotations

import httpx
from loguru import logger

from stock_radar.mcp_servers.market_data.config import FINNHUB_BASE_URL, SERVER_NAME
from stock_radar.mcp_servers.market_data.exceptions import ApiError, TickerNotFoundError
from stock_radar.models.market_data import EarningsTranscriptResponse


class FinnhubClient:
    """Async client for the Finnhub REST API.

    Handles earnings transcript retrieval. Authenticates via the
    ``X-Finnhub-Token`` header.

    Args:
        api_key: Finnhub API key.
        http_client: Shared ``httpx.AsyncClient`` instance.
    """

    def __init__(self, api_key: str, http_client: httpx.AsyncClient) -> None:
        self._api_key = api_key
        self._http = http_client

    async def get_earnings_transcript(
        self,
        ticker: str,
        quarter: int,
        year: int,
    ) -> EarningsTranscriptResponse:
        """Fetch an earnings call transcript for a specific quarter.

        Uses a two-step flow: first lists available transcripts for the
        ticker to find the ID matching the requested quarter/year, then
        fetches the full transcript by ID.

        Args:
            ticker: Stock ticker symbol.
            quarter: Fiscal quarter (1-4).
            year: Fiscal year.

        Returns:
            Parsed transcript with concatenated speaker text.

        Raises:
            TickerNotFoundError: If no transcript is available.
            ApiError: On HTTP errors (including 403 for premium-only access).
        """
        # Step 1: find the transcript ID for the requested quarter/year.
        listing = await self._request(
            "/stock/transcripts/list",
            params={"symbol": ticker},
        )

        transcript_id = None
        for entry in listing.get("transcripts", []):
            if entry.get("quarter") == quarter and entry.get("year") == year:
                transcript_id = entry.get("id")
                break

        if transcript_id is None:
            raise TickerNotFoundError(
                f"No earnings transcript for {ticker} Q{quarter} {year}."
            )

        # Step 2: fetch the full transcript by ID.
        data = await self._request(
            "/stock/transcripts",
            params={"id": transcript_id},
        )

        transcript_entries = data.get("transcript", [])
        if not transcript_entries:
            raise TickerNotFoundError(
                f"Empty transcript for {ticker} Q{quarter} {year} (id={transcript_id})."
            )

        content = self._format_transcript(transcript_entries)

        return EarningsTranscriptResponse(
            ticker=ticker,
            quarter=quarter,
            year=year,
            date=data.get("time", ""),
            content=content,
        )

    async def _request(self, endpoint: str, params: dict[str, str]) -> dict:
        """Make an authenticated request to the Finnhub API.

        Args:
            endpoint: API path (e.g. ``"/stock/transcripts"``).
            params: Query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            ApiError: On HTTP errors (403 for premium-only endpoints, etc.).
            TickerNotFoundError: On 302 redirects (deprecated endpoints).
        """
        url = f"{FINNHUB_BASE_URL}{endpoint}"
        headers = {"X-Finnhub-Token": self._api_key}

        logger.debug(
            "Finnhub request: {endpoint} {params}",
            endpoint=endpoint,
            params=params,
            server=SERVER_NAME,
        )

        try:
            response = await self._http.get(url, params=params, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.is_redirect:
                raise TickerNotFoundError(
                    f"Finnhub resource not found (302 redirect): {endpoint}"
                ) from exc
            if exc.response.status_code == 403:
                raise ApiError(
                    f"Finnhub 403: endpoint {endpoint} requires a premium plan"
                ) from exc
            raise ApiError(
                f"Finnhub HTTP {exc.response.status_code}: " f"{exc.response.text[:200]}"
            ) from exc

        return response.json()

    @staticmethod
    def _format_transcript(entries: list[dict]) -> str:
        """Format raw transcript entries into readable text.

        Each speaker's remarks are prefixed with their name and joined
        into a single string.

        Args:
            entries: List of ``{"name": str, "speech": list[str]}`` dicts.

        Returns:
            Formatted transcript text.
        """
        parts = []
        for entry in entries:
            name = entry.get("name", "Unknown")
            speech = " ".join(entry.get("speech", []))
            parts.append(f"{name}: {speech}")
        return "\n\n".join(parts)
