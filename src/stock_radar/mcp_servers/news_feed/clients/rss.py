"""Google News RSS client for news search fallback."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, urlparse

import httpx
from loguru import logger

from stock_radar.mcp_servers.news_feed.config import RSS_BASE_URL, SERVER_NAME
from stock_radar.mcp_servers.news_feed.exceptions import ApiError
from stock_radar.models.news_feed import NewsArticle, NewsResponse


class RssNewsClient:
    """Async client for Google News RSS search feeds.

    Used as a fallback when Alpha Vantage returns no results for a query.
    RSS articles carry no sentiment data; all sentiment fields default to
    neutral values.

    Args:
        http_client: Shared ``httpx.AsyncClient`` instance.
    """

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    async def search_news(self, query: str, limit: int = 50) -> NewsResponse:
        """Search Google News RSS for articles matching a query.

        Args:
            query: Search query string.
            limit: Maximum number of articles to return.

        Returns:
            Parsed news response with ``source="rss"``.

        Raises:
            ApiError: On HTTP errors from the RSS endpoint.
        """
        params = urlencode({"q": f"{query} stock", "hl": "en-US", "gl": "US", "ceid": "US:en"})
        url = f"{RSS_BASE_URL}?{params}"

        logger.debug("RSS News request: {query}", query=query, server=SERVER_NAME)

        try:
            response = await self._http.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ApiError(
                f"RSS HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc

        articles = self._parse_xml(response.text)[:limit]
        return NewsResponse(
            query=query,
            articles=articles,
            total_fetched=len(articles),
            source="rss",
        )

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _parse_xml(xml_text: str) -> list[NewsArticle]:
        """Parse RSS XML into ``NewsArticle`` instances.

        Args:
            xml_text: Raw XML string from the RSS feed.

        Returns:
            List of parsed articles. Articles with missing title or link are
            skipped.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        articles = []
        for item in root.findall(".//item"):
            title = item.findtext("title") or ""
            url = item.findtext("link") or ""
            time_published = item.findtext("pubDate") or ""
            raw_description = item.findtext("description") or ""
            source_el = item.find("source")
            source_name = source_el.text if source_el is not None else ""

            # Skip malformed entries
            if not title or not url:
                continue

            summary = re.sub(r"<[^>]+>", " ", raw_description).strip()
            # Collapse multiple spaces left by stripped tags
            summary = re.sub(r"\s{2,}", " ", summary)

            source_domain = urlparse(url).netloc

            articles.append(
                NewsArticle(
                    title=title,
                    url=url,
                    time_published=time_published,
                    authors=[],
                    summary=summary,
                    source=source_name or source_domain,
                    source_domain=source_domain,
                    topics=[],
                    overall_sentiment_score=0.0,
                    overall_sentiment_label="Neutral",
                    ticker_sentiment=[],
                )
            )
        return articles
