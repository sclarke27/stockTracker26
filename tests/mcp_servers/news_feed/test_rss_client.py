"""Tests for the Google News RSS client."""

from __future__ import annotations

import httpx
import pytest
import respx

from stock_radar.mcp_servers.news_feed.clients.rss import RssNewsClient
from stock_radar.mcp_servers.news_feed.config import RSS_BASE_URL

RSS_SEARCH_URL = RSS_BASE_URL


SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News Search</title>
    <link>https://news.google.com</link>
    <description>Google News</description>
    <item>
      <title>Apple Stock Surges After Earnings Beat</title>
      <link>https://bloomberg.com/apple-earnings</link>
      <pubDate>Wed, 15 Jan 2025 12:00:00 GMT</pubDate>
      <description>&lt;a href="https://bloomberg.com/apple-earnings"&gt;Apple Stock Surges&lt;/a&gt;
      Apple beat earnings estimates for the fourth quarter.</description>
      <source url="https://bloomberg.com">Bloomberg</source>
    </item>
    <item>
      <title>Tech Giants Rally as Fed Signals Rate Cut</title>
      <link>https://reuters.com/tech-rally</link>
      <pubDate>Tue, 14 Jan 2025 09:00:00 GMT</pubDate>
      <description>Tech stocks rallied broadly on Fed commentary.</description>
      <source url="https://reuters.com">Reuters</source>
    </item>
  </channel>
</rss>
"""

SAMPLE_RSS_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News Search</title>
    <item></item>
  </channel>
</rss>
"""

SAMPLE_RSS_NO_ITEMS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News Search</title>
  </channel>
</rss>
"""


@pytest.fixture()
def rss_client() -> RssNewsClient:
    http_client = httpx.AsyncClient()
    return RssNewsClient(http_client=http_client)


class TestSearchNews:
    """Tests for RssNewsClient.search_news()."""

    @respx.mock
    async def test_parses_articles(self, rss_client: RssNewsClient) -> None:
        respx.get(RSS_SEARCH_URL).mock(return_value=httpx.Response(200, text=SAMPLE_RSS_XML))
        result = await rss_client.search_news("apple stock")
        assert result.query == "apple stock"
        assert result.source == "rss"
        assert result.total_fetched == 2
        assert len(result.articles) == 2

    @respx.mock
    async def test_article_title_parsed(self, rss_client: RssNewsClient) -> None:
        respx.get(RSS_SEARCH_URL).mock(return_value=httpx.Response(200, text=SAMPLE_RSS_XML))
        result = await rss_client.search_news("apple stock")
        assert result.articles[0].title == "Apple Stock Surges After Earnings Beat"

    @respx.mock
    async def test_article_url_parsed(self, rss_client: RssNewsClient) -> None:
        respx.get(RSS_SEARCH_URL).mock(return_value=httpx.Response(200, text=SAMPLE_RSS_XML))
        result = await rss_client.search_news("apple stock")
        assert result.articles[0].url == "https://bloomberg.com/apple-earnings"

    @respx.mock
    async def test_source_domain_extracted(self, rss_client: RssNewsClient) -> None:
        respx.get(RSS_SEARCH_URL).mock(return_value=httpx.Response(200, text=SAMPLE_RSS_XML))
        result = await rss_client.search_news("apple")
        assert result.articles[0].source_domain == "bloomberg.com"
        assert result.articles[1].source_domain == "reuters.com"

    @respx.mock
    async def test_html_stripped_from_description(self, rss_client: RssNewsClient) -> None:
        respx.get(RSS_SEARCH_URL).mock(return_value=httpx.Response(200, text=SAMPLE_RSS_XML))
        result = await rss_client.search_news("apple")
        summary = result.articles[0].summary
        assert "<" not in summary
        assert ">" not in summary
        assert "Apple beat earnings estimates" in summary

    @respx.mock
    async def test_sentiment_defaults_to_neutral(self, rss_client: RssNewsClient) -> None:
        respx.get(RSS_SEARCH_URL).mock(return_value=httpx.Response(200, text=SAMPLE_RSS_XML))
        result = await rss_client.search_news("apple")
        article = result.articles[0]
        assert article.overall_sentiment_score == 0.0
        assert article.overall_sentiment_label == "Neutral"
        assert article.ticker_sentiment == []
        assert article.topics == []

    @respx.mock
    async def test_limit_respected(self, rss_client: RssNewsClient) -> None:
        respx.get(RSS_SEARCH_URL).mock(return_value=httpx.Response(200, text=SAMPLE_RSS_XML))
        result = await rss_client.search_news("apple", limit=1)
        assert len(result.articles) == 1

    @respx.mock
    async def test_empty_feed_returns_empty(self, rss_client: RssNewsClient) -> None:
        respx.get(RSS_SEARCH_URL).mock(return_value=httpx.Response(200, text=SAMPLE_RSS_NO_ITEMS))
        result = await rss_client.search_news("nothing")
        assert result.articles == []
        assert result.total_fetched == 0

    @respx.mock
    async def test_http_error_raises(self, rss_client: RssNewsClient) -> None:
        from stock_radar.mcp_servers.news_feed.exceptions import ApiError

        respx.get(RSS_SEARCH_URL).mock(return_value=httpx.Response(503, text="Service Unavailable"))
        with pytest.raises(ApiError):
            await rss_client.search_news("apple")
