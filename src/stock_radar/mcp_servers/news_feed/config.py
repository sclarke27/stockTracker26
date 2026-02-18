"""Constants for the news feed MCP server."""

from __future__ import annotations

# Cache TTLs (seconds)
CACHE_TTL_NEWS = 3_600  # 1 hour
CACHE_TTL_SENTIMENT_SUMMARY = 14_400  # 4 hours

# Alpha Vantage rate limits (shared free-tier limits)
AV_RATE_LIMIT_PER_MINUTE = 5
AV_RATE_LIMIT_PER_DAY = 500

# Alpha Vantage base URL
AV_BASE_URL = "https://www.alphavantage.co/query"

# Google News RSS base URL
RSS_BASE_URL = "https://news.google.com/rss/search"

# Alpha Vantage request limits
AV_DEFAULT_LIMIT = 50
AV_MAX_LIMIT = 1_000

# Sentiment classification thresholds (matches AV's own labeling)
SENTIMENT_BULLISH_THRESHOLD = 0.15
SENTIMENT_BEARISH_THRESHOLD = -0.15

# Number of top topics to include in sentiment summary
SENTIMENT_TOP_TOPICS_COUNT = 10

# Server identity for logging
SERVER_NAME = "news-feed-mcp"
