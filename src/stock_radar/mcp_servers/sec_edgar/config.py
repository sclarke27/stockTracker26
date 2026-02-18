"""Constants for the SEC EDGAR MCP server."""

from __future__ import annotations

# Server identity for logging
SERVER_NAME = "sec-edgar-mcp"

# SEC EDGAR base URLs
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

# Cache TTLs (seconds). None = never expires.
CACHE_TTL_FILINGS = 86_400  # 24 hours
CACHE_TTL_FILING_TEXT: int | None = None  # Never expires (filings are immutable)
CACHE_TTL_INSIDER_TRANSACTIONS = 86_400  # 24 hours
CACHE_TTL_FILING_SEARCH = 3_600  # 1 hour

# SEC EDGAR rate limit
SEC_RATE_LIMIT_PER_SECOND = 10

# Filing text defaults
DEFAULT_MAX_FILING_TEXT_LENGTH = 100_000  # characters

# Insider transaction defaults
DEFAULT_INSIDER_FILING_LIMIT = 20  # Max Form 4 filings to parse per request

# User-Agent header template (SEC requires contact info)
SEC_USER_AGENT_TEMPLATE = "StockRadar {email}"
