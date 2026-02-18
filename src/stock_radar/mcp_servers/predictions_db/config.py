"""Constants for the predictions database MCP server."""

from __future__ import annotations

# Server identity for logging
SERVER_NAME = "predictions-db-mcp"

# Scoring thresholds
NEUTRAL_RETURN_THRESHOLD_PCT = 2.0  # Neutral predictions get PARTIAL if |return| < this

# Default query limits
DEFAULT_QUERY_LIMIT = 50
DEFAULT_ACCURACY_LOOKBACK_DAYS = 90

HORIZON_BUFFER_DAYS = 1  # Days past horizon before eligible for scoring
