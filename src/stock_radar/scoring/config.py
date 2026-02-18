"""Constants for the prediction scoring runner."""

from __future__ import annotations

# Server name for structured logging.
SERVER_NAME = "scoring-runner"

# Days past horizon before a prediction is eligible for scoring.
# Ensures the market has closed and Alpha Vantage data has propagated.
HORIZON_BUFFER_DAYS = 1

# Calendar days threshold for choosing compact vs full price history.
# Alpha Vantage "compact" returns ~100 trading days ≈ ~140 calendar days.
# 130 provides a safety margin.
COMPACT_HISTORY_CALENDAR_DAYS = 130
