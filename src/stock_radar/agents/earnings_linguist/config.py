"""Constants for the Earnings Linguist agent."""

from __future__ import annotations

# Agent identity
AGENT_NAME = "earnings_linguist"
SIGNAL_TYPE = "earnings_sentiment"

# Default prediction horizon
DEFAULT_HORIZON_DAYS = 5

# Escalation thresholds
ESCALATION_CONFIDENCE_THRESHOLD = 0.3
ESCALATION_TRANSCRIPT_LENGTH = 24_000  # characters (~6K tokens)
