"""Constants for the SEC Filing Pattern Analyzer agent."""

from __future__ import annotations

# Agent identity
AGENT_NAME = "sec_filing_analyzer"
SIGNAL_TYPE = "sec_filing_pattern"

# Default prediction horizon
DEFAULT_HORIZON_DAYS = 15

# Escalation thresholds
ESCALATION_CONFIDENCE_THRESHOLD = 0.3
ESCALATION_FILING_COUNT = 30  # escalate if > 30 filings (complex synthesis)
ESCALATION_INSIDER_TRANSACTION_COUNT = 20  # escalate if many insider transactions

# Data fetch limits
LOOKBACK_DAYS = 90  # default filing window to analyze
MAX_FILINGS = 50  # cap to keep prompt context manageable
MAX_INSIDER_TRANSACTIONS = 30
