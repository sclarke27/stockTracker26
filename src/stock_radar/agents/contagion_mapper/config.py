"""Constants for the Cross-Sector Contagion Mapper agent."""

from __future__ import annotations

# Agent identity
AGENT_NAME = "contagion_mapper"
SIGNAL_TYPE = "contagion_signal"

# Default prediction horizon
DEFAULT_HORIZON_DAYS = 5

# Escalation thresholds
ESCALATION_CONFIDENCE_THRESHOLD = 0.3

# Data fetch limits
MAX_TRIGGER_NEWS = 5  # recent news articles for the trigger company
MAX_TARGET_NEWS = 5  # recent news articles for the target company
