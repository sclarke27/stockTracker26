"""Constants for the Narrative vs Price Divergence agent."""

from __future__ import annotations

# Agent identity
AGENT_NAME = "narrative_divergence"
SIGNAL_TYPE = "narrative_price_divergence"

# Default prediction horizon
DEFAULT_HORIZON_DAYS = 10

# Escalation thresholds
ESCALATION_CONFIDENCE_THRESHOLD = 0.3
ESCALATION_MIN_ARTICLES = 5  # escalate if fewer articles (noisy signal)

# Data fetch limits
MAX_TOP_ARTICLES = 5  # top articles to include in prompt context
