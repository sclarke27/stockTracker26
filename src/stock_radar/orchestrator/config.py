"""Constants for the orchestrator cycle."""

from __future__ import annotations

# Server identity
ORCHESTRATOR_NAME = "stock-radar-orchestrator"

# Which agents run for which coverage tier.
DEEP_AGENTS: list[str] = [
    "earnings_linguist",
    "narrative_divergence",
    "sec_filing_analyzer",
    "contagion_mapper",
]
LIGHT_AGENTS: list[str] = [
    "narrative_divergence",
    "sec_filing_analyzer",
]

# Phase identifiers (used in PhaseResult.phase).
PHASE_INGESTION = "ingestion"
PHASE_ANALYSIS = "analysis"
PHASE_SCORING = "scoring"
