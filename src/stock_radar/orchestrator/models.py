"""Pydantic result models for orchestrator cycle tracking."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PhaseResult(BaseModel):
    """Outcome of a single orchestrator phase (ingestion, analysis, scoring)."""

    phase: str = Field(description="Phase identifier")
    started_at: str = Field(description="ISO-8601 timestamp when phase started")
    completed_at: str = Field(description="ISO-8601 timestamp when phase completed")
    duration_seconds: float = Field(description="Wall-clock duration in seconds")
    success: bool = Field(description="Whether the phase completed without fatal error")
    error: str | None = Field(default=None, description="Error message if phase failed")


class AgentRunResult(BaseModel):
    """Outcome of a single agent batch run within the analysis phase."""

    agent_name: str = Field(description="Agent identifier (e.g. 'earnings_linguist')")
    tier: str = Field(description="Coverage tier ('deep' or 'light')")
    tickers_run: int = Field(description="Number of tickers processed")
    predictions_generated: int = Field(description="Predictions successfully produced")
    errors: int = Field(description="Number of ticker-level errors")
    duration_seconds: float = Field(description="Wall-clock duration in seconds")


class CycleResult(BaseModel):
    """Aggregated outcome of a full orchestrator cycle."""

    cycle_id: str = Field(description="Unique cycle identifier (UUID)")
    started_at: str = Field(description="ISO-8601 timestamp when cycle started")
    completed_at: str = Field(description="ISO-8601 timestamp when cycle completed")
    duration_seconds: float = Field(description="Total wall-clock duration in seconds")
    quarter: int = Field(description="Fiscal quarter being analyzed")
    year: int = Field(description="Fiscal year being analyzed")
    phases: list[PhaseResult] = Field(description="Results for each phase")
    agent_runs: list[AgentRunResult] = Field(description="Results for each agent batch run")
    total_predictions: int = Field(description="Sum of predictions across all agents")
    total_errors: int = Field(description="Sum of errors across all agents")


RelationshipType = Literal[
    "supplier",
    "customer",
    "competitor",
    "same_sector",
    "distribution_partner",
]


class ContagionPair(BaseModel):
    """A single trigger→target relationship for contagion analysis."""

    trigger: str = Field(description="Trigger ticker symbol")
    target: str = Field(description="Target ticker symbol")
    relationship: RelationshipType = Field(description="Nature of the relationship")


class ContagionPairsConfig(BaseModel):
    """Root model for contagion_pairs.yaml."""

    pairs: list[ContagionPair] = Field(description="All configured contagion pairs")
