"""Pydantic models for the prediction scoring runner."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScoringOutcome(BaseModel):
    """Result of scoring a single prediction.

    When a prediction is successfully scored, ``status`` and ``return_pct``
    are populated. When scoring is skipped (e.g. price data unavailable),
    ``skipped`` is True and ``skip_reason`` explains why.
    """

    prediction_id: str = Field(description="ID of the prediction that was scored")
    ticker: str = Field(description="Stock ticker symbol")
    status: str | None = Field(
        default=None,
        description="Scoring outcome: CORRECT, INCORRECT, PARTIAL, or None if skipped",
    )
    return_pct: float | None = Field(
        default=None, description="Actual percentage return, or None if skipped"
    )
    skipped: bool = Field(default=False, description="True if prediction could not be scored")
    skip_reason: str | None = Field(
        default=None,
        description="Reason the prediction was skipped, if applicable",
    )


class ScoringResult(BaseModel):
    """Summary of a complete scoring loop run.

    Captures timing, counts, and per-prediction outcomes for a single
    invocation of the scoring loop.
    """

    started_at: str = Field(description="ISO timestamp when scoring started")
    completed_at: str = Field(description="ISO timestamp when scoring completed")
    duration_seconds: float = Field(ge=0, description="Total scoring duration in seconds")
    predictions_found: int = Field(
        ge=0, description="Number of unscored predictions past their horizon"
    )
    predictions_scored: int = Field(ge=0, description="Number of predictions successfully scored")
    predictions_skipped: int = Field(
        ge=0, description="Number of predictions skipped due to errors"
    )
    outcomes: list[ScoringOutcome] = Field(description="Per-prediction scoring outcomes")
