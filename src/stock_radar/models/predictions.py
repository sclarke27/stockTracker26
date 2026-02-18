"""Pydantic models for the predictions DB MCP server."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["BULLISH", "BEARISH", "NEUTRAL"]
"""Valid prediction direction values."""

ScoringStatus = Literal["CORRECT", "INCORRECT", "PARTIAL"]
"""Valid scoring outcome values."""


class PredictionRecord(BaseModel):
    """Full prediction row, including optional scoring fields.

    Represents a single prediction made by an analysis agent. Scoring fields
    are None until the prediction horizon has elapsed and the prediction has
    been evaluated against actual price movement.
    """

    id: str = Field(description="Unique prediction identifier")
    ticker: str = Field(description="Stock ticker symbol")
    agent_name: str = Field(description="Name of the agent that generated this prediction")
    signal_type: str = Field(description="Type of signal detected (e.g. earnings_sentiment)")
    direction: Direction = Field(description="Predicted price direction")
    confidence: float = Field(ge=0.0, le=1.0, description="Agent confidence score (0.0 to 1.0)")
    reasoning: str = Field(description="Agent reasoning supporting the prediction")
    prediction_date: str = Field(description="Date the prediction was made (YYYY-MM-DD)")
    horizon_days: int = Field(gt=0, description="Number of days until prediction is evaluated")
    created_at: str = Field(description="ISO 8601 timestamp when the record was created")

    # Scoring fields — populated after horizon elapses
    scored_at: str | None = Field(
        default=None, description="ISO 8601 timestamp when scoring occurred"
    )
    actual_price_close: float | None = Field(
        default=None, description="Closing price on prediction date"
    )
    actual_price_at_horizon: float | None = Field(
        default=None, description="Closing price at horizon date"
    )
    return_pct: float | None = Field(
        default=None, description="Percentage return over the horizon period"
    )
    status: ScoringStatus | None = Field(
        default=None, description="Scoring outcome after evaluation"
    )


class LogPredictionResponse(BaseModel):
    """Response returned after logging a new prediction."""

    prediction_id: str = Field(description="ID of the newly created prediction")
    created_at: str = Field(description="ISO 8601 timestamp of creation")


class ScorePredictionResponse(BaseModel):
    """Response returned after scoring a prediction against actual results."""

    prediction_id: str = Field(description="ID of the scored prediction")
    status: ScoringStatus = Field(description="Scoring outcome")
    return_pct: float = Field(description="Actual percentage return over the horizon")
    direction: Direction = Field(description="Original predicted direction")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Original agent confidence score (0.0 to 1.0)"
    )


class PredictionHistoryResponse(BaseModel):
    """Paginated list of prediction records."""

    predictions: list[PredictionRecord] = Field(description="List of prediction records")
    total_count: int = Field(ge=0, description="Total number of matching predictions")


class PendingScoringResponse(BaseModel):
    """Predictions awaiting scoring (unscored, past horizon)."""

    predictions: list[PredictionRecord] = Field(
        description="Unscored predictions whose horizon has elapsed"
    )
    total_count: int = Field(ge=0, description="Number of pending predictions")


class AgentStats(BaseModel):
    """Accuracy and performance statistics for a single agent and signal type."""

    agent_name: str = Field(description="Name of the analysis agent")
    signal_type: str = Field(description="Signal type produced by the agent")
    total: int = Field(ge=0, description="Total number of predictions")
    scored: int = Field(ge=0, description="Number of predictions that have been scored")
    correct: int = Field(ge=0, description="Number of predictions scored as CORRECT")
    accuracy_pct: float = Field(ge=0.0, description="Accuracy percentage (correct / scored * 100)")
    avg_confidence: float = Field(
        ge=0.0, le=1.0, description="Average confidence across all predictions"
    )
    avg_return_when_correct: float = Field(
        description="Average percentage return for correct predictions"
    )


class AgentAccuracyResponse(BaseModel):
    """Aggregated accuracy statistics across all agents."""

    agent_stats: list[AgentStats] = Field(description="Per-agent accuracy statistics")
