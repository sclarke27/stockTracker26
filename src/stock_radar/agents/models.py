"""Base Pydantic models shared by all analysis agents."""

from __future__ import annotations

from pydantic import BaseModel, Field

from stock_radar.models.predictions import Direction


class AgentInput(BaseModel):
    """Base input for all analysis agents."""

    ticker: str = Field(description="Stock ticker symbol")
    quarter: int = Field(ge=1, le=4, description="Fiscal quarter (1-4)")
    year: int = Field(gt=2000, description="Fiscal year")


class AnalysisResult(BaseModel):
    """Result from agent analysis, before prediction logging.

    Contains the core prediction fields plus metadata about which
    LLM produced the analysis.
    """

    ticker: str = Field(description="Stock ticker symbol")
    direction: Direction = Field(description="Predicted price direction")
    confidence: float = Field(ge=0.0, le=1.0, description="Agent confidence (0.0 to 1.0)")
    reasoning: str = Field(description="Detailed reasoning supporting the prediction")
    horizon_days: int = Field(gt=0, description="Prediction horizon in days")
    model_used: str = Field(description="LLM model that produced this analysis")
    escalated: bool = Field(default=False, description="Whether Claude API was used")


class AgentOutput(BaseModel):
    """Full output after prediction logging (includes prediction_id)."""

    prediction_id: str = Field(description="ID from predictions-db after logging")
    result: AnalysisResult = Field(description="The analysis result")
    similar_past_reasoning: list[str] = Field(
        default_factory=list,
        description="Similar historical reasoning from vector store",
    )
