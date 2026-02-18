"""Pydantic models for the Earnings Linguist agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from stock_radar.agents.earnings_linguist.config import DEFAULT_HORIZON_DAYS
from stock_radar.agents.models import AgentInput


class SentimentIndicator(BaseModel):
    """A single sentiment signal detected in an earnings transcript."""

    category: Literal[
        "hedging",
        "confidence_shift",
        "tone_change",
        "forward_guidance",
        "risk_language",
    ] = Field(description="Category of sentiment signal")
    quote: str = Field(description="Relevant quote from the transcript")
    interpretation: str = Field(description="What this signal means for investors")
    impact: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        description="Directional impact of this signal"
    )


class EarningsAnalysis(BaseModel):
    """Structured output the LLM must produce.

    This model's JSON schema is embedded in the system prompt, making
    it the single source of truth for the expected output format.
    """

    overall_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        description="Overall sentiment assessment"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the assessment (0.0 to 1.0)"
    )
    sentiment_indicators: list[SentimentIndicator] = Field(
        description="Individual sentiment signals detected"
    )
    quarter_over_quarter_shift: str | None = Field(
        default=None, description="How tone changed vs prior quarter (if available)"
    )
    key_risks: list[str] = Field(description="Key risks identified in the transcript")
    key_opportunities: list[str] = Field(
        description="Key opportunities identified in the transcript"
    )
    reasoning_summary: str = Field(description="Concise summary of the analysis reasoning")
    horizon_days: int = Field(
        default=DEFAULT_HORIZON_DAYS, gt=0, description="Suggested prediction horizon in days"
    )


class EarningsLinguistInput(AgentInput):
    """Input for the Earnings Linguist agent."""

    transcript_content: str = Field(description="Full earnings call transcript text")
    prior_transcript_content: str | None = Field(
        default=None, description="Previous quarter transcript for comparison"
    )
    company_name: str = Field(default="", description="Company name for context")
