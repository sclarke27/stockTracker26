"""Pydantic models for the Narrative vs Price Divergence agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from stock_radar.agents.models import AgentInput
from stock_radar.agents.narrative_divergence.config import DEFAULT_HORIZON_DAYS


class NarrativeDivergenceInput(AgentInput):
    """Input for the Narrative vs Price Divergence agent.

    Combines news sentiment data (from news-feed-mcp) with recent price
    action (from market-data-mcp) to enable divergence detection.
    """

    sentiment_score: float = Field(description="Average news sentiment score (-1.0 to 1.0)")
    article_count: int = Field(description="Number of articles in the sentiment window")
    average_sentiment_label: str = Field(
        description="Human-readable sentiment label (e.g. 'Somewhat-Bullish')"
    )
    price_return_30d: float = Field(
        description="30-day price return as a decimal (e.g. -0.08 for -8%)"
    )
    price_return_7d: float = Field(
        description="7-day price return as a decimal (e.g. 0.02 for +2%)"
    )
    top_articles: list[dict] = Field(
        default_factory=list,
        description="Top article snippets [{title, summary, sentiment_score}]",
    )
    time_from: str | None = Field(
        default=None,
        description="Start of sentiment window (AV format 'YYYYMMDDTHHMM')",
    )
    time_to: str | None = Field(
        default=None,
        description="End of sentiment window (AV format 'YYYYMMDDTHHMM')",
    )


class NarrativeAnalysis(BaseModel):
    """Structured output the LLM must produce for narrative divergence analysis.

    This model's JSON schema is embedded in the system prompt, making it the
    single source of truth for the expected LLM output format.
    """

    divergence_detected: bool = Field(
        description="Whether a meaningful divergence between narrative and price exists"
    )
    divergence_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Strength of the divergence signal (0.0 = none, 1.0 = extreme)",
    )
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        description="Predicted directional impact for the target ticker"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment (0.0 to 1.0)",
    )
    narrative_summary: str = Field(
        description="Summary of what the news sentiment is saying about the stock"
    )
    price_action_summary: str = Field(
        description="Summary of recent price action (trend, magnitude)"
    )
    divergence_explanation: str = Field(
        description="Explanation of why the narrative and price action diverge"
    )
    key_catalysts: list[str] = Field(description="Key news events or themes driving the narrative")
    horizon_days: int = Field(
        default=DEFAULT_HORIZON_DAYS,
        gt=0,
        description="Suggested prediction horizon in days",
    )
    reasoning_summary: str = Field(description="Concise summary of the full analysis reasoning")
