"""Pydantic models for the Cross-Sector Contagion Mapper agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from stock_radar.agents.contagion_mapper.config import DEFAULT_HORIZON_DAYS
from stock_radar.agents.models import AgentInput


class ContagionInput(AgentInput):
    """Input for the Cross-Sector Contagion Mapper agent.

    The base AgentInput.ticker represents the **target** company being
    evaluated for contagion impact. The trigger_ticker is the company
    that experienced the original shock event.
    """

    trigger_ticker: str = Field(
        description="Ticker of the company that experienced the trigger event"
    )
    trigger_company_name: str = Field(description="Full name of the trigger company")
    trigger_event_summary: str = Field(
        description="Brief description of what happened at the trigger company"
    )
    target_company_name: str = Field(description="Full name of the target company (same as ticker)")
    relationship_type: Literal[
        "supplier",
        "customer",
        "competitor",
        "same_sector",
        "distribution_partner",
    ] = Field(description="Nature of the relationship between trigger and target companies")
    trigger_recent_news: list[dict] = Field(
        default_factory=list,
        description="Recent news articles about the trigger company",
    )
    target_recent_news: list[dict] = Field(
        default_factory=list,
        description="Recent news articles about the target company",
    )
    trigger_sector: str = Field(description="Sector of the trigger company")
    target_sector: str = Field(description="Sector of the target company")


class ContagionAnalysis(BaseModel):
    """Structured output the LLM must produce for contagion analysis.

    This model's JSON schema is embedded in the system prompt, making it the
    single source of truth for the expected LLM output format.
    """

    contagion_likely: bool = Field(
        description="Whether meaningful contagion impact is likely on the target"
    )
    contagion_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated probability that the trigger event will impact the target (0-1)",
    )
    contagion_mechanism: str = Field(
        description="Explanation of how the shock propagates from trigger to target"
    )
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        description="Predicted directional impact on the target company's stock"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the contagion assessment (0.0 to 1.0)",
    )
    affected_business_segments: list[str] = Field(
        description="Specific business segments of the target company likely to be impacted"
    )
    timeline_days: int = Field(
        gt=0,
        description="Expected number of days before the contagion impact appears in price",
    )
    mitigating_factors: list[str] = Field(
        description="Factors that reduce the likelihood or magnitude of contagion"
    )
    amplifying_factors: list[str] = Field(
        description="Factors that increase the likelihood or magnitude of contagion"
    )
    horizon_days: int = Field(
        default=DEFAULT_HORIZON_DAYS,
        gt=0,
        description="Suggested prediction horizon in days",
    )
    reasoning_summary: str = Field(description="Concise summary of the full analysis reasoning")
