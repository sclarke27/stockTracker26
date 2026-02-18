"""Pydantic models for the SEC Filing Pattern Analyzer agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from stock_radar.agents.models import AgentInput
from stock_radar.agents.sec_filing_analyzer.config import DEFAULT_HORIZON_DAYS


class FilingPattern(BaseModel):
    """A recurring pattern detected across recent SEC filings."""

    pattern_type: Literal[
        "insider_buying_cluster",
        "insider_selling_cluster",
        "unusual_8k_frequency",
        "s1_amendment",
        "late_filing",
        "executive_departure",
    ] = Field(description="Type of filing pattern detected")
    description: str = Field(description="Human-readable description of the pattern")
    significance: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Significance level of this pattern as a signal"
    )
    filing_dates: list[str] = Field(
        description="Filing dates associated with this pattern (ISO 8601)"
    )


class InsiderSummary(BaseModel):
    """Aggregate summary of insider transaction activity."""

    net_shares_acquired: float = Field(
        description=(
            "Net shares acquired by insiders (positive = net buying, negative = net selling)"
        )
    )
    total_transactions: int = Field(description="Total number of insider transactions")
    unique_insiders: int = Field(description="Number of distinct insiders who transacted")
    largest_transaction_shares: float = Field(
        description="Absolute share count of the largest single transaction"
    )


class SecFilingInput(AgentInput):
    """Input for the SEC Filing Pattern Analyzer agent.

    Combines recent SEC filings and insider transaction data for
    pattern analysis via the LLM.
    """

    recent_filings: list[dict] = Field(
        description="Recent SEC filings [{form_type, filed_at, description, url}]"
    )
    insider_transactions: list[dict] = Field(
        description=(
            "Insider transactions [{insider_name, transaction_type, shares, date, price_per_share}]"
        )
    )
    filing_count: int = Field(description="Total number of filings in the window")
    insider_transaction_count: int = Field(
        description="Total number of insider transactions in the window"
    )
    lookback_days: int = Field(description="Number of days of filing history analyzed")


class SecFilingAnalysis(BaseModel):
    """Structured output the LLM must produce for SEC filing pattern analysis.

    This model's JSON schema is embedded in the system prompt, making it the
    single source of truth for the expected LLM output format.
    """

    patterns_detected: list[FilingPattern] = Field(
        description="Distinct filing patterns identified in the data"
    )
    insider_summary: InsiderSummary = Field(
        description="Aggregated summary of insider transaction activity"
    )
    insider_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        description="Directional sentiment from insider activity alone"
    )
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        description="Overall predicted price direction based on all filing signals"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment (0.0 to 1.0)",
    )
    risk_flags: list[str] = Field(description="Specific risk indicators identified in the filings")
    key_findings: list[str] = Field(
        description="Most significant findings from the filing analysis"
    )
    horizon_days: int = Field(
        default=DEFAULT_HORIZON_DAYS,
        gt=0,
        description="Suggested prediction horizon in days",
    )
    reasoning_summary: str = Field(description="Concise summary of the full analysis reasoning")
