"""Pydantic models for data ingestion pipeline results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCallResult(BaseModel):
    """Result of a single MCP tool invocation."""

    tool_name: str = Field(description="Name of the MCP tool called")
    ticker: str = Field(description="Stock ticker the tool was called for")
    success: bool = Field(description="Whether the call succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_ms: float = Field(ge=0, description="Call duration in milliseconds")


class TickerResult(BaseModel):
    """Aggregated results for all tool calls on a single ticker."""

    ticker: str = Field(description="Stock ticker symbol")
    tier: str = Field(description="Coverage tier (deep or light)")
    results: list[ToolCallResult] = Field(description="Individual tool call results")
    success_count: int = Field(ge=0, description="Number of successful calls")
    error_count: int = Field(ge=0, description="Number of failed calls")


class PipelineResult(BaseModel):
    """Summary of a complete pipeline run."""

    started_at: str = Field(description="ISO timestamp when pipeline started")
    completed_at: str = Field(description="ISO timestamp when pipeline completed")
    duration_seconds: float = Field(ge=0, description="Total pipeline duration in seconds")
    tickers_processed: int = Field(ge=0, description="Number of tickers processed")
    total_calls: int = Field(ge=0, description="Total MCP tool calls made")
    total_errors: int = Field(ge=0, description="Total failed tool calls")
    ticker_results: list[TickerResult] = Field(description="Per-ticker results")
