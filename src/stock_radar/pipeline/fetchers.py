"""Data fetchers that call MCP server tools for each coverage tier.

Each fetcher function orchestrates multiple MCP tool calls for a single ticker,
collecting results into a structured TickerResult.  Individual tool failures are
isolated — one broken call never prevents the remaining tools from running.
"""

from __future__ import annotations

import time

from fastmcp import Client
from loguru import logger

from stock_radar.pipeline.config import (
    DEEP_EDGAR_TOOLS,
    DEEP_MARKET_TOOLS,
    LIGHT_EDGAR_TOOLS,
    LIGHT_MARKET_TOOLS,
    TIER_DEEP,
    TIER_LIGHT,
)
from stock_radar.pipeline.models import TickerResult, ToolCallResult


async def _safe_call(
    client: Client,
    tool_name: str,
    args: dict,
    ticker: str,
) -> ToolCallResult:
    """Call a single MCP tool, catching errors and returning a structured result.

    Never raises — all exceptions are caught and returned as a failed
    ToolCallResult so that callers can continue processing remaining tools.

    Args:
        client: FastMCP Client connected to the appropriate server.
        tool_name: Name of the MCP tool to call.
        args: Arguments to pass to the tool.
        ticker: Ticker symbol (for logging and result tagging).

    Returns:
        ToolCallResult indicating success or failure.
    """
    start = time.monotonic()
    try:
        await client.call_tool(tool_name, args)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("Tool call succeeded", tool=tool_name, ticker=ticker)
        return ToolCallResult(
            tool_name=tool_name,
            ticker=ticker,
            success=True,
            duration_ms=round(elapsed_ms, 2),
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        error_msg = str(exc)
        logger.warning("Tool call failed", tool=tool_name, ticker=ticker, error=error_msg)
        return ToolCallResult(
            tool_name=tool_name,
            ticker=ticker,
            success=False,
            error=error_msg,
            duration_ms=round(elapsed_ms, 2),
        )


async def fetch_deep(
    market_client: Client,
    edgar_client: Client,
    ticker: str,
    quarter: int,
    year: int,
) -> TickerResult:
    """Fetch all deep-tier data for a single ticker.

    Calls 6 MCP tools across market-data and sec-edgar servers.
    Individual tool failures are isolated — other tools still run.

    Args:
        market_client: FastMCP Client for market-data-mcp.
        edgar_client: FastMCP Client for sec-edgar-mcp.
        ticker: Stock ticker symbol.
        quarter: Fiscal quarter for transcript (1-4).
        year: Fiscal year for transcript.

    Returns:
        TickerResult with results from all 6 tool calls.
    """
    results: list[ToolCallResult] = []

    # Market data tools — each has its own argument signature.
    market_args: dict[str, dict] = {
        "get_price_history": {"ticker": ticker, "outputsize": "compact"},
        "get_quote": {"ticker": ticker},
        "get_company_info": {"ticker": ticker},
        "get_earnings_transcript": {"ticker": ticker, "quarter": quarter, "year": year},
    }
    for tool_name in DEEP_MARKET_TOOLS:
        result = await _safe_call(market_client, tool_name, market_args[tool_name], ticker)
        results.append(result)

    # SEC EDGAR tools — all take only ticker.
    edgar_args: dict = {"ticker": ticker}
    for tool_name in DEEP_EDGAR_TOOLS:
        result = await _safe_call(edgar_client, tool_name, edgar_args, ticker)
        results.append(result)

    success_count = sum(1 for r in results if r.success)
    error_count = len(results) - success_count

    return TickerResult(
        ticker=ticker,
        tier=TIER_DEEP,
        results=results,
        success_count=success_count,
        error_count=error_count,
    )


async def fetch_light(
    market_client: Client,
    edgar_client: Client,
    ticker: str,
) -> TickerResult:
    """Fetch light-tier data for a single ticker.

    Calls 2 MCP tools: quote from market-data and filings from sec-edgar.

    Args:
        market_client: FastMCP Client for market-data-mcp.
        edgar_client: FastMCP Client for sec-edgar-mcp.
        ticker: Stock ticker symbol.

    Returns:
        TickerResult with results from 2 tool calls.
    """
    results: list[ToolCallResult] = []

    for tool_name in LIGHT_MARKET_TOOLS:
        result = await _safe_call(market_client, tool_name, {"ticker": ticker}, ticker)
        results.append(result)

    for tool_name in LIGHT_EDGAR_TOOLS:
        result = await _safe_call(edgar_client, tool_name, {"ticker": ticker}, ticker)
        results.append(result)

    success_count = sum(1 for r in results if r.success)
    error_count = len(results) - success_count

    return TickerResult(
        ticker=ticker,
        tier=TIER_LIGHT,
        results=results,
        success_count=success_count,
        error_count=error_count,
    )
