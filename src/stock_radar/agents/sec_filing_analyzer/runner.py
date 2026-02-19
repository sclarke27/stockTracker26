"""Runner for the SEC Filing Pattern Analyzer agent."""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client
from loguru import logger

from stock_radar.agents.models import AgentOutput
from stock_radar.agents.sec_filing_analyzer.agent import SecFilingAnalyzerAgent
from stock_radar.agents.sec_filing_analyzer.config import (
    LOOKBACK_DAYS,
    MAX_FILINGS,
    MAX_INSIDER_TRANSACTIONS,
)
from stock_radar.agents.sec_filing_analyzer.models import SecFilingInput
from stock_radar.config.loader import load_settings
from stock_radar.config.settings import AppSettings
from stock_radar.llm.factory import create_anthropic_client, create_ollama_client
from stock_radar.mcp_servers.predictions_db.server import (
    create_server as create_predictions_server,
)
from stock_radar.mcp_servers.sec_edgar.server import create_server as create_sec_edgar_server
from stock_radar.mcp_servers.vector_store.server import (
    create_server as create_vector_store_server,
)
from stock_radar.utils.logging import setup_logging
from stock_radar.utils.mcp import get_tool_text


def _load_settings() -> AppSettings:
    """Load application settings from config.

    Returns:
        Populated AppSettings instance.
    """
    return load_settings()


async def run_sec_filing_analyzer(
    ticker: str,
    quarter: int,
    year: int,
    settings: AppSettings | None = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> AgentOutput:
    """Run the SEC Filing Pattern Analyzer for a single ticker.

    Creates in-process MCP server instances, fetches recent filings and
    insider transactions, then runs the agent analysis pipeline.

    Args:
        ticker: Stock ticker symbol.
        quarter: Fiscal quarter (1-4) — used for prediction tracking.
        year: Fiscal year — used for prediction tracking.
        settings: Application settings, or None to load from config.
        lookback_days: Days of SEC filing history to analyze.

    Returns:
        Agent output with prediction and reasoning.
    """
    if settings is None:
        settings = _load_settings()

    sf_settings = settings.agents.sec_filing_analyzer

    # Create LLM clients
    ollama_model = sf_settings.ollama_model or settings.ollama.default_model
    ollama_client = create_ollama_client(settings.ollama, model=ollama_model)

    anthropic_client = None
    if settings.api_keys.anthropic:
        anthropic_client = create_anthropic_client(
            api_key=settings.api_keys.anthropic,
            model=sf_settings.anthropic_model,
        )

    # Create in-process MCP servers
    sec_edgar_server = create_sec_edgar_server()
    predictions_server = create_predictions_server()
    vector_store_server = create_vector_store_server()

    async with (
        Client(sec_edgar_server) as sec_client,
        Client(predictions_server) as predictions_client,
        Client(vector_store_server) as vector_store_client,
    ):
        # Fetch recent SEC filings
        filings_result = await sec_client.call_tool(
            "get_filings",
            {
                "ticker": ticker,
                "form_types": ["8-K", "10-K", "10-Q", "S-1", "SC 13D"],
                "limit": MAX_FILINGS,
            },
        )
        filings_data = json.loads(get_tool_text(filings_result))
        recent_filings: list[dict] = filings_data.get("filings", [])

        # Fetch insider transactions (non-fatal if unavailable)
        insider_transactions: list[dict] = []
        try:
            insider_result = await sec_client.call_tool(
                "get_insider_transactions",
                {"ticker": ticker, "limit": MAX_INSIDER_TRANSACTIONS},
            )
            insider_data = json.loads(get_tool_text(insider_result))
            insider_transactions = insider_data.get("transactions", [])
        except Exception:
            logger.warning(
                "Could not fetch insider transactions (non-fatal)",
                ticker=ticker,
            )

        input_data = SecFilingInput(
            ticker=ticker,
            quarter=quarter,
            year=year,
            recent_filings=recent_filings,
            insider_transactions=insider_transactions,
            filing_count=len(recent_filings),
            insider_transaction_count=len(insider_transactions),
            lookback_days=lookback_days,
        )

        agent = SecFilingAnalyzerAgent(settings=sf_settings)
        return await agent.run(
            input_data,
            ollama_client,
            anthropic_client,
            predictions_client,
            vector_store_client,
        )


async def run_batch(
    tickers: list[str],
    quarter: int,
    year: int,
    settings: AppSettings | None = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> list[AgentOutput]:
    """Run the SEC Filing Pattern Analyzer for multiple tickers.

    Processes tickers sequentially to respect SEC EDGAR rate limits.

    Args:
        tickers: List of stock ticker symbols.
        quarter: Fiscal quarter.
        year: Fiscal year.
        settings: Application settings, or None to load from config.
        lookback_days: Days of filing history to analyze.

    Returns:
        List of agent outputs (one per ticker, skipping failures).
    """
    if settings is None:
        settings = _load_settings()

    results: list[AgentOutput] = []
    for ticker in tickers:
        try:
            output = await run_sec_filing_analyzer(
                ticker, quarter, year, settings=settings, lookback_days=lookback_days
            )
            results.append(output)
        except Exception:
            logger.error(
                "SEC Filing Analyzer failed for ticker",
                ticker=ticker,
                quarter=quarter,
                year=year,
            )

    return results


def main() -> None:
    """CLI entry point for the SEC Filing Pattern Analyzer agent."""
    setup_logging()
    result = asyncio.run(run_sec_filing_analyzer("AAPL", 1, 2025))
    logger.info(
        "Result",
        prediction_id=result.prediction_id,
        direction=result.result.direction,
    )
