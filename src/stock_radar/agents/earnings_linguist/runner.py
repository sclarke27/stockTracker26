"""Runner for the Earnings Linguist agent."""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client
from loguru import logger

from stock_radar.agents.earnings_linguist.agent import EarningsLinguistAgent
from stock_radar.agents.earnings_linguist.models import EarningsLinguistInput
from stock_radar.agents.exceptions import TranscriptNotFoundError
from stock_radar.agents.models import AgentOutput
from stock_radar.config.loader import load_config
from stock_radar.config.settings import AppSettings
from stock_radar.llm.factory import create_anthropic_client, create_ollama_client
from stock_radar.mcp_servers.market_data.server import create_server as create_market_server
from stock_radar.mcp_servers.predictions_db.server import create_server as create_predictions_server
from stock_radar.mcp_servers.vector_store.server import create_server as create_vector_store_server
from stock_radar.utils.logging import setup_logging


def _load_settings() -> AppSettings:
    """Load application settings from config.

    Returns:
        Populated AppSettings instance.
    """
    config = load_config()
    return AppSettings(**config)


async def run_earnings_linguist(
    ticker: str,
    quarter: int,
    year: int,
    settings: AppSettings | None = None,
    prior_quarter: tuple[int, int] | None = None,
) -> AgentOutput:
    """Run the Earnings Linguist agent for a single ticker.

    Creates in-process MCP server instances, fetches the transcript,
    and runs the agent analysis pipeline.

    Args:
        ticker: Stock ticker symbol.
        quarter: Fiscal quarter (1-4).
        year: Fiscal year.
        settings: Application settings, or None to load from config.
        prior_quarter: Optional (quarter, year) tuple for prior comparison.

    Returns:
        Agent output with prediction and reasoning.

    Raises:
        TranscriptNotFoundError: If the transcript is not available.
    """
    if settings is None:
        settings = _load_settings()

    el_settings = settings.agents.earnings_linguist

    # Create LLM clients
    ollama_model = el_settings.ollama_model or settings.ollama.default_model
    ollama_client = create_ollama_client(settings.ollama, model=ollama_model)

    anthropic_client = None
    if settings.api_keys.anthropic:
        anthropic_client = create_anthropic_client(
            api_key=settings.api_keys.anthropic,
            model=el_settings.anthropic_model,
        )

    # Create in-process MCP servers
    market_server = create_market_server()
    predictions_server = create_predictions_server()
    vector_store_server = create_vector_store_server()

    async with (
        Client(market_server) as market_client,
        Client(predictions_server) as predictions_client,
        Client(vector_store_server) as vector_store_client,
    ):
        # Fetch current quarter transcript
        try:
            result = await market_client.call_tool(
                "get_earnings_transcript",
                {"ticker": ticker, "quarter": quarter, "year": year},
            )
            transcript_data = json.loads(result.content[0].text)
            transcript_content = transcript_data["content"]
        except Exception as exc:
            raise TranscriptNotFoundError(
                f"Failed to fetch transcript for {ticker} Q{quarter} {year}: {exc}"
            ) from exc

        # Fetch company name from company info (best-effort)
        company_name = ""
        try:
            info_result = await market_client.call_tool("get_company_info", {"ticker": ticker})
            info_data = json.loads(info_result.content[0].text)
            company_name = info_data.get("name", "")
        except Exception:
            logger.warning("Could not fetch company name", ticker=ticker)

        # Optionally fetch prior quarter transcript
        prior_content = None
        if prior_quarter:
            try:
                prior_result = await market_client.call_tool(
                    "get_earnings_transcript",
                    {
                        "ticker": ticker,
                        "quarter": prior_quarter[0],
                        "year": prior_quarter[1],
                    },
                )
                prior_data = json.loads(prior_result.content[0].text)
                prior_content = prior_data["content"]
            except Exception:
                logger.warning(
                    "Prior quarter transcript not available",
                    ticker=ticker,
                    quarter=prior_quarter[0],
                    year=prior_quarter[1],
                )

        input_data = EarningsLinguistInput(
            ticker=ticker,
            quarter=quarter,
            year=year,
            transcript_content=transcript_content,
            prior_transcript_content=prior_content,
            company_name=company_name,
        )

        agent = EarningsLinguistAgent(settings=el_settings)
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
) -> list[AgentOutput]:
    """Run the Earnings Linguist agent for multiple tickers.

    Processes tickers sequentially to respect rate limits.

    Args:
        tickers: List of stock ticker symbols.
        quarter: Fiscal quarter.
        year: Fiscal year.
        settings: Application settings, or None to load from config.

    Returns:
        List of agent outputs (one per ticker, skipping failures).
    """
    if settings is None:
        settings = _load_settings()

    results: list[AgentOutput] = []
    for ticker in tickers:
        try:
            output = await run_earnings_linguist(ticker, quarter, year, settings=settings)
            results.append(output)
        except Exception:
            logger.error(
                "Earnings Linguist failed for ticker",
                ticker=ticker,
                quarter=quarter,
                year=year,
            )

    return results


def main() -> None:
    """CLI entry point for the Earnings Linguist agent."""
    setup_logging()
    result = asyncio.run(run_earnings_linguist("AAPL", 4, 2024))
    logger.info(
        "Result",
        prediction_id=result.prediction_id,
        direction=result.result.direction,
    )
