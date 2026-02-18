"""Pipeline runner -- orchestrates data ingestion across all watchlist tickers."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastmcp import Client
from loguru import logger

from stock_radar.mcp_servers.market_data.server import create_server as create_market_server
from stock_radar.mcp_servers.sec_edgar.server import create_server as create_edgar_server
from stock_radar.pipeline.fetchers import fetch_deep, fetch_light
from stock_radar.pipeline.models import PipelineResult, TickerResult
from stock_radar.pipeline.quarter import current_quarter
from stock_radar.pipeline.watchlist import Watchlist, load_watchlist
from stock_radar.utils.logging import setup_logging


async def run_pipeline(watchlist: Watchlist | None = None) -> PipelineResult:
    """Run the full data ingestion pipeline.

    Creates in-process MCP server instances and iterates through all
    watchlist tickers, fetching data per coverage tier.

    Args:
        watchlist: Pre-loaded watchlist, or None to load from default path.

    Returns:
        PipelineResult summarizing the entire run.
    """
    setup_logging()
    started_at = datetime.now(UTC).isoformat()

    if watchlist is None:
        watchlist = load_watchlist()

    quarter, year = current_quarter()
    logger.info(
        "Pipeline starting",
        deep_count=len(watchlist.deep),
        light_count=len(watchlist.light),
        quarter=quarter,
        year=year,
    )

    ticker_results: list[TickerResult] = []

    market_server = create_market_server()
    edgar_server = create_edgar_server()

    async with Client(market_server) as market_client, Client(edgar_server) as edgar_client:
        # Deep-tier tickers
        for entry in watchlist.deep:
            logger.info("Fetching deep-tier data", ticker=entry.symbol)
            result = await fetch_deep(market_client, edgar_client, entry.symbol, quarter, year)
            ticker_results.append(result)

        # Light-tier tickers
        for entry in watchlist.light:
            logger.info("Fetching light-tier data", ticker=entry.symbol)
            result = await fetch_light(market_client, edgar_client, entry.symbol)
            ticker_results.append(result)

    completed_at = datetime.now(UTC).isoformat()
    total_calls = sum(len(r.results) for r in ticker_results)
    total_errors = sum(r.error_count for r in ticker_results)

    # Compute actual duration from timestamps.
    start_dt = datetime.fromisoformat(started_at)
    end_dt = datetime.fromisoformat(completed_at)
    duration_seconds = round((end_dt - start_dt).total_seconds(), 2)

    pipeline_result = PipelineResult(
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        tickers_processed=len(ticker_results),
        total_calls=total_calls,
        total_errors=total_errors,
        ticker_results=ticker_results,
    )

    logger.info(
        "Pipeline complete",
        tickers=pipeline_result.tickers_processed,
        calls=total_calls,
        errors=total_errors,
        duration_s=pipeline_result.duration_seconds,
    )

    return pipeline_result


def main() -> None:
    """CLI entry point for the data ingestion pipeline."""
    asyncio.run(run_pipeline())
