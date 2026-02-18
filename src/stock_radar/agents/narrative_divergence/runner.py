"""Runner for the Narrative vs Price Divergence agent."""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client
from loguru import logger

from stock_radar.agents.models import AgentOutput
from stock_radar.agents.narrative_divergence.agent import NarrativeDivergenceAgent
from stock_radar.agents.narrative_divergence.config import MAX_TOP_ARTICLES
from stock_radar.agents.narrative_divergence.models import NarrativeDivergenceInput
from stock_radar.config.loader import load_config
from stock_radar.config.settings import AppSettings
from stock_radar.llm.factory import create_anthropic_client, create_ollama_client
from stock_radar.mcp_servers.market_data.server import create_server as create_market_server
from stock_radar.mcp_servers.news_feed.server import create_server as create_news_feed_server
from stock_radar.mcp_servers.predictions_db.server import (
    create_server as create_predictions_server,
)
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
    config = load_config()
    return AppSettings(**config)


def _compute_price_return(prices: list[dict], lookback_days: int) -> float:
    """Compute price return over a lookback window from a price history list.

    Prices are assumed to be ordered most-recent-first.

    Args:
        prices: List of price dicts with a 'close' key.
        lookback_days: Number of days to look back.

    Returns:
        Return as a decimal (e.g. -0.08 for -8%), or 0.0 if insufficient data.
    """
    if len(prices) < lookback_days + 1:
        return 0.0

    current_price = prices[0].get("close", 0.0)
    past_price = prices[lookback_days].get("close", 0.0)

    if past_price == 0.0:
        return 0.0

    return (current_price - past_price) / past_price


async def run_narrative_divergence(
    ticker: str,
    quarter: int,
    year: int,
    settings: AppSettings | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
) -> AgentOutput:
    """Run the Narrative vs Price Divergence agent for a single ticker.

    Creates in-process MCP server instances, fetches news sentiment and
    price history, then runs the agent analysis pipeline.

    Args:
        ticker: Stock ticker symbol.
        quarter: Fiscal quarter (1-4) — used for prediction tracking.
        year: Fiscal year — used for prediction tracking.
        settings: Application settings, or None to load from config.
        time_from: Start of news sentiment window (AV format 'YYYYMMDDTHHMM').
        time_to: End of news sentiment window (AV format 'YYYYMMDDTHHMM').

    Returns:
        Agent output with prediction and reasoning.
    """
    if settings is None:
        settings = _load_settings()

    nd_settings = settings.agents.narrative_divergence

    # Create LLM clients
    ollama_model = nd_settings.ollama_model or settings.ollama.default_model
    ollama_client = create_ollama_client(settings.ollama, model=ollama_model)

    anthropic_client = None
    if settings.api_keys.anthropic:
        anthropic_client = create_anthropic_client(
            api_key=settings.api_keys.anthropic,
            model=nd_settings.anthropic_model,
        )

    # Create in-process MCP servers
    news_feed_server = create_news_feed_server()
    market_server = create_market_server()
    predictions_server = create_predictions_server()
    vector_store_server = create_vector_store_server()

    async with (
        Client(news_feed_server) as news_client,
        Client(market_server) as market_client,
        Client(predictions_server) as predictions_client,
        Client(vector_store_server) as vector_store_client,
    ):
        # Fetch sentiment summary
        sentiment_kwargs: dict = {"ticker": ticker}
        if time_from:
            sentiment_kwargs["time_from"] = time_from
        if time_to:
            sentiment_kwargs["time_to"] = time_to

        sentiment_result = await news_client.call_tool("get_sentiment_summary", sentiment_kwargs)
        sentiment_data = json.loads(get_tool_text(sentiment_result))

        article_count: int = sentiment_data.get("article_count", 0)
        sentiment_score: float = sentiment_data.get("average_sentiment_score", 0.0)
        average_sentiment_label: str = sentiment_data.get("average_sentiment_label", "Neutral")

        # Fetch top articles for LLM context (best-effort)
        top_articles: list[dict] = []
        try:
            news_kwargs: dict = {"ticker": ticker, "limit": MAX_TOP_ARTICLES}
            if time_from:
                news_kwargs["time_from"] = time_from
            news_result = await news_client.call_tool("get_news", news_kwargs)
            news_data = json.loads(get_tool_text(news_result))
            top_articles = news_data.get("articles", [])[:MAX_TOP_ARTICLES]
        except Exception:
            logger.warning("Could not fetch top articles for context", ticker=ticker)

        # Fetch price history and compute returns
        price_return_30d = 0.0
        price_return_7d = 0.0
        try:
            price_result = await market_client.call_tool(
                "get_price_history",
                {"ticker": ticker, "outputsize": "compact"},
            )
            price_data = json.loads(get_tool_text(price_result))
            prices: list[dict] = price_data.get("prices", [])
            price_return_30d = _compute_price_return(prices, lookback_days=30)
            price_return_7d = _compute_price_return(prices, lookback_days=7)
        except Exception:
            logger.warning("Could not fetch price history for returns", ticker=ticker)

        input_data = NarrativeDivergenceInput(
            ticker=ticker,
            quarter=quarter,
            year=year,
            sentiment_score=sentiment_score,
            article_count=article_count,
            average_sentiment_label=average_sentiment_label,
            price_return_30d=price_return_30d,
            price_return_7d=price_return_7d,
            top_articles=top_articles,
            time_from=time_from,
            time_to=time_to,
        )

        agent = NarrativeDivergenceAgent(settings=nd_settings)
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
    time_from: str | None = None,
    time_to: str | None = None,
) -> list[AgentOutput]:
    """Run the Narrative vs Price Divergence agent for multiple tickers.

    Processes tickers sequentially to respect API rate limits.

    Args:
        tickers: List of stock ticker symbols.
        quarter: Fiscal quarter.
        year: Fiscal year.
        settings: Application settings, or None to load from config.
        time_from: Start of news sentiment window.
        time_to: End of news sentiment window.

    Returns:
        List of agent outputs (one per ticker, skipping failures).
    """
    if settings is None:
        settings = _load_settings()

    results: list[AgentOutput] = []
    for ticker in tickers:
        try:
            output = await run_narrative_divergence(
                ticker, quarter, year, settings=settings, time_from=time_from, time_to=time_to
            )
            results.append(output)
        except Exception:
            logger.error(
                "Narrative Divergence agent failed for ticker",
                ticker=ticker,
                quarter=quarter,
                year=year,
            )

    return results


def main() -> None:
    """CLI entry point for the Narrative vs Price Divergence agent."""
    setup_logging()
    result = asyncio.run(run_narrative_divergence("AAPL", 1, 2025))
    logger.info(
        "Result",
        prediction_id=result.prediction_id,
        direction=result.result.direction,
    )
