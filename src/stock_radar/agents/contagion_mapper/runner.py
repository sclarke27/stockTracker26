"""Runner for the Cross-Sector Contagion Mapper agent."""

from __future__ import annotations

import asyncio
import json
from typing import Literal

from fastmcp import Client
from loguru import logger

from stock_radar.agents.contagion_mapper.agent import ContagionMapperAgent
from stock_radar.agents.contagion_mapper.config import MAX_TARGET_NEWS, MAX_TRIGGER_NEWS
from stock_radar.agents.contagion_mapper.models import ContagionInput
from stock_radar.agents.models import AgentOutput
from stock_radar.config.loader import load_settings
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

RelationshipType = Literal[
    "supplier",
    "customer",
    "competitor",
    "same_sector",
    "distribution_partner",
]


def _load_settings() -> AppSettings:
    """Load application settings from config.

    Returns:
        Populated AppSettings instance.
    """
    return load_settings()


async def run_contagion_mapper(
    trigger_ticker: str,
    target_ticker: str,
    relationship_type: RelationshipType,
    quarter: int,
    year: int,
    settings: AppSettings | None = None,
) -> AgentOutput:
    """Run the Cross-Sector Contagion Mapper for a trigger→target pair.

    Creates in-process MCP server instances, fetches company info and
    recent news for both companies, then runs the agent analysis pipeline.

    Args:
        trigger_ticker: Ticker of the company that had the trigger event.
        target_ticker: Ticker of the company being assessed for impact.
        relationship_type: Nature of relationship between the two companies.
        quarter: Fiscal quarter (1-4) — used for prediction tracking.
        year: Fiscal year — used for prediction tracking.
        settings: Application settings, or None to load from config.

    Returns:
        Agent output with prediction and reasoning. The prediction is logged
        under the target ticker (the company being impacted).
    """
    if settings is None:
        settings = _load_settings()

    cm_settings = settings.agents.contagion_mapper

    # Create LLM clients
    ollama_model = cm_settings.ollama_model or settings.ollama.default_model
    ollama_client = create_ollama_client(settings.ollama, model=ollama_model)

    anthropic_client = None
    if settings.api_keys.anthropic:
        anthropic_client = create_anthropic_client(
            api_key=settings.api_keys.anthropic,
            model=cm_settings.anthropic_model,
        )

    # Create in-process MCP servers
    market_server = create_market_server()
    news_feed_server = create_news_feed_server()
    predictions_server = create_predictions_server()
    vector_store_server = create_vector_store_server()

    async with (
        Client(market_server) as market_client,
        Client(news_feed_server) as news_client,
        Client(predictions_server) as predictions_client,
        Client(vector_store_server) as vector_store_client,
    ):
        # Fetch company info for both trigger and target (best-effort)
        trigger_company_name = trigger_ticker
        target_company_name = target_ticker
        trigger_sector = ""
        target_sector = ""

        try:
            trigger_info_result = await market_client.call_tool(
                "get_company_info", {"ticker": trigger_ticker}
            )
            trigger_info = json.loads(get_tool_text(trigger_info_result))
            trigger_company_name = trigger_info.get("name", trigger_ticker)
            trigger_sector = trigger_info.get("sector", "")
        except Exception:
            logger.warning(
                "Could not fetch trigger company info (non-fatal)",
                ticker=trigger_ticker,
            )

        try:
            target_info_result = await market_client.call_tool(
                "get_company_info", {"ticker": target_ticker}
            )
            target_info = json.loads(get_tool_text(target_info_result))
            target_company_name = target_info.get("name", target_ticker)
            target_sector = target_info.get("sector", "")
        except Exception:
            logger.warning(
                "Could not fetch target company info (non-fatal)",
                ticker=target_ticker,
            )

        # Fetch trigger event news (best-effort)
        trigger_recent_news: list[dict] = []
        try:
            trigger_news_result = await news_client.call_tool(
                "search_news",
                {"query": trigger_ticker, "limit": MAX_TRIGGER_NEWS},
            )
            trigger_news_data = json.loads(get_tool_text(trigger_news_result))
            trigger_recent_news = trigger_news_data.get("articles", [])[:MAX_TRIGGER_NEWS]
        except Exception:
            logger.warning(
                "Could not fetch trigger news (non-fatal)",
                ticker=trigger_ticker,
            )

        # Fetch target recent news (best-effort)
        target_recent_news: list[dict] = []
        try:
            target_news_result = await news_client.call_tool(
                "get_news",
                {"ticker": target_ticker, "limit": MAX_TARGET_NEWS},
            )
            target_news_data = json.loads(get_tool_text(target_news_result))
            target_recent_news = target_news_data.get("articles", [])[:MAX_TARGET_NEWS]
        except Exception:
            logger.warning(
                "Could not fetch target news (non-fatal)",
                ticker=target_ticker,
            )

        # Summarize trigger event from news (use the most recent headline as summary)
        trigger_event_summary = (
            trigger_recent_news[0].get("summary", f"{trigger_ticker} experienced a market event.")
            if trigger_recent_news
            else f"{trigger_ticker} experienced a market event."
        )

        input_data = ContagionInput(
            ticker=target_ticker,  # target is the prediction subject
            quarter=quarter,
            year=year,
            trigger_ticker=trigger_ticker,
            trigger_company_name=trigger_company_name,
            trigger_event_summary=trigger_event_summary,
            target_company_name=target_company_name,
            relationship_type=relationship_type,
            trigger_recent_news=trigger_recent_news,
            target_recent_news=target_recent_news,
            trigger_sector=trigger_sector,
            target_sector=target_sector,
        )

        agent = ContagionMapperAgent(settings=cm_settings)
        return await agent.run(
            input_data,
            ollama_client,
            anthropic_client,
            predictions_client,
            vector_store_client,
        )


async def run_batch(
    pairs: list[tuple[str, str, RelationshipType]],
    quarter: int,
    year: int,
    settings: AppSettings | None = None,
) -> list[AgentOutput]:
    """Run the Contagion Mapper for multiple trigger→target pairs.

    Processes pairs sequentially to respect rate limits.

    Args:
        pairs: List of (trigger_ticker, target_ticker, relationship_type) tuples.
        quarter: Fiscal quarter.
        year: Fiscal year.
        settings: Application settings, or None to load from config.

    Returns:
        List of agent outputs (one per pair, skipping failures).
    """
    if settings is None:
        settings = _load_settings()

    results: list[AgentOutput] = []
    for trigger, target, rel_type in pairs:
        try:
            output = await run_contagion_mapper(
                trigger, target, rel_type, quarter, year, settings=settings
            )
            results.append(output)
        except Exception:
            logger.error(
                "Contagion Mapper failed for pair",
                trigger=trigger,
                target=target,
                relationship_type=rel_type,
            )

    return results


def main() -> None:
    """CLI entry point for the Cross-Sector Contagion Mapper agent."""
    setup_logging()
    result = asyncio.run(
        run_contagion_mapper(
            trigger_ticker="NVDA",
            target_ticker="AMD",
            relationship_type="competitor",
            quarter=1,
            year=2025,
        )
    )
    logger.info(
        "Result",
        prediction_id=result.prediction_id,
        direction=result.result.direction,
    )
