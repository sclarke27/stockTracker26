"""Dashboard MCP server — tools, lifespan, and response models."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastmcp import Client, Context, FastMCP
from loguru import logger
from pydantic import BaseModel

from stock_radar.mcp_servers.dashboard.config import (
    ACCURACY_LOOKBACK_DAYS,
    ACTIVE_SIGNALS_LIMIT,
    MIN_SIGNAL_CONFIDENCE,
    RECENT_PREDICTIONS_LIMIT,
    SERVER_NAME,
)
from stock_radar.mcp_servers.market_data.server import create_server as create_market_server
from stock_radar.mcp_servers.news_feed.server import create_server as create_news_feed_server
from stock_radar.mcp_servers.predictions_db.server import (
    create_server as create_predictions_server,
)
from stock_radar.models.predictions import Direction
from stock_radar.utils.mcp import get_tool_text

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SignalSummary(BaseModel):
    """Summary of a single active prediction signal."""

    prediction_id: str
    ticker: str
    agent_name: str
    signal_type: str
    direction: Direction
    confidence: float
    reasoning: str
    horizon_days: int
    prediction_date: str
    current_price: float | None = None


class ActiveSignalsResponse(BaseModel):
    """Response for get_active_signals."""

    signals: list[SignalSummary]
    total_count: int


class WatchlistEntry(BaseModel):
    """A single ticker entry in the watchlist."""

    ticker: str
    company_name: str
    current_price: float | None = None
    active_signal_count: int
    latest_direction: Direction | None = None


class WatchlistResponse(BaseModel):
    """Response for get_watchlist."""

    entries: list[WatchlistEntry]


class AgentStatus(BaseModel):
    """Performance summary for one agent."""

    agent_name: str
    signal_type: str
    total_predictions: int
    scored: int
    accuracy_pct: float | None = None
    avg_confidence: float | None = None


class AgentStatusResponse(BaseModel):
    """Response for get_agent_status."""

    agents: list[AgentStatus]
    as_of_days: int


class PredictionDetailResponse(BaseModel):
    """Response for get_prediction_detail."""

    prediction: dict | None
    price_history: list[dict]


class TickerSummaryResponse(BaseModel):
    """Aggregated summary for a single ticker."""

    ticker: str
    company_name: str
    current_price: float | None = None
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    recent_predictions: list[dict]


# ---------------------------------------------------------------------------
# Server dependencies
# ---------------------------------------------------------------------------


@dataclass
class ServerDeps:
    """Lifespan-managed MCP clients shared across all tool invocations."""

    predictions_client: Client
    market_client: Client
    news_client: Client


def _deps(ctx: Context) -> ServerDeps:
    """Extract server dependencies from the tool context."""
    return ctx.fastmcp._lifespan_result  # type: ignore[attr-defined,return-value]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ServerDeps]:
    """Initialize in-process MCP clients and hold them for the server lifetime.

    Creates three in-process FastMCP servers (predictions-db, market-data,
    news-feed) and opens Client connections to each. The connections persist
    for the lifetime of the HTTP server — not created per-request.
    """
    predictions_server = create_predictions_server()
    market_server = create_market_server()
    news_feed_server = create_news_feed_server()

    async with (
        Client(predictions_server) as predictions_client,
        Client(market_server) as market_client,
        Client(news_feed_server) as news_client,
    ):
        logger.info("Dashboard MCP server started", server=SERVER_NAME)
        yield ServerDeps(
            predictions_client=predictions_client,
            market_client=market_client,
            news_client=news_client,
        )

    logger.info("Dashboard MCP server stopped", server=SERVER_NAME)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


async def get_active_signals(ctx: Context, limit: int = ACTIVE_SIGNALS_LIMIT) -> str:
    """Get active (unscored) prediction signals above the confidence threshold.

    Fetches unscored predictions from the predictions DB, filters to those
    with confidence >= MIN_SIGNAL_CONFIDENCE, and enriches each with the
    current quote price from market data (best-effort).

    Args:
        ctx: FastMCP tool context.
        limit: Maximum number of signals to return.

    Returns:
        JSON-serialized ActiveSignalsResponse.
    """
    deps = _deps(ctx)

    raw = await deps.predictions_client.call_tool(
        "get_prediction_history",
        {"limit": limit, "scored_only": False},
    )
    history = json.loads(get_tool_text(raw))
    predictions = history.get("predictions", [])

    # Filter by confidence threshold
    predictions = [p for p in predictions if p.get("confidence", 0.0) >= MIN_SIGNAL_CONFIDENCE]

    signals: list[SignalSummary] = []
    for pred in predictions:
        ticker = pred.get("ticker", "")
        current_price: float | None = None
        try:
            quote_raw = await deps.market_client.call_tool("get_quote", {"ticker": ticker})
            quote = json.loads(get_tool_text(quote_raw))
            current_price = quote.get("price")
        except Exception:
            logger.debug("Could not fetch quote for signal (non-fatal)", ticker=ticker)

        signals.append(
            SignalSummary(
                prediction_id=pred.get("id", ""),
                ticker=ticker,
                agent_name=pred.get("agent_name", ""),
                signal_type=pred.get("signal_type", ""),
                direction=pred.get("direction", "NEUTRAL"),
                confidence=pred.get("confidence", 0.0),
                reasoning=pred.get("reasoning", ""),
                horizon_days=pred.get("horizon_days", 0),
                prediction_date=pred.get("prediction_date", ""),
                current_price=current_price,
            )
        )

    response = ActiveSignalsResponse(signals=signals, total_count=len(signals))
    return response.model_dump_json()


async def get_watchlist(ctx: Context) -> str:
    """Get the watchlist of tickers with active predictions.

    Retrieves all unscored predictions, deduplicates by ticker, and fetches
    current quote and company info for each (best-effort). The latest
    direction and signal count are derived from the predictions.

    Args:
        ctx: FastMCP tool context.

    Returns:
        JSON-serialized WatchlistResponse.
    """
    deps = _deps(ctx)

    raw = await deps.predictions_client.call_tool(
        "get_prediction_history",
        {"limit": RECENT_PREDICTIONS_LIMIT, "scored_only": False},
    )
    history = json.loads(get_tool_text(raw))
    predictions = history.get("predictions", [])

    # Group by ticker preserving insertion order
    ticker_preds: dict[str, list[dict]] = {}
    for pred in predictions:
        ticker = pred.get("ticker", "")
        if ticker not in ticker_preds:
            ticker_preds[ticker] = []
        ticker_preds[ticker].append(pred)

    entries: list[WatchlistEntry] = []
    for ticker, preds in ticker_preds.items():
        current_price: float | None = None
        company_name: str = ticker  # fallback

        try:
            quote_raw = await deps.market_client.call_tool("get_quote", {"ticker": ticker})
            quote = json.loads(get_tool_text(quote_raw))
            current_price = quote.get("price")
        except Exception:
            logger.debug("Could not fetch quote for watchlist entry (non-fatal)", ticker=ticker)

        try:
            info_raw = await deps.market_client.call_tool("get_company_info", {"ticker": ticker})
            info = json.loads(get_tool_text(info_raw))
            company_name = info.get("name", ticker)
        except Exception:
            logger.debug(
                "Could not fetch company info for watchlist entry (non-fatal)", ticker=ticker
            )

        # Latest direction = most recent prediction's direction
        latest_direction: Direction | None = preds[0].get("direction") if preds else None

        entries.append(
            WatchlistEntry(
                ticker=ticker,
                company_name=company_name,
                current_price=current_price,
                active_signal_count=len(preds),
                latest_direction=latest_direction,
            )
        )

    response = WatchlistResponse(entries=entries)
    return response.model_dump_json()


async def get_agent_status(ctx: Context) -> str:
    """Get performance statistics for all analysis agents.

    Calls the predictions DB accuracy endpoint with a fixed lookback window
    and maps the per-agent stats to a dashboard-friendly shape.

    Args:
        ctx: FastMCP tool context.

    Returns:
        JSON-serialized AgentStatusResponse.
    """
    deps = _deps(ctx)

    raw = await deps.predictions_client.call_tool(
        "get_agent_accuracy",
        {"days_lookback": ACCURACY_LOOKBACK_DAYS},
    )
    accuracy = json.loads(get_tool_text(raw))
    agent_stats = accuracy.get("agent_stats", [])

    agents: list[AgentStatus] = []
    for stats in agent_stats:
        agents.append(
            AgentStatus(
                agent_name=stats.get("agent_name", ""),
                signal_type=stats.get("signal_type", ""),
                total_predictions=stats.get("total", 0),
                scored=stats.get("scored", 0),
                accuracy_pct=stats.get("accuracy_pct"),
                avg_confidence=stats.get("avg_confidence"),
            )
        )

    response = AgentStatusResponse(agents=agents, as_of_days=ACCURACY_LOOKBACK_DAYS)
    return response.model_dump_json()


async def get_prediction_detail(ctx: Context, prediction_id: str) -> str:
    """Get full detail for a single prediction including price context.

    Fetches the prediction record by ID and enriches it with recent price
    history for the ticker (best-effort, for chart rendering).

    Args:
        ctx: FastMCP tool context.
        prediction_id: The unique prediction ID to look up.

    Returns:
        JSON-serialized PredictionDetailResponse.
    """
    deps = _deps(ctx)

    raw = await deps.predictions_client.call_tool(
        "get_prediction_history",
        {"limit": RECENT_PREDICTIONS_LIMIT},
    )
    history = json.loads(get_tool_text(raw))
    predictions = history.get("predictions", [])

    # Find the specific prediction
    matched: dict | None = next(
        (p for p in predictions if p.get("id") == prediction_id),
        None,
    )

    price_history: list[dict] = []
    if matched:
        ticker = matched.get("ticker", "")
        try:
            ph_raw = await deps.market_client.call_tool(
                "get_price_history",
                {"ticker": ticker, "outputsize": "compact"},
            )
            ph_data = json.loads(get_tool_text(ph_raw))
            price_history = ph_data.get("prices", [])
        except Exception:
            logger.debug(
                "Could not fetch price history for prediction detail (non-fatal)",
                ticker=ticker,
            )

    response = PredictionDetailResponse(prediction=matched, price_history=price_history)
    return response.model_dump_json()


async def get_ticker_summary(ctx: Context, ticker: str) -> str:
    """Get an aggregated summary for a single ticker.

    Combines company info, current quote, recent predictions, and news
    sentiment into a single response suitable for a detail view.

    Args:
        ctx: FastMCP tool context.
        ticker: Stock ticker symbol to summarize.

    Returns:
        JSON-serialized TickerSummaryResponse.
    """
    deps = _deps(ctx)

    # Company info (best-effort)
    company_name: str = ticker
    try:
        info_raw = await deps.market_client.call_tool("get_company_info", {"ticker": ticker})
        info = json.loads(get_tool_text(info_raw))
        company_name = info.get("name", ticker)
    except Exception:
        logger.debug("Could not fetch company info for ticker summary (non-fatal)", ticker=ticker)

    # Current quote (best-effort)
    current_price: float | None = None
    try:
        quote_raw = await deps.market_client.call_tool("get_quote", {"ticker": ticker})
        quote = json.loads(get_tool_text(quote_raw))
        current_price = quote.get("price")
    except Exception:
        logger.debug("Could not fetch quote for ticker summary (non-fatal)", ticker=ticker)

    # Recent predictions for ticker
    pred_raw = await deps.predictions_client.call_tool(
        "get_prediction_history",
        {"ticker": ticker, "limit": RECENT_PREDICTIONS_LIMIT},
    )
    pred_history = json.loads(get_tool_text(pred_raw))
    recent_predictions = pred_history.get("predictions", [])

    # Sentiment summary (best-effort)
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    try:
        sent_raw = await deps.news_client.call_tool(
            "get_sentiment_summary",
            {"ticker": ticker},
        )
        sentiment = json.loads(get_tool_text(sent_raw))
        sentiment_score = sentiment.get("average_sentiment_score")
        sentiment_label = sentiment.get("average_sentiment_label")
    except Exception:
        logger.debug("Could not fetch sentiment for ticker summary (non-fatal)", ticker=ticker)

    response = TickerSummaryResponse(
        ticker=ticker,
        company_name=company_name,
        current_price=current_price,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        recent_predictions=recent_predictions,
    )
    return response.model_dump_json()


# ---------------------------------------------------------------------------
# Tool registration list
# ---------------------------------------------------------------------------

TOOLS = [
    get_active_signals,
    get_watchlist,
    get_agent_status,
    get_prediction_detail,
    get_ticker_summary,
]


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server(name: str = SERVER_NAME) -> FastMCP:
    """Create and return a fully-wired dashboard FastMCP server instance.

    Returns a fresh instance each time — useful for testing where each
    test needs its own isolated server with independent lifespan state.
    The server uses HTTP transport (not stdio) and manages in-process
    client connections via lifespan.

    Args:
        name: Server name (defaults to SERVER_NAME constant).

    Returns:
        Configured FastMCP instance with all tools registered.
    """
    server = FastMCP(name, lifespan=lifespan)
    for tool_fn in TOOLS:
        server.tool()(tool_fn)  # type: ignore[arg-type]
    return server


# Module-level server instance for production use (stdio / HTTP run).
_server = create_server()
