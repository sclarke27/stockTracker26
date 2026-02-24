"""FastMCP server assembly for predictions database tools."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from fastmcp import Context, FastMCP
from loguru import logger

from stock_radar.mcp_servers.predictions_db.config import (
    DEFAULT_ACCURACY_LOOKBACK_DAYS,
    DEFAULT_QUERY_LIMIT,
    HORIZON_BUFFER_DAYS,
    NEUTRAL_RETURN_THRESHOLD_PCT,
    SERVER_NAME,
)
from stock_radar.mcp_servers.predictions_db.store import PredictionsStore
from stock_radar.models.predictions import (
    AgentAccuracyResponse,
    AgentStats,
    LogPredictionResponse,
    PendingScoringResponse,
    PredictionHistoryResponse,
    PredictionRecord,
    ScorePredictionResponse,
    ScoringStatus,
)
from stock_radar.utils.logging import setup_logging


@dataclass
class ServerDeps:
    """Shared dependencies initialized during server lifespan."""

    store: PredictionsStore


def _get_db_path() -> str:
    """Return the predictions database path from config.

    Extracted as a function so tests can patch it with a temporary path.
    """
    from stock_radar.config.loader import load_settings

    settings = load_settings()
    return settings.predictions.db_path


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ServerDeps]:
    """Initialize and tear down server dependencies.

    Creates the PredictionsStore backed by SQLite.
    Available to tools via ``ctx.fastmcp._lifespan_result``.
    """
    setup_logging()

    db_path = _get_db_path()

    store = PredictionsStore(db_path=db_path)
    await store.initialize()

    logger.info("Predictions DB MCP server started", server=SERVER_NAME)

    deps = ServerDeps(store=store)
    try:
        yield deps
    finally:
        await store.close()
        logger.info("Predictions DB MCP server stopped", server=SERVER_NAME)


def _deps(ctx: Context) -> ServerDeps:
    """Extract server dependencies from the tool context."""
    return ctx.fastmcp._lifespan_result  # type: ignore[attr-defined,return-value]


# ---------------------------------------------------------------------------
# Tool functions (plain async functions, registered on a server via
# ``create_server()`` or the module-level ``mcp`` instance).
# ---------------------------------------------------------------------------


async def log_prediction(
    ctx: Context,
    ticker: str,
    agent_name: str,
    signal_type: str,
    direction: str,
    confidence: float,
    reasoning: str,
    horizon_days: int,
    prediction_date: str | None = None,
) -> str:
    """Log a new prediction from an analysis agent.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).
        agent_name: Name of the agent generating the prediction.
        signal_type: Type of signal detected (e.g. earnings_sentiment).
        direction: Predicted direction (BULLISH, BEARISH, or NEUTRAL).
        confidence: Confidence score between 0.0 and 1.0.
        reasoning: Agent reasoning supporting the prediction.
        horizon_days: Number of days until the prediction is evaluated.
        prediction_date: Date of the prediction (YYYY-MM-DD). Defaults to today.
    """
    deps = _deps(ctx)

    prediction_id = str(uuid.uuid4())
    if prediction_date is None:
        prediction_date = date.today().isoformat()
    created_at = datetime.now(UTC).isoformat()

    await deps.store.insert(
        {
            "id": prediction_id,
            "ticker": ticker,
            "agent_name": agent_name,
            "signal_type": signal_type,
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
            "prediction_date": prediction_date,
            "horizon_days": horizon_days,
            "created_at": created_at,
        }
    )

    response = LogPredictionResponse(
        prediction_id=prediction_id,
        created_at=created_at,
    )
    return response.model_dump_json()


async def score_prediction(
    ctx: Context,
    prediction_id: str,
    actual_price_close: float,
    actual_price_at_horizon: float,
) -> str:
    """Score a prediction against actual price movement.

    Args:
        prediction_id: ID of the prediction to score.
        actual_price_close: Closing price on the prediction date.
        actual_price_at_horizon: Closing price at the horizon date.
    """
    deps = _deps(ctx)

    prediction = await deps.store.get_by_id(prediction_id)
    if prediction is None:
        raise ValueError(f"Prediction not found: {prediction_id}")

    return_pct = (actual_price_at_horizon / actual_price_close - 1) * 100
    direction = prediction["direction"]

    status: ScoringStatus
    if direction == "BULLISH" and return_pct > 0 or direction == "BEARISH" and return_pct < 0:
        status = "CORRECT"
    elif direction == "NEUTRAL" and abs(return_pct) < NEUTRAL_RETURN_THRESHOLD_PCT:
        status = "PARTIAL"
    else:
        status = "INCORRECT"

    scored_at = datetime.now(UTC).isoformat()

    await deps.store.update_score(
        prediction_id,
        scored_at,
        actual_price_close,
        actual_price_at_horizon,
        return_pct,
        status,
    )

    response = ScorePredictionResponse(
        prediction_id=prediction_id,
        status=status,
        return_pct=round(return_pct, 4),
        direction=direction,
        confidence=prediction["confidence"],
    )
    return response.model_dump_json()


async def get_prediction_history(
    ctx: Context,
    ticker: str | None = None,
    agent_name: str | None = None,
    signal_type: str | None = None,
    limit: int = DEFAULT_QUERY_LIMIT,
    offset: int = 0,
    scored_only: bool = False,
) -> str:
    """Get prediction history with optional filters and pagination.

    Args:
        ticker: Filter by stock ticker symbol.
        agent_name: Filter by agent name.
        signal_type: Filter by signal type.
        limit: Maximum number of records to return (default 50).
        offset: Number of records to skip for pagination.
        scored_only: If true, only return scored predictions.
    """
    deps = _deps(ctx)

    rows, total_count = await deps.store.query(
        ticker=ticker,
        agent_name=agent_name,
        signal_type=signal_type,
        scored_only=scored_only,
        limit=limit,
        offset=offset,
    )

    predictions = [PredictionRecord(**row) for row in rows]

    response = PredictionHistoryResponse(
        predictions=predictions,
        total_count=total_count,
    )
    return response.model_dump_json()


async def get_agent_accuracy(
    ctx: Context,
    agent_name: str | None = None,
    signal_type: str | None = None,
    days_lookback: int = DEFAULT_ACCURACY_LOOKBACK_DAYS,
) -> str:
    """Get accuracy statistics for analysis agents.

    Args:
        agent_name: Filter to a specific agent.
        signal_type: Filter to a specific signal type.
        days_lookback: Number of days to look back (default 90).
    """
    deps = _deps(ctx)

    since_date = (date.today() - timedelta(days=days_lookback)).isoformat()

    rows = await deps.store.get_accuracy_stats(
        agent_name=agent_name,
        signal_type=signal_type,
        since_date=since_date,
    )

    agent_stats = [AgentStats(**row) for row in rows]

    response = AgentAccuracyResponse(agent_stats=agent_stats)
    return response.model_dump_json()


async def get_pending_scoring(
    ctx: Context,
    as_of_date: str | None = None,
    buffer_days: int = HORIZON_BUFFER_DAYS,
) -> str:
    """Get predictions that are unscored and past their horizon.

    Finds predictions eligible for scoring — those where the prediction
    date plus horizon days (plus a buffer) has elapsed.

    Args:
        as_of_date: Reference date (YYYY-MM-DD). Defaults to today.
        buffer_days: Days past horizon before eligible. Defaults to 1.
    """
    deps = _deps(ctx)

    if as_of_date is None:
        as_of_date = date.today().isoformat()

    rows = await deps.store.query_pending_scoring(
        as_of_date=as_of_date,
        buffer_days=buffer_days,
    )
    predictions = [PredictionRecord(**row) for row in rows]

    response = PendingScoringResponse(
        predictions=predictions,
        total_count=len(predictions),
    )
    return response.model_dump_json()


# All tool functions in registration order.
TOOLS = [
    log_prediction,
    score_prediction,
    get_prediction_history,
    get_agent_accuracy,
    get_pending_scoring,
]


def create_server(name: str = "predictions-db") -> FastMCP:
    """Create a new FastMCP server instance with all tools registered.

    Returns a fresh instance each time -- useful for testing where each
    test needs its own isolated server with independent lifespan state.
    """
    server = FastMCP(name, lifespan=lifespan)
    for tool_fn in TOOLS:
        server.tool()(tool_fn)  # type: ignore[arg-type]
    return server


# Module-level server instance for production use.
mcp = create_server()
