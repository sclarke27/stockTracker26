"""Prediction scoring loop -- scores unscored predictions against actual prices."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta

from fastmcp import Client
from loguru import logger

from stock_radar.mcp_servers.market_data.server import (
    create_server as create_market_server,
)
from stock_radar.mcp_servers.predictions_db.server import (
    create_server as create_predictions_server,
)
from stock_radar.scoring.config import (
    COMPACT_HISTORY_CALENDAR_DAYS,
    HORIZON_BUFFER_DAYS,
    SERVER_NAME,
)
from stock_radar.scoring.date_utils import build_price_map, find_closest_trading_day
from stock_radar.scoring.models import ScoringOutcome, ScoringResult
from stock_radar.utils.logging import setup_logging

_log = logger.bind(server=SERVER_NAME)


async def run_scoring_loop(
    as_of_date: str | None = None,
) -> ScoringResult:
    """Run the prediction scoring loop.

    Fetches all unscored predictions past their horizon, groups by ticker,
    fetches price history once per ticker, and scores each prediction.

    Args:
        as_of_date: Reference date (YYYY-MM-DD). Defaults to today.

    Returns:
        ScoringResult summarizing the run.
    """
    setup_logging()
    started_at = datetime.now(UTC).isoformat()

    if as_of_date is None:
        as_of_date = date.today().isoformat()

    _log.info("Scoring loop starting", as_of_date=as_of_date)

    predictions_server = create_predictions_server()
    market_server = create_market_server()

    outcomes: list[ScoringOutcome] = []

    async with (
        Client(predictions_server) as pred_client,
        Client(market_server) as market_client,
    ):
        # 1. Fetch all pending predictions.
        pending = await _get_pending_predictions(pred_client, as_of_date)

        _log.info("Pending predictions found", count=len(pending))

        if not pending:
            return _build_result(started_at, outcomes)

        # 2. Group by ticker to minimize API calls.
        by_ticker: dict[str, list[dict]] = defaultdict(list)
        for pred in pending:
            by_ticker[pred["ticker"]].append(pred)

        # 3. Score each ticker group.
        for ticker, ticker_preds in by_ticker.items():
            ticker_outcomes = await _score_ticker_predictions(
                market_client=market_client,
                pred_client=pred_client,
                ticker=ticker,
                predictions=ticker_preds,
                as_of_date=as_of_date,
            )
            outcomes.extend(ticker_outcomes)

    return _build_result(started_at, outcomes)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _get_pending_predictions(
    pred_client: Client,
    as_of_date: str,
) -> list[dict]:
    """Fetch all pending predictions from the predictions-db server.

    Returns empty list on failure (never raises).
    """
    try:
        result = await pred_client.call_tool(
            "get_pending_scoring",
            {"as_of_date": as_of_date, "buffer_days": HORIZON_BUFFER_DAYS},
        )
        data = json.loads(result.content[0].text)
        return data["predictions"]
    except Exception as exc:
        _log.error("Failed to fetch pending predictions", error=str(exc))
        return []


async def _score_ticker_predictions(
    market_client: Client,
    pred_client: Client,
    ticker: str,
    predictions: list[dict],
    as_of_date: str,
) -> list[ScoringOutcome]:
    """Score all predictions for a single ticker.

    Fetches price history once, then scores each prediction individually.
    If the price fetch fails, all predictions for this ticker are skipped.

    Args:
        market_client: FastMCP Client for market-data-mcp.
        pred_client: FastMCP Client for predictions-db-mcp.
        ticker: Stock ticker symbol.
        predictions: All unscored predictions for this ticker.
        as_of_date: Reference date string.

    Returns:
        List of ScoringOutcome -- one per prediction.
    """
    # Determine if we need full history.
    oldest_prediction_date = min(p["prediction_date"] for p in predictions)
    outputsize = _choose_outputsize(oldest_prediction_date, as_of_date)

    price_map = await _fetch_price_map(market_client, ticker, outputsize)

    if price_map is None:
        _log.warning("Price data unavailable, skipping ticker", ticker=ticker)
        return [
            ScoringOutcome(
                prediction_id=p["id"],
                ticker=ticker,
                skipped=True,
                skip_reason="price_data_unavailable",
            )
            for p in predictions
        ]

    outcomes: list[ScoringOutcome] = []
    for pred in predictions:
        outcome = await _score_single_prediction(pred_client, pred, price_map, ticker)
        outcomes.append(outcome)

    return outcomes


def _choose_outputsize(oldest_prediction_date: str, as_of_date: str) -> str:
    """Choose 'compact' or 'full' based on how old the oldest prediction is.

    Args:
        oldest_prediction_date: Earliest prediction_date across predictions.
        as_of_date: Reference date (today).

    Returns:
        ``"compact"`` if the oldest prediction is within the compact window,
        ``"full"`` otherwise.
    """
    oldest = date.fromisoformat(oldest_prediction_date)
    today = date.fromisoformat(as_of_date)
    calendar_days_ago = (today - oldest).days
    return "compact" if calendar_days_ago <= COMPACT_HISTORY_CALENDAR_DAYS else "full"


async def _fetch_price_map(
    market_client: Client,
    ticker: str,
    outputsize: str,
) -> dict[str, float] | None:
    """Fetch price history and convert to a date->close map.

    Returns ``None`` on any failure so the caller can skip the ticker.
    """
    try:
        result = await market_client.call_tool(
            "get_price_history",
            {"ticker": ticker, "outputsize": outputsize},
        )
        data = json.loads(result.content[0].text)
        return build_price_map(data["bars"])
    except Exception as exc:
        _log.warning(
            "Failed to fetch price history",
            ticker=ticker,
            outputsize=outputsize,
            error=str(exc),
        )
        return None


async def _score_single_prediction(
    pred_client: Client,
    prediction: dict,
    price_map: dict[str, float],
    ticker: str,
) -> ScoringOutcome:
    """Score a single prediction using aligned trading-day prices.

    Args:
        pred_client: FastMCP Client for predictions-db-mcp.
        prediction: Prediction record dict.
        price_map: Date->close map for this ticker.
        ticker: Ticker symbol (for logging).

    Returns:
        ScoringOutcome with status and return_pct, or skipped=True.
    """
    prediction_id = prediction["id"]
    prediction_date = prediction["prediction_date"]
    horizon_date = _compute_horizon_date(prediction_date, prediction["horizon_days"])

    # Align both dates to actual trading days.
    actual_pred_date = find_closest_trading_day(prediction_date, price_map, direction="backward")
    actual_horizon_date = find_closest_trading_day(horizon_date, price_map, direction="backward")

    if actual_pred_date is None:
        return ScoringOutcome(
            prediction_id=prediction_id,
            ticker=ticker,
            skipped=True,
            skip_reason="prediction_date_not_in_history",
        )

    if actual_horizon_date is None:
        return ScoringOutcome(
            prediction_id=prediction_id,
            ticker=ticker,
            skipped=True,
            skip_reason="horizon_date_not_in_history",
        )

    price_close = price_map[actual_pred_date]
    price_at_horizon = price_map[actual_horizon_date]

    try:
        result = await pred_client.call_tool(
            "score_prediction",
            {
                "prediction_id": prediction_id,
                "actual_price_close": price_close,
                "actual_price_at_horizon": price_at_horizon,
            },
        )
        data = json.loads(result.content[0].text)
        return ScoringOutcome(
            prediction_id=prediction_id,
            ticker=ticker,
            status=data["status"],
            return_pct=data["return_pct"],
        )
    except Exception as exc:
        _log.warning(
            "Failed to score prediction",
            prediction_id=prediction_id,
            ticker=ticker,
            error=str(exc),
        )
        return ScoringOutcome(
            prediction_id=prediction_id,
            ticker=ticker,
            skipped=True,
            skip_reason=f"score_call_failed: {exc}",
        )


def _compute_horizon_date(prediction_date: str, horizon_days: int) -> str:
    """Compute the horizon date from prediction_date and horizon_days.

    Args:
        prediction_date: ISO date string (YYYY-MM-DD).
        horizon_days: Number of calendar days from prediction_date.

    Returns:
        ISO date string of the horizon date.
    """
    base = date.fromisoformat(prediction_date)
    return (base + timedelta(days=horizon_days)).isoformat()


def _build_result(started_at: str, outcomes: list[ScoringOutcome]) -> ScoringResult:
    """Build the final ScoringResult from collected outcomes.

    Args:
        started_at: ISO timestamp when the scoring loop started.
        outcomes: All scoring outcomes from this run.

    Returns:
        Complete ScoringResult with timing and counts.
    """
    completed_at = datetime.now(UTC).isoformat()
    start_dt = datetime.fromisoformat(started_at)
    end_dt = datetime.fromisoformat(completed_at)
    duration_seconds = round((end_dt - start_dt).total_seconds(), 2)

    scored = sum(1 for o in outcomes if not o.skipped)
    skipped = sum(1 for o in outcomes if o.skipped)

    result = ScoringResult(
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        predictions_found=len(outcomes),
        predictions_scored=scored,
        predictions_skipped=skipped,
        outcomes=outcomes,
    )

    _log.info(
        "Scoring loop complete",
        found=result.predictions_found,
        scored=result.predictions_scored,
        skipped=result.predictions_skipped,
        duration_s=result.duration_seconds,
    )
    return result


def main() -> None:
    """CLI entry point for the prediction scoring loop."""
    asyncio.run(run_scoring_loop())
