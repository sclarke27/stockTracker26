"""Tests for the prediction scoring runner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from stock_radar.scoring.models import ScoringOutcome, ScoringResult
from stock_radar.scoring.runner import (
    _build_result,
    _choose_outputsize,
    _compute_horizon_date,
    _fetch_price_map,
    _get_pending_predictions,
    _score_single_prediction,
    run_scoring_loop,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pending_prediction(**overrides: object) -> dict:
    """Build a fake pending prediction dict."""
    base: dict = {
        "id": "pred-001",
        "ticker": "AAPL",
        "agent_name": "earnings_linguist",
        "signal_type": "earnings_sentiment",
        "direction": "BULLISH",
        "confidence": 0.85,
        "reasoning": "Strong forward guidance.",
        "prediction_date": "2026-01-10",
        "horizon_days": 5,
        "created_at": "2026-01-10T12:00:00+00:00",
        "scored_at": None,
        "actual_price_close": None,
        "actual_price_at_horizon": None,
        "return_pct": None,
        "status": None,
    }
    base.update(overrides)
    return base


def _mock_tool_result(data: dict) -> MagicMock:
    """Create a mock MCP tool result with JSON content."""
    content_item = MagicMock()
    content_item.text = json.dumps(data)
    result = MagicMock()
    result.content = [content_item]
    return result


# ---------------------------------------------------------------------------
# Test _compute_horizon_date
# ---------------------------------------------------------------------------


class TestComputeHorizonDate:
    """Tests for the horizon date computation helper."""

    def test_basic_computation(self) -> None:
        assert _compute_horizon_date("2026-01-10", 5) == "2026-01-15"

    def test_crosses_month_boundary(self) -> None:
        assert _compute_horizon_date("2026-01-28", 5) == "2026-02-02"

    def test_zero_horizon(self) -> None:
        """Horizon of 0 returns same date (degenerate case)."""
        assert _compute_horizon_date("2026-01-10", 0) == "2026-01-10"


# ---------------------------------------------------------------------------
# Test _choose_outputsize
# ---------------------------------------------------------------------------


class TestChooseOutputsize:
    """Tests for the outputsize selection helper."""

    def test_recent_uses_compact(self) -> None:
        """Predictions within 130 days use compact."""
        result = _choose_outputsize("2026-01-01", "2026-03-01")
        assert result == "compact"

    def test_old_uses_full(self) -> None:
        """Predictions older than 130 days use full."""
        result = _choose_outputsize("2025-06-01", "2026-02-01")
        assert result == "full"

    def test_boundary_at_130_days(self) -> None:
        """Exactly 130 days uses compact (<=)."""
        result = _choose_outputsize("2025-10-01", "2026-02-08")
        assert result == "compact"


# ---------------------------------------------------------------------------
# Test _get_pending_predictions
# ---------------------------------------------------------------------------


class TestGetPendingPredictions:
    """Tests for the pending predictions fetcher."""

    async def test_returns_predictions_list(self) -> None:
        """Successfully parses predictions from tool response."""
        pred = _make_pending_prediction()
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = _mock_tool_result(
            {"predictions": [pred], "total_count": 1}
        )

        result = await _get_pending_predictions(mock_client, "2026-02-01")
        assert len(result) == 1
        assert result[0]["id"] == "pred-001"

    async def test_returns_empty_on_error(self) -> None:
        """Returns empty list on tool call failure (never raises)."""
        mock_client = AsyncMock()
        mock_client.call_tool.side_effect = RuntimeError("Connection failed")

        result = await _get_pending_predictions(mock_client, "2026-02-01")
        assert result == []


# ---------------------------------------------------------------------------
# Test _fetch_price_map
# ---------------------------------------------------------------------------


class TestFetchPriceMap:
    """Tests for the price map fetcher."""

    async def test_returns_price_map(self) -> None:
        """Parses price history into date->close dict."""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = _mock_tool_result(
            {
                "ticker": "AAPL",
                "bars": [
                    {"date": "2026-01-15", "close": 155.0},
                    {"date": "2026-01-14", "close": 154.0},
                ],
                "last_refreshed": "2026-01-15",
            }
        )

        result = await _fetch_price_map(mock_client, "AAPL", "compact")
        assert result == {"2026-01-15": 155.0, "2026-01-14": 154.0}

    async def test_returns_none_on_error(self) -> None:
        """Returns None on tool call failure."""
        mock_client = AsyncMock()
        mock_client.call_tool.side_effect = RuntimeError("API error")

        result = await _fetch_price_map(mock_client, "AAPL", "compact")
        assert result is None


# ---------------------------------------------------------------------------
# Test _score_single_prediction
# ---------------------------------------------------------------------------


class TestScoreSinglePrediction:
    """Tests for scoring a single prediction."""

    async def test_successful_score(self) -> None:
        """Returns outcome with status and return_pct on success."""
        pred = _make_pending_prediction(prediction_date="2026-01-10", horizon_days=5)
        price_map = {
            "2026-01-10": 150.0,  # prediction date
            "2026-01-15": 155.0,  # horizon date
        }
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = _mock_tool_result(
            {
                "prediction_id": "pred-001",
                "status": "CORRECT",
                "return_pct": 3.33,
                "direction": "BULLISH",
                "confidence": 0.85,
            }
        )

        outcome = await _score_single_prediction(mock_client, pred, price_map, "AAPL")
        assert outcome.status == "CORRECT"
        assert outcome.return_pct == 3.33
        assert outcome.skipped is False

    async def test_skipped_when_prediction_date_not_in_history(self) -> None:
        """Returns skipped outcome when prediction date can't be aligned."""
        pred = _make_pending_prediction(prediction_date="2025-06-01", horizon_days=5)
        # Price map doesn't contain dates anywhere near 2025-06-01
        price_map = {"2026-01-10": 150.0}

        mock_client = AsyncMock()
        outcome = await _score_single_prediction(mock_client, pred, price_map, "AAPL")
        assert outcome.skipped is True
        assert "prediction_date" in outcome.skip_reason

    async def test_skipped_when_horizon_date_not_in_history(self) -> None:
        """Returns skipped outcome when horizon date can't be aligned."""
        # horizon_days=10 puts horizon on 2026-01-20, which is 10 days from the
        # only price entry (2026-01-10) -- beyond MAX_SEARCH_DAYS=7 so no match.
        pred = _make_pending_prediction(prediction_date="2026-01-10", horizon_days=10)
        # Price map has prediction date but nothing within 7 days of horizon date
        price_map = {"2026-01-10": 150.0}

        mock_client = AsyncMock()
        outcome = await _score_single_prediction(mock_client, pred, price_map, "AAPL")
        assert outcome.skipped is True
        assert "horizon_date" in outcome.skip_reason

    async def test_skipped_when_score_call_fails(self) -> None:
        """Returns skipped outcome when score_prediction tool fails."""
        pred = _make_pending_prediction(prediction_date="2026-01-10", horizon_days=5)
        price_map = {"2026-01-10": 150.0, "2026-01-15": 155.0}

        mock_client = AsyncMock()
        mock_client.call_tool.side_effect = RuntimeError("DB error")

        outcome = await _score_single_prediction(mock_client, pred, price_map, "AAPL")
        assert outcome.skipped is True
        assert "score_call_failed" in outcome.skip_reason


# ---------------------------------------------------------------------------
# Test _build_result
# ---------------------------------------------------------------------------


class TestBuildResult:
    """Tests for the result builder."""

    def test_empty_outcomes(self) -> None:
        """Zero outcomes results in zero counts."""
        result = _build_result("2026-01-15T10:00:00+00:00", [])
        assert isinstance(result, ScoringResult)
        assert result.predictions_found == 0
        assert result.predictions_scored == 0
        assert result.predictions_skipped == 0

    def test_counts_scored_and_skipped(self) -> None:
        """Correctly counts scored vs skipped outcomes."""
        outcomes = [
            ScoringOutcome(prediction_id="a", ticker="AAPL", status="CORRECT", return_pct=2.0),
            ScoringOutcome(prediction_id="b", ticker="MSFT", skipped=True, skip_reason="no data"),
            ScoringOutcome(prediction_id="c", ticker="GOOG", status="INCORRECT", return_pct=-1.0),
        ]
        result = _build_result("2026-01-15T10:00:00+00:00", outcomes)
        assert result.predictions_found == 3
        assert result.predictions_scored == 2
        assert result.predictions_skipped == 1

    def test_timestamps_populated(self) -> None:
        """started_at and completed_at are ISO strings."""
        result = _build_result("2026-01-15T10:00:00+00:00", [])
        assert result.started_at == "2026-01-15T10:00:00+00:00"
        assert result.completed_at  # non-empty
        assert result.duration_seconds >= 0


# ---------------------------------------------------------------------------
# Test run_scoring_loop (integration-style with mocks)
# ---------------------------------------------------------------------------


class TestRunScoringLoop:
    """Integration tests for run_scoring_loop with mocked MCP."""

    async def test_returns_scoring_result(self) -> None:
        """run_scoring_loop returns a ScoringResult."""
        with (
            patch("stock_radar.scoring.runner.create_predictions_server"),
            patch("stock_radar.scoring.runner.create_market_server"),
            patch("stock_radar.scoring.runner.Client") as mock_client_cls,
            patch("stock_radar.scoring.runner.setup_logging"),
        ):
            # Mock the async context manager
            mock_pred_client = AsyncMock()
            mock_market_client = AsyncMock()

            # get_pending_scoring returns empty list
            mock_pred_client.call_tool.return_value = _mock_tool_result(
                {"predictions": [], "total_count": 0}
            )

            # Make Client() return different mocks for pred and market
            contexts = [mock_pred_client, mock_market_client]
            mock_client_cls.return_value.__aenter__ = AsyncMock(side_effect=contexts)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await run_scoring_loop(as_of_date="2026-02-01")

        assert isinstance(result, ScoringResult)
        assert result.predictions_found == 0

    async def test_empty_pending_returns_zero_counts(self) -> None:
        """No pending predictions results in zero counts."""
        with (
            patch("stock_radar.scoring.runner.create_predictions_server"),
            patch("stock_radar.scoring.runner.create_market_server"),
            patch("stock_radar.scoring.runner.Client") as mock_client_cls,
            patch("stock_radar.scoring.runner.setup_logging"),
        ):
            mock_pred_client = AsyncMock()
            mock_market_client = AsyncMock()
            mock_pred_client.call_tool.return_value = _mock_tool_result(
                {"predictions": [], "total_count": 0}
            )

            contexts = [mock_pred_client, mock_market_client]
            mock_client_cls.return_value.__aenter__ = AsyncMock(side_effect=contexts)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await run_scoring_loop(as_of_date="2026-02-01")

        assert result.predictions_scored == 0
        assert result.predictions_skipped == 0
        assert result.outcomes == []
