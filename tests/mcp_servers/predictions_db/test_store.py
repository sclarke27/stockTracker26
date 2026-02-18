"""Tests for the predictions database async SQLite store layer."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from aiosqlite import IntegrityError

from stock_radar.mcp_servers.predictions_db.store import PredictionsStore


def _make_prediction(**overrides: object) -> dict:
    """Build a prediction dict with sensible defaults.

    Any field can be overridden via keyword arguments.
    """
    base: dict = {
        "id": str(uuid.uuid4()),
        "ticker": "AAPL",
        "agent_name": "earnings_linguist",
        "signal_type": "earnings_sentiment",
        "direction": "BULLISH",
        "confidence": 0.85,
        "reasoning": "Strong language signals in Q4 transcript.",
        "prediction_date": "2026-02-17",
        "horizon_days": 5,
        "created_at": datetime.now(UTC).isoformat(),
    }
    base.update(overrides)
    return base


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[PredictionsStore]:
    """Provide an initialized PredictionsStore backed by a temporary database."""
    db_path = str(tmp_path / "test_predictions.db")
    s = PredictionsStore(db_path)
    await s.initialize()
    yield s
    await s.close()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestStoreInitialize:
    """Tests for database initialization."""

    async def test_creates_db_file(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "new_predictions.db")
        assert not Path(db_path).exists()
        s = PredictionsStore(db_path)
        await s.initialize()
        assert Path(db_path).exists()
        await s.close()

    async def test_creates_predictions_table(self, store: PredictionsStore) -> None:
        # If the table exists we should be able to query without error.
        rows, total = await store.query()
        assert rows == []
        assert total == 0

    async def test_idempotent(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "idempotent.db")
        s = PredictionsStore(db_path)
        await s.initialize()
        # Calling a second time must not raise.
        await s.initialize()
        rows, total = await s.query()
        assert rows == []
        assert total == 0
        await s.close()


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


class TestStoreInsert:
    """Tests for inserting predictions."""

    async def test_insert_and_retrieve(self, store: PredictionsStore) -> None:
        pred = _make_prediction()
        await store.insert(pred)
        row = await store.get_by_id(pred["id"])
        assert row is not None
        assert row["id"] == pred["id"]
        assert row["ticker"] == "AAPL"
        assert row["agent_name"] == "earnings_linguist"
        assert row["signal_type"] == "earnings_sentiment"
        assert row["direction"] == "BULLISH"
        assert row["confidence"] == pytest.approx(0.85)
        assert row["reasoning"] == pred["reasoning"]
        assert row["prediction_date"] == "2026-02-17"
        assert row["horizon_days"] == 5
        assert row["created_at"] == pred["created_at"]

    async def test_scoring_fields_are_null(self, store: PredictionsStore) -> None:
        pred = _make_prediction()
        await store.insert(pred)
        row = await store.get_by_id(pred["id"])
        assert row is not None
        assert row["scored_at"] is None
        assert row["actual_price_close"] is None
        assert row["actual_price_at_horizon"] is None
        assert row["return_pct"] is None
        assert row["status"] is None

    async def test_duplicate_id_raises(self, store: PredictionsStore) -> None:
        pred = _make_prediction()
        await store.insert(pred)
        with pytest.raises(IntegrityError):
            await store.insert(pred)

    async def test_insert_multiple(self, store: PredictionsStore) -> None:
        pred_a = _make_prediction(ticker="AAPL")
        pred_b = _make_prediction(ticker="MSFT")
        pred_c = _make_prediction(ticker="GOOG")
        await store.insert(pred_a)
        await store.insert(pred_b)
        await store.insert(pred_c)
        rows, total = await store.query()
        assert total == 3
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# Get by ID
# ---------------------------------------------------------------------------


class TestStoreGetById:
    """Tests for retrieving a single prediction."""

    async def test_existing_returns_dict(self, store: PredictionsStore) -> None:
        pred = _make_prediction()
        await store.insert(pred)
        row = await store.get_by_id(pred["id"])
        assert isinstance(row, dict)
        assert row["id"] == pred["id"]

    async def test_nonexistent_returns_none(self, store: PredictionsStore) -> None:
        result = await store.get_by_id("does-not-exist")
        assert result is None


# ---------------------------------------------------------------------------
# Update score
# ---------------------------------------------------------------------------


class TestStoreUpdateScore:
    """Tests for scoring a prediction."""

    async def test_score_existing(self, store: PredictionsStore) -> None:
        pred = _make_prediction()
        await store.insert(pred)
        ok = await store.update_score(
            prediction_id=pred["id"],
            scored_at="2026-02-22T12:00:00+00:00",
            actual_price_close=175.50,
            actual_price_at_horizon=180.25,
            return_pct=2.7,
            status="CORRECT",
        )
        assert ok is True
        row = await store.get_by_id(pred["id"])
        assert row is not None
        assert row["scored_at"] == "2026-02-22T12:00:00+00:00"
        assert row["actual_price_close"] == pytest.approx(175.50)
        assert row["actual_price_at_horizon"] == pytest.approx(180.25)
        assert row["return_pct"] == pytest.approx(2.7)
        assert row["status"] == "CORRECT"

    async def test_nonexistent_returns_false(self, store: PredictionsStore) -> None:
        ok = await store.update_score(
            prediction_id="ghost",
            scored_at="2026-02-22T12:00:00+00:00",
            actual_price_close=100.0,
            actual_price_at_horizon=105.0,
            return_pct=5.0,
            status="CORRECT",
        )
        assert ok is False

    async def test_rescoring_overwrites(self, store: PredictionsStore) -> None:
        pred = _make_prediction()
        await store.insert(pred)
        await store.update_score(
            prediction_id=pred["id"],
            scored_at="2026-02-22T12:00:00+00:00",
            actual_price_close=175.50,
            actual_price_at_horizon=180.25,
            return_pct=2.7,
            status="CORRECT",
        )
        await store.update_score(
            prediction_id=pred["id"],
            scored_at="2026-02-23T08:00:00+00:00",
            actual_price_close=170.00,
            actual_price_at_horizon=165.00,
            return_pct=-2.9,
            status="INCORRECT",
        )
        row = await store.get_by_id(pred["id"])
        assert row is not None
        assert row["scored_at"] == "2026-02-23T08:00:00+00:00"
        assert row["status"] == "INCORRECT"
        assert row["return_pct"] == pytest.approx(-2.9)

    async def test_scored_fields_all_populated(self, store: PredictionsStore) -> None:
        pred = _make_prediction()
        await store.insert(pred)
        await store.update_score(
            prediction_id=pred["id"],
            scored_at="2026-02-22T12:00:00+00:00",
            actual_price_close=175.50,
            actual_price_at_horizon=180.25,
            return_pct=2.7,
            status="CORRECT",
        )
        row = await store.get_by_id(pred["id"])
        assert row is not None
        assert row["scored_at"] is not None
        assert row["actual_price_close"] is not None
        assert row["actual_price_at_horizon"] is not None
        assert row["return_pct"] is not None
        assert row["status"] is not None

    async def test_original_fields_unchanged(self, store: PredictionsStore) -> None:
        pred = _make_prediction()
        await store.insert(pred)
        await store.update_score(
            prediction_id=pred["id"],
            scored_at="2026-02-22T12:00:00+00:00",
            actual_price_close=175.50,
            actual_price_at_horizon=180.25,
            return_pct=2.7,
            status="CORRECT",
        )
        row = await store.get_by_id(pred["id"])
        assert row is not None
        assert row["ticker"] == pred["ticker"]
        assert row["agent_name"] == pred["agent_name"]
        assert row["signal_type"] == pred["signal_type"]
        assert row["direction"] == pred["direction"]
        assert row["confidence"] == pytest.approx(pred["confidence"])
        assert row["reasoning"] == pred["reasoning"]
        assert row["prediction_date"] == pred["prediction_date"]
        assert row["horizon_days"] == pred["horizon_days"]
        assert row["created_at"] == pred["created_at"]


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestStoreQuery:
    """Tests for querying predictions with filters."""

    async def test_no_filters_returns_all(self, store: PredictionsStore) -> None:
        for _ in range(3):
            await store.insert(_make_prediction())
        rows, total = await store.query()
        assert len(rows) == 3
        assert total == 3

    async def test_filter_by_ticker(self, store: PredictionsStore) -> None:
        await store.insert(_make_prediction(ticker="AAPL"))
        await store.insert(_make_prediction(ticker="MSFT"))
        await store.insert(_make_prediction(ticker="AAPL"))
        rows, total = await store.query(ticker="AAPL")
        assert total == 2
        assert all(r["ticker"] == "AAPL" for r in rows)

    async def test_filter_by_agent_name(self, store: PredictionsStore) -> None:
        await store.insert(_make_prediction(agent_name="earnings_linguist"))
        await store.insert(_make_prediction(agent_name="contagion_mapper"))
        rows, total = await store.query(agent_name="contagion_mapper")
        assert total == 1
        assert rows[0]["agent_name"] == "contagion_mapper"

    async def test_filter_by_signal_type(self, store: PredictionsStore) -> None:
        await store.insert(_make_prediction(signal_type="earnings_sentiment"))
        await store.insert(_make_prediction(signal_type="insider_pattern"))
        rows, total = await store.query(signal_type="insider_pattern")
        assert total == 1
        assert rows[0]["signal_type"] == "insider_pattern"

    async def test_scored_only(self, store: PredictionsStore) -> None:
        pred_scored = _make_prediction()
        pred_unscored = _make_prediction()
        await store.insert(pred_scored)
        await store.insert(pred_unscored)
        await store.update_score(
            prediction_id=pred_scored["id"],
            scored_at="2026-02-22T12:00:00+00:00",
            actual_price_close=175.0,
            actual_price_at_horizon=180.0,
            return_pct=2.8,
            status="CORRECT",
        )
        rows, total = await store.query(scored_only=True)
        assert total == 1
        assert rows[0]["id"] == pred_scored["id"]

    async def test_limit_and_offset(self, store: PredictionsStore) -> None:
        for i in range(5):
            await store.insert(
                _make_prediction(
                    created_at=f"2026-02-{17 - i:02d}T00:00:00+00:00",
                )
            )
        rows, total = await store.query(limit=2, offset=1)
        assert len(rows) == 2
        assert total == 5

    async def test_total_count_ignores_pagination(self, store: PredictionsStore) -> None:
        for _ in range(10):
            await store.insert(_make_prediction())
        rows, total = await store.query(limit=3, offset=0)
        assert len(rows) == 3
        assert total == 10

    async def test_newest_first_ordering(self, store: PredictionsStore) -> None:
        await store.insert(_make_prediction(created_at="2026-02-10T00:00:00+00:00"))
        await store.insert(_make_prediction(created_at="2026-02-15T00:00:00+00:00"))
        await store.insert(_make_prediction(created_at="2026-02-12T00:00:00+00:00"))
        rows, _ = await store.query()
        dates = [r["created_at"] for r in rows]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# Accuracy stats
# ---------------------------------------------------------------------------


class TestStoreGetAccuracyStats:
    """Tests for accuracy statistics aggregation."""

    async def _insert_scored(
        self,
        store: PredictionsStore,
        *,
        agent_name: str = "earnings_linguist",
        signal_type: str = "earnings_sentiment",
        status: str = "CORRECT",
        return_pct: float = 3.0,
        scored_at: str = "2026-02-20T00:00:00+00:00",
        prediction_date: str = "2026-02-17",
    ) -> None:
        """Insert a prediction and immediately score it."""
        pred = _make_prediction(
            agent_name=agent_name,
            signal_type=signal_type,
            prediction_date=prediction_date,
        )
        await store.insert(pred)
        await store.update_score(
            prediction_id=pred["id"],
            scored_at=scored_at,
            actual_price_close=100.0,
            actual_price_at_horizon=100.0 + return_pct,
            return_pct=return_pct,
            status=status,
        )

    async def test_single_agent_stats(self, store: PredictionsStore) -> None:
        await self._insert_scored(store, status="CORRECT", return_pct=5.0)
        await self._insert_scored(store, status="INCORRECT", return_pct=-2.0)
        stats = await store.get_accuracy_stats()
        assert len(stats) == 1
        row = stats[0]
        assert row["agent_name"] == "earnings_linguist"
        assert row["signal_type"] == "earnings_sentiment"
        assert row["total"] == 2
        assert row["scored"] == 2
        assert row["correct"] == 1
        assert row["accuracy_pct"] == pytest.approx(50.0)
        assert row["avg_confidence"] > 0
        assert row["avg_return_when_correct"] == pytest.approx(5.0)

    async def test_multiple_agents(self, store: PredictionsStore) -> None:
        await self._insert_scored(store, agent_name="earnings_linguist", status="CORRECT")
        await self._insert_scored(
            store,
            agent_name="contagion_mapper",
            signal_type="cross_sector",
            status="INCORRECT",
            return_pct=-1.5,
        )
        stats = await store.get_accuracy_stats()
        assert len(stats) == 2
        agents = {s["agent_name"] for s in stats}
        assert agents == {"earnings_linguist", "contagion_mapper"}

    async def test_no_scored_predictions(self, store: PredictionsStore) -> None:
        # Insert unscored predictions only.
        await store.insert(_make_prediction())
        await store.insert(_make_prediction())
        stats = await store.get_accuracy_stats()
        assert len(stats) == 1
        row = stats[0]
        assert row["scored"] == 0
        assert row["accuracy_pct"] == pytest.approx(0.0)

    async def test_filter_by_agent_name(self, store: PredictionsStore) -> None:
        await self._insert_scored(store, agent_name="earnings_linguist", status="CORRECT")
        await self._insert_scored(
            store,
            agent_name="contagion_mapper",
            signal_type="cross_sector",
            status="CORRECT",
        )
        stats = await store.get_accuracy_stats(agent_name="earnings_linguist")
        assert len(stats) == 1
        assert stats[0]["agent_name"] == "earnings_linguist"

    async def test_since_date_filter(self, store: PredictionsStore) -> None:
        await self._insert_scored(
            store,
            status="CORRECT",
            prediction_date="2026-01-15",
        )
        await self._insert_scored(
            store,
            status="CORRECT",
            prediction_date="2026-02-15",
        )
        stats = await store.get_accuracy_stats(since_date="2026-02-01")
        assert len(stats) == 1
        row = stats[0]
        assert row["scored"] == 1


# ---------------------------------------------------------------------------
# Query pending scoring
# ---------------------------------------------------------------------------


class TestStoreQueryPendingScoring:
    """Tests for the query_pending_scoring method."""

    async def test_returns_empty_when_no_predictions(self, store: PredictionsStore) -> None:
        """Empty database returns empty list."""
        results = await store.query_pending_scoring(as_of_date="2026-03-01")
        assert results == []

    async def test_returns_unscored_past_horizon(self, store: PredictionsStore) -> None:
        """Unscored prediction with elapsed horizon+buffer is returned."""
        pred = _make_prediction(prediction_date="2026-01-01", horizon_days=5)
        await store.insert(pred)

        # as_of_date = Jan 8 → horizon was Jan 6, buffer=1 → eligible on Jan 7
        results = await store.query_pending_scoring(as_of_date="2026-01-08")
        assert len(results) == 1
        assert results[0]["id"] == pred["id"]

    async def test_excludes_future_horizon(self, store: PredictionsStore) -> None:
        """Prediction whose horizon has not yet elapsed is excluded."""
        pred = _make_prediction(prediction_date="2026-03-01", horizon_days=10)
        await store.insert(pred)

        # as_of_date = Mar 5 → horizon is Mar 11, not yet elapsed
        results = await store.query_pending_scoring(as_of_date="2026-03-05")
        assert results == []

    async def test_excludes_within_buffer(self, store: PredictionsStore) -> None:
        """Horizon elapsed but within buffer days is excluded."""
        pred = _make_prediction(prediction_date="2026-01-01", horizon_days=5)
        await store.insert(pred)

        # Horizon = Jan 6. buffer=1, so eligible on Jan 7.
        # as_of_date = Jan 6 → not yet eligible.
        results = await store.query_pending_scoring(as_of_date="2026-01-06")
        assert results == []

    async def test_excludes_already_scored(self, store: PredictionsStore) -> None:
        """Already-scored predictions are excluded."""
        pred = _make_prediction(prediction_date="2026-01-01", horizon_days=5)
        await store.insert(pred)
        await store.update_score(pred["id"], "2026-01-08T00:00:00", 150.0, 155.0, 3.33, "CORRECT")

        results = await store.query_pending_scoring(as_of_date="2026-02-01")
        assert results == []

    async def test_mixed_predictions_filters_correctly(self, store: PredictionsStore) -> None:
        """Mix of scored, unscored, future, and eligible — only eligible returned."""
        # Eligible: unscored, past horizon + buffer
        eligible = _make_prediction(id="eligible-1", prediction_date="2026-01-01", horizon_days=5)
        await store.insert(eligible)

        # Scored: past horizon but already scored
        scored = _make_prediction(id="scored-1", prediction_date="2026-01-01", horizon_days=5)
        await store.insert(scored)
        await store.update_score("scored-1", "2026-01-08T00:00:00", 150.0, 155.0, 3.33, "CORRECT")

        # Future: horizon not yet elapsed
        future = _make_prediction(id="future-1", prediction_date="2026-03-01", horizon_days=10)
        await store.insert(future)

        results = await store.query_pending_scoring(as_of_date="2026-01-15")
        assert len(results) == 1
        assert results[0]["id"] == "eligible-1"

    async def test_orders_by_prediction_date_asc(self, store: PredictionsStore) -> None:
        """Results are ordered oldest prediction_date first."""
        newer = _make_prediction(id="newer", prediction_date="2026-01-10", horizon_days=3)
        older = _make_prediction(id="older", prediction_date="2026-01-05", horizon_days=3)
        await store.insert(newer)
        await store.insert(older)

        results = await store.query_pending_scoring(as_of_date="2026-01-20")
        assert len(results) == 2
        assert results[0]["id"] == "older"
        assert results[1]["id"] == "newer"

    async def test_buffer_days_zero(self, store: PredictionsStore) -> None:
        """buffer_days=0 returns predictions exactly at horizon."""
        pred = _make_prediction(prediction_date="2026-01-01", horizon_days=5)
        await store.insert(pred)

        # Horizon = Jan 6. buffer=0 → eligible on Jan 6 itself.
        results = await store.query_pending_scoring(as_of_date="2026-01-06", buffer_days=0)
        assert len(results) == 1
