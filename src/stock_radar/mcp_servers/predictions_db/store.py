"""Async SQLite store for prediction records and scoring."""

from __future__ import annotations

import aiosqlite
from loguru import logger

from stock_radar.mcp_servers.predictions_db.config import SERVER_NAME

_log = logger.bind(server=SERVER_NAME)

# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                      TEXT PRIMARY KEY,
    ticker                  TEXT NOT NULL,
    agent_name              TEXT NOT NULL,
    signal_type             TEXT NOT NULL,
    direction               TEXT NOT NULL,
    confidence              REAL NOT NULL,
    reasoning               TEXT NOT NULL,
    prediction_date         TEXT NOT NULL,
    horizon_days            INTEGER NOT NULL,
    created_at              TEXT NOT NULL,
    scored_at               TEXT,
    actual_price_close      REAL,
    actual_price_at_horizon REAL,
    return_pct              REAL,
    status                  TEXT
)
"""

_CREATE_INDEXES_SQL: list[str] = [
    """
    CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date
        ON predictions (ticker, prediction_date)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_predictions_agent_signal
        ON predictions (agent_name, signal_type)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_predictions_status
        ON predictions (status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_predictions_created_desc
        ON predictions (created_at DESC)
    """,
]

# Columns written during insertion (excludes scoring columns).
_INSERT_COLUMNS = [
    "id",
    "ticker",
    "agent_name",
    "signal_type",
    "direction",
    "confidence",
    "reasoning",
    "prediction_date",
    "horizon_days",
    "created_at",
]


class PredictionsStore:
    """Async SQLite store for prediction lifecycle management.

    Handles insertion of new predictions, scoring updates, filtered queries,
    and accuracy-statistics aggregation.  Uses WAL mode and row-factory for
    ergonomic dict-like access.

    Args:
        db_path: Filesystem path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open the database connection, enable WAL, and create schema.

        Safe to call multiple times (``CREATE … IF NOT EXISTS``).
        """
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(_CREATE_TABLE_SQL)
        for idx_sql in _CREATE_INDEXES_SQL:
            await self._db.execute(idx_sql)
        await self._db.commit()
        _log.info("Predictions store initialized at {path}", path=self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def insert(self, record: dict) -> None:
        """Insert a new prediction record.

        Only the 10 core (non-scoring) columns are written; scoring fields
        are left ``NULL`` until :meth:`update_score` is called.

        Args:
            record: Dictionary whose keys match ``_INSERT_COLUMNS``.

        Raises:
            aiosqlite.IntegrityError: If the ``id`` already exists.
        """
        placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
        columns = ", ".join(_INSERT_COLUMNS)
        values = tuple(record[col] for col in _INSERT_COLUMNS)
        await self._db.execute(
            f"INSERT INTO predictions ({columns}) VALUES ({placeholders})",
            values,
        )
        await self._db.commit()
        _log.debug(
            "Inserted prediction {id} for {ticker}",
            id=record["id"],
            ticker=record["ticker"],
        )

    async def update_score(
        self,
        prediction_id: str,
        scored_at: str,
        actual_price_close: float,
        actual_price_at_horizon: float,
        return_pct: float,
        status: str,
    ) -> bool:
        """Update the scoring columns of an existing prediction.

        Args:
            prediction_id: Primary-key ``id`` of the prediction.
            scored_at: ISO-8601 timestamp of when scoring occurred.
            actual_price_close: Closing price on the prediction date.
            actual_price_at_horizon: Closing price at the horizon date.
            return_pct: Percentage return over the horizon.
            status: Outcome label (e.g. ``CORRECT``, ``INCORRECT``,
                ``PARTIAL``).

        Returns:
            ``True`` if a row was updated, ``False`` if the id was not found.
        """
        cursor = await self._db.execute(
            """
            UPDATE predictions
            SET scored_at = ?,
                actual_price_close = ?,
                actual_price_at_horizon = ?,
                return_pct = ?,
                status = ?
            WHERE id = ?
            """,
            (
                scored_at,
                actual_price_close,
                actual_price_at_horizon,
                return_pct,
                status,
                prediction_id,
            ),
        )
        await self._db.commit()
        updated = cursor.rowcount > 0
        if updated:
            _log.debug(
                "Scored prediction {id} as {status}",
                id=prediction_id,
                status=status,
            )
        return updated

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_by_id(self, prediction_id: str) -> dict | None:
        """Retrieve a single prediction by its id.

        Args:
            prediction_id: Primary-key ``id`` to look up.

        Returns:
            A dict of the row, or ``None`` if not found.
        """
        async with self._db.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (prediction_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def query(
        self,
        *,
        ticker: str | None = None,
        agent_name: str | None = None,
        signal_type: str | None = None,
        scored_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Query predictions with optional filters and pagination.

        Args:
            ticker: Filter by stock ticker symbol.
            agent_name: Filter by the agent that produced the prediction.
            signal_type: Filter by signal type.
            scored_only: When ``True``, return only scored predictions.
            limit: Maximum rows to return.
            offset: Number of rows to skip (for pagination).

        Returns:
            A tuple of ``(rows, total_count)`` where *total_count* reflects
            all matching rows regardless of *limit*/*offset*.
        """
        where_clauses: list[str] = []
        params: list[object] = []

        if ticker is not None:
            where_clauses.append("ticker = ?")
            params.append(ticker)
        if agent_name is not None:
            where_clauses.append("agent_name = ?")
            params.append(agent_name)
        if signal_type is not None:
            where_clauses.append("signal_type = ?")
            params.append(signal_type)
        if scored_only:
            where_clauses.append("scored_at IS NOT NULL")

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Total count (ignores pagination).
        async with self._db.execute(
            f"SELECT COUNT(*) FROM predictions {where_sql}",
            params,
        ) as cursor:
            count_row = await cursor.fetchone()
            total = count_row[0]

        # Paginated data.
        async with self._db.execute(
            f"SELECT * FROM predictions {where_sql} " f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ) as cursor:
            rows = [dict(r) for r in await cursor.fetchall()]

        return rows, total

    async def query_pending_scoring(
        self,
        *,
        as_of_date: str,
        buffer_days: int = 1,
    ) -> list[dict]:
        """Query unscored predictions whose horizon has elapsed.

        Finds predictions where scored_at IS NULL and the horizon date
        (prediction_date + horizon_days) is at least buffer_days before
        as_of_date. Results are ordered oldest prediction_date first.

        Args:
            as_of_date: Reference date string (YYYY-MM-DD), typically today.
            buffer_days: Minimum days past the horizon before scoring.
                Defaults to 1 to ensure market data is available.

        Returns:
            List of prediction dicts ready for scoring, oldest first.
        """
        async with self._db.execute(
            """
            SELECT *
            FROM predictions
            WHERE scored_at IS NULL
              AND date(prediction_date, '+' || horizon_days || ' days')
                  <= date(?, '-' || ? || ' days')
            ORDER BY prediction_date ASC
            """,
            (as_of_date, buffer_days),
        ) as cursor:
            return [dict(r) for r in await cursor.fetchall()]

    async def get_accuracy_stats(
        self,
        *,
        agent_name: str | None = None,
        signal_type: str | None = None,
        since_date: str | None = None,
    ) -> list[dict]:
        """Compute accuracy statistics grouped by agent and signal type.

        Args:
            agent_name: Restrict to a single agent.
            signal_type: Restrict to a single signal type.
            since_date: Only include predictions scored on or after this
                ISO-8601 date string (e.g. ``"2026-02-01"``).

        Returns:
            A list of dicts, one per ``(agent_name, signal_type)`` group,
            each containing ``total``, ``scored``, ``correct``,
            ``accuracy_pct``, ``avg_confidence``, and
            ``avg_return_when_correct``.
        """
        where_clauses: list[str] = []
        params: list[object] = []

        if agent_name is not None:
            where_clauses.append("agent_name = ?")
            params.append(agent_name)
        if signal_type is not None:
            where_clauses.append("signal_type = ?")
            params.append(signal_type)
        if since_date is not None:
            where_clauses.append("prediction_date >= ?")
            params.append(since_date)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT
                agent_name,
                signal_type,
                COUNT(*) AS total,
                SUM(CASE WHEN scored_at IS NOT NULL THEN 1 ELSE 0 END)
                    AS scored,
                SUM(CASE WHEN status = 'CORRECT' THEN 1 ELSE 0 END)
                    AS correct,
                CASE
                    WHEN SUM(CASE WHEN scored_at IS NOT NULL THEN 1 ELSE 0 END) = 0
                    THEN 0.0
                    ELSE ROUND(
                        100.0
                        * SUM(CASE WHEN status = 'CORRECT' THEN 1 ELSE 0 END)
                        / SUM(CASE WHEN scored_at IS NOT NULL THEN 1 ELSE 0 END),
                        2
                    )
                END AS accuracy_pct,
                ROUND(AVG(confidence), 4) AS avg_confidence,
                CASE
                    WHEN SUM(CASE WHEN status = 'CORRECT' THEN 1 ELSE 0 END) = 0
                    THEN 0.0
                    ELSE ROUND(
                        SUM(
                            CASE WHEN status = 'CORRECT' THEN return_pct ELSE 0 END
                        )
                        / SUM(CASE WHEN status = 'CORRECT' THEN 1 ELSE 0 END),
                        4
                    )
                END AS avg_return_when_correct
            FROM predictions
            {where_sql}
            GROUP BY agent_name, signal_type
        """

        async with self._db.execute(sql, params) as cursor:
            return [dict(r) for r in await cursor.fetchall()]
