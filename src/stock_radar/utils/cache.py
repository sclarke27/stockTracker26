"""Async SQLite cache with TTL support for API responses."""

from __future__ import annotations

import time

import aiosqlite


class Cache:
    """Async SQLite cache with per-entry TTL.

    Stores serialized JSON responses keyed by a deterministic string
    built from the tool name and its parameters. Expired entries are
    invisible to reads (lazy expiration) and can be bulk-removed with
    ``clear_expired()``.

    Args:
        db_path: Filesystem path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the database connection and create the cache table.

        Enables WAL mode for better concurrent read performance.
        """
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl INTEGER
            )
            """)
        await self._db.commit()

    async def get(self, key: str) -> str | None:
        """Retrieve cached data if the entry exists and has not expired.

        Args:
            key: Cache key.

        Returns:
            The cached JSON string, or ``None`` on a miss or expiration.
        """
        assert self._db is not None, "Cache.initialize() must be called before use"
        now = time.time()
        async with self._db.execute(
            """
            SELECT data FROM cache
            WHERE key = ?
              AND (ttl IS NULL OR created_at + ttl > ?)
            """,
            (key, now),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    async def set(self, key: str, data: str, ttl: int | None = None) -> None:
        """Store or overwrite a cache entry.

        Args:
            key: Cache key.
            data: Serialized JSON data to cache.
            ttl: Time-to-live in seconds, or ``None`` to never expire.
        """
        assert self._db is not None, "Cache.initialize() must be called before use"
        now = time.time()
        await self._db.execute(
            """
            INSERT INTO cache (key, data, created_at, ttl)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                data = excluded.data,
                created_at = excluded.created_at,
                ttl = excluded.ttl
            """,
            (key, data, now, ttl),
        )
        await self._db.commit()

    async def delete(self, key: str) -> None:
        """Delete a specific cache entry.

        Args:
            key: Cache key to remove. No error if the key does not exist.
        """
        assert self._db is not None, "Cache.initialize() must be called before use"
        await self._db.execute("DELETE FROM cache WHERE key = ?", (key,))
        await self._db.commit()

    async def clear_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        assert self._db is not None, "Cache.initialize() must be called before use"
        now = time.time()
        cursor = await self._db.execute(
            """
            DELETE FROM cache
            WHERE ttl IS NOT NULL AND created_at + ttl <= ?
            """,
            (now,),
        )
        await self._db.commit()
        return cursor.rowcount

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    @staticmethod
    def make_key(prefix: str, **kwargs: object) -> str:
        """Build a deterministic cache key from a prefix and parameters.

        Parameters are sorted alphabetically to ensure the same inputs
        always produce the same key regardless of argument order.

        Args:
            prefix: Key namespace (e.g. ``"quote"``, ``"price_history"``).
            **kwargs: Key-value pairs to include in the key.

        Returns:
            A string like ``"quote:ticker=AAPL"`` or
            ``"price_history:outputsize=compact&ticker=AAPL"``.
        """
        params = "&".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{prefix}:{params}"
