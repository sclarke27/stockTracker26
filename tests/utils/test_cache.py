"""Tests for the async SQLite cache layer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from stock_radar.utils.cache import Cache


@pytest.fixture()
async def cache(tmp_path: Path) -> Cache:
    """Provide an initialized cache backed by a temporary database."""
    db_path = str(tmp_path / "test_cache.db")
    c = Cache(db_path)
    await c.initialize()
    yield c
    await c.close()


class TestCacheInitialize:
    """Tests for cache initialization."""

    async def test_initialize_creates_table(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "init_test.db")
        c = Cache(db_path)
        await c.initialize()
        # Should be able to get without error (table exists).
        result = await c.get("nonexistent")
        assert result is None
        await c.close()

    async def test_initialize_creates_db_file(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "new.db")
        assert not Path(db_path).exists()
        c = Cache(db_path)
        await c.initialize()
        assert Path(db_path).exists()
        await c.close()


class TestCacheGetSet:
    """Tests for basic get/set operations."""

    async def test_set_and_get_roundtrip(self, cache: Cache) -> None:
        data = json.dumps({"ticker": "AAPL", "price": 150.0})
        await cache.set("quote:ticker=AAPL", data, ttl=900)
        result = await cache.get("quote:ticker=AAPL")
        assert result == data

    async def test_get_nonexistent_key_returns_none(self, cache: Cache) -> None:
        result = await cache.get("does_not_exist")
        assert result is None

    async def test_overwrite_existing_key(self, cache: Cache) -> None:
        await cache.set("key1", "old_value", ttl=900)
        await cache.set("key1", "new_value", ttl=900)
        result = await cache.get("key1")
        assert result == "new_value"

    async def test_ttl_none_never_expires(self, cache: Cache) -> None:
        await cache.set("permanent", "data", ttl=None)
        result = await cache.get("permanent")
        assert result == "data"


class TestCacheTTL:
    """Tests for TTL expiration behavior."""

    async def test_expired_entry_returns_none(self, cache: Cache) -> None:
        # Set with a TTL of 0 seconds (immediately expired).
        await cache.set("expired_key", "stale", ttl=0)
        # Small sleep to ensure time passes.
        await asyncio.sleep(0.05)
        result = await cache.get("expired_key")
        assert result is None

    async def test_non_expired_entry_returns_data(self, cache: Cache) -> None:
        await cache.set("fresh_key", "fresh_data", ttl=3600)
        result = await cache.get("fresh_key")
        assert result == "fresh_data"


class TestCacheDelete:
    """Tests for cache deletion."""

    async def test_delete_existing_key(self, cache: Cache) -> None:
        await cache.set("to_delete", "data", ttl=900)
        await cache.delete("to_delete")
        result = await cache.get("to_delete")
        assert result is None

    async def test_delete_nonexistent_key_no_error(self, cache: Cache) -> None:
        # Should not raise.
        await cache.delete("nonexistent")


class TestCacheClearExpired:
    """Tests for expired entry cleanup."""

    async def test_clear_expired_removes_old_entries(self, cache: Cache) -> None:
        await cache.set("expired1", "data1", ttl=0)
        await cache.set("expired2", "data2", ttl=0)
        await cache.set("fresh", "data3", ttl=3600)
        await asyncio.sleep(0.05)
        removed = await cache.clear_expired()
        assert removed == 2
        # Fresh entry should still be there.
        assert await cache.get("fresh") == "data3"

    async def test_clear_expired_ignores_permanent(self, cache: Cache) -> None:
        await cache.set("permanent", "data", ttl=None)
        removed = await cache.clear_expired()
        assert removed == 0
        assert await cache.get("permanent") == "data"


class TestCacheMakeKey:
    """Tests for deterministic cache key generation."""

    def test_single_param(self) -> None:
        key = Cache.make_key("quote", ticker="AAPL")
        assert key == "quote:ticker=AAPL"

    def test_multiple_params_sorted(self) -> None:
        key = Cache.make_key("price_history", ticker="AAPL", outputsize="compact")
        assert key == "price_history:outputsize=compact&ticker=AAPL"

    def test_same_params_different_order_same_key(self) -> None:
        key1 = Cache.make_key("search", keywords="apple", limit="10")
        key2 = Cache.make_key("search", limit="10", keywords="apple")
        assert key1 == key2

    def test_no_params(self) -> None:
        key = Cache.make_key("all_tickers")
        assert key == "all_tickers:"
