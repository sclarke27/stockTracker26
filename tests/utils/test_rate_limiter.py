"""Tests for the sliding-window rate limiter."""

from __future__ import annotations

import asyncio
import time

import pytest

from stock_radar.utils.rate_limiter import RateLimiter, RateLimitExceededError


class TestRateLimiter:
    """Tests for per-minute and per-day limiting."""

    async def test_first_request_passes_immediately(self) -> None:
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=500)
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    async def test_requests_within_limit_pass(self) -> None:
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=500)
        for _ in range(5):
            await limiter.acquire()

    async def test_exceeding_minute_limit_waits(self) -> None:
        limiter = RateLimiter(requests_per_minute=2, requests_per_day=500)
        # Use up the per-minute allowance.
        await limiter.acquire()
        await limiter.acquire()
        # The third request should wait. We verify it takes > 0 seconds
        # but we don't want to actually wait 60s in a test, so we use
        # wait_time() instead to check the limiter knows it needs to wait.
        wait = await limiter.wait_time()
        assert wait > 0

    async def test_daily_limit_raises(self) -> None:
        limiter = RateLimiter(requests_per_minute=100, requests_per_day=3)
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        with pytest.raises(RateLimitExceededError):
            await limiter.acquire()

    async def test_daily_remaining(self) -> None:
        limiter = RateLimiter(requests_per_minute=100, requests_per_day=10)
        assert limiter.daily_remaining == 10
        await limiter.acquire()
        assert limiter.daily_remaining == 9
        await limiter.acquire()
        assert limiter.daily_remaining == 8

    async def test_wait_time_zero_when_available(self) -> None:
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=500)
        wait = await limiter.wait_time()
        assert wait == 0.0

    async def test_concurrent_access(self) -> None:
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=500)

        async def acquire_one() -> None:
            await limiter.acquire()

        # Launch 5 concurrent acquisitions — all should succeed
        # since the limit is 5/min.
        await asyncio.gather(*[acquire_one() for _ in range(5)])
        assert limiter.daily_remaining == 495


class TestPerSecondLimiting:
    """Tests for per-second rate limiting (e.g. SEC EDGAR 10/sec)."""

    async def test_requests_within_second_limit_pass(self) -> None:
        limiter = RateLimiter(
            requests_per_minute=600,
            requests_per_day=50_000,
            requests_per_second=10,
        )
        for _ in range(10):
            await limiter.acquire()

    async def test_exceeding_second_limit_reports_wait(self) -> None:
        limiter = RateLimiter(
            requests_per_minute=600,
            requests_per_day=50_000,
            requests_per_second=2,
        )
        await limiter.acquire()
        await limiter.acquire()
        wait = await limiter.wait_time()
        assert wait > 0

    async def test_none_second_limit_skips_check(self) -> None:
        """When requests_per_second is None, no per-second check occurs."""
        limiter = RateLimiter(
            requests_per_minute=5,
            requests_per_day=500,
            requests_per_second=None,
        )
        # Should behave like the old limiter — 5 requests pass immediately.
        for _ in range(5):
            await limiter.acquire()

    async def test_backward_compatible_without_second_param(self) -> None:
        """Existing callers that don't pass requests_per_second still work."""
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=500)
        assert limiter._second_limit is None
        for _ in range(5):
            await limiter.acquire()
