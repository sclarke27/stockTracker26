"""Sliding-window rate limiter for external API calls."""

from __future__ import annotations

import asyncio
import time
from collections import deque

from stock_radar.mcp_servers.market_data.exceptions import RateLimitExceededError


class RateLimiter:
    """Async-safe sliding-window rate limiter.

    Tracks request timestamps and enforces both per-minute and per-day
    limits. When the per-minute limit is hit, ``acquire()`` sleeps until
    a slot opens. When the per-day limit is hit, it raises
    ``RateLimitExceededError`` (cannot recover by waiting).

    Args:
        requests_per_minute: Maximum requests allowed per 60-second window.
        requests_per_day: Maximum requests allowed per 24-hour window.
    """

    _MINUTE_WINDOW = 60.0
    _DAY_WINDOW = 86_400.0

    def __init__(
        self,
        requests_per_minute: int,
        requests_per_day: int,
    ) -> None:
        self._minute_limit = requests_per_minute
        self._daily_limit = requests_per_day
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available, then record the request.

        Raises:
            RateLimitExceededError: If the daily limit is exhausted.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                self._purge_old(now)

                if len(self._timestamps) >= self._daily_limit:
                    raise RateLimitExceededError(
                        f"Daily API limit of {self._daily_limit} requests exhausted."
                    )

                minute_count = self._count_in_window(now, self._MINUTE_WINDOW)
                if minute_count < self._minute_limit:
                    self._timestamps.append(now)
                    return

                wait = self._compute_minute_wait(now)

            # Sleep outside the lock so other coroutines can check state.
            await asyncio.sleep(wait)

    async def wait_time(self) -> float:
        """Return seconds until the next request slot is available.

        Returns:
            0.0 if a slot is immediately available, otherwise the number
            of seconds to wait for the per-minute window to open.
        """
        async with self._lock:
            now = time.monotonic()
            self._purge_old(now)

            minute_count = self._count_in_window(now, self._MINUTE_WINDOW)
            if minute_count < self._minute_limit:
                return 0.0

            return self._compute_minute_wait(now)

    @property
    def daily_remaining(self) -> int:
        """Number of remaining requests in the current 24-hour window."""
        now = time.monotonic()
        self._purge_old(now)
        return max(0, self._daily_limit - len(self._timestamps))

    def _purge_old(self, now: float) -> None:
        """Remove timestamps older than the daily window."""
        cutoff = now - self._DAY_WINDOW
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def _count_in_window(self, now: float, window: float) -> int:
        """Count timestamps within the given time window."""
        cutoff = now - window
        return sum(1 for ts in self._timestamps if ts >= cutoff)

    def _compute_minute_wait(self, now: float) -> float:
        """Compute how long to wait for a minute-window slot to open."""
        cutoff = now - self._MINUTE_WINDOW
        # Find the oldest timestamp in the minute window.
        for ts in self._timestamps:
            if ts >= cutoff:
                return self._MINUTE_WINDOW - (now - ts) + 0.01
        return 0.0
