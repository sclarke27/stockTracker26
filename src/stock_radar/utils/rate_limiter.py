"""Sliding-window rate limiter for external API calls."""

from __future__ import annotations

import asyncio
import time
from collections import deque


class RateLimitExceededError(Exception):
    """Raised when an API rate limit is exhausted and cannot recover by waiting."""


class RateLimiter:
    """Async-safe sliding-window rate limiter.

    Tracks request timestamps and enforces per-second, per-minute, and
    per-day limits. When a short-window limit (second or minute) is hit,
    ``acquire()`` sleeps until a slot opens. When the per-day limit is
    hit, it raises ``RateLimitExceededError`` (cannot recover by waiting).

    Args:
        requests_per_minute: Maximum requests allowed per 60-second window.
        requests_per_day: Maximum requests allowed per 24-hour window.
        requests_per_second: Maximum requests per 1-second window, or
            ``None`` to skip per-second limiting.
    """

    _SECOND_WINDOW = 1.0
    _MINUTE_WINDOW = 60.0
    _DAY_WINDOW = 86_400.0

    def __init__(
        self,
        requests_per_minute: int,
        requests_per_day: int,
        requests_per_second: int | None = None,
    ) -> None:
        self._second_limit = requests_per_second
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

                # Check per-second limit first (tightest window).
                if self._second_limit is not None:
                    second_count = self._count_in_window(now, self._SECOND_WINDOW)
                    if second_count >= self._second_limit:
                        wait = self._compute_window_wait(now, self._SECOND_WINDOW)
                        # Release lock before sleeping.
                        await asyncio.sleep(0)  # Yield to event loop
                        break_to_sleep = wait
                    else:
                        break_to_sleep = None
                else:
                    break_to_sleep = None

                if break_to_sleep is not None:
                    pass  # Will sleep below after releasing lock
                else:
                    minute_count = self._count_in_window(now, self._MINUTE_WINDOW)
                    if minute_count < self._minute_limit:
                        self._timestamps.append(now)
                        return

                    break_to_sleep = self._compute_window_wait(now, self._MINUTE_WINDOW)

            # Sleep outside the lock so other coroutines can check state.
            await asyncio.sleep(break_to_sleep)

    async def wait_time(self) -> float:
        """Return seconds until the next request slot is available.

        Returns:
            0.0 if a slot is immediately available, otherwise the number
            of seconds to wait for a window to open.
        """
        async with self._lock:
            now = time.monotonic()
            self._purge_old(now)

            # Check per-second window first.
            if self._second_limit is not None:
                second_count = self._count_in_window(now, self._SECOND_WINDOW)
                if second_count >= self._second_limit:
                    return self._compute_window_wait(now, self._SECOND_WINDOW)

            minute_count = self._count_in_window(now, self._MINUTE_WINDOW)
            if minute_count < self._minute_limit:
                return 0.0

            return self._compute_window_wait(now, self._MINUTE_WINDOW)

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

    def _compute_window_wait(self, now: float, window: float) -> float:
        """Compute how long to wait for a slot to open in the given window."""
        cutoff = now - window
        # Find the oldest timestamp in the window.
        for ts in self._timestamps:
            if ts >= cutoff:
                return window - (now - ts) + 0.01
        return 0.0
