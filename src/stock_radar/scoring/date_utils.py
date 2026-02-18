"""Date alignment utilities for mapping target dates to trading days."""

from __future__ import annotations

from datetime import date, timedelta

# Maximum calendar days to search in any direction.
MAX_SEARCH_DAYS = 7


def find_closest_trading_day(
    target_date: str,
    price_map: dict[str, float],
    *,
    direction: str = "backward",
) -> str | None:
    """Find the closest available trading day to a target date.

    Markets are closed on weekends and holidays. Given a target date and a
    dict of ``{YYYY-MM-DD: close_price}`` from price history, this returns
    the nearest available trading day, searching in the specified direction.

    Args:
        target_date: ISO date string (YYYY-MM-DD) to seek.
        price_map: Dict mapping date strings to closing prices.
        direction: Search direction if target_date is not in price_map.
            ``"backward"`` (default) searches toward earlier dates.
            ``"forward"`` searches toward later dates.
            ``"nearest"`` tries backward first at each delta, then forward.

    Returns:
        The closest available date string, or ``None`` if no match is
        found within :data:`MAX_SEARCH_DAYS`.
    """
    # Exact match — no search needed.
    if target_date in price_map:
        return target_date

    target = date.fromisoformat(target_date)

    if direction == "backward":
        for delta in range(1, MAX_SEARCH_DAYS + 1):
            candidate = (target - timedelta(days=delta)).isoformat()
            if candidate in price_map:
                return candidate

    elif direction == "forward":
        for delta in range(1, MAX_SEARCH_DAYS + 1):
            candidate = (target + timedelta(days=delta)).isoformat()
            if candidate in price_map:
                return candidate

    elif direction == "nearest":
        for delta in range(1, MAX_SEARCH_DAYS + 1):
            backward = (target - timedelta(days=delta)).isoformat()
            if backward in price_map:
                return backward
            forward = (target + timedelta(days=delta)).isoformat()
            if forward in price_map:
                return forward

    return None


def build_price_map(bars: list[dict]) -> dict[str, float]:
    """Build a date-to-close-price lookup from a list of OHLCV bar dicts.

    Args:
        bars: List of dicts with ``date`` and ``close`` keys. Order does
            not matter — the result is a flat dict lookup.

    Returns:
        Dict mapping ISO date strings to closing prices.
    """
    return {bar["date"]: bar["close"] for bar in bars}
