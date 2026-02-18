"""Tests for scoring date alignment utilities."""

from __future__ import annotations

from stock_radar.scoring.date_utils import build_price_map, find_closest_trading_day


class TestFindClosestTradingDay:
    """Tests for the find_closest_trading_day function."""

    def test_exact_match_returns_target(self) -> None:
        """Target date found in price_map is returned as-is."""
        price_map = {"2026-01-15": 150.0, "2026-01-16": 151.0}
        result = find_closest_trading_day("2026-01-15", price_map)
        assert result == "2026-01-15"

    def test_backward_finds_prior_trading_day(self) -> None:
        """Saturday maps to prior Friday with backward direction."""
        # Jan 17, 2026 is a Saturday
        price_map = {"2026-01-16": 150.0}  # Friday
        result = find_closest_trading_day("2026-01-17", price_map, direction="backward")
        assert result == "2026-01-16"

    def test_backward_skips_weekend(self) -> None:
        """Sunday maps to prior Friday with backward direction."""
        # Jan 18, 2026 is a Sunday
        price_map = {"2026-01-16": 150.0}  # Friday
        result = find_closest_trading_day("2026-01-18", price_map, direction="backward")
        assert result == "2026-01-16"

    def test_forward_finds_next_trading_day(self) -> None:
        """Saturday maps to next Monday with forward direction."""
        # Jan 17, 2026 is a Saturday
        price_map = {"2026-01-19": 152.0}  # Monday
        result = find_closest_trading_day("2026-01-17", price_map, direction="forward")
        assert result == "2026-01-19"

    def test_nearest_prefers_backward(self) -> None:
        """Nearest direction checks backward first at equal distance."""
        price_map = {
            "2026-01-14": 149.0,  # 1 day before
            "2026-01-16": 151.0,  # 1 day after
        }
        result = find_closest_trading_day("2026-01-15", price_map, direction="nearest")
        assert result == "2026-01-14"

    def test_nearest_finds_forward_when_no_backward(self) -> None:
        """Nearest falls through to forward when no backward match."""
        price_map = {"2026-01-20": 155.0}  # 5 days after
        result = find_closest_trading_day("2026-01-15", price_map, direction="nearest")
        assert result == "2026-01-20"

    def test_returns_none_when_not_found(self) -> None:
        """No matching dates within search window returns None."""
        price_map = {"2025-01-01": 100.0}  # Way too far
        result = find_closest_trading_day("2026-01-15", price_map)
        assert result is None

    def test_max_search_days_boundary(self) -> None:
        """Date exactly 7 days away is found; 8 days away is not."""
        price_map = {"2026-01-08": 100.0}  # Exactly 7 days before Jan 15
        result = find_closest_trading_day("2026-01-15", price_map, direction="backward")
        assert result == "2026-01-08"

        price_map_far = {"2026-01-07": 100.0}  # 8 days before
        result_far = find_closest_trading_day("2026-01-15", price_map_far, direction="backward")
        assert result_far is None

    def test_holiday_gap_finds_nearest(self) -> None:
        """Multi-day gap (e.g., holiday + weekend) still finds nearest."""
        # 3 consecutive non-trading days
        price_map = {"2026-01-12": 148.0}  # Monday before a gap
        result = find_closest_trading_day("2026-01-15", price_map, direction="backward")
        assert result == "2026-01-12"

    def test_empty_price_map_returns_none(self) -> None:
        """Empty price map returns None."""
        result = find_closest_trading_day("2026-01-15", {})
        assert result is None

    def test_default_direction_is_backward(self) -> None:
        """Default direction parameter is backward."""
        price_map = {"2026-01-14": 149.0}
        result = find_closest_trading_day("2026-01-15", price_map)
        assert result == "2026-01-14"


class TestBuildPriceMap:
    """Tests for the build_price_map function."""

    def test_basic_mapping(self) -> None:
        """Builds correct date-to-close mapping."""
        bars = [
            {"date": "2026-01-15", "close": 150.0, "open": 149.0},
            {"date": "2026-01-16", "close": 151.0, "open": 150.0},
        ]
        result = build_price_map(bars)
        assert result == {"2026-01-15": 150.0, "2026-01-16": 151.0}

    def test_empty_bars(self) -> None:
        """Empty bar list returns empty dict."""
        assert build_price_map([]) == {}

    def test_ignores_extra_fields(self) -> None:
        """Extra fields in bars are ignored."""
        bars = [
            {"date": "2026-01-15", "close": 150.0, "open": 149.0, "volume": 1000000},
        ]
        result = build_price_map(bars)
        assert result == {"2026-01-15": 150.0}
