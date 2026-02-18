"""Tests for watchlist loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from stock_radar.pipeline.watchlist import Watchlist, WatchlistTicker, load_watchlist


class TestWatchlistTicker:
    """Tests for the WatchlistTicker Pydantic model."""

    def test_symbol_only_defaults_name_to_empty(self) -> None:
        """Constructing with only a symbol should default name to an empty string."""
        ticker = WatchlistTicker(symbol="AAPL")
        assert ticker.symbol == "AAPL"
        assert ticker.name == ""

    def test_symbol_and_name(self) -> None:
        """Constructing with both symbol and name should store both."""
        ticker = WatchlistTicker(symbol="MSFT", name="Microsoft Corp")
        assert ticker.symbol == "MSFT"
        assert ticker.name == "Microsoft Corp"

    def test_missing_symbol_raises_validation_error(self) -> None:
        """Omitting the required symbol field should raise ValidationError."""
        with pytest.raises(ValidationError):
            WatchlistTicker()  # type: ignore[call-arg]


class TestWatchlist:
    """Tests for the Watchlist Pydantic model."""

    def test_deep_and_light_lists(self) -> None:
        """Constructing with both deep and light lists should store them."""
        deep = [WatchlistTicker(symbol="AAPL", name="Apple Inc")]
        light = [WatchlistTicker(symbol="AMD", name="Advanced Micro Devices")]
        watchlist = Watchlist(deep=deep, light=light)
        assert len(watchlist.deep) == 1
        assert len(watchlist.light) == 1
        assert watchlist.deep[0].symbol == "AAPL"
        assert watchlist.light[0].symbol == "AMD"

    def test_light_defaults_to_empty_list(self) -> None:
        """Omitting light should default to an empty list."""
        deep = [WatchlistTicker(symbol="AAPL")]
        watchlist = Watchlist(deep=deep)
        assert watchlist.light == []

    def test_missing_deep_raises_validation_error(self) -> None:
        """Omitting the required deep field should raise ValidationError."""
        with pytest.raises(ValidationError):
            Watchlist()  # type: ignore[call-arg]


class TestLoadWatchlist:
    """Tests for the load_watchlist function."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """Loading a well-formed YAML file should return a populated Watchlist."""
        content = (
            "deep:\n"
            "  - symbol: AAPL\n"
            "    name: Apple Inc\n"
            "  - symbol: MSFT\n"
            "    name: Microsoft Corp\n"
            "light:\n"
            "  - symbol: AMD\n"
            "    name: Advanced Micro Devices\n"
        )
        watchlist_file = tmp_path / "watchlist.yaml"
        watchlist_file.write_text(content)

        result = load_watchlist(watchlist_file)

        assert len(result.deep) == 2
        assert result.deep[0].symbol == "AAPL"
        assert result.deep[0].name == "Apple Inc"
        assert result.deep[1].symbol == "MSFT"
        assert result.deep[1].name == "Microsoft Corp"
        assert len(result.light) == 1
        assert result.light[0].symbol == "AMD"
        assert result.light[0].name == "Advanced Micro Devices"

    def test_load_without_light_defaults_to_empty(self, tmp_path: Path) -> None:
        """Loading a YAML file with no light section should default light to an empty list."""
        content = "deep:\n" "  - symbol: NVDA\n" "    name: NVIDIA Corp\n"
        watchlist_file = tmp_path / "watchlist.yaml"
        watchlist_file.write_text(content)

        result = load_watchlist(watchlist_file)

        assert len(result.deep) == 1
        assert result.deep[0].symbol == "NVDA"
        assert result.light == []

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Passing a path to a non-existent file should raise FileNotFoundError."""
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_watchlist(missing)

    def test_load_with_name_omitted_defaults_to_empty(self, tmp_path: Path) -> None:
        """Tickers without a name field should default name to an empty string."""
        content = (
            "deep:\n"
            "  - symbol: AAPL\n"
            "    name: Apple Inc\n"
            "  - symbol: GOOG\n"
            "light:\n"
            "  - symbol: TSLA\n"
        )
        watchlist_file = tmp_path / "watchlist.yaml"
        watchlist_file.write_text(content)

        result = load_watchlist(watchlist_file)

        assert result.deep[0].name == "Apple Inc"
        assert result.deep[1].symbol == "GOOG"
        assert result.deep[1].name == ""
        assert result.light[0].symbol == "TSLA"
        assert result.light[0].name == ""
