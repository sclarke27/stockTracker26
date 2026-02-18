"""Tests for the pipeline runner that orchestrates data ingestion."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stock_radar.pipeline.config import TIER_DEEP, TIER_LIGHT
from stock_radar.pipeline.models import PipelineResult, TickerResult, ToolCallResult
from stock_radar.pipeline.runner import run_pipeline
from stock_radar.pipeline.watchlist import Watchlist, WatchlistTicker


def _make_ticker_result(
    ticker: str,
    tier: str,
    success: int,
    errors: int,
) -> TickerResult:
    """Build a TickerResult with the specified success/error counts.

    Args:
        ticker: Stock ticker symbol.
        tier: Coverage tier (deep or light).
        success: Number of successful tool calls.
        errors: Number of failed tool calls.

    Returns:
        TickerResult populated with mock ToolCallResults.
    """
    results: list[ToolCallResult] = []
    for i in range(success):
        results.append(
            ToolCallResult(
                tool_name=f"tool_{i}",
                ticker=ticker,
                success=True,
                duration_ms=10.0,
            )
        )
    for i in range(errors):
        results.append(
            ToolCallResult(
                tool_name=f"error_tool_{i}",
                ticker=ticker,
                success=False,
                error="mock error",
                duration_ms=5.0,
            )
        )
    return TickerResult(
        ticker=ticker,
        tier=tier,
        results=results,
        success_count=success,
        error_count=errors,
    )


def _make_watchlist(
    deep_symbols: list[str],
    light_symbols: list[str] | None = None,
) -> Watchlist:
    """Build a Watchlist from symbol lists.

    Args:
        deep_symbols: Ticker symbols for the deep tier.
        light_symbols: Ticker symbols for the light tier (defaults to empty).

    Returns:
        Watchlist with WatchlistTicker entries.
    """
    deep = [WatchlistTicker(symbol=s) for s in deep_symbols]
    light = [WatchlistTicker(symbol=s) for s in (light_symbols or [])]
    return Watchlist(deep=deep, light=light)


@pytest.fixture()
def _mock_pipeline_deps():
    """Mock all external dependencies for run_pipeline tests.

    Patches server creation, Client context managers, current_quarter,
    and setup_logging so that tests never touch real MCP servers.
    """
    # Each Client() call must return a separate async context manager instance.
    # We use side_effect to produce a fresh one per call.
    mock_market_client = AsyncMock()
    mock_edgar_client = AsyncMock()

    def _client_factory(*args, **kwargs):
        """Return a new async context manager mock for each Client() call."""
        ctx = AsyncMock()
        # First call is for market server, second for edgar server.
        # We distinguish by tracking call order via side_effect list.
        return ctx

    # Build two distinct context managers — one per server.
    market_ctx = AsyncMock()
    market_ctx.__aenter__ = AsyncMock(return_value=mock_market_client)
    market_ctx.__aexit__ = AsyncMock(return_value=False)

    edgar_ctx = AsyncMock()
    edgar_ctx.__aenter__ = AsyncMock(return_value=mock_edgar_client)
    edgar_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client_cls = MagicMock(side_effect=[market_ctx, edgar_ctx])

    with (
        patch("stock_radar.pipeline.runner.create_market_server"),
        patch("stock_radar.pipeline.runner.create_edgar_server"),
        patch("stock_radar.pipeline.runner.Client", mock_client_cls),
        patch("stock_radar.pipeline.runner.current_quarter", return_value=(4, 2025)),
        patch("stock_radar.pipeline.runner.setup_logging"),
    ):
        yield


# ---------------------------------------------------------------------------
# TestRunPipeline
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_mock_pipeline_deps")
class TestRunPipeline:
    """Tests for the run_pipeline orchestrator."""

    async def test_processes_deep_and_light_tickers(self) -> None:
        """Deep and light fetchers are called the correct number of times."""
        watchlist = _make_watchlist(["AAPL", "MSFT"], ["NVDA"])
        deep_result = _make_ticker_result("X", TIER_DEEP, success=6, errors=0)
        light_result = _make_ticker_result("X", TIER_LIGHT, success=2, errors=0)

        with (
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                return_value=deep_result,
            ) as mock_deep,
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
                return_value=light_result,
            ) as mock_light,
        ):
            await run_pipeline(watchlist=watchlist)

            assert mock_deep.call_count == 2
            assert mock_light.call_count == 1

    async def test_returns_pipeline_result(self) -> None:
        """run_pipeline returns a PipelineResult with correct tickers_processed."""
        watchlist = _make_watchlist(["AAPL"], ["NVDA"])
        deep_result = _make_ticker_result("AAPL", TIER_DEEP, success=6, errors=0)
        light_result = _make_ticker_result("NVDA", TIER_LIGHT, success=2, errors=0)

        with (
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                return_value=deep_result,
            ),
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
                return_value=light_result,
            ),
        ):
            result = await run_pipeline(watchlist=watchlist)

            assert isinstance(result, PipelineResult)
            assert result.tickers_processed == 2

    async def test_total_calls_aggregated(self) -> None:
        """total_calls is the sum of all tool call results across tickers."""
        watchlist = _make_watchlist(["AAPL"], ["NVDA"])
        deep_result = _make_ticker_result("AAPL", TIER_DEEP, success=5, errors=1)
        light_result = _make_ticker_result("NVDA", TIER_LIGHT, success=2, errors=0)

        with (
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                return_value=deep_result,
            ),
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
                return_value=light_result,
            ),
        ):
            result = await run_pipeline(watchlist=watchlist)

            # deep: 5+1=6 results, light: 2 results → 8 total
            assert result.total_calls == 8

    async def test_total_errors_aggregated(self) -> None:
        """total_errors is the sum of error_count across all ticker results."""
        watchlist = _make_watchlist(["AAPL", "MSFT"])
        deep_aapl = _make_ticker_result("AAPL", TIER_DEEP, success=4, errors=2)
        deep_msft = _make_ticker_result("MSFT", TIER_DEEP, success=5, errors=1)

        call_count = 0

        async def _fetch_deep_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return deep_aapl if call_count == 1 else deep_msft

        with (
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                side_effect=_fetch_deep_side_effect,
            ),
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
            ),
        ):
            result = await run_pipeline(watchlist=watchlist)

            assert result.total_errors == 3  # 2 + 1

    async def test_empty_watchlist(self) -> None:
        """A watchlist with no tickers produces an empty result."""
        watchlist = _make_watchlist([], [])

        with (
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
            ) as mock_deep,
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
            ) as mock_light,
        ):
            result = await run_pipeline(watchlist=watchlist)

            assert result.tickers_processed == 0
            assert result.total_calls == 0
            assert result.total_errors == 0
            mock_deep.assert_not_called()
            mock_light.assert_not_called()

    async def test_deep_only_watchlist(self) -> None:
        """A watchlist with only deep tickers produces correct results."""
        watchlist = _make_watchlist(["AAPL", "MSFT"])
        deep_result = _make_ticker_result("X", TIER_DEEP, success=6, errors=0)

        with (
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                return_value=deep_result,
            ) as mock_deep,
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
            ) as mock_light,
        ):
            result = await run_pipeline(watchlist=watchlist)

            assert mock_deep.call_count == 2
            mock_light.assert_not_called()
            assert result.tickers_processed == 2
            assert result.total_calls == 12  # 6 * 2

    async def test_light_heavy_watchlist(self) -> None:
        """A watchlist with one deep and multiple light tickers works correctly."""
        watchlist = _make_watchlist(["AAPL"], ["NVDA", "AMD", "GOOG"])
        deep_result = _make_ticker_result("AAPL", TIER_DEEP, success=6, errors=0)
        light_result = _make_ticker_result("X", TIER_LIGHT, success=2, errors=0)

        with (
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                return_value=deep_result,
            ) as mock_deep,
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
                return_value=light_result,
            ) as mock_light,
        ):
            result = await run_pipeline(watchlist=watchlist)

            assert mock_deep.call_count == 1
            assert mock_light.call_count == 3
            assert result.tickers_processed == 4
            assert result.total_calls == 12  # 6 + 2*3

    async def test_uses_current_quarter_for_deep(self) -> None:
        """fetch_deep receives the quarter and year from current_quarter()."""
        watchlist = _make_watchlist(["TSLA"])
        deep_result = _make_ticker_result("TSLA", TIER_DEEP, success=6, errors=0)

        with (
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                return_value=deep_result,
            ) as mock_deep,
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
            ),
        ):
            await run_pipeline(watchlist=watchlist)

            mock_deep.assert_called_once()
            call_args = mock_deep.call_args
            # Positional args: (market_client, edgar_client, ticker, quarter, year)
            assert call_args.args[2] == "TSLA"
            assert call_args.args[3] == 4  # quarter from mocked current_quarter
            assert call_args.args[4] == 2025  # year from mocked current_quarter

    async def test_loads_default_watchlist_when_none(self) -> None:
        """When watchlist=None, load_watchlist() is called to get the default."""
        default_watchlist = _make_watchlist(["AAPL"])
        deep_result = _make_ticker_result("AAPL", TIER_DEEP, success=6, errors=0)

        with (
            patch(
                "stock_radar.pipeline.runner.load_watchlist",
                return_value=default_watchlist,
            ) as mock_load,
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                return_value=deep_result,
            ),
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
            ),
        ):
            result = await run_pipeline(watchlist=None)

            mock_load.assert_called_once()
            assert result.tickers_processed == 1

    async def test_uses_provided_watchlist(self) -> None:
        """When a watchlist is passed directly, load_watchlist() is NOT called."""
        watchlist = _make_watchlist(["AAPL"])
        deep_result = _make_ticker_result("AAPL", TIER_DEEP, success=6, errors=0)

        with (
            patch(
                "stock_radar.pipeline.runner.load_watchlist",
            ) as mock_load,
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                return_value=deep_result,
            ),
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
            ),
        ):
            await run_pipeline(watchlist=watchlist)

            mock_load.assert_not_called()

    async def test_timestamps_are_iso_format(self) -> None:
        """started_at and completed_at should be valid ISO 8601 timestamps."""
        watchlist = _make_watchlist(["AAPL"])
        deep_result = _make_ticker_result("AAPL", TIER_DEEP, success=6, errors=0)

        with (
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                return_value=deep_result,
            ),
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
            ),
        ):
            result = await run_pipeline(watchlist=watchlist)

            # Both should parse without raising ValueError.
            start_dt = datetime.fromisoformat(result.started_at)
            end_dt = datetime.fromisoformat(result.completed_at)
            assert start_dt is not None
            assert end_dt is not None

    async def test_duration_is_non_negative(self) -> None:
        """duration_seconds should be >= 0."""
        watchlist = _make_watchlist(["AAPL"])
        deep_result = _make_ticker_result("AAPL", TIER_DEEP, success=6, errors=0)

        with (
            patch(
                "stock_radar.pipeline.runner.fetch_deep",
                new_callable=AsyncMock,
                return_value=deep_result,
            ),
            patch(
                "stock_radar.pipeline.runner.fetch_light",
                new_callable=AsyncMock,
            ),
        ):
            result = await run_pipeline(watchlist=watchlist)

            assert result.duration_seconds >= 0
