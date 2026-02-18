"""Tests for data fetchers that call MCP server tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import Client

from stock_radar.pipeline.config import (
    DEEP_EDGAR_TOOLS,
    DEEP_MARKET_TOOLS,
    LIGHT_EDGAR_TOOLS,
    LIGHT_MARKET_TOOLS,
)
from stock_radar.pipeline.fetchers import _safe_call, fetch_deep, fetch_light
from stock_radar.pipeline.models import TickerResult, ToolCallResult


def _make_client(*, fail_tools: set[str] | None = None) -> AsyncMock:
    """Create a mock FastMCP Client.

    Args:
        fail_tools: Set of tool names that should raise on call.

    Returns:
        AsyncMock configured to behave like a FastMCP Client.
    """
    client = AsyncMock(spec=Client)
    fail_tools = fail_tools or set()

    async def mock_call_tool(tool_name: str, args: dict | None = None) -> MagicMock:
        if tool_name in fail_tools:
            raise Exception(f"Simulated failure for {tool_name}")
        result = MagicMock()
        result.content = [MagicMock(text='{"status": "ok"}')]
        return result

    client.call_tool = AsyncMock(side_effect=mock_call_tool)
    return client


# ---------------------------------------------------------------------------
# TestSafeCall
# ---------------------------------------------------------------------------


class TestSafeCall:
    """Tests for the _safe_call helper function."""

    async def test_successful_call(self) -> None:
        """A successful tool call returns success=True and error=None."""
        client = _make_client()

        result = await _safe_call(client, "get_quote", {"ticker": "AAPL"}, "AAPL")

        assert isinstance(result, ToolCallResult)
        assert result.success is True
        assert result.error is None
        assert result.tool_name == "get_quote"
        assert result.ticker == "AAPL"

    async def test_failed_call_returns_error(self) -> None:
        """When client.call_tool raises, returns success=False with error message."""
        client = _make_client(fail_tools={"get_quote"})

        result = await _safe_call(client, "get_quote", {"ticker": "AAPL"}, "AAPL")

        assert result.success is False
        assert result.error is not None
        assert "Simulated failure" in result.error

    async def test_duration_is_recorded(self) -> None:
        """Duration in milliseconds should be a positive number."""
        client = _make_client()

        result = await _safe_call(client, "get_quote", {"ticker": "AAPL"}, "AAPL")

        assert result.duration_ms >= 0

    async def test_never_raises(self) -> None:
        """Even with an unexpected exception, _safe_call returns a result instead of raising."""
        client = AsyncMock(spec=Client)
        client.call_tool = AsyncMock(side_effect=RuntimeError("unexpected crash"))

        result = await _safe_call(client, "get_quote", {"ticker": "AAPL"}, "AAPL")

        assert isinstance(result, ToolCallResult)
        assert result.success is False
        assert "unexpected crash" in result.error

    async def test_logs_on_success_and_failure(self) -> None:
        """Logger should be called for both success and failure cases."""
        success_client = _make_client()
        fail_client = _make_client(fail_tools={"get_quote"})

        with patch("stock_radar.pipeline.fetchers.logger") as mock_logger:
            await _safe_call(success_client, "get_quote", {"ticker": "AAPL"}, "AAPL")
            mock_logger.info.assert_called_once()

        with patch("stock_radar.pipeline.fetchers.logger") as mock_logger:
            await _safe_call(fail_client, "get_quote", {"ticker": "AAPL"}, "AAPL")
            mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# TestFetchDeep
# ---------------------------------------------------------------------------


class TestFetchDeep:
    """Tests for the fetch_deep function."""

    async def test_calls_all_six_tools(self) -> None:
        """fetch_deep should invoke exactly 6 MCP tools (4 market + 2 edgar)."""
        market = _make_client()
        edgar = _make_client()

        await fetch_deep(market, edgar, "AAPL", quarter=1, year=2025)

        assert market.call_tool.call_count == len(DEEP_MARKET_TOOLS)
        assert edgar.call_tool.call_count == len(DEEP_EDGAR_TOOLS)
        total_calls = market.call_tool.call_count + edgar.call_tool.call_count
        assert total_calls == 6

    async def test_returns_deep_tier(self) -> None:
        """The returned TickerResult should have tier='deep'."""
        market = _make_client()
        edgar = _make_client()

        result = await fetch_deep(market, edgar, "AAPL", quarter=1, year=2025)

        assert isinstance(result, TickerResult)
        assert result.tier == "deep"

    async def test_all_success_counts(self) -> None:
        """When all tools succeed, success_count=6 and error_count=0."""
        market = _make_client()
        edgar = _make_client()

        result = await fetch_deep(market, edgar, "AAPL", quarter=1, year=2025)

        assert result.success_count == 6
        assert result.error_count == 0
        assert len(result.results) == 6

    async def test_partial_failure_continues(self) -> None:
        """When one market tool fails, the rest still run. Counts reflect partial failure."""
        market = _make_client(fail_tools={"get_quote"})
        edgar = _make_client()

        result = await fetch_deep(market, edgar, "AAPL", quarter=1, year=2025)

        assert result.success_count == 5
        assert result.error_count == 1
        assert len(result.results) == 6

    async def test_transcript_gets_quarter_and_year(self) -> None:
        """get_earnings_transcript should be called with quarter and year args."""
        market = _make_client()
        edgar = _make_client()

        await fetch_deep(market, edgar, "TSLA", quarter=3, year=2024)

        transcript_calls = [
            call
            for call in market.call_tool.call_args_list
            if call.args[0] == "get_earnings_transcript"
        ]
        assert len(transcript_calls) == 1
        args = transcript_calls[0].args[1]
        assert args["ticker"] == "TSLA"
        assert args["quarter"] == 3
        assert args["year"] == 2024

    async def test_price_history_gets_compact(self) -> None:
        """get_price_history should be called with outputsize='compact'."""
        market = _make_client()
        edgar = _make_client()

        await fetch_deep(market, edgar, "MSFT", quarter=2, year=2025)

        history_calls = [
            call for call in market.call_tool.call_args_list if call.args[0] == "get_price_history"
        ]
        assert len(history_calls) == 1
        args = history_calls[0].args[1]
        assert args["ticker"] == "MSFT"
        assert args["outputsize"] == "compact"

    async def test_all_tools_fail(self) -> None:
        """When all 6 tools fail, success_count=0 and error_count=6."""
        all_tools = set(DEEP_MARKET_TOOLS) | set(DEEP_EDGAR_TOOLS)
        market = _make_client(fail_tools=all_tools)
        edgar = _make_client(fail_tools=all_tools)

        result = await fetch_deep(market, edgar, "AAPL", quarter=1, year=2025)

        assert result.success_count == 0
        assert result.error_count == 6

    async def test_results_contain_correct_tool_names(self) -> None:
        """All 6 tool names from config should appear in the results."""
        market = _make_client()
        edgar = _make_client()

        result = await fetch_deep(market, edgar, "AAPL", quarter=1, year=2025)

        result_tool_names = {r.tool_name for r in result.results}
        expected = set(DEEP_MARKET_TOOLS) | set(DEEP_EDGAR_TOOLS)
        assert result_tool_names == expected


# ---------------------------------------------------------------------------
# TestFetchLight
# ---------------------------------------------------------------------------


class TestFetchLight:
    """Tests for the fetch_light function."""

    async def test_calls_two_tools(self) -> None:
        """fetch_light should invoke exactly 2 MCP tools (1 market + 1 edgar)."""
        market = _make_client()
        edgar = _make_client()

        await fetch_light(market, edgar, "AAPL")

        assert market.call_tool.call_count == len(LIGHT_MARKET_TOOLS)
        assert edgar.call_tool.call_count == len(LIGHT_EDGAR_TOOLS)
        total_calls = market.call_tool.call_count + edgar.call_tool.call_count
        assert total_calls == 2

    async def test_returns_light_tier(self) -> None:
        """The returned TickerResult should have tier='light'."""
        market = _make_client()
        edgar = _make_client()

        result = await fetch_light(market, edgar, "AAPL")

        assert isinstance(result, TickerResult)
        assert result.tier == "light"

    async def test_all_success(self) -> None:
        """When both tools succeed, success_count=2 and error_count=0."""
        market = _make_client()
        edgar = _make_client()

        result = await fetch_light(market, edgar, "NVDA")

        assert result.success_count == 2
        assert result.error_count == 0
        assert len(result.results) == 2

    async def test_one_failure(self) -> None:
        """When one tool fails, success_count=1 and error_count=1."""
        market = _make_client(fail_tools={"get_quote"})
        edgar = _make_client()

        result = await fetch_light(market, edgar, "GOOG")

        assert result.success_count == 1
        assert result.error_count == 1
        assert len(result.results) == 2

    async def test_correct_tool_names(self) -> None:
        """Results should contain 'get_quote' and 'get_filings'."""
        market = _make_client()
        edgar = _make_client()

        result = await fetch_light(market, edgar, "AMD")

        result_tool_names = {r.tool_name for r in result.results}
        expected = set(LIGHT_MARKET_TOOLS) | set(LIGHT_EDGAR_TOOLS)
        assert result_tool_names == expected
