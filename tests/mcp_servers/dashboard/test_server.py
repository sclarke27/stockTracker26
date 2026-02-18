"""Tests for the dashboard MCP server tools."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent

from stock_radar.mcp_servers.dashboard.server import (
    ServerDeps,
    create_server,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_result(data: dict | list) -> MagicMock:
    """Wrap a dict/list as a mock CallToolResult with TextContent."""
    return MagicMock(content=[MagicMock(spec=TextContent, text=json.dumps(data))])


def _prediction(
    pred_id: str = "pred-001",
    ticker: str = "AAPL",
    agent_name: str = "earnings_linguist",
    signal_type: str = "earnings_sentiment",
    direction: str = "BULLISH",
    confidence: float = 0.8,
    horizon_days: int = 5,
    scored_at: str | None = None,
) -> dict:
    return {
        "id": pred_id,
        "ticker": ticker,
        "agent_name": agent_name,
        "signal_type": signal_type,
        "direction": direction,
        "confidence": confidence,
        "reasoning": "Strong buy signal.",
        "prediction_date": "2024-06-01",
        "horizon_days": horizon_days,
        "created_at": "2024-06-01T12:00:00Z",
        "scored_at": scored_at,
        "actual_price_close": None,
        "actual_price_at_horizon": None,
        "return_pct": None,
        "status": None,
    }


def _quote(ticker: str = "AAPL", price: float = 150.0) -> dict:
    return {
        "ticker": ticker,
        "price": price,
        "open": 148.0,
        "high": 152.0,
        "low": 147.0,
        "volume": 1000000,
        "timestamp": "2024-06-01T16:00:00Z",
    }


def _company_info(ticker: str = "AAPL", name: str = "Apple Inc.") -> dict:
    return {
        "ticker": ticker,
        "name": name,
        "description": f"{name} is a technology company.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "US",
        "exchange": "NASDAQ",
        "currency": "USD",
    }


def _agent_stats(
    agent_name: str = "earnings_linguist",
    signal_type: str = "earnings_sentiment",
    total: int = 10,
    scored: int = 8,
    correct: int = 6,
    accuracy_pct: float = 75.0,
    avg_confidence: float = 0.72,
) -> dict:
    return {
        "agent_name": agent_name,
        "signal_type": signal_type,
        "total": total,
        "scored": scored,
        "correct": correct,
        "accuracy_pct": accuracy_pct,
        "avg_confidence": avg_confidence,
        "avg_return_when_correct": 2.5,
    }


def _sentiment_summary(ticker: str = "AAPL") -> dict:
    return {
        "ticker": ticker,
        "article_count": 20,
        "average_sentiment_score": 0.4,
        "average_sentiment_label": "Bullish",
        "bullish_count": 14,
        "bearish_count": 3,
        "neutral_count": 3,
        "top_topics": ["earnings", "AI"],
    }


def _price_history(ticker: str = "AAPL") -> dict:
    return {
        "ticker": ticker,
        "prices": [
            {
                "date": "2024-06-01",
                "open": 148.0,
                "high": 152.0,
                "low": 147.0,
                "close": 150.0,
                "volume": 1000000,
            },
            {
                "date": "2024-05-31",
                "open": 145.0,
                "high": 149.0,
                "low": 144.0,
                "close": 147.0,
                "volume": 900000,
            },
        ],
        "outputsize": "compact",
    }


# ---------------------------------------------------------------------------
# Fixture: mock clients
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_clients() -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    """Return (predictions_client, market_client, news_client) as AsyncMocks."""
    pred = AsyncMock()
    market = AsyncMock()
    news = AsyncMock()
    return pred, market, news


# ---------------------------------------------------------------------------
# Helper: call a tool with mocked lifespan deps
# ---------------------------------------------------------------------------


async def _call_tool(
    tool_name: str,
    args: dict,
    pred_client: AsyncMock,
    market_client: AsyncMock,
    news_client: AsyncMock,
) -> dict:
    """Call a dashboard tool via in-process Client with a mocked lifespan.

    Creates a fresh server instance per call so each test gets an isolated
    lifespan with its own deps. Patches the server's ``_lifespan`` to skip
    real MCP server connections and inject controlled mock clients instead.
    The tool functions run exactly as in production — only dep injection
    is replaced.
    """
    from fastmcp import Client

    deps = ServerDeps(
        predictions_client=pred_client,
        market_client=market_client,
        news_client=news_client,
    )

    @asynccontextmanager
    async def _mock_lifespan(server):
        yield deps

    # Fresh server per call — no shared lifespan state between tests.
    server = create_server()
    with patch.object(server, "_lifespan", _mock_lifespan):
        async with Client(server) as client:
            result = await client.call_tool(tool_name, args)
            content = result.content[0]
            assert isinstance(content, TextContent)
            return json.loads(content.text)


# ---------------------------------------------------------------------------
# Tests: create_server
# ---------------------------------------------------------------------------


class TestCreateServer:
    def test_create_server_returns_fastmcp(self) -> None:
        """create_server() returns a FastMCP instance."""
        from fastmcp import FastMCP

        server = create_server()
        assert isinstance(server, FastMCP)

    def test_create_server_custom_name(self) -> None:
        """create_server() accepts a custom name."""
        server = create_server(name="test-dashboard")
        assert server.name == "test-dashboard"


# ---------------------------------------------------------------------------
# Tests: get_active_signals
# ---------------------------------------------------------------------------


class TestGetActiveSignals:
    async def test_returns_active_signals_response_structure(self, mock_clients: tuple) -> None:
        """get_active_signals returns correct response structure."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {"predictions": [_prediction(confidence=0.8)], "total_count": 1}
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))
        market_client.call_tool = AsyncMock(return_value=_text_result(_quote()))

        data = await _call_tool(
            "get_active_signals", {"limit": 10}, pred_client, market_client, news_client
        )

        assert "signals" in data
        assert "total_count" in data
        assert data["total_count"] == 1
        assert len(data["signals"]) == 1

    async def test_signal_fields_present(self, mock_clients: tuple) -> None:
        """Each signal has all required fields."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {
            "predictions": [_prediction(pred_id="pred-xyz", ticker="TSLA", confidence=0.75)],
            "total_count": 1,
        }
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))
        market_client.call_tool = AsyncMock(return_value=_text_result(_quote("TSLA", 200.0)))

        data = await _call_tool("get_active_signals", {}, pred_client, market_client, news_client)

        sig = data["signals"][0]
        assert sig["prediction_id"] == "pred-xyz"
        assert sig["ticker"] == "TSLA"
        assert sig["direction"] == "BULLISH"
        assert sig["confidence"] == 0.75
        assert sig["current_price"] == 200.0

    async def test_filters_low_confidence_signals(self, mock_clients: tuple) -> None:
        """Signals below MIN_SIGNAL_CONFIDENCE are excluded."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {
            "predictions": [
                _prediction(pred_id="pred-high", confidence=0.8),
                _prediction(pred_id="pred-low", confidence=0.3),
            ],
            "total_count": 2,
        }
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))
        market_client.call_tool = AsyncMock(return_value=_text_result(_quote()))

        data = await _call_tool("get_active_signals", {}, pred_client, market_client, news_client)

        ids = [s["prediction_id"] for s in data["signals"]]
        assert "pred-high" in ids
        assert "pred-low" not in ids

    async def test_price_fetch_failure_uses_none(self, mock_clients: tuple) -> None:
        """If market data fetch fails, current_price is None (non-fatal)."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {"predictions": [_prediction(confidence=0.9)], "total_count": 1}
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))
        market_client.call_tool = AsyncMock(side_effect=Exception("market unavailable"))

        data = await _call_tool("get_active_signals", {}, pred_client, market_client, news_client)

        assert data["signals"][0]["current_price"] is None

    async def test_empty_predictions_returns_empty_list(self, mock_clients: tuple) -> None:
        """Empty predictions DB returns empty signals list."""
        pred_client, market_client, news_client = mock_clients

        pred_client.call_tool = AsyncMock(
            return_value=_text_result({"predictions": [], "total_count": 0})
        )

        data = await _call_tool("get_active_signals", {}, pred_client, market_client, news_client)

        assert data["signals"] == []
        assert data["total_count"] == 0


# ---------------------------------------------------------------------------
# Tests: get_watchlist
# ---------------------------------------------------------------------------


class TestGetWatchlist:
    async def test_returns_watchlist_entries(self, mock_clients: tuple) -> None:
        """get_watchlist returns deduplicated ticker entries."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {
            "predictions": [
                _prediction(pred_id="p1", ticker="AAPL", direction="BULLISH"),
                _prediction(pred_id="p2", ticker="AAPL", direction="BEARISH"),
                _prediction(pred_id="p3", ticker="MSFT", direction="BULLISH"),
            ],
            "total_count": 3,
        }
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))

        def market_side_effect(tool_name, args):
            ticker = args.get("ticker", "AAPL")
            if tool_name == "get_quote":
                return _text_result(_quote(ticker))
            return _text_result(_company_info(ticker, f"{ticker} Corp"))

        market_client.call_tool = AsyncMock(side_effect=market_side_effect)

        data = await _call_tool("get_watchlist", {}, pred_client, market_client, news_client)

        assert "entries" in data
        tickers = [e["ticker"] for e in data["entries"]]
        assert tickers.count("AAPL") == 1
        assert "MSFT" in tickers

    async def test_watchlist_signal_count(self, mock_clients: tuple) -> None:
        """active_signal_count reflects number of predictions per ticker."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {
            "predictions": [
                _prediction(pred_id="p1", ticker="NVDA"),
                _prediction(pred_id="p2", ticker="NVDA"),
                _prediction(pred_id="p3", ticker="NVDA"),
            ],
            "total_count": 3,
        }
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))

        def market_side_effect(tool_name, args):
            if tool_name == "get_quote":
                return _text_result(_quote("NVDA", 900.0))
            return _text_result(_company_info("NVDA", "NVIDIA Corporation"))

        market_client.call_tool = AsyncMock(side_effect=market_side_effect)

        data = await _call_tool("get_watchlist", {}, pred_client, market_client, news_client)

        nvda_entry = next(e for e in data["entries"] if e["ticker"] == "NVDA")
        assert nvda_entry["active_signal_count"] == 3

    async def test_market_fetch_failure_is_nonfatal(self, mock_clients: tuple) -> None:
        """Market data failures do not crash get_watchlist."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {"predictions": [_prediction(ticker="FAIL")], "total_count": 1}
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))
        market_client.call_tool = AsyncMock(side_effect=Exception("market down"))

        data = await _call_tool("get_watchlist", {}, pred_client, market_client, news_client)

        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["current_price"] is None
        assert entry["company_name"] == "FAIL"  # falls back to ticker


# ---------------------------------------------------------------------------
# Tests: get_agent_status
# ---------------------------------------------------------------------------


class TestGetAgentStatus:
    async def test_returns_agent_status_response(self, mock_clients: tuple) -> None:
        """get_agent_status returns all agent stats."""
        pred_client, market_client, news_client = mock_clients

        accuracy_data = {
            "agent_stats": [
                _agent_stats("earnings_linguist", "earnings_sentiment"),
                _agent_stats(
                    "contagion_mapper",
                    "contagion_signal",
                    total=5,
                    scored=3,
                    correct=2,
                    accuracy_pct=66.7,
                ),
            ]
        }
        pred_client.call_tool = AsyncMock(return_value=_text_result(accuracy_data))

        data = await _call_tool("get_agent_status", {}, pred_client, market_client, news_client)

        assert "agents" in data
        assert "as_of_days" in data
        assert len(data["agents"]) == 2
        agent_names = [a["agent_name"] for a in data["agents"]]
        assert "earnings_linguist" in agent_names

    async def test_agent_status_fields_mapped(self, mock_clients: tuple) -> None:
        """AgentStatus fields are correctly mapped from AgentStats."""
        pred_client, market_client, news_client = mock_clients

        accuracy_data = {
            "agent_stats": [
                _agent_stats(
                    total=20, scored=15, correct=12, accuracy_pct=80.0, avg_confidence=0.72
                )
            ]
        }
        pred_client.call_tool = AsyncMock(return_value=_text_result(accuracy_data))

        data = await _call_tool("get_agent_status", {}, pred_client, market_client, news_client)

        agent = data["agents"][0]
        assert agent["total_predictions"] == 20
        assert agent["scored"] == 15
        assert agent["accuracy_pct"] == 80.0
        assert agent["avg_confidence"] == 0.72


# ---------------------------------------------------------------------------
# Tests: get_prediction_detail
# ---------------------------------------------------------------------------


class TestGetPredictionDetail:
    async def test_returns_prediction_and_price_history(self, mock_clients: tuple) -> None:
        """get_prediction_detail returns prediction dict + price history."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {
            "predictions": [_prediction(pred_id="pred-abc", ticker="AMD")],
            "total_count": 1,
        }
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))
        market_client.call_tool = AsyncMock(return_value=_text_result(_price_history("AMD")))

        data = await _call_tool(
            "get_prediction_detail",
            {"prediction_id": "pred-abc"},
            pred_client,
            market_client,
            news_client,
        )

        assert "prediction" in data
        assert "price_history" in data
        assert data["prediction"]["id"] == "pred-abc"
        assert isinstance(data["price_history"], list)

    async def test_prediction_not_found_returns_none(self, mock_clients: tuple) -> None:
        """If prediction_id not found, prediction field is None."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {"predictions": [], "total_count": 0}
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))
        market_client.call_tool = AsyncMock(return_value=_text_result(_price_history()))

        data = await _call_tool(
            "get_prediction_detail",
            {"prediction_id": "nonexistent"},
            pred_client,
            market_client,
            news_client,
        )

        assert data["prediction"] is None

    async def test_price_history_failure_is_nonfatal(self, mock_clients: tuple) -> None:
        """Price history fetch failure returns empty list rather than crashing."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {
            "predictions": [_prediction(pred_id="pred-xyz", ticker="TSLA")],
            "total_count": 1,
        }
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))
        market_client.call_tool = AsyncMock(side_effect=Exception("price history unavailable"))

        data = await _call_tool(
            "get_prediction_detail",
            {"prediction_id": "pred-xyz"},
            pred_client,
            market_client,
            news_client,
        )

        assert data["prediction"]["id"] == "pred-xyz"
        assert data["price_history"] == []


# ---------------------------------------------------------------------------
# Tests: get_ticker_summary
# ---------------------------------------------------------------------------


class TestGetTickerSummary:
    async def test_returns_full_ticker_summary(self, mock_clients: tuple) -> None:
        """get_ticker_summary aggregates company info, quote, predictions, sentiment."""
        pred_client, market_client, news_client = mock_clients

        pred_data = {"predictions": [_prediction(ticker="AAPL")], "total_count": 1}
        pred_client.call_tool = AsyncMock(return_value=_text_result(pred_data))

        def market_side_effect(tool_name, args):
            if tool_name == "get_quote":
                return _text_result(_quote("AAPL", 155.0))
            return _text_result(_company_info("AAPL", "Apple Inc."))

        market_client.call_tool = AsyncMock(side_effect=market_side_effect)
        news_client.call_tool = AsyncMock(return_value=_text_result(_sentiment_summary("AAPL")))

        data = await _call_tool(
            "get_ticker_summary", {"ticker": "AAPL"}, pred_client, market_client, news_client
        )

        assert data["ticker"] == "AAPL"
        assert data["company_name"] == "Apple Inc."
        assert data["current_price"] == 155.0
        assert data["sentiment_score"] == 0.4
        assert isinstance(data["recent_predictions"], list)

    async def test_sentiment_failure_is_nonfatal(self, mock_clients: tuple) -> None:
        """Sentiment fetch failure returns None scores without crashing."""
        pred_client, market_client, news_client = mock_clients

        pred_client.call_tool = AsyncMock(
            return_value=_text_result({"predictions": [], "total_count": 0})
        )

        def market_side_effect(tool_name, args):
            if tool_name == "get_quote":
                return _text_result(_quote())
            return _text_result(_company_info())

        market_client.call_tool = AsyncMock(side_effect=market_side_effect)
        news_client.call_tool = AsyncMock(side_effect=Exception("news API down"))

        data = await _call_tool(
            "get_ticker_summary", {"ticker": "AAPL"}, pred_client, market_client, news_client
        )

        assert data["sentiment_score"] is None
        assert data["sentiment_label"] is None

    async def test_company_info_failure_falls_back_to_ticker(self, mock_clients: tuple) -> None:
        """Company info failure falls back to ticker as company_name."""
        pred_client, market_client, news_client = mock_clients

        pred_client.call_tool = AsyncMock(
            return_value=_text_result({"predictions": [], "total_count": 0})
        )

        def market_side_effect(tool_name, args):
            if tool_name == "get_quote":
                return _text_result(_quote("XYZ", 10.0))
            raise Exception("company info unavailable")

        market_client.call_tool = AsyncMock(side_effect=market_side_effect)
        news_client.call_tool = AsyncMock(return_value=_text_result(_sentiment_summary("XYZ")))

        data = await _call_tool(
            "get_ticker_summary", {"ticker": "XYZ"}, pred_client, market_client, news_client
        )

        assert data["ticker"] == "XYZ"
        assert data["company_name"] == "XYZ"
