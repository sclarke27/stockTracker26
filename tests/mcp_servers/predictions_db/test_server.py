"""Integration tests for the predictions-db MCP server."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from stock_radar.mcp_servers.predictions_db.server import create_server
from tests.mcp_servers.conftest import get_tool_text

MOCK_ENV = {
    "ALPHA_VANTAGE_API_KEY": "unused",
    "ANTHROPIC_API_KEY": "unused",
    "OPENAI_API_KEY": "unused",
    "SEC_EDGAR_EMAIL": "test@example.com",
}


async def _log_prediction(client: Client, **overrides: object) -> dict:
    """Log a prediction with sensible defaults, returning the parsed response.

    Any keyword argument overrides the default value.
    """
    args: dict = {
        "ticker": "AAPL",
        "agent_name": "earnings_linguist",
        "signal_type": "earnings_sentiment",
        "direction": "BULLISH",
        "confidence": 0.85,
        "reasoning": "Strong language signals.",
        "horizon_days": 5,
    }
    args.update(overrides)
    result = await client.call_tool("log_prediction", args)
    return json.loads(get_tool_text(result))


class TestLogPrediction:
    """Tests for the log_prediction tool."""

    async def test_log_prediction_returns_id_and_timestamp(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                data = await _log_prediction(client)

        assert "prediction_id" in data
        # UUID4 format: 8-4-4-4-12 hex characters
        parts = data["prediction_id"].split("-")
        assert len(parts) == 5
        assert "created_at" in data

    async def test_log_prediction_persists(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await _log_prediction(client)
                result = await client.call_tool("get_prediction_history", {})

        data = json.loads(get_tool_text(result))
        assert data["total_count"] == 1
        assert data["predictions"][0]["ticker"] == "AAPL"

    async def test_log_prediction_with_custom_date(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await _log_prediction(client, prediction_date="2026-01-15")
                result = await client.call_tool("get_prediction_history", {})

        data = json.loads(get_tool_text(result))
        assert data["predictions"][0]["prediction_date"] == "2026-01-15"

    async def test_log_prediction_defaults_date_to_today(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await _log_prediction(client)
                result = await client.call_tool("get_prediction_history", {})

        data = json.loads(get_tool_text(result))
        assert data["predictions"][0]["prediction_date"] == date.today().isoformat()


class TestScorePrediction:
    """Tests for the score_prediction tool."""

    async def test_score_bullish_correct(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                log_data = await _log_prediction(client, direction="BULLISH")
                result = await client.call_tool(
                    "score_prediction",
                    {
                        "prediction_id": log_data["prediction_id"],
                        "actual_price_close": 100.0,
                        "actual_price_at_horizon": 110.0,
                    },
                )

        data = json.loads(get_tool_text(result))
        assert data["status"] == "CORRECT"
        assert data["return_pct"] == pytest.approx(10.0, abs=0.01)
        assert data["direction"] == "BULLISH"

    async def test_score_bullish_incorrect(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                log_data = await _log_prediction(client, direction="BULLISH")
                result = await client.call_tool(
                    "score_prediction",
                    {
                        "prediction_id": log_data["prediction_id"],
                        "actual_price_close": 100.0,
                        "actual_price_at_horizon": 90.0,
                    },
                )

        data = json.loads(get_tool_text(result))
        assert data["status"] == "INCORRECT"

    async def test_score_bearish_correct(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                log_data = await _log_prediction(client, direction="BEARISH")
                result = await client.call_tool(
                    "score_prediction",
                    {
                        "prediction_id": log_data["prediction_id"],
                        "actual_price_close": 100.0,
                        "actual_price_at_horizon": 90.0,
                    },
                )

        data = json.loads(get_tool_text(result))
        assert data["status"] == "CORRECT"

    async def test_score_neutral_partial(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                log_data = await _log_prediction(client, direction="NEUTRAL")
                result = await client.call_tool(
                    "score_prediction",
                    {
                        "prediction_id": log_data["prediction_id"],
                        "actual_price_close": 100.0,
                        "actual_price_at_horizon": 101.0,
                    },
                )

        data = json.loads(get_tool_text(result))
        # 1% return is below the 2% neutral threshold -> PARTIAL
        assert data["status"] == "PARTIAL"

    async def test_score_nonexistent_prediction(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                with pytest.raises(ToolError, match="not found"):
                    await client.call_tool(
                        "score_prediction",
                        {
                            "prediction_id": "00000000-0000-0000-0000-000000000000",
                            "actual_price_close": 100.0,
                            "actual_price_at_horizon": 110.0,
                        },
                    )


class TestGetPredictionHistory:
    """Tests for the get_prediction_history tool."""

    async def test_history_no_filters(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await _log_prediction(client, ticker="AAPL")
                await _log_prediction(client, ticker="MSFT")
                await _log_prediction(client, ticker="GOOG")
                result = await client.call_tool("get_prediction_history", {})

        data = json.loads(get_tool_text(result))
        assert data["total_count"] == 3
        assert len(data["predictions"]) == 3

    async def test_history_filter_by_ticker(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await _log_prediction(client, ticker="AAPL")
                await _log_prediction(client, ticker="MSFT")
                result = await client.call_tool(
                    "get_prediction_history",
                    {"ticker": "AAPL"},
                )

        data = json.loads(get_tool_text(result))
        assert data["total_count"] == 1
        assert data["predictions"][0]["ticker"] == "AAPL"

    async def test_history_scored_only(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                log1 = await _log_prediction(client)
                await _log_prediction(client)
                # Score only the first prediction.
                await client.call_tool(
                    "score_prediction",
                    {
                        "prediction_id": log1["prediction_id"],
                        "actual_price_close": 100.0,
                        "actual_price_at_horizon": 110.0,
                    },
                )
                result = await client.call_tool(
                    "get_prediction_history",
                    {"scored_only": True},
                )

        data = json.loads(get_tool_text(result))
        assert data["total_count"] == 1
        assert data["predictions"][0]["status"] == "CORRECT"

    async def test_history_pagination(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                for i in range(5):
                    await _log_prediction(client, ticker=f"T{i}")
                result = await client.call_tool(
                    "get_prediction_history",
                    {"limit": 2, "offset": 1},
                )

        data = json.loads(get_tool_text(result))
        assert len(data["predictions"]) == 2
        assert data["total_count"] == 5


class TestGetAgentAccuracy:
    """Tests for the get_agent_accuracy tool."""

    async def test_accuracy_single_agent(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                # Log 3 predictions for the same agent.
                log1 = await _log_prediction(client, direction="BULLISH")
                log2 = await _log_prediction(client, direction="BULLISH")
                await _log_prediction(client, direction="BULLISH")

                # Score 2: one correct (price up), one incorrect (price down).
                await client.call_tool(
                    "score_prediction",
                    {
                        "prediction_id": log1["prediction_id"],
                        "actual_price_close": 100.0,
                        "actual_price_at_horizon": 110.0,
                    },
                )
                await client.call_tool(
                    "score_prediction",
                    {
                        "prediction_id": log2["prediction_id"],
                        "actual_price_close": 100.0,
                        "actual_price_at_horizon": 90.0,
                    },
                )
                result = await client.call_tool("get_agent_accuracy", {})

        data = json.loads(get_tool_text(result))
        assert len(data["agent_stats"]) == 1
        stats = data["agent_stats"][0]
        assert stats["agent_name"] == "earnings_linguist"
        assert stats["total"] == 3
        assert stats["scored"] == 2
        assert stats["correct"] == 1
        assert stats["accuracy_pct"] == pytest.approx(50.0, abs=0.1)

    async def test_accuracy_multiple_agents(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await _log_prediction(
                    client,
                    agent_name="earnings_linguist",
                    signal_type="earnings_sentiment",
                )
                await _log_prediction(
                    client,
                    agent_name="sec_filing_analyzer",
                    signal_type="filing_anomaly",
                )
                result = await client.call_tool("get_agent_accuracy", {})

        data = json.loads(get_tool_text(result))
        agent_names = {s["agent_name"] for s in data["agent_stats"]}
        assert agent_names == {"earnings_linguist", "sec_filing_analyzer"}

    async def test_accuracy_empty(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool("get_agent_accuracy", {})

        data = json.loads(get_tool_text(result))
        assert data["agent_stats"] == []


class TestToolReturnsValidJson:
    """Tests that tool responses are well-formed JSON."""

    async def test_all_tools_return_valid_json(self, tmp_db: str) -> None:
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool(
                    "log_prediction",
                    {
                        "ticker": "AAPL",
                        "agent_name": "earnings_linguist",
                        "signal_type": "earnings_sentiment",
                        "direction": "BULLISH",
                        "confidence": 0.85,
                        "reasoning": "Strong language signals.",
                        "horizon_days": 5,
                    },
                )

        parsed = json.loads(get_tool_text(result))
        assert isinstance(parsed, dict)


class TestGetPendingScoring:
    """Tests for the get_pending_scoring tool."""

    async def test_returns_empty_when_none_pending(self, tmp_db: str) -> None:
        """No predictions at all returns empty list."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                result = await client.call_tool("get_pending_scoring", {"as_of_date": "2026-03-01"})
                data = json.loads(get_tool_text(result))

        assert data["predictions"] == []
        assert data["total_count"] == 0

    async def test_returns_pending_past_horizon(self, tmp_db: str) -> None:
        """Prediction with elapsed horizon is returned."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                logged = await _log_prediction(
                    client,
                    prediction_date="2026-01-01",
                    horizon_days=5,
                )

                result = await client.call_tool("get_pending_scoring", {"as_of_date": "2026-01-15"})
                data = json.loads(get_tool_text(result))

        assert data["total_count"] == 1
        assert data["predictions"][0]["id"] == logged["prediction_id"]

    async def test_excludes_future_horizons(self, tmp_db: str) -> None:
        """Predictions whose horizon has not elapsed are excluded."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await _log_prediction(
                    client,
                    prediction_date="2026-03-01",
                    horizon_days=10,
                )

                result = await client.call_tool("get_pending_scoring", {"as_of_date": "2026-03-05"})
                data = json.loads(get_tool_text(result))

        assert data["total_count"] == 0

    async def test_excludes_scored_predictions(self, tmp_db: str) -> None:
        """Already-scored predictions are excluded."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                logged = await _log_prediction(
                    client,
                    prediction_date="2026-01-01",
                    horizon_days=5,
                )

                # Score it
                await client.call_tool(
                    "score_prediction",
                    {
                        "prediction_id": logged["prediction_id"],
                        "actual_price_close": 150.0,
                        "actual_price_at_horizon": 155.0,
                    },
                )

                result = await client.call_tool("get_pending_scoring", {"as_of_date": "2026-01-15"})
                data = json.loads(get_tool_text(result))

        assert data["total_count"] == 0

    async def test_response_structure(self, tmp_db: str) -> None:
        """Response contains predictions list and total_count."""
        with (
            patch.dict("os.environ", MOCK_ENV),
            patch(
                "stock_radar.mcp_servers.predictions_db.server._get_db_path",
                return_value=tmp_db,
            ),
        ):
            async with Client(create_server()) as client:
                await _log_prediction(
                    client,
                    prediction_date="2026-01-01",
                    horizon_days=5,
                )

                result = await client.call_tool("get_pending_scoring", {"as_of_date": "2026-01-15"})
                data = json.loads(get_tool_text(result))

        assert "predictions" in data
        assert "total_count" in data
        assert isinstance(data["predictions"], list)
        pred = data["predictions"][0]
        assert "id" in pred
        assert "ticker" in pred
        assert "prediction_date" in pred
        assert "horizon_days" in pred
