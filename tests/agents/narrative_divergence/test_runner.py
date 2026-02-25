"""Tests for the Narrative vs Price Divergence runner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from stock_radar.agents.models import AgentOutput, AnalysisResult
from stock_radar.agents.narrative_divergence.models import NarrativeDivergenceInput
from stock_radar.agents.narrative_divergence.runner import run_narrative_divergence
from tests.agents.helpers import mock_tool_response as _mock_tool_response


def _sample_sentiment_response() -> str:
    return json.dumps(
        {
            "ticker": "AAPL",
            "article_count": 14,
            "average_sentiment_score": 0.45,
            "average_sentiment_label": "Somewhat-Bullish",
            "breakdown": {"bullish": 8, "bearish": 2, "neutral": 4},
            "top_topics": [],
        }
    )


def _sample_news_response() -> str:
    return json.dumps(
        {
            "ticker": "AAPL",
            "articles": [
                {
                    "title": "Apple beats earnings",
                    "url": "https://example.com/1",
                    "time_published": "20240315T120000",
                    "authors": [],
                    "summary": "Apple beat expectations.",
                    "source": "Reuters",
                    "source_domain": "reuters.com",
                    "topics": [],
                    "overall_sentiment_score": 0.6,
                    "overall_sentiment_label": "Bullish",
                    "ticker_sentiment": [],
                }
            ],
            "total_fetched": 1,
            "source": "alpha_vantage",
        }
    )


def _sample_price_history_response() -> str:
    """Price history with enough data points to compute 7d and 30d returns."""
    # 35 days of daily data (most recent first)
    prices = []
    for i in range(35):
        prices.append(
            {
                "date": f"2024-{(3 - i // 30):02d}-{(15 - (i % 30)):02d}",
                "open": 180.0 - i * 0.3,
                "high": 182.0 - i * 0.3,
                "low": 178.0 - i * 0.3,
                "close": 180.0 - i * 0.3,
                "volume": 50000000,
            }
        )
    return json.dumps(
        {
            "ticker": "AAPL",
            "interval": "daily",
            "prices": prices,
        }
    )


def _sample_output() -> AgentOutput:
    return AgentOutput(
        prediction_id="pred-nd-001",
        result=AnalysisResult(
            ticker="AAPL",
            direction="BULLISH",
            confidence=0.75,
            reasoning="Bullish sentiment diverges from price decline.",
            horizon_days=10,
            model_used="qwen3:32b",
        ),
    )


class TestRunNarrativeDivergence:

    @patch("stock_radar.agents.narrative_divergence.runner.create_predictions_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_vector_store_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_news_feed_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_market_server")
    async def test_successful_run(self, mock_market, mock_news, mock_vs, mock_pred) -> None:
        """End-to-end: fetches data from MCP servers and runs the agent."""
        with (
            patch("stock_radar.agents.narrative_divergence.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.narrative_divergence.runner.NarrativeDivergenceAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.narrative_divergence.runner._load_settings"),
        ):
            news_client = AsyncMock()

            def news_side_effect(tool_name, args):
                if tool_name == "get_sentiment_summary":
                    return _mock_tool_response(_sample_sentiment_response())
                return _mock_tool_response(_sample_news_response())

            news_client.call_tool = AsyncMock(side_effect=news_side_effect)

            market_client = AsyncMock()
            market_client.call_tool = AsyncMock(
                return_value=_mock_tool_response(_sample_price_history_response())
            )
            pred_client = AsyncMock()
            vs_client = AsyncMock()

            mock_contexts = [news_client, market_client, pred_client, vs_client]
            context_idx = 0

            class _MockContext:
                def __init__(self, server):
                    nonlocal context_idx
                    self._client = mock_contexts[context_idx]
                    context_idx += 1

                async def __aenter__(self):
                    return self._client

                async def __aexit__(self, *args):
                    pass

            mock_client_cls.side_effect = _MockContext

            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=_sample_output())
            mock_agent_cls.return_value = mock_agent

            output = await run_narrative_divergence("AAPL", 4, 2024)

            assert isinstance(output, AgentOutput)
            assert output.prediction_id == "pred-nd-001"

    @patch("stock_radar.agents.narrative_divergence.runner.create_predictions_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_vector_store_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_news_feed_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_market_server")
    async def test_agent_called_with_narrative_divergence_input(
        self, mock_market, mock_news, mock_vs, mock_pred
    ) -> None:
        """The agent's run() receives a NarrativeDivergenceInput."""
        with (
            patch("stock_radar.agents.narrative_divergence.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.narrative_divergence.runner.NarrativeDivergenceAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.narrative_divergence.runner._load_settings"),
        ):
            news_client = AsyncMock()

            def news_side_effect(tool_name, args):
                if tool_name == "get_sentiment_summary":
                    return _mock_tool_response(_sample_sentiment_response())
                return _mock_tool_response(_sample_news_response())

            news_client.call_tool = AsyncMock(side_effect=news_side_effect)
            market_client = AsyncMock()
            market_client.call_tool = AsyncMock(
                return_value=_mock_tool_response(_sample_price_history_response())
            )
            pred_client = AsyncMock()
            vs_client = AsyncMock()

            mock_contexts = [news_client, market_client, pred_client, vs_client]
            context_idx = 0

            class _MockContext:
                def __init__(self, server):
                    nonlocal context_idx
                    self._client = mock_contexts[context_idx]
                    context_idx += 1

                async def __aenter__(self):
                    return self._client

                async def __aexit__(self, *args):
                    pass

            mock_client_cls.side_effect = _MockContext

            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=_sample_output())
            mock_agent_cls.return_value = mock_agent

            await run_narrative_divergence("AAPL", 4, 2024)

            mock_agent.run.assert_called_once()
            call_args = mock_agent.run.call_args
            input_arg = call_args[0][0]
            assert isinstance(input_arg, NarrativeDivergenceInput)
            assert input_arg.ticker == "AAPL"

    @patch("stock_radar.agents.narrative_divergence.runner.create_predictions_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_vector_store_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_news_feed_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_market_server")
    async def test_sentiment_fetch_failure_raises(
        self, mock_market, mock_news, mock_vs, mock_pred
    ) -> None:
        """If sentiment fetch fails, the error propagates."""
        with (
            patch("stock_radar.agents.narrative_divergence.runner.Client") as mock_client_cls,
            patch("stock_radar.agents.narrative_divergence.runner._load_settings"),
        ):
            news_client = AsyncMock()
            news_client.call_tool = AsyncMock(side_effect=Exception("News feed unavailable"))

            class _MockContext:
                def __init__(self, server):
                    self._client = news_client

                async def __aenter__(self):
                    return self._client

                async def __aexit__(self, *args):
                    pass

            mock_client_cls.side_effect = _MockContext

            with pytest.raises(Exception, match="News feed unavailable"):
                await run_narrative_divergence("AAPL", 4, 2024)

    @patch("stock_radar.agents.narrative_divergence.runner.create_predictions_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_vector_store_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_news_feed_server")
    @patch("stock_radar.agents.narrative_divergence.runner.create_market_server")
    async def test_price_history_insufficient_falls_back(
        self, mock_market, mock_news, mock_vs, mock_pred
    ) -> None:
        """If price history has fewer than 7 data points, returns still work."""
        with (
            patch("stock_radar.agents.narrative_divergence.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.narrative_divergence.runner.NarrativeDivergenceAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.narrative_divergence.runner._load_settings"),
        ):
            news_client = AsyncMock()

            def news_side_effect(tool_name, args):
                if tool_name == "get_sentiment_summary":
                    return _mock_tool_response(_sample_sentiment_response())
                return _mock_tool_response(_sample_news_response())

            news_client.call_tool = AsyncMock(side_effect=news_side_effect)

            # Only 3 price points — not enough for 7d or 30d returns
            short_history = json.dumps(
                {
                    "ticker": "AAPL",
                    "interval": "daily",
                    "prices": [
                        {"date": "2024-03-15", "close": 175.0},
                        {"date": "2024-03-14", "close": 174.0},
                        {"date": "2024-03-13", "close": 173.0},
                    ],
                }
            )
            market_client = AsyncMock()
            market_client.call_tool = AsyncMock(return_value=_mock_tool_response(short_history))
            pred_client = AsyncMock()
            vs_client = AsyncMock()

            mock_contexts = [news_client, market_client, pred_client, vs_client]
            context_idx = 0

            class _MockContext:
                def __init__(self, server):
                    nonlocal context_idx
                    self._client = mock_contexts[context_idx]
                    context_idx += 1

                async def __aenter__(self):
                    return self._client

                async def __aexit__(self, *args):
                    pass

            mock_client_cls.side_effect = _MockContext

            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=_sample_output())
            mock_agent_cls.return_value = mock_agent

            # Should not raise — returns 0.0 for unavailable returns
            output = await run_narrative_divergence("AAPL", 4, 2024)
            assert output.prediction_id == "pred-nd-001"
