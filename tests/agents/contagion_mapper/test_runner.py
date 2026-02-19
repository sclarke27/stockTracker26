"""Tests for the Cross-Sector Contagion Mapper runner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from stock_radar.agents.contagion_mapper.models import ContagionInput
from stock_radar.agents.contagion_mapper.runner import run_contagion_mapper
from stock_radar.agents.models import AgentOutput, AnalysisResult
from tests.agents.helpers import mock_tool_response as _mock_tool_response


def _sample_company_info(ticker: str, name: str, sector: str) -> str:
    return json.dumps(
        {
            "ticker": ticker,
            "name": name,
            "description": f"{name} is a technology company.",
            "sector": sector,
            "industry": "Semiconductors",
            "country": "US",
            "exchange": "NASDAQ",
            "currency": "USD",
        }
    )


def _sample_news_response(ticker: str) -> str:
    return json.dumps(
        {
            "ticker": ticker,
            "articles": [
                {
                    "title": f"{ticker} headline",
                    "url": "https://example.com/1",
                    "time_published": "20240601T120000",
                    "authors": [],
                    "summary": f"{ticker} news summary.",
                    "source": "Reuters",
                    "source_domain": "reuters.com",
                    "topics": [],
                    "overall_sentiment_score": -0.5,
                    "overall_sentiment_label": "Bearish",
                    "ticker_sentiment": [],
                }
            ],
            "total_fetched": 1,
            "source": "alpha_vantage",
        }
    )


def _sample_search_news_response() -> str:
    return json.dumps(
        {
            "query": "NVDA",
            "articles": [
                {
                    "title": "NVDA misses guidance",
                    "url": "https://example.com/2",
                    "time_published": "20240601T110000",
                    "authors": [],
                    "summary": "NVDA cuts forward guidance by 20%.",
                    "source": "CNBC",
                    "source_domain": "cnbc.com",
                    "topics": [],
                    "overall_sentiment_score": -0.7,
                    "overall_sentiment_label": "Bearish",
                    "ticker_sentiment": [],
                }
            ],
            "total_fetched": 1,
            "source": "alpha_vantage",
        }
    )


def _sample_output() -> AgentOutput:
    return AgentOutput(
        prediction_id="pred-cm-001",
        result=AnalysisResult(
            ticker="AMD",
            direction="BEARISH",
            confidence=0.70,
            reasoning="NVDA miss signals sector softness affecting AMD.",
            horizon_days=5,
            model_used="llama3.1:8b",
        ),
    )


class TestRunContagionMapper:

    @patch("stock_radar.agents.contagion_mapper.runner.create_predictions_server")
    @patch("stock_radar.agents.contagion_mapper.runner.create_vector_store_server")
    @patch("stock_radar.agents.contagion_mapper.runner.create_news_feed_server")
    @patch("stock_radar.agents.contagion_mapper.runner.create_market_server")
    async def test_successful_run(self, mock_market, mock_news, mock_vs, mock_pred) -> None:
        """End-to-end: fetches company info + news and runs agent."""
        with (
            patch("stock_radar.agents.contagion_mapper.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.contagion_mapper.runner.ContagionMapperAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.contagion_mapper.runner._load_settings"),
        ):
            market_client = AsyncMock()

            def market_side_effect(tool_name, args):
                ticker = args.get("ticker", "")
                if ticker == "NVDA":
                    return _mock_tool_response(
                        _sample_company_info("NVDA", "NVIDIA Corporation", "Semiconductors")
                    )
                return _mock_tool_response(
                    _sample_company_info("AMD", "Advanced Micro Devices", "Semiconductors")
                )

            market_client.call_tool = AsyncMock(side_effect=market_side_effect)

            news_client = AsyncMock()

            def news_side_effect(tool_name, args):
                if tool_name == "search_news":
                    return _mock_tool_response(_sample_search_news_response())
                ticker = args.get("ticker", "")
                return _mock_tool_response(_sample_news_response(ticker))

            news_client.call_tool = AsyncMock(side_effect=news_side_effect)

            pred_client = AsyncMock()
            vs_client = AsyncMock()

            mock_contexts = [market_client, news_client, pred_client, vs_client]
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

            output = await run_contagion_mapper(
                trigger_ticker="NVDA",
                target_ticker="AMD",
                relationship_type="competitor",
                quarter=3,
                year=2024,
            )

            assert isinstance(output, AgentOutput)
            assert output.prediction_id == "pred-cm-001"

    @patch("stock_radar.agents.contagion_mapper.runner.create_predictions_server")
    @patch("stock_radar.agents.contagion_mapper.runner.create_vector_store_server")
    @patch("stock_radar.agents.contagion_mapper.runner.create_news_feed_server")
    @patch("stock_radar.agents.contagion_mapper.runner.create_market_server")
    async def test_agent_called_with_contagion_input(
        self, mock_market, mock_news, mock_vs, mock_pred
    ) -> None:
        """The agent's run() receives a ContagionInput with correct ticker fields."""
        with (
            patch("stock_radar.agents.contagion_mapper.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.contagion_mapper.runner.ContagionMapperAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.contagion_mapper.runner._load_settings"),
        ):
            market_client = AsyncMock()

            def market_side_effect(tool_name, args):
                ticker = args.get("ticker", "")
                if ticker == "NVDA":
                    return _mock_tool_response(
                        _sample_company_info("NVDA", "NVIDIA Corporation", "Semiconductors")
                    )
                return _mock_tool_response(
                    _sample_company_info("AMD", "Advanced Micro Devices", "Semiconductors")
                )

            market_client.call_tool = AsyncMock(side_effect=market_side_effect)

            news_client = AsyncMock()

            def news_side_effect(tool_name, args):
                if tool_name == "search_news":
                    return _mock_tool_response(_sample_search_news_response())
                ticker = args.get("ticker", "")
                return _mock_tool_response(_sample_news_response(ticker))

            news_client.call_tool = AsyncMock(side_effect=news_side_effect)

            pred_client = AsyncMock()
            vs_client = AsyncMock()

            mock_contexts = [market_client, news_client, pred_client, vs_client]
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

            await run_contagion_mapper(
                trigger_ticker="NVDA",
                target_ticker="AMD",
                relationship_type="competitor",
                quarter=3,
                year=2024,
            )

            mock_agent.run.assert_called_once()
            call_args = mock_agent.run.call_args
            input_arg = call_args[0][0]
            assert isinstance(input_arg, ContagionInput)
            assert input_arg.ticker == "AMD"  # target is the base ticker
            assert input_arg.trigger_ticker == "NVDA"
            assert input_arg.relationship_type == "competitor"

    @patch("stock_radar.agents.contagion_mapper.runner.create_predictions_server")
    @patch("stock_radar.agents.contagion_mapper.runner.create_vector_store_server")
    @patch("stock_radar.agents.contagion_mapper.runner.create_news_feed_server")
    @patch("stock_radar.agents.contagion_mapper.runner.create_market_server")
    async def test_company_info_failure_uses_ticker_as_name(
        self, mock_market, mock_news, mock_vs, mock_pred
    ) -> None:
        """If company info fetch fails, falls back to ticker as company name."""
        with (
            patch("stock_radar.agents.contagion_mapper.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.contagion_mapper.runner.ContagionMapperAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.contagion_mapper.runner._load_settings"),
        ):
            market_client = AsyncMock()
            market_client.call_tool = AsyncMock(side_effect=Exception("Company info unavailable"))

            news_client = AsyncMock()

            def news_side_effect(tool_name, args):
                if tool_name == "search_news":
                    return _mock_tool_response(_sample_search_news_response())
                ticker = args.get("ticker", "UNKNOWN")
                return _mock_tool_response(_sample_news_response(ticker))

            news_client.call_tool = AsyncMock(side_effect=news_side_effect)

            pred_client = AsyncMock()
            vs_client = AsyncMock()

            mock_contexts = [market_client, news_client, pred_client, vs_client]
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

            output = await run_contagion_mapper(
                trigger_ticker="NVDA",
                target_ticker="AMD",
                relationship_type="competitor",
                quarter=3,
                year=2024,
            )

            assert output.prediction_id == "pred-cm-001"
            call_args = mock_agent.run.call_args
            input_arg = call_args[0][0]
            # Falls back to ticker as company name
            assert input_arg.trigger_company_name == "NVDA"
            assert input_arg.target_company_name == "AMD"
