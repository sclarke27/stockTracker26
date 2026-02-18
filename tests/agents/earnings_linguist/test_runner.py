"""Tests for the Earnings Linguist runner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stock_radar.agents.earnings_linguist.models import (
    EarningsAnalysis,
    SentimentIndicator,
)
from stock_radar.agents.earnings_linguist.runner import run_earnings_linguist
from stock_radar.agents.exceptions import TranscriptNotFoundError
from stock_radar.agents.models import AgentOutput, AnalysisResult


def _sample_transcript_response() -> str:
    """JSON string mimicking market-data-mcp get_earnings_transcript response."""
    return json.dumps(
        {
            "ticker": "AAPL",
            "quarter": 4,
            "year": 2024,
            "date": "2024-10-31",
            "content": "Tim Cook: We had a record-breaking quarter with strong revenue growth.",
        }
    )


def _sample_analysis() -> EarningsAnalysis:
    return EarningsAnalysis(
        overall_sentiment="BULLISH",
        confidence=0.85,
        sentiment_indicators=[
            SentimentIndicator(
                category="forward_guidance",
                quote="record-breaking quarter",
                interpretation="Strong performance language",
                impact="BULLISH",
            )
        ],
        key_risks=["Supply chain"],
        key_opportunities=["New products"],
        reasoning_summary="Strong forward guidance with optimistic tone.",
        horizon_days=5,
    )


def _mock_tool_response(data: str) -> MagicMock:
    return MagicMock(content=[MagicMock(text=data)])


class TestRunEarningsLinguist:

    @patch("stock_radar.agents.earnings_linguist.runner.create_predictions_server")
    @patch("stock_radar.agents.earnings_linguist.runner.create_market_server")
    @patch("stock_radar.agents.earnings_linguist.runner.create_vector_store_server")
    async def test_successful_run(
        self, mock_vs_server, mock_market_server, mock_pred_server
    ) -> None:
        """End-to-end: fetches transcript, analyzes, logs prediction."""
        with (
            patch("stock_radar.agents.earnings_linguist.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.earnings_linguist.runner.EarningsLinguistAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.earnings_linguist.runner._load_settings"),
        ):
            # Set up the mock MCP clients
            market_client = AsyncMock()
            market_client.call_tool = AsyncMock(
                return_value=_mock_tool_response(_sample_transcript_response())
            )

            pred_client = AsyncMock()
            pred_client.call_tool = AsyncMock(
                return_value=_mock_tool_response(
                    json.dumps({"prediction_id": "pred-001", "created_at": "2024-01-01T00:00:00Z"})
                )
            )

            vs_client = AsyncMock()

            async def vs_side_effect(tool_name, args):
                if tool_name == "store_embedding":
                    return _mock_tool_response(
                        json.dumps(
                            {
                                "document_id": "d1",
                                "document_type": "reasoning",
                                "collection_name": "reasoning",
                            }
                        )
                    )
                return _mock_tool_response(
                    json.dumps({"query": "q", "document_type": "reasoning", "results": []})
                )

            vs_client.call_tool = AsyncMock(side_effect=vs_side_effect)

            # Client() context manager returns our mock clients in order
            mock_contexts = [market_client, pred_client, vs_client]
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

            # Mock the agent
            mock_result = AnalysisResult(
                ticker="AAPL",
                direction="BULLISH",
                confidence=0.85,
                reasoning="Strong forward guidance.",
                horizon_days=5,
                model_used="llama3.1:8b",
            )
            mock_output = AgentOutput(
                prediction_id="pred-001",
                result=mock_result,
            )
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_output)
            mock_agent_cls.return_value = mock_agent

            output = await run_earnings_linguist("AAPL", 4, 2024)

            assert isinstance(output, AgentOutput)
            assert output.prediction_id == "pred-001"

    @patch("stock_radar.agents.earnings_linguist.runner.create_market_server")
    @patch("stock_radar.agents.earnings_linguist.runner.create_predictions_server")
    @patch("stock_radar.agents.earnings_linguist.runner.create_vector_store_server")
    async def test_transcript_not_found(self, mock_vs, mock_pred, mock_market) -> None:
        """When transcript not available, raises TranscriptNotFoundError."""
        with (
            patch("stock_radar.agents.earnings_linguist.runner.Client") as mock_client_cls,
            patch("stock_radar.agents.earnings_linguist.runner._load_settings"),
        ):
            market_client = AsyncMock()
            market_client.call_tool = AsyncMock(side_effect=Exception("No transcript found"))

            class _MockContext:
                def __init__(self, server):
                    self._client = market_client

                async def __aenter__(self):
                    return self._client

                async def __aexit__(self, *args):
                    pass

            mock_client_cls.side_effect = _MockContext

            with pytest.raises(TranscriptNotFoundError):
                await run_earnings_linguist("AAPL", 4, 2024)

    @patch("stock_radar.agents.earnings_linguist.runner.create_predictions_server")
    @patch("stock_radar.agents.earnings_linguist.runner.create_market_server")
    @patch("stock_radar.agents.earnings_linguist.runner.create_vector_store_server")
    async def test_agent_run_is_called_with_correct_input(
        self, mock_vs_server, mock_market_server, mock_pred_server
    ) -> None:
        """The agent's run() is called with an EarningsLinguistInput for the given ticker."""
        from stock_radar.agents.earnings_linguist.models import EarningsLinguistInput

        with (
            patch("stock_radar.agents.earnings_linguist.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.earnings_linguist.runner.EarningsLinguistAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.earnings_linguist.runner._load_settings"),
        ):
            market_client = AsyncMock()
            market_client.call_tool = AsyncMock(
                return_value=_mock_tool_response(_sample_transcript_response())
            )
            pred_client = AsyncMock()
            vs_client = AsyncMock()

            mock_contexts = [market_client, pred_client, vs_client]
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

            mock_result = AnalysisResult(
                ticker="MSFT",
                direction="BULLISH",
                confidence=0.9,
                reasoning="Strong.",
                horizon_days=5,
                model_used="llama3.1:8b",
            )
            mock_output = AgentOutput(prediction_id="pred-002", result=mock_result)
            mock_agent = AsyncMock()
            mock_agent.run = AsyncMock(return_value=mock_output)
            mock_agent_cls.return_value = mock_agent

            # Override transcript with MSFT
            msft_data = json.dumps(
                {
                    "ticker": "MSFT",
                    "quarter": 1,
                    "year": 2025,
                    "date": "2025-01-29",
                    "content": "Satya Nadella: Azure growth is exceptional.",
                }
            )
            market_client.call_tool = AsyncMock(return_value=_mock_tool_response(msft_data))

            await run_earnings_linguist("MSFT", 1, 2025)

            # Verify run() was called and the first arg is EarningsLinguistInput
            mock_agent.run.assert_called_once()
            call_args = mock_agent.run.call_args
            input_arg = call_args[0][0]
            assert isinstance(input_arg, EarningsLinguistInput)
            assert input_arg.ticker == "MSFT"
            assert input_arg.quarter == 1
            assert input_arg.year == 2025
