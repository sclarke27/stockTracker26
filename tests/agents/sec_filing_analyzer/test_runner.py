"""Tests for the SEC Filing Pattern Analyzer runner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from stock_radar.agents.models import AgentOutput, AnalysisResult
from stock_radar.agents.sec_filing_analyzer.models import SecFilingInput
from stock_radar.agents.sec_filing_analyzer.runner import run_sec_filing_analyzer
from tests.agents.helpers import mock_tool_response as _mock_tool_response


def _sample_filings_response() -> str:
    return json.dumps(
        {
            "ticker": "TSLA",
            "filings": [
                {
                    "form_type": "8-K",
                    "filed_at": "2024-06-15",
                    "description": "Material event",
                    "url": "https://sec.gov/...",
                },
                {
                    "form_type": "10-Q",
                    "filed_at": "2024-05-10",
                    "description": "Quarterly report",
                    "url": "https://sec.gov/...",
                },
            ],
            "total_count": 2,
        }
    )


def _sample_insider_response() -> str:
    return json.dumps(
        {
            "ticker": "TSLA",
            "transactions": [
                {
                    "insider_name": "Elon Musk",
                    "title": "CEO",
                    "transaction_type": "S",
                    "shares": 100000,
                    "price_per_share": 200.0,
                    "total_value": 20000000.0,
                    "date": "2024-06-10",
                    "form_type": "Form 4",
                }
            ],
            "total_count": 1,
        }
    )


def _sample_output() -> AgentOutput:
    return AgentOutput(
        prediction_id="pred-sec-001",
        result=AnalysisResult(
            ticker="TSLA",
            direction="BEARISH",
            confidence=0.75,
            reasoning="Insider selling indicates bearish outlook.",
            horizon_days=15,
            model_used="llama3.1:8b",
        ),
    )


class TestRunSecFilingAnalyzer:

    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_predictions_server")
    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_vector_store_server")
    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_sec_edgar_server")
    async def test_successful_run(self, mock_sec, mock_vs, mock_pred) -> None:
        """End-to-end: fetches filings + insider transactions and runs agent."""
        with (
            patch("stock_radar.agents.sec_filing_analyzer.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.sec_filing_analyzer.runner.SecFilingAnalyzerAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.sec_filing_analyzer.runner._load_settings"),
        ):
            sec_client = AsyncMock()

            def sec_side_effect(tool_name, args):
                if tool_name == "get_filings":
                    return _mock_tool_response(_sample_filings_response())
                return _mock_tool_response(_sample_insider_response())

            sec_client.call_tool = AsyncMock(side_effect=sec_side_effect)
            pred_client = AsyncMock()
            vs_client = AsyncMock()

            mock_contexts = [sec_client, pred_client, vs_client]
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

            output = await run_sec_filing_analyzer("TSLA", 2, 2024)

            assert isinstance(output, AgentOutput)
            assert output.prediction_id == "pred-sec-001"

    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_predictions_server")
    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_vector_store_server")
    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_sec_edgar_server")
    async def test_agent_called_with_sec_filing_input(self, mock_sec, mock_vs, mock_pred) -> None:
        """The agent's run() receives a SecFilingInput."""
        with (
            patch("stock_radar.agents.sec_filing_analyzer.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.sec_filing_analyzer.runner.SecFilingAnalyzerAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.sec_filing_analyzer.runner._load_settings"),
        ):
            sec_client = AsyncMock()

            def sec_side_effect(tool_name, args):
                if tool_name == "get_filings":
                    return _mock_tool_response(_sample_filings_response())
                return _mock_tool_response(_sample_insider_response())

            sec_client.call_tool = AsyncMock(side_effect=sec_side_effect)
            pred_client = AsyncMock()
            vs_client = AsyncMock()

            mock_contexts = [sec_client, pred_client, vs_client]
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

            await run_sec_filing_analyzer("TSLA", 2, 2024)

            mock_agent.run.assert_called_once()
            call_args = mock_agent.run.call_args
            input_arg = call_args[0][0]
            assert isinstance(input_arg, SecFilingInput)
            assert input_arg.ticker == "TSLA"

    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_predictions_server")
    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_vector_store_server")
    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_sec_edgar_server")
    async def test_filings_fetch_failure_raises(self, mock_sec, mock_vs, mock_pred) -> None:
        """If the SEC EDGAR fetch fails, the error propagates."""
        with (
            patch("stock_radar.agents.sec_filing_analyzer.runner.Client") as mock_client_cls,
            patch("stock_radar.agents.sec_filing_analyzer.runner._load_settings"),
        ):
            sec_client = AsyncMock()
            sec_client.call_tool = AsyncMock(side_effect=Exception("SEC EDGAR unavailable"))

            class _MockContext:
                def __init__(self, server):
                    self._client = sec_client

                async def __aenter__(self):
                    return self._client

                async def __aexit__(self, *args):
                    pass

            mock_client_cls.side_effect = _MockContext

            with pytest.raises(Exception, match="SEC EDGAR unavailable"):
                await run_sec_filing_analyzer("TSLA", 2, 2024)

    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_predictions_server")
    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_vector_store_server")
    @patch("stock_radar.agents.sec_filing_analyzer.runner.create_sec_edgar_server")
    async def test_insider_fetch_failure_uses_empty_list(
        self, mock_sec, mock_vs, mock_pred
    ) -> None:
        """If insider transaction fetch fails, proceeds with empty list (non-fatal)."""
        with (
            patch("stock_radar.agents.sec_filing_analyzer.runner.Client") as mock_client_cls,
            patch(
                "stock_radar.agents.sec_filing_analyzer.runner.SecFilingAnalyzerAgent"
            ) as mock_agent_cls,
            patch("stock_radar.agents.sec_filing_analyzer.runner._load_settings"),
        ):
            sec_client = AsyncMock()

            def sec_side_effect(tool_name, args):
                if tool_name == "get_filings":
                    return _mock_tool_response(_sample_filings_response())
                raise Exception("Insider fetch failed")

            sec_client.call_tool = AsyncMock(side_effect=sec_side_effect)
            pred_client = AsyncMock()
            vs_client = AsyncMock()

            mock_contexts = [sec_client, pred_client, vs_client]
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

            # Should not raise — insider failure is non-fatal
            output = await run_sec_filing_analyzer("TSLA", 2, 2024)
            assert output.prediction_id == "pred-sec-001"

            # Verify input had empty insider list
            call_args = mock_agent.run.call_args
            input_arg = call_args[0][0]
            assert isinstance(input_arg, SecFilingInput)
            assert input_arg.insider_transactions == []
            assert input_arg.insider_transaction_count == 0
