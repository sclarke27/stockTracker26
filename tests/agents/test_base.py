"""Tests for the BaseAgent abstract class."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from stock_radar.agents.base import BaseAgent
from stock_radar.agents.exceptions import EscalationError
from stock_radar.agents.models import AgentInput, AgentOutput, AnalysisResult
from stock_radar.llm.base import LlmClient

# --- Test helpers ---


def _make_result(**overrides: object) -> AnalysisResult:
    defaults = {
        "ticker": "AAPL",
        "direction": "BULLISH",
        "confidence": 0.85,
        "reasoning": "Strong signals.",
        "horizon_days": 5,
        "model_used": "llama3.1:8b",
        "escalated": False,
    }
    defaults.update(overrides)
    return AnalysisResult(**defaults)


def _make_input(**overrides: object) -> AgentInput:
    defaults = {"ticker": "AAPL", "quarter": 4, "year": 2024}
    defaults.update(overrides)
    return AgentInput(**defaults)


def _mock_predictions_client(prediction_id: str = "pred-001") -> AsyncMock:
    """Create a mock FastMCP Client for predictions-db-mcp."""
    client = AsyncMock()
    client.call_tool = AsyncMock(
        return_value=MagicMock(
            content=[
                MagicMock(
                    text=json.dumps(
                        {
                            "prediction_id": prediction_id,
                            "created_at": "2024-01-01T00:00:00Z",
                        }
                    )
                )
            ]
        )
    )
    return client


def _mock_vector_store_client(
    similar_results: list[dict] | None = None,
) -> AsyncMock:
    """Create a mock FastMCP Client for vector-store-mcp."""
    client = AsyncMock()
    if similar_results is None:
        similar_results = []
    # store_embedding returns success, search_similar returns results
    store_response = MagicMock(
        content=[
            MagicMock(
                text=json.dumps(
                    {
                        "document_id": "d1",
                        "document_type": "reasoning",
                        "collection_name": "reasoning",
                    }
                )
            )
        ]
    )
    search_response = MagicMock(
        content=[
            MagicMock(
                text=json.dumps(
                    {
                        "query": "q",
                        "document_type": "reasoning",
                        "results": similar_results,
                    }
                )
            )
        ]
    )

    async def call_tool_side_effect(tool_name: str, args: dict) -> MagicMock:
        if tool_name == "store_embedding":
            return store_response
        if tool_name == "search_similar":
            return search_response
        raise ValueError(f"Unexpected tool: {tool_name}")

    client.call_tool = AsyncMock(side_effect=call_tool_side_effect)
    return client


class _StubAgent(BaseAgent):
    """Concrete agent for testing BaseAgent lifecycle."""

    agent_name = "stub_agent"
    signal_type = "stub_signal"

    def __init__(
        self,
        result: AnalysisResult | None = None,
        escalate: bool = False,
    ) -> None:
        self._result = result or _make_result()
        self._escalate = escalate
        self.analyze_calls: list[LlmClient] = []

    async def analyze(
        self,
        input_data: AgentInput,
        llm_client: LlmClient,
    ) -> AnalysisResult:
        self.analyze_calls.append(llm_client)
        return self._result

    def should_escalate(
        self,
        input_data: AgentInput,
        initial_result: AnalysisResult | None = None,
    ) -> bool:
        if initial_result is None:
            return self._escalate
        # Post-analysis: escalate if confidence < 0.3
        return initial_result.confidence < 0.3


class TestBaseAgentRun:
    """Tests for the run() lifecycle method."""

    @pytest.mark.asyncio
    async def test_happy_path_no_escalation(self) -> None:
        """Normal flow: analyze with Ollama, log, store, return output."""
        agent = _StubAgent()
        ollama = AsyncMock(spec=LlmClient)
        pred_client = _mock_predictions_client("pred-001")
        vs_client = _mock_vector_store_client()

        output = await agent.run(_make_input(), ollama, None, pred_client, vs_client)

        assert isinstance(output, AgentOutput)
        assert output.prediction_id == "pred-001"
        assert output.result.direction == "BULLISH"
        assert len(agent.analyze_calls) == 1
        assert agent.analyze_calls[0] is ollama

    @pytest.mark.asyncio
    async def test_pre_analysis_escalation(self) -> None:
        """When should_escalate returns True pre-analysis, use Anthropic."""
        result = _make_result(model_used="claude-sonnet-4-20250514")
        agent = _StubAgent(result=result, escalate=True)
        ollama = AsyncMock(spec=LlmClient)
        anthropic = AsyncMock(spec=LlmClient)
        pred_client = _mock_predictions_client()
        vs_client = _mock_vector_store_client()

        output = await agent.run(_make_input(), ollama, anthropic, pred_client, vs_client)

        assert output.result.escalated is True
        assert len(agent.analyze_calls) == 1
        assert agent.analyze_calls[0] is anthropic

    @pytest.mark.asyncio
    async def test_pre_analysis_escalation_no_anthropic_raises(self) -> None:
        """Escalation needed but no Anthropic client raises EscalationError."""
        agent = _StubAgent(escalate=True)
        ollama = AsyncMock(spec=LlmClient)

        with pytest.raises(EscalationError, match="no Anthropic client"):
            await agent.run(
                _make_input(),
                ollama,
                None,
                _mock_predictions_client(),
                _mock_vector_store_client(),
            )

    @pytest.mark.asyncio
    async def test_post_analysis_escalation_on_low_confidence(self) -> None:
        """Low confidence from Ollama triggers post-analysis escalation."""
        low_conf_result = _make_result(confidence=0.2)
        high_conf_result = _make_result(confidence=0.9, model_used="claude-sonnet-4-20250514")

        # Agent returns low confidence first, then high confidence
        call_count = 0

        class _EscalatingAgent(_StubAgent):
            async def analyze(self, input_data, llm_client):
                nonlocal call_count
                self.analyze_calls.append(llm_client)
                call_count += 1
                if call_count == 1:
                    return low_conf_result
                return high_conf_result

        agent = _EscalatingAgent(escalate=False)
        ollama = AsyncMock(spec=LlmClient)
        anthropic = AsyncMock(spec=LlmClient)
        pred_client = _mock_predictions_client()
        vs_client = _mock_vector_store_client()

        output = await agent.run(_make_input(), ollama, anthropic, pred_client, vs_client)

        assert output.result.escalated is True
        assert output.result.confidence == 0.9
        assert len(agent.analyze_calls) == 2
        assert agent.analyze_calls[0] is ollama
        assert agent.analyze_calls[1] is anthropic

    @pytest.mark.asyncio
    async def test_post_analysis_no_anthropic_keeps_original(self) -> None:
        """Low confidence but no Anthropic keeps original result (warning only)."""
        low_conf_result = _make_result(confidence=0.2)
        agent = _StubAgent(result=low_conf_result)
        ollama = AsyncMock(spec=LlmClient)
        pred_client = _mock_predictions_client()
        vs_client = _mock_vector_store_client()

        output = await agent.run(_make_input(), ollama, None, pred_client, vs_client)

        assert output.result.confidence == 0.2
        assert output.result.escalated is False

    @pytest.mark.asyncio
    async def test_logs_prediction(self) -> None:
        """Verify prediction is logged with correct args."""
        agent = _StubAgent()
        pred_client = _mock_predictions_client()
        vs_client = _mock_vector_store_client()

        await agent.run(
            _make_input(),
            AsyncMock(spec=LlmClient),
            None,
            pred_client,
            vs_client,
        )

        pred_client.call_tool.assert_called_once()
        call_args = pred_client.call_tool.call_args
        assert call_args[0][0] == "log_prediction"
        args = call_args[0][1]
        assert args["ticker"] == "AAPL"
        assert args["agent_name"] == "stub_agent"
        assert args["signal_type"] == "stub_signal"
        assert args["direction"] == "BULLISH"

    @pytest.mark.asyncio
    async def test_stores_reasoning(self) -> None:
        """Verify reasoning is stored in vector store."""
        agent = _StubAgent()
        pred_client = _mock_predictions_client("pred-xyz")
        vs_client = _mock_vector_store_client()

        await agent.run(
            _make_input(),
            AsyncMock(spec=LlmClient),
            None,
            pred_client,
            vs_client,
        )

        # Find the store_embedding call
        store_calls = [
            c for c in vs_client.call_tool.call_args_list if c[0][0] == "store_embedding"
        ]
        assert len(store_calls) == 1
        args = store_calls[0][0][1]
        assert args["document_type"] == "reasoning"
        assert args["document_id"] == "pred-xyz"
        assert args["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_fetches_similar_reasoning(self) -> None:
        """Similar past reasoning is included in output."""
        similar = [
            {
                "document_id": "r1",
                "content": "Prior bullish signal.",
                "metadata": {},
                "distance": 0.1,
            },
            {
                "document_id": "r2",
                "content": "Another signal.",
                "metadata": {},
                "distance": 0.2,
            },
        ]
        agent = _StubAgent()
        pred_client = _mock_predictions_client()
        vs_client = _mock_vector_store_client(similar_results=similar)

        output = await agent.run(
            _make_input(),
            AsyncMock(spec=LlmClient),
            None,
            pred_client,
            vs_client,
        )

        assert len(output.similar_past_reasoning) == 2
        assert "Prior bullish signal." in output.similar_past_reasoning

    @pytest.mark.asyncio
    async def test_similar_search_failure_non_fatal(self) -> None:
        """If similar search fails, output still succeeds with empty list."""
        agent = _StubAgent()
        pred_client = _mock_predictions_client()
        vs_client = AsyncMock()

        # store_embedding succeeds, search_similar fails
        async def side_effect(tool_name, args):
            if tool_name == "store_embedding":
                return MagicMock(
                    content=[
                        MagicMock(
                            text=json.dumps(
                                {
                                    "document_id": "d1",
                                    "document_type": "reasoning",
                                    "collection_name": "reasoning",
                                }
                            )
                        )
                    ]
                )
            raise RuntimeError("Search failed")

        vs_client.call_tool = AsyncMock(side_effect=side_effect)

        output = await agent.run(
            _make_input(),
            AsyncMock(spec=LlmClient),
            None,
            pred_client,
            vs_client,
        )

        assert output.similar_past_reasoning == []
        assert output.prediction_id is not None
