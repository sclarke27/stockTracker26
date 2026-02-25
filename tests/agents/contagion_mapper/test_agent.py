"""Tests for the Cross-Sector Contagion Mapper agent."""

from __future__ import annotations

from unittest.mock import AsyncMock

from stock_radar.agents.contagion_mapper.agent import ContagionMapperAgent
from stock_radar.agents.contagion_mapper.config import (
    AGENT_NAME,
    ESCALATION_CONFIDENCE_THRESHOLD,
    SIGNAL_TYPE,
)
from stock_radar.agents.contagion_mapper.models import (
    ContagionAnalysis,
    ContagionInput,
)
from stock_radar.agents.models import AnalysisResult
from stock_radar.config.settings import ContagionMapperSettings
from stock_radar.llm.base import LlmClient
from stock_radar.llm.models import LlmRequest


def _sample_input(**overrides) -> ContagionInput:
    defaults = {
        "ticker": "AMD",
        "quarter": 3,
        "year": 2024,
        "trigger_ticker": "NVDA",
        "trigger_company_name": "NVIDIA Corporation",
        "trigger_event_summary": "NVDA missed earnings guidance.",
        "target_company_name": "Advanced Micro Devices",
        "relationship_type": "competitor",
        "trigger_recent_news": [],
        "target_recent_news": [],
        "trigger_sector": "Semiconductors",
        "target_sector": "Semiconductors",
    }
    defaults.update(overrides)
    return ContagionInput(**defaults)


def _sample_analysis(**overrides) -> ContagionAnalysis:
    defaults = {
        "contagion_likely": True,
        "contagion_probability": 0.65,
        "contagion_mechanism": "Same customer base reduces revenue for both.",
        "direction": "BEARISH",
        "confidence": 0.70,
        "affected_business_segments": ["Data Center"],
        "timeline_days": 3,
        "mitigating_factors": [],
        "amplifying_factors": ["Same end markets"],
        "horizon_days": 5,
        "reasoning_summary": "NVDA miss signals broad sector softness impacting AMD.",
    }
    defaults.update(overrides)
    return ContagionAnalysis(**defaults)


def _mock_llm_client(analysis: ContagionAnalysis | None = None) -> AsyncMock:
    client = AsyncMock(spec=LlmClient)
    client.generate_structured = AsyncMock(return_value=analysis or _sample_analysis())
    return client


class TestContagionMapperAgentProperties:
    """Test agent identity properties."""

    def test_agent_name(self) -> None:
        assert ContagionMapperAgent().agent_name == AGENT_NAME

    def test_signal_type(self) -> None:
        assert ContagionMapperAgent().signal_type == SIGNAL_TYPE


class TestContagionMapperAnalyze:
    """Tests for the analyze() method."""

    async def test_analyze_returns_analysis_result(self) -> None:
        agent = ContagionMapperAgent()
        llm = _mock_llm_client()
        result = await agent.analyze(_sample_input(), llm)

        assert isinstance(result, AnalysisResult)
        assert result.ticker == "AMD"
        assert result.direction == "BEARISH"
        assert result.confidence == 0.70
        assert result.horizon_days == 5

    async def test_analyze_calls_generate_structured_with_correct_model(self) -> None:
        agent = ContagionMapperAgent()
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        llm.generate_structured.assert_called_once()
        call_args = llm.generate_structured.call_args
        assert call_args[0][1] is ContagionAnalysis

    async def test_analyze_uses_temperature_from_settings(self) -> None:
        settings = ContagionMapperSettings(temperature=0.3)
        agent = ContagionMapperAgent(settings=settings)
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        call_args = llm.generate_structured.call_args
        request: LlmRequest = call_args[0][0]
        assert request.temperature == 0.3

    async def test_analyze_uses_max_tokens_from_settings(self) -> None:
        settings = ContagionMapperSettings(max_tokens=512)
        agent = ContagionMapperAgent(settings=settings)
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        call_args = llm.generate_structured.call_args
        request: LlmRequest = call_args[0][0]
        assert request.max_tokens == 512

    async def test_analyze_maps_reasoning_summary(self) -> None:
        agent = ContagionMapperAgent()
        analysis = _sample_analysis(reasoning_summary="Specific contagion reasoning.")
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.reasoning == "Specific contagion reasoning."

    async def test_analyze_maps_direction(self) -> None:
        agent = ContagionMapperAgent()
        analysis = _sample_analysis(direction="BULLISH", confidence=0.6)
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.direction == "BULLISH"

    async def test_analyze_maps_horizon_days(self) -> None:
        agent = ContagionMapperAgent()
        analysis = _sample_analysis(horizon_days=7)
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.horizon_days == 7

    async def test_ticker_is_target_in_result(self) -> None:
        """Result ticker should be the target (AMD), not the trigger (NVDA)."""
        agent = ContagionMapperAgent()
        llm = _mock_llm_client()
        result = await agent.analyze(_sample_input(), llm)

        assert result.ticker == "AMD"  # target
        # not NVDA (the trigger)


class TestContagionMapperShouldEscalate:
    """Tests for the should_escalate() method."""

    def test_no_pre_escalation(self) -> None:
        """Contagion always uses Ollama pre-analysis (well-defined task)."""
        agent = ContagionMapperAgent()
        inp = _sample_input()
        assert agent.should_escalate(inp) is False

    def test_post_escalation_on_low_confidence(self) -> None:
        agent = ContagionMapperAgent()
        result = AnalysisResult(
            ticker="AMD",
            direction="NEUTRAL",
            confidence=ESCALATION_CONFIDENCE_THRESHOLD - 0.01,
            reasoning="r",
            horizon_days=5,
            model_used="qwen3:32b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is True

    def test_no_post_escalation_on_high_confidence(self) -> None:
        agent = ContagionMapperAgent()
        result = AnalysisResult(
            ticker="AMD",
            direction="BEARISH",
            confidence=0.85,
            reasoning="r",
            horizon_days=5,
            model_used="qwen3:32b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is False

    def test_settings_override_confidence_threshold(self) -> None:
        settings = ContagionMapperSettings(escalation_confidence_threshold=0.5)
        agent = ContagionMapperAgent(settings=settings)
        result = AnalysisResult(
            ticker="AMD",
            direction="NEUTRAL",
            confidence=0.45,
            reasoning="r",
            horizon_days=5,
            model_used="qwen3:32b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is True
