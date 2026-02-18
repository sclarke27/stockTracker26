"""Tests for the Narrative vs Price Divergence agent."""

from __future__ import annotations

from unittest.mock import AsyncMock

from stock_radar.agents.models import AnalysisResult
from stock_radar.agents.narrative_divergence.agent import NarrativeDivergenceAgent
from stock_radar.agents.narrative_divergence.config import (
    AGENT_NAME,
    ESCALATION_CONFIDENCE_THRESHOLD,
    ESCALATION_MIN_ARTICLES,
    SIGNAL_TYPE,
)
from stock_radar.agents.narrative_divergence.models import (
    NarrativeAnalysis,
    NarrativeDivergenceInput,
)
from stock_radar.config.settings import NarrativeDivergenceSettings
from stock_radar.llm.base import LlmClient
from stock_radar.llm.models import LlmRequest


def _sample_input(**overrides) -> NarrativeDivergenceInput:
    defaults = {
        "ticker": "AAPL",
        "quarter": 4,
        "year": 2024,
        "sentiment_score": 0.55,
        "article_count": 15,
        "average_sentiment_label": "Somewhat-Bullish",
        "price_return_30d": -0.10,
        "price_return_7d": -0.04,
        "top_articles": [],
    }
    defaults.update(overrides)
    return NarrativeDivergenceInput(**defaults)


def _sample_analysis(**overrides) -> NarrativeAnalysis:
    defaults = {
        "divergence_detected": True,
        "divergence_strength": 0.65,
        "direction": "BULLISH",
        "confidence": 0.75,
        "narrative_summary": "Bullish sentiment in the news.",
        "price_action_summary": "Price down 10% in 30 days.",
        "divergence_explanation": "Market underreacting to positive news.",
        "key_catalysts": ["Analyst upgrade"],
        "horizon_days": 10,
        "reasoning_summary": "Sentiment bullish, price bearish — potential mean reversion.",
    }
    defaults.update(overrides)
    return NarrativeAnalysis(**defaults)


def _mock_llm_client(analysis: NarrativeAnalysis | None = None) -> AsyncMock:
    client = AsyncMock(spec=LlmClient)
    client.generate_structured = AsyncMock(return_value=analysis or _sample_analysis())
    return client


class TestNarrativeDivergenceAgentProperties:
    """Test agent identity properties."""

    def test_agent_name(self) -> None:
        assert NarrativeDivergenceAgent().agent_name == AGENT_NAME

    def test_signal_type(self) -> None:
        assert NarrativeDivergenceAgent().signal_type == SIGNAL_TYPE


class TestNarrativeDivergenceAnalyze:
    """Tests for the analyze() method."""

    async def test_analyze_returns_analysis_result(self) -> None:
        agent = NarrativeDivergenceAgent()
        llm = _mock_llm_client()
        result = await agent.analyze(_sample_input(), llm)

        assert isinstance(result, AnalysisResult)
        assert result.ticker == "AAPL"
        assert result.direction == "BULLISH"
        assert result.confidence == 0.75
        assert result.horizon_days == 10

    async def test_analyze_calls_generate_structured_with_correct_model(self) -> None:
        agent = NarrativeDivergenceAgent()
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        llm.generate_structured.assert_called_once()
        call_args = llm.generate_structured.call_args
        assert call_args[0][1] is NarrativeAnalysis

    async def test_analyze_uses_temperature_from_settings(self) -> None:
        settings = NarrativeDivergenceSettings(temperature=0.5)
        agent = NarrativeDivergenceAgent(settings=settings)
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        call_args = llm.generate_structured.call_args
        request: LlmRequest = call_args[0][0]
        assert request.temperature == 0.5

    async def test_analyze_uses_max_tokens_from_settings(self) -> None:
        settings = NarrativeDivergenceSettings(max_tokens=1024)
        agent = NarrativeDivergenceAgent(settings=settings)
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        call_args = llm.generate_structured.call_args
        request: LlmRequest = call_args[0][0]
        assert request.max_tokens == 1024

    async def test_analyze_maps_reasoning_summary(self) -> None:
        agent = NarrativeDivergenceAgent()
        analysis = _sample_analysis(reasoning_summary="Specific reasoning text.")
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.reasoning == "Specific reasoning text."

    async def test_analyze_maps_direction_from_analysis(self) -> None:
        agent = NarrativeDivergenceAgent()
        analysis = _sample_analysis(direction="BEARISH", confidence=0.6)
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.direction == "BEARISH"

    async def test_analyze_maps_horizon_days(self) -> None:
        agent = NarrativeDivergenceAgent()
        analysis = _sample_analysis(horizon_days=15)
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.horizon_days == 15


class TestNarrativeDivergenceShouldEscalate:
    """Tests for the should_escalate() method."""

    def test_no_pre_escalation_for_sufficient_articles(self) -> None:
        agent = NarrativeDivergenceAgent()
        inp = _sample_input(article_count=ESCALATION_MIN_ARTICLES + 1)
        assert agent.should_escalate(inp) is False

    def test_pre_escalation_when_too_few_articles(self) -> None:
        agent = NarrativeDivergenceAgent()
        inp = _sample_input(article_count=ESCALATION_MIN_ARTICLES - 1)
        assert agent.should_escalate(inp) is True

    def test_no_pre_escalation_at_exact_threshold(self) -> None:
        """Exactly at threshold — not strictly less than, so no escalation."""
        agent = NarrativeDivergenceAgent()
        inp = _sample_input(article_count=ESCALATION_MIN_ARTICLES)
        # article_count < ESCALATION_MIN_ARTICLES: 5 < 5 is False
        assert agent.should_escalate(inp) is False

    def test_post_escalation_on_low_confidence(self) -> None:
        agent = NarrativeDivergenceAgent()
        result = AnalysisResult(
            ticker="AAPL",
            direction="NEUTRAL",
            confidence=ESCALATION_CONFIDENCE_THRESHOLD - 0.01,
            reasoning="r",
            horizon_days=10,
            model_used="llama3.1:8b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is True

    def test_no_post_escalation_on_high_confidence(self) -> None:
        agent = NarrativeDivergenceAgent()
        result = AnalysisResult(
            ticker="AAPL",
            direction="BULLISH",
            confidence=0.9,
            reasoning="r",
            horizon_days=10,
            model_used="llama3.1:8b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is False

    def test_settings_override_article_threshold(self) -> None:
        settings = NarrativeDivergenceSettings(escalation_min_articles=10)
        agent = NarrativeDivergenceAgent(settings=settings)
        # 8 articles < 10 threshold → escalate
        inp = _sample_input(article_count=8)
        assert agent.should_escalate(inp) is True

    def test_settings_override_confidence_threshold(self) -> None:
        settings = NarrativeDivergenceSettings(escalation_confidence_threshold=0.5)
        agent = NarrativeDivergenceAgent(settings=settings)
        result = AnalysisResult(
            ticker="AAPL",
            direction="NEUTRAL",
            confidence=0.45,
            reasoning="r",
            horizon_days=10,
            model_used="llama3.1:8b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is True
