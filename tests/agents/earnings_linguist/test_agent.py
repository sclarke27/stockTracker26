"""Tests for the Earnings Linguist agent."""

from __future__ import annotations

from unittest.mock import AsyncMock

from stock_radar.agents.earnings_linguist.agent import EarningsLinguistAgent
from stock_radar.agents.earnings_linguist.config import (
    AGENT_NAME,
    ESCALATION_CONFIDENCE_THRESHOLD,
    ESCALATION_TRANSCRIPT_LENGTH,
    SIGNAL_TYPE,
)
from stock_radar.agents.earnings_linguist.models import (
    EarningsAnalysis,
    EarningsLinguistInput,
    SentimentIndicator,
)
from stock_radar.agents.models import AnalysisResult
from stock_radar.config.settings import EarningsLinguistSettings
from stock_radar.llm.base import LlmClient
from stock_radar.llm.models import LlmRequest


def _sample_analysis(**overrides) -> EarningsAnalysis:
    """Create a sample EarningsAnalysis for testing."""
    defaults = {
        "overall_sentiment": "BULLISH",
        "confidence": 0.85,
        "sentiment_indicators": [
            SentimentIndicator(
                category="forward_guidance",
                quote="We expect strong Q1 results.",
                interpretation="Positive forward guidance from management.",
                impact="BULLISH",
            )
        ],
        "key_risks": ["Supply chain delays"],
        "key_opportunities": ["New product launch in Q1"],
        "reasoning_summary": "Management tone is optimistic with strong forward guidance.",
        "horizon_days": 5,
    }
    defaults.update(overrides)
    return EarningsAnalysis(**defaults)


def _sample_input(**overrides) -> EarningsLinguistInput:
    """Create a sample input."""
    defaults = {
        "ticker": "AAPL",
        "quarter": 4,
        "year": 2024,
        "transcript_content": (
            "Tim Cook: We had an outstanding quarter" " with revenue growth across all segments."
        ),
        "company_name": "Apple Inc.",
    }
    defaults.update(overrides)
    return EarningsLinguistInput(**defaults)


def _mock_llm_client(analysis: EarningsAnalysis | None = None) -> AsyncMock:
    """Create a mock LLM client that returns a predetermined analysis."""
    client = AsyncMock(spec=LlmClient)
    client.generate_structured = AsyncMock(return_value=analysis or _sample_analysis())
    return client


class TestEarningsLinguistAgentProperties:
    """Test agent identity properties."""

    def test_agent_name(self) -> None:
        agent = EarningsLinguistAgent()
        assert agent.agent_name == AGENT_NAME

    def test_signal_type(self) -> None:
        agent = EarningsLinguistAgent()
        assert agent.signal_type == SIGNAL_TYPE


class TestEarningsLinguistAnalyze:
    """Tests for the analyze() method."""

    async def test_analyze_returns_analysis_result(self) -> None:
        """analyze() calls LLM and maps EarningsAnalysis to AnalysisResult."""
        agent = EarningsLinguistAgent()
        llm = _mock_llm_client()
        input_data = _sample_input()

        result = await agent.analyze(input_data, llm)

        assert isinstance(result, AnalysisResult)
        assert result.ticker == "AAPL"
        assert result.direction == "BULLISH"
        assert result.confidence == 0.85
        assert result.horizon_days == 5
        assert (
            "optimistic" in result.reasoning.lower()
            or "forward guidance" in result.reasoning.lower()
        )

    async def test_analyze_calls_generate_structured(self) -> None:
        """analyze() calls generate_structured with EarningsAnalysis model."""
        agent = EarningsLinguistAgent()
        llm = _mock_llm_client()
        input_data = _sample_input()

        await agent.analyze(input_data, llm)

        llm.generate_structured.assert_called_once()
        call_args = llm.generate_structured.call_args
        # Second positional arg is the response model class
        assert call_args[0][1] is EarningsAnalysis

    async def test_analyze_includes_prior_transcript(self) -> None:
        """When prior transcript is provided, it's included in the prompt."""
        agent = EarningsLinguistAgent()
        llm = _mock_llm_client()
        input_data = _sample_input(prior_transcript_content="Prior quarter was weak.")

        await agent.analyze(input_data, llm)

        # Verify the request messages contain the prior transcript
        call_args = llm.generate_structured.call_args
        request = call_args[0][0]  # LlmRequest
        assert isinstance(request, LlmRequest)
        user_msg = request.messages[1].content
        assert "Prior quarter was weak." in user_msg

    async def test_analyze_uses_correct_temperature(self) -> None:
        """Temperature from settings is passed to the LLM request."""
        settings = EarningsLinguistSettings(temperature=0.5)
        agent = EarningsLinguistAgent(settings=settings)
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        call_args = llm.generate_structured.call_args
        request = call_args[0][0]
        assert request.temperature == 0.5

    async def test_analyze_uses_correct_max_tokens(self) -> None:
        """Max tokens from settings is passed to the LLM request."""
        settings = EarningsLinguistSettings(max_tokens=2048)
        agent = EarningsLinguistAgent(settings=settings)
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        call_args = llm.generate_structured.call_args
        request = call_args[0][0]
        assert request.max_tokens == 2048

    async def test_analyze_maps_reasoning_summary(self) -> None:
        """The reasoning_summary from EarningsAnalysis becomes result.reasoning."""
        agent = EarningsLinguistAgent()
        analysis = _sample_analysis(reasoning_summary="Detailed reasoning text here.")
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.reasoning == "Detailed reasoning text here."

    async def test_analyze_uses_default_horizon_when_missing(self) -> None:
        """Falls back to DEFAULT_HORIZON_DAYS when horizon_days is missing from analysis."""
        from stock_radar.agents.earnings_linguist.config import DEFAULT_HORIZON_DAYS

        agent = EarningsLinguistAgent()
        # horizon_days defaults to DEFAULT_HORIZON_DAYS in the model already,
        # but we test that it passes through correctly.
        analysis = _sample_analysis(horizon_days=DEFAULT_HORIZON_DAYS)
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.horizon_days == DEFAULT_HORIZON_DAYS


class TestEarningsLinguistShouldEscalate:
    """Tests for the should_escalate() method."""

    def test_no_escalation_for_short_transcript(self) -> None:
        """Normal-length transcript does not trigger escalation."""
        agent = EarningsLinguistAgent()
        input_data = _sample_input(transcript_content="Short transcript.")
        assert agent.should_escalate(input_data) is False

    def test_escalates_on_long_transcript(self) -> None:
        """Transcript exceeding threshold triggers pre-analysis escalation."""
        agent = EarningsLinguistAgent()
        long_text = "x" * (ESCALATION_TRANSCRIPT_LENGTH + 1)
        input_data = _sample_input(transcript_content=long_text)
        assert agent.should_escalate(input_data) is True

    def test_escalates_on_combined_length(self) -> None:
        """Combined current + prior transcript exceeding threshold triggers escalation."""
        agent = EarningsLinguistAgent()
        half = "x" * (ESCALATION_TRANSCRIPT_LENGTH // 2 + 1)
        input_data = _sample_input(
            transcript_content=half,
            prior_transcript_content=half,
        )
        assert agent.should_escalate(input_data) is True

    def test_post_analysis_escalation_low_confidence(self) -> None:
        """Low confidence in initial result triggers post-analysis escalation."""
        agent = EarningsLinguistAgent()
        result = AnalysisResult(
            ticker="AAPL",
            direction="NEUTRAL",
            confidence=ESCALATION_CONFIDENCE_THRESHOLD - 0.01,
            reasoning="r",
            horizon_days=5,
            model_used="llama3.1:8b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is True

    def test_no_post_escalation_high_confidence(self) -> None:
        """High confidence does not trigger post-analysis escalation."""
        agent = EarningsLinguistAgent()
        result = AnalysisResult(
            ticker="AAPL",
            direction="BULLISH",
            confidence=0.9,
            reasoning="r",
            horizon_days=5,
            model_used="llama3.1:8b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is False
