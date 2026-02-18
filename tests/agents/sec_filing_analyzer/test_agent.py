"""Tests for the SEC Filing Pattern Analyzer agent."""

from __future__ import annotations

from unittest.mock import AsyncMock

from stock_radar.agents.models import AnalysisResult
from stock_radar.agents.sec_filing_analyzer.agent import SecFilingAnalyzerAgent
from stock_radar.agents.sec_filing_analyzer.config import (
    AGENT_NAME,
    ESCALATION_CONFIDENCE_THRESHOLD,
    ESCALATION_FILING_COUNT,
    ESCALATION_INSIDER_TRANSACTION_COUNT,
    SIGNAL_TYPE,
)
from stock_radar.agents.sec_filing_analyzer.models import (
    FilingPattern,
    InsiderSummary,
    SecFilingAnalysis,
    SecFilingInput,
)
from stock_radar.config.settings import SecFilingAnalyzerSettings
from stock_radar.llm.base import LlmClient
from stock_radar.llm.models import LlmRequest


def _sample_input(**overrides) -> SecFilingInput:
    defaults = {
        "ticker": "TSLA",
        "quarter": 2,
        "year": 2024,
        "recent_filings": [
            {"form_type": "8-K", "filed_at": "2024-06-15", "description": "Material event"}
        ],
        "insider_transactions": [
            {
                "insider_name": "Elon Musk",
                "transaction_type": "S",
                "shares": 50000,
                "date": "2024-06-10",
            }
        ],
        "filing_count": 1,
        "insider_transaction_count": 1,
        "lookback_days": 90,
    }
    defaults.update(overrides)
    return SecFilingInput(**defaults)


def _sample_analysis(**overrides) -> SecFilingAnalysis:
    defaults = {
        "patterns_detected": [
            FilingPattern(
                pattern_type="insider_selling_cluster",
                description="CEO sold large block.",
                significance="HIGH",
                filing_dates=["2024-06-10"],
            )
        ],
        "insider_summary": InsiderSummary(
            net_shares_acquired=-50000.0,
            total_transactions=1,
            unique_insiders=1,
            largest_transaction_shares=50000.0,
        ),
        "insider_sentiment": "BEARISH",
        "direction": "BEARISH",
        "confidence": 0.78,
        "risk_flags": ["CEO sell"],
        "key_findings": ["Large insider sale"],
        "horizon_days": 15,
        "reasoning_summary": "Insider selling indicates negative outlook.",
    }
    defaults.update(overrides)
    return SecFilingAnalysis(**defaults)


def _mock_llm_client(analysis: SecFilingAnalysis | None = None) -> AsyncMock:
    client = AsyncMock(spec=LlmClient)
    client.generate_structured = AsyncMock(return_value=analysis or _sample_analysis())
    return client


class TestSecFilingAnalyzerAgentProperties:
    """Test agent identity properties."""

    def test_agent_name(self) -> None:
        assert SecFilingAnalyzerAgent().agent_name == AGENT_NAME

    def test_signal_type(self) -> None:
        assert SecFilingAnalyzerAgent().signal_type == SIGNAL_TYPE


class TestSecFilingAnalyzerAnalyze:
    """Tests for the analyze() method."""

    async def test_analyze_returns_analysis_result(self) -> None:
        agent = SecFilingAnalyzerAgent()
        llm = _mock_llm_client()
        result = await agent.analyze(_sample_input(), llm)

        assert isinstance(result, AnalysisResult)
        assert result.ticker == "TSLA"
        assert result.direction == "BEARISH"
        assert result.confidence == 0.78
        assert result.horizon_days == 15

    async def test_analyze_calls_generate_structured_with_correct_model(self) -> None:
        agent = SecFilingAnalyzerAgent()
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        llm.generate_structured.assert_called_once()
        call_args = llm.generate_structured.call_args
        assert call_args[0][1] is SecFilingAnalysis

    async def test_analyze_uses_temperature_from_settings(self) -> None:
        settings = SecFilingAnalyzerSettings(temperature=0.2)
        agent = SecFilingAnalyzerAgent(settings=settings)
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        call_args = llm.generate_structured.call_args
        request: LlmRequest = call_args[0][0]
        assert request.temperature == 0.2

    async def test_analyze_uses_max_tokens_from_settings(self) -> None:
        settings = SecFilingAnalyzerSettings(max_tokens=1024)
        agent = SecFilingAnalyzerAgent(settings=settings)
        llm = _mock_llm_client()

        await agent.analyze(_sample_input(), llm)

        call_args = llm.generate_structured.call_args
        request: LlmRequest = call_args[0][0]
        assert request.max_tokens == 1024

    async def test_analyze_maps_reasoning_summary(self) -> None:
        agent = SecFilingAnalyzerAgent()
        analysis = _sample_analysis(reasoning_summary="Detailed SEC analysis reasoning.")
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.reasoning == "Detailed SEC analysis reasoning."

    async def test_analyze_maps_direction(self) -> None:
        agent = SecFilingAnalyzerAgent()
        analysis = _sample_analysis(direction="BULLISH", confidence=0.8)
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.direction == "BULLISH"

    async def test_analyze_maps_horizon_days(self) -> None:
        agent = SecFilingAnalyzerAgent()
        analysis = _sample_analysis(horizon_days=20)
        llm = _mock_llm_client(analysis)

        result = await agent.analyze(_sample_input(), llm)

        assert result.horizon_days == 20


class TestSecFilingAnalyzerShouldEscalate:
    """Tests for the should_escalate() method."""

    def test_no_pre_escalation_below_filing_threshold(self) -> None:
        agent = SecFilingAnalyzerAgent()
        inp = _sample_input(filing_count=ESCALATION_FILING_COUNT - 1, insider_transaction_count=1)
        assert agent.should_escalate(inp) is False

    def test_pre_escalation_when_filing_count_exceeds_threshold(self) -> None:
        agent = SecFilingAnalyzerAgent()
        inp = _sample_input(filing_count=ESCALATION_FILING_COUNT + 1, insider_transaction_count=1)
        assert agent.should_escalate(inp) is True

    def test_pre_escalation_when_insider_count_exceeds_threshold(self) -> None:
        agent = SecFilingAnalyzerAgent()
        inp = _sample_input(
            filing_count=5,
            insider_transaction_count=ESCALATION_INSIDER_TRANSACTION_COUNT + 1,
        )
        assert agent.should_escalate(inp) is True

    def test_no_pre_escalation_at_exact_filing_threshold(self) -> None:
        """Exactly at threshold — not strictly greater than, so no escalation."""
        agent = SecFilingAnalyzerAgent()
        inp = _sample_input(filing_count=ESCALATION_FILING_COUNT, insider_transaction_count=1)
        # filing_count > ESCALATION_FILING_COUNT: 30 > 30 is False
        assert agent.should_escalate(inp) is False

    def test_post_escalation_on_low_confidence(self) -> None:
        agent = SecFilingAnalyzerAgent()
        result = AnalysisResult(
            ticker="TSLA",
            direction="NEUTRAL",
            confidence=ESCALATION_CONFIDENCE_THRESHOLD - 0.01,
            reasoning="r",
            horizon_days=15,
            model_used="llama3.1:8b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is True

    def test_no_post_escalation_on_high_confidence(self) -> None:
        agent = SecFilingAnalyzerAgent()
        result = AnalysisResult(
            ticker="TSLA",
            direction="BEARISH",
            confidence=0.8,
            reasoning="r",
            horizon_days=15,
            model_used="llama3.1:8b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is False

    def test_settings_override_filing_threshold(self) -> None:
        settings = SecFilingAnalyzerSettings(escalation_filing_count=10)
        agent = SecFilingAnalyzerAgent(settings=settings)
        inp = _sample_input(filing_count=11, insider_transaction_count=1)
        assert agent.should_escalate(inp) is True

    def test_settings_override_confidence_threshold(self) -> None:
        settings = SecFilingAnalyzerSettings(escalation_confidence_threshold=0.6)
        agent = SecFilingAnalyzerAgent(settings=settings)
        result = AnalysisResult(
            ticker="TSLA",
            direction="NEUTRAL",
            confidence=0.55,
            reasoning="r",
            horizon_days=15,
            model_used="llama3.1:8b",
        )
        assert agent.should_escalate(_sample_input(), initial_result=result) is True
