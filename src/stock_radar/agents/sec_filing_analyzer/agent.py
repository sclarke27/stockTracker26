"""SEC Filing Pattern Analyzer agent."""

from __future__ import annotations

from stock_radar.agents.base import BaseAgent
from stock_radar.agents.models import AgentInput, AnalysisResult
from stock_radar.agents.sec_filing_analyzer.config import (
    AGENT_NAME,
    DEFAULT_HORIZON_DAYS,
    ESCALATION_FILING_COUNT,
    ESCALATION_INSIDER_TRANSACTION_COUNT,
    SIGNAL_TYPE,
)
from stock_radar.agents.sec_filing_analyzer.models import (
    SecFilingAnalysis,
    SecFilingInput,
)
from stock_radar.agents.sec_filing_analyzer.prompt import build_messages
from stock_radar.config.settings import SecFilingAnalyzerSettings
from stock_radar.llm.base import LlmClient
from stock_radar.llm.models import LlmRequest


class SecFilingAnalyzerAgent(BaseAgent):
    """Analyzes SEC filings and insider transactions for predictive patterns.

    Detects insider buying/selling clusters, unusual 8-K frequency,
    late filings, and executive departures that often precede price moves.
    """

    def __init__(self, settings: SecFilingAnalyzerSettings | None = None) -> None:
        self._settings = settings or SecFilingAnalyzerSettings()

    @property
    def agent_name(self) -> str:
        return AGENT_NAME

    @property
    def signal_type(self) -> str:
        return SIGNAL_TYPE

    async def analyze(
        self,
        input_data: AgentInput,
        llm_client: LlmClient,
    ) -> AnalysisResult:
        """Analyze SEC filing patterns using the LLM.

        Builds a prompt from filings and insider transactions, sends it
        to the LLM, and maps the structured response to an AnalysisResult.

        Args:
            input_data: Must be a SecFilingInput.
            llm_client: LLM client to use for inference.

        Returns:
            Analysis result with prediction and reasoning.
        """
        sf_input = input_data if isinstance(input_data, SecFilingInput) else None

        recent_filings = sf_input.recent_filings if sf_input else []
        insider_transactions = sf_input.insider_transactions if sf_input else []
        filing_count = sf_input.filing_count if sf_input else 0
        insider_transaction_count = sf_input.insider_transaction_count if sf_input else 0
        lookback_days = sf_input.lookback_days if sf_input else 90

        messages = build_messages(
            ticker=input_data.ticker,
            recent_filings=recent_filings,
            insider_transactions=insider_transactions,
            filing_count=filing_count,
            insider_transaction_count=insider_transaction_count,
            lookback_days=lookback_days,
        )

        request = LlmRequest(
            messages=messages,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )

        analysis: SecFilingAnalysis = await llm_client.generate_structured(
            request, SecFilingAnalysis
        )

        return AnalysisResult(
            ticker=input_data.ticker,
            direction=analysis.direction,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning_summary,
            horizon_days=analysis.horizon_days or DEFAULT_HORIZON_DAYS,
            model_used=getattr(llm_client, "_model", "unknown"),
        )

    def should_escalate(
        self,
        input_data: AgentInput,
        initial_result: AnalysisResult | None = None,
    ) -> bool:
        """Check if escalation to Claude API is needed.

        Pre-analysis escalation (initial_result is None):
        - Filing count exceeds threshold (complex synthesis task).
        - Insider transaction count exceeds threshold.

        Post-analysis escalation (initial_result provided):
        - Confidence below threshold.

        Args:
            input_data: The analysis input.
            initial_result: Result from first analysis attempt, if available.

        Returns:
            True if escalation is warranted.
        """
        if initial_result is not None:
            return initial_result.confidence < self._settings.escalation_confidence_threshold

        sf_input = input_data if isinstance(input_data, SecFilingInput) else None
        if sf_input is None:
            return False

        filing_threshold = self._settings.escalation_filing_count or ESCALATION_FILING_COUNT
        insider_threshold = ESCALATION_INSIDER_TRANSACTION_COUNT
        return (
            sf_input.filing_count > filing_threshold
            or sf_input.insider_transaction_count > insider_threshold
        )
