"""Earnings Linguist agent — sentiment analysis of earnings transcripts."""

from __future__ import annotations

from stock_radar.agents.base import BaseAgent
from stock_radar.agents.earnings_linguist.config import (
    AGENT_NAME,
    DEFAULT_HORIZON_DAYS,
    SIGNAL_TYPE,
)
from stock_radar.agents.earnings_linguist.models import (
    EarningsAnalysis,
    EarningsLinguistInput,
)
from stock_radar.agents.earnings_linguist.prompt import build_messages
from stock_radar.agents.models import AgentInput, AnalysisResult
from stock_radar.config.settings import EarningsLinguistSettings
from stock_radar.llm.base import LlmClient
from stock_radar.llm.models import LlmRequest


class EarningsLinguistAgent(BaseAgent):
    """Analyzes earnings call transcripts for sentiment signals.

    Detects hedging language, confidence shifts, tone changes,
    and quarter-over-quarter sentiment evolution.
    """

    def __init__(self, settings: EarningsLinguistSettings | None = None) -> None:
        self._settings = settings or EarningsLinguistSettings()

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
        """Analyze an earnings transcript using the LLM.

        Builds a prompt from the transcript, sends it to the LLM,
        and maps the structured response to an AnalysisResult.

        Args:
            input_data: Must be an EarningsLinguistInput.
            llm_client: LLM client to use for inference.

        Returns:
            Analysis result with prediction and reasoning.
        """
        el_input = input_data if isinstance(input_data, EarningsLinguistInput) else None

        transcript = el_input.transcript_content if el_input else ""
        prior = el_input.prior_transcript_content if el_input else None
        company = el_input.company_name if el_input else ""

        messages = build_messages(
            ticker=input_data.ticker,
            transcript=transcript,
            prior_transcript=prior,
            company_name=company,
        )

        request = LlmRequest(
            messages=messages,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )

        analysis: EarningsAnalysis = await llm_client.generate_structured(request, EarningsAnalysis)

        return AnalysisResult(
            ticker=input_data.ticker,
            direction=analysis.overall_sentiment,
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
        - Transcript length exceeds threshold.
        - Combined current + prior transcript exceeds threshold.

        Post-analysis escalation (initial_result provided):
        - Confidence below threshold.

        Args:
            input_data: The analysis input.
            initial_result: Result from first analysis attempt, if available.

        Returns:
            True if escalation is warranted.
        """
        conf_threshold = self._settings.escalation_confidence_threshold

        if initial_result is not None:
            return initial_result.confidence < conf_threshold

        # Pre-analysis: check transcript length
        el_input = input_data if isinstance(input_data, EarningsLinguistInput) else None
        if el_input is None:
            return False

        total_length = len(el_input.transcript_content)
        if el_input.prior_transcript_content:
            total_length += len(el_input.prior_transcript_content)

        threshold = self._settings.escalation_transcript_length
        return total_length > threshold
