"""Cross-Sector Contagion Mapper agent."""

from __future__ import annotations

from stock_radar.agents.base import BaseAgent
from stock_radar.agents.contagion_mapper.config import (
    AGENT_NAME,
    DEFAULT_HORIZON_DAYS,
    SIGNAL_TYPE,
)
from stock_radar.agents.contagion_mapper.models import (
    ContagionAnalysis,
    ContagionInput,
)
from stock_radar.agents.contagion_mapper.prompt import build_messages
from stock_radar.agents.models import AgentInput, AnalysisResult
from stock_radar.config.settings import ContagionMapperSettings
from stock_radar.llm.base import LlmClient
from stock_radar.llm.models import LlmRequest


class ContagionMapperAgent(BaseAgent):
    """Maps shock propagation from one company to related companies.

    Analyzes the relationship between a trigger company (which had an event)
    and a target company to determine whether contagion will occur and in
    which direction it will affect the target's stock price.
    """

    def __init__(self, settings: ContagionMapperSettings | None = None) -> None:
        self._settings = settings or ContagionMapperSettings()

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
        """Analyze cross-sector contagion potential using the LLM.

        Builds a prompt from trigger/target company context, sends it to
        the LLM, and maps the structured response to an AnalysisResult.

        Args:
            input_data: Must be a ContagionInput.
            llm_client: LLM client to use for inference.

        Returns:
            Analysis result with prediction and reasoning. The result ticker
            is the target company (the one being analyzed for contagion impact).
        """
        cm_input = input_data if isinstance(input_data, ContagionInput) else None

        trigger_ticker = cm_input.trigger_ticker if cm_input else ""
        trigger_company_name = cm_input.trigger_company_name if cm_input else ""
        trigger_event_summary = cm_input.trigger_event_summary if cm_input else ""
        target_company_name = cm_input.target_company_name if cm_input else ""
        relationship_type = cm_input.relationship_type if cm_input else "same_sector"
        trigger_recent_news = cm_input.trigger_recent_news if cm_input else []
        target_recent_news = cm_input.target_recent_news if cm_input else []
        trigger_sector = cm_input.trigger_sector if cm_input else ""
        target_sector = cm_input.target_sector if cm_input else ""

        messages = build_messages(
            trigger_ticker=trigger_ticker,
            trigger_company_name=trigger_company_name,
            target_ticker=input_data.ticker,
            target_company_name=target_company_name,
            relationship_type=relationship_type,
            trigger_event_summary=trigger_event_summary,
            trigger_recent_news=trigger_recent_news,
            target_recent_news=target_recent_news,
            trigger_sector=trigger_sector,
            target_sector=target_sector,
        )

        request = LlmRequest(
            messages=messages,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )

        analysis: ContagionAnalysis = await llm_client.generate_structured(
            request, ContagionAnalysis
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
        - Contagion analysis is well-defined; always use Ollama initially.

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

        # Pre-analysis: no escalation — contagion is a well-scoped task
        return False
