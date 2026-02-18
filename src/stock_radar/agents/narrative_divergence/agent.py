"""Narrative vs Price Divergence agent."""

from __future__ import annotations

from stock_radar.agents.base import BaseAgent
from stock_radar.agents.models import AgentInput, AnalysisResult
from stock_radar.agents.narrative_divergence.config import (
    AGENT_NAME,
    DEFAULT_HORIZON_DAYS,
    ESCALATION_MIN_ARTICLES,
    SIGNAL_TYPE,
)
from stock_radar.agents.narrative_divergence.models import (
    NarrativeAnalysis,
    NarrativeDivergenceInput,
)
from stock_radar.agents.narrative_divergence.prompt import build_messages
from stock_radar.config.settings import NarrativeDivergenceSettings
from stock_radar.llm.base import LlmClient
from stock_radar.llm.models import LlmRequest


class NarrativeDivergenceAgent(BaseAgent):
    """Detects divergences between news narrative and stock price action.

    When news sentiment strongly diverges from recent price movement, the
    market may be mis-pricing the stock. This agent quantifies the divergence
    and predicts whether the price will converge to the narrative or vice versa.
    """

    def __init__(self, settings: NarrativeDivergenceSettings | None = None) -> None:
        self._settings = settings or NarrativeDivergenceSettings()

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
        """Analyze narrative vs price divergence using the LLM.

        Builds a prompt from sentiment and price data, sends it to the LLM,
        and maps the structured response to an AnalysisResult.

        Args:
            input_data: Must be a NarrativeDivergenceInput.
            llm_client: LLM client to use for inference.

        Returns:
            Analysis result with prediction and reasoning.
        """
        nd_input = input_data if isinstance(input_data, NarrativeDivergenceInput) else None

        # Extract fields with safe defaults if wrong input type
        sentiment_score = nd_input.sentiment_score if nd_input else 0.0
        article_count = nd_input.article_count if nd_input else 0
        average_sentiment_label = nd_input.average_sentiment_label if nd_input else "Neutral"
        price_return_30d = nd_input.price_return_30d if nd_input else 0.0
        price_return_7d = nd_input.price_return_7d if nd_input else 0.0
        top_articles = nd_input.top_articles if nd_input else []
        time_from = nd_input.time_from if nd_input else None
        time_to = nd_input.time_to if nd_input else None

        messages = build_messages(
            ticker=input_data.ticker,
            sentiment_score=sentiment_score,
            article_count=article_count,
            average_sentiment_label=average_sentiment_label,
            price_return_30d=price_return_30d,
            price_return_7d=price_return_7d,
            top_articles=top_articles,
            time_from=time_from,
            time_to=time_to,
        )

        request = LlmRequest(
            messages=messages,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )

        analysis: NarrativeAnalysis = await llm_client.generate_structured(
            request, NarrativeAnalysis
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
        - Too few articles to form a reliable signal.

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

        # Pre-analysis: check article count
        nd_input = input_data if isinstance(input_data, NarrativeDivergenceInput) else None
        if nd_input is None:
            return False

        min_articles = self._settings.escalation_min_articles or ESCALATION_MIN_ARTICLES
        return nd_input.article_count < min_articles
