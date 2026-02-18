"""Abstract base class for all analysis agents."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from fastmcp import Client
from loguru import logger

from stock_radar.agents.exceptions import EscalationError
from stock_radar.agents.models import AgentInput, AgentOutput, AnalysisResult
from stock_radar.llm.base import LlmClient


class BaseAgent(ABC):
    """Abstract base for analysis agents.

    Implements the Template Method pattern: the ``run()`` method defines
    the full agent lifecycle (analyze -> escalate -> log -> store), while
    subclasses implement the analysis-specific ``analyze()`` and
    ``should_escalate()`` methods.

    MCP clients are passed into ``run()`` rather than created internally,
    following the same dependency-injection pattern as the pipeline runner.
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique name identifying this agent (e.g. 'earnings_linguist')."""

    @property
    @abstractmethod
    def signal_type(self) -> str:
        """Signal type this agent produces (e.g. 'earnings_sentiment')."""

    @abstractmethod
    async def analyze(
        self,
        input_data: AgentInput,
        llm_client: LlmClient,
    ) -> AnalysisResult:
        """Run the agent's core analysis logic.

        Args:
            input_data: Structured input for this analysis.
            llm_client: LLM client to use (may be Ollama or Anthropic).

        Returns:
            Analysis result with prediction and reasoning.
        """

    @abstractmethod
    def should_escalate(
        self,
        input_data: AgentInput,
        initial_result: AnalysisResult | None = None,
    ) -> bool:
        """Determine whether to escalate to a more powerful LLM.

        Called twice:
        1. Pre-analysis (initial_result=None): based on input characteristics.
        2. Post-analysis (initial_result provided): based on result quality.

        Args:
            input_data: The analysis input.
            initial_result: Result from first analysis attempt, if available.

        Returns:
            True if escalation is warranted.
        """

    async def run(
        self,
        input_data: AgentInput,
        ollama_client: LlmClient,
        anthropic_client: LlmClient | None,
        predictions_client: Client,
        vector_store_client: Client,
    ) -> AgentOutput:
        """Execute the full agent lifecycle.

        Steps:
        1. Pre-analysis escalation check.
        2. Run analysis with the chosen LLM.
        3. Post-analysis escalation check (low confidence retry).
        4. Log prediction to predictions-db-mcp.
        5. Store reasoning in vector-store-mcp.
        6. Fetch similar past reasoning for context.
        7. Return structured output.

        Args:
            input_data: Structured input for this analysis.
            ollama_client: Local LLM client (default).
            anthropic_client: Cloud LLM client (escalation), or None.
            predictions_client: FastMCP client for predictions-db-mcp.
            vector_store_client: FastMCP client for vector-store-mcp.

        Returns:
            Complete agent output with prediction ID and reasoning.

        Raises:
            EscalationError: If escalation is needed but no Anthropic client.
        """
        logger.info(
            "Agent starting",
            agent=self.agent_name,
            ticker=input_data.ticker,
            quarter=input_data.quarter,
            year=input_data.year,
        )

        # Step 1: Pre-analysis escalation check
        if self.should_escalate(input_data):
            if anthropic_client is None:
                raise EscalationError(
                    f"Escalation needed for {input_data.ticker} "
                    f"but no Anthropic client configured"
                )
            logger.info(
                "Pre-analysis escalation triggered",
                agent=self.agent_name,
                ticker=input_data.ticker,
            )
            result = await self.analyze(input_data, anthropic_client)
            result = result.model_copy(update={"escalated": True})
        else:
            # Step 2: Run with local LLM
            result = await self.analyze(input_data, ollama_client)

            # Step 3: Post-analysis escalation check
            if not result.escalated and self.should_escalate(input_data, result):
                if anthropic_client is not None:
                    logger.info(
                        "Post-analysis escalation triggered",
                        agent=self.agent_name,
                        ticker=input_data.ticker,
                        initial_confidence=result.confidence,
                    )
                    result = await self.analyze(input_data, anthropic_client)
                    result = result.model_copy(update={"escalated": True})
                else:
                    logger.warning(
                        "Post-analysis escalation needed but no Anthropic client",
                        agent=self.agent_name,
                        ticker=input_data.ticker,
                    )

        # Step 4: Log prediction
        prediction_id = await self._log_prediction(predictions_client, result)

        # Step 5: Store reasoning
        await self._store_reasoning(vector_store_client, result, prediction_id)

        # Step 6: Fetch similar past reasoning
        similar = await self._fetch_similar_reasoning(
            vector_store_client, result.reasoning, result.ticker
        )

        logger.info(
            "Agent complete",
            agent=self.agent_name,
            ticker=input_data.ticker,
            direction=result.direction,
            confidence=result.confidence,
            escalated=result.escalated,
            prediction_id=prediction_id,
        )

        return AgentOutput(
            prediction_id=prediction_id,
            result=result,
            similar_past_reasoning=similar,
        )

    async def _log_prediction(
        self,
        client: Client,
        result: AnalysisResult,
    ) -> str:
        """Log a prediction to predictions-db-mcp.

        Args:
            client: FastMCP client for predictions-db-mcp.
            result: Analysis result to log.

        Returns:
            The prediction ID from the database.
        """
        tool_result = await client.call_tool(
            "log_prediction",
            {
                "ticker": result.ticker,
                "agent_name": self.agent_name,
                "signal_type": self.signal_type,
                "direction": result.direction,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "horizon_days": result.horizon_days,
            },
        )
        data = json.loads(tool_result.content[0].text)
        return data["prediction_id"]

    async def _store_reasoning(
        self,
        client: Client,
        result: AnalysisResult,
        prediction_id: str,
    ) -> None:
        """Store reasoning in vector-store-mcp for similarity search.

        Args:
            client: FastMCP client for vector-store-mcp.
            result: Analysis result whose reasoning to store.
            prediction_id: ID to use as document ID.
        """
        await client.call_tool(
            "store_embedding",
            {
                "document_type": "reasoning",
                "document_id": prediction_id,
                "content": result.reasoning,
                "ticker": result.ticker,
                "metadata": {
                    "agent_name": self.agent_name,
                    "signal_type": self.signal_type,
                    "direction": result.direction,
                    "confidence": str(result.confidence),
                },
            },
        )

    async def _fetch_similar_reasoning(
        self,
        client: Client,
        query: str,
        ticker: str,
    ) -> list[str]:
        """Search vector store for similar past reasoning.

        Args:
            client: FastMCP client for vector-store-mcp.
            query: Text to search for similarity.
            ticker: Filter results to this ticker.

        Returns:
            List of similar reasoning strings (may be empty).
        """
        try:
            tool_result = await client.call_tool(
                "search_similar",
                {
                    "document_type": "reasoning",
                    "query": query,
                    "n_results": 3,
                    "ticker": ticker,
                },
            )
            data = json.loads(tool_result.content[0].text)
            return [r["content"] for r in data.get("results", [])]
        except Exception:
            logger.debug(
                "Similar reasoning search failed (non-fatal)",
                agent=self.agent_name,
                ticker=ticker,
            )
            return []
