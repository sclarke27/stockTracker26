"""Phase implementations for the orchestrator cycle.

Each phase wraps the corresponding subsystem call, measures timing,
and returns a structured PhaseResult.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from loguru import logger

from stock_radar.agents.contagion_mapper.runner import run_batch as cm_run_batch
from stock_radar.agents.earnings_linguist.runner import run_batch as el_run_batch
from stock_radar.agents.narrative_divergence.runner import run_batch as nd_run_batch
from stock_radar.agents.sec_filing_analyzer.runner import run_batch as sf_run_batch
from stock_radar.config.settings import AppSettings
from stock_radar.orchestrator.config import (
    PHASE_ANALYSIS,
    PHASE_INGESTION,
    PHASE_SCORING,
)
from stock_radar.orchestrator.models import AgentRunResult, ContagionPair, PhaseResult
from stock_radar.pipeline.runner import run_pipeline
from stock_radar.scoring.runner import run_scoring_loop


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Phase 1 — Data Ingestion
# ---------------------------------------------------------------------------


async def run_ingestion() -> PhaseResult:
    """Run the data ingestion pipeline and wrap the result.

    Returns:
        PhaseResult capturing timing and success/failure.
    """
    started_at = _now_iso()
    t0 = time.monotonic()
    try:
        await run_pipeline()
        elapsed = time.monotonic() - t0
        logger.info("Ingestion phase completed in {:.1f}s", elapsed)
        return PhaseResult(
            phase=PHASE_INGESTION,
            started_at=started_at,
            completed_at=_now_iso(),
            duration_seconds=elapsed,
            success=True,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("Ingestion phase failed after {:.1f}s: {}", elapsed, exc)
        return PhaseResult(
            phase=PHASE_INGESTION,
            started_at=started_at,
            completed_at=_now_iso(),
            duration_seconds=elapsed,
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Phase 2 — Analysis Agents
# ---------------------------------------------------------------------------


def _is_agent_enabled(agent_name: str, settings: AppSettings) -> bool:
    """Check whether an agent is enabled in settings."""
    agent_settings = getattr(settings.agents, agent_name, None)
    if agent_settings is None:
        return False
    return getattr(agent_settings, "enabled", True)


async def _run_ticker_agent_batch(
    agent_name: str,
    runner_fn: object,
    tickers: list[str],
    tier: str,
    quarter: int,
    year: int,
    settings: AppSettings,
) -> AgentRunResult:
    """Run a ticker-based agent batch and wrap the result.

    Args:
        agent_name: Agent identifier for logging and result tracking.
        runner_fn: The async run_batch callable for this agent.
        tickers: Tickers to process.
        tier: Coverage tier label ('deep' or 'light').
        quarter: Fiscal quarter.
        year: Fiscal year.
        settings: Application settings.

    Returns:
        AgentRunResult capturing counts and timing.
    """
    t0 = time.monotonic()
    try:
        outputs = await runner_fn(
            tickers=tickers,
            quarter=quarter,
            year=year,
            settings=settings,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            "Agent {} ({}) processed {} tickers → {} predictions in {:.1f}s",
            agent_name,
            tier,
            len(tickers),
            len(outputs),
            elapsed,
        )
        return AgentRunResult(
            agent_name=agent_name,
            tier=tier,
            tickers_run=len(tickers),
            predictions_generated=len(outputs),
            errors=0,
            duration_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("Agent {} ({}) failed after {:.1f}s: {}", agent_name, tier, elapsed, exc)
        return AgentRunResult(
            agent_name=agent_name,
            tier=tier,
            tickers_run=len(tickers),
            predictions_generated=0,
            errors=len(tickers),
            duration_seconds=elapsed,
        )


async def _run_contagion_agent(
    pairs: list[ContagionPair],
    quarter: int,
    year: int,
    settings: AppSettings,
) -> AgentRunResult:
    """Run the contagion mapper agent and wrap the result."""
    t0 = time.monotonic()
    pair_tuples = [(p.trigger, p.target, p.relationship) for p in pairs]
    try:
        outputs = await cm_run_batch(
            pairs=pair_tuples,
            quarter=quarter,
            year=year,
            settings=settings,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            "Agent contagion_mapper processed {} pairs → {} predictions in {:.1f}s",
            len(pairs),
            len(outputs),
            elapsed,
        )
        return AgentRunResult(
            agent_name="contagion_mapper",
            tier="deep",
            tickers_run=len(pairs),
            predictions_generated=len(outputs),
            errors=0,
            duration_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("Agent contagion_mapper failed after {:.1f}s: {}", elapsed, exc)
        return AgentRunResult(
            agent_name="contagion_mapper",
            tier="deep",
            tickers_run=len(pairs),
            predictions_generated=0,
            errors=len(pairs),
            duration_seconds=elapsed,
        )


async def run_analysis(
    deep_tickers: list[str],
    light_tickers: list[str],
    contagion_pairs: list[ContagionPair],
    quarter: int,
    year: int,
    settings: AppSettings,
) -> tuple[PhaseResult, list[AgentRunResult]]:
    """Run all analysis agents for both coverage tiers.

    Deep-tier agents run against deep_tickers; light-tier agents run
    against light_tickers.  Contagion mapper receives explicit pairs.
    Each agent failure is isolated — other agents continue.

    Args:
        deep_tickers: Tickers for deep coverage (all 4 agents).
        light_tickers: Tickers for light coverage (narrative + SEC only).
        contagion_pairs: Pairs for the contagion mapper.
        quarter: Fiscal quarter.
        year: Fiscal year.
        settings: Application settings (agent enabled flags are respected).

    Returns:
        Tuple of (PhaseResult, list of AgentRunResult).
    """
    started_at = _now_iso()
    t0 = time.monotonic()
    agent_runs: list[AgentRunResult] = []

    # --- Deep tier: all 4 agents ---
    if deep_tickers:
        if _is_agent_enabled("earnings_linguist", settings):
            result = await _run_ticker_agent_batch(
                agent_name="earnings_linguist",
                runner_fn=el_run_batch,
                tickers=deep_tickers,
                tier="deep",
                quarter=quarter,
                year=year,
                settings=settings,
            )
            agent_runs.append(result)

        if _is_agent_enabled("narrative_divergence", settings):
            result = await _run_ticker_agent_batch(
                agent_name="narrative_divergence",
                runner_fn=nd_run_batch,
                tickers=deep_tickers,
                tier="deep",
                quarter=quarter,
                year=year,
                settings=settings,
            )
            agent_runs.append(result)

        if _is_agent_enabled("sec_filing_analyzer", settings):
            result = await _run_ticker_agent_batch(
                agent_name="sec_filing_analyzer",
                runner_fn=sf_run_batch,
                tickers=deep_tickers,
                tier="deep",
                quarter=quarter,
                year=year,
                settings=settings,
            )
            agent_runs.append(result)

    if _is_agent_enabled("contagion_mapper", settings) and contagion_pairs:
        result = await _run_contagion_agent(
            pairs=contagion_pairs,
            quarter=quarter,
            year=year,
            settings=settings,
        )
        agent_runs.append(result)

    # --- Light tier: narrative_divergence + sec_filing_analyzer only ---
    if light_tickers:
        if _is_agent_enabled("narrative_divergence", settings):
            result = await _run_ticker_agent_batch(
                agent_name="narrative_divergence",
                runner_fn=nd_run_batch,
                tickers=light_tickers,
                tier="light",
                quarter=quarter,
                year=year,
                settings=settings,
            )
            agent_runs.append(result)

        if _is_agent_enabled("sec_filing_analyzer", settings):
            result = await _run_ticker_agent_batch(
                agent_name="sec_filing_analyzer",
                runner_fn=sf_run_batch,
                tickers=light_tickers,
                tier="light",
                quarter=quarter,
                year=year,
                settings=settings,
            )
            agent_runs.append(result)

    elapsed = time.monotonic() - t0
    logger.info("Analysis phase completed in {:.1f}s with {} agent runs", elapsed, len(agent_runs))

    return (
        PhaseResult(
            phase=PHASE_ANALYSIS,
            started_at=started_at,
            completed_at=_now_iso(),
            duration_seconds=elapsed,
            success=True,
        ),
        agent_runs,
    )


# ---------------------------------------------------------------------------
# Phase 3 — Prediction Scoring
# ---------------------------------------------------------------------------


async def run_scoring() -> PhaseResult:
    """Run the prediction scoring loop and wrap the result.

    Returns:
        PhaseResult capturing timing and success/failure.
    """
    started_at = _now_iso()
    t0 = time.monotonic()
    try:
        await run_scoring_loop()
        elapsed = time.monotonic() - t0
        logger.info("Scoring phase completed in {:.1f}s", elapsed)
        return PhaseResult(
            phase=PHASE_SCORING,
            started_at=started_at,
            completed_at=_now_iso(),
            duration_seconds=elapsed,
            success=True,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("Scoring phase failed after {:.1f}s: {}", elapsed, exc)
        return PhaseResult(
            phase=PHASE_SCORING,
            started_at=started_at,
            completed_at=_now_iso(),
            duration_seconds=elapsed,
            success=False,
            error=str(exc),
        )
