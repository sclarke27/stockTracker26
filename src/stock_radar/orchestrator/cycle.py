"""Top-level orchestrator cycle — ties ingestion, analysis, and scoring together.

A single invocation of ``run_cycle()`` performs one complete pass:
    1. Data ingestion (pipeline)
    2. Analysis agents (deep + light tiers)
    3. Prediction scoring

Designed to be triggered by cron — runs once and exits.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from loguru import logger

from stock_radar.config.loader import load_config
from stock_radar.config.settings import AppSettings
from stock_radar.orchestrator.models import ContagionPairsConfig, CycleResult
from stock_radar.orchestrator.phases import run_analysis, run_ingestion, run_scoring
from stock_radar.pipeline.quarter import current_quarter
from stock_radar.pipeline.watchlist import load_watchlist


def load_settings() -> AppSettings:
    """Load application settings from config.

    Returns:
        Populated AppSettings instance.
    """
    config = load_config()
    return AppSettings(**config)


def _load_contagion_pairs(path: str) -> ContagionPairsConfig:
    """Load contagion pair definitions from a YAML file.

    Args:
        path: Filesystem path to contagion_pairs.yaml.

    Returns:
        Validated ContagionPairsConfig.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    raw = yaml.safe_load(Path(path).read_text())
    return ContagionPairsConfig(**raw)


async def run_cycle() -> CycleResult:
    """Execute one full orchestrator cycle.

    Phases:
        1. **Ingestion** — refresh market data via the pipeline.
        2. **Analysis** — run agents against deep and light ticker tiers.
        3. **Scoring** — score mature predictions against actual prices.

    Each phase is independent: a failure in one does not prevent the next
    from running (e.g. stale data can still be scored).

    Returns:
        CycleResult with timing, phase outcomes, and agent run details.
    """
    cycle_id = str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat()
    t0 = time.monotonic()

    logger.info("Starting orchestrator cycle {}", cycle_id)

    # --- Load configuration ---
    settings = load_settings()
    watchlist = load_watchlist()
    quarter, year = current_quarter()

    contagion_pairs_config = _load_contagion_pairs(settings.orchestrator.contagion_pairs_path)

    deep_tickers = [t.symbol for t in watchlist.deep]
    light_tickers = [t.symbol for t in watchlist.light]

    logger.info(
        "Cycle config: quarter=Q{}/{}, deep={}, light={}, contagion_pairs={}",
        quarter,
        year,
        len(deep_tickers),
        len(light_tickers),
        len(contagion_pairs_config.pairs),
    )

    phases = []
    agent_runs = []

    # --- Phase 1: Ingestion ---
    if not settings.orchestrator.skip_ingestion:
        ingestion_result = await run_ingestion()
        phases.append(ingestion_result)
        if not ingestion_result.success:
            logger.warning("Ingestion failed — continuing with stale data")
    else:
        logger.info("Skipping ingestion phase (skip_ingestion=True)")

    # --- Phase 2: Analysis ---
    analysis_result, analysis_agent_runs = await run_analysis(
        deep_tickers=deep_tickers,
        light_tickers=light_tickers,
        contagion_pairs=contagion_pairs_config.pairs,
        quarter=quarter,
        year=year,
        settings=settings,
    )
    phases.append(analysis_result)
    agent_runs.extend(analysis_agent_runs)

    # --- Phase 3: Scoring ---
    if not settings.orchestrator.skip_scoring:
        scoring_result = await run_scoring()
        phases.append(scoring_result)
    else:
        logger.info("Skipping scoring phase (skip_scoring=True)")

    # --- Aggregate ---
    elapsed = time.monotonic() - t0
    completed_at = datetime.now(UTC).isoformat()
    total_predictions = sum(r.predictions_generated for r in agent_runs)
    total_errors = sum(r.errors for r in agent_runs)

    result = CycleResult(
        cycle_id=cycle_id,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=elapsed,
        quarter=quarter,
        year=year,
        phases=phases,
        agent_runs=agent_runs,
        total_predictions=total_predictions,
        total_errors=total_errors,
    )

    logger.info(
        "Cycle {} completed in {:.1f}s — {} predictions, {} errors",
        cycle_id,
        elapsed,
        total_predictions,
        total_errors,
    )

    return result
