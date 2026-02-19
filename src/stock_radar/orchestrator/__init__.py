"""Orchestrator — cron-triggered analysis cycle.

Entry point for ``stock-radar-orchestrate`` CLI command.
Runs one full cycle (ingestion → analysis → scoring) and exits.
"""

from __future__ import annotations

import asyncio
import sys

from loguru import logger

from stock_radar.orchestrator.cycle import run_cycle
from stock_radar.utils.logging import setup_logging


def main() -> None:
    """CLI entry point for the orchestrator.

    Runs a single orchestrator cycle synchronously and exits with
    code 0 on success, 1 if any errors occurred.
    """
    setup_logging()
    logger.info("stock-radar-orchestrate starting")

    result = asyncio.run(run_cycle())

    if result.total_errors > 0:
        logger.warning(
            "Cycle completed with {} errors across {} predictions",
            result.total_errors,
            result.total_predictions,
        )
        sys.exit(1)

    logger.info(
        "Cycle completed successfully: {} predictions in {:.1f}s",
        result.total_predictions,
        result.duration_seconds,
    )
    sys.exit(0)
