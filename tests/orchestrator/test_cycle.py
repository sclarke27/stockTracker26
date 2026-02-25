"""Tests for the top-level orchestrator cycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stock_radar.config.settings import AppSettings, OrchestratorSettings
from stock_radar.orchestrator.config import PHASE_ANALYSIS, PHASE_INGESTION, PHASE_SCORING
from stock_radar.orchestrator.cycle import run_cycle
from stock_radar.orchestrator.models import AgentRunResult, ContagionPairsConfig, PhaseResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_PREFIX = "stock_radar.orchestrator.cycle"


def _make_settings(**overrides: object) -> AppSettings:
    """Build an AppSettings with sensible defaults for testing."""
    defaults = {
        "api_keys": {"alpha_vantage": "test"},
        "sec_edgar": {"user_agent_email": "test@test.com"},
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


def _ok_phase(phase: str) -> PhaseResult:
    return PhaseResult(
        phase=phase,
        started_at="2025-01-15T10:00:00",
        completed_at="2025-01-15T10:05:00",
        duration_seconds=300.0,
        success=True,
    )


def _failed_phase(phase: str) -> PhaseResult:
    return PhaseResult(
        phase=phase,
        started_at="2025-01-15T10:00:00",
        completed_at="2025-01-15T10:00:05",
        duration_seconds=5.0,
        success=False,
        error="Something went wrong",
    )


def _agent_run(name: str = "earnings_linguist", predictions: int = 5) -> AgentRunResult:
    return AgentRunResult(
        agent_name=name,
        tier="deep",
        tickers_run=10,
        predictions_generated=predictions,
        errors=0,
        duration_seconds=10.0,
    )


def _mock_watchlist() -> MagicMock:
    deep_ticker = MagicMock()
    deep_ticker.symbol = "AAPL"
    light_ticker = MagicMock()
    light_ticker.symbol = "WMT"
    wl = MagicMock()
    wl.deep = [deep_ticker]
    wl.light = [light_ticker]
    return wl


def _base_patches():
    """Return a dict of common patches for run_cycle tests."""
    return {
        "load_settings": patch(
            f"{_PATCH_PREFIX}.load_settings",
            return_value=_make_settings(),
        ),
        "load_watchlist": patch(
            f"{_PATCH_PREFIX}.load_watchlist",
            return_value=_mock_watchlist(),
        ),
        "load_contagion_pairs": patch(
            f"{_PATCH_PREFIX}._load_contagion_pairs",
            return_value=ContagionPairsConfig(pairs=[]),
        ),
        "current_quarter": patch(
            f"{_PATCH_PREFIX}.current_quarter",
            return_value=(4, 2024),
        ),
        "run_ingestion": patch(
            f"{_PATCH_PREFIX}.run_ingestion",
            new_callable=AsyncMock,
            return_value=_ok_phase(PHASE_INGESTION),
        ),
        "run_analysis": patch(
            f"{_PATCH_PREFIX}.run_analysis",
            new_callable=AsyncMock,
            return_value=(_ok_phase(PHASE_ANALYSIS), [_agent_run()]),
        ),
        "run_scoring": patch(
            f"{_PATCH_PREFIX}.run_scoring",
            new_callable=AsyncMock,
            return_value=_ok_phase(PHASE_SCORING),
        ),
    }


class TestRunCycle:
    """Tests for the top-level run_cycle function."""

    @pytest.mark.asyncio
    async def test_executes_all_three_phases_in_order(self) -> None:
        patches = _base_patches()
        call_order: list[str] = []

        async def track_ingestion() -> PhaseResult:
            call_order.append("ingestion")
            return _ok_phase(PHASE_INGESTION)

        async def track_analysis(**kwargs: object) -> tuple[PhaseResult, list[AgentRunResult]]:
            call_order.append("analysis")
            return (_ok_phase(PHASE_ANALYSIS), [_agent_run()])

        async def track_scoring() -> PhaseResult:
            call_order.append("scoring")
            return _ok_phase(PHASE_SCORING)

        patches["run_ingestion"] = patch(
            f"{_PATCH_PREFIX}.run_ingestion",
            new_callable=AsyncMock,
            side_effect=track_ingestion,
        )
        patches["run_analysis"] = patch(
            f"{_PATCH_PREFIX}.run_analysis",
            new_callable=AsyncMock,
            side_effect=track_analysis,
        )
        patches["run_scoring"] = patch(
            f"{_PATCH_PREFIX}.run_scoring",
            new_callable=AsyncMock,
            side_effect=track_scoring,
        )

        with (
            patches["load_settings"],
            patches["load_watchlist"],
            patches["load_contagion_pairs"],
            patches["current_quarter"],
            patches["run_ingestion"],
            patches["run_analysis"],
            patches["run_scoring"],
        ):
            result = await run_cycle()

        assert call_order == ["ingestion", "analysis", "scoring"]
        assert len(result.phases) == 3
        assert result.quarter == 4
        assert result.year == 2024

    @pytest.mark.asyncio
    async def test_skips_ingestion_when_configured(self) -> None:
        settings = _make_settings(
            orchestrator=OrchestratorSettings(skip_ingestion=True),
        )
        patches = _base_patches()
        patches["load_settings"] = patch(
            f"{_PATCH_PREFIX}.load_settings",
            return_value=settings,
        )

        with (
            patches["load_settings"],
            patches["load_watchlist"],
            patches["load_contagion_pairs"],
            patches["current_quarter"],
            patches["run_ingestion"] as mock_ingest,
            patches["run_analysis"],
            patches["run_scoring"],
        ):
            result = await run_cycle()

        mock_ingest.assert_not_called()
        # Should have analysis + scoring but not ingestion
        phase_names = [p.phase for p in result.phases]
        assert PHASE_INGESTION not in phase_names
        assert PHASE_ANALYSIS in phase_names
        assert PHASE_SCORING in phase_names

    @pytest.mark.asyncio
    async def test_skips_scoring_when_configured(self) -> None:
        settings = _make_settings(
            orchestrator=OrchestratorSettings(skip_scoring=True),
        )
        patches = _base_patches()
        patches["load_settings"] = patch(
            f"{_PATCH_PREFIX}.load_settings",
            return_value=settings,
        )

        with (
            patches["load_settings"],
            patches["load_watchlist"],
            patches["load_contagion_pairs"],
            patches["current_quarter"],
            patches["run_ingestion"],
            patches["run_analysis"],
            patches["run_scoring"] as mock_score,
        ):
            result = await run_cycle()

        mock_score.assert_not_called()
        phase_names = [p.phase for p in result.phases]
        assert PHASE_SCORING not in phase_names
        assert PHASE_INGESTION in phase_names

    @pytest.mark.asyncio
    async def test_continues_to_scoring_even_if_analysis_fails(self) -> None:
        patches = _base_patches()
        patches["run_analysis"] = patch(
            f"{_PATCH_PREFIX}.run_analysis",
            new_callable=AsyncMock,
            return_value=(_failed_phase(PHASE_ANALYSIS), []),
        )

        with (
            patches["load_settings"],
            patches["load_watchlist"],
            patches["load_contagion_pairs"],
            patches["current_quarter"],
            patches["run_ingestion"],
            patches["run_analysis"],
            patches["run_scoring"] as mock_score,
        ):
            result = await run_cycle()

        mock_score.assert_called_once()
        assert len(result.phases) == 3

    @pytest.mark.asyncio
    async def test_returns_valid_cycle_result_with_timing(self) -> None:
        patches = _base_patches()

        with (
            patches["load_settings"],
            patches["load_watchlist"],
            patches["load_contagion_pairs"],
            patches["current_quarter"],
            patches["run_ingestion"],
            patches["run_analysis"],
            patches["run_scoring"],
        ):
            result = await run_cycle()

        assert result.cycle_id  # non-empty UUID string
        assert result.started_at
        assert result.completed_at
        assert result.duration_seconds >= 0
        assert result.total_predictions == 5  # from _agent_run default
        assert result.total_errors == 0
