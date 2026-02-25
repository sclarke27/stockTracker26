"""Tests for orchestrator phase implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stock_radar.config.settings import (
    AgentsSettings,
    AppSettings,
    ContagionMapperSettings,
    EarningsLinguistSettings,
    NarrativeDivergenceSettings,
    SecFilingAnalyzerSettings,
)
from stock_radar.orchestrator.config import PHASE_ANALYSIS, PHASE_INGESTION, PHASE_SCORING
from stock_radar.orchestrator.models import ContagionPair
from stock_radar.orchestrator.phases import run_analysis, run_ingestion, run_scoring

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: object) -> AppSettings:
    """Build an AppSettings with sensible defaults for testing."""
    defaults = {
        "api_keys": {"alpha_vantage": "test"},
        "sec_edgar": {"user_agent_email": "test@test.com"},
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


def _make_agent_output(prediction_id: str = "pred-1") -> MagicMock:
    """Create a mock AgentOutput."""
    output = MagicMock()
    output.prediction_id = prediction_id
    return output


# ---------------------------------------------------------------------------
# run_ingestion
# ---------------------------------------------------------------------------


class TestRunIngestion:
    """Tests for the ingestion phase."""

    @pytest.mark.asyncio
    async def test_calls_pipeline_and_wraps_result(self) -> None:
        mock_pipeline_result = MagicMock()
        with patch(
            "stock_radar.orchestrator.phases.run_pipeline",
            new_callable=AsyncMock,
            return_value=mock_pipeline_result,
        ) as mock_run:
            result = await run_ingestion()
            mock_run.assert_called_once()

        assert result.phase == PHASE_INGESTION
        assert result.success is True
        assert result.error is None
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_handles_pipeline_failure_gracefully(self) -> None:
        with patch(
            "stock_radar.orchestrator.phases.run_pipeline",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Connection refused"),
        ):
            result = await run_ingestion()

        assert result.phase == PHASE_INGESTION
        assert result.success is False
        assert "Connection refused" in (result.error or "")


# ---------------------------------------------------------------------------
# run_analysis
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    """Tests for the analysis phase."""

    @pytest.mark.asyncio
    async def test_runs_deep_agents_for_deep_tickers(self) -> None:
        settings = _make_settings()
        deep_tickers = ["AAPL", "NVDA"]
        light_tickers = ["WMT"]
        pairs: list[ContagionPair] = []

        with (
            patch(
                "stock_radar.orchestrator.phases.el_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ) as mock_el,
            patch(
                "stock_radar.orchestrator.phases.nd_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ),
            patch(
                "stock_radar.orchestrator.phases.sf_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ),
            patch(
                "stock_radar.orchestrator.phases.cm_run_batch",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            phase_result, agent_runs = await run_analysis(
                deep_tickers=deep_tickers,
                light_tickers=light_tickers,
                contagion_pairs=pairs,
                quarter=4,
                year=2024,
                settings=settings,
            )

        # Earnings linguist should only be called with deep tickers
        mock_el.assert_called_once()
        call_args = mock_el.call_args
        assert call_args[1]["tickers"] == deep_tickers

        assert phase_result.phase == PHASE_ANALYSIS
        assert phase_result.success is True

    @pytest.mark.asyncio
    async def test_runs_light_agents_for_light_tickers(self) -> None:
        settings = _make_settings()
        deep_tickers = ["AAPL"]
        light_tickers = ["WMT", "COST"]
        pairs: list[ContagionPair] = []

        with (
            patch(
                "stock_radar.orchestrator.phases.el_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ),
            patch(
                "stock_radar.orchestrator.phases.nd_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ) as mock_nd,
            patch(
                "stock_radar.orchestrator.phases.sf_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ) as mock_sf,
            patch(
                "stock_radar.orchestrator.phases.cm_run_batch",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            _, agent_runs = await run_analysis(
                deep_tickers=deep_tickers,
                light_tickers=light_tickers,
                contagion_pairs=pairs,
                quarter=4,
                year=2024,
                settings=settings,
            )

        # narrative_divergence called twice (deep + light)
        assert mock_nd.call_count == 2
        # sec_filing_analyzer called twice (deep + light)
        assert mock_sf.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_disabled_agents(self) -> None:
        agents = AgentsSettings(
            earnings_linguist=EarningsLinguistSettings(enabled=False),
            narrative_divergence=NarrativeDivergenceSettings(enabled=True),
            sec_filing_analyzer=SecFilingAnalyzerSettings(enabled=True),
            contagion_mapper=ContagionMapperSettings(enabled=False),
        )
        settings = _make_settings(agents=agents)

        with (
            patch(
                "stock_radar.orchestrator.phases.el_run_batch",
                new_callable=AsyncMock,
            ) as mock_el,
            patch(
                "stock_radar.orchestrator.phases.nd_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ),
            patch(
                "stock_radar.orchestrator.phases.sf_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ),
            patch(
                "stock_radar.orchestrator.phases.cm_run_batch",
                new_callable=AsyncMock,
            ) as mock_cm,
        ):
            _, agent_runs = await run_analysis(
                deep_tickers=["AAPL"],
                light_tickers=[],
                contagion_pairs=[],
                quarter=4,
                year=2024,
                settings=settings,
            )

        mock_el.assert_not_called()
        mock_cm.assert_not_called()

    @pytest.mark.asyncio
    async def test_continues_after_single_agent_failure(self) -> None:
        settings = _make_settings()

        with (
            patch(
                "stock_radar.orchestrator.phases.el_run_batch",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Ollama down"),
            ),
            patch(
                "stock_radar.orchestrator.phases.nd_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ) as mock_nd,
            patch(
                "stock_radar.orchestrator.phases.sf_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ),
            patch(
                "stock_radar.orchestrator.phases.cm_run_batch",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            phase_result, agent_runs = await run_analysis(
                deep_tickers=["AAPL"],
                light_tickers=[],
                contagion_pairs=[],
                quarter=4,
                year=2024,
                settings=settings,
            )

        # Phase succeeds overall even if one agent fails
        assert phase_result.success is True
        # Other agents still ran
        mock_nd.assert_called()
        # The failed agent run should have errors > 0
        el_run = next((r for r in agent_runs if r.agent_name == "earnings_linguist"), None)
        assert el_run is not None
        assert el_run.errors > 0

    @pytest.mark.asyncio
    async def test_loads_and_passes_contagion_pairs(self) -> None:
        settings = _make_settings()
        pairs = [
            ContagionPair(trigger="NVDA", target="AMD", relationship="competitor"),
            ContagionPair(trigger="AAPL", target="META", relationship="competitor"),
        ]

        with (
            patch(
                "stock_radar.orchestrator.phases.el_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ),
            patch(
                "stock_radar.orchestrator.phases.nd_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ),
            patch(
                "stock_radar.orchestrator.phases.sf_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output()],
            ),
            patch(
                "stock_radar.orchestrator.phases.cm_run_batch",
                new_callable=AsyncMock,
                return_value=[_make_agent_output(), _make_agent_output()],
            ) as mock_cm,
        ):
            _, agent_runs = await run_analysis(
                deep_tickers=["NVDA", "AAPL"],
                light_tickers=[],
                contagion_pairs=pairs,
                quarter=4,
                year=2024,
                settings=settings,
            )

        mock_cm.assert_called_once()
        call_kwargs = mock_cm.call_args[1]
        # Should pass tuples of (trigger, target, relationship)
        passed_pairs = call_kwargs["pairs"]
        assert len(passed_pairs) == 2
        assert passed_pairs[0] == ("NVDA", "AMD", "competitor")


# ---------------------------------------------------------------------------
# run_scoring
# ---------------------------------------------------------------------------


class TestRunScoring:
    """Tests for the scoring phase."""

    @pytest.mark.asyncio
    async def test_calls_scoring_loop_and_wraps_result(self) -> None:
        mock_scoring_result = MagicMock()
        with patch(
            "stock_radar.orchestrator.phases.run_scoring_loop",
            new_callable=AsyncMock,
            return_value=mock_scoring_result,
        ) as mock_run:
            result = await run_scoring()
            mock_run.assert_called_once()

        assert result.phase == PHASE_SCORING
        assert result.success is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_handles_scoring_failure_gracefully(self) -> None:
        with patch(
            "stock_radar.orchestrator.phases.run_scoring_loop",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Database locked"),
        ):
            result = await run_scoring()

        assert result.phase == PHASE_SCORING
        assert result.success is False
        assert "Database locked" in (result.error or "")
