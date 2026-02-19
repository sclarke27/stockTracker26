"""Tests for orchestrator Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.orchestrator.models import (
    AgentRunResult,
    ContagionPair,
    ContagionPairsConfig,
    CycleResult,
    PhaseResult,
)


class TestPhaseResult:
    """Tests for PhaseResult model."""

    def test_valid_phase_result(self) -> None:
        result = PhaseResult(
            phase="ingestion",
            started_at="2025-01-15T10:00:00",
            completed_at="2025-01-15T10:05:00",
            duration_seconds=300.0,
            success=True,
        )
        assert result.phase == "ingestion"
        assert result.success is True
        assert result.error is None

    def test_phase_result_with_error(self) -> None:
        result = PhaseResult(
            phase="analysis",
            started_at="2025-01-15T10:00:00",
            completed_at="2025-01-15T10:00:05",
            duration_seconds=5.0,
            success=False,
            error="Pipeline connection failed",
        )
        assert result.success is False
        assert result.error == "Pipeline connection failed"

    def test_phase_result_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            PhaseResult(
                phase="ingestion",
                started_at="2025-01-15T10:00:00",
                # missing completed_at, duration_seconds, success
            )


class TestAgentRunResult:
    """Tests for AgentRunResult model."""

    def test_valid_agent_run_result(self) -> None:
        result = AgentRunResult(
            agent_name="earnings_linguist",
            tier="deep",
            tickers_run=10,
            predictions_generated=8,
            errors=2,
            duration_seconds=45.5,
        )
        assert result.agent_name == "earnings_linguist"
        assert result.tier == "deep"
        assert result.tickers_run == 10
        assert result.predictions_generated == 8
        assert result.errors == 2


class TestCycleResult:
    """Tests for CycleResult model."""

    def test_valid_cycle_result(self) -> None:
        phase = PhaseResult(
            phase="ingestion",
            started_at="2025-01-15T10:00:00",
            completed_at="2025-01-15T10:05:00",
            duration_seconds=300.0,
            success=True,
        )
        agent_run = AgentRunResult(
            agent_name="earnings_linguist",
            tier="deep",
            tickers_run=10,
            predictions_generated=8,
            errors=2,
            duration_seconds=45.5,
        )
        result = CycleResult(
            cycle_id="abc-123",
            started_at="2025-01-15T10:00:00",
            completed_at="2025-01-15T11:00:00",
            duration_seconds=3600.0,
            quarter=4,
            year=2024,
            phases=[phase],
            agent_runs=[agent_run],
            total_predictions=8,
            total_errors=2,
        )
        assert result.cycle_id == "abc-123"
        assert result.quarter == 4
        assert result.year == 2024
        assert len(result.phases) == 1
        assert len(result.agent_runs) == 1
        assert result.total_predictions == 8
        assert result.total_errors == 2

    def test_cycle_result_empty_runs(self) -> None:
        result = CycleResult(
            cycle_id="empty-cycle",
            started_at="2025-01-15T10:00:00",
            completed_at="2025-01-15T10:00:01",
            duration_seconds=1.0,
            quarter=1,
            year=2025,
            phases=[],
            agent_runs=[],
            total_predictions=0,
            total_errors=0,
        )
        assert result.total_predictions == 0
        assert len(result.phases) == 0


class TestContagionPair:
    """Tests for ContagionPair model."""

    def test_valid_contagion_pair(self) -> None:
        pair = ContagionPair(
            trigger="NVDA",
            target="AMD",
            relationship="competitor",
        )
        assert pair.trigger == "NVDA"
        assert pair.target == "AMD"
        assert pair.relationship == "competitor"

    def test_invalid_relationship_type(self) -> None:
        with pytest.raises(ValidationError):
            ContagionPair(
                trigger="NVDA",
                target="AMD",
                relationship="invalid_type",
            )

    def test_all_valid_relationship_types(self) -> None:
        valid_types = [
            "supplier",
            "customer",
            "competitor",
            "same_sector",
            "distribution_partner",
        ]
        for rel_type in valid_types:
            pair = ContagionPair(
                trigger="AAPL",
                target="MSFT",
                relationship=rel_type,
            )
            assert pair.relationship == rel_type


class TestContagionPairsConfig:
    """Tests for ContagionPairsConfig model."""

    def test_valid_config(self) -> None:
        config = ContagionPairsConfig(
            pairs=[
                ContagionPair(trigger="NVDA", target="AMD", relationship="competitor"),
                ContagionPair(trigger="JPM", target="V", relationship="distribution_partner"),
            ]
        )
        assert len(config.pairs) == 2

    def test_load_from_yaml(self, tmp_path: object) -> None:
        """Test loading contagion pairs from a YAML file."""
        from pathlib import Path

        import yaml

        yaml_content = {
            "pairs": [
                {"trigger": "NVDA", "target": "AMD", "relationship": "competitor"},
                {"trigger": "AAPL", "target": "META", "relationship": "competitor"},
            ]
        }
        yaml_path = Path(str(tmp_path)) / "pairs.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        raw = yaml.safe_load(yaml_path.read_text())
        config = ContagionPairsConfig(**raw)
        assert len(config.pairs) == 2
        assert config.pairs[0].trigger == "NVDA"
        assert config.pairs[1].target == "META"
