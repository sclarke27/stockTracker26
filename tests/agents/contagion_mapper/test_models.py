"""Tests for Cross-Sector Contagion Mapper agent models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.agents.contagion_mapper.models import (
    ContagionAnalysis,
    ContagionInput,
)


def _sample_input(**overrides) -> ContagionInput:
    defaults = {
        "ticker": "AMD",  # target ticker
        "quarter": 3,
        "year": 2024,
        "trigger_ticker": "NVDA",
        "trigger_company_name": "NVIDIA Corporation",
        "trigger_event_summary": "NVDA missed earnings by 15%, guidance cut.",
        "target_company_name": "Advanced Micro Devices",
        "relationship_type": "competitor",
        "trigger_recent_news": [],
        "target_recent_news": [],
        "trigger_sector": "Semiconductors",
        "target_sector": "Semiconductors",
    }
    defaults.update(overrides)
    return ContagionInput(**defaults)


def _sample_analysis(**overrides) -> ContagionAnalysis:
    defaults = {
        "contagion_likely": True,
        "contagion_probability": 0.68,
        "contagion_mechanism": "Competitor miss validates market softness, AMD likely affected.",
        "direction": "BEARISH",
        "confidence": 0.72,
        "affected_business_segments": ["Data Center GPU", "AI accelerators"],
        "timeline_days": 3,
        "mitigating_factors": ["AMD gaining market share from NVDA"],
        "amplifying_factors": ["Same customer base", "Same end markets"],
        "horizon_days": 5,
        "reasoning_summary": "NVDA miss signals sector-wide slowdown affecting AMD.",
    }
    defaults.update(overrides)
    return ContagionAnalysis(**defaults)


class TestContagionInput:
    """Tests for ContagionInput model."""

    def test_valid_construction(self) -> None:
        inp = _sample_input()
        assert inp.ticker == "AMD"
        assert inp.trigger_ticker == "NVDA"
        assert inp.relationship_type == "competitor"

    def test_all_relationship_types_accepted(self) -> None:
        valid_types = [
            "supplier",
            "customer",
            "competitor",
            "same_sector",
            "distribution_partner",
        ]
        for rt in valid_types:
            inp = _sample_input(relationship_type=rt)
            assert inp.relationship_type == rt

    def test_invalid_relationship_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            _sample_input(relationship_type="unknown")  # type: ignore[arg-type]

    def test_news_lists_default_empty(self) -> None:
        inp = _sample_input()
        assert inp.trigger_recent_news == []
        assert inp.target_recent_news == []

    def test_news_lists_accept_dicts(self) -> None:
        news = [{"title": "NVDA misses earnings", "sentiment_score": -0.7}]
        inp = _sample_input(trigger_recent_news=news)
        assert len(inp.trigger_recent_news) == 1

    def test_missing_ticker_raises(self) -> None:
        with pytest.raises(ValidationError):
            ContagionInput(
                quarter=3,
                year=2024,
                trigger_ticker="NVDA",
                trigger_company_name="NVIDIA",
                trigger_event_summary="Missed earnings.",
                target_company_name="AMD",
                relationship_type="competitor",
                trigger_recent_news=[],
                target_recent_news=[],
                trigger_sector="Semiconductors",
                target_sector="Semiconductors",
            )  # type: ignore[call-arg]

    def test_target_ticker_is_base_ticker(self) -> None:
        """The base AgentInput.ticker should be the target for prediction tracking."""
        inp = _sample_input()
        assert inp.ticker == "AMD"  # target
        assert inp.trigger_ticker == "NVDA"  # source of shock


class TestContagionAnalysis:
    """Tests for ContagionAnalysis model (LLM structured output)."""

    def test_valid_construction(self) -> None:
        analysis = _sample_analysis()
        assert analysis.contagion_likely is True
        assert analysis.contagion_probability == 0.68
        assert analysis.direction == "BEARISH"

    def test_direction_literal_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _sample_analysis(direction="SIDEWAYS")  # type: ignore[arg-type]

    def test_all_directions_accepted(self) -> None:
        for direction in ("BULLISH", "BEARISH", "NEUTRAL"):
            analysis = _sample_analysis(direction=direction)
            assert analysis.direction == direction

    def test_probability_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _sample_analysis(contagion_probability=1.5)
        with pytest.raises(ValidationError):
            _sample_analysis(contagion_probability=-0.1)

    def test_confidence_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _sample_analysis(confidence=1.5)
        with pytest.raises(ValidationError):
            _sample_analysis(confidence=-0.1)

    def test_no_contagion_case(self) -> None:
        analysis = _sample_analysis(
            contagion_likely=False,
            contagion_probability=0.12,
            direction="NEUTRAL",
            confidence=0.6,
        )
        assert analysis.contagion_likely is False
        assert analysis.direction == "NEUTRAL"

    def test_empty_mitigating_factors_valid(self) -> None:
        analysis = _sample_analysis(mitigating_factors=[])
        assert analysis.mitigating_factors == []

    def test_empty_affected_segments_valid(self) -> None:
        analysis = _sample_analysis(affected_business_segments=[])
        assert analysis.affected_business_segments == []
