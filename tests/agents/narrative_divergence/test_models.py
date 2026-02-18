"""Tests for Narrative vs Price Divergence agent models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.agents.narrative_divergence.models import (
    NarrativeAnalysis,
    NarrativeDivergenceInput,
)


def _sample_input(**overrides) -> NarrativeDivergenceInput:
    """Create a minimal valid NarrativeDivergenceInput."""
    defaults = {
        "ticker": "AAPL",
        "quarter": 4,
        "year": 2024,
        "sentiment_score": 0.45,
        "article_count": 12,
        "average_sentiment_label": "Somewhat-Bullish",
        "price_return_30d": -0.08,
        "price_return_7d": -0.03,
        "top_articles": [],
    }
    defaults.update(overrides)
    return NarrativeDivergenceInput(**defaults)


def _sample_analysis(**overrides) -> NarrativeAnalysis:
    """Create a minimal valid NarrativeAnalysis."""
    defaults = {
        "divergence_detected": True,
        "divergence_strength": 0.72,
        "direction": "BULLISH",
        "confidence": 0.78,
        "narrative_summary": "News is broadly bullish on new product pipeline.",
        "price_action_summary": "Stock has fallen 8% over 30 days.",
        "divergence_explanation": "Market appears to be discounting positive news.",
        "key_catalysts": ["Product launch", "Analyst upgrade"],
        "horizon_days": 10,
        "reasoning_summary": "Clear divergence between sentiment and price.",
    }
    defaults.update(overrides)
    return NarrativeAnalysis(**defaults)


class TestNarrativeDivergenceInput:
    """Tests for NarrativeDivergenceInput model."""

    def test_valid_construction(self) -> None:
        inp = _sample_input()
        assert inp.ticker == "AAPL"
        assert inp.sentiment_score == 0.45
        assert inp.article_count == 12
        assert inp.price_return_30d == -0.08

    def test_optional_time_window_defaults_none(self) -> None:
        inp = _sample_input()
        assert inp.time_from is None
        assert inp.time_to is None

    def test_optional_time_window_can_be_set(self) -> None:
        inp = _sample_input(time_from="20240101T000000", time_to="20240201T000000")
        assert inp.time_from == "20240101T000000"
        assert inp.time_to == "20240201T000000"

    def test_top_articles_defaults_empty(self) -> None:
        inp = _sample_input()
        assert inp.top_articles == []

    def test_top_articles_accepts_list_of_dicts(self) -> None:
        articles = [{"title": "AAPL hits new high", "sentiment_score": 0.6}]
        inp = _sample_input(top_articles=articles)
        assert len(inp.top_articles) == 1
        assert inp.top_articles[0]["title"] == "AAPL hits new high"

    def test_missing_required_ticker_raises(self) -> None:
        with pytest.raises(ValidationError):
            NarrativeDivergenceInput(
                quarter=4,
                year=2024,
                sentiment_score=0.4,
                article_count=5,
                average_sentiment_label="Neutral",
                price_return_30d=0.0,
                price_return_7d=0.0,
                top_articles=[],
            )  # type: ignore[call-arg]

    def test_negative_price_return_valid(self) -> None:
        inp = _sample_input(price_return_30d=-0.25, price_return_7d=-0.10)
        assert inp.price_return_30d == -0.25

    def test_positive_price_return_valid(self) -> None:
        inp = _sample_input(price_return_30d=0.15, price_return_7d=0.05)
        assert inp.price_return_30d == 0.15


class TestNarrativeAnalysis:
    """Tests for NarrativeAnalysis model (LLM structured output)."""

    def test_valid_construction(self) -> None:
        analysis = _sample_analysis()
        assert analysis.divergence_detected is True
        assert analysis.divergence_strength == 0.72
        assert analysis.direction == "BULLISH"
        assert analysis.confidence == 0.78

    def test_direction_must_be_valid_literal(self) -> None:
        with pytest.raises(ValidationError):
            _sample_analysis(direction="SIDEWAYS")  # type: ignore[arg-type]

    def test_all_direction_values_accepted(self) -> None:
        for direction in ("BULLISH", "BEARISH", "NEUTRAL"):
            analysis = _sample_analysis(direction=direction)
            assert analysis.direction == direction

    def test_confidence_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _sample_analysis(confidence=1.5)
        with pytest.raises(ValidationError):
            _sample_analysis(confidence=-0.1)

    def test_divergence_strength_valid_range(self) -> None:
        analysis = _sample_analysis(divergence_strength=0.0)
        assert analysis.divergence_strength == 0.0
        analysis = _sample_analysis(divergence_strength=1.0)
        assert analysis.divergence_strength == 1.0

    def test_key_catalysts_can_be_empty(self) -> None:
        analysis = _sample_analysis(key_catalysts=[])
        assert analysis.key_catalysts == []

    def test_no_divergence_detected(self) -> None:
        analysis = _sample_analysis(
            divergence_detected=False,
            divergence_strength=0.05,
            direction="NEUTRAL",
        )
        assert analysis.divergence_detected is False
        assert analysis.direction == "NEUTRAL"
