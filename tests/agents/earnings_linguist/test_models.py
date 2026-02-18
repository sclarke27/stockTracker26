"""Tests for Earnings Linguist models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_radar.agents.earnings_linguist.models import (
    EarningsAnalysis,
    EarningsLinguistInput,
    SentimentIndicator,
)


class TestSentimentIndicator:
    def test_construction(self) -> None:
        ind = SentimentIndicator(
            category="hedging",
            quote="We believe growth may slow.",
            interpretation="Management hedging on growth outlook.",
            impact="BEARISH",
        )
        assert ind.category == "hedging"
        assert ind.impact == "BEARISH"

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SentimentIndicator(
                category="invalid_cat",
                quote="q",
                interpretation="i",
                impact="BULLISH",
            )

    def test_invalid_impact_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SentimentIndicator(
                category="hedging",
                quote="q",
                interpretation="i",
                impact="UP",
            )


class TestEarningsAnalysis:
    def test_construction(self) -> None:
        analysis = EarningsAnalysis(
            overall_sentiment="BULLISH",
            confidence=0.85,
            sentiment_indicators=[
                SentimentIndicator(
                    category="forward_guidance",
                    quote="We expect strong Q1.",
                    interpretation="Positive forward guidance.",
                    impact="BULLISH",
                )
            ],
            key_risks=["Supply chain delays"],
            key_opportunities=["New product launch"],
            reasoning_summary="Management tone is optimistic.",
            horizon_days=5,
        )
        assert analysis.overall_sentiment == "BULLISH"
        assert len(analysis.sentiment_indicators) == 1
        assert analysis.quarter_over_quarter_shift is None

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            EarningsAnalysis(
                overall_sentiment="BULLISH",
                confidence=1.5,
                sentiment_indicators=[],
                key_risks=[],
                key_opportunities=[],
                reasoning_summary="r",
            )

    def test_with_qoq_shift(self) -> None:
        analysis = EarningsAnalysis(
            overall_sentiment="BEARISH",
            confidence=0.6,
            sentiment_indicators=[],
            quarter_over_quarter_shift="Tone shifted from optimistic to cautious.",
            key_risks=["Margin pressure"],
            key_opportunities=[],
            reasoning_summary="Cautious tone.",
        )
        assert analysis.quarter_over_quarter_shift is not None

    def test_default_horizon_days(self) -> None:
        analysis = EarningsAnalysis(
            overall_sentiment="NEUTRAL",
            confidence=0.5,
            sentiment_indicators=[],
            key_risks=[],
            key_opportunities=[],
            reasoning_summary="Mixed signals.",
        )
        assert analysis.horizon_days == 5


class TestEarningsLinguistInput:
    def test_construction(self) -> None:
        inp = EarningsLinguistInput(
            ticker="AAPL",
            quarter=4,
            year=2024,
            transcript_content="Tim Cook: Great quarter...",
            company_name="Apple Inc.",
        )
        assert inp.ticker == "AAPL"
        assert inp.transcript_content.startswith("Tim Cook")
        assert inp.prior_transcript_content is None

    def test_with_prior_transcript(self) -> None:
        inp = EarningsLinguistInput(
            ticker="AAPL",
            quarter=4,
            year=2024,
            transcript_content="Current quarter...",
            prior_transcript_content="Prior quarter...",
        )
        assert inp.prior_transcript_content == "Prior quarter..."
