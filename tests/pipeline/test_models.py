"""Tests for data ingestion pipeline Pydantic models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from stock_radar.pipeline.models import (
    PipelineResult,
    TickerResult,
    ToolCallResult,
)


class TestToolCallResult:
    """Tests for the ToolCallResult model."""

    def test_successful_result(self) -> None:
        result = ToolCallResult(
            tool_name="get_price_history",
            ticker="AAPL",
            success=True,
            duration_ms=123.45,
        )
        assert result.tool_name == "get_price_history"
        assert result.ticker == "AAPL"
        assert result.success is True
        assert result.error is None
        assert result.duration_ms == 123.45

    def test_failed_result_with_error(self) -> None:
        result = ToolCallResult(
            tool_name="get_quote",
            ticker="MSFT",
            success=False,
            error="API rate limit exceeded",
            duration_ms=5000.0,
        )
        assert result.success is False
        assert result.error == "API rate limit exceeded"

    def test_duration_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError, match="duration_ms"):
            ToolCallResult(
                tool_name="get_quote",
                ticker="AAPL",
                success=True,
                duration_ms=-1.0,
            )

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ToolCallResult(
                tool_name="get_quote",
                # missing ticker, success, duration_ms
            )


class TestTickerResult:
    """Tests for the TickerResult model."""

    def test_construction_with_mixed_results(self) -> None:
        results = [
            ToolCallResult(
                tool_name="get_price_history",
                ticker="AAPL",
                success=True,
                duration_ms=200.0,
            ),
            ToolCallResult(
                tool_name="get_quote",
                ticker="AAPL",
                success=False,
                error="Timeout",
                duration_ms=5000.0,
            ),
            ToolCallResult(
                tool_name="get_company_info",
                ticker="AAPL",
                success=True,
                duration_ms=150.0,
            ),
        ]
        ticker_result = TickerResult(
            ticker="AAPL",
            tier="deep",
            results=results,
            success_count=2,
            error_count=1,
        )
        assert ticker_result.ticker == "AAPL"
        assert ticker_result.tier == "deep"
        assert len(ticker_result.results) == 3
        assert ticker_result.success_count == 2
        assert ticker_result.error_count == 1

    def test_empty_results_list(self) -> None:
        ticker_result = TickerResult(
            ticker="TSLA",
            tier="light",
            results=[],
            success_count=0,
            error_count=0,
        )
        assert ticker_result.results == []
        assert ticker_result.success_count == 0
        assert ticker_result.error_count == 0

    def test_counts_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError, match="success_count"):
            TickerResult(
                ticker="AAPL",
                tier="deep",
                results=[],
                success_count=-1,
                error_count=0,
            )

        with pytest.raises(ValidationError, match="error_count"):
            TickerResult(
                ticker="AAPL",
                tier="deep",
                results=[],
                success_count=0,
                error_count=-1,
            )


class TestPipelineResult:
    """Tests for the PipelineResult model."""

    def test_construction_with_ticker_results(self) -> None:
        tool_result = ToolCallResult(
            tool_name="get_quote",
            ticker="NVDA",
            success=True,
            duration_ms=100.0,
        )
        ticker_result = TickerResult(
            ticker="NVDA",
            tier="light",
            results=[tool_result],
            success_count=1,
            error_count=0,
        )
        pipeline = PipelineResult(
            started_at="2025-01-15T09:30:00Z",
            completed_at="2025-01-15T09:31:00Z",
            duration_seconds=60.0,
            tickers_processed=1,
            total_calls=1,
            total_errors=0,
            ticker_results=[ticker_result],
        )
        assert pipeline.started_at == "2025-01-15T09:30:00Z"
        assert pipeline.completed_at == "2025-01-15T09:31:00Z"
        assert pipeline.duration_seconds == 60.0
        assert pipeline.tickers_processed == 1
        assert pipeline.total_calls == 1
        assert pipeline.total_errors == 0
        assert len(pipeline.ticker_results) == 1
        assert pipeline.ticker_results[0].ticker == "NVDA"

    def test_serialization_to_json(self) -> None:
        tool_result = ToolCallResult(
            tool_name="get_quote",
            ticker="GOOG",
            success=True,
            duration_ms=85.2,
        )
        ticker_result = TickerResult(
            ticker="GOOG",
            tier="deep",
            results=[tool_result],
            success_count=1,
            error_count=0,
        )
        pipeline = PipelineResult(
            started_at="2025-01-15T09:30:00Z",
            completed_at="2025-01-15T09:30:30Z",
            duration_seconds=30.0,
            tickers_processed=1,
            total_calls=1,
            total_errors=0,
            ticker_results=[ticker_result],
        )
        json_str = pipeline.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["started_at"] == "2025-01-15T09:30:00Z"
        assert parsed["tickers_processed"] == 1
        assert len(parsed["ticker_results"]) == 1
        assert parsed["ticker_results"][0]["ticker"] == "GOOG"

        # Roundtrip: deserialize back to model
        restored = PipelineResult.model_validate_json(json_str)
        assert restored == pipeline

    def test_zero_ticker_pipeline_run(self) -> None:
        pipeline = PipelineResult(
            started_at="2025-01-15T09:30:00Z",
            completed_at="2025-01-15T09:30:00Z",
            duration_seconds=0.0,
            tickers_processed=0,
            total_calls=0,
            total_errors=0,
            ticker_results=[],
        )
        assert pipeline.tickers_processed == 0
        assert pipeline.ticker_results == []

    def test_numeric_fields_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError, match="duration_seconds"):
            PipelineResult(
                started_at="2025-01-15T09:30:00Z",
                completed_at="2025-01-15T09:30:00Z",
                duration_seconds=-1.0,
                tickers_processed=0,
                total_calls=0,
                total_errors=0,
                ticker_results=[],
            )

        with pytest.raises(ValidationError, match="tickers_processed"):
            PipelineResult(
                started_at="2025-01-15T09:30:00Z",
                completed_at="2025-01-15T09:30:00Z",
                duration_seconds=0.0,
                tickers_processed=-1,
                total_calls=0,
                total_errors=0,
                ticker_results=[],
            )

        with pytest.raises(ValidationError, match="total_calls"):
            PipelineResult(
                started_at="2025-01-15T09:30:00Z",
                completed_at="2025-01-15T09:30:00Z",
                duration_seconds=0.0,
                tickers_processed=0,
                total_calls=-1,
                total_errors=0,
                ticker_results=[],
            )

        with pytest.raises(ValidationError, match="total_errors"):
            PipelineResult(
                started_at="2025-01-15T09:30:00Z",
                completed_at="2025-01-15T09:30:00Z",
                duration_seconds=0.0,
                tickers_processed=0,
                total_calls=0,
                total_errors=-1,
                ticker_results=[],
            )
