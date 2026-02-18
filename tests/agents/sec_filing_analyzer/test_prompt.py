"""Tests for SEC Filing Pattern Analyzer prompt builders."""

from __future__ import annotations

from stock_radar.agents.sec_filing_analyzer.prompt import (
    build_messages,
    build_system_prompt,
    build_user_prompt,
)
from stock_radar.llm.models import LlmMessage


class TestBuildSystemPrompt:
    """Tests for build_system_prompt()."""

    def test_returns_string(self) -> None:
        prompt = build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_schema_fields(self) -> None:
        prompt = build_system_prompt()
        assert "patterns_detected" in prompt
        assert "insider_summary" in prompt
        assert "insider_sentiment" in prompt

    def test_contains_analysis_guidance(self) -> None:
        prompt = build_system_prompt()
        assert "insider" in prompt.lower()
        assert "filing" in prompt.lower()

    def test_contains_json_instruction(self) -> None:
        prompt = build_system_prompt()
        assert "json" in prompt.lower()


class TestBuildUserPrompt:
    """Tests for build_user_prompt()."""

    def test_returns_string(self) -> None:
        prompt = build_user_prompt(
            ticker="TSLA",
            recent_filings=[],
            insider_transactions=[],
            filing_count=0,
            insider_transaction_count=0,
            lookback_days=90,
        )
        assert isinstance(prompt, str)

    def test_contains_ticker(self) -> None:
        prompt = build_user_prompt(
            ticker="NVDA",
            recent_filings=[],
            insider_transactions=[],
            filing_count=0,
            insider_transaction_count=0,
            lookback_days=90,
        )
        assert "NVDA" in prompt

    def test_contains_filing_count(self) -> None:
        prompt = build_user_prompt(
            ticker="AAPL",
            recent_filings=[{"form_type": "8-K", "filed_at": "2024-01-15", "description": "X"}],
            insider_transactions=[],
            filing_count=5,
            insider_transaction_count=0,
            lookback_days=90,
        )
        assert "5" in prompt

    def test_includes_filing_details(self) -> None:
        filings = [{"form_type": "10-K", "filed_at": "2024-03-15", "description": "Annual report"}]
        prompt = build_user_prompt(
            ticker="AAPL",
            recent_filings=filings,
            insider_transactions=[],
            filing_count=1,
            insider_transaction_count=0,
            lookback_days=90,
        )
        assert "10-K" in prompt

    def test_includes_insider_transaction_details(self) -> None:
        txns = [
            {
                "insider_name": "John Smith",
                "transaction_type": "P",
                "shares": 10000,
                "date": "2024-06-01",
            }
        ]
        prompt = build_user_prompt(
            ticker="AAPL",
            recent_filings=[],
            insider_transactions=txns,
            filing_count=0,
            insider_transaction_count=1,
            lookback_days=90,
        )
        assert "John Smith" in prompt

    def test_no_filings_produces_valid_prompt(self) -> None:
        prompt = build_user_prompt(
            ticker="AAPL",
            recent_filings=[],
            insider_transactions=[],
            filing_count=0,
            insider_transaction_count=0,
            lookback_days=90,
        )
        assert "AAPL" in prompt
        assert len(prompt) > 50


class TestBuildMessages:
    """Tests for build_messages()."""

    def test_returns_two_messages(self) -> None:
        messages = build_messages(
            ticker="TSLA",
            recent_filings=[],
            insider_transactions=[],
            filing_count=0,
            insider_transaction_count=0,
            lookback_days=90,
        )
        assert len(messages) == 2

    def test_first_message_is_system(self) -> None:
        messages = build_messages(
            ticker="TSLA",
            recent_filings=[],
            insider_transactions=[],
            filing_count=0,
            insider_transaction_count=0,
            lookback_days=90,
        )
        assert isinstance(messages[0], LlmMessage)
        assert messages[0].role == "system"

    def test_second_message_is_user(self) -> None:
        messages = build_messages(
            ticker="TSLA",
            recent_filings=[],
            insider_transactions=[],
            filing_count=0,
            insider_transaction_count=0,
            lookback_days=90,
        )
        assert messages[1].role == "user"
        assert "TSLA" in messages[1].content
