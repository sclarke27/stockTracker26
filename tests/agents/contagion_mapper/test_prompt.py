"""Tests for Cross-Sector Contagion Mapper prompt builders."""

from __future__ import annotations

from stock_radar.agents.contagion_mapper.prompt import (
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
        assert "contagion_likely" in prompt
        assert "contagion_probability" in prompt
        assert "contagion_mechanism" in prompt

    def test_contains_analysis_guidance(self) -> None:
        prompt = build_system_prompt()
        assert "contagion" in prompt.lower()
        assert "sector" in prompt.lower()

    def test_contains_json_instruction(self) -> None:
        prompt = build_system_prompt()
        assert "json" in prompt.lower()


class TestBuildUserPrompt:
    """Tests for build_user_prompt()."""

    def test_returns_string(self) -> None:
        prompt = build_user_prompt(
            trigger_ticker="NVDA",
            trigger_company_name="NVIDIA",
            target_ticker="AMD",
            target_company_name="Advanced Micro Devices",
            relationship_type="competitor",
            trigger_event_summary="NVDA missed earnings.",
            trigger_recent_news=[],
            target_recent_news=[],
            trigger_sector="Semiconductors",
            target_sector="Semiconductors",
        )
        assert isinstance(prompt, str)

    def test_contains_both_tickers(self) -> None:
        prompt = build_user_prompt(
            trigger_ticker="NVDA",
            trigger_company_name="NVIDIA",
            target_ticker="AMD",
            target_company_name="Advanced Micro Devices",
            relationship_type="competitor",
            trigger_event_summary="NVDA missed earnings.",
            trigger_recent_news=[],
            target_recent_news=[],
            trigger_sector="Semiconductors",
            target_sector="Semiconductors",
        )
        assert "NVDA" in prompt
        assert "AMD" in prompt

    def test_contains_relationship_type(self) -> None:
        prompt = build_user_prompt(
            trigger_ticker="TSMC",
            trigger_company_name="Taiwan Semiconductor",
            target_ticker="AAPL",
            target_company_name="Apple Inc.",
            relationship_type="supplier",
            trigger_event_summary="TSMC earthquake damage.",
            trigger_recent_news=[],
            target_recent_news=[],
            trigger_sector="Semiconductors",
            target_sector="Consumer Electronics",
        )
        assert "supplier" in prompt.lower()

    def test_includes_trigger_event_summary(self) -> None:
        prompt = build_user_prompt(
            trigger_ticker="NVDA",
            trigger_company_name="NVIDIA",
            target_ticker="AMD",
            target_company_name="Advanced Micro Devices",
            relationship_type="competitor",
            trigger_event_summary="NVDA product recall announced.",
            trigger_recent_news=[],
            target_recent_news=[],
            trigger_sector="Semiconductors",
            target_sector="Semiconductors",
        )
        assert "NVDA product recall" in prompt

    def test_includes_trigger_news_titles(self) -> None:
        news = [{"title": "NVDA reports massive earnings miss", "sentiment_score": -0.8}]
        prompt = build_user_prompt(
            trigger_ticker="NVDA",
            trigger_company_name="NVIDIA",
            target_ticker="AMD",
            target_company_name="Advanced Micro Devices",
            relationship_type="competitor",
            trigger_event_summary="NVDA miss.",
            trigger_recent_news=news,
            target_recent_news=[],
            trigger_sector="Semiconductors",
            target_sector="Semiconductors",
        )
        assert "NVDA reports massive earnings miss" in prompt

    def test_includes_sector_information(self) -> None:
        prompt = build_user_prompt(
            trigger_ticker="NVDA",
            trigger_company_name="NVIDIA",
            target_ticker="AMD",
            target_company_name="Advanced Micro Devices",
            relationship_type="same_sector",
            trigger_event_summary="Event.",
            trigger_recent_news=[],
            target_recent_news=[],
            trigger_sector="Semiconductors",
            target_sector="Semiconductors",
        )
        assert "Semiconductors" in prompt


class TestBuildMessages:
    """Tests for build_messages()."""

    def test_returns_two_messages(self) -> None:
        messages = build_messages(
            trigger_ticker="NVDA",
            trigger_company_name="NVIDIA",
            target_ticker="AMD",
            target_company_name="Advanced Micro Devices",
            relationship_type="competitor",
            trigger_event_summary="Event.",
            trigger_recent_news=[],
            target_recent_news=[],
            trigger_sector="Semiconductors",
            target_sector="Semiconductors",
        )
        assert len(messages) == 2

    def test_first_message_is_system(self) -> None:
        messages = build_messages(
            trigger_ticker="NVDA",
            trigger_company_name="NVIDIA",
            target_ticker="AMD",
            target_company_name="Advanced Micro Devices",
            relationship_type="competitor",
            trigger_event_summary="Event.",
            trigger_recent_news=[],
            target_recent_news=[],
            trigger_sector="Semiconductors",
            target_sector="Semiconductors",
        )
        assert isinstance(messages[0], LlmMessage)
        assert messages[0].role == "system"

    def test_second_message_is_user(self) -> None:
        messages = build_messages(
            trigger_ticker="NVDA",
            trigger_company_name="NVIDIA",
            target_ticker="AMD",
            target_company_name="Advanced Micro Devices",
            relationship_type="competitor",
            trigger_event_summary="Event.",
            trigger_recent_news=[],
            target_recent_news=[],
            trigger_sector="Semiconductors",
            target_sector="Semiconductors",
        )
        assert messages[1].role == "user"
        assert "NVDA" in messages[1].content
        assert "AMD" in messages[1].content
