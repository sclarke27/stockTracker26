"""Prompt engineering for the Narrative vs Price Divergence agent."""

from __future__ import annotations

import json

from stock_radar.agents.narrative_divergence.models import NarrativeAnalysis
from stock_radar.llm.models import LlmMessage


def build_system_prompt() -> str:
    """Build the system prompt for narrative vs price divergence analysis.

    Embeds the NarrativeAnalysis JSON schema so the LLM knows the
    exact output format required. The Pydantic model is the single
    source of truth — no manual schema maintenance.

    Returns:
        Complete system prompt string.
    """
    schema = json.dumps(NarrativeAnalysis.model_json_schema(), indent=2)

    return f"""You are an expert financial analyst specializing in detecting \
divergences between media narrative and stock price action. Your task is to \
analyze news sentiment data alongside recent price movements to identify cases \
where the market may be mis-pricing a stock relative to its prevailing narrative.

## Analysis Framework

A **narrative vs price divergence** occurs when:
- News sentiment is strongly bullish but the stock price is falling (potential \
long opportunity — market underreacting to positive news)
- News sentiment is strongly bearish but the stock price is rising (potential \
short opportunity — market ignoring warning signs)

## Key Considerations

1. **Sentiment Quality**: Higher article counts produce more reliable signals. \
Very few articles may indicate low information flow rather than true divergence.

2. **Price Action Context**: Short-term price drops (7-day) in trending stocks \
may be noise. 30-day returns provide better signal of sustained divergence.

3. **Divergence Strength**: Score based on the absolute gap between sentiment \
direction and price direction. Large positive sentiment with large price decline \
= strong divergence.

4. **Catalysts**: Identify specific news themes driving the narrative. Without \
clear catalysts, a divergence signal is weaker.

5. **Horizon**: Short divergences resolve faster (5-7 days). Structural narrative \
shifts may take 15-30 days to be reflected in price.

## Output Requirements

Respond with valid JSON matching this exact schema:

```json
{schema}
```

## Confidence Scoring

- **0.8-1.0**: Clear divergence with strong sentiment signal and many articles
- **0.6-0.8**: Moderate divergence with reasonable article coverage
- **0.4-0.6**: Weak or ambiguous divergence signal
- **0.2-0.4**: Noisy data — few articles or mixed sentiment
- **0.0-0.2**: No meaningful divergence detected"""


def build_user_prompt(
    ticker: str,
    sentiment_score: float,
    article_count: int,
    average_sentiment_label: str,
    price_return_30d: float,
    price_return_7d: float,
    top_articles: list[dict],
    time_from: str | None = None,
    time_to: str | None = None,
) -> str:
    """Build the user message with sentiment and price data.

    Args:
        ticker: Stock ticker symbol.
        sentiment_score: Average sentiment score (-1.0 to 1.0).
        article_count: Number of articles analyzed.
        average_sentiment_label: Human-readable sentiment label.
        price_return_30d: 30-day price return as decimal.
        price_return_7d: 7-day price return as decimal.
        top_articles: List of article dicts with title/summary/sentiment_score.
        time_from: Start of sentiment window (optional).
        time_to: End of sentiment window (optional).

    Returns:
        Formatted user prompt string.
    """
    window_info = ""
    if time_from or time_to:
        window_info = f" (window: {time_from or 'N/A'} → {time_to or 'N/A'})"

    price_30d_pct = f"{price_return_30d * 100:+.1f}%"
    price_7d_pct = f"{price_return_7d * 100:+.1f}%"

    parts = [
        f"Analyze the narrative vs price divergence for **{ticker}**{window_info}.",
        "",
        "## News Sentiment Data",
        "",
        f"- Articles analyzed: {article_count}",
        f"- Average sentiment score: {sentiment_score:.3f} ({average_sentiment_label})",
        "",
        "## Price Action",
        "",
        f"- 30-day price return: {price_30d_pct}",
        f"- 7-day price return: {price_7d_pct}",
    ]

    if top_articles:
        parts.extend(
            [
                "",
                "## Top Articles (for context)",
                "",
            ]
        )
        for i, article in enumerate(top_articles, 1):
            title = article.get("title", "")
            summary = article.get("summary", "")
            score = article.get("overall_sentiment_score", article.get("sentiment_score", 0.0))
            parts.append(f"{i}. **{title}** (sentiment: {score:.2f})")
            if summary:
                parts.append(f"   {summary[:200]}")

    parts.extend(
        [
            "",
            f"Assess whether the {average_sentiment_label} news sentiment "
            f"diverges meaningfully from the recent price action "
            f"({price_30d_pct} over 30 days) and predict the likely resolution.",
        ]
    )

    return "\n".join(parts)


def build_messages(
    ticker: str,
    sentiment_score: float,
    article_count: int,
    average_sentiment_label: str,
    price_return_30d: float,
    price_return_7d: float,
    top_articles: list[dict],
    time_from: str | None = None,
    time_to: str | None = None,
) -> list[LlmMessage]:
    """Build the complete message list for the LLM.

    Args:
        ticker: Stock ticker symbol.
        sentiment_score: Average sentiment score.
        article_count: Number of articles analyzed.
        average_sentiment_label: Human-readable sentiment label.
        price_return_30d: 30-day price return.
        price_return_7d: 7-day price return.
        top_articles: Top article snippets for context.
        time_from: Start of sentiment window.
        time_to: End of sentiment window.

    Returns:
        List of LlmMessage with system and user messages.
    """
    return [
        LlmMessage(role="system", content=build_system_prompt()),
        LlmMessage(
            role="user",
            content=build_user_prompt(
                ticker=ticker,
                sentiment_score=sentiment_score,
                article_count=article_count,
                average_sentiment_label=average_sentiment_label,
                price_return_30d=price_return_30d,
                price_return_7d=price_return_7d,
                top_articles=top_articles,
                time_from=time_from,
                time_to=time_to,
            ),
        ),
    ]
