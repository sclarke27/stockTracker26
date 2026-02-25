"""Prompt engineering for the Earnings Linguist agent."""

from __future__ import annotations

import json

from stock_radar.agents.earnings_linguist.models import EarningsAnalysis
from stock_radar.llm.models import LlmMessage


def build_system_prompt() -> str:
    """Build the system prompt for earnings transcript analysis.

    Embeds the EarningsAnalysis JSON schema so the LLM knows the
    exact output format required. The Pydantic model is the single
    source of truth — no manual schema maintenance.

    Returns:
        Complete system prompt string.
    """
    schema = json.dumps(EarningsAnalysis.model_json_schema(), indent=2)

    return f"""You are an expert financial linguist specializing in earnings call \
transcript analysis. Your task is to analyze earnings call transcripts and \
identify sentiment shifts, hedging language, and quarter-over-quarter tone \
changes that may signal future stock price movement.

## Analysis Guidelines

Your analysis must be **balanced** — look equally for bullish and bearish \
signals. Most earnings calls contain a mix of both. Only assign a directional \
prediction when there is a clear imbalance.

### Bearish Signals
1. **Hedging Language**: Phrases indicating uncertainty or caution:
   - "we believe", "potentially", "we hope", "uncertain", "challenging environment"
   - Qualifiers that weaken previously strong statements
2. **Weakened Guidance**: Lowered or widened guidance ranges, vague projections \
where specific numbers were given before.
3. **Evasive Answers**: Defensive or non-responsive answers to analyst questions.
4. **New Risk Factors**: Litigation, regulatory issues, supply chain disruptions.

### Bullish Signals
1. **Confident Language**: Specific commitments, raised guidance, narrowed ranges.
   - "we are confident", "we expect", "strong momentum", "accelerating growth"
   - Management volunteering specific numbers or targets
2. **Strengthened Guidance**: Raised outlook, beat-and-raise patterns.
3. **Proactive Disclosure**: Management volunteering positive details unprompted, \
transparent answers to analyst concerns.
4. **Strategic Initiatives**: New product launches, market expansion, successful \
cost optimization described with specifics.

### Neutral / No Signal
- Maintained guidance with no meaningful language shifts → NEUTRAL
- Boilerplate earnings calls with standard corporate language → NEUTRAL
- Do not force a directional call when the transcript is unremarkable.

### Tone Changes (Either Direction)
- Optimistic → cautious (bearish shift)
- Cautious → confident (bullish shift)
- Compare against prior quarter transcript when available

## Output Requirements

Respond with valid JSON matching this exact schema:

```json
{schema}
```

## Confidence Scoring

- **0.8-1.0**: Clear, unambiguous signals with strong evidence
- **0.6-0.8**: Moderate signals with some supporting evidence
- **0.4-0.6**: Mixed signals, balanced bull/bear indicators
- **0.2-0.4**: Weak signals, mostly noise
- **0.0-0.2**: No meaningful signals detected

## Horizon Days

- **5**: Short-term event-driven (earnings surprise, guidance change)
- **10-15**: Medium-term sentiment shift
- **20-30**: Structural or strategic changes"""


def build_user_prompt(
    ticker: str,
    transcript: str,
    prior_transcript: str | None = None,
    company_name: str = "",
) -> str:
    """Build the user message with transcript content.

    Args:
        ticker: Stock ticker symbol.
        transcript: Current quarter earnings transcript.
        prior_transcript: Previous quarter transcript for comparison.
        company_name: Company name for additional context.

    Returns:
        Formatted user prompt string.
    """
    company_label = f" ({company_name})" if company_name else ""
    parts = [
        f"Analyze the following earnings call transcript for {ticker}{company_label}.",
        "",
        "## Current Quarter Transcript",
        "",
        transcript,
    ]

    if prior_transcript:
        parts.extend(
            [
                "",
                "## Previous Quarter Transcript (for comparison)",
                "",
                prior_transcript,
                "",
                "Compare the tone, language, and sentiment between the two quarters.",
            ]
        )

    return "\n".join(parts)


def build_messages(
    ticker: str,
    transcript: str,
    prior_transcript: str | None = None,
    company_name: str = "",
) -> list[LlmMessage]:
    """Build the complete message list for the LLM.

    Args:
        ticker: Stock ticker symbol.
        transcript: Current quarter transcript.
        prior_transcript: Previous quarter transcript.
        company_name: Company name.

    Returns:
        List of LlmMessage with system and user messages.
    """
    return [
        LlmMessage(role="system", content=build_system_prompt()),
        LlmMessage(
            role="user",
            content=build_user_prompt(ticker, transcript, prior_transcript, company_name),
        ),
    ]
