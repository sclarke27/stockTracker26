"""Prompt engineering for the SEC Filing Pattern Analyzer agent."""

from __future__ import annotations

import json

from stock_radar.agents.sec_filing_analyzer.models import SecFilingAnalysis
from stock_radar.llm.models import LlmMessage


def build_system_prompt() -> str:
    """Build the system prompt for SEC filing pattern analysis.

    Embeds the SecFilingAnalysis JSON schema so the LLM knows the
    exact output format required. The Pydantic model is the single
    source of truth — no manual schema maintenance.

    Returns:
        Complete system prompt string.
    """
    schema = json.dumps(SecFilingAnalysis.model_json_schema(), indent=2)

    return f"""You are an expert financial analyst specializing in SEC filing \
analysis and insider transaction interpretation. Your task is to analyze recent \
SEC filings and insider transactions to identify patterns that may predict \
future stock price movement.

## Analysis Framework

### Important: Routine Activity Is Not a Signal

Most SEC filings and insider transactions are **routine**. Do not treat normal \
activity as a signal. Only flag patterns that are genuinely unusual in context.

- Executives regularly sell shares for taxes, diversification, and liquidity. \
**Routine selling is NOT bearish.** Only flag selling that is unusual in timing, \
size, or clustering relative to the company's historical pattern.
- Most 8-K filings report ordinary events (board changes, compensation plans). \
Only unusual frequency or content warrants attention.
- A single insider selling is almost never a signal on its own.

### Bullish Patterns

1. **Insider Buying Cluster**: Multiple insiders purchasing shares in a short \
window — strong bullish signal. Insiders risk their own money, so voluntary \
buying reflects genuine confidence.

2. **Large Open-Market Purchase by CEO/CFO**: Especially meaningful when the \
executive buys >1% of their holdings voluntarily (not options exercise).

3. **Clean Filing Cadence**: On-time 10-K/10-Q with no amendments or NT filings \
indicates well-functioning internal controls — supportive of stability.

### Bearish Patterns

1. **Unusual Insider Selling Cluster**: Multiple C-suite insiders selling in a \
short window **outside** of pre-planned 10b5-1 programs and at unusual volumes \
relative to their history. One-off sales are typically routine.

2. **Late Filing (NT 10-K/10-Q)**: Filing past deadline suggests internal \
control issues or auditor disagreements.

3. **Executive Departure + Red Flags**: C-suite departure combined with other \
signals (late filings, restatements) — not departure alone.

### Context Matters

- **Planned sales** (10b5-1 programs): Insiders pre-schedule these — they are \
NOT informative about near-term outlook.
- **Options exercises + immediate sale**: Routine tax optimization, not bearish.
- **Single insider selling**: Almost always routine. Require a cluster or \
unusual pattern to flag.

### Signal Strength

- Multiple aligned signals (e.g., buying cluster + raised guidance) → HIGH \
confidence bullish
- Multiple aligned signals (e.g., selling cluster + late filing + departure) → \
HIGH confidence bearish
- Single clear signal → MEDIUM confidence
- Routine filings and normal transactions → NEUTRAL direction, LOW confidence
- When in doubt, default to NEUTRAL — false signals are worse than missed ones

## Output Requirements

Respond with valid JSON matching this exact schema:

```json
{schema}
```

## Confidence Scoring

- **0.8-1.0**: Multiple aligned, unambiguous signals (e.g., buying cluster + \
strong fundamentals, or selling cluster + late filing + departure)
- **0.6-0.8**: Single strong signal with corroborating evidence
- **0.4-0.6**: Moderate signals requiring interpretation
- **0.2-0.4**: Weak or routine filing activity — likely NEUTRAL
- **0.0-0.2**: No meaningful patterns — normal filing cadence, output NEUTRAL"""


def build_user_prompt(
    ticker: str,
    recent_filings: list[dict],
    insider_transactions: list[dict],
    filing_count: int,
    insider_transaction_count: int,
    lookback_days: int,
) -> str:
    """Build the user message with SEC filing and insider transaction data.

    Args:
        ticker: Stock ticker symbol.
        recent_filings: List of recent filings with form_type, filed_at, description.
        insider_transactions: List of insider transactions with name, type, shares, date.
        filing_count: Total filing count in window.
        insider_transaction_count: Total insider transaction count.
        lookback_days: Number of days of history analyzed.

    Returns:
        Formatted user prompt string.
    """
    parts = [
        f"Analyze the SEC filings and insider transactions for **{ticker}** "
        f"over the past {lookback_days} days.",
        "",
        f"## SEC Filings ({filing_count} total)",
        "",
    ]

    if recent_filings:
        for filing in recent_filings:
            form_type = filing.get("form_type", "Unknown")
            filed_at = filing.get("filed_at", "Unknown date")
            description = filing.get("description", "")
            parts.append(f"- **{form_type}** filed {filed_at}: {description}")
    else:
        parts.append("No filings found in this window.")

    parts.extend(
        [
            "",
            f"## Insider Transactions ({insider_transaction_count} total)",
            "",
        ]
    )

    if insider_transactions:
        for txn in insider_transactions:
            name = txn.get("insider_name", "Unknown")
            title = txn.get("title", "")
            txn_type = txn.get("transaction_type", "?")
            shares = txn.get("shares", 0)
            date = txn.get("date", "Unknown date")
            price = txn.get("price_per_share")

            txn_label = "Purchased" if txn_type in ("P", "A") else "Sold"
            price_str = f" @ ${price:.2f}" if price else ""
            title_str = f" ({title})" if title else ""
            parts.append(f"- {name}{title_str}: {txn_label} {shares:,} shares{price_str} on {date}")
    else:
        parts.append("No insider transactions found in this window.")

    parts.extend(
        [
            "",
            f"Identify patterns in these filings and transactions that may predict "
            f"the near-term price direction of {ticker}.",
        ]
    )

    return "\n".join(parts)


def build_messages(
    ticker: str,
    recent_filings: list[dict],
    insider_transactions: list[dict],
    filing_count: int,
    insider_transaction_count: int,
    lookback_days: int,
) -> list[LlmMessage]:
    """Build the complete message list for the LLM.

    Args:
        ticker: Stock ticker symbol.
        recent_filings: List of recent SEC filings.
        insider_transactions: List of insider transactions.
        filing_count: Total filing count.
        insider_transaction_count: Total insider transaction count.
        lookback_days: Days of history analyzed.

    Returns:
        List of LlmMessage with system and user messages.
    """
    return [
        LlmMessage(role="system", content=build_system_prompt()),
        LlmMessage(
            role="user",
            content=build_user_prompt(
                ticker=ticker,
                recent_filings=recent_filings,
                insider_transactions=insider_transactions,
                filing_count=filing_count,
                insider_transaction_count=insider_transaction_count,
                lookback_days=lookback_days,
            ),
        ),
    ]
