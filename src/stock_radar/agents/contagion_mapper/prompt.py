"""Prompt engineering for the Cross-Sector Contagion Mapper agent."""

from __future__ import annotations

import json

from stock_radar.agents.contagion_mapper.models import ContagionAnalysis
from stock_radar.llm.models import LlmMessage


def build_system_prompt() -> str:
    """Build the system prompt for cross-sector contagion analysis.

    Embeds the ContagionAnalysis JSON schema so the LLM knows the
    exact output format required. The Pydantic model is the single
    source of truth — no manual schema maintenance.

    Returns:
        Complete system prompt string.
    """
    schema = json.dumps(ContagionAnalysis.model_json_schema(), indent=2)

    return f"""You are an expert financial analyst specializing in cross-sector \
contagion analysis. Your task is to determine whether a shock event at one \
company (the trigger) will propagate to and impact another related company \
(the target).

## What is Contagion?

Contagion occurs when an event at one company causes ripple effects at related \
companies. Contagion can be **positive or negative** — analyze both directions \
with equal weight.

### Negative Contagion Examples
1. **Supply Chain Disruption**: Supplier problems → customer shortages or \
cost increases.
2. **Customer Weakness**: Major customer misses earnings → reduced orders \
for suppliers.
3. **Industry-Wide Headwinds**: Bellwether miss reflects macro conditions \
affecting all peers.

### Positive Contagion Examples
1. **Competitor Stumble = Market Share Gain**: Competitor's scandal, recall, \
or miss creates opportunity for the target to capture share.
2. **Customer Strength**: Major customer beats and raises → increased orders \
for suppliers.
3. **Industry Tailwind**: Bellwether beats on strong demand → positive signal \
for all peers in the sector.
4. **Distribution Expansion**: Partner grows market share → more shelf space \
and reach for vendors.

### Neutral / No Contagion
- The trigger event is idiosyncratic (company-specific problem with no \
supply chain or competitive linkage) → NEUTRAL
- The relationship is too indirect to have meaningful impact → NEUTRAL

## Analysis Framework

For each relationship type:
- **Supplier → Customer**: Does the trigger's problem affect the target's \
supply chain or input costs?
- **Customer → Supplier**: Does the trigger's reduced demand hit the target's \
revenue?
- **Competitor → Competitor**: Does the miss reflect industry conditions or \
competitive repositioning?
- **Same Sector**: Is this a macro/regulatory issue affecting all peers?
- **Distribution Partner**: Does the trigger's weakness reduce the target's \
distribution reach?

## Key Considerations

- **Time the contagion**: Supply chain impacts take 1-2 quarters; sentiment \
contagion can be immediate (1-5 days).
- **Structural vs. idiosyncratic**: Is the trigger event company-specific or \
industry-wide? Company-specific events often have no contagion.
- **Direction is not always negative**: A competitor's weakness can be bullish \
for the target. A customer's strength can be bullish for suppliers. Always \
consider both directions before concluding.
- **Default to NEUTRAL**: If the contagion pathway is unclear or indirect, \
output NEUTRAL rather than forcing a directional call.

## Output Requirements

Respond with valid JSON matching this exact schema:

```json
{schema}
```

## Confidence Scoring

- **0.8-1.0**: Strong, direct linkage with clear contagion mechanism
- **0.6-0.8**: Likely contagion with moderate evidence
- **0.4-0.6**: Possible contagion, ambiguous signals
- **0.2-0.4**: Weak linkage, limited contagion expected
- **0.0-0.2**: No meaningful contagion pathway identified"""


def build_user_prompt(
    trigger_ticker: str,
    trigger_company_name: str,
    target_ticker: str,
    target_company_name: str,
    relationship_type: str,
    trigger_event_summary: str,
    trigger_recent_news: list[dict],
    target_recent_news: list[dict],
    trigger_sector: str,
    target_sector: str,
) -> str:
    """Build the user message with contagion analysis context.

    Args:
        trigger_ticker: Ticker of the company with the trigger event.
        trigger_company_name: Full name of trigger company.
        target_ticker: Ticker of the company being assessed for impact.
        target_company_name: Full name of target company.
        relationship_type: Relationship between the two companies.
        trigger_event_summary: What happened at the trigger company.
        trigger_recent_news: Recent news about the trigger.
        target_recent_news: Recent news about the target.
        trigger_sector: Sector of the trigger company.
        target_sector: Sector of the target company.

    Returns:
        Formatted user prompt string.
    """
    parts = [
        f"Analyze the contagion risk from **{trigger_ticker}** "
        f"({trigger_company_name}) to **{target_ticker}** ({target_company_name}).",
        "",
        f"**Relationship**: {relationship_type.replace('_', ' ').title()}",
        f"**Trigger Sector**: {trigger_sector}",
        f"**Target Sector**: {target_sector}",
        "",
        f"## Trigger Event: {trigger_ticker}",
        "",
        trigger_event_summary,
    ]

    if trigger_recent_news:
        parts.extend(["", f"### Recent {trigger_ticker} News", ""])
        for article in trigger_recent_news:
            title = article.get("title", "")
            score = article.get("overall_sentiment_score", article.get("sentiment_score", 0.0))
            summary = article.get("summary", "")
            parts.append(f"- **{title}** (sentiment: {score:.2f})")
            if summary:
                parts.append(f"  {summary[:200]}")

    parts.extend(["", f"## Target Company: {target_ticker}", ""])

    if target_recent_news:
        parts.extend([f"### Recent {target_ticker} News", ""])
        for article in target_recent_news:
            title = article.get("title", "")
            score = article.get("overall_sentiment_score", article.get("sentiment_score", 0.0))
            summary = article.get("summary", "")
            parts.append(f"- **{title}** (sentiment: {score:.2f})")
            if summary:
                parts.append(f"  {summary[:200]}")
    else:
        parts.append(f"No recent {target_ticker} news available.")

    parts.extend(
        [
            "",
            f"Assess whether the trigger event at {trigger_ticker} will propagate "
            f"to {target_ticker} through the {relationship_type.replace('_', ' ')} "
            f"relationship, and predict the likely direction and timeline.",
        ]
    )

    return "\n".join(parts)


def build_messages(
    trigger_ticker: str,
    trigger_company_name: str,
    target_ticker: str,
    target_company_name: str,
    relationship_type: str,
    trigger_event_summary: str,
    trigger_recent_news: list[dict],
    target_recent_news: list[dict],
    trigger_sector: str,
    target_sector: str,
) -> list[LlmMessage]:
    """Build the complete message list for the LLM.

    Args:
        trigger_ticker: Ticker of the trigger company.
        trigger_company_name: Full name of trigger company.
        target_ticker: Ticker of the target company.
        target_company_name: Full name of target company.
        relationship_type: Relationship type between companies.
        trigger_event_summary: Description of the trigger event.
        trigger_recent_news: Recent news about the trigger.
        target_recent_news: Recent news about the target.
        trigger_sector: Sector of the trigger company.
        target_sector: Sector of the target company.

    Returns:
        List of LlmMessage with system and user messages.
    """
    return [
        LlmMessage(role="system", content=build_system_prompt()),
        LlmMessage(
            role="user",
            content=build_user_prompt(
                trigger_ticker=trigger_ticker,
                trigger_company_name=trigger_company_name,
                target_ticker=target_ticker,
                target_company_name=target_company_name,
                relationship_type=relationship_type,
                trigger_event_summary=trigger_event_summary,
                trigger_recent_news=trigger_recent_news,
                target_recent_news=target_recent_news,
                trigger_sector=trigger_sector,
                target_sector=target_sector,
            ),
        ),
    ]
