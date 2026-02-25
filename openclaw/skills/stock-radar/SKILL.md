---
name: stock-radar
description: Stock Radar — AI-powered stock market analysis. Run analysis cycles, check predictions, query agent status.
user-invocable: true
metadata: {"openclaw": {"always": false, "requires": {"bins": ["stock-radar-orchestrate"]}}}
---

# Stock Radar

You manage a Stock Radar system — an AI-powered stock market analysis tool that uses LLMs to identify tradeable patterns from unstructured data (earnings transcripts, SEC filings, news).

## System Layout

The project lives at `/home/pi/stock-radar`. All commands must be run from this directory with the Python virtualenv activated:

```
cd /home/pi/stock-radar && source .venv/bin/activate
```

## Available Commands

### Full orchestrator cycle (ingestion + analysis + scoring)

```bash
stock-radar-orchestrate
```

Runs three phases in sequence:
1. **Ingestion** — fetches market data, SEC filings, earnings transcripts from external APIs
2. **Analysis** — runs 4 AI agents against the watchlist tickers
3. **Scoring** — scores mature predictions against actual price movements

Exit code 0 on success, 1 if any errors occurred.

### Individual components

```bash
stock-radar-ingest                   # Data ingestion only
stock-radar-earnings-linguist        # Earnings sentiment agent
stock-radar-narrative-divergence     # Narrative vs price divergence agent
stock-radar-sec-filing-analyzer      # SEC filing pattern agent
stock-radar-contagion-mapper         # Cross-sector contagion agent
stock-radar-score                    # Score mature predictions
```

### Dashboard API

```bash
stock-radar-dashboard-mcp            # HTTP API on port 8082
```

The dashboard API is a persistent service (runs via systemd). It provides read-only access to active signals, watchlist status, and agent performance.

## Watchlist

The system tracks stocks in two tiers:

- **Deep** (10 tickers): AAPL, MSFT, GOOGL, AMZN, NVDA, META, JPM, V, NOW, NFLX — all 4 agents run against these
- **Light** (11 tickers): AMD, BA, DIS, CRM, INTC, PFE, XOM, WMT, KO, DASH, WBD — only narrative divergence and SEC filing analysis

The watchlist is configured in `config/watchlist.yaml`.

## Analysis Agents

| Agent | What It Detects | Horizon |
|-------|-----------------|---------|
| Earnings Linguist | Sentiment shifts, hedging language in earnings calls | 5 days |
| Narrative Divergence | Media narrative vs actual price action gaps | 10 days |
| SEC Filing Analyzer | Unusual filing patterns, insider transaction clusters | 15 days |
| Contagion Mapper | Cross-sector signal propagation via supply chains | 5 days |

## Configuration

- Main config: `config/default.yaml`
- Watchlist: `config/watchlist.yaml`
- Contagion pairs: `config/contagion_pairs.yaml`
- Environment variables: `.env` (API keys for Alpha Vantage, SEC EDGAR)

## Data Storage

| Path | Contents |
|------|----------|
| `data/stock_radar.db` | SQLite cache for market data and SEC filings |
| `data/predictions.db` | Prediction tracking and scoring results |
| `data/chroma_data/` | ChromaDB vector embeddings |

## LLM Configuration

All inference runs on Ollama (`qwen3:32b`) on the desktop (10.0.0.15). Cloud LLM escalation (Claude/OpenAI) is disabled by default (`ollama_only: true` in config). To re-enable cloud escalation, set `ollama_only: false` in `config/default.yaml`.

## When Reporting Results

When running the orchestrator or individual agents, always report:
- Number of predictions generated
- Any errors that occurred and what caused them
- Total execution time
- For scoring runs: how many predictions were scored and the accuracy breakdown

## Troubleshooting

- If ingestion fails, agents can still run on cached (stale) data
- If an individual agent fails, the others continue running
- Check `logs/` directory for detailed log files
- The Ollama server must be running on the desktop (RTX 4090) at `http://10.0.0.15:11434`
- Verify Ollama connectivity: `curl http://10.0.0.15:11434/api/tags`
