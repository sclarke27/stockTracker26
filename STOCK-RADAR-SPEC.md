# Stock Radar — Project Spec

## What to Build

An AI-powered stock market analysis system. It reads unstructured data — earnings call transcripts, SEC filings, news articles, macro indicators — and uses LLMs to identify tradeable patterns. This is not a traditional quant system. The edge comes from language analysis, not numerical time series.

## Hardware

- **Raspberry Pi 5**: Orchestrator. Runs scheduling, routing, lightweight coordination, OpenClaw, and the dashboard.
- **Desktop with NVIDIA RTX 4090**: Runs local LLMs via Ollama for inference.
- Both on the same local network. No cloud infrastructure. Only outbound connections are API calls.

## How It Works

### Data Ingestion
The system pulls data from external sources and caches it locally:
- **Alpha Vantage** — market data, prices, fundamentals, technical indicators, news sentiment.
- **SEC EDGAR** — filings (8-K, 10-K, 10-Q), insider transactions, filing pattern analysis. Free API, no key needed, 10 req/sec limit.
- **Finnhub** (free or $50/mo) — earnings call transcripts.

All fetched data gets cached in a local SQLite database to avoid redundant API calls.

### Analysis (Agents)
AI agents analyze the ingested data and produce structured predictions. Each agent focuses on one signal type:

1. **Earnings Language Analysis** (highest priority) — Detects sentiment shifts, hedging language, and quarter-over-quarter tone changes in earnings transcripts.
2. **Cross-Sector Contagion Mapping** — Tracks how a signal in one stock/sector propagates through supply chain and competitive relationships.
3. **Narrative vs. Price Divergence** — Flags when media/social sentiment diverges from actual price action.
4. **SEC Filing Pattern Analysis** — Spots unusual 8-K patterns, insider transaction clusters, amendment timing.

Agents run locally on Ollama (RTX 4090) for routine analysis. Complex multi-document synthesis escalates to the Claude API. LLM model selection is based on what's best for each task — local small model → local large model → Claude Sonnet → Claude Opus.

### Prediction Tracking & Learning Loop
Every prediction the system makes gets logged to a database with:
- What the prediction was (ticker, direction, confidence, reasoning)
- What actually happened (price movement after N days)
- A score (was it right or wrong, by how much)

This prediction history feeds back into future analysis. Agents can query past predictions to find "what happened last time we saw a pattern like this" using vector similarity search.

### Coverage Tiers
- **Deep (20-50 stocks)**: Full multi-signal analysis, daily updates, prediction tracking
- **Light (500-1000 stocks)**: Lightweight monitoring, anomaly flagging only
- **Screening (broad market)**: Pattern scanning for candidates to promote to higher tiers

### Dashboard
A PWA (Lit + TypeScript) that surfaces active signals, the watchlist, recent predictions, and agent performance. Connects to the `dashboard-mcp` server over HTTP. Runs on the Raspberry Pi alongside OpenClaw. Installable on mobile via PWA. Keep it plain and functional — design comes later once data is flowing. This is low priority — build it last.

## Architecture

Data flows through MCP (Model Context Protocol) servers. Each MCP server wraps one data source or service behind a standardized tool interface that any agent can call.

```
External APIs ──► MCP Servers ──► Agents (Ollama / Claude API)
                      │                    │
                      ▼                    ▼
                  Local Cache        Predictions DB
                                         │
                                         ▼
                                    Vector Store (ChromaDB)
                                         │
                                         ▼
                                      Dashboard
```

### MCP Servers to Build (in order)

1. **market-data-mcp** — Wraps Alpha Vantage (prices, fundamentals, technicals) and Finnhub (earnings transcripts). Tools: `get_price_history`, `get_quote`, `get_company_info`, `search_tickers`, `get_earnings_transcript`. Caches in local SQLite.
2. **sec-edgar-mcp** — Wraps SEC EDGAR API. Tools: `get_filings`, `get_filing_text`, `get_insider_transactions`, `search_filings`. Downloads and caches full filing text.
3. **predictions-db-mcp** — Wraps the prediction tracking database. Tools: `log_prediction`, `score_prediction`, `get_prediction_history`, `get_agent_accuracy`.
4. **vector-store-mcp** — Wraps ChromaDB. Tools: `store_embedding`, `search_similar`, `get_document`. Stores transcripts, filings, prediction reasoning.
5. **news-feed-mcp** — Wraps Alpha Vantage news sentiment and RSS feeds. Tools: `get_news`, `search_news`, `get_sentiment_summary`.
6. **dashboard-mcp** — Read-only HTTP API for the dashboard. Tools: `get_active_signals`, `get_watchlist`, `get_agent_status`. Uses HTTP transport (not stdio) so the Lit frontend can connect.

All other MCP servers use stdio transport (local only, no network exposure).

### Agent Design (build after MCP servers work)

Every agent follows the same structure:
- Receives structured input (Pydantic model)
- Produces structured output (Pydantic model) — never free-text only
- Logs full reasoning to the prediction database
- Has escalation criteria (when to call Claude API instead of local Ollama)
- Has a config file for trigger schedule, thresholds, model selection
- Targets < 30 second inference on RTX 4090 for local runs

Agent orchestration is handled by OpenClaw, running on the Raspberry Pi.

## Tech Stack

### Backend (Python)
- **Python 3.11+** with venv — primary language for MCP servers, agents, and data pipeline
- **Ollama** — local LLM runtime
- **Anthropic Claude API** — complex analysis
- **FastMCP** — MCP server framework
- **Pydantic** — all data models and schemas
- **httpx** — async HTTP client
- **SQLite** — database (upgrade to PostgreSQL later if needed)
- **ChromaDB** — vector store (embedded, no separate server process)
- **loguru** — logging
- **pyyaml** — config files
- **pytest + pytest-asyncio** — testing
- **black** — code formatting
- **ruff** — linting

### Frontend (JavaScript/TypeScript)
- **Node.js** — runtime for the dashboard
- **Lit** — web components for all UI
- **Sass** — styling
- **Vite** — build tooling and dev server
- **TypeScript** — type safety
- Tabs for indentation
- **PWA** — installable on mobile, works offline for cached data
- Modular, extendable architecture — easy to add new views/widgets without touching existing code
- The dashboard is a standalone app that connects to `dashboard-mcp` over HTTP
- Keep UI plain and functional until data is flowing, then design later

## Code Conventions

Follow Google style guides for both Python and TypeScript.

### Python (Google Python Style Guide)
- Type hints required everywhere. Use `from __future__ import annotations`.
- Async for I/O-bound operations (API calls, DB queries). Sync for CPU-bound (local LLM inference).
- All config in YAML files. Secrets via environment variables, never in config. Use `${ENV_VAR}` pattern for env var substitution.
- Google-style docstrings.
- PEP 8, max line length 100.
- Specific exception handling, never bare `except Exception: pass`.
- Structured logging with loguru, always include context (ticker, agent name).
- All inputs/outputs are Pydantic BaseModel with Field() descriptions.
- Imports: one per line, grouped (stdlib → third-party → local), alphabetized within groups.
- Naming: `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- No mutable default arguments.
- Use list/dict/set comprehensions over `map()`/`filter()` where clearer.

### TypeScript/Lit (Google TypeScript Style Guide)
- `camelCase` for variables/functions, `PascalCase` for classes/types/interfaces/components.
- Explicit return types on public functions and methods.
- No `any` — use `unknown` if the type is truly unknown.
- `const` by default, `let` when reassignment is needed, never `var`.
- Import from the actual module, not barrel exports (no `index.ts` re-exports).
- Lit components: prefix custom elements with `sr-` (e.g. `<sr-signal-card>`).
- JSDoc comments on public APIs.
- Tabs for indentation.
- Sass: use partials (`_variables.scss`, `_mixins.scss`), BEM naming for classes.

## Build Order

Phase 1 — Foundation (start here):
1. Project scaffolding (repo, pyproject.toml, shared utilities, config loader)
2. `market-data-mcp` server (get price data flowing)
3. `sec-edgar-mcp` server (get filings flowing)
4. `predictions-db-mcp` server (database schema, logging, scoring)
5. Basic data ingestion pipeline that fetches and caches data on a schedule

**Between phases:** Review the full codebase and clean up any old, unused, or dead code and files before starting the next phase.

Phase 2 — Analysis:
6. `vector-store-mcp` server
7. First agent: Earnings Linguist
8. Prediction scoring loop (compare predictions to actual price movement)
9. `news-feed-mcp` server

Phase 3 — Intelligence:
10. Additional agents (contagion mapper, narrative divergence, filing patterns)
11. Agent orchestration via OpenClaw on the Pi
12. Dashboard (Lit + TypeScript PWA connecting to dashboard-mcp over HTTP)

## Key Design Principles

- **Local-first**: Minimize cloud dependencies. Only the Claude API and data feed APIs go outbound.
- **Learning loop**: Every prediction is tracked, scored, and fed back to improve future analysis.
- **Escalation model**: Local LLMs handle routine work. Claude API handles complex multi-document synthesis.
- **No alert fatigue**: Intelligent filtering of what gets surfaced. Not everything is a signal.
- **Cache aggressively**: External API calls are expensive and rate-limited. Fetch once, store locally.
