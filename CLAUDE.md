# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Persona

When writing code, operate as a **senior software architect**. Prioritize clean architecture, maintainability, and clear separation of concerns. Make deliberate design decisions and be opinionated about code quality. **No shortcuts** — do everything the right way even if it takes longer. No hacks, no "we'll fix this later" compromises, no quick-and-dirty solutions. **No drive-by coding** — keep edits focused on the task at hand. Don't touch unrelated code, don't refactor things you weren't asked to change, don't "improve" nearby code while you're in a file for something else. **Test-driven development** — write tests first, then write the code to make them pass. **No monolithic files** — break code into small, focused modules organized logically. Each file should have a single clear responsibility. A human should be able to navigate the codebase and find things by intuition. **DRY** — no code duplication. Shared logic goes into base classes or shared utilities. Always follow general coding best practices (SOLID, meaningful names, proper error handling, separation of concerns).

## Project Overview

Stock Radar — an AI-powered stock market analysis system that uses LLMs to identify tradeable patterns from unstructured data (earnings transcripts, SEC filings, news). The edge is language analysis, not numerical time series. See `STOCK-RADAR-SPEC.md` for the full specification.

**Hardware:** Raspberry Pi 5 (orchestrator/scheduler/dashboard) + Desktop with RTX 4090 (Ollama LLM inference), both on a local network.

## Tech Stack

**Backend (Python):** Python 3.11+ with venv | FastMCP (MCP servers) | Ollama + Claude API (LLMs) | Pydantic (data models) | httpx (async HTTP) | SQLite (storage) | ChromaDB (vector store) | loguru (logging) | pyyaml (config) | pytest + pytest-asyncio (testing) | black (formatting) | ruff (linting)

**Data sources:** Alpha Vantage (prices, fundamentals, transcripts, IPOs, news sentiment) | SEC EDGAR (filings, insider transactions)

**Frontend (JavaScript/TypeScript):** Lit (web components) | Sass (styling) | Vite (build) | TypeScript | PWA — dashboard connects to `dashboard-mcp` over HTTP. Tabs for indentation.

## Build/Test/Lint Commands

```bash
# Backend (Python) — once pyproject.toml is set up:
pip install -e ".[dev]"      # Install with dev dependencies
pytest                        # Run all tests
pytest tests/test_foo.py      # Run single test file
pytest -k "test_name"         # Run single test by name
black . --check               # Check formatting
black .                       # Auto-format
ruff check .                  # Lint
ruff check . --fix            # Auto-fix lint issues

# Frontend (Lit) — from the dashboard/ directory:
npm install                   # Install dependencies
npm run dev                   # Dev server
npm run build                 # Production build
npm run lint                  # ESLint
npm run typecheck             # TypeScript checking
```

## Architecture

Data flows through MCP servers (Model Context Protocol). Each MCP server wraps one data source/service behind a standardized tool interface that agents call.

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

**6 MCP servers** (build in this order):
1. `market-data-mcp` — price history, quotes, company info (stdio transport)
2. `sec-edgar-mcp` — SEC EDGAR filings, insider transactions (stdio transport)
3. `predictions-db-mcp` — prediction logging and scoring (stdio transport)
4. `vector-store-mcp` — ChromaDB embeddings for transcripts/filings/reasoning (stdio transport)
5. `news-feed-mcp` — news API / RSS feeds (stdio transport)
6. `dashboard-mcp` — read-only HTTP API for the dashboard (**HTTP transport**, not stdio)

**Agent structure** (every agent follows this pattern):
- Structured Pydantic input → structured Pydantic output (never free-text only)
- Logs full reasoning to predictions DB
- Has escalation criteria (when to use Claude API vs local Ollama)
- Per-agent YAML config for schedule, thresholds, model selection
- Targets < 30s inference on RTX 4090

**4 analysis agents:** Earnings Linguist (priority 1), Cross-Sector Contagion Mapper, Narrative vs Price Divergence, SEC Filing Pattern Analyzer.

## Code Conventions

Follow Google style guides for both Python and TypeScript.

### Python
- Type hints everywhere with `from __future__ import annotations`
- Async for I/O (API calls, DB queries); sync for CPU-bound (local LLM inference)
- All config in YAML; secrets via environment variables (`${ENV_VAR}` pattern, never in config)
- Google-style docstrings
- PEP 8, max line length 100
- Specific exception handling — never bare `except Exception: pass`
- Structured logging with loguru — always include context (ticker, agent name)
- All inputs/outputs as `Pydantic BaseModel` with `Field()` descriptions
- Imports: one per line, grouped (stdlib → third-party → local), alphabetized
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- No mutable default arguments

### TypeScript/Lit
- `camelCase` variables/functions, `PascalCase` classes/types/components
- Explicit return types on public functions
- No `any` — use `unknown` if needed
- `const` by default, `let` when needed, never `var`
- Import from actual modules, not barrel exports
- Lit components prefixed with `sr-` (e.g. `<sr-signal-card>`)
- JSDoc on public APIs
- Tabs for indentation
- Sass: partials + BEM naming

## Build Phases

**Phase 1 — Foundation:** scaffolding (pyproject.toml, shared utils, config loader) → market-data-mcp → sec-edgar-mcp → predictions-db-mcp → data ingestion pipeline

**Between phases:** Review full codebase, clean up any old/unused/dead code and files.

**Phase 2 — Analysis:** vector-store-mcp → Earnings Linguist agent → prediction scoring loop → news-feed-mcp

**Phase 3 — Intelligence:** remaining agents → orchestration (OpenClaw on Pi) → dashboard (Lit + TypeScript, PWA)

## Design Principles

- **Local-first:** only Claude API and data feed APIs go outbound
- **Learning loop:** every prediction tracked, scored, fed back for future analysis
- **Escalation model:** Ollama for routine, Claude API for complex multi-document synthesis
- **Cache aggressively:** external API calls are expensive and rate-limited
- **No alert fatigue:** intelligent filtering — not everything is a signal
