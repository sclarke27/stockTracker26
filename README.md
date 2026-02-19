# Stock Radar

AI-powered stock market analysis system that uses LLMs to identify tradeable
patterns from unstructured data (earnings transcripts, SEC filings, news).
The edge is language analysis, not numerical time series.

---

## Hardware

Two machines on the same local network:

- **Desktop (RTX 4090)** -- runs Ollama for local LLM inference.
- **Raspberry Pi 5** -- runs OpenClaw, the orchestrator, dashboard API, and dashboard UI.

Only outbound connections are API calls to Alpha Vantage, Finnhub, SEC EDGAR,
and Anthropic.

---

## Prerequisites

- Python 3.11+
- Node.js 22+ and pnpm
- Ollama installed on the desktop (https://ollama.ai)
- OpenClaw installed on the Pi (https://openclaw.ai)

---

## 1. Clone and install

```bash
git clone <repo-url> stock-radar
cd stock-radar

# Python backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Dashboard frontend
cd dashboard
pnpm install
pnpm run build
cd ..
```

## 2. Environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Required variables:

| Variable              | Description                                    | Where to get it                                  |
|-----------------------|------------------------------------------------|--------------------------------------------------|
| ALPHA_VANTAGE_API_KEY | Market data, prices, fundamentals, news        | https://www.alphavantage.co/support/#api-key     |
| FINNHUB_API_KEY       | Earnings call transcripts                      | https://finnhub.io/register                      |
| SEC_EDGAR_EMAIL       | Contact email for SEC EDGAR User-Agent header  | Use your own email                               |

At least one of these cloud LLM keys is needed for escalation:

| Variable              | Description                                    | Where to get it                                  |
|-----------------------|------------------------------------------------|--------------------------------------------------|
| ANTHROPIC_API_KEY     | Claude API -- top-tier escalation              | https://console.anthropic.com/                   |
| OPENAI_API_KEY        | OpenAI API -- mid-tier escalation              | https://platform.openai.com/api-keys             |

Optional:

| Variable    | Default                    | Description              |
|-------------|----------------------------|--------------------------|
| OLLAMA_HOST | http://localhost:11434     | Ollama server address    |

## 3. Ollama setup (desktop)

Install the models the agents use:

```bash
ollama pull llama3.1:8b
ollama pull llama3.1:70b
```

If the Pi needs to reach Ollama over the network, start Ollama bound to
all interfaces and set `OLLAMA_HOST` in `.env` to the desktop's IP:

```bash
# On the desktop
OLLAMA_HOST=0.0.0.0 ollama serve

# In .env on the Pi
OLLAMA_HOST=http://192.168.x.x:11434
```

Also update `config/default.yaml` to match:

```yaml
ollama:
  host: "http://192.168.x.x:11434"
```

## 4. Configuration

All config lives in `config/`. Defaults work out of the box.

### config/default.yaml

Main settings: API keys (loaded from env vars), Ollama connection, database
paths, agent parameters, scoring, and orchestrator toggles.

Key sections you might want to adjust:

```yaml
# Disable an agent
agents:
  contagion_mapper:
    enabled: false

# Skip phases during a cycle
orchestrator:
  skip_ingestion: true   # use stale data
  skip_scoring: true     # skip prediction scoring
```

### config/watchlist.yaml

Tickers to track, organized by coverage tier:

- **deep** (10 tickers) -- all 4 agents run against these. Full data
  ingestion: price history, quotes, company info, earnings transcripts,
  SEC filings, insider transactions.
- **light** (10 tickers) -- only narrative divergence and SEC filing
  analyzer. Lighter data fetch: quotes and filings only.

Edit this file to change which stocks are tracked.

### config/contagion_pairs.yaml

Explicit ticker relationships for the contagion mapper agent. Each pair
defines a trigger and target with a relationship type (supplier, customer,
competitor, same_sector, distribution_partner).

---

## 5. Running the system

### Full orchestrator cycle (recommended)

Runs all three phases in sequence: data ingestion, agent analysis, prediction
scoring. Designed to be triggered by cron.

```bash
source .venv/bin/activate
stock-radar-orchestrate
```

Exit code 0 on success, 1 if any errors occurred.

### Individual components

Each phase can also be run independently:

```bash
# Data ingestion only
stock-radar-ingest

# Individual agents
stock-radar-earnings-linguist
stock-radar-narrative-divergence
stock-radar-sec-filing-analyzer
stock-radar-contagion-mapper

# Score mature predictions
stock-radar-score
```

### Dashboard

Start the dashboard API server:

```bash
stock-radar-dashboard-mcp
```

This starts an HTTP server on port 8080. The dashboard frontend connects
to it.

Serve the built frontend:

```bash
cd dashboard
pnpm run preview
```

The dashboard is a PWA -- once loaded in a browser, it can be installed
on mobile devices.

For development with hot reload:

```bash
cd dashboard
pnpm run dev
```

Dev server runs on port 5173.

---

## 6. OpenClaw setup (Pi)

OpenClaw handles scheduling, monitoring, and notifications for Stock Radar.
It runs on the Raspberry Pi and triggers the orchestrator on a cron schedule.

### Install OpenClaw

```bash
npm install -g openclaw@latest
openclaw onboard --install-daemon
```

### Configure OpenClaw

Copy the reference config files from this repo to your OpenClaw directory:

```bash
# Gateway config
cp openclaw/openclaw.json ~/.openclaw/openclaw.json

# Cron jobs
cp openclaw/cron-jobs.json ~/.openclaw/cron/jobs.json

# Stock Radar skill
cp -r openclaw/skills/stock-radar ~/.openclaw/skills/
```

Edit `~/.openclaw/openclaw.json` and update:

- `agents.defaults.workspace` -- path to your stock-radar checkout
- `models.providers.ollama-desktop.baseUrl` -- your desktop's IP address
- `channels` -- uncomment and configure Telegram/Slack/etc. for notifications

### Install the MCP adapter plugin

This lets OpenClaw query the dashboard API directly:

```bash
openclaw plugins install mcp-adapter
openclaw gateway restart
```

### Verify cron jobs

```bash
openclaw cron list
```

You should see two jobs:

- **Stock Radar -- Daily Cycle** -- runs at 6:00 AM Eastern, weekdays
- **Stock Radar -- Score Predictions** -- runs at 5:00 PM Eastern, weekdays

Test the daily cycle immediately:

```bash
openclaw cron run "Stock Radar — Daily Cycle" --now
```

### Notifications

To receive cycle results via Telegram, edit `~/.openclaw/openclaw.json`:

```json5
"channels": {
  "telegram": {
    "enabled": true,
    "dmPolicy": "pairing"
  }
}
```

Then update the cron job delivery to target Telegram:

```bash
openclaw cron edit <job-id> --announce --channel telegram --to "<chat-id>"
```

### Dashboard API (systemd)

The dashboard API should run as a persistent service, independent of OpenClaw:

```ini
[Unit]
Description=Stock Radar Dashboard API
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/stock-radar
ExecStart=/path/to/stock-radar/.venv/bin/stock-radar-dashboard-mcp
Restart=always
EnvironmentFile=/path/to/stock-radar/.env

[Install]
WantedBy=multi-user.target
```

---

## 7. Data storage

All data is stored locally. The `data/` directory is created automatically
on first run:

| Path                  | Contents                                       |
|-----------------------|------------------------------------------------|
| data/stock_radar.db   | SQLite cache for market data and SEC filings   |
| data/predictions.db   | Prediction tracking and scoring results        |
| data/chroma_data/     | ChromaDB vector embeddings for semantic search |

These paths are configurable in `config/default.yaml`.

---

## 8. Architecture

```
External APIs --> MCP Servers --> Agents (Ollama / Claude API)
                      |                    |
                      v                    v
                  Local Cache        Predictions DB
                                         |
                                         v
                                    Vector Store (ChromaDB)
                                         |
                                         v
                                      Dashboard
```

### MCP Servers

Each data source is wrapped in an MCP server with a standardized tool
interface:

1. market-data-mcp -- Alpha Vantage prices/quotes + Finnhub transcripts
2. sec-edgar-mcp -- SEC EDGAR filings and insider transactions
3. predictions-db-mcp -- prediction logging and scoring
4. vector-store-mcp -- ChromaDB semantic search
5. news-feed-mcp -- Alpha Vantage news sentiment + RSS feeds
6. dashboard-mcp -- read-only HTTP API for the dashboard (port 8080)

Servers 1-5 use stdio transport (in-process). Server 6 uses HTTP.

### Analysis Agents

| Agent                | Signal Type              | Horizon | Coverage     |
|----------------------|--------------------------|---------|--------------|
| Earnings Linguist    | Earnings sentiment       | 5 days  | Deep only    |
| Narrative Divergence | Narrative vs price gap   | 10 days | Deep + Light |
| SEC Filing Analyzer  | Filing pattern anomalies | 15 days | Deep + Light |
| Contagion Mapper     | Cross-sector contagion   | 5 days  | Deep only    |

Each agent follows a tiered escalation model: Ollama (local, fast) for
routine analysis, OpenAI for mid-tier escalation, Claude API for the most
complex multi-document synthesis. Configure one or both cloud providers.
If only one is configured, it handles all escalation.

### Orchestrator cycle

```
OpenClaw (cron schedule) --> stock-radar-orchestrate
                                Phase 1: Ingestion    -- fetch market data, filings, transcripts
                                Phase 2: Analysis     -- run agents against deep and light tickers
                                Phase 3: Scoring      -- score mature predictions against actual prices
```

OpenClaw triggers the orchestrator on a cron schedule (weekdays at 6 AM
Eastern). Each phase is fault-tolerant. A failure in one does not block
the next.

---

## 9. Development

```bash
# Run all tests (707 tests)
pytest

# Run a specific test file
pytest tests/orchestrator/test_cycle.py

# Formatting
black .
black . --check

# Linting
ruff check .
ruff check . --fix

# Frontend
cd dashboard
pnpm run typecheck
pnpm run lint
```

---

## 10. Project layout

```
config/
  default.yaml            Main configuration
  watchlist.yaml           Tracked tickers by coverage tier
  contagion_pairs.yaml     Cross-sector ticker relationships

openclaw/
  openclaw.json            Reference OpenClaw gateway config (copy to ~/.openclaw/)
  cron-jobs.json           Cron job definitions (copy to ~/.openclaw/cron/jobs.json)
  skills/stock-radar/      OpenClaw skill for Stock Radar interaction
    SKILL.md               Skill definition and instructions

src/stock_radar/
  agents/                  4 analysis agents (earnings, narrative, SEC, contagion)
  config/                  Settings models and YAML loader
  llm/                     LLM client abstraction (Ollama + Anthropic)
  mcp_servers/             6 MCP servers (market data, SEC, predictions, vectors, news, dashboard)
  orchestrator/            Cron-triggered analysis cycle
  pipeline/                Data ingestion pipeline and watchlist
  scoring/                 Prediction scoring loop
  utils/                   Caching, logging, rate limiting

dashboard/
  src/components/          Lit web components (sr-app, sr-signal-card, etc.)
  src/api/                 Dashboard API client (JSON-RPC over HTTP)
  src/styles/              Sass partials (variables, mixins, reset)

tests/                     707 unit tests mirroring src/ structure
```
