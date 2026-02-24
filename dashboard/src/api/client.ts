/**
 * Typed API client for dashboard-mcp.
 *
 * Communicates with the FastMCP HTTP server using JSON-RPC 2.0 over the
 * streamable-http transport endpoint. Manages the MCP session lifecycle:
 * sends an `initialize` handshake on first use, captures the session ID,
 * and includes it on all subsequent tool calls.
 */

import type {
	ActiveSignalsResponse,
	AgentStatusResponse,
	PredictionDetailResponse,
	TickerSummaryResponse,
	WatchlistResponse,
} from "./types";

const MCP_ENDPOINT =
	(import.meta.env["VITE_DASHBOARD_URL"] as string | undefined) ??
	`http://${window.location.hostname}:8082/mcp`;

const MCP_HEADERS: Record<string, string> = {
	"Content-Type": "application/json",
	Accept: "application/json, text/event-stream",
};

let _requestId = 1;
let _sessionId: string | null = null;
let _initPromise: Promise<void> | null = null;

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

/**
 * Parse a Server-Sent Events text body and return the last JSON-RPC message.
 *
 * SSE format: lines of `event: <type>\ndata: <json>\n\n`.
 * We collect all `data:` lines and parse the last complete one.
 */
function parseSseJson(text: string): Record<string, unknown> {
	const dataLines: string[] = [];
	for (const line of text.split("\n")) {
		if (line.startsWith("data: ")) {
			dataLines.push(line.slice(6));
		}
	}
	if (dataLines.length === 0) {
		throw new Error("No data lines found in SSE response");
	}
	return JSON.parse(dataLines[dataLines.length - 1]) as Record<string, unknown>;
}

/** Send a JSON-RPC request to the MCP endpoint and return the parsed body. */
async function mcpPost(body: Record<string, unknown>): Promise<{
	json: Record<string, unknown>;
	headers: Headers;
}> {
	const headers: Record<string, string> = { ...MCP_HEADERS };
	if (_sessionId) {
		headers["Mcp-Session-Id"] = _sessionId;
	}

	const res = await fetch(MCP_ENDPOINT, {
		method: "POST",
		headers,
		body: JSON.stringify(body),
	});

	if (!res.ok) {
		const text = await res.text().catch(() => "");
		throw new Error(`MCP error ${res.status}: ${text}`);
	}

	const contentType = res.headers.get("content-type") ?? "";
	let json: Record<string, unknown>;

	if (contentType.includes("text/event-stream")) {
		const text = await res.text();
		json = parseSseJson(text);
	} else {
		json = (await res.json()) as Record<string, unknown>;
	}

	return { json, headers: res.headers };
}

/**
 * Perform the MCP initialize + initialized handshake.
 *
 * Called lazily on the first tool call. Captures the session ID from the
 * response headers for use in all subsequent requests.
 */
async function initialize(): Promise<void> {
	// Step 1: initialize — server returns capabilities + session ID header
	const { json, headers } = await mcpPost({
		jsonrpc: "2.0",
		id: _requestId++,
		method: "initialize",
		params: {
			protocolVersion: "2025-03-26",
			capabilities: {},
			clientInfo: { name: "stock-radar-dashboard", version: "1.0.0" },
		},
	});

	const error = json["error"] as { message?: string } | undefined;
	if (error) {
		throw new Error(`MCP initialize failed: ${error.message ?? "unknown"}`);
	}

	_sessionId = headers.get("mcp-session-id");
	if (!_sessionId) {
		throw new Error("Server did not return Mcp-Session-Id header");
	}

	// Step 2: initialized notification — no response expected
	await mcpPost({
		jsonrpc: "2.0",
		method: "notifications/initialized",
	});
}

/** Ensure the session is initialized (idempotent, single-flight). */
async function ensureSession(): Promise<void> {
	if (_sessionId) return;
	if (!_initPromise) {
		_initPromise = initialize().catch((err) => {
			_initPromise = null;
			throw err;
		});
	}
	await _initPromise;
}

// ---------------------------------------------------------------------------
// Tool calling
// ---------------------------------------------------------------------------

/**
 * Call a tool on the dashboard-mcp server.
 *
 * Automatically initializes the MCP session on the first call.
 *
 * @param tool - Tool name as registered on the server.
 * @param args - Tool arguments dict.
 * @returns Parsed JSON response from the tool.
 */
async function callTool<T>(tool: string, args: Record<string, unknown> = {}): Promise<T> {
	await ensureSession();

	const { json } = await mcpPost({
		jsonrpc: "2.0",
		id: _requestId++,
		method: "tools/call",
		params: { name: tool, arguments: args },
	});

	const error = json["error"] as { message?: string } | undefined;
	if (error) {
		throw new Error(`Tool error: ${error.message ?? "unknown"}`);
	}

	const result = json["result"] as { content?: Array<{ text?: string }> } | undefined;
	const text = result?.content?.[0]?.text;
	if (!text) {
		throw new Error("Empty response from dashboard-mcp");
	}

	return JSON.parse(text) as T;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Fetch active prediction signals above the confidence threshold. */
export async function getActiveSignals(limit?: number): Promise<ActiveSignalsResponse> {
	const args: Record<string, unknown> = {};
	if (limit !== undefined) args["limit"] = limit;
	return callTool<ActiveSignalsResponse>("get_active_signals", args);
}

/** Fetch the watchlist of tickers with active predictions. */
export async function getWatchlist(): Promise<WatchlistResponse> {
	return callTool<WatchlistResponse>("get_watchlist");
}

/** Fetch performance stats for all analysis agents. */
export async function getAgentStatus(): Promise<AgentStatusResponse> {
	return callTool<AgentStatusResponse>("get_agent_status");
}

/** Fetch full detail for a single prediction. */
export async function getPredictionDetail(predictionId: string): Promise<PredictionDetailResponse> {
	return callTool<PredictionDetailResponse>("get_prediction_detail", {
		prediction_id: predictionId,
	});
}

/** Fetch aggregated summary for a single ticker. */
export async function getTickerSummary(ticker: string): Promise<TickerSummaryResponse> {
	return callTool<TickerSummaryResponse>("get_ticker_summary", { ticker });
}
