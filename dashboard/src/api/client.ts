/**
 * Typed API client for dashboard-mcp.
 *
 * Communicates with the FastMCP HTTP server using JSON-RPC 2.0 over the
 * streamable-http transport endpoint. Each call serialises to a tools/call
 * request and deserialises the text payload from the first content item.
 */

import type {
	ActiveSignalsResponse,
	AgentStatusResponse,
	PredictionDetailResponse,
	TickerSummaryResponse,
	WatchlistResponse,
} from "./types";

const BASE_URL = (import.meta.env["VITE_DASHBOARD_URL"] as string | undefined) ?? "http://localhost:8081";

let _requestId = 1;

/**
 * Call a tool on the dashboard-mcp server.
 *
 * @param tool - Tool name as registered on the server.
 * @param args - Tool arguments dict.
 * @returns Parsed JSON response from the tool.
 */
async function callTool<T>(tool: string, args: Record<string, unknown> = {}): Promise<T> {
	const res = await fetch(`${BASE_URL}/mcp/`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			jsonrpc: "2.0",
			id: _requestId++,
			method: "tools/call",
			params: { name: tool, arguments: args },
		}),
	});

	if (!res.ok) {
		throw new Error(`Dashboard API error: ${res.status} ${res.statusText}`);
	}

	const envelope = (await res.json()) as {
		result?: { content?: Array<{ text?: string }> };
		error?: { message?: string };
	};

	if (envelope.error) {
		throw new Error(`Tool error: ${envelope.error.message ?? "unknown"}`);
	}

	const text = envelope.result?.content?.[0]?.text;
	if (!text) {
		throw new Error("Empty response from dashboard-mcp");
	}

	return JSON.parse(text) as T;
}

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
