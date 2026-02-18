/**
 * TypeScript interfaces matching dashboard-mcp server response models.
 *
 * Keep in sync with stock_radar/mcp_servers/dashboard/server.py models.
 */

export type Direction = 'BULLISH' | 'BEARISH' | 'NEUTRAL';

export interface SignalSummary {
	prediction_id: string;
	ticker: string;
	agent_name: string;
	signal_type: string;
	direction: Direction;
	confidence: number;
	reasoning: string;
	horizon_days: number;
	prediction_date: string;
	current_price: number | null;
}

export interface ActiveSignalsResponse {
	signals: SignalSummary[];
	total_count: number;
}

export interface WatchlistEntry {
	ticker: string;
	company_name: string;
	current_price: number | null;
	active_signal_count: number;
	latest_direction: Direction | null;
}

export interface WatchlistResponse {
	entries: WatchlistEntry[];
}

export interface AgentStatus {
	agent_name: string;
	signal_type: string;
	total_predictions: number;
	scored: number;
	accuracy_pct: number | null;
	avg_confidence: number | null;
}

export interface AgentStatusResponse {
	agents: AgentStatus[];
	as_of_days: number;
}

export interface PredictionDetailResponse {
	prediction: Record<string, unknown> | null;
	price_history: Array<Record<string, unknown>>;
}

export interface TickerSummaryResponse {
	ticker: string;
	company_name: string;
	current_price: number | null;
	sentiment_score: number | null;
	sentiment_label: string | null;
	recent_predictions: Array<Record<string, unknown>>;
}
