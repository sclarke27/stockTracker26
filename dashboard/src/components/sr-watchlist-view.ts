/**
 * sr-watchlist-view — table of tickers with active predictions.
 */

import { LitElement, css, html } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { getWatchlist } from '../api/client';
import type { WatchlistEntry } from '../api/types';

@customElement('sr-watchlist-view')
export class SrWatchlistView extends LitElement {
	static override styles = css`
		:host {
			display: block;
		}

		.title {
			font-size: 1.125rem;
			font-weight: 600;
			margin-bottom: 1.5rem;
			color: var(--sr-text, #e2e8f0);
		}

		.loading,
		.empty,
		.error {
			padding: 3rem;
			text-align: center;
			color: var(--sr-text-muted, #8892a4);
			font-size: 0.875rem;
		}

		.error {
			color: #ef4444;
			background: rgba(239, 68, 68, 0.08);
			border: 1px solid rgba(239, 68, 68, 0.2);
			border-radius: 8px;
		}

		.table-wrap {
			background: var(--sr-surface, #1a1d27);
			border: 1px solid var(--sr-border, #2d3150);
			border-radius: 8px;
			overflow: hidden;
		}

		table {
			width: 100%;
			border-collapse: collapse;
			font-size: 0.875rem;
		}

		th {
			padding: 0.75rem 1rem;
			text-align: left;
			color: var(--sr-text-muted, #8892a4);
			font-weight: 500;
			border-bottom: 1px solid var(--sr-border, #2d3150);
		}

		td {
			padding: 0.75rem 1rem;
			border-bottom: 1px solid rgba(45, 49, 80, 0.5);
			color: var(--sr-text, #e2e8f0);
		}

		tr:last-child td {
			border-bottom: none;
		}

		tr:hover td {
			background: rgba(34, 38, 58, 0.5);
			cursor: pointer;
		}

		.ticker {
			font-family: var(--sr-font-mono, monospace);
			font-size: 0.875rem;
			font-weight: 700;
			background: rgba(79, 142, 247, 0.15);
			color: var(--sr-accent, #4f8ef7);
			border: 1px solid rgba(79, 142, 247, 0.3);
			border-radius: 4px;
			padding: 0.125rem 0.5rem;
		}

		.direction--BULLISH { color: #22c55e; }
		.direction--BEARISH { color: #ef4444; }
		.direction--NEUTRAL { color: #94a3b8; }

		.price {
			font-family: var(--sr-font-mono, monospace);
		}

		.signal-count {
			font-weight: 600;
			color: var(--sr-accent, #4f8ef7);
		}
	`;

	@state() private _entries: WatchlistEntry[] = [];
	@state() private _loading = true;
	@state() private _error: string | null = null;

	override connectedCallback(): void {
		super.connectedCallback();
		void this._load();
	}

	private async _load(): Promise<void> {
		this._loading = true;
		this._error = null;
		try {
			const res = await getWatchlist();
			this._entries = res.entries;
		} catch (err) {
			this._error = err instanceof Error ? err.message : 'Failed to load watchlist';
		} finally {
			this._loading = false;
		}
	}

	private _onRowClick(ticker: string): void {
		window.location.hash = `#/ticker/${ticker}`;
	}

	override render() {
		if (this._loading) return html`<div class="loading">Loading watchlist…</div>`;
		if (this._error) return html`<div class="error">${this._error}</div>`;

		return html`
			<h2 class="title">Watchlist</h2>
			${this._entries.length === 0
				? html`<div class="empty">No tickers with active predictions.</div>`
				: html`
					<div class="table-wrap">
						<table>
							<thead>
								<tr>
									<th>Ticker</th>
									<th>Company</th>
									<th>Price</th>
									<th>Signals</th>
									<th>Direction</th>
								</tr>
							</thead>
							<tbody>
								${this._entries.map(
									(e) => html`
										<tr @click=${() => this._onRowClick(e.ticker)}>
											<td><span class="ticker">${e.ticker}</span></td>
											<td>${e.company_name}</td>
											<td class="price">${e.current_price != null ? `$${e.current_price.toFixed(2)}` : '—'}</td>
											<td><span class="signal-count">${e.active_signal_count}</span></td>
											<td>
												${e.latest_direction
													? html`<span class="direction--${e.latest_direction}">${e.latest_direction}</span>`
													: html`—`}
											</td>
										</tr>
									`,
								)}
							</tbody>
						</table>
					</div>
				`}
		`;
	}
}

declare global {
	interface HTMLElementTagNameMap {
		'sr-watchlist-view': SrWatchlistView;
	}
}
