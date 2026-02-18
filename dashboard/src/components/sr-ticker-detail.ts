/**
 * sr-ticker-detail — per-ticker deep-dive view.
 */

import { LitElement, css, html } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { getTickerSummary } from '../api/client';
import type { TickerSummaryResponse } from '../api/types';

@customElement('sr-ticker-detail')
export class SrTickerDetail extends LitElement {
	static override styles = css`
		:host {
			display: block;
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

		.header {
			display: flex;
			align-items: flex-start;
			justify-content: space-between;
			gap: 1rem;
			margin-bottom: 1.5rem;
			flex-wrap: wrap;
		}

		.ticker {
			font-family: var(--sr-font-mono, monospace);
			font-size: 1.5rem;
			font-weight: 700;
			color: var(--sr-accent, #4f8ef7);
		}

		.company-name {
			font-size: 1rem;
			color: var(--sr-text-muted, #8892a4);
			margin-top: 0.25rem;
		}

		.price {
			font-family: var(--sr-font-mono, monospace);
			font-size: 1.5rem;
			font-weight: 700;
			color: var(--sr-text, #e2e8f0);
		}

		.sentiment {
			font-size: 0.875rem;
			color: var(--sr-text-muted, #8892a4);
			margin-top: 0.25rem;
		}

		.section {
			margin-top: 1.5rem;
		}

		.section-title {
			font-size: 0.875rem;
			font-weight: 600;
			color: var(--sr-text-muted, #8892a4);
			text-transform: uppercase;
			letter-spacing: 0.05em;
			margin-bottom: 0.75rem;
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

		tr:last-child td { border-bottom: none; }

		.direction--BULLISH { color: #22c55e; }
		.direction--BEARISH { color: #ef4444; }
		.direction--NEUTRAL { color: #94a3b8; }

		.back-link {
			display: inline-flex;
			align-items: center;
			gap: 0.25rem;
			font-size: 0.875rem;
			color: var(--sr-accent, #4f8ef7);
			margin-bottom: 1.5rem;
			background: none;
			border: none;
			padding: 0;
			cursor: pointer;
		}

		.back-link:hover {
			text-decoration: underline;
		}
	`;

	@property({ type: String })
	ticker = '';

	@state() private _data: TickerSummaryResponse | null = null;
	@state() private _loading = true;
	@state() private _error: string | null = null;

	override updated(changed: Map<string, unknown>): void {
		if (changed.has('ticker') && this.ticker) {
			void this._load();
		}
	}

	private async _load(): Promise<void> {
		this._loading = true;
		this._error = null;
		this._data = null;
		try {
			this._data = await getTickerSummary(this.ticker);
		} catch (err) {
			this._error = err instanceof Error ? err.message : 'Failed to load ticker summary';
		} finally {
			this._loading = false;
		}
	}

	override render() {
		if (this._loading) return html`<div class="loading">Loading ${this.ticker}…</div>`;
		if (this._error) return html`<div class="error">${this._error}</div>`;
		if (!this._data) return html``;

		const d = this._data;
		return html`
			<button class="back-link" @click=${() => { window.location.hash = '#/'; }}>
				← Back to Signals
			</button>

			<div class="header">
				<div>
					<div class="ticker">${d.ticker}</div>
					<div class="company-name">${d.company_name}</div>
				</div>
				<div>
					${d.current_price != null
						? html`<div class="price">$${d.current_price.toFixed(2)}</div>`
						: html``}
					${d.sentiment_score != null
						? html`<div class="sentiment">
								Sentiment: ${d.sentiment_label ?? ''} (${d.sentiment_score.toFixed(2)})
							</div>`
						: html``}
				</div>
			</div>

			${d.recent_predictions.length > 0
				? html`
					<div class="section">
						<div class="section-title">Recent Predictions</div>
						<div class="table-wrap">
							<table>
								<thead>
									<tr>
										<th>Date</th>
										<th>Agent</th>
										<th>Direction</th>
										<th>Confidence</th>
										<th>Horizon</th>
									</tr>
								</thead>
								<tbody>
									${d.recent_predictions.map((p) => {
										const direction = String(p['direction'] ?? '');
										return html`
											<tr>
												<td>${String(p['prediction_date'] ?? '')}</td>
												<td>${String(p['agent_name'] ?? '')}</td>
												<td class="direction--${direction}">${direction}</td>
												<td>${p['confidence'] != null ? `${Math.round(Number(p['confidence']) * 100)}%` : '—'}</td>
												<td>${String(p['horizon_days'] ?? '')}d</td>
											</tr>
										`;
									})}
								</tbody>
							</table>
						</div>
					</div>
				`
				: html`<div class="empty">No predictions for ${d.ticker}.</div>`}
		`;
	}
}

declare global {
	interface HTMLElementTagNameMap {
		'sr-ticker-detail': SrTickerDetail;
	}
}
