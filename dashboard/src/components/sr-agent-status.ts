/**
 * sr-agent-status — agent performance table.
 */

import { LitElement, css, html } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { getAgentStatus } from '../api/client';
import type { AgentStatus } from '../api/types';

@customElement('sr-agent-status')
export class SrAgentStatus extends LitElement {
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

		.accuracy-high { color: #22c55e; font-weight: 600; }
		.accuracy-mid  { color: #f59e0b; font-weight: 600; }
		.accuracy-low  { color: #ef4444; font-weight: 600; }

		.agent-name {
			font-weight: 500;
		}

		.signal-type {
			font-size: 0.75rem;
			color: var(--sr-text-muted, #8892a4);
		}

		.lookback {
			font-size: 0.8125rem;
			color: var(--sr-text-muted, #8892a4);
			margin-top: 0.75rem;
		}
	`;

	@state() private _agents: AgentStatus[] = [];
	@state() private _asOfDays = 90;
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
			const res = await getAgentStatus();
			this._agents = res.agents;
			this._asOfDays = res.as_of_days;
		} catch (err) {
			this._error = err instanceof Error ? err.message : 'Failed to load agent status';
		} finally {
			this._loading = false;
		}
	}

	private _accuracyClass(pct: number | null): string {
		if (pct === null) return '';
		if (pct >= 60) return 'accuracy-high';
		if (pct >= 40) return 'accuracy-mid';
		return 'accuracy-low';
	}

	override render() {
		if (this._loading) return html`<div class="loading">Loading agent status…</div>`;
		if (this._error) return html`<div class="error">${this._error}</div>`;

		return html`
			<h2 class="title">Agent Performance</h2>
			${this._agents.length === 0
				? html`<div class="empty">No agent data available.</div>`
				: html`
					<div class="table-wrap">
						<table>
							<thead>
								<tr>
									<th>Agent</th>
									<th>Total</th>
									<th>Scored</th>
									<th>Accuracy</th>
									<th>Avg Confidence</th>
								</tr>
							</thead>
							<tbody>
								${this._agents.map(
									(a) => html`
										<tr>
											<td>
												<div class="agent-name">${a.agent_name}</div>
												<div class="signal-type">${a.signal_type}</div>
											</td>
											<td>${a.total_predictions}</td>
											<td>${a.scored}</td>
											<td class=${this._accuracyClass(a.accuracy_pct)}>
												${a.accuracy_pct != null ? `${a.accuracy_pct.toFixed(1)}%` : '—'}
											</td>
											<td>${a.avg_confidence != null ? `${(a.avg_confidence * 100).toFixed(0)}%` : '—'}</td>
										</tr>
									`,
								)}
							</tbody>
						</table>
					</div>
					<p class="lookback">Last ${this._asOfDays} days</p>
				`}
		`;
	}
}

declare global {
	interface HTMLElementTagNameMap {
		'sr-agent-status': SrAgentStatus;
	}
}
