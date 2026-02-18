/**
 * sr-signals-view — grid of active prediction signals (home view).
 */

import { LitElement, css, html } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { getActiveSignals } from '../api/client';
import type { SignalSummary } from '../api/types';
import './sr-signal-card';

@customElement('sr-signals-view')
export class SrSignalsView extends LitElement {
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

		.grid {
			display: grid;
			grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
			gap: 1rem;
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

		.count {
			font-size: 0.875rem;
			color: var(--sr-text-muted, #8892a4);
			margin-bottom: 1rem;
		}
	`;

	@state() private _signals: SignalSummary[] = [];
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
			const res = await getActiveSignals();
			this._signals = res.signals;
		} catch (err) {
			this._error = err instanceof Error ? err.message : 'Failed to load signals';
		} finally {
			this._loading = false;
		}
	}

	override render() {
		if (this._loading) return html`<div class="loading">Loading signals…</div>`;
		if (this._error) return html`<div class="error">${this._error}</div>`;

		return html`
			<h2 class="title">Active Signals</h2>
			${this._signals.length === 0
				? html`<div class="empty">No active signals above the confidence threshold.</div>`
				: html`
					<p class="count">${this._signals.length} signal${this._signals.length !== 1 ? 's' : ''}</p>
					<div class="grid">
						${this._signals.map(
							(s) => html`<sr-signal-card .signal=${s}></sr-signal-card>`,
						)}
					</div>
				`}
		`;
	}
}

declare global {
	interface HTMLElementTagNameMap {
		'sr-signals-view': SrSignalsView;
	}
}
