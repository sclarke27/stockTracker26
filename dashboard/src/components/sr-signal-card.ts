/**
 * sr-signal-card — single prediction signal card.
 */

import { LitElement, css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import type { SignalSummary } from '../api/types';

/** Format confidence (0–1) as a percentage string. */
function fmtConfidence(c: number): string {
	return `${Math.round(c * 100)}%`;
}

/** Map confidence to a tier class suffix. */
function confidenceTier(c: number): string {
	if (c >= 0.7) return 'high';
	if (c >= 0.5) return 'mid';
	return 'low';
}

@customElement('sr-signal-card')
export class SrSignalCard extends LitElement {
	static override styles = css`
		:host {
			display: block;
		}

		.card {
			background: var(--sr-surface, #1a1d27);
			border: 1px solid var(--sr-border, #2d3150);
			border-radius: 8px;
			padding: 1rem;
			display: flex;
			flex-direction: column;
			gap: 0.75rem;
			transition: border-color 0.2s;
		}

		.card:hover {
			border-color: var(--sr-accent, #4f8ef7);
		}

		.header {
			display: flex;
			align-items: center;
			justify-content: space-between;
			gap: 0.5rem;
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

		.direction {
			font-size: 0.75rem;
			font-weight: 600;
			letter-spacing: 0.05em;
			text-transform: uppercase;
			border: 1px solid;
			border-radius: 4px;
			padding: 0.125rem 0.5rem;
		}

		.direction--BULLISH {
			color: #22c55e;
			border-color: rgba(34, 197, 94, 0.3);
			background: rgba(34, 197, 94, 0.1);
		}

		.direction--BEARISH {
			color: #ef4444;
			border-color: rgba(239, 68, 68, 0.3);
			background: rgba(239, 68, 68, 0.1);
		}

		.direction--NEUTRAL {
			color: #94a3b8;
			border-color: rgba(148, 163, 184, 0.3);
			background: rgba(148, 163, 184, 0.1);
		}

		.meta {
			display: flex;
			flex-direction: column;
			gap: 0.25rem;
			font-size: 0.75rem;
			color: var(--sr-text-muted, #8892a4);
		}

		.confidence-row {
			display: flex;
			align-items: center;
			gap: 0.5rem;
		}

		.confidence-label {
			font-size: 0.75rem;
			color: var(--sr-text-muted, #8892a4);
			width: 2.5rem;
			flex-shrink: 0;
		}

		.confidence-bar {
			flex: 1;
			height: 4px;
			background: var(--sr-border, #2d3150);
			border-radius: 2px;
			overflow: hidden;
		}

		.confidence-fill {
			height: 100%;
			border-radius: 2px;
			transition: width 0.3s;
		}

		.confidence-fill--high { background: #22c55e; }
		.confidence-fill--mid  { background: #f59e0b; }
		.confidence-fill--low  { background: #ef4444; }

		.reasoning {
			font-size: 0.8125rem;
			color: var(--sr-text, #e2e8f0);
			line-height: 1.5;
			display: -webkit-box;
			-webkit-line-clamp: 3;
			-webkit-box-orient: vertical;
			overflow: hidden;
		}

		.price {
			font-family: var(--sr-font-mono, monospace);
			font-size: 0.875rem;
			color: var(--sr-text, #e2e8f0);
		}
	`;

	@property({ attribute: false })
	signal: SignalSummary | undefined;

	override render() {
		if (!this.signal) return html``;
		const s = this.signal;
		const tier = confidenceTier(s.confidence);
		return html`
			<div class="card">
				<div class="header">
					<span class="ticker">${s.ticker}</span>
					<span class="direction direction--${s.direction}">${s.direction}</span>
				</div>

				<div class="confidence-row">
					<span class="confidence-label">${fmtConfidence(s.confidence)}</span>
					<div class="confidence-bar">
						<div
							class="confidence-fill confidence-fill--${tier}"
							style="width: ${s.confidence * 100}%"
						></div>
					</div>
				</div>

				<div class="meta">
					<span>${s.agent_name} · ${s.horizon_days}d horizon · ${s.prediction_date}</span>
					${s.current_price != null
						? html`<span class="price">$${s.current_price.toFixed(2)}</span>`
						: html``}
				</div>

				<p class="reasoning">${s.reasoning}</p>
			</div>
		`;
	}
}

declare global {
	interface HTMLElementTagNameMap {
		'sr-signal-card': SrSignalCard;
	}
}
