/**
 * sr-app — root shell with hash router and sidebar navigation.
 */

import { LitElement, css, html } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import './sr-signals-view';
import './sr-watchlist-view';
import './sr-agent-status';
import './sr-ticker-detail';

type Route = 'signals' | 'watchlist' | 'agents' | 'ticker';

interface ParsedRoute {
	name: Route;
	param?: string;
}

function parseHash(hash: string): ParsedRoute {
	const path = hash.replace(/^#[/]?/, '');
	if (path.startsWith('ticker/')) {
		return { name: 'ticker', param: path.slice('ticker/'.length) };
	}
	if (path === 'watchlist') return { name: 'watchlist' };
	if (path === 'agents') return { name: 'agents' };
	return { name: 'signals' };
}

@customElement('sr-app')
export class SrApp extends LitElement {
	static override styles = css`
		:host {
			display: flex;
			min-height: 100vh;
			background: var(--sr-bg, #0f1117);
			color: var(--sr-text, #e2e8f0);
			font-family: var(--sr-font, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif);
		}

		/* CSS custom properties for child components */
		:host {
			--sr-bg: #0f1117;
			--sr-surface: #1a1d27;
			--sr-surface-raised: #22263a;
			--sr-border: #2d3150;
			--sr-text: #e2e8f0;
			--sr-text-muted: #8892a4;
			--sr-accent: #4f8ef7;
			--sr-font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
			--sr-font-mono: 'JetBrains Mono', 'Fira Code', monospace;
		}

		nav {
			width: 200px;
			flex-shrink: 0;
			background: var(--sr-surface, #1a1d27);
			border-right: 1px solid var(--sr-border, #2d3150);
			padding: 1.5rem 0;
			display: flex;
			flex-direction: column;
		}

		.nav-brand {
			padding: 0 1rem 1.5rem;
			font-size: 1rem;
			font-weight: 700;
			color: var(--sr-accent, #4f8ef7);
			border-bottom: 1px solid var(--sr-border, #2d3150);
			margin-bottom: 0.5rem;
		}

		.nav-link {
			display: block;
			padding: 0.625rem 1rem;
			font-size: 0.875rem;
			color: var(--sr-text-muted, #8892a4);
			text-decoration: none;
			transition: color 0.15s, background 0.15s;
		}

		.nav-link:hover {
			color: var(--sr-text, #e2e8f0);
			background: var(--sr-surface-raised, #22263a);
			text-decoration: none;
		}

		.nav-link--active {
			color: var(--sr-accent, #4f8ef7);
			background: rgba(79, 142, 247, 0.1);
			font-weight: 500;
		}

		main {
			flex: 1;
			padding: 2rem;
			overflow: auto;
			max-width: 1400px;
		}

		/* Narrow screens: stack nav on top */
		@media (max-width: 767px) {
			:host {
				flex-direction: column;
			}

			nav {
				width: 100%;
				flex-direction: row;
				align-items: center;
				padding: 0;
				border-right: none;
				border-bottom: 1px solid var(--sr-border, #2d3150);
			}

			.nav-brand {
				padding: 0.75rem 1rem;
				border-bottom: none;
				border-right: 1px solid var(--sr-border, #2d3150);
				margin-bottom: 0;
			}

			.nav-link {
				padding: 0.75rem 0.875rem;
			}

			main {
				padding: 1rem;
			}
		}
	`;

	@state() private _route: ParsedRoute = parseHash(window.location.hash);

	override connectedCallback(): void {
		super.connectedCallback();
		window.addEventListener('hashchange', this._onHashChange);
	}

	override disconnectedCallback(): void {
		super.disconnectedCallback();
		window.removeEventListener('hashchange', this._onHashChange);
	}

	private readonly _onHashChange = (): void => {
		this._route = parseHash(window.location.hash);
	};

	private _navClass(route: Route): string {
		return `nav-link${this._route.name === route ? ' nav-link--active' : ''}`;
	}

	private _renderView() {
		switch (this._route.name) {
			case 'watchlist':
				return html`<sr-watchlist-view></sr-watchlist-view>`;
			case 'agents':
				return html`<sr-agent-status></sr-agent-status>`;
			case 'ticker':
				return html`<sr-ticker-detail ticker=${this._route.param ?? ''}></sr-ticker-detail>`;
			default:
				return html`<sr-signals-view></sr-signals-view>`;
		}
	}

	override render() {
		return html`
			<nav>
				<div class="nav-brand">📡 Stock Radar</div>
				<a href="#/" class=${this._navClass('signals')}>Signals</a>
				<a href="#/watchlist" class=${this._navClass('watchlist')}>Watchlist</a>
				<a href="#/agents" class=${this._navClass('agents')}>Agents</a>
			</nav>
			<main>${this._renderView()}</main>
		`;
	}
}

declare global {
	interface HTMLElementTagNameMap {
		'sr-app': SrApp;
	}
}
