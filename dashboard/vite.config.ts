import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
	plugins: [
		VitePWA({
			registerType: 'autoUpdate',
			manifest: {
				name: 'Stock Radar',
				short_name: 'StockRadar',
				description: 'AI-powered stock market signal dashboard',
				theme_color: '#0f1117',
				background_color: '#0f1117',
				display: 'standalone',
				icons: [
					{
						src: '/favicon.svg',
						sizes: 'any',
						type: 'image/svg+xml',
						purpose: 'any',
					},
				],
			},
			workbox: {
				globPatterns: ['**/*.{js,css,html,svg,png,woff2}'],
				runtimeCaching: [
					{
						urlPattern: /^http:\/\/localhost:8081/,
						handler: 'NetworkFirst',
						options: {
							cacheName: 'dashboard-api',
							networkTimeoutSeconds: 5,
						},
					},
				],
			},
		}),
	],
	css: {
		preprocessorOptions: {
			scss: {
				api: 'modern-compiler',
			},
		},
	},
	server: {
		port: 5173,
		host: true,
	},
});
