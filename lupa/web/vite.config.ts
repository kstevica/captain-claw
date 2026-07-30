import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // The BFF serves this directory as the SPA. Nothing else lives in it,
    // so (unlike flight-deck's static dir) emptying it is safe.
    outDir: '../api/static',
    emptyOutDir: true,
  },
  server: {
    port: 25190,
    proxy: {
      '/api': { target: 'http://localhost:25180', changeOrigin: true },
    },
  },
})
