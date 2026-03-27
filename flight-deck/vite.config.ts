import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: '../captain_claw/flight_deck/static',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:23180',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:23180',
        ws: true,
      },
      '/fd': {
        target: 'http://localhost:25080',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
