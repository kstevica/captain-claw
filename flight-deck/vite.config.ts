import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: '../captain_claw/flight_deck/static',
    // CRITICAL: do NOT empty this directory on build.
    //
    // The same static/ directory holds the React-built dashboard assets AND
    // the standalone glasses pages (glasses_view.html, glasses_mobile.html,
    // glasses_input.html, glasses_enroll.html, glasses_person.html,
    // glasses_settings.html). Vite's default emptyOutDir wipes all of them
    // before writing the new React bundle — silently breaking every glasses
    // route until someone notices FileNotFoundError in FD's logs.
    //
    // Trade-off: stale hashed asset filenames from previous builds will
    // accumulate. They never clash (hashes are content-derived), but the
    // directory grows over time. Periodic cleanup: delete every file in
    // static/assets/ whose hash doesn't appear in the new index.html, e.g.
    //   rm -rf captain_claw/flight_deck/static/assets/
    //   npm run build
    // when the disk usage ever becomes annoying.
    emptyOutDir: false,
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
