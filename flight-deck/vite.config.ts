import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The smart-glasses HUD is its own page (hud.html → src/hud/main.tsx), served
// by Flight Deck at /hud/. In dev, rewrite /hud and /hud/<screen> to that page;
// the manifest and service worker come from the FD server (proxied below).
function hudDevRewrite(): Plugin {
  return {
    name: 'hud-dev-rewrite',
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        const url = req.url || ''
        const m = /^\/hud(\/[^?]*)?(\?.*)?$/.exec(url)
        if (m && !/^\/hud\/(manifest\.webmanifest|sw\.js)$/.test(url.split('?')[0])) {
          req.url = '/hud.html' + (m[2] || '')
        }
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), hudDevRewrite()],
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
    rollupOptions: {
      // Two pages: the dashboard and the lean smart-glasses HUD. They share
      // only small chunks (react, the auth store) — the HUD must never pull
      // the dashboard's three.js / web-llm / xyflow graph.
      input: {
        index: fileURLToPath(new URL('./index.html', import.meta.url)),
        hud: fileURLToPath(new URL('./hud.html', import.meta.url)),
      },
    },
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
      // Scheduler + glasses bridge endpoints live on the FD server too.
      '/scheduler': { target: 'http://localhost:25080', changeOrigin: true },
      '/glasses': { target: 'http://localhost:25080', changeOrigin: true },
      '/deck': { target: 'http://localhost:25080', changeOrigin: true },
      '/hud/manifest.webmanifest': { target: 'http://localhost:25080', changeOrigin: true },
      '/hud/sw.js': { target: 'http://localhost:25080', changeOrigin: true },
    },
  },
})
