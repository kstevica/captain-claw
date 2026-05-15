import { create } from 'zustand'
import { useAuthStore } from './authStore'

/**
 * Store for agent-coded apps (the new code-app runtime).
 *
 * The store is intentionally tiny — it doesn't try to mirror every
 * field the server returns. Only the bits the *UI* needs in more
 * than one place live here:
 *
 *   - The list of apps, refreshed periodically by the page.
 *   - Which slug the user has selected (persisted to localStorage so
 *     navigating away and back re-opens the same app).
 *
 * Logs, manifests, and per-app actions stay co-located with the
 * page that uses them. Hoisting them here would invite stale-state
 * bugs that a smaller store sidesteps.
 */

export interface CodeAppSummary {
  slug: string
  manifest: Record<string, unknown>
  has_backend: boolean
  has_frontend: boolean
  running: boolean
  pid: number | null
  idle_seconds: number | null
  has_error: boolean
}

const SELECTED_SLUG_KEY = 'fd:code-app:selected-slug'

function loadSelected(): string | null {
  try { return localStorage.getItem(SELECTED_SLUG_KEY) } catch { return null }
}

function persistSelected(slug: string | null) {
  try {
    if (slug) localStorage.setItem(SELECTED_SLUG_KEY, slug)
    else localStorage.removeItem(SELECTED_SLUG_KEY)
  } catch { /* ignore */ }
}

/**
 * Build headers for an FD API call. Matches the pattern used by every
 * other FD client (see ``app-runtime/authoring.ts``): bearer token
 * pulled from ``useAuthStore``. We deliberately do *not* rely on
 * cookies — FD's ``get_current_user`` dependency expects the token in
 * the ``Authorization`` header, and cookie-based calls 401.
 */
export function fdAuthHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extra,
  }
}

interface AppCodeState {
  apps: CodeAppSummary[]
  selectedSlug: string | null
  loading: boolean
  error: string | null
  // The page bumps this to force the iframe to remount (Reload / Restart).
  iframeNonce: number

  refresh: () => Promise<void>
  selectSlug: (slug: string) => void
  bumpIframe: () => void
}

export const useAppCodeStore = create<AppCodeState>((set, get) => ({
  apps: [],
  selectedSlug: loadSelected(),
  loading: false,
  error: null,
  iframeNonce: 0,

  refresh: async () => {
    set({ loading: true, error: null })
    try {
      const r = await fetch('/fd/code-apps', { headers: fdAuthHeaders() })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const data = await r.json()
      const apps: CodeAppSummary[] = data.apps || []
      // If our remembered slug is gone, fall back to the first app.
      const cur = get().selectedSlug
      const stillExists = cur && apps.some((a) => a.slug === cur)
      const selectedSlug = stillExists ? cur : (apps[0]?.slug ?? null)
      if (selectedSlug !== cur) persistSelected(selectedSlug)
      set({ apps, selectedSlug, loading: false })
    } catch (e) {
      set({ error: (e as Error).message || 'Failed to load code-apps', loading: false })
    }
  },

  selectSlug: (slug) => {
    persistSelected(slug)
    set({ selectedSlug: slug, iframeNonce: get().iframeNonce + 1 })
  },

  bumpIframe: () => set({ iframeNonce: get().iframeNonce + 1 }),
}))
