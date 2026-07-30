import { create } from 'zustand'
import { api, post, setAccessToken, onSessionChange } from './api'

// ── types ────────────────────────────────────────────────────────────

export interface Pack {
  slug: string
  name: string
  tagline: string
  theme: Record<string, string>
  vocabulary: Record<string, string>
  intake: { types: { id: string; label: string; description: string; default_max_agents: number }[] }
  quality: Record<string, unknown>
  briefs?: { presets: { id: string; label: string; hours: number }[] }
  onboarding_md?: string
}

export interface User { id: string; email?: string; display_name?: string; role?: string }

export interface Stream {
  id: string; title: string; pack: string; vfs_project: string
  created_at: string; updated_at: string; rounds?: Round[] | number
}

export interface Round {
  stream_id: string; session_id: string; round_no: number; kind: string; created_at: string
}

// ── pack (branding, vocabulary — everything vertical-specific) ───────

interface PackState { pack: Pack | null; load: () => Promise<void> }

/** The desk slug from the URL — /desks/<slug> activates that pack. */
export function deskSlugFromPath(): string | null {
  const m = window.location.pathname.match(/^\/desks\/([a-z0-9-]+)/)
  return m ? m[1] : null
}

export const usePack = create<PackState>((set) => ({
  pack: null,
  load: async () => {
    let pack: Pack | null = null
    const desk = deskSlugFromPath()
    if (desk) {
      // Published desk manifests are public — plain fetch, no auth needed.
      const r = await fetch(`/api/packs/${desk}`)
      if (r.ok) pack = (await r.json()).pack as Pack
    }
    if (!pack) pack = await api<Pack>('/api/pack')
    // Project the pack theme onto the CSS variables + document identity.
    const root = document.documentElement
    const map: Record<string, string> = {
      accent: '--lp-accent', accent_soft: '--lp-accent-soft', bg: '--lp-bg',
      surface: '--lp-surface', border: '--lp-border', text: '--lp-text',
      text_dim: '--lp-text-dim',
    }
    for (const [k, v] of Object.entries(pack.theme ?? {})) {
      if (map[k]) root.style.setProperty(map[k], v)
    }
    document.title = pack.name
    set({ pack })
  },
}))

/** Vocabulary lookup with a safe fallback — the shell never hardcodes terms. */
export function useVocab(): (key: string, fallback?: string) => string {
  const pack = usePack((s) => s.pack)
  return (key, fallback) => pack?.vocabulary?.[key] ?? fallback ?? key
}

// ── auth ─────────────────────────────────────────────────────────────

interface AuthState {
  user: User | null
  ready: boolean
  boot: () => Promise<void>
  login: (email: string, password: string, register?: boolean) => Promise<void>
  logout: () => Promise<void>
}

export const useAuth = create<AuthState>((set) => ({
  user: null,
  ready: false,
  boot: async () => {
    onSessionChange((token, user) => {
      if (!token) set({ user: null })
      else if (user) set({ user: user as User })
    })
    // Try a silent refresh (the httpOnly cookie survives reloads).
    try {
      const r = await fetch('/api/auth/refresh', { method: 'POST', credentials: 'same-origin' })
      if (r.ok) {
        const data = await r.json()
        setAccessToken(data.access_token ?? null)
        set({ user: data.user ?? null })
      }
    } catch { /* FD down or signed out — the login screen handles it */ }
    set({ ready: true })
  },
  login: async (email, password, register = false) => {
    const action = register ? 'register' : 'login'
    const r = await fetch(`/api/auth/${action}`, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(register ? { email, password, display_name: email.split('@')[0] }
                                    : { email, password }),
    })
    if (!r.ok) {
      let detail = 'sign-in failed'
      try { detail = (await r.json()).detail ?? detail } catch { /* not json */ }
      throw new Error(String(detail))
    }
    const data = await r.json()
    setAccessToken(data.access_token ?? null)
    set({ user: data.user ?? null })
  },
  logout: async () => {
    try { await fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin' }) }
    catch { /* best-effort */ }
    setAccessToken(null)
    set({ user: null })
  },
}))

// ── streams ──────────────────────────────────────────────────────────

interface StreamsState {
  streams: Stream[]
  load: (pack?: string) => Promise<void>
  create: (title: string, pack?: string) => Promise<Stream>
}

export const useStreams = create<StreamsState>((set) => ({
  streams: [],
  load: async (pack?: string) => {
    const q = pack ? `?pack=${encodeURIComponent(pack)}` : ''
    const data = await api<{ streams: Stream[] }>(`/api/streams${q}`)
    set({ streams: data.streams })
  },
  create: async (title: string, pack?: string) => {
    const s = await post<Stream>('/api/streams', { title, pack: pack ?? '' })
    set((st) => ({ streams: [s, ...st.streams] }))
    return s
  },
}))
