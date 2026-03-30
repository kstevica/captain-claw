import { create } from 'zustand'

const STORAGE_KEY = 'fd:botport-connection'

interface ConnectionConfig {
  botportUrl: string  // e.g. "http://localhost:23180" or "https://botport.example.com"
}

interface ConnectionStore extends ConnectionConfig {
  setBotportUrl: (url: string) => void
}

function loadConfig(): ConnectionConfig {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    return { botportUrl: stored.botportUrl || '' }
  } catch {
    return { botportUrl: '' }
  }
}

function persist(config: ConnectionConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
}

const initial = loadConfig()

export const useConnectionStore = create<ConnectionStore>((set) => ({
  ...initial,

  setBotportUrl: (botportUrl) => {
    persist({ botportUrl })
    set({ botportUrl })
  },
}))

/**
 * Get the base URL for REST API calls.
 * - If botportUrl is set, use it directly (e.g. "http://192.168.1.50:23180")
 * - If empty, use relative "/api" which relies on the Vite dev proxy
 */
export function getApiBase(): string {
  const { botportUrl } = useConnectionStore.getState()
  if (!botportUrl) return '/api'
  // Ensure no trailing slash, append /api
  const base = botportUrl.replace(/\/+$/, '')
  return `${base}/api`
}

/**
 * Get the WebSocket URL for the BotPort dashboard.
 * - If botportUrl is set, derive ws:// from http://
 * - If empty, use relative ws path via Vite proxy
 */
export function getWsUrl(): string {
  const { botportUrl } = useConnectionStore.getState()
  if (!botportUrl) {
    // No BotPort URL configured — return empty to skip connection.
    // The Flight Deck server doesn't have /ws/dashboard.
    return ''
  }
  const wsBase = botportUrl
    .replace(/^http:/, 'ws:')
    .replace(/^https:/, 'wss:')
    .replace(/\/+$/, '')
  return `${wsBase}/ws/dashboard`
}
