/**
 * Settings sync service — bridges Zustand stores with server-side per-user settings.
 *
 * On login: fetches all settings from server, hydrates stores.
 * On store mutation: debounced write to server.
 * One-time migration: reads localStorage fd:* keys, uploads, clears.
 */

import { useAuthStore } from '../stores/authStore'

const FD = '/fd'

// All localStorage keys that should be migrated
const MIGRATEABLE_KEYS = [
  'fd:theme',
  'fd:botport-connection',
  'fd:agent-groups',
  'fd:agent-positions',
  'fd:agent-layout-mode',
  'fd:local-agents',
  'fd:container-descriptions',
  'fd:container-names',
  'fd:container-forwarding-tasks',
  'fd:container-consult-approval',
  'fd:process-descriptions',
  'fd:process-names',
  'fd:process-forwarding-tasks',
  'fd:process-consult-approval',
  'fd:pipelines',
  'fd:pinned-messages',
  'fd:pinned-files',
  'fd:shared-clipboard',
  'fd:chat-panel-width',
  'fd:director-panel-width',
  'fd:tool-panel-width',
  'fd:director-open',
  'fd:agent-presets',
  'fd:container-view',
  'fd:process-view',
  'fd:local-agent-view',
]

function _headers(): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

/** Fetch all settings for the current user from the server. */
export async function fetchSettings(): Promise<Record<string, string>> {
  const res = await fetch(`${FD}/settings`, {
    headers: _headers(),
    credentials: 'include',
  })
  if (!res.ok) return {}
  return res.json()
}

/** Save settings to server (partial merge). */
export async function saveSettings(settings: Record<string, string>): Promise<void> {
  await fetch(`${FD}/settings`, {
    method: 'PUT',
    headers: _headers(),
    credentials: 'include',
    body: JSON.stringify({ settings }),
  })
}

/** Delete a single setting from the server. */
export async function deleteSetting(key: string): Promise<void> {
  await fetch(`${FD}/settings/${encodeURIComponent(key)}`, {
    method: 'DELETE',
    headers: _headers(),
    credentials: 'include',
  })
}

// ── Debounced save ──

let _pending: Record<string, string> = {}
let _timer: ReturnType<typeof setTimeout> | null = null

/** Queue a setting change for debounced save. */
export function queueSave(key: string, value: string): void {
  _pending[key] = value
  if (_timer) clearTimeout(_timer)
  _timer = setTimeout(_flush, 300)
}

async function _flush(): Promise<void> {
  const batch = { ..._pending }
  _pending = {}
  _timer = null
  if (Object.keys(batch).length === 0) return
  try {
    await saveSettings(batch)
  } catch {
    // Silently fail — settings will re-sync next load
  }
}

// ── Migration ──

/** Migrate localStorage fd:* keys to server. Called once on first auth login. */
export async function migrateFromLocalStorage(): Promise<void> {
  const settings: Record<string, string> = {}
  for (const key of MIGRATEABLE_KEYS) {
    const val = localStorage.getItem(key)
    if (val !== null) {
      settings[key] = val
    }
  }
  if (Object.keys(settings).length === 0) return
  try {
    await saveSettings(settings)
    // Clear migrated keys
    for (const key of MIGRATEABLE_KEYS) {
      localStorage.removeItem(key)
    }
  } catch {
    // Keep localStorage as fallback if server save fails
  }
}

// ── Hydrate stores ──

type StoreHydrator = (settings: Record<string, string>) => void

const _hydrators: StoreHydrator[] = []

/** Register a store hydrator function. Called during store initialization. */
export function registerHydrator(fn: StoreHydrator): void {
  _hydrators.push(fn)
}

/**
 * Full sync: fetch settings from server, hydrate all registered stores.
 * If no server settings exist, attempt migration from localStorage.
 */
export async function hydrateAllStores(): Promise<void> {
  let settings = await fetchSettings()

  // If server has no settings, try migrating from localStorage
  if (Object.keys(settings).length === 0) {
    await migrateFromLocalStorage()
    settings = await fetchSettings()
  }

  for (const hydrate of _hydrators) {
    try {
      hydrate(settings)
    } catch {
      // Don't let one store failure block others
    }
  }

  // Clear all fd:* localStorage keys so stale data from previous
  // sessions doesn't bleed into the authenticated session
  for (const key of MIGRATEABLE_KEYS) {
    localStorage.removeItem(key)
  }
}
