import { useAuthStore } from '../../stores/authStore'
import { useUIStore } from '../../stores/uiStore'

// What the Admin and Users pages share: the deck's settings (kiosk lock,
// waitlist, branding — Admin → Settings), its plans (Admin → Plans), the way
// onto the Users page and the search its lists share.

export function adminHeaders(): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

// Kiosk (simple-chat) lock for the deck: nobody, every non-admin, or the users
// marked on the Users page. Admins are never locked.
export type KioskMode = 'off' | 'all' | 'selected'

export interface SystemConfig {
  docker_spawn_enabled: boolean
  kiosk_mode: KioskMode
  kiosk_new_users: boolean
  // The mode is the `--simple-chat` default — no admin has picked one yet.
  kiosk_mode_from_env?: boolean
  // While self-registration is closed, sign-up takes waitlist applications.
  waitlist_enabled: boolean
  // Blank: applicants join under the instance name (waitlistName).
  waitlist_cohort: string
  // Branding: the panels' title and the sign-in page (the logo: BrandingCard).
  instance_name: string
  instance_description: string
  // "Powered by Captain" (captain-claw.com · GitHub) in the sign-in page's footer.
  show_powered_by: boolean
  // A process agent idle this many minutes is put to sleep (0 = never).
  agent_idle_stop_minutes: number
  // Custom tools (customTools.ts): on for this deck, and whether a tool shared
  // with everybody waits for an admin.
  external_tools_enabled: boolean
  external_tools_everybody_approval: boolean
  // Only run custom tools inside the OS sandbox (Linux, Landlock).
  external_tools_require_sandbox: boolean
}

export const CONFIG_DEFAULTS: SystemConfig = {
  docker_spawn_enabled: true, kiosk_mode: 'off', kiosk_new_users: false,
  waitlist_enabled: true, waitlist_cohort: '',
  instance_name: 'Captain', instance_description: 'Agent Management Console',
  show_powered_by: true, agent_idle_stop_minutes: 0,
  external_tools_enabled: false, external_tools_everybody_approval: true, external_tools_require_sandbox: false,
}

// What applicants join: the waitlist name an admin set, else the instance name.
export function waitlistName(cfg: SystemConfig): string {
  return cfg.waitlist_cohort || cfg.instance_name || 'Captain'
}

export function kioskModeNote(mode: KioskMode): string | null {
  if (mode === 'all') return 'Kiosk mode is on for everyone (Admin console → Settings → Kiosk mode), so this user is locked either way.'
  if (mode === 'off') return 'Kiosk mode is off (Admin console → Settings → Kiosk mode); this takes effect under "Selected users".'
  return null
}

export async function fetchConfig(): Promise<SystemConfig> {
  const res = await fetch('/fd/admin/config', {
    headers: adminHeaders(), credentials: 'include',
  })
  if (!res.ok) return { ...CONFIG_DEFAULTS }
  return { ...CONFIG_DEFAULTS, ...(await res.json()) }
}

export async function saveConfig(cfg: Partial<SystemConfig>): Promise<SystemConfig> {
  const res = await fetch('/fd/admin/config', {
    method: 'PUT', headers: adminHeaders(), credentials: 'include',
    body: JSON.stringify(cfg),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(typeof body.detail === 'string' ? body.detail : 'Failed to save config')
  }
  return { ...CONFIG_DEFAULTS, ...(await res.json()) }
}

export interface PlanLimits {
  max_agents: number
  max_storage_mb: number
  requests_per_minute: number
  spawns_per_hour: number
  messages_per_day: number  // 0 = unlimited
  tokens_per_day: number    // 0 = unlimited
}

export async function fetchPlans(): Promise<Record<string, PlanLimits>> {
  const res = await fetch('/fd/admin/plans', {
    headers: adminHeaders(), credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to fetch plans')
  const data = await res.json()
  return data.plans
}

// ── The Users page ──

export type UsersTab = 'users' | 'invitations' | 'waitlist'

export const USERS_TAB_KEY = 'fd:users-tab'

// Open the Users page on a given tab (Admin → Settings links to the waitlist).
export function openUsersTab(tab: UsersTab) {
  try { sessionStorage.setItem(USERS_TAB_KEY, tab) } catch { /* storage blocked */ }
  useUIStore.getState().setView('users')
}

// Every word must appear somewhere in the item's text.
export function matchesQuery(query: string, ...fields: (string | null | undefined)[]): boolean {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean)
  if (words.length === 0) return true
  const hay = fields.filter(Boolean).join(' ').toLowerCase()
  return words.every((w) => hay.includes(w))
}
