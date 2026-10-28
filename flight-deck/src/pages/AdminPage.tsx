import { useEffect, useState, useCallback } from 'react'
import {
  Shield,
  BarChart3,
  RefreshCw,
  Loader2,
  Settings2,
  Key,
  Save,
  Check,
  Plug,
  MessageSquare,
  ClipboardList,
  UserCog,
} from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import { DeckProfileDefaults } from '../components/profile/DeckProfileDefaults'
import { SharedTelegramBotAdmin } from '../components/admin/SharedTelegramBotAdmin'
import { SharedCredentialsAdmin } from '../components/admin/SharedCredentialsAdmin'
import { BrandingAdmin } from '../components/admin/BrandingAdmin'
import { Switch } from '../components/admin/Switch'
import { IdleAgentsSetting, IDLE_AGENTS_HINT, IDLE_AGENTS_TITLE } from '../components/admin/IdleAgentsSetting'
import { ByokSetting } from '../components/admin/ByokSetting'
import {
  adminHeaders, CONFIG_DEFAULTS, fetchConfig, fetchPlans, openUsersTab, saveConfig, waitlistName,
  type KioskMode, type PlanLimits, type SystemConfig,
} from '../components/admin/deckConfig'
import { ADMIN_TAB_KEY } from '../services/sharedIntegrations'

// Users, invitations and the waitlist have a page of their own (UsersPage).
type AdminTab = 'usage' | 'plans' | 'settings' | 'integrations'

// Other pages may open Admin on a given tab (Connections → Telegram sends an
// admin to Integrations, see ADMIN_TAB_KEY); otherwise it opens on Usage.
function takeRequestedTab(): AdminTab {
  try {
    const t = sessionStorage.getItem(ADMIN_TAB_KEY)
    sessionStorage.removeItem(ADMIN_TAB_KEY)
    if (t === 'plans' || t === 'settings' || t === 'integrations') return t
  } catch { /* storage blocked */ }
  return 'usage'
}

// ── Types ──

const KIOSK_MODES: { value: KioskMode; label: string; hint: string }[] = [
  { value: 'off', label: 'Off', hint: 'Everyone gets the normal layout.' },
  { value: 'all', label: 'Everyone', hint: 'Every non-admin is locked to simple chat.' },
  { value: 'selected', label: 'Selected users', hint: 'Only the users marked on the Users page are locked.' },
]

interface UsageLog {
  id: number
  user_id: string
  event_type: string
  detail: string
  created_at: string
}

interface CostUserRow {
  user_id: string
  email: string | null
  display_name: string | null
  runs: number
  usd: number
  priced: number
}
interface CostKindRow {
  run_kind: string
  runs: number
  usd: number
}
interface CostRollup {
  by_user: CostUserRow[]
  by_kind: CostKindRow[]
  total_usd: number
}

// ── API helpers ──
async function fetchUsageSummary(userId?: string, since?: string): Promise<Record<string, number>> {
  const params = new URLSearchParams()
  if (userId) params.set('user_id', userId)
  if (since) params.set('since', since)
  const res = await fetch(`/fd/admin/usage/summary?${params}`, {
    headers: adminHeaders(), credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to fetch usage')
  const data = await res.json()
  return data.summary
}

async function fetchUsageLogs(userId?: string, eventType?: string, limit = 50): Promise<UsageLog[]> {
  const params = new URLSearchParams()
  if (userId) params.set('user_id', userId)
  if (eventType) params.set('event_type', eventType)
  params.set('limit', String(limit))
  const res = await fetch(`/fd/admin/usage?${params}`, {
    headers: adminHeaders(), credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to fetch usage logs')
  const data = await res.json()
  return data.logs
}

async function fetchCostRollup(since?: string, until?: string): Promise<CostRollup> {
  const params = new URLSearchParams()
  if (since) params.set('since', since)
  if (until) params.set('until', until)
  const qs = params.toString() ? `?${params}` : ''
  const res = await fetch(`/fd/admin/costs${qs}`, { headers: adminHeaders(), credentials: 'include' })
  if (!res.ok) throw new Error('Failed to fetch cost rollup')
  return res.json()
}

async function fetchProviderKeys(): Promise<Record<string, string>> {
  const res = await fetch('/fd/admin/provider-keys', {
    headers: adminHeaders(), credentials: 'include',
  })
  if (!res.ok) return {}
  const data = await res.json()
  return data.keys || {}
}

async function saveProviderKeysApi(keys: Record<string, string>): Promise<Record<string, string>> {
  const res = await fetch('/fd/admin/provider-keys', {
    method: 'PUT', headers: adminHeaders(), credentials: 'include',
    body: JSON.stringify({ keys }),
  })
  if (!res.ok) throw new Error('Failed to save provider keys')
  const data = await res.json()
  return data.keys || {}
}

const LLM_PROVIDERS = [
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'gemini', label: 'Gemini' },
  { value: 'openrouter', label: 'OpenRouter' },
  { value: 'xai', label: 'xAI' },
  { value: 'litert', label: 'LiteRT (local Gemma)' },
] as const

async function updatePlan(plan: string, limits: Partial<PlanLimits>): Promise<void> {
  const res = await fetch(`/fd/admin/plans/${plan}`, {
    method: 'PUT', headers: adminHeaders(), credentials: 'include',
    body: JSON.stringify(limits),
  })
  if (!res.ok) throw new Error('Failed to update plan')
}

// ── Components ──

const PLAN_FIELD_LABELS: Record<string, string> = {
  max_agents: 'Max Agents',
  max_storage_mb: 'Storage (MB)',
  requests_per_minute: 'Requests / min',
  spawns_per_hour: 'Spawns / hour',
  messages_per_day: 'Messages / day',
  tokens_per_day: 'AI tokens / day',
}

function PlanCard({
  name, limits, onSave,
}: {
  name: string
  limits: PlanLimits
  onSave: (plan: string, limits: Partial<PlanLimits>) => Promise<void>
}) {
  const [draft, setDraft] = useState({ ...limits })
  const [saving, setSaving] = useState(false)
  const changed = Object.keys(PLAN_FIELD_LABELS).some(
    (k) => draft[k as keyof PlanLimits] !== limits[k as keyof PlanLimits]
  )

  const handleSave = async () => {
    setSaving(true)
    try { await onSave(name, draft) } catch { /* ignore */ }
    setSaving(false)
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-3">
      <h3 className="text-sm font-semibold text-zinc-200 capitalize">{name}</h3>
      <div className="grid grid-cols-2 gap-3">
        {Object.entries(PLAN_FIELD_LABELS).map(([key, label]) => (
          <div key={key}>
            <label className="block text-xs text-zinc-500 mb-1">{label}</label>
            <input
              type="number"
              min={0}
              value={draft[key as keyof PlanLimits]}
              onChange={(e) => setDraft({ ...draft, [key]: parseInt(e.target.value) || 0 })}
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
            />
          </div>
        ))}
      </div>
      <p className="text-[11px] text-zinc-500">
        Daily limits: 0 is unlimited. A day is a UTC day; admins are never held to them.
      </p>
      {changed && (
        <button
          onClick={handleSave}
          disabled={saving}
          className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
        >
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
      )}
    </div>
  )
}

// ── Main Page ──

export function AdminPage() {
  const currentUser = useAuthStore((s) => s.user)
  const [tab, setTab] = useState<AdminTab>(takeRequestedTab)
  const [plans, setPlans] = useState<Record<string, PlanLimits>>({})
  const [config, setConfig] = useState<SystemConfig>({ ...CONFIG_DEFAULTS })
  const [idleSaving, setIdleSaving] = useState(false)
  const [providerKeys, setProviderKeys] = useState<Record<string, string>>({})
  const [providerKeysDraft, setProviderKeysDraft] = useState<Record<string, string>>({})
  const [keySaveFlash, setKeySaveFlash] = useState<string | null>(null)
  const [usageSummary, setUsageSummary] = useState<Record<string, number>>({})
  const [usageLogs, setUsageLogs] = useState<UsageLog[]>([])
  const [costRollup, setCostRollup] = useState<CostRollup | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  // Bumped by Refresh: the Integrations cards load on their own and reload on it.
  const [refreshTick, setRefreshTick] = useState(0)
  const [cohortDraft, setCohortDraft] = useState('')
  const signUpMode = useAuthStore((s) => s.signUp.mode)

  const loadSettings = useCallback(async () => {
    try {
      const [plansData, cfgData, keys] = await Promise.all([fetchPlans(), fetchConfig(), fetchProviderKeys()])
      setPlans(plansData)
      setConfig(cfgData)
      setCohortDraft(cfgData.waitlist_cohort)
      setProviderKeys(keys)
      setProviderKeysDraft(keys)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load settings')
    }
  }, [])

  const loadUsage = useCallback(async () => {
    try {
      const [summary, logs, costs] = await Promise.all([
        fetchUsageSummary(), fetchUsageLogs(), fetchCostRollup().catch(() => null),
      ])
      setUsageSummary(summary)
      setUsageLogs(logs)
      setCostRollup(costs)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load usage')
    }
  }, [])

  useEffect(() => {
    setLoading(true)
    Promise.all([loadSettings(), loadUsage()]).finally(() => setLoading(false))
  }, [loadSettings, loadUsage])

  const refresh = () => {
    setRefreshTick((t) => t + 1)
    setLoading(true)
    Promise.all([loadSettings(), loadUsage()]).finally(() => setLoading(false))
  }

  if (!currentUser || currentUser.role !== 'admin') {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-zinc-500 text-sm">Admin access required</div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Shield className="h-5 w-5 text-violet-400" />
            <h1 className="text-lg font-semibold text-zinc-100">Admin Panel</h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => openUsersTab('users')}
              title="Users, invitations and the waitlist"
              className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <UserCog className="h-3.5 w-3.5" />
              Users
            </button>
            <button
              onClick={refresh}
              disabled={loading}
              className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {error && (
          <div className="rounded-md border border-red-900/50 bg-red-950/20 px-4 py-2 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 overflow-x-auto border-b border-zinc-800">
          <button
            onClick={() => setTab('usage')}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm border-b-2 transition-colors ${
              tab === 'usage'
                ? 'border-violet-500 text-zinc-100'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <BarChart3 className="h-4 w-4" />
            Usage
          </button>
          <button
            onClick={() => setTab('plans')}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm border-b-2 transition-colors ${
              tab === 'plans'
                ? 'border-violet-500 text-zinc-100'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <Settings2 className="h-4 w-4" />
            Plans
          </button>
          <button
            onClick={() => setTab('settings')}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm border-b-2 transition-colors ${
              tab === 'settings'
                ? 'border-violet-500 text-zinc-100'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <Settings2 className="h-4 w-4" />
            Settings
          </button>
          <button
            onClick={() => setTab('integrations')}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm border-b-2 transition-colors ${
              tab === 'integrations'
                ? 'border-violet-500 text-zinc-100'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <Plug className="h-4 w-4" />
            Integrations
          </button>
        </div>

        {loading && (
          <div className="flex justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
          </div>
        )}

        {/* Usage tab */}
        {!loading && tab === 'usage' && (
          <div className="space-y-6">
            {/* Summary cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(usageSummary).map(([type, count]) => (
                <div key={type} className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                  <div className="text-2xl font-bold text-zinc-100">{count}</div>
                  <div className="text-xs text-zinc-500 mt-1">{type.replace(/_/g, ' ')}</div>
                </div>
              ))}
              {Object.keys(usageSummary).length === 0 && (
                <div className="col-span-full text-center text-sm text-zinc-500 py-4">
                  No usage data yet
                </div>
              )}
            </div>

            {/* Team spend (cost rollup) */}
            {costRollup && (
              <div>
                <div className="flex items-baseline gap-3 mb-3">
                  <h3 className="text-sm font-medium text-zinc-300">Team spend</h3>
                  <span className="text-lg font-semibold text-emerald-400">${costRollup.total_usd.toFixed(2)}</span>
                  <span className="text-xs text-zinc-500">total across all users</span>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden">
                    <div className="px-3 py-2 text-xs font-medium text-zinc-400 border-b border-zinc-800">By user</div>
                    {costRollup.by_user.length === 0 && (
                      <p className="px-3 py-3 text-xs text-zinc-500">No costs recorded yet</p>
                    )}
                    {costRollup.by_user.map((u) => (
                      <div key={u.user_id} className="flex items-center gap-2 px-3 py-2 text-sm border-b border-zinc-800/50 last:border-0">
                        <span className="flex-1 truncate text-zinc-300">{u.display_name || u.email || u.user_id.slice(0, 8)}</span>
                        <span className="text-xs text-zinc-500">{u.runs} run{u.runs === 1 ? '' : 's'}</span>
                        <span className="w-16 text-right font-mono text-emerald-400">${(u.usd || 0).toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                  <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden">
                    <div className="px-3 py-2 text-xs font-medium text-zinc-400 border-b border-zinc-800">By run kind</div>
                    {costRollup.by_kind.length === 0 && (
                      <p className="px-3 py-3 text-xs text-zinc-500">No costs recorded yet</p>
                    )}
                    {costRollup.by_kind.map((k) => (
                      <div key={k.run_kind} className="flex items-center gap-2 px-3 py-2 text-sm border-b border-zinc-800/50 last:border-0">
                        <span className="flex-1 truncate text-zinc-300 capitalize">{k.run_kind || 'other'}</span>
                        <span className="text-xs text-zinc-500">{k.runs} run{k.runs === 1 ? '' : 's'}</span>
                        <span className="w-16 text-right font-mono text-emerald-400">${(k.usd || 0).toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Recent logs */}
            <div>
              <h3 className="text-sm font-medium text-zinc-300 mb-3">Recent Activity</h3>
              <div className="space-y-1">
                {usageLogs.map((log) => (
                  <div key={log.id} className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-zinc-800/50 text-sm">
                    <span className="text-xs text-zinc-600 font-mono w-40 shrink-0">
                      {new Date(log.created_at).toLocaleString()}
                    </span>
                    <span className="px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400 text-xs">
                      {log.event_type}
                    </span>
                    <span className="text-zinc-500 text-xs truncate flex-1">
                      {(() => {
                        try {
                          const d = JSON.parse(log.detail)
                          return Object.entries(d).map(([k, v]) => `${k}: ${v}`).join(', ')
                        } catch { return log.detail }
                      })()}
                    </span>
                    <span className="text-xs text-zinc-600 font-mono shrink-0">{log.user_id.slice(0, 8)}</span>
                  </div>
                ))}
                {usageLogs.length === 0 && (
                  <p className="text-center text-sm text-zinc-500 py-4">No activity logs yet</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Settings tab */}
        {!loading && tab === 'settings' && (
          <div className="space-y-4">
            <BrandingAdmin
              name={config.instance_name}
              description={config.instance_description}
              poweredBy={config.show_powered_by}
              onSave={async (patch) => {
                const next = await saveConfig(patch)
                setConfig(next)
                // A waitlist without a name of its own goes by the instance name.
                const { signUp, setSignUp } = useAuthStore.getState()
                setSignUp({ ...signUp, cohort: waitlistName(next) })
                return next
              }}
            />

            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-4">
              <h3 className="text-sm font-semibold text-zinc-200">Spawn Settings</h3>
              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <div className="text-sm text-zinc-300">Docker container spawning</div>
                  <div className="text-xs text-zinc-500 mt-0.5">Allow users to spawn agents as Docker containers. When disabled, only local process agents are available.</div>
                </div>
                <button
                  onClick={async () => {
                    const next = !config.docker_spawn_enabled
                    try {
                      await saveConfig({ docker_spawn_enabled: next })
                      setConfig({ ...config, docker_spawn_enabled: next })
                      // Update the auth store so the spawner page reacts immediately
                      useAuthStore.getState().setDockerSpawnEnabled(next)
                    } catch { setError('Failed to save config') }
                  }}
                  className={`relative inline-flex h-6 w-11 shrink-0 rounded-full border-2 border-transparent transition-colors ${
                    config.docker_spawn_enabled ? 'bg-violet-600' : 'bg-zinc-700'
                  }`}
                >
                  <span className={`inline-block h-5 w-5 rounded-full bg-white transition-transform ${
                    config.docker_spawn_enabled ? 'translate-x-5' : 'translate-x-0'
                  }`} />
                </button>
              </label>
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-sm text-zinc-300">{IDLE_AGENTS_TITLE}</div>
                  <div className="text-xs text-zinc-500 mt-0.5">{IDLE_AGENTS_HINT}</div>
                </div>
                <IdleAgentsSetting
                  minutes={config.agent_idle_stop_minutes}
                  busy={idleSaving}
                  onSave={async (m) => {
                    setIdleSaving(true)
                    try {
                      setConfig(await saveConfig({ agent_idle_stop_minutes: m }))
                    } catch (e) {
                      setError(e instanceof Error ? e.message : 'Failed to save config')
                    } finally {
                      setIdleSaving(false)
                    }
                  }}
                />
              </div>
            </div>

            {/* Kiosk mode (simple chat) */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-4">
              <div>
                <h3 className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                  <MessageSquare className="h-4 w-4 text-violet-400" />
                  Kiosk mode (simple chat)
                </h3>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Locks users to the chat-first simple layout: they can chat, start an agent from an archetype,
                  manage their own connections and profile, and sign out. Admins are never locked.
                  A user already signed in picks up a change within about 15 minutes, or on reload.
                </p>
              </div>
              <div>
                <div className="inline-flex rounded-md border border-zinc-700 p-0.5" role="radiogroup" aria-label="Kiosk mode">
                  {KIOSK_MODES.map((m) => (
                    <button
                      key={m.value}
                      type="button"
                      role="radio"
                      aria-checked={config.kiosk_mode === m.value}
                      onClick={async () => {
                        if (config.kiosk_mode === m.value && !config.kiosk_mode_from_env) return
                        try {
                          const next = await saveConfig({ kiosk_mode: m.value })
                          setConfig(next)
                          useAuthStore.getState().setSimpleChatOnly(next.kiosk_mode === 'all')
                        } catch { setError('Failed to save config') }
                      }}
                      className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
                        config.kiosk_mode === m.value
                          ? 'bg-violet-600 text-white'
                          : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
                <p className="mt-1.5 text-xs text-zinc-500">
                  {KIOSK_MODES.find((m) => m.value === config.kiosk_mode)?.hint}
                  {config.kiosk_mode_from_env && ' Set by --simple-chat until you pick a mode here.'}
                </p>
              </div>
              <label className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-sm text-zinc-300">New users start in kiosk mode</div>
                  <div className="text-xs text-zinc-500 mt-0.5">
                    Marks accounts created from now on (Add user, sign-up or a waitlist approval). Counts under "Selected users";
                    with "Everyone" every non-admin is locked anyway.
                  </div>
                </div>
                <Switch
                  on={config.kiosk_new_users}
                  label="New users start in kiosk mode"
                  onToggle={async () => {
                    try {
                      setConfig(await saveConfig({ kiosk_new_users: !config.kiosk_new_users }))
                    } catch { setError('Failed to save config') }
                  }}
                />
              </label>
            </div>

            {/* Waitlist (sign-up while registration is closed) */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-4">
              <div>
                <h3 className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                  <ClipboardList className="h-4 w-4 text-violet-400" />
                  Waitlist
                </h3>
                <p className="text-xs text-zinc-500 mt-0.5">
                  While registration is closed, the sign-up page takes applications instead: name, email, a
                  password, and the name and email of whoever recommended them. You approve them on the
                  Users page, under{' '}
                  <button onClick={() => openUsersTab('waitlist')} className="text-violet-400 underline underline-offset-2 hover:no-underline">
                    Waitlist
                  </button>.
                  {signUpMode === 'register' && ' Self-registration is open on this deck (FD_REGISTRATION_OPEN), so the waitlist isn\'t shown.'}
                </p>
              </div>
              <label className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-sm text-zinc-300">Take waitlist applications</div>
                  <div className="text-xs text-zinc-500 mt-0.5">
                    Off: the sign-in page says sign-up is closed and that an administrator creates accounts
                    (Users → Add user, or Invitations). Applications already in stay listed.
                  </div>
                </div>
                <Switch
                  on={config.waitlist_enabled}
                  label="Take waitlist applications"
                  onToggle={async () => {
                    try {
                      const next = await saveConfig({ waitlist_enabled: !config.waitlist_enabled })
                      setConfig(next)
                      const { signUp, setSignUp } = useAuthStore.getState()
                      if (signUp.mode !== 'register') {
                        setSignUp({ mode: next.waitlist_enabled ? 'waitlist' : 'closed', cohort: waitlistName(next) })
                      }
                    } catch { setError('Failed to save config') }
                  }}
                />
              </label>
              <div>
                <label className="block text-sm text-zinc-300 mb-1">Waitlist name</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={cohortDraft}
                    onChange={(e) => setCohortDraft(e.target.value)}
                    maxLength={60}
                    placeholder={config.instance_name}
                    className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                  />
                  <button
                    onClick={async () => {
                      try {
                        const next = await saveConfig({ waitlist_cohort: cohortDraft })
                        setConfig(next)
                        setCohortDraft(next.waitlist_cohort)
                        const { signUp, setSignUp } = useAuthStore.getState()
                        setSignUp({ ...signUp, cohort: waitlistName(next) })
                      } catch (e) { setError(e instanceof Error ? e.message : 'Failed to save config') }
                    }}
                    disabled={cohortDraft.trim() === config.waitlist_cohort}
                    className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
                  >
                    <Save className="h-3.5 w-3.5" />
                    Save
                  </button>
                </div>
                <p className="mt-1 text-xs text-zinc-500">
                  Leave it blank to use the instance name ({config.instance_name}). Shown on the sign-up page
                  and saved with each application; changing it doesn't move earlier applicants.
                </p>
              </div>
            </div>

            {/* Provider API Keys */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-zinc-200">Provider API Keys</h3>
                <p className="text-xs text-zinc-500 mt-0.5">System-wide API keys for LLM providers. These are used as defaults when spawning agents.</p>
              </div>
              {LLM_PROVIDERS.map((p) => (
                <div key={p.value}>
                  <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-zinc-300">
                    <Key className="h-3.5 w-3.5 text-zinc-500" />
                    {p.label}
                    {providerKeys[p.value] && <span className="text-xs text-emerald-500/70 font-normal">configured</span>}
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={providerKeysDraft[p.value] || ''}
                      onChange={(e) => setProviderKeysDraft({ ...providerKeysDraft, [p.value]: e.target.value })}
                      placeholder={providerKeys[p.value] ? '••••••••' : `${p.label} API key`}
                      className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs font-mono text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                    />
                    <button
                      onClick={async () => {
                        try {
                          const val = providerKeysDraft[p.value] || ''
                          const updated = await saveProviderKeysApi({ [p.value]: val })
                          setProviderKeys(updated)
                          setProviderKeysDraft(updated)
                          setKeySaveFlash(p.value)
                          setTimeout(() => setKeySaveFlash(null), 1500)
                        } catch { setError('Failed to save provider key') }
                      }}
                      className={`flex items-center gap-1 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                        keySaveFlash === p.value
                          ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-400'
                          : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
                      }`}
                    >
                      {keySaveFlash === p.value ? <Check className="h-3 w-3" /> : <Save className="h-3 w-3" />}
                      {keySaveFlash === p.value ? 'Saved' : 'Save'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Owner profile defaults every user's agents receive. Not behind
            `loading`: it loads on its own, and a Refresh must not unmount it
            and drop a long unsaved draft. */}
        {tab === 'settings' && <DeckProfileDefaults />}

        {/* Set up once, used by everyone: the team Telegram bot (with its
            access requests) and the shared Google / Microsoft / Jev / Soniox
            credentials and MCP servers.
            They load on their own, like the profile defaults above. */}
        {tab === 'integrations' && (
          <div className="space-y-4">
            <SharedTelegramBotAdmin refreshTick={refreshTick} />
            <SharedCredentialsAdmin refreshTick={refreshTick} />
          </div>
        )}

        {/* Plans tab */}
        {!loading && tab === 'plans' && <div className="mb-4"><ByokSetting /></div>}
        {!loading && tab === 'plans' && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {Object.entries(plans).map(([name, limits]) => (
              <PlanCard
                key={name}
                name={name}
                limits={limits}
                onSave={async (plan, newLimits) => {
                  await updatePlan(plan, newLimits)
                  await loadSettings()
                }}
              />
            ))}
            {Object.keys(plans).length === 0 && (
              <p className="col-span-full text-center text-sm text-zinc-500 py-8">No plans configured</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
