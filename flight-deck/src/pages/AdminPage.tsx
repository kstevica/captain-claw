import { useEffect, useState, useCallback } from 'react'
import {
  Shield,
  Users,
  BarChart3,
  RefreshCw,
  Trash2,
  ChevronDown,
  ChevronRight,
  Crown,
  Loader2,
  Settings2,
  Key,
  Save,
  Check,
} from 'lucide-react'
import { useAuthStore } from '../stores/authStore'

// ── Types ──

interface UserInfo {
  id: string
  email: string
  display_name: string
  role: string
  created_at: string
  updated_at: string
  metadata: string
}

interface UsageLog {
  id: number
  user_id: string
  event_type: string
  detail: string
  created_at: string
}

interface PlanLimits {
  max_agents: number
  max_storage_mb: number
  requests_per_minute: number
  spawns_per_hour: number
}

// ── API helpers ──

function _headers(): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function fetchUsers(limit = 100, offset = 0): Promise<{ users: UserInfo[]; total: number }> {
  const res = await fetch(`/fd/admin/users?limit=${limit}&offset=${offset}`, {
    headers: _headers(), credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to fetch users')
  return res.json()
}

async function updateUser(userId: string, patch: Record<string, unknown>): Promise<void> {
  const res = await fetch(`/fd/admin/users/${userId}`, {
    method: 'PUT', headers: _headers(), credentials: 'include',
    body: JSON.stringify(patch),
  })
  if (!res.ok) throw new Error('Failed to update user')
}

async function deleteUser(userId: string): Promise<void> {
  const res = await fetch(`/fd/admin/users/${userId}`, {
    method: 'DELETE', headers: _headers(), credentials: 'include',
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || 'Failed to delete user')
  }
}

async function fetchUsageSummary(userId?: string, since?: string): Promise<Record<string, number>> {
  const params = new URLSearchParams()
  if (userId) params.set('user_id', userId)
  if (since) params.set('since', since)
  const res = await fetch(`/fd/admin/usage/summary?${params}`, {
    headers: _headers(), credentials: 'include',
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
    headers: _headers(), credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to fetch usage logs')
  const data = await res.json()
  return data.logs
}

async function fetchConfig(): Promise<Record<string, boolean>> {
  const res = await fetch('/fd/admin/config', {
    headers: _headers(), credentials: 'include',
  })
  if (!res.ok) return { docker_spawn_enabled: true }
  return res.json()
}

async function saveConfig(cfg: Record<string, boolean>): Promise<void> {
  const res = await fetch('/fd/admin/config', {
    method: 'PUT', headers: _headers(), credentials: 'include',
    body: JSON.stringify(cfg),
  })
  if (!res.ok) throw new Error('Failed to save config')
}

async function fetchProviderKeys(): Promise<Record<string, string>> {
  const res = await fetch('/fd/admin/provider-keys', {
    headers: _headers(), credentials: 'include',
  })
  if (!res.ok) return {}
  const data = await res.json()
  return data.keys || {}
}

async function saveProviderKeysApi(keys: Record<string, string>): Promise<Record<string, string>> {
  const res = await fetch('/fd/admin/provider-keys', {
    method: 'PUT', headers: _headers(), credentials: 'include',
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

async function fetchPlans(): Promise<Record<string, PlanLimits>> {
  const res = await fetch('/fd/admin/plans', {
    headers: _headers(), credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to fetch plans')
  const data = await res.json()
  return data.plans
}

async function updatePlan(plan: string, limits: Partial<PlanLimits>): Promise<void> {
  const res = await fetch(`/fd/admin/plans/${plan}`, {
    method: 'PUT', headers: _headers(), credentials: 'include',
    body: JSON.stringify(limits),
  })
  if (!res.ok) throw new Error('Failed to update plan')
}

// ── Components ──

function UserRow({
  user, plans, onUpdate, onDelete, currentUserId,
}: {
  user: UserInfo
  plans: Record<string, PlanLimits>
  onUpdate: () => void
  onDelete: (id: string) => void
  currentUserId: string
}) {
  const [expanded, setExpanded] = useState(false)
  const [plan, setPlan] = useState(() => {
    try { return JSON.parse(user.metadata || '{}').plan || 'free' } catch { return 'free' }
  })
  const [role, setRole] = useState(user.role)
  const [saving, setSaving] = useState(false)
  const isSelf = user.id === currentUserId

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateUser(user.id, { plan, role })
      onUpdate()
    } catch { /* ignore */ }
    setSaving(false)
  }

  return (
    <div className="border border-zinc-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-zinc-800/50 transition-colors text-left"
      >
        {expanded ? <ChevronDown className="h-4 w-4 text-zinc-500 shrink-0" /> : <ChevronRight className="h-4 w-4 text-zinc-500 shrink-0" />}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-zinc-200 truncate">{user.display_name || user.email}</span>
            {user.role === 'admin' && <Crown className="h-3.5 w-3.5 text-amber-400" />}
            {isSelf && <span className="text-xs text-zinc-500">(you)</span>}
          </div>
          <div className="text-xs text-zinc-500">{user.email}</div>
        </div>
        <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400 capitalize">{plan}</span>
      </button>

      {expanded && (
        <div className="border-t border-zinc-800 px-4 py-3 bg-zinc-900/50 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-zinc-500 mb-1">Plan</label>
              <select
                value={plan}
                onChange={(e) => setPlan(e.target.value)}
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              >
                {Object.keys(plans).map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-zinc-500 mb-1">Role</label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value)}
                disabled={isSelf}
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none disabled:opacity-50"
              >
                <option value="user">user</option>
                <option value="admin">admin</option>
              </select>
            </div>
          </div>

          {plans[plan] && (
            <div className="text-xs text-zinc-500 grid grid-cols-2 gap-1">
              <span>Max agents: {plans[plan].max_agents}</span>
              <span>Storage: {plans[plan].max_storage_mb} MB</span>
              <span>Requests/min: {plans[plan].requests_per_minute}</span>
              <span>Spawns/hour: {plans[plan].spawns_per_hour}</span>
            </div>
          )}

          <div className="text-xs text-zinc-600">
            Joined: {new Date(user.created_at).toLocaleDateString()}
          </div>

          <div className="flex items-center gap-2 pt-1">
            <button
              onClick={handleSave}
              disabled={saving}
              className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
            >
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
            {!isSelf && (
              <button
                onClick={() => { if (confirm(`Delete user ${user.email}? This cannot be undone.`)) onDelete(user.id) }}
                className="rounded-md border border-red-900/50 px-3 py-1.5 text-xs text-red-400 hover:bg-red-950/50"
              >
                <Trash2 className="h-3 w-3 inline mr-1" />
                Delete
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

const PLAN_FIELD_LABELS: Record<string, string> = {
  max_agents: 'Max Agents',
  max_storage_mb: 'Storage (MB)',
  requests_per_minute: 'Requests / min',
  spawns_per_hour: 'Spawns / hour',
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
  const [tab, setTab] = useState<'users' | 'usage' | 'plans' | 'settings'>('users')
  const [users, setUsers] = useState<UserInfo[]>([])
  const [totalUsers, setTotalUsers] = useState(0)
  const [plans, setPlans] = useState<Record<string, PlanLimits>>({})
  const [config, setConfig] = useState<Record<string, boolean>>({ docker_spawn_enabled: true })
  const [providerKeys, setProviderKeys] = useState<Record<string, string>>({})
  const [providerKeysDraft, setProviderKeysDraft] = useState<Record<string, string>>({})
  const [keySaveFlash, setKeySaveFlash] = useState<string | null>(null)
  const [usageSummary, setUsageSummary] = useState<Record<string, number>>({})
  const [usageLogs, setUsageLogs] = useState<UsageLog[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const loadUsers = useCallback(async () => {
    try {
      const [usersData, plansData, cfgData, keys] = await Promise.all([fetchUsers(), fetchPlans(), fetchConfig(), fetchProviderKeys()])
      setUsers(usersData.users)
      setTotalUsers(usersData.total)
      setPlans(plansData)
      setConfig(cfgData)
      setProviderKeys(keys)
      setProviderKeysDraft(keys)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load users')
    }
  }, [])

  const loadUsage = useCallback(async () => {
    try {
      const [summary, logs] = await Promise.all([fetchUsageSummary(), fetchUsageLogs()])
      setUsageSummary(summary)
      setUsageLogs(logs)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load usage')
    }
  }, [])

  useEffect(() => {
    setLoading(true)
    Promise.all([loadUsers(), loadUsage()]).finally(() => setLoading(false))
  }, [loadUsers, loadUsage])

  const handleDelete = async (userId: string) => {
    try {
      await deleteUser(userId)
      loadUsers()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete user')
    }
  }

  const refresh = () => {
    setLoading(true)
    Promise.all([loadUsers(), loadUsage()]).finally(() => setLoading(false))
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
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="h-5 w-5 text-violet-400" />
            <h1 className="text-lg font-semibold text-zinc-100">Admin Panel</h1>
          </div>
          <button
            onClick={refresh}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {error && (
          <div className="rounded-md border border-red-900/50 bg-red-950/20 px-4 py-2 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 border-b border-zinc-800">
          <button
            onClick={() => setTab('users')}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm border-b-2 transition-colors ${
              tab === 'users'
                ? 'border-violet-500 text-zinc-100'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <Users className="h-4 w-4" />
            Users ({totalUsers})
          </button>
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
        </div>

        {loading && (
          <div className="flex justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
          </div>
        )}

        {/* Users tab */}
        {!loading && tab === 'users' && (
          <div className="space-y-2">
            {users.map((u) => (
              <UserRow
                key={u.id}
                user={u}
                plans={plans}
                onUpdate={loadUsers}
                onDelete={handleDelete}
                currentUserId={currentUser.id}
              />
            ))}
            {users.length === 0 && (
              <p className="text-center text-sm text-zinc-500 py-8">No users found</p>
            )}
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

        {/* Plans tab */}
        {!loading && tab === 'plans' && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {Object.entries(plans).map(([name, limits]) => (
              <PlanCard
                key={name}
                name={name}
                limits={limits}
                onSave={async (plan, newLimits) => {
                  await updatePlan(plan, newLimits)
                  await loadUsers()
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
