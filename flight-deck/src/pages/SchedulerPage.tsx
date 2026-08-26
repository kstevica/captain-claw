import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlarmClock,
  Loader2,
  RefreshCw,
  AlertCircle,
  Plus,
  Play,
  Trash2,
  Power,
  X,
  Check,
  MessageSquare,
  Radio,
  KeyRound,
} from 'lucide-react'
import { useAuthStore } from '../stores/authStore'

// ── Types ──

interface Job {
  id: string
  name: string
  schedule: string
  agent_slug: string
  agent_auth: string
  prompt: string
  flow_id?: string
  delivery_kind: 'whatsapp' | 'channel'
  delivery_target: string
  enabled: number
  ignore_quiet_hours: number
  created_at: string
  updated_at: string
  next_run_at: number | null
  last_run_at: number | null
  last_status: string
  last_result: string
}

interface AgentOption {
  id: string
  name: string
  port: number
}

// ── Auth helper ──
// The /scheduler routes are gated by FD_GLASSES_BRIDGE_TOKEN (via
// _check_token), NOT FD's admin auth. When that env var is unset (the
// common self-hosted case) no token is needed. When it IS set, the user
// pastes it once here; we persist it and append ?t= to every request.
const TOKEN_KEY = 'fd.scheduler.token'
function getToken(): string {
  try { return localStorage.getItem(TOKEN_KEY) || '' } catch { return '' }
}
function setToken(v: string) {
  try { v ? localStorage.setItem(TOKEN_KEY, v) : localStorage.removeItem(TOKEN_KEY) } catch { /* ignore */ }
}
function withToken(path: string): string {
  const t = getToken()
  if (!t) return path
  return path + (path.includes('?') ? '&' : '?') + 't=' + encodeURIComponent(t)
}
async function api<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  // Scheduler routes now require an authenticated caller. Send the FD JWT so
  // logged-in team members are authorized directly; the ?t= glasses token
  // (withToken) still covers glasses/bridge callers.
  const jwt = useAuthStore.getState().token
  const authHeader: Record<string, string> = jwt ? { Authorization: `Bearer ${jwt}` } : {}
  const r = await fetch(withToken(path), {
    ...init,
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json', ...authHeader, ...(init?.headers || {}) },
  })
  if (!r.ok) {
    const detail = await r.text().catch(() => '')
    throw new Error(`${r.status} ${detail || r.statusText}`)
  }
  // Some endpoints return {ok:true} only.
  const text = await r.text()
  return (text ? JSON.parse(text) : {}) as T
}

// ── Helpers ──

function fmtEpoch(epoch: number | null): string {
  if (!epoch) return '—'
  try {
    const d = new Date(epoch * 1000)
    const pad = (n: number) => String(n).padStart(2, '0')
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
  } catch { return '—' }
}
function relFuture(epoch: number | null): string {
  if (!epoch) return ''
  const secs = Math.round(epoch - Date.now() / 1000)
  if (secs < 0) return 'due'
  if (secs < 60) return `in ${secs}s`
  if (secs < 3600) return `in ${Math.round(secs / 60)}m`
  if (secs < 86400) return `in ${Math.round(secs / 3600)}h`
  return `in ${Math.round(secs / 86400)}d`
}

const SCHEDULE_HINT = 'every 30m · every 2h · daily 08:00 · weekly mon 09:00 · in 10m · once 2026-12-25T09:00:00'

const EMPTY_FORM = {
  name: '',
  schedule: 'daily 08:00',
  agent_slug: '',
  agent_auth: '',
  prompt: '',
  flow_id: '',
  delivery_kind: 'whatsapp' as 'whatsapp' | 'channel',
  delivery_target: '',
  ignore_quiet_hours: false,
}
type FormState = typeof EMPTY_FORM

// ── Page ──

export function SchedulerPage() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [agents, setAgents] = useState<AgentOption[]>([])
  const [flowList, setFlowList] = useState<{ id: string; name: string }[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const action: 'prompt' | 'flow' = form.flow_id ? 'flow' : 'prompt'
  const [editingId, setEditingId] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [runResult, setRunResult] = useState<Record<string, string>>({})
  const [tokenInput, setTokenInput] = useState(getToken())
  const [showToken, setShowToken] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const data = await api<Job[]>('/scheduler/jobs')
      setJobs(Array.isArray(data) ? data : [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  const loadAgents = useCallback(async () => {
    try {
      const data = await api<AgentOption[]>('/glasses/agents')
      setAgents(Array.isArray(data) ? data : [])
    } catch {
      setAgents([])
    }
  }, [])

  useEffect(() => { load(); loadAgents() }, [load, loadAgents])
  useEffect(() => {
    import('../services/flowsApi')
      .then((m) => m.listFlows())
      .then((fs) => setFlowList(fs.map((f) => ({ id: f.id, name: f.name }))))
      .catch(() => setFlowList([]))
  }, [])

  const resetForm = () => {
    setForm(EMPTY_FORM)
    setEditingId(null)
    setShowForm(false)
  }

  const startEdit = (job: Job) => {
    setForm({
      name: job.name,
      schedule: job.schedule,
      agent_slug: job.agent_slug,
      agent_auth: job.agent_auth,
      prompt: job.prompt,
      flow_id: job.flow_id || '',
      delivery_kind: job.delivery_kind,
      delivery_target: job.delivery_target,
      ignore_quiet_hours: job.ignore_quiet_hours === 1,
    })
    setEditingId(job.id)
    setShowForm(true)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const submit = async () => {
    if (!form.schedule.trim() || (!form.prompt.trim() && !form.flow_id) || !form.delivery_target.trim()) {
      setError('schedule, a prompt or flow, and a delivery target are required')
      return
    }
    setSaving(true)
    setError('')
    try {
      if (editingId) {
        await api(`/scheduler/jobs/${editingId}`, {
          method: 'PATCH', body: JSON.stringify(form),
        })
      } else {
        await api('/scheduler/jobs', {
          method: 'POST', body: JSON.stringify({ ...form, enabled: true }),
        })
      }
      resetForm()
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const toggleEnabled = async (job: Job) => {
    setBusyId(job.id)
    try {
      await api(`/scheduler/jobs/${job.id}`, {
        method: 'PATCH', body: JSON.stringify({ enabled: job.enabled !== 1 }),
      })
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyId(null)
    }
  }

  const runNow = async (job: Job) => {
    setBusyId(job.id)
    setRunResult((m) => ({ ...m, [job.id]: 'running…' }))
    try {
      const res = await api<{ status: string; result_preview: string }>(
        `/scheduler/jobs/${job.id}/run`, { method: 'POST' },
      )
      setRunResult((m) => ({
        ...m,
        [job.id]: `${res.status} — ${res.result_preview || '(no text)'}`,
      }))
      await load()
    } catch (e) {
      setRunResult((m) => ({ ...m, [job.id]: 'error: ' + (e instanceof Error ? e.message : String(e)) }))
    } finally {
      setBusyId(null)
    }
  }

  const remove = async (job: Job) => {
    if (!confirm(`Delete job "${job.name || job.id}"?`)) return
    setBusyId(job.id)
    try {
      await api(`/scheduler/jobs/${job.id}`, { method: 'DELETE' })
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyId(null)
    }
  }

  const saveToken = () => {
    setToken(tokenInput.trim())
    setShowToken(false)
    load(); loadAgents()
  }

  const sorted = useMemo(
    () => [...jobs].sort((a, b) => (a.name || a.id).localeCompare(b.name || b.id)),
    [jobs],
  )

  return (
    <div className="h-full overflow-y-auto bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-4xl px-4 py-6">
        {/* Header */}
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-400">
            <AlarmClock className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <h1 className="text-xl font-semibold">Scheduler</h1>
            <p className="text-sm text-zinc-500">
              Run agent prompts on a timer; deliver replies to WhatsApp or a channel.
            </p>
          </div>
          <button onClick={() => setShowToken((v) => !v)}
            className="rounded-lg border border-zinc-800 p-2 text-zinc-500 hover:text-zinc-300" title="API token">
            <KeyRound className="h-4 w-4" />
          </button>
          <button onClick={load}
            className="rounded-lg border border-zinc-800 p-2 text-zinc-400 hover:text-zinc-200" title="Refresh">
            <RefreshCw className="h-4 w-4" />
          </button>
          <button onClick={() => { resetForm(); setShowForm(true) }}
            className="flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3 py-2 text-sm font-medium text-emerald-950 hover:bg-emerald-400">
            <Plus className="h-4 w-4" /> New job
          </button>
        </div>

        {showToken && (
          <div className="mb-4 rounded-xl border border-zinc-800 bg-zinc-900 p-3">
            <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-zinc-500">
              FD_GLASSES_BRIDGE_TOKEN (only if your server sets one)
            </label>
            <div className="flex gap-2">
              <input value={tokenInput} onChange={(e) => setTokenInput(e.target.value)}
                placeholder="leave empty if no token is configured"
                className="flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm focus:border-emerald-500/50 focus:outline-none" />
              <button onClick={saveToken}
                className="rounded-lg bg-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-600">Save</button>
            </div>
          </div>
        )}

        {error && (
          <div className="mb-4 flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">
            <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" />
            <span className="break-all">{error}</span>
          </div>
        )}

        {/* Create / edit form */}
        {showForm && (
          <div className="mb-6 rounded-xl border border-zinc-800 bg-zinc-900 p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold">{editingId ? 'Edit job' : 'New job'}</h2>
              <button onClick={resetForm} className="text-zinc-500 hover:text-zinc-300"><X className="h-4 w-4" /></button>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Field label="Name">
                <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="Morning briefing" className={inputCls} />
              </Field>
              <Field label="Schedule" hint={SCHEDULE_HINT}>
                <input value={form.schedule} onChange={(e) => setForm({ ...form, schedule: e.target.value })}
                  className={inputCls} />
              </Field>
              <Field label="Agent">
                {agents.length > 0 ? (
                  <select value={form.agent_slug} onChange={(e) => setForm({ ...form, agent_slug: e.target.value })}
                    className={inputCls}>
                    <option value="">— pick a running agent —</option>
                    {agents.map((a) => (
                      <option key={a.id} value={a.id}>{a.name} ({a.id})</option>
                    ))}
                  </select>
                ) : (
                  <input value={form.agent_slug} onChange={(e) => setForm({ ...form, agent_slug: e.target.value })}
                    placeholder="agent slug" className={inputCls} />
                )}
              </Field>
              <Field label="Agent auth token" hint="from agent config.yaml web.auth_token (optional)">
                <input value={form.agent_auth} onChange={(e) => setForm({ ...form, agent_auth: e.target.value })}
                  placeholder="optional override" className={inputCls} />
              </Field>
              <Field label="Delivery">
                <select value={form.delivery_kind}
                  onChange={(e) => setForm({ ...form, delivery_kind: e.target.value as 'whatsapp' | 'channel' })}
                  className={inputCls}>
                  <option value="whatsapp">WhatsApp (WAID)</option>
                  <option value="channel">Channel (glasses / bridges)</option>
                </select>
              </Field>
              <Field label={form.delivery_kind === 'whatsapp' ? 'WhatsApp number (no +)' : 'Channel name'}>
                <input value={form.delivery_target} onChange={(e) => setForm({ ...form, delivery_target: e.target.value })}
                  placeholder={form.delivery_kind === 'whatsapp' ? '385976707736' : 'skchannel'}
                  className={inputCls} />
              </Field>
            </div>
            <div className="mt-3 flex items-center gap-1 rounded-lg border border-zinc-700 p-0.5 text-xs w-fit">
              <button
                onClick={() => setForm({ ...form, flow_id: '' })}
                className={`rounded-md px-2.5 py-1 ${action === 'prompt' ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}
              >Prompt</button>
              <button
                onClick={() => setForm({ ...form, prompt: '', flow_id: form.flow_id || (flowList[0]?.id || '') })}
                className={`rounded-md px-2.5 py-1 ${action === 'flow' ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}
              >Flow</button>
            </div>
            {action === 'flow' ? (
              <Field label="Run flow" className="mt-2">
                <select value={form.flow_id} onChange={(e) => setForm({ ...form, flow_id: e.target.value })} className={inputCls}>
                  <option value="">(pick a flow)</option>
                  {flowList.map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
                </select>
                <p className="mt-1 text-[11px] text-zinc-600">Runs the flow on schedule and delivers its output. Use self-contained flows (no “ask user” steps).</p>
              </Field>
            ) : (
              <Field label="Prompt" className="mt-2">
                <textarea value={form.prompt} onChange={(e) => setForm({ ...form, prompt: e.target.value })}
                  rows={3} placeholder="Compile my morning briefing: today's calendar, flagged emails, and one follow-up I owe."
                  className={inputCls + ' resize-y'} />
              </Field>
            )}
            <label className="mt-3 flex items-center gap-2 text-sm text-zinc-400">
              <input type="checkbox" checked={form.ignore_quiet_hours}
                onChange={(e) => setForm({ ...form, ignore_quiet_hours: e.target.checked })}
                className="h-4 w-4 rounded border-zinc-600 bg-zinc-800" />
              Ignore quiet hours (FD_SCHEDULER_QUIET_HOURS)
            </label>
            <div className="mt-4 flex gap-2">
              <button onClick={submit} disabled={saving}
                className="flex items-center gap-1.5 rounded-lg bg-emerald-500 px-4 py-2 text-sm font-medium text-emerald-950 hover:bg-emerald-400 disabled:opacity-50">
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                {editingId ? 'Save changes' : 'Create job'}
              </button>
              <button onClick={resetForm} className="rounded-lg border border-zinc-700 px-4 py-2 text-sm hover:bg-zinc-800">Cancel</button>
            </div>
          </div>
        )}

        {/* Job list */}
        {loading ? (
          <div className="flex items-center justify-center py-16 text-zinc-500">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : sorted.length === 0 ? (
          <div className="rounded-xl border border-dashed border-zinc-800 py-16 text-center text-zinc-500">
            No scheduled jobs yet. Click “New job” to create one.
          </div>
        ) : (
          <div className="space-y-3">
            {sorted.map((job) => (
              <div key={job.id}
                className={`rounded-xl border bg-zinc-900 p-4 ${job.enabled === 1 ? 'border-zinc-800' : 'border-zinc-800/50 opacity-60'}`}>
                <div className="flex items-start gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold truncate">{job.name || '(unnamed)'}</span>
                      <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-xs text-cyan-400/80">{job.schedule}</span>
                      <span className="flex items-center gap-1 rounded bg-zinc-800 px-1.5 py-0.5 text-xs text-zinc-400">
                        {job.delivery_kind === 'whatsapp'
                          ? <><MessageSquare className="h-3 w-3" /> {job.delivery_target}</>
                          : <><Radio className="h-3 w-3" /> {job.delivery_target}</>}
                      </span>
                    </div>
                    <p className="mt-1 line-clamp-2 text-sm text-zinc-400">{job.prompt}</p>
                    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs text-zinc-500">
                      <span>agent: <span className="text-zinc-400">{job.agent_slug || '—'}</span></span>
                      <span>next: <span className="text-zinc-400">{fmtEpoch(job.next_run_at)} {relFuture(job.next_run_at)}</span></span>
                      {job.last_run_at ? <span>last: <span className="text-zinc-400">{fmtEpoch(job.last_run_at)}</span></span> : null}
                      {job.last_status ? (
                        <span>status: <span className={job.last_status.startsWith('ok') ? 'text-emerald-400' : 'text-amber-400'}>{job.last_status}</span></span>
                      ) : null}
                    </div>
                    {runResult[job.id] && (
                      <div className="mt-2 rounded-lg bg-zinc-950 p-2 font-mono text-xs text-zinc-400 break-words">
                        {runResult[job.id]}
                      </div>
                    )}
                  </div>
                  <div className="flex flex-shrink-0 items-center gap-1">
                    <button onClick={() => runNow(job)} disabled={busyId === job.id}
                      className="rounded-lg border border-zinc-700 p-2 text-zinc-400 hover:border-emerald-500/40 hover:text-emerald-400 disabled:opacity-50" title="Run now">
                      {busyId === job.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                    </button>
                    <button onClick={() => toggleEnabled(job)} disabled={busyId === job.id}
                      className={`rounded-lg border border-zinc-700 p-2 hover:bg-zinc-800 disabled:opacity-50 ${job.enabled === 1 ? 'text-emerald-400' : 'text-zinc-500'}`}
                      title={job.enabled === 1 ? 'Disable' : 'Enable'}>
                      <Power className="h-4 w-4" />
                    </button>
                    <button onClick={() => startEdit(job)}
                      className="rounded-lg border border-zinc-700 px-3 py-2 text-xs hover:bg-zinc-800">Edit</button>
                    <button onClick={() => remove(job)} disabled={busyId === job.id}
                      className="rounded-lg border border-zinc-700 p-2 text-zinc-500 hover:border-red-500/40 hover:text-red-400 disabled:opacity-50" title="Delete">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Small presentational helpers ──

const inputCls =
  'w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-100 placeholder-zinc-600 focus:border-emerald-500/50 focus:outline-none'

function Field({ label, hint, className, children }:
  { label: string; hint?: string; className?: string; children: React.ReactNode }) {
  return (
    <div className={className}>
      <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-zinc-500">{label}</label>
      {children}
      {hint && <p className="mt-1 text-xs text-zinc-600">{hint}</p>}
    </div>
  )
}
