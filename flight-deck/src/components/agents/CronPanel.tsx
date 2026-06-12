import { useState, useEffect, useCallback } from 'react'
import {
  Clock, Loader2, AlertTriangle, RefreshCw, X, Play, Pause, Trash2, CheckCircle2, Plus,
} from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'

interface CronJob {
  id: string
  kind: string
  payload: Record<string, unknown>
  schedule: unknown
  schedule_text?: string
  session_id: string
  enabled: boolean
  created_at?: string
  updated_at?: string
  last_run_at?: string | null
  next_run_at?: string | null
  last_status?: string
  last_error?: string
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { ...(init?.headers as Record<string, string> | undefined) }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
  let res = await fetch(`/fd${path}`, { ...init, headers, credentials: 'include' })
  if (res.status === 401 && authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) {
      const t2 = useAuthStore.getState().token
      const h2: Record<string, string> = { ...(init?.headers as Record<string, string> | undefined) }
      if (t2) h2['Authorization'] = `Bearer ${t2}`
      res = await fetch(`/fd${path}`, { ...init, headers: h2, credentials: 'include' })
    }
  }
  if (!res.ok) {
    const b = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(b.error || b.detail || `${res.status}`)
  }
  return res.json()
}

function relTime(iso?: string | null): string {
  if (!iso) return ''
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return ''
  const s = Math.round((t - Date.now()) / 1000)
  const a = Math.abs(s)
  const unit =
    a < 60 ? `${a}s` : a < 3600 ? `${Math.round(a / 60)}m` : a < 86400 ? `${Math.round(a / 3600)}h` : `${Math.round(a / 86400)}d`
  return s >= 0 ? `in ${unit}` : `${unit} ago`
}

function taskDesc(job: CronJob): string {
  const p = (job.payload || {}) as Record<string, unknown>
  return (
    (p.text as string) || (p.path as string) || (p.workflow as string) || job.kind || '(task)'
  )
}

interface CronPanelProps {
  host: string
  port: number
  auth?: string
  agentName: string
  onClose: () => void
}

export function CronPanel({ host, port, auth, agentName, onClose }: CronPanelProps) {
  const [jobs, setJobs] = useState<CronJob[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [newSchedule, setNewSchedule] = useState('')
  const [newTask, setNewTask] = useState('')
  const [createBusy, setCreateBusy] = useState(false)
  const tokenQs = auth ? `?token=${encodeURIComponent(auth)}` : ''
  const base = `/agent-cron/${host}/${port}`

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fdFetch<CronJob[]>(`${base}${tokenQs}`)
      setJobs(Array.isArray(data) ? data : [])
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [host, port, auth])

  useEffect(() => { refresh() }, [refresh])

  const act = useCallback(
    async (job: CronJob, kind: 'run' | 'pause' | 'resume' | 'remove') => {
      if (kind === 'remove' && !confirm(`Remove cron job ${job.id.slice(0, 8)}?`)) return
      setBusy(job.id + kind)
      try {
        if (kind === 'remove') {
          await fdFetch(`${base}/${job.id}${tokenQs}`, { method: 'DELETE' })
        } else {
          await fdFetch(`${base}/${job.id}/${kind}${tokenQs}`, { method: 'POST' })
        }
        await refresh()
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusy(null)
      }
    },
    [base, tokenQs, refresh],
  )

  const createJob = useCallback(async () => {
    const schedule = newSchedule.trim()
    const task = newTask.trim()
    if (!schedule || !task) { setError('Schedule and task are both required.'); return }
    setCreateBusy(true)
    try {
      await fdFetch(`${base}${tokenQs}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: 'prompt', schedule, task }),
      })
      setNewSchedule(''); setNewTask(''); setCreating(false); setError('')
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setCreateBusy(false)
    }
  }, [base, tokenQs, newSchedule, newTask, refresh])

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <Clock className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          <div className="min-w-0 flex-1">
            <div className="text-sm font-semibold text-zinc-100">Cron jobs</div>
            <div className="truncate text-xs text-zinc-500">{agentName}</div>
          </div>
          <button
            onClick={() => setCreating((v) => !v)}
            className={`inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium ${creating ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300' : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'}`}
            title="Create a new cron job"
          >
            <Plus className="h-3.5 w-3.5" /> New job
          </button>
          <button onClick={refresh} className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Refresh">
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button onClick={onClose} className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* New-job form */}
        {creating && (
          <div className="border-b border-zinc-800 bg-zinc-950/40 p-3 space-y-2">
            <div>
              <label className="mb-1 block text-[11px] uppercase tracking-wider text-zinc-500">Schedule</label>
              <input
                value={newSchedule}
                onChange={(e) => setNewSchedule(e.target.value)}
                placeholder="in 5m  ·  every 15m  ·  daily 09:00  ·  weekly mon 10:00"
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-emerald-500/50 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-[11px] uppercase tracking-wider text-zinc-500">Task</label>
              <textarea
                value={newTask}
                onChange={(e) => setNewTask(e.target.value)}
                rows={3}
                placeholder="What should the agent do when this fires? e.g. Check my inbox and summarize anything new."
                className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-emerald-500/50 focus:outline-none"
              />
            </div>
            <div className="flex items-center justify-end gap-2">
              <button onClick={() => { setCreating(false); setError('') }} className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
                Cancel
              </button>
              <button
                onClick={createJob}
                disabled={createBusy || !newSchedule.trim() || !newTask.trim()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-300 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-100 disabled:opacity-40 dark:border-emerald-700/60 dark:bg-emerald-950/40 dark:text-emerald-300 dark:hover:bg-emerald-900/40"
              >
                {createBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />} Create job
              </button>
            </div>
          </div>
        )}

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-3">
          {error && (
            <div className="mb-3 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-300">
              <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
            </div>
          )}
          {loading && jobs.length === 0 ? (
            <div className="flex items-center justify-center gap-2 py-10 text-sm text-zinc-500">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading…
            </div>
          ) : jobs.length === 0 ? (
            <div className="py-10 text-center text-sm text-zinc-500">No cron jobs scheduled.</div>
          ) : (
            <div className="space-y-2">
              {jobs.map((job) => {
                const paused = !job.enabled
                return (
                  <div key={job.id} className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
                    <div className="mb-1.5 flex items-center gap-2 text-xs">
                      <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-emerald-600 dark:text-emerald-400/80">{job.kind}</span>
                      <span className="font-mono text-zinc-500">{job.id.slice(0, 8)}</span>
                      {paused ? (
                        <span className="rounded-full bg-amber-100 px-2 py-0.5 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400">paused</span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-0.5 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400">
                          <CheckCircle2 className="h-3 w-3" /> active
                        </span>
                      )}
                      <span className="ml-auto text-zinc-500">
                        {job.schedule_text || '—'}
                        {job.next_run_at && !paused ? ` · ${relTime(job.next_run_at)}` : ''}
                      </span>
                    </div>
                    <p className="whitespace-pre-wrap break-words text-sm text-zinc-200">{taskDesc(job)}</p>
                    {job.last_status && (
                      <div className="mt-1 text-[11px] text-zinc-500">
                        last: {job.last_status}
                        {job.last_run_at ? ` (${relTime(job.last_run_at)})` : ''}
                        {job.last_error ? <span className="text-red-600 dark:text-red-400/80"> — {job.last_error}</span> : null}
                      </div>
                    )}
                    <div className="mt-2 flex items-center gap-1">
                      <button
                        onClick={() => act(job, 'run')}
                        disabled={busy === job.id + 'run'}
                        className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40"
                      >
                        {busy === job.id + 'run' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />} Run now
                      </button>
                      {paused ? (
                        <button
                          onClick={() => act(job, 'resume')}
                          disabled={busy === job.id + 'resume'}
                          className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-emerald-600 hover:bg-zinc-800 disabled:opacity-40 dark:text-emerald-400"
                        >
                          <Play className="h-3 w-3" /> Resume
                        </button>
                      ) : (
                        <button
                          onClick={() => act(job, 'pause')}
                          disabled={busy === job.id + 'pause'}
                          className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-amber-600 hover:bg-zinc-800 disabled:opacity-40 dark:text-amber-400"
                        >
                          <Pause className="h-3 w-3" /> Pause
                        </button>
                      )}
                      <button
                        onClick={() => act(job, 'remove')}
                        disabled={busy === job.id + 'remove'}
                        className="ml-auto inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-red-600 hover:bg-red-100 hover:text-red-700 disabled:opacity-40 dark:text-red-400/80 dark:hover:bg-red-950/30 dark:hover:text-red-300"
                      >
                        <Trash2 className="h-3 w-3" /> Remove
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
