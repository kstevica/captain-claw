import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  CalendarDays,
  Inbox,
  Lightbulb,
  Sparkles,
  Clock,
  RefreshCw,
  Loader2,
  AlertCircle,
  Check,
  Ban,
  X,
  Eye,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useContainerStore } from '../stores/containerStore'
import { useProcessStore } from '../stores/processStore'
import { useLocalAgentStore } from '../stores/localAgentStore'
import { useAuthStore } from '../stores/authStore'
import { useGroupStore } from '../stores/groupStore'
import { useChatStore } from '../stores/chatStore'

// ── Types ──

interface AgentTarget {
  id: string
  name: string
  description: string
  model: string
  kind: 'docker' | 'process' | 'local'
  host: string
  port: number
  auth: string
}

interface PendingInsight {
  id: string
  content: string
  category: string
  importance: number
  created_at: string
}

interface Reflection {
  timestamp?: string
  summary?: string
  topics_reviewed?: string[]
}

interface CronJob {
  id: string
  kind: string
  payload: Record<string, unknown>
  schedule: { _text?: string; type?: string }
  enabled: boolean
  next_run_at?: string | null
  last_run_at?: string | null
  last_status?: string | null
}

interface Intuition {
  id: string
  content: string
  thread_type?: string
  confidence?: number
  created_at?: string
}

interface AgentToday {
  target: AgentTarget
  loading: boolean
  error: string | null
  insights: PendingInsight[]
  reflection: Reflection | null
  reflectionsList: Reflection[]
  crons: CronJob[]
  intuitions: Intuition[]
}

// ── Helpers ──

function fdAuthHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = {}
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

async function jget(url: string): Promise<unknown> {
  const res = await fetch(url, { headers: fdAuthHeaders(), credentials: 'include' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

function relativeTime(iso?: string | null): string {
  if (!iso) return ''
  const t = new Date(iso).getTime()
  if (!t) return ''
  const diff = Date.now() - t
  const abs = Math.abs(diff)
  const s = Math.floor(abs / 1000)
  if (s < 60) return diff >= 0 ? `${s}s ago` : `in ${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return diff >= 0 ? `${m}m ago` : `in ${m}m`
  const h = Math.floor(m / 60)
  if (h < 24) return diff >= 0 ? `${h}h ago` : `in ${h}h`
  const d = Math.floor(h / 24)
  return diff >= 0 ? `${d}d ago` : `in ${d}d`
}

function isToday(iso?: string | null): boolean {
  if (!iso) return false
  const t = new Date(iso)
  if (isNaN(t.getTime())) return false
  const now = new Date()
  return (
    t.getFullYear() === now.getFullYear() &&
    t.getMonth() === now.getMonth() &&
    t.getDate() === now.getDate()
  )
}

// ── Page ──

export function TodayPage() {
  const containers = useContainerStore((s) => s.containers)
  const processes = useProcessStore((s) => s.processes)
  const localAgents = useLocalAgentStore((s) => s.agents)
  const groups = useGroupStore((s) => s.groups)
  const chatSessions = useChatStore((s) => s.sessions)

  const [data, setData] = useState<Map<string, AgentToday>>(new Map())
  const [refreshing, setRefreshing] = useState(false)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [detail, setDetail] = useState<{ target: AgentTarget; kind: 'insights' | 'intuitions' | 'crons' | 'reflections' } | null>(null)
  const [search, setSearch] = useState('')
  const [modelFilter, setModelFilter] = useState('')
  const [groupFilter, setGroupFilter] = useState('')
  const [roleFilter, setRoleFilter] = useState('')

  const targets: AgentTarget[] = useMemo(() => {
    const out: AgentTarget[] = []
    for (const c of containers) {
      if (c.web_port) {
        out.push({
          id: c.id,
          name: c.agent_name || c.name,
          description: c.description || '',
          model: chatSessions.get(c.id)?.activeModel || '',
          kind: 'docker',
          host: 'localhost',
          port: c.web_port,
          auth: c.web_auth || '',
        })
      }
    }
    for (const p of processes) {
      if (p.web_port) {
        out.push({
          id: p.slug,
          name: p.name,
          description: p.description || '',
          model: p.model || '',
          kind: 'process',
          host: 'localhost',
          port: p.web_port,
          auth: p.web_auth || '',
        })
      }
    }
    for (const a of localAgents) {
      out.push({
        id: a.id,
        name: a.name,
        description: a.description || '',
        model: chatSessions.get(a.id)?.activeModel || '',
        kind: 'local',
        host: a.host,
        port: a.port,
        auth: a.authToken || '',
      })
    }
    return out
  }, [containers, processes, localAgents, chatSessions])

  // Filter option lists
  const modelOptions = useMemo(() => {
    const set = new Set<string>()
    for (const t of targets) if (t.model) set.add(t.model)
    return Array.from(set).sort()
  }, [targets])

  const groupOptions = useMemo(
    () => groups.filter((g) => g.type === 'group').sort((a, b) => a.name.localeCompare(b.name)),
    [groups]
  )
  const roleOptions = useMemo(
    () => groups.filter((g) => g.type === 'role').sort((a, b) => a.name.localeCompare(b.name)),
    [groups]
  )

  const filteredTargets = useMemo(() => {
    const q = search.trim().toLowerCase()
    const groupMembers = groupFilter ? new Set(groups.find((g) => g.id === groupFilter)?.agentIds || []) : null
    const roleMembers = roleFilter ? new Set(groups.find((g) => g.id === roleFilter)?.agentIds || []) : null
    return targets.filter((t) => {
      if (q && !t.name.toLowerCase().includes(q) && !t.description.toLowerCase().includes(q)) return false
      if (modelFilter && t.model !== modelFilter) return false
      if (groupMembers && !groupMembers.has(t.id)) return false
      if (roleMembers && !roleMembers.has(t.id)) return false
      return true
    })
  }, [targets, search, modelFilter, groupFilter, roleFilter, groups])

  const fetchAgent = useCallback(async (target: AgentTarget): Promise<AgentToday> => {
    const tokenQs = target.auth ? `token=${encodeURIComponent(target.auth)}` : ''
    const base = `/fd`
    const hp = `${encodeURIComponent(target.host)}/${target.port}`
    const insightsUrl = `${base}/agent-insights/${hp}/pending${tokenQs ? '?' + tokenQs : ''}`
    const reflectionsUrl = `${base}/agent-reflections/${hp}?limit=10${tokenQs ? '&' + tokenQs : ''}`
    const cronUrl = `${base}/agent-cron/${hp}${tokenQs ? '?' + tokenQs : ''}`
    const intuitionsUrl = `${base}/agent-intuitions/${hp}?limit=20${tokenQs ? '&' + tokenQs : ''}`

    const result: AgentToday = {
      target,
      loading: false,
      error: null,
      insights: [],
      reflection: null,
      reflectionsList: [],
      crons: [],
      intuitions: [],
    }

    const settled = await Promise.allSettled([
      jget(insightsUrl),
      jget(reflectionsUrl),
      jget(cronUrl),
      jget(intuitionsUrl),
    ])

    const errors: string[] = []

    if (settled[0].status === 'fulfilled') {
      const v = settled[0].value as { items?: PendingInsight[] }
      result.insights = Array.isArray(v?.items) ? v.items : []
    } else errors.push(`insights: ${settled[0].reason}`)

    if (settled[1].status === 'fulfilled') {
      const v = settled[1].value as Reflection[] | { error?: string }
      const list = Array.isArray(v) ? v : []
      result.reflectionsList = list
      result.reflection = list[0] || null
    } else errors.push(`reflections: ${settled[1].reason}`)

    if (settled[2].status === 'fulfilled') {
      const v = settled[2].value as CronJob[] | { error?: string }
      result.crons = Array.isArray(v) ? v : []
    } else errors.push(`cron: ${settled[2].reason}`)

    if (settled[3].status === 'fulfilled') {
      const v = settled[3].value as { items?: Intuition[] }
      result.intuitions = Array.isArray(v?.items) ? v.items : []
    } else errors.push(`intuitions: ${settled[3].reason}`)

    if (errors.length === 4) result.error = errors.join('; ')
    return result
  }, [])

  const fetchAll = useCallback(async () => {
    setRefreshing(true)
    const results = await Promise.all(
      targets.map(async (t) => {
        try {
          return await fetchAgent(t)
        } catch (e) {
          return {
            target: t, loading: false,
            error: e instanceof Error ? e.message : String(e),
            insights: [], reflection: null, reflectionsList: [], crons: [], intuitions: [],
          } as AgentToday
        }
      })
    )
    const m = new Map<string, AgentToday>()
    for (const r of results) m.set(r.target.id, r)
    setData(m)
    setRefreshing(false)
  }, [targets, fetchAgent])

  useEffect(() => {
    fetchAll()
    // Light auto-refresh every 60s
    const t = setInterval(fetchAll, 60000)
    return () => clearInterval(t)
  }, [fetchAll])

  // Close detail modal on Escape
  useEffect(() => {
    if (!detail) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setDetail(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [detail])

  // Aggregate counts (over filtered set)
  const agg = useMemo(() => {
    let insights = 0
    let intuitions = 0
    let cronsToday = 0
    let cronsActive = 0
    let reflections = 0
    for (const t of filteredTargets) {
      const d = data.get(t.id)
      if (!d) continue
      insights += d.insights.length
      intuitions += d.intuitions.length
      reflections += d.reflectionsList.length
      for (const c of d.crons) {
        if (c.enabled) cronsActive += 1
        if (isToday(c.next_run_at) || isToday(c.last_run_at)) cronsToday += 1
      }
    }
    return { insights, intuitions, cronsToday, cronsActive, reflections, agents: filteredTargets.length }
  }, [data, filteredTargets])

  const approveInsight = useCallback(async (target: AgentTarget, id: string) => {
    setBusyId(id)
    try {
      const params = new URLSearchParams()
      if (target.auth) params.set('token', target.auth)
      params.set('supersede', '1')
      const res = await fetch(
        `/fd/agent-insights/${encodeURIComponent(target.host)}/${target.port}/pending/${encodeURIComponent(id)}/approve?${params}`,
        { method: 'POST', headers: fdAuthHeaders(), credentials: 'include' }
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setData((prev) => {
        const next = new Map(prev)
        const cur = next.get(target.id)
        if (cur) {
          next.set(target.id, { ...cur, insights: cur.insights.filter((i) => i.id !== id) })
        }
        return next
      })
    } catch (e) {
      alert(`Approve failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusyId(null)
    }
  }, [])

  const rejectInsight = useCallback(async (target: AgentTarget, id: string) => {
    setBusyId(id)
    try {
      const params = new URLSearchParams()
      if (target.auth) params.set('token', target.auth)
      const res = await fetch(
        `/fd/agent-insights/${encodeURIComponent(target.host)}/${target.port}/pending/${encodeURIComponent(id)}/reject?${params}`,
        { method: 'POST', headers: fdAuthHeaders(), credentials: 'include' }
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setData((prev) => {
        const next = new Map(prev)
        const cur = next.get(target.id)
        if (cur) {
          next.set(target.id, { ...cur, insights: cur.insights.filter((i) => i.id !== id) })
        }
        return next
      })
    } catch (e) {
      alert(`Reject failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusyId(null)
    }
  }, [])

  if (targets.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center bg-zinc-950 text-zinc-500">
        <CalendarDays className="mb-3 h-10 w-10 text-zinc-700" />
        <div className="text-sm">No agents reachable yet.</div>
        <div className="mt-1 text-xs text-zinc-600">Spawn a Captain Claw agent to populate Today.</div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-zinc-950">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-6 py-4">
        <div className="flex items-center gap-3">
          <CalendarDays className="h-5 w-5 text-violet-400" />
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">Today</h1>
            <p className="text-xs text-zinc-500">
              Pending insights, intuitions, scheduled work and reflections across all Flight Deck agents
            </p>
          </div>
        </div>
        <button
          onClick={fetchAll}
          disabled={refreshing}
          className="flex items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
        >
          {refreshing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-6 py-3">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search name or description…"
          className="min-w-[220px] flex-1 rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />
        <select
          value={modelFilter}
          onChange={(e) => setModelFilter(e.target.value)}
          className="rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
        >
          <option value="">All models</option>
          {modelOptions.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <select
          value={groupFilter}
          onChange={(e) => setGroupFilter(e.target.value)}
          className="rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
        >
          <option value="">All groups</option>
          {groupOptions.map((g) => (
            <option key={g.id} value={g.id}>{g.name}</option>
          ))}
        </select>
        <select
          value={roleFilter}
          onChange={(e) => setRoleFilter(e.target.value)}
          className="rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
        >
          <option value="">All roles</option>
          {roleOptions.map((g) => (
            <option key={g.id} value={g.id}>{g.name}</option>
          ))}
        </select>
        {(search || modelFilter || groupFilter || roleFilter) && (
          <button
            onClick={() => { setSearch(''); setModelFilter(''); setGroupFilter(''); setRoleFilter('') }}
            className="rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
          >
            Clear
          </button>
        )}
        <span className="ml-auto text-[10px] text-zinc-500">
          {filteredTargets.length} / {targets.length} agents
        </span>
      </div>

      {/* Aggregate strip */}
      <div className="grid grid-cols-2 gap-3 border-b border-zinc-800 px-6 py-4 sm:grid-cols-5">
        <StatPill icon={<Inbox className="h-4 w-4" />} label="Pending insights" value={agg.insights} tone="amber" />
        <StatPill icon={<Sparkles className="h-4 w-4" />} label="Intuitions" value={agg.intuitions} tone="violet" />
        <StatPill icon={<Clock className="h-4 w-4" />} label="Crons today" value={agg.cronsToday} tone="sky" />
        <StatPill icon={<Clock className="h-4 w-4" />} label="Crons active" value={agg.cronsActive} tone="emerald" />
        <StatPill icon={<Lightbulb className="h-4 w-4" />} label="Reflections" value={agg.reflections} tone="rose" />
      </div>

      {/* Agent cards */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {filteredTargets.length === 0 && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-6 text-center text-xs text-zinc-500">
            No agents match your filters.
          </div>
        )}
        {filteredTargets.map((t) => {
          const d = data.get(t.id)
          return (
            <div key={t.id} className="rounded-xl border border-zinc-800 bg-zinc-900/60 overflow-hidden">
              <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2.5">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-zinc-100">{t.name}</span>
                  <span className="rounded-full border border-zinc-700 bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-300">
                    {t.kind}
                  </span>
                  <span className="text-[10px] text-zinc-500">{t.host}:{t.port}</span>
                </div>
                {d?.loading && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
              </div>

              {d?.error && (
                <div className="flex items-start gap-2 px-4 py-3 text-xs text-red-300">
                  <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                  {d.error}
                </div>
              )}

              {d && !d.error && (
                <div className="grid grid-cols-1 gap-px bg-zinc-800 lg:grid-cols-2 xl:grid-cols-4">
                  {/* Insights */}
                  <Section
                    icon={<Inbox className="h-3.5 w-3.5 text-amber-500" />}
                    title="Pending insights"
                    count={d.insights.length}
                    empty="No pending insights"
                    onView={() => setDetail({ target: t, kind: 'insights' })}
                  >
                    {d.insights.slice(0, 5).map((it) => (
                      <div key={it.id} className="rounded-md border border-zinc-800 bg-zinc-950/50 p-2">
                        <div className="text-[11px] text-zinc-200">{it.content}</div>
                        <div className="mt-1 flex items-center justify-between text-[9px] text-zinc-500">
                          <span>{it.category} · imp {it.importance}</span>
                          <div className="flex gap-1">
                            <button
                              disabled={busyId === it.id}
                              onClick={() => rejectInsight(t, it.id)}
                              className="rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-400 hover:border-red-500/30 hover:bg-red-500/10 hover:text-red-500 disabled:opacity-40"
                            >
                              <Ban className="h-3 w-3" />
                            </button>
                            <button
                              disabled={busyId === it.id}
                              onClick={() => approveInsight(t, it.id)}
                              className="rounded border border-emerald-500/30 bg-emerald-500/15 px-1.5 py-0.5 text-emerald-600 hover:bg-emerald-500/25 disabled:opacity-40"
                            >
                              <Check className="h-3 w-3" />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </Section>

                  {/* Intuitions */}
                  <Section
                    icon={<Sparkles className="h-3.5 w-3.5 text-violet-500" />}
                    title="Intuitions"
                    count={d.intuitions.length}
                    empty="No recent intuitions"
                    onView={() => setDetail({ target: t, kind: 'intuitions' })}
                  >
                    {d.intuitions.slice(0, 5).map((it) => (
                      <div key={it.id} className="rounded-md border border-zinc-800 bg-zinc-950/50 p-2">
                        <div className="text-[11px] text-zinc-200">{it.content}</div>
                        <div className="mt-1 text-[9px] text-zinc-500">
                          {it.thread_type || 'association'}
                          {typeof it.confidence === 'number' ? ` · ${(it.confidence * 100).toFixed(0)}%` : ''}
                          {it.created_at ? ` · ${relativeTime(it.created_at)}` : ''}
                        </div>
                      </div>
                    ))}
                  </Section>

                  {/* Cron */}
                  <Section
                    icon={<Clock className="h-3.5 w-3.5 text-sky-500" />}
                    title="Cron jobs"
                    count={d.crons.length}
                    empty="No cron jobs"
                    onView={() => setDetail({ target: t, kind: 'crons' })}
                  >
                    {d.crons.slice(0, 6).map((c) => (
                      <div key={c.id} className="rounded-md border border-zinc-800 bg-zinc-950/50 p-2">
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate text-[11px] text-zinc-200">
                            {c.kind}: {(c.payload?.text as string) || (c.payload?.prompt as string) || c.id.slice(0, 8)}
                          </span>
                          <span className={`shrink-0 rounded-full border px-1.5 py-0.5 text-[8px] ${
                            c.enabled
                              ? 'border-emerald-500/30 bg-emerald-500/15 text-emerald-600'
                              : 'border-zinc-700 bg-zinc-800 text-zinc-400'
                          }`}>
                            {c.enabled ? 'on' : 'off'}
                          </span>
                        </div>
                        <div className="mt-1 text-[9px] text-zinc-500">
                          {c.schedule?._text || c.schedule?.type || ''}
                          {c.next_run_at ? ` · next ${relativeTime(c.next_run_at)}` : ''}
                        </div>
                      </div>
                    ))}
                  </Section>

                  {/* Reflections */}
                  <Section
                    icon={<Lightbulb className="h-3.5 w-3.5 text-rose-500" />}
                    title="Latest reflection"
                    count={d.reflectionsList.length}
                    empty="No reflections yet"
                    onView={() => setDetail({ target: t, kind: 'reflections' })}
                  >
                    {d.reflection && (
                      <div className="rounded-md border border-zinc-800 bg-zinc-950/50 p-2">
                        <div className="text-[11px] text-zinc-200 line-clamp-4 whitespace-pre-wrap">
                          {d.reflection.summary || '(no summary)'}
                        </div>
                        {d.reflection.timestamp && (
                          <div className="mt-1 text-[9px] text-zinc-500">{relativeTime(d.reflection.timestamp)}</div>
                        )}
                        {d.reflection.topics_reviewed && d.reflection.topics_reviewed.length > 0 && (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {d.reflection.topics_reviewed.slice(0, 4).map((tp) => (
                              <span key={tp} className="rounded-full border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 text-[8px] text-zinc-300">
                                {tp}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </Section>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Detail modal */}
      {detail && (() => {
        const d = data.get(detail.target.id)
        if (!d) return null
        const titles: Record<typeof detail.kind, string> = {
          insights: 'Pending insights',
          intuitions: 'Intuitions',
          crons: 'Cron jobs',
          reflections: 'Reflections',
        }
        return (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
            onClick={() => setDetail(null)}
          >
            <div
              className="flex max-h-[85vh] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
                <div>
                  <div className="text-sm font-semibold text-zinc-100">{titles[detail.kind]}</div>
                  <div className="text-[10px] text-zinc-500">{detail.target.name} · {detail.target.host}:{detail.target.port}</div>
                </div>
                <button
                  onClick={() => setDetail(null)}
                  className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="flex-1 space-y-2 overflow-y-auto p-4">
                {detail.kind === 'insights' && d.insights.map((it) => (
                  <div key={it.id} className="rounded-md border border-zinc-800 bg-zinc-950/50 p-3">
                    <div className="fd-markdown text-xs text-zinc-200 leading-relaxed">
                      <Markdown remarkPlugins={[remarkGfm]}>{it.content}</Markdown>
                    </div>
                    <div className="mt-2 flex items-center justify-between text-[10px] text-zinc-500">
                      <span>{it.category} · imp {it.importance} · {relativeTime(it.created_at)}</span>
                      <div className="flex gap-1">
                        <button
                          disabled={busyId === it.id}
                          onClick={() => rejectInsight(detail.target, it.id)}
                          className="flex items-center gap-1 rounded border border-zinc-700 px-2 py-1 text-zinc-300 hover:border-red-500/30 hover:bg-red-500/10 hover:text-red-500 disabled:opacity-40"
                        >
                          <Ban className="h-3 w-3" /> Reject
                        </button>
                        <button
                          disabled={busyId === it.id}
                          onClick={() => approveInsight(detail.target, it.id)}
                          className="flex items-center gap-1 rounded border border-emerald-500/30 bg-emerald-500/15 px-2 py-1 text-emerald-600 hover:bg-emerald-500/25 disabled:opacity-40"
                        >
                          <Check className="h-3 w-3" /> Approve
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
                {detail.kind === 'intuitions' && d.intuitions.map((it) => (
                  <div key={it.id} className="rounded-md border border-zinc-800 bg-zinc-950/50 p-3">
                    <div className="fd-markdown text-xs text-zinc-200 leading-relaxed">
                      <Markdown remarkPlugins={[remarkGfm]}>{it.content}</Markdown>
                    </div>
                    <div className="mt-1.5 text-[10px] text-zinc-500">
                      {it.thread_type || 'association'}
                      {typeof it.confidence === 'number' ? ` · ${(it.confidence * 100).toFixed(0)}%` : ''}
                      {it.created_at ? ` · ${relativeTime(it.created_at)}` : ''}
                    </div>
                  </div>
                ))}
                {detail.kind === 'crons' && d.crons.map((c) => (
                  <div key={c.id} className="rounded-md border border-zinc-800 bg-zinc-950/50 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-medium text-zinc-200">
                        {c.kind}: {(c.payload?.text as string) || (c.payload?.prompt as string) || c.id.slice(0, 8)}
                      </span>
                      <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[9px] ${
                        c.enabled
                          ? 'border-emerald-500/30 bg-emerald-500/15 text-emerald-600'
                          : 'border-zinc-700 bg-zinc-800 text-zinc-400'
                      }`}>
                        {c.enabled ? 'enabled' : 'disabled'}
                      </span>
                    </div>
                    <div className="mt-1.5 text-[10px] text-zinc-500">
                      Schedule: {c.schedule?._text || c.schedule?.type || '—'}
                      {c.next_run_at && <> · next {relativeTime(c.next_run_at)}</>}
                      {c.last_run_at && <> · last {relativeTime(c.last_run_at)}</>}
                      {c.last_status && <> · {c.last_status}</>}
                    </div>
                  </div>
                ))}
                {detail.kind === 'reflections' && d.reflectionsList.map((r, i) => (
                  <div key={i} className="rounded-md border border-zinc-800 bg-zinc-950/50 p-3">
                    <div className="fd-markdown text-xs text-zinc-200 leading-relaxed">
                      <Markdown remarkPlugins={[remarkGfm]}>{r.summary || '(no summary)'}</Markdown>
                    </div>
                    {r.timestamp && (
                      <div className="mt-1.5 text-[10px] text-zinc-500">{relativeTime(r.timestamp)}</div>
                    )}
                    {r.topics_reviewed && r.topics_reviewed.length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {r.topics_reviewed.map((tp) => (
                          <span key={tp} className="rounded-full border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 text-[9px] text-zinc-300">
                            {tp}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )
      })()}
    </div>
  )
}

// ── Subcomponents ──

function StatPill({
  icon, label, value, tone,
}: {
  icon: React.ReactNode
  label: string
  value: number
  tone: 'amber' | 'violet' | 'sky' | 'emerald' | 'rose'
}) {
  const tones: Record<string, string> = {
    amber: 'text-amber-600 border-amber-500/30 bg-amber-500/15',
    violet: 'text-violet-600 border-violet-500/30 bg-violet-500/15',
    sky: 'text-sky-600 border-sky-500/30 bg-sky-500/15',
    emerald: 'text-emerald-600 border-emerald-500/30 bg-emerald-500/15',
    rose: 'text-rose-600 border-rose-500/30 bg-rose-500/15',
  }
  return (
    <div className={`flex items-center gap-3 rounded-lg border px-3 py-2 ${tones[tone]}`}>
      {icon}
      <div>
        <div className="text-lg font-semibold leading-none">{value}</div>
        <div className="mt-1 text-[10px] uppercase tracking-wide opacity-70">{label}</div>
      </div>
    </div>
  )
}

function Section({
  icon, title, count, empty, onView, children,
}: {
  icon: React.ReactNode
  title: string
  count: number
  empty: string
  onView?: () => void
  children?: React.ReactNode
}) {
  const hasChildren = Array.isArray(children) ? children.length > 0 : Boolean(children)
  return (
    <div className="flex flex-col bg-zinc-900/40 p-3">
      <div className="mb-2 flex items-center gap-1.5">
        {icon}
        <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">{title}</span>
        <span className="text-[9px] text-zinc-500">({count})</span>
        {onView && count > 0 && (
          <button
            onClick={onView}
            className="ml-auto flex items-center gap-1 rounded border border-violet-500/30 bg-violet-500/10 px-1.5 py-0.5 text-[9px] font-medium text-violet-600 hover:bg-violet-500/20"
          >
            <Eye className="h-2.5 w-2.5" />
            View
          </button>
        )}
      </div>
      <div className="space-y-1.5">
        {hasChildren ? children : <div className="text-[10px] italic text-zinc-500">{empty}</div>}
      </div>
    </div>
  )
}
