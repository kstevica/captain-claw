import { useState, useEffect, useCallback } from 'react'
import { Tags, Loader2, AlertTriangle, RefreshCw, X, Sparkles, Search, Maximize2, Minimize2, Download, Combine, Trash2 } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'

interface TopicMsg { role: string; channel?: string; excerpt: string; ts: string }
interface Topic {
  id: string
  label: string
  summary: string
  keywords?: string
  msg_count?: number
  last_seen?: string
  messages?: TopicMsg[]
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

interface TopicsPanelProps {
  host: string
  port: number
  auth?: string
  agentName: string
  onClose: () => void
}

const BACKFILL_WINDOWS: { label: string; hours: number }[] = [
  { label: 'Last 24h', hours: 24 },
  { label: 'Last 7 days', hours: 168 },
  { label: 'Last 30 days', hours: 720 },
  { label: 'All history', hours: 0 },
]

function relTime(iso?: string): string {
  if (!iso) return ''
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return ''
  const s = Math.floor((Date.now() - t) / 1000)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}

export function TopicsPanel({ host, port, auth, agentName, onClose }: TopicsPanelProps) {
  const [topics, setTopics] = useState<Topic[]>([])
  const [selected, setSelected] = useState<Topic | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingTopic, setLoadingTopic] = useState(false)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [hours, setHours] = useState(168)
  const [backfilling, setBackfilling] = useState(false)
  const [note, setNote] = useState('')
  const [fullscreen, setFullscreen] = useState(false)
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [combining, setCombining] = useState(false)
  const [refreshing, setRefreshing] = useState(false)

  const tokenQs = auth ? `?token=${encodeURIComponent(auth)}` : ''

  const refresh = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const qp = new URLSearchParams()
      if (auth) qp.set('token', auth)
      if (query.trim()) qp.set('q', query.trim())
      const data = await fdFetch<{ topics: Topic[] }>(`/agent-topics/${host}/${port}?${qp.toString()}`)
      setTopics(data.topics || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [host, port, auth, query])

  useEffect(() => { refresh() }, [refresh])

  const openTopic = async (t: Topic) => {
    setSelected(t); setLoadingTopic(true)
    try {
      const data = await fdFetch<{ topic: Topic }>(`/agent-topic/${host}/${port}/${encodeURIComponent(t.id)}${tokenQs}`)
      setSelected(data.topic)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoadingTopic(false)
    }
  }

  const runBackfill = async () => {
    setBackfilling(true); setNote('')
    // Each call classifies one batch and reports `remaining`; keep going until
    // the backlog is drained (capped so a bug can't loop forever).
    let total = 0
    let topicsTotal = 0
    try {
      for (let i = 0; i < 100; i++) {
        const qp = new URLSearchParams()
        if (auth) qp.set('token', auth)
        qp.set('hours', String(hours))
        const r = await fdFetch<{ ok?: boolean; error?: string; classified: number; topics_touched: number; remaining: number }>(
          `/agent-topics-backfill/${host}/${port}?${qp.toString()}`, { method: 'POST' },
        )
        if (r.error) { setNote(r.error); break }
        total += r.classified
        topicsTotal += r.topics_touched || 0
        setNote(r.remaining
          ? `Classifying… ${total} done · ${topicsTotal} topic update(s) · ${r.remaining} left`
          : `Done — ${total} message(s), ${topicsTotal} topic update(s)` +
            (topicsTotal === 0 ? ' (model returned no usable topics — check agent logs)' : ''))
        await refresh()
        if (!r.remaining || r.classified === 0) break
      }
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e))
    } finally {
      setBackfilling(false)
    }
  }

  const runReset = async () => {
    if (!window.confirm('Clear ALL topics and their messages and start fresh? Backfill progress is reset too, so Generate will reconsider every message.')) return
    setBackfilling(true); setNote('')
    try {
      const qp = new URLSearchParams()
      if (auth) qp.set('token', auth)
      const r = await fdFetch<{ cleared_topics: number }>(
        `/agent-topics-reset/${host}/${port}?${qp.toString()}`, { method: 'POST' },
      )
      setSelected(null); setSel(new Set())
      setNote(`Reset — cleared ${r.cleared_topics} topic(s). Click Generate to rebuild.`)
      await refresh()
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e))
    } finally {
      setBackfilling(false)
    }
  }

  const toggleSel = (id: string) => setSel((prev) => {
    const next = new Set(prev)
    if (next.has(id)) next.delete(id); else next.add(id)
    return next
  })

  const runRefresh = async () => {
    if (!selected) return
    setRefreshing(true)
    try {
      await fdFetch(`/agent-topic-refresh/${host}/${port}/${encodeURIComponent(selected.id)}${tokenQs}`, { method: 'POST' })
      const data = await fdFetch<{ topic: Topic }>(`/agent-topic/${host}/${port}/${encodeURIComponent(selected.id)}${tokenQs}`)
      setSelected(data.topic)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRefreshing(false)
    }
  }

  // Merge selected topics into the first (most-recent) one.
  const combineTargets = topics.filter((t) => sel.has(t.id))
  const runCombine = async () => {
    if (combineTargets.length < 2) return
    const target = combineTargets[0]
    const sources = combineTargets.slice(1).map((t) => t.id)
    setCombining(true); setNote('')
    try {
      const qp = new URLSearchParams()
      if (auth) qp.set('token', auth)
      await fdFetch(`/agent-topics-combine/${host}/${port}?${qp.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_id: target.id, source_ids: sources }),
      })
      setNote(`Combined ${sources.length + 1} topics into "${target.label}"`)
      setSel(new Set())
      if (selected && sources.includes(selected.id)) setSelected(null)
      await refresh()
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e))
    } finally {
      setCombining(false)
    }
  }

  const exportMd = () => {
    if (!selected) return
    const lines: string[] = [`# ${selected.label}`, '']
    if (selected.summary) lines.push(selected.summary, '')
    if (selected.keywords) lines.push(`**Tags:** ${selected.keywords.split(',').filter(Boolean).join(', ')}`, '')
    lines.push(`_${agentName} · ${(selected.messages || []).length} messages_`, '', '---', '')
    for (const m of selected.messages || []) {
      lines.push(`### ${m.role}${m.ts ? ` · ${m.ts}` : ''}`, '', m.excerpt, '')
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `topic-${selected.id || 'conversation'}.md`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="flex flex-col border border-zinc-700/50 bg-zinc-900 shadow-2xl"
        style={fullscreen
          ? { width: '100vw', height: '100vh', borderRadius: 0 }
          : { width: '80vw', maxWidth: '1000px', height: '78vh', borderRadius: '0.75rem' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3.5 shrink-0">
          <div className="flex items-center gap-2">
            <Tags className="h-4 w-4 text-sky-400" />
            <span className="text-sm font-semibold text-zinc-100">Topics</span>
            <span className="text-xs text-zinc-500">· {agentName}</span>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={refresh} className="rounded-md p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200" title="Refresh">
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button onClick={() => setFullscreen((v) => !v)} className="rounded-md p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200" title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}>
              {fullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </button>
            <button onClick={onClose} className="rounded-md p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Backfill bar */}
        <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-5 py-2.5 shrink-0">
          <Sparkles className="h-3.5 w-3.5 text-zinc-500" />
          <span className="text-[11px] text-zinc-500">Generate topics for untagged messages:</span>
          <select value={hours} onChange={(e) => setHours(Number(e.target.value))}
            className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200">
            {BACKFILL_WINDOWS.map((w) => <option key={w.hours} value={w.hours}>{w.label}</option>)}
          </select>
          <button onClick={runBackfill} disabled={backfilling}
            className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40">
            {backfilling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
            Generate
          </button>
          <button onClick={runReset} disabled={backfilling} title="Clear all topics and backfill progress, then rebuild"
            className="flex items-center gap-1 rounded-lg border border-zinc-700 px-2 py-1 text-[11px] text-zinc-400 hover:bg-rose-950/40 hover:text-rose-300 disabled:opacity-40">
            <Trash2 className="h-3.5 w-3.5" /> Reset
          </button>
          {note && <span className="text-[11px] text-zinc-400">{note}</span>}
        </div>

        {error && (
          <div className="flex items-center gap-2 border-b border-rose-900/40 bg-rose-950/20 px-5 py-2 text-xs text-rose-400">
            <AlertTriangle className="h-3.5 w-3.5" /> {error}
          </div>
        )}

        {/* Body: topic list | messages */}
        <div className="flex min-h-0 flex-1">
          {/* List */}
          <div className="flex w-2/5 flex-col border-r border-zinc-800">
            <div className="flex items-center gap-1.5 border-b border-zinc-800 px-3 py-2">
              <Search className="h-3.5 w-3.5 text-zinc-600" />
              <input
                value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search topics…"
                className="w-full bg-transparent text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none"
              />
            </div>
            {combineTargets.length >= 2 && (
              <div className="flex items-center gap-2 border-b border-zinc-800 bg-zinc-800/40 px-3 py-1.5">
                <span className="text-[11px] text-zinc-400">{combineTargets.length} selected</span>
                <button onClick={runCombine} disabled={combining}
                  className="flex items-center gap-1 rounded-md bg-sky-600 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-sky-500 disabled:opacity-40">
                  {combining ? <Loader2 className="h-3 w-3 animate-spin" /> : <Combine className="h-3 w-3" />}
                  Combine → “{combineTargets[0].label}”
                </button>
                <button onClick={() => setSel(new Set())} className="ml-auto text-[10px] text-zinc-500 hover:text-zinc-300">clear</button>
              </div>
            )}
            <div className="flex-1 overflow-auto">
              {loading ? (
                <div className="p-4 text-center text-xs text-zinc-500"><Loader2 className="mx-auto h-4 w-4 animate-spin" /></div>
              ) : topics.length === 0 ? (
                <div className="p-4 text-center text-xs text-zinc-600">No topics yet. Chat a while, or run Generate above.</div>
              ) : (
                topics.map((t) => (
                  <div key={t.id}
                    className={`flex items-center gap-1.5 border-b border-zinc-800/50 hover:bg-zinc-800/50 ${selected?.id === t.id ? 'bg-zinc-800/70' : ''}`}>
                    <input type="checkbox" checked={sel.has(t.id)} onChange={() => toggleSel(t.id)}
                      title="select to combine"
                      className="ml-2 shrink-0 rounded border-zinc-700 bg-zinc-950 accent-sky-600" />
                    <button onClick={() => openTopic(t)} className="min-w-0 flex-1 px-2 py-2 text-left">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-xs font-medium text-zinc-100">{t.label}</span>
                        <span className="shrink-0 text-[10px] text-zinc-600">{relTime(t.last_seen)}</span>
                      </div>
                      <div className="truncate text-[11px] text-zinc-500">{t.summary}</div>
                      {!!t.msg_count && <div className="text-[10px] text-zinc-600">{t.msg_count} msgs</div>}
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Messages */}
          <div className="flex w-3/5 flex-col overflow-auto">
            {!selected ? (
              <div className="flex h-full items-center justify-center text-xs text-zinc-600">Select a topic to see its messages.</div>
            ) : (
              <div className="flex flex-col gap-2 p-4">
                <div className="flex items-start justify-between gap-2">
                  <div className="text-sm font-semibold text-zinc-100">{selected.label}</div>
                  <div className="flex shrink-0 items-center gap-1.5">
                    <button onClick={runRefresh} disabled={refreshing} title="Re-pull full message text from the live session"
                      className="flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800 disabled:opacity-40">
                      {refreshing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />} Refresh
                    </button>
                    <button onClick={exportMd} title="Export this conversation as Markdown"
                      className="flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800">
                      <Download className="h-3.5 w-3.5" /> .md
                    </button>
                  </div>
                </div>
                {selected.summary && (
                  <div className="fd-markdown text-xs text-zinc-400">
                    <Markdown remarkPlugins={[remarkGfm]}>{selected.summary}</Markdown>
                  </div>
                )}
                {selected.keywords && (
                  <div className="flex flex-wrap gap-1">
                    {selected.keywords.split(',').filter(Boolean).map((k) => (
                      <span key={k} className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400">{k}</span>
                    ))}
                  </div>
                )}
                <div className="mt-1 border-t border-zinc-800 pt-2">
                  {loadingTopic ? (
                    <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />
                  ) : (selected.messages || []).length === 0 ? (
                    <div className="text-xs text-zinc-600">No messages stored.</div>
                  ) : (
                    (selected.messages || []).map((m, i) => (
                      <div key={i} className="border-b border-zinc-800/40 py-1.5 last:border-0">
                        <div className="flex items-center gap-2 text-[10px] text-zinc-600">
                          <span className={m.role === 'user' ? 'text-sky-400' : m.role === 'narration' ? 'text-zinc-500' : 'text-emerald-400'}>{m.role}</span>
                          <span>{relTime(m.ts)}</span>
                        </div>
                        <div className="fd-markdown text-xs text-zinc-300">
                          <Markdown remarkPlugins={[remarkGfm]}>{m.excerpt}</Markdown>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
