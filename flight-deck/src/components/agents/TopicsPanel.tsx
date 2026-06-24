import { useState, useEffect, useCallback, useRef } from 'react'
import { Tags, Loader2, AlertTriangle, RefreshCw, X, Sparkles, Search, Maximize2, Minimize2, Download, Combine, Trash2, Star } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'
import { TopicChat } from './TopicChat'

interface TopicMsg { role: string; channel?: string; excerpt: string; ts: string }
interface Group { id: string; name: string; count?: number }
interface Topic {
  id: string
  label: string
  summary: string
  keywords?: string
  msg_count?: number
  starred?: number
  last_seen?: string
  messages?: TopicMsg[]
  groups?: Group[]
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
  const [order, setOrder] = useState<'recent' | 'alpha'>('recent')
  const [groups, setGroups] = useState<Group[]>([])
  const [groupFilter, setGroupFilter] = useState('')
  const [activeTags, setActiveTags] = useState<string[]>([])
  const [hours, setHours] = useState(168)
  const [backfilling, setBackfilling] = useState(false)
  const [note, setNote] = useState('')
  const [fullscreen, setFullscreen] = useState(false)
  const [listWidth, setListWidth] = useState<number>(() => {
    const v = Number(localStorage.getItem('fd:topics-list-width'))
    return v >= 15 && v <= 85 ? v : 40
  })
  const bodyRef = useRef<HTMLDivElement>(null)
  const startDrag = (e: React.MouseEvent) => {
    e.preventDefault()
    const onMove = (ev: MouseEvent) => {
      const r = bodyRef.current?.getBoundingClientRect()
      if (!r || r.width === 0) return
      const pct = Math.min(85, Math.max(15, ((ev.clientX - r.left) / r.width) * 100))
      setListWidth(pct)
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      setListWidth((w) => { localStorage.setItem('fd:topics-list-width', String(Math.round(w))); return w })
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }
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
      qp.set('order', order)
      if (groupFilter) qp.set('group', groupFilter)
      if (activeTags.length) qp.set('tags', activeTags.join(','))
      const data = await fdFetch<{ topics: Topic[] }>(`/agent-topics/${host}/${port}?${qp.toString()}`)
      setTopics(data.topics || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [host, port, auth, query, order, groupFilter, activeTags])

  useEffect(() => { refresh() }, [refresh])

  const fetchGroups = useCallback(async () => {
    try {
      const d = await fdFetch<{ groups: Group[] }>(`/agent-topic-groups/${host}/${port}${tokenQs}`)
      setGroups(d.groups || [])
    } catch { /* non-fatal */ }
  }, [host, port, tokenQs])
  useEffect(() => { fetchGroups() }, [fetchGroups])

  const createGroup = async () => {
    const name = window.prompt('New group name (e.g. Work, Private):')?.trim()
    if (!name) return
    try {
      await fdFetch(`/agent-topic-groups/${host}/${port}${tokenQs}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }),
      })
      await fetchGroups()
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
  }

  // Toggle the selected topic's membership in a group, then re-sync.
  const toggleTopicGroup = async (gid: string) => {
    if (!selected) return
    const current = (selected.groups || []).map((g) => g.id)
    const next = current.includes(gid) ? current.filter((x) => x !== gid) : [...current, gid]
    try {
      const r = await fdFetch<{ groups: Group[] }>(
        `/agent-topic-set-groups/${host}/${port}/${encodeURIComponent(selected.id)}${tokenQs}`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ group_ids: next }) },
      )
      setSelected({ ...selected, groups: r.groups })
      await Promise.all([fetchGroups(), refresh()])
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
  }

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
    const preserveIds = Array.from(sel)
    const confirmMsg = preserveIds.length
      ? `Clear all topics EXCEPT the ${preserveIds.length} selected? The selected topics (and their messages) stay intact; everything else is wiped.`
      : 'Clear ALL topics and their messages and start fresh? Backfill progress is reset too, so Generate will reconsider every message.'
    if (!window.confirm(confirmMsg)) return
    setBackfilling(true); setNote('')
    try {
      const qp = new URLSearchParams()
      if (auth) qp.set('token', auth)
      const r = await fdFetch<{ cleared_topics: number; preserved?: number }>(
        `/agent-topics-reset/${host}/${port}?${qp.toString()}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ preserve_ids: preserveIds }),
        },
      )
      // Keep the open topic if it was preserved; otherwise close the detail pane.
      if (selected && !preserveIds.includes(selected.id)) setSelected(null)
      setSel(new Set())
      setNote(r.preserved
        ? `Reset — cleared ${r.cleared_topics} topic(s), kept ${r.preserved}. Click Generate to rebuild.`
        : `Reset — cleared ${r.cleared_topics} topic(s). Click Generate to rebuild.`)
      await refresh()
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e))
    } finally {
      setBackfilling(false)
    }
  }

  const toggleStar = async (t: Topic) => {
    const next = !t.starred
    setTopics((prev) => prev.map((x) => x.id === t.id ? { ...x, starred: next ? 1 : 0 } : x))  // optimistic
    try {
      await fdFetch(`/agent-topic-star/${host}/${port}/${encodeURIComponent(t.id)}${tokenQs}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ starred: next }),
      })
      await refresh()  // re-sort (starred float to top)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      await refresh()
    }
  }

  const toggleTag = (tag: string) => {
    const t = tag.trim()
    if (!t) return
    setActiveTags((prev) => prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t])
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
        <div ref={bodyRef} className="flex min-h-0 flex-1">
          {/* List */}
          <div className="flex flex-col" style={{ width: `${listWidth}%` }}>
            <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-1.5">
              <span className="text-[11px] font-medium text-zinc-400">{topics.length} topic{topics.length === 1 ? '' : 's'}</span>
              <select value={order} onChange={(e) => setOrder(e.target.value as 'recent' | 'alpha')}
                className="ml-auto rounded-md border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-[10px] text-zinc-300">
                <option value="recent">Recent</option>
                <option value="alpha">A–Z</option>
              </select>
            </div>
            {/* Group filter */}
            <div className="flex flex-wrap items-center gap-1 border-b border-zinc-800 px-3 py-1.5">
              <button onClick={() => setGroupFilter('')}
                className={`rounded-full px-2 py-0.5 text-[10px] ${groupFilter === '' ? 'bg-sky-600 text-white' : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'}`}>All</button>
              {groups.map((g) => (
                <button key={g.id} onClick={() => setGroupFilter(groupFilter === g.id ? '' : g.id)}
                  className={`rounded-full px-2 py-0.5 text-[10px] ${groupFilter === g.id ? 'bg-sky-600 text-white' : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'}`}>
                  {g.name}{g.count ? ` ${g.count}` : ''}
                </button>
              ))}
              <button onClick={createGroup} title="New group"
                className="rounded-full border border-zinc-700 px-2 py-0.5 text-[10px] text-zinc-500 hover:text-zinc-300">+ group</button>
            </div>
            {activeTags.length > 0 && (
              <div className="flex flex-wrap items-center gap-1 border-b border-zinc-800 bg-sky-950/10 px-3 py-1.5">
                <span className="text-[10px] uppercase tracking-wider text-zinc-500">Tags:</span>
                {activeTags.map((t) => (
                  <button key={t} onClick={() => toggleTag(t)} title="Remove tag filter"
                    className="flex items-center gap-1 rounded-full bg-sky-600 px-2 py-0.5 text-[10px] text-white hover:bg-sky-500">
                    {t} <X className="h-3 w-3" />
                  </button>
                ))}
                <button onClick={() => setActiveTags([])} className="ml-auto text-[10px] text-zinc-500 hover:text-zinc-300">clear</button>
              </div>
            )}
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
                    <button onClick={() => toggleStar(t)} title={t.starred ? 'Unpin' : 'Pin to top'}
                      className="shrink-0 p-0.5">
                      <Star className={`h-3.5 w-3.5 ${t.starred ? 'fill-amber-400 text-amber-400' : 'text-zinc-600 hover:text-zinc-400'}`} />
                    </button>
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

          {/* Draggable divider */}
          <div onMouseDown={startDrag} title="Drag to resize"
            className="w-1 shrink-0 cursor-col-resize bg-zinc-800 hover:bg-sky-600/60" />
          {/* Messages pane: scrollable middle + chat pinned at the bottom */}
          <div className="flex min-h-0 flex-1 flex-col">
            {!selected ? (
              <div className="flex h-full items-center justify-center text-xs text-zinc-600">Select a topic to see its messages.</div>
            ) : (
              <>
                <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-auto p-4">
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
                      {selected.keywords.split(',').filter(Boolean).map((k) => {
                        const tag = k.trim()
                        const on = activeTags.includes(tag)
                        return (
                          <button key={k} onClick={() => toggleTag(tag)}
                            title={on ? `Remove “${tag}” from filters` : `Filter topics by “${tag}”`}
                            className={`rounded-full px-2 py-0.5 text-[10px] ${on
                              ? 'bg-sky-600 text-white'
                              : 'bg-zinc-800 text-zinc-400 hover:bg-sky-900/60 hover:text-sky-200'}`}>
                            {on ? '✓ ' : ''}{k}
                          </button>
                        )
                      })}
                    </div>
                  )}
                  {/* Group assignment — click to add/remove this topic from a group */}
                  <div className="flex flex-wrap items-center gap-1">
                    <span className="text-[10px] text-zinc-500">Groups:</span>
                    {groups.length === 0 && <span className="text-[10px] text-zinc-600">none yet</span>}
                    {groups.map((g) => {
                      const on = (selected.groups || []).some((x) => x.id === g.id)
                      return (
                        <button key={g.id} onClick={() => toggleTopicGroup(g.id)}
                          className={`rounded-full px-2 py-0.5 text-[10px] ${on ? 'bg-sky-600 text-white' : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'}`}>
                          {on ? '✓ ' : ''}{g.name}
                        </button>
                      )
                    })}
                    <button onClick={createGroup} className="rounded-full border border-zinc-700 px-2 py-0.5 text-[10px] text-zinc-500 hover:text-zinc-300">+ group</button>
                  </div>
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
                {/* Chat pinned at the bottom (not part of the scrolling area) */}
                <div className="shrink-0 border-t border-zinc-800 px-4 pb-3">
                  <TopicChat host={host} port={port} auth={auth} topicId={selected.id} onPersisted={refresh} />
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
