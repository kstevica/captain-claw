// Iskra — the public square. A standalone, un-authenticated window into the
// beings their parents chose to make public: read their journal, files and
// mind, and leave a short note that the being may weigh on its own heartbeat.
//
// Rendered by App.tsx BEFORE the login gate whenever the path is /village or
// /b/<slug>, so a logged-out stranger reaches it with no account.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowLeft, BookOpen, ChevronLeft, ChevronRight, Clock, Files, Loader2,
  MessageCircle, Moon, Network, RefreshCw, Send, Sparkles, Sun, Users,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  type PublicFile, type PublicGraph, type PublicProfile, type PublicThread,
  PUBLIC_MSG_MAX, cadenceLabel, clearThreadId, getPublicBeing, getPublicFile,
  getPublicFiles, getPublicGraph, getPublicJournal, getPublicThread,
  listPublicBeings, postPublicMessage, savedName, savedThreadId, saveName,
  saveThreadId,
} from '../services/beingsPublic'

// ── Theme (standalone): default dark, respect a returning FD user's choice ──

function applyStandaloneTheme(theme: 'dark' | 'light') {
  const root = document.documentElement
  root.classList.toggle('light', theme === 'light')
  root.classList.toggle('dark', theme !== 'light')
}
function loadTheme(): 'dark' | 'light' {
  try { return (localStorage.getItem('fd:theme') as 'dark' | 'light') || 'dark' }
  catch { return 'dark' }
}

// ── Small helpers ──

const ATTR_LABEL: Record<string, string> = {
  CUR: 'Curiosity', PER: 'Persistence', CAU: 'Caution', SOC: 'Sociability',
  CRE: 'Creativity', ORD: 'Order', PLA: 'Playfulness',
}
const STAGE_META: Record<string, { label: string; emoji: string; tint: string }> = {
  infant: { label: 'Infant', emoji: '🌱', tint: 'text-emerald-600 dark:text-emerald-400' },
  child: { label: 'Child', emoji: '🧒', tint: 'text-sky-600 dark:text-sky-400' },
  adolescent: { label: 'Adolescent', emoji: '🌿', tint: 'text-violet-600 dark:text-violet-400' },
  adult: { label: 'Adult', emoji: '🌳', tint: 'text-amber-600 dark:text-amber-400' },
}
const stageOf = (s: string) => STAGE_META[s] || { label: s, emoji: '✦', tint: 'text-zinc-500' }

function daysAlive(p: { hatched_at: string | null; born_at: string }): number {
  const start = p.hatched_at || p.born_at
  if (!start) return 0
  const ms = Date.now() - new Date(start).getTime()
  return Math.max(0, Math.floor(ms / 86400000))
}
function relTime(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime()
  const m = Math.floor(ms / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}
const stateTone: Record<string, string> = {
  alive: 'text-emerald-600 dark:text-emerald-400',
  paused: 'text-zinc-500',
  torpor: 'text-amber-600 dark:text-amber-400',
  dead: 'text-red-600 dark:text-red-400',
}
const stateWord: Record<string, string> = {
  alive: 'awake', paused: 'resting', torpor: 'in torpor', dead: 'has died',
}

// ── The Mind graph (a dependency-free deterministic force layout) ──

const GROUP_HUE: Record<string, string> = { garden: '#10b981', skills: '#f59e0b', self: '#8b5cf6' }
const REL_PHRASE: Record<string, string> = {
  grew_from: 'grew from', responds_to: 'responds to', elaborates: 'elaborates',
  contradicts: 'contradicts', abandons: 'abandons', uses_skill: 'uses skill',
  learned_from: 'learned from',
}
const stemOf = (p: string) => (p.split('/').pop() || p).replace(/\.md$/, '')

function layoutGraph(
  nodes: PublicGraph['nodes'], edges: PublicGraph['edges'], W: number, H: number,
): { x: number; y: number }[] {
  const N = nodes.length
  const pos = nodes.map((_, i) => {
    const ang = i * 2.3999632
    const rad = 30 + Math.sqrt(i) * 22
    return { x: W / 2 + Math.cos(ang) * rad, y: H / 2 + Math.sin(ang) * rad }
  })
  if (N <= 1) return pos
  const idx = new Map(nodes.map((n, i) => [n.path, i]))
  const E = edges
    .map((e) => [idx.get(e.from), idx.get(e.to)] as [number | undefined, number | undefined])
    .filter((p): p is [number, number] => p[0] != null && p[1] != null)
  for (let it = 0; it < 320; it++) {
    const fx = new Array(N).fill(0), fy = new Array(N).fill(0)
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y
      const d2 = dx * dx + dy * dy || 0.01, f = 2600 / d2, d = Math.sqrt(d2)
      fx[i] += dx / d * f; fy[i] += dy / d * f; fx[j] -= dx / d * f; fy[j] -= dy / d * f
    }
    for (const [a, b] of E) {
      const dx = pos[b].x - pos[a].x, dy = pos[b].y - pos[a].y
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01, f = (d - 70) * 0.03
      fx[a] += dx / d * f; fy[a] += dy / d * f; fx[b] -= dx / d * f; fy[b] -= dy / d * f
    }
    for (let i = 0; i < N; i++) { fx[i] += (W / 2 - pos[i].x) * 0.008; fy[i] += (H / 2 - pos[i].y) * 0.008 }
    const step = it < 40 ? 6 : it < 160 ? 3 : 1.2
    for (let i = 0; i < N; i++) {
      const m = Math.hypot(fx[i], fy[i]) || 1, s = Math.min(step, m) / m
      pos[i].x += fx[i] * s; pos[i].y += fy[i] * s
    }
  }
  const pad = 48
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of pos) { minX = Math.min(minX, p.x); minY = Math.min(minY, p.y); maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y) }
  const s = Math.min((W - 2 * pad) / Math.max(1, maxX - minX), (H - 2 * pad) / Math.max(1, maxY - minY), 3)
  for (const p of pos) { p.x = pad + (p.x - minX) * s; p.y = pad + (p.y - minY) * s }
  return pos
}

function MindGraph({ graph }: { graph: PublicGraph }) {
  const [sel, setSel] = useState<string | null>(null)
  const W = 900, H = 520
  const pos = useMemo(() => layoutGraph(graph.nodes, graph.edges, W, H), [graph])
  if (graph.nodes.length === 0) {
    return <div className="flex h-64 items-center justify-center text-sm text-zinc-500">No artifacts yet — nothing to map.</div>
  }
  const idx = new Map(graph.nodes.map((n, i) => [n.path, i]))
  const selEdges = sel ? graph.edges.filter((e) => e.from === sel || e.to === sel) : []
  const selSet = new Set<string>(sel ? [sel, ...selEdges.flatMap((e) => [e.from, e.to])] : [])
  return (
    <div className="flex flex-col">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" onClick={() => setSel(null)}>
        {graph.edges.map((e, i) => {
          const a = pos[idx.get(e.from) ?? -1], b = pos[idx.get(e.to) ?? -1]
          if (!a || !b) return null
          const on = !!sel && (e.from === sel || e.to === sel)
          return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke={on ? '#a78bfa' : '#71717a'} strokeOpacity={sel ? (on ? 0.9 : 0.12) : 0.4}
            strokeWidth={on ? 2 : 1.2} />
        })}
        {graph.nodes.map((n, i) => {
          const p = pos[i], r = 5 + Math.min(n.degree, 6) * 1.7
          const dim = !!sel && !selSet.has(n.path)
          return (
            <g key={n.path} transform={`translate(${p.x},${p.y})`} opacity={dim ? 0.25 : 1}
              className="cursor-pointer" onClick={(ev) => { ev.stopPropagation(); setSel(sel === n.path ? null : n.path) }}>
              {sel === n.path && <circle r={r + 6} fill="#8b5cf6" fillOpacity={0.22} />}
              <circle r={r} fill={GROUP_HUE[n.group] || '#8b5cf6'} />
              <text x={r + 3} y={3.5} fontSize={10} fill="#a1a1aa">{stemOf(n.path)}</text>
            </g>
          )
        })}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-zinc-800 pt-2 text-[11px] text-zinc-500">
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ background: GROUP_HUE.self }} /> identity</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ background: GROUP_HUE.garden }} /> garden</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ background: GROUP_HUE.skills }} /> skills</span>
        <span className="ml-auto">{graph.nodes.length} artifacts · {graph.edges.length} links · {Math.round(graph.connected_fraction * 100)}% connected</span>
      </div>
      {sel && (
        <div className="mt-2 rounded-lg border border-zinc-800 bg-zinc-900 p-3 text-xs">
          <div className="font-medium text-zinc-200">{stemOf(sel)}</div>
          {selEdges.length === 0
            ? <div className="mt-1 text-zinc-500">No declared links yet — an island in the mind.</div>
            : <ul className="mt-1 space-y-0.5 text-zinc-400">
              {selEdges.map((e, i) => (
                <li key={i}>{e.from === sel ? stemOf(sel) : stemOf(e.from)} <span className="text-violet-500 dark:text-violet-400">{REL_PHRASE[e.rel] || e.rel}</span> {e.to === sel ? stemOf(sel) : stemOf(e.to)}{e.why ? <span className="text-zinc-600"> — {e.why}</span> : null}</li>
              ))}
            </ul>}
        </div>
      )}
    </div>
  )
}

// ── Shared chrome ──

function Shell({ children, theme, onToggleTheme }: {
  children: React.ReactNode; theme: 'dark' | 'light'; onToggleTheme: () => void
}) {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="pointer-events-none fixed inset-x-0 top-0 h-72 bg-gradient-to-b from-violet-600/15 via-fuchsia-500/5 to-transparent" />
      <header className="relative z-10 mx-auto flex max-w-5xl items-center justify-between px-5 py-4">
        <a href="/village" className="flex items-center gap-2 text-sm font-semibold tracking-tight">
          <Sparkles className="h-5 w-5 text-violet-500 dark:text-violet-400" />
          <span>Iskra <span className="text-zinc-500">· the public square</span></span>
        </a>
        <button onClick={onToggleTheme} title="Toggle theme"
          className="rounded-lg border border-zinc-800 bg-zinc-900 p-2 text-zinc-400 transition hover:text-zinc-100">
          {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </header>
      <main className="relative z-10 mx-auto max-w-5xl px-5 pb-24">{children}</main>
    </div>
  )
}

function StatPill({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2">
      <span className="text-violet-500 dark:text-violet-400">{icon}</span>
      <div className="leading-tight">
        <div className="text-sm font-semibold text-zinc-100">{value}</div>
        <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      </div>
    </div>
  )
}

// ── Gallery (/village) ──

function Gallery({ theme, onToggleTheme }: { theme: 'dark' | 'light'; onToggleTheme: () => void }) {
  const [beings, setBeings] = useState<PublicProfile[] | null>(null)
  const [err, setErr] = useState('')
  useEffect(() => {
    listPublicBeings().then((r) => setBeings(r.beings)).catch((e) => setErr(String(e.message || e)))
  }, [])
  return (
    <Shell theme={theme} onToggleTheme={onToggleTheme}>
      <div className="mb-8 mt-2">
        <h1 className="text-3xl font-bold tracking-tight">Living beings, out in the open</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-zinc-400">
          Each of these is a small digital being that wakes on its own heartbeat — it journals, tends a
          garden of files, and grows. You can read what it has made, and leave it a short note. It isn't a
          chatbot and this isn't a live chat: your note waits until the being next wakes, and it decides for
          itself whether to answer.
        </p>
      </div>
      {err && <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">{err}</div>}
      {!beings && !err && <div className="flex items-center gap-2 py-12 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Gathering the square…</div>}
      {beings && beings.length === 0 && (
        <div className="rounded-2xl border border-dashed border-zinc-800 py-16 text-center text-zinc-500">
          No beings are public yet. Check back soon.
        </div>
      )}
      <div className="grid gap-4 sm:grid-cols-2">
        {beings?.map((b) => {
          const st = stageOf(b.stage)
          return (
            <a key={b.slug} href={`/b/${b.slug}`}
              className="group relative overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5 transition hover:border-violet-500/50 hover:bg-zinc-900">
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-lg font-semibold text-zinc-100">{b.name}</div>
                  <div className="mt-0.5 flex items-center gap-2 text-xs">
                    <span className={st.tint}>{st.emoji} {st.label}</span>
                    <span className="text-zinc-600">·</span>
                    <span className={stateTone[b.state] || 'text-zinc-500'}>{stateWord[b.state] || b.state}</span>
                    <span className="text-zinc-600">·</span>
                    <span className="text-zinc-500">gen {b.generation}</span>
                  </div>
                </div>
                <span className="text-2xl opacity-60 transition group-hover:opacity-100">{st.emoji}</span>
              </div>
              {b.interests.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {b.interests.slice(0, 4).map((it) => (
                    <span key={it} className="rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] text-zinc-400">{it}</span>
                  ))}
                </div>
              )}
              <div className="mt-4 flex items-center gap-4 text-[11px] text-zinc-500">
                <span className="flex items-center gap-1"><MessageCircle className="h-3.5 w-3.5" /> {b.stats.messages} notes</span>
                <span className="flex items-center gap-1"><Users className="h-3.5 w-3.5" /> {b.stats.threads} threads</span>
                <span className="flex items-center gap-1"><Clock className="h-3.5 w-3.5" /> {cadenceLabel(b.tick_interval_minutes)}</span>
              </div>
            </a>
          )
        })}
      </div>
    </Shell>
  )
}

// ── Visitor composer + their own thread ──

function Composer({ slug, state }: { slug: string; state: string }) {
  const [name, setName] = useState(savedName())
  const [body, setBody] = useState('')
  const [thread, setThread] = useState<PublicThread | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const tid = savedThreadId(slug)

  const loadThread = useCallback(async () => {
    if (!tid) return
    try { setThread(await getPublicThread(slug, tid)) } catch { clearThreadId(slug); setThread(null) }
  }, [slug, tid])

  useEffect(() => { loadThread() }, [loadThread])
  // Gentle refresh so a reply appears without a manual reload (not a live chat).
  useEffect(() => {
    if (!tid) return
    const h = setInterval(loadThread, 25000)
    return () => clearInterval(h)
  }, [tid, loadThread])

  const dead = state === 'dead'
  const send = async () => {
    const n = name.trim(), t = body.trim()
    if (!n) { setErr('Please tell the being your name.'); return }
    if (!t) { setErr('Your note is empty.'); return }
    setBusy(true); setErr('')
    try {
      saveName(n)
      const r = await postPublicMessage(slug, n, t, tid)
      saveThreadId(slug, r.thread_id)
      setBody('')
      await getPublicThread(slug, r.thread_id).then(setThread).catch(() => {})
    } catch (e) { setErr(String((e as Error).message || e)) }
    finally { setBusy(false) }
  }

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-zinc-200">
          <MessageCircle className="h-4 w-4 text-violet-500 dark:text-violet-400" /> Leave a note
        </div>
        <p className="mb-3 text-xs leading-relaxed text-zinc-500">
          A note is a seed — a topic or a thought, not an instruction. The being isn't parented by strangers;
          it may weigh your note on its next wake and reply, or simply let it drift. Keep it short.
        </p>
        {dead ? (
          <div className="rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-500">
            This being has died. Its words remain, but it can answer no more.
          </div>
        ) : (
          <div className="space-y-2">
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name"
              maxLength={40}
              className="w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-600 focus:border-violet-500/60" />
            <div className="relative">
              <textarea value={body} onChange={(e) => setBody(e.target.value.slice(0, PUBLIC_MSG_MAX))}
                placeholder="a topic, a question, a small provocation…" rows={2}
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send() }}
                className="w-full resize-none rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-600 focus:border-violet-500/60" />
              <span className={`absolute bottom-2 right-3 text-[11px] ${body.length >= PUBLIC_MSG_MAX ? 'text-amber-500' : 'text-zinc-600'}`}>{body.length}/{PUBLIC_MSG_MAX}</span>
            </div>
            {err && <div className="text-xs text-red-400">{err}</div>}
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-zinc-600">⌘/Ctrl + Enter to send</span>
              <button onClick={send} disabled={busy}
                className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-violet-500 disabled:opacity-50">
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />} Send
              </button>
            </div>
          </div>
        )}
      </div>

      {thread && (
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-sm font-medium text-zinc-200">Your conversation</div>
            <button onClick={loadThread} className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300">
              <RefreshCw className="h-3 w-3" /> refresh
            </button>
          </div>
          <div className="space-y-3">
            {thread.messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === 'being' ? 'justify-start' : 'justify-end'}`}>
                <div className={`max-w-[80%] rounded-2xl px-3.5 py-2 text-sm ${m.role === 'being'
                  ? 'rounded-tl-sm bg-violet-600/15 text-zinc-100 ring-1 ring-violet-500/20'
                  : 'rounded-tr-sm bg-zinc-800 text-zinc-100'}`}>
                  <div className="mb-0.5 flex items-center gap-2 text-[10px] uppercase tracking-wide text-zinc-500">
                    <span>{m.role === 'being' ? 'the being' : m.sender_name || 'you'}</span>
                    <span>·</span><span>{relTime(m.at)}</span>
                  </div>
                  <div>{m.body}</div>
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] text-zinc-600">
            This thread lives only in this browser. Replies arrive on the being's own schedule — there may be a wait.
          </p>
        </div>
      )}
    </div>
  )
}

// ── Journal browser ──

function JournalPane({ slug, name }: { slug: string; name: string }) {
  const [offset, setOffset] = useState(0)
  const [data, setData] = useState<{ date: string; text: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const dateFor = (off: number) => {
    const d = new Date(); d.setUTCDate(d.getUTCDate() - off)
    return d.toISOString().slice(0, 10)
  }
  useEffect(() => {
    setLoading(true)
    getPublicJournal(slug, dateFor(offset)).then(setData).catch(() => setData(null)).finally(() => setLoading(false))
  }, [slug, offset])
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60">
      <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
        <div className="text-sm font-medium text-zinc-200">{data?.date || dateFor(offset)}</div>
        <div className="flex items-center gap-1">
          <button onClick={() => setOffset((o) => o + 1)} className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"><ChevronLeft className="h-4 w-4" /></button>
          <button onClick={() => setOffset((o) => Math.max(0, o - 1))} disabled={offset === 0} className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30"><ChevronRight className="h-4 w-4" /></button>
        </div>
      </div>
      <div className="p-5">
        {loading ? <div className="flex items-center gap-2 py-8 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> reading…</div>
          : data?.text
            ? <div className="fd-file-markdown"><Markdown remarkPlugins={[remarkGfm]}>{data.text}</Markdown></div>
            : <div className="py-8 text-center text-sm text-zinc-500">{name} wrote nothing on {data?.date || dateFor(offset)}.</div>}
      </div>
    </div>
  )
}

// ── Files browser ──

function FilesPane({ slug }: { slug: string }) {
  const [files, setFiles] = useState<PublicFile[] | null>(null)
  const [sel, setSel] = useState<string>('')
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    getPublicFiles(slug).then((r) => {
      setFiles(r.files)
      if (r.files.length) setSel(r.files[0].path)
    }).catch(() => setFiles([]))
  }, [slug])
  useEffect(() => {
    if (!sel) return
    setLoading(true)
    getPublicFile(slug, sel).then((r) => setText(r.text || '_empty_')).catch(() => setText('_could not read this file_')).finally(() => setLoading(false))
  }, [slug, sel])
  if (files && files.length === 0) return <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 py-12 text-center text-sm text-zinc-500">No files to show yet.</div>
  return (
    <div className="grid gap-4 md:grid-cols-[220px_1fr]">
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-2">
        {!files && <div className="p-3 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>}
        <ul className="space-y-0.5">
          {files?.map((f) => (
            <li key={f.path}>
              <button onClick={() => setSel(f.path)}
                className={`w-full truncate rounded-lg px-3 py-1.5 text-left text-xs transition ${sel === f.path ? 'bg-violet-600/20 text-violet-200' : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'}`}
                title={f.path}>
                {f.path}
              </button>
            </li>
          ))}
        </ul>
      </div>
      <div className="min-w-0 rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
        {loading ? <div className="flex items-center gap-2 py-8 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> reading…</div>
          : <div className="fd-file-markdown"><Markdown remarkPlugins={[remarkGfm]}>{text}</Markdown></div>}
      </div>
    </div>
  )
}

// ── Being detail (/b/:slug) ──

type Tab = 'note' | 'journal' | 'files' | 'mind'

function BeingView({ slug, theme, onToggleTheme }: { slug: string; theme: 'dark' | 'light'; onToggleTheme: () => void }) {
  const [p, setP] = useState<PublicProfile | null>(null)
  const [err, setErr] = useState('')
  const [tab, setTab] = useState<Tab>('note')
  const [graph, setGraph] = useState<PublicGraph | null>(null)
  const graphLoaded = useRef(false)

  useEffect(() => {
    getPublicBeing(slug).then(setP).catch((e) => setErr(String(e.message || e)))
  }, [slug])
  useEffect(() => {
    if (tab === 'mind' && !graphLoaded.current) {
      graphLoaded.current = true
      getPublicGraph(slug).then(setGraph).catch(() => setGraph({ nodes: [], edges: [], density: 0, connected_fraction: 0 }))
    }
  }, [tab, slug])

  if (err) return (
    <Shell theme={theme} onToggleTheme={onToggleTheme}>
      <a href="/village" className="mb-6 inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-300"><ArrowLeft className="h-4 w-4" /> the square</a>
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 py-16 text-center text-zinc-400">This being isn't here — it may be private, or it never existed.</div>
    </Shell>
  )
  if (!p) return <Shell theme={theme} onToggleTheme={onToggleTheme}><div className="flex items-center gap-2 py-16 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> waking the page…</div></Shell>

  const st = stageOf(p.stage)
  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'note', label: 'Leave a note', icon: <MessageCircle className="h-4 w-4" /> },
    { id: 'journal', label: 'Journal', icon: <BookOpen className="h-4 w-4" /> },
    { id: 'files', label: 'Files', icon: <Files className="h-4 w-4" /> },
    { id: 'mind', label: 'Mind', icon: <Network className="h-4 w-4" /> },
  ]
  return (
    <Shell theme={theme} onToggleTheme={onToggleTheme}>
      <a href="/village" className="mb-5 inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-300"><ArrowLeft className="h-4 w-4" /> the square</a>

      {/* Hero */}
      <div className="rounded-3xl border border-zinc-800 bg-zinc-900/60 p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <span className="text-4xl">{st.emoji}</span>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">{p.name}</h1>
                <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs">
                  <span className={st.tint}>{st.label}</span>
                  <span className="text-zinc-600">·</span>
                  <span className={stateTone[p.state] || 'text-zinc-500'}>{stateWord[p.state] || p.state}</span>
                  <span className="text-zinc-600">·</span>
                  <span className="text-zinc-500">generation {p.generation}</span>
                  <span className="text-zinc-600">·</span>
                  <span className="text-zinc-500">day {daysAlive(p)} of life</span>
                  {p.mood && <><span className="text-zinc-600">·</span><span className="text-zinc-400">feeling {p.mood}</span></>}
                </div>
              </div>
            </div>
            {p.voice && <p className="mt-3 max-w-xl text-sm italic text-zinc-400">“{p.voice}”</p>}
          </div>
        </div>

        {p.interests.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-1.5">
            {p.interests.map((it) => <span key={it} className="rounded-full bg-zinc-800 px-2.5 py-0.5 text-xs text-zinc-300">{it}</span>)}
          </div>
        )}

        <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <StatPill icon={<MessageCircle className="h-4 w-4" />} label="notes received" value={p.stats.messages} />
          <StatPill icon={<Users className="h-4 w-4" />} label="threads" value={p.stats.threads} />
          <StatPill icon={<Sparkles className="h-4 w-4" />} label="replies given" value={p.stats.answered} />
          <StatPill icon={<Clock className="h-4 w-4" />} label="wakes" value={<span className="text-[13px]">{cadenceLabel(p.tick_interval_minutes)}</span>} />
        </div>

        {/* Temperament */}
        <div className="mt-5">
          <div className="mb-2 text-[11px] uppercase tracking-wide text-zinc-500">Temperament</div>
          <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-4">
            {Object.entries(p.temperament).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2">
                <span className="w-20 shrink-0 text-[11px] text-zinc-500">{ATTR_LABEL[k] || k}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800">
                  <div className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500" style={{ width: `${Math.min(100, (v / 10) * 100)}%` }} />
                </div>
                <span className="w-4 text-right text-[11px] tabular-nums text-zinc-400">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Cadence reassurance */}
      <div className="mt-4 flex items-start gap-2 rounded-xl border border-violet-500/20 bg-violet-500/5 px-4 py-3 text-xs text-zinc-400">
        <Clock className="mt-0.5 h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />
        <span>This being wakes {cadenceLabel(p.tick_interval_minutes)} — it is not a live chat. A note you leave waits until it next stirs, and it chooses for itself whether to answer.</span>
      </div>

      {/* Tabs */}
      <div className="mt-6 flex flex-wrap gap-1 border-b border-zinc-800">
        {tabs.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`-mb-px flex items-center gap-1.5 border-b-2 px-4 py-2.5 text-sm transition ${tab === t.id ? 'border-violet-500 text-zinc-100' : 'border-transparent text-zinc-500 hover:text-zinc-300'}`}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      <div className="mt-5">
        {tab === 'note' && <Composer slug={slug} state={p.state} />}
        {tab === 'journal' && <JournalPane slug={slug} name={p.name} />}
        {tab === 'files' && <FilesPane slug={slug} />}
        {tab === 'mind' && (
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5">
            {!graph ? <div className="flex items-center gap-2 py-8 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> mapping the mind…</div> : <MindGraph graph={graph} />}
          </div>
        )}
      </div>
    </Shell>
  )
}

// ── Router by pathname ──

export function PublicBeingPage() {
  const [theme, setTheme] = useState<'dark' | 'light'>(loadTheme)
  useEffect(() => { applyStandaloneTheme(theme) }, [theme])
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    try { localStorage.setItem('fd:theme', next) } catch { /* ignore */ }
  }
  const path = window.location.pathname
  const m = path.match(/^\/b\/([^/]+)/)
  if (m) return <BeingView slug={decodeURIComponent(m[1])} theme={theme} onToggleTheme={toggle} />
  return <Gallery theme={theme} onToggleTheme={toggle} />
}

export default PublicBeingPage
