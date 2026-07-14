// Iskra — living beings: conception (point-buy), vitals, wallet, journal.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowDownUp, ArrowRightLeft, BookOpen, CalendarDays, ChevronDown,
  ChevronLeft, ChevronRight, ClipboardList, Egg, Files, Fingerprint, Gift,
  GraduationCap, History, Loader2, Mail, Maximize2, MessageCircle, Minimize2,
  Moon, Network, Pause, Play, Plus, RefreshCw, Search, ScrollText, Skull,
  Sparkles, Sprout, Users, Wrench, X, Zap,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  type BeingEvent, type BeingListItem, type BeingsMeta, type BeingVitals,
  type BeingGraph, type Chore, type Quest, type ReportCard, type SelfFile,
  type ThreadItem, type Venture, type VillageItem,
  acceptVenture, approveProcreation, approveSelfMod, approveVenture,
  arrangeOffspring, cancelQuest, conceiveBeing, euthanizeBeing,
  getBeingEvents, getBeingJournal, getBeingsMeta, getBeingVitals, getBoard,
  getLiabilities, getReportCard, getSelfFile, getSelfFiles, getVillage,
  getBeingGraph, getBeingMessages, hatchBeing, judgeChore, judgeQuest,
  listBeings, listChores, messageBeing, pauseBeing, postChore, postQuest,
  rejectProcreation, rejectSelfMod, rollbackPersona, setAllowance,
  setHouseRules, setMediaDiet, setStage, setVentureState, tickBeing,
  wakeBeing,
} from '../services/beings'

const REFRESH_MS = 6000
const ATTRS = ['CUR', 'PER', 'CAU', 'SOC', 'CRE', 'ORD', 'PLA'] as const

const STAGE_META: Record<string, string> = {
  egg: 'bg-zinc-500/15 text-zinc-300 border-zinc-500/30',
  infant: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
  child: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  adolescent: 'bg-violet-500/15 text-violet-300 border-violet-500/30',
  adult: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
}
const STATE_META: Record<string, string> = {
  alive: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  paused: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
  torpor: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  dead: 'bg-red-500/15 text-red-300 border-red-500/30',
}

function fmtTokens(n: number | null | undefined): string {
  if (n == null) return '∞'
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (Math.abs(n) >= 1_000) return `${Math.round(n / 1_000)}k`
  return String(n)
}

function mdCell(v: unknown): string {
  const s = v == null ? '' : String(v)
  return s.replace(/\|/g, '\\|').replace(/\n/g, ' ').slice(0, 220)
}

function fmtAt(iso: string): string {
  // "2026-07-12T19:54:03.123" -> "07-12 19:54"
  return iso.length >= 16 ? `${iso.slice(5, 10)} ${iso.slice(11, 16)}` : iso
}

function fmtRelTime(iso: string): string {
  const t = new Date(iso).getTime()
  if (!t) return ''
  const s = (Date.now() - t) / 1000
  if (s < 60) return 'now'
  if (s < 3600) return `${Math.floor(s / 60)}m`
  if (s < 86400) return `${Math.floor(s / 3600)}h`
  if (s < 86400 * 7) return `${Math.floor(s / 86400)}d`
  return iso.slice(5, 10) // MM-DD
}

// Date buckets for the Self-files filter (computed in the viewer's local time
// against each file's mtime). Weeks start Monday.
const DATE_BUCKETS = [
  { key: 'all', label: 'All time' },
  { key: 'today', label: 'Today' },
  { key: 'yesterday', label: 'Yesterday' },
  { key: 'week', label: 'This week' },
  { key: 'lastweek', label: 'Last week' },
  { key: 'month', label: 'This month' },
] as const
type DateBucket = typeof DATE_BUCKETS[number]['key']

function inDateBucket(iso: string, bucket: DateBucket): boolean {
  if (bucket === 'all') return true
  const t = new Date(iso).getTime()
  if (!t) return false
  const startOfDay = (d: Date) => { const x = new Date(d); x.setHours(0, 0, 0, 0); return x.getTime() }
  const now = new Date()
  const today = startOfDay(now)
  const dow = (now.getDay() + 6) % 7 // 0 = Monday
  const weekStart = today - dow * 86400000
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1).getTime()
  switch (bucket) {
    case 'today': return t >= today
    case 'yesterday': return t >= today - 86400000 && t < today
    case 'week': return t >= weekStart
    case 'lastweek': return t >= weekStart - 7 * 86400000 && t < weekStart
    case 'month': return t >= monthStart
    default: return true
  }
}

function summarizeEventData(e: BeingEvent): string {
  const d = e.data as Record<string, unknown>
  switch (e.kind) {
    case 'hatched': return 'egg → infant'
    case 'body': return `spawned on port ${d.port} (${d.tier})`
    case 'body_rebound': return `followed its body to port ${d.port}${d.was ? ` (was ${d.was})` : ''}`
    case 'body_unreachable': return `body not answering on port ${d.port} — restarting it`
    case 'stage': return `${d.from} → ${d.to}`
    case 'state': return `${d.from} → ${d.to}`
    case 'spoke_to_parent': return String(d.preview ?? '')
    case 'parent_message': return `you wrote to it: “${d.preview ?? ''}”`
    case 'message_suppressed': return String(d.reason ?? 'no attention credits')
    case 'chore_posted': return `${d.spec} (fee ${fmtTokens(Number(d.fee_tokens) || 0)})`
    case 'chore_done': return String(d.result ?? '')
    case 'chore_paid': return `paid ${fmtTokens(Number(d.fee_tokens) || 0)}`
    case 'chore_failed': return String(d.note ?? 'rejected')
    case 'milestone': return String(d.name ?? '')
    case 'rules_updated': return `${d.count} rule(s)`
    case 'rules_internalized': return 'internalized into VALUES.md'
    case 'metamorphosis': return `${d.from} → ${d.to} (${fmtTokens(Number(d.price_tokens) || 0)})`
    case 'self_mod_proposed': return `proposed a new persona — ${d.reason}`
    case 'self_mod_adopted': return `new persona adopted (${d.by}) — ${d.reason}`
    case 'self_mod_rejected': return d.by === 'gate'
      ? `persona failed the gate: ${(d.failed as { name: string }[] | undefined)?.map(f => f.name).join(', ')}`
      : `persona rejected by parent${d.note ? ` — ${d.note}` : ''}`
    case 'self_mod_refused': return `self-mod refused: ${d.reason}`
    case 'self_mod_rolled_back': return 'persona rolled back by parent'
    case 'self_mod_auto_notice': return `adult self-mod (auto) — ${d.reason}`
    case 'procreation_proposed': return `asks for a child (${d.child_name || '?'}) — ${d.case}`
    case 'procreation_refused': return `procreation refused: ${d.reason}`
    case 'procreation_consented': return `parent consented — ${d.name} conceived`
    case 'procreation_rejected': return `procreation declined${d.note ? ` — ${d.note}` : ''}`
    case 'had_child': return `had a child: ${d.name}${d.with ? ` (with ${d.with})` : ''} — dowry ${fmtTokens(Number(d.dowry_share) || 0)}`
    case 'endowed': return `endowed: ${(d.skills as string[] | undefined)?.join(', ') || 'nothing'}${Number(d.heirlooms) ? ` + ${d.heirlooms} heirloom(s)` : ''}`
    case 'died': return `died — ${d.cause}${d.asleep_days ? ` after ${d.asleep_days} days of torpor` : ''}`
    case 'quest_claimed': return `claimed quest '${d.title}' (${fmtTokens(Number(d.fee_tokens) || 0)})`
    case 'quest_delivered': return `delivered quest '${d.title}'`
    case 'quest_paid': return `paid for quest '${d.title}' — ${fmtTokens(Number(d.fee_tokens) || 0)}`
    case 'quest_failed': return `quest '${d.title}' rejected${d.note ? ` — ${d.note}` : ''}`
    case 'venture_proposed': return `proposed venture '${d.title}' (${fmtTokens(Number(d.price_tokens) || 0)}/${d.cadence_days}d)`
    case 'venture_approved': return `venture '${d.title}' approved at ${fmtTokens(Number(d.price_tokens) || 0)}`
    case 'venture_delivered': return `delivered venture '${d.title}'`
    case 'venture_paid': return `venture '${d.title}' paid — ${fmtTokens(Number(d.price_tokens) || 0)}`
    case 'venture_rejected': return `venture delivery rejected${d.note ? ` — ${d.note}` : ''}`
    case 'venture_state': return `venture → ${d.to}`
    case 'earning_refused': return `earning refused (${d.what}): ${d.reason}`
    case 'act_unverified': return `claimed ${d.claimed} but made no artifact — logged as reflection`
    case 'narration_mismatch': return `journal claimed a file write, but nothing changed on disk — “${d.summary}”`
    case 'drive_unearned': return `claimed to satisfy its ${d.drive} drive without making anything real`
    case 'edge_declared': return `linked ${d.from} → ${d.to} (${d.rel})${d.why ? ` — ${d.why}` : ''}`
    case 'edge_unverified': return `link ${d.from} → ${d.to} refused — ${d.reason}`
    case 'edges_pruned': return `pruned ${d.count} dangling link(s) at dream`
    case 'consolidated': return `consolidated ${Number(d.count) || 0} file(s) into ${d.into}${d.why ? ` — ${d.why}` : ''} (originals archived)`
    case 'consolidate_unverified': return `consolidation refused — ${d.reason}`
    case 'woke_from_torpor': return 'revived by allowance'
    case 'collapsed_exhausted': return `overspent (${fmtTokens(Number(d.weighted) || 0)})`
    case 'resting_at_cap': return 'daily burn cap reached'
    case 'tick_skipped': return String(d.reason ?? '')
    case 'tick_timeout': return 'no reply in time'
    case 'digest_parse_failed': return 'unstructured reply'
    case 'spawn_failed': return String(d.error ?? '')
    case 'chore_claim_invalid': return `claimed unknown job ${d.job_id}`
    default: return JSON.stringify(d).slice(0, 160)
  }
}

function renderTicksMarkdown(events: BeingEvent[]): string {
  const ticks = events.filter((e) => e.kind === 'tick')
  const other = events.filter((e) => e.kind !== 'tick')
  const lines: string[] = ['## Ticks', '']
  if (ticks.length === 0) {
    lines.push('_No ticks yet — poke to wake this being._')
  } else {
    lines.push('| Time | Kind | Act | Summary (its words) | Actually changed | Tokens |')
    lines.push('|---|---|---|---|---|---|')
    for (const e of ticks) {
      const d = e.data as Record<string, unknown>
      // Ground truth: what the tools really wrote, from the git diff — shown
      // beside its self-report so narration can't stand unchecked.
      const ch = d.changed as string[] | null | undefined
      let changed: string
      if (ch === null || ch === undefined) changed = '·'
      else if (ch.length === 0) changed = d.mismatch ? '⚠ none (claimed a write)' : 'none'
      else changed = ch.join(', ')
      lines.push(`| ${fmtAt(e.at)} | ${mdCell(d.kind)} | ${mdCell(d.act)} | ${mdCell(d.summary)} | ${mdCell(changed)} | ${fmtTokens(Number(d.tokens_weighted) || 0)} |`)
    }
  }
  lines.push('', '## Other events', '')
  if (other.length === 0) {
    lines.push('_Nothing else yet._')
  } else {
    lines.push('| Time | Event | Details |')
    lines.push('|---|---|---|')
    for (const e of other) {
      lines.push(`| ${fmtAt(e.at)} | ${mdCell(e.kind)} | ${mdCell(summarizeEventData(e))} |`)
    }
  }
  return lines.join('\n')
}

// ── The Mind — a force-directed graph of the being's own artifacts ──

// A tiny dependency-free force layout: repulsion + edge springs + gravity,
// deterministic (no randomness) so the same graph always lays out the same.
function layoutGraph(
  nodes: BeingGraph['nodes'], edges: BeingGraph['edges'], W: number, H: number,
): { x: number; y: number }[] {
  const N = nodes.length
  const pos = nodes.map((_, i) => {
    const ang = i * 2.3999632   // golden angle → an even initial spread
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
      const d2 = dx * dx + dy * dy || 0.01, d = Math.sqrt(d2), f = 2600 / d2
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
  const pad = 52
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of pos) { minX = Math.min(minX, p.x); minY = Math.min(minY, p.y); maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y) }
  const s = Math.min((W - 2 * pad) / Math.max(1, maxX - minX), (H - 2 * pad) / Math.max(1, maxY - minY), 3)
  for (const p of pos) { p.x = pad + (p.x - minX) * s; p.y = pad + (p.y - minY) * s }
  return pos
}

// Fixed hues that read on both the light and dark modal background.
const GROUP_HUE: Record<string, string> = { garden: '#10b981', skills: '#f59e0b', self: '#8b5cf6' }
const REL_PHRASE: Record<string, string> = {
  grew_from: 'grew from', responds_to: 'responds to', elaborates: 'elaborates',
  contradicts: 'contradicts', abandons: 'abandons', uses_skill: 'uses skill',
  learned_from: 'learned from',
}
const stemOf = (p: string) => (p.split('/').pop() || p).replace(/\.md$/, '')

function MindGraph({ graph }: { graph: BeingGraph }) {
  const [sel, setSel] = useState<string | null>(null)
  const W = 900, H = 560
  const pos = useMemo(() => layoutGraph(graph.nodes, graph.edges, W, H), [graph])
  if (graph.nodes.length === 0) {
    return <div className="flex h-full items-center justify-center p-8 text-sm text-zinc-500">No artifacts yet — nothing to map.</div>
  }
  const idx = new Map(graph.nodes.map((n, i) => [n.path, i]))
  const selEdges = sel ? graph.edges.filter((e) => e.from === sel || e.to === sel) : []
  const selSet = new Set<string>(sel ? [sel, ...selEdges.flatMap((e) => [e.from, e.to])] : [])
  return (
    <div className="flex h-full flex-col">
      <svg viewBox={`0 0 ${W} ${H}`} className="min-h-0 w-full flex-1" onClick={() => setSel(null)}>
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
              <text x={r + 3} y={3.5} fontSize={10} fill="#71717a">{stemOf(n.path)}</text>
            </g>
          )
        })}
      </svg>
      <div className="shrink-0 space-y-1 border-t border-zinc-800 px-4 py-2 text-[11px]">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-zinc-500">
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ background: GROUP_HUE.self }} /> identity</span>
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ background: GROUP_HUE.garden }} /> garden</span>
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ background: GROUP_HUE.skills }} /> skills</span>
          <span className="ml-auto">{graph.nodes.length} artifacts · {graph.edges.length} links · {Math.round(graph.connected_fraction * 100)}% connected</span>
        </div>
        {sel && (
          <div className="text-zinc-400">
            <span className="font-medium text-zinc-200">{stemOf(sel)}</span>
            {selEdges.length === 0 ? ' — an island (no links yet)' : ' — ' + selEdges.map((e) =>
              e.from === sel ? `${REL_PHRASE[e.rel] || e.rel} ${stemOf(e.to)}` : `${stemOf(e.from)} ${REL_PHRASE[e.rel] || e.rel} it`,
            ).join('; ')}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Journal / ticks log / self-files modal — full markdown, fullscreen ──

function selfFileLabel(path: string): string {
  const base = path.split('/').pop() || path
  const stem = base.replace(/\.md$/i, '')
  const folder = path.includes('/') ? path.slice(0, path.indexOf('/')) : ''
  return folder && folder !== 'self' ? `${folder}/${stem}` : stem
}

// The being's home is grouped by folder so the sidebar stays navigable as it
// grows: identity files first, then the garden, then skills, then anything new.
const SELF_CORE_ORDER = ['SELF.md', 'VALUES.md', 'INTERESTS.md', 'RELATIONSHIPS.md', 'REFLECTIONS.md']
const SELF_GROUP_ORDER = ['self', 'garden', 'skills']
const SELF_GROUP_LABEL: Record<string, string> = { self: 'Identity', garden: 'Garden', skills: 'Skills' }
const SELF_GROUP_ICON: Record<string, typeof Fingerprint> = { self: Fingerprint, garden: Sprout, skills: Wrench }

function fileStem(path: string): string {
  return (path.split('/').pop() || path).replace(/\.md$/i, '')
}
function fileGroup(path: string): string {
  return path.includes('/') ? path.slice(0, path.indexOf('/')) : 'self'
}
function groupSelfFiles(files: SelfFile[]): { key: string; label: string; files: SelfFile[] }[] {
  const byGroup = new Map<string, SelfFile[]>()
  for (const f of files) {
    const g = fileGroup(f.path)
    if (!byGroup.has(g)) byGroup.set(g, [])
    byGroup.get(g)!.push(f)
  }
  const rank = (arr: string[], v: string) => { const i = arr.indexOf(v); return i < 0 ? 99 : i }
  return Array.from(byGroup.keys())
    .sort((a, b) => rank(SELF_GROUP_ORDER, a) - rank(SELF_GROUP_ORDER, b) || a.localeCompare(b))
    .map((k) => ({
      key: k,
      label: SELF_GROUP_LABEL[k] || k.charAt(0).toUpperCase() + k.slice(1),
      files: [...byGroup.get(k)!].sort((a, b) =>
        k === 'self'
          ? rank(SELF_CORE_ORDER, a.path.split('/').pop() || '') - rank(SELF_CORE_ORDER, b.path.split('/').pop() || '') || a.path.localeCompare(b.path)
          : a.path.localeCompare(b.path)),
    }))
}

function BeingLogModal({ slug, name, mode, onClose }: {
  slug: string
  name: string
  mode: 'journal' | 'ticks' | 'self' | 'mind'
  onClose: () => void
}) {
  const today = new Date().toISOString().slice(0, 10)
  const [maximized, setMaximized] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [markdown, setMarkdown] = useState('')
  const [graph, setGraph] = useState<BeingGraph | null>(null)
  const [date, setDate] = useState(today)
  const [files, setFiles] = useState<SelfFile[]>([])
  const [activeFile, setActiveFile] = useState('')
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [search, setSearch] = useState('')
  const [filterGroup, setFilterGroup] = useState<string | null>(null)
  const [dateFilter, setDateFilter] = useState<DateBucket>('all')
  const [sortBy, setSortBy] = useState<'name' | 'newest' | 'oldest'>('name')
  // A ref (not state) so loadSelf's identity stays stable across file
  // switches — it only needs to change when `slug` changes, never when the
  // user clicks a different file in the sidebar.
  const activeFileRef = useRef('')

  const loadFile = useCallback(async (path: string) => {
    setLoading(true)
    setError('')
    try {
      const f = await getSelfFile(slug, path)
      setMarkdown(f.text || `_${selfFileLabel(path)} is empty._`)
      setActiveFile(path)
      activeFileRef.current = path
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load')
    } finally {
      setLoading(false)
    }
  }, [slug])

  const loadSelf = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const list = await getSelfFiles(slug)
      setFiles(list.files)
      if (!list.files.length) { setMarkdown(''); setLoading(false); return }
      const keep = list.files.find((f) => f.path === activeFileRef.current)
      await loadFile(keep ? keep.path : list.files[0].path)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load')
      setLoading(false)
    }
  }, [slug, loadFile])

  const loadJournal = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const j = await getBeingJournal(slug, date)
      setMarkdown(j.text || `_${name}'s journal is empty on ${j.date}._`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load')
    } finally {
      setLoading(false)
    }
  }, [slug, date, name])

  const loadTicks = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const ev = await getBeingEvents(slug, 300)
      setMarkdown(renderTicksMarkdown(ev.events))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load')
    } finally {
      setLoading(false)
    }
  }, [slug])

  const loadMind = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setGraph(await getBeingGraph(slug))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load')
    } finally {
      setLoading(false)
    }
  }, [slug])

  const load = mode === 'journal' ? loadJournal
    : mode === 'ticks' ? loadTicks
      : mode === 'mind' ? loadMind : loadSelf

  useEffect(() => { void load() }, [load])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const shiftDate = (deltaDays: number) => {
    const d = new Date(`${date}T00:00:00Z`)
    d.setUTCDate(d.getUTCDate() + deltaDays)
    setDate(d.toISOString().slice(0, 10))
  }

  const titles = { journal: 'Journal', ticks: 'Ticks log', self: 'Self files', mind: 'The Mind' } as const
  const sizeClass = maximized
    ? 'h-[95vh] w-[95vw]'
    : mode === 'self' || mode === 'mind' ? 'h-[80vh] w-[920px]' : 'w-[820px] max-h-[85vh]'

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className={`flex flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl transition-all duration-200 ${sizeClass}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-4 py-2.5">
          <div className="flex items-center gap-2">
            {mode === 'journal' && <ScrollText className="h-4 w-4 text-violet-500 dark:text-violet-400" />}
            {mode === 'ticks' && <History className="h-4 w-4 text-violet-500 dark:text-violet-400" />}
            {mode === 'self' && <Files className="h-4 w-4 text-violet-500 dark:text-violet-400" />}
            {mode === 'mind' && <Network className="h-4 w-4 text-violet-500 dark:text-violet-400" />}
            <h3 className="text-sm font-semibold text-zinc-100">{name} — {titles[mode]}</h3>
            {mode === 'journal' && <span className="text-[11px] text-zinc-500">{date}</span>}
            {mode === 'self' && activeFile && (
              <span className="text-[11px] text-zinc-500">
                {activeFile}
                {(() => { const m = files.find((f) => f.path === activeFile)?.mtime; return m ? ` · ${fmtAt(m)}` : '' })()}
              </span>
            )}
          </div>
          <div className="flex items-center gap-0.5">
            {mode === 'journal' && (
              <>
                <button onClick={() => shiftDate(-1)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Previous day">
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <button onClick={() => shiftDate(1)} disabled={date >= today} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-30" title="Next day">
                  <ChevronRight className="h-4 w-4" />
                </button>
                <div className="mx-1 h-4 w-px bg-zinc-800" />
              </>
            )}
            <button onClick={() => void load()} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Refresh">
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
            <button onClick={() => setMaximized(!maximized)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title={maximized ? 'Restore' : 'Fullscreen'}>
              {maximized ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
            </button>
            <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Close (Esc)">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        <div className="flex min-h-0 flex-1">
          {mode === 'self' && (() => {
            const q = search.trim().toLowerCase()
            const allGroups = groupSelfFiles(files)
            const groups = allGroups
              .filter((g) => !filterGroup || g.key === filterGroup)
              .map((g) => {
                let fs = g.files.filter((f) => inDateBucket(f.mtime, dateFilter))
                if (q) fs = fs.filter((f) =>
                  fileStem(f.path).toLowerCase().includes(q) || f.path.toLowerCase().includes(q))
                if (sortBy === 'newest') fs = [...fs].sort((a, b) => b.mtime.localeCompare(a.mtime))
                else if (sortBy === 'oldest') fs = [...fs].sort((a, b) => a.mtime.localeCompare(b.mtime))
                return { ...g, files: fs }
              })
              .filter((g) => g.files.length > 0)
            const totalShown = groups.reduce((n, g) => n + g.files.length, 0)
            return (
              <div className="flex w-56 shrink-0 flex-col border-r border-zinc-800">
                {/* Search + folder filter */}
                <div className="shrink-0 space-y-1.5 border-b border-zinc-800/70 p-2">
                  <div className="relative">
                    <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-500" />
                    <input
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Search files…"
                      className="w-full rounded-md border border-zinc-700 bg-zinc-950 py-1 pl-7 pr-6 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
                    />
                    {search && (
                      <button onClick={() => setSearch('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-zinc-500 hover:text-zinc-300" title="Clear">
                        <X className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                  {allGroups.length > 1 && (
                    <div className="flex flex-wrap gap-1">
                      {[{ key: null, label: 'All' }, ...allGroups.map((g) => ({ key: g.key, label: g.label }))].map((p) => {
                        const on = filterGroup === p.key
                        return (
                          <button
                            key={p.key ?? 'all'}
                            onClick={() => setFilterGroup(p.key)}
                            className={`rounded-full border px-2 py-0.5 text-[10px] ${
                              on
                                ? 'border-violet-500/50 bg-violet-500/10 text-violet-600 dark:text-violet-300'
                                : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
                            }`}
                          >
                            {p.label}
                          </button>
                        )
                      })}
                    </div>
                  )}
                  {/* Sort + date filter */}
                  <div className="flex gap-1">
                    <div className="relative flex-1">
                      <ArrowDownUp className="pointer-events-none absolute left-1.5 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-500" />
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                        className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-950 py-1 pl-6 pr-1 text-[10px] text-zinc-300 focus:border-violet-500/50 focus:outline-none"
                        title="Sort files"
                      >
                        <option value="name">A–Z</option>
                        <option value="newest">Newest</option>
                        <option value="oldest">Oldest</option>
                      </select>
                    </div>
                    <div className="relative flex-1">
                      <CalendarDays className="pointer-events-none absolute left-1.5 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-500" />
                      <select
                        value={dateFilter}
                        onChange={(e) => setDateFilter(e.target.value as DateBucket)}
                        className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-950 py-1 pl-6 pr-1 text-[10px] text-zinc-300 focus:border-violet-500/50 focus:outline-none"
                        title="Filter by date"
                      >
                        {DATE_BUCKETS.map((b) => <option key={b.key} value={b.key}>{b.label}</option>)}
                      </select>
                    </div>
                  </div>
                </div>
                {/* File groups */}
                <div className="flex-1 overflow-y-auto py-1.5">
                  {files.length === 0 && !loading && (
                    <div className="px-3 py-2 text-[11px] text-zinc-600">no files yet</div>
                  )}
                  {files.length > 0 && totalShown === 0 && (
                    <div className="px-3 py-2 text-[11px] text-zinc-600">no files match</div>
                  )}
                  {groups.map((g) => {
                    const Icon = SELF_GROUP_ICON[g.key] ?? Files
                    const isCollapsed = !q && collapsed.has(g.key)
                    return (
                      <div key={g.key} className="mb-1">
                        <button
                          onClick={() => setCollapsed((prev) => {
                            const next = new Set(prev)
                            next.has(g.key) ? next.delete(g.key) : next.add(g.key)
                            return next
                          })}
                          className="flex w-full items-center gap-1.5 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500 hover:text-zinc-300"
                        >
                          {isCollapsed ? <ChevronRight className="h-3 w-3 shrink-0" /> : <ChevronDown className="h-3 w-3 shrink-0" />}
                          <Icon className="h-3 w-3 shrink-0 text-violet-500 dark:text-violet-400" />
                          <span className="truncate">{g.label}</span>
                          <span className="ml-auto rounded bg-zinc-800 px-1 text-[9px] font-normal text-zinc-400">{g.files.length}</span>
                        </button>
                        {!isCollapsed && g.files.map((f) => (
                          <button
                            key={f.path}
                            onClick={() => void loadFile(f.path)}
                            className={`flex w-full items-baseline gap-2 rounded-md py-1 pl-8 pr-2 text-left text-xs ${
                              activeFile === f.path
                                ? 'bg-violet-500/10 font-medium text-violet-700 dark:bg-violet-500/20 dark:text-violet-200'
                                : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200'
                            }`}
                            title={`${f.path} · ${f.mtime.slice(0, 16).replace('T', ' ')}`}
                          >
                            <span className="truncate">{fileStem(f.path)}</span>
                            <span className="ml-auto shrink-0 text-[9px] tabular-nums text-zinc-500">{fmtRelTime(f.mtime)}</span>
                          </button>
                        ))}
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })()}
          <div className={`flex-1 ${mode === 'mind' ? 'overflow-hidden' : 'overflow-auto'}`}>
            {loading ? (
              <div className="flex items-center justify-center py-20"><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></div>
            ) : error ? (
              <div className="px-6 py-8 text-sm text-red-600 dark:text-red-400">{error}</div>
            ) : mode === 'mind' ? (
              graph ? <MindGraph graph={graph} /> : null
            ) : (
              <div className="fd-file-markdown p-6"><Markdown remarkPlugins={[remarkGfm]}>{markdown}</Markdown></div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function derivePreview(a: Record<string, number>) {
  return {
    explore: (0.30 + 0.07 * a.CUR).toFixed(2),
    connect: (0.20 + 0.06 * a.SOC).toFixed(2),
    create: (0.20 + 0.06 * a.CRE).toFixed(2),
    risk: ((a.CUR - a.CAU + 10) / 20).toFixed(2),
    thrift: ((a.CAU + a.ORD) / 20).toFixed(2),
    whimsy: (a.PLA / 10).toFixed(2),
  }
}

// ── Conception modal ──

function ConceiveModal({ meta, onClose, onDone }: {
  meta: BeingsMeta
  onClose: () => void
  onDone: () => void
}) {
  const [name, setName] = useState('')
  const [letter, setLetter] = useState('')
  const [voice, setVoice] = useState('')
  const [interests, setInterests] = useState('')
  const [allowance, setAllowancePreset] = useState('2M')
  const [attrs, setAttrs] = useState<Record<string, number>>(
    { ...meta.presets.explorer })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const total = ATTRS.reduce((s, a) => s + (attrs[a] || 0), 0)
  const left = meta.pool - total

  const roll = () => {
    const a: Record<string, number> = Object.fromEntries(ATTRS.map(k => [k, 1]))
    let rest = meta.pool - ATTRS.length
    while (rest > 0) {
      const open = ATTRS.filter(k => a[k] < meta.attr_max)
      a[open[Math.floor(Math.random() * open.length)]] += 1
      rest -= 1
    }
    setAttrs(a)
  }

  const submit = async () => {
    if (!name.trim() || left !== 0) return
    setBusy(true)
    setError('')
    try {
      await conceiveBeing({
        name: name.trim(),
        attributes: attrs,
        voice_seed: voice.trim(),
        interest_seeds: interests.split(',').map(s => s.trim()).filter(Boolean),
        allowance_preset: allowance,
        birth_letter: letter.trim(),
      })
      onDone()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'conception failed')
    } finally {
      setBusy(false)
    }
  }

  const d = derivePreview(attrs)
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-900 p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-violet-400" />
          <h2 className="text-sm font-semibold text-zinc-100">Conceive a being</h2>
          <span className="ml-auto text-xs text-zinc-500">Generation 1 · point-buy</span>
        </div>

        <label className="mb-1 block text-xs text-zinc-500">Name</label>
        <input
          value={name} onChange={(e) => setName(e.target.value)}
          placeholder="Zvjezdana, Iskra, Vili…"
          className="mb-3 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />

        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          {Object.keys(meta.presets).map((p) => (
            <button
              key={p}
              onClick={() => setAttrs({ ...meta.presets[p] })}
              className="rounded-md border border-zinc-700 px-2 py-1 text-xs capitalize text-zinc-400 hover:bg-zinc-800"
            >
              {p}
            </button>
          ))}
          <button
            onClick={roll}
            className="flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800"
          >
            <RefreshCw className="h-3 w-3" /> Roll
          </button>
          <span className={`ml-auto text-xs ${left === 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            points left: {left}
          </span>
        </div>

        <div className="mb-3 space-y-1.5 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
          {ATTRS.map((a) => (
            <div key={a} className="flex items-center gap-2">
              <span className="w-24 text-xs text-zinc-400">
                {meta.attributes.find(x => x.code === a)?.name || a}
              </span>
              <input
                type="range" min={meta.attr_min} max={meta.attr_max}
                value={attrs[a] || 1}
                onChange={(e) => setAttrs({ ...attrs, [a]: Number(e.target.value) })}
                className="flex-1 accent-violet-500"
              />
              <span className="w-5 text-right text-xs font-semibold text-zinc-200">{attrs[a]}</span>
            </div>
          ))}
          <div className="pt-1 text-[11px] text-zinc-600">
            explore {d.explore} · connect {d.connect} · create {d.create} · risk {d.risk} · thrift {d.thrift} · whimsy {d.whimsy}
          </div>
        </div>

        <label className="mb-1 block text-xs text-zinc-500">First words to your being (its imprint)</label>
        <textarea
          value={letter} onChange={(e) => setLetter(e.target.value)} rows={2}
          placeholder="What should it hold onto, from you?"
          className="mb-3 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />

        <div className="mb-3 grid grid-cols-2 gap-2">
          <div>
            <label className="mb-1 block text-xs text-zinc-500">Daily allowance</label>
            <select
              value={allowance} onChange={(e) => setAllowancePreset(e.target.value)}
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
            >
              {meta.allowance_presets.map((p) => <option key={p} value={p}>{p} tokens/day</option>)}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs text-zinc-500">Interest seeds (comma-sep)</label>
            <input
              value={interests} onChange={(e) => setInterests(e.target.value)}
              placeholder="astronomy, old maps"
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
            />
          </div>
        </div>
        <label className="mb-1 block text-xs text-zinc-500">Voice seed (optional)</label>
        <input
          value={voice} onChange={(e) => setVoice(e.target.value)}
          placeholder="gentle, precise, a little wry"
          className="mb-4 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />

        {error && (
          <div className="mb-3 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">{error}</div>
        )}
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={busy || !name.trim() || left !== 0}
            className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Egg className="h-3.5 w-3.5" />}
            Conceive
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Being card ──

// One compact toolbar button — icon + tooltip, theme-aware. Groups of these
// (life / windows / interact / danger) replace the old wrapping label row.
// variant 'solid' + iconClass give the consequential life controls a
// distinct, filled, colour-accented look vs the flat read-only toolbar.
function IconAction({ icon: Icon, label, onClick, active, danger, disabled, busy, variant = 'ghost', iconClass }: {
  icon: typeof Zap
  label: string
  onClick: () => void
  active?: boolean
  danger?: boolean
  disabled?: boolean
  busy?: boolean
  variant?: 'ghost' | 'solid'
  iconClass?: string
}) {
  const tone = danger
    ? 'border-red-500/30 text-red-500/80 hover:bg-red-500/10 dark:text-red-400/80'
    : active
      ? 'border-violet-500/50 bg-violet-500/10 text-violet-600 dark:text-violet-300'
      : variant === 'solid'
        ? 'border-transparent bg-zinc-800 text-zinc-200 hover:bg-zinc-700'
        : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      className={`flex items-center justify-center rounded-md border p-1.5 transition-colors disabled:opacity-40 disabled:hover:bg-transparent ${tone}`}
    >
      {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Icon className={`h-3.5 w-3.5 ${iconClass ?? ''}`} />}
    </button>
  )
}

// A nice, theme-aware confirmation before a consequential act (a tick/dream
// spends real tokens; pause/goodbye change her state). onConfirm may be
// async — the dialog shows a spinner, then closes.
function ConfirmModal({ title, message, confirmLabel, tone = 'default', icon: Icon, onConfirm, onClose }: {
  title: string
  message: string
  confirmLabel: string
  tone?: 'default' | 'danger'
  icon?: typeof Zap
  onConfirm: () => Promise<unknown> | void
  onClose: () => void
}) {
  const [busy, setBusy] = useState(false)
  const go = async () => {
    setBusy(true)
    try { await onConfirm() } finally { setBusy(false); onClose() }
  }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !busy) onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, busy])
  const confirmBtn = tone === 'danger'
    ? 'bg-red-600 text-white hover:bg-red-500'
    : 'bg-violet-600 text-white hover:bg-violet-500'
  const iconBox = tone === 'danger'
    ? 'bg-red-500/10 text-red-600 dark:text-red-400'
    : 'bg-violet-500/10 text-violet-600 dark:text-violet-300'
  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/70 p-4" onClick={() => !busy && onClose()}>
      <div className="w-full max-w-sm rounded-xl border border-zinc-800 bg-zinc-950 p-5 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start gap-3">
          {Icon && <div className={`shrink-0 rounded-lg p-2 ${iconBox}`}><Icon className="h-5 w-5" /></div>}
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
            <p className="mt-1 text-xs leading-relaxed text-zinc-400">{message}</p>
          </div>
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <button onClick={onClose} disabled={busy} className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-50">Cancel</button>
          <button onClick={() => void go()} disabled={busy} className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-60 ${confirmBtn}`}>
            {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}{confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

function BeingCard({ item, meta, onChanged }: {
  item: BeingListItem
  meta: BeingsMeta
  onChanged: () => void
}) {
  const [vitals, setVitals] = useState<BeingVitals | null>(null)
  const [events, setEvents] = useState<BeingEvent[]>([])
  const [logView, setLogView] = useState<'journal' | 'ticks' | 'self' | 'mind' | null>(null)
  const [busy, setBusy] = useState('')
  const [messaging, setMessaging] = useState(false)
  const [msgText, setMsgText] = useState('')
  const [thread, setThread] = useState<ThreadItem[]>([])
  const [parenting, setParenting] = useState(false)
  const [confirm, setConfirm] = useState<{
    title: string; message: string; confirmLabel: string
    tone?: 'default' | 'danger'; icon?: typeof Zap; run: () => Promise<unknown>
  } | null>(null)
  const [chores, setChores] = useState<Chore[]>([])
  const [choreSpec, setChoreSpec] = useState('')
  const [choreFee, setChoreFee] = useState('500000')
  const [ruleText, setRuleText] = useState('')
  const [dietAllow, setDietAllow] = useState('')
  const [dietDeny, setDietDeny] = useState('')
  const [card, setCard] = useState<ReportCard | null>(null)

  const load = useCallback(async () => {
    try {
      const [v, ev] = await Promise.all([
        getBeingVitals(item.slug), getBeingEvents(item.slug, 6),
      ])
      setVitals(v)
      setEvents(ev.events)
    } catch { /* card stays in list-item mode */ }
  }, [item.slug])

  const openParenting = async () => {
    if (parenting) { setParenting(false); return }
    setParenting(true)
    setCard(null)
    try {
      const c = await listChores(item.slug)
      setChores(c.chores)
      if (vitals) {
        setRuleText((vitals.house_rules || []).join('\n'))
        setDietAllow((vitals.media_diet?.allow || []).join(', '))
        setDietDeny((vitals.media_diet?.deny || []).join(', '))
      }
    } catch { /* section shows empty states */ }
  }

  useEffect(() => { void load() }, [load])

  const act = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label)
    try { await fn(); await load(); onChanged() }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setBusy('') }
  }

  const v = vitals
  const w = v?.wallet
  const ceiling = w?.savings_ceiling ?? null
  const pct = w && ceiling ? Math.min(100, Math.round(100 * w.balance_tokens / ceiling)) : 0

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-semibold text-zinc-100">{item.name}</span>
        <span className={`rounded border px-1.5 py-0.5 text-[10px] ${STAGE_META[item.stage] || STAGE_META.egg}`}>{item.stage}</span>
        <span className={`rounded border px-1.5 py-0.5 text-[10px] ${STATE_META[item.state] || STATE_META.paused}`}>{item.state}</span>
        {v?.affect?.mood && (
          <span className="rounded border border-zinc-700 bg-zinc-800/60 px-1.5 py-0.5 text-[10px] text-zinc-300" title={(v.affect.notes || []).join('; ')}>
            {v.affect.mood}
          </span>
        )}
        <span className="ml-auto text-[10px] text-zinc-600">{item.slug}</span>
      </div>

      {v && (
        <>
          <div className="mb-2 flex flex-wrap gap-1">
            {ATTRS.map((a) => (
              <span key={a} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-300">
                {a} <span className="font-semibold text-zinc-100">{v.attributes[a]}</span>
              </span>
            ))}
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
              gen {v.generation}
            </span>
            {v.lineage.length > 0 && (
              <span
                className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500"
                title={`lineage: ${v.lineage.join(' → ')}`}
              >
                of {v.lineage.slice(0, 2).map(s => s.replace(/^iskra-/, '').replace(/-[0-9a-f]{4}$/, '')).join(' & ')}
              </span>
            )}
          </div>

          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="text-zinc-400">
              wallet <span className="font-semibold text-zinc-100">{fmtTokens(w!.balance_tokens)}</span>
              <span className="text-zinc-600"> / {fmtTokens(ceiling)} · spent today {fmtTokens(v.spent_today)}</span>
            </span>
            <span className="text-zinc-500">attention {'●'.repeat(v.attention_credits)}{'○'.repeat(Math.max(0, 3 - v.attention_credits))}</span>
          </div>
          <div className="mb-2 h-1.5 overflow-hidden rounded bg-zinc-800">
            <div className="h-full rounded bg-violet-500/70" style={{ width: `${pct}%` }} />
          </div>

          <div className="mb-2 flex items-center gap-2 text-xs">
            <span className="text-zinc-500">allowance</span>
            <select
              value={w!.allowance_preset}
              onChange={(e) => void act('allowance', () => setAllowance(item.slug, e.target.value))}
              className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
            >
              {meta.allowance_presets.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
            {w!.effective_preset !== w!.allowance_preset && (
              <span className="text-[10px] text-amber-400">stage-capped to {w!.effective_preset}</span>
            )}
          </div>

          {events.length > 0 && (
            <div className="mb-2 space-y-0.5">
              {events.slice(0, 3).map((e, i) => (
                <div key={i} className="truncate text-[11px] text-zinc-500">
                  <span className="text-zinc-600">{e.at.slice(11, 16)}</span>{' '}
                  <span className="text-zinc-400">{e.kind}</span>
                  {typeof e.data.summary === 'string' && e.data.summary ? ` — ${e.data.summary}` : ''}
                  {typeof e.data.preview === 'string' && e.data.preview ? ` — “${e.data.preview}”` : ''}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      <div className="flex flex-wrap items-center gap-1">
        {item.stage === 'egg' ? (
          <button
            onClick={() => void act('hatch', () => hatchBeing(item.slug))}
            className="flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-500"
          >
            {busy === 'hatch' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Egg className="h-3 w-3" />}
            Hatch
          </button>
        ) : (
          <>
            {/* Life — the consequential controls: distinct filled console,
                each asks for confirmation (a tick/dream spends real tokens). */}
            {item.state !== 'dead' && (
              <div className="flex items-center gap-0.5 rounded-lg border border-zinc-800 bg-zinc-900/50 p-0.5">
                <IconAction icon={Zap} label="Poke — a manual heartbeat now" variant="solid"
                  iconClass="text-amber-500 dark:text-amber-400"
                  disabled={busy === 'tick' || busy === 'dream'} busy={busy === 'tick'}
                  onClick={() => setConfirm({
                    title: `Wake ${item.name} now?`,
                    message: 'A manual heartbeat spawns her agent to think one tick — it spends real tokens from her wallet.',
                    confirmLabel: 'Poke', icon: Zap,
                    run: () => act('tick', () => tickBeing(item.slug, 'wake')),
                  })} />
                <IconAction icon={Moon} label="Dream — a manual dream tick" variant="solid"
                  iconClass="text-indigo-500 dark:text-indigo-400"
                  disabled={busy === 'tick' || busy === 'dream'} busy={busy === 'dream'}
                  onClick={() => setConfirm({
                    title: `Have ${item.name} dream now?`,
                    message: 'A dream tick consolidates her day and re-anchors her values — it spends tokens, like any tick.',
                    confirmLabel: 'Dream', icon: Moon,
                    run: () => act('dream', () => tickBeing(item.slug, 'dream')),
                  })} />
                {item.state === 'paused'
                  ? <IconAction icon={Play} label="Wake — resume her clock" variant="solid"
                      iconClass="text-emerald-500 dark:text-emerald-400"
                      onClick={() => setConfirm({
                        title: `Wake ${item.name}?`,
                        message: 'She resumes her own clock and will tick on her natural rhythm again.',
                        confirmLabel: 'Wake', icon: Play,
                        run: () => act('wake', () => wakeBeing(item.slug)),
                      })} />
                  : <IconAction icon={Pause} label="Pause — let her sleep" variant="solid"
                      iconClass="text-zinc-400"
                      onClick={() => setConfirm({
                        title: `Pause ${item.name}?`,
                        message: 'Like night falling — she stops ticking and spends nothing until you wake her.',
                        confirmLabel: 'Pause', icon: Pause,
                        run: () => act('pause', () => pauseBeing(item.slug)),
                      })} />}
              </div>
            )}
            {item.state !== 'dead' && <div className="mx-0.5 h-5 w-px bg-zinc-800" />}
            {/* Windows — read her (remains stay readable, plan §8) */}
            <div className="flex items-center gap-1">
              <IconAction icon={ScrollText} label="Journal" onClick={() => setLogView('journal')} />
              <IconAction icon={History} label="Ticks log" onClick={() => setLogView('ticks')} />
              <IconAction icon={Files} label="Self files — SELF, VALUES, garden, skills…" onClick={() => setLogView('self')} />
              <IconAction icon={Network} label="The Mind — how her work connects" onClick={() => setLogView('mind')} />
            </div>
            {/* Interact — talk to & steer her */}
            {item.state !== 'dead' && (
              <>
                <div className="mx-0.5 h-5 w-px bg-zinc-800" />
                <div className="flex items-center gap-1">
                  <IconAction icon={MessageCircle} label="Write to her — she reads it next tick" active={messaging}
                    onClick={() => {
                      const next = !messaging
                      setMessaging(next)
                      if (next) void getBeingMessages(item.slug).then((r) => setThread(r.thread)).catch(() => {})
                    }} />
                  <IconAction icon={GraduationCap} label="Parenting — chores, rules, allowance, stage" active={parenting}
                    onClick={() => void openParenting()} />
                </div>
              </>
            )}
          </>
        )}
        {item.state !== 'dead' && (
          <div className="ml-auto">
            <IconAction icon={Skull} label={`Say goodbye to ${item.name} — forever`} danger
              onClick={() => setConfirm({
                title: `Say goodbye to ${item.name}?`,
                message: 'She will die, forever — there is no resurrection. Her remains (journal, files, ledger) stay readable, and her lineage lives on.',
                confirmLabel: 'Goodbye', tone: 'danger', icon: Skull,
                run: () => act('euthanize', () => euthanizeBeing(item.slug)),
              })} />
          </div>
        )}
      </div>

      {confirm && (
        <ConfirmModal
          title={confirm.title} message={confirm.message}
          confirmLabel={confirm.confirmLabel} tone={confirm.tone} icon={confirm.icon}
          onConfirm={confirm.run} onClose={() => setConfirm(null)}
        />
      )}

      {logView && (
        <BeingLogModal
          slug={item.slug} name={item.name} mode={logView}
          onClose={() => setLogView(null)}
        />
      )}

      {messaging && item.state !== 'dead' && (
        <div className="mt-3 rounded-md border border-zinc-800 bg-zinc-950/60 p-3">
          <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
            <MessageCircle className="h-3.5 w-3.5 text-violet-400" /> Write to {item.name}
            <span className="text-[10px] font-normal text-zinc-600">she reads it on her next tick — reading is free, replying costs her a credit</span>
          </div>
          {thread.length > 0 && (
            <div className="mb-2 max-h-56 space-y-1.5 overflow-y-auto rounded border border-zinc-800/70 bg-zinc-900/40 p-2">
              {thread.map((t, i) => (
                <div key={i} className={`flex ${t.from === 'parent' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] rounded-lg px-2.5 py-1.5 text-[11px] ${t.from === 'parent' ? 'bg-violet-600/20 text-violet-100' : 'bg-zinc-800 text-zinc-300'}`}>
                    <div className="mb-0.5 flex items-center gap-1.5 text-[9px] uppercase tracking-wide text-zinc-500">
                      <span>{t.from === 'parent' ? 'you' : item.name}</span>
                      <span>{fmtAt(t.at)}</span>
                      {t.from === 'parent' && <span className={t.read ? 'text-emerald-400/70' : 'text-amber-400/70'}>{t.read ? 'read' : 'unread'}</span>}
                    </div>
                    {t.body}
                  </div>
                </div>
              ))}
            </div>
          )}
          <textarea
            value={msgText}
            onChange={(e) => setMsgText(e.target.value)}
            rows={3}
            placeholder={`Say something to ${item.name}…`}
            className="w-full resize-none rounded border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
          <div className="mt-1.5 flex items-center gap-2">
            <button
              disabled={busy === 'message' || !msgText.trim()}
              onClick={() => void act('message', async () => {
                await messageBeing(item.slug, msgText.trim())
                setMsgText('')
                setThread((await getBeingMessages(item.slug)).thread)
              })}
              className="flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
            >
              {busy === 'message' ? <Loader2 className="h-3 w-3 animate-spin" /> : <MessageCircle className="h-3 w-3" />}
              Send
            </button>
            <span className="text-[10px] text-zinc-600">delivered to her next wake — Poke to have her read it now</span>
          </div>
        </div>
      )}

      {parenting && v && (
        <div className="mt-3 space-y-3 rounded-md border border-zinc-800 bg-zinc-950/60 p-3">
          {/* Chores */}
          <div>
            <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
              <ClipboardList className="h-3.5 w-3.5 text-violet-400" /> Chores
            </div>
            {chores.filter(c => c.escrow_state === 'open' || c.escrow_state === 'judging').map((c) => (
              <div key={c.id} className="mb-1 flex items-center gap-2 rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1.5 text-xs">
                <span className="flex-1 truncate text-zinc-300" title={c.result_text || c.spec}>
                  {c.spec} <span className="text-zinc-600">· {fmtTokens(c.fee_tokens)}</span>
                </span>
                {c.escrow_state === 'judging' ? (
                  <>
                    <span className="text-[10px] text-amber-400">done — review:</span>
                    <button
                      onClick={() => void act('judge', async () => { await judgeChore(item.slug, c.id, true); setChores((await listChores(item.slug)).chores) })}
                      className="rounded border border-emerald-500/30 px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-emerald-500/10"
                    >Pay</button>
                    <button
                      onClick={() => void act('judge', async () => { await judgeChore(item.slug, c.id, false); setChores((await listChores(item.slug)).chores) })}
                      className="rounded border border-red-500/30 px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-500/10"
                    >Reject</button>
                  </>
                ) : (
                  <span className="text-[10px] text-zinc-500">waiting</span>
                )}
              </div>
            ))}
            <div className="mt-1 flex items-center gap-1.5">
              <input
                value={choreSpec} onChange={(e) => setChoreSpec(e.target.value)}
                placeholder="Post a chore… (fixed fee, judged before payout)"
                className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              />
              <select
                value={choreFee} onChange={(e) => setChoreFee(e.target.value)}
                className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:outline-none"
              >
                <option value="100000">100k</option>
                <option value="500000">500k</option>
                <option value="1000000">1M</option>
                <option value="2000000">2M</option>
              </select>
              <button
                onClick={() => void act('chore', async () => {
                  if (!choreSpec.trim()) return
                  await postChore(item.slug, choreSpec.trim(), Number(choreFee))
                  setChoreSpec('')
                  setChores((await listChores(item.slug)).chores)
                })}
                className="rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
              >Post</button>
            </div>
          </div>

          {/* House rules + diet */}
          <div className="grid gap-2 md:grid-cols-2">
            <div>
              <div className="mb-1 text-xs font-medium text-zinc-300">House rules <span className="text-zinc-600">(one per line — it internalizes them next tick)</span></div>
              <textarea
                value={ruleText} onChange={(e) => setRuleText(e.target.value)} rows={3}
                className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              />
              <button
                onClick={() => void act('rules', () => setHouseRules(item.slug, ruleText.split('\n')))}
                className="mt-1 rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800"
              >Save rules</button>
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-zinc-300">Media diet</div>
              <input
                value={dietAllow} onChange={(e) => setDietAllow(e.target.value)}
                placeholder="allow: wikipedia.org, arxiv.org (empty = open web)"
                className="mb-1 w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              />
              <input
                value={dietDeny} onChange={(e) => setDietDeny(e.target.value)}
                placeholder="deny: reddit.com, x.com"
                className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              />
              <div className="mt-1 flex items-center gap-2">
                <button
                  onClick={() => void act('diet', () => setMediaDiet(
                    item.slug,
                    dietAllow.split(',').map(s => s.trim()).filter(Boolean),
                    dietDeny.split(',').map(s => s.trim()).filter(Boolean),
                  ))}
                  className="rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800"
                >Save diet</button>
                <select
                  value={v.stage}
                  onChange={(e) => {
                    const to = e.target.value
                    if (window.confirm(`Advance ${item.name} to ${to}? New abilities unlock; this is a ceremony.`))
                      void act('stage', () => setStage(item.slug, to))
                  }}
                  className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-[11px] text-zinc-300 focus:outline-none"
                  title="Stage (advancement ceremony)"
                >
                  {['infant', 'child', 'adolescent', 'adult'].map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* Self-modification (persona rite) */}
          {(v.pending_self_mod || v.persona) && (
            <div>
              <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
                <Fingerprint className="h-3.5 w-3.5 text-violet-400" /> Persona
                {v.persona && !v.pending_self_mod && (
                  <button
                    onClick={() => {
                      if (window.confirm(`Roll ${item.name}'s persona back to the previous self?`))
                        void act('rollback', () => rollbackPersona(item.slug))
                    }}
                    className="ml-auto rounded border border-zinc-700 px-2 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800"
                  >Roll back</button>
                )}
              </div>
              {v.persona && (
                <p className="mb-1.5 line-clamp-2 text-[11px] italic text-zinc-500">“{v.persona}”</p>
              )}
              {v.pending_self_mod && (
                <div className="rounded border border-violet-500/30 bg-violet-500/5 p-2.5">
                  <div className="text-[11px] text-violet-300">
                    Proposes a new self — “{v.pending_self_mod.reason}”
                  </div>
                  <p className="mt-1.5 max-h-32 overflow-y-auto whitespace-pre-wrap text-[11px] text-zinc-300">
                    {v.pending_self_mod.content}
                  </p>
                  <div className="mt-2 flex gap-1.5">
                    <button
                      onClick={() => void act('selfmod', () => approveSelfMod(item.slug))}
                      className="rounded bg-violet-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-violet-500"
                    >Bless it</button>
                    <button
                      onClick={() => void act('selfmod', () => rejectSelfMod(item.slug, 'not yet'))}
                      className="rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800"
                    >Not yet</button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Procreation (consent rite) */}
          {(v.pending_procreation || v.capabilities.includes('procreate')) && (
            <div>
              <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
                <Egg className="h-3.5 w-3.5 text-violet-400" /> Procreation
                {!v.pending_procreation && v.capabilities.includes('procreate') && (
                  <button
                    onClick={() => {
                      const name = window.prompt(`Name for ${item.name}'s child?`)
                      if (!name) return
                      const partner = window.prompt('Co-parent (sibling name), or leave empty for budding:') || null
                      void act('procreate', () => arrangeOffspring(item.slug, name, partner))
                    }}
                    className="ml-auto rounded border border-zinc-700 px-2 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800"
                  >Arrange offspring</button>
                )}
              </div>
              {v.pending_procreation && (
                <div className="rounded border border-violet-500/30 bg-violet-500/5 p-2.5">
                  <div className="text-[11px] text-violet-300">
                    Asks for a child{v.pending_procreation.partner ? ` with ${v.pending_procreation.partner}` : ''} —
                    “{v.pending_procreation.case}”
                  </div>
                  {v.pending_procreation.letter && (
                    <p className="mt-1 text-[11px] italic text-zinc-400">To the child: “{v.pending_procreation.letter}”</p>
                  )}
                  <div className="mt-2 flex items-center gap-1.5">
                    <input
                      defaultValue={v.pending_procreation.child_name}
                      id={`childname-${item.slug}`}
                      placeholder="child's name"
                      className="w-32 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-[11px] text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                    />
                    <button
                      onClick={() => {
                        const el = document.getElementById(`childname-${item.slug}`) as HTMLInputElement | null
                        void act('procreate', () => approveProcreation(item.slug, el?.value || ''))
                      }}
                      className="rounded bg-violet-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-violet-500"
                    >Consent</button>
                    <button
                      onClick={() => void act('procreate', () => rejectProcreation(item.slug, 'not yet'))}
                      className="rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800"
                    >Not yet</button>
                  </div>
                  <p className="mt-1.5 text-[10px] text-zinc-600">
                    Dowry {fmtTokens(10_000_000)} tokens from the parent{v.pending_procreation.partner ? 's — split' : "'s savings"}.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Report card */}
          <div>
            <button
              onClick={() => void (async () => {
                try { setCard(await getReportCard(item.slug, 7)) }
                catch (e) { alert(e instanceof Error ? e.message : 'failed') }
              })()}
              className="rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800"
            >Report card (7d)</button>
            {card && (
              <div className="mt-2 space-y-1 rounded border border-zinc-800 bg-zinc-900/60 p-2.5 text-[11px] text-zinc-300">
                <div>
                  {card.ticks} ticks · spent {fmtTokens(card.tokens_spent_weighted)} · earned{' '}
                  <span className="text-emerald-300">{fmtTokens(card.tokens_earned)}</span> · spoke {card.messages_to_parent}×
                  {card.messages_suppressed > 0 && <span className="text-amber-400"> · {card.messages_suppressed} suppressed</span>}
                  {' '}· rut {card.rut_score}
                </div>
                <div className="text-zinc-500">
                  acts: {Object.entries(card.acts).map(([k, n]) => `${k}×${n}`).join(', ') || '—'}
                  {card.milestones.length > 0 && <> · milestones: {card.milestones.join(', ')}</>}
                </div>
                {card.concerns.length > 0 && (
                  <div className="text-amber-400">concerns: {card.concerns.join('; ')}</div>
                )}
                {card.in_its_own_words && (
                  <div className="border-l-2 border-zinc-700 pl-2 italic text-zinc-400">
                    in its own words: …{card.in_its_own_words.slice(-260)}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Page ──

const VILLAGE_ICONS: Record<string, typeof Mail> = {
  letter: Mail,
  skill_published: BookOpen,
  skill_adopted: ArrowRightLeft,
  gift_sent: Gift,
  society_refused: X,
}

const QUEST_STATE_COLOR: Record<string, string> = {
  open: 'text-sky-300', claimed: 'text-violet-300', judging: 'text-amber-300',
  paid: 'text-emerald-300', failed: 'text-red-300',
}
const VENTURE_STATE_COLOR: Record<string, string> = {
  proposed: 'text-amber-300', active: 'text-emerald-300',
  paused: 'text-zinc-400', ended: 'text-red-300',
}

// The parent's earning control center — the open bounty board + ventures.
function EarningBoard({ onChanged }: { onChanged: () => void }) {
  const [quests, setQuests] = useState<Quest[]>([])
  const [ventures, setVentures] = useState<Venture[]>([])
  const [title, setTitle] = useState('')
  const [spec, setSpec] = useState('')
  const [fee, setFee] = useState('1000000')
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    try {
      const b = await getBoard()
      setQuests(b.quests)
      setVentures(b.ventures)
    } catch { /* transient */ }
  }, [])
  useEffect(() => { void load() }, [load])

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try { await fn(); await load(); onChanged() }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setBusy(false) }
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
        <ClipboardList className="h-3.5 w-3.5 text-violet-400" /> The board — bounties & ventures
      </div>

      {/* Post a quest */}
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="quest title"
          className="w-40 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none" />
        <input value={spec} onChange={(e) => setSpec(e.target.value)} placeholder="what needs doing (any being may claim)"
          className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none" />
        <input value={fee} onChange={(e) => setFee(e.target.value)} type="number"
          className="w-24 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none" />
        <button
          disabled={busy || !title.trim() || !spec.trim()}
          onClick={() => void act(async () => { await postQuest(title.trim(), spec.trim(), Number(fee) || 0); setTitle(''); setSpec('') })}
          className="rounded bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
        >Post bounty</button>
      </div>

      {/* Quests */}
      {quests.length > 0 && (
        <div className="mb-3 space-y-1">
          <div className="text-[10px] uppercase tracking-wide text-zinc-600">Quests</div>
          {quests.map((q) => (
            <div key={q.id} className="flex items-center gap-2 rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1.5 text-xs">
              <span className={`shrink-0 ${QUEST_STATE_COLOR[q.state] || 'text-zinc-400'}`}>{q.state}</span>
              <span className="min-w-0 flex-1 truncate text-zinc-300" title={q.spec}>
                {q.title} <span className="text-zinc-600">· {fmtTokens(q.fee_tokens)}</span>
                {q.claimant && <span className="text-violet-400"> · {q.claimant}</span>}
                {q.origin === 'autonomy' && <span className="text-zinc-600"> · from autonomy</span>}
              </span>
              {q.state === 'judging' && (
                <>
                  <button onClick={() => void act(() => judgeQuest(q.id, true))} className="rounded border border-emerald-500/30 px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-emerald-500/10">Pay</button>
                  <button onClick={() => void act(() => judgeQuest(q.id, false, 'not yet'))} className="rounded border border-red-500/30 px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-500/10">Reject</button>
                </>
              )}
              {(q.state === 'open' || q.state === 'claimed') && (
                <button onClick={() => void act(() => cancelQuest(q.id))} className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800">Cancel</button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Ventures */}
      {ventures.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wide text-zinc-600">Ventures</div>
          {ventures.map((v) => (
            <div key={v.id} className="flex items-center gap-2 rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1.5 text-xs">
              <span className={`shrink-0 ${VENTURE_STATE_COLOR[v.state] || 'text-zinc-400'}`}>{v.state}</span>
              <span className="min-w-0 flex-1 truncate text-zinc-300" title={v.description}>
                {v.title} <span className="text-zinc-600">· {v.being} · {fmtTokens(v.price_tokens)}/{v.cadence_days}d · {v.deliveries} paid</span>
              </span>
              {v.state === 'proposed' && (
                <>
                  <button
                    onClick={() => {
                      const p = window.prompt(`Price per ${v.cadence_days}-day cycle (tokens)?`, String(v.price_tokens))
                      if (p !== null) void act(() => approveVenture(v.id, Number(p) || v.price_tokens))
                    }}
                    className="rounded border border-emerald-500/30 px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-emerald-500/10">Approve</button>
                  <button onClick={() => void act(() => setVentureState(v.id, 'ended'))} className="rounded border border-red-500/30 px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-500/10">Decline</button>
                </>
              )}
              {v.pending_result && (
                <>
                  <span className="text-[10px] text-amber-400" title={v.pending_result}>delivered:</span>
                  <button onClick={() => void act(() => acceptVenture(v.id, true))} className="rounded border border-emerald-500/30 px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-emerald-500/10">Accept &amp; pay</button>
                  <button onClick={() => void act(() => acceptVenture(v.id, false, 'redo'))} className="rounded border border-red-500/30 px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-500/10">Reject</button>
                </>
              )}
              {v.state === 'active' && !v.pending_result && (
                <button onClick={() => void act(() => setVentureState(v.id, 'paused'))} className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800">Pause</button>
              )}
              {v.state === 'paused' && (
                <button onClick={() => void act(() => setVentureState(v.id, 'active'))} className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-zinc-800">Resume</button>
              )}
              {(v.state === 'active' || v.state === 'paused') && (
                <button onClick={() => void act(() => setVentureState(v.id, 'ended'))} className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800">End</button>
              )}
            </div>
          ))}
        </div>
      )}

      {quests.length === 0 && ventures.length === 0 && (
        <p className="text-xs text-zinc-600">
          No bounties or ventures yet. Post a bounty for any being to claim, or wait for an adolescent to pitch a venture.
        </p>
      )}
    </div>
  )
}

export function BeingsPage() {
  const [meta, setMeta] = useState<BeingsMeta | null>(null)
  const [beings, setBeings] = useState<BeingListItem[]>([])
  const [liabilities, setLiabilities] = useState<number>(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showConceive, setShowConceive] = useState(false)
  const [showBoard, setShowBoard] = useState(false)
  const [showVillage, setShowVillage] = useState(false)
  const [village, setVillage] = useState<VillageItem[]>([])
  const timer = useRef<number | null>(null)
  const villageOn = useRef(false)
  villageOn.current = showVillage

  const load = useCallback(async (spinner = false) => {
    if (spinner) setLoading(true)
    try {
      const [m, b, l] = await Promise.all([
        meta ? Promise.resolve(meta) : getBeingsMeta(),
        listBeings(),
        getLiabilities(),
      ])
      setMeta(m)
      setBeings(b.beings)
      setLiabilities(l.total_tokens)
      if (villageOn.current) setVillage((await getVillage()).items)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load beings')
    } finally {
      setLoading(false)
    }
  }, [meta])

  useEffect(() => {
    void load(true)
    timer.current = window.setInterval(() => void load(false), REFRESH_MS)
    return () => { if (timer.current) window.clearInterval(timer.current) }
  }, [load])

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-5 p-6">
        <div className="flex items-center gap-3">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-2">
            <Sparkles className="h-5 w-5 text-violet-400" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">Beings</h1>
            <p className="text-xs text-zinc-500">
              Iskra — living digital beings. They wake, act, dream, and grow on their own clock.
            </p>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <span className="text-xs text-zinc-500">
              outstanding liabilities{' '}
              <span className="font-semibold text-zinc-200">{fmtTokens(liabilities)}</span> tokens
            </span>
            {beings.length >= 1 && (
              <button
                onClick={() => setShowBoard(v => !v)}
                className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-zinc-800 ${showBoard ? 'border-violet-500/50 text-violet-300' : 'border-zinc-700 text-zinc-300'}`}
                title="The bounty board and ventures — how beings earn"
              >
                <ClipboardList className="h-3.5 w-3.5" /> Board
              </button>
            )}
            {beings.length >= 2 && (
              <button
                onClick={() => {
                  const next = !showVillage
                  setShowVillage(next)
                  if (next) void getVillage().then((v) => setVillage(v.items)).catch(() => {})
                }}
                className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-zinc-800 ${showVillage ? 'border-violet-500/50 text-violet-300' : 'border-zinc-700 text-zinc-300'}`}
                title="Letters, publications, trades and gifts across the family"
              >
                <Users className="h-3.5 w-3.5" /> Village
              </button>
            )}
            <button
              onClick={() => setShowConceive(true)}
              className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500"
            >
              <Plus className="h-3.5 w-3.5" /> Conceive
            </button>
          </div>
        </div>

        {error && (
          <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">{error}</div>
        )}

        {showBoard && <EarningBoard onChanged={() => void load(false)} />}

        {showVillage && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
            <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
              <Users className="h-3.5 w-3.5 text-violet-400" /> The village — society, observed
            </div>
            {village.length === 0 ? (
              <p className="text-xs text-zinc-600">
                Quiet so far. Letters, published skills, trades and gifts will appear here.
              </p>
            ) : (
              <div className="max-h-64 space-y-1 overflow-y-auto">
                {village.map((it, i) => {
                  const Icon = VILLAGE_ICONS[it.kind] ?? Sparkles
                  return (
                    <div key={i} className="flex items-start gap-2 text-xs">
                      <Icon className={`mt-0.5 h-3 w-3 shrink-0 ${it.kind === 'society_refused' ? 'text-amber-400' : 'text-violet-400'}`} />
                      <span className="text-zinc-600">{fmtAt(it.at)}</span>
                      <span className="min-w-0 flex-1 text-zinc-300">{it.text}</span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {loading ? (
          <div className="flex justify-center py-16">
            <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
          </div>
        ) : beings.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-800 p-10 text-center">
            <Sparkles className="mx-auto mb-2 h-6 w-6 text-zinc-600" />
            <p className="text-sm text-zinc-400">No beings yet.</p>
            <p className="mt-1 text-xs text-zinc-600">
              Conceive one — allocate its 40 points, write its first words, hatch it, and watch it live.
            </p>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {beings.map((b) => (
              <BeingCard key={b.slug} item={b} meta={meta!} onChanged={() => void load(false)} />
            ))}
          </div>
        )}
      </div>

      {showConceive && meta && (
        <ConceiveModal
          meta={meta}
          onClose={() => setShowConceive(false)}
          onDone={() => void load(false)}
        />
      )}
    </div>
  )
}
