import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Network, Play, Sparkles, Plus, Trash2, ThumbsUp, ThumbsDown,
  Loader2, Check, X, Wrench, Maximize2, Minimize2, Download, Paperclip, FileText, Image as ImageIcon,
  SlidersHorizontal, Eye, ScanSearch, AlertTriangle, RefreshCw, Square, CornerDownRight, Users,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useBasnaStore, parseAnalysis, type BasnaSession, type BasnaRun, type ProgressEvent, type BasnaAnalysis } from '../stores/basnaStore'
import { VatraDelegation } from '../components/VatraDelegation'
import { useTierConfig, TIER_ORDER, PROVIDERS } from '../services/tierConfig'

const COGNITIVE_MODES = ['neutra', 'ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian']

// Demo / debug tasks — each exercises a different path through the pipeline.
const BASNA_EXAMPLES: { label: string; text: string }[] = [
  {
    label: 'EU expansion (complex)',
    text: `We're a 20-person B2B SaaS selling AI-powered contract-review software to US mid-market law firms (~$2M ARR, growing ~8%/month). Decide whether we should expand into the EU in the next two quarters.

Weigh, with evidence: (1) GDPR and the EU AI Act's implications for a legal-AI tool — obligations, timelines, and risk classification; (2) the competitive landscape in DACH and France — who's already there and how we'd differentiate; (3) data-residency and localization costs (hosting, language, support, legal); (4) the realistic revenue opportunity versus the distraction risk to our US growth.

End with ONE clear recommendation (go / no-go / phased) and the three specific conditions that would flip the decision.`,
  },
  {
    label: 'Data store choice',
    text: "We're adding an append-only event log: ~50M records/month, written continuously, queried by time-range and user_id, with daily aggregations. Pick ONE data store — Postgres, ClickHouse, or DynamoDB — and give the single most important reason.",
  },
  {
    label: 'Quick fact',
    text: 'What does the SQL keyword EXPLAIN do? One sentence.',
  },
  {
    label: 'Brainstorm options',
    text: 'Brainstorm ways to cut cold-start latency for a serverless API. List distinct approaches — breadth over depth.',
  },
  {
    label: 'Migration risk',
    text: 'Name the single biggest risk in migrating a monolith to microservices.',
  },
]

const DIFFICULTY_COLOR: Record<string, string> = {
  trivial: 'text-emerald-700 dark:text-emerald-300',
  moderate: 'text-amber-700 dark:text-amber-300',
  hard: 'text-rose-700 dark:text-rose-300',
}

function Badge({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={`rounded-full border border-zinc-700/60 bg-zinc-800/60 px-2 py-0.5 text-[11px] font-medium ${className}`}>
      {children}
    </span>
  )
}

function WeightBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
      <div className="h-full rounded-full bg-sky-500" style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  )
}

function timeAgo(iso?: string): string {
  if (!iso) return ''
  const t = new Date(iso).getTime()
  if (isNaN(t)) return ''
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000))
  if (s < 60) return 'now'
  const m = Math.floor(s / 60); if (m < 60) return `${m}m`
  const h = Math.floor(m / 60); if (h < 24) return `${h}h`
  const d = Math.floor(h / 24); if (d < 7) return `${d}d`
  return new Date(t).toLocaleDateString()
}

const STATUS_DOT: Record<string, string> = {
  routing: 'bg-sky-400', routed: 'bg-zinc-400', running: 'bg-amber-400',
  done: 'bg-emerald-500', error: 'bg-rose-500',
}

// Confidence colour for the run list: green high, amber mid, rose low.
function confColor(c: number): string {
  if (c >= 0.7) return 'text-emerald-600 dark:text-emerald-400'
  if (c >= 0.4) return 'text-amber-600 dark:text-amber-400'
  return 'text-rose-600 dark:text-rose-400'
}

// Agent-started runs carry {source:'agent', origin_platform} in their config.
function agentOrigin(config?: string): string | null {
  if (!config) return null
  try {
    const c = JSON.parse(config)
    if (c && c.source === 'agent') return c.origin_platform || 'agent'
  } catch { /* ignore */ }
  return null
}

function isVatra(config?: string): boolean {
  if (!config) return false
  try { return JSON.parse(config)?.mode === 'vatra' } catch { return false }
}

// Deepen runs carry {kind:'deepen', parent_session_id} in their config.
function parentIdOf(config?: string): string | null {
  if (!config) return null
  try {
    const c = JSON.parse(config)
    if (c && c.kind === 'deepen' && c.parent_session_id) return String(c.parent_session_id)
  } catch { /* ignore */ }
  return null
}

function SessionCard({ s, active, onOpen, onDelete, onCancel }: {
  s: BasnaSession; active: boolean; onOpen: () => void; onDelete: () => void; onCancel: () => void
}) {
  const working = s.status === 'running' || s.status === 'routing' || s.status === 'routed'
  const origin = agentOrigin(s.config)
  return (
    <div
      onClick={onOpen}
      className={`group relative cursor-pointer rounded-lg border p-2.5 pl-3 transition-colors ${
        active ? 'border-sky-600/60 bg-sky-950/30' : 'border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800/50'
      }`}
    >
      {active && <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-sky-500" />}
      <div className="flex items-start gap-2">
        <span className="mt-0.5 shrink-0">
          {working
            ? <Loader2 className="h-3 w-3 animate-spin text-amber-400" />
            : <span className={`block h-2 w-2 rounded-full ${STATUS_DOT[s.status] || 'bg-zinc-500'}`} />}
        </span>
        <div className="min-w-0 flex-1">
          <p className="line-clamp-2 text-xs font-medium leading-snug text-zinc-200" title={s.intent || ''}>{s.title || s.intent || '(untitled)'}</p>
          {s.title && s.intent && (
            <p className="mt-0.5 line-clamp-1 text-[10px] leading-snug text-zinc-500" title={s.intent}>{s.intent}</p>
          )}
        </div>
        {working && (
          <button
            onClick={(e) => { e.stopPropagation(); onCancel() }}
            title="Stop this run"
            className="shrink-0 rounded p-0.5 text-amber-500 transition-colors hover:text-rose-400"
          >
            <Square className="h-3.5 w-3.5 fill-current" />
          </button>
        )}
        <button
          onClick={(e) => { e.stopPropagation(); onDelete() }}
          className="shrink-0 rounded p-0.5 text-zinc-600 opacity-0 transition-opacity hover:text-rose-400 group-hover:opacity-100"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="mt-1.5 flex items-center gap-1.5 pl-5 text-[10px] text-zinc-500">
        {origin && (
          <span
            title={`Started by an agent${origin !== 'agent' ? ` from ${origin}` : ''}`}
            className="flex shrink-0 items-center gap-1 rounded bg-violet-500/15 border border-violet-500/25 px-1.5 py-0.5 font-medium text-violet-700 dark:text-violet-300"
          >
            <Sparkles className="h-2.5 w-2.5" />agent{origin !== 'agent' ? `·${origin}` : ''}
          </span>
        )}
        {isVatra(s.config) && (
          <span title="Collaborative (Vatra) run" className="flex shrink-0 items-center gap-1 rounded border border-violet-500/25 bg-violet-500/15 px-1.5 py-0.5 font-medium text-violet-700 dark:text-violet-300">
            <Users className="h-2.5 w-2.5" />vatra
          </span>
        )}
        {s.domain && <span className="truncate rounded bg-zinc-800/70 px-1.5 py-0.5 text-zinc-400">{s.domain}</span>}
        {s.difficulty && <span className={`font-medium ${DIFFICULTY_COLOR[s.difficulty] || 'text-zinc-400'}`}>{s.difficulty}</span>}
        {s.status === 'done' && s.confidence > 0 && (
          <span className={`font-medium ${confColor(s.confidence)}`}>· {Math.round(s.confidence * 100)}%</span>
        )}
        <span className="ml-auto shrink-0 tabular-nums text-zinc-600">{timeAgo(s.updated_at || s.created_at)}</span>
      </div>
    </div>
  )
}

function analysisToMarkdown(a: BasnaAnalysis): string {
  const out: string[] = ['# Analysis', '']
  if (a.agreement?.length) {
    out.push('## Agreement', '')
    for (const x of a.agreement) out.push(`- ${x}`)
    out.push('')
  }
  if (a.differences?.length) {
    out.push('## Key differences', '')
    for (const d of a.differences) {
      out.push(`### ${d.point}`)
      for (const p of d.positions || []) out.push(`- **${p.by}** — ${p.stance}`)
      out.push('')
    }
  }
  if (a.unique?.length) {
    out.push('## Unique insights', '')
    for (const u of a.unique) out.push(`- **${u.by}** — ${u.insight}`)
    out.push('')
  }
  if (a.blind_spots?.length) {
    out.push('## Blind spots — covered by none', '')
    for (const b of a.blind_spots) out.push(`- ${b}`)
    out.push('')
  }
  return out.join('\n').trim()
}

function formatProgress(events: ProgressEvent[]): string {
  return events.map((e) => {
    const t = e.ts ? new Date(e.ts * 1000).toLocaleTimeString([], { hour12: false }) : ''
    return `${t}  ${e.stage.toUpperCase().padEnd(10)} ${e.message}`
  }).join('\n')
}

function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'basna'
}

function downloadMarkdown(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

type ViewMode = 'markdown' | 'html' | 'text'

function fileExt(name: string): string {
  const m = name.toLowerCase().match(/\.([a-z0-9]+)$/)
  return m ? m[1] : ''
}

// File types we can preview in-app — markdown, html, plain text, and scripts.
// Anything else stays download-only.
const VIEWABLE_EXTS = new Set([
  'md', 'markdown', 'txt', 'text', 'log', 'html', 'htm',
  'json', 'csv', 'tsv', 'xml', 'yaml', 'yml', 'toml', 'ini',
  'py', 'sh', 'bash', 'js', 'mjs', 'ts', 'tsx', 'jsx', 'css', 'sql',
])
function isViewable(name: string): boolean { return VIEWABLE_EXTS.has(fileExt(name)) }
function viewModeForFile(name: string): ViewMode {
  const e = fileExt(name)
  if (e === 'md' || e === 'markdown') return 'markdown'
  if (e === 'html' || e === 'htm') return 'html'
  return 'text'
}

// Fullscreen-capable preview modal. Renders by mode: markdown (with GFM tables),
// raw HTML (sandboxed iframe), or plain text / source. The maximise button grows
// it to near-fullscreen; click the backdrop or ✕ to close.
function FileModal({ title, content, mode, onClose }: {
  title: string; content: string; mode: ViewMode; onClose: () => void
}) {
  const [maximized, setMaximized] = useState(false)
  const ext = fileExt(title)
  const exportName = ext ? title : `${slugify(title)}.md`
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className={`flex flex-col rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl ${
          maximized ? 'h-[96vh] w-[97vw] max-w-none' : 'max-h-[90vh] w-full max-w-4xl'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-800 px-4 py-3">
          <span className="truncate text-sm font-medium text-zinc-200">{title}</span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setMaximized((m) => !m)}
              title={maximized ? 'Restore' : 'Maximise'}
              className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              {maximized ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </button>
            <button
              onClick={() => downloadMarkdown(exportName, content)}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-200 hover:bg-zinc-800"
            >
              <Download className="h-3.5 w-3.5" /> Export {ext ? `.${ext}` : '.md'}
            </button>
            <button onClick={onClose} className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        {mode === 'html' ? (
          <iframe
            title={title}
            sandbox=""
            srcDoc={content}
            className="min-h-[60vh] w-full flex-1 rounded-b-xl bg-white"
          />
        ) : mode === 'markdown' ? (
          <div className="fd-markdown flex-1 overflow-auto p-5 text-sm text-zinc-200 leading-relaxed">
            <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
          </div>
        ) : (
          <pre className="flex-1 overflow-auto whitespace-pre-wrap break-words p-5 font-mono text-xs leading-relaxed text-zinc-300">
            {content}
          </pre>
        )}
      </div>
    </div>
  )
}

// Compact fullscreen + export buttons reused by the truth card and agent cards.
function OutputActions({ title, content, onView }: { title: string; content: string; onView: (t: string, c: string) => void }) {
  return (
    <div className="flex items-center gap-1">
      <button onClick={() => onView(title, content)} title="Fullscreen" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
        <Maximize2 className="h-3.5 w-3.5" />
      </button>
      <button onClick={() => downloadMarkdown(`${slugify(title)}.md`, content)} title="Export .md" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
        <Download className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

function AgentRow({ run, onFeedback, onView }: { run: BasnaRun; onFeedback: (success: boolean) => void; onView: (t: string, c: string) => void }) {
  const scored = run.success !== null
  let actions: { tool: string; detail?: string }[] = []
  try { actions = JSON.parse(run.actions || '[]') } catch { actions = [] }
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-zinc-200">{run.role || run.archetype_id}</span>
          <Badge className="text-sky-700 dark:text-sky-300">{run.tier}</Badge>
          {scored && (
            run.success === 1
              ? <Badge className="text-emerald-700 dark:text-emerald-300">success</Badge>
              : <Badge className="text-rose-700 dark:text-rose-300">fail</Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          {run.output && <OutputActions title={run.role || run.archetype_id} content={run.output} onView={onView} />}
          <span className="text-[11px] text-zinc-500">{(run.latency_ms / 1000).toFixed(1)}s</span>
        </div>
      </div>
      <div className="mt-2 flex items-center gap-2">
        <span className="w-16 shrink-0 text-[11px] text-zinc-500">weight {run.weight_at_run.toFixed(2)}</span>
        <WeightBar value={run.weight_at_run} />
      </div>
      {run.output && (
        <div className="fd-markdown mt-2 max-h-48 overflow-auto text-xs text-zinc-400 leading-relaxed">
          <Markdown remarkPlugins={[remarkGfm]}>{run.output}</Markdown>
        </div>
      )}
      {actions.length > 0 && (
        <div className="mt-2 rounded-md border border-zinc-800 bg-zinc-950/40 p-2">
          <div className="mb-1 flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">Activity ({actions.length})</span>
            <button
              onClick={() => downloadMarkdown(
                `${slugify(run.role || run.archetype_id)}-activity.md`,
                actions.map((a) => `- ${a.tool}${a.detail ? ': ' + a.detail : ''}`).join('\n'),
              )}
              title="Export activity"
              className="rounded p-0.5 text-zinc-600 hover:text-zinc-300"
            >
              <Download className="h-3 w-3" />
            </button>
          </div>
          <div className="space-y-0.5">
            {actions.map((a, i) => (
              <div key={i} className="flex items-baseline gap-2 text-[11px]">
                <Wrench className="h-3 w-3 shrink-0 text-zinc-600" />
                <span className="font-mono text-zinc-400">{a.tool}</span>
                {a.detail && <span className="truncate text-zinc-600">{a.detail}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="mt-2 flex items-center gap-2">
        <span className="text-[11px] text-zinc-500">Was this contribution good?</span>
        <button
          onClick={() => onFeedback(true)}
          className={`rounded p-1 transition-colors ${run.success === 1 ? 'text-emerald-400' : 'text-zinc-500 hover:text-emerald-400'}`}
          title="Mark as good"
        >
          <ThumbsUp className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => onFeedback(false)}
          className={`rounded p-1 transition-colors ${run.success === 0 ? 'text-rose-400' : 'text-zinc-500 hover:text-rose-400'}`}
          title="Mark as poor"
        >
          <ThumbsDown className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}

// Compact token count: 1_234 → "1.2k", 244_461 → "244k".
function fmtTok(n?: number): string {
  if (!n || n <= 0) return '0'
  if (n >= 10000) return Math.round(n / 1000) + 'k'
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k'
  return String(n)
}

interface LiveAgent { role: string; actions: ProgressEvent[]; usage?: ProgressEvent; done?: boolean; ok?: boolean }

// Live per-agent panels built from the streaming progress events while the run
// is in flight — each agent's tool calls and running LLM token usage, before the
// final persisted runs (with output + feedback) take over at the end.
function LiveAgentsPanel({ agents }: { agents: LiveAgent[] }) {
  return (
    <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
      {agents.map((a) => (
        <div key={a.role} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-2">
              {a.done
                ? (a.ok === false
                    ? <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
                    : <Check className="h-3.5 w-3.5 shrink-0 text-emerald-500" />)
                : <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-sky-400" />}
              <span className="truncate text-sm font-medium text-zinc-200">{a.role}</span>
            </div>
            {a.usage && (
              <span className="shrink-0 font-mono text-[10px] tabular-nums text-zinc-500" title="LLM tokens (prompt → completion)">
                {fmtTok(a.usage.prompt_tokens)}→{fmtTok(a.usage.completion_tokens)} tok
              </span>
            )}
          </div>
          <div className="mt-1.5 text-[10px] font-semibold uppercase tracking-wide text-zinc-600">
            Activity ({a.actions.length})
          </div>
          <div className="mt-1 max-h-48 space-y-0.5 overflow-auto">
            {a.actions.slice(-14).map((ev) => (
              <div key={ev.i} className="flex items-baseline gap-2 text-[11px]">
                {ev.stage === 'narration'
                  ? <Sparkles className="h-3 w-3 shrink-0 text-zinc-500" />
                  : <Wrench className="h-3 w-3 shrink-0 text-zinc-600" />}
                <span className="shrink-0 font-mono text-zinc-400">{ev.tool}</span>
                {ev.detail && <span className="truncate text-zinc-600">{ev.detail}</span>}
              </div>
            ))}
            {a.actions.length === 0 && <div className="text-[11px] text-zinc-600">working…</div>}
          </div>
        </div>
      ))}
    </div>
  )
}

export function BasnaPage() {
  const {
    sessions, activeSession, routePlan, runs, lastExecute, progress, attachments,
    routing, executing, recompiling, error,
    routerTier, maxAgents, setRouterTier, setMaxAgents, addFiles, removeFile, downloadFile, fetchFileText,
    updateSelected, loadSessions, pollRunning, selectSession, newSession, route, saveTitle, execute, recompile, sendFeedback, deleteSession, cancelSession, deepenSession,
  } = useBasnaStore()
  const { tiers, registry, envVars } = useTierConfig()

  const [intent, setIntent] = useState('')
  const [title, setTitle] = useState('')
  const [agentOnly, setAgentOnly] = useState(false)
  const [deepening, setDeepening] = useState(false)
  const [modal, setModal] = useState<{ title: string; content: string; mode: ViewMode } | null>(null)
  const viewFull = (title: string, content: string) => setModal({ title, content, mode: 'markdown' })
  // Preview a generated file: fetch its text and render by type (md/html/text).
  const viewFile = async (name: string) => {
    const text = await fetchFileText(name)
    setModal({ title: name, content: text, mode: viewModeForFile(name) })
  }
  const [editing, setEditing] = useState<Record<number, boolean>>({})
  const progressRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const handlePaste = (e: React.ClipboardEvent) => {
    const imgs: File[] = []
    for (const it of Array.from(e.clipboardData?.items || [])) {
      if (it.type.startsWith('image/')) {
        const f = it.getAsFile()
        if (f) {
          const ext = it.type.split('/')[1] || 'png'
          imgs.push(f.name && f.name !== 'image.png' ? f : new File([f], `pasted-${Date.now()}.${ext}`, { type: it.type }))
        }
      }
    }
    if (imgs.length) addFiles(imgs)
  }

  // Keep the progress log pinned to the newest line as events stream in.
  useEffect(() => {
    const el = progressRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [progress.length])

  useEffect(() => { loadSessions() }, [loadSessions])
  useEffect(() => { setIntent(activeSession?.intent || '') }, [activeSession?.id, activeSession?.intent])
  useEffect(() => { setTitle(activeSession?.title || '') }, [activeSession?.id, activeSession?.title])

  // Live monitor: while any run (incl. agent-started) is mid-flight, poll the
  // list status + the open session's progress every few seconds; stop when idle.
  const anyRunning = sessions.some((s) => ['routing', 'routed', 'running'].includes(s.status))
  useEffect(() => {
    if (!anyRunning || executing) return
    const iv = setInterval(() => { pollRunning() }, 4000)
    return () => clearInterval(iv)
  }, [anyRunning, executing, pollRunning])

  const visibleSessions = agentOnly ? sessions.filter((s) => agentOrigin(s.config)) : sessions

  // A run already in flight (e.g. a deepen that route+ran server-side) must not
  // be re-routed or re-run. 'routed' stays runnable — that's the normal Route→Run step.
  const activeBusy = !!activeSession && (activeSession.status === 'running' || activeSession.status === 'routing')
  const canRoute = intent.trim().length > 0 && !routing && !activeBusy
  const canRun = !!routePlan && !!activeSession && !executing && !activeBusy
  const truth = lastExecute?.truth ?? activeSession?.truth ?? ''
  const confidence = lastExecute?.confidence ?? activeSession?.confidence ?? 0
  const analysis = lastExecute?.analysis ?? parseAnalysis(activeSession?.analysis)
  // Subject for download filenames: the run's title, else the first words of the
  // task — so analysis/truth export as "<subject>-analysis.md" / "…-compiled-truth.md".
  const subject = (activeSession?.title || '').trim()
    || (activeSession?.intent || '').trim().split(/\s+/).slice(0, 8).join(' ')
    || 'basna'

  // Group the streaming progress into live per-agent panels (preserving the
  // order each agent first appears). `usage` events update the running token
  // counter; everything else is a tool call / narration in that agent's feed.
  const liveAgents = useMemo<LiveAgent[]>(() => {
    const byRole = new Map<string, LiveAgent>()
    const order: string[] = []
    for (const ev of progress) {
      if (!ev.agent) continue
      let a = byRole.get(ev.agent)
      if (!a) { a = { role: ev.agent, actions: [] }; byRole.set(ev.agent, a); order.push(ev.agent) }
      if (ev.stage === 'usage') a.usage = ev
      else if (ev.stage === 'dispatch') { a.done = true; a.ok = ev.ok !== false }
      else a.actions.push(ev)
    }
    return order.map((r) => byRole.get(r) as LiveAgent)
  }, [progress])

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex shrink-0 items-center gap-2 border-b border-zinc-700/50 bg-zinc-900/50 px-4 py-3 md:px-6">
        <Network className="h-5 w-5 text-sky-600 dark:text-sky-400" />
        <div>
          <h1 className="text-sm font-semibold text-zinc-100">Basna</h1>
          <p className="text-[11px] text-zinc-500">Route → spawn the minimal team → merge by reliability</p>
        </div>
        <button
          onClick={() => { newSession(); setIntent('') }}
          className="ml-auto flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
        >
          <Plus className="h-3.5 w-3.5" /> New
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Session list */}
        <div className="flex w-80 shrink-0 flex-col overflow-hidden border-r border-zinc-800 lg:w-96">
          <div className="flex items-center gap-2 px-3 pt-3 pb-1.5">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Runs</span>
            {sessions.some((s) => agentOrigin(s.config)) && (
              <button
                onClick={() => setAgentOnly((v) => !v)}
                title="Show only agent-started runs"
                className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium border transition-colors ${
                  agentOnly
                    ? 'border-violet-500/40 bg-violet-500/15 text-violet-700 dark:text-violet-300'
                    : 'border-zinc-700 text-zinc-500 hover:text-zinc-300'
                }`}
              >
                <Sparkles className="h-2.5 w-2.5" /> agent
              </button>
            )}
            {visibleSessions.length > 0 && <span className="ml-auto text-[10px] tabular-nums text-zinc-600">{visibleSessions.length}</span>}
          </div>
          <div className="flex-1 space-y-1.5 overflow-auto px-3 pb-3">
          {visibleSessions.length === 0 && <p className="px-1 py-2 text-xs text-zinc-600">{agentOnly ? 'No agent-started runs yet.' : 'No runs yet — describe a task and Route.'}</p>}
          {visibleSessions.map((s) => (
            <SessionCard
              key={s.id}
              s={s}
              active={activeSession?.id === s.id}
              onOpen={() => selectSession(s.id)}
              onDelete={() => {
                const raw = (s.title || s.intent || '').trim()
                const label = raw.slice(0, 80)
                if (window.confirm(`Delete this Basna run?${label ? `\n\n"${label}${raw.length > 80 ? '…' : ''}"` : ''}`)) {
                  deleteSession(s.id)
                }
              }}
              onCancel={() => cancelSession(s.id)}
            />
          ))}
          </div>
        </div>

        {/* Detail */}
        <div className="flex-1 overflow-auto p-4 md:p-6">
          <div className="mx-auto max-w-3xl space-y-5">
            {/* Intent + controls */}
            <div
              className={`rounded-lg border bg-zinc-900/50 p-4 ${dragOver ? 'border-sky-500' : 'border-zinc-800'}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files) }}
            >
              {activeSession && (() => {
                const pid = parentIdOf(activeSession.config)
                const parent = pid ? sessions.find((s) => s.id === pid) : null
                const children = sessions.filter((s) => parentIdOf(s.config) === activeSession.id)
                if (!parent && children.length === 0) return null
                const linkCls = 'flex items-center gap-1 text-left text-violet-700 hover:underline dark:text-violet-300'
                return (
                  <div className="mb-3 flex flex-col gap-1 rounded-lg border border-violet-300 bg-violet-50 px-3 py-2 text-[11px] dark:border-violet-900/40 dark:bg-violet-950/20">
                    {parent && (
                      <button onClick={() => selectSession(parent.id)} className={linkCls}>
                        <CornerDownRight className="h-3 w-3 shrink-0" />
                        <span className="truncate">deepened from “{parent.title || parent.intent || 'run'}”</span>
                      </button>
                    )}
                    {children.map((ch) => (
                      <button key={ch.id} onClick={() => selectSession(ch.id)} className={linkCls}>
                        <ScanSearch className="h-3 w-3 shrink-0" />
                        <span className="truncate">
                          deepened into “{ch.title || ch.intent || 'run'}”{ch.status !== 'done' ? ` · ${ch.status}` : ''}
                        </span>
                      </button>
                    ))}
                  </div>
                )
              })()}
              <label className="mb-1.5 block text-xs font-medium text-zinc-400">
                Title <span className="font-normal text-zinc-600">— optional, auto-generated from the task if blank</span>
              </label>
              <div className="mb-3 flex items-center gap-2">
                <input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g. Q3 competitor scan"
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
                />
                {activeSession && title.trim() !== (activeSession.title || '') && (
                  <button
                    onClick={() => saveTitle(title)}
                    title="Save title"
                    className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800"
                  >
                    <Check className="h-3.5 w-3.5" /> Save
                  </button>
                )}
              </div>
              <label className="mb-1.5 block text-xs font-medium text-zinc-400">Task / intent</label>
              <textarea
                value={intent}
                onChange={(e) => setIntent(e.target.value)}
                onPaste={handlePaste}
                rows={9}
                placeholder="Describe the task, or attach/drop/paste files. The router picks the smallest team that can answer it well."
                className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 p-2.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
              />

              {/* Attachments */}
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
                >
                  <Paperclip className="h-3.5 w-3.5" /> Attach
                </button>
                <input
                  ref={fileInputRef} type="file" multiple className="hidden"
                  onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }}
                />
                {attachments.filter((a) => a.kind !== 'generated').map((a) => (
                  <span key={a.name} className="flex items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-800/60 px-2 py-0.5 text-[11px] text-zinc-300">
                    {a.mime.startsWith('image/') ? <ImageIcon className="h-3 w-3 text-zinc-500" /> : <FileText className="h-3 w-3 text-zinc-500" />}
                    {a.name}
                    <span className="text-zinc-600">{Math.max(1, Math.round(a.size / 1024))}kb</span>
                    <button onClick={() => removeFile(a.name)} className="text-zinc-500 hover:text-rose-400"><X className="h-3 w-3" /></button>
                  </span>
                ))}
              </div>
              {!intent.trim() && (
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <span className="text-[11px] text-zinc-600">Try:</span>
                  {BASNA_EXAMPLES.map((ex) => (
                    <button
                      key={ex.label}
                      onClick={() => setIntent(ex.text)}
                      title={ex.text}
                      className="rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-1 text-[11px] text-sky-300 hover:bg-sky-500/20 transition-colors"
                    >
                      {ex.label}
                    </button>
                  ))}
                </div>
              )}
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2 text-xs text-zinc-400">
                  Router tier
                  <select
                    value={routerTier}
                    onChange={(e) => setRouterTier(e.target.value)}
                    title="Which Library tier picks the archetypes"
                    className="rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-sky-600 focus:outline-none"
                  >
                    {TIER_ORDER.filter((t) => tiers[t]).map((t) => (
                      <option key={t} value={t}>{registry?.tiers[t]?.label || t}</option>
                    ))}
                    {Object.keys(tiers).length === 0 && <option value="reason">reason</option>}
                  </select>
                </label>
                <label className="flex items-center gap-2 text-xs text-zinc-400">
                  Max agents
                  <input
                    type="number" min={1} max={10} value={maxAgents}
                    onChange={(e) => setMaxAgents(Number(e.target.value))}
                    className="w-16 rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-sky-600 focus:outline-none"
                  />
                </label>
                <div className="ml-auto flex items-center gap-2">
                  <button
                    onClick={() => route(intent, tiers, title)}
                    disabled={!canRoute}
                    className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800 disabled:opacity-40"
                  >
                    {routing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                    Route
                  </button>
                  <button
                    onClick={() => execute(tiers, envVars)}
                    disabled={!canRun}
                    className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
                  >
                    {executing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                    Run ensemble
                  </button>
                </div>
              </div>
              {error && (
                <div className="mt-3 flex items-start gap-2 rounded-lg border border-rose-900/50 bg-rose-950/30 p-2.5 text-xs text-rose-300">
                  <X className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {error}
                </div>
              )}
            </div>

            {/* Route plan */}
            {routePlan && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                    {routePlan.mode === 'vatra' ? 'Team plan' : 'Route plan'}
                  </span>
                  <Badge className="text-zinc-300">{routePlan.domain}</Badge>
                  {routePlan.mode === 'vatra' ? (
                    <Badge className="text-violet-600 dark:text-violet-300">collaborative</Badge>
                  ) : (
                    <>
                      <Badge className={DIFFICULTY_COLOR[routePlan.difficulty] || 'text-zinc-300'}>{routePlan.difficulty}</Badge>
                      <Badge className="text-sky-700 dark:text-sky-300">{routePlan.merge_kind}</Badge>
                    </>
                  )}
                  {routePlan.source && <Badge className="text-zinc-500">{routePlan.source}</Badge>}
                  <span className="ml-auto text-[11px] text-zinc-600">
                    {routePlan.selected.length} {routePlan.mode === 'vatra' ? 'owner(s)' : 'agent(s)'}
                  </span>
                </div>
                {routePlan.rationale && <p className="mb-3 text-xs text-zinc-400">{routePlan.rationale}</p>}
                <div className="space-y-2">
                  {routePlan.selected.map((sel, idx) => {
                    const tc = tiers[sel.tier]
                    const arch = registry?.archetypes.find((a) => a.id === sel.archetype_id)
                    const dispProvider = sel.provider ?? tc?.provider ?? ''
                    const dispModel = sel.model ?? tc?.model ?? ''
                    const isOpen = !!editing[idx]
                    const fld = 'w-full rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-xs text-zinc-200 focus:border-sky-600 focus:outline-none'
                    const lbl = 'mb-0.5 block text-[10px] font-medium text-zinc-500'
                    return (
                    <div key={sel.archetype_id} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-medium text-zinc-200">{sel.role || sel.archetype_id}</span>
                        <div className="flex items-center gap-1.5">
                          <Badge className="text-sky-700 dark:text-sky-300">{sel.tier}</Badge>
                          <button
                            onClick={() => setEditing((e) => ({ ...e, [idx]: !e[idx] }))}
                            title="Edit agent"
                            className={`rounded p-1 ${isOpen ? 'text-sky-400' : 'text-zinc-500 hover:text-zinc-200'}`}
                          >
                            <SlidersHorizontal className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                      <p className="mt-0.5 font-mono text-[11px] text-zinc-600">
                        {dispModel ? `${dispProvider}/${dispModel}` : `${sel.tier} tier (model from server)`}
                      </p>
                      {sel.why && <p className="mt-1 text-xs text-zinc-500">{sel.why}</p>}
                      <div className="mt-2 flex items-center gap-2">
                        <span className="w-20 shrink-0 text-[11px] text-zinc-500">prior {sel.prior_weight.toFixed(2)}</span>
                        <WeightBar value={sel.prior_weight} />
                      </div>

                      {isOpen && (
                        <div className="mt-3 space-y-2 border-t border-zinc-800 pt-3">
                          <div>
                            <label className={lbl}>Role</label>
                            <input className={fld} value={sel.role}
                              onChange={(e) => updateSelected(idx, { role: e.target.value })} />
                          </div>
                          <div>
                            <label className={lbl}>Tier</label>
                            <div className="flex flex-wrap gap-1.5">
                              {TIER_ORDER.filter((t) => tiers[t]).map((t) => (
                                <button
                                  key={t}
                                  onClick={() => updateSelected(idx, {
                                    tier: t, provider: undefined, model: undefined, api_key: undefined,
                                    base_url: undefined, max_context: undefined, max_tokens: undefined,
                                  })}
                                  className={`rounded-full border px-2.5 py-0.5 text-[11px] ${
                                    sel.tier === t ? 'border-sky-500 bg-sky-500/15 text-sky-300'
                                      : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'
                                  }`}
                                >
                                  {registry?.tiers[t]?.label || t}
                                </button>
                              ))}
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <label className={lbl}>Provider</label>
                              <select className={fld} value={dispProvider}
                                onChange={(e) => updateSelected(idx, { provider: e.target.value })}>
                                {dispProvider && !PROVIDERS.includes(dispProvider) && <option value={dispProvider}>{dispProvider}</option>}
                                {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
                              </select>
                            </div>
                            <div>
                              <label className={lbl}>Model</label>
                              <input className={fld} value={dispModel} placeholder="(tier)"
                                onChange={(e) => updateSelected(idx, { model: e.target.value })} />
                            </div>
                          </div>
                          <div>
                            <label className={lbl}>API key</label>
                            <input className={fld} type="password" value={sel.api_key ?? ''}
                              placeholder="leave blank to use the tier key"
                              onChange={(e) => updateSelected(idx, { api_key: e.target.value })} />
                          </div>
                          <div>
                            <label className={lbl}>Base URL</label>
                            <input className={fld} value={sel.base_url ?? tc?.base_url ?? ''} placeholder="(tier)"
                              onChange={(e) => updateSelected(idx, { base_url: e.target.value })} />
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <label className={lbl}>Input ctx</label>
                              <input className={fld} type="number" value={sel.max_context ?? tc?.input_ctx ?? 0}
                                onChange={(e) => updateSelected(idx, { max_context: Number(e.target.value) || 0 })} />
                            </div>
                            <div>
                              <label className={lbl}>Output ctx</label>
                              <input className={fld} type="number" value={sel.max_tokens ?? tc?.output_ctx ?? 0}
                                onChange={(e) => updateSelected(idx, { max_tokens: Number(e.target.value) || 0 })} />
                            </div>
                          </div>
                          <div>
                            <label className={lbl}>Cognitive mode</label>
                            {(() => {
                              const cm = sel.cognitive_mode ?? arch?.cognitive_mode ?? 'neutra'
                              return (
                                <select className={fld} value={cm}
                                  onChange={(e) => updateSelected(idx, { cognitive_mode: e.target.value })}>
                                  {cm && !COGNITIVE_MODES.includes(cm) && <option value={cm}>{cm}</option>}
                                  {COGNITIVE_MODES.map((m) => <option key={m} value={m}>{m}</option>)}
                                </select>
                              )
                            })()}
                          </div>
                          <div>
                            <label className={lbl}>Fleet instructions (system prompt)</label>
                            <textarea className={`${fld} resize-y font-mono`} rows={16}
                              value={sel.fleet_instructions ?? arch?.fleet_instructions ?? ''}
                              onChange={(e) => updateSelected(idx, { fleet_instructions: e.target.value })} />
                          </div>
                          <div>
                            <label className={lbl}>Extra task instructions (appended to the prompt)</label>
                            <textarea className={`${fld} resize-y`} rows={8} value={sel.extra ?? ''}
                              placeholder="optional — e.g. focus areas, output format, constraints"
                              onChange={(e) => updateSelected(idx, { extra: e.target.value })} />
                          </div>
                        </div>
                      )}
                    </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Vatra (collaborative mode): the decomposition + delegation blackboard. */}
            {routePlan?.mode === 'vatra' && activeSession && (
              <VatraDelegation
                sessionId={activeSession.id}
                subtasks={routePlan.subtasks}
                active={activeBusy || executing}
              />
            )}

            {/* Live per-agent panels — actions + running LLM usage as they stream.
                Show while THIS client executes OR the active run is in flight
                (e.g. a deepen that route+ran server-side — executing is false there). */}
            {(executing || activeBusy) && liveAgents.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                  Agents working ({liveAgents.length})
                </span>
                <LiveAgentsPanel agents={liveAgents} />
              </div>
            )}

            {/* Live progress log */}
            {(executing || progress.length > 0) && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="mb-2 flex items-center gap-2">
                  {executing
                    ? <Loader2 className="h-4 w-4 animate-spin text-sky-400" />
                    : <Check className="h-4 w-4 text-emerald-500" />}
                  <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Progress</span>
                  {progress.length > 30 && (
                    <span className="text-[10px] text-zinc-600">showing last 30 of {progress.length}</span>
                  )}
                  <button
                    onClick={() => downloadMarkdown('basna-progress.md', formatProgress(progress))}
                    title="Export progress log"
                    className="ml-auto rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                  >
                    <Download className="h-3.5 w-3.5" />
                  </button>
                </div>
                <div ref={progressRef} className="max-h-72 space-y-1 overflow-auto">
                  {progress.slice(-30).map((ev) => (
                    <div key={ev.i} className="flex items-baseline gap-2 text-xs">
                      <span className="shrink-0 font-mono text-[10px] tabular-nums text-zinc-600">
                        {ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString([], { hour12: false }) : ''}
                      </span>
                      <span className="w-16 shrink-0 font-mono text-[10px] uppercase text-zinc-600">{ev.stage}</span>
                      <span className={
                        ev.stage === 'narration' ? 'text-zinc-100 font-medium'
                          : ev.ok === false ? 'text-rose-400'
                          : 'text-zinc-400'
                      }>{ev.message}</span>
                    </div>
                  ))}
                  {executing && progress.length === 0 && (
                    <div className="text-xs text-zinc-500">Starting…</div>
                  )}
                </div>
              </div>
            )}

            {/* Cross-agent analysis */}
            {analysis && (
              ((analysis.agreement?.length || analysis.differences?.length || analysis.unique?.length || analysis.blind_spots?.length) ? (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <ScanSearch className="h-4 w-4 text-violet-600 dark:text-violet-400" />
                  <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Analysis</span>
                  <div className="ml-auto">
                    <OutputActions title={`${subject} — Analysis`} content={analysisToMarkdown(analysis)} onView={viewFull} />
                  </div>
                </div>

                {!!analysis.agreement?.length && (
                  <div>
                    <div className="mb-1 text-xs font-semibold text-emerald-700 dark:text-emerald-300">Agreement</div>
                    <ul className="ml-4 list-disc space-y-1 text-sm text-zinc-200">
                      {analysis.agreement.map((a, i) => <li key={i}>{a}</li>)}
                    </ul>
                  </div>
                )}

                {!!analysis.differences?.length && (
                  <div>
                    <div className="mb-1 text-xs font-semibold text-amber-700 dark:text-amber-300">Key differences</div>
                    <div className="space-y-2">
                      {analysis.differences.map((d, i) => (
                        <div key={i} className="text-sm">
                          <div className="text-zinc-200">{d.point}</div>
                          {!!d.positions?.length && (
                            <div className="ml-3 mt-0.5 space-y-0.5">
                              {d.positions.map((p, j) => (
                                <div key={j} className="text-zinc-400">
                                  <span className="font-mono text-zinc-400">{p.by}</span> — {p.stance}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {!!analysis.unique?.length && (
                  <div>
                    <div className="mb-1 text-xs font-semibold text-sky-700 dark:text-sky-300">Unique insights</div>
                    <ul className="ml-4 list-disc space-y-1 text-sm text-zinc-200">
                      {analysis.unique.map((u, i) => (
                        <li key={i}><span className="font-mono text-zinc-400">{u.by}</span> — {u.insight}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {!!analysis.blind_spots?.length && (
                  <div className="rounded-md border border-rose-300 bg-rose-50 p-3 dark:border-rose-900/40 dark:bg-rose-950/20">
                    <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-rose-700 dark:text-rose-300">
                      <AlertTriangle className="h-3.5 w-3.5" /> Blind spots — covered by none
                    </div>
                    <ul className="ml-4 list-disc space-y-1 text-sm text-rose-800 dark:text-rose-200/90">
                      {analysis.blind_spots.map((b, i) => <li key={i}>{b}</li>)}
                    </ul>
                    {activeSession?.status === 'done' && truth && (
                      <button
                        onClick={async () => {
                          if (!activeSession) return
                          setDeepening(true)
                          try { await deepenSession(activeSession.id) }
                          catch { /* surfaced by store error path */ }
                          finally { setDeepening(false) }
                        }}
                        disabled={deepening}
                        title="Spawn a follow-up run focused on these blind spots, seeded with this run's result"
                        className="mt-3 flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
                      >
                        {deepening ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ScanSearch className="h-3.5 w-3.5" />}
                        Investigate blind spots
                      </button>
                    )}
                  </div>
                )}
              </div>
              ) : null)
            )}

            {/* Truth */}
            {truth && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <Check className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                  <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Compiled truth</span>
                  {lastExecute?.method && <Badge className="text-sky-700 dark:text-sky-300">{lastExecute.method}</Badge>}
                  <span className="ml-auto flex items-center gap-2 text-[11px] text-zinc-500">
                    confidence {(confidence * 100).toFixed(0)}%
                    <span className="h-1.5 w-20 overflow-hidden rounded-full bg-zinc-800">
                      <span className="block h-full rounded-full bg-emerald-500" style={{ width: `${Math.round(confidence * 100)}%` }} />
                    </span>
                  </span>
                  <button
                    onClick={() => recompile(tiers)}
                    disabled={recompiling}
                    title="Recompile the truth + analysis from the agent outputs"
                    className="flex items-center gap-1 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${recompiling ? 'animate-spin' : ''}`} />
                  </button>
                  <OutputActions title={`${subject} — Compiled truth`} content={truth} onView={viewFull} />
                </div>
                <div className="fd-markdown text-sm text-zinc-200 leading-relaxed">
                  <Markdown remarkPlugins={[remarkGfm]}>{truth}</Markdown>
                </div>
              </div>
            )}

            {/* Recovery: agents produced outputs but the merge didn't finish */}
            {!truth && !executing && runs.length > 0 && (
              <div className="flex items-center gap-3 rounded-lg border border-yellow-400 bg-yellow-100 p-4 dark:border-amber-500/40 dark:bg-amber-400/10">
                <AlertTriangle className="h-4 w-4 shrink-0 text-orange-600 dark:text-orange-400" />
                <span className="text-xs font-medium text-orange-700 dark:text-orange-300">
                  No compiled truth — the merge may have stalled or failed. The {runs.length} agent output(s) are saved; recompile from them.
                </span>
                <button
                  onClick={() => recompile(tiers)}
                  disabled={recompiling}
                  className="ml-auto flex shrink-0 items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
                >
                  {recompiling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                  Compile truth
                </button>
              </div>
            )}

            {/* Generated files — sits between the compiled truth and the agents */}
            {attachments.some((a) => a.kind === 'generated') && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">Generated files</div>
                <div className="space-y-1">
                  {attachments.filter((a) => a.kind === 'generated').map((a) => (
                    <div key={a.name} className="flex items-center gap-2 text-xs">
                      <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                      <span className="truncate text-zinc-300">{a.name}</span>
                      {a.agent && (
                        <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400" title={`Generated by ${a.agent}`}>
                          {a.agent}
                        </span>
                      )}
                      <span className="shrink-0 text-zinc-600">{Math.max(1, Math.round(a.size / 1024))}kb</span>
                      <div className="ml-auto flex shrink-0 items-center gap-0.5">
                        {isViewable(a.name) && (
                          <button
                            onClick={() => viewFile(a.name)}
                            title="View"
                            className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                          >
                            <Eye className="h-3.5 w-3.5" />
                          </button>
                        )}
                        <button
                          onClick={() => downloadFile(a.name)}
                          title="Download"
                          className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                        >
                          <Download className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Agent runs */}
            {runs.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Agents ({runs.length})</span>
                {runs.map((run) => (
                  <AgentRow key={run.id} run={run} onFeedback={(s) => sendFeedback(run.id, s)} onView={viewFull} />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {modal && <FileModal title={modal.title} content={modal.content} mode={modal.mode} onClose={() => setModal(null)} />}
    </div>
  )
}
