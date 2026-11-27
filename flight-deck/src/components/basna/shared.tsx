import { useState } from 'react'
import { Maximize2, Minimize2, Download, X } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { BasnaAnalysis, ProgressEvent } from '../../stores/basnaStore'

// ── Shared helpers + tiny components for the Basna page family ──────────────
// Split out of BasnaPage so the sidebar / workspace / report components can
// use them without circular imports.

export const DIFFICULTY_COLOR: Record<string, string> = {
  trivial: 'text-emerald-700 dark:text-emerald-300',
  moderate: 'text-amber-700 dark:text-amber-300',
  hard: 'text-rose-700 dark:text-rose-300',
}

export function Badge({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={`inline-flex items-center rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] font-medium ${className}`}>
      {children}
    </span>
  )
}

export function WeightBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800/80">
      <div className="h-full rounded-full bg-sky-500" style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  )
}

export function timeAgo(iso?: string): string {
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

export const STATUS_DOT: Record<string, string> = {
  routing: 'bg-sky-400', routed: 'bg-zinc-400', running: 'bg-amber-400',
  done: 'bg-emerald-500', error: 'bg-rose-500',
}

// Confidence colour for the run list: green high, amber mid, rose low.
export function confColor(c: number): string {
  if (c >= 0.7) return 'text-emerald-600 dark:text-emerald-400'
  if (c >= 0.4) return 'text-amber-600 dark:text-amber-400'
  return 'text-rose-600 dark:text-rose-400'
}

// Agent-started runs carry {source:'agent', origin_platform} in their config.
export function agentOrigin(config?: string): string | null {
  if (!config) return null
  try {
    const c = JSON.parse(config)
    if (c && c.source === 'agent') return c.origin_platform || 'agent'
  } catch { /* ignore */ }
  return null
}

export function isVatra(config?: string): boolean {
  if (!config) return false
  try { return JSON.parse(config)?.mode === 'vatra' } catch { return false }
}

// Deepen runs carry {kind:'deepen', parent_session_id} in their config.
export function parentIdOf(config?: string): string | null {
  if (!config) return null
  try {
    const c = JSON.parse(config)
    if (c && c.kind === 'deepen' && c.parent_session_id) return String(c.parent_session_id)
  } catch { /* ignore */ }
  return null
}

// The VFS folder this run OWNS — never a reference/prior folder it only read.
// Resolution order:
//  1. A pinned folder recorded in config (continuation rounds, future runs).
//  2. The folder the team WROTE to. Reference folders (prior-run knowledge) and
//     read-only links are never written, so any write/datastore target is this
//     run's own folder — unlike reads, which routinely hit reference folders.
//  3. The backend's deterministic auto-name `<mode>-<sid8>`.
export function runVfsProject(
  session: { id: string; config?: string },
  progress?: ProgressEvent[],
): string {
  try {
    const cfg = JSON.parse(session.config || '{}')
    if (cfg && cfg.vfs_project) return String(cfg.vfs_project)
  } catch { /* ignore */ }
  if (progress) {
    for (const ev of progress) {
      // Only trust WRITE-side events — the write tool and datastore ops target
      // this run's own folder; reads (web/glob/grep) routinely hit reference
      // folders, so parsing those would grab the wrong folder.
      if (ev.tool !== 'write' && ev.tool !== 'datastore') continue
      const m = `${ev.detail || ''} ${ev.message || ''}`.match(/vfs:([A-Za-z0-9_.-]+)\//)
      if (m) return m[1]
    }
  }
  return (isVatra(session.config) ? 'vatra-' : 'basna-') + session.id.slice(0, 8)
}

export function analysisToMarkdown(a: BasnaAnalysis): string {
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

export function formatProgress(events: ProgressEvent[]): string {
  return events.map((e) => {
    const t = e.ts ? new Date(e.ts * 1000).toLocaleTimeString([], { hour12: false }) : ''
    return `${t}  ${e.stage.toUpperCase().padEnd(10)} ${e.message}`
  }).join('\n')
}

export function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'basna'
}

export function downloadMarkdown(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export type ViewMode = 'markdown' | 'html' | 'text'

export function fileExt(name: string): string {
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
export function isViewable(name: string): boolean { return VIEWABLE_EXTS.has(fileExt(name)) }
export function viewModeForFile(name: string): ViewMode {
  const e = fileExt(name)
  if (e === 'md' || e === 'markdown') return 'markdown'
  if (e === 'html' || e === 'htm') return 'html'
  return 'text'
}

// Fullscreen-capable preview modal. Renders by mode: markdown (with GFM tables),
// raw HTML (sandboxed iframe), or plain text / source. The maximise button grows
// it to near-fullscreen; click the backdrop or ✕ to close.
export function FileModal({ title, content, mode, onClose }: {
  title: string; content: string; mode: ViewMode; onClose: () => void
}) {
  const [maximized, setMaximized] = useState(false)
  const ext = fileExt(title)
  const exportName = ext ? title : `${slugify(title)}.md`
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={`flex w-full flex-col border border-zinc-800 bg-zinc-950 shadow-2xl ${
          maximized ? 'h-[100dvh] wide:h-[96vh] wide:w-[97vw] wide:max-w-none wide:rounded-2xl' : 'max-h-[92dvh] rounded-t-2xl wide:max-h-[90vh] wide:max-w-4xl wide:rounded-2xl'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-5 py-3.5">
          <span className="min-w-0 flex-1 truncate text-base font-semibold text-zinc-100">{title}</span>
          <button
            onClick={() => setMaximized((m) => !m)}
            title={maximized ? 'Restore' : 'Maximise'}
            aria-label={maximized ? 'Restore' : 'Maximise'}
            className="hidden rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 wide:block"
          >
            {maximized ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
          <button
            onClick={() => downloadMarkdown(exportName, content)}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-xl border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:border-zinc-600 hover:bg-zinc-800"
          >
            <Download className="h-3.5 w-3.5" /> Export {ext ? `.${ext}` : '.md'}
          </button>
          <button onClick={onClose} aria-label="Close" title="Close" className="rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>
        {mode === 'html' ? (
          <iframe
            title={title}
            sandbox=""
            srcDoc={content}
            className="min-h-[60vh] w-full flex-1 rounded-b-2xl bg-white"
          />
        ) : mode === 'markdown' ? (
          <div className="fd-markdown flex-1 overflow-auto px-6 py-5 text-sm leading-relaxed text-zinc-200">
            <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
          </div>
        ) : (
          <pre className="flex-1 overflow-auto whitespace-pre-wrap break-words px-6 py-5 font-mono text-xs leading-relaxed text-zinc-300">
            {content}
          </pre>
        )}
      </div>
    </div>
  )
}

// Compact fullscreen + export buttons reused by the report and agent cards.
export function OutputActions({ title, content, onView }: { title: string; content: string; onView: (t: string, c: string) => void }) {
  return (
    <div className="flex items-center gap-1">
      <button onClick={() => onView(title, content)} title="Fullscreen" aria-label="Fullscreen" className="rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
        <Maximize2 className="h-3.5 w-3.5" />
      </button>
      <button onClick={() => downloadMarkdown(`${slugify(title)}.md`, content)} title="Export .md" aria-label="Export .md" className="rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
        <Download className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

// Compact token count: 1_234 → "1.2k", 244_461 → "244k", 1_619_000 → "1.62M".
export function fmtTok(n?: number): string {
  if (!n || n <= 0) return '0'
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M'
  if (n >= 10_000) return Math.round(n / 1000) + 'k'
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k'
  return String(n)
}

export interface LiveAgent {
  role: string; actions: ProgressEvent[]; usage?: ProgressEvent; done?: boolean; ok?: boolean; group?: string
  // Keep on course: the latest Jev course check, and whether the agent is off
  // track (two checks in a row) until a later check says otherwise.
  course?: ProgressEvent; offTrack?: boolean
}

// Group the streaming progress into live per-agent panels (preserving the
// order each agent first appears). `usage` events update the running token
// counter; everything else is a tool call / narration in that agent's feed.
export function buildLiveAgents(progress: ProgressEvent[]): LiveAgent[] {
  const byRole = new Map<string, LiveAgent>()
  const order: string[] = []
  for (const ev of progress) {
    if (!ev.agent) continue
    let a = byRole.get(ev.agent)
    if (!a) { a = { role: ev.agent, actions: [] }; byRole.set(ev.agent, a); order.push(ev.agent) }
    if (ev.group) a.group = ev.group  // Vatra grouped mode: the owner's phase letter
    // Any fresh activity after a dispatch (e.g. the Vatra review round) flips the
    // card back to "working" so the spinner returns; the next dispatch marks done.
    if (ev.stage === 'usage') { a.usage = ev; a.done = false }
    else if (ev.stage === 'dispatch') { a.done = true; a.ok = ev.ok !== false }
    else if (ev.stage === 'course') {
      a.course = ev
      if (typeof ev.p_on_track === 'number' && ev.p_on_track > 0.25) a.offTrack = false
    }
    else if (ev.stage === 'course_off_track') { a.offTrack = true }
    else { a.actions.push(ev); a.done = false }
  }
  return order.map((r) => byRole.get(r) as LiveAgent)
}
