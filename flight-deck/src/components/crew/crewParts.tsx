import { memo, useState, type ReactNode } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertTriangle, Check, CheckCircle2, CircleDashed, CirclePause, Clock, Copy, GitMerge, GitPullRequest, Loader2, MessageCircleQuestion, type LucideIcon,
} from 'lucide-react'
import { inputCls } from '../console/parts'
import { untrustedUrlTransform } from '../../utils/untrustedContent'
import type { CrewResults, CrewStatus, CrewTask } from '../../services/crewApi'
import type { HostedSite } from '../../services/hostingApi'

// ── What the Crew page's panes share ──
//
// The Admin console's look (console/parts.tsx): cards, soft tinted icons and the
// deck's violet, with what a person wants up front and git's words behind a click.

export const STATUS: Record<CrewStatus, { label: string; tone: string; dot: string; icon: LucideIcon }> = {
  queued: { label: 'Waiting for a slot', tone: 'bg-zinc-500/15 text-zinc-400', dot: 'bg-zinc-500', icon: Clock },
  running: { label: 'Working', tone: 'bg-sky-500/15 text-sky-600 dark:text-sky-300', dot: 'bg-sky-500', icon: Loader2 },
  waiting: { label: 'Needs you', tone: 'bg-amber-500/15 text-amber-700 dark:text-amber-300', dot: 'bg-amber-500', icon: MessageCircleQuestion },
  ready: { label: 'Ready for review', tone: 'bg-teal-500/15 text-teal-700 dark:text-teal-300', dot: 'bg-teal-500', icon: CheckCircle2 },
  failed: { label: 'Failed', tone: 'bg-red-500/15 text-red-700 dark:text-red-300', dot: 'bg-red-500', icon: AlertTriangle },
  stopped: { label: 'Stopped', tone: 'bg-zinc-500/15 text-zinc-400', dot: 'bg-zinc-500', icon: CirclePause },
  pr_open: { label: 'Pull request open', tone: 'bg-violet-500/15 text-violet-700 dark:text-violet-300', dot: 'bg-violet-500', icon: GitPullRequest },
  delivered: { label: 'Landed', tone: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300', dot: 'bg-emerald-500', icon: GitMerge },
  closed: { label: 'Closed', tone: 'bg-zinc-500/15 text-zinc-500', dot: 'bg-zinc-600', icon: CircleDashed },
}
export const LIVE: CrewStatus[] = ['queued', 'running']
export const DONE: CrewStatus[] = ['delivered', 'closed']

export const fieldClass = inputCls
export const primaryBtn = 'inline-flex items-center justify-center gap-1.5 rounded-xl bg-violet-600 px-3.5 py-2 text-sm font-medium text-white shadow-sm shadow-violet-900/20 transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50'
export const quietBtn = 'inline-flex items-center justify-center gap-1.5 rounded-xl border border-zinc-700 bg-zinc-900/60 px-3.5 py-2 text-sm font-medium text-zinc-200 transition-colors hover:border-zinc-600 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50'
export const dangerBtn = 'inline-flex items-center justify-center gap-1.5 rounded-xl border border-zinc-700 px-3.5 py-2 text-sm font-medium text-zinc-400 transition-colors hover:border-red-500/40 hover:bg-red-500/10 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-50 dark:hover:text-red-400'
export const sureDangerBtn = 'inline-flex items-center justify-center gap-1.5 rounded-xl border border-red-500/50 bg-red-500/10 px-3.5 py-2 text-sm font-medium text-red-700 transition-colors hover:bg-red-500/20 disabled:cursor-not-allowed disabled:opacity-50 dark:text-red-300'

export const errorOf = (e: unknown) => (e instanceof Error ? e.message : 'Something went wrong')

/** A deck path as an address to send someone. */
export const fullUrl = (url: string) => new URL(url, window.location.href).href

/** Has this task made a page worth putting online? */
export function hasPage(results?: CrewResults): boolean {
  if (!results) return false
  return results.whole ? !!results.publish?.subdir : !!results.files?.some((n) => n.endsWith('index.html'))
}

/** The task a site came from, if it came from one. */
export function taskOfSite(site: HostedSite, tasks: CrewTask[]): CrewTask | undefined {
  return tasks.find((t) => t.results?.link?.name === site.name)
    ?? (site.project === 'crew' ? tasks.find((t) => t.results?.publish?.subdir === site.subdir) : undefined)
}

/** Nothing of it is left to decide: it landed, it closed, or it has nothing to land. */
export function isFinished(t: CrewTask): boolean {
  if (DONE.includes(t.status)) return true
  if (t.status !== 'ready') return false
  const e = t.evidence || {}
  return t.kind === 'scout' || !(e.commits || e.dirty || e.stray)
}

/** It wants this person: a question, a failure, or gone quiet. */
export const needsYou = (t: CrewTask) => t.status === 'waiting' || t.status === 'failed' || t.stalled
export const isWorking = (t: CrewTask) => LIVE.includes(t.status) && !t.stalled

export function statusLabel(t: Pick<CrewTask, 'status' | 'stalled' | 'kind' | 'evidence'>): string {
  if (t.stalled) return 'Quiet for a while'
  if (t.status === 'ready' && t.kind === 'scout') return 'Report ready'
  if (t.status === 'ready' && !(t.evidence?.commits || t.evidence?.dirty)) return 'No changes'
  return (STATUS[t.status] ?? STATUS.stopped).label
}

const URGENCY: CrewStatus[] = ['waiting', 'failed', 'ready', 'pr_open', 'running', 'queued', 'stopped', 'delivered', 'closed']

/** What wants the person first, then the latest. */
export function byUrgency(a: CrewTask, b: CrewTask): number {
  return URGENCY.indexOf(a.status) - URGENCY.indexOf(b.status) || (b.updated_at ?? 0) - (a.updated_at ?? 0)
}

export function ago(epoch?: number): string {
  if (!epoch) return ''
  const s = Math.max(0, Date.now() / 1000 - epoch)
  if (s < 60) return 'now'
  if (s < 3600) return `${Math.floor(s / 60)}m`
  if (s < 86400) return `${Math.floor(s / 3600)}h`
  return `${Math.floor(s / 86400)}d`
}

export function statParts(stat?: string): { files: number; added: number; removed: number } {
  const n = (re: RegExp) => Number(re.exec(stat || '')?.[1] ?? 0)
  return { files: n(/(\d+) files? changed/), added: n(/(\d+) insertions?/), removed: n(/(\d+) deletions?/) }
}

export function StatusDot({ task }: { task: Pick<CrewTask, 'status' | 'stalled'> }) {
  const s = STATUS[task.status] ?? STATUS.stopped
  const running = task.status === 'running' && !task.stalled
  return (
    <span className="relative mt-[5px] flex h-2 w-2 shrink-0">
      {running && <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${s.dot}`} />}
      <span className={`relative inline-flex h-2 w-2 rounded-full ${task.stalled ? STATUS.waiting.dot : s.dot}`} />
    </span>
  )
}

export function StatusPill({ task }: { task: Pick<CrewTask, 'status' | 'stalled' | 'kind' | 'evidence'> }) {
  const s = STATUS[task.status] ?? STATUS.stopped
  return (
    <span className={`inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-0.5 text-[11px] font-medium ${task.stalled ? STATUS.waiting.tone : s.tone}`}>
      {task.status === 'running' && !task.stalled && <Loader2 className="h-3 w-3 animate-spin" />}
      {statusLabel(task)}
    </span>
  )
}

/** A task's state as a tinted square: what the rows on the overview lead with. */
export function StatusChip({ task, size = 'md' }: { task: Pick<CrewTask, 'status' | 'stalled'>; size?: 'md' | 'lg' }) {
  const s = task.stalled ? STATUS.waiting : (STATUS[task.status] ?? STATUS.stopped)
  const Icon = s.icon
  return (
    <span className={`flex shrink-0 items-center justify-center ${s.tone} ${size === 'lg' ? 'h-11 w-11 rounded-2xl' : 'h-9 w-9 rounded-xl'}`}>
      <Icon className={`${size === 'lg' ? 'h-5 w-5' : 'h-[18px] w-[18px]'} ${task.status === 'running' && !task.stalled ? 'animate-spin' : ''}`} />
    </span>
  )
}

/** One of a few choices, with a word on what it means. */
export function ChoiceTile({ icon: Icon, title, hint, on, disabled, onClick }: {
  icon?: LucideIcon; title: ReactNode; hint?: ReactNode; on: boolean; disabled?: boolean; onClick: () => void
}) {
  return (
    <button
      type="button" disabled={disabled} onClick={onClick} aria-pressed={on}
      className={`flex w-full min-w-0 items-start gap-3 rounded-2xl border p-3.5 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${on
        ? 'border-violet-500/60 bg-violet-500/[0.08] ring-2 ring-violet-500/15' : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-800/40'}`}
    >
      {Icon && (
        <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${on ? 'bg-violet-500/15 text-violet-600 dark:text-violet-300' : 'bg-zinc-800/80 text-zinc-400'}`}>
          <Icon className="h-[18px] w-[18px]" />
        </span>
      )}
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium text-zinc-100">{title}</span>
        {hint && <span className="mt-0.5 block text-xs leading-relaxed text-zinc-500">{hint}</span>}
      </span>
    </button>
  )
}

export function ErrorLine({ text }: { text: string }) {
  if (!text) return null
  return (
    <p className="flex items-start gap-1.5 rounded-xl bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> <span className="min-w-0 break-words">{text}</span>
    </p>
  )
}

/** A path or an address, ready to copy. */
export function CopyChip({ text, children }: { text: string; children?: ReactNode }) {
  const [done, setDone] = useState(false)
  const copy = () => {
    navigator.clipboard?.writeText(text).then(() => {
      setDone(true)
      setTimeout(() => setDone(false), 1500)
    }).catch(() => {})
  }
  return (
    <button
      type="button" onClick={copy} title="Copy"
      className="inline-flex min-w-0 max-w-full items-center gap-1.5 rounded-lg bg-zinc-800/70 px-2 py-1 text-left text-xs text-zinc-300 transition-colors hover:bg-zinc-800"
    >
      <span className="min-w-0 truncate">{children ?? text}</span>
      {done ? <Check className="h-3 w-3 shrink-0 text-emerald-600 dark:text-emerald-400" /> : <Copy className="h-3 w-3 shrink-0 text-zinc-500" />}
    </button>
  )
}

const MARKDOWN: Components = {
  a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
  table: ({ node: _node, ...props }) => <div className="crew-md-table"><table {...props} /></div>,
}

/** What a worker wrote. It reads files and the web, so what it writes is treated as
 *  someone else's: links work, and nothing makes this browser fetch anything. */
export const CrewMarkdown = memo(function CrewMarkdown({ text }: { text: string }) {
  return (
    <div className="fd-file-markdown crew-md">
      <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={MARKDOWN}>{text}</Markdown>
    </div>
  )
})
