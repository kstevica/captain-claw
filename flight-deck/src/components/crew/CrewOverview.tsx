import { useMemo, type ReactNode } from 'react'
import {
  Bot, CheckCircle2, ChevronRight, ExternalLink, Globe, Hourglass, Loader2, MessageCircleQuestion, Plus, type LucideIcon,
} from 'lucide-react'
import { Card, StatTile } from '../console/parts'
import type { CrewLimit, CrewTask, CrewUsage } from '../../services/crewApi'
import type { HostedSite } from '../../services/hostingApi'
import { isLive, SiteIcon } from './CrewSites'
import { StatusChip, ago, byUrgency, isFinished, isWorking, needsYou, primaryBtn, quietBtn, statusLabel } from './crewParts'

// ── Where Crew opens ──
//
// What's going on, in words a person would use: what needs them first, what's ready
// to look at, who's working, what's online, and how much of each agent's limit is left.

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`

function Hero({ needs, review, working, none, onOpen, onNew }: {
  needs: CrewTask[]; review: CrewTask[]; working: CrewTask[]; none: boolean; onOpen: (id: string) => void; onNew: () => void
}) {
  const first = needs[0] ?? review[0]
  const headline = needs.length ? `${plural(needs.length, 'task needs', 'tasks need')} you`
    : review.length ? `${plural(review.length, 'task')} ready for you to look at`
      : working.length ? `${plural(working.length, 'agent')} at work`
        : none ? 'Put a coding agent to work' : 'All caught up'
  const sub = needs.length ? (needs[0].note || needs[0].title)
    : review.length ? 'See what it did, then merge it or send it back.'
      : working.length ? working[0].title
        : none ? 'Describe a task and an agent does it in a copy of the project. Nothing lands until you\'ve looked at it. Or ask your agent in chat: "have Claude Code fix the login test".'
          : 'Nothing needs you right now.'
  return (
    <section className="fd-rise relative overflow-hidden rounded-2xl border border-violet-500/20 bg-gradient-to-br from-violet-500/[0.12] via-transparent to-sky-500/[0.06] p-5 wide:p-6">
      <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-violet-700 dark:text-violet-300">
        <span className="relative flex h-2 w-2">
          {working.length > 0 && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-400 opacity-60" />}
          <span className="relative inline-flex h-2 w-2 rounded-full bg-violet-500" />
        </span>
        {working.length ? `${working.length} working now` : 'Your crew'}
      </p>
      <h3 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">{headline}</h3>
      <p className="mt-1 line-clamp-2 max-w-2xl text-sm leading-relaxed text-zinc-400">{sub}</p>
      <div className="mt-4 flex flex-wrap gap-2">
        {first && <button onClick={() => onOpen(first.id)} className={primaryBtn}>{first.status === 'waiting' ? 'Answer it' : 'Look at it'} <ChevronRight className="h-4 w-4" /></button>}
        <button onClick={onNew} className={first ? quietBtn : primaryBtn}><Plus className="h-4 w-4" /> New task</button>
      </div>
    </section>
  )
}

export function TaskLine({ task, onOpen }: { task: CrewTask; onOpen: (id: string) => void }) {
  const about = [task.project_name || task.repo_label, task.harness_label].filter(Boolean).join(' · ')
  return (
    <button onClick={() => onOpen(task.id)} className="group flex w-full items-center gap-3 px-5 py-3 text-left transition-colors hover:bg-zinc-800/30">
      <StatusChip task={task} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium text-zinc-100">{task.title}</span>
        <span className="block truncate text-xs text-zinc-500">{task.note || statusLabel(task)}{about ? ` · ${about}` : ''}</span>
      </span>
      <span className="shrink-0 text-xs tabular-nums text-zinc-600">{ago(task.updated_at)}</span>
      <ChevronRight className="h-4 w-4 shrink-0 text-zinc-600 transition-transform group-hover:translate-x-0.5 group-hover:text-zinc-300" />
    </button>
  )
}

function TaskCard({ title, icon, tasks, onOpen }: { title: string; icon: LucideIcon; tasks: CrewTask[]; onOpen: (id: string) => void }) {
  if (!tasks.length) return null
  return (
    <Card title={title} icon={icon} bodyClassName="divide-y divide-zinc-800/70">
      {tasks.map((t) => <TaskLine key={t.id} task={t} onOpen={onOpen} />)}
    </Card>
  )
}

export function SitesCard({ sites, onSites, onPublish }: { sites: HostedSite[]; onSites: () => void; onPublish: () => void }) {
  const live = sites.filter(isLive)
  return (
    <Card
      title="Live sites" icon={Globe} hint={sites.length ? `${live.length} of ${sites.length} online` : undefined} bodyClassName="p-2"
      action={<button onClick={onPublish} className="rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200" title="Publish a site" aria-label="Publish a site"><Plus className="h-4 w-4" /></button>}
    >
      {live.slice(0, 4).map((s) => (
        <div key={s.name} className="group flex items-center gap-2.5 rounded-xl px-3 py-2 transition-colors hover:bg-zinc-800/40">
          <SiteIcon site={s} />
          <button onClick={onSites} className="min-w-0 flex-1 text-left">
            <span className="block truncate text-sm font-medium text-zinc-100">{s.name}</span>
            <span className="block truncate text-xs text-zinc-500">{s.kind === 'app' ? 'App' : 'Page'}{s.visit_count ? ` · ${plural(s.visit_count, 'visit')}` : ''}</span>
          </button>
          <a href={s.url} target="_blank" rel="noreferrer" className="rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200" title={`Open ${s.name}`} aria-label={`Open ${s.name}`}><ExternalLink className="h-4 w-4" /></a>
        </div>
      ))}
      {!live.length && <p className="px-3 py-2 text-sm text-zinc-500">{sites.length ? 'All of them are stopped.' : 'Nothing online. When a task builds a page, publish it here.'}</p>}
      <button onClick={onSites} className="flex w-full items-center gap-1 rounded-xl px-3 py-2 text-left text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800/40 hover:text-zinc-200">
        {sites.length ? 'Start, stop or remove them' : 'Open Live sites'} <ChevronRight className="h-3.5 w-3.5" />
      </button>
    </Card>
  )
}

const WINDOWS: Record<string, string> = {
  five_hour: '5 hours', seven_day: 'This week', seven_day_opus: 'This week, Opus', seven_day_sonnet: 'This week, Sonnet',
}

function LimitBar({ name, limit }: { name: string; limit: CrewLimit }) {
  const pct = limit.used === null ? null : Math.round(limit.used * 100)
  const stopped = !!limit.status && !limit.status.startsWith('allowed')
  const tone = stopped || (pct ?? 0) >= 90 ? 'bg-red-500' : limit.status === 'allowed_warning' || (pct ?? 0) >= 75 ? 'bg-amber-500' : 'bg-gradient-to-r from-violet-500 to-fuchsia-500'
  const resets = limit.resets_at ? new Date(limit.resets_at * 1000).toLocaleString(undefined, { weekday: 'short', hour: 'numeric', minute: '2-digit' }) : ''
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="text-zinc-400">{WINDOWS[name] ?? name.replace(/_/g, ' ')}</span>
        <span className="tabular-nums text-zinc-200">{pct === null ? limit.status : `${pct}% used`}</span>
      </div>
      {pct !== null && (
        <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-zinc-800">
          <div className={`h-full rounded-full ${tone}`} style={{ width: `${Math.min(100, Math.max(2, pct))}%` }} />
        </div>
      )}
      {resets && <p className="mt-1 text-[11px] text-zinc-500">Resets {resets}</p>}
    </div>
  )
}

function compactTokens(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1000 ? `${Math.round(n / 1000)}k` : String(n)
}

/** Each CLI's limits, as it last reported them while working: there's no asking
 *  between runs, so the card says how old they are. */
function AgentsCard({ rows }: { rows: CrewUsage[] }) {
  if (!rows.length) return null
  return (
    <Card title="Your coding agents" icon={Bot} hint="This last week" bodyClassName="divide-y divide-zinc-800/70">
      {rows.map((u) => (
        <div key={u.id} className="space-y-3 p-5" title={u.model || undefined}>
          <div>
            <p className="text-sm font-medium text-zinc-100">{u.label}</p>
            <p className="text-xs text-zinc-500" title={u.cost_usd ? `About $${u.cost_usd.toFixed(2)} at API prices` : undefined}>
              {plural(u.tasks, 'task')} · {compactTokens(u.tokens)} tokens
            </p>
          </div>
          {Object.entries(u.limits).map(([name, limit]) => <LimitBar key={name} name={name} limit={limit} />)}
          {u.seen_at > 0 && Object.keys(u.limits).length > 0 && <p className="text-[11px] text-zinc-600">As of its last run, {ago(u.seen_at) === 'now' ? 'just now' : `${ago(u.seen_at)} ago`}</p>}
        </div>
      ))}
    </Card>
  )
}

export function CrewOverview({ tasks, sites, usage, onOpen, onNew, onSites, onPublish, children }: {
  tasks: CrewTask[]; sites: HostedSite[]; usage: CrewUsage[]; onOpen: (id: string) => void; onNew: () => void
  onSites: () => void; onPublish: () => void
  /** On a phone, where there's no side rail: the projects. */
  children?: ReactNode
}) {
  const sorted = useMemo(() => [...tasks].sort(byUrgency), [tasks])
  const needs = sorted.filter(needsYou)
  const review = sorted.filter((t) => (t.status === 'ready' || t.status === 'pr_open') && !isFinished(t) && !t.stalled)
  const working = sorted.filter(isWorking)
  const live = sites.filter(isLive).length
  return (
    <div className="@container min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-5xl space-y-5 p-4 wide:p-6">
        <Hero needs={needs} review={review} working={working} none={!tasks.length} onOpen={onOpen} onNew={onNew} />
        <div className="grid grid-cols-2 gap-3 @xl:grid-cols-4">
          <StatTile icon={Loader2} tone="sky" label="Working" value={working.length} />
          <StatTile icon={MessageCircleQuestion} tone="amber" label="Needs you" value={needs.length} />
          <StatTile icon={CheckCircle2} tone="emerald" label="To look at" value={review.length} />
          <StatTile icon={Globe} tone="violet" label="Live sites" value={live} onClick={onSites} />
        </div>
        <div className="grid grid-cols-1 gap-5 @3xl:grid-cols-5">
          <div className="min-w-0 space-y-5 @3xl:col-span-3">
            <TaskCard title="Needs you" icon={MessageCircleQuestion} tasks={needs} onOpen={onOpen} />
            <TaskCard title="Ready to look at" icon={CheckCircle2} tasks={review} onOpen={onOpen} />
            <TaskCard title="Working now" icon={Hourglass} tasks={working} onOpen={onOpen} />
            {!needs.length && !review.length && !working.length && tasks.length > 0 && (
              <p className="rounded-2xl border border-dashed border-zinc-800 p-6 text-center text-sm text-zinc-500">Nothing open. Finished tasks stay under their projects.</p>
            )}
            {children}
          </div>
          <div className="min-w-0 space-y-5 @3xl:col-span-2">
            <SitesCard sites={sites} onSites={onSites} onPublish={onPublish} />
            <AgentsCard rows={usage} />
          </div>
        </div>
      </div>
    </div>
  )
}
