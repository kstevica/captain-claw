import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react'
import {
  AlertTriangle, ArrowLeft, Bot, Check, CheckCircle2, ChevronRight, CircleDot, FileDiff, FolderOpen, FolderPlus, GitMerge, GitPullRequest, Globe, KeyRound,
  LayoutGrid, Loader2, MessageCircleQuestion, MessageSquare, Plus, RefreshCw, ScrollText, Search, Send, Ship, Sparkles, Square, TerminalSquare, Trash2, Wrench, X, type LucideIcon,
} from 'lucide-react'
import { Card, EmptyState, Pill, Segmented, StatTile, labelCls } from '../console/parts'
import { FullscreenButton, libraryIconBtn } from '../library/libraryParts'
import {
  crewApi, type CrewDispatch, type CrewEnv, type CrewEvent, type CrewHarness, type CrewMe, type CrewRepos, type CrewTask, type CrewTerminalTarget, type CrewUsage,
} from '../../services/crewApi'
import { hostingApi, type HostedSite } from '../../services/hostingApi'
import { CrewActivity } from './CrewActivity'
import { CrewChanges } from './CrewChanges'
import { EnvironmentCard } from './CrewEnvironment'
import { CrewFiles } from './CrewFiles'
import { CrewOverview, SitesCard, TaskLine } from './CrewOverview'
import { CrewSites, PublishSheet, isLive } from './CrewSites'
import {
  ChoiceTile, CopyChip, CrewMarkdown, DONE, ErrorLine, LIVE, StatusChip, StatusDot, StatusPill, ago, byUrgency, dangerBtn, errorOf, fieldClass, isFinished, isWorking, needsYou,
  primaryBtn, quietBtn, statParts, statusLabel, sureDangerBtn, taskOfSite,
} from './crewParts'

// ── Crew ──
//
// Coding agents working for this user: what needs them, what's ready to look at,
// who's working and what's online. An overview first, the projects and their tasks on
// the side; on a phone the overview holds the projects too. An agent's own chat can
// start and steer the same tasks, so this page polls rather than assuming it made them.

const CrewTerminal = lazy(() => import('./CrewTerminal'))

const POLL_LIST_MS = 4000
const POLL_ENVS_MS = 20000
const POLL_EVENTS_MS = 2000
const POLL_USAGE_MS = 30000
const COLLAPSED_KEY = 'fd:crew:collapsed'

function readCollapsed(): string[] {
  try {
    const got = JSON.parse(localStorage.getItem(COLLAPSED_KEY) || '[]')
    return Array.isArray(got) ? got : []
  } catch {
    return []
  }
}

function saveCollapsed(keys: string[]) {
  try { localStorage.setItem(COLLAPSED_KEY, JSON.stringify(keys)) } catch { /* only a convenience */ }
}

const plural = (n: number, one: string) => `${n} ${one}${n === 1 ? '' : 's'}`

// ── One task ─────────────────────────────────────────────────────────────

/** A step to be sure: the button asks once more before it does. */
function SureButton({ icon, label, sure, busy, danger, primary, onClick }: {
  icon: ReactNode; label: string; sure: string; busy: boolean; danger?: boolean; primary?: boolean; onClick: () => void
}) {
  const [asking, setAsking] = useState(false)
  useEffect(() => {
    if (!asking) return
    const t = setTimeout(() => setAsking(false), 6000)
    return () => clearTimeout(t)
  }, [asking])
  if (asking) {
    return (
      <button
        disabled={busy}
        className={danger ? sureDangerBtn : primaryBtn}
        onClick={() => { setAsking(false); onClick() }}
      >
        <Check className="h-4 w-4" /> {sure}
      </button>
    )
  }
  return <button disabled={busy} className={primary ? primaryBtn : danger ? dangerBtn : quietBtn} onClick={() => setAsking(true)}>{icon} {label}</button>
}

function Outcome({ task }: { task: CrewTask }) {
  if (!task.note) return null
  const good = task.status === 'ready' || task.status === 'delivered' || task.status === 'pr_open'
  const look = task.status === 'waiting' ? 'border-amber-500/30 bg-amber-500/[0.06] text-amber-800 dark:text-amber-200'
    : task.status === 'failed' ? 'border-red-500/30 bg-red-500/[0.06] text-red-800 dark:text-red-200'
      : good ? 'border-teal-500/25 bg-teal-500/[0.06] text-teal-900 dark:text-teal-100' : 'border-zinc-800 bg-zinc-900/40 text-zinc-300'
  const Icon = task.status === 'waiting' ? MessageCircleQuestion : task.status === 'failed' ? AlertTriangle : good ? CheckCircle2 : CircleDot
  return (
    <p className={`flex items-start gap-2.5 rounded-2xl border px-4 py-3 text-sm leading-relaxed ${look}`}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0 opacity-80" />
      <span className="min-w-0 break-words">{task.note}</span>
    </p>
  )
}

/** What git shows, not what the agent says: in words, and only what matters. */
function WhatChanged({ task }: { task: CrewTask }) {
  const e = task.evidence || {}
  const { files } = statParts(e.stat)
  return (
    <div className="flex flex-wrap items-center gap-1.5" title="What git shows, not what the agent says">
      {files > 0 && <Pill>{plural(files, 'file')} changed</Pill>}
      {files === 0 && !e.dirty && <Pill>No changes yet</Pill>}
      {!!e.dirty && <Pill tone="amber">{e.dirty === 1 ? '1 change' : `${e.dirty} changes`} not saved in git</Pill>}
      {e.off_branch && <Pill tone="red">It left its branch, for {e.off_branch}</Pill>}
      {!!e.stray && <Pill tone="red">{plural(e.stray, 'commit')} off its branch</Pill>}
      {task.delivery?.url && <a href={task.delivery.url} target="_blank" rel="noreferrer" className="contents"><Pill tone="violet" icon={GitPullRequest}>Pull request #{task.delivery.pr}</Pill></a>}
    </div>
  )
}

/** Which model, which branch, how many tokens: there when someone wants them. */
function Details({ task }: { task: CrewTask }) {
  const tokens = task.tokens_in + task.tokens_out
  const commits = task.evidence?.commits
  const { added, removed } = statParts(task.evidence?.stat)
  const rows: [string, string][] = [
    ['Agent', task.harness_label],
    ...(task.kind === 'clone' ? [] : [['Model', task.model_used || task.model || 'Its default'] as [string, string]]),
    ...(task.env?.port ? [['Its port', String(task.env.port)] as [string, string]] : []),
    ...(task.env?.copied?.length ? [['Its own copies', task.env.copied.join(', ')] as [string, string]] : []),
    ...(task.effort ? [['Effort', task.effort] as [string, string]] : []),
    ['Working in', task.repo_label],
    ...(task.branch ? [['Branch', task.branch] as [string, string]] : []),
    ...(commits ? [['Commits', `${commits}${added || removed ? `, +${added} −${removed}` : ''}`] as [string, string]] : []),
    ...(tokens ? [['Tokens', `${Math.round(tokens / 1000)}k`] as [string, string]] : []),
    ...(task.cost_usd ? [['At API prices', `about $${task.cost_usd.toFixed(2)}`] as [string, string]] : []),
    ...(task.turn > 1 ? [['Turns', String(task.turn)] as [string, string]] : []),
  ]
  return (
    <details className="group/details text-xs">
      <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-zinc-500 transition-colors hover:text-zinc-300 [&::-webkit-details-marker]:hidden">
        <ChevronRight className="h-3.5 w-3.5 transition-transform group-open/details:rotate-90" /> Details
      </summary>
      <dl className="mt-2 grid gap-x-6 gap-y-1.5 rounded-xl border border-zinc-800 bg-zinc-900/40 p-3 sm:grid-cols-2">
        {rows.map(([k, v]) => (
          <div key={k} className="flex min-w-0 items-baseline gap-2">
            <dt className="shrink-0 text-zinc-500">{k}</dt>
            <dd className="min-w-0 truncate text-zinc-300" title={v}>{v}</dd>
          </div>
        ))}
      </dl>
    </details>
  )
}

type TaskTab = 'activity' | 'changes' | 'files' | 'report'

function TaskView({ taskId, canTerminal, onBack, onChanged, onTerminal }: {
  taskId: string; canTerminal: boolean; onBack?: () => void; onChanged: () => void; onTerminal: (task: CrewTask) => void
}) {
  const [task, setTask] = useState<CrewTask | null>(null)
  const [events, setEvents] = useState<CrewEvent[]>([])
  const [tab, setTab] = useState<TaskTab>('activity')
  const [diff, setDiff] = useState<{ committed: string; uncommitted: string; note?: string } | null>(null)
  const [steer, setSteer] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const after = useRef(0)

  const load = useCallback(async () => {
    const [fresh, more] = await Promise.all([crewApi.task(taskId), crewApi.events(taskId, after.current)])
    setTask(fresh)
    if (more.events.length) {
      after.current = more.events[more.events.length - 1].n
      setEvents((old) => [...old, ...more.events])
    }
  }, [taskId])

  useEffect(() => {
    after.current = 0
    setEvents([]); setTask(null); setTab('activity'); setDiff(null); setError(''); setSteer('')
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    const tick = async () => {
      try { await load() } catch (e) { if (!stopped) setError(errorOf(e)) }
      if (!stopped) timer = setTimeout(tick, POLL_EVENTS_MS)
    }
    tick()
    return () => { stopped = true; clearTimeout(timer) }
  }, [load])

  useEffect(() => {
    if (tab !== 'changes') return
    crewApi.diff(taskId).then(setDiff).catch((e) => setError(errorOf(e)))
  }, [tab, taskId, task?.evidence?.commits, task?.evidence?.dirty])

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError('')
    try {
      await fn()
      await load()
      onChanged()
    } catch (e) {
      setError(errorOf(e))
    }
    setBusy(false)
  }

  if (!task) {
    return <div className="flex flex-1 items-center justify-center text-zinc-500">{error ? <ErrorLine text={error} /> : <Loader2 className="h-5 w-5 animate-spin" />}</div>
  }

  const live = LIVE.includes(task.status)
  const finished = DONE.includes(task.status)
  const e = task.evidence || {}
  const ship = task.kind === 'ship'
  const cloneOnly = task.kind === 'clone'
  const landable = ship && !live && !finished && !!e.commits && !e.dirty && !e.stray
  const unlanded = !!e.commits || !!e.dirty || !!e.stray
  const hasPr = !!task.delivery?.pr

  const send = (ev: FormEvent) => {
    ev.preventDefault()
    const text = steer.trim()
    if (!text) return
    setSteer('')
    act(() => crewApi.steer(task.id, text))
  }

  const tabs: [TaskTab, string, LucideIcon][] = [
    ['activity', 'Activity', MessageSquare],
    ...(ship ? [['changes', 'Changes', FileDiff] as [TaskTab, string, LucideIcon]] : []),
    ...(cloneOnly || task.kind === 'setup' ? [] : [['files', 'Files', FolderOpen] as [TaskTab, string, LucideIcon]]),
    ...(task.report ? [['report', 'Report', ScrollText] as [TaskTab, string, LucideIcon]] : []),
  ]
  const about = [task.harness_label, task.project_name || task.repo_label, task.from_agent ? 'from your agent' : '', ago(task.updated_at) === 'now' ? 'just now' : `${ago(task.updated_at)} ago`].filter(Boolean)

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="max-h-[45dvh] shrink-0 space-y-3 overflow-y-auto border-b border-zinc-800 p-4 wide:max-h-none wide:p-5">
        <div className="flex items-start gap-3">
          {onBack && <button onClick={onBack} className={libraryIconBtn} aria-label="Back"><ArrowLeft className="h-4 w-4" /></button>}
          <StatusChip task={task} size="lg" />
          <div className="min-w-0 flex-1">
            <h3 className="break-words text-lg font-semibold leading-snug tracking-tight text-zinc-100">{task.title}</h3>
            <p className="mt-0.5 text-xs text-zinc-500">{about.join(' · ')}</p>
          </div>
          <span className="hidden wide:inline-flex"><StatusPill task={task} /></span>
        </div>
        <div className="wide:hidden"><StatusPill task={task} /></div>

        <Outcome task={task} />
        {ship && (task.turn > 0 || finished) && <WhatChanged task={task} />}
        <Details task={task} />

        {(live || landable || (!live && !finished)) && (
          <div className="flex flex-wrap gap-2 pt-0.5">
            {canTerminal && task.has_worktree && !cloneOnly && (
              <button className={quietBtn} onClick={() => onTerminal(task)}><TerminalSquare className="h-4 w-4" /> Terminal</button>
            )}
            {live && <button disabled={busy} className={quietBtn} onClick={() => act(() => crewApi.stop(task.id))}><Square className="h-4 w-4" /> Stop</button>}
            {landable && (
              <SureButton
                primary icon={<GitMerge className="h-4 w-4" />} busy={busy}
                label={hasPr ? 'Merge the pull request' : `Merge into ${task.base_branch || 'its branch'}`}
                sure="Yes, merge it" onClick={() => act(() => crewApi.merge(task.id))}
              />
            )}
            {landable && (
              <button disabled={busy} className={quietBtn} onClick={() => act(() => crewApi.pr(task.id))}><GitPullRequest className="h-4 w-4" /> {hasPr ? 'Update the pull request' : 'Open a pull request'}</button>
            )}
            {!live && !finished && (
              unlanded ? (
                <SureButton danger icon={<Trash2 className="h-4 w-4" />} busy={busy} label="Discard" sure="Yes, throw this work away" onClick={() => act(() => crewApi.close(task.id, true))} />
              ) : (
                <button disabled={busy} className={quietBtn} onClick={() => act(() => crewApi.close(task.id))}><X className="h-4 w-4" /> Close</button>
              )
            )}
          </div>
        )}
        <ErrorLine text={error} />
      </div>

      <div className="shrink-0 px-4 pt-3 wide:px-5">
        <Segmented
          label="What to show" value={tab} onChange={setTab}
          options={tabs.map(([id, label, Icon]) => ({ value: id, label: <span className="inline-flex items-center gap-1.5"><Icon className="h-3.5 w-3.5" />{label}</span> }))}
        />
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {tab === 'activity' && <CrewActivity events={events} live={live} />}
        {tab === 'changes' && <CrewChanges diff={diff} evidence={e} />}
        {tab === 'files' && <CrewFiles task={task} />}
        {tab === 'report' && <div className="min-h-0 flex-1 overflow-y-auto p-5"><CrewMarkdown text={task.report ?? ''} /></div>}
      </div>

      {!finished && !cloneOnly && (
        <form onSubmit={send} className="shrink-0 border-t border-zinc-800 p-3">
          <div className="flex items-end gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/60 p-1.5 transition-colors focus-within:border-violet-500/60 focus-within:ring-2 focus-within:ring-violet-500/15">
            <textarea
              value={steer}
              onChange={(ev) => setSteer(ev.target.value)}
              onKeyDown={(ev) => { if (ev.key === 'Enter' && !ev.shiftKey) send(ev) }}
              rows={Math.min(5, steer.split('\n').length)}
              placeholder={task.status === 'waiting' ? 'Your answer…' : task.status === 'preparing' ? 'Tell it something, for when it starts…' : live ? 'Tell it something, for when this turn ends…' : 'Tell it to go on…'}
              aria-label="Tell the agent"
              className="min-w-0 flex-1 resize-none bg-transparent px-2.5 py-1.5 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none"
            />
            <button type="submit" disabled={busy || !steer.trim()} className={`${primaryBtn} h-9 w-9 shrink-0 !p-0`} aria-label="Send"><Send className="h-4 w-4" /></button>
          </div>
        </form>
      )}
    </div>
  )
}

// ── One project ──────────────────────────────────────────────────────────

interface Project {
  key: string
  name: string
  tasks: CrewTask[]
  paths: string[]
}

const MARKS = [
  'bg-violet-500/15 text-violet-700 dark:text-violet-300',
  'bg-sky-500/15 text-sky-700 dark:text-sky-300',
  'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  'bg-fuchsia-500/15 text-fuchsia-700 dark:text-fuchsia-300',
  'bg-cyan-500/15 text-cyan-700 dark:text-cyan-300',
]

/** A project's first letter on its own colour, the same every time. */
function ProjectMark({ name, big }: { name: string; big?: boolean }) {
  const hue = [...name].reduce((n, c) => n + c.charCodeAt(0), 0) % MARKS.length
  return (
    <span className={`flex shrink-0 items-center justify-center font-semibold uppercase ${MARKS[hue]} ${big ? 'h-11 w-11 rounded-2xl text-base' : 'h-6 w-6 rounded-lg text-[11px]'}`}>
      {name.trim().charAt(0) || '?'}
    </span>
  )
}

function ProjectView({ project, sites, envs, me, onOpen, onNew, onChanged, onEnvChanged, onTerminal, onBack, onSites, onPublish }: {
  project: Project; sites: HostedSite[]; envs: CrewEnv[]; me: CrewMe | null; onOpen: (id: string) => void; onNew: () => void; onChanged: () => void
  onEnvChanged: () => void; onTerminal: (env: CrewEnv) => void; onBack?: () => void; onSites: () => void; onPublish: () => void
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [done, setDone] = useState('')
  useEffect(() => { setDone(''); setError('') }, [project.key])
  const tasks = useMemo(() => [...project.tasks].sort(byUrgency), [project.tasks])
  const finished = tasks.filter(isFinished).length
  const count = (...statuses: CrewTask['status'][]) => tasks.filter((t) => statuses.includes(t.status) && !isFinished(t) && !t.stalled).length
  const ownSites = useMemo(() => sites.filter((s) => s.project === project.name || !!taskOfSite(s, project.tasks)), [sites, project])
  const tidy = async () => {
    setBusy(true); setError(''); setDone('')
    try {
      const got = await crewApi.tidy(project.key)
      setDone(`Cleared ${got.cleared}. Their reports and files stay in your VFS, under crew/.`)
      onChanged()
    } catch (e) {
      setError(errorOf(e))
    }
    setBusy(false)
  }
  return (
    <div className="@container min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl space-y-5 p-4 wide:p-6">
        <div className="flex flex-wrap items-start gap-3">
          {onBack && <button onClick={onBack} className={libraryIconBtn} aria-label="Back"><ArrowLeft className="h-4 w-4" /></button>}
          <ProjectMark name={project.name} big />
          <div className="min-w-0 flex-1 basis-48">
            <h3 className="break-words text-xl font-semibold tracking-tight text-zinc-100">{project.name}</h3>
            <p className="mt-0.5 text-sm text-zinc-500">{tasks.length ? plural(tasks.length, 'task') : 'No tasks yet'}</p>
          </div>
          <button onClick={onNew} className={primaryBtn}><Plus className="h-4 w-4" /> New task here</button>
        </div>

        <div className="grid grid-cols-2 gap-3 @xl:grid-cols-4">
          <StatTile icon={Loader2} tone="sky" label="Working" value={tasks.filter(isWorking).length} />
          <StatTile icon={MessageCircleQuestion} tone="amber" label="Needs you" value={tasks.filter(needsYou).length + count('stopped')} />
          <StatTile icon={CheckCircle2} tone="emerald" label="To look at" value={count('ready', 'pr_open')} />
          <StatTile icon={Check} label="Finished" value={finished} />
        </div>

        {done && <p className="rounded-xl bg-emerald-500/10 px-3 py-2 text-xs text-emerald-800 dark:text-emerald-200">{done}</p>}
        <ErrorLine text={error} />

        {envs.map((env) => <EnvironmentCard key={env.id} env={env} me={me} onChanged={onEnvChanged} onOpenTask={onOpen} onTerminal={onTerminal} />)}

        {tasks.length === 0 ? (
          <EmptyState icon={Ship} title="No tasks here yet">Start one, or ask your agent in chat.</EmptyState>
        ) : (
          <Card
            title="Tasks" icon={Ship} bodyClassName="divide-y divide-zinc-800/70"
            action={finished > 0 ? <SureButton icon={<Sparkles className="h-4 w-4" />} busy={busy} label={`Tidy up ${finished} finished`} sure="Yes, clear them off this page" onClick={tidy} /> : undefined}
          >
            {tasks.map((t) => <TaskLine key={t.id} task={t} onOpen={onOpen} />)}
          </Card>
        )}
        {ownSites.length > 0 && <SitesCard sites={ownSites} onSites={onSites} onPublish={onPublish} />}
        {project.paths.length > 0 && (
          <details className="group/where text-xs">
            <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-zinc-500 transition-colors hover:text-zinc-300 [&::-webkit-details-marker]:hidden">
              <ChevronRight className="h-3.5 w-3.5 transition-transform group-open/where:rotate-90" /> Where its files are
            </summary>
            <div className="mt-2 flex flex-wrap gap-1.5">{project.paths.map((p) => <CopyChip key={p} text={p} />)}</div>
          </details>
        )}
      </div>
    </div>
  )
}

// ── A new task ───────────────────────────────────────────────────────────

/** A worker sends the model the code it reads, so whose model it is matters. */
function FreeModelsNote({ url }: { url?: string }) {
  return (
    <span className="block font-normal leading-relaxed text-zinc-500">
      Free models need no key, and your code goes to them. They differ in what happens to it:
      {' '}<a href={url} target="_blank" rel="noreferrer" className="underline">check each one</a>.
    </span>
  )
}

function harnessHint(h: CrewHarness): string {
  if (!h.installed) return 'Not installed on this computer'
  if (!h.login.ok) return 'Needs a login'
  return h.login.how ? `Ready, on ${h.login.how}` : 'Ready'
}

function NewTask({ me, initialProject, initialUrl, onCancel, onMade, onLogins }: {
  me: CrewMe; initialProject: string; initialUrl: string; onCancel: () => void; onMade: (id: string) => void; onLogins: () => void
}) {
  const usable = useMemo(() => (me.harnesses ?? []).filter((h) => h.installed && h.login.ok), [me.harnesses])
  const [harness, setHarness] = useState(usable[0]?.id ?? '')
  const [brief, setBrief] = useState('')
  const [kind, setKind] = useState<'ship' | 'scout'>('ship')
  const [deliver, setDeliver] = useState('branch')
  const [where, setWhere] = useState<'project' | 'url' | 'path'>(initialUrl ? 'url' : 'project')
  const [repos, setRepos] = useState<CrewRepos | null>(null)
  const [project, setProject] = useState(initialProject)
  const [folder, setFolder] = useState('')
  const [url, setUrl] = useState(initialUrl)
  const [path, setPath] = useState('')
  const [model, setModel] = useState('')
  const [effort, setEffort] = useState('')
  const chosen = me.harnesses?.find((h) => h.id === harness)
  const needsModel = !!chosen?.login.free && !chosen.default_model
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    crewApi.repos().then((r) => {
      setRepos(r)
      if (r.projects[0]) setProject((p) => p || r.projects[0].project)
    }).catch((e) => setError(errorOf(e)))
  }, [])

  const folders = repos?.projects.find((p) => p.project === project)?.folders ?? []
  const named = where === 'project' ? !!project.trim() : where === 'url' ? !!url.trim() : !!path.trim()

  const submit = async (ev: FormEvent) => {
    ev.preventDefault()
    setBusy(true); setError('')
    const body: CrewDispatch = { harness, brief: brief.trim(), kind, deliver: kind === 'ship' ? deliver : 'branch', model: model.trim(), effort: effort.trim() }
    if (where === 'project') { body.project = project.trim(); body.folder = folder.trim() }
    if (where === 'url') body.repo_url = url.trim()
    if (where === 'path') body.repo_path = path.trim()
    try {
      const made = await crewApi.dispatch(body)
      onMade(made.id)
    } catch (e) {
      setError(errorOf(e))
      setBusy(false)
    }
  }

  const modelField = (
    <label className={labelCls}>
      Model
      <input
        list="crew-models" value={model} onChange={(e) => setModel(e.target.value)} aria-label="Model"
        placeholder={chosen?.default_model ? `Your default: ${chosen.default_model}` : needsModel ? 'Choose one of its free models' : 'Its default'}
        className={`${fieldClass} mt-1.5 font-normal`}
      />
      <datalist id="crew-models">{chosen?.models?.map((m) => <option key={m} value={m} />)}</datalist>
      {chosen?.login.free && <span className="mt-1.5 block"><FreeModelsNote url={me.free_terms_url} /></span>}
    </label>
  )

  return (
    <form onSubmit={submit} className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-2xl space-y-6 p-4 wide:p-6">
          <div className="flex items-start gap-3">
            <button type="button" onClick={onCancel} className={libraryIconBtn} aria-label="Back"><ArrowLeft className="h-4 w-4" /></button>
            <div className="min-w-0">
              <h3 className="text-xl font-semibold tracking-tight text-zinc-100">New task</h3>
              <p className="mt-0.5 text-sm text-zinc-500">An agent does it in a copy of the project. Nothing lands until you've looked at it.</p>
            </div>
          </div>

          {!usable.length && (
            <p className="rounded-2xl border border-amber-500/30 bg-amber-500/[0.06] px-4 py-3 text-sm text-amber-800 dark:text-amber-200">
              No coding agent is ready yet. <button type="button" onClick={onLogins} className="underline">Add a login</button>{' '}
              or ask an admin to install one.
            </p>
          )}

          <label className={labelCls}>
            What to do
            <textarea
              value={brief} onChange={(e) => setBrief(e.target.value)} rows={5} required autoFocus
              placeholder="The whole task. It sees only this: the goal, what done means, anything already decided."
              className={`${fieldClass} mt-1.5 resize-y font-normal`}
            />
          </label>

          <fieldset>
            <legend className={labelCls}>Who does it</legend>
            <div className="grid gap-2 sm:grid-cols-2">
              {(me.harnesses ?? []).map((h) => (
                <ChoiceTile key={h.id} icon={Bot} title={h.label} hint={harnessHint(h)} disabled={!h.installed || !h.login.ok} on={harness === h.id} onClick={() => setHarness(h.id)} />
              ))}
            </div>
          </fieldset>
          {needsModel && modelField}

          <fieldset>
            <legend className={labelCls}>Where</legend>
            <Segmented
              label="Where" value={where} onChange={setWhere}
              options={[
                { value: 'project' as const, label: 'A project' },
                { value: 'url' as const, label: 'A repository address' },
                ...(me.trusted ? [{ value: 'path' as const, label: 'A folder here' }] : []),
              ]}
            />
            {where === 'project' && (
              <div className="grid gap-2 pt-2 sm:grid-cols-2">
                <input list="crew-projects" value={project} onChange={(e) => setProject(e.target.value)} placeholder="Project (a new name makes one)" aria-label="Project" className={fieldClass} />
                <datalist id="crew-projects">{repos?.projects.map((p) => <option key={p.project} value={p.project} />)}</datalist>
                <input list="crew-folders" value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="Folder (optional)" aria-label="Folder" className={fieldClass} />
                <datalist id="crew-folders">{folders.map((f) => <option key={f} value={f} />)}</datalist>
              </div>
            )}
            {where === 'url' && <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://github.com/you/project" aria-label="Repository address" className={`${fieldClass} mt-2`} />}
            {where === 'path' && <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="/path/to/a/git/repository" aria-label="Folder" className={`${fieldClass} mt-2`} />}
          </fieldset>

          <fieldset>
            <legend className={labelCls}>What kind of task</legend>
            <div className="grid gap-2 sm:grid-cols-2">
              <ChoiceTile icon={Wrench} title="Change the code" hint="It makes the change. You decide if it lands." on={kind === 'ship'} onClick={() => setKind('ship')} />
              <ChoiceTile icon={Search} title="Only look, and report" hint="It reads the project and changes nothing." on={kind === 'scout'} onClick={() => setKind('scout')} />
            </div>
          </fieldset>

          {kind === 'ship' && (
            <fieldset>
              <legend className={labelCls}>When it's done</legend>
              <Segmented
                label="When it's done" value={deliver} onChange={setDeliver}
                options={[{ value: 'branch', label: 'Leave it for me to look at' }, { value: 'pr', label: 'Open a pull request' }]}
              />
            </fieldset>
          )}

          <details className="group/more text-xs">
            <summary className="inline-flex cursor-pointer list-none items-center gap-1 font-medium text-zinc-500 transition-colors hover:text-zinc-300 [&::-webkit-details-marker]:hidden">
              <ChevronRight className="h-3.5 w-3.5 transition-transform group-open/more:rotate-90" /> More options
            </summary>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              {!needsModel && modelField}
              <label className={labelCls}>
                Effort
                <input value={effort} onChange={(e) => setEffort(e.target.value)} placeholder="Like high" aria-label="Effort" className={`${fieldClass} mt-1.5 font-normal`} />
              </label>
            </div>
          </details>
          <ErrorLine text={error} />
        </div>
      </div>
      <div className="flex shrink-0 justify-end gap-2 border-t border-zinc-800 bg-zinc-900/40 px-4 py-3">
        <button type="button" onClick={onCancel} className={quietBtn}>Cancel</button>
        <button type="submit" disabled={busy || !harness || !brief.trim() || !named || (needsModel && !model.trim())} className={primaryBtn}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />} Start
        </button>
      </div>
    </form>
  )
}

// ── Logins ───────────────────────────────────────────────────────────────

const LOGINS: { name: string; label: string; hint: string; placeholder: string; multiline?: boolean }[] = [
  { name: 'claude', label: 'Claude Code', hint: 'Run claude setup-token on your own computer and paste the token (uses your plan), or paste an Anthropic API key.', placeholder: 'sk-ant-…' },
  { name: 'codex', label: 'Codex', hint: 'Paste an OpenAI API key, or the contents of ~/.codex/auth.json after codex login. A login used here shouldn\'t also be used elsewhere: Codex refreshes it in place.', placeholder: 'sk-… or {"auth_mode": …}', multiline: true },
  { name: 'github', label: 'GitHub', hint: 'A token that can read and push your repositories. Only Flight Deck uses it, to clone, push and open pull requests; the agents never see it.', placeholder: 'github_pat_…' },
  { name: 'anthropic', label: 'Anthropic key', hint: 'For Pi, and OpenCode\'s paid models.', placeholder: 'sk-ant-…' },
  { name: 'openai', label: 'OpenAI key', hint: 'For Pi, and OpenCode\'s paid models.', placeholder: 'sk-…' },
  { name: 'openrouter', label: 'OpenRouter key', hint: 'For Pi, and OpenCode\'s paid models.', placeholder: 'sk-or-…' },
  { name: 'gemini', label: 'Google Gemini key', hint: 'For Pi, and OpenCode\'s paid models.', placeholder: 'AI…' },
  { name: 'xai', label: 'xAI key', hint: 'For Pi, and OpenCode\'s paid models.', placeholder: 'xai-…' },
]

function SavedPill({ hint }: { hint?: string }) {
  return <Pill tone="emerald" icon={Check}>saved{hint ? ` ${hint}` : ''}</Pill>
}

function LoginRow({ login, saved, onChanged }: {
  login: typeof LOGINS[number]; saved?: { hint: string }; onChanged: () => void
}) {
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError('')
    try { await fn(); setValue(''); onChanged() } catch (e) { setError(errorOf(e)) }
    setBusy(false)
  }
  return (
    <div className="space-y-2 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-zinc-100">{login.label}</span>
        {saved && <SavedPill hint={saved.hint} />}
        {saved && <button disabled={busy} onClick={() => run(() => crewApi.deleteLogin(login.name))} className="ml-auto text-xs text-zinc-500 hover:text-red-600 dark:hover:text-red-300">Remove</button>}
      </div>
      <p className="text-xs leading-relaxed text-zinc-500">{login.hint}</p>
      <div className="flex items-start gap-2">
        {login.multiline ? (
          <textarea value={value} onChange={(e) => setValue(e.target.value)} rows={1} placeholder={saved ? 'Replace it…' : login.placeholder} aria-label={login.label} autoComplete="off" spellCheck={false} className={`${fieldClass} resize-y font-mono text-xs`} />
        ) : (
          <input type="password" value={value} onChange={(e) => setValue(e.target.value)} placeholder={saved ? 'Replace it…' : login.placeholder} aria-label={login.label} autoComplete="off" className={fieldClass} />
        )}
        <button disabled={busy || !value.trim()} onClick={() => run(() => crewApi.saveLogin(login.name, value.trim()))} className={quietBtn}>Save</button>
      </div>
      <ErrorLine text={error} />
    </div>
  )
}

/** A model server of their own. The key is sealed here, and reaches a worker only while it runs. */
function OwnModelCard({ own, onChanged }: { own?: CrewMe['own_model']; onChanged: () => void }) {
  const [url, setUrl] = useState(own?.url ?? '')
  const [key, setKey] = useState('')
  const [models, setModels] = useState((own?.models ?? []).join(', '))
  const [context, setContext] = useState(own?.context ? String(own.context) : '')
  const [found, setFound] = useState<string[]>([])
  const [checked, setChecked] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setError(''); setChecked('')
    try { await fn() } catch (e) { setError(errorOf(e)) }
    setBusy(false)
  }
  const check = () => run(async () => {
    const got = await crewApi.checkOwnModel(url.trim(), key.trim())
    if (!got.ok) { setFound([]); setError(got.error); return }
    setUrl(got.url)
    setFound(got.models)
    if (!models.trim() && got.models[0]) setModels(got.models[0])
    setChecked(got.models.length ? 'It answered. Pick what to use below.' : 'It answered, but listed no models.')
  })
  const save = () => run(async () => {
    const saved = await crewApi.saveOwnModel({ url: url.trim(), key: key.trim(), models: models.trim(), context: context.trim() })
    setUrl(saved.url)
    setKey('')
    setChecked(`Saved. OpenCode and Pi can run on ${saved.names.join(', ')}.`)
    onChanged()
  })
  const remove = () => run(async () => {
    await crewApi.deleteOwnModel()
    setUrl(''); setKey(''); setModels(''); setContext(''); setFound([])
    onChanged()
  })
  return (
    <div className="space-y-2.5 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-zinc-100">Your own model</span>
        {own && <SavedPill />}
        {own && <button disabled={busy} onClick={remove} className="ml-auto text-xs text-zinc-500 hover:text-red-600 dark:hover:text-red-300">Remove</button>}
      </div>
      <p className="text-xs leading-relaxed text-zinc-500">
        A server that speaks the OpenAI API, like llama.cpp or Ollama, for OpenCode and Pi. It has to be reachable from the computer running Flight Deck, and able to call tools.
      </p>
      <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="http://192.168.1.20:8080/v1" aria-label="Its address" autoComplete="off" spellCheck={false} className={fieldClass} />
      <input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder={own?.has_key ? 'Key saved. Type to replace it…' : 'Key, if it needs one'} aria-label="Its key" autoComplete="off" className={fieldClass} />
      <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_9rem]">
        <input value={models} onChange={(e) => setModels(e.target.value)} placeholder="Model name, as the server calls it" aria-label="Model name" autoComplete="off" spellCheck={false} className={fieldClass} />
        <input value={context} onChange={(e) => setContext(e.target.value)} inputMode="numeric" placeholder="Context (optional)" aria-label="Context size in tokens" className={fieldClass} />
      </div>
      {found.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {found.map((m) => <button key={m} type="button" onClick={() => setModels(m)} className={`rounded-lg border px-2 py-0.5 font-mono text-[11px] transition-colors ${models === m ? 'border-violet-500/60 bg-violet-500/10 text-violet-700 dark:text-violet-300' : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`}>{m}</button>)}
        </div>
      )}
      {checked && <p className="rounded-xl bg-emerald-500/10 px-3 py-2 text-xs text-emerald-800 dark:text-emerald-200">{checked}</p>}
      <ErrorLine text={error} />
      <div className="flex flex-wrap gap-2">
        <button disabled={busy || !url.trim()} onClick={check} className={quietBtn}>Check connection</button>
        <button disabled={busy || !url.trim() || !models.trim()} onClick={save} className={primaryBtn}>Save</button>
        {busy && <Loader2 className="my-auto h-4 w-4 animate-spin text-zinc-500" />}
      </div>
    </div>
  )
}

function HarnessLine({ h, termsUrl, onChanged }: { h: CrewHarness; termsUrl?: string; onChanged: () => void }) {
  const [model, setModel] = useState(h.default_model)
  const [error, setError] = useState('')
  useEffect(() => setModel(h.default_model), [h.default_model])
  const save = async () => {
    if (model.trim() === h.default_model) return
    setError('')
    try {
      await crewApi.saveModel(h.id, model.trim())
      onChanged()
    } catch (e) {
      setError(errorOf(e))
    }
  }
  return (
    <div className="space-y-2 p-4">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
        <span className="font-medium text-zinc-100">{h.label}</span>
        {!h.installed ? (
          <Pill tone="amber">not installed</Pill>
        ) : h.login.ok ? (
          <span className="text-xs text-zinc-500">ready, on {h.login.how}{h.version ? ` · ${h.version}` : ''}</span>
        ) : (
          <Pill tone="amber">needs a login</Pill>
        )}
      </div>
      {!h.installed && <p className="text-xs text-zinc-500">An admin can run: <code className="rounded bg-zinc-800 px-1 py-0.5 text-[11px]">{h.install}</code></p>}
      {h.installed && (
        <div className="text-xs text-zinc-500">
          <input
            list={`crew-models-${h.id}`} value={model} onChange={(e) => setModel(e.target.value)} onBlur={save}
            onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
            placeholder={h.login.free ? 'Default model: choose one of its free models' : 'Default model: its own'} aria-label={`Default model for ${h.label}`}
            className={`${fieldClass} !py-1.5 text-xs`}
          />
          <datalist id={`crew-models-${h.id}`}>{h.models?.map((m) => <option key={m} value={m} />)}</datalist>
          {h.login.free && <p className="mt-1"><FreeModelsNote url={termsUrl} /></p>}
          <ErrorLine text={error} />
        </div>
      )}
    </div>
  )
}

function Logins({ me, onBack, onChanged }: { me: CrewMe; onBack: () => void; onChanged: () => void }) {
  return (
    <div className="@container min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl space-y-5 p-4 wide:p-6">
        <div className="flex items-start gap-3">
          <button onClick={onBack} className={libraryIconBtn} aria-label="Back"><ArrowLeft className="h-4 w-4" /></button>
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-violet-500/10 text-violet-600 dark:text-violet-400"><KeyRound className="h-5 w-5" /></span>
          <div className="min-w-0">
            <h3 className="text-xl font-semibold tracking-tight text-zinc-100">Logins</h3>
            <p className="mt-0.5 max-w-2xl text-sm leading-relaxed text-zinc-500">
              {me.trusted
                ? 'Without a saved login, your agents use the ones already on this computer.'
                : 'Your agents run on your own accounts. Keys from "Your own AI keys" count too.'}
              {' '}What you save here is stored encrypted and never shown again.
            </p>
          </div>
        </div>
        <Card title="Your coding agents" icon={Bot} bodyClassName="divide-y divide-zinc-800/70">
          {(me.harnesses ?? []).map((h) => <HarnessLine key={h.id} h={h} termsUrl={me.free_terms_url} onChanged={onChanged} />)}
        </Card>
        <OwnModelCard own={me.own_model} onChanged={onChanged} />
        <div className="grid gap-3 @2xl:grid-cols-2">
          {LOGINS.map((l) => <LoginRow key={l.name} login={l} saved={me.logins?.[l.name]} onChanged={onChanged} />)}
        </div>
      </div>
    </div>
  )
}

// ── The side rail ────────────────────────────────────────────────────────

function RailLink({ icon: Icon, label, on, badge, onClick }: { icon: LucideIcon; label: string; on: boolean; badge?: ReactNode; onClick: () => void }) {
  return (
    <button
      onClick={onClick} aria-current={on ? 'page' : undefined}
      className={`group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors ${on ? 'bg-violet-500/10 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200'}`}
    >
      <Icon className={`h-[18px] w-[18px] shrink-0 ${on ? 'text-violet-600 dark:text-violet-400' : 'text-zinc-500 group-hover:text-zinc-300'}`} />
      <span className="min-w-0 flex-1 truncate text-sm font-medium">{label}</span>
      {badge}
    </button>
  )
}

function TaskRow({ task, active, onOpen }: { task: CrewTask; active: boolean; onOpen: () => void }) {
  return (
    <button
      onClick={onOpen} aria-current={active ? 'page' : undefined}
      className={`flex w-full items-start gap-2.5 rounded-xl px-2.5 py-2 text-left transition-colors ${active ? 'bg-violet-500/10 ring-1 ring-inset ring-violet-500/30' : 'hover:bg-zinc-800/60'}`}
    >
      <StatusDot task={task} />
      <span className="min-w-0 flex-1">
        <span className="line-clamp-2 break-words text-[13px] leading-snug text-zinc-200">{task.title}</span>
        <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-zinc-500">
          <span className="min-w-0 truncate">{statusLabel(task)}</span>
          <span className="ml-auto shrink-0 tabular-nums">{ago(task.updated_at)}</span>
        </span>
      </span>
    </button>
  )
}

function ProjectGroup({ project, collapsed, selectedTask, selectedProject, onToggle, onProject, onTask, onNew }: {
  project: Project; collapsed: boolean; selectedTask: string; selectedProject: boolean
  onToggle: () => void; onProject: () => void; onTask: (id: string) => void; onNew: () => void
}) {
  const [showFinished, setShowFinished] = useState(false)
  const open = useMemo(() => project.tasks.filter((t) => !isFinished(t)).sort(byUrgency), [project.tasks])
  const finished = useMemo(() => project.tasks.filter(isFinished).sort(byUrgency), [project.tasks])
  const working = project.tasks.filter(isWorking).length
  const wanting = project.tasks.filter(needsYou).length
  const finishedOpen = showFinished || finished.some((t) => t.id === selectedTask)
  return (
    <div>
      <div className={`group flex items-center gap-0.5 rounded-xl pr-1 transition-colors ${selectedProject ? 'bg-violet-500/10' : 'hover:bg-zinc-800/60'}`}>
        <button onClick={onProject} className="flex min-w-0 flex-1 items-center gap-2.5 py-2 pl-2.5 text-left">
          <ProjectMark name={project.name} />
          <span className="min-w-0 truncate text-[13px] font-semibold text-zinc-200">{project.name}</span>
          {working > 0 && <span className="inline-flex shrink-0 items-center gap-0.5 text-[11px] text-sky-600 dark:text-sky-300" title={`${working} working`}><Loader2 className="h-3 w-3 animate-spin" />{working}</span>}
          {wanting > 0 && <span className="shrink-0 rounded-full bg-amber-500/15 px-1.5 text-[11px] font-medium text-amber-700 dark:text-amber-300" title={`${wanting} need you`}>{wanting}</span>}
        </button>
        <button onClick={onNew} className="rounded-lg p-1 text-zinc-500 opacity-0 transition-opacity hover:text-violet-600 focus:opacity-100 group-hover:opacity-100 dark:hover:text-violet-300" title={`New task in ${project.name}`} aria-label={`New task in ${project.name}`}>
          <Plus className="h-3.5 w-3.5" />
        </button>
        <button onClick={onToggle} className="rounded-lg p-1 text-zinc-500 hover:text-zinc-300" aria-label={collapsed ? `Open ${project.name}` : `Fold ${project.name}`} aria-expanded={!collapsed}>
          <ChevronRight className={`h-3.5 w-3.5 transition-transform ${collapsed ? '' : 'rotate-90'}`} />
        </button>
      </div>
      {!collapsed && (
        <div className="space-y-0.5 pb-1.5 pl-3 pt-0.5">
          {open.map((t) => <TaskRow key={t.id} task={t} active={selectedTask === t.id} onOpen={() => onTask(t.id)} />)}
          {finished.length > 0 && (
            <>
              <button onClick={() => setShowFinished((v) => !v)} className="flex items-center gap-1 px-2.5 py-1 text-[11px] text-zinc-500 transition-colors hover:text-zinc-300">
                <ChevronRight className={`h-3 w-3 transition-transform ${finishedOpen ? 'rotate-90' : ''}`} /> Finished ({finished.length})
              </button>
              {finishedOpen && finished.map((t) => <TaskRow key={t.id} task={t} active={selectedTask === t.id} onOpen={() => onTask(t.id)} />)}
            </>
          )}
          {!project.tasks.length && <p className="px-2.5 py-1 text-[11px] text-zinc-500">No tasks yet.</p>}
        </div>
      )}
    </div>
  )
}

function NewProjectLine({ onMade }: { onMade: (key: string) => void }) {
  const [name, setName] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  if (name === null) {
    return <button onClick={() => setName('')} className={`${libraryIconBtn} !p-1`} title="New project" aria-label="New project"><FolderPlus className="h-3.5 w-3.5" /></button>
  }
  const submit = async (ev: FormEvent) => {
    ev.preventDefault()
    setBusy(true); setError('')
    try {
      const made = await crewApi.newProject(name.trim())
      setName(null)
      onMade(made.key)
    } catch (e) {
      setError(errorOf(e))
    }
    setBusy(false)
  }
  return (
    <form onSubmit={submit} className="absolute inset-x-2 z-10 mt-7 space-y-2 rounded-2xl border border-zinc-700 bg-zinc-900 p-2.5 shadow-xl shadow-black/30">
      <input
        autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="Project name" aria-label="Project name"
        onKeyDown={(e) => { if (e.key === 'Escape') setName(null) }} className={fieldClass}
      />
      <ErrorLine text={error} />
      <div className="flex justify-end gap-1.5">
        <button type="button" onClick={() => setName(null)} className={quietBtn}>Cancel</button>
        <button type="submit" disabled={busy || !name.trim()} className={primaryBtn}>Make it</button>
      </div>
    </form>
  )
}

// ── The page ─────────────────────────────────────────────────────────────

type Pane = 'home' | 'sites' | 'task' | 'project' | 'new' | 'logins' | 'terminal'

export function CrewBoard({ phone = false, onClose }: { phone?: boolean; onClose?: () => void }) {
  const [me, setMe] = useState<CrewMe | null>(null)
  const [tasks, setTasks] = useState<CrewTask[] | null>(null)
  const [sites, setSites] = useState<HostedSite[] | null>(null)
  const [repos, setRepos] = useState<CrewRepos | null>(null)
  const [usage, setUsage] = useState<CrewUsage[]>([])
  const [selected, setSelected] = useState('')
  const [projectKey, setProjectKey] = useState('')
  const [pane, setPane] = useState<Pane>('home')
  const [publishing, setPublishing] = useState(false)
  const [newIn, setNewIn] = useState('')
  const [newUrl, setNewUrl] = useState('')
  const [collapsed, setCollapsed] = useState<string[]>(readCollapsed)
  const [error, setError] = useState('')
  const [meError, setMeError] = useState('')
  const [listed, setListed] = useState<Record<string, string[]>>({})
  const [envs, setEnvs] = useState<CrewEnv[]>([])
  const [shell, setShell] = useState<{ target: CrewTerminalTarget; title: string; back: Pane } | null>(null)

  // The models come apart: a CLI can take half a minute to say, and the page doesn't wait for it.
  const loadMe = useCallback(() => {
    setMeError('')
    crewApi.me().then(setMe).catch((e) => setMeError(errorOf(e)))
    crewApi.models().then(setListed).catch(() => {})
  }, [])
  const loadRepos = useCallback(() => crewApi.repos().then(setRepos).catch(() => {}), [])
  const loadEnvs = useCallback(() => crewApi.envs().then((r) => setEnvs(r.envs)).catch(() => {}), [])
  const loadSites = useCallback(() => hostingApi.list().then((r) => setSites(r.entries)).catch(() => setSites((s) => s ?? [])), [])
  const loadTasks = useCallback(async () => {
    try {
      const { tasks: list } = await crewApi.tasks()
      setTasks(list)
      setError('')
    } catch (e) {
      setError(errorOf(e))
    }
  }, [])
  const refreshSites = useCallback(async () => { await Promise.all([loadSites(), loadTasks()]) }, [loadSites, loadTasks])

  useEffect(() => { loadMe() }, [loadMe])
  useEffect(() => {
    if (me && !me.allowed) return
    loadTasks()
    loadSites()
    loadRepos()
    const t = setInterval(() => { loadTasks(); loadSites() }, POLL_LIST_MS)
    return () => clearInterval(t)
  }, [loadTasks, loadSites, loadRepos, me])
  useEffect(() => {
    if (me && !me.allowed) return
    const load = () => { crewApi.usage().then((u) => setUsage(u.harnesses)).catch(() => {}) }
    load()
    const t = setInterval(load, POLL_USAGE_MS)
    return () => clearInterval(t)
  }, [me])
  // It takes a look in each project's folder: often enough to see a setup land while someone watches, no more.
  useEffect(() => {
    if (me && !me.allowed) return
    loadEnvs()
    if (pane !== 'project') return
    const t = setInterval(loadEnvs, POLL_ENVS_MS)
    return () => clearInterval(t)
  }, [loadEnvs, me, pane])

  const { projects, idle } = useMemo(() => {
    const groups = new Map<string, Project>()
    for (const t of tasks ?? []) {
      const key = t.project_key || t.repo_label
      const group = groups.get(key) ?? { key, name: t.project_name || t.repo_label, tasks: [], paths: [] }
      group.tasks.push(t)
      groups.set(key, group)
    }
    const unused: Project[] = []
    for (const p of repos?.projects ?? []) {
      const paths = Object.values(p.paths ?? {}).filter(Boolean)
      const group = groups.get(p.key)
      if (group) group.paths = paths
      else unused.push({ key: p.key, name: p.project, tasks: [], paths })
    }
    // A repository that's here with no task on the page is still a project of theirs.
    for (const e of envs) {
      if (!groups.has(e.project_key) && !unused.some((p) => p.key === e.project_key)) {
        unused.push({ key: e.project_key, name: e.project_name, tasks: [], paths: [] })
      }
    }
    const latest = (p: Project) => Math.max(...p.tasks.map((t) => t.updated_at ?? 0))
    return { projects: [...groups.values()].sort((a, b) => latest(b) - latest(a)), idle: unused }
  }, [tasks, repos, envs])

  const meListed = useMemo(() => me && { ...me, harnesses: me.harnesses?.map((h) => ({ ...h, models: listed[h.id] ?? h.models })) }, [me, listed])
  const all = [...projects, ...idle]
  const current = all.find((p) => p.key === projectKey)
  const working = (tasks ?? []).filter(isWorking).length
  const wanting = (tasks ?? []).filter(needsYou).length
  const live = (sites ?? []).filter(isLive).length

  // A new task is aimed at a Code project by name, and at a repository by the address it came from.
  const inProject = (key?: string, name?: string) => (key?.startsWith('project:') ? name ?? '' : '')
  const addressOf = (list?: CrewTask[]) => list?.find((t) => t.repo_url)?.repo_url ?? ''
  const selectedTask = tasks?.find((t) => t.id === selected)
  const open = (id: string) => { setSelected(id); setPane('task') }
  const openProject = (key: string) => { setProjectKey(key); setSelected(''); setPane('project') }
  const startIn = (name: string, address = '') => { setNewIn(name); setNewUrl(address); setPane('new') }
  const openShell = (target: CrewTerminalTarget, title: string) => { setShell({ target, title, back: pane }); setPane('terminal') }
  const goHome = () => { setSelected(''); setPane('home') }
  const goSites = () => { setSelected(''); setPane('sites') }
  const toggle = (key: string) => {
    setCollapsed((old) => {
      const next = old.includes(key) ? old.filter((k) => k !== key) : [...old, key]
      saveCollapsed(next)
      return next
    })
  }

  if (me && !me.allowed) {
    return (
      <div className="flex h-full flex-col bg-zinc-950">
        <Header onClose={onClose} />
        <div className="flex flex-1 items-center justify-center p-6">
          <EmptyState icon={Ship} title={me.offered ? 'Crew needs an admin\'s approval' : 'Crew is turned off'}>
            {me.offered ? 'An admin needs to approve Crew for your account first.' : 'Crew is turned off on this Flight Deck.'}
          </EmptyState>
        </div>
      </div>
    )
  }

  const group = (p: Project) => (
    <ProjectGroup
      key={p.key} project={p} collapsed={collapsed.includes(p.key)} selectedTask={pane === 'task' ? selected : ''}
      selectedProject={pane === 'project' && projectKey === p.key} onToggle={() => toggle(p.key)} onProject={() => openProject(p.key)}
      onTask={open} onNew={() => startIn(inProject(p.key, p.name), addressOf(p.tasks))}
    />
  )
  const projectList = (
    <>
      {projects.map(group)}
      {idle.length > 0 && (
        <details className="pt-1">
          <summary className="cursor-pointer px-2.5 py-1 text-[11px] text-zinc-500 hover:text-zinc-300">Projects with no tasks ({idle.length})</summary>
          <div className="space-y-1">{idle.map(group)}</div>
        </details>
      )}
    </>
  )
  const newProject = <NewProjectLine onMade={(key) => { loadRepos(); openProject(key) }} />
  const backOnPhone = phone ? goHome : undefined

  return (
    <div className="flex h-full min-h-0 flex-col bg-zinc-950">
      <Header
        onClose={onClose} onRefresh={() => { loadTasks(); loadSites(); loadMe(); loadRepos() }} onLogins={() => setPane('logins')}
        onNew={() => (selectedTask
          ? startIn(inProject(selectedTask.project_key, selectedTask.project_name), selectedTask.repo_url ?? '')
          : startIn(inProject(current?.key, current?.name), addressOf(current?.tasks)))}
        summary={[working ? `${working} working` : '', wanting ? `${wanting} need${wanting === 1 ? 's' : ''} you` : '', live ? `${live} live` : ''].filter(Boolean).join(' · ')}
      />
      {(error || (me && meError)) && <div className="p-2"><ErrorLine text={error || meError} /></div>}
      <div className="flex min-h-0 flex-1">
        {!phone && (
          <aside className="relative flex w-72 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/40">
            <nav className="shrink-0 space-y-0.5 p-2" aria-label="Crew">
              <RailLink icon={LayoutGrid} label="Overview" on={pane === 'home'} onClick={goHome} />
              <RailLink
                icon={Globe} label="Live sites" on={pane === 'sites'} onClick={goSites}
                badge={live > 0 ? <span className="rounded-full bg-emerald-500/15 px-1.5 text-[11px] font-medium tabular-nums text-emerald-700 dark:text-emerald-300">{live}</span> : undefined}
              />
            </nav>
            <div className="relative flex shrink-0 items-center px-4 pb-1 pt-2">
              <span className="mr-auto text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Projects</span>
              {newProject}
            </div>
            <div className="min-h-0 flex-1 space-y-1 overflow-y-auto px-2 pb-3">
              {tasks === null && <p className="p-3 text-sm text-zinc-500">Loading…</p>}
              {tasks?.length === 0 && <p className="px-2.5 py-2 text-xs leading-relaxed text-zinc-500">Tasks gather here by project, once you've started one.</p>}
              {projectList}
            </div>
          </aside>
        )}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          {pane === 'home' && (
            <CrewOverview tasks={tasks ?? []} sites={sites ?? []} usage={usage} onOpen={open} onNew={() => startIn('')} onSites={goSites} onPublish={() => setPublishing(true)}>
              {phone && (projects.length > 0 || idle.length > 0) && (
                <Card title="Projects" action={newProject} bodyClassName="relative space-y-1 p-2">{projectList}</Card>
              )}
            </CrewOverview>
          )}
          {pane === 'sites' && (
            <CrewSites sites={sites} tasks={tasks ?? []} me={me} onBack={backOnPhone} onChanged={refreshSites} onOpenTask={open} onPublish={() => setPublishing(true)} />
          )}
          {(pane === 'new' || pane === 'logins') && !meListed && (
            <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-center text-sm text-zinc-500">
              {meError
                ? <><ErrorLine text={meError} /><button onClick={loadMe} className={quietBtn}>Try again</button></>
                : <><Loader2 className="h-5 w-5 animate-spin" /> Loading…</>}
            </div>
          )}
          {pane === 'new' && meListed && <NewTask key={`${newIn}|${newUrl}`} me={meListed} initialProject={newIn} initialUrl={newUrl} onCancel={goHome} onLogins={() => setPane('logins')} onMade={(id) => { loadTasks(); loadRepos(); open(id) }} />}
          {pane === 'logins' && meListed && <Logins me={meListed} onBack={goHome} onChanged={loadMe} />}
          {pane === 'project' && current && (
            <ProjectView
              project={current} sites={sites ?? []} envs={envs.filter((e) => e.project_key === current.key)} me={meListed} onOpen={open}
              onNew={() => startIn(inProject(current.key, current.name), addressOf(current.tasks) || envs.find((e) => e.project_key === current.key)?.repo_url || '')}
              onChanged={() => { loadTasks(); loadRepos(); loadEnvs() }} onEnvChanged={loadEnvs} onTerminal={(e) => openShell({ env: e.id }, e.label)}
              onBack={backOnPhone} onSites={goSites} onPublish={() => setPublishing(true)}
            />
          )}
          {pane === 'project' && !current && (
            <div className="flex flex-1 items-center justify-center p-6"><EmptyState icon={Ship} title="That project isn't here any more" /></div>
          )}
          {pane === 'task' && selected && (
            <TaskView
              key={selected} taskId={selected} canTerminal={!!me?.terminal} onBack={backOnPhone} onChanged={() => { loadTasks(); loadEnvs() }}
              onTerminal={(t) => openShell({ task: t.id }, t.title)}
            />
          )}
          {pane === 'terminal' && shell && (
            <Suspense fallback={<div className="flex flex-1 items-center justify-center text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>}>
              <CrewTerminal key={shell.target.env || shell.target.task} target={shell.target} title={shell.title} onBack={() => setPane(shell.back)} />
            </Suspense>
          )}
        </div>
      </div>
      {publishing && (
        <PublishSheet
          tasks={tasks ?? []} sites={sites ?? []} me={me} onClose={() => setPublishing(false)}
          onDone={() => { setPublishing(false); refreshSites(); goSites() }}
        />
      )}
    </div>
  )
}

function Header({ onClose, onRefresh, onLogins, onNew, summary }: {
  onClose?: () => void; onRefresh?: () => void; onLogins?: () => void; onNew?: () => void; summary?: string
}) {
  return (
    <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 bg-zinc-900/40 px-4 py-3">
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-lg shadow-violet-500/25">
        <Ship className="h-5 w-5" />
      </span>
      <div className="mr-auto min-w-0">
        <h2 className="truncate text-base font-semibold tracking-tight text-zinc-100">Crew</h2>
        <p className="truncate text-xs text-zinc-500">{summary || 'Coding agents working for you'}</p>
      </div>
      {onNew && <button onClick={onNew} className={primaryBtn} aria-label="New task"><Plus className="h-4 w-4" /><span className="hidden sm:inline">New task</span></button>}
      {onLogins && <button onClick={onLogins} className={libraryIconBtn} title="Logins" aria-label="Logins"><KeyRound className="h-4 w-4" /></button>}
      {onRefresh && <button onClick={onRefresh} className={`${libraryIconBtn} hidden wide:block`} title="Refresh" aria-label="Refresh"><RefreshCw className="h-4 w-4" /></button>}
      <FullscreenButton />
      {onClose && <button onClick={onClose} className={libraryIconBtn} title="Close" aria-label="Close"><X className="h-4 w-4" /></button>}
    </div>
  )
}
