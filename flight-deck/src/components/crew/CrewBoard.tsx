import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react'
import {
  AlertTriangle, ArrowLeft, Check, CheckCircle2, ChevronRight, CircleDot, FileDiff, FolderGit2, FolderOpen, FolderPlus,
  GitBranch, GitMerge, GitPullRequest, KeyRound, Loader2, MessageSquare, Plus, RefreshCw, ScrollText, Send, Ship, Sparkles,
  Square, Trash2, X,
} from 'lucide-react'
import { FullscreenButton, libraryIconBtn } from '../library/libraryParts'
import {
  crewApi, type CrewDispatch, type CrewEvent, type CrewHarness, type CrewLimit, type CrewMe, type CrewRepos, type CrewTask,
  type CrewUsage,
} from '../../services/crewApi'
import { CrewActivity } from './CrewActivity'
import { CrewChanges } from './CrewChanges'
import { CrewFiles } from './CrewFiles'
import {
  CopyChip, CrewMarkdown, DONE, ErrorLine, LIVE, StatusDot, StatusPill, ago, byUrgency, errorOf, fieldClass, isFinished,
  primaryBtn, quietBtn, statParts, statusLabel,
} from './crewParts'

// ── Crew ──
//
// Coding agents working for this user, by project: what each is doing, what git says
// it did, and what's waiting for them. The projects and their tasks on the left, one
// task or one project on the right; on a phone, one or the other. An agent's own chat
// can start and steer the same tasks, so this page polls rather than assuming it made
// them.

const POLL_LIST_MS = 4000
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
  const tone = danger ? 'border-red-500/50 text-red-700 hover:bg-red-500/10 dark:text-red-300' : ''
  if (asking) {
    return (
      <button
        disabled={busy}
        className={danger ? `${quietBtn} ${tone}` : primaryBtn}
        onClick={() => { setAsking(false); onClick() }}
      >
        <Check className="h-3.5 w-3.5" /> {sure}
      </button>
    )
  }
  return <button disabled={busy} className={primary ? primaryBtn : danger ? `${quietBtn} ${tone}` : quietBtn} onClick={() => setAsking(true)}>{icon} {label}</button>
}

function Outcome({ task }: { task: CrewTask }) {
  if (!task.note) return null
  const good = task.status === 'ready' || task.status === 'delivered' || task.status === 'pr_open'
  const look = task.status === 'waiting' ? 'bg-amber-500/10 text-amber-800 dark:text-amber-200'
    : task.status === 'failed' ? 'bg-red-500/10 text-red-800 dark:text-red-200'
      : good ? 'bg-teal-500/10 text-teal-900 dark:text-teal-100' : 'bg-zinc-800/50 text-zinc-300'
  const Icon = task.status === 'waiting' || task.status === 'failed' ? AlertTriangle : good ? CheckCircle2 : CircleDot
  return (
    <p className={`flex items-start gap-2 rounded-xl px-3 py-2 text-sm ${look}`}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0 opacity-70" />
      <span className="min-w-0 break-words">{task.note}</span>
    </p>
  )
}

function GitFacts({ task }: { task: CrewTask }) {
  const e = task.evidence || {}
  const { files, added, removed } = statParts(e.stat)
  const chip = 'inline-flex items-center gap-1 rounded-md bg-zinc-800/60 px-2 py-0.5'
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs text-zinc-400" title="What git shows, not what the agent says">
      <span className={chip}><CircleDot className="h-3 w-3" />{e.commits ?? 0} commit{e.commits === 1 ? '' : 's'}</span>
      {files > 0 && <span className={chip}>{files} file{files === 1 ? '' : 's'}</span>}
      {added > 0 && <span className={`${chip} font-mono text-emerald-700 dark:text-emerald-400`}>+{added}</span>}
      {removed > 0 && <span className={`${chip} font-mono text-red-700 dark:text-red-400`}>−{removed}</span>}
      {!!e.dirty && <span className={`${chip} text-amber-700 dark:text-amber-300`}>{e.dirty} uncommitted</span>}
      {e.off_branch && <span className={`${chip} text-red-700 dark:text-red-300`}>left its branch: on {e.off_branch}</span>}
      {!!e.stray && <span className={`${chip} text-red-700 dark:text-red-300`}>{e.stray} commit{e.stray === 1 ? '' : 's'} off its branch</span>}
      {task.delivery?.url && <a href={task.delivery.url} target="_blank" rel="noreferrer" className={`${chip} text-teal-700 underline dark:text-teal-300`}>pull request #{task.delivery.pr}</a>}
    </div>
  )
}

type TaskTab = 'activity' | 'changes' | 'files' | 'report'

function TaskView({ taskId, onBack, onChanged }: { taskId: string; onBack?: () => void; onChanged: () => void }) {
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

  const tabs: [TaskTab, string, typeof Ship][] = [
    ['activity', 'Activity', MessageSquare],
    ...(ship ? [['changes', 'Changes', FileDiff] as [TaskTab, string, typeof Ship]] : []),
    ['files', 'Files', FolderOpen],
    ...(task.report ? [['report', 'Report', ScrollText] as [TaskTab, string, typeof Ship]] : []),
  ]

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="space-y-2.5 border-b border-zinc-800 p-4">
        <div className="flex items-start gap-2">
          {onBack && <button onClick={onBack} className={libraryIconBtn} aria-label="Back to the list"><ArrowLeft className="h-4 w-4" /></button>}
          <div className="min-w-0 flex-1">
            <h3 className="break-words text-[15px] font-semibold leading-snug text-zinc-100">{task.title}</h3>
            <p className="mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-xs text-zinc-500">
              <span title={task.model_used ? 'As the CLI reports it' : undefined}>{task.harness_label}{(task.model_used || task.model) ? ` · ${task.model_used || task.model}` : ''}</span>
              <span>{task.repo_label}</span>
              {task.branch && <span className="inline-flex min-w-0 items-center gap-1"><GitBranch className="h-3 w-3 shrink-0" /><span className="truncate">{task.branch}</span></span>}
              {task.from_agent && <span>from your agent</span>}
              {task.turn > 1 && <span>turn {task.turn}</span>}
              {(task.tokens_in + task.tokens_out) > 0 && <span>{Math.round((task.tokens_in + task.tokens_out) / 1000)}k tokens</span>}
            </p>
          </div>
          <StatusPill task={task} />
        </div>

        <Outcome task={task} />
        {ship && (task.turn > 0 || finished) && <GitFacts task={task} />}

        {(live || landable || (!live && !finished)) && (
          <div className="flex flex-wrap gap-1.5 pt-0.5">
            {live && <button disabled={busy} className={quietBtn} onClick={() => act(() => crewApi.stop(task.id))}><Square className="h-3.5 w-3.5" /> Stop</button>}
            {landable && (
              <SureButton
                primary icon={<GitMerge className="h-3.5 w-3.5" />} busy={busy}
                label={hasPr ? 'Merge the pull request' : `Merge into ${task.base_branch || 'its branch'}`}
                sure="Yes, merge it" onClick={() => act(() => crewApi.merge(task.id))}
              />
            )}
            {landable && (
              <button disabled={busy} className={quietBtn} onClick={() => act(() => crewApi.pr(task.id))}><GitPullRequest className="h-3.5 w-3.5" /> {hasPr ? 'Update the pull request' : 'Open a pull request'}</button>
            )}
            {!live && !finished && (
              unlanded ? (
                <SureButton danger icon={<Trash2 className="h-3.5 w-3.5" />} busy={busy} label="Discard" sure="Yes, throw this work away" onClick={() => act(() => crewApi.close(task.id, true))} />
              ) : (
                <button disabled={busy} className={quietBtn} onClick={() => act(() => crewApi.close(task.id))}><X className="h-3.5 w-3.5" /> Close</button>
              )
            )}
          </div>
        )}
        <ErrorLine text={error} />
      </div>

      <div className="flex shrink-0 gap-4 border-b border-zinc-800 px-4 text-xs">
        {tabs.map(([id, label, Icon]) => (
          <button key={id} onClick={() => setTab(id)} className={`-mb-px inline-flex items-center gap-1.5 border-b-2 py-2.5 font-medium transition-colors ${tab === id ? 'border-teal-500 text-zinc-100' : 'border-transparent text-zinc-500 hover:text-zinc-300'}`}>
            <Icon className="h-3.5 w-3.5" /> {label}
          </button>
        ))}
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {tab === 'activity' && <CrewActivity events={events} live={live} />}
        {tab === 'changes' && <CrewChanges diff={diff} evidence={e} />}
        {tab === 'files' && <CrewFiles task={task} />}
        {tab === 'report' && <div className="min-h-0 flex-1 overflow-y-auto p-5"><CrewMarkdown text={task.report ?? ''} /></div>}
      </div>

      {!finished && (
        <form onSubmit={send} className="flex shrink-0 items-end gap-2 border-t border-zinc-800 p-3">
          <textarea
            value={steer}
            onChange={(ev) => setSteer(ev.target.value)}
            onKeyDown={(ev) => { if (ev.key === 'Enter' && !ev.shiftKey) send(ev) }}
            rows={Math.min(5, steer.split('\n').length)}
            placeholder={task.status === 'waiting' ? 'Your answer…' : live ? 'Tell it something, for when this turn ends…' : 'Tell it to go on…'}
            aria-label="Tell the agent"
            className={`${fieldClass} resize-none`}
          />
          <button type="submit" disabled={busy || !steer.trim()} className={`${primaryBtn} h-[38px] w-[38px] !px-0`} aria-label="Send"><Send className="h-4 w-4" /></button>
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

function ProjectView({ project, onOpen, onNew, onChanged, onBack }: {
  project: Project; onOpen: (id: string) => void; onNew: () => void; onChanged: () => void; onBack?: () => void
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [done, setDone] = useState('')
  useEffect(() => { setDone(''); setError('') }, [project.key])
  const tasks = useMemo(() => [...project.tasks].sort(byUrgency), [project.tasks])
  const finished = tasks.filter(isFinished).length
  const count = (...statuses: CrewTask['status'][]) => tasks.filter((t) => statuses.includes(t.status) && !isFinished(t)).length
  const tiles: [string, number, string][] = [
    ['Working', count('running', 'queued'), 'text-sky-600 dark:text-sky-300'],
    ['Needs you', count('waiting', 'failed', 'stopped'), 'text-amber-700 dark:text-amber-300'],
    ['To land', count('ready', 'pr_open'), 'text-teal-700 dark:text-teal-300'],
    ['Finished', finished, 'text-zinc-400'],
  ]
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
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
      <div className="flex items-start gap-2">
        {onBack && <button onClick={onBack} className={libraryIconBtn} aria-label="Back to the list"><ArrowLeft className="h-4 w-4" /></button>}
        <FolderGit2 className="mt-1 h-5 w-5 shrink-0 text-teal-600 dark:text-teal-400" />
        <div className="min-w-0 flex-1">
          <h3 className="break-words text-lg font-semibold text-zinc-100">{project.name}</h3>
          {project.paths.length > 0 && <div className="mt-1 flex flex-wrap gap-1.5">{project.paths.map((p) => <CopyChip key={p} text={p} />)}</div>}
        </div>
        <button onClick={onNew} className={primaryBtn}><Plus className="h-4 w-4" /> New task here</button>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {tiles.map(([label, n, tone]) => (
          <div key={label} className="rounded-xl border border-zinc-800 p-3">
            <p className={`text-xl font-semibold tabular-nums ${n ? tone : 'text-zinc-600'}`}>{n}</p>
            <p className="text-xs text-zinc-500">{label}</p>
          </div>
        ))}
      </div>

      {finished > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <SureButton icon={<Sparkles className="h-3.5 w-3.5" />} busy={busy} label={`Tidy up ${finished} finished`} sure="Yes, clear them off this page" onClick={tidy} />
          <span className="text-xs text-zinc-500">Work that hasn't landed stays where it is.</span>
        </div>
      )}
      {done && <p className="rounded-lg bg-teal-500/10 px-3 py-2 text-xs text-teal-800 dark:text-teal-200">{done}</p>}
      <ErrorLine text={error} />

      {tasks.length === 0 ? (
        <p className="rounded-xl border border-dashed border-zinc-800 p-6 text-center text-sm text-zinc-500">No tasks here yet. Start one, or ask your agent in chat.</p>
      ) : (
        <ul className="divide-y divide-zinc-800/70 overflow-hidden rounded-xl border border-zinc-800">
          {tasks.map((t) => (
            <li key={t.id}>
              <button onClick={() => onOpen(t.id)} className="flex w-full items-start gap-2.5 px-3 py-2.5 text-left transition-colors hover:bg-zinc-800/50">
                <StatusDot task={t} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-zinc-200">{t.title}</span>
                  {t.note && <span className="block truncate text-xs text-zinc-500">{t.note}</span>}
                </span>
                <StatusPill task={t} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ── A new task ───────────────────────────────────────────────────────────

/** A worker sends the model the code it reads, so whose model it is matters. */
function FreeModelsNote({ url }: { url?: string }) {
  return (
    <span className="block font-normal text-zinc-500">
      Free models need no key, and your code goes to them. They differ in what happens to it:
      {' '}<a href={url} target="_blank" rel="noreferrer" className="underline">check each one</a>.
    </span>
  )
}

function NewTask({ me, initialProject, onCancel, onMade, onLogins }: {
  me: CrewMe; initialProject: string; onCancel: () => void; onMade: (id: string) => void; onLogins: () => void
}) {
  const usable = useMemo(() => (me.harnesses ?? []).filter((h) => h.installed && h.login.ok), [me.harnesses])
  const [harness, setHarness] = useState(usable[0]?.id ?? '')
  const [brief, setBrief] = useState('')
  const [kind, setKind] = useState<'ship' | 'scout'>('ship')
  const [deliver, setDeliver] = useState('branch')
  const [where, setWhere] = useState<'project' | 'url' | 'path'>('project')
  const [repos, setRepos] = useState<CrewRepos | null>(null)
  const [project, setProject] = useState(initialProject)
  const [folder, setFolder] = useState('')
  const [url, setUrl] = useState('')
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

  const choice = (on: boolean) => `rounded-lg border px-3 py-1.5 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${on ? 'border-teal-500 bg-teal-500/10 text-teal-700 dark:text-teal-300' : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`

  return (
    <form onSubmit={submit} className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
      <div className="flex items-center gap-2">
        <button type="button" onClick={onCancel} className={libraryIconBtn} aria-label="Back"><ArrowLeft className="h-4 w-4" /></button>
        <h3 className="text-[15px] font-semibold text-zinc-100">New task</h3>
      </div>

      {!usable.length && (
        <p className="rounded-xl bg-amber-500/10 px-3 py-2 text-sm text-amber-800 dark:text-amber-200">
          No coding agent is ready yet. <button type="button" onClick={onLogins} className="underline">Add a login</button>{' '}
          or ask an admin to install one.
        </p>
      )}

      <label className="block space-y-1.5 text-xs font-medium text-zinc-400">
        <span>What to do</span>
        <textarea
          value={brief} onChange={(e) => setBrief(e.target.value)} rows={6} required autoFocus
          placeholder="The whole task. It sees only this: the goal, what done means, anything already decided."
          className={`${fieldClass} resize-y font-normal`}
        />
      </label>

      <fieldset className="space-y-1.5">
        <legend className="text-xs font-medium text-zinc-400">Who does it</legend>
        <div className="flex flex-wrap gap-1.5">
          {(me.harnesses ?? []).map((h) => (
            <button
              key={h.id} type="button" disabled={!h.installed || !h.login.ok} onClick={() => setHarness(h.id)} className={choice(harness === h.id)}
              title={!h.installed ? 'Not installed' : !h.login.ok ? 'No login yet' : h.login.how}
            >
              {h.label}{!h.installed ? ' · not installed' : !h.login.ok ? ' · no login' : ''}
            </button>
          ))}
        </div>
      </fieldset>

      <label className="block space-y-1.5 text-xs font-medium text-zinc-400">
        <span>Model</span>
        <input
          list="crew-models" value={model} onChange={(e) => setModel(e.target.value)} aria-label="Model"
          placeholder={chosen?.default_model ? `Your default: ${chosen.default_model}` : needsModel ? 'Choose one of its free models' : 'Its default'}
          className={`${fieldClass} font-normal`}
        />
        <datalist id="crew-models">{chosen?.models?.map((m) => <option key={m} value={m} />)}</datalist>
        {chosen?.login.free && <FreeModelsNote url={me.free_terms_url} />}
      </label>

      <fieldset className="space-y-1.5">
        <legend className="text-xs font-medium text-zinc-400">Where</legend>
        <div className="flex flex-wrap gap-1.5">
          {([['project', 'A project'], ['url', 'A repository address'], ...(me.trusted ? [['path', 'A folder on this computer']] : [])] as [typeof where, string][]).map(([id, label]) => (
            <button key={id} type="button" onClick={() => setWhere(id)} className={choice(where === id)}>{label}</button>
          ))}
        </div>
        {where === 'project' && (
          <div className="grid gap-2 pt-1 sm:grid-cols-2">
            <input list="crew-projects" value={project} onChange={(e) => setProject(e.target.value)} placeholder="Project (a new name makes one)" aria-label="Project" className={fieldClass} />
            <datalist id="crew-projects">{repos?.projects.map((p) => <option key={p.project} value={p.project} />)}</datalist>
            <input list="crew-folders" value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="Folder (optional)" aria-label="Folder" className={fieldClass} />
            <datalist id="crew-folders">{folders.map((f) => <option key={f} value={f} />)}</datalist>
          </div>
        )}
        {where === 'url' && <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://github.com/you/project" aria-label="Repository address" className={`${fieldClass} mt-1`} />}
        {where === 'path' && <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="/path/to/a/git/repository" aria-label="Folder" className={`${fieldClass} mt-1`} />}
      </fieldset>

      <fieldset className="space-y-1.5">
        <legend className="text-xs font-medium text-zinc-400">Kind</legend>
        <div className="flex flex-wrap gap-1.5">
          <button type="button" onClick={() => setKind('ship')} className={choice(kind === 'ship')}>Change the code</button>
          <button type="button" onClick={() => setKind('scout')} className={choice(kind === 'scout')}>Only investigate, and report</button>
        </div>
      </fieldset>

      {kind === 'ship' && (
        <fieldset className="space-y-1.5">
          <legend className="text-xs font-medium text-zinc-400">When it's done</legend>
          <div className="flex flex-wrap gap-1.5">
            <button type="button" onClick={() => setDeliver('branch')} className={choice(deliver === 'branch')}>Leave it on a branch for me</button>
            <button type="button" onClick={() => setDeliver('pr')} className={choice(deliver === 'pr')}>Open a pull request</button>
          </div>
        </fieldset>
      )}

      <details className="text-xs text-zinc-400">
        <summary className="cursor-pointer font-medium">Effort</summary>
        <input value={effort} onChange={(e) => setEffort(e.target.value)} placeholder="Effort, e.g. high" aria-label="Effort" className={`${fieldClass} mt-2 sm:w-1/2`} />
      </details>

      <ErrorLine text={error} />
      <button type="submit" disabled={busy || !harness || !brief.trim() || !named || (needsModel && !model.trim())} className={primaryBtn}>
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />} Start
      </button>
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
    <div className="space-y-1.5 rounded-xl border border-zinc-800 p-3">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-zinc-200">{login.label}</span>
        {saved && <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] text-emerald-700 dark:text-emerald-300"><Check className="h-3 w-3" /> saved {saved.hint}</span>}
        {saved && <button disabled={busy} onClick={() => run(() => crewApi.deleteLogin(login.name))} className="ml-auto text-xs text-zinc-500 hover:text-red-600 dark:hover:text-red-300">Remove</button>}
      </div>
      <p className="text-xs text-zinc-500">{login.hint}</p>
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
    <div className="space-y-2 rounded-xl border border-zinc-800 p-3">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-zinc-200">Your own model</span>
        {own && <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] text-emerald-700 dark:text-emerald-300"><Check className="h-3 w-3" /> saved</span>}
        {own && <button disabled={busy} onClick={remove} className="ml-auto text-xs text-zinc-500 hover:text-red-600 dark:hover:text-red-300">Remove</button>}
      </div>
      <p className="text-xs text-zinc-500">
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
          {found.map((m) => <button key={m} type="button" onClick={() => setModels(m)} className={`rounded-md border px-2 py-0.5 font-mono text-[11px] transition-colors ${models === m ? 'border-teal-500 bg-teal-500/10 text-teal-700 dark:text-teal-300' : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`}>{m}</button>)}
        </div>
      )}
      {checked && <p className="rounded-lg bg-teal-500/10 px-3 py-2 text-xs text-teal-800 dark:text-teal-200">{checked}</p>}
      <ErrorLine text={error} />
      <div className="flex flex-wrap gap-1.5">
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
    <li className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
        <span className="font-medium text-zinc-200">{h.label}</span>
        {!h.installed ? (
          <span className="text-xs text-amber-700 dark:text-amber-300">not installed. An admin can run: <code className="rounded bg-zinc-800 px-1 py-0.5 text-[11px]">{h.install}</code></span>
        ) : h.login.ok ? (
          <span className="text-xs text-zinc-500">ready, on {h.login.how}{h.version ? ` · ${h.version}` : ''}</span>
        ) : (
          <span className="text-xs text-amber-700 dark:text-amber-300">needs a login</span>
        )}
      </div>
      {h.installed && (
        <div className="text-xs text-zinc-500">
          <input
            list={`crew-models-${h.id}`} value={model} onChange={(e) => setModel(e.target.value)} onBlur={save}
            onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
            placeholder={h.login.free ? 'Default model: choose one of its free models' : 'Default model: its own'} aria-label={`Default model for ${h.label}`}
            className={`${fieldClass} !py-1 text-xs`}
          />
          <datalist id={`crew-models-${h.id}`}>{h.models?.map((m) => <option key={m} value={m} />)}</datalist>
          {h.login.free && <p className="mt-1"><FreeModelsNote url={termsUrl} /></p>}
          <ErrorLine text={error} />
        </div>
      )}
    </li>
  )
}

function Logins({ me, onBack, onChanged }: { me: CrewMe; onBack: () => void; onChanged: () => void }) {
  return (
    <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
      <div className="flex items-center gap-2">
        <button onClick={onBack} className={libraryIconBtn} aria-label="Back"><ArrowLeft className="h-4 w-4" /></button>
        <h3 className="text-sm font-semibold text-zinc-100">Logins</h3>
      </div>
      <ul className="space-y-3 rounded-xl border border-zinc-800 p-3">{(me.harnesses ?? []).map((h) => <HarnessLine key={h.id} h={h} termsUrl={me.free_terms_url} onChanged={onChanged} />)}</ul>
      <p className="text-xs text-zinc-500">
        {me.trusted
          ? 'Without a saved login, your agents use the ones already on this computer.'
          : 'Your agents run on your own accounts. Keys from "Your own AI keys" count too.'}
        {' '}What you save here is stored encrypted and never shown again.
      </p>
      <OwnModelCard own={me.own_model} onChanged={onChanged} />
      {LOGINS.map((l) => <LoginRow key={l.name} login={l} saved={me.logins?.[l.name]} onChanged={onChanged} />)}
    </div>
  )
}

// ── The board ────────────────────────────────────────────────────────────

function TaskRow({ task, active, onOpen }: { task: CrewTask; active: boolean; onOpen: () => void }) {
  return (
    <button
      onClick={onOpen}
      className={`flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors ${active ? 'bg-teal-500/10 ring-1 ring-inset ring-teal-500/40' : 'hover:bg-zinc-800/60'}`}
    >
      <StatusDot task={task} />
      <span className="min-w-0 flex-1">
        <span className="line-clamp-2 break-words text-[13px] leading-snug text-zinc-200">{task.title}</span>
        <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-zinc-500">
          <span className="min-w-0 truncate">{statusLabel(task)} · {task.harness_label}</span>
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
  const working = project.tasks.filter((t) => t.status === 'running').length
  const wanting = project.tasks.filter((t) => t.status === 'waiting' || t.status === 'failed').length
  const landing = open.filter((t) => t.status === 'ready' || t.status === 'pr_open').length
  const finishedOpen = showFinished || finished.some((t) => t.id === selectedTask)
  return (
    <div>
      <div className={`group flex items-center gap-0.5 rounded-lg pr-1 transition-colors ${selectedProject ? 'bg-zinc-800/70' : 'hover:bg-zinc-800/40'}`}>
        <button onClick={onToggle} className="rounded p-1.5 text-zinc-500 hover:text-zinc-300" aria-label={collapsed ? `Open ${project.name}` : `Fold ${project.name}`} aria-expanded={!collapsed}>
          <ChevronRight className={`h-3.5 w-3.5 transition-transform ${collapsed ? '' : 'rotate-90'}`} />
        </button>
        <button onClick={onProject} className="flex min-w-0 flex-1 items-center gap-1.5 py-1.5 text-left">
          <span className="min-w-0 truncate text-[13px] font-semibold text-zinc-200">{project.name}</span>
          {working > 0 && <span className="inline-flex shrink-0 items-center gap-0.5 text-[11px] text-sky-600 dark:text-sky-300" title={`${working} working`}><Loader2 className="h-3 w-3 animate-spin" />{working}</span>}
          {wanting > 0 && <span className="shrink-0 rounded-full bg-amber-500/15 px-1.5 text-[11px] font-medium text-amber-700 dark:text-amber-300" title={`${wanting} need you`}>{wanting}</span>}
          {landing > 0 && <span className="shrink-0 rounded-full bg-teal-500/15 px-1.5 text-[11px] font-medium text-teal-700 dark:text-teal-300" title={`${landing} to land`}>{landing}</span>}
        </button>
        <button onClick={onNew} className="rounded p-1 text-zinc-500 opacity-0 transition-opacity hover:text-teal-600 focus:opacity-100 group-hover:opacity-100 dark:hover:text-teal-300" title={`New task in ${project.name}`} aria-label={`New task in ${project.name}`}>
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>
      {!collapsed && (
        <div className="ml-3 space-y-0.5 border-l border-zinc-800 pb-1 pl-1.5 pt-0.5">
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

const WINDOWS: Record<string, string> = {
  five_hour: '5 hours', seven_day: 'This week', seven_day_opus: 'This week, Opus', seven_day_sonnet: 'This week, Sonnet',
}

function LimitBar({ name, limit }: { name: string; limit: CrewLimit }) {
  const pct = limit.used === null ? null : Math.round(limit.used * 100)
  const stopped = !!limit.status && !limit.status.startsWith('allowed')
  const tone = stopped || (pct ?? 0) >= 90 ? 'bg-red-500' : limit.status === 'allowed_warning' || (pct ?? 0) >= 75 ? 'bg-amber-500' : 'bg-teal-500'
  const resets = limit.resets_at ? new Date(limit.resets_at * 1000).toLocaleString(undefined, { weekday: 'short', hour: 'numeric', minute: '2-digit' }) : ''
  return (
    <div>
      <div className="flex items-baseline gap-2 text-[11px]">
        <span className="text-zinc-400">{WINDOWS[name] ?? name.replace(/_/g, ' ')}</span>
        <span className="ml-auto text-zinc-500">{resets && `resets ${resets}`}</span>
        <span className="font-medium tabular-nums text-zinc-300">{pct === null ? limit.status : `${pct}%`}</span>
      </div>
      {pct !== null && (
        <div className="mt-1 h-1 overflow-hidden rounded-full bg-zinc-800">
          <div className={`h-full rounded-full ${tone}`} style={{ width: `${Math.min(100, Math.max(2, pct))}%` }} />
        </div>
      )}
    </div>
  )
}

function compactTokens(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1000 ? `${Math.round(n / 1000)}k` : String(n)
}

/** Each CLI's model and limits, as it last reported them while working: there's no
 *  asking between runs, so the panel says how old they are. */
function UsagePanel({ rows }: { rows: CrewUsage[] }) {
  if (!rows.length) return null
  return (
    <div className="max-h-52 shrink-0 space-y-2.5 overflow-y-auto border-t border-zinc-800 px-3.5 py-2.5">
      {rows.map((u) => (
        <div key={u.id} className="space-y-1.5">
          <p className="flex items-baseline gap-1.5 text-xs">
            <span className="shrink-0 font-medium text-zinc-300">{u.label}</span>
            {u.model && <span className="min-w-0 truncate font-mono text-[11px] text-zinc-500">{u.model}</span>}
          </p>
          {Object.entries(u.limits).map(([name, limit]) => <LimitBar key={name} name={name} limit={limit} />)}
          <p className="text-[11px] text-zinc-500" title={u.cost_usd ? `About $${u.cost_usd.toFixed(2)} at API prices` : undefined}>
            {u.tasks} task{u.tasks === 1 ? '' : 's'} · {compactTokens(u.tokens)} tokens in the last week
            {u.seen_at > 0 && Object.keys(u.limits).length > 0 && ` · limits as of ${ago(u.seen_at)} ago`}
          </p>
        </div>
      ))}
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
    <form onSubmit={submit} className="absolute inset-x-2 top-8 z-10 space-y-1.5 rounded-xl border border-zinc-700 bg-zinc-900 p-2 shadow-lg">
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

export function CrewBoard({ phone = false, onClose }: { phone?: boolean; onClose?: () => void }) {
  const [me, setMe] = useState<CrewMe | null>(null)
  const [tasks, setTasks] = useState<CrewTask[] | null>(null)
  const [repos, setRepos] = useState<CrewRepos | null>(null)
  const [usage, setUsage] = useState<CrewUsage[]>([])
  const [selected, setSelected] = useState('')
  const [projectKey, setProjectKey] = useState('')
  const [pane, setPane] = useState<'task' | 'project' | 'new' | 'logins'>('task')
  const [newIn, setNewIn] = useState('')
  const [collapsed, setCollapsed] = useState<string[]>(readCollapsed)
  const [error, setError] = useState('')
  const [meError, setMeError] = useState('')
  const [listed, setListed] = useState<Record<string, string[]>>({})

  // The models come apart: a CLI can take half a minute to say, and the page doesn't wait for it.
  const loadMe = useCallback(() => {
    setMeError('')
    crewApi.me().then(setMe).catch((e) => setMeError(errorOf(e)))
    crewApi.models().then(setListed).catch(() => {})
  }, [])
  const loadRepos = useCallback(() => crewApi.repos().then(setRepos).catch(() => {}), [])
  const loadTasks = useCallback(async () => {
    try {
      const { tasks: list } = await crewApi.tasks()
      setTasks(list)
      setError('')
    } catch (e) {
      setError(errorOf(e))
    }
  }, [])

  useEffect(() => { loadMe() }, [loadMe])
  useEffect(() => {
    if (me && !me.allowed) return
    loadTasks()
    loadRepos()
    const t = setInterval(loadTasks, POLL_LIST_MS)
    return () => clearInterval(t)
  }, [loadTasks, loadRepos, me])
  useEffect(() => {
    if (me && !me.allowed) return
    const load = () => { crewApi.usage().then((u) => setUsage(u.harnesses)).catch(() => {}) }
    load()
    const t = setInterval(load, POLL_USAGE_MS)
    return () => clearInterval(t)
  }, [me])

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
    const latest = (p: Project) => Math.max(...p.tasks.map((t) => t.updated_at ?? 0))
    return { projects: [...groups.values()].sort((a, b) => latest(b) - latest(a)), idle: unused }
  }, [tasks, repos])

  const meListed = useMemo(() => me && { ...me, harnesses: me.harnesses?.map((h) => ({ ...h, models: listed[h.id] ?? h.models })) }, [me, listed])
  const all = [...projects, ...idle]
  const current = all.find((p) => p.key === projectKey)
  const working = (tasks ?? []).filter((t) => t.status === 'running').length
  const wanting = (tasks ?? []).filter((t) => t.status === 'waiting').length

  // A new task can only be aimed at a Code project by name: the rest came with an address.
  const inProject = (key?: string, name?: string) => (key?.startsWith('project:') ? name ?? '' : '')
  const selectedTask = tasks?.find((t) => t.id === selected)
  const open = (id: string) => { setSelected(id); setPane('task') }
  const openProject = (key: string) => { setProjectKey(key); setSelected(''); setPane('project') }
  const startIn = (name: string) => { setNewIn(name); setPane('new') }
  const toggle = (key: string) => {
    setCollapsed((old) => {
      const next = old.includes(key) ? old.filter((k) => k !== key) : [...old, key]
      saveCollapsed(next)
      return next
    })
  }
  const back = () => { setSelected(''); setPane('task') }
  const atList = pane === 'task' && !selected
  const showList = !phone || atList
  const showPane = !phone || !atList

  if (me && !me.allowed) {
    return (
      <div className="flex h-full flex-col bg-zinc-900">
        <Header onClose={onClose} />
        <div className="flex flex-1 items-center justify-center p-6 text-center text-sm text-zinc-400">
          {me.offered ? 'An admin needs to approve Crew for your account first.' : 'Crew is turned off on this Flight Deck.'}
        </div>
      </div>
    )
  }

  const group = (p: Project) => (
    <ProjectGroup
      key={p.key} project={p} collapsed={collapsed.includes(p.key)} selectedTask={pane === 'task' ? selected : ''}
      selectedProject={pane === 'project' && projectKey === p.key} onToggle={() => toggle(p.key)} onProject={() => openProject(p.key)}
      onTask={open} onNew={() => startIn(inProject(p.key, p.name))}
    />
  )

  return (
    <div className="flex h-full min-h-0 flex-col bg-zinc-900">
      <Header
        onClose={onClose} onRefresh={() => { loadTasks(); loadMe(); loadRepos() }} onLogins={() => setPane('logins')}
        onNew={() => startIn(selectedTask ? inProject(selectedTask.project_key, selectedTask.project_name) : inProject(current?.key, current?.name))}
        summary={[working ? `${working} working` : '', wanting ? `${wanting} need${wanting === 1 ? 's' : ''} you` : ''].filter(Boolean).join(' · ')}
      />
      {(error || (me && meError)) && <div className="p-2"><ErrorLine text={error || meError} /></div>}
      <div className="flex min-h-0 flex-1">
        {showList && (
          <div className={`relative flex min-h-0 flex-col ${phone ? 'flex-1' : 'w-80 shrink-0 border-r border-zinc-800'}`}>
            <div className="flex shrink-0 items-center px-3.5 pb-1 pt-2.5">
              <span className="mr-auto text-[11px] font-medium uppercase tracking-wide text-zinc-500">Projects</span>
              <NewProjectLine onMade={(key) => { loadRepos(); openProject(key) }} />
            </div>
            <div className="min-h-0 flex-1 space-y-1 overflow-y-auto px-1.5 pb-3">
              {tasks === null && <p className="p-3 text-sm text-zinc-500">Loading…</p>}
              {tasks?.length === 0 && (
                <div className="space-y-2 p-3 text-sm text-zinc-400">
                  <p>No tasks yet.</p>
                  <p className="text-xs text-zinc-500">Start one here, or ask your agent in chat: "have Claude Code fix the login test".</p>
                </div>
              )}
              {projects.map(group)}
              {idle.length > 0 && (
                <details className="pt-1">
                  <summary className="cursor-pointer px-2 py-1 text-[11px] text-zinc-500 hover:text-zinc-300">Projects with no tasks ({idle.length})</summary>
                  <div className="space-y-1">{idle.map(group)}</div>
                </details>
              )}
            </div>
            <UsagePanel rows={usage} />
          </div>
        )}
        {showPane && (
          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            {(pane === 'new' || pane === 'logins') && !meListed && (
              <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-center text-sm text-zinc-500">
                {meError
                  ? <><ErrorLine text={meError} /><button onClick={loadMe} className={quietBtn}>Try again</button></>
                  : <><Loader2 className="h-5 w-5 animate-spin" /> Loading…</>}
              </div>
            )}
            {pane === 'new' && meListed && <NewTask key={newIn} me={meListed} initialProject={newIn} onCancel={back} onLogins={() => setPane('logins')} onMade={(id) => { loadTasks(); loadRepos(); open(id) }} />}
            {pane === 'logins' && meListed && <Logins me={meListed} onBack={back} onChanged={loadMe} />}
            {pane === 'project' && current && (
              <ProjectView project={current} onOpen={open} onNew={() => startIn(inProject(current.key, current.name))} onChanged={() => { loadTasks(); loadRepos() }} onBack={phone ? back : undefined} />
            )}
            {pane === 'task' && selected && <TaskView key={selected} taskId={selected} onBack={phone ? back : undefined} onChanged={loadTasks} />}
            {((pane === 'task' && !selected) || (pane === 'project' && !current)) && (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 p-6 text-center">
                <Ship className="h-8 w-8 text-zinc-700" />
                <p className="text-sm text-zinc-500">Choose a project or a task, or start a new one.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function Header({ onClose, onRefresh, onLogins, onNew, summary }: {
  onClose?: () => void; onRefresh?: () => void; onLogins?: () => void; onNew?: () => void; summary?: string
}) {
  return (
    <div className="flex shrink-0 items-center gap-1 border-b border-zinc-800 px-3.5 py-2.5">
      <Ship className="h-4 w-4 text-teal-600 dark:text-teal-400" />
      <h2 className="pl-1 text-sm font-semibold text-zinc-100">Crew</h2>
      <span className="mr-auto min-w-0 truncate pl-2 text-xs text-zinc-500">{summary}</span>
      {onNew && <button onClick={onNew} className={primaryBtn}><Plus className="h-4 w-4" /> New task</button>}
      {onLogins && <button onClick={onLogins} className={libraryIconBtn} title="Logins" aria-label="Logins"><KeyRound className="h-4 w-4" /></button>}
      {onRefresh && <button onClick={onRefresh} className={libraryIconBtn} title="Refresh" aria-label="Refresh"><RefreshCw className="h-4 w-4" /></button>}
      <FullscreenButton />
      {onClose && <button onClick={onClose} className={libraryIconBtn} title="Close" aria-label="Close"><X className="h-4 w-4" /></button>}
    </div>
  )
}
