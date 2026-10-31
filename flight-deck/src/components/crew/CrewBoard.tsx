import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react'
import {
  AlertTriangle, ArrowLeft, Check, CircleDot, FileDiff, GitBranch, GitMerge, GitPullRequest, KeyRound, Loader2,
  MessageSquare, Plus, RefreshCw, Send, Ship, Square, Trash2, X,
} from 'lucide-react'
import { FullscreenButton, libraryIconBtn } from '../library/libraryParts'
import {
  crewApi, type CrewDispatch, type CrewEvent, type CrewHarness, type CrewMe, type CrewRepos, type CrewStatus, type CrewTask,
} from '../../services/crewApi'

// ── Crew ──
//
// Coding agents working for this user: what each is doing, what git says it did,
// and what's waiting for them. The list on the left, one task on the right; on a
// phone, one or the other. An agent's own chat can start and steer the same
// tasks, so this page polls rather than assuming it made them.

const POLL_LIST_MS = 4000
const POLL_EVENTS_MS = 2000

const STATUS: Record<CrewStatus, { label: string; tone: string }> = {
  queued: { label: 'Waiting for a slot', tone: 'bg-zinc-500/15 text-zinc-400' },
  running: { label: 'Working', tone: 'bg-sky-500/15 text-sky-600 dark:text-sky-300' },
  waiting: { label: 'Needs you', tone: 'bg-amber-500/15 text-amber-700 dark:text-amber-300' },
  ready: { label: 'Ready for review', tone: 'bg-teal-500/15 text-teal-700 dark:text-teal-300' },
  failed: { label: 'Failed', tone: 'bg-red-500/15 text-red-700 dark:text-red-300' },
  stopped: { label: 'Stopped', tone: 'bg-zinc-500/15 text-zinc-400' },
  pr_open: { label: 'Pull request open', tone: 'bg-violet-500/15 text-violet-700 dark:text-violet-300' },
  delivered: { label: 'Landed', tone: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300' },
  closed: { label: 'Closed', tone: 'bg-zinc-500/15 text-zinc-500' },
}
const LIVE: CrewStatus[] = ['queued', 'running']
const DONE: CrewStatus[] = ['delivered', 'closed']

const fieldClass = 'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-teal-500 focus:outline-none'
const primaryBtn = 'inline-flex items-center justify-center gap-1.5 rounded-md bg-teal-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-teal-500 disabled:cursor-not-allowed disabled:opacity-50'
const quietBtn = 'inline-flex items-center justify-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50'

const errorOf = (e: unknown) => (e instanceof Error ? e.message : 'Something went wrong')

function StatusPill({ task }: { task: Pick<CrewTask, 'status' | 'stalled'> }) {
  const s = STATUS[task.status] ?? STATUS.stopped
  return (
    <span className={`inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-medium ${task.stalled ? STATUS.waiting.tone : s.tone}`}>
      {task.status === 'running' && <Loader2 className="h-3 w-3 animate-spin" />}
      {task.stalled ? 'Quiet for a while' : s.label}
    </span>
  )
}

function ErrorLine({ text }: { text: string }) {
  if (!text) return null
  return (
    <p className="flex items-start gap-1.5 rounded-md bg-red-500/10 px-2.5 py-1.5 text-xs text-red-700 dark:text-red-300">
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> <span className="min-w-0 break-words">{text}</span>
    </p>
  )
}

// ── One task ─────────────────────────────────────────────────────────────

function Activity({ events }: { events: CrewEvent[] }) {
  const end = useRef<HTMLDivElement>(null)
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = box.current
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 240) end.current?.scrollIntoView({ block: 'end' })
  }, [events.length])
  if (!events.length) return <p className="p-4 text-sm text-zinc-500">Nothing yet.</p>
  return (
    <div ref={box} className="min-h-0 flex-1 space-y-1.5 overflow-y-auto p-3 font-mono text-xs leading-relaxed">
      {events.map((e) => <EventLine key={e.n} event={e} />)}
      <div ref={end} />
    </div>
  )
}

function EventLine({ event: e }: { event: CrewEvent }) {
  if (e.kind === 'turn') {
    return (
      <div className="mt-2 rounded-md bg-zinc-800/60 px-2.5 py-1.5 font-sans text-zinc-300">
        <span className="mr-1.5 font-medium text-zinc-500">{e.turn <= 1 ? 'Task' : `Told it (turn ${e.turn})`}</span>
        <span className="whitespace-pre-wrap break-words">{e.text.length > 1200 ? `${e.text.slice(0, 1200)}…` : e.text}</span>
      </div>
    )
  }
  if (e.kind === 'text') {
    // Its CREW-STATUS line is for the watcher: what it says shows as the status below.
    const said = e.text.replace(/^[\s*_`>]*CREW-STATUS:.*$/gim, '').trim()
    return said ? <p className="whitespace-pre-wrap break-words font-sans text-[13px] text-zinc-200">{said}</p> : null
  }
  if (e.kind === 'tool') {
    return (
      <p className="break-words text-sky-700 dark:text-sky-300">
        <span className="font-semibold">{e.tool}</span> <span className="whitespace-pre-wrap text-zinc-400">{e.text}</span>
      </p>
    )
  }
  if (e.kind === 'error') return <pre className="whitespace-pre-wrap break-words text-red-700 dark:text-red-300">{e.text}</pre>
  if (e.kind === 'status') {
    return <p className="break-words font-sans font-medium text-teal-700 dark:text-teal-300">{e.text}</p>
  }
  if (!e.text.trim()) return null
  return (
    <details className="text-zinc-500">
      <summary className="cursor-pointer truncate">{e.text.split('\n')[0].slice(0, 140) || 'output'}</summary>
      <pre className="mt-1 max-h-72 overflow-auto whitespace-pre-wrap break-words pl-3">{e.text}</pre>
    </details>
  )
}

function DiffView({ text }: { text: string }) {
  return (
    <pre className="overflow-x-auto p-3 font-mono text-xs leading-relaxed">
      {text.split('\n').map((line, i) => {
        const tone = line.startsWith('+') && !line.startsWith('+++') ? 'text-emerald-700 dark:text-emerald-400'
          : line.startsWith('-') && !line.startsWith('---') ? 'text-red-700 dark:text-red-400'
            : line.startsWith('@@') ? 'text-sky-700 dark:text-sky-400'
              : line.startsWith('diff ') ? 'font-semibold text-zinc-200' : 'text-zinc-500'
        return <div key={i} className={tone}>{line || ' '}</div>
      })}
    </pre>
  )
}

/** A step to be sure: the button asks once more before it does. */
function SureButton({ icon, label, sure, busy, danger, onClick }: {
  icon: ReactNode; label: string; sure: string; busy: boolean; danger?: boolean; onClick: () => void
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
  return <button disabled={busy} className={danger ? `${quietBtn} ${tone}` : quietBtn} onClick={() => setAsking(true)}>{icon} {label}</button>
}

function TaskView({ taskId, onBack, onChanged }: { taskId: string; onBack?: () => void; onChanged: () => void }) {
  const [task, setTask] = useState<CrewTask | null>(null)
  const [events, setEvents] = useState<CrewEvent[]>([])
  const [tab, setTab] = useState<'activity' | 'changes' | 'report'>('activity')
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

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="space-y-2 border-b border-zinc-800 p-3">
        <div className="flex items-start gap-2">
          {onBack && <button onClick={onBack} className={libraryIconBtn} aria-label="Back to the list"><ArrowLeft className="h-4 w-4" /></button>}
          <div className="min-w-0 flex-1">
            <h3 className="break-words text-sm font-semibold text-zinc-100">{task.title}</h3>
            <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-xs text-zinc-500">
              <span>{task.harness_label}{task.model ? ` · ${task.model}` : ''}</span>
              <span>{task.repo_label}</span>
              {task.branch && <span className="inline-flex items-center gap-1"><GitBranch className="h-3 w-3" />{task.branch}</span>}
              {task.from_agent && <span>from your agent</span>}
              {task.turn > 1 && <span>turn {task.turn}</span>}
              {(task.tokens_in + task.tokens_out) > 0 && <span>{Math.round((task.tokens_in + task.tokens_out) / 1000)}k tokens</span>}
            </p>
          </div>
          <StatusPill task={task} />
        </div>

        {task.note && (
          <p className={`break-words rounded-md px-2.5 py-1.5 text-sm ${task.status === 'waiting' ? 'bg-amber-500/10 text-amber-800 dark:text-amber-200' : 'bg-zinc-800/50 text-zinc-300'}`}>
            {task.note}
          </p>
        )}

        {ship && (task.turn > 0 || finished) && (
          <p className="flex flex-wrap items-center gap-x-3 text-xs text-zinc-400" title="What git shows, not what the agent says">
            <span className="inline-flex items-center gap-1"><CircleDot className="h-3 w-3" />{e.commits ?? 0} commit{e.commits === 1 ? '' : 's'}</span>
            {!!e.dirty && <span className="text-amber-700 dark:text-amber-300">{e.dirty} uncommitted</span>}
            {e.stat && <span>{e.stat}</span>}
            {e.off_branch && <span className="text-red-700 dark:text-red-300">left its branch: on {e.off_branch}</span>}
            {!!e.stray && <span className="text-red-700 dark:text-red-300">{e.stray} commit{e.stray === 1 ? '' : 's'} off its branch</span>}
            {task.delivery?.url && <a href={task.delivery.url} target="_blank" rel="noreferrer" className="text-teal-700 underline dark:text-teal-300">pull request #{task.delivery.pr}</a>}
          </p>
        )}

        <div className="flex flex-wrap gap-1.5">
          {live && <button disabled={busy} className={quietBtn} onClick={() => act(() => crewApi.stop(task.id))}><Square className="h-3.5 w-3.5" /> Stop</button>}
          {landable && (
            <SureButton
              icon={<GitMerge className="h-3.5 w-3.5" />} busy={busy}
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
        <ErrorLine text={error} />
      </div>

      <div className="flex shrink-0 gap-1 border-b border-zinc-800 px-3 pt-1.5 text-xs">
        {([['activity', 'Activity', MessageSquare], ...(ship ? [['changes', 'Changes', FileDiff]] : []), ...(task.report ? [['report', 'Report', Ship]] : [])] as [typeof tab, string, typeof Ship][]).map(([id, label, Icon]) => (
          <button key={id} onClick={() => setTab(id)} className={`inline-flex items-center gap-1.5 rounded-t-md px-2.5 py-1.5 ${tab === id ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}>
            <Icon className="h-3.5 w-3.5" /> {label}
          </button>
        ))}
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {tab === 'activity' && <Activity events={events} />}
        {tab === 'changes' && (
          <div className="min-h-0 flex-1 overflow-y-auto">
            {!diff && <p className="p-4 text-sm text-zinc-500">Reading…</p>}
            {diff?.note && <p className="px-3 pt-3 text-xs text-zinc-500">{diff.note}</p>}
            {diff && !diff.committed && !diff.uncommitted && !diff.note && <p className="p-4 text-sm text-zinc-500">No changes.</p>}
            {diff?.committed && <DiffView text={diff.committed} />}
            {diff?.uncommitted && (
              <>
                <p className="px-3 pt-3 text-xs font-medium text-amber-700 dark:text-amber-300">Not committed yet</p>
                <DiffView text={diff.uncommitted} />
              </>
            )}
          </div>
        )}
        {tab === 'report' && <pre className="min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap break-words p-3 font-sans text-sm text-zinc-200">{task.report}</pre>}
      </div>

      {!finished && (
        <form onSubmit={send} className="flex shrink-0 items-end gap-2 border-t border-zinc-800 p-2.5">
          <textarea
            value={steer}
            onChange={(ev) => setSteer(ev.target.value)}
            onKeyDown={(ev) => { if (ev.key === 'Enter' && !ev.shiftKey) send(ev) }}
            rows={Math.min(5, steer.split('\n').length)}
            placeholder={task.status === 'waiting' ? 'Your answer…' : live ? 'Tell it something, for when this turn ends…' : 'Tell it to go on…'}
            aria-label="Tell the agent"
            className={`${fieldClass} resize-none`}
          />
          <button type="submit" disabled={busy || !steer.trim()} className={primaryBtn} aria-label="Send"><Send className="h-4 w-4" /></button>
        </form>
      )}
    </div>
  )
}

// ── A new task ───────────────────────────────────────────────────────────

function NewTask({ me, onCancel, onMade, onLogins }: {
  me: CrewMe; onCancel: () => void; onMade: (id: string) => void; onLogins: () => void
}) {
  const usable = useMemo(() => (me.harnesses ?? []).filter((h) => h.installed && h.login.ok), [me.harnesses])
  const [harness, setHarness] = useState(usable[0]?.id ?? '')
  const [brief, setBrief] = useState('')
  const [kind, setKind] = useState<'ship' | 'scout'>('ship')
  const [deliver, setDeliver] = useState('branch')
  const [where, setWhere] = useState<'project' | 'url' | 'path'>('project')
  const [repos, setRepos] = useState<CrewRepos | null>(null)
  const [project, setProject] = useState('')
  const [folder, setFolder] = useState('')
  const [url, setUrl] = useState('')
  const [path, setPath] = useState('')
  const [model, setModel] = useState('')
  const [effort, setEffort] = useState('')
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

  return (
    <form onSubmit={submit} className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
      <div className="flex items-center gap-2">
        <button type="button" onClick={onCancel} className={libraryIconBtn} aria-label="Back"><ArrowLeft className="h-4 w-4" /></button>
        <h3 className="text-sm font-semibold text-zinc-100">New task</h3>
      </div>

      {!usable.length && (
        <p className="rounded-md bg-amber-500/10 px-3 py-2 text-sm text-amber-800 dark:text-amber-200">
          No coding agent is ready yet. <button type="button" onClick={onLogins} className="underline">Add a login</button>{' '}
          or ask an admin to install one.
        </p>
      )}

      <label className="block space-y-1 text-xs text-zinc-400">
        <span>Who does it</span>
        <select value={harness} onChange={(e) => setHarness(e.target.value)} className={fieldClass}>
          {(me.harnesses ?? []).map((h) => (
            <option key={h.id} value={h.id} disabled={!h.installed || !h.login.ok}>
              {h.label}{!h.installed ? ' (not installed)' : !h.login.ok ? ' (no login yet)' : ''}
            </option>
          ))}
        </select>
      </label>

      <label className="block space-y-1 text-xs text-zinc-400">
        <span>What to do</span>
        <textarea
          value={brief} onChange={(e) => setBrief(e.target.value)} rows={6} required
          placeholder="The whole task. It sees only this: the goal, what done means, anything already decided."
          className={`${fieldClass} resize-y`}
        />
      </label>

      <fieldset className="space-y-1.5">
        <legend className="text-xs text-zinc-400">Where</legend>
        <div className="flex flex-wrap gap-1.5">
          {([['project', 'A project'], ['url', 'A repository address'], ...(me.trusted ? [['path', 'A folder on this computer']] : [])] as [typeof where, string][]).map(([id, label]) => (
            <button key={id} type="button" onClick={() => setWhere(id)} className={`rounded-md border px-2.5 py-1 text-xs ${where === id ? 'border-teal-500 bg-teal-500/10 text-teal-700 dark:text-teal-300' : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`}>{label}</button>
          ))}
        </div>
        {where === 'project' && (
          <div className="grid gap-2 sm:grid-cols-2">
            <input list="crew-projects" value={project} onChange={(e) => setProject(e.target.value)} placeholder="Project (a new name makes one)" aria-label="Project" className={fieldClass} />
            <datalist id="crew-projects">{repos?.projects.map((p) => <option key={p.project} value={p.project} />)}</datalist>
            <input list="crew-folders" value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="Folder (optional)" aria-label="Folder" className={fieldClass} />
            <datalist id="crew-folders">{folders.map((f) => <option key={f} value={f} />)}</datalist>
          </div>
        )}
        {where === 'url' && <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://github.com/you/project" aria-label="Repository address" className={fieldClass} />}
        {where === 'path' && <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="/path/to/a/git/repository" aria-label="Folder" className={fieldClass} />}
      </fieldset>

      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block space-y-1 text-xs text-zinc-400">
          <span>Kind</span>
          <select value={kind} onChange={(e) => setKind(e.target.value as 'ship' | 'scout')} className={fieldClass}>
            <option value="ship">Change the code</option>
            <option value="scout">Only investigate, and report</option>
          </select>
        </label>
        {kind === 'ship' && (
          <label className="block space-y-1 text-xs text-zinc-400">
            <span>When it's done</span>
            <select value={deliver} onChange={(e) => setDeliver(e.target.value)} className={fieldClass}>
              <option value="branch">Leave it on a branch for me</option>
              <option value="pr">Open a pull request</option>
            </select>
          </label>
        )}
      </div>

      <details className="text-xs text-zinc-400">
        <summary className="cursor-pointer">Model and effort</summary>
        <div className="mt-2 grid gap-2 sm:grid-cols-2">
          <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="Model (its default if empty)" aria-label="Model" className={fieldClass} />
          <input value={effort} onChange={(e) => setEffort(e.target.value)} placeholder="Effort, e.g. high" aria-label="Effort" className={fieldClass} />
        </div>
      </details>

      <ErrorLine text={error} />
      <button type="submit" disabled={busy || !harness || !brief.trim() || !named} className={primaryBtn}>
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
  { name: 'anthropic', label: 'Anthropic key', hint: 'For OpenCode and Pi.', placeholder: 'sk-ant-…' },
  { name: 'openai', label: 'OpenAI key', hint: 'For OpenCode and Pi.', placeholder: 'sk-…' },
  { name: 'openrouter', label: 'OpenRouter key', hint: 'For OpenCode and Pi.', placeholder: 'sk-or-…' },
  { name: 'gemini', label: 'Google Gemini key', hint: 'For OpenCode and Pi.', placeholder: 'AI…' },
  { name: 'xai', label: 'xAI key', hint: 'For OpenCode and Pi.', placeholder: 'xai-…' },
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
    <div className="space-y-1.5 rounded-lg border border-zinc-800 p-3">
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

function HarnessLine({ h }: { h: CrewHarness }) {
  return (
    <li className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
      <span className="font-medium text-zinc-200">{h.label}</span>
      {!h.installed ? (
        <span className="text-xs text-amber-700 dark:text-amber-300">not installed. An admin can run: <code className="rounded bg-zinc-800 px-1 py-0.5 text-[11px]">{h.install}</code></span>
      ) : h.login.ok ? (
        <span className="text-xs text-zinc-500">ready, on {h.login.how}{h.version ? ` · ${h.version}` : ''}</span>
      ) : (
        <span className="text-xs text-amber-700 dark:text-amber-300">needs a login</span>
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
      <ul className="space-y-1 rounded-lg border border-zinc-800 p-3">{(me.harnesses ?? []).map((h) => <HarnessLine key={h.id} h={h} />)}</ul>
      <p className="text-xs text-zinc-500">
        {me.trusted
          ? 'Without a saved login, your agents use the ones already on this computer.'
          : 'Your agents run on your own accounts. Keys from "Your own AI keys" count too.'}
        {' '}What you save here is stored encrypted and never shown again.
      </p>
      {LOGINS.map((l) => <LoginRow key={l.name} login={l} saved={me.logins?.[l.name]} onChanged={onChanged} />)}
    </div>
  )
}

// ── The board ────────────────────────────────────────────────────────────

export function CrewBoard({ phone = false, onClose }: { phone?: boolean; onClose?: () => void }) {
  const [me, setMe] = useState<CrewMe | null>(null)
  const [tasks, setTasks] = useState<CrewTask[] | null>(null)
  const [selected, setSelected] = useState('')
  const [pane, setPane] = useState<'task' | 'new' | 'logins'>('task')
  const [error, setError] = useState('')

  const loadMe = useCallback(() => crewApi.me().then(setMe).catch((e) => setError(errorOf(e))), [])
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
    const t = setInterval(loadTasks, POLL_LIST_MS)
    return () => clearInterval(t)
  }, [loadTasks, me])

  const open = (id: string) => { setSelected(id); setPane('task') }
  const showList = !phone || (pane === 'task' && !selected)
  const showPane = !phone || pane !== 'task' || !!selected

  if (me && !me.allowed) {
    return (
      <div className="flex h-full flex-col">
        <Header onClose={onClose} />
        <div className="flex flex-1 items-center justify-center p-6 text-center text-sm text-zinc-400">
          {me.offered ? 'An admin needs to approve Crew for your account first.' : 'Crew is turned off on this Flight Deck.'}
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-zinc-900">
      <Header onClose={onClose} onRefresh={() => { loadTasks(); loadMe() }} onLogins={() => setPane('logins')} onNew={() => setPane('new')} />
      {error && <div className="p-2"><ErrorLine text={error} /></div>}
      <div className="flex min-h-0 flex-1">
        {showList && (
          <div className={`min-h-0 overflow-y-auto ${phone ? 'flex-1' : 'w-80 shrink-0 border-r border-zinc-800'}`}>
            {tasks === null && <p className="p-4 text-sm text-zinc-500">Loading…</p>}
            {tasks?.length === 0 && (
              <div className="space-y-2 p-5 text-sm text-zinc-400">
                <p>No tasks yet.</p>
                <p className="text-xs text-zinc-500">Start one here, or ask your agent in chat: "have Claude Code fix the login test".</p>
              </div>
            )}
            {tasks?.map((t) => (
              <button
                key={t.id}
                onClick={() => open(t.id)}
                className={`block w-full space-y-1 border-b border-zinc-800/70 px-3 py-2.5 text-left transition-colors hover:bg-zinc-800/50 ${selected === t.id && pane === 'task' ? 'bg-zinc-800/70' : ''}`}
              >
                <span className="line-clamp-2 break-words text-sm text-zinc-200">{t.title}</span>
                <span className="flex items-center gap-2">
                  <StatusPill task={t} />
                  <span className="min-w-0 truncate text-xs text-zinc-500">{t.harness_label} · {t.repo_label}</span>
                </span>
              </button>
            ))}
          </div>
        )}
        {showPane && (
          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            {pane === 'new' && me && <NewTask me={me} onCancel={() => setPane('task')} onLogins={() => setPane('logins')} onMade={(id) => { loadTasks(); open(id) }} />}
            {pane === 'logins' && me && <Logins me={me} onBack={() => setPane('task')} onChanged={loadMe} />}
            {pane === 'task' && (selected ? (
              <TaskView key={selected} taskId={selected} onBack={phone ? () => setSelected('') : undefined} onChanged={loadTasks} />
            ) : (
              <div className="flex flex-1 items-center justify-center p-6 text-sm text-zinc-500">Choose a task, or start a new one.</div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Header({ onClose, onRefresh, onLogins, onNew }: {
  onClose?: () => void; onRefresh?: () => void; onLogins?: () => void; onNew?: () => void
}) {
  return (
    <div className="flex shrink-0 items-center gap-1 border-b border-zinc-800 px-3 py-2">
      <Ship className="h-4 w-4 text-teal-600 dark:text-teal-400" />
      <h2 className="mr-auto pl-1 text-sm font-semibold text-zinc-100">Crew</h2>
      {onNew && <button onClick={onNew} className={primaryBtn}><Plus className="h-4 w-4" /> New task</button>}
      {onLogins && <button onClick={onLogins} className={libraryIconBtn} title="Logins" aria-label="Logins"><KeyRound className="h-4 w-4" /></button>}
      {onRefresh && <button onClick={onRefresh} className={libraryIconBtn} title="Refresh" aria-label="Refresh"><RefreshCw className="h-4 w-4" /></button>}
      <FullscreenButton />
      {onClose && <button onClick={onClose} className={libraryIconBtn} title="Close" aria-label="Close"><X className="h-4 w-4" /></button>}
    </div>
  )
}
