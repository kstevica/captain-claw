import { useMemo, useState, type ReactNode } from 'react'
import { AlertCircle, Clock, Compass, Coffee, Moon, Sparkles } from 'lucide-react'
import { AgentPicture } from '../agents/AgentPicture'
import { useActivityStore } from '../../stores/activityStore'
import { useAuthStore } from '../../stores/authStore'
import { useTaskboardStore } from '../../stores/taskboardStore'
import type { OpenAsk } from '../../services/asksApi'
import { resolveProposal } from '../../services/notifications'
import { taskboardApi, type TaskAction, type TaskItem } from '../../services/taskboardApi'
import type { TaskRun } from '../../services/schedulerApi'
import { ago, countWord, isoSeconds, useToday, type WaitingAction } from '../home/todayData'
import { Calm, Draft, List, Loading, Pill, Row, Section, Trouble, WaitingCard } from '../home/TodayParts'
import { allFailed, allSettled } from '../home/todayLists'

// ── Centre: no chat open — Today ──
//
// The morning briefing: a greeting and one line on what needs the person,
// the agents to talk to, then what waits for them (proposals to approve,
// Taskboard tasks that are their move, open questions), today's open work,
// and what got handled on its own (done tasks, scheduled runs) — each with
// "See what happened". Every answer here is the same call the bell and the
// Taskboard make; open questions are answered in their own window. With no
// agent at all, the caller's empty card shows instead.

/** What the home needs of an agent row (SimpleLayout's agents fit it). */
export interface HomeAgent {
  id: string
  kind: 'docker' | 'process' | 'local' | 'shared'
  name: string
  description: string
  state: 'running' | 'starting' | 'stopped' | 'unknown'
  reachable: boolean
  asleep?: boolean
  ownerName?: string
}

const isUp = (a: HomeAgent) => a.state === 'running' || a.state === 'starting' || !!a.asleep

function greeting(): string {
  const h = new Date().getHours()
  if (h < 5) return 'Working late'
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}

/** One thing waiting on the person. */
type Waiting =
  | { kind: 'proposal'; key: string; action: WaitingAction; task: TaskItem | null; at: number }
  | { kind: 'task'; key: string; task: TaskItem; at: number }
  | { kind: 'ask'; key: string; ask: OpenAsk; at: number }

/** One thing done without the person. */
type Handled =
  | { kind: 'task'; key: string; task: TaskItem; at: number }
  | { kind: 'run'; key: string; run: TaskRun; at: number }

/** The plain sentence for a proposal, by what it would do. */
function proposalHeadline(kind: string, who: string): string {
  if (kind === 'reply.on_origin') return `${who} wants to send this reply. Looks good?`
  if (kind === 'mail.draft') return `${who} wrote this email. Save it to your drafts?`
  if (kind === 'nudge') return `${who} wants to send this reminder. Looks good?`
  return `${who} wants to go ahead with this. Looks good?`
}

/** The plain sentence for a Taskboard task that is the person's move, and its two answers. */
function taskAsk(t: TaskItem, who: string): { headline: string; yes: [string, TaskAction]; no: [string, TaskAction] } {
  if (t.status === 'failed') return { headline: 'This one didn’t go through.', yes: ['Got it', 'ack_failed'], no: ['Dismiss', 'dismiss'] }
  if (t.flags?.probably_done) return { headline: `${who} thinks this is done. Close it?`, yes: ['Yes, it’s done', 'done'], no: ['Not yet', 'not_done'] }
  if (t.flags?.agent_declined) return { headline: `${who} couldn’t do this one — it’s over to you.`, yes: ['Mark done', 'done'], no: ['Dismiss', 'dismiss'] }
  if (t.flags?.partly_done) return { headline: `${who} got part of the way. The rest is yours.`, yes: ['Mark done', 'done'], no: ['Dismiss', 'dismiss'] }
  if (t.flags?.overdue || (t.due_at && t.due_at * 1000 < Date.now())) return { headline: 'This is past its due date.', yes: ['Mark done', 'done'], no: ['Dismiss', 'dismiss'] }
  return { headline: 'This one’s your move.', yes: ['Mark done', 'done'], no: ['Dismiss', 'dismiss'] }
}

/** Where an open task stands, in a few words. */
function standing(t: TaskItem, who: string): string {
  if (t.in_line) return t.in_line.ahead > 0 ? `Next in line with ${who}` : `${who} is on it`
  if (t.lane === 'working' || t.owes_next === 'agent') return `${who} is on it`
  if (t.lane === 'waiting' || t.owes_next === 'them') return 'Waiting to hear back'
  return 'Open'
}

function Mark({ tone }: { tone: 'working' | 'waiting' | 'open' }) {
  const cls = tone === 'working' ? 'bg-violet-500 dark:bg-violet-400' : tone === 'waiting' ? 'bg-sky-500 dark:bg-sky-400' : 'bg-zinc-500'
  return <span className={`h-2 w-2 rounded-full ${cls}`} />
}

/** What happened, for the expander: the words as the agent left them. */
function Happened({ lines, result, actions }: { lines: string[]; result: string; actions: ReactNode }) {
  return (
    <div className="rounded-xl bg-zinc-950/70 px-4 py-3 text-sm">
      {result
        ? <p className="max-h-56 overflow-y-auto whitespace-pre-wrap break-words leading-relaxed text-zinc-300">{result}</p>
        : <p className="italic text-zinc-500">No summary was left for this one.</p>}
      {lines.length > 0 && <p className="mt-2 text-xs text-zinc-500">{lines.join(' · ')}</p>}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">{actions}</div>
    </div>
  )
}

const linkCls = 'font-medium text-violet-600 transition-colors hover:text-violet-500 dark:text-violet-400 dark:hover:text-violet-300'

export function AgentHome<A extends HomeAgent>({ agents, sessionInfo, booting, switching, onOpen, onGuide, empty }: {
  agents: A[]
  sessionInfo: Map<string, { busy: boolean; unread: boolean }>
  booting: Set<string>
  switching: Set<string>
  onOpen: (a: A) => void
  locked?: boolean
  /** Kiosk only: open the Getting started guide. */
  onGuide?: () => void
  /** Shown instead when there is no agent at all. */
  empty: ReactNode
}) {
  const user = useAuthStore((s) => s.user)
  const showTask = useTaskboardStore((s) => s.showTask)
  const fetchSummary = useTaskboardStore((s) => s.fetchSummary)
  const openActivity = useActivityStore((s) => s.openActivity)
  const { data, reload } = useToday()
  // Answered here: gone at once, before the next read confirms it.
  const [answered, setAnswered] = useState<Set<string>>(() => new Set())
  const [busy, setBusy] = useState<{ key: string; which: 'yes' | 'no' } | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})

  const shown = useMemo(() => agents.filter(isUp), [agents])

  // An agent's name by its slug (a process agent's id is `proc-<slug>`).
  const nameOf = useMemo(() => {
    const names = new Map<string, string>()
    for (const a of agents) {
      names.set(a.id, a.name)
      if (a.id.startsWith('proc-')) names.set(a.id.slice(5), a.name)
    }
    return (slug: string, fallback = '') => names.get(slug) || fallback || 'Captain'
  }, [agents])

  const waiting = useMemo<Waiting[]>(() => {
    const out: Waiting[] = []
    const tasks = data.needsMe.data ?? []
    const used = new Set<string>()
    for (const ask of data.asks.data ?? []) out.push({ kind: 'ask', key: `ask:${ask.id}`, ask, at: ask.created })
    for (const action of data.proposals.data ?? []) {
      const task = tasks.find((t) => t.proposal?.id === action.id || t.id === action.ref_id || t.id === action.payload?.task_id) ?? null
      if (task) used.add(task.id)
      out.push({ kind: 'proposal', key: `proposal:${action.id}`, action, task, at: isoSeconds(action.created_at) ?? 0 })
    }
    for (const task of tasks) {
      if (used.has(task.id)) continue
      // Its proposal waits but the proposals didn't load: the task's own copy of it.
      if (task.proposal?.status === 'awaiting_approval' && !data.proposals.data) {
        const p = task.proposal
        out.push({
          kind: 'proposal', key: `proposal:${p.id}`, task, at: task.last_event_at ?? task.updated_at,
          action: { id: p.id, source: p.source ?? '', kind: p.kind ?? '', title: p.title, target: task.agent_slug, ref_id: task.id, created_at: '', payload: p.payload ?? {} },
        })
        continue
      }
      out.push({ kind: 'task', key: `task:${task.id}`, task, at: task.last_event_at ?? task.updated_at })
    }
    return out.filter((w) => !answered.has(w.key))
  }, [data.asks.data, data.proposals.data, data.needsMe.data, answered])

  const priorities = useMemo(() => (data.open.data ?? []).filter((t) => !answered.has(`task:${t.id}`)), [data.open.data, answered])

  const handled = useMemo<Handled[]>(() => {
    const runs = data.runs.data
    const out: Handled[] = []
    for (const run of runs ?? []) out.push({ kind: 'run', key: `run:${run.id}`, run, at: run.finished_at ?? run.started_at })
    for (const task of data.done.data ?? []) {
      // A scheduled run is on the board too: shown once, as its run.
      if (runs && task.kind === 'scheduled') continue
      out.push({ kind: 'task', key: `done:${task.id}`, task, at: task.closed_at ?? task.updated_at })
    }
    return out.sort((a, b) => b.at - a.at).slice(0, 8)
  }, [data.runs.data, data.done.data])

  if (agents.length === 0) return <>{empty}</>

  const firstName = (user?.display_name || '').trim().split(/\s+/)[0]
  // A section shows what it has (or that it's empty) once every one of its
  // sources has answered or failed: one still loading may hold the very thing.
  const waitingSources = [data.needsMe, data.proposals, data.asks]
  const waitingFailed = allFailed(...waitingSources)
  const waitingLoaded = allSettled(...waitingSources) && !waitingFailed
  const handledFailed = allFailed(data.done, data.runs)
  const handledLoaded = allSettled(data.done, data.runs) && !handledFailed
  const handledToday = handled.filter((h) => h.kind === 'task' || h.run.status === 'ok').length

  const summary = !waitingLoaded
    ? waitingFailed ? 'Couldn’t catch up on everything just now — the rest of your day is below.' : 'Catching up on your day…'
    : waiting.length > 0
      ? `${countWord(waiting.length)} ${waiting.length === 1 ? 'thing needs' : 'things need'} a quick look from you${handledToday ? ` — and ${countWord(handledToday).toLowerCase()} already handled.` : '. Everything else is moving.'}`
      : handledToday
        ? `Nothing needs you right now. ${countWord(handledToday)} ${handledToday === 1 ? 'thing' : 'things'} handled since yesterday.`
        : 'Nothing needs you right now. Enjoy the quiet.'

  const today = new Date().toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long' })

  // One answer to one waiting thing: the call, then the board's count and a fresh read.
  const answer = async (key: string, which: 'yes' | 'no', call: () => Promise<unknown>) => {
    setBusy({ key, which })
    setErrors((now) => {
      const rest = { ...now }
      delete rest[key]
      return rest
    })
    try {
      await call()
      setAnswered((s) => new Set(s).add(key))
      void fetchSummary()
      void reload()
    } catch (e) {
      setErrors((now) => ({ ...now, [key]: e instanceof Error ? e.message : String(e) }))
    } finally {
      setBusy(null)
    }
  }
  const pill = (key: string, which: 'yes' | 'no', label: string, call: () => Promise<unknown>) => (
    <Pill key={which} tone={which === 'yes' ? 'primary' : 'quiet'} busy={busy?.key === key && busy.which === which}
      disabled={busy !== null} onClick={() => void answer(key, which, call)}>
      {label}
    </Pill>
  )

  const waitingCard = (w: Waiting) => {
    if (w.kind === 'ask') {
      // Answered in its own window (PersonAsks): it opens over every screen
      // and stays in front until answered, so a button here could never be
      // reached. The card only says it's there, and goes once it's answered.
      const { ask } = w
      const who = ask.agent_name || ask.title || 'Captain'
      const first = ask.questions[0]
      return (
        <WaitingCard key={w.key} who={who} when={ago(w.at)}
          headline={`${who} has a question for you.`}
          title={first ? `${first.question}${ask.questions.length > 1 ? ` (and ${ask.questions.length - 1} more)` : ''}` : ask.title}
          actions={<span className="text-sm text-zinc-500">Answer it in the question window on this screen.</span>} />
      )
    }
    if (w.kind === 'proposal') {
      const { action, task } = w
      const who = nameOf(action.target, task?.agent_name)
      const p = action.payload ?? {}
      const to = p.to_options?.find((o) => o.address === p.to)?.display || p.to || ''
      const text = p.text || ''
      const taskRef = task?.id ?? `autonomy:${action.id}`
      return (
        <WaitingCard key={w.key} who={who} when={ago(w.at)} error={errors[w.key]}
          headline={proposalHeadline(action.kind, who)}
          title={task?.title || action.title}
          actions={[
            // Approve only with its words in sight: a proposal without them is read on its task.
            text
              ? pill(w.key, 'yes', 'Approve', () => resolveProposal(action.id, true))
              : <Pill key="review" tone="primary" onClick={() => showTask(taskRef)}>Review</Pill>,
            pill(w.key, 'no', 'Dismiss', () => resolveProposal(action.id, false)),
            action.source === 'taskboard' && text
              ? <Pill key="edit" onClick={() => showTask(taskRef)} title="Change the wording or who it goes to, on its task">Edit first</Pill>
              : null,
          ]}>
          {text && <Draft text={text} to={to} />}
        </WaitingCard>
      )
    }
    const { task } = w
    const who = task.agent_name || nameOf(task.agent_slug)
    const q = taskAsk(task, who)
    const context = task.flags?.blocker || task.flags?.why || task.ask
    return (
      <WaitingCard key={w.key} who={who} when={ago(w.at)} error={errors[w.key]}
        headline={q.headline}
        title={task.title}
        actions={[
          pill(w.key, 'yes', q.yes[0], () => taskboardApi.action(task.id, q.yes[1])),
          pill(w.key, 'no', q.no[0], () => taskboardApi.action(task.id, q.no[1])),
          <Pill key="open" onClick={() => showTask(task.id)}>Details</Pill>,
        ]}>
        {context && context !== task.title && <p className="mt-2 line-clamp-2 text-sm text-zinc-500">{context}</p>}
      </WaitingCard>
    )
  }

  const handledRow = (h: Handled) => {
    if (h.kind === 'run') {
      const { run } = h
      const ok = run.status === 'ok'
      const who = nameOf(run.agent_slug)
      const sent = run.deliveries.filter((d) => d.ok).map((d) => d.via === 'chat' ? 'its chat' : d.via)
      return (
        <Row key={h.key} title={run.job_name || 'Scheduled task'} aside={ago(h.at)}
          mark={ok ? <Clock className="h-4 w-4 text-emerald-600 dark:text-emerald-400" /> : <AlertCircle className="h-4 w-4 text-amber-600 dark:text-amber-400" />}
          sub={ok ? `${who} ran it on schedule` : `${who} ran it, but it didn’t finish`}
          detail={<Happened result={run.result}
            lines={[run.trigger === 'manual' ? 'Run by hand' : 'Ran on schedule', sent.length ? `Sent to ${sent.join(', ')}` : '']
              .filter(Boolean)}
            actions={<>
              <button className={linkCls} onClick={() => showTask(`sched:job:${run.job_id}`)}>Open in Tasks</button>
              <button className={linkCls} onClick={() => openActivity({ slug: run.agent_slug })}>Open the activity log</button>
            </>} />} />
      )
    }
    const { task } = h
    const who = task.agent_name || nameOf(task.agent_slug)
    return (
      <Row key={h.key} title={task.title} aside={ago(h.at)}
        mark={<Sparkles className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />}
        sub={task.closed_by === 'user' ? 'You closed it' : `${who} took care of it`}
        detail={<Happened result={task.result_text}
          lines={[task.ask && task.ask !== task.title ? `Asked: ${task.ask}` : '', task.channel ? `Came in on ${task.channel}` : ''].filter(Boolean)}
          actions={<>
            <button className={linkCls} onClick={() => showTask(task.id)}>Open in Tasks</button>
            {task.agent_slug && <button className={linkCls} onClick={() => openActivity({ slug: task.agent_slug })}>Open the activity log</button>}
          </>} />} />
    )
  }

  // The agents to talk to: a pill each, the way into a chat.
  const crewPill = (a: A) => {
    const info = sessionInfo.get(a.id)
    const starting = (booting.has(a.id) && a.state !== 'stopped') || switching.has(a.id) || a.state === 'starting'
    const openable = !switching.has(a.id) && ((a.reachable && !starting) || sessionInfo.has(a.id))
    const resting = !!a.asleep && !starting && !info?.busy && !info?.unread
    const state = starting ? 'Starting' : info?.busy ? 'Working' : info?.unread ? 'New reply' : a.asleep ? 'Asleep' : ''
    const dot = starting || info?.busy ? 'bg-amber-500 dark:bg-amber-400 animate-pulse' : info?.unread ? 'bg-sky-500 dark:bg-sky-400' : a.asleep ? '' : 'bg-emerald-500 dark:bg-emerald-400'
    return (
      <li key={a.id}>
        <button onClick={() => onOpen(a)} disabled={!openable}
          title={openable ? (a.asleep ? `Chat with ${a.name} — it wakes when you open it` : `Chat with ${a.name}`) : `${a.name} — ${state || 'not reachable yet'}`}
          className="group flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/60 py-1 pl-1 pr-3.5 text-sm text-zinc-300 transition-colors hover:border-zinc-700 hover:text-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/50 disabled:cursor-default disabled:opacity-60">
          <span className="relative flex h-7 w-7 shrink-0 items-center justify-center overflow-hidden rounded-full bg-zinc-800 text-[11px] font-semibold text-zinc-400">
            <AgentPicture agentId={a.id} name={a.name} size="sm" />
          </span>
          <span className="max-w-[12rem] truncate font-medium">{a.name}</span>
          {resting
            ? <Moon className="h-3 w-3 text-indigo-500 dark:text-indigo-400" aria-label="Asleep" />
            : <span className={`h-1.5 w-1.5 rounded-full ${dot}`} aria-label={state || 'Ready'} />}
        </button>
      </li>
    )
  }

  const off = agents.length - shown.length

  return (
    <div className="h-full overflow-y-auto bg-zinc-950/80">
      <div className="mx-auto w-full max-w-2xl px-6 pb-16 pt-12 wide:pt-16">
        <header className="fd-rise">
          <p className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-500">{today}</p>
          <h1 className="mt-3 text-3xl font-semibold tracking-tight text-zinc-100 wide:text-4xl">
            {greeting()}{firstName ? `, ${firstName}` : ''}.
          </h1>
          <p className="mt-3 text-[15px] leading-relaxed text-zinc-400" aria-live="polite">{summary}</p>

          {shown.length > 0 ? (
            <ul className="mt-6 flex flex-wrap gap-2" aria-label="Start a conversation">
              {shown.map(crewPill)}
            </ul>
          ) : (
            <p className="mt-6 text-sm text-zinc-500">None of your agents is on — switch one on from the list on the left to chat.</p>
          )}
        </header>

        <Section title="Waiting for you" count={waiting.length} delay={40}
          hint={waiting.length ? 'A yes or no from you and Captain takes it from there.' : undefined}>
          {!waitingLoaded
            ? waitingFailed ? <Trouble what="what’s waiting for you" error={data.needsMe.error} onRetry={() => void reload()} /> : <Loading />
            : waiting.length === 0
              ? <Calm icon={<Coffee className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />}>Nothing waiting — Captain’s on it.</Calm>
              : <ul className="space-y-3">{waiting.map(waitingCard)}</ul>}
          {waitingLoaded && (data.needsMe.error || data.proposals.error || data.asks.error) && (
            <p className="mt-2 text-xs text-zinc-500">Some of it couldn’t load just now — this list may be missing a thing or two.</p>
          )}
        </Section>

        <Section title="Today’s priorities" count={priorities.length} delay={80}
          hint={priorities.length ? 'Open work, most pressing first.' : undefined}>
          {data.open.data === null
            ? data.open.error ? <Trouble what="today’s priorities" error={data.open.error} onRetry={() => void reload()} /> : <Loading rows={1} />
            : priorities.length === 0
              ? <Calm>A clear slate. Ask one of your agents for something and it shows up here.</Calm>
              : (
                <>
                  <List>
                    {priorities.slice(0, 6).map((t) => {
                      const who = t.agent_name || nameOf(t.agent_slug)
                      const s = standing(t, who)
                      return (
                        <Row key={t.id} title={t.title}
                          aside={t.due_at ? `Due ${new Date(t.due_at * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}` : ago(t.last_event_at ?? t.updated_at)}
                          mark={<Mark tone={s === 'Waiting to hear back' ? 'waiting' : s === 'Open' ? 'open' : 'working'} />}
                          sub={s} />
                      )
                    })}
                  </List>
                  {priorities.length > 6 && (
                    <button onClick={() => useTaskboardStore.getState().openTaskboard()} className={`mt-2 text-sm ${linkCls}`}>
                      {priorities.length - 6} more in Tasks
                    </button>
                  )}
                </>
              )}
        </Section>

        <Section title="Handled automatically" count={handled.length} delay={120}
          hint={handled.length ? 'Done without you since yesterday.' : undefined}>
          {!handledLoaded
            ? handledFailed ? <Trouble what="what got handled" error={data.done.error} onRetry={() => void reload()} /> : <Loading rows={1} />
            : handled.length === 0
              ? <Calm icon={<Sparkles className="h-4 w-4 shrink-0 text-violet-600 dark:text-violet-400" />}>Nothing yet today. What your agents finish on their own lands here.</Calm>
              : <List>{handled.map(handledRow)}</List>}
          {handledLoaded && (data.done.error || data.runs.error) && (
            <p className="mt-2 text-xs text-zinc-500">Some of it couldn’t load just now — this list may be missing a thing or two.</p>
          )}
        </Section>

        <footer className="mt-14 flex flex-col items-center gap-2 text-center text-xs text-zinc-500">
          <p>
            {off > 0 && <>{off === 1 ? 'One more agent is off: switch it on' : `${off} more agents are off: switch them on`} from the list on the left. </>}
            What they save for you lands in Library, Data and Folders, bottom left.
          </p>
          {onGuide && (
            <button onClick={onGuide} className="flex items-center gap-1.5 font-medium transition-colors hover:text-violet-600 dark:hover:text-violet-400">
              <Compass className="h-3.5 w-3.5" />
              New here? Getting started
            </button>
          )}
        </footer>
      </div>
    </div>
  )
}
