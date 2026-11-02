import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  AlertTriangle, ArrowRight, CalendarClock, Check, CheckCircle2, ChevronDown, ExternalLink, FileCode2, Hammer, MessageSquare, Plug,
  Plus, RefreshCw, Sparkles, Waypoints,
} from 'lucide-react'
import {
  forgeApi, type ForgeApp, type ForgeConnect, type ForgeConnectResult, type ForgePlan, type ForgeSchedule,
} from '../../services/agentForge'
import { crewApi, type CrewMe, type CrewModelChoices, type CrewTask } from '../../services/crewApi'
import {
  buildSchedule, describeSchedule, localTimeZone, parseSchedule, schedulerApi, type DeliveryOptions, type ScheduleForm,
} from '../../services/schedulerApi'
import type { Archetype } from '../../services/tierConfig'
import { useChatStore } from '../../stores/chatStore'
import { useCrewStore } from '../../stores/crewStore'
import { useProcessStore } from '../../stores/processStore'
import { useScheduledStore } from '../../stores/scheduledStore'
import { useUIStore } from '../../stores/uiStore'
import { Button, Card, EmptyState, Field, Notice, Segmented, Sheet, Switch, inputCls } from '../console/parts'
import { ModelChooser, NOTHING_CHOSEN, sourceOf, type ModelChoice } from '../crew/CrewModelChoice'
import { StatusPill } from '../crew/crewParts'
import { SchedulePicker } from '../scheduler/SchedulePicker'
import { useSchedulePreview } from '../scheduler/schedulePreview'
import { AgentBadge, Chip, Eyebrow, ErrorNote, Fold } from './forgeParts'
import { errorOf } from './forgeUtil'

// Step four: the team is up, and what's left is on one page, in the order you'd
// do it. Each suggestion says why and how you'd use it, and reads as a sentence;
// the detail is a click away. Nothing is made until you press the button for it.

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`

const CONNECT_LABEL: Record<ForgeConnect, string> = { both: 'API and socket', api: 'API only', wss: 'Socket only' }
const BUILT = new Set(['ready', 'pr_open', 'delivered'])
const BROKEN = new Set(['failed', 'stopped', 'closed', 'gone'])

function openAgentChat(slug: string, name: string) {
  const p = useProcessStore.getState().processes.find((x) => x.slug === slug)
  useChatStore.getState().openChat(`proc-${slug}`, p?.name || name, 'localhost', p?.web_port || 0, p?.web_auth || '')
}

export function ForgeReady({ plan, archetypes, onChanged, onClose, onAnother }: {
  plan: ForgePlan
  archetypes: Archetype[]
  onChanged: () => Promise<void>
  onClose: () => void
  onAnother: () => void
}) {
  const recs = plan.recommendations
  const schedules = recs?.schedules || []
  const apps = recs?.apps || []
  const agents = plan.created.filter((a) => a.ok)
  const missing = plan.created.filter((a) => !a.ok)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const again = async () => {
    setBusy(true)
    setErr('')
    try {
      await forgeApi.recommend(plan.id)
      await onChanged()
    } catch (e) {
      setErr(errorOf(e))
      setBusy(false)
    }
  }
  const chat = (slug: string, name: string) => { openAgentChat(slug, name); onClose() }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-4xl space-y-5 px-4 py-5 wide:px-6 wide:py-7">
        <section className="fd-rise relative overflow-hidden rounded-2xl border border-emerald-500/20 bg-gradient-to-br from-emerald-500/[0.10] via-transparent to-violet-500/[0.08] p-5 wide:p-6">
          <Eyebrow icon={CheckCircle2} tone="emerald">Your team is ready</Eyebrow>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">
            {agents.length === 1 ? `${agents[0].name} is ready to work` : `${agents.length} agents are ready to work`}
          </h2>
          {recs?.usage && <p className="mt-1 max-w-3xl whitespace-pre-wrap text-sm leading-relaxed text-zinc-400">{recs.usage}</p>}
          <ul className="mt-4 grid gap-2 wide:grid-cols-2">
            {agents.map((a, i) => (
              <li key={a.id}>
                <button onClick={() => chat(a.slug, a.name)} className="group flex w-full items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-950/60 p-2.5 text-left transition-colors hover:border-violet-500/40">
                  <AgentBadge archetype={a.archetype} archetypes={archetypes} />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium text-zinc-100">{a.name}</span>
                    <span className="block truncate text-xs text-zinc-500">{a.role}</span>
                  </span>
                  <span className={`inline-flex shrink-0 items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium transition-colors ${
                    i === 0 ? 'bg-violet-600 text-white group-hover:bg-violet-500' : 'text-zinc-400 group-hover:text-zinc-200'
                  }`}>
                    <MessageSquare className="h-3.5 w-3.5" /> Chat
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {missing.length > 0 && (
            <p className="mt-3 flex items-start gap-1.5 text-xs leading-relaxed text-amber-700 dark:text-amber-300">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{missing.map((a) => `${a.name}: ${a.error}`).join(' ')}</span>
            </p>
          )}
          {plan.team && (
            <p className="mt-3 text-xs leading-relaxed text-zinc-500">
              They share the folder <code className="rounded bg-zinc-800 px-1 py-0.5">vfs:{plan.team}/</code>{plan.files.length > 0 && <>, with your files under <code className="rounded bg-zinc-800 px-1 py-0.5">inputs/</code></>}, and one set of data tables.
            </p>
          )}
        </section>

        {plan.recommend_error ? (
          <Notice tone="amber">
            <p>Captain couldn't work out what to schedule or build: {plan.recommend_error}</p>
            <Button icon={RefreshCw} busy={busy} onClick={again} className="mt-2">Try again</Button>
            {err && <p className="mt-2 text-red-700 dark:text-red-300">{err}</p>}
          </Notice>
        ) : (
          <>
            <Schedules plan={plan} agents={agents} schedules={schedules} onChanged={onChanged} onClose={onClose} />
            <Apps plan={plan} agents={agents} apps={apps} archetypes={archetypes} onChanged={onChanged} onClose={onClose} />
          </>
        )}

        <div className="flex justify-center pb-2">
          <Button variant="ghost" icon={Plus} onClick={onAnother}>Forge another team</Button>
        </div>
      </div>
    </div>
  )
}

// ── Scheduled tasks ──

interface Draft {
  id: string
  name: string
  agent: string
  prompt: string
  delivery: ForgeSchedule['delivery']
  form: ScheduleForm
  keep: boolean
  /** Captain's schedule the scheduler couldn't read, until the person picks a time. */
  unread: string
}

function draftOf(s: ForgeSchedule): Draft {
  return {
    id: s.id, name: s.name, agent: s.agent, prompt: s.prompt, delivery: s.delivery,
    form: parseSchedule(s.schedule_problem ? '' : s.schedule, null),
    keep: !s.schedule_problem, unread: s.schedule_problem ? s.schedule : '',
  }
}

function Schedules({ plan, agents, schedules, onChanged, onClose }: {
  plan: ForgePlan; agents: ForgePlan['created']; schedules: ForgeSchedule[]; onChanged: () => Promise<void>; onClose: () => void
}) {
  const [drafts, setDrafts] = useState<Draft[]>(() => schedules.map(draftOf))
  const [open, setOpen] = useState<Set<string>>(() => new Set(schedules.filter((s) => s.schedule_problem).map((s) => s.id)))
  const [options, setOptions] = useState<DeliveryOptions | null>(null)
  const [problems, setProblems] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const tz = useMemo(() => localTimeZone(), [])

  useEffect(() => { schedulerApi.deliveryOptions().then(setOptions).catch(() => {}) }, [])

  const made = (id: string) => schedules.find((s) => s.id === id)?.job_id || ''
  const patch = (id: string, p: Partial<Draft>) => setDrafts((list) => list.map((d) => (d.id === id ? { ...d, ...p } : d)))
  const toggleOpen = (id: string) => setOpen((old) => { const n = new Set(old); if (n.has(id)) n.delete(id); else n.add(id); return n })
  const chosen = drafts.filter((d) => d.keep && !made(d.id))
  const waiting = drafts.filter((d) => !made(d.id))
  const madeCount = drafts.length - waiting.length

  // A way to send it that this person hasn't connected isn't one the task can use.
  const usable = (d: Draft): ForgeSchedule['delivery'] => {
    const out = { ...d.delivery, email: d.delivery.email && options?.email.available !== false, telegram: d.delivery.telegram && options?.telegram.available !== false }
    return out.chat || out.email || out.telegram ? out : { ...out, chat: true }
  }
  const create = async () => {
    setBusy(true)
    setErr('')
    setProblems({})
    try {
      const items = chosen.map((d) => {
        const rec = schedules.find((s) => s.id === d.id) as ForgeSchedule
        return { ...rec, name: d.name.trim() || rec.name, agent: d.agent, prompt: d.prompt, schedule: buildSchedule(d.form), delivery: usable(d) }
      })
      const { results } = await forgeApi.schedules(plan.id, items, tz)
      const failed = results.filter((r) => !r.ok)
      setProblems(Object.fromEntries(failed.map((r) => [r.id, r.error || "It couldn't be made."])))
      setOpen((old) => new Set([...old, ...failed.map((r) => r.id)]))
      await onChanged()
    } catch (e) {
      setErr(errorOf(e))
    } finally {
      setBusy(false)
    }
  }
  const openTasks = (id?: string) => {
    onClose()
    useScheduledStore.getState().openTasks(id)
    useUIStore.getState().setView('scheduler')
  }

  if (!schedules.length) {
    return (
      <Card className="fd-rise" icon={CalendarClock} title="What runs by itself">
        <EmptyState icon={CalendarClock} title="Nothing needs a schedule">
          Your agents work when you ask them. You can add a task any time, under Scheduled tasks.
        </EmptyState>
      </Card>
    )
  }
  return (
    <Card
      className="fd-rise"
      icon={CalendarClock}
      title="What runs by itself"
      hint={madeCount === drafts.length ? 'All set. Change them any time under Scheduled tasks.' : 'Captain suggests these. Keep the ones you want.'}
      action={madeCount > 0 ? <Button variant="ghost" icon={ExternalLink} className="!px-2.5 !py-1 !text-xs" onClick={() => openTasks()}>Scheduled tasks</Button> : undefined}
      bodyClassName=""
    >
      <ul className="divide-y divide-zinc-800/70">
        {drafts.map((d) => (
          <ScheduleRow
            key={d.id}
            draft={d}
            why={schedules.find((s) => s.id === d.id)?.why || ''}
            usage={schedules.find((s) => s.id === d.id)?.usage || ''}
            jobId={made(d.id)}
            problem={problems[d.id] || ''}
            agents={agents}
            options={options}
            tz={tz}
            open={open.has(d.id)}
            disabled={busy}
            onToggle={() => toggleOpen(d.id)}
            onChange={(p) => patch(d.id, p)}
            onOpenTask={() => openTasks(made(d.id))}
          />
        ))}
      </ul>
      {waiting.length > 0 && (
        <div className="flex flex-col gap-2 border-t border-zinc-800/80 px-5 py-3 wide:flex-row wide:items-center wide:justify-between">
          <p className="text-xs text-zinc-500">{chosen.length ? `Keeping ${chosen.length} of ${waiting.length}. You can change them any time.` : 'None kept yet.'}</p>
          {err && <ErrorNote>{err}</ErrorNote>}
          <Button variant="primary" icon={CalendarClock} busy={busy} disabled={!chosen.length} onClick={create}>
            {chosen.length ? `Create ${plural(chosen.length, 'task')}` : 'Create tasks'}
          </Button>
        </div>
      )}
    </Card>
  )
}

function ScheduleRow({ draft, why, usage, jobId, problem, agents, options, tz, open, disabled, onToggle, onChange, onOpenTask }: {
  draft: Draft; why: string; usage: string; jobId: string; problem: string; agents: ForgePlan['created']; options: DeliveryOptions | null; tz: string
  open: boolean; disabled: boolean; onToggle: () => void; onChange: (p: Partial<Draft>) => void; onOpenTask: () => void
}) {
  const schedule = buildSchedule(draft.form)
  const agentName = agents.find((a) => a.slug === draft.agent)?.name || draft.agent
  const sends = [draft.delivery.chat && 'chat', draft.delivery.email && 'email', draft.delivery.telegram && 'Telegram'].filter(Boolean).join(', ')
  return (
    <li className={`px-5 py-4 transition-opacity ${!jobId && !draft.keep && !open ? 'opacity-60' : ''}`}>
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex w-9 shrink-0 justify-center">
          {jobId
            ? <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-700 dark:text-emerald-300"><Check className="h-3 w-3" /></span>
            : <Switch on={draft.keep} busy={disabled} label={`Keep ${draft.name}`} onChange={(keep) => onChange({ keep })} />}
        </span>
        <button type="button" onClick={onToggle} aria-expanded={open} className="group min-w-0 flex-1 text-left">
          <span className="flex items-center gap-2">
            <span className="min-w-0 truncate text-sm font-medium text-zinc-100">{draft.name}</span>
            <ChevronDown className={`h-3.5 w-3.5 shrink-0 text-zinc-600 transition-transform group-hover:text-zinc-300 ${open ? '' : '-rotate-90'}`} />
          </span>
          <span className="mt-0.5 block text-xs font-medium text-violet-700 dark:text-violet-300">
            {draft.unread ? 'Pick a time' : describeSchedule(schedule)} · {agentName}{sends ? ` · to ${sends}` : ''}
          </span>
          {why && <span className="mt-1 block text-xs leading-relaxed text-zinc-400">{why}</span>}
          {usage && <span className="mt-0.5 block text-xs leading-relaxed text-zinc-500">{usage}</span>}
        </button>
      </div>
      {jobId && (
        <div className="mt-1 pl-12"><button onClick={onOpenTask} className="text-xs font-medium text-emerald-700 hover:underline dark:text-emerald-300">Scheduled. See it under Scheduled tasks</button></div>
      )}
      {problem && !jobId && <div className="mt-2 pl-12"><ErrorNote>{problem}</ErrorNote></div>}
      {open && !jobId && (
        <ScheduleEditor draft={draft} agents={agents} options={options} tz={tz} disabled={disabled} onChange={onChange} />
      )}
    </li>
  )
}

function ScheduleEditor({ draft, agents, options, tz, disabled, onChange }: {
  draft: Draft; agents: ForgePlan['created']; options: DeliveryOptions | null; tz: string; disabled: boolean; onChange: (p: Partial<Draft>) => void
}) {
  const preview = useSchedulePreview(buildSchedule(draft.form), tz)
  const setDelivery = (key: 'chat' | 'email' | 'telegram') => onChange({ delivery: { ...draft.delivery, [key]: !draft.delivery[key] } })
  return (
    <div className="fd-rise mt-3 space-y-3 rounded-xl border border-zinc-800 bg-zinc-950/50 p-3 wide:ml-12">
      {draft.unread && (
        <p className="text-xs leading-relaxed text-amber-700 dark:text-amber-300">Captain suggested “{draft.unread}”, which the scheduler can't do. Pick the closest time and it'll be kept.</p>
      )}
      <Field label="When">
        <SchedulePicker
          form={draft.form}
          preview={preview}
          tz={tz}
          onChange={(update) => onChange({ form: update(draft.form), unread: '', keep: draft.unread ? true : draft.keep })}
        />
      </Field>
      <div className="grid gap-3 wide:grid-cols-2">
        <Field label="Name">
          <input value={draft.name} onChange={(e) => onChange({ name: e.target.value })} disabled={disabled} className={inputCls} />
        </Field>
        <Field label="Who does it">
          <select value={draft.agent} onChange={(e) => onChange({ agent: e.target.value })} disabled={disabled} className={inputCls}>
            {agents.map((a) => <option key={a.slug} value={a.slug}>{a.name}</option>)}
          </select>
        </Field>
      </div>
      <Field label="Where the result goes" hint={options && (!options.email.available || !options.telegram.available) ? 'Connect your mail or Telegram under Connections to use them here.' : undefined}>
        <div className="flex flex-wrap gap-1.5">
          <Chip on={draft.delivery.chat} disabled={disabled} onClick={() => setDelivery('chat')}>Chat</Chip>
          <Chip on={draft.delivery.email && options?.email.available !== false} disabled={disabled || options?.email.available === false} onClick={() => setDelivery('email')}>Email</Chip>
          <Chip on={draft.delivery.telegram && options?.telegram.available !== false} disabled={disabled || options?.telegram.available === false} onClick={() => setDelivery('telegram')}>Telegram</Chip>
        </div>
      </Field>
      <Field label="What the agent is told each time">
        <textarea value={draft.prompt} onChange={(e) => onChange({ prompt: e.target.value })} disabled={disabled} rows={4} className={`${inputCls} resize-y leading-relaxed`} />
      </Field>
    </div>
  )
}

// ── Apps ──

function Apps({ plan, agents, apps, archetypes, onChanged, onClose }: {
  plan: ForgePlan; agents: ForgePlan['created']; apps: ForgeApp[]; archetypes: Archetype[]; onChanged: () => Promise<void>; onClose: () => void
}) {
  const [me, setMe] = useState<CrewMe | null>(null)
  const [choices, setChoices] = useState<CrewModelChoices | null>(null)
  const [choicesErr, setChoicesErr] = useState('')

  useEffect(() => {
    crewApi.me().then((m) => {
      setMe(m)
      if (m.allowed) crewApi.modelChoices().then(setChoices).catch((e) => setChoicesErr(errorOf(e)))
    }).catch(() => setMe({ offered: false, allowed: false }))
  }, [])

  // While a build runs, its status follows from the plan.
  const building = apps.some((a) => a.task && ['preparing', 'queued', 'running', 'waiting'].includes(a.task.status))
  useEffect(() => {
    if (!building) return
    const t = window.setInterval(() => { void onChanged() }, 4000)
    return () => window.clearInterval(t)
  }, [building, onChanged])

  if (!apps.length) {
    return (
      <Card className="fd-rise" icon={Waypoints} title="Software to build">
        <EmptyState icon={Sparkles} title="No software needed">
          Your agents' own tools cover this. If that changes, give Crew a task from the Crew page.
        </EmptyState>
      </Card>
    )
  }
  return (
    <Card
      className="fd-rise"
      icon={Waypoints}
      title="Software to build"
      hint="Crew's coding agents build these, for your agents first: an API they call and a socket where the app hands them work."
      bodyClassName="space-y-3 p-4 wide:p-5"
    >
      {me && !me.allowed && (
        <Notice tone="amber">{me.offered ? "Crew isn't approved for your account yet, so these can't be built from here. An admin can approve it on the Users page." : "Crew is turned off on this deck, so these can't be built from here."}</Notice>
      )}
      {apps.map((a) => <AppCard key={a.id} plan={plan} agents={agents} app={a} archetypes={archetypes} me={me} choices={choices} choicesErr={choicesErr} onChanged={onChanged} onClose={onClose} />)}
    </Card>
  )
}

/** Build, then connect: where this app has got to. */
function Track({ built, active, connected }: { built: boolean; active: boolean; connected: boolean }) {
  const mark = (state: 'done' | 'now' | 'next', n: number) => (
    <span className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-semibold ${
      state === 'done' ? 'bg-violet-600 text-white'
        : state === 'now' ? 'bg-violet-500/15 text-violet-700 ring-1 ring-violet-500/50 dark:text-violet-300'
          : 'bg-zinc-800 text-zinc-500'
    }`}>
      {state === 'done' ? <Check className="h-3 w-3" /> : n}
    </span>
  )
  const first = built ? 'done' : 'now'
  const second = connected ? 'done' : built || active ? 'now' : 'next'
  return (
    <ol className="flex items-center gap-1.5 text-xs" aria-label="Progress">
      <li className="flex items-center gap-1.5">{mark(first, 1)}<span className={first === 'now' ? 'font-medium text-zinc-200' : 'text-zinc-500'}>Build</span></li>
      <li aria-hidden className={`h-px w-6 ${built ? 'bg-violet-500/60' : 'bg-zinc-800'}`} />
      <li className="flex items-center gap-1.5">{mark(second, 2)}<span className={second === 'now' ? 'font-medium text-zinc-200' : 'text-zinc-500'}>Connect</span></li>
    </ol>
  )
}

/** What the person does next, in a sentence. */
function nextStep(status: string, connected: boolean): string {
  const then = connected ? '' : ', then start the app and connect your agents'
  switch (status) {
    case 'preparing': case 'queued': case 'running': return "Crew is building this. You can close this window; it carries on, and its agent is told when it's done."
    case 'waiting': return 'Crew has a question for you.'
    case 'ready': return `Crew has finished. Look at what it did and land it${then}.`
    case 'pr_open': return `A pull request is open. Land it${then}.`
    case 'delivered': return connected ? 'It has landed.' : 'It has landed. Start the app, then connect your agents.'
    case 'failed': return "Crew couldn't finish. See what happened in Crew, or build it again."
    case 'gone': return "This task isn't on the Crew page any more. You can build it again."
    default: return 'This build stopped. You can build it again.'
  }
}

function AppCard({ plan, agents, app, archetypes, me, choices, choicesErr, onChanged, onClose }: {
  plan: ForgePlan; agents: ForgePlan['created']; app: ForgeApp; archetypes: Archetype[]; me: CrewMe | null; choices: CrewModelChoices | null
  choicesErr: string; onChanged: () => Promise<void>; onClose: () => void
}) {
  const usable = (me?.harnesses || []).filter((h) => h.installed)
  const [harnessId, setHarnessId] = useState('')
  const harness = usable.find((h) => h.id === harnessId) || usable.find((h) => h.login.ok) || usable[0]
  const [model, setModel] = useState('')
  const [modelChoice, setModelChoice] = useState<ModelChoice>(NOTHING_CHOSEN)
  const [brief, setBrief] = useState(app.brief)
  const [connect, setConnect] = useState<ForgeConnect>(app.connect)
  const [optionsOpen, setOptionsOpen] = useState(false)
  const [briefOpen, setBriefOpen] = useState(false)
  const [connectNow, setConnectNow] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const task = app.task
  const started = !!task && !BROKEN.has(task.status)
  const built = !!task && BUILT.has(task.status)
  const names = Object.fromEntries(agents.map((a) => [a.slug, a.name]))
  // With no login of its own the CLI has to run on something else, so that choice can't stay hidden.
  const showOptions = optionsOpen || (!!harness && !harness.login.ok)

  const build = async () => {
    if (!harness) return
    setBusy(true)
    setErr('')
    try {
      const source = sourceOf(modelChoice, choices, harness.id)
      await forgeApi.build(plan.id, app.id, { harness: harness.id, brief, connect, ...(source ? { model_source: source } : { model: model.trim() }) })
      await onChanged()
    } catch (e) {
      setErr(errorOf(e))
    } finally {
      setBusy(false)
    }
  }
  const openCrew = () => {
    onClose()
    useCrewStore.getState().openCrew()
    useUIStore.getState().setView('crew')
  }

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex w-full min-w-0 items-start gap-3 wide:w-auto wide:flex-1">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400"><FileCode2 className="h-[18px] w-[18px]" /></span>
          <div className="min-w-0">
            <h4 className="text-sm font-semibold text-zinc-100">{app.name}</h4>
            {app.summary && <p className="mt-0.5 text-sm leading-relaxed text-zinc-300">{app.summary}</p>}
            {app.why && <p className="mt-0.5 text-xs leading-relaxed text-zinc-500">{app.why}</p>}
          </div>
        </div>
        <Track built={built} active={started} connected={!!app.connection} />
      </div>

      {app.agents.length > 0 && (
        <ul className="mt-3 space-y-1.5">
          {app.agents.map((s) => {
            const a = agents.find((x) => x.slug === s)
            return (
              <li key={s} className="flex items-start gap-2 text-xs leading-relaxed text-zinc-400">
                {a && <AgentBadge archetype={a.archetype} archetypes={archetypes} size="sm" />}
                <span className="min-w-0 pt-1"><span className="font-medium text-zinc-300">{names[s] || s}</span>{app.agent_use[s] ? `: ${app.agent_use[s]}` : ''}</span>
              </li>
            )
          })}
        </ul>
      )}

      {task && (
        <div className="mt-3 flex flex-col gap-2 rounded-xl border border-zinc-800/70 p-3 wide:flex-row wide:items-center">
          {task.status === 'gone'
            ? <span className="shrink-0 text-xs font-medium text-zinc-500">Gone</span>
            : <StatusPill task={task as unknown as CrewTask} />}
          <p className="min-w-0 flex-1 text-xs leading-relaxed text-zinc-400">{nextStep(task.status, !!app.connection)}</p>
          {task.status !== 'gone' && <Button icon={ExternalLink} className="shrink-0 !px-2.5 !py-1 !text-xs" onClick={openCrew}>Open in Crew</Button>}
        </div>
      )}

      {!started && me?.allowed && (
        usable.length === 0 ? (
          <p className="mt-3 text-xs leading-relaxed text-zinc-500">No coding CLI is installed on this computer yet. An admin can install one; the Crew page says how.</p>
        ) : (
          <div className="mt-3 space-y-3">
            <Fold
              label={<>Options <span className="font-normal text-zinc-600">· {harness?.label} · {CONNECT_LABEL[connect]}</span></>}
              open={showOptions}
              onToggle={() => setOptionsOpen((v) => !v)}
            >
              <div className="space-y-3 rounded-xl border border-zinc-800 p-3">
                {usable.length > 1 && (
                  <Field label="Built by">
                    <Segmented label="Built by" value={harness?.id || ''} onChange={setHarnessId} options={usable.map((h) => ({ value: h.id, label: h.label }))} />
                  </Field>
                )}
                <Field label="How agents connect">
                  <Segmented label="How agents connect" value={connect} onChange={setConnect} options={(Object.keys(CONNECT_LABEL) as ForgeConnect[]).map((c) => ({ value: c, label: CONNECT_LABEL[c] }))} />
                </Field>
                <Field label="Model">
                  <ModelChooser
                    choice={modelChoice} choices={choices} harness={harness} own={me.own_model} loadError={choicesErr}
                    onChange={(next) => setModelChoice((was) => ({ ...was, ...next }))} onLogins={openCrew}
                  >
                    <input value={model} onChange={(e) => setModel(e.target.value)} aria-label="Its model" placeholder={harness?.default_model ? `Your default: ${harness.default_model}` : 'Its default'} className={inputCls} />
                  </ModelChooser>
                </Field>
              </div>
            </Fold>
            {err && <ErrorNote>{err}</ErrorNote>}
            <div className="flex flex-wrap gap-2">
              <Button variant="primary" icon={Hammer} busy={busy} disabled={!harness} onClick={build}>{task ? 'Build it again' : 'Build it with Crew'}</Button>
              <Button onClick={() => setBriefOpen(true)} disabled={busy}>What Crew is told</Button>
            </div>
          </div>
        )
      )}

      {(built || connectNow || !!app.connection) && <ConnectForm plan={plan} app={app} agents={agents} onChanged={onChanged} />}
      {started && !built && !connectNow && !app.connection && (
        <button onClick={() => setConnectNow(true)} className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-300">
          Is it already running? Connect your agents <ArrowRight className="h-3 w-3" />
        </button>
      )}

      {briefOpen && <BriefSheet app={app} brief={brief} onBrief={setBrief} onClose={() => setBriefOpen(false)} />}
    </div>
  )
}

function BriefSheet({ app, brief, onBrief, onClose }: { app: ForgeApp; brief: string; onBrief: (b: string) => void; onClose: () => void }) {
  const lines = (title: string, rows: ReactNode[]) => rows.length > 0 && (
    <div>
      <p className="mb-1.5 text-xs font-medium text-zinc-400">{title}</p>
      <ul className="space-y-1 text-xs leading-relaxed text-zinc-500">{rows}</ul>
    </div>
  )
  return (
    <Sheet
      wide
      title={`What Crew is told about ${app.name}`}
      subtitle="Change anything. The rules for how agents connect are added to this, and can't be edited."
      onClose={onClose}
      footer={<Button variant="primary" icon={Check} onClick={onClose}>Done</Button>}
    >
      <div className="space-y-4">
        <Field label="What to build">
          <textarea value={brief} onChange={(e) => onBrief(e.target.value)} rows={12} className={`${inputCls} resize-y leading-relaxed`} />
        </Field>
        {lines('What agents call', app.endpoints.map((e) => <li key={e.method + e.path}><code className="text-zinc-300">{e.method} {e.path}</code>{e.purpose ? `: ${e.purpose}` : ''}</li>))}
        {lines('What the app sends agents', app.events.map((e) => <li key={e.name}><code className="text-zinc-300">{e.name}</code>{e.purpose ? `: ${e.purpose}` : ''}</li>))}
      </div>
    </Sheet>
  )
}

function ConnectForm({ plan, app, agents, onChanged }: {
  plan: ForgePlan; app: ForgeApp; agents: ForgePlan['created']; onChanged: () => Promise<void>
}) {
  const [editing, setEditing] = useState(!app.connection)
  const [url, setUrl] = useState(app.connection?.url || '')
  const [key, setKey] = useState('')
  const [chosen, setChosen] = useState<Set<string>>(() => new Set(app.connection?.agents?.length ? app.connection.agents : app.agents))
  const [results, setResults] = useState<ForgeConnectResult[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const c = app.connection

  const connect = async () => {
    setBusy(true)
    setErr('')
    setResults([])
    try {
      const res = await forgeApi.connect(plan.id, app.id, { url: url.trim(), key: key.trim(), agents: [...chosen] })
      setResults(res.results)
      setKey('')
      const procs = useProcessStore.getState()
      for (const [slug, text] of Object.entries(res.instructions)) {
        void procs.setFleetInstructions(slug, text).then(() => useChatStore.getState().resendFleetIdentity(`proc-${slug}`))
      }
      if (res.results.every((r) => !r.error)) setEditing(false)
      await onChanged()
    } catch (e) {
      setErr(errorOf(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mt-3 space-y-2.5 border-t border-zinc-800/80 pt-3">
      <p className="flex items-center gap-1.5 text-sm font-medium text-zinc-200"><Plug className="h-3.5 w-3.5 text-violet-500" /> Connect your agents</p>
      {c && (
        <Notice tone="emerald">
          <span className="font-medium">{plural(c.agents.length, 'agent')} connected</span> to {c.url}
          {c.socket ? ', over its socket' : ''}{c.api_calls ? `${c.socket ? ' and' : ','} with ${plural(c.api_calls, 'API call')}` : ''}.
          {!editing && <button onClick={() => setEditing(true)} className="ml-1.5 font-medium underline">Change</button>}
        </Notice>
      )}
      {editing && (
        <>
          {!c && <p className="text-xs leading-relaxed text-zinc-500">Once the app is running, give its address and key. Crew's report says where the key is. You can publish the app from Crew, under Live sites, if your account allows it.</p>}
          <div className="grid gap-2 wide:grid-cols-2">
            <Field label="Address">
              <input value={url} onChange={(e) => setUrl(e.target.value)} disabled={busy} placeholder="https://…" className={inputCls} />
            </Field>
            <Field label="Key">
              <input value={key} onChange={(e) => setKey(e.target.value)} disabled={busy} type="password" autoComplete="off" placeholder="The app's AGENT_API_KEY" className={inputCls} />
            </Field>
          </div>
          <Field label="Who connects">
            <div className="flex flex-wrap gap-1.5">
              {agents.map((a) => (
                <Chip key={a.slug} on={chosen.has(a.slug)} disabled={busy} onClick={() => setChosen((old) => { const n = new Set(old); if (n.has(a.slug)) n.delete(a.slug); else n.add(a.slug); return n })}>
                  {a.name}
                </Chip>
              ))}
            </div>
          </Field>
          {err && <ErrorNote>{err}</ErrorNote>}
          {results.some((r) => r.error) && (
            <ul className="space-y-0.5 text-xs">
              {results.map((r) => (
                <li key={r.slug} className={r.error ? 'text-red-700 dark:text-red-300' : 'text-zinc-400'}>
                  {r.name}: {r.error || [r.linked ? 'socket linked' : '', r.api_calls ? `${plural(r.api_calls, 'API call')} registered` : ''].filter(Boolean).join(', ')}
                </li>
              ))}
            </ul>
          )}
          <Button variant="primary" icon={Plug} busy={busy} disabled={!url.trim() || !key.trim() || chosen.size === 0} onClick={connect}>Connect</Button>
        </>
      )}
    </div>
  )
}
