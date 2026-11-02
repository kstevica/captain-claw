import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, ClipboardEvent, DragEvent, ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  AlertTriangle, ArrowLeft, Bot, CalendarClock, Check, ChevronDown, ChevronRight, Hammer, HelpCircle, Loader2,
  Paperclip, Plug, RefreshCw, Sparkles, Trash2, X,
} from 'lucide-react'
import { listArchetypes } from '../../services/archetypes'
import { crewApi, type CrewMe, type CrewModelChoices } from '../../services/crewApi'
import { localTimeZone } from '../../services/schedulerApi'
import {
  forgeApi, type ForgeApp, type ForgeConnect, type ForgeConnectResult, type ForgeDraftAgent, type ForgeFeedEvent,
  type ForgePlan, type ForgePlanRow, type ForgeQuestion, type ForgeSchedule,
} from '../../services/agentForge'
import type { Archetype } from '../../services/tierConfig'
import { useChatStore } from '../../stores/chatStore'
import { useProcessStore } from '../../stores/processStore'
import { useIsMobile } from '../../hooks/useMediaQuery'
import { ModelChooser, NOTHING_CHOSEN, sourceOf, type ModelChoice } from '../crew/CrewModelChoice'

// ── Forge agents ─────────────────────────────────────────────────────
//
// The other way to make agents: instead of picking an archetype, say what you
// need and attach what you have. Captain plans a team with Vatra's planner and
// stops at the plan; you change it, answer its questions, and approve. Only then
// are the agents created. It then recommends what should run on a schedule and
// what software Crew should build for the agents to connect to, and you choose.
//
// The plan lives in Flight Deck, so closing this dialog loses nothing: planning
// goes on, the bell says when it's ready, and the dialog opens where it left off.

const MAX_FILES = 10
const MAX_FILE_BYTES = 25 * 1024 * 1024

const fieldCls = 'w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-base text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none disabled:opacity-60 wide:text-sm'
const labelCls = 'mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500'
const primaryCls = 'inline-flex items-center justify-center gap-2 rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50'
const quietCls = 'inline-flex items-center justify-center gap-2 rounded-lg border border-zinc-700 px-3 py-2 text-sm text-zinc-300 transition-colors hover:border-zinc-600 hover:bg-zinc-800 disabled:opacity-50'

const errorOf = (e: unknown) => (e instanceof Error ? e.message : String(e))
const WORKING = new Set(['planning', 'creating', 'recommending'])

function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-700 dark:text-red-300">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> <span className="min-w-0">{children}</span>
    </p>
  )
}

function Chip({ on, disabled, onClick, children }: { on: boolean; disabled?: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={on}
      className={`rounded-lg border px-2.5 py-1 text-xs transition-colors disabled:opacity-50 ${
        on ? 'border-violet-500/60 bg-violet-500/15 text-violet-700 dark:text-violet-200'
          : 'border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200'
      }`}
    >
      {children}
    </button>
  )
}

const STEPS = ['Ask', 'Plan', 'Create', 'Schedule and build']

function stepOf(plan: ForgePlan | null): number {
  if (!plan) return 0
  if (plan.status === 'planning') return 1
  if (plan.status === 'awaiting_plan') return 1
  if (plan.status === 'error') return plan.created.length ? 2 : 1
  if (plan.status === 'creating') return 2
  return 3
}

// ── Dialog ───────────────────────────────────────────────────────────

export function ForgeAgentsDialog({ initialPlanId, onClose, onCreated }: {
  initialPlanId?: string
  onClose: () => void
  /** Agents that have just been created, so the caller can watch them start. */
  onCreated?: (slugs: string[]) => void
}) {
  const [planId, setPlanId] = useState(initialPlanId || '')
  const [plan, setPlan] = useState<ForgePlan | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const { isMobile } = useIsMobile()
  const told = useRef(new Set<string>())

  const reload = useCallback(async () => {
    if (!planId) return
    try {
      setPlan(await forgeApi.get(planId))
      setLoadErr('')
    } catch (e) {
      setLoadErr(errorOf(e))
    }
  }, [planId])

  useEffect(() => {
    if (!planId) return
    let live = true
    forgeApi.get(planId)
      .then((p) => { if (live) setPlan(p) })
      .catch((e) => { if (live) setLoadErr(errorOf(e)) })
    return () => { live = false }
  }, [planId])

  // While it works, follow it. A plan left working by a restart has nothing to follow.
  const working = !!plan && WORKING.has(plan.status) && !plan.stalled
  useEffect(() => {
    if (!working) return
    const t = window.setInterval(() => { void reload() }, 2500)
    return () => window.clearInterval(t)
  }, [working, reload])

  // New agents reach the kiosk's list as they're made.
  useEffect(() => {
    const fresh = (plan?.created || []).filter((a) => a.ok && !told.current.has(a.slug))
    if (!fresh.length) return
    for (const a of fresh) told.current.add(a.slug)
    const procs = useProcessStore.getState()
    for (const a of fresh) if (a.instructions) void procs.setFleetInstructions(a.slug, a.instructions)
    void procs.fetchProcesses()
    onCreated?.(fresh.map((a) => a.slug))
  }, [plan?.created, onCreated])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const startOver = () => { setPlanId(''); setPlan(null); setLoadErr('') }
  const step = stepOf(plan)

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={isMobile ? undefined : onClose}>
      <div
        role="dialog"
        aria-label="Forge agents"
        className="flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-900 pb-[env(safe-area-inset-bottom)] shadow-2xl wide:h-[min(92vh,60rem)] wide:max-w-6xl wide:rounded-2xl wide:border wide:border-zinc-800 wide:pb-0"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
            <Hammer className="h-4 w-4" />
          </span>
          <div className="min-w-0">
            <h3 className="truncate text-base font-semibold text-zinc-100">Forge agents</h3>
            <ol className="flex items-center gap-1 overflow-hidden text-[11px] text-zinc-500" aria-label="Steps">
              {STEPS.map((s, i) => (
                <li key={s} className={`flex shrink-0 items-center gap-1 ${i === step ? 'font-medium text-violet-700 dark:text-violet-300' : ''} ${isMobile && i !== step ? 'hidden' : ''}`}>
                  {i > 0 && !isMobile && <ChevronRight className="h-3 w-3 text-zinc-700" />}
                  {s}
                </li>
              ))}
            </ol>
          </div>
          {plan && (
            <button onClick={startOver} className="ml-auto hidden rounded-lg px-2.5 py-1.5 text-xs text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 wide:block">
              New request
            </button>
          )}
          <button onClick={onClose} aria-label="Close" title="Close. A plan keeps going without this open." className={`${plan ? 'wide:ml-0' : ''} ml-auto rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200`}>
            <X className="h-5 w-5" />
          </button>
        </div>

        {loadErr && !plan && (
          <div className="p-5">
            <ErrorNote>{loadErr}</ErrorNote>
            <button onClick={startOver} className={`${quietCls} mt-3`}><ArrowLeft className="h-4 w-4" /> Back</button>
          </div>
        )}
        {!planId && <RequestStep onStarted={setPlanId} />}
        {planId && !plan && !loadErr && (
          <p className="flex flex-1 items-center justify-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</p>
        )}
        {plan && (WORKING.has(plan.status) || plan.status === 'error') && <WorkingStep plan={plan} onChanged={reload} onStartOver={startOver} />}
        {plan && plan.status === 'awaiting_plan' && <ReviewStep key={plan.id + plan.agents.map((a) => a.id).join()} plan={plan} onChanged={reload} />}
        {plan && plan.status === 'ready' && <ReadyStep plan={plan} onChanged={reload} />}
      </div>
    </div>,
    document.body,
  )
}

// ── Step 1: what you need ────────────────────────────────────────────

function RequestStep({ onStarted }: { onStarted: (id: string) => void }) {
  const [request, setRequest] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [recent, setRecent] = useState<ForgePlanRow[]>([])
  const picker = useRef<HTMLInputElement>(null)
  const { isMobile } = useIsMobile()

  useEffect(() => { forgeApi.list().then(setRecent).catch(() => {}) }, [])

  const add = (list: File[]) => {
    if (!list.length) return
    const tooBig = list.find((f) => f.size > MAX_FILE_BYTES)
    if (tooBig) { setErr(`${tooBig.name} is over 25 MB.`); return }
    setErr('')
    setFiles((old) => {
      const names = new Set(old.map((f) => f.name))
      const next = [...old, ...list.filter((f) => !names.has(f.name))]
      if (next.length > MAX_FILES) setErr(`Attach at most ${MAX_FILES} files.`)
      return next.slice(0, MAX_FILES)
    })
  }
  const onPaste = (e: ClipboardEvent) => {
    const list = Array.from(e.clipboardData?.files ?? [])
    if (list.length) { e.preventDefault(); add(list) }
  }
  const onDrop = (e: DragEvent) => {
    const list = Array.from(e.dataTransfer?.files ?? [])
    if (list.length) { e.preventDefault(); add(list) }
  }
  const onPick = (e: ChangeEvent<HTMLInputElement>) => { add(Array.from(e.target.files ?? [])); e.target.value = '' }

  const forget = async (id: string) => {
    try {
      await forgeApi.forget(id)
      setRecent((old) => old.filter((r) => r.id !== id))
    } catch (e) {
      setErr(errorOf(e))
    }
  }

  const submit = async () => {
    if (!request.trim() || busy) return
    setBusy(true)
    setErr('')
    try {
      onStarted((await forgeApi.start(request.trim(), files)).id)
    } catch (e) {
      setErr(errorOf(e))
      setBusy(false)
    }
  }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 wide:px-5" onDragOver={(e) => { if (e.dataTransfer?.types.includes('Files')) e.preventDefault() }} onDrop={onDrop}>
      <div className="mx-auto max-w-3xl space-y-3">
        <p className="text-sm leading-relaxed text-zinc-400">
          Say what you need done, in your own words, and attach what you have. Captain plans which agents you need and
          what each one does. Nothing is created until you've seen the plan and approved it.
        </p>
        <label className="block">
          <span className={labelCls}>What do you need?</span>
          <textarea
            autoFocus={!isMobile}
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            onPaste={onPaste}
            disabled={busy}
            rows={isMobile ? 8 : 10}
            placeholder="For example: every quarter we go through our funds' reports, work out what they really invested in, and fill in a standard Excel for Risk. It takes days…"
            className={`${fieldCls} resize-y leading-relaxed`}
          />
        </label>

        <div>
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={() => picker.current?.click()} disabled={busy} className={quietCls}>
              <Paperclip className="h-4 w-4" /> Attach files
            </button>
            <span className="text-xs text-zinc-600">{isMobile ? `Up to ${MAX_FILES}, 25 MB each.` : `Or drop or paste them here. Up to ${MAX_FILES}, 25 MB each.`}</span>
            <input ref={picker} type="file" multiple hidden onChange={onPick} />
          </div>
          {files.length > 0 && (
            <ul className="mt-2 flex flex-wrap gap-1.5">
              {files.map((f) => (
                <li key={f.name} className="flex max-w-full items-center gap-1 rounded-lg border border-zinc-700/80 bg-zinc-800/60 py-1 pl-2.5 pr-1 text-xs text-zinc-300">
                  <span className="truncate">{f.name}</span>
                  <button onClick={() => setFiles((old) => old.filter((x) => x !== f))} disabled={busy} aria-label={`Remove ${f.name}`} className="shrink-0 rounded p-0.5 text-zinc-500 hover:bg-zinc-700 hover:text-zinc-200">
                    <X className="h-3 w-3" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {err && <ErrorNote>{err}</ErrorNote>}
        <button onClick={submit} disabled={!request.trim() || busy} className={`${primaryCls} w-full wide:w-auto`}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          {busy ? 'Starting…' : 'Plan my agents'}
        </button>

        {recent.length > 0 && (
          <div className="border-t border-zinc-800 pt-3">
            <p className={labelCls}>Earlier plans</p>
            <ul className="space-y-1">
              {recent.slice(0, 6).map((r) => (
                <li key={r.id} className="flex items-center gap-1">
                  <button onClick={() => onStarted(r.id)} className="flex min-w-0 flex-1 items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm text-zinc-300 hover:bg-zinc-800">
                    <span className="min-w-0 flex-1 truncate">{r.title || 'Untitled'}</span>
                    <span className="shrink-0 text-xs text-zinc-500">{planLabel(r)}</span>
                  </button>
                  <button
                    onClick={() => forget(r.id)}
                    aria-label={`Forget the plan: ${r.title || 'Untitled'}`}
                    title="Forget this plan. Any agents, tasks and apps it made stay."
                    className="shrink-0 rounded-lg p-1.5 text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

function planLabel(r: ForgePlanRow): string {
  if (r.status === 'awaiting_plan') return 'Waiting for you'
  if (r.status === 'ready') return `${r.created} agent${r.created === 1 ? '' : 's'}`
  if (r.status === 'error') return 'Stopped'
  return 'Working…'
}

// ── Working: planning, creating, recommending ────────────────────────

function WorkingStep({ plan, onChanged, onStartOver }: { plan: ForgePlan; onChanged: () => Promise<void>; onStartOver: () => void }) {
  const [events, setEvents] = useState<ForgeFeedEvent[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const failed = plan.status === 'error'
  const stopped = failed || plan.stalled

  useEffect(() => {
    let live = true
    const read = () => forgeApi.feed(plan.id).then((f) => { if (live) setEvents(f.events || []) }).catch(() => {})
    void read()
    if (stopped) return () => { live = false }
    const t = window.setInterval(read, 1500)
    return () => { live = false; window.clearInterval(t) }
  }, [plan.id, stopped])

  const shown = useMemo(() => events.filter((e) => e.message && !['usage', 'llm', 'action', 'narration'].includes(e.stage)), [events])
  const phase = [...events].reverse().find((e) => e.stage === 'phase')?.message
  const headline = plan.status === 'planning' ? 'Planning your team' : plan.status === 'creating' ? 'Creating your agents' : 'Working out what to schedule and build'
  // What carries on from here depends on how far it got. Going on with an approved team makes
  // only the agents it doesn't have yet.
  const resume = async () => {
    setBusy(true)
    setErr('')
    try {
      if (plan.status === 'recommending') await forgeApi.recommend(plan.id)
      else if (plan.status === 'creating' || plan.created.length > 0) await forgeApi.approve(plan.id, [], plan.overview)
      else await forgeApi.replan(plan.id, '', [])
      await onChanged()
    } catch (e) {
      setErr(errorOf(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 wide:px-5">
      <div className="mx-auto max-w-2xl space-y-3">
        {!stopped && (
          <div className="flex items-center gap-3">
            <Loader2 className="h-5 w-5 shrink-0 animate-spin text-violet-500" />
            <div className="min-w-0">
              <p className="text-sm font-medium text-zinc-100">{headline}</p>
              <p className="truncate text-xs text-zinc-500">{phase || 'Starting…'}</p>
            </div>
          </div>
        )}
        {!stopped && <p className="text-xs text-zinc-500">You can close this. It carries on, and the bell tells you when it needs you.</p>}
        {failed && <ErrorNote>{plan.error || 'It stopped without saying why.'}</ErrorNote>}
        {plan.stalled && <ErrorNote>This stopped when Flight Deck restarted. Nothing is lost; carry on from where it was.</ErrorNote>}
        {err && <ErrorNote>{err}</ErrorNote>}
        {stopped && (
          <div className="flex flex-wrap gap-2">
            <button onClick={resume} disabled={busy} className={primaryCls}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />} Try again
            </button>
            <button onClick={onStartOver} disabled={busy} className={quietCls}>New request</button>
          </div>
        )}
        {plan.created.length > 0 && (
          <ul className="space-y-1">
            {plan.created.map((a) => <CreatedRow key={a.id} agent={a} />)}
          </ul>
        )}
        {shown.length > 0 && (
          <ol className="space-y-1 border-t border-zinc-800 pt-3 text-xs text-zinc-500">
            {shown.slice(-40).map((e, i) => (
              <li key={i} className={e.stage === 'phase' ? 'pt-1.5 font-medium text-zinc-300' : e.ok === false ? 'text-amber-700 dark:text-amber-300' : ''}>{e.message}</li>
            ))}
          </ol>
        )}
      </div>
    </div>
  )
}

function CreatedRow({ agent }: { agent: ForgePlan['created'][number] }) {
  return (
    <li className="flex items-start gap-2 rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm">
      {agent.ok ? <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" /> : <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />}
      <span className="min-w-0">
        <span className="text-zinc-200">{agent.name}</span>
        <span className="ml-1.5 text-xs text-zinc-500">{agent.role}</span>
        {agent.error && <span className="block text-xs text-red-700 dark:text-red-300">{agent.error}</span>}
      </span>
    </li>
  )
}

// ── Step 2: the plan ─────────────────────────────────────────────────

function ReviewStep({ plan, onChanged }: { plan: ForgePlan; onChanged: () => Promise<void> }) {
  const [agents, setAgents] = useState<ForgeDraftAgent[]>(plan.agents)
  const [overview, setOverview] = useState(plan.overview)
  const [questions, setQuestions] = useState<ForgeQuestion[]>(plan.questions)
  const [note, setNote] = useState('')
  const [selected, setSelected] = useState(plan.agents[0]?.id || '')
  const [tab, setTab] = useState<'team' | 'questions'>('team')
  const [archetypes, setArchetypes] = useState<Archetype[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const { isMobile } = useIsMobile()

  useEffect(() => { listArchetypes().then((r) => setArchetypes(r.archetypes || [])).catch(() => {}) }, [])

  const current = agents.find((a) => a.id === selected) || agents[0]
  const patch = (p: Partial<ForgeDraftAgent>) => setAgents((list) => list.map((a) => (a.id === current?.id ? { ...a, ...p } : a)))
  const remove = (id: string) => {
    setAgents((list) => list.filter((a) => a.id !== id).map((a) => ({ ...a, consumes_from: a.consumes_from.filter((c) => c !== id) })))
    if (selected === id) setSelected('')
  }
  const families = useMemo(() => {
    const byFamily = new Map<string, Archetype[]>()
    for (const a of archetypes) byFamily.set(a.family || 'Other', [...(byFamily.get(a.family || 'Other') || []), a])
    return [...byFamily.entries()].sort(([x], [y]) => x.localeCompare(y))
  }, [archetypes])

  const answered = (q: ForgeQuestion) => (q.selected || []).length > 0 || !!(q.other || '').trim()
  const toAnswer = questions.filter((q) => !answered(q)).length
  const hasAnswers = questions.some(answered)
  const wantsReplan = hasAnswers || !!note.trim()

  const act = async (run: () => Promise<unknown>) => {
    setBusy(true)
    setErr('')
    try {
      await run()
      await onChanged()
    } catch (e) {
      setErr(errorOf(e))
      setBusy(false)
    }
  }

  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 wide:px-5">
        {plan.rationale && <p className="text-sm leading-relaxed text-zinc-300">{plan.rationale}</p>}
        {plan.room < plan.max_team && plan.agents.length >= plan.room && (
          <p className="mt-2 text-xs leading-relaxed text-amber-700 dark:text-amber-300">
            Your plan has room for {plan.room} more agent{plan.room === 1 ? '' : 's'}, so the team was kept to that size. An admin can raise the limit.
          </p>
        )}
        <label className="mt-3 block">
          <span className={labelCls}>How the team works together</span>
          <textarea value={overview} onChange={(e) => setOverview(e.target.value)} rows={isMobile ? 4 : 2} disabled={busy} className={`${fieldCls} resize-y leading-relaxed`} />
        </label>

        {questions.length > 0 && (
          <div className="mt-3 flex items-center gap-1 border-b border-zinc-800" role="tablist">
            <TabButton on={tab === 'team'} onClick={() => setTab('team')}>The team</TabButton>
            <TabButton on={tab === 'questions'} onClick={() => setTab('questions')}>
              <HelpCircle className="h-3.5 w-3.5" /> Questions
              {toAnswer > 0 && <span className="rounded-full bg-amber-500/20 px-1.5 text-[10px] font-semibold text-amber-700 dark:text-amber-300">{toAnswer}</span>}
            </TabButton>
          </div>
        )}

        {tab === 'questions' && questions.length > 0 ? (
          <div className="mt-3 space-y-3">
            <p className="text-xs text-zinc-500">Answer what matters, then plan again so the team reflects it. Anything you leave blank uses the planner's own judgement.</p>
            {questions.map((q) => (
              <QuestionCard key={q.id} q={q} disabled={busy} onChange={(next) => setQuestions((all) => all.map((x) => (x.id === q.id ? next : x)))} />
            ))}
          </div>
        ) : (
          <div className="mt-3 flex flex-col gap-3 wide:flex-row wide:items-start">
            <ul className="shrink-0 space-y-1.5 wide:w-72" aria-label="Planned agents">
              {agents.map((a) => (
                <li key={a.id}>
                  <button
                    onClick={() => setSelected(a.id)}
                    aria-current={a.id === current?.id}
                    className={`flex w-full items-start gap-2.5 rounded-xl border p-2.5 text-left transition-colors ${
                      a.id === current?.id ? 'border-violet-500/60 bg-violet-500/10' : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700'
                    }`}
                  >
                    <Bot className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-medium text-zinc-100">{a.name}</span>
                      <span className="block truncate text-xs text-zinc-500">{archetypes.find((x) => x.id === a.archetype)?.role || a.archetype}</span>
                    </span>
                  </button>
                </li>
              ))}
              {agents.length === 0 && <li className="px-2 py-3 text-sm text-zinc-500">You've removed every agent. Plan again to get a new team.</li>}
            </ul>
            {current && (
              <div className="min-w-0 flex-1 space-y-3 rounded-xl border border-zinc-800 bg-zinc-950/40 p-3">
                <div className="grid gap-3 wide:grid-cols-2">
                  <label className="block">
                    <span className={labelCls}>Name</span>
                    <input value={current.name} onChange={(e) => patch({ name: e.target.value })} disabled={busy} className={fieldCls} />
                  </label>
                  <label className="block">
                    <span className={labelCls}>Based on</span>
                    <select value={current.archetype} onChange={(e) => patch({ archetype: e.target.value })} disabled={busy} className={fieldCls}>
                      {!archetypes.some((a) => a.id === current.archetype) && <option value={current.archetype}>{current.archetype}</option>}
                      {families.map(([family, list]) => (
                        <optgroup key={family} label={family}>
                          {list.map((a) => <option key={a.id} value={a.id}>{a.role || a.id}</option>)}
                        </optgroup>
                      ))}
                    </select>
                  </label>
                </div>
                <label className="block">
                  <span className={labelCls}>Its job</span>
                  <textarea value={current.mandate} onChange={(e) => patch({ mandate: e.target.value })} rows={7} disabled={busy} className={`${fieldCls} resize-y leading-relaxed`} />
                </label>
                <label className="block">
                  <span className={labelCls}>What it delivers</span>
                  <textarea value={current.produces} onChange={(e) => patch({ produces: e.target.value })} rows={2} disabled={busy} className={`${fieldCls} resize-y leading-relaxed`} />
                </label>
                {agents.length > 1 && (
                  <div>
                    <span className={labelCls}>Builds on the work of</span>
                    <div className="flex flex-wrap gap-1.5">
                      {agents.filter((a) => a.id !== current.id).map((a) => {
                        const on = current.consumes_from.includes(a.id)
                        return (
                          <Chip key={a.id} on={on} disabled={busy} onClick={() => patch({ consumes_from: on ? current.consumes_from.filter((c) => c !== a.id) : [...current.consumes_from, a.id] })}>
                            {a.name}
                          </Chip>
                        )
                      })}
                    </div>
                  </div>
                )}
                <label className="block">
                  <span className={labelCls}>How teammates get its work</span>
                  <textarea value={current.hand_off_notes} onChange={(e) => patch({ hand_off_notes: e.target.value })} rows={3} disabled={busy} className={`${fieldCls} resize-y leading-relaxed`} />
                </label>
                <button onClick={() => remove(current.id)} disabled={busy} className="inline-flex items-center gap-1.5 text-xs text-zinc-500 hover:text-red-600 disabled:opacity-50 dark:hover:text-red-400">
                  <Trash2 className="h-3.5 w-3.5" /> Leave this agent out
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="shrink-0 space-y-2 border-t border-zinc-800 px-4 py-3 wide:px-5">
        {err && <ErrorNote>{err}</ErrorNote>}
        <div className="flex flex-col gap-2 wide:flex-row wide:items-center">
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            disabled={busy}
            aria-label="What to change"
            placeholder={hasAnswers ? 'Anything else to change? Your answers go in too.' : 'Want something different? Say what, and plan again.'}
            className={`${fieldCls} wide:flex-1`}
          />
          <div className="flex gap-2">
            <button onClick={() => act(() => forgeApi.replan(plan.id, note.trim(), questions))} disabled={busy || !wantsReplan} className={`${quietCls} flex-1 wide:flex-none`}>
              <RefreshCw className="h-4 w-4" /> Plan again
            </button>
            <button onClick={() => act(() => forgeApi.approve(plan.id, agents, overview))} disabled={busy || agents.length === 0} className={`${primaryCls} flex-1 wide:flex-none`}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
              Create {agents.length} agent{agents.length === 1 ? '' : 's'}
            </button>
          </div>
        </div>
      </div>
    </>
  )
}

function TabButton({ on, onClick, children }: { on: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      role="tab"
      aria-selected={on}
      onClick={onClick}
      className={`-mb-px flex shrink-0 items-center gap-1.5 whitespace-nowrap border-b-2 px-3 py-2 text-sm transition-colors ${
        on ? 'border-violet-500 font-medium text-violet-700 dark:text-violet-300' : 'border-transparent text-zinc-500 hover:text-zinc-300'
      }`}
    >
      {children}
    </button>
  )
}

function QuestionCard({ q, disabled, onChange }: { q: ForgeQuestion; disabled: boolean; onChange: (q: ForgeQuestion) => void }) {
  const selected = q.selected || []
  const toggle = (opt: string) => {
    const has = selected.includes(opt)
    onChange({ ...q, selected: q.multi ? (has ? selected.filter((o) => o !== opt) : [...selected, opt]) : (has ? [] : [opt]) })
  }
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/40 p-3">
      <p className="text-sm font-medium leading-snug text-zinc-200">{q.question}</p>
      {q.options.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {q.options.map((o) => <Chip key={o} on={selected.includes(o)} disabled={disabled} onClick={() => toggle(o)}>{o}</Chip>)}
        </div>
      )}
      <input value={q.other || ''} onChange={(e) => onChange({ ...q, other: e.target.value })} disabled={disabled} aria-label={`Your own answer: ${q.question}`} placeholder="Your own answer" className={`${fieldCls} mt-2`} />
    </div>
  )
}

// ── Step 3: what to schedule, what to build ──────────────────────────

function ReadyStep({ plan, onChanged }: { plan: ForgePlan; onChanged: () => Promise<void> }) {
  const recs = plan.recommendations
  const schedules = recs?.schedules || []
  const apps = recs?.apps || []
  const [tab, setTab] = useState<'team' | 'schedules' | 'apps'>('team')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const agents = plan.created.filter((a) => a.ok)

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

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-1 overflow-x-auto border-b border-zinc-800 px-3 wide:px-4" role="tablist">
        <TabButton on={tab === 'team'} onClick={() => setTab('team')}><Bot className="h-3.5 w-3.5" /> Your team</TabButton>
        <TabButton on={tab === 'schedules'} onClick={() => setTab('schedules')}><CalendarClock className="h-3.5 w-3.5" /> Scheduled tasks ({schedules.length})</TabButton>
        <TabButton on={tab === 'apps'} onClick={() => setTab('apps')}><Plug className="h-3.5 w-3.5" /> Apps to build ({apps.length})</TabButton>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 wide:px-5">
        {tab === 'team' && (
          <div className="mx-auto max-w-3xl space-y-3">
            {plan.recommend_error && (
              <div className="space-y-2">
                <ErrorNote>Your agents are created, but the recommendations didn't work: {plan.recommend_error}</ErrorNote>
                <button onClick={again} disabled={busy} className={quietCls}>
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />} Try again
                </button>
                {err && <ErrorNote>{err}</ErrorNote>}
              </div>
            )}
            {recs?.usage && (
              <div>
                <p className={labelCls}>How to use this team</p>
                <p className="whitespace-pre-wrap text-sm leading-relaxed text-zinc-300">{recs.usage}</p>
              </div>
            )}
            <div>
              <p className={labelCls}>Agents</p>
              <ul className="space-y-1">{plan.created.map((a) => <CreatedRow key={a.id} agent={a} />)}</ul>
            </div>
            {plan.team && (
              <p className="text-xs leading-relaxed text-zinc-500">
                They share the folder <code className="rounded bg-zinc-800 px-1 py-0.5">vfs:{plan.team}/</code>, where your files are under <code className="rounded bg-zinc-800 px-1 py-0.5">inputs/</code>, and one set of data tables.
              </p>
            )}
          </div>
        )}
        {tab === 'schedules' && <SchedulesTab plan={plan} agents={agents} schedules={schedules} onChanged={onChanged} />}
        {tab === 'apps' && <AppsTab plan={plan} agents={agents} apps={apps} onChanged={onChanged} />}
      </div>
    </div>
  )
}

function SchedulesTab({ plan, agents, schedules, onChanged }: {
  plan: ForgePlan; agents: ForgePlan['created']; schedules: ForgeSchedule[]; onChanged: () => Promise<void>
}) {
  const [drafts, setDrafts] = useState<ForgeSchedule[]>(schedules)
  const [kept, setKept] = useState<Set<string>>(() => new Set(schedules.filter((s) => !s.job_id).map((s) => s.id)))
  const [problems, setProblems] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  useEffect(() => {
    // Only what the server now knows was made; an edit in progress stays.
    setDrafts((old) => old.map((d) => ({ ...d, job_id: schedules.find((s) => s.id === d.id)?.job_id })))
  }, [schedules])

  if (!schedules.length) {
    return <p className="mx-auto max-w-3xl text-sm text-zinc-500">Nothing here needs to run on a schedule: your agents work when you ask them. You can add a task later from Scheduled tasks.</p>
  }
  const open = drafts.filter((d) => !d.job_id)
  const chosen = open.filter((d) => kept.has(d.id))

  const make = async () => {
    setBusy(true)
    setErr('')
    setProblems({})
    try {
      const { results } = await forgeApi.schedules(plan.id, chosen, localTimeZone())
      setProblems(Object.fromEntries(results.filter((r) => !r.ok).map((r) => [r.id, r.error || 'It could not be made.'])))
      await onChanged()
    } catch (e) {
      setErr(errorOf(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-3">
      {drafts.map((d) => (
        <ScheduleCard
          key={d.id}
          draft={d}
          agents={agents}
          kept={kept.has(d.id)}
          problem={problems[d.id] || ''}
          disabled={busy}
          onKeep={(on) => setKept((old) => { const n = new Set(old); if (on) n.add(d.id); else n.delete(d.id); return n })}
          onChange={(next) => setDrafts((list) => list.map((x) => (x.id === d.id ? next : x)))}
        />
      ))}
      {err && <ErrorNote>{err}</ErrorNote>}
      {open.length > 0 && (
        <button onClick={make} disabled={busy || chosen.length === 0} className={`${primaryCls} w-full wide:w-auto`}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <CalendarClock className="h-4 w-4" />}
          Create {chosen.length} scheduled task{chosen.length === 1 ? '' : 's'}
        </button>
      )}
    </div>
  )
}

function ScheduleCard({ draft, agents, kept, problem, disabled, onKeep, onChange }: {
  draft: ForgeSchedule; agents: ForgePlan['created']; kept: boolean; problem: string; disabled: boolean
  onKeep: (on: boolean) => void; onChange: (next: ForgeSchedule) => void
}) {
  const [open, setOpen] = useState(false)
  const made = !!draft.job_id
  const setDelivery = (key: 'chat' | 'email' | 'telegram') => onChange({ ...draft, delivery: { ...draft.delivery, [key]: !draft.delivery[key] } })
  const off = disabled || made
  return (
    <div className={`rounded-xl border p-3 ${made ? 'border-emerald-500/30 bg-emerald-500/5' : kept ? 'border-zinc-700 bg-zinc-950/40' : 'border-zinc-800 opacity-70'}`}>
      <div className="flex items-start gap-2.5">
        {made ? <Check className="mt-1 h-4 w-4 shrink-0 text-emerald-500" /> : (
          <input type="checkbox" checked={kept} onChange={(e) => onKeep(e.target.checked)} disabled={disabled} aria-label={`Keep ${draft.name}`} className="mt-1 h-4 w-4 shrink-0 accent-violet-600" />
        )}
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-zinc-100">{draft.name}{made && <span className="ml-2 text-xs font-normal text-emerald-700 dark:text-emerald-300">Created</span>}</p>
          {draft.why && <p className="mt-0.5 text-xs leading-relaxed text-zinc-400">{draft.why}</p>}
          {draft.usage && <p className="mt-1 text-xs leading-relaxed text-zinc-500"><span className="font-medium text-zinc-400">How you'd use it: </span>{draft.usage}</p>}
        </div>
      </div>
      <div className="mt-2.5 grid gap-2 wide:grid-cols-2">
        <label className="block">
          <span className={labelCls}>Agent</span>
          <select value={draft.agent} onChange={(e) => onChange({ ...draft, agent: e.target.value })} disabled={off} className={fieldCls}>
            {agents.map((a) => <option key={a.slug} value={a.slug}>{a.name}</option>)}
          </select>
        </label>
        <label className="block">
          <span className={labelCls}>When</span>
          <input
            value={draft.schedule}
            onChange={(e) => onChange({ ...draft, schedule: e.target.value, schedule_problem: '' })}
            disabled={off}
            placeholder="weekdays 07:30"
            className={fieldCls}
          />
        </label>
      </div>
      {(problem || draft.schedule_problem) && !made && (
        <p className="mt-1.5 text-xs text-red-700 dark:text-red-300">{problem || draft.schedule_problem}</p>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[11px] uppercase tracking-wider text-zinc-500">Send to</span>
        <Chip on={draft.delivery.chat} disabled={off} onClick={() => setDelivery('chat')}>Chat</Chip>
        <Chip on={draft.delivery.email} disabled={off} onClick={() => setDelivery('email')}>Email</Chip>
        <Chip on={draft.delivery.telegram} disabled={off} onClick={() => setDelivery('telegram')}>Telegram</Chip>
      </div>
      <button onClick={() => setOpen(!open)} aria-expanded={open} className="mt-2 inline-flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300">
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />} What the agent is told
      </button>
      {open && (
        <textarea value={draft.prompt} onChange={(e) => onChange({ ...draft, prompt: e.target.value })} disabled={off} rows={6} aria-label="What the agent is told" className={`${fieldCls} mt-1.5 resize-y leading-relaxed`} />
      )}
    </div>
  )
}

function AppsTab({ plan, agents, apps, onChanged }: {
  plan: ForgePlan; agents: ForgePlan['created']; apps: ForgeApp[]; onChanged: () => Promise<void>
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
    const t = window.setInterval(() => { void onChanged() }, 5000)
    return () => window.clearInterval(t)
  }, [building, onChanged])

  if (!apps.length) {
    return <p className="mx-auto max-w-3xl text-sm text-zinc-500">No software needed: your agents' own tools cover this. If that changes, you can give Crew a task from the Crew page.</p>
  }
  return (
    <div className="mx-auto max-w-3xl space-y-3">
      <p className="text-xs leading-relaxed text-zinc-500">
        Crew's coding agents build these in a repository of their own. Each is built for agents first: an HTTP API they call and a WebSocket where the app hands them work. Once it runs, connect your agents to it here.
      </p>
      {me && !me.allowed && (
        <ErrorNote>{me.offered ? "Crew isn't approved for your account yet, so these can't be built from here. An admin can approve it on the Users page." : 'Crew is turned off on this deck, so these can\'t be built from here.'}</ErrorNote>
      )}
      {apps.map((a) => <AppCard key={a.id} plan={plan} agents={agents} app={a} me={me} choices={choices} choicesErr={choicesErr} onChanged={onChanged} />)}
    </div>
  )
}

const CONNECT_LABEL: Record<ForgeConnect, string> = { both: 'API and socket', api: 'API only', wss: 'Socket only' }
const FINISHED_BADLY = new Set(['failed', 'stopped', 'closed', 'gone'])

function AppCard({ plan, agents, app, me, choices, choicesErr, onChanged }: {
  plan: ForgePlan; agents: ForgePlan['created']; app: ForgeApp; me: CrewMe | null; choices: CrewModelChoices | null
  choicesErr: string; onChanged: () => Promise<void>
}) {
  const usable = (me?.harnesses || []).filter((h) => h.installed)
  const [harnessId, setHarnessId] = useState('')
  const harness = usable.find((h) => h.id === harnessId) || usable.find((h) => h.login.ok) || usable[0]
  const [model, setModel] = useState('')
  const [modelChoice, setModelChoice] = useState<ModelChoice>(NOTHING_CHOSEN)
  const [brief, setBrief] = useState(app.brief)
  const [connect, setConnect] = useState<ForgeConnect>(app.connect)
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const task = app.task
  const started = !!task && !FINISHED_BADLY.has(task.status)
  const names = Object.fromEntries(agents.map((a) => [a.slug, a.name]))

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

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/40 p-3">
      <p className="text-sm font-medium text-zinc-100">{app.name}</p>
      {app.summary && <p className="mt-0.5 text-sm leading-relaxed text-zinc-300">{app.summary}</p>}
      {app.why && <p className="mt-1 text-xs leading-relaxed text-zinc-500">{app.why}</p>}
      {app.agents.length > 0 && (
        <ul className="mt-2 space-y-0.5 text-xs text-zinc-400">
          {app.agents.map((s) => (
            <li key={s}><span className="font-medium text-zinc-300">{names[s] || s}</span>{app.agent_use[s] ? `: ${app.agent_use[s]}` : ''}</li>
          ))}
        </ul>
      )}

      <button onClick={() => setOpen(!open)} aria-expanded={open} className="mt-2 inline-flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300">
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />} What Crew is told
      </button>
      {open && (
        <div className="mt-1.5 space-y-2">
          <textarea value={brief} onChange={(e) => setBrief(e.target.value)} disabled={busy || started} rows={8} aria-label="What Crew is told" className={`${fieldCls} resize-y leading-relaxed`} />
          {app.endpoints.length > 0 && (
            <div>
              <p className={labelCls}>What agents call</p>
              <ul className="space-y-0.5 text-xs text-zinc-400">
                {app.endpoints.map((e) => <li key={e.method + e.path}><code className="text-zinc-300">{e.method} {e.path}</code>{e.purpose ? `: ${e.purpose}` : ''}</li>)}
              </ul>
            </div>
          )}
          {app.events.length > 0 && (
            <div>
              <p className={labelCls}>What the app sends agents</p>
              <ul className="space-y-0.5 text-xs text-zinc-400">
                {app.events.map((e) => <li key={e.name}><code className="text-zinc-300">{e.name}</code>{e.purpose ? `: ${e.purpose}` : ''}</li>)}
              </ul>
            </div>
          )}
          <p className="text-xs text-zinc-500">The fixed rules for how agents connect and what to hand over are added to this.</p>
        </div>
      )}

      {task && (
        <p className={`mt-2.5 text-xs ${FINISHED_BADLY.has(task.status) ? 'text-red-700 dark:text-red-300' : 'text-zinc-400'}`}>
          <span className="font-medium">Crew: {task.status.replace('_', ' ')}</span>
          {task.note ? ` · ${task.note}` : ''}
          {task.branch ? ` · on branch ${task.branch}` : ''}
        </p>
      )}

      {!started && me?.allowed && (
        usable.length === 0 ? (
          <p className="mt-2.5 text-xs text-zinc-500">No coding CLI is installed on this computer yet. An admin can install one; the Crew page says how.</p>
        ) : (
          <div className="mt-2.5 space-y-2">
            <div className="grid gap-2 wide:grid-cols-2">
              <label className="block">
                <span className={labelCls}>Built by</span>
                <select value={harness?.id || ''} onChange={(e) => setHarnessId(e.target.value)} disabled={busy} className={fieldCls}>
                  {usable.map((h) => <option key={h.id} value={h.id}>{h.label}{h.login.ok ? '' : ' (no login)'}</option>)}
                </select>
              </label>
              <label className="block">
                <span className={labelCls}>Agents connect by</span>
                <select value={connect} onChange={(e) => setConnect(e.target.value as ForgeConnect)} disabled={busy} className={fieldCls}>
                  {(Object.keys(CONNECT_LABEL) as ForgeConnect[]).map((c) => <option key={c} value={c}>{CONNECT_LABEL[c]}</option>)}
                </select>
              </label>
            </div>
            <div>
              <span className={labelCls}>Model</span>
              <ModelChooser
                choice={modelChoice} choices={choices} harness={harness} own={me.own_model} loadError={choicesErr}
                onChange={(next) => setModelChoice((was) => ({ ...was, ...next }))} onLogins={() => setErr('Add a login on the Crew page, under Logins.')}
              >
                <input value={model} onChange={(e) => setModel(e.target.value)} aria-label="Its model" placeholder={harness?.default_model ? `Your default: ${harness.default_model}` : 'Its default'} className={fieldCls} />
              </ModelChooser>
            </div>
            {err && <ErrorNote>{err}</ErrorNote>}
            <button onClick={build} disabled={busy || !harness} className={`${primaryCls} w-full wide:w-auto`}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Hammer className="h-4 w-4" />}
              {task ? 'Build it again' : 'Build it with Crew'}
            </button>
          </div>
        )
      )}

      {started && <ConnectForm plan={plan} app={app} agents={agents} onChanged={onChanged} />}
    </div>
  )
}

function ConnectForm({ plan, app, agents, onChanged }: {
  plan: ForgePlan; app: ForgeApp; agents: ForgePlan['created']; onChanged: () => Promise<void>
}) {
  const [url, setUrl] = useState(app.connection?.url || '')
  const [key, setKey] = useState('')
  const [chosen, setChosen] = useState<Set<string>>(() => new Set(app.connection?.agents?.length ? app.connection.agents : app.agents))
  const [results, setResults] = useState<ForgeConnectResult[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

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
      await onChanged()
    } catch (e) {
      setErr(errorOf(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mt-3 space-y-2 border-t border-zinc-800 pt-3">
      <p className="flex items-center gap-1.5 text-sm font-medium text-zinc-200"><Plug className="h-3.5 w-3.5 text-violet-500" /> Connect your agents</p>
      {app.connection ? (
        <p className="text-xs leading-relaxed text-emerald-700 dark:text-emerald-300">
          Connected to {app.connection.url}{app.connection.socket ? ' over its socket' : ''}{app.connection.api_calls ? `${app.connection.socket ? ' and' : ' with'} ${app.connection.api_calls} API calls` : ''}. Connect again if the address or key changes.
        </p>
      ) : (
        <p className="text-xs leading-relaxed text-zinc-500">
          When the app is running, give its address and key. The key is in what Crew reports when it finishes. You can publish the app from Crew, under Live sites, if your account allows it.
        </p>
      )}
      <div className="grid gap-2 wide:grid-cols-2">
        <input value={url} onChange={(e) => setUrl(e.target.value)} disabled={busy} aria-label="The app's address" placeholder="https://… or wss://…" className={fieldCls} />
        <input value={key} onChange={(e) => setKey(e.target.value)} disabled={busy} type="password" autoComplete="off" aria-label="The app's key" placeholder="The app's AGENT_API_KEY" className={fieldCls} />
      </div>
      <div className="flex flex-wrap gap-1.5">
        {agents.map((a) => (
          <Chip key={a.slug} on={chosen.has(a.slug)} disabled={busy} onClick={() => setChosen((old) => { const n = new Set(old); if (n.has(a.slug)) n.delete(a.slug); else n.add(a.slug); return n })}>
            {a.name}
          </Chip>
        ))}
      </div>
      {err && <ErrorNote>{err}</ErrorNote>}
      {results.length > 0 && (
        <ul className="space-y-0.5 text-xs">
          {results.map((r) => (
            <li key={r.slug} className={r.error ? 'text-red-700 dark:text-red-300' : 'text-zinc-400'}>
              {r.name}: {r.error || [r.linked ? 'socket linked' : '', r.api_calls ? `${r.api_calls} API calls registered` : ''].filter(Boolean).join(', ') || 'nothing to connect'}
            </li>
          ))}
        </ul>
      )}
      <button onClick={connect} disabled={busy || !url.trim() || !key.trim() || chosen.size === 0} className={quietCls}>
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plug className="h-4 w-4" />} Connect
      </button>
    </div>
  )
}
