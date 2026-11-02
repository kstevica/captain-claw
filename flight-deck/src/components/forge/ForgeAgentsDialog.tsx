import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Check, Hammer, Loader2, Plus, X } from 'lucide-react'
import { listArchetypes } from '../../services/archetypes'
import { forgeApi, type ForgePlan } from '../../services/agentForge'
import type { Archetype } from '../../services/tierConfig'
import { useProcessStore } from '../../stores/processStore'
import { useIsMobile } from '../../hooks/useMediaQuery'
import { DialogFullscreen } from '../library/dialogFullscreen'
import { FullscreenButton } from '../library/libraryParts'
import { ForgeAsk } from './ForgeAsk'
import { ForgeReady } from './ForgeReady'
import { ForgeReview } from './ForgeReview'
import { ForgeWorking } from './ForgeWorking'
import { ErrorNote } from './forgeParts'
import { errorOf } from './forgeUtil'

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

const WORKING = new Set(['planning', 'creating', 'recommending'])

const STEPS = ['Describe', 'Review', 'Create', 'Finish']

function stepOf(plan: ForgePlan | null): number {
  if (!plan) return 0
  if (plan.status === 'planning' || plan.status === 'awaiting_plan') return 1
  if (plan.status === 'error') return plan.approved.length ? 2 : 1
  if (plan.status === 'creating') return 2
  return 3
}

/** Where you are, drawn: numbered marks joined by a line, or a bar on a phone. */
function Steps({ current, working }: { current: number; working: boolean }) {
  return (
    <ol aria-label="Steps" className="hidden items-center gap-1.5 wide:flex">
      {STEPS.map((s, i) => (
        <li key={s} aria-current={i === current ? 'step' : undefined} className="flex items-center gap-1.5">
          {i > 0 && <span className={`h-px w-5 transition-colors ${i <= current ? 'bg-violet-500/60' : 'bg-zinc-800'}`} />}
          <span className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-semibold transition-colors ${
            i < current ? 'bg-violet-600 text-white'
              : i === current ? 'bg-violet-500/15 text-violet-700 ring-1 ring-violet-500/50 dark:text-violet-300'
                : 'bg-zinc-800 text-zinc-500'
          }`}>
            {i < current ? <Check className="h-3 w-3" /> : i === current && working ? <Loader2 className="h-3 w-3 animate-spin" /> : i + 1}
          </span>
          <span className={`text-xs ${i === current ? 'font-medium text-zinc-200' : 'text-zinc-500'}`}>{s}</span>
        </li>
      ))}
    </ol>
  )
}

export function ForgeAgentsDialog({ initialPlanId, onClose, onCreated }: {
  initialPlanId?: string
  onClose: () => void
  /** Agents that have just been created, so the caller can watch them start. */
  onCreated?: (slugs: string[]) => void
}) {
  const [planId, setPlanId] = useState(initialPlanId || '')
  const [plan, setPlan] = useState<ForgePlan | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const [archetypes, setArchetypes] = useState<Archetype[]>([])
  const [fullscreen, setFullscreen] = useState(false)
  const { isMobile } = useIsMobile()
  const told = useRef(new Set<string>())
  const self = useRef<HTMLDivElement>(null)

  useEffect(() => { listArchetypes().then((r) => setArchetypes(r.archetypes || [])).catch(() => {}) }, [])

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
    const t = window.setInterval(() => { void reload() }, 2000)
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
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // A sheet open over this takes its own Escape.
      const over = [...document.querySelectorAll('[aria-modal="true"]')].some((el) => el !== self.current)
      if (!over) onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const startOver = () => { setPlanId(''); setPlan(null); setLoadErr('') }
  const step = stepOf(plan)
  const host = useMemo(() => (isMobile ? null : { fullscreen, toggle: () => setFullscreen((v) => !v) }), [isMobile, fullscreen])
  const big = fullscreen && !isMobile

  let body: ReactNode = null
  if (!planId) body = <ForgeAsk onStarted={setPlanId} />
  else if (loadErr && !plan) body = <div className="p-5"><ErrorNote>{loadErr}</ErrorNote></div>
  else if (!plan) body = <p className="flex flex-1 items-center justify-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</p>
  else if (plan.status === 'awaiting_plan') body = <ForgeReview key={plan.id + plan.agents.map((a) => a.id + a.name).join()} plan={plan} archetypes={archetypes} onChanged={reload} />
  else if (plan.status === 'ready') body = <ForgeReady plan={plan} archetypes={archetypes} onChanged={reload} onClose={onClose} onAnother={startOver} />
  else body = <ForgeWorking plan={plan} archetypes={archetypes} onChanged={reload} onStartOver={startOver} onClose={onClose} />

  return createPortal(
    <div className={`fd-fade fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center ${big ? '' : 'wide:p-4'}`} onClick={isMobile ? undefined : onClose}>
      <div
        ref={self}
        role="dialog"
        aria-modal="true"
        aria-label="Forge agents"
        className={`fd-sheet fd-card-wide flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 pb-[env(safe-area-inset-bottom)] shadow-2xl wide:pb-0 ${
          big ? '' : 'wide:h-[min(92vh,62rem)] wide:max-w-6xl wide:rounded-2xl wide:border wide:border-zinc-800'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        <DialogFullscreen.Provider value={host}>
          <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
              <Hammer className="h-[18px] w-[18px]" />
            </span>
            <div className="min-w-0 flex-1">
              <h3 className="truncate text-base font-semibold text-zinc-100">Forge agents</h3>
              <p className="truncate text-xs text-zinc-500">
                {isMobile && <span className="font-medium text-violet-700 dark:text-violet-300">{step + 1} of {STEPS.length} · {STEPS[step]}</span>}
                {isMobile && plan?.title ? ' · ' : ''}
                {plan?.title || (isMobile ? '' : 'Say what you need and Captain plans the team.')}
              </p>
            </div>
            <Steps current={step} working={working} />
            <div className="flex shrink-0 items-center gap-0.5">
              {plan && (
                <button onClick={startOver} aria-label="New request" title="New request. This plan stays under Earlier plans." className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                  <Plus className="h-4 w-4" />
                </button>
              )}
              <FullscreenButton />
              <button onClick={onClose} aria-label="Close" title={working ? 'Close. It carries on without this open.' : 'Close'} className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                <X className="h-[18px] w-[18px]" />
              </button>
            </div>
          </div>
          {isMobile && (
            <div className="flex shrink-0 gap-1 px-4 pt-2" aria-hidden>
              {STEPS.map((s, i) => <span key={s} className={`h-1 flex-1 rounded-full transition-colors ${i <= step ? 'bg-violet-500' : 'bg-zinc-800'}`} />)}
            </div>
          )}
          {body}
        </DialogFullscreen.Provider>
      </div>
    </div>,
    document.body,
  )
}
