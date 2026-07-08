import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Network, Play, Sparkles, Check, X, Paperclip, FileText, Image as ImageIcon,
  SlidersHorizontal, ScanSearch, RefreshCw, CornerDownRight, Users, HelpCircle,
  Gauge, ListChecks, Brain, FolderSearch, Loader2, ChevronDown, ChevronRight,
} from 'lucide-react'
import { useBasnaStore, parseAnalysis, apiVatraSkipAgent, type FolderMode } from '../stores/basnaStore'
import { VatraTeamPlan } from '../components/VatraDelegation'
import { QualityControls } from '../components/QualityControls'
import { BasnaWizardModal } from '../components/agents/BasnaWizardModal'
import { KnowledgePicker } from '../components/agents/KnowledgePicker'
import { ReferenceFolderPicker } from '../components/agents/ReferenceFolderPicker'
import { deriveCost } from '../components/CostCard'
import { useTierConfig, TIER_ORDER } from '../services/tierConfig'
import {
  buildLiveAgents, FileModal, isVatra, parentIdOf, runVfsProject, type ViewMode,
} from '../components/basna/shared'
import { RunsSidebar } from '../components/basna/RunsSidebar'
import { RoutePlanEditor } from '../components/basna/RoutePlanEditor'
import { RunWorkspace } from '../components/basna/RunWorkspace'
import { RunReport } from '../components/basna/RunReport'

const HELP_MD = `
# Basna, Vatra & Deep mode

Two ways to put a **fleet of specialist agents** on one task — plus an optional
**Deep** mode that spends extra compute to reach frontier‑grade quality on cheaper models.

---

## Basna — the independent ensemble

A router picks the **minimal set of specialist archetypes** for your task, spawns them
fresh, runs them **in parallel and blind to each other**, then **merges** their answers
weighted by each archetype's learned reliability. An LLM only synthesises when the
answers genuinely conflict.

- **Best for:** truth‑finding — "what's true", options, verification, analysis.
- **Why it works:** independent experts make *uncorrelated* errors, so merging cancels them.
- **It learns:** every run scores each archetype, so routing and weighting improve over time.

**Route → review the team → Run ensemble.**

---

## Vatra — the collaborating team

A **Lead** decomposes the task into owned sub‑pieces; specialists work **on a shared
blackboard** and can **delegate sub‑questions to each other** without blocking; a dedicated
**reporter** assembles everything into one deliverable.

- **Best for:** building a **multi‑part artifact** whose pieces depend on each other
  (a design doc, a report, a plan).
- **Difference from Basna:** Basna *merges* independent answers; Vatra *collaborates* toward
  one stitched result.

**Plan team → review → Run team.**

---

## ⚡ Deep mode (Frontier Horizon)

Off by default. When on, the system **spends test‑time compute, gated by verifiers**, to
simulate a much stronger model. Three layers:

1. **Per‑worker depth.**
   - *Basna:* each worker answers **N times** (self‑consistency vote) and a panel of
     **adversarial critics** (distinct cognitive lenses) must not refute it; a failed check
     triggers a fix pass. Set the **samples** number to widen the vote.
   - *Vatra:* each specialist's slice is **verified by the critic panel and revised once**
     if refuted (no vote — that would flood the blackboard).
2. **The closer.** After the answer/deliverable is assembled, the critic panel reviews the
   **final** result and **revises it once** if a majority refute it — the self‑correction
   step a single pass lacks.
3. **Critics never grade themselves** — they run on a *different* model than the worker.

> Deep mode is **much slower and costlier** (many extra model calls). Use it when quality
> matters more than speed/cost.

---

## 🧭 Plan mode (Basna) — think way ahead

A planner breaks the task into **ordered steps**; each step is **driven to a verified
result before the next begins**; a step that can't be verified triggers a **re‑plan**; the
deliverable is synthesised from the verified steps. This is the long‑horizon lever — the
system plans, checks itself, and recovers instead of charging ahead.

- **Steps engine** — how each step runs:
  - **single** — one generation per step (fast).
  - **Basna ensemble** — a full Basna fleet per step.
  - **Vatra team** — a full Vatra team per step (strongest, slowest).
- **parallel** — the planner emits a **dependency graph**, so independent steps run at the
  same time and each step sees only what it depends on.

---

### Rule of thumb
| Want… | Use |
|---|---|
| The true answer / options, fast | **Basna** |
| One assembled multi‑part artifact | **Vatra** |
| Frontier‑grade quality, cost no object | add **Deep** |
| A hard task that needs multi‑step reasoning | **Plan** |
`.trim()

// Demo / debug tasks — each exercises a different path through the pipeline.
const BASNA_EXAMPLES: { label: string; text: string }[] = [
  {
    label: 'EU expansion (complex)',
    text: `We're a 20-person B2B SaaS selling AI-powered contract-review software to US mid-market law firms (~$2M ARR, growing ~8%/month). Decide whether we should expand into the EU in the next two quarters.

Weigh, with evidence: (1) GDPR and the EU AI Act's implications for a legal-AI tool — obligations, timelines, and risk classification; (2) the competitive landscape in DACH and France — who's already there and how we'd differentiate; (3) data-residency and localization costs (hosting, language, support, legal); (4) the realistic revenue opportunity versus the distraction risk to our US growth.

End with ONE clear recommendation (go / no-go / phased) and the three specific conditions that would flip the decision.`,
  },
  {
    label: 'Data store choice',
    text: "We're adding an append-only event log: ~50M records/month, written continuously, queried by time-range and user_id, with daily aggregations. Pick ONE data store — Postgres, ClickHouse, or DynamoDB — and give the single most important reason.",
  },
  {
    label: 'Quick fact',
    text: 'What does the SQL keyword EXPLAIN do? One sentence.',
  },
  {
    label: 'Brainstorm options',
    text: 'Brainstorm ways to cut cold-start latency for a serverless API. List distinct approaches — breadth over depth.',
  },
  {
    label: 'Migration risk',
    text: 'Name the single biggest risk in migrating a monolith to microservices.',
  },
]

/**
 * R12 intent brief: the clarified, editable task the team was routed on. Editing
 * it and re-routing re-selects the team against the new brief. The original
 * request always governs — this is a faithful clarification, never a rewrite of
 * the goal. Shown only when the intent_brief lever produced a brief.
 */
function BriefEditor({ brief, busy, onReroute }: {
  brief: string; busy: boolean; onReroute: (edited: string) => void
}) {
  const [draft, setDraft] = useState(brief)
  const [open, setOpen] = useState(true)
  useEffect(() => { setDraft(brief) }, [brief])
  const changed = draft.trim() !== brief.trim()
  return (
    <div className="rounded-lg border border-amber-300/70 bg-amber-50/70 p-4 dark:border-amber-700/40 dark:bg-amber-900/10">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <Sparkles className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400" />
        <span className="text-xs font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-300">Task brief</span>
        <span className="text-[11px] text-zinc-500">the team was selected from this — edit &amp; re-route to re-pick it</span>
        <button onClick={() => setOpen((o) => !o)} className="ml-auto text-[11px] text-zinc-500 hover:text-zinc-300">
          {open ? 'Hide' : 'Show'}
        </button>
      </div>
      {open && (
        <>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={Math.min(18, Math.max(6, draft.split('\n').length + 1))}
            spellCheck={false}
            className="w-full resize-y rounded-md border border-zinc-300 bg-white/70 p-2.5 font-mono text-[11px] leading-relaxed text-zinc-800 focus:border-amber-500 focus:outline-none dark:border-zinc-700 dark:bg-zinc-950/60 dark:text-zinc-200"
          />
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <button
              onClick={() => onReroute(draft)}
              disabled={busy || !draft.trim()}
              className="flex items-center gap-1.5 rounded-lg bg-amber-500 px-3 py-1.5 text-xs font-medium text-zinc-950 hover:bg-amber-400 disabled:opacity-40"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              Re-route on this brief
            </button>
            {changed && <span className="text-[11px] text-amber-600 dark:text-amber-400">edited — re-route to apply</span>}
            <span className="ml-auto text-[10px] text-zinc-500">the original request always governs on conflict</span>
          </div>
        </>
      )}
    </div>
  )
}

// ── Stage rail: the four collapsing stages of a run ──────────────────────────

type Stage = 'define' | 'plan' | 'run' | 'done'
type PanelId = 'define' | 'setup' | 'plan'

function StageChip({ n, label, summary, state, busy, open, onClick }: {
  n: number
  label: string
  summary?: string
  state: 'done' | 'active' | 'idle' | 'failed'
  busy?: boolean   // show a spinner (only the Run chip while live)
  open?: boolean
  onClick?: () => void
}) {
  const tone = state === 'done'
    ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
    : state === 'active'
      ? 'border-violet-500/50 bg-violet-500/15 text-violet-700 dark:text-violet-300'
      : state === 'failed'
        ? 'border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300'
        : 'border-zinc-700 bg-zinc-900/50 text-zinc-500'
  return (
    <button
      onClick={onClick}
      disabled={!onClick}
      className={`flex min-w-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors ${tone} ${onClick ? 'hover:brightness-110' : 'cursor-default'}`}
    >
      {busy
        ? <Loader2 className="h-3 w-3 shrink-0 animate-spin" />
        : state === 'done'
          ? <Check className="h-3 w-3 shrink-0" />
          : state === 'failed'
            ? <X className="h-3 w-3 shrink-0" />
            : <span className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full text-[9px] ${
                state === 'active' ? 'bg-violet-500/30' : 'bg-zinc-800 text-zinc-400'}`}>{n}</span>}
      {label}
      {summary && <span className="max-w-52 truncate font-normal opacity-70">· {summary}</span>}
      {onClick && (open ? <ChevronDown className="h-3 w-3 shrink-0 opacity-60" /> : <ChevronRight className="h-3 w-3 shrink-0 opacity-60" />)}
    </button>
  )
}

const QUALITY_LABEL: Record<string, string> = { off: 'Basic', balanced: 'Balanced', thorough: 'Thorough', custom: 'Custom' }

export function BasnaPage() {
  const {
    sessions, activeSession, routePlan, runs, lastExecute, progress, attachments,
    routing, planning, executing, recompiling, error,
    routerTier, maxAgents, setRouterTier, setMaxAgents, maxParallel, setMaxParallel, executionGroups, setExecutionGroups, sharedDatastore, setSharedDatastore, folderMode, setFolderMode, newFolderName, setNewFolderName, existingFolder, setExistingFolder, projects, projectsLoading, loadProjects, knowledgeSessionIds, toggleKnowledgeSession, knowledgeIncludeBoard, setKnowledgeIncludeBoard, referenceFolders, toggleReferenceFolder, deep, deepSamples, setDeep, setDeepSamples, planMode, planSteps, setPlanMode, setPlanSteps, planComplex, setPlanComplex, planDag, setPlanDag, runPlan, quality, setQuality, addFiles, removeFile,
    updateSelected, updateSubtask, removeSubtask, setGroupInstruction, loadSessions, pollRunning, selectSession, newSession, route, planVatra, runVatra, fillGaps, saveTitle, execute, recompile, sendFeedback, deleteSession, cancelSession, deepenSession, continueSession,
  } = useBasnaStore()
  const { tiers, registry, envVars } = useTierConfig()

  const [intent, setIntent] = useState('')
  const [title, setTitle] = useState('')
  // Compose mode for a NEW run; a selected run locks to its own mode (effectiveMode).
  const [composeMode, setComposeMode] = useState<'basna' | 'vatra'>(
    () => ((typeof localStorage !== 'undefined' && localStorage.getItem('basna.composeMode')) as 'basna' | 'vatra') || 'basna',
  )
  // Optional user-fixed team: archetype ids the route/plan MUST use (empty = auto).
  const [team, setTeam] = useState<string[]>([])
  const [teamOpen, setTeamOpen] = useState(false)
  const [tuning, setTuning] = useState(false)  // router tier + max agents disclosure
  const toggleTeam = (id: string) =>
    setTeam((t) => (t.includes(id) ? t.filter((x) => x !== id) : [...t, id]))
  const [deepening, setDeepening] = useState(false)
  const [modal, setModal] = useState<{ title: string; content: string; mode: ViewMode } | null>(null)
  const viewFull = (t: string, content: string) => setModal({ title: t, content, mode: 'markdown' })
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const handlePaste = (e: React.ClipboardEvent) => {
    const imgs: File[] = []
    for (const it of Array.from(e.clipboardData?.items || [])) {
      if (it.type.startsWith('image/')) {
        const f = it.getAsFile()
        if (f) {
          const ext = it.type.split('/')[1] || 'png'
          imgs.push(f.name && f.name !== 'image.png' ? f : new File([f], `pasted-${Date.now()}.${ext}`, { type: it.type }))
        }
      }
    }
    if (imgs.length) addFiles(imgs)
  }

  useEffect(() => { loadSessions() }, [loadSessions])
  // Prefetch VFS folders once so the "Existing folder" picker is populated the
  // moment the user opens it (a native <select> won't repaint options that
  // arrive while it's already open).
  useEffect(() => { loadProjects() }, [loadProjects])
  // Load a selected run's intent/title, but DON'T wipe the textarea when the
  // selection is cleared (New and mode-switch handle clearing explicitly) — this
  // lets switching modes keep what you've typed.
  useEffect(() => { if (activeSession) setIntent(activeSession.intent || '') }, [activeSession?.id, activeSession?.intent])
  useEffect(() => { if (activeSession) setTitle(activeSession.title || '') }, [activeSession?.id, activeSession?.title])

  // Live monitor: while any run (incl. agent-started) is mid-flight, poll the
  // list status + the open session's progress every few seconds; stop when idle.
  const anyRunning = sessions.some((s) => ['routing', 'routed', 'running'].includes(s.status))
  // Vatra runs render the collaboration panel, not the Basna route-plan editor
  // (their route's `selected` carries no tier/prior_weight). Read from config so
  // it's correct even before the Lead has decomposed (route is still empty).
  const vatraMode = !!activeSession && isVatra(activeSession.config)
  // The mode the controls act in: a selected run is locked to its own mode; with
  // no run selected, the toggle (composeMode) decides.
  const effectiveMode: 'basna' | 'vatra' = activeSession ? (vatraMode ? 'vatra' : 'basna') : composeMode
  // Plan step engine: simple = one model per step; complex = a full run of the
  // selected mode (Basna ensemble / Vatra team) per step.
  const planStepMode: 'llm' | 'ensemble' | 'vatra' =
    !planComplex ? 'llm' : effectiveMode === 'vatra' ? 'vatra' : 'ensemble'
  // The three effort levels are mutually exclusive; derive one from the two flags.
  const strategy: 'standard' | 'deep' | 'plan' = planMode ? 'plan' : deep ? 'deep' : 'standard'
  const setStrategy = (sId: 'standard' | 'deep' | 'plan') => {
    if (sId === 'standard') { setDeep(false); setPlanMode(false) }
    else if (sId === 'deep') setDeep(true)   // setter clears Plan
    else setPlanMode(true)                   // setter clears Deep
  }
  // Keep the toggle in sync with whatever run is open.
  useEffect(() => {
    if (activeSession) setComposeMode(isVatra(activeSession.config) ? 'vatra' : 'basna')
  }, [activeSession?.id, activeSession?.config])
  const pickMode = (m: 'basna' | 'vatra') => {
    if (m === effectiveMode) return
    setComposeMode(m)
    try { localStorage.setItem('basna.composeMode', m) } catch { /* ignore */ }
    // A run is locked to its mode, so switching means composing a fresh one — but
    // keep the typed intent/title (the cleared selection no longer wipes them).
    if (activeSession) newSession()
  }
  const [wizardOpen, setWizardOpen] = useState(false)
  const [showKnowledge, setShowKnowledge] = useState(false)
  const [showRefFolders, setShowRefFolders] = useState(false)
  // Wizard handoff: set the task + mode, then prepare (route/plan) — the user
  // reviews the plan and runs it via the normal panel.
  const onWizardPrepare = ({ intent: wi, title: wt, mode }: { intent: string; title: string; mode: 'basna' | 'vatra' }) => {
    if (!wi.trim()) return
    newSession()
    setIntent(wi); setTitle(wt); setTeam([])
    setComposeMode(mode)
    try { localStorage.setItem('basna.composeMode', mode) } catch { /* ignore */ }
    setWizardOpen(false)
    if (mode === 'vatra') planVatra(wi, tiers, wt, [])
    else route(wi, tiers, wt, [])
  }
  // Creds for the wizard's recommend call — same tier the router uses.
  const _rtc = tiers[routerTier]
  const wizardCreds = _rtc?.model
    ? { provider: _rtc.provider, model: _rtc.model, api_key: _rtc.api_key || undefined, base_url: _rtc.base_url || undefined }
    : {}
  useEffect(() => {
    if (!anyRunning || executing) return
    const iv = setInterval(() => { pollRunning() }, 4000)
    return () => clearInterval(iv)
  }, [anyRunning, executing, pollRunning])

  // A run already in flight (e.g. a deepen that route+ran server-side) must not
  // be re-routed or re-run. 'routed' stays runnable — that's the normal Route→Run step.
  const activeBusy = !!activeSession && (activeSession.status === 'running' || activeSession.status === 'routing')
  const canRoute = intent.trim().length > 0 && !routing && !planning && !activeBusy
  const canRun = !!routePlan && !!activeSession && !executing && !planning && !activeBusy
  // While a team is being summoned (routing/planning/running), lock team selection.
  const teamLocked = routing || planning || executing || activeBusy
  // Collapse the team selector the moment a route/plan/run starts.
  useEffect(() => { if (teamLocked) setTeamOpen(false) }, [teamLocked])
  const truth = lastExecute?.truth ?? activeSession?.truth ?? ''
  const confidence = lastExecute?.confidence ?? activeSession?.confidence ?? 0
  const analysis = lastExecute?.analysis ?? parseAnalysis(activeSession?.analysis)
  // Run cost (tokens + $ + $/hour): from the terminal `cost` progress event, or
  // the execute response. Shown once a run has finished.
  const runCost = useMemo(() => deriveCost(progress, lastExecute?.cost), [progress, lastExecute])
  // Subject for download filenames: the run's title, else the first words of the
  // task — so analysis/truth export as "<subject>-analysis.md" / "…-compiled-truth.md".
  const subject = (activeSession?.title || '').trim()
    || (activeSession?.intent || '').trim().split(/\s+/).slice(0, 8).join(' ')
    || 'basna'

  const liveAgents = useMemo(() => buildLiveAgents(progress), [progress])

  // The active high-level stage — the most recent `phase` banner the backend
  // emitted (Planning / Intro / Main / Synthesizing / Step x/y …). Shown stickily
  // in the run header so it never scrolls out of view under action spam.
  const currentPhase = useMemo<string | null>(() => {
    for (let i = progress.length - 1; i >= 0; i--) {
      if (progress[i].stage === 'phase') return progress[i].message
    }
    return null
  }, [progress])

  // The VFS folder this run reads/writes — for the Files + Datastore panels.
  const runProject = useMemo(
    () => (activeSession ? runVfsProject(activeSession, progress) : ''),
    [activeSession?.id, activeSession?.config, progress.length], // eslint-disable-line react-hooks/exhaustive-deps
  )

  // ── The stage machine: define → plan → run → done ──────────────────────────
  const running = executing || activeBusy
  const finished = !!activeSession && (activeSession.status === 'done' || activeSession.status === 'error')
  const stage: Stage = running ? 'run' : finished ? 'done' : routePlan ? 'plan' : 'define'

  // Which panels are expanded. Each stage transition sets sensible defaults;
  // the chips let the user reopen any collapsed stage at any time.
  const [open, setOpen] = useState<Record<PanelId, boolean>>({ define: true, setup: false, plan: true })
  const prevStage = useRef<Stage | null>(null)
  useEffect(() => {
    if (prevStage.current === stage) return
    prevStage.current = stage
    if (stage === 'define') setOpen({ define: true, setup: false, plan: false })
    else if (stage === 'plan') setOpen({ define: false, setup: false, plan: true })
    else setOpen({ define: false, setup: false, plan: false })
  }, [stage])
  const toggle = (p: PanelId) => setOpen((o) => ({ ...o, [p]: !o[p] }))

  // Also collapse everything when switching between sessions in the same stage
  // family (e.g. two finished runs) so the report stays front and center.
  useEffect(() => {
    if (stage === 'done' || stage === 'run') setOpen({ define: false, setup: false, plan: false })
  }, [activeSession?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const planAgentCount = vatraMode ? (routePlan?.subtasks?.length || 0) : (routePlan?.selected.length || 0)
  const setupSummary = [
    effectiveMode === 'vatra' ? 'Vatra' : 'Basna',
    strategy[0].toUpperCase() + strategy.slice(1),
    QUALITY_LABEL[quality.profile] || quality.profile,
    ...(knowledgeSessionIds.length ? [`${knowledgeSessionIds.length} prior`] : []),
    ...(referenceFolders.length ? [`${referenceFolders.length} ref`] : []),
    ...(team.length ? [`${team.length} pinned`] : []),
  ].join(' · ')

  const onDelete = (s: typeof sessions[number]) => {
    const raw = (s.title || s.intent || '').trim()
    const label = raw.slice(0, 80)
    if (window.confirm(`Delete this Basna run?${label ? `\n\n"${label}${raw.length > 80 ? '…' : ''}"` : ''}`)) {
      deleteSession(s.id)
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex shrink-0 items-center gap-2 border-b border-zinc-700/50 bg-zinc-900/50 px-4 py-3 md:px-6">
        <Network className="h-5 w-5 text-sky-600 dark:text-sky-400" />
        <div>
          <h1 className="text-sm font-semibold text-zinc-100">Basna</h1>
          <p className="text-[11px] text-zinc-500">Route → spawn the minimal team → merge by reliability</p>
        </div>
        <button
          onClick={() => viewFull('Basna, Vatra & Deep mode', HELP_MD)}
          title="What are Basna, Vatra and Deep mode?"
          className="ml-auto flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800"
        >
          <HelpCircle className="h-3.5 w-3.5" /> Help
        </button>
      </div>
      {wizardOpen && (
        <BasnaWizardModal
          sessions={sessions}
          creds={wizardCreds}
          onClose={() => setWizardOpen(false)}
          onPrepare={onWizardPrepare}
        />
      )}

      <div className="flex flex-1 overflow-hidden">
        <RunsSidebar
          sessions={sessions}
          activeId={activeSession?.id}
          onSelect={selectSession}
          onDelete={onDelete}
          onCancel={cancelSession}
          onNew={() => { newSession(); setIntent(''); setTitle(''); setTeam([]) }}
          onWizard={() => setWizardOpen(true)}
        />

        {/* Detail */}
        <div className="flex-1 overflow-auto p-4 md:p-6">
          <div className="mx-auto w-[92%] max-w-[2000px] space-y-3">
            {/* Deepen lineage — jump to the parent / child runs of this one. */}
            {activeSession && (() => {
              const pid = parentIdOf(activeSession.config)
              const parent = pid ? sessions.find((s) => s.id === pid) : null
              const children = sessions.filter((s) => parentIdOf(s.config) === activeSession.id)
              if (!parent && children.length === 0) return null
              const linkCls = 'flex items-center gap-1 text-left text-violet-700 hover:underline dark:text-violet-300'
              return (
                <div className="flex flex-col gap-1 rounded-lg border border-violet-300 bg-violet-50 px-3 py-2 text-[11px] dark:border-violet-900/40 dark:bg-violet-950/20">
                  {parent && (
                    <button onClick={() => selectSession(parent.id)} className={linkCls}>
                      <CornerDownRight className="h-3 w-3 shrink-0" />
                      <span className="truncate">deepened from “{parent.title || parent.intent || 'run'}”</span>
                    </button>
                  )}
                  {children.map((ch) => (
                    <button key={ch.id} onClick={() => selectSession(ch.id)} className={linkCls}>
                      <ScanSearch className="h-3 w-3 shrink-0" />
                      <span className="truncate">
                        deepened into “{ch.title || ch.intent || 'run'}”{ch.status !== 'done' ? ` · ${ch.status}` : ''}
                      </span>
                    </button>
                  ))}
                </div>
              )
            })()}

            {/* Stage rail — the run's lifecycle at a glance; click a chip to reopen it. */}
            <div className="flex flex-wrap items-center gap-1.5">
              <StageChip
                n={1} label="Task"
                state={intent.trim() ? 'done' : stage === 'define' ? 'active' : 'idle'}
                summary={title.trim() || (intent.trim() ? intent.trim().split(/\s+/).slice(0, 5).join(' ') : undefined)}
                open={open.define}
                onClick={() => toggle('define')}
              />
              <ChevronRight className="h-3 w-3 text-zinc-700" />
              <StageChip
                n={2} label="Setup"
                state={intent.trim() ? 'done' : 'idle'}
                summary={setupSummary}
                open={open.setup}
                onClick={() => toggle('setup')}
              />
              <ChevronRight className="h-3 w-3 text-zinc-700" />
              <StageChip
                n={3} label="Plan"
                state={routePlan ? (stage === 'plan' ? 'active' : 'done') : 'idle'}
                summary={routePlan ? `${planAgentCount || routePlan.selected.length} agents` : 'optional'}
                open={open.plan}
                onClick={routePlan ? () => toggle('plan') : undefined}
              />
              <ChevronRight className="h-3 w-3 text-zinc-700" />
              <StageChip
                n={4} label="Run"
                state={stage === 'run' ? 'active' : stage === 'done' ? (activeSession?.status === 'error' ? 'failed' : 'done') : 'idle'}
                busy={stage === 'run'}
                summary={stage === 'run' ? (currentPhase || 'working') : stage === 'done' ? (activeSession?.status === 'error' ? 'failed' : 'finished') : undefined}
              />
            </div>

            {/* Stage 1 — Task: title, the task itself, attachments. */}
            {open.define && (
              <div
                className={`rounded-lg border bg-zinc-900/50 p-4 ${dragOver ? 'border-sky-500' : 'border-zinc-800'}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files) }}
              >
                <label className="mb-1.5 block text-xs font-medium text-zinc-400">
                  Title <span className="font-normal text-zinc-600">— optional, auto-generated from the task if blank</span>
                </label>
                <div className="mb-3 flex items-center gap-2">
                  <input
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="e.g. Q3 competitor scan"
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
                  />
                  {activeSession && title.trim() !== (activeSession.title || '') && (
                    <button
                      onClick={() => saveTitle(title)}
                      title="Save title"
                      className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800"
                    >
                      <Check className="h-3.5 w-3.5" /> Save
                    </button>
                  )}
                </div>
                <label className="mb-1.5 block text-xs font-medium text-zinc-400">Task / intent</label>
                <textarea
                  value={intent}
                  onChange={(e) => setIntent(e.target.value)}
                  onPaste={handlePaste}
                  rows={7}
                  placeholder="Describe the task, or attach/drop/paste files. The router picks the smallest team that can answer it well."
                  className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 p-2.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
                />

                {/* Attachments */}
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
                  >
                    <Paperclip className="h-3.5 w-3.5" /> Attach
                  </button>
                  <input
                    ref={fileInputRef} type="file" multiple className="hidden"
                    onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }}
                  />
                  {attachments.filter((a) => a.kind !== 'generated').map((a) => (
                    <span key={a.name} className="flex items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-800/60 px-2 py-0.5 text-[11px] text-zinc-300">
                      {a.mime.startsWith('image/') ? <ImageIcon className="h-3 w-3 text-zinc-500" /> : <FileText className="h-3 w-3 text-zinc-500" />}
                      {a.name}
                      <span className="text-zinc-600">{Math.max(1, Math.round(a.size / 1024))}kb</span>
                      <button onClick={() => removeFile(a.name)} className="text-zinc-500 hover:text-rose-400"><X className="h-3 w-3" /></button>
                    </span>
                  ))}
                </div>
                {!intent.trim() && (
                  <div className="mt-2 flex flex-wrap items-center gap-1.5">
                    <span className="text-[11px] text-zinc-600">Try:</span>
                    {BASNA_EXAMPLES.map((ex) => (
                      <button
                        key={ex.label}
                        onClick={() => setIntent(ex.text)}
                        title={ex.text}
                        className="rounded-full border border-sky-300 bg-sky-50 px-2.5 py-1 text-[11px] text-sky-700 transition-colors hover:bg-sky-100 dark:border-sky-500/30 dark:bg-sky-500/10 dark:text-sky-300 dark:hover:bg-sky-500/20"
                      >
                        {ex.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Stage 2 — Setup: mode, effort, quality, knowledge, folders, tuning. */}
            {open.setup && (
              <div className="space-y-3 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                {/* Mode selector — the two ways to run a team, clearly distinguished. */}
                <div className="grid grid-cols-2 gap-2">
                  {([
                    { id: 'basna', icon: Network, name: 'Basna', sub: 'Independent ensemble — agents answer blind, merged by reliability', on: 'border-sky-400 bg-sky-50 dark:border-sky-500/70 dark:bg-sky-950/30', dot: 'text-sky-600 dark:text-sky-400' },
                    { id: 'vatra', icon: Users, name: 'Vatra', sub: 'Collaborative team — a Lead splits the work, a reporter assembles it', on: 'border-violet-400 bg-violet-50 dark:border-violet-500/70 dark:bg-violet-950/30', dot: 'text-violet-600 dark:text-violet-400' },
                  ] as const).map((m) => {
                    const Icon = m.icon
                    const sel = effectiveMode === m.id
                    return (
                      <button
                        key={m.id}
                        onClick={() => pickMode(m.id)}
                        className={`flex items-start gap-2 rounded-lg border p-2.5 text-left transition-colors ${
                          sel ? m.on : 'border-zinc-800 bg-zinc-900/40 hover:bg-zinc-800/40'}`}
                      >
                        <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${sel ? m.dot : 'text-zinc-500'}`} />
                        <span className="min-w-0">
                          <span className={`block text-xs font-semibold ${sel ? 'text-zinc-100' : 'text-zinc-300'}`}>
                            {m.name}{sel && <span className="ml-1.5 text-[10px] font-normal text-zinc-500">selected</span>}
                          </span>
                          <span className="block text-[11px] leading-snug text-zinc-500">{m.sub}</span>
                        </span>
                      </button>
                    )
                  })}
                </div>

                {/* Effort / strategy */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Effort</span>
                  <div className="inline-flex rounded-lg border border-zinc-700 bg-zinc-900/50 p-0.5">
                    {([
                      { id: 'standard', label: 'Standard', icon: Gauge, on: 'bg-sky-600 text-white' },
                      { id: 'deep', label: 'Deep', icon: ScanSearch, on: 'bg-amber-500 text-zinc-950' },
                      { id: 'plan', label: 'Plan', icon: ListChecks, on: 'bg-emerald-600 text-white' },
                    ] as const).map((st) => {
                      const Icon = st.icon
                      const sel = strategy === st.id
                      return (
                        <button
                          key={st.id}
                          onClick={() => setStrategy(st.id)}
                          className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                            sel ? st.on : 'text-zinc-400 hover:text-zinc-200'}`}
                        >
                          <Icon className="h-3.5 w-3.5" /> {st.label}
                        </button>
                      )
                    })}
                  </div>
                  <button
                    onClick={() => setTuning((t) => !t)}
                    title="Router tier and team size"
                    className={`ml-auto flex items-center gap-1.5 rounded-md px-2 py-1 text-xs ${
                      tuning ? 'bg-zinc-800 text-zinc-200' : 'text-zinc-500 hover:text-zinc-300'}`}
                  >
                    <SlidersHorizontal className="h-3.5 w-3.5" /> Tuning
                  </button>
                </div>

                {/* What the chosen effort does + its options */}
                <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 px-3 py-2.5 text-xs text-zinc-400">
                  {strategy === 'standard' && (
                    <p className="leading-relaxed">
                      A minimal team answers once — {effectiveMode === 'vatra'
                        ? 'a Lead splits the work and a reporter assembles it.'
                        : 'independent agents, merged by reliability.'}
                    </p>
                  )}
                  {strategy === 'deep' && (
                    <div className="space-y-2">
                      <p className="leading-relaxed">
                        A diverse critic panel reviews each answer and revises it if refuted{effectiveMode === 'basna'
                          ? ', and every agent self-consistency-votes' : ''}. Much stronger, much slower &amp; costlier.
                      </p>
                      {effectiveMode === 'basna' && (
                        <label className="flex items-center gap-2 text-zinc-400">
                          Samples / agent
                          <input
                            type="number" min={2} max={8} value={deepSamples}
                            onChange={(e) => setDeepSamples(Number(e.target.value))}
                            title="Self-consistency vote width per agent"
                            className="w-14 rounded border border-amber-700/50 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-amber-500 focus:outline-none"
                          />
                        </label>
                      )}
                    </div>
                  )}
                  {strategy === 'plan' && (
                    <div className="space-y-2.5">
                      <p className="leading-relaxed">
                        Break the task into steps, drive each to a <span className="text-zinc-300">verified</span> result before the next, re-plan on failure, then synthesize.
                      </p>
                      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
                        <label className="flex items-center gap-2 text-zinc-400">
                          Steps
                          <input
                            type="number" min={1} max={12} value={planSteps}
                            onChange={(e) => setPlanSteps(Number(e.target.value))}
                            title="Max steps in the plan"
                            className="w-14 rounded border border-emerald-700/50 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-emerald-500 focus:outline-none"
                          />
                        </label>
                        <div className="flex items-center gap-2 text-zinc-400">
                          Each step
                          <div className="inline-flex rounded-md border border-zinc-700 bg-zinc-950/50 p-0.5">
                            {([
                              { complex: false, label: 'simple' },
                              { complex: true, label: effectiveMode === 'vatra' ? 'Vatra team' : 'Basna ensemble' },
                            ]).map((opt) => (
                              <button
                                key={opt.label}
                                onClick={() => setPlanComplex(opt.complex)}
                                title={opt.complex
                                  ? `Each step runs a full ${effectiveMode === 'vatra' ? 'Vatra team' : 'Basna ensemble'} — strongest, slowest.`
                                  : 'One fast model per step.'}
                                className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors ${
                                  planComplex === opt.complex ? 'bg-emerald-600 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}
                              >
                                {opt.label}
                              </button>
                            ))}
                          </div>
                        </div>
                        <label
                          className="flex items-center gap-1.5 text-zinc-400"
                          title="Planner emits a dependency graph — independent steps run in parallel, each seeing only what it depends on."
                        >
                          <input
                            type="checkbox" checked={planDag}
                            onChange={(e) => setPlanDag(e.target.checked)}
                            className="accent-emerald-500"
                          />
                          parallel
                        </label>
                      </div>
                    </div>
                  )}
                </div>

                {/* Quality levers (opt-in cross-pollination) */}
                <QualityControls scope="research" value={quality} onChange={setQuality} />

                {/* Prior-run knowledge (opt-in) — seed this run with earlier reports + gaps */}
                <div>
                  <button
                    onClick={() => setShowKnowledge((v) => !v)}
                    className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                      knowledgeSessionIds.length > 0
                        ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-300'
                        : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'
                    }`}
                    title="Use the knowledge (report + gaps/blind spots) of prior finished runs"
                  >
                    <Brain className="h-3.5 w-3.5" />
                    {knowledgeSessionIds.length > 0 ? `Prior knowledge: ${knowledgeSessionIds.length}` : 'Use prior-run knowledge'}
                  </button>
                  {showKnowledge && (
                    <div className="mt-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                      <KnowledgePicker
                        sessions={sessions}
                        selectedIds={knowledgeSessionIds}
                        onToggle={toggleKnowledgeSession}
                        includeBoard={knowledgeIncludeBoard}
                        onIncludeBoard={setKnowledgeIncludeBoard}
                      />
                    </div>
                  )}
                </div>

                {/* Reference folders (read-only) — agents check these before web search */}
                <div>
                  <button
                    onClick={() => { setShowRefFolders((v) => !v); if (projects.length === 0) loadProjects() }}
                    className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                      referenceFolders.length > 0
                        ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                        : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'
                    }`}
                    title="Read-only VFS folders agents search before web search (prior-knowledge runs' folders are auto-included)"
                  >
                    <FolderSearch className="h-3.5 w-3.5" />
                    {referenceFolders.length > 0 ? `Reference folders: ${referenceFolders.length}` : 'Reference folders (read-only)'}
                  </button>
                  {showRefFolders && (
                    <div className="mt-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                      <ReferenceFolderPicker projects={projects} selected={referenceFolders} onToggle={toggleReferenceFolder} />
                    </div>
                  )}
                </div>

                {/* Tuning (router tier + team size) */}
                {tuning && (
                  <div className="flex flex-wrap items-center gap-5 rounded-lg border border-zinc-800 bg-zinc-900/30 px-3 py-2 text-xs">
                    <label className="flex items-center gap-2 text-zinc-400">
                      Router tier
                      <select
                        value={routerTier}
                        onChange={(e) => setRouterTier(e.target.value)}
                        title="Which Library tier picks the team / leads it"
                        className="rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-sky-600 focus:outline-none"
                      >
                        {TIER_ORDER.filter((t) => tiers[t]).map((t) => (
                          <option key={t} value={t}>{registry?.tiers[t]?.label || t}</option>
                        ))}
                        {Object.keys(tiers).length === 0 && <option value="reason">reason</option>}
                      </select>
                    </label>
                    <label className="flex items-center gap-2 text-zinc-400">
                      Max agents
                      <input
                        type="number" min={1} max={10} value={maxAgents}
                        onChange={(e) => setMaxAgents(Number(e.target.value))}
                        disabled={team.length > 0}
                        title={team.length > 0 ? 'Ignored — the team is fixed by your selection' : 'Team size cap'}
                        className="w-16 rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-sky-600 focus:outline-none disabled:opacity-40"
                      />
                    </label>
                    <label className="flex items-center gap-2 text-zinc-400">
                      Max parallel
                      <input
                        type="number" min={0} max={16} value={maxParallel}
                        onChange={(e) => setMaxParallel(Number(e.target.value))}
                        title="How many agents run their turn at once. 0 = all at once. Lower it (e.g. 2) for local models to avoid running the serving box out of memory."
                        className="w-16 rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-sky-600 focus:outline-none"
                      />
                    </label>
                    {effectiveMode === 'vatra' && (
                      <label
                        className="flex cursor-pointer items-center gap-1.5 text-zinc-400"
                        title="Run owners in ordered phases A→B→C→D (research/design first, review/assembly last) with a barrier between groups, instead of all at once. Opt-in."
                      >
                        <input
                          type="checkbox" checked={executionGroups}
                          onChange={(e) => setExecutionGroups(e.target.checked)}
                          className="h-3.5 w-3.5 rounded border-zinc-700 bg-zinc-950/60 accent-violet-600"
                        />
                        Grouped
                      </label>
                    )}
                    {/* Run folder — new (auto/custom name) or an existing VFS folder */}
                    <label
                      className="flex items-center gap-2 text-zinc-400"
                      title="Where this run's agents read/write shared files (and the shared datastore, if on). New = a fresh folder; Existing = build on a folder from a previous run."
                    >
                      Folder
                      <select
                        value={folderMode}
                        onChange={(e) => {
                          const m = e.target.value as FolderMode
                          setFolderMode(m)
                          if (m === 'existing') loadProjects()
                        }}
                        className="rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-sky-600 focus:outline-none"
                      >
                        <option value="new">New</option>
                        <option value="existing">Existing</option>
                      </select>
                      {folderMode === 'new' ? (
                        <input
                          value={newFolderName}
                          onChange={(e) => setNewFolderName(e.target.value)}
                          placeholder="auto-name"
                          title="Leave blank to auto-name (basna-…/vatra-…), or type a folder name to reuse across runs."
                          className="w-32 rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 placeholder-zinc-600 focus:border-sky-600 focus:outline-none"
                        />
                      ) : (
                        <span className="flex items-center gap-1">
                          <select
                            value={existingFolder}
                            onChange={(e) => setExistingFolder(e.target.value)}
                            className="w-44 rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-sky-600 focus:outline-none"
                          >
                            <option value="">
                              {projectsLoading ? 'Loading…' : projects.length === 0 ? 'No folders found' : '— pick folder —'}
                            </option>
                            {projects.map((p) => (
                              <option key={p.name} value={p.name}>
                                {p.name}{p.files ? ` (${p.files})` : ''}
                              </option>
                            ))}
                          </select>
                          <button
                            type="button"
                            onClick={() => loadProjects()}
                            title="Refresh folder list"
                            className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
                          >
                            <RefreshCw className={`h-3 w-3 ${projectsLoading ? 'animate-spin' : ''}`} />
                          </button>
                        </span>
                      )}
                    </label>
                    {/* Shared datastore — one relational store in the folder for all agents */}
                    <label
                      className="flex cursor-pointer items-center gap-1.5 text-zinc-400"
                      title="Bind every agent in this run to ONE relational datastore stored in the run's VFS folder, so they collaborate through shared tables. Off = each agent keeps a private datastore."
                    >
                      <input
                        type="checkbox" checked={sharedDatastore}
                        onChange={(e) => setSharedDatastore(e.target.checked)}
                        className="h-3.5 w-3.5 rounded border-zinc-700 bg-zinc-950/60 accent-violet-600"
                      />
                      Shared datastore
                    </label>
                  </div>
                )}
              </div>
            )}

            {/* Action bar — pin a team, then route/plan/run. Hidden while live. */}
            {!running && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setTeamOpen((o) => !o)}
                  disabled={teamLocked}
                  className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs disabled:opacity-40 ${
                    team.length > 0
                      ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-300'
                      : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`}
                  title={teamLocked ? 'Locked while running' : 'Pin exactly which archetypes the team must use (optional)'}
                >
                  <Users className="h-3.5 w-3.5" />
                  {team.length > 0 ? `Team: ${team.length}` : 'Select team'}
                </button>
                <div className="ml-auto flex items-center gap-2">
                  {planMode ? (
                    <button
                      onClick={() => runPlan(intent, tiers, title, envVars, team, planStepMode)}
                      disabled={!canRoute}
                      title={team.length > 0 && !planComplex
                        ? 'A team is selected but "simple" steps do not route — switch to a full ensemble/team step.'
                        : `Decompose → verify each step (${planComplex ? (effectiveMode === 'vatra' ? 'Vatra team' : 'Basna ensemble') : 'single model'}/step) → re-plan → synthesize.`}
                      className="flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-40"
                    >
                      {executing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                      Run plan
                    </button>
                  ) : effectiveMode === 'vatra' ? (
                    <>
                      <button
                        onClick={() => planVatra(intent, tiers, title, team)}
                        disabled={!canRoute || vatraMode}
                        title="Decompose the task into owned pieces — review the team, then Run team."
                        className="flex items-center gap-1.5 rounded-lg border border-violet-300 px-3 py-1.5 text-xs font-medium text-violet-700 hover:bg-violet-100 disabled:opacity-40 dark:border-violet-700/70 dark:text-violet-300 dark:hover:bg-violet-800/30"
                      >
                        {planning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                        Plan team
                      </button>
                      <button
                        onClick={() => runVatra(tiers, envVars)}
                        disabled={!canRun}
                        className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
                      >
                        {(executing || activeBusy) ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                        Run team
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => route(intent, tiers, title, team)}
                        disabled={!canRoute}
                        title="Select the minimal archetype team — review/edit it, then Run ensemble."
                        className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800 disabled:opacity-40"
                      >
                        {routing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                        Route
                      </button>
                      <button
                        onClick={() => execute(tiers, envVars)}
                        disabled={!canRun}
                        className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
                      >
                        {executing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                        Run ensemble
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Optional fixed team — pick archetypes the route/plan MUST use. */}
            {teamOpen && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                <div className="mb-2 flex items-center gap-2">
                  <span className="text-[11px] text-zinc-400">
                    Pick the archetypes the {effectiveMode === 'vatra' ? 'Lead' : 'router'} must use — every one selected is
                    used and instructed for this task. Leave empty for automatic selection.
                  </span>
                  {team.length > 0 && (
                    <button onClick={() => setTeam([])} disabled={teamLocked} className="ml-auto shrink-0 text-[11px] text-zinc-500 hover:text-zinc-300 disabled:opacity-40">Clear</button>
                  )}
                </div>
                <div className={`grid max-h-64 grid-cols-1 gap-1 overflow-auto sm:grid-cols-2 ${teamLocked ? 'pointer-events-none opacity-50' : ''}`}>
                  {(registry?.archetypes || []).map((a) => {
                    const on = team.includes(a.id)
                    return (
                      <button
                        key={a.id}
                        onClick={() => toggleTeam(a.id)}
                        disabled={teamLocked}
                        className={`flex items-start gap-2 rounded border p-1.5 text-left text-xs transition-colors ${
                          on ? 'border-violet-500/50 bg-violet-500/10' : 'border-zinc-800 hover:bg-zinc-800/50'}`}
                      >
                        <span className={`mt-0.5 flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded border ${
                          on ? 'border-violet-500 bg-violet-500 text-white' : 'border-zinc-600'}`}>
                          {on && <Check className="h-2.5 w-2.5" />}
                        </span>
                        <span className="min-w-0">
                          <span className="block text-zinc-200">{a.role || a.id}</span>
                          <span className="block text-[10px] text-zinc-500">{a.family || a.id}</span>
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-start gap-2 rounded-lg border border-rose-900/50 bg-rose-950/30 p-2.5 text-xs text-rose-300">
                <X className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {error}
              </div>
            )}

            {/* Stage 3 — Plan review: the team the router/Lead picked, editable. */}
            {open.plan && routePlan && (
              <>
                {/* R12 intent brief — editable; re-routing on an edited brief re-selects the team. */}
                {routePlan.brief && (
                  <BriefEditor
                    brief={routePlan.brief}
                    busy={routing || planning}
                    onReroute={(edited) =>
                      (effectiveMode === 'vatra' ? planVatra : route)(intent, tiers, title, team, edited)}
                  />
                )}
                {vatraMode && activeSession ? (
                  <VatraTeamPlan
                    subtasks={routePlan.subtasks}
                    sharedContext={routePlan.shared_context}
                    editable={!activeBusy && !executing}
                    groupInstructions={routePlan.group_instructions}
                    onUpdateSubtask={updateSubtask}
                    onRemoveSubtask={removeSubtask}
                    onSetGroupInstruction={setGroupInstruction}
                  />
                ) : !vatraMode ? (
                  <RoutePlanEditor
                    routePlan={routePlan}
                    tiers={tiers}
                    registry={registry}
                    onUpdateSelected={updateSelected}
                  />
                ) : null}
              </>
            )}

            {/* Stage 4 — Run: the live workspace (board / agents / progress). */}
            {stage === 'run' && activeSession && (
              <RunWorkspace
                session={activeSession}
                vatraMode={vatraMode}
                running
                subtasks={routePlan?.subtasks}
                liveAgents={liveAgents}
                progress={progress}
                currentPhase={currentPhase}
                runCost={runCost}
                project={runProject}
                onSkip={vatraMode ? (role) => { void apiVatraSkipAgent(activeSession.id, role) } : undefined}
                onStop={() => cancelSession(activeSession.id)}
              />
            )}
            {/* A fresh execute before the session flips to running (progress already streaming). */}
            {stage !== 'run' && executing && !activeSession && progress.length > 0 && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 text-xs text-zinc-500">Starting…</div>
            )}

            {/* Stage 4 done — the report takes over: Report | Analysis | Gaps | Board | Files | Agents | Log. */}
            {stage === 'done' && activeSession && (
              <RunReport
                session={activeSession}
                vatraMode={vatraMode}
                truth={truth}
                confidence={confidence}
                method={lastExecute?.method}
                analysis={analysis}
                runs={runs}
                runCost={runCost}
                subject={subject}
                project={runProject}
                progress={progress}
                subtasks={routePlan?.subtasks}
                recompiling={recompiling}
                onRecompile={() => recompile(tiers)}
                onView={viewFull}
                onFeedback={sendFeedback}
                deepening={deepening}
                onDeepen={async () => {
                  setDeepening(true)
                  try { await deepenSession(activeSession.id) }
                  catch { /* surfaced by store error path */ }
                  finally { setDeepening(false) }
                }}
                onFillGaps={async () => {
                  setDeepening(true)
                  try { await fillGaps(activeSession.id) }
                  catch { /* surfaced by store error path */ }
                  finally { setDeepening(false) }
                }}
                onContinue={async ({ instruction, kind, sameCast }) => {
                  await continueSession(activeSession.id, {
                    instruction, kind, sameCast, vatra: vatraMode,
                  })
                }}
              />
            )}
          </div>
        </div>
      </div>

      {modal && <FileModal title={modal.title} content={modal.content} mode={modal.mode} onClose={() => setModal(null)} />}
    </div>
  )
}
