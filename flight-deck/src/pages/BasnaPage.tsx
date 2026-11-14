import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Play, Sparkles, Check, X, Paperclip, FileText, Image as ImageIcon,
  SlidersHorizontal, ScanSearch, RefreshCw, CornerDownRight, Users, HelpCircle,
  Gauge, ListChecks, Brain, FolderSearch, Loader2, ChevronDown, ChevronRight,
  Folder, ChevronLeft, ClipboardList, Database, Trash2, Scale, Inbox,
} from 'lucide-react'
import { useBasnaStore, parseAnalysis, apiVatraSkipAgent, type FolderMode, type BasnaSession, type Group0Plan } from '../stores/basnaStore'
import { ShareModal } from '../components/common/ShareModal'
import { useBasnaProjectStore, UNFILED_ID, type BasnaProject } from '../stores/basnaProjectStore'
import { VatraTeamPlan } from '../components/VatraDelegation'
import { VatraGroup0Plan } from '../components/VatraGroup0Plan'
import { QualityControls } from '../components/QualityControls'
import { BasnaWizardModal } from '../components/agents/BasnaWizardModal'
import { KnowledgePicker } from '../components/agents/KnowledgePicker'
import { ReferenceFolderPicker } from '../components/agents/ReferenceFolderPicker'
import { deriveCost } from '../components/CostCard'
import { archetypeMatches, useTierConfig, TIER_ORDER } from '../services/tierConfig'
import {
  buildLiveAgents, FileModal, isVatra, parentIdOf, runVfsProject, type ViewMode,
} from '../components/basna/shared'
import { RunsSidebar } from '../components/basna/RunsSidebar'
import { RoutePlanEditor } from '../components/basna/RoutePlanEditor'
import { RunWorkspace } from '../components/basna/RunWorkspace'
import { RunReport } from '../components/basna/RunReport'
import { ProjectPicker } from '../components/basna/ProjectPicker'
import { ProjectDetails } from '../components/basna/ProjectDetails'
import { ProjectFiles, type ProjectSource } from '../components/basna/ProjectFiles'
import { ProjectDatastore } from '../components/basna/ProjectDatastore'
import { JudgeCalibrationModal } from '../components/basna/JudgeCalibrationModal'
import { IconSquare, SectionLabel, TeamHeader } from '../components/basna/look'
import {
  MODE_LOOK, MODE_TILE_ON, checkCls, chipToggleCls, ctaCls, headerIconBtnCls, iconBtnCls, segBtnCls, segFrameCls, smallInputCls, toggleBtnCls,
} from '../components/basna/lookCls'
import { Button, Notice, Page, Segmented, inputCls, labelCls } from '../components/console/parts'
import { FullscreenButton } from '../components/library/libraryParts'

// The console input without its full width, for a control that shares its row.
const fitInputCls = inputCls.replace('w-full ', '')

// A run's project bundle id (config.project_id), '' when unfiled.
function projectIdOf(config?: string): string {
  try { return JSON.parse(config || '{}').project_id || '' } catch { return '' }
}

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
    <section className="rounded-2xl border border-amber-500/30 bg-amber-500/[0.05] p-4 wide:p-5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-amber-600 dark:text-amber-400" />
          <span className="text-[13px] font-semibold text-amber-700 dark:text-amber-300">Task brief</span>
        </span>
        <span className="text-xs text-zinc-500">the team was selected from this — edit &amp; re-route to re-pick it</span>
        <button onClick={() => setOpen((o) => !o)} className="ml-auto rounded-lg px-2 py-1 text-xs font-medium text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
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
            className={`${inputCls} mt-3 resize-y font-mono text-xs leading-relaxed`}
          />
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              onClick={() => onReroute(draft)}
              disabled={busy || !draft.trim()}
              className="inline-flex items-center gap-1.5 rounded-xl bg-amber-500 px-3.5 py-2 text-sm font-medium text-black shadow-sm transition-colors hover:bg-amber-400 disabled:opacity-40"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
              Re-route on this brief
            </button>
            {changed && <span className="text-xs text-amber-700 dark:text-amber-300">edited — re-route to apply</span>}
            <span className="ml-auto text-[11px] text-zinc-500">the original request always governs on conflict</span>
          </div>
        </>
      )}
    </section>
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
    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
    : state === 'active'
      ? 'border-violet-500/50 bg-violet-500/15 text-violet-700 dark:text-violet-300'
      : state === 'failed'
        ? 'border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300'
        : 'border-zinc-800 bg-zinc-900/60 text-zinc-500'
  return (
    <button
      onClick={onClick}
      disabled={!onClick}
      aria-expanded={onClick ? !!open : undefined}
      className={`flex min-w-0 items-center gap-1.5 rounded-full border px-2.5 py-1.5 text-xs font-medium transition-colors wide:px-3 ${tone} ${onClick ? 'hover:brightness-110' : 'cursor-default'}`}
    >
      {busy
        ? <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
        : state === 'done'
          ? <Check className="h-3.5 w-3.5 shrink-0" />
          : state === 'failed'
            ? <X className="h-3.5 w-3.5 shrink-0" />
            : <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold ${
                state === 'active' ? 'bg-violet-500/25' : 'bg-zinc-800 text-zinc-400'}`}>{n}</span>}
      {label}
      {summary && <span className="hidden max-w-52 truncate font-normal opacity-75 wide:inline">· {summary}</span>}
      {onClick && (open ? <ChevronDown className="h-3.5 w-3.5 shrink-0 opacity-60" /> : <ChevronRight className="h-3.5 w-3.5 shrink-0 opacity-60" />)}
    </button>
  )
}

const QUALITY_LABEL: Record<string, string> = { off: 'Basic', balanced: 'Balanced', thorough: 'Thorough', custom: 'Custom' }

/** `mode` locks the page to Vatra or Basna: only that mode's runs, no switch
 *  to the other (each has its own entry in the kiosk's More and in the full
 *  layout). Without it, both, and a new run opens on the last one picked.
 *  `onClose`: in a kiosk dialog, its Close. */
export function BasnaPage({ mode, onClose }: { mode?: 'basna' | 'vatra'; onClose?: () => void } = {}) {
  const {
    sessions, activeSession: storeSession, routePlan: storeRoutePlan, runs, lastExecute, progress, attachments,
    routing, planning, executing, recompiling, resuming, error,
    routerTier, maxAgents, setRouterTier, setMaxAgents, maxParallel, setMaxParallel, executionGroups, setExecutionGroups, groupedReview, setGroupedReview, sharedDatastore, setSharedDatastore, folderMode, setFolderMode, newFolderName, setNewFolderName, existingFolder, setExistingFolder, projects, projectsLoading, loadProjects, knowledgeSessionIds, toggleKnowledgeSession, knowledgeIncludeBoard, setKnowledgeIncludeBoard, referenceFolders, toggleReferenceFolder, deep, deepSamples, setDeep, setDeepSamples, planMode, planSteps, setPlanMode, setPlanSteps, planComplex, setPlanComplex, planDag, setPlanDag, runPlan, quality, setQuality, addFiles, removeFile,
    updateSelected, updateSubtask, removeSubtask, setGroupInstruction, loadSessions, pollRunning, selectSession, newSession, resetDraft, route, planVatra, runVatra, approveVatraPlan, replanVatraGroups, cancelVatraPlan, approvingPlan, fillGaps, saveTitle, execute, recompile, resumeSession, sendFeedback, deleteSession, cancelSession, deepenSession, continueSession, setActiveProjectId,
  } = useBasnaStore()
  const { tiers, registry, envVars } = useTierConfig()
  // Locked to one mode: a run of the other one — left open in the shared store by
  // the other page, or opened from a link — is never shown here; it's put down.
  const ofPageMode = (config?: string) => !mode || isVatra(config) === (mode === 'vatra')
  const foreign = !!storeSession && !ofPageMode(storeSession.config)
  const activeSession = foreign ? null : storeSession
  const routePlan = foreign ? null : storeRoutePlan
  useEffect(() => { if (foreign) newSession() }, [foreign, storeSession?.id]) // eslint-disable-line react-hooks/exhaustive-deps
  // The runs this page shows: this mode's only, when locked.
  const modeSessions = useMemo(
    () => (mode ? sessions.filter((s) => isVatra(s.config) === (mode === 'vatra')) : sessions),
    [sessions, mode],
  )
  const {
    projects: bundleProjects, current: currentProject, loading: bundlesLoading,
    loadProjects: loadBundles, createProject, updateProject, deleteProject, select: selectProject,
  } = useBasnaProjectStore()
  const [projectTab, setProjectTab] = useState<'details' | 'run' | 'files' | 'datastore'>('run')
  const [savingProject, setSavingProject] = useState(false)
  const [shareTarget, setShareTarget] = useState<BasnaSession | null>(null)

  // Load the project bundles once; the picker + scoping read from them.
  useEffect(() => { loadBundles() }, [loadBundles])
  // New runs launched here belong to the open project (Unfiled → no project_id).
  useEffect(() => {
    setActiveProjectId(currentProject && currentProject.id !== UNFILED_ID ? currentProject.id : '')
  }, [currentProject?.id, setActiveProjectId])
  // Default to the Run tab whenever a project is opened.
  useEffect(() => { if (currentProject) setProjectTab('run') }, [currentProject?.id])
  // Run count per project (+ Unfiled) for the picker cards.
  const runCounts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const s of modeSessions) {
      const pid = projectIdOf(s.config) || UNFILED_ID
      c[pid] = (c[pid] || 0) + 1
    }
    return c
  }, [modeSessions])
  // The runs shown in the sidebar — scoped to the open project (Unfiled = no id).
  const scopedSessions = useMemo(() => {
    if (!currentProject) return modeSessions
    if (currentProject.id === UNFILED_ID) return modeSessions.filter((s) => !projectIdOf(s.config))
    return modeSessions.filter((s) => projectIdOf(s.config) === currentProject.id)
  }, [modeSessions, currentProject?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // File/datastore sources for the project Files + Datastore tabs: the project's
  // shared folder plus every run's own write folder, deduped (continuation chains
  // share the root's folder) and labelled by owner.
  const projectSources = useMemo<ProjectSource[]>(() => {
    if (!currentProject || currentProject.id === UNFILED_ID) return []
    const out: ProjectSource[] = []
    if (currentProject.vfs_folder) out.push({ label: 'Project files', folder: currentProject.vfs_folder, owner: 'project' })
    for (const s of scopedSessions) {
      const folder = runVfsProject(s)
      if (folder) out.push({ label: s.title || s.intent || 'run', folder, owner: 'run' })
    }
    const seen = new Set<string>()
    return out.filter((x) => (seen.has(x.folder) ? false : (seen.add(x.folder), true)))
  }, [currentProject?.id, currentProject?.vfs_folder, scopedSessions]) // eslint-disable-line react-hooks/exhaustive-deps

  const [intent, setIntent] = useState('')
  const [title, setTitle] = useState('')
  // Compose mode for a NEW run; a selected run locks to its own mode (effectiveMode).
  // A page locked to one mode composes in that mode only.
  const [composeMode, setComposeMode] = useState<'basna' | 'vatra'>(
    () => mode || ((typeof localStorage !== 'undefined' && localStorage.getItem('basna.composeMode')) as 'basna' | 'vatra') || 'basna',
  )
  // Optional user-fixed team: archetype ids the route/plan MUST use (empty = auto).
  const [team, setTeam] = useState<string[]>([])
  const [teamQuery, setTeamQuery] = useState('')
  const [teamOpen, setTeamOpen] = useState(false)
  const [tuning, setTuning] = useState(false)  // router tier + max agents disclosure
  const toggleTeam = (id: string) =>
    setTeam((t) => (t.includes(id) ? t.filter((x) => x !== id) : [...t, id]))
  const [deepening, setDeepening] = useState(false)
  const [modal, setModal] = useState<{ title: string; content: string; mode: ViewMode } | null>(null)
  const viewFull = (t: string, content: string) => setModal({ title: t, content, mode: 'markdown' })
  const [calibrationOpen, setCalibrationOpen] = useState(false)
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

  // Moving to a DIFFERENT project starts a clean draft — one project's team
  // plan, selected run, reference folders, prior-knowledge picks and task text
  // must never bleed into another. Track the last *non-null* project id so that
  // (a) visiting the project picker and returning to the SAME project keeps the
  // draft, and (b) a remount doesn't wipe it — only a real project→project
  // change resets.
  const lastProjectId = useRef<string | undefined>(undefined)
  useEffect(() => {
    const id = currentProject?.id
    if (!id) return  // the picker (no project open) never wipes a draft
    if (lastProjectId.current !== undefined && lastProjectId.current !== id) {
      resetDraft()
      setIntent(''); setTitle(''); setTeam([])
    }
    lastProjectId.current = id
  }, [currentProject?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // Live monitor: while any run (incl. agent-started) is mid-flight, poll the
  // list status + the open session's progress every few seconds; stop when idle.
  const anyRunning = sessions.some((s) => ['routing', 'routed', 'running', 'planning', 'awaiting_plan'].includes(s.status))
  // Vatra runs render the collaboration panel, not the Basna route-plan editor
  // (their route's `selected` carries no tier/prior_weight). Read from config so
  // it's correct even before the Lead has decomposed (route is still empty).
  const vatraMode = !!activeSession && isVatra(activeSession.config)
  // The mode the controls act in: a selected run is locked to its own mode; with
  // no run selected, the toggle (composeMode) decides.
  const effectiveMode: 'basna' | 'vatra' = mode ?? (activeSession ? (vatraMode ? 'vatra' : 'basna') : composeMode)
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
  // Keep the toggle in sync with whatever run is open (never while locked).
  useEffect(() => {
    if (activeSession && !mode) setComposeMode(isVatra(activeSession.config) ? 'vatra' : 'basna')
  }, [activeSession?.id, activeSession?.config, mode])
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
  const onWizardPrepare = ({ intent: wi, title: wt, mode: picked }: { intent: string; title: string; mode: 'basna' | 'vatra' }) => {
    if (!wi.trim()) return
    const m = mode ?? picked  // a locked page prepares in its own mode only
    newSession()
    setIntent(wi); setTitle(wt); setTeam([])
    setComposeMode(m)
    try { localStorage.setItem('basna.composeMode', m) } catch { /* ignore */ }
    setWizardOpen(false)
    if (m === 'vatra') planVatra(wi, tiers, wt, [])
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
  // Group 0 pre-phase: 'planning' = the Long Horizon Planner is drafting the plan
  // (a live run); 'awaiting_plan' = drafted, paused at the approval gate.
  const group0Planning = !!activeSession && activeSession.status === 'planning'
  const awaitingPlan = !!activeSession && activeSession.status === 'awaiting_plan'
  const running = executing || activeBusy || group0Planning
  // The editable Group 0 plan draft lives here (lifted) so the top-bar Execute button
  // sends the user's edits. Seeded from the persisted plan when the gate opens; cleared
  // when it closes. Not reseeded mid-approve so an in-flight Execute never clobbers edits.
  const [group0Draft, setGroup0Draft] = useState<Group0Plan | null>(null)
  useEffect(() => {
    if (awaitingPlan && routePlan?.group0_plan && !approvingPlan) setGroup0Draft(routePlan.group0_plan)
    else if (!awaitingPlan) setGroup0Draft(null)
  }, [awaitingPlan, routePlan?.group0_plan, approvingPlan])
  // Whether the user re-grouped an agent or answered a clarifying question vs the drafted
  // plan. Either makes the plan's mandates/dependencies stale (they were written for the
  // old phasing / before the answers), so we surface a Re-plan action to regenerate it.
  const planDirty = useMemo(() => {
    const base = routePlan?.group0_plan
    if (!awaitingPlan || !group0Draft || !base) return false
    const byId = new Map((base.agents || []).map((a) => [a.subtask_id, a.group || '']))
    const groupsChanged = (group0Draft.agents || []).some((a) => (byId.get(a.subtask_id) ?? '') !== (a.group || ''))
    const answered = (group0Draft.questions || []).some((q) => (q.selected?.length || 0) > 0 || !!(q.other || '').trim())
    return groupsChanged || answered
  }, [awaitingPlan, group0Draft, routePlan?.group0_plan])
  const finished = !!activeSession && (activeSession.status === 'done' || activeSession.status === 'error')
  const stage: Stage = running ? 'run' : finished ? 'done' : routePlan ? 'plan' : 'define'
  // A stopped/failed run can be resumed from its checkpoints: finished agents are
  // restored (no re-spend), only the missing ones re-run. Shown while it's idle
  // (not actively executing) so the user decides when it's stuck.
  const resumable = !!activeSession
    && (activeSession.status === 'cancelled' || activeSession.status === 'error')
    && !executing && !resuming

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
    ...(mode ? [] : [effectiveMode === 'vatra' ? 'Vatra' : 'Basna']),
    strategy[0].toUpperCase() + strategy.slice(1),
    QUALITY_LABEL[quality.profile] || quality.profile,
    ...(knowledgeSessionIds.length ? [`${knowledgeSessionIds.length} prior`] : []),
    ...(referenceFolders.length ? [`${referenceFolders.length} ref`] : []),
    ...(team.length ? [`${team.length} pinned`] : []),
  ].join(' · ')

  const onDelete = (s: typeof sessions[number]) => {
    const raw = (s.title || s.intent || '').trim()
    const label = raw.slice(0, 80)
    if (window.confirm(`Delete this ${mode === 'vatra' ? 'Vatra' : 'Basna'} run?${label ? `\n\n"${label}${raw.length > 80 ? '…' : ''}"` : ''}`)) {
      deleteSession(s.id)
    }
  }

  const onDeleteProject = (p: BasnaProject) => {
    const n = runCounts[p.id] || 0
    const msg = n > 0
      ? `Delete project "${p.name}"?\n\nIt has ${n} run(s). Click OK to delete the project AND its runs, or Cancel to keep everything.`
      : `Delete project "${p.name}"?`
    if (window.confirm(msg)) deleteProject(p.id, n > 0)
  }

  // ── The header: the mode's icon, name and a line on it (the kiosk opens a
  // mode), or Basna's own in the full layout — and the page's actions. ──
  const look = MODE_LOOK[mode ? effectiveMode : 'basna']
  const pageTitle = mode ? look.name : 'Basna'
  const pageBlurb = mode ? look.blurb : 'Bundle runs into projects — one theme, one shared folder.'
  const headerActions = (
    <>
      <Button
        icon={HelpCircle}
        onClick={() => viewFull('Basna, Vatra & Deep mode', HELP_MD)}
        title="What are Basna, Vatra and Deep mode?"
        aria-label="Help"
        className="px-2.5 wide:px-3"
      >
        <span className="hidden wide:inline">Help</span>
      </Button>
      <Button
        icon={Scale}
        onClick={() => setCalibrationOpen(true)}
        title="How often the automatic judge agrees with your thumbs votes — and export the pairs"
        aria-label="Judge calibration"
        className="px-2.5 wide:px-3"
      >
        <span className="hidden wide:inline">Judge calibration</span>
      </Button>
      <FullscreenButton className={headerIconBtnCls} />
      {onClose && <Button variant="ghost" icon={X} onClick={onClose} aria-label="Close" title="Close" className="px-2.5" />}
    </>
  )

  // ── Project picker: the projects-first landing (no project open yet) ──
  if (!currentProject) {
    return (
      <div className="@container h-full overflow-y-auto">
        <Page full={!!onClose}>
          <TeamHeader icon={look.icon} tone={look.tone} title={pageTitle} description={pageBlurb} actions={headerActions} />
          <ProjectPicker
            projects={bundleProjects}
            counts={runCounts}
            busy={bundlesLoading}
            accent={look.tone === 'sky' ? 'sky' : 'violet'}
            onSelect={selectProject}
            onCreate={async (name, description, instructions) => {
              const p = await createProject(name, description, instructions)
              if (p) selectProject(p)
            }}
            onDelete={onDeleteProject}
          />
        </Page>
        {modal && <FileModal title={modal.title} content={modal.content} mode={modal.mode} onClose={() => setModal(null)} />}
        {calibrationOpen && <JudgeCalibrationModal mode={mode} onClose={() => setCalibrationOpen(false)} />}
      </div>
    )
  }

  const isUnfiled = currentProject.id === UNFILED_ID
  // A tab's own pane: it scrolls beside the runs list on a larger screen; on a
  // phone the whole page scrolls as one column.
  const paneCls = '@container min-w-0 @3xl:flex-1 @3xl:overflow-y-auto'
  const runCta = ctaCls(effectiveMode === 'vatra' ? 'violet' : 'sky')

  return (
    <div className="@container flex h-full flex-col">
      {shareTarget && (
        <ShareModal
          resourceType="basna"
          resourceId={shareTarget.id}
          resourceName={shareTarget.title || shareTarget.intent || `${mode === 'vatra' ? 'Vatra' : 'Basna'} run`}
          onClose={() => setShareTarget(null)}
        />
      )}
      {/* Header — the mode, the project breadcrumb and its Details/Run/Files/Datastore tabs */}
      <div className="shrink-0 border-b border-zinc-800/80 px-4 pb-4 pt-4 wide:px-8 wide:pt-6">
        <TeamHeader icon={look.icon} tone={look.tone} title={pageTitle} description={pageBlurb} actions={headerActions} compact />
        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-3">
          <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1 text-sm">
            <button
              onClick={() => selectProject(null)}
              title="All projects"
              className="flex shrink-0 items-center gap-1 rounded-lg py-1 pl-1 pr-2 font-medium text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
            >
              <ChevronLeft className="h-4 w-4" /> Projects
            </button>
            <span className="text-zinc-700" aria-hidden>/</span>
            <span className="flex min-w-0 items-center gap-1.5 px-1 font-medium text-zinc-100" title={currentProject.name}>
              {isUnfiled
                ? <Inbox className="h-4 w-4 shrink-0 text-zinc-500" />
                : <Folder className={`h-4 w-4 shrink-0 ${look.tone === 'sky' ? 'text-sky-600 dark:text-sky-400' : 'text-violet-600 dark:text-violet-400'}`} />}
              <span className="truncate">{currentProject.name}</span>
            </span>
          </nav>
          {!isUnfiled && (
            <div className="wide:ml-auto">
              <Segmented
                label="Project view"
                value={projectTab}
                onChange={setProjectTab}
                options={[
                  { value: 'details', label: <><ClipboardList className="hidden h-3.5 w-3.5 wide:block" />Details</> },
                  { value: 'run', label: <><Play className="hidden h-3.5 w-3.5 wide:block" />Run</> },
                  { value: 'files', label: <><FileText className="hidden h-3.5 w-3.5 wide:block" />Files</> },
                  { value: 'datastore', label: <><Database className="hidden h-3.5 w-3.5 wide:block" />Datastore</> },
                ]}
              />
            </div>
          )}
        </div>
      </div>
      {wizardOpen && (
        <BasnaWizardModal
          sessions={modeSessions}
          creds={wizardCreds}
          mode={mode}
          onClose={() => setWizardOpen(false)}
          onPrepare={onWizardPrepare}
        />
      )}

      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto @3xl:flex-row @3xl:overflow-hidden">
        {projectTab === 'run' && (
          <RunsSidebar
            mode={mode}
            sessions={scopedSessions}
            activeId={activeSession?.id}
            onSelect={selectSession}
            onDelete={onDelete}
            onCancel={cancelSession}
            onNew={() => { newSession(); setIntent(''); setTitle(''); setTeam([]) }}
            onWizard={() => setWizardOpen(true)}
            onShare={setShareTarget}
          />
        )}

        {/* Details tab — the project's theme. */}
        {projectTab === 'details' && !isUnfiled && (
          <div className={paneCls}>
            <div className="mx-auto w-full max-w-4xl px-4 py-6 wide:px-8 wide:py-8">
              <ProjectDetails
                project={currentProject}
                saving={savingProject}
                onSave={async (fields) => {
                  setSavingProject(true)
                  try { await updateProject(currentProject.id, fields) }
                  finally { setSavingProject(false) }
                }}
              />
            </div>
          </div>
        )}

        {/* Files tab — the project folder + every run's folder, grouped by owner. */}
        {projectTab === 'files' && !isUnfiled && (
          <div className={paneCls}>
            <div className="mx-auto w-full max-w-5xl px-4 py-6 wide:px-8 wide:py-8">
              <ProjectFiles sources={projectSources} uploadFolder={currentProject.vfs_folder} />
            </div>
          </div>
        )}

        {/* Datastore tab — tables across the project folder + every run, by owner. */}
        {projectTab === 'datastore' && !isUnfiled && (
          <div className={paneCls}>
            <div className="mx-auto w-full max-w-5xl px-4 py-6 wide:px-8 wide:py-8">
              <ProjectDatastore sources={projectSources} />
            </div>
          </div>
        )}

        {/* Detail (Run tab) */}
        {projectTab === 'run' && (
        <div className={paneCls}>
          <div className="mx-auto w-full max-w-[1600px] space-y-4 px-4 py-5 wide:px-6 wide:py-6">
            {/* Deepen lineage — jump to the parent / child runs of this one. */}
            {activeSession && (() => {
              const pid = parentIdOf(activeSession.config)
              const parent = pid ? modeSessions.find((s) => s.id === pid) : null
              const children = modeSessions.filter((s) => parentIdOf(s.config) === activeSession.id)
              if (!parent && children.length === 0) return null
              const linkCls = 'flex min-w-0 items-center gap-1.5 text-left text-violet-700 hover:underline dark:text-violet-300'
              return (
                <div className="flex flex-col gap-1.5 rounded-xl border border-violet-500/25 bg-violet-500/[0.05] px-3.5 py-2.5 text-xs">
                  {parent && (
                    <button onClick={() => selectSession(parent.id)} className={linkCls}>
                      <CornerDownRight className="h-3.5 w-3.5 shrink-0" />
                      <span className="truncate">deepened from “{parent.title || parent.intent || 'run'}”</span>
                    </button>
                  )}
                  {children.map((ch) => (
                    <button key={ch.id} onClick={() => selectSession(ch.id)} className={linkCls}>
                      <ScanSearch className="h-3.5 w-3.5 shrink-0" />
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
              <ChevronRight className="hidden h-3.5 w-3.5 shrink-0 text-zinc-600 wide:block" />
              <StageChip
                n={2} label="Setup"
                state={intent.trim() ? 'done' : 'idle'}
                summary={setupSummary}
                open={open.setup}
                onClick={() => toggle('setup')}
              />
              <ChevronRight className="hidden h-3.5 w-3.5 shrink-0 text-zinc-600 wide:block" />
              <StageChip
                n={3} label="Plan"
                state={routePlan ? (stage === 'plan' ? 'active' : 'done') : 'idle'}
                summary={routePlan ? `${planAgentCount || routePlan.selected.length} agents` : 'optional'}
                open={open.plan}
                onClick={routePlan ? () => toggle('plan') : undefined}
              />
              <ChevronRight className="hidden h-3.5 w-3.5 shrink-0 text-zinc-600 wide:block" />
              <StageChip
                n={4} label="Run"
                state={stage === 'run' ? 'active' : stage === 'done' ? (activeSession?.status === 'error' ? 'failed' : 'done') : 'idle'}
                busy={stage === 'run'}
                summary={stage === 'run' ? (currentPhase || 'working') : stage === 'done' ? (activeSession?.status === 'error' ? 'failed' : 'finished') : undefined}
              />
            </div>

            {/* A stopped/failed run can be resumed from its checkpoints — restore the
                finished agents (no re-spend) and re-run only the missing ones. */}
            {resumable && (
              <div className="flex flex-col gap-3 rounded-2xl border border-amber-500/30 bg-amber-500/[0.06] p-4 wide:flex-row wide:items-center wide:justify-between">
                <div className="flex min-w-0 items-start gap-3">
                  <IconSquare icon={RefreshCw} tone="amber" size="sm" />
                  <div className="min-w-0">
                    <div className="text-sm font-semibold text-amber-700 dark:text-amber-300">
                      {activeSession?.status === 'error' ? 'Run failed' : 'Run stopped'}
                    </div>
                    <div className="mt-0.5 text-xs leading-relaxed text-zinc-500">
                      Resume from where it left off — finished {vatraMode ? 'specialists' : 'agents'} are
                      restored from checkpoints (no re-run, no re-spend); only the missing ones re-run,
                      then it synthesizes.
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => resumeSession(tiers, envVars, vatraMode)}
                  disabled={resuming}
                  className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-xl bg-amber-500 px-3.5 py-2 text-sm font-medium text-black shadow-sm transition-colors hover:bg-amber-400 disabled:opacity-40"
                >
                  {resuming ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                  Resume
                </button>
              </div>
            )}

            {/* Stage 1 — Task: title, the task itself, attachments. */}
            {open.define && (
              <section
                className={`rounded-2xl border bg-zinc-900/40 p-4 transition-colors wide:p-5 ${
                  dragOver ? 'border-violet-500/60 ring-2 ring-violet-500/15' : 'border-zinc-800'}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files) }}
              >
                <label className={labelCls}>
                  Title <span className="font-normal text-zinc-600">— optional, auto-generated from the task if blank</span>
                </label>
                <div className="mb-4 flex items-center gap-2">
                  <input
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="e.g. Q3 competitor scan"
                    className={inputCls}
                  />
                  {activeSession && title.trim() !== (activeSession.title || '') && (
                    <Button icon={Check} onClick={() => saveTitle(title)} title="Save title" className="shrink-0">
                      Save
                    </Button>
                  )}
                </div>
                <label className={labelCls}>Task / intent</label>
                <textarea
                  value={intent}
                  onChange={(e) => setIntent(e.target.value)}
                  onPaste={handlePaste}
                  rows={7}
                  placeholder={mode === 'vatra'
                    ? 'Describe the task, or attach/drop/paste files. A Lead splits it into pieces and the team builds one result.'
                    : 'Describe the task, or attach/drop/paste files. The router picks the smallest team that can answer it well.'}
                  className={`${inputCls} resize-y leading-relaxed`}
                />

                {/* Attachments */}
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Button icon={Paperclip} onClick={() => fileInputRef.current?.click()} className="px-3 py-1.5 text-xs">
                    Attach
                  </Button>
                  <input
                    ref={fileInputRef} type="file" multiple className="hidden"
                    onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }}
                  />
                  {attachments.filter((a) => a.kind !== 'generated').map((a) => (
                    <span key={a.name} className="flex max-w-full items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-950/60 py-1 pl-2.5 pr-1.5 text-xs text-zinc-300">
                      {a.mime.startsWith('image/') ? <ImageIcon className="h-3.5 w-3.5 shrink-0 text-zinc-500" /> : <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />}
                      <span className="truncate">{a.name}</span>
                      <span className="shrink-0 text-zinc-600">{Math.max(1, Math.round(a.size / 1024))}kb</span>
                      <button onClick={() => removeFile(a.name)} aria-label={`Remove ${a.name}`} className="shrink-0 rounded-full p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-red-600 dark:hover:text-red-400"><X className="h-3 w-3" /></button>
                    </span>
                  ))}
                </div>
                {!intent.trim() && (
                  <div className="mt-4 flex flex-wrap items-center gap-1.5 border-t border-zinc-800/80 pt-4">
                    <span className="mr-1 text-xs text-zinc-500">Try:</span>
                    {BASNA_EXAMPLES.map((ex) => (
                      <button
                        key={ex.label}
                        onClick={() => setIntent(ex.text)}
                        title={ex.text}
                        className="rounded-full border border-zinc-800 bg-zinc-950/60 px-2.5 py-1 text-xs text-zinc-400 transition-colors hover:border-violet-500/50 hover:text-violet-700 dark:hover:text-violet-300"
                      >
                        {ex.label}
                      </button>
                    ))}
                  </div>
                )}
              </section>
            )}

            {/* Stage 2 — Setup: mode, effort, quality, knowledge, folders, tuning. */}
            {open.setup && (
              <section className="space-y-5 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 wide:p-5">
                {/* Mode selector — the two ways to run a team, clearly distinguished.
                    A page locked to one mode has no switch to the other. */}
                {!mode && (
                <div>
                  <SectionLabel className="mb-2.5">Mode</SectionLabel>
                  <div className="grid gap-2.5 @xl:grid-cols-2">
                    {([
                      { id: 'basna', sub: 'Independent ensemble — agents answer blind, merged by reliability' },
                      { id: 'vatra', sub: 'Collaborative team — a Lead splits the work, a reporter assembles it' },
                    ] as const).map((m) => {
                      const ml = MODE_LOOK[m.id]
                      const sel = effectiveMode === m.id
                      return (
                        <button
                          key={m.id}
                          onClick={() => pickMode(m.id)}
                          aria-pressed={sel}
                          className={`group relative flex min-w-0 items-center gap-3.5 rounded-2xl border p-3.5 text-left transition-all hover:-translate-y-px ${
                            sel ? MODE_TILE_ON[ml.tone] : 'border-zinc-800 bg-zinc-950/40 hover:border-zinc-700 hover:bg-zinc-800/60 hover:shadow-lg hover:shadow-black/10'}`}
                        >
                          <IconSquare icon={ml.icon} tone={sel ? ml.tone : 'zinc'} />
                          <span className="min-w-0 flex-1">
                            <span className="block text-sm font-semibold text-zinc-100">{ml.name}</span>
                            <span className="mt-0.5 block text-xs leading-snug text-zinc-500">{m.sub}</span>
                          </span>
                          {sel && (
                            <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-white ${ml.tone === 'sky' ? 'bg-sky-600' : 'bg-violet-600'}`}>
                              <Check className="h-3 w-3" /><span className="sr-only">selected</span>
                            </span>
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>
                )}

                {/* Effort / strategy */}
                <div>
                  <div className="mb-2.5 flex items-center justify-between gap-3">
                    <SectionLabel>Effort</SectionLabel>
                    <button
                      onClick={() => setTuning((t) => !t)}
                      title="Router tier and team size"
                      aria-pressed={tuning}
                      className={toggleBtnCls(tuning)}
                    >
                      <SlidersHorizontal className="h-3.5 w-3.5" /> Tuning
                    </button>
                  </div>
                  <Segmented
                    label="Effort"
                    value={strategy}
                    onChange={setStrategy}
                    options={[
                      { value: 'standard', label: <><Gauge className="h-3.5 w-3.5" />Standard</> },
                      { value: 'deep', label: <><ScanSearch className="h-3.5 w-3.5" />Deep</> },
                      { value: 'plan', label: <><ListChecks className="h-3.5 w-3.5" />Plan</> },
                    ]}
                  />

                  {/* What the chosen effort does + its options */}
                  <div className="mt-3 rounded-xl border border-zinc-800 bg-zinc-950/40 px-4 py-3 text-xs text-zinc-400">
                    {strategy === 'standard' && (
                      <p className="leading-relaxed">
                        A minimal team answers once — {effectiveMode === 'vatra'
                          ? 'a Lead splits the work and a reporter assembles it.'
                          : 'independent agents, merged by reliability.'}
                      </p>
                    )}
                    {strategy === 'deep' && (
                      <div className="space-y-3">
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
                              className={`${smallInputCls} w-16`}
                            />
                          </label>
                        )}
                      </div>
                    )}
                    {strategy === 'plan' && (
                      <div className="space-y-3">
                        <p className="leading-relaxed">
                          Break the task into steps, drive each to a <span className="text-zinc-200">verified</span> result before the next, re-plan on failure, then synthesize.
                        </p>
                        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
                          <label className="flex items-center gap-2 text-zinc-400">
                            Steps
                            <input
                              type="number" min={1} max={12} value={planSteps}
                              onChange={(e) => setPlanSteps(Number(e.target.value))}
                              title="Max steps in the plan"
                              className={`${smallInputCls} w-16`}
                            />
                          </label>
                          <div className="flex items-center gap-2 text-zinc-400">
                            Each step
                            <div className={segFrameCls}>
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
                                  className={segBtnCls(planComplex === opt.complex)}
                                >
                                  {opt.label}
                                </button>
                              ))}
                            </div>
                          </div>
                          <label
                            className="flex cursor-pointer items-center gap-2 text-zinc-400"
                            title="Planner emits a dependency graph — independent steps run in parallel, each seeing only what it depends on."
                          >
                            <input
                              type="checkbox" checked={planDag}
                              onChange={(e) => setPlanDag(e.target.checked)}
                              className={checkCls}
                            />
                            parallel
                          </label>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Quality levers (opt-in cross-pollination) */}
                <QualityControls scope="research" value={quality} onChange={setQuality} />

                {/* What the team can read: prior runs' knowledge and reference folders. */}
                <div>
                  <SectionLabel className="mb-2.5">Knowledge</SectionLabel>
                  <div className="flex flex-wrap gap-2">
                    {/* Prior-run knowledge (opt-in) — seed this run with earlier reports + gaps */}
                    <button
                      onClick={() => setShowKnowledge((v) => !v)}
                      aria-expanded={showKnowledge}
                      className={chipToggleCls(knowledgeSessionIds.length > 0, 'violet')}
                      title="Use the knowledge (report + gaps/blind spots) of prior finished runs"
                    >
                      <Brain className="h-4 w-4" />
                      {knowledgeSessionIds.length > 0 ? `Prior knowledge: ${knowledgeSessionIds.length}` : 'Use prior-run knowledge'}
                      <ChevronDown className={`h-3.5 w-3.5 opacity-60 transition-transform ${showKnowledge ? 'rotate-180' : ''}`} />
                    </button>
                    {/* Reference folders (read-only) — agents check these before web search */}
                    <button
                      onClick={() => { setShowRefFolders((v) => !v); if (projects.length === 0) loadProjects() }}
                      aria-expanded={showRefFolders}
                      className={chipToggleCls(referenceFolders.length > 0, 'emerald')}
                      title="Read-only VFS folders agents search before web search (prior-knowledge runs' folders are auto-included)"
                    >
                      <FolderSearch className="h-4 w-4" />
                      {referenceFolders.length > 0 ? `Reference folders: ${referenceFolders.length}` : 'Reference folders (read-only)'}
                      <ChevronDown className={`h-3.5 w-3.5 opacity-60 transition-transform ${showRefFolders ? 'rotate-180' : ''}`} />
                    </button>
                  </div>
                  {showKnowledge && (
                    <div className="mt-3 rounded-xl border border-zinc-800 bg-zinc-950/40 p-3">
                      <KnowledgePicker
                        sessions={scopedSessions}
                        selectedIds={knowledgeSessionIds}
                        onToggle={toggleKnowledgeSession}
                        includeBoard={knowledgeIncludeBoard}
                        onIncludeBoard={setKnowledgeIncludeBoard}
                      />
                    </div>
                  )}
                  {showRefFolders && (
                    <div className="mt-3 rounded-xl border border-zinc-800 bg-zinc-950/40 p-3">
                      <ReferenceFolderPicker projects={projects} selected={referenceFolders} onToggle={toggleReferenceFolder} />
                    </div>
                  )}
                </div>

                {/* Tuning (router tier + team size) */}
                {tuning && (
                  <div className="rounded-xl border border-zinc-800 bg-zinc-950/40 p-4">
                    <SectionLabel className="mb-3">Tuning</SectionLabel>
                    <div className="grid gap-x-5 gap-y-4 @lg:grid-cols-2 @4xl:grid-cols-3">
                      <label className="block min-w-0">
                        <span className={labelCls}>Router tier</span>
                        <select
                          value={routerTier}
                          onChange={(e) => setRouterTier(e.target.value)}
                          title="Which Library tier picks the team / leads it"
                          className={inputCls}
                        >
                          {TIER_ORDER.filter((t) => tiers[t]).map((t) => (
                            <option key={t} value={t}>{registry?.tiers[t]?.label || t}</option>
                          ))}
                          {Object.keys(tiers).length === 0 && <option value="reason">reason</option>}
                        </select>
                      </label>
                      <label className="block min-w-0">
                        <span className={labelCls}>Max agents</span>
                        <input
                          type="number" min={1} max={10} value={maxAgents}
                          onChange={(e) => setMaxAgents(Number(e.target.value))}
                          disabled={team.length > 0}
                          title={team.length > 0 ? 'Ignored — the team is fixed by your selection' : 'Team size cap'}
                          className={inputCls}
                        />
                      </label>
                      <label className="block min-w-0">
                        <span className={labelCls}>Max parallel</span>
                        <input
                          type="number" min={0} max={16} value={maxParallel}
                          onChange={(e) => setMaxParallel(Number(e.target.value))}
                          title="How many agents run their turn at once. 0 = all at once. Lower it (e.g. 2) for local models to avoid running the serving box out of memory."
                          className={inputCls}
                        />
                      </label>
                      {/* Run folder — new (auto/custom name) or an existing VFS folder */}
                      <label
                        className="block min-w-0 @lg:col-span-2"
                        title="Where this run's agents read/write shared files (and the shared datastore, if on). New = a fresh folder; Existing = build on a folder from a previous run."
                      >
                        <span className={labelCls}>Folder</span>
                        <span className="flex items-center gap-2">
                          <select
                            value={folderMode}
                            onChange={(e) => {
                              const m = e.target.value as FolderMode
                              setFolderMode(m)
                              if (m === 'existing') loadProjects()
                            }}
                            className={`${fitInputCls} w-32 shrink-0`}
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
                              className={`${fitInputCls} min-w-0 flex-1`}
                            />
                          ) : (
                            <>
                              <select
                                value={existingFolder}
                                onChange={(e) => setExistingFolder(e.target.value)}
                                className={`${fitInputCls} min-w-0 flex-1`}
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
                                className={iconBtnCls}
                              >
                                <RefreshCw className={`h-3.5 w-3.5 ${projectsLoading ? 'animate-spin' : ''}`} />
                              </button>
                            </>
                          )}
                        </span>
                      </label>
                    </div>
                    <div className="mt-4 flex flex-wrap gap-x-6 gap-y-3 border-t border-zinc-800/80 pt-4 text-sm text-zinc-300">
                      {effectiveMode === 'vatra' && (
                        <label
                          className="flex cursor-pointer items-center gap-2"
                          title="Run owners in ordered phases A→B→C→D (research/design first, review/assembly last) with a barrier between groups, instead of all at once. Opt-in."
                        >
                          <input
                            type="checkbox" checked={executionGroups}
                            onChange={(e) => setExecutionGroups(e.target.checked)}
                            className={checkCls}
                          />
                          Grouped
                        </label>
                      )}
                      {effectiveMode === 'vatra' && executionGroups && (
                        <label
                          className="flex cursor-pointer items-center gap-2"
                          title="After the final group, every owner revises its piece against the whole team's work — early-group owners never see later groups' output otherwise. Adds one dispatch per owner. Opt-in."
                        >
                          <input
                            type="checkbox" checked={groupedReview}
                            onChange={(e) => setGroupedReview(e.target.checked)}
                            className={checkCls}
                          />
                          Review round
                        </label>
                      )}
                      {/* Shared datastore — one relational store in the folder for all agents */}
                      <label
                        className="flex cursor-pointer items-center gap-2"
                        title="Bind every agent in this run to ONE relational datastore stored in the run's VFS folder, so they collaborate through shared tables. Off = each agent keeps a private datastore."
                      >
                        <input
                          type="checkbox" checked={sharedDatastore}
                          onChange={(e) => setSharedDatastore(e.target.checked)}
                          className={checkCls}
                        />
                        Shared datastore
                      </label>
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* Gate action bar — the coordination plan is drafted and awaiting the user;
                Execute approves (runs Group A) and Cancel discards. Replaces Plan/Run team
                since the run is already in motion. */}
            {awaitingPlan && (
              <div className="flex flex-wrap items-center justify-end gap-2">
                {planDirty && (
                  <span className="mr-auto text-xs text-amber-700 dark:text-amber-300">
                    Plan edited — re-plan to fold your groups/answers into the coordination.
                  </span>
                )}
                <Button variant="danger" icon={Trash2} onClick={() => cancelVatraPlan()} disabled={approvingPlan}>
                  Cancel
                </Button>
                {planDirty && group0Draft && (
                  <button
                    onClick={() => replanVatraGroups(group0Draft, tiers, envVars)}
                    disabled={approvingPlan}
                    title="Regenerate the coordination plan for the new grouping / answers"
                    className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-amber-500/50 bg-amber-500/10 px-3.5 py-2 text-sm font-medium text-amber-700 transition-colors hover:bg-amber-500/20 disabled:opacity-40 dark:text-amber-300"
                  >
                    {approvingPlan ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                    Re-plan
                  </button>
                )}
                <button
                  onClick={() => approveVatraPlan(group0Draft ?? undefined, tiers, envVars)}
                  disabled={approvingPlan}
                  title={planDirty ? 'Run with the new groups but the current (pre-re-plan) mandates' : 'Run Group A'}
                  className={ctaCls('violet')}
                >
                  {approvingPlan ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  Execute
                </button>
              </div>
            )}

            {/* Action bar — pin a team, then route/plan/run. Hidden while live or at the gate. */}
            {!running && !awaitingPlan && (
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => setTeamOpen((o) => !o)}
                  disabled={teamLocked}
                  aria-expanded={teamOpen}
                  className={chipToggleCls(team.length > 0, 'violet')}
                  title={teamLocked ? 'Locked while running' : 'Pin exactly which archetypes the team must use (optional)'}
                >
                  <Users className="h-4 w-4" />
                  {team.length > 0 ? `Team: ${team.length}` : 'Select team'}
                </button>
                <div className="ml-auto flex flex-wrap items-center justify-end gap-2">
                  {planMode ? (
                    <button
                      onClick={() => runPlan(intent, tiers, title, envVars, team, planStepMode)}
                      disabled={!canRoute}
                      title={team.length > 0 && !planComplex
                        ? 'A team is selected but "simple" steps do not route — switch to a full ensemble/team step.'
                        : `Decompose → verify each step (${planComplex ? (effectiveMode === 'vatra' ? 'Vatra team' : 'Basna ensemble') : 'single model'}/step) → re-plan → synthesize.`}
                      className={runCta}
                    >
                      {executing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                      Run plan
                    </button>
                  ) : effectiveMode === 'vatra' ? (
                    <>
                      <Button
                        onClick={() => planVatra(intent, tiers, title, team)}
                        disabled={!canRoute || vatraMode}
                        title="Decompose the task into owned pieces — review the team, then Run team."
                      >
                        {planning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                        Plan team
                      </Button>
                      <button
                        onClick={() => runVatra(tiers, envVars)}
                        disabled={!canRun}
                        className={runCta}
                      >
                        {(executing || activeBusy) ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                        Run team
                      </button>
                    </>
                  ) : (
                    <>
                      <Button
                        onClick={() => route(intent, tiers, title, team)}
                        disabled={!canRoute}
                        title="Select the minimal archetype team — review/edit it, then Run ensemble."
                      >
                        {routing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                        Route
                      </Button>
                      <button
                        onClick={() => execute(tiers, envVars)}
                        disabled={!canRun}
                        className={runCta}
                      >
                        {executing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                        Run ensemble
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Optional fixed team — pick archetypes the route/plan MUST use. */}
            {teamOpen && (
              <section className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 wide:p-5">
                <div className="mb-3 flex items-start gap-3">
                  <p className="min-w-0 flex-1 text-xs leading-relaxed text-zinc-500">
                    Pick the archetypes the {effectiveMode === 'vatra' ? 'Lead' : 'router'} must use — every one selected is
                    used and instructed for this task. Leave empty for automatic selection.
                  </p>
                  {team.length > 0 && (
                    <button onClick={() => setTeam([])} disabled={teamLocked} className="shrink-0 rounded-lg px-2 py-1 text-xs font-medium text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40">Clear</button>
                  )}
                </div>
                <input
                  value={teamQuery}
                  onChange={(e) => setTeamQuery(e.target.value)}
                  placeholder="Filter by role, family or skill…"
                  aria-label="Filter archetypes"
                  className={`${inputCls} mb-3`}
                />
                <div className={`grid max-h-72 grid-cols-1 gap-1.5 overflow-auto @xl:grid-cols-2 @5xl:grid-cols-3 ${teamLocked ? 'pointer-events-none opacity-50' : ''}`}>
                  {(registry?.archetypes || []).filter((a) => team.includes(a.id) || archetypeMatches(a, teamQuery)).map((a) => {
                    const on = team.includes(a.id)
                    return (
                      <button
                        key={a.id}
                        onClick={() => toggleTeam(a.id)}
                        disabled={teamLocked}
                        className={`flex items-start gap-2.5 rounded-xl border px-3 py-2 text-left transition-colors ${
                          on ? 'border-violet-500/50 bg-violet-500/10' : 'border-zinc-800 bg-zinc-950/40 hover:border-zinc-700 hover:bg-zinc-800/50'}`}
                      >
                        <span className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border ${
                          on ? 'border-violet-500 bg-violet-600 text-white' : 'border-zinc-600'}`}>
                          {on && <Check className="h-3 w-3" />}
                        </span>
                        <span className="min-w-0">
                          <span className="block truncate text-sm text-zinc-200">{a.role || a.id}</span>
                          <span className="block truncate text-[11px] text-zinc-500">{a.family || a.id}</span>
                        </span>
                      </button>
                    )
                  })}
                </div>
              </section>
            )}

            {error && <Notice tone="red">{error}</Notice>}

            {/* Group 0 gate — the Long Horizon Planner's coordination plan, awaiting
                the user's review/edit before any Group-A worker runs. The Execute/Cancel
                CTAs live in the action bar above (replacing Plan/Run team). */}
            {awaitingPlan && activeSession && routePlan && group0Draft && (
              <VatraGroup0Plan
                plan={group0Draft}
                subtasks={routePlan.subtasks}
                onChange={setGroup0Draft}
                dirty={planDirty}
              />
            )}

            {/* Stage 3 — Plan review: the team the router/Lead picked, editable. */}
            {open.plan && routePlan && !awaitingPlan && (
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
              <div className="flex items-center gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 text-sm text-zinc-500">
                <Loader2 className="h-4 w-4 animate-spin" /> Starting…
              </div>
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
                onDeepen={async (instruction) => {
                  setDeepening(true)
                  try { await deepenSession(activeSession.id, instruction) }
                  catch { /* surfaced by store error path */ }
                  finally { setDeepening(false) }
                }}
                onFillGaps={async (instruction) => {
                  setDeepening(true)
                  try { await fillGaps(activeSession.id, instruction) }
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
        )}
      </div>

      {modal && <FileModal title={modal.title} content={modal.content} mode={modal.mode} onClose={() => setModal(null)} />}
      {calibrationOpen && <JudgeCalibrationModal mode={mode} onClose={() => setCalibrationOpen(false)} />}
    </div>
  )
}
