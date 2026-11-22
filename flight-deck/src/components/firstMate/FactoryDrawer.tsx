import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Archive, ArchiveRestore, ChevronLeft, ChevronRight, Database, FlaskConical, Gauge, History, Loader2, Mail, Maximize2, Minimize2, Play, ScrollText, ShieldCheck, Square, ThumbsDown, ThumbsUp, X } from 'lucide-react'
import {
  approve,
  approveRun,
  archiveFactory,
  archiveFinishedRuns,
  archiveRun,
  listRuns,
  caseDraft,
  decline,
  rateRun,
  rejectRun,
  setTrust,
  editDraft,
  getFactory,
  newDraft,
  patchFactory,
  runNow,
  stopRun,
  continueRun,
  tokensText,
  triggersText,
  whenText,
  HEALTH_LABELS,
  RUN_LABELS,
  SIGNAL_LABELS,
  getMyAgent,
  RUNNER_LABELS,
  STATE_LABELS,
  TRUST_HINTS,
  TRUST_LABELS,
  type DraftFields,
  type FactoryDetail,
  type FactoryEvent,
  type Intent,
  type Run,
  type Trigger,
  type Trust,
  type Version,
} from '../../services/firstMate'
import { ErrorBox, Pill, Section, errText, type PillTone } from './ui'
import { Checks, pct, type CaseSeed } from './Checks'
import { RunResult } from './RunResult'
import { RunProgress } from './RunProgress'
import { CapPicker } from './CapPicker'
import { FactoryData } from './FactoryData'
import { FactoryResults, TimingChips } from './FactoryResults'
import { FactoryRecipe } from './FactoryRecipe'
import { AddDataPanel } from '../context/AddData'
import { CharterText, GoalChecklist, GoalList, RunnerPicker, SchedulePicker } from './CharterEditor'
import { checksOf, goalKind, goalsOf, type Goal } from '../../utils/charterGoals'

// One factory, in a drawer over the page: its charter (the live version, and
// the draft being written), its runs, and its history. Approving a draft is
// the owner's own act, against the version they see: the server refuses an
// approval once the draft changed under it.

export type Tab = 'charter' | 'recipe' | 'results' | 'data' | 'runs' | 'record' | 'checks' | 'history'

const FULL_KEY = 'fd:fm-drawer-full'

const RUN_TONE: Record<Run['status'], PillTone> = {
  queued: 'zinc', running: 'violet', done: 'green', failed: 'red', cancelled: 'zinc',
}

function eventText(e: FactoryEvent): string {
  const d = e.detail
  switch (e.event) {
    case 'created': return 'Created'
    case 'draft': return `Draft v${String(d.version ?? '')} started${d.note ? `: ${String(d.note)}` : ''}`
    case 'draft_edited': return `Draft v${String(d.version ?? '')} changed`
    case 'submitted': return `Draft v${String(d.version ?? '')} sent for approval`
    case 'approved': return `Version ${String(d.version ?? '')} approved${d.grants_added ? ` (grants added: ${(d.grants_added as string[]).join(', ')})` : ''}`
    case 'declined': return `Draft v${String(d.version ?? '')} declined${d.reason ? `: ${String(d.reason)}` : ''}`
    case 'state': return `${STATE_LABELS[d.to as keyof typeof STATE_LABELS] ?? String(d.to ?? '')}${d.reason ? `: ${String(d.reason)}` : ''}`
    case 'trust': return `${e.actor === 'record' ? 'Its record raised trust to' : e.actor === 'deck' ? 'Trust lowered to' : 'Trust:'} ${TRUST_LABELS[d.to as keyof typeof TRUST_LABELS] ?? String(d.to ?? '')}${d.pinned ? ' (pinned)' : ''}`
    case 'updated': return `Changed ${Object.keys(d).join(', ').replace(/_/g, ' ')}`
    case 'run_started': return 'Run started'
    case 'run_continued': return 'Run continued where it stopped'
    case 'archived': return d.paused ? 'Archived, and paused' : 'Archived'
    case 'restored': return 'Restored to the list'
    case 'run_finished': return `Run ${RUN_LABELS[d.status as Run['status']]?.toLowerCase() ?? String(d.status ?? '')}`
    case 'recipe_edited': return 'Edited the recipe by hand'
    case 'recipe_forgotten': return 'Forgot the recipe: the next run builds a new one'
    case 'checked_forgotten': return `Forgot the ${String(d.items ?? '')} items its runs had checked`
    case 'case_added': return 'Test case added'
    case 'case_deleted': return 'Test case deleted'
    case 'check_started': return `Check of v${String(d.version ?? '')} started (${String(d.cases ?? '')} cases)`
    case 'check_finished': return d.status === 'done'
      ? `Check of v${String(d.version ?? '')}: ${pct(d.score as number | null)}`
      : `Check of v${String(d.version ?? '')} ${d.status === 'void' ? 'called off' : 'failed'}`
    default: return e.event.replace(/_/g, ' ')
  }
}

/** The editable part of a draft, as the form holds it: its goals each on or off. */
interface Form { charter: string; goals: Goal[]; schedule: string; runner: Version['runner']; leftovers: boolean; loosen: boolean; searches: string; expert: string; veto: boolean }

function formOf(v: Version): Form {
  const sched = v.triggers.find((t): t is Extract<Trigger, { kind: 'schedule' }> => t.kind === 'schedule')
  return {
    charter: v.charter,
    goals: goalsOf(v.checks),
    schedule: sched?.schedule ?? '',
    runner: v.runner,
    leftovers: (v.runner_config?.leftovers ?? 'agent') !== 'none',
    loosen: (v.runner_config?.loosen ?? 'short') !== 'never',
    searches: v.runner_config?.search_budget == null ? '' : String(v.runner_config.search_budget),
    expert: typeof v.runner_config?.expert === 'string' ? v.runner_config.expert : '',
    veto: v.runner_config?.expert_veto === true,
  }
}

/** The runner settings a form asks for: the version's own while its runner stays. */
function runnerConfigOf(form: Form, v: Version): Record<string, unknown> {
  if (form.runner === 'funnel') {
    const { search_budget: _was, expert: _who, expert_veto: _veto, ...rest } = (v.runner === 'funnel' ? v.runner_config : {}) as Record<string, unknown>
    void _was; void _who; void _veto
    return { ...rest, leftovers: form.leftovers ? 'agent' : 'none', loosen: form.loosen ? 'short' : 'never',
      ...(form.searches.trim() ? { search_budget: Math.min(100000, Number(form.searches)) } : {}),
      ...(form.expert ? { expert: form.expert, ...(form.veto ? { expert_veto: true } : {}) } : {}) }
  }
  return form.runner === v.runner ? v.runner_config : {}
}

function fieldsOf(form: Form, v: Version): DraftFields {
  const others = v.triggers.filter((t) => t.kind !== 'schedule')
  const schedule = form.schedule.trim()
  return {
    charter: form.charter,
    checks: { ...v.checks, ...checksOf(form.goals) },
    triggers: [...(schedule ? [{ kind: 'schedule' as const, schedule, tz: '' }] : []), ...others],
    runner: form.runner,
    runner_config: runnerConfigOf(form, v),
  }
}

/** Whether a form asks for what the draft already holds: every field runnerConfigOf saves counts,
 *  the expert and its veto included (the veto is saved only with an expert). */
const same = (a: Form, b: Form) => a.charter === b.charter && a.schedule.trim() === b.schedule.trim()
  && a.runner === b.runner && (a.runner !== 'funnel' || (a.leftovers === b.leftovers && a.loosen === b.loosen && a.searches.trim() === b.searches.trim()
    && a.expert === b.expert && (!a.expert || a.veto === b.veto)))
  && JSON.stringify(checksOf(a.goals)) === JSON.stringify(checksOf(b.goals))

export function FactoryDrawer({ factoryId, initialTab = 'charter', onClose, onChanged, onNotice }: {
  factoryId: string
  /** Open on this tab (a proposal opens on Runs). */
  initialTab?: Tab
  onClose: () => void
  onChanged: () => void
  onNotice: (text: string) => void
}) {
  const [data, setData] = useState<FactoryDetail | null>(null)
  const [tab, setTab] = useState<Tab>(initialTab)
  // The owner's agents, for picking the one that works beside a list factory's runs.
  const [agents, setAgents] = useState<{ slug: string; name: string }[]>([])
  useEffect(() => {
    let stop = false
    getMyAgent().then((a) => { if (!stop) setAgents(a.agents) }).catch(() => {})
    return () => { stop = true }
  }, [])
  // Full screen, remembered in this browser for the next factory opened.
  const [full, setFull] = useState(() => { try { return localStorage.getItem(FULL_KEY) === '1' } catch { return false } })
  const toggleFull = () => setFull((v) => {
    try { localStorage.setItem(FULL_KEY, v ? '0' : '1') } catch { /* private window */ }
    return !v
  })
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [form, setForm] = useState<Form | null>(null)
  const [note, setNote] = useState('')
  const [record, setRecord] = useState(false)
  const [withData, setWithData] = useState(false)
  // The Runs tab's archived view: the runs taken off it, to look at or restore.
  const [showArchived, setShowArchived] = useState(false)
  const [archivedList, setArchivedList] = useState<Run[] | null>(null)
  const [seed, setSeed] = useState<CaseSeed | null>(null)
  const seedUsed = useCallback(() => setSeed(null), [])

  const load = useCallback(async () => {
    try {
      const d = await getFactory(factoryId)
      setData(d)
      setForm(d.draft ? formOf(d.draft) : null)
      setError(null)
    } catch (e) {
      setError(errText(e))
    }
  }, [factoryId])
  useEffect(() => { void load() }, [load])
  const loadArchived = useCallback(async () => {
    try {
      setArchivedList(await listRuns(factoryId, true))
    } catch (e) {
      setError(errText(e))
    }
  }, [factoryId])
  useEffect(() => { if (showArchived) void loadArchived() }, [showArchived, loadArchived])

  // Escape closes the drawer only: taken first and marked handled, so the
  // dialog under it (the kiosk's) stays open. A viewer or table opened over it
  // (data-above-drawer) closes first, on its own.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented || document.querySelector('[data-above-drawer]')) return
      e.preventDefault()
      onClose()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  // While a run or a check is going, look again every few seconds.
  const going = !!data?.runs.some((r) => r.status === 'queued' || r.status === 'running')
  const checking = !!data?.check
  useEffect(() => {
    if (!going && !checking) return
    const t = setInterval(() => { void load() }, 5000)
    return () => clearInterval(t)
  }, [going, checking, load])

  const act = async (key: string, fn: () => Promise<unknown>, done?: string) => {
    setBusy(key); setError(null)
    try {
      await fn()
      if (done) onNotice(done)
      await load()
      onChanged()
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(null)
    }
  }
  const setCap = (tokens: number) => void act('cap', () => patchFactory(factoryId, { run_tokens: tokens }),
    tokens < 0 ? 'No cap: each run goes until it’s done or stopped'
      : tokens ? `Each run may now use ${tokensText(tokens)}` : 'Each run uses the cap it inherits')

  const f = data
  const draft = f?.draft ?? null
  const dirty = !!(draft && form && !same(form, formOf(draft)))
  const canRun = !!f?.live && ['running', 'paused', 'stopped'].includes(f.state) && !going && !f.archived_at

  // A live factory with cases checks a new version before it goes live.
  const gate = f?.check?.purpose === 'approve' && f.check.version === draft?.version ? f.check : null
  const last = f?.scorecard.check
  const held = !gate && !!draft && last?.purpose === 'approve' && last.version === draft.version
    && last.version_sha === draft.sha && last.status === 'done' ? last : null  // edited since: approving checks again
  const cases = f?.scorecard.cases ?? 0

  const saveDraft = () => act('save', () => editDraft(factoryId, fieldsOf(form!, draft!)), 'Draft saved')
  const approveDraft = (skipCheck = false) => act('approve', async () => {
    const latest = dirty ? (await editDraft(factoryId, fieldsOf(form!, draft!))).draft! : draft!
    const got = await approve(factoryId, latest.version, latest.sha, skipCheck)
    onNotice(got.check ? `Checking version ${latest.version} on its ${got.check.cases.length} cases before it goes live`
      : f?.version ? `Version ${latest.version} is live` : `${f!.title} is approved and running`)
  })
  const makeCase = (runId: string) => act(`case:${runId}`, async () => {
    setSeed(await caseDraft(runId))
    setTab('checks')
  })

  const tabCls = (t: Tab) => `shrink-0 whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
    tab === t ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:bg-[var(--fd-hover)] hover:text-zinc-200'}`
  const input = 'w-full rounded-xl border border-[var(--fd-line)] bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none'

  const headBtn = 'flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200'
  const wrap = full ? 'w-full wide:px-8' : ''
  return createPortal(
    <div className={`fd-fade fixed inset-0 z-[60] flex justify-end bg-black/30 ${full ? '' : 'wide:p-3'}`} onClick={onClose}>
      <aside
        role="dialog"
        aria-label={f ? `Factory: ${f.title}` : 'Factory'}
        className={`fd-drawer fd-dialog @container flex h-full w-full flex-col overflow-hidden bg-zinc-950 ${full ? ''
          : 'max-w-xl border-l border-[var(--fd-line)] wide:rounded-[var(--fd-radius-panel)] wide:border'}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-1 px-3 pt-3">
          <button onClick={onClose} title="Back to First Mate" aria-label="Back to First Mate"
            className="inline-flex h-9 items-center gap-1 rounded-full pl-2 pr-3 text-[13px] font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100">
            <ChevronLeft className="h-4 w-4" />First Mate
          </button>
          <span className="flex-1" />
          <button onClick={toggleFull} className={`${headBtn} hidden wide:flex`} aria-pressed={full}
            title={full ? 'Exit full screen' : 'Full screen'} aria-label={full ? 'Exit full screen' : 'Full screen'}>
            {full ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
          <button onClick={onClose} title="Close" aria-label="Close" className={headBtn}>
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className={`px-5 pb-3 pt-1 ${wrap}`}>
          {f ? (
            <>
              <div className="flex flex-wrap items-center gap-1.5">
                <Pill tone={f.state === 'running' ? 'green' : f.state === 'paused_budget' ? 'amber' : 'zinc'}>{STATE_LABELS[f.state]}</Pill>
                {f.live && <Pill tone="violet">{RUNNER_LABELS[f.live.runner]}</Pill>}
                {f.live && <Pill tone="zinc">v{f.live.version}</Pill>}
              </div>
              <h2 className="mt-2 text-xl font-bold leading-snug tracking-tight text-zinc-100">{f.title}</h2>
              {f.state_reason && f.state_reason !== 'archived' && <p className="mt-0.5 text-xs text-zinc-500">{f.state_reason}</p>}
              {f.archived_at && (
                <div className="mt-3 flex flex-wrap items-center gap-3 rounded-xl border border-dashed border-[var(--fd-line)] bg-zinc-500/[0.06] px-3 py-2.5">
                  <Archive className="h-4 w-4 shrink-0 text-zinc-500" />
                  <p className="min-w-0 flex-1 text-xs text-zinc-400">
                    <span className="font-semibold text-zinc-200">Archived {whenText(f.archived_at)}.</span>{' '}
                    It doesn't start on its own and isn't on your list; its runs and results are kept.
                  </p>
                  <button onClick={() => void act('restore', () => archiveFactory(factoryId, false),
                    `${f.title} is back on your list, paused: resume it when you want it to run`)} disabled={!!busy}
                    className="inline-flex h-8 items-center gap-1.5 rounded-full border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] px-3 text-xs font-medium text-zinc-200 transition-colors hover:border-violet-500/40">
                    {busy === 'restore' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ArchiveRestore className="h-3.5 w-3.5" />}Restore
                  </button>
                </div>
              )}
            </>
          ) : <Loader2 className="mt-1 h-4 w-4 animate-spin text-zinc-500" />}
        </div>
        {f && (
          <div role="tablist" className={`flex flex-wrap gap-1 px-5 pb-3 ${wrap}`}>
            <button role="tab" aria-selected={tab === 'charter'} className={tabCls('charter')} onClick={() => setTab('charter')}>Charter</button>
            {f.live?.runner === 'funnel' && (
              <button role="tab" aria-selected={tab === 'recipe'} className={tabCls('recipe')} onClick={() => setTab('recipe')}>Recipe</button>
            )}
            <button role="tab" aria-selected={tab === 'results'} className={tabCls('results')} onClick={() => setTab('results')}>Results</button>
            <button role="tab" aria-selected={tab === 'data'} className={tabCls('data')} onClick={() => setTab('data')}>
              Data{f.data?.length ? ` · ${f.data.length}` : ''}
            </button>
            <button role="tab" aria-selected={tab === 'runs'} className={tabCls('runs')} onClick={() => setTab('runs')}>
              Runs{f.runs.length ? ` · ${f.runs.length}` : ''}
            </button>
            <button role="tab" aria-selected={tab === 'record'} className={tabCls('record')} onClick={() => setTab('record')}>Record</button>
            <button role="tab" aria-selected={tab === 'checks'} className={tabCls('checks')} onClick={() => setTab('checks')}>
              Checks{cases ? ` · ${cases}` : ''}
            </button>
            <button role="tab" aria-selected={tab === 'history'} className={tabCls('history')} onClick={() => setTab('history')}>History</button>
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto border-t border-[var(--fd-line)] text-sm">
          <div className={`space-y-5 px-5 pb-8 pt-4 ${wrap}`}>
          {error && <ErrorBox text={error} onDismiss={() => setError(null)} />}

          {f && tab === 'charter' && (
            <>
              {draft && form && (
                <section className="rounded-2xl border border-violet-500/25 bg-[var(--fd-surface-raised)] shadow-sm">
                  <header className="flex items-center gap-3 rounded-t-2xl border-b border-[var(--fd-line)] bg-gradient-to-r from-violet-500/[0.08] via-fuchsia-500/[0.04] to-transparent px-4 py-3">
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-md shadow-violet-500/20">
                      <ScrollText className="h-4 w-4" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <h3 className="text-sm font-semibold text-zinc-100">{f.version ? `Draft v${draft.version}` : 'Your charter'}</h3>
                      <p className="text-[11px] text-zinc-500">Read it, switch goals on or off, then approve. Nothing runs before that.</p>
                    </div>
                    <Pill tone="amber">{f.version ? 'Not live yet' : 'Waiting for your approval'}</Pill>
                  </header>
                  <div className="space-y-6 px-4 py-4">
                    <div className="grid gap-6 @6xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] @6xl:gap-8">
                      <CharterText value={form.charter} onChange={(charter) => setForm({ ...form, charter })} />
                      <div className="space-y-6">
                        <GoalList goals={form.goals} onChange={(goals) => setForm({ ...form, goals })} />
                        <SchedulePicker value={form.schedule} onChange={(schedule) => setForm({ ...form, schedule })} />
                        {(form.runner === 'bat' || form.runner === 'funnel') && (
                          <RunnerPicker value={form.runner} leftovers={form.leftovers} loosen={form.loosen}
                            listy={form.goals.some((g) => g.on && goalKind(g.text) === 'countable')}
                            searches={form.searches} onSearches={(searches) => setForm({ ...form, searches })}
                            expert={form.expert} agents={agents} onExpert={(expert) => setForm({ ...form, expert })}
                            veto={form.veto} onVeto={(veto) => setForm({ ...form, veto })}
                            onChange={(runner, leftovers, loosen) => setForm({ ...form, runner, leftovers, loosen })} />
                        )}
                      </div>
                    </div>
                  {gate && (
                    <div className="flex items-start gap-2 rounded-xl border border-violet-500/30 bg-violet-500/[0.06] px-3 py-2 text-xs text-zinc-300">
                      <Loader2 className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin text-violet-500" />
                      <span>
                        Being checked on its cases: {Object.values(gate.results).filter((r) => r.score != null || r.note).length} of {gate.cases.length} graded.
                        {' '}It goes live if it scores about as well as version {f.version}
                        {gate.baseline != null ? ` (${pct(gate.baseline)})` : ''}. Changing it calls the check off.
                      </span>
                    </div>
                  )}
                  {held && (
                    <div className="rounded-xl border border-amber-500/30 bg-amber-500/[0.06] px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
                      {held.note || `It scored ${pct(held.score)} on its cases and stayed a draft.`}
                      {' '}Look at its check on the Checks tab, change it, or approve it anyway.
                    </div>
                  )}
                  </div>
                  <div className="sticky bottom-0 z-[1] flex flex-wrap items-center justify-end gap-2 rounded-b-2xl border-t border-[var(--fd-line)] bg-[var(--fd-surface-raised)]/95 px-4 py-3 backdrop-blur">
                    {dirty && <span className="mr-auto inline-flex items-center gap-1.5 text-[11px] text-amber-700 dark:text-amber-300"><span className="h-1.5 w-1.5 rounded-full bg-amber-500" />Changes not saved</span>}
                    {!dirty && f.version > 0 && cases > 0 && !gate && !held && (
                      <span className="mr-auto text-[11px] text-zinc-500">It's checked on its {cases} {cases === 1 ? 'case' : 'cases'} before it goes live.</span>
                    )}
                    <button onClick={() => void act('decline', () => decline(factoryId, draft.version), 'Draft discarded')}
                      disabled={!!busy} className="fd-btn-quiet h-8 px-3 text-xs">
                      {busy === 'decline' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null} Discard draft
                    </button>
                    <button onClick={() => void saveDraft()} disabled={!!busy || !dirty} className="fd-btn-quiet h-8 px-3 text-xs">
                      {busy === 'save' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null} Save
                    </button>
                    <button onClick={() => void approveDraft(!!held)} disabled={!!busy || !form.charter.trim() || !!gate}
                      className="inline-flex h-9 items-center gap-1.5 rounded-full bg-violet-600 px-4 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-violet-500 disabled:opacity-60">
                      {busy === 'approve' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                      {!f.version ? 'Approve and start' : held ? 'Approve anyway' : gate ? 'Being checked' : 'Approve this version'}
                    </button>
                  </div>
                </section>
              )}

              {f.live ? (
                <LiveVersion version={f.live} trust={f.trust} pinned={f.trust_pinned} approvedAt={f.live.decided_at}
                  cap={<CapPicker id="fm-cap" cap={f.run_cap} own={f.run_tokens ?? 0} busy={busy === 'cap'} onPick={setCap} />}
                  onChange={draft ? undefined : () => void act('draft', () => newDraft(factoryId))}
                  busy={busy === 'draft'} trustBusy={busy === 'trust'}
                  onTrust={(level, pin) => void act('trust', () => setTrust(factoryId, level, pin),
                    pin === undefined ? `Trust: ${TRUST_LABELS[level]}` : pin ? 'Trust pinned' : 'Trust unpinned')} />
              ) : !draft ? (
                <p className="text-zinc-500">This factory has no draft. Start one to write its charter again.</p>
              ) : null}
              {!f.live && !draft && (
                <button onClick={() => void act('draft', () => newDraft(factoryId))} disabled={!!busy} className="fd-btn-quiet h-8 px-3 text-xs">
                  Start a new draft
                </button>
              )}
            </>
          )}

          {f && tab === 'runs' && (
            <>
              <div className="fd-card flex flex-wrap items-center gap-2 p-3">
                <input className={`${input} min-w-0 flex-1`} value={note} placeholder="A note for this run (optional)"
                  onChange={(e) => setNote(e.target.value)} disabled={!canRun} />
                <button onClick={() => void act('run', async () => { await runNow(factoryId, note, [], record); setNote('') }, record ? 'Run started, recording what it reads' : 'Run started')}
                  disabled={!canRun || !!busy}
                  title={canRun ? 'Start a run now' : going ? 'A run is going' : 'Approve a charter first'}
                  className="inline-flex h-9 items-center gap-1.5 rounded-full bg-violet-600 px-4 text-xs font-semibold text-white transition-colors hover:bg-violet-500 disabled:opacity-60">
                  {busy === 'run' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />} Run now
                </button>
                <button onClick={() => setWithData(true)} disabled={!canRun || !!busy}
                  title={canRun ? 'Start a run with extra data for this run alone' : going ? 'A run is going' : 'Approve a charter first'}
                  className="inline-flex h-9 items-center gap-1.5 rounded-full border border-[var(--fd-line)] px-3.5 text-xs font-medium text-zinc-300 transition-colors hover:border-violet-500/40 hover:text-zinc-100 disabled:opacity-60">
                  <Database className="h-3.5 w-3.5" /> Run with data…
                </button>
                {f.live?.runner === 'funnel' && (
                  <label className="inline-flex h-9 cursor-pointer items-center gap-1.5 px-1 text-xs text-zinc-400"
                    title="Keeps every search and page this run reads (in its folder, shared/funnel/tape), so the run can be replayed later as a test set">
                    <input type="checkbox" checked={record} onChange={(e) => setRecord(e.target.checked)} disabled={!canRun} />
                    Record this run
                  </label>
                )}
                {(f.data?.length ?? 0) > 0 && (
                  <p className="w-full text-xs text-zinc-500">
                    Every run gets the factory's {f.data.length} file{f.data.length === 1 ? '' : 's'} of extra data.{' '}
                    <button onClick={() => setTab('data')} className="font-medium text-violet-700 hover:underline dark:text-violet-300">See them</button>
                  </p>
                )}
                {f.live && (
                  <div className="flex w-full flex-wrap items-center gap-2 text-xs text-zinc-500">
                    <label htmlFor="fm-cap-runs">Each run may use</label>
                    <CapPicker id="fm-cap-runs" cap={f.run_cap} own={f.run_tokens ?? 0} busy={busy === 'cap'} onPick={setCap} />
                    <span>{f.run_cap.tokens < 0 ? 'A run going now goes on with no cap; Continue does too.' : 'A run going now follows a change from its next step; Continue gives a stopped one this much more.'}</span>
                  </div>
                )}
              </div>
              {withData && (
                <AddDataPanel title={`Run ${f.title} with extra data`}
                  subtitle={f.data?.length
                    ? `For this run alone, beside the factory's own ${f.data.length} file${f.data.length === 1 ? '' : 's'}.`
                    : 'For this run alone. To give every run something, add it on the Data tab.'}
                  action={(n) => (n ? `Run with ${n} file${n === 1 ? '' : 's'}` : 'Run with data')}
                  onDone={async (staged) => {
                    setError(null)
                    await runNow(factoryId, note, staged.map((s) => s.id), record)
                    setNote('')
                    onNotice(`Run started with ${staged.length} file${staged.length === 1 ? '' : 's'} of extra data`)
                    void load()
                    onChanged()
                  }}
                  onClose={() => setWithData(false)} />
              )}
              {(() => {
                const finished = f.runs.filter((r) => ['done', 'failed', 'cancelled'].includes(r.status)).length
                const archivedCount = f.archived_runs ?? 0
                if (!finished && !archivedCount && !showArchived) return null
                return (
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="text-zinc-500">{showArchived ? 'Archived runs: off Runs and Results, kept whole.' : `${f.runs.length} on the list`}</span>
                    <span className="flex-1" />
                    {!showArchived && finished > 0 && (
                      <button onClick={() => {
                        if (window.confirm(`Archive ${finished} finished run${finished === 1 ? '' : 's'}? They leave Runs and Results; nothing is deleted.`)) {
                          void act('archive-finished', () => archiveFinishedRuns(factoryId), `${finished} run${finished === 1 ? '' : 's'} archived`)
                        }
                      }} disabled={!!busy}
                        className="inline-flex h-7 items-center gap-1.5 rounded-full border border-[var(--fd-line)] px-2.5 font-medium text-zinc-400 transition-colors hover:border-violet-500/40 hover:text-zinc-200">
                        {busy === 'archive-finished' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Archive className="h-3 w-3" />}Archive finished runs
                      </button>
                    )}
                    {(archivedCount > 0 || showArchived) && (
                      <button onClick={() => setShowArchived((v) => !v)} aria-pressed={showArchived}
                        className={`inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 font-medium transition-colors ${showArchived
                          ? 'border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-200' : 'border-[var(--fd-line)] text-zinc-400 hover:text-zinc-200'}`}>
                        <Archive className="h-3 w-3" />{showArchived ? 'Back to the list' : `Archived · ${archivedCount}`}
                      </button>
                    )}
                  </div>
                )
              })()}
              {showArchived ? (
                archivedList === null ? (
                  <p className="flex items-center gap-1.5 text-xs text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" />Loading…</p>
                ) : archivedList.length === 0 ? (
                  <p className="text-zinc-500">No archived runs.</p>
                ) : (
                  <ul className="space-y-2.5">
                    {archivedList.map((r) => (
                      <RunItem key={r.id} run={r} busy={false} deciding={false} onStop={() => {}} continuing={false}
                        onApprove={() => {}} onReject={() => {}} onRate={(kind) => void act(`rate:${r.id}`, () => rateRun(r.id, kind))}
                        making={false} archived archiving={busy === `archive:${r.id}`}
                        onArchive={() => void act(`archive:${r.id}`, async () => { await archiveRun(r.id, false); await loadArchived() },
                          'Run restored to the list')} />
                    ))}
                  </ul>
                )
              ) : f.runs.length === 0 ? (
                <p className="text-zinc-500">No runs yet.</p>
              ) : (
                <ul className="space-y-2.5">
                  {f.runs.map((r) => (
                    <RunItem key={r.id} run={r} busy={busy === `stop:${r.id}`} deciding={busy === `decide:${r.id}`}
                      archiving={busy === `archive:${r.id}`}
                      onArchive={['done', 'failed', 'cancelled'].includes(r.status)
                        ? () => void act(`archive:${r.id}`, () => archiveRun(r.id), 'Run archived: find it under Archived')
                        : undefined}
                      onStop={() => void act(`stop:${r.id}`, () => stopRun(r.id), 'Run stopped')}
                      continuing={busy === `continue:${r.id}`}
                      onContinue={r.can_continue
                        ? () => void act(`continue:${r.id}`, () => continueRun(r.id), 'Going on where it stopped')
                        : undefined}
                      onApprove={(ids, total) => void act(`decide:${r.id}`, () => approveRun(r.id, ids.length === total ? undefined : ids),
                        `Carrying out ${ids.length} ${ids.length === 1 ? 'action' : 'actions'}, once`)}
                      onReject={() => void act(`decide:${r.id}`, () => rejectRun(r.id), 'Rejected: nothing will be done')}
                      onRate={(kind) => void act(`rate:${r.id}`, () => rateRun(r.id, kind))}
                      onMakeCase={r.trigger === 'events' && r.status === 'done' ? () => void makeCase(r.id) : undefined}
                      making={busy === `case:${r.id}`} />
                  ))}
                </ul>
              )}
            </>
          )}

          {f && tab === 'results' && <FactoryResults factoryId={factoryId} onNotice={onNotice} />}

          {f && tab === 'recipe' && <FactoryRecipe factoryId={factoryId} onNotice={onNotice} onChanged={() => void load()} />}

          {f && tab === 'data' && (
            <FactoryData factoryId={factoryId} title={f.title} items={f.data ?? []} onNotice={onNotice}
              onChanged={() => { void load(); onChanged() }} />
          )}

          {f && tab === 'record' && <Record detail={f} />}

          {f && tab === 'checks' && (
            <Checks factoryId={factoryId} live={f.version} seed={seed} onSeedUsed={seedUsed}
              onChanged={() => { void load(); onChanged() }} onNotice={onNotice} />
          )}

          {f && tab === 'history' && (
            <Section icon={History} title="History" hint="What happened to this factory, newest first">
              {f.events.length ? (
                <ol className="fd-card divide-y divide-[var(--fd-line)] overflow-hidden p-0">
                  {f.events.map((e) => (
                    <li key={e.id} className="flex gap-4 px-4 py-2.5 text-[13px] transition-colors hover:bg-[var(--fd-hover)]">
                      <span className="w-32 shrink-0 text-xs tabular-nums text-zinc-500">{whenText(e.at)}</span>
                      <span className="min-w-0 flex-1 text-zinc-300">{eventText(e)}</span>
                    </li>
                  ))}
                </ol>
              ) : <p className="text-xs text-zinc-500">Nothing yet.</p>}
            </Section>
          )}
          </div>
        </div>
      </aside>
    </div>,
    document.body,
  )
}

const TRUST_ORDER: Trust[] = ['propose', 'commit', 'direct']

function LiveVersion({ version: v, cap, trust, pinned, approvedAt, onChange, busy, trustBusy, onTrust }: {
  version: Version
  /** The factory's cap per run, to change in place. */
  cap: React.ReactNode
  trust: FactoryDetail['trust']
  pinned: boolean
  approvedAt: string | null
  onChange?: () => void
  busy: boolean
  trustBusy: boolean
  onTrust: (level: Trust, pin?: boolean) => void
}) {
  const outward = v.grants.email || v.grants.spend || v.grants.accounts
  const grants = (['email', 'organize', 'spend', 'accounts'] as const).filter((g) => v.grants[g])
  const row = 'flex gap-3 py-2'
  const key = 'w-28 shrink-0 text-xs text-zinc-500'
  return (
    <section className="rounded-2xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] shadow-sm">
      <header className="flex items-center gap-3 rounded-t-2xl border-b border-[var(--fd-line)] bg-gradient-to-r from-emerald-500/[0.07] via-violet-500/[0.03] to-transparent px-4 py-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)]">
          <ScrollText className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-zinc-100">Live charter · version {v.version}</h3>
          <p className="text-[11px] text-zinc-500">{approvedAt ? `Approved ${whenText(approvedAt)}. ` : ''}What every run works to.</p>
        </div>
        <Pill tone="green">Live</Pill>
        {onChange && (
          <button onClick={onChange} disabled={busy} className="fd-btn-quiet h-8 px-3 text-xs">
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null} Change it
          </button>
        )}
      </header>
      <div className="grid gap-4 px-4 py-4 @6xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] @6xl:items-start @6xl:gap-8">
      <blockquote className="whitespace-pre-wrap rounded-xl border-l-2 border-violet-500/50 bg-[var(--fd-hover)] px-3 py-2 leading-relaxed text-zinc-200">
        {v.charter}
      </blockquote>
      <div className="divide-y divide-[var(--fd-line)] text-[13px]">
        {(v.checks.contract.length > 0 || (v.checks.off?.length ?? 0) > 0) && (
          <div className={row}>
            <span className={key}>Done when</span>
            <GoalChecklist checks={v.checks} />
          </div>
        )}
        <div className={row}><span className={key}>Starts</span><span className="text-zinc-300">{triggersText(v.triggers)}</span></div>
        <div className={row}>
          <label className={key} htmlFor="fm-cap">Cap per run</label>
          <span className="min-w-0 flex-1">
            {cap}
            <span className="mt-1 block text-[11px] text-zinc-500">Tokens one run may use before it stops. Changing it needs no approval, and a run going now follows it from its next step. No cap suits a local model; on a paid API a run can cost a lot.</span>
          </span>
        </div>
        <div className={row}>
          <span className={key}>May act</span>
          <span className="text-zinc-300">
            {grants.length ? grants.join(', ') : 'Nothing outward: a run never sends, spends or signs up'}
          </span>
        </div>
        <div className={row}>
          <span className={key}>Trust</span>
          <span className="min-w-0 flex-1 text-zinc-300">
            <span role="radiogroup" aria-label="Trust" className="inline-flex flex-wrap gap-1">
              {TRUST_ORDER.map((level) => {
                const blocked = level === 'direct' && outward
                return (
                  <button key={level} role="radio" aria-checked={trust === level} aria-label={TRUST_LABELS[level]}
                    disabled={trustBusy || blocked || trust === level}
                    title={blocked ? 'A factory that may send email, spend or open accounts always rehearses first' : TRUST_HINTS[level]}
                    onClick={() => onTrust(level)}
                    className={`rounded-full px-2.5 py-1 text-xs transition-colors disabled:cursor-default ${
                      trust === level ? 'bg-violet-600 text-white' : blocked ? 'text-zinc-600' : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100'}`}>
                    {TRUST_LABELS[level]}
                  </button>
                )
              })}
            </span>
            <span className="mt-1 block text-xs text-zinc-500">{TRUST_HINTS[trust]}</span>
            <label className="mt-1.5 flex items-center gap-2 text-xs text-zinc-500">
              <input type="checkbox" checked={pinned} disabled={trustBusy} onChange={(e) => onTrust(trust, e.target.checked)}
                className="h-3.5 w-3.5 accent-violet-600" />
              Pin it: its record won’t raise it (a doubtful carry-out still lowers it)
            </label>
          </span>
        </div>
      </div>
      </div>
    </section>
  )
}

const HEALTH_TONE: Record<FactoryDetail['scorecard']['health'], PillTone> = {
  green: 'green', amber: 'amber', red: 'red', new: 'zinc',
}

function Record({ detail: d }: { detail: FactoryDetail }) {
  const s = d.scorecard
  const l = d.ladder
  const pct = (x: number | null) => (x == null ? '—' : `${Math.round(x * 100)}%`)
  const cell = 'fd-card px-4 py-3.5'
  const big = 'text-2xl font-semibold tracking-tight text-zinc-100'
  const small = 'text-[11px] text-zinc-500'
  return (
    <div className="space-y-7">
      <Section icon={Gauge} title="How it's doing" action={<Pill tone={HEALTH_TONE[s.health]}>{HEALTH_LABELS[s.health]}</Pill>}
        hint={s.health_why || undefined}>
      <div className="grid grid-cols-2 gap-2.5 @3xl:grid-cols-4">
        <div className={cell}><div className={big}>{pct(s.satisfaction)}</div>
          <div className={small}>Satisfaction · {s.signals} {s.signals === 1 ? 'signal' : 'signals'} · target {pct(s.target)}</div></div>
        <div className={cell}><div className={big}>{s.week.done}/{s.week.runs}</div><div className={small}>Runs done this week</div></div>
        <div className={cell}><div className={big}>{s.fortnight.doubts}/{s.fortnight.carry_outs}</div>
          <div className={small}>Doubtful carry-outs, two weeks</div></div>
        <div className={cell}><div className={big}>{s.week.tokens_per_run ? tokensText(s.week.tokens_per_run).replace(' tokens', '') : '—'}</div>
          <div className={small}>Tokens per run{s.week.median_minutes == null ? ''
            : s.week.median_minutes < 1 ? ' · under a minute' : ` · ${Math.round(s.week.median_minutes)} min`}</div></div>
      </div>
      </Section>
      {l && (
        <Section icon={ShieldCheck} title="Trust" hint="How much it may do without asking you">
          <div className="fd-card px-4 py-3.5 text-[13px]">
            <div className="flex flex-wrap items-center gap-2">
              <Pill tone="violet">{TRUST_LABELS[l.level]}</Pill>
              {l.next && <span className="text-zinc-400">→ {TRUST_LABELS[l.next]} after {l.needed} settled results at {pct(l.bar)}</span>}
              {l.pinned && <Pill tone="zinc">Pinned</Pill>}
            </div>
            {l.next && (
              <div className="mt-2 flex items-center gap-2">
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800" role="progressbar"
                  aria-valuenow={l.counted} aria-valuemin={0} aria-valuemax={l.needed} aria-label="Settled results at this level">
                  <div className={`h-full rounded-full ${l.value != null && l.value < l.bar ? 'bg-amber-500' : 'bg-violet-500'}`}
                    style={{ width: `${Math.round((Math.min(l.counted, l.needed) / l.needed) * 100)}%` }} />
                </div>
                <span className="shrink-0 text-xs text-zinc-500">{l.counted}/{l.needed} · {pct(l.value)}</span>
              </div>
            )}
            {l.why && <p className="mt-1.5 text-xs text-zinc-500">{l.why}</p>}
            {!l.calibration.ready && l.next && !l.why.includes('agree with you') && (
              <p className="mt-1 text-xs text-zinc-500">
                Trust moves on its own only once its checks agree with your labels: {l.calibration.why}.
              </p>
            )}
          </div>
        </Section>
      )}
      {d.drafts.length > 0 && (
        <Section icon={Mail} title="Its drafts in your mailbox"
          hint={`${s.drafts.sent} sent as written · ${s.drafts.edited} sent after edits · ${s.drafts.deleted} deleted · ${s.drafts.waiting} waiting`}>
          <ul className="fd-card divide-y divide-[var(--fd-line)] overflow-hidden p-0 text-[13px]">
            {d.drafts.slice(0, 8).map((x) => (
              <li key={x.id} className="flex gap-2 px-4 py-2.5">
                <span className="min-w-0 flex-1 truncate text-zinc-300">{x.subject || '(no subject)'}</span>
                <span className="shrink-0 text-xs text-zinc-500">
                  {{ waiting: 'Waiting', sent: 'Sent as written', edited: `Sent, ${pct(x.similarity)} kept`, deleted: 'Deleted', expired: 'Left alone' }[x.status] ?? x.status}
                </span>
              </li>
            ))}
          </ul>
        </Section>
      )}
      <Section icon={ThumbsUp} title="What you did with its results" hint="Your ratings, approvals and sent drafts: they set its score">
        {d.signals.length === 0 ? (
          <p className="rounded-[var(--fd-radius-card)] border border-dashed border-zinc-800 px-4 py-4 text-sm text-zinc-500">Nothing yet. Rate a run, approve or reject what it would do, or send its drafts.</p>
        ) : (
          <ol className="fd-card divide-y divide-[var(--fd-line)] overflow-hidden p-0">
            {d.signals.map((g) => (
              <li key={g.id} className="flex gap-3 px-4 py-2.5 text-[13px]">
                <span className="w-28 shrink-0 text-xs text-zinc-500">{whenText(g.at)}</span>
                <span className="min-w-0 flex-1 text-zinc-300">{SIGNAL_LABELS[g.kind] ?? g.kind}</span>
                <span className={`shrink-0 text-xs ${g.value >= 0.7 ? 'text-emerald-600 dark:text-emerald-400' : g.value < 0.3 ? 'text-red-600 dark:text-red-300' : 'text-zinc-500'}`}>
                  {Math.round(g.value * 100)}%
                </span>
              </li>
            ))}
          </ol>
        )}
      </Section>
    </div>
  )
}

const INTENT_DONE: Record<Exclude<Intent['status'], 'open'>, { tone: PillTone; label: string }> = {
  approved: { tone: 'violet', label: 'Approved' },
  carried_out: { tone: 'green', label: 'Carried out' },
  rejected: { tone: 'zinc', label: 'Rejected' },
  dropped: { tone: 'zinc', label: 'Not done' },
}

function RunItem({ run: r, busy, deciding, onStop, onContinue, continuing, onApprove, onReject, onRate, onMakeCase, making,
  onArchive, archiving = false, archived = false }: {
  run: Run
  busy: boolean
  deciding: boolean
  /** Off Runs and Results (a finished run), or back (``archived``). */
  onArchive?: () => void
  archiving?: boolean
  archived?: boolean
  onStop: () => void
  /** A stopped run that can go on where it stopped. */
  onContinue?: () => void
  continuing: boolean
  onApprove: (ids: number[], total: number) => void
  onReject: () => void
  onRate: (kind: 'up' | 'down' | '') => void
  /** A run started on fixed input (new mail) can become a test case. */
  onMakeCase?: () => void
  making: boolean
}) {
  const going = r.status === 'queued' || r.status === 'running'
  const waiting = (r.intents ?? []).filter((i) => i.status === 'open')
  // Its steps, agents and activity: open while it's going, a click away after.
  const traceable = (r.runner === 'bat' || r.runner === 'funnel') && !!r.child_id
  const [open, setOpen] = useState(going)
  useEffect(() => { if (going) setOpen(true) }, [going])  // a run that goes on opens again
  const [unticked, setUnticked] = useState<Set<number>>(new Set())
  const chosen = waiting.filter((i) => !unticked.has(i.id)).map((i) => i.id)
  const toggle = (id: number) => setUnticked((s) => {
    const n = new Set(s)
    if (n.has(id)) n.delete(id)
    else n.add(id)
    return n
  })
  return (
    <li className="fd-card p-3 @2xl:p-4">
      <div className="flex flex-wrap items-center gap-2">
        {traceable && (
          <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
            title={open ? 'Hide its progress' : 'Show its steps, agents and activity'}
            aria-label={open ? 'Hide its progress' : 'Show its progress'}
            className="-ml-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            <ChevronRight className={`h-4 w-4 transition-transform ${open ? 'rotate-90' : ''}`} />
          </button>
        )}
        <Pill tone={r.waiting ? 'amber' : RUN_TONE[r.status]}>{r.waiting ? 'Waiting for you' : RUN_LABELS[r.status]}</Pill>
        {r.kind === 'rehearsal' && <Pill tone="zinc" title="Anything outward was recorded, not done">Rehearsal</Pill>}
        {r.kind === 'carry_out' && <Pill tone="violet" title="Did what you approved, once">Carry-out</Pill>}
        {r.kind === 'carry_out' && r.proof?.state && (
          <Pill tone={r.proof.state === 'confirmed' ? 'green' : r.proof.state === 'doubt' ? 'red' : 'zinc'}
            title="What the tool-call register and Gmail/Calendar showed it did">
            {{ confirmed: 'Confirmed', doubt: 'Check it', nothing: 'Did nothing outside', unknown: 'Not checked' }[r.proof.state]}
          </Pill>
        )}
        {r.verdict === 'leak' && <Pill tone="red" title="The tool-call register shows an outward call that ran">Reached outside</Pill>}
        <span className="text-xs text-zinc-500">{whenText(r.created_at)}</span>
        {r.tokens > 0 && <span className="text-xs text-zinc-500">· {tokensText(r.tokens)}</span>}
        {r.trigger !== 'manual' && <span className="text-xs text-zinc-500">· {r.trigger}</span>}
        <span className="flex-1" />
        {onMakeCase && (
          <button onClick={onMakeCase} disabled={making} title="Make a test case from what this run got"
            aria-label="Make a test case"
            className="flex h-7 w-7 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            {making ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FlaskConical className="h-3.5 w-3.5" />}
          </button>
        )}
        {(r.status === 'done' || r.status === 'failed') && (
          <span className="inline-flex items-center gap-0.5" aria-label="How was this run?">
            <button onClick={() => onRate(r.thumbs === 'thumbs_up' ? '' : 'up')} aria-pressed={r.thumbs === 'thumbs_up'}
              title="Good result" aria-label="Good result"
              className={`flex h-7 w-7 items-center justify-center rounded-full transition-colors ${
                r.thumbs === 'thumbs_up' ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`}>
              <ThumbsUp className="h-3.5 w-3.5" />
            </button>
            <button onClick={() => onRate(r.thumbs === 'thumbs_down' ? '' : 'down')} aria-pressed={r.thumbs === 'thumbs_down'}
              title="Bad result" aria-label="Bad result"
              className={`flex h-7 w-7 items-center justify-center rounded-full transition-colors ${
                r.thumbs === 'thumbs_down' ? 'bg-red-500/10 text-red-600 dark:text-red-300' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`}>
              <ThumbsDown className="h-3.5 w-3.5" />
            </button>
          </span>
        )}
        {onArchive && (
          <button onClick={onArchive} disabled={archiving}
            title={archived ? 'Restore it to Runs and Results' : 'Archive: off Runs and Results, kept whole'}
            aria-label={archived ? 'Restore this run' : 'Archive this run'}
            className="flex h-7 w-7 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            {archiving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : archived ? <ArchiveRestore className="h-3.5 w-3.5" /> : <Archive className="h-3.5 w-3.5" />}
          </button>
        )}
        {onContinue && !archived && (
          <button onClick={onContinue} disabled={continuing}
            title="Go on where it stopped: finished steps stay finished, with your cap per run again"
            className="inline-flex h-7 items-center gap-1 rounded-full bg-violet-600 px-2.5 text-xs font-semibold text-white transition-colors hover:bg-violet-500 disabled:opacity-60">
            {continuing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />} Continue
          </button>
        )}
        {going && (
          <button onClick={onStop} disabled={busy}
            className="inline-flex h-7 items-center gap-1 rounded-full px-2.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100">
            {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Square className="h-3 w-3" />} Stop
          </button>
        )}
      </div>
      {r.waiting && r.note && <p className="mt-1.5 text-xs text-amber-700 dark:text-amber-300">{r.note}</p>}
      {!going && !(traceable && open) && <TimingChips timing={r.timing} className="mt-2" />}
      {traceable && open && <RunProgress runId={r.id} going={going} stopped={!!onContinue} />}
      {r.kind === 'carry_out' && r.proof?.state && (r.proof.note || (r.proof.actions?.length ?? 0) > 0) && (
        <div className={`mt-2 rounded-xl px-3 py-2 text-xs ${r.proof.state === 'doubt' ? 'bg-red-500/[0.07] text-red-700 dark:text-red-300' : 'bg-[var(--fd-hover)] text-zinc-400'}`}>
          {r.proof.note && <p>{r.proof.note}</p>}
          {(r.proof.actions?.length ?? 0) > 0 && (
            <ul className={`${r.proof.note ? 'mt-1.5' : ''} space-y-0.5 text-zinc-400`}>
              {r.proof.actions!.map((a, n) => <li key={n}>• {a}</li>)}
            </ul>
          )}
        </div>
      )}
      {r.kind === 'rehearsal' && !going && r.intents && (
        r.intents.length ? (
          <div className={`mt-2 rounded-xl px-3 py-2 ${waiting.length ? 'border border-amber-500/30 bg-amber-500/[0.06]' : 'bg-[var(--fd-hover)]'}`}>
            <p className="text-[11px] font-semibold text-zinc-400">
              {waiting.length ? 'It would do this. Nothing of it is done until you say so.' : 'It would have done this'}
            </p>
            <ul className="mt-1.5 space-y-1.5 text-[13px] text-zinc-300">
              {r.intents.slice(0, 20).map((i) => (
                <li key={i.id} className="flex gap-2">
                  {i.status === 'open' ? (
                    <input type="checkbox" checked={!unticked.has(i.id)} onChange={() => toggle(i.id)} disabled={deciding}
                      aria-label={`Carry out: ${i.what}`} className="mt-0.5 h-3.5 w-3.5 shrink-0 accent-violet-600" />
                  ) : <span className="w-3.5 shrink-0 text-zinc-500">{i.kind === 'commit' ? '⏸' : '•'}</span>}
                  <span className="min-w-0 flex-1">
                    <span className="flex flex-wrap items-center gap-1.5">
                      {i.what}
                      {i.status !== 'open' && <Pill tone={INTENT_DONE[i.status].tone}>{INTENT_DONE[i.status].label}</Pill>}
                    </span>
                    {i.detail && <span className="mt-0.5 block whitespace-pre-wrap text-xs leading-relaxed text-zinc-500">{i.detail}</span>}
                  </span>
                </li>
              ))}
              {r.intents.length > 20 && <li className="text-xs text-zinc-500">and {r.intents.length - 20} more</li>}
            </ul>
            {waiting.length > 0 && (
              <div className="mt-2.5 flex flex-wrap items-center justify-end gap-2 border-t border-amber-500/20 pt-2.5">
                <button onClick={onReject} disabled={deciding} className="fd-btn-quiet h-8 px-3 text-xs">Reject</button>
                <button onClick={() => onApprove(chosen, waiting.length)} disabled={deciding || chosen.length === 0}
                  className="inline-flex h-8 items-center gap-1.5 rounded-full bg-violet-600 px-3.5 text-xs font-semibold text-white transition-colors hover:bg-violet-500 disabled:opacity-60">
                  {deciding ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                  Carry out {chosen.length === waiting.length ? 'all' : `${chosen.length} of ${waiting.length}`}, once
                </button>
              </div>
            )}
          </div>
        ) : <p className="mt-1.5 text-xs text-zinc-500">It needed nothing outside.</p>
      )}
      {r.error && !going && <p className="mt-1.5 text-xs text-zinc-500">{r.error}</p>}
      {r.result && <RunResult text={r.result} />}
    </li>
  )
}
