import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Anchor, BellDot, CalendarDays, Factory as FactoryIcon, Loader2, MoreHorizontal, PanelRightOpen, Pause, Play, Plus,
  RefreshCw, Square, Trash2, Archive, X, ShieldCheck, Ship,
} from 'lucide-react'
import {
  createFactory,
  deleteFactory,
  getReport,
  getStatus,
  getTemplates,
  listFactories,
  runNow,
  saveDeckSettings,
  savePrefs,
  setState,
  tokensText,
  triggersText,
  whenText,
  RUN_LABELS,
  RUNNER_LABELS,
  HEALTH_LABELS,
  STATE_LABELS,
  TRUST_LABELS,
  type FactoryCard,
  type OwnerState,
  type Status,
  type Template,
  type WeekReport,
} from '../services/firstMate'
import { useFirstMateStore } from '../stores/firstMateStore'
import { QueueMenu, type MenuEntry } from '../components/agents/queue/QueueMenu'
import { FactoryDrawer } from '../components/firstMate/FactoryDrawer'
import { EmptyBand, ErrorBox, Pill, Section, Switch, Toast, errText, type PillTone } from '../components/firstMate/ui'

// First Mate: standing jobs ("factories") that run for the user. Each has a
// charter they approve, runs through Bat, and is checked every run. See
// docs/first-mate-plan.md.
//
// On top, "Needs you": charters waiting for approval, factories over budget,
// runs waiting on the owner. Then the factories, one row each, with Run now
// and a ⋯ menu; a row opens its drawer (charter, runs, history). A deck with
// First Mate off shows why, and its admins get the switch.
//
// `onClose`: shown in the kiosk's dialog, a close button in the header and
// Escape closes it (not while the drawer or the New factory picker is open).

const STATE_TONE: Record<FactoryCard['state'], PillTone> = {
  draft: 'zinc', awaiting_charter: 'amber', running: 'green', paused: 'zinc', paused_budget: 'amber',
  stopped: 'zinc', retired: 'zinc',
}

export function FirstMatePage({ onClose }: { onClose?: () => void }) {
  const [status, setStatusData] = useState<Status | null>(null)
  const [factories, setFactories] = useState<FactoryCard[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [opened, setOpened] = useState<string | null>(null)
  const [openedTab, setOpenedTab] = useState<'charter' | 'runs'>('charter')
  const openAt = (id: string, tab: 'charter' | 'runs' = 'charter') => { setOpenedTab(tab); setOpened(id) }
  const [picking, setPicking] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [scrolled, setScrolled] = useState(false)
  const reloadNav = useFirstMateStore((s) => s.load)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const s = await getStatus()
      setStatusData(s)
      setFactories(s.available ? await listFactories() : [])
      setError(null)
    } catch (e) {
      setError(errText(e))
    } finally {
      setLoading(false)
    }
  }, [])
  useEffect(() => { void load() }, [load])

  useEffect(() => {
    if (!notice) return
    const t = setTimeout(() => setNotice(null), 5000)
    return () => clearTimeout(t)
  }, [notice])

  // A run going somewhere: look again now and then, so its row moves on.
  const going = factories.some((f) => f.last_run && (f.last_run.status === 'queued' || f.last_run.status === 'running'))
  useEffect(() => {
    if (!going || opened) return
    const t = setInterval(() => { void listFactories().then(setFactories).catch(() => {}) }, 8000)
    return () => clearInterval(t)
  }, [going, opened])

  const holdRef = useRef(false)
  holdRef.current = !!opened || picking
  useEffect(() => {
    if (!onClose) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !e.defaultPrevented && !holdRef.current) onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const act = async (key: string, fn: () => Promise<unknown>, done?: string) => {
    setBusy(key); setError(null)
    try {
      await fn()
      if (done) setNotice(done)
      await load()
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(null)
    }
  }

  const deckSwitch = async (change: { enabled?: boolean; admins_only?: boolean }) => {
    await act('deck', () => saveDeckSettings(change))
    void reloadNav()
  }

  const streak = (f: FactoryCard) => f.state === 'paused' && /failed runs in a row/.test(f.state_reason)
  const needs = useMemo(() => factories.filter((f) => f.proposal || f.draft || f.state === 'paused_budget' || streak(f)
    || f.last_run?.waiting), [factories])

  const create = async (body: { template?: string; title?: string; charter?: string }) => {
    setBusy('create'); setError(null)
    try {
      const f = await createFactory(body)
      setPicking(false)
      await load()
      openAt(f.id)
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(null)
    }
  }

  const max = status?.settings.max_factories ?? 8
  const atCap = factories.length >= max

  return (
    <div className="@container relative flex h-full min-h-0 flex-col bg-zinc-950 text-zinc-200">
      <header className={`shrink-0 border-b transition-colors ${scrolled ? 'border-[var(--fd-line)]' : 'border-transparent'}`}>
        <div className="mx-auto w-full max-w-4xl px-5 pb-4 pt-5 wide:px-8 wide:pb-5 wide:pt-7">
          <div className="flex items-center gap-2">
            <h2 className="min-w-0 flex-1 text-xl font-bold tracking-tight text-zinc-100">First Mate</h2>
            <button onClick={() => void load()} disabled={loading} title="Check again" aria-label="Check again"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-60">
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            {onClose && (
              <button onClick={onClose} title="Close" aria-label="Close"
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
          <p className="mt-0.5 max-w-2xl text-sm leading-relaxed text-zinc-500">
            Standing jobs that run for you. Each one has a charter you approve, runs through Bat, and is checked every time.
          </p>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto" onScroll={(e) => setScrolled(e.currentTarget.scrollTop > 4)}>
        <div className="mx-auto w-full max-w-4xl space-y-9 px-5 pb-24 wide:px-8">
          {error && <ErrorBox text={error} onDismiss={() => setError(null)} />}
          {!status && loading && (
            <div className="flex h-48 items-center justify-center"><Loader2 className="h-5 w-5 animate-spin text-zinc-500" /></div>
          )}

          {status && !status.available && (
            <section className="fd-card p-5">
              <div className="flex items-start gap-3">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[0.7rem] bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)]">
                  <Anchor className="h-4 w-4" />
                </span>
                <div className="min-w-0 flex-1">
                  <h3 className="text-[15px] font-semibold text-zinc-100">Not on yet</h3>
                  <p className="mt-0.5 text-sm text-zinc-400">
                    {status.is_admin && !status.settings.enabled ? 'First Mate is off on this deck. Turn it on below.' : status.reason}
                  </p>
                </div>
              </div>
              {status.is_admin && (
                <DeckSettings className="mt-3" status={status} busy={busy === 'deck'} onChange={(c) => void deckSwitch(c)} />
              )}
            </section>
          )}

          {status?.available && (
            <>
              {needs.length > 0 && (
                <Section icon={BellDot} tone="amber" title="Needs you" count={needs.length}
                  hint="Nothing runs on a charter you haven't approved.">
                  <div className="fd-card overflow-hidden">
                    <ul className="divide-y divide-[var(--fd-line)]">
                      {needs.map((f) => (
                        <li key={f.id}>
                          <button onClick={() => openAt(f.id, f.proposal ? 'runs' : 'charter')}
                            className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-[var(--fd-hover)]">
                            <span className="min-w-0 flex-1">
                              <span className="block text-[15px] font-medium text-zinc-100">{f.title}</span>
                              <span className="block text-xs text-zinc-500">
                                {f.proposal ? `It would do ${f.proposal.count} ${f.proposal.count === 1 ? 'thing' : 'things'}: carry them out or reject them`
                                  : f.draft ? (f.version ? `Draft v${f.draft.version} waits for your approval` : 'Its charter waits for your approval')
                                  : f.state === 'paused_budget' ? 'Used its share of this week’s tokens'
                                    : streak(f) ? `Paused after ${f.state_reason}: see why, then resume it`
                                      : 'A run waits for you on the Bat page'}
                              </span>
                            </span>
                            <PanelRightOpen className="h-4 w-4 shrink-0 text-zinc-500" />
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                </Section>
              )}

              <Section icon={Ship} title="Your First Mate"
                hint={status.prefs.on ? 'Schedules and new mail start your factories’ runs.' : 'Off: your factories run only when you press Run now.'}>
                <YourFirstMate status={status} busy={busy === 'prefs'}
                  onChange={(change) => void act('prefs', () => savePrefs(change),
                    change.on === undefined ? 'Weekly budget saved' : change.on ? 'First Mate is on' : 'First Mate is off: running factories paused')} />
              </Section>

              <Section icon={FactoryIcon} title="Factories"
                hint={`${factories.length} of ${max}. Each run is a Bat run: its checks, its reviewers, its questions to you.`}
                action={!picking && !atCap ? (
                  <button onClick={() => setPicking(true)} className="fd-btn-quiet h-8 shrink-0 px-3 text-xs">
                    <Plus className="h-3.5 w-3.5" /> New factory
                  </button>
                ) : undefined}>
                {picking && (
                  <NewFactory busy={busy === 'create'} onCreate={(b) => void create(b)} onCancel={() => setPicking(false)} />
                )}
                {factories.length === 0 && !picking ? (
                  <EmptyBand text="Set up your first standing job" onAdd={() => setPicking(true)} />
                ) : factories.length > 0 && (
                  <div className={`fd-card overflow-hidden ${picking ? 'mt-4' : ''}`}>
                    <ul className="divide-y divide-[var(--fd-line)]">
                      {factories.map((f) => (
                        <FactoryRow key={f.id} factory={f} busy={busy === f.id} autoOn={status.prefs.on}
                          onOpen={() => openAt(f.id)}
                          onRun={() => void act(f.id, () => runNow(f.id), `${f.title}: run started`)}
                          onState={(s) => void act(f.id, () => setState(f.id, s), `${f.title}: ${STATE_LABELS[s].toLowerCase()}`)}
                          onDelete={() => {
                            if (window.confirm(`Delete “${f.title}” for good, with its history?`)) {
                              void act(f.id, () => deleteFactory(f.id), `${f.title} deleted`)
                            }
                          }} />
                      ))}
                    </ul>
                  </div>
                )}
              </Section>

              {factories.some((f) => f.version > 0) && (
                <Section icon={CalendarDays} title="The week"
                  hint="Each Monday morning First Mate sends last week's report to your Line.">
                  <Week onOpen={(id) => openAt(id)} />
                </Section>
              )}

              {status.is_admin && (
                <Section icon={ShieldCheck} title="This deck" hint="Only admins see this.">
                  <div className="fd-card px-4 py-1.5"><DeckSettings status={status} busy={busy === 'deck'} onChange={(c) => void deckSwitch(c)} /></div>
                </Section>
              )}
            </>
          )}
        </div>
      </div>

      {notice && <Toast text={notice} onDismiss={() => setNotice(null)} />}
      {opened && (
        <FactoryDrawer factoryId={opened} initialTab={openedTab} onClose={() => setOpened(null)}
          onChanged={() => { void listFactories().then(setFactories).catch(() => {}) }}
          onNotice={setNotice} />
      )}
    </div>
  )
}

const HEALTH_TONE_WEEK: Record<WeekReport['factories'][number]['health'], PillTone> = {
  green: 'green', amber: 'amber', red: 'red', new: 'zinc',
}

/** Last week (what Monday's report said) or this week so far: totals, a row per factory, what's worth a look. */
function Week({ onOpen }: { onOpen: (id: string) => void }) {
  const [which, setWhich] = useState<'last' | 'this'>('last')
  const [data, setData] = useState<WeekReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    setData(null)
    getReport(which).then((r) => { if (live) { setData(r); setError(null) } })
      .catch((e) => { if (live) setError(errText(e)) })
    return () => { live = false }
  }, [which])
  const pct = (x: number | null) => (x == null ? '—' : `${Math.round(x * 100)}%`)
  const tab = (w: 'last' | 'this') => `rounded-full px-3 py-1 text-xs font-medium transition-colors ${
    which === w ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:bg-[var(--fd-hover)] hover:text-zinc-200'}`
  const t = data?.totals
  return (
    <div className="fd-card space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <div role="tablist" className="flex gap-1">
          <button role="tab" aria-selected={which === 'last'} className={tab('last')} onClick={() => setWhich('last')}>Last week</button>
          <button role="tab" aria-selected={which === 'this'} className={tab('this')} onClick={() => setWhich('this')}>This week so far</button>
        </div>
        <span className="flex-1" />
        {data && <span className="text-xs text-zinc-500">{data.title.replace("First Mate's week: ", '')}</span>}
      </div>
      {error && <ErrorBox text={error} onDismiss={() => setError(null)} />}
      {!data && !error && <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />}
      {data && t && (
        <>
          <p className="text-[13px] text-zinc-300">
            {t.runs} {t.runs === 1 ? 'run' : 'runs'}, {t.done} done
            {t.carry_outs > 0 && <> · {t.carry_outs} carried out{t.doubts > 0 && <span className="text-red-600 dark:text-red-300">, {t.doubts} doubtful</span>}</>}
            {' '}· {tokensText(t.tokens).replace(' tokens', '')} of {tokensText(t.budget)}
            {(t.drafts.sent + t.drafts.edited + t.drafts.deleted) > 0 && (
              <span className="block text-xs text-zinc-500">
                Drafts: {t.drafts.sent} sent as written, {t.drafts.edited} after edits, {t.drafts.deleted} deleted
              </span>
            )}
          </p>
          <ul className="divide-y divide-[var(--fd-line)] rounded-xl border border-[var(--fd-line)]">
            {data.factories.map((x) => (
              <li key={x.id}>
                <button onClick={() => onOpen(x.id)} className="w-full px-3 py-2.5 text-left transition-colors hover:bg-[var(--fd-hover)]">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-zinc-100">{x.title}</span>
                    <Pill tone={HEALTH_TONE_WEEK[x.health]}>{HEALTH_LABELS[x.health]}</Pill>
                    <Pill tone="zinc">{TRUST_LABELS[x.trust]}</Pill>
                  </div>
                  <div className="mt-0.5 text-xs text-zinc-500">
                    {x.runs} {x.runs === 1 ? 'run' : 'runs'}, {x.done} done · {tokensText(x.tokens).replace(' tokens', '')} of its {tokensText(x.share).replace(' tokens', '')} share
                    {x.satisfaction != null && <> · {pct(x.satisfaction)} satisfied</>}
                    {x.cases_score != null && <> · {pct(x.cases_score)} on its cases</>}
                  </div>
                  {x.moved.length > 0 && (
                    <ul className="mt-1 space-y-0.5 text-xs text-zinc-400">
                      {x.moved.map((m, n) => <li key={n}>• {m}</li>)}
                    </ul>
                  )}
                </button>
              </li>
            ))}
          </ul>
          {data.look.length > 0 && (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/[0.06] px-3 py-2">
              <p className="text-[11px] font-semibold text-amber-800 dark:text-amber-200">Worth a look</p>
              <ul className="mt-1 space-y-0.5 text-xs text-zinc-300">
                {data.look.map((m, n) => <li key={n}>• {m}</li>)}
              </ul>
            </div>
          )}
          {which === 'last' && (
            <p className="text-[11px] text-zinc-500">
              {data.sent.week !== data.end.slice(0, 10) ? 'It goes to your Line on Monday morning.'
                : data.sent.sent ? `Sent to your Line ${whenText(data.sent.at)}.`
                  : 'Nothing happened that week, so no report went out.'}
            </p>
          )}
        </>
      )}
    </div>
  )
}

const BUDGETS = [5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000, 250_000_000]

function YourFirstMate({ status, busy, onChange }: {
  status: Status
  busy: boolean
  onChange: (change: { on?: boolean; weekly_tokens?: number }) => void
}) {
  const { on } = status.prefs
  const { used, budget } = status.week
  const pct = budget > 0 ? Math.min(100, Math.round((used / budget) * 100)) : 0
  const options = BUDGETS.includes(budget) ? BUDGETS : [...BUDGETS, budget].sort((a, b) => a - b)
  const row = 'flex items-center gap-4 py-2.5'
  return (
    <div className="fd-card divide-y divide-[var(--fd-line)] px-4 py-1.5">
      <div className={row}>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-zinc-200">Run my factories on their own</div>
          <div className="text-xs text-zinc-500">
            {on ? 'On their schedules and when new mail comes in. Turning it off pauses them and stops their runs.'
              : 'Turn it on and running factories start on their schedules and on new mail.'}
          </div>
        </div>
        <Switch on={on} disabled={busy} label="Run my factories on their own" onChange={(v) => onChange({ on: v })} />
      </div>
      <div className="py-2.5">
        <div className="flex items-center gap-3">
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-zinc-200">This week</div>
            <div className="text-xs text-zinc-500">
              {tokensText(used)} of {tokensText(budget)}, shared between your factories by their share. The week starts Monday.
            </div>
          </div>
          <label className="sr-only" htmlFor="fm-budget">Weekly budget</label>
          <select id="fm-budget" value={budget} disabled={busy}
            onChange={(e) => onChange({ weekly_tokens: Number(e.target.value) })}
            className="h-8 rounded-full border border-[var(--fd-line)] bg-zinc-950 px-3 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none">
            {options.map((n) => <option key={n} value={n}>{tokensText(n).replace(' tokens', '')} a week</option>)}
          </select>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-800" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label="This week's tokens">
          <div className={`h-full rounded-full ${pct >= 100 ? 'bg-amber-500' : 'bg-violet-500'}`} style={{ width: `${pct}%` }} />
        </div>
      </div>
      {status.calibration && (
        <div className={row}>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-zinc-200">Its checks against your labels</div>
            <div className="text-xs text-zinc-500">
              {status.calibration.ready
                ? `They agree with you (kappa ${status.calibration.kappa?.toFixed(2)} on ${status.calibration.n} results), so a factory's record can raise its trust on its own.`
                : `Trust moves only by hand until they agree with you: ${status.calibration.why}. Your 👍/👎, approvals and what you do with its drafts are the labels.`}
            </div>
          </div>
          <Pill tone={status.calibration.ready ? 'green' : 'zinc'}>
            {status.calibration.ready ? 'Calibrated' : `${Math.min(status.calibration.n, status.calibration.labels_bar)}/${status.calibration.labels_bar}`}
          </Pill>
        </div>
      )}
    </div>
  )
}

function DeckSettings({ status, busy, onChange, className = '' }: {
  status: Status
  busy: boolean
  onChange: (change: { enabled?: boolean; admins_only?: boolean }) => void
  className?: string
}) {
  const s = status.settings
  const row = 'flex items-center gap-4 py-2.5'
  return (
    <div className={`divide-y divide-[var(--fd-line)] ${className}`}>
      <div className={row}>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-zinc-200">First Mate on this deck</div>
          <div className="text-xs text-zinc-500">Off: nobody sees it but admins, and nothing runs.</div>
        </div>
        <Switch on={s.enabled} disabled={busy} label="First Mate on this deck" onChange={(on) => onChange({ enabled: on })} />
      </div>
      <div className={row}>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-zinc-200">Admins only</div>
          <div className="text-xs text-zinc-500">While First Mate is being tried out, only admins use it.</div>
        </div>
        <Switch on={s.admins_only} disabled={busy} label="Admins only" onChange={(on) => onChange({ admins_only: on })} />
      </div>
    </div>
  )
}

function FactoryRow({ factory: f, busy, autoOn, onOpen, onRun, onState, onDelete }: {
  factory: FactoryCard
  busy: boolean
  /** The owner's switch: off, nothing starts on its own, so no "next run". */
  autoOn: boolean
  onOpen: () => void
  onRun: () => void
  onState: (s: OwnerState) => void
  onDelete: () => void
}) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const run = f.last_run
  const going = !!run && (run.status === 'queued' || run.status === 'running')
  const canRun = !!f.live && ['running', 'paused', 'stopped'].includes(f.state) && !going

  const menu: MenuEntry[] = [
    { label: 'Open', icon: PanelRightOpen, onSelect: onOpen, title: 'Its charter, runs and history' },
    ...(canRun ? [{ label: 'Run now', icon: Play, onSelect: onRun }] : []),
    'divider',
    ...(f.state === 'running' ? [
      { label: 'Pause', icon: Pause, onSelect: () => onState('paused') },
      { label: 'Stop', icon: Square, onSelect: () => onState('stopped') },
    ] : ['paused', 'stopped', 'paused_budget'].includes(f.state) ? [
      { label: 'Resume', icon: Play, onSelect: () => onState('running') },
    ] : []),
    ...(f.state !== 'draft' && f.state !== 'awaiting_charter' && f.state !== 'retired'
      ? [{ label: 'Retire', icon: Archive, onSelect: () => onState('retired') }] : []),
    ...(['draft', 'awaiting_charter', 'retired'].includes(f.state)
      ? ['divider' as const, { label: 'Delete', icon: Trash2, danger: true, onSelect: onDelete }] : []),
  ]

  const meta: React.ReactNode[] = []
  if (f.live) meta.push(<span key="t">{triggersText(f.live.triggers)}</span>)
  if (f.live && autoOn && f.state === 'running' && f.next_at && !going) {
    const due = new Date(f.next_at).getTime() <= Date.now()
    meta.push(<span key="n" className="text-zinc-400">{due ? 'Due now' : `Next ${whenText(f.next_at)}`}</span>)
  }
  if (f.proposal) {
    meta.push(<span key="p" className="text-amber-700 dark:text-amber-300">
      {f.proposal.count} {f.proposal.count === 1 ? 'thing waits' : 'things wait'} for your yes
    </span>)
  }
  if (f.pending_events > 0) {
    meta.push(<span key="e" className="text-zinc-400">{f.pending_events} {f.pending_events === 1 ? 'event waits' : 'events wait'} for the next run</span>)
  }
  if (run) {
    meta.push(
      <span key="r" className={run.waiting ? 'text-amber-700 dark:text-amber-300' : 'text-zinc-400'}>
        {run.waiting ? 'Waiting for you' : going ? 'Running now' : `Last run ${RUN_LABELS[run.status].toLowerCase()}`}
        {!going && run.ended_at ? ` · ${whenText(run.ended_at)}` : ''}
      </span>,
    )
  } else if (f.live) {
    meta.push(<span key="none">No runs yet</span>)
  }
  if (f.satisfaction != null) {
    meta.push(<span key="s" title="The average of what you did with its last results">{Math.round(f.satisfaction * 100)}% satisfied</span>)
  }
  if (f.live && f.share_tokens > 0) {
    meta.push(
      <span key="w" title="Its tokens this week, of its share of your weekly budget"
        className={f.week_tokens >= f.share_tokens ? 'text-amber-700 dark:text-amber-300' : ''}>
        {tokensText(f.week_tokens).replace(' tokens', '')} of {tokensText(f.share_tokens)} this week
      </span>,
    )
  }

  return (
    <li className="group relative flex gap-3 px-4 py-3 transition-colors hover:bg-[var(--fd-hover)]">
      <span className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
        going ? 'bg-violet-500/15 text-violet-600 dark:text-violet-300'
          : f.state === 'running' ? 'bg-emerald-500/12 text-emerald-600 dark:text-emerald-400' : 'bg-zinc-500/10 text-zinc-500'}`}>
        {going ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : f.state === 'running' ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
      </span>
      <div className="min-w-0 flex-1">
        <button onClick={onOpen} title="Open: its charter, runs and history"
          className="block w-full rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40">
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="text-[15px] font-medium leading-snug text-zinc-100">{f.title}</span>
            {f.state !== 'running' && <Pill tone={STATE_TONE[f.state]}>{STATE_LABELS[f.state]}</Pill>}
            {f.draft && f.version > 0 && <Pill tone="amber">Draft waiting</Pill>}
            {f.live && (f.health === 'amber' || f.health === 'red') && (
              <Pill tone={f.health === 'red' ? 'red' : 'amber'} title={f.health_why}>{HEALTH_LABELS[f.health]}</Pill>
            )}
            {f.live && <Pill tone="zinc" title={`Trust: ${TRUST_LABELS[f.trust]}`}>{RUNNER_LABELS[f.live.runner]}</Pill>}
          </span>
          {(f.live ?? f.draft)?.charter && (
            <span className="mt-0.5 line-clamp-2 text-[13px] leading-relaxed text-zinc-400">{(f.live ?? f.draft)!.charter}</span>
          )}
          {meta.length > 0 && <span className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-zinc-500">{meta}</span>}
        </button>
      </div>
      <div className="flex shrink-0 items-start gap-1">
        {canRun && (
          <button onClick={onRun} disabled={busy} title="Run now"
            className="hidden h-8 items-center gap-1 rounded-full px-3 text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100 wide:inline-flex">
            <Play className="h-3.5 w-3.5" /> Run now
          </button>
        )}
        <button
          onClick={(e) => setAnchor(anchor ? null : e.currentTarget)}
          disabled={busy}
          aria-label={`${f.title}: actions`}
          aria-haspopup="menu"
          aria-expanded={!!anchor}
          className={`flex h-8 w-8 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 focus:opacity-100 group-hover:opacity-100 pointer-coarse:opacity-100 ${
            anchor || busy ? 'bg-zinc-800 text-zinc-200 opacity-100' : 'opacity-0'}`}
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <MoreHorizontal className="h-4 w-4" />}
        </button>
        {anchor && <QueueMenu anchor={anchor} label={`${f.title}: actions`} items={menu} onClose={() => setAnchor(null)} />}
      </div>
    </li>
  )
}

function NewFactory({ busy, onCreate, onCancel }: {
  busy: boolean
  onCreate: (body: { template?: string; title?: string; charter?: string }) => void
  onCancel: () => void
}) {
  const [templates, setTemplates] = useState<Template[] | null>(null)
  const [scratch, setScratch] = useState(false)
  const [title, setTitle] = useState('')
  const [charter, setCharter] = useState('')
  useEffect(() => { void getTemplates().then(setTemplates).catch(() => setTemplates([])) }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return
      e.preventDefault()
      onCancel()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onCancel])

  const input = 'w-full rounded-xl border border-[var(--fd-line)] bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none'
  return (
    <div className="fd-card space-y-4 p-4">
      <div className="flex items-center gap-2">
        <h4 className="flex-1 text-sm font-semibold text-zinc-100">Start from</h4>
        <button onClick={onCancel} aria-label="Cancel"
          className="flex h-8 w-8 items-center justify-center rounded-full text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
          <X className="h-4 w-4" />
        </button>
      </div>
      {templates === null ? (
        <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />
      ) : (
        <div className="grid gap-2 @lg:grid-cols-2">
          {templates.map((t) => (
            <button key={t.id} onClick={() => onCreate({ template: t.id })} disabled={busy}
              className="rounded-xl border border-[var(--fd-line)] px-3.5 py-3 text-left transition-colors hover:border-violet-500/40 hover:bg-[var(--fd-hover)] disabled:opacity-60">
              <span className="block text-sm font-medium text-zinc-100">{t.title}</span>
              <span className="mt-0.5 block text-xs leading-relaxed text-zinc-500">{t.blurb}</span>
              <span className="mt-1.5 block text-[11px] text-zinc-500">{triggersText(t.triggers)}</span>
            </button>
          ))}
          <button onClick={() => setScratch(true)} disabled={busy}
            className="rounded-xl border border-dashed border-zinc-800 px-3.5 py-3 text-left transition-colors hover:border-violet-500/40 hover:bg-[var(--fd-hover)]">
            <span className="block text-sm font-medium text-zinc-100">From scratch</span>
            <span className="mt-0.5 block text-xs text-zinc-500">Write the charter yourself.</span>
          </button>
        </div>
      )}
      {scratch && (
        <form className="space-y-3 border-t border-[var(--fd-line)] pt-4"
          onSubmit={(e) => { e.preventDefault(); onCreate({ title, charter }) }}>
          <input className={input} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Name, like “Invoice chaser”" autoFocus />
          <textarea className={`${input} resize-y leading-relaxed`} rows={5} value={charter} onChange={(e) => setCharter(e.target.value)}
            placeholder="What it does each run, and what it must never do" />
          <div className="flex justify-end">
            <button type="submit" disabled={busy || !title.trim() || !charter.trim()}
              className="inline-flex h-8 items-center gap-1.5 rounded-full bg-violet-600 px-3.5 text-xs font-semibold text-white transition-colors hover:bg-violet-500 disabled:opacity-60">
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null} Create draft
            </button>
          </div>
        </form>
      )}
      <p className="text-xs text-zinc-500">A new factory starts as a draft. Nothing runs until you approve its charter.</p>
    </div>
  )
}
