import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, Anchor, Archive, BellDot, CalendarDays, Coins, Factory as FactoryIcon, Loader2, MoreHorizontal, PanelRightOpen, Pause, Play, Plus, Power, RefreshCw, ShieldCheck, Ship, SlidersHorizontal, Square, Trash2, Trophy, X,
} from 'lucide-react'
import {
  createFactory,
  deleteFactory,
  draftFactory,
  getMyAgent,
  getReport,
  makeMyAgent,
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
  type MyAgent,
  type Template,
  type WeekReport,
} from '../services/firstMate'
import { getCourse } from '../services/course'
import { useFirstMateStore } from '../stores/firstMateStore'
import { QueueMenu, type MenuEntry } from '../components/agents/queue/QueueMenu'
import { FactoryDrawer, type Tab as DrawerTab } from '../components/firstMate/FactoryDrawer'
import { EmptyBand, ErrorBox, Pill, Section, Switch, Toast, errText, type PillTone } from '../components/firstMate/ui'
import { FullscreenButton } from '../components/library/libraryParts'

// First Mate: standing jobs ("factories") that run for the user. Each has a
// charter they approve, runs through Bat, and is checked every run. See
// docs/first-mate-plan.md.
//
// On top, four tiles: the factories, the runs going, this week's tokens, and
// the owner's switch. Then "Needs you" (charters waiting for approval,
// factories over budget, runs waiting on the owner), the factories as cards
// with Run now and a ⋯ menu (a card opens its drawer: charter, runs, history),
// and the week. Beside them on a wide screen: the owner's settings and, for
// admins, the deck's switches. A deck with First Mate off shows why, and its
// admins get the switch.
//
// `onClose`: shown in the kiosk's dialog, a close button in the header and
// Escape closes it (not while the drawer or the New factory picker is open).

const HEAD_BTN = 'flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-60'

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
  const [openedTab, setOpenedTab] = useState<DrawerTab>('charter')
  const openAt = (id: string, tab: DrawerTab = 'charter') => { setOpenedTab(tab); setOpened(id) }
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

  const [draftError, setDraftError] = useState<string | null>(null)
  const describe = async (words: string, goalId: string) => {
    setBusy('draft'); setDraftError(null)
    try {
      const f = await draftFactory(goalId ? { words, goal_id: goalId } : { words })
      setPicking(false)
      setNotice(`Drafted “${f.title}”. Look it over and approve it.`)
      await load()
      openAt(f.id)
    } catch (e) {
      setDraftError(errText(e))
    } finally {
      setBusy(null)
    }
  }

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
      <header className={`shrink-0 border-b transition-colors ${scrolled ? 'border-[var(--fd-line)] bg-zinc-950/90 backdrop-blur' : 'border-transparent'}`}>
        <div className="mx-auto flex w-full max-w-6xl items-center gap-3 px-5 py-4 wide:px-8 wide:py-5">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-sm">
            <Ship className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="text-xl font-bold leading-tight tracking-tight text-zinc-100">First Mate</h2>
            <p className="truncate text-[13px] text-zinc-500">Standing jobs that run for you, each on a charter you approve, checked every time.</p>
          </div>
          <button onClick={() => void load()} disabled={loading} title="Check again" aria-label="Check again" className={HEAD_BTN}>
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <FullscreenButton className={HEAD_BTN} />
          {onClose && (
            <button onClick={onClose} title="Close" aria-label="Close" className={HEAD_BTN}>
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto" onScroll={(e) => setScrolled(e.currentTarget.scrollTop > 4)}>
        <div className="mx-auto w-full max-w-6xl space-y-8 px-5 pb-24 pt-1 wide:px-8">
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
              <Overview status={status} factories={factories} busy={busy === 'prefs'}
                onSwitch={(on) => void act('prefs', () => savePrefs({ on }), on ? 'First Mate is on' : 'First Mate is off: running factories paused')} />
              <div className="grid items-start gap-8 @4xl:grid-cols-[minmax(0,1fr)_19rem]">
              <div className="min-w-0 space-y-9">
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

              <Section icon={FactoryIcon} title="Factories"
                hint={`${factories.length} of ${max}. Each run is a Bat run: its checks, its reviewers, its questions to you.`}
                action={!picking && !atCap ? (
                  <button onClick={() => setPicking(true)} className="fd-btn-quiet h-8 shrink-0 px-3 text-xs">
                    <Plus className="h-3.5 w-3.5" /> New factory
                  </button>
                ) : undefined}>
                {picking && (
                  <NewFactory busy={busy === 'create'} drafting={busy === 'draft'} draftError={draftError}
                    onCreate={(b) => void create(b)} onDescribe={(w, g) => void describe(w, g)}
                    onCancel={() => { setPicking(false); setDraftError(null) }} />
                )}
                {factories.length === 0 && !picking ? (
                  <EmptyBand text="Set up your first standing job" onAdd={() => setPicking(true)} />
                ) : factories.length > 0 && (
                  <div className={picking ? 'mt-4' : ''}>
                    <ul className="space-y-3">
                      {factories.map((f) => (
                        <FactoryRow key={f.id} factory={f} busy={busy === f.id} autoOn={status.prefs.on}
                          onOpen={() => openAt(f.id)}
                          onResults={() => openAt(f.id, 'results')}
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

              </div>
              <aside className="space-y-8 @4xl:sticky @4xl:top-2">
                <Section icon={SlidersHorizontal} title="Your settings">
                  <MateSettings status={status} busy={busy === 'prefs'}
                    onChange={(change) => void act('prefs', () => savePrefs(change), 'Weekly budget saved')}
                    onNotice={setNotice} />
                </Section>
                {status.is_admin && (
                  <Section icon={ShieldCheck} title="This deck" hint="Only admins see this.">
                    <div className="fd-card px-4 py-1"><DeckSettings status={status} busy={busy === 'deck'} onChange={(c) => void deckSwitch(c)} /></div>
                  </Section>
                )}
              </aside>
              </div>
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

/** Which agent is First Mate: picked from the owner's agents, or made for it. Its model drafts charters. */
function MateAgent({ row, onNotice }: { row: string; onNotice: (text: string) => void }) {
  const [data, setData] = useState<MyAgent | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { void getMyAgent().then(setData).catch((e) => setError(errText(e))) }, [])
  const pick = async (slug: string) => {
    setBusy(true); setError(null)
    try {
      await savePrefs({ agent: slug })
      setData(await getMyAgent())
      onNotice(slug ? 'First Mate now thinks on that agent’s model' : 'First Mate drafts on your tier set')
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(false)
    }
  }
  const make = async () => {
    setBusy(true); setError(null)
    try {
      const made = await makeMyAgent()
      setData({ current: made.current, agents: made.agents ?? (await getMyAgent()).agents })
      onNotice(`Made “${made.name}”. Talk to it like any agent; it uses the first_mate tool.`)
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(false)
    }
  }
  const cur = data?.current
  return (
    <div className={row}>
      <div className="min-w-0">
        <div className="text-sm font-medium text-zinc-200">Its agent</div>
        <div className="text-xs text-zinc-500">
          {!data ? 'Loading…'
            : !cur?.slug ? 'None yet: drafting uses your tier set. Pick an agent, or make one, to talk to First Mate and draft on its model.'
              : !cur.exists ? `${cur.name} is gone. Drafting uses your tier set until you pick another.`
                : `${cur.name} ${cur.running ? 'is running' : 'is asleep'}: talk to it in its chat, and it drafts on its own model.`}
        </div>
        {error && <div role="alert" className="mt-1 text-xs text-red-700 dark:text-red-300">{error}</div>}
      </div>
      {data && (
        <div className="flex w-full items-center gap-2">
          <label className="sr-only" htmlFor="fm-agent">First Mate's agent</label>
          <select id="fm-agent" value={cur?.exists ? cur.slug : ''} disabled={busy} onChange={(e) => void pick(e.target.value)}
            className="h-8 min-w-0 flex-1 rounded-full border border-[var(--fd-line)] bg-zinc-950 px-3 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none">
            <option value="">None</option>
            {data.agents.map((a) => <option key={a.slug} value={a.slug}>{a.name}</option>)}
          </select>
          {!cur?.exists && (
            <button onClick={() => void make()} disabled={busy} className="fd-btn-quiet h-8 shrink-0 px-3 text-xs" title="Make an agent called First Mate">
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />} Make one
            </button>
          )}
        </div>
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
            {' '}· {t.budget > 0 ? <>{tokensText(t.tokens).replace(' tokens', '')} of {tokensText(t.budget)}</> : <>{tokensText(t.tokens)} (no weekly limit)</>}
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

/** The four tiles on top: factories, runs going, this week's tokens, and the
 *  owner's switch (off: factories run only when Run now is pressed). */
function Overview({ status, factories, busy, onSwitch }: {
  status: Status
  factories: FactoryCard[]
  busy: boolean
  onSwitch: (on: boolean) => void
}) {
  const live = factories.filter((f) => f.version > 0 && f.state !== 'retired')
  const running = live.filter((f) => f.state === 'running').length
  const going = factories.filter((f) => f.last_run && (f.last_run.status === 'queued' || f.last_run.status === 'running')).length
  const { used, budget } = status.week
  const pct = budget > 0 ? Math.min(100, Math.round((used / budget) * 100)) : 0
  const { on } = status.prefs
  const tile = 'fd-card relative flex min-w-0 flex-col gap-1 overflow-hidden px-4 py-3.5'
  const label = 'flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500'
  const value = 'text-2xl font-semibold tabular-nums leading-tight tracking-tight text-zinc-100'
  return (
    <div className="grid grid-cols-2 gap-3 @3xl:grid-cols-4">
      <div className={tile}>
        <span className={label}><FactoryIcon className="h-3.5 w-3.5" />Factories</span>
        <span className={value}>{live.length}</span>
        <span className="truncate text-xs text-zinc-500">{running} running{factories.length > live.length ? ` · ${factories.length - live.length} not live` : ''}</span>
      </div>
      <div className={tile}>
        <span className={label}><Activity className="h-3.5 w-3.5" />Runs going</span>
        <span className={`${value} flex items-center gap-2`}>
          {going}
          {going > 0 && <span className="relative flex h-2.5 w-2.5"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-400 opacity-60" /><span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-violet-500" /></span>}
        </span>
        <span className="truncate text-xs text-zinc-500">{going ? 'working now' : 'none right now'}</span>
      </div>
      <div className={tile}>
        <span className={label}><Coins className="h-3.5 w-3.5" />This week</span>
        <span className={value}>{tokensText(used).replace(' tokens', '')}</span>
        <span className="truncate text-xs text-zinc-500">{budget > 0 ? `of ${tokensText(budget)}` : 'tokens · no weekly limit'}</span>
        {budget > 0 && (
          <span className="absolute inset-x-0 bottom-0 h-1 bg-zinc-800" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label="This week's tokens">
            <span className={`block h-full ${pct >= 100 ? 'bg-amber-500' : 'bg-violet-500'}`} style={{ width: `${pct}%` }} />
          </span>
        )}
      </div>
      <div className={`${tile} ${on ? 'ring-1 ring-inset ring-emerald-500/30' : ''}`}>
        <span className={label}><Power className="h-3.5 w-3.5" />On their own</span>
        <span className="flex items-center justify-between gap-2">
          <span className={`text-2xl font-semibold leading-tight tracking-tight ${on ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-400'}`}>{on ? 'On' : 'Off'}</span>
          <Switch on={on} disabled={busy} label="Run my factories on their own" onChange={onSwitch} />
        </span>
        <span className="text-xs leading-snug text-zinc-500">{on ? 'Schedules and new mail start runs.' : 'Only when you press Run now.'}</span>
      </div>
    </div>
  )
}

/** The owner's settings: the weekly budget, First Mate's agent, and how far its checks agree with them. */
function MateSettings({ status, busy, onChange, onNotice }: {
  status: Status
  busy: boolean
  onChange: (change: { weekly_tokens?: number }) => void
  onNotice: (text: string) => void
}) {
  const { budget } = status.week
  const unlimited = budget < 0
  const options = unlimited || BUDGETS.includes(budget) ? BUDGETS : [...BUDGETS, budget].sort((a, b) => a - b)
  // A narrow column: each setting's words above its control.
  const stack = 'flex flex-col items-start gap-2 py-3'
  const row = 'flex items-start gap-3 py-3'
  return (
    <div className="fd-card divide-y divide-[var(--fd-line)] px-4 py-1">
      <div className={stack}>
        <div className="min-w-0">
          <div className="text-sm font-medium text-zinc-200">Weekly budget</div>
          <div className="text-xs text-zinc-500">
            {unlimited ? 'No limit: fine on a local model; on a paid API it can cost a lot.'
              : 'Shared between your factories by their share. The week starts Monday.'}
          </div>
        </div>
        <label className="sr-only" htmlFor="fm-budget">Weekly budget</label>
        <select id="fm-budget" value={budget} disabled={busy}
          onChange={(e) => onChange({ weekly_tokens: Number(e.target.value) })}
          className="h-8 w-full rounded-full border border-[var(--fd-line)] bg-zinc-950 px-3 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none">
          {options.map((n) => <option key={n} value={n}>{tokensText(n).replace(' tokens', '')} a week</option>)}
          <option value={-1}>No weekly limit</option>
        </select>
      </div>
      <MateAgent row={stack} onNotice={onNotice} />
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

function FactoryRow({ factory: f, busy, autoOn, onOpen, onResults, onRun, onState, onDelete }: {
  factory: FactoryCard
  busy: boolean
  /** The owner's switch: off, nothing starts on its own, so no "next run". */
  autoOn: boolean
  onOpen: () => void
  /** Its Results tab. */
  onResults: () => void
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
  if (f.live && f.share_tokens < 0 && f.week_tokens > 0) {
    meta.push(<span key="w" title="Its tokens this week; your week has no limit">{tokensText(f.week_tokens)} this week</span>)
  } else if (f.live && f.share_tokens > 0) {
    meta.push(
      <span key="w" title="Its tokens this week, of its share of your weekly budget"
        className={f.week_tokens >= f.share_tokens ? 'text-amber-700 dark:text-amber-300' : ''}>
        {tokensText(f.week_tokens).replace(' tokens', '')} of {tokensText(f.share_tokens)} this week
      </span>,
    )
  }

  const initial = (f.title.trim()[0] || '?').toUpperCase()
  return (
    <li className={`group relative flex gap-3.5 overflow-hidden rounded-2xl border bg-[var(--fd-surface-raised)] p-4 transition-all hover:-translate-y-px hover:shadow-md ${
      going ? 'border-violet-500/30' : 'border-[var(--fd-line)]'}`}>
      <span className={`relative mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-sm font-semibold ${
        going ? 'bg-violet-500/15 text-violet-700 dark:text-violet-300'
          : f.state === 'running' ? 'bg-emerald-500/12 text-emerald-700 dark:text-emerald-300' : 'bg-zinc-500/10 text-zinc-500'}`}
        title={going ? 'A run is going' : STATE_LABELS[f.state]}>
        {initial}
        <span className={`absolute -bottom-0.5 -right-0.5 flex h-4 w-4 items-center justify-center rounded-full ring-2 ring-[var(--fd-surface-raised)] ${
          going ? 'bg-violet-500 text-white' : f.state === 'running' ? 'bg-emerald-500 text-white' : 'bg-zinc-400 text-white dark:bg-zinc-600'}`}>
          {going ? <Loader2 className="h-2.5 w-2.5 animate-spin" /> : f.state === 'running' ? <Play className="h-2 w-2 fill-current" /> : <Pause className="h-2 w-2" />}
        </span>
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
        {f.latest_result && (
          <button onClick={onResults} title="Its results: open the table, download it, add it to the Library"
            className={`mt-2.5 inline-flex max-w-full items-center gap-2 rounded-xl border py-1.5 pl-2 pr-3 text-left text-xs transition-colors ${
              f.latest_result.status === 'done' ? 'border-emerald-500/25 bg-emerald-500/[0.06] hover:border-emerald-500/50'
                : 'border-[var(--fd-line)] bg-zinc-900/40 hover:border-violet-500/40'}`}>
            <Trophy className={`h-3.5 w-3.5 shrink-0 ${f.latest_result.status === 'done' ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-500'}`} />
            <span className="shrink-0 text-zinc-500">{f.latest_result.status === 'done' ? 'Result' : going ? 'So far' : 'Last result'}</span>
            <span className="min-w-0 truncate font-medium text-zinc-200">{f.latest_result.name}</span>
            <span className="shrink-0 text-zinc-500">
              {[f.latest_result.rows !== null ? `${f.latest_result.rows.toLocaleString()} row${f.latest_result.rows === 1 ? '' : 's'}` : '',
                f.latest_result.files > 1 ? `+${f.latest_result.files - 1} more` : ''].filter(Boolean).join(' · ')}
            </span>
          </button>
        )}
      </div>
      <div className="flex shrink-0 items-start gap-1">
        {canRun && (
          <button onClick={onRun} disabled={busy} title="Run now"
            className="hidden h-8 items-center gap-1 rounded-full border border-[var(--fd-line)] px-3 text-xs font-medium text-zinc-300 transition-colors hover:border-violet-500/40 hover:bg-violet-500/10 hover:text-violet-700 dark:hover:text-violet-200 wide:inline-flex">
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
      {going && (
        <span className="absolute inset-x-0 bottom-0 h-0.5 overflow-hidden bg-violet-500/15" aria-hidden>
          <span className="fm-running-bar block h-full w-1/3 rounded-full bg-violet-500" />
        </span>
      )}
    </li>
  )
}

function NewFactory({ busy, drafting, draftError, onCreate, onDescribe, onCancel }: {
  busy: boolean
  drafting: boolean
  draftError: string | null
  onCreate: (body: { template?: string; title?: string; charter?: string }) => void
  onDescribe: (words: string, goalId: string) => void
  onCancel: () => void
}) {
  const [templates, setTemplates] = useState<Template[] | null>(null)
  const [scratch, setScratch] = useState(false)
  const [title, setTitle] = useState('')
  const [charter, setCharter] = useState('')
  const [words, setWords] = useState('')
  const [goalId, setGoalId] = useState('')
  const [goals, setGoals] = useState<{ id: string; title: string }[]>([])
  useEffect(() => { void getTemplates().then(setTemplates).catch(() => setTemplates([])) }, [])
  useEffect(() => {
    void getCourse().then((c) => setGoals([...c.goals.mine, ...c.goals.deck]
      .filter((g) => g.status === 'active').map((g) => ({ id: g.id, title: g.title })))).catch(() => setGoals([]))
  }, [])
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
        <h4 className="flex-1 text-sm font-semibold text-zinc-100">New factory</h4>
        <button onClick={onCancel} aria-label="Cancel"
          className="flex h-8 w-8 items-center justify-center rounded-full text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
          <X className="h-4 w-4" />
        </button>
      </div>
      <form className="space-y-2.5 rounded-xl border border-violet-500/30 bg-[var(--fd-accent-soft)] p-3.5"
        onSubmit={(e) => { e.preventDefault(); if (words.trim().length >= 8 || goalId) onDescribe(words.trim(), goalId) }}>
        <label htmlFor="fm-describe" className="block text-sm font-medium text-zinc-100">Describe it</label>
        <textarea id="fm-describe" className={`${input} resize-y leading-relaxed`} rows={3} value={words}
          onChange={(e) => setWords(e.target.value)} disabled={drafting}
          placeholder="Every weekday morning, draft replies to clients waiting on an invoice question. Never send anything." />
        <div className="flex flex-wrap items-center gap-2">
          {goals.length > 0 && (
            <>
              <label htmlFor="fm-goal" className="sr-only">For a Course goal</label>
              <select id="fm-goal" value={goalId} onChange={(e) => setGoalId(e.target.value)} disabled={drafting}
                className="h-8 max-w-[16rem] rounded-full border border-[var(--fd-line)] bg-zinc-950 px-3 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none">
                <option value="">Not for a goal</option>
                {goals.map((g) => <option key={g.id} value={g.id}>For: {g.title}</option>)}
              </select>
            </>
          )}
          <span className="flex-1" />
          <button type="submit" disabled={drafting || busy || (words.trim().length < 8 && !goalId)}
            className="inline-flex h-8 items-center gap-1.5 rounded-full bg-violet-600 px-3.5 text-xs font-semibold text-white transition-colors hover:bg-violet-500 disabled:opacity-60">
            {drafting ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Drafting…</> : 'Draft it'}
          </button>
        </div>
        {draftError && <p role="alert" className="text-xs text-red-700 dark:text-red-300">{draftError}</p>}
        <p className="text-[11px] text-zinc-500">
          First Mate writes the charter, when it starts, what a good run does and a few sealed test cases. You approve it before anything runs.
        </p>
      </form>
      <h4 className="text-xs font-semibold text-zinc-400">Or start from a template</h4>
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
