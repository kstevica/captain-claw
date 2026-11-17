import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { FlaskConical, Loader2, Play, Square, ThumbsDown, ThumbsUp, X } from 'lucide-react'
import {
  approve,
  approveRun,
  caseDraft,
  decline,
  rateRun,
  rejectRun,
  setTrust,
  editDraft,
  getFactory,
  newDraft,
  runNow,
  stopRun,
  scheduleText,
  tokensText,
  triggersText,
  whenText,
  HEALTH_LABELS,
  RUN_LABELS,
  SIGNAL_LABELS,
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
import { ErrorBox, Pill, errText, type PillTone } from './ui'
import { Checks, pct, type CaseSeed } from './Checks'
import { RunResult } from './RunResult'

// One factory, in a drawer over the page: its charter (the live version, and
// the draft being written), its runs, and its history. Approving a draft is
// the owner's own act, against the version they see: the server refuses an
// approval once the draft changed under it.

type Tab = 'charter' | 'runs' | 'record' | 'checks' | 'history'

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
    case 'run_finished': return `Run ${RUN_LABELS[d.status as Run['status']]?.toLowerCase() ?? String(d.status ?? '')}`
    case 'case_added': return 'Test case added'
    case 'case_deleted': return 'Test case deleted'
    case 'check_started': return `Check of v${String(d.version ?? '')} started (${String(d.cases ?? '')} cases)`
    case 'check_finished': return d.status === 'done'
      ? `Check of v${String(d.version ?? '')}: ${pct(d.score as number | null)}`
      : `Check of v${String(d.version ?? '')} ${d.status === 'void' ? 'called off' : 'failed'}`
    default: return e.event.replace(/_/g, ' ')
  }
}

/** The editable part of a draft, as the form holds it. */
interface Form { charter: string; checks: string; schedule: string; runTokens: string }

function formOf(v: Version): Form {
  const sched = v.triggers.find((t): t is Extract<Trigger, { kind: 'schedule' }> => t.kind === 'schedule')
  return {
    charter: v.charter,
    checks: v.checks.contract.join('\n'),
    schedule: sched?.schedule ?? '',
    runTokens: v.budget.run_tokens ? String(v.budget.run_tokens) : '',
  }
}

function fieldsOf(form: Form, v: Version): DraftFields {
  const others = v.triggers.filter((t) => t.kind !== 'schedule')
  const schedule = form.schedule.trim()
  const tokens = form.runTokens.trim().replace(/[,_\s]/g, '')
  return {
    charter: form.charter,
    checks: { ...v.checks, contract: form.checks.split('\n').map((s) => s.trim()).filter(Boolean) },
    triggers: [...(schedule ? [{ kind: 'schedule' as const, schedule, tz: '' }] : []), ...others],
    budget: { run_tokens: tokens ? Number(tokens) : null },
  }
}

const same = (a: Form, b: Form) => a.charter === b.charter && a.checks === b.checks
  && a.schedule === b.schedule && a.runTokens === b.runTokens

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
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [form, setForm] = useState<Form | null>(null)
  const [note, setNote] = useState('')
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

  // Escape closes the drawer only: taken first and marked handled, so the
  // dialog under it (the kiosk's) stays open.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return
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

  const f = data
  const draft = f?.draft ?? null
  const dirty = !!(draft && form && !same(form, formOf(draft)))
  const canRun = !!f?.live && ['running', 'paused', 'stopped'].includes(f.state) && !going

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

  const tabCls = (t: Tab) => `rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
    tab === t ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:bg-[var(--fd-hover)] hover:text-zinc-200'}`
  const label = 'mb-1.5 block text-xs font-semibold text-zinc-400'
  const input = 'w-full rounded-xl border border-[var(--fd-line)] bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none'

  return createPortal(
    <div className="fd-fade fixed inset-0 z-[60] flex justify-end bg-black/30 wide:p-3" onClick={onClose}>
      <aside
        role="dialog"
        aria-label={f ? `Factory: ${f.title}` : 'Factory'}
        className="fd-drawer fd-dialog flex h-full w-full max-w-xl flex-col overflow-hidden border-l border-[var(--fd-line)] bg-zinc-950 wide:rounded-[var(--fd-radius-panel)] wide:border"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3 px-5 pb-3 pt-5">
          <div className="min-w-0 flex-1">
            {f ? (
              <>
                <div className="flex flex-wrap items-center gap-1.5">
                  <Pill tone={f.state === 'running' ? 'green' : f.state === 'paused_budget' ? 'amber' : 'zinc'}>{STATE_LABELS[f.state]}</Pill>
                  {f.live && <Pill tone="violet">{RUNNER_LABELS[f.live.runner]}</Pill>}
                  {f.live && <Pill tone="zinc">v{f.live.version}</Pill>}
                </div>
                <h2 className="mt-2 text-lg font-bold leading-snug tracking-tight text-zinc-100">{f.title}</h2>
                {f.state_reason && <p className="mt-0.5 text-xs text-zinc-500">{f.state_reason}</p>}
              </>
            ) : <Loader2 className="mt-1 h-4 w-4 animate-spin text-zinc-500" />}
          </div>
          <button onClick={onClose} title="Close" aria-label="Close"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>
        {f && (
          <div role="tablist" className="flex gap-1 px-5 pb-3">
            <button role="tab" aria-selected={tab === 'charter'} className={tabCls('charter')} onClick={() => setTab('charter')}>Charter</button>
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

        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto border-t border-[var(--fd-line)] px-5 pb-8 pt-4 text-sm">
          {error && <ErrorBox text={error} onDismiss={() => setError(null)} />}

          {f && tab === 'charter' && (
            <>
              {draft && form && (
                <section className="fd-card space-y-4 p-4">
                  <div className="flex items-center gap-2">
                    <h3 className="flex-1 text-sm font-semibold text-zinc-100">
                      {f.version ? `Draft v${draft.version}` : 'Your charter'}
                    </h3>
                    <Pill tone="amber">{f.version ? 'Not live yet' : 'Waiting for your approval'}</Pill>
                  </div>
                  <div>
                    <label className={label} htmlFor="fm-charter">What it does, each run</label>
                    <textarea id="fm-charter" rows={7} className={`${input} resize-y leading-relaxed`} value={form.charter}
                      onChange={(e) => setForm({ ...form, charter: e.target.value })} />
                  </div>
                  <div>
                    <label className={label} htmlFor="fm-checks">Done when (one per line)</label>
                    <textarea id="fm-checks" rows={3} className={`${input} resize-y`} value={form.checks}
                      placeholder="Bat writes its own checks when you leave this empty"
                      onChange={(e) => setForm({ ...form, checks: e.target.value })} />
                  </div>
                  <div className="grid gap-3 wide:grid-cols-2">
                    <div>
                      <label className={label} htmlFor="fm-schedule">When it runs</label>
                      <input id="fm-schedule" className={input} value={form.schedule} placeholder="weekdays 08:00"
                        onChange={(e) => setForm({ ...form, schedule: e.target.value })} />
                      <p className="mt-1 text-[11px] text-zinc-500">
                        {form.schedule.trim() ? scheduleText(form.schedule) : 'Empty: only when you press Run now'}.
                        {' '}It starts on its own while your First Mate is on.
                      </p>
                    </div>
                    <div>
                      <label className={label} htmlFor="fm-tokens">Token cap per run</label>
                      <input id="fm-tokens" inputMode="numeric" className={input} value={form.runTokens} placeholder="Your default"
                        onChange={(e) => setForm({ ...form, runTokens: e.target.value })} />
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
                  <div className="flex flex-wrap items-center justify-end gap-2 border-t border-[var(--fd-line)] pt-3">
                    {f.version > 0 && cases > 0 && !gate && !held && (
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
                      className="inline-flex h-8 items-center gap-1.5 rounded-full bg-violet-600 px-3.5 text-xs font-semibold text-white transition-colors hover:bg-violet-500 disabled:opacity-60">
                      {busy === 'approve' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                      {!f.version ? 'Approve and start' : held ? 'Approve anyway' : gate ? 'Being checked' : 'Approve this version'}
                    </button>
                  </div>
                </section>
              )}

              {f.live ? (
                <LiveVersion version={f.live} trust={f.trust} pinned={f.trust_pinned} approvedAt={f.live.decided_at}
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
                <button onClick={() => void act('run', async () => { await runNow(factoryId, note); setNote('') }, 'Run started')}
                  disabled={!canRun || !!busy}
                  title={canRun ? 'Start a run now' : going ? 'A run is going' : 'Approve a charter first'}
                  className="inline-flex h-9 items-center gap-1.5 rounded-full bg-violet-600 px-4 text-xs font-semibold text-white transition-colors hover:bg-violet-500 disabled:opacity-60">
                  {busy === 'run' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />} Run now
                </button>
              </div>
              {f.runs.length === 0 ? (
                <p className="text-zinc-500">No runs yet.</p>
              ) : (
                <ul className="space-y-2.5">
                  {f.runs.map((r) => (
                    <RunItem key={r.id} run={r} busy={busy === `stop:${r.id}`} deciding={busy === `decide:${r.id}`}
                      onStop={() => void act(`stop:${r.id}`, () => stopRun(r.id), 'Run stopped')}
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

          {f && tab === 'record' && <Record detail={f} />}

          {f && tab === 'checks' && (
            <Checks factoryId={factoryId} live={f.version} seed={seed} onSeedUsed={seedUsed}
              onChanged={() => { void load(); onChanged() }} onNotice={onNotice} />
          )}

          {f && tab === 'history' && (
            <ol className="space-y-2">
              {f.events.map((e) => (
                <li key={e.id} className="flex gap-3 text-[13px]">
                  <span className="w-28 shrink-0 text-xs text-zinc-500">{whenText(e.at)}</span>
                  <span className="min-w-0 flex-1 text-zinc-300">{eventText(e)}</span>
                </li>
              ))}
            </ol>
          )}
        </div>
      </aside>
    </div>,
    document.body,
  )
}

const TRUST_ORDER: Trust[] = ['propose', 'commit', 'direct']

function LiveVersion({ version: v, trust, pinned, approvedAt, onChange, busy, trustBusy, onTrust }: {
  version: Version
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
    <section className="space-y-3">
      <div className="flex items-center gap-2">
        <h3 className="flex-1 text-sm font-semibold text-zinc-100">Live: version {v.version}</h3>
        {onChange && (
          <button onClick={onChange} disabled={busy} className="fd-btn-quiet h-8 px-3 text-xs">
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null} Change it
          </button>
        )}
      </div>
      <blockquote className="whitespace-pre-wrap rounded-xl border-l-2 border-violet-500/50 bg-[var(--fd-hover)] px-3 py-2 leading-relaxed text-zinc-200">
        {v.charter}
      </blockquote>
      <div className="divide-y divide-[var(--fd-line)] text-[13px]">
        {v.checks.contract.length > 0 && (
          <div className={row}>
            <span className={key}>Done when</span>
            <ul className="min-w-0 flex-1 list-disc space-y-0.5 pl-4 text-zinc-300">
              {v.checks.contract.map((c) => <li key={c}>{c}</li>)}
            </ul>
          </div>
        )}
        <div className={row}><span className={key}>Starts</span><span className="text-zinc-300">{triggersText(v.triggers)}</span></div>
        <div className={row}>
          <span className={key}>Cap per run</span>
          <span className="text-zinc-300">{v.budget.run_tokens ? tokensText(v.budget.run_tokens) : 'Your default'}</span>
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
        {approvedAt && <div className={row}><span className={key}>Approved</span><span className="text-zinc-300">{whenText(approvedAt)}</span></div>}
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
  const cell = 'rounded-xl bg-[var(--fd-hover)] px-3 py-2.5'
  const big = 'text-lg font-semibold text-zinc-100'
  const small = 'text-[11px] text-zinc-500'
  return (
    <div className="space-y-5">
      <section className="flex items-start gap-3">
        <Pill tone={HEALTH_TONE[s.health]}>{HEALTH_LABELS[s.health]}</Pill>
        {s.health_why && <p className="text-[13px] text-zinc-400">{s.health_why}</p>}
      </section>
      <section className="grid grid-cols-2 gap-2 wide:grid-cols-4">
        <div className={cell}><div className={big}>{pct(s.satisfaction)}</div>
          <div className={small}>Satisfaction · {s.signals} {s.signals === 1 ? 'signal' : 'signals'} · target {pct(s.target)}</div></div>
        <div className={cell}><div className={big}>{s.week.done}/{s.week.runs}</div><div className={small}>Runs done this week</div></div>
        <div className={cell}><div className={big}>{s.fortnight.doubts}/{s.fortnight.carry_outs}</div>
          <div className={small}>Doubtful carry-outs, two weeks</div></div>
        <div className={cell}><div className={big}>{s.week.tokens_per_run ? tokensText(s.week.tokens_per_run).replace(' tokens', '') : '—'}</div>
          <div className={small}>Tokens per run{s.week.median_minutes == null ? ''
            : s.week.median_minutes < 1 ? ' · under a minute' : ` · ${Math.round(s.week.median_minutes)} min`}</div></div>
      </section>
      {l && (
        <section>
          <h3 className="mb-1.5 text-xs font-semibold text-zinc-400">Trust</h3>
          <div className="rounded-xl bg-[var(--fd-hover)] px-3 py-2.5 text-[13px]">
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
        </section>
      )}
      {d.drafts.length > 0 && (
        <section>
          <h3 className="mb-1.5 text-xs font-semibold text-zinc-400">Its drafts in your mailbox</h3>
          <p className="mb-2 text-xs text-zinc-500">
            {s.drafts.sent} sent as written · {s.drafts.edited} sent after edits · {s.drafts.deleted} deleted · {s.drafts.waiting} waiting
          </p>
          <ul className="space-y-1 text-[13px]">
            {d.drafts.slice(0, 8).map((x) => (
              <li key={x.id} className="flex gap-2">
                <span className="min-w-0 flex-1 truncate text-zinc-300">{x.subject || '(no subject)'}</span>
                <span className="shrink-0 text-xs text-zinc-500">
                  {{ waiting: 'Waiting', sent: 'Sent as written', edited: `Sent, ${pct(x.similarity)} kept`, deleted: 'Deleted', expired: 'Left alone' }[x.status] ?? x.status}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
      <section>
        <h3 className="mb-1.5 text-xs font-semibold text-zinc-400">What you did with its results</h3>
        {d.signals.length === 0 ? (
          <p className="text-xs text-zinc-500">Nothing yet. Rate a run, approve or reject what it would do, or send its drafts.</p>
        ) : (
          <ol className="space-y-1">
            {d.signals.map((g) => (
              <li key={g.id} className="flex gap-3 text-[13px]">
                <span className="w-28 shrink-0 text-xs text-zinc-500">{whenText(g.at)}</span>
                <span className="min-w-0 flex-1 text-zinc-300">{SIGNAL_LABELS[g.kind] ?? g.kind}</span>
                <span className={`shrink-0 text-xs ${g.value >= 0.7 ? 'text-emerald-600 dark:text-emerald-400' : g.value < 0.3 ? 'text-red-600 dark:text-red-300' : 'text-zinc-500'}`}>
                  {Math.round(g.value * 100)}%
                </span>
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  )
}

const INTENT_DONE: Record<Exclude<Intent['status'], 'open'>, { tone: PillTone; label: string }> = {
  approved: { tone: 'violet', label: 'Approved' },
  carried_out: { tone: 'green', label: 'Carried out' },
  rejected: { tone: 'zinc', label: 'Rejected' },
  dropped: { tone: 'zinc', label: 'Not done' },
}

function RunItem({ run: r, busy, deciding, onStop, onApprove, onReject, onRate, onMakeCase, making }: {
  run: Run
  busy: boolean
  deciding: boolean
  onStop: () => void
  onApprove: (ids: number[], total: number) => void
  onReject: () => void
  onRate: (kind: 'up' | 'down' | '') => void
  /** A run started on fixed input (new mail) can become a test case. */
  onMakeCase?: () => void
  making: boolean
}) {
  const going = r.status === 'queued' || r.status === 'running'
  const waiting = (r.intents ?? []).filter((i) => i.status === 'open')
  const [unticked, setUnticked] = useState<Set<number>>(new Set())
  const chosen = waiting.filter((i) => !unticked.has(i.id)).map((i) => i.id)
  const toggle = (id: number) => setUnticked((s) => {
    const n = new Set(s)
    if (n.has(id)) n.delete(id)
    else n.add(id)
    return n
  })
  return (
    <li className="fd-card p-3">
      <div className="flex items-center gap-2">
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
        {going && (
          <button onClick={onStop} disabled={busy}
            className="inline-flex h-7 items-center gap-1 rounded-full px-2.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100">
            {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Square className="h-3 w-3" />} Stop
          </button>
        )}
      </div>
      {r.waiting && r.note && <p className="mt-1.5 text-xs text-amber-700 dark:text-amber-300">{r.note}</p>}
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
