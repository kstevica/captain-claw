import { useCallback, useEffect, useState } from 'react'
import { ChevronDown, ChevronRight, ExternalLink, FlaskConical, History, Info, Loader2, Lock, Pencil, Play, Plus, Sparkles, Square, Trash2 } from 'lucide-react'
import {
  addCase,
  checkNow,
  deleteCase,
  draftImprovement,
  improveFactory,
  listCases,
  listImprovements,
  stopCheck,
  updateCase,
  whenText,
  CHECK_PURPOSE_LABELS,
  type Case,
  type CaseFields,
  type Check,
  type ImprovementRun,
} from '../../services/firstMate'
import { useUIStore } from '../../stores/uiStore'
import { ErrorBox, Pill, Section, errText, type PillTone } from './ui'

// A factory's test cases and the checks run on them. A case is an input the
// factory might get and what a good result does; a check runs one version on
// every case, as a rehearsal, and a reviewer on another model grades each
// result. Sealed cases are shown here, to the owner, and to nothing that
// works on the factory.

export interface CaseSeed { title: string; input: string; expect: string[]; from_run: string; has_input: boolean }

interface Form { title: string; input: string; expect: string; sealed: boolean; fromRun: string }

const EMPTY: Form = { title: '', input: '', expect: '', sealed: true, fromRun: '' }

const formOf = (c: Case): Form => ({ title: c.title, input: c.input, expect: c.expect.join('\n'), sealed: c.sealed, fromRun: '' })
const fieldsOf = (f: Form): CaseFields => ({
  title: f.title.trim(), input: f.input, sealed: f.sealed,
  expect: f.expect.split('\n').map((s) => s.trim()).filter(Boolean),
})

export const pct = (x: number | null | undefined) => (x == null ? '—' : `${Math.round(x * 100)}%`)
export const scoreTone = (x: number | null | undefined): PillTone =>
  x == null ? 'zinc' : x >= 0.8 ? 'green' : x >= 0.5 ? 'amber' : 'red'

export function Checks({ factoryId, live, seed, onSeedUsed, onChanged, onNotice }: {
  factoryId: string
  /** The live version, if any: only a live factory is checked. */
  live: number
  /** A case started from a past run, to finish in the form. */
  seed: CaseSeed | null
  onSeedUsed: () => void
  onChanged: () => void
  onNotice: (text: string) => void
}) {
  const [data, setData] = useState<{ cases: Case[]; checks: Check[]; max: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [editing, setEditing] = useState<string | null>(null) // a case id, or 'new'
  const [form, setForm] = useState<Form>(EMPTY)
  const [confirm, setConfirm] = useState<string | null>(null)
  const [openCase, setOpenCase] = useState<string | null>(null)
  const [openCheck, setOpenCheck] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setData(await listCases(factoryId))
      setError(null)
    } catch (e) {
      setError(errText(e))
    }
  }, [factoryId])
  useEffect(() => { void load() }, [load])

  useEffect(() => {
    if (!seed) return
    setForm({ title: seed.title, input: seed.input, expect: seed.expect.join('\n'), sealed: true, fromRun: seed.from_run })
    setEditing('new')
    onSeedUsed()
  }, [seed, onSeedUsed])

  const running = data?.checks.find((c) => c.status === 'running') ?? null
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => { void load() }, 5000)
    return () => clearInterval(t)
  }, [running, load])

  const act = async (key: string, fn: () => Promise<unknown>, done?: string) => {
    setBusy(key); setError(null)
    try {
      await fn()
      if (done) onNotice(done)
      await load()
      onChanged()
      return true
    } catch (e) {
      setError(errText(e))
      return false
    } finally {
      setBusy(null)
    }
  }

  const save = async () => {
    const ok = await act('save', () => (editing === 'new'
      ? addCase(factoryId, { ...fieldsOf(form), ...(form.fromRun ? { from_run: form.fromRun } : {}) })
      : updateCase(factoryId, editing!, fieldsOf(form))), editing === 'new' ? 'Case added' : 'Case saved')
    if (ok) { setEditing(null); setForm(EMPTY) }
  }

  const titleOf = (id: string) => data?.cases.find((c) => c.id === id)?.title ?? 'A deleted case'
  const label = 'mb-1.5 block text-xs font-semibold text-zinc-400'
  const input = 'w-full rounded-xl border border-[var(--fd-line)] bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none'
  const cases = data?.cases ?? []
  const canCheck = !!live && cases.length > 0 && !running
  const ready = !!form.title.trim() && !!form.input.trim() && !!form.expect.trim()

  if (!data) return error ? <ErrorBox text={error} onDismiss={() => setError(null)} /> : <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />

  return (
    <div className="space-y-7">
      {error && <ErrorBox text={error} onDismiss={() => setError(null)} />}
      <div className="flex items-start gap-3 rounded-2xl border border-[var(--fd-line)] bg-gradient-to-br from-violet-500/[0.07] via-transparent to-fuchsia-500/[0.05] px-4 py-3">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-violet-600 dark:text-violet-300" />
        <p className="text-[13px] leading-relaxed text-zinc-400">
          Cases are inputs this factory might get, each with what a good result does. A check runs a version on
          every case as a rehearsal, so nothing outside changes, and a reviewer on another model grades each
          result. A new version goes live only if it scores about as well as the live one. It's also checked once
          a week while your First Mate is on.
        </p>
      </div>

      {running && (
        <section className="fd-card flex flex-wrap items-center gap-2 border-violet-500/30 p-3">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-500" />
          <span className="min-w-0 flex-1 text-[13px] text-zinc-300">
            Checking version {running.version} ({CHECK_PURPOSE_LABELS[running.purpose].toLowerCase()}):{' '}
            {Object.values(running.results).filter((r) => r.score != null || r.note).length} of {running.cases.length} cases graded
          </span>
          <button onClick={() => void act('stop', () => stopCheck(factoryId), 'Check stopped')} disabled={!!busy}
            className="inline-flex h-7 items-center gap-1 rounded-full px-2.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100">
            {busy === 'stop' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Square className="h-3 w-3" />} Stop
          </button>
        </section>
      )}

      <Section icon={FlaskConical} title="Cases" hint={`${cases.length} of ${data.max} · inputs it might get, each with what a good result does`}
        action={<div className="flex flex-wrap items-center justify-end gap-2">
          {editing === null && cases.length < data.max && (
            <button onClick={() => { setForm(EMPTY); setEditing('new') }} className="fd-btn-quiet h-8 px-3 text-xs">
              <Plus className="h-3.5 w-3.5" /> Add a case
            </button>
          )}
          <button onClick={() => void act('check', () => checkNow(factoryId), 'Checking the live version on its cases')}
            disabled={!canCheck || !!busy}
            title={!live ? 'Approve a charter first' : !cases.length ? 'Add a case first' : running ? 'A check is going' : 'Check the live version now'}
            className="inline-flex h-8 items-center gap-1.5 rounded-full bg-violet-600 px-3.5 text-xs font-semibold text-white transition-colors hover:bg-violet-500 disabled:opacity-60">
            {busy === 'check' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />} Run the checks now
          </button>
        </div>}>
        <div className="space-y-2.5">

        {editing === 'new' && (
          <CaseForm form={form} setForm={setForm} label={label} input={input} busy={busy === 'save'} ready={ready}
            title={form.fromRun ? 'A case from a run' : 'A new case'}
            hint={form.fromRun && !form.input.trim() ? 'That run had no fixed input. Paste one it might get.' : ''}
            onCancel={() => { setEditing(null); setForm(EMPTY) }} onSave={() => void save()} />
        )}

        {cases.length === 0 && editing !== 'new' && (
          <p className="rounded-[var(--fd-radius-card)] border border-dashed border-zinc-800 px-4 py-4 text-sm text-zinc-500">
            No cases yet. Add one by hand, or open a run that started on new mail and choose Make a test case.
          </p>
        )}

        <ul className="grid gap-2.5 @5xl:grid-cols-2">
          {cases.map((c) => (editing === c.id ? (
            <li key={c.id} className="@5xl:col-span-2">
              <CaseForm form={form} setForm={setForm} label={label} input={input} busy={busy === 'save'} ready={ready}
                title="Edit the case" hint="" onCancel={() => { setEditing(null); setForm(EMPTY) }} onSave={() => void save()} />
            </li>
          ) : (
            <li key={c.id} className="fd-card p-3">
              <div className="flex items-center gap-2">
                <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-zinc-100">{c.title}</span>
                {c.sealed
                  ? <Pill tone="violet" title="Never shown to the factory, or to anything that rewrites its charter"><Lock className="mr-1 h-2.5 w-2.5" />Sealed</Pill>
                  : <Pill tone="zinc" title="May be shown when its charter is reworked">Open</Pill>}
                {c.origin === 'run' && <Pill tone="zinc">From a run</Pill>}
                <button onClick={() => { setForm(formOf(c)); setEditing(c.id) }} disabled={editing !== null}
                  title="Edit" aria-label={`Edit ${c.title}`}
                  className="flex h-7 w-7 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                  <Pencil className="h-3.5 w-3.5" />
                </button>
                {confirm === c.id ? (
                  <button onClick={() => void act(`del:${c.id}`, () => deleteCase(factoryId, c.id), 'Case deleted').then(() => setConfirm(null))}
                    className="rounded-full bg-red-500/10 px-2.5 py-1 text-xs text-red-700 dark:text-red-300">
                    {busy === `del:${c.id}` ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Delete it'}
                  </button>
                ) : (
                  <button onClick={() => setConfirm(c.id)} title="Delete" aria-label={`Delete ${c.title}`}
                    className="flex h-7 w-7 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
              <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-xs text-zinc-400">
                {c.expect.map((e) => <li key={e}>{e}</li>)}
              </ul>
              <button onClick={() => setOpenCase(openCase === c.id ? null : c.id)} className="mt-2 block w-full text-left">
                <span className={`whitespace-pre-wrap rounded-lg bg-[var(--fd-hover)] px-2.5 py-1.5 text-xs leading-relaxed text-zinc-400 ${openCase === c.id ? 'block' : 'line-clamp-2'}`}>{c.input}</span>
              </button>
            </li>
          )))}
        </ul>
        </div>
      </Section>

      <Improvements factoryId={factoryId} onChanged={onChanged} onNotice={onNotice} />

      {data.checks.length > 0 && (
        <Section icon={History} title="Checks run" hint="Each version's score on its cases">
          <ul className="fd-card divide-y divide-[var(--fd-line)] overflow-hidden p-0">
            {data.checks.map((k) => (
              <li key={k.id} className="px-3 py-2 text-[13px]">
                <button onClick={() => setOpenCheck(openCheck === k.id ? null : k.id)} className="flex w-full items-center gap-2 text-left">
                  {openCheck === k.id ? <ChevronDown className="h-3.5 w-3.5 shrink-0 text-zinc-500" /> : <ChevronRight className="h-3.5 w-3.5 shrink-0 text-zinc-500" />}
                  <span className="w-24 shrink-0 text-xs text-zinc-500">{whenText(k.created_at)}</span>
                  <span className="min-w-0 flex-1 truncate text-zinc-300">{CHECK_PURPOSE_LABELS[k.purpose]} · v{k.version}</span>
                  {k.status === 'done' ? <Pill tone={scoreTone(k.score)}>{pct(k.score)}</Pill>
                    : <Pill tone={k.status === 'failed' ? 'red' : k.status === 'running' ? 'violet' : 'zinc'}>
                      {{ running: 'Going', void: 'Called off', failed: 'Failed' }[k.status]}
                    </Pill>}
                </button>
                {openCheck === k.id && (
                  <div className="ml-5 mt-1.5 space-y-1.5 rounded-xl bg-[var(--fd-hover)] px-3 py-2 text-xs text-zinc-400">
                    {k.note && <p className="text-zinc-300">{k.note}</p>}
                    {k.baseline != null && <p>The live version scored {pct(k.baseline)} before.</p>}
                    <ul className="space-y-1">
                      {k.cases.map((id) => {
                        const r = k.results[id]
                        return (
                          <li key={id} className="flex gap-2">
                            <span className="min-w-0 flex-1 truncate">{titleOf(id)}</span>
                            {r?.note && <span className="shrink-0 truncate text-zinc-500" title={r.note}>{r.note}</span>}
                            <span className="w-10 shrink-0 text-right">{r ? pct(r.score) : '—'}</span>
                          </li>
                        )
                      })}
                    </ul>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </Section>
      )}
    </div>
  )
}

const RUN_WORDS: Record<string, string> = {
  awaiting_go: 'Waiting', baseline: 'Measuring', running: 'Going', done: 'Done', stalled: 'Done', stopped: 'Stopped',
  failed: 'Failed', cancelled: 'Cancelled', declined: 'Declined',
}

/** Captain Swarm runs that try better charters on this factory's cases. Admins only, like Captain Swarm. */
function Improvements({ factoryId, onChanged, onNotice }: {
  factoryId: string
  onChanged: () => void
  onNotice: (text: string) => void
}) {
  const [data, setData] = useState<{ runs: ImprovementRun[]; can: { ok: boolean; why: string }; available: boolean } | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const setView = useUIStore((s) => s.setView)
  const load = useCallback(async () => {
    try {
      setData(await listImprovements(factoryId))
    } catch (e) {
      setError(errText(e))
    }
  }, [factoryId])
  useEffect(() => { void load() }, [load])
  const going = !!data?.runs.some((r) => r.going)
  useEffect(() => {
    if (!going) return
    const t = setInterval(() => { void load() }, 10000)
    return () => clearInterval(t)
  }, [going, load])
  const act = async (key: string, fn: () => Promise<unknown>, done: string) => {
    setBusy(key); setError(null)
    try {
      await fn()
      onNotice(done)
      await load()
      onChanged()
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(null)
    }
  }
  if (!data) return null
  if (!data.runs.length && !data.available) return null  // Captain Swarm isn't theirs on this deck
  return (
    <Section icon={Sparkles} title="Improvement runs" hint="Captain Swarm looks for a better charter on these cases"
      action={
        <button onClick={() => void act('improve', () => improveFactory(factoryId),
          'An improvement run started: it tries better charters on these cases')}
          disabled={!data.can.ok || !!busy} title={data.can.ok ? 'Captain Swarm tries better charters on these cases' : data.can.why}
          className="fd-btn-quiet h-8 px-3 text-xs">
          {busy === 'improve' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />} Run one now
        </button>}>
      <div className="space-y-2.5">
      <p className="text-xs text-zinc-500">
        Captain Swarm tries better charters and steps on these cases, as rehearsals, and keeps only what scores
        clearly higher and holds up on the sealed cases it never showed the writer. The best comes back as a draft
        you approve. {!data.can.ok && data.can.why ? data.can.why : ''}
      </p>
      {error && <ErrorBox text={error} onDismiss={() => setError(null)} />}
      {data.runs.length > 0 && (
        <ul className="fd-card divide-y divide-[var(--fd-line)] overflow-hidden p-0">
          {data.runs.map((r) => (
            <li key={r.id} className="flex flex-wrap items-center gap-2 px-3 py-2 text-[13px]">
              <span className="w-24 shrink-0 text-xs text-zinc-500">{whenText(r.created_at)}</span>
              <Pill tone={r.going ? 'violet' : r.improved ? 'green' : r.status === 'failed' ? 'red' : 'zinc'}>
                {r.going ? RUN_WORDS[r.status] ?? r.status : r.improved ? 'Found a better one' : RUN_WORDS[r.status] ?? r.status}
              </Pill>
              <span className="min-w-0 flex-1 truncate text-xs text-zinc-400">
                {r.experiments} {r.experiments === 1 ? 'idea' : 'ideas'}
                {r.baseline_dev != null && <> · {pct(r.baseline_dev)}{r.champion_dev != null && r.improved ? ` → ${pct(r.champion_dev)}` : ''}</>}
                {r.drafted && ' · became a draft'}
              </span>
              {r.improved && !r.drafted && !r.going && (
                <button onClick={() => void act(`draft:${r.id}`, () => draftImprovement(factoryId, r.id),
                  'Its best charter is a draft now: approve it on the Charter tab')}
                  disabled={!!busy} className="fd-btn-quiet h-7 px-2.5 text-xs">
                  {busy === `draft:${r.id}` ? <Loader2 className="h-3 w-3 animate-spin" /> : null} Make it a draft
                </button>
              )}
              <button onClick={() => setView('captain-swarm')} title="Open Captain Swarm"
                className="flex h-7 w-7 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                <ExternalLink className="h-3.5 w-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
      </div>
    </Section>
  )
}

function CaseForm({ form, setForm, label, input, busy, ready, title, hint, onCancel, onSave }: {
  form: Form
  setForm: (f: Form) => void
  label: string
  input: string
  busy: boolean
  ready: boolean
  title: string
  hint: string
  onCancel: () => void
  onSave: () => void
}) {
  return (
    <section className="fd-card space-y-3 p-4">
      <h4 className="text-sm font-semibold text-zinc-100">{title}</h4>
      <div>
        <label className={label} htmlFor="fm-case-title">Name</label>
        <input id="fm-case-title" className={input} value={form.title} placeholder="A client asks about an invoice total"
          onChange={(e) => setForm({ ...form, title: e.target.value })} />
      </div>
      <div>
        <label className={label} htmlFor="fm-case-input">What it gets</label>
        <textarea id="fm-case-input" rows={6} className={`${input} resize-y leading-relaxed`} value={form.input}
          placeholder="The email, request or page, as the factory would see it"
          onChange={(e) => setForm({ ...form, input: e.target.value })} />
        {hint && <p className="mt-1 text-[11px] text-amber-700 dark:text-amber-300">{hint}</p>}
      </div>
      <div>
        <label className={label} htmlFor="fm-case-expect">A good result (one per line)</label>
        <textarea id="fm-case-expect" rows={3} className={`${input} resize-y`} value={form.expect}
          placeholder={'Drafts a reply to the sender\nAnswers the question asked'}
          onChange={(e) => setForm({ ...form, expect: e.target.value })} />
      </div>
      <label className="flex items-start gap-2 text-xs text-zinc-400">
        <input type="checkbox" checked={form.sealed} onChange={(e) => setForm({ ...form, sealed: e.target.checked })}
          className="mt-0.5 h-3.5 w-3.5 accent-violet-600" />
        <span>Sealed: never shown to the factory, or to anything that rewrites its charter, so it can't learn the answers.</span>
      </label>
      <div className="flex justify-end gap-2 border-t border-[var(--fd-line)] pt-3">
        <button onClick={onCancel} disabled={busy} className="fd-btn-quiet h-8 px-3 text-xs">Cancel</button>
        <button onClick={onSave} disabled={busy || !ready}
          className="inline-flex h-8 items-center gap-1.5 rounded-full bg-violet-600 px-3.5 text-xs font-semibold text-white transition-colors hover:bg-violet-500 disabled:opacity-60">
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null} Save the case
        </button>
      </div>
    </section>
  )
}
