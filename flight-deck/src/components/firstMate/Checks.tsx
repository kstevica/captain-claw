import { useCallback, useEffect, useState } from 'react'
import { ChevronDown, ChevronRight, Loader2, Lock, Pencil, Play, Plus, Square, Trash2 } from 'lucide-react'
import {
  addCase,
  checkNow,
  deleteCase,
  listCases,
  stopCheck,
  updateCase,
  whenText,
  CHECK_PURPOSE_LABELS,
  type Case,
  type CaseFields,
  type Check,
} from '../../services/firstMate'
import { ErrorBox, Pill, errText, type PillTone } from './ui'

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
    <div className="space-y-5">
      {error && <ErrorBox text={error} onDismiss={() => setError(null)} />}
      <p className="text-[13px] leading-relaxed text-zinc-400">
        Cases are inputs this factory might get, each with what a good result does. A check runs a version on
        every case as a rehearsal, so nothing outside changes, and a reviewer on another model grades each
        result. A new version goes live only if it scores about as well as the live one. It's also checked once
        a week while your First Mate is on.
      </p>

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

      <section className="space-y-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="flex-1 text-sm font-semibold text-zinc-100">Cases · {cases.length} of {data.max}</h3>
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
        </div>

        {editing === 'new' && (
          <CaseForm form={form} setForm={setForm} label={label} input={input} busy={busy === 'save'} ready={ready}
            title={form.fromRun ? 'A case from a run' : 'A new case'}
            hint={form.fromRun && !form.input.trim() ? 'That run had no fixed input. Paste one it might get.' : ''}
            onCancel={() => { setEditing(null); setForm(EMPTY) }} onSave={() => void save()} />
        )}

        {cases.length === 0 && editing !== 'new' && (
          <p className="text-xs text-zinc-500">
            No cases yet. Add one by hand, or open a run that started on new mail and choose Make a test case.
          </p>
        )}

        <ul className="space-y-2.5">
          {cases.map((c) => (editing === c.id ? (
            <li key={c.id}>
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
      </section>

      {data.checks.length > 0 && (
        <section>
          <h3 className="mb-2 text-sm font-semibold text-zinc-100">Checks</h3>
          <ul className="space-y-1.5">
            {data.checks.map((k) => (
              <li key={k.id} className="text-[13px]">
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
        </section>
      )}
    </div>
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
