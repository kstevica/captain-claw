import { useEffect, useState } from 'react'
import { Hammer, ArrowLeft, ScanSearch, RefreshCw, TriangleAlert, Check } from 'lucide-react'
import { api } from '../../services/fileLibraryApi'
import type { CrewMe } from '../../services/crewApi'
import { customToolsApi, type SpecReview } from '../../services/customToolsApi'
import { Dialog, Button, Notice, Segmented, inputCls, labelCls } from '../console/parts'

// A new tool in two steps. 1: the person names it and writes what it should
// do. 2: Flight Deck's check (the person's model, with how custom tools run,
// what agents already do and the person's own tools) answers with a fuller
// spec and why; the person picks the checked spec or their own, edits it, and
// a Crew worker builds it. It goes live after they approve the merge.

let harnessCache: CrewMe | null = null

export function HarnessPicker({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  const [me, setMe] = useState<CrewMe | null>(harnessCache)
  useEffect(() => {
    if (harnessCache) return
    api<CrewMe>('/fd/crew/me').then((m) => { harnessCache = m; setMe(m) }).catch(() => setMe(null))
  }, [])
  const usable = (me?.harnesses || []).filter((h) => h.installed)
  useEffect(() => {
    if (!value && usable.length) onChange((usable.find((h) => h.login?.ok) || usable[0]).id)
  }, [value, usable, onChange])
  return (
    <select aria-label="Coding agent" className={`${inputCls.replace('w-full', 'w-auto max-w-full')}`} value={value} onChange={(e) => onChange(e.target.value)}>
      {usable.length === 0 && <option value="">No coding agent installed</option>}
      {usable.map((h) => <option key={h.id} value={h.id}>{h.label}{h.login?.ok ? '' : ' (needs a login in Crew)'}</option>)}
    </select>
  )
}

const SLUG = /^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$/
const MAX_SPEC = 8000
const PLACEHOLDER = `Read a CSV file and add up one column.

Input: the file, and the name of the column to add up.
Returns: how many rows it read, and the total.
Example: a file with the rows 5, 7.5 and 2.5 in "amount" gives 3 rows and 15.
When the column isn't there: say which columns the file has.`

type Step = 'write' | 'check'
type Which = 'checked' | 'mine'
type Checked = Extract<SpecReview, { spec: string }>

function Steps({ step }: { step: Step }) {
  const item = (n: number, label: string, on: boolean, done: boolean) => (
    <span className={`flex items-center gap-2 ${on ? 'text-zinc-100' : 'text-zinc-500'}`}>
      <span className={`flex h-5 w-5 items-center justify-center rounded-full text-[11px] font-semibold ${
        on ? 'bg-violet-600 text-white' : done ? 'bg-violet-500/15 text-violet-600 dark:text-violet-400' : 'bg-zinc-800 text-zinc-500'}`}>
        {done ? <Check className="h-3 w-3" /> : n}
      </span>
      {label}
    </span>
  )
  return (
    <div className="mb-4 flex shrink-0 items-center gap-3 text-xs font-medium">
      {item(1, 'Describe it', step === 'write', step === 'check')}
      <span className="h-px w-6 bg-zinc-800" />
      {item(2, 'Check it and build', step === 'check', false)}
    </div>
  )
}

function SpecBox({ value, onChange, label, placeholder, autoFocus }: {
  value: string; onChange: (v: string) => void; label: string; placeholder?: string; autoFocus?: boolean
}) {
  return (
    <div className="flex min-h-[18rem] flex-1 flex-col">
      <textarea
        aria-label={label}
        autoFocus={autoFocus}
        className={`${inputCls} min-h-0 flex-1 resize-none font-mono text-[13px] leading-relaxed`}
        value={value}
        maxLength={MAX_SPEC}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
      <div className="mt-1 text-right text-[11px] tabular-nums text-zinc-500">{value.length} / {MAX_SPEC}</div>
    </div>
  )
}

export function NewToolDialog({ taken = [], onClose, onStarted }: {
  taken?: string[]
  onClose: () => void
  onStarted: (message: string) => void
}) {
  const [step, setStep] = useState<Step>('write')
  const [slug, setSlug] = useState('')
  const [title, setTitle] = useState('')
  const [what, setWhat] = useState('')           // the person's own spec
  const [review, setReview] = useState<Checked | null>(null)
  const [checked, setChecked] = useState('')     // the checked spec, as the person edits it
  const [which, setWhich] = useState<Which>('checked')
  const [checkError, setCheckError] = useState('')
  const [harness, setHarness] = useState('')
  const [busy, setBusy] = useState<'' | 'check' | 'build'>('')
  const [error, setError] = useState('')

  const nameTaken = taken.includes(slug)
  const ok = SLUG.test(slug) && !nameTaken && what.trim().length >= 10
  const useChecked = !!review && which === 'checked'
  const spec = useChecked ? checked : what

  const check = async () => {
    setBusy('check')
    setError('')
    setCheckError('')
    try {
      const r = await customToolsApi.review({ slug, title, what })
      if (r.error !== undefined) {
        // No spec came back: say why, and the person can still build theirs.
        setReview(null)
        setCheckError(r.error)
      } else {
        setReview(r)
        setChecked(r.spec)
        setWhich(r.unchanged ? 'mine' : 'checked')
        if (!title.trim() && r.title) setTitle(r.title)
      }
      setStep('check')
    } catch (e) {
      // Flight Deck refused the name or the text: fixed here, in step one.
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setBusy('')
    }
  }

  const build = async () => {
    setBusy('build')
    setError('')
    try {
      const started = await customToolsApi.build({ slug, title, what: spec, harness })
      onStarted(`${started.task.harness_label || 'A coding agent'} is building “${slug}”. Approve its merge on the Crew page when it’s done; it goes live once its tests and examples pass.`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
      setBusy('')
    }
  }

  const footer = step === 'write' ? (
    <>
      <Button variant="ghost" onClick={onClose}>Cancel</Button>
      <Button variant="primary" icon={ScanSearch} busy={busy === 'check'} disabled={!ok} onClick={check}>
        {busy === 'check' ? 'Checking… up to a minute' : 'Check it'}
      </Button>
    </>
  ) : (
    <>
      <Button variant="ghost" icon={ArrowLeft} className="mr-auto" disabled={!!busy} onClick={() => { setError(''); setStep('write') }}>Back</Button>
      <HarnessPicker value={harness} onChange={setHarness} />
      <Button variant="primary" icon={Hammer} busy={busy === 'build'} disabled={!harness || spec.trim().length < 10} onClick={build}>Build it</Button>
    </>
  )

  return (
    <Dialog title="New custom tool" icon={Hammer} onClose={onClose} footer={footer} large keepOpen={!!what.trim() || !!busy}>
      <Steps step={step} />
      {error && <div className="mb-4 shrink-0"><Notice tone="red">{error}</Notice></div>}

      {step === 'write' ? (
        <div className="flex min-h-0 flex-1 flex-col gap-4">
          <div className="grid shrink-0 gap-4 wide:grid-cols-2">
            <div>
              <label className={labelCls} htmlFor="xt-new-name">Name</label>
              <input id="xt-new-name" className={`${inputCls} font-mono`} value={slug} maxLength={48} autoComplete="off"
                onChange={(e) => setSlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-'))} />
              <p className={`mt-1.5 text-[11px] leading-relaxed ${nameTaken ? 'text-red-500' : 'text-zinc-500'}`}>
                {nameTaken ? `You already have a tool called ${slug}: improve it from its page, or pick another name.`
                  : 'Lowercase letters, digits and dashes, like invoice-parse. Agents call it by this name.'}
              </p>
            </div>
            <div>
              <label className={labelCls} htmlFor="xt-new-title">Title</label>
              <input id="xt-new-title" className={inputCls} value={title} maxLength={80} onChange={(e) => setTitle(e.target.value)} />
              <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-500">A few words, like “Read an invoice PDF”. Left empty, the check suggests one.</p>
            </div>
          </div>
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
              <span className="text-xs font-medium text-zinc-400">What it should do</span>
              <span className="text-[11px] text-zinc-500">Its input, what it returns, an example with the exact answer, and what to do with bad input.</span>
            </div>
            <SpecBox label="What it should do" value={what} onChange={setWhat} placeholder={PLACEHOLDER} autoFocus />
            <p className="mt-1 shrink-0 text-[11px] leading-relaxed text-zinc-500">
              Next, Flight Deck checks it against how custom tools run, what your agents already do without one, and the tools you have,
              and suggests what to add or change. You choose what gets built.
            </p>
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-5 wide:flex-row">
          {/* What the check said */}
          <div className="space-y-3 wide:w-[38%] wide:shrink-0 wide:overflow-y-auto wide:pr-1">
            <div className="text-sm">
              <div className="font-semibold text-zinc-100">{title || slug}</div>
              <div className="font-mono text-[11px] text-zinc-500">{slug}</div>
            </div>
            {checkError ? (
              <>
                <Notice tone="amber">The check didn’t work: {checkError}</Notice>
                <Button icon={RefreshCw} busy={busy === 'check'} onClick={check}>Check again</Button>
              </>
            ) : review && (
              <>
                <p className="text-[11px] text-zinc-500">Checked by {review.model}.</p>
                {review.overlap && (
                  <div className="flex gap-2 rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-xs leading-relaxed text-amber-700 dark:text-amber-300">
                    <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
                    <div><span className="font-semibold">Already there? </span>{review.overlap}</div>
                  </div>
                )}
                {review.suggestions.length === 0 ? (
                  <Notice tone="emerald">Nothing to add: it reads complete.</Notice>
                ) : (
                  <div>
                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">What the check changed</h3>
                    <ol className="space-y-2">
                      {review.suggestions.map((s, i) => (
                        <li key={i} className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
                          <div className="text-xs font-medium leading-relaxed text-zinc-200">{s.change}</div>
                          {s.why && <div className="mt-1 text-[11px] leading-relaxed text-zinc-500">{s.why}</div>}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </>
            )}
          </div>

          {/* What gets built */}
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-medium text-zinc-400">The spec the coding agent gets</span>
              {review && !review.unchanged && (
                <Segmented label="Which spec" value={which} onChange={setWhich}
                  options={[{ value: 'checked', label: 'Checked' }, { value: 'mine', label: 'Yours' }]} />
              )}
            </div>
            <SpecBox label="The spec the coding agent gets" value={spec} onChange={useChecked ? setChecked : setWhat} />
            <p className="mt-1 shrink-0 text-[11px] leading-relaxed text-zinc-500">
              Edit it as you like. The coding agent writes the tool, its tests and its examples from this; it goes live after you approve the merge on the Crew page.
            </p>
          </div>
        </div>
      )}
    </Dialog>
  )
}
