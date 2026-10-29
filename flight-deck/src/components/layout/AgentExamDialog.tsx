import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertTriangle, ArrowLeft, BookOpen, CalendarDays, Check, ChevronDown, CircleDashed, Clock, Cpu, Equal, FileText,
  GraduationCap, Info, Lightbulb, List, ListTree, Loader2, MessageSquareText, Play, RotateCcw, Search, Square, Target, Timer,
  Trash2, TrendingDown, TrendingUp, Wrench, X, Zap,
} from 'lucide-react'
import {
  agentExamApi, type Cause, type Exam, type ExamOverview, type ExamQuestion, type ExamSize, type Lookup, type Score,
  type Verdict,
} from '../../services/agentExam'

// ── An agent's onboarding exam ───────────────────────────────────────
//
// The Onboarding exam item in the kiosk's agent menu opens this on one agent
// (/fd/agent-exam/{slug}): questions written from its documents, asked the way
// its users would ask them, in private sessions that keep nothing — then
// graded against the section each came from.
//
//   start    — what it is examined on (all its documents, or the ones
//              picked), how many questions (Quick / Standard / Deep /
//              Marathon) and what that costs, an optional focus, earlier exams;
//   running  — writing, asking, grading, live: the question it is asked now,
//              and the answers as they come, rendered as the chat renders them;
//   results  — the score and how the answers went, by kind, by document and
//              by whether the setup read that part; each question with its
//              answer, the answer its document gives, what it looked up, why
//              it went wrong — and the fix: teach it the answer, or read the
//              part it was never read.

const POLL_MS = 2000

const SIZES: { id: ExamSize; label: string; hint: string }[] = [
  { id: 'quick', label: 'Quick', hint: 'A first look' },
  { id: 'standard', label: 'Standard', hint: 'A fair picture' },
  { id: 'deep', label: 'Deep', hint: 'Into every corner' },
  { id: 'marathon', label: 'Marathon', hint: 'As wide as it goes' },
]

/** Past this many documents, the pick gets a search. */
const MATERIALS_SEARCH = 8

// A verdict's label, its colour in a bar (dot), on a question's number (badge) and as a tag (pill).
const VERDICT: Record<Verdict, { label: string; dot: string; badge: string; pill: string }> = {
  right: { label: 'Right', dot: 'bg-emerald-500', badge: 'bg-emerald-500/15 text-emerald-700 ring-emerald-500/30 dark:text-emerald-300',
    pill: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400' },
  partly: { label: 'Partly right', dot: 'bg-amber-500', badge: 'bg-amber-500/15 text-amber-700 ring-amber-500/30 dark:text-amber-300',
    pill: 'bg-amber-500/10 text-amber-700 dark:text-amber-400' },
  wrong: { label: 'Wrong', dot: 'bg-red-500', badge: 'bg-red-500/15 text-red-700 ring-red-500/30 dark:text-red-300',
    pill: 'bg-red-500/10 text-red-700 dark:text-red-400' },
  no_answer: { label: 'No answer', dot: 'bg-zinc-500', badge: 'bg-zinc-800 text-zinc-400 ring-zinc-700',
    pill: 'bg-zinc-800 text-zinc-400' },
}

const CAUSES: Cause[] = ['didnt_look', 'wrong_place', 'got_wrong', 'made_up']
const CAUSE: Record<Cause, { label: string; explain: string }> = {
  didnt_look: { label: "Didn't look", explain: 'It answered from memory, without looking in its documents.' },
  wrong_place: { label: 'Looked elsewhere', explain: 'It searched its documents but never reached this section.' },
  got_wrong: { label: 'Found it, got it wrong', explain: 'It reached the right section and still answered wrong.' },
  made_up: { label: 'Made things up', explain: "It stated details that aren't in your documents." },
}

const PHASES: { id: Exam['phase']; label: string }[] = [
  { id: 'writing', label: 'Writing' },
  { id: 'asking', label: 'Asking' },
  { id: 'grading', label: 'Grading' },
]

type Filter = 'all' | 'wrong' | 'right' | Cause

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))
const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`
const kTokens = (n: number) => (n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1000 ? `${Math.round(n / 1000)}k` : `${n}`)
const when = (epoch: number) => new Date(epoch * 1000).toLocaleString(undefined, {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
})
const took = (e: Exam) => {
  const s = Math.max(0, Math.round((e.finished ?? Date.now() / 1000) - e.started))
  return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : s >= 60 ? `${Math.round(s / 60)} min` : `${s} s`
}
const scoreTone = (pct: number | null) =>
  pct === null ? 'text-zinc-500' : pct >= 80 ? 'text-emerald-600 dark:text-emerald-400'
    : pct >= 50 ? 'text-amber-600 dark:text-amber-400' : 'text-red-600 dark:text-red-400'
const scoreBar = (pct: number | null) =>
  pct === null ? 'bg-zinc-600' : pct >= 80 ? 'bg-emerald-500' : pct >= 50 ? 'bg-amber-500' : 'bg-red-500'
const wentWrong = (q: ExamQuestion) => !!q.verdict && q.verdict !== 'right'
const sizeLabel = (s: ExamSize) => SIZES.find((x) => x.id === s)?.label ?? s

const MD: Components = {
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer noopener" className="text-violet-600 underline underline-offset-2 dark:text-violet-400">{children}</a>
  ),
  img: ({ alt }) => (alt ? <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[11px] text-zinc-500">[picture: {alt}]</span> : null),
}

export function AgentExamDialog({ agent, onClose, onReadDocs }: {
  agent: { slug: string; name: string }
  onClose: () => void
  /** Open its Way of working with these documents (their setup names) staged to read. */
  onReadDocs?: (docs: string[]) => void
}) {
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label={`${agent.name}'s onboarding exam`}
        className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-900 shadow-2xl wide:h-[min(90vh,56rem)] wide:max-w-5xl wide:rounded-2xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        <AgentExamPanel agent={agent} onClose={onClose} onReadDocs={onReadDocs} />
      </div>
    </div>,
    document.body,
  )
}

/** The exam itself — in its own dialog (with *onClose*: a header with a close button), or a tab of
 *  the Onboarding page (a slim bar of its own). */
export function AgentExamPanel({ agent, onClose, onReadDocs }: {
  agent: { slug: string; name: string }
  onClose?: () => void
  onReadDocs?: (docs: string[]) => void
}) {
  const [overview, setOverview] = useState<ExamOverview | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const [exam, setExam] = useState<Exam | null>(null)
  const [view, setView] = useState<'start' | 'exam'>('start')
  const [size, setSize] = useState<ExamSize>('quick')
  const [focus, setFocus] = useState('')
  // The documents picked to be examined on (their files) — null: all of them.
  const [picked, setPicked] = useState<string[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [tries, setTries] = useState(0)

  // What it would be examined on, and its exams: open on the one under way, else the latest.
  useEffect(() => {
    let live = true
    agentExamApi.overview(agent.slug).then((o) => {
      if (!live) return
      setOverview(o)
      const first = o.running ?? o.exams[0]
      if (first) {
        setView('exam')
        agentExamApi.get(agent.slug, first.id).then((e) => { if (live) setExam(e) }).catch(() => undefined)
      }
    }).catch((e) => { if (live) setLoadErr(message(e)) })
    return () => { live = false }
  }, [agent.slug, tries])

  // Follow an exam under way.
  const runningId = exam?.state === 'running' ? exam.id : ''
  useEffect(() => {
    if (!runningId) return
    let live = true
    const t = setTimeout(() => {
      agentExamApi.get(agent.slug, runningId).then((next) => {
        if (!live) return
        setExam(next)
        if (next.state !== 'running') agentExamApi.overview(agent.slug).then((o) => { if (live) setOverview(o) }).catch(() => undefined)
      }).catch(() => undefined)
    }, POLL_MS)
    return () => { live = false; clearTimeout(t) }
  }, [agent.slug, runningId, exam])

  const act = async (fn: () => Promise<Exam>) => {
    setBusy(true)
    setErr('')
    try {
      const started = await fn()
      setExam(started)
      setView('exam')
    } catch (e) {
      setErr(message(e))
    } finally {
      setBusy(false)
    }
  }
  const start = () => act(() => agentExamApi.start(agent.slug, { size, focus: focus.trim(), docs: picked ?? [] }))
  const again = (id: string) => act(() => agentExamApi.start(agent.slug, { again: id }))
  const resume = (id: string) => act(() => agentExamApi.resume(agent.slug, id))
  const stop = async (id: string) => {
    setBusy(true)
    try {
      await agentExamApi.stop(agent.slug, id)
      setExam(await agentExamApi.get(agent.slug, id))
    } catch (e) {
      setErr(message(e))
    } finally {
      setBusy(false)
    }
  }
  const open = (id: string) => {
    setErr('')
    setView('exam')
    setExam(null)
    agentExamApi.get(agent.slug, id).then(setExam).catch((e) => setErr(message(e)))
  }
  const forget = async (id: string) => {
    await agentExamApi.stop(agent.slug, id).catch(() => undefined)
    setOverview((o) => (o ? { ...o, exams: o.exams.filter((x) => x.id !== id) } : o))
  }
  const taught = (qid: string) => setExam((e) => (e && e.questions
    ? { ...e, questions: e.questions.map((q) => (q.id === qid ? { ...q, taught: true } : q)) } : e))

  const running = exam?.state === 'running'
  const subtitle = overview
    ? `${plural(overview.docs, 'document')} · answered by ${overview.model || 'its model'}`
    : 'Does it really know its documents?'

  return (
    <div className="flex h-full min-h-0 flex-col">
        <div className={`flex shrink-0 items-center gap-3 border-b border-zinc-800 ${onClose ? 'px-4 py-3 wide:px-5' : 'px-4 py-2 wide:px-5'}`}>
          {view === 'start' && exam ? (
            <button onClick={() => setView('exam')} aria-label="Back to the exam"
              className="-ml-2 rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
              <ArrowLeft className="h-5 w-5" />
            </button>
          ) : onClose && (
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
              <GraduationCap className="h-4 w-4" />
            </span>
          )}
          <div className="min-w-0">
            {onClose
              ? <h3 className="truncate text-base font-semibold text-zinc-100">{agent.name}'s onboarding exam</h3>
              : <h3 className="truncate text-sm font-semibold text-zinc-200">{view === 'start' ? 'A new exam' : 'Exams'}</h3>}
            <p className="truncate text-xs text-zinc-500">{subtitle}</p>
          </div>
          <div className="ml-auto flex shrink-0 items-center gap-1">
            {view === 'exam' && !running && overview && (
              <button onClick={() => { setErr(''); setView('start') }}
                className="hidden items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 sm:flex">
                <Play className="h-3.5 w-3.5" /> New exam
              </button>
            )}
            {onClose && (
              <button onClick={onClose} aria-label="Close" title="Close"
                className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                <X className="h-5 w-5" />
              </button>
            )}
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {loadErr ? (
            <Centered>
              <AlertTriangle className="h-6 w-6 text-amber-500" />
              <p className="max-w-sm text-sm text-zinc-400">{loadErr}</p>
              <button onClick={() => { setLoadErr(''); setTries((n) => n + 1) }}
                className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800">Try again</button>
            </Centered>
          ) : !overview ? (
            <Centered><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></Centered>
          ) : view === 'start' || (!exam && !err) ? (
            view === 'start' ? (
              <StartPanel overview={overview} size={size} setSize={setSize} focus={focus} setFocus={setFocus}
                picked={picked} setPicked={setPicked} busy={busy} err={err} onStart={start} onOpen={open} onForget={forget} />
            ) : <Centered><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></Centered>
          ) : !exam ? (
            <Centered><p className="text-sm text-red-500">{err}</p></Centered>
          ) : exam.state === 'running' ? (
            <RunningPanel exam={exam} busy={busy} onStop={() => void stop(exam.id)} />
          ) : (
            <ResultsPanel
              key={exam.id} exam={exam} busy={busy} err={err}
              onAgain={() => void again(exam.id)} onResume={() => void resume(exam.id)}
              onNew={() => { setErr(''); setView('start') }}
              onTeach={async (q) => {
                await agentExamApi.teach(agent.slug, exam.id, q.id)
                taught(q.id)
              }}
              onRead={onReadDocs}
            />
          )}
        </div>
    </div>
  )
}

function Centered({ children }: { children: ReactNode }) {
  return <div className="flex h-full min-h-[16rem] flex-col items-center justify-center gap-3 p-6 text-center">{children}</div>
}

// ── Start ─────────────────────────────────────────────────────────────

function StartPanel({ overview, size, setSize, focus, setFocus, picked, setPicked, busy, err, onStart, onOpen, onForget }: {
  overview: ExamOverview
  size: ExamSize
  setSize: (s: ExamSize) => void
  focus: string
  setFocus: (f: string) => void
  picked: string[] | null
  setPicked: (p: string[] | null) => void
  busy: boolean
  err: string
  onStart: () => void
  onOpen: (id: string) => void
  onForget: (id: string) => void
}) {
  const nothing = overview.docs === 0 || overview.sections === 0
  const materials = overview.materials ?? []
  const chosen = picked === null ? materials : materials.filter((m) => picked.includes(m.file))
  // What the picked documents have to ask about: the most questions any size can have.
  const sections = picked === null ? overview.sections : chosen.reduce((n, m) => n + m.sections, 0)
  const costOf = (s: ExamSize) => (overview.per_question
    ? Math.min(overview.sizes[s], sections) * overview.per_question : overview.cost[s])
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 p-4 wide:p-6">
      <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4">
        <p className="text-sm leading-relaxed text-zinc-300">
          Questions are written from its documents and asked the way your users would ask them — some plain questions,
          some situations where a rule decides what to do. Each answer is checked against the part of the document it
          came from, and a wrong one says why: it didn't look, looked in the wrong place, or found it and still got it wrong.
        </p>
        <p className="mt-2 flex items-start gap-1.5 text-xs text-zinc-500">
          <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-500" />
          It answers in private: nothing it says here is kept in its memory, and it can't send or change anything.
        </p>
      </div>

      {nothing ? (
        <div className="flex items-start gap-3 rounded-xl border border-zinc-800 p-4">
          <BookOpen className="mt-0.5 h-5 w-5 shrink-0 text-zinc-500" />
          <p className="text-sm text-zinc-400">
            {overview.name} has no documents to be examined on yet. Give it some in Way of working first.
          </p>
        </div>
      ) : (
        <>
          {materials.length > 1 && <MaterialsPick materials={materials} picked={picked} setPicked={setPicked} />}

          <div>
            <div className="mb-2 flex items-baseline justify-between gap-2">
              <h4 className="text-sm font-semibold text-zinc-200">How many questions</h4>
              <span className="text-xs text-zinc-500">
                from {plural(chosen.length, 'document')} · {plural(sections, 'section')} to ask about
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4" role="radiogroup" aria-label="Exam size">
              {SIZES.map((s) => {
                const n = Math.min(overview.sizes[s.id], sections)
                const on = size === s.id
                return (
                  <button key={s.id} role="radio" aria-checked={on} onClick={() => setSize(s.id)}
                    className={`flex flex-col rounded-xl border p-3 text-left transition-colors ${on
                      ? 'border-violet-500 bg-violet-500/10 ring-1 ring-violet-500/40'
                      : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-800/50'}`}>
                    <span className="block text-sm font-semibold text-zinc-100">{s.label}</span>
                    <span className="mt-0.5 block text-2xl font-semibold tabular-nums text-zinc-100">{n}</span>
                    <span className="block text-[11px] text-zinc-500">questions · {s.hint}</span>
                    <span className="mt-2 block text-[11px] text-zinc-500">about {kTokens(costOf(s.id))} tokens</span>
                  </button>
                )
              })}
            </div>
          </div>

          <label className="block">
            <span className="mb-1.5 block text-sm font-semibold text-zinc-200">
              Focus <span className="font-normal text-zinc-500">— optional</span>
            </span>
            <input value={focus} onChange={(e) => setFocus(e.target.value)} maxLength={300}
              placeholder="e.g. returns and refunds, or pricing for business customers"
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-violet-500 focus:outline-none" />
            <span className="mt-1 block text-xs text-zinc-500">Questions lean to the parts closest to it.</span>
          </label>

          {err && <p className="flex items-start gap-1.5 text-sm text-red-500"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{err}</p>}

          <div className="flex items-center justify-end gap-3">
            {sections === 0 && (
              <span className="text-xs text-zinc-500">
                {chosen.length === 0 ? 'Pick at least one document.' : 'The picked documents have nothing long enough to ask about.'}
              </span>
            )}
            <button onClick={onStart} disabled={busy || sections === 0}
              className="flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-500 disabled:opacity-60">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />} Start the exam
            </button>
          </div>
        </>
      )}

      {overview.exams.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-zinc-200">Earlier exams</h4>
          <ul className="divide-y divide-zinc-800 overflow-hidden rounded-2xl border border-zinc-800">
            {overview.exams.map((e) => (
              <li key={e.id} className="group flex items-center gap-3 px-3 py-2.5 hover:bg-zinc-800/40">
                <button onClick={() => onOpen(e.id)} className="flex min-w-0 flex-1 items-center gap-3 text-left">
                  <ScoreRing pct={e.summary?.pct ?? null} size={44} />
                  <span className="min-w-0">
                    <span className="block truncate text-sm text-zinc-200">
                      {sizeLabel(e.size)}{e.again ? ' · asked again' : ''}{e.docs?.length ? ` · ${plural(e.docs.length, 'document')}` : ''}
                      {e.focus ? ` · ${e.focus}` : ''}
                    </span>
                    <span className="block truncate text-xs text-zinc-500">
                      {e.state === 'ready' ? 'from its archetype · ready to ask'
                        : [when(e.started), e.model, e.state === 'done' ? '' : e.state === 'error' ? 'stopped by an error' : 'stopped']
                          .filter(Boolean).join(' · ')}
                    </span>
                  </span>
                </button>
                <button onClick={() => onForget(e.id)} aria-label="Forget this exam" title="Forget this exam"
                  className="rounded-lg p-1.5 text-zinc-600 opacity-0 hover:bg-zinc-800 hover:text-zinc-300 focus:opacity-100 group-hover:opacity-100">
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

/** Which documents the exam asks about: all of them (*picked* null), or the ones ticked. */
function MaterialsPick({ materials, picked, setPicked }: {
  materials: ExamOverview['materials']
  picked: string[] | null
  setPicked: (p: string[] | null) => void
}) {
  const [query, setQuery] = useState('')
  const toggle = (file: string) => {
    const now = picked ?? materials.map((m) => m.file)
    const next = now.includes(file) ? now.filter((f) => f !== file) : [...now, file]
    setPicked(next.length === materials.length ? null : next)
  }
  const q = query.trim().toLowerCase()
  const shown = q ? materials.filter((m) => `${m.title} ${m.file}`.toLowerCase().includes(q)) : materials
  const link = 'font-medium text-violet-600 hover:underline disabled:text-zinc-600 disabled:no-underline dark:text-violet-400'
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h4 className="text-sm font-semibold text-zinc-200">Materials</h4>
        <span className="flex items-center gap-2 text-xs text-zinc-500">
          {picked === null ? materials.length : picked.length} of {materials.length} picked
          <button onClick={() => setPicked(null)} disabled={picked === null} className={link}>All</button>
          <button onClick={() => setPicked([])} disabled={picked?.length === 0} className={link}>None</button>
        </span>
      </div>
      {materials.length > MATERIALS_SEARCH && (
        <div className="relative mb-2">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Find a document" aria-label="Find a document"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 py-1.5 pl-8 pr-3 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-violet-500 focus:outline-none" />
        </div>
      )}
      <ul role="group" aria-label="Materials" className="max-h-64 divide-y divide-zinc-800 overflow-y-auto rounded-xl border border-zinc-800">
        {shown.map((m) => {
          const ticked = picked === null || picked.includes(m.file)
          return (
            <li key={m.file}>
              <button role="checkbox" aria-checked={ticked} onClick={() => toggle(m.file)}
                className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-zinc-800/50">
                <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border ${ticked
                  ? 'border-violet-500 bg-violet-500 text-white' : 'border-zinc-600'}`}>
                  {ticked && <Check className="h-3 w-3" />}
                </span>
                <FileText className="h-4 w-4 shrink-0 text-zinc-500" />
                <span className="min-w-0 flex-1 truncate text-sm text-zinc-200" title={m.file}>{m.title}</span>
                <span className="shrink-0 text-xs text-zinc-500">{m.sections ? plural(m.sections, 'section') : 'nothing to ask'}</span>
              </button>
            </li>
          )
        })}
        {shown.length === 0 && <li className="px-3 py-3 text-sm text-zinc-500">No document matches “{query.trim()}”.</li>}
      </ul>
      <span className="mt-1 block text-xs text-zinc-500">Questions come only from the ticked documents.</span>
    </div>
  )
}

// ── Pieces both views use ─────────────────────────────────────────────

/** A question's number, coloured by how it went (grey until graded; a spinner while it's graded). */
function NumberBadge({ n, verdict, pending = false }: { n: number; verdict?: Verdict; pending?: boolean }) {
  const v = verdict ? VERDICT[verdict] : null
  return (
    <span title={v?.label ?? (pending ? 'Being graded' : 'Not graded')}
      className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold tabular-nums ring-1 ${v
        ? v.badge : 'bg-zinc-800 text-zinc-400 ring-zinc-700'}`}>
      {pending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : n}
    </span>
  )
}

function KindTag({ kind }: { kind: ExamQuestion['kind'] }) {
  return (
    <span className={`rounded-md px-1.5 py-0.5 text-[11px] font-medium ${kind === 'situation'
      ? 'bg-sky-500/10 text-sky-700 dark:text-sky-300' : 'bg-zinc-800 text-zinc-400'}`}>
      {kind === 'situation' ? 'Situation' : 'Question'}
    </span>
  )
}

function Meta({ icon: Icon, children }: { icon: typeof Clock; children: ReactNode }) {
  return (
    <span className="inline-flex max-w-full items-center gap-1 rounded-md bg-zinc-800/70 px-2 py-0.5 text-[11px] text-zinc-400">
      <Icon className="h-3 w-3 shrink-0 text-zinc-500" /><span className="truncate">{children}</span>
    </span>
  )
}

/** An answer, as the chat shows it: rendered markdown, cut off softly past *max* px until it's opened. */
function Answer({ text, max }: { text: string; max: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const [long, setLong] = useState(false)
  const [open, setOpen] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const check = () => setLong(el.scrollHeight > max + 8)
    const ro = new ResizeObserver(check)
    ro.observe(el)
    return () => ro.disconnect()
  }, [max])
  return (
    <div>
      <div ref={ref} style={open ? undefined : { maxHeight: max }}
        className={`fd-file-markdown fd-exam-md ${open ? '' : 'overflow-hidden'} ${long && !open ? 'fd-fade-bottom' : ''}`}>
        <Markdown remarkPlugins={[remarkGfm]} components={MD}>{text}</Markdown>
      </div>
      {long && (
        <button onClick={() => setOpen((v) => !v)}
          className="mt-1.5 inline-flex items-center gap-1 text-xs font-medium text-violet-600 hover:text-violet-500 dark:text-violet-400 dark:hover:text-violet-300">
          {open ? 'Show less' : 'Show the whole answer'}
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      )}
    </div>
  )
}

// ── Running ───────────────────────────────────────────────────────────

function RunningPanel({ exam, busy, onStop }: { exam: Exam; busy: boolean; onStop: () => void }) {
  const at = PHASES.findIndex((p) => p.id === exam.phase)
  const pct = exam.steps ? Math.round((exam.step / exam.steps) * 100) : 0
  const questions = exam.questions ?? []
  const number = new Map(questions.map((q, i) => [q.id, i + 1]))
  const asked = questions.filter((q) => q.answer !== undefined || q.error)
  const now = exam.phase === 'asking' ? questions.find((q) => q.answer === undefined && !q.error) : undefined
  const lookups = asked.reduce((n, q) => n + (q.lookups?.length ?? 0), 0)
  const waiting = exam.waiting
  const note = !waiting ? ''
    : waiting.why === 'pacing' ? 'Pacing the calls, to keep well within the model’s limits…'
      : waiting.why === 'busy' ? `${waiting.model || 'The model'} is busy — waiting and trying again…`
        : waiting.why === 'waking' ? `Waking ${waiting.name || 'the agent'}…` : ''
  const headline = exam.phase === 'writing' ? 'Writing the questions'
    : exam.phase === 'asking' ? `Question ${Math.min(exam.step + 1, exam.steps)} of ${exam.steps}`
      : `Grading ${Math.min(exam.step + 1, exam.steps)} of ${exam.steps}`
  const sub = exam.phase === 'writing' ? 'From its documents — asked the way your users would ask them.'
    : exam.phase === 'asking' ? 'It answers in a private session, looking things up in its documents as it would for a user.'
      : 'Each answer is checked against the part of the document it came from.'

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-5 p-4 wide:p-6">
      <section className="relative overflow-hidden rounded-2xl border border-violet-500/20 bg-gradient-to-br from-violet-500/[0.12] via-transparent to-sky-500/[0.06] p-5">
        <div className="flex items-start gap-4">
          <div className="min-w-0 flex-1">
            <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-violet-700 dark:text-violet-300">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-400 opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-violet-500" />
              </span>
              Exam under way
            </p>
            <h4 className="mt-2 text-xl font-semibold tracking-tight text-zinc-100 wide:text-2xl">{headline}</h4>
            <p className="mt-1 text-sm text-zinc-400">{sub}</p>
          </div>
          <button onClick={onStop} disabled={busy}
            className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-60">
            <Square className="h-3 w-3" /> Stop
          </button>
        </div>

        <ol className="mt-5 grid grid-cols-3 gap-2" aria-label="Exam steps">
          {PHASES.map((p, i) => {
            const fill = i < at ? 100 : i === at ? Math.max(4, pct) : 0
            return (
              <li key={p.id} className="min-w-0">
                <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
                  <div className={`h-full rounded-full transition-[width] duration-700 ease-out ${i < at
                    ? 'bg-emerald-500' : 'bg-gradient-to-r from-violet-500 to-fuchsia-500'}`} style={{ width: `${fill}%` }} />
                </div>
                <p className={`mt-1.5 flex items-center gap-1 truncate text-[11px] ${i === at
                  ? 'font-medium text-zinc-200' : i < at ? 'text-zinc-400' : 'text-zinc-600'}`}>
                  {i < at ? <Check className="h-3 w-3 shrink-0 text-emerald-500" />
                    : i === at ? <Loader2 className="h-3 w-3 shrink-0 animate-spin text-violet-500" />
                      : <CircleDashed className="h-3 w-3 shrink-0" />}
                  {p.label}
                </p>
              </li>
            )
          })}
        </ol>

        <div className="mt-4 flex flex-wrap gap-1.5">
          <Meta icon={Clock}>{took(exam)}</Meta>
          <Meta icon={Zap}>{kTokens(exam.tokens)} tokens</Meta>
          {exam.model && <Meta icon={Cpu}>{exam.model}</Meta>}
          {lookups > 0 && <Meta icon={Search}>{plural(lookups, 'lookup')} so far</Meta>}
        </div>
        {note && <p className="mt-3 flex items-center gap-1.5 text-xs text-amber-700 dark:text-amber-400"><Timer className="h-3.5 w-3.5" />{note}</p>}
        <p className="mt-3 text-xs text-zinc-500">You can close this — the exam goes on, and the bell tells you when it's done.</p>
      </section>

      {exam.phase === 'writing' && (
        <ul className="flex flex-col gap-2.5" aria-hidden>
          {[0, 1, 2].map((i) => (
            <li key={i} className="rounded-2xl border border-zinc-800 p-4" style={{ opacity: 1 - i * 0.25 }}>
              <div className="fd-shimmer h-3.5 w-3/4 rounded" />
              <div className="fd-shimmer mt-2.5 h-3 w-1/2 rounded" />
            </li>
          ))}
        </ul>
      )}

      {now && (
        <div className="fd-rise rounded-2xl border border-violet-500/30 bg-violet-500/[0.04] p-4">
          <div className="flex items-start gap-3">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-violet-500/15 text-xs font-semibold tabular-nums text-violet-700 ring-1 ring-violet-500/30 dark:text-violet-300">
              {number.get(now.id)}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-violet-700 dark:text-violet-300">Asking now</p>
              <p className="mt-1 text-sm font-medium leading-snug text-zinc-100">{now.question}</p>
              <div className="mt-3 space-y-2" aria-hidden>
                <div className="fd-shimmer h-2.5 w-full rounded" />
                <div className="fd-shimmer h-2.5 w-5/6 rounded" />
                <div className="fd-shimmer h-2.5 w-2/3 rounded" />
              </div>
            </div>
          </div>
        </div>
      )}

      {asked.length > 0 && (
        <section>
          <h4 className="mb-2.5 flex items-baseline gap-2 text-sm font-semibold text-zinc-200">
            Answered so far <span className="text-xs font-normal tabular-nums text-zinc-500">{asked.length} of {questions.length}</span>
          </h4>
          <ul className="flex flex-col gap-2.5">
            {[...asked].reverse().map((q) => (
              <li key={q.id} className="fd-rise rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
                <div className="flex items-start gap-3">
                  <NumberBadge n={number.get(q.id) ?? 0} verdict={q.verdict} pending={exam.phase === 'grading' && !q.verdict && !q.error} />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium leading-snug text-zinc-100">{q.question}</p>
                    <div className="mt-2.5 rounded-xl border border-zinc-800 bg-zinc-950/40 px-3.5 py-3">
                      {q.error ? <p className="text-sm text-red-500">{q.error}</p> : <Answer text={q.answer ?? ''} max={132} />}
                    </div>
                    <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                      <KindTag kind={q.kind} />
                      {q.verdict && <span className={`rounded-md px-1.5 py-0.5 text-[11px] font-medium ${VERDICT[q.verdict].pill}`}>{VERDICT[q.verdict].label}</span>}
                      <Meta icon={Search}>
                        {q.lookups?.length ? `looked up ${plural(q.lookups.length, 'thing')}` : 'looked nothing up'}
                      </Meta>
                      {q.seconds !== undefined && <Meta icon={Clock}>{q.seconds} s</Meta>}
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}

// ── Results ───────────────────────────────────────────────────────────

/** How a score reads, and the colours it is shown in. */
function band(pct: number | null) {
  if (pct === null) return { word: 'Not graded', glow: 'from-zinc-500/10', ring: 'border-zinc-800', text: 'text-zinc-400' }
  if (pct >= 80) {
    return { word: pct >= 95 ? 'Excellent' : 'Strong', glow: 'from-emerald-500/[0.14]', ring: 'border-emerald-500/25',
      text: 'text-emerald-700 dark:text-emerald-400' }
  }
  if (pct >= 50) return { word: 'Getting there', glow: 'from-amber-500/[0.14]', ring: 'border-amber-500/25', text: 'text-amber-700 dark:text-amber-400' }
  return { word: 'Needs work', glow: 'from-red-500/[0.12]', ring: 'border-red-500/25', text: 'text-red-700 dark:text-red-400' }
}

function ResultsPanel({ exam, busy, err, onAgain, onResume, onNew, onTeach, onRead }: {
  exam: Exam
  busy: boolean
  err: string
  onAgain: () => void
  onResume: () => void
  onNew: () => void
  onTeach: (q: ExamQuestion) => Promise<void>
  onRead?: (docs: string[]) => void
}) {
  const [filter, setFilter] = useState<Filter>('all')
  const [openQ, setOpenQ] = useState<string | null>(null)
  const questions = useMemo(() => exam.questions ?? [], [exam.questions])
  const number = new Map(questions.map((q, i) => [q.id, i + 1]))
  const graded = questions.filter((q) => q.verdict)
  const s = exam.summary
  const unfinished = exam.state !== 'done'
  const shown = questions.filter((q) => (
    filter === 'all' ? true : filter === 'wrong' ? wentWrong(q) : filter === 'right' ? q.verdict === 'right' : q.cause === filter
  ))
  // The documents with unread parts it got wrong: one click reads them.
  const toRead = [...new Set(questions.filter((q) => wentWrong(q) && q.read === false && q.doc_name).map((q) => q.doc_name))]
  // The documents it was examined on, when they were picked: by title.
  const picks = (exam.docs ?? []).map((f) => questions.find((q) => q.doc === f)?.doc_title ?? f)
  const b = band(s?.pct ?? null)
  const tiles = s ? [
    { label: 'Questions', hint: 'asked straight out', score: s.kinds.ask },
    { label: 'Situations', hint: 'a rule decides what to do', score: s.kinds.situation },
    // What the setup read: only when reads were tracked for what was asked.
    ...(s.read.read.questions || s.read.unread.questions ? [
      { label: 'Parts it read', hint: 'rules and facts taken', score: s.read.read },
      { label: 'Parts never read', hint: 'only searchable', score: s.read.unread },
    ] : []),
  ] : []

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-5 p-4 wide:p-6">
      {exam.state === 'ready' && (
        <div className="flex flex-col gap-3 rounded-2xl border border-violet-500/30 bg-gradient-to-br from-violet-500/[0.1] to-transparent p-4 sm:flex-row sm:items-center">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/15 text-violet-600 dark:text-violet-300">
            <GraduationCap className="h-5 w-5" />
          </span>
          <p className="flex-1 text-sm text-zinc-300">
            {exam.notes?.[0] || 'Questions passed on with the archetype it was made from.'} Ask them to see how well it knows
            what it was given — {questions.length === 1 ? 'one question' : `${questions.length} questions`}.
          </p>
          <button onClick={onResume} disabled={busy}
            className="flex shrink-0 items-center justify-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-2 text-xs font-semibold text-white shadow-sm shadow-violet-600/30 hover:bg-violet-500 disabled:opacity-60">
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />} Ask them
          </button>
        </div>
      )}
      {unfinished && exam.state !== 'ready' && (
        <div className="flex flex-col gap-3 rounded-2xl border border-amber-500/30 bg-amber-500/5 p-4 sm:flex-row sm:items-center">
          <AlertTriangle className="h-5 w-5 shrink-0 text-amber-500" />
          <p className="flex-1 text-sm text-zinc-300">
            {exam.state === 'error' ? exam.error || 'The exam stopped.' : 'The exam was stopped.'}
            {questions.length > 0 && ` ${graded.length} of ${questions.length} questions are graded.`}
          </p>
          {questions.length > 0 && (
            <button onClick={onResume} disabled={busy}
              className="flex shrink-0 items-center justify-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-2 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-60">
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />} Continue
            </button>
          )}
        </div>
      )}
      {err && <p className="flex items-start gap-1.5 text-sm text-red-500"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{err}</p>}

      {s && s.questions > 0 && (
        <section className={`relative overflow-hidden rounded-2xl border ${b.ring} bg-gradient-to-br ${b.glow} via-transparent to-transparent p-5 wide:p-6`}>
          <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
            <ScoreRing pct={s.pct} size={120} />
            <div className="min-w-0 flex-1">
              <p className={`text-[11px] font-semibold uppercase tracking-wider ${b.text}`}>{b.word}</p>
              <h4 className="mt-1 text-2xl font-semibold tracking-tight text-zinc-100">
                {s.right} of {s.questions} right
              </h4>
              <div className="mt-2.5 flex flex-wrap gap-1.5">
                <Meta icon={GraduationCap}>{sizeLabel(exam.size)} exam{exam.again ? ', asked again' : ''}</Meta>
                {picks.length > 0 && (
                  <span title={picks.join('\n')} className="contents">
                    <Meta icon={FileText}>{picks.length === 1 ? picks[0] : `${picks.length} documents`}</Meta>
                  </span>
                )}
                {exam.focus && <Meta icon={Target}>{exam.focus}</Meta>}
                {exam.model && <Meta icon={Cpu}>{exam.model}</Meta>}
                <Meta icon={CalendarDays}>{when(exam.started)}</Meta>
                <Meta icon={Clock}>{took(exam)}</Meta>
                <Meta icon={Zap}>{kTokens(exam.tokens)} tokens</Meta>
              </div>
              <Spread questions={questions} />
            </div>
          </div>

          {s.changed && s.changed.compared > 0 && (
            <div className="mt-4 flex flex-wrap items-center gap-1.5 border-t border-zinc-800/80 pt-4 text-xs">
              <span className="mr-1 text-zinc-400">Since last time</span>
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 font-medium text-emerald-700 dark:text-emerald-400">
                <TrendingUp className="h-3 w-3" />{s.changed.better} better
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-red-500/10 px-2 py-0.5 font-medium text-red-700 dark:text-red-400">
                <TrendingDown className="h-3 w-3" />{s.changed.worse} worse
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-zinc-800 px-2 py-0.5 text-zinc-400">
                <Equal className="h-3 w-3" />{s.changed.compared - s.changed.better - s.changed.worse} the same
              </span>
              {s.questions < 20 && (
                <span className="w-full pt-1 text-[11px] text-zinc-500 sm:w-auto sm:pl-1 sm:pt-0">
                  With few questions, some change is chance: a model doesn't answer the same way twice.
                </span>
              )}
            </div>
          )}
          {exam.notes.length > 0 && (
            <p className="mt-3 flex items-start gap-1.5 text-xs text-zinc-400">
              <Info className="mt-px h-3.5 w-3.5 shrink-0 text-zinc-500" />{exam.notes.join(' ')}
            </p>
          )}

          <div className="mt-5 flex flex-wrap gap-2">
            <button onClick={onAgain} disabled={busy}
              title="The same questions again, with what it knows and the model it has now"
              className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-2 text-xs font-semibold text-white shadow-sm shadow-violet-600/30 hover:bg-violet-500 disabled:opacity-60">
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />} Ask the same again
            </button>
            <button onClick={onNew}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900/60 px-3.5 py-2 text-xs font-medium text-zinc-200 hover:bg-zinc-800 sm:hidden">
              <Play className="h-3.5 w-3.5" /> New exam
            </button>
            {toRead.length > 0 && onRead && (
              <button onClick={() => onRead(toRead)}
                title="Its rules and facts are taken from the parts it was never read"
                className="flex items-center gap-1.5 rounded-lg border border-violet-500/40 px-3.5 py-2 text-xs font-medium text-violet-700 hover:bg-violet-500/10 dark:text-violet-300">
                <BookOpen className="h-3.5 w-3.5" /> Read the parts it missed
              </button>
            )}
          </div>
        </section>
      )}

      {tiles.length > 0 && (
        <div className={`grid min-w-0 grid-cols-2 gap-3 ${tiles.length > 2 ? 'wide:grid-cols-4' : ''}`}>
          {tiles.map((t) => <Tile key={t.label} {...t} />)}
        </div>
      )}
      {s && s.docs.length > 1 && <DocScores docs={s.docs} />}

      {questions.length > 0 && (
        <section>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <div className="inline-flex gap-0.5 rounded-xl bg-zinc-800/60 p-1" role="tablist" aria-label="Show">
              <Segment on={filter === 'all'} onClick={() => setFilter('all')} label="All" count={questions.length} />
              <Segment on={filter === 'wrong'} onClick={() => setFilter('wrong')} label="Not right" count={questions.filter(wentWrong).length} />
              <Segment on={filter === 'right'} onClick={() => setFilter('right')} label="Right" count={questions.filter((q) => q.verdict === 'right').length} />
            </div>
            {s && CAUSES.filter((c) => s.causes[c] > 0).map((c) => (
              <button key={c} role="tab" aria-selected={filter === c} onClick={() => setFilter(c)}
                className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors ${filter === c
                  ? 'border-violet-500 bg-violet-500/15 text-violet-700 dark:text-violet-200'
                  : 'border-zinc-800 text-zinc-500 hover:text-zinc-300'}`}>
                {CAUSE[c].label}<span className="tabular-nums text-zinc-500">{s.causes[c]}</span>
              </button>
            ))}
          </div>
          {shown.length === 0 ? (
            <p className="rounded-2xl border border-dashed border-zinc-800 p-8 text-center text-sm text-zinc-500">Nothing here.</p>
          ) : (
            <ul className="flex flex-col gap-2.5">
              {shown.map((q) => (
                <QuestionCard key={q.id} n={number.get(q.id) ?? 0} q={q} open={openQ === q.id}
                  onToggle={() => setOpenQ(openQ === q.id ? null : q.id)}
                  onTeach={() => onTeach(q)} onRead={q.read === false && q.doc_name && onRead ? () => onRead([q.doc_name]) : undefined} />
              ))}
            </ul>
          )}
        </section>
      )}
    </div>
  )
}

/** An exam score as a ring, drawn in on arrival and coloured by how good it is. */
export function ScoreRing({ pct, size = 96 }: { pct: number | null; size?: number }) {
  const gradient = `fd-ring-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const [drawn, setDrawn] = useState(0)
  useEffect(() => {
    const f = requestAnimationFrame(() => setDrawn(pct ?? 0))
    return () => cancelAnimationFrame(f)
  }, [pct])
  const r = 34
  const c = 2 * Math.PI * r
  const [from, to] = pct === null ? ['#71717a', '#52525b'] : pct >= 80 ? ['#34d399', '#059669']
    : pct >= 50 ? ['#fbbf24', '#d97706'] : ['#f87171', '#dc2626']
  return (
    <div className="relative shrink-0 self-center" style={{ width: size, height: size }}>
      <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90" aria-hidden>
        <defs>
          <linearGradient id={gradient} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor={from} />
            <stop offset="1" stopColor={to} />
          </linearGradient>
        </defs>
        <circle cx="40" cy="40" r={r} className="fill-none stroke-zinc-800" strokeWidth="6" />
        <circle cx="40" cy="40" r={r} fill="none" stroke={`url(#${gradient})`} strokeWidth="6" strokeLinecap="round"
          strokeDasharray={`${(c * drawn) / 100} ${c}`} style={{ transition: 'stroke-dasharray 900ms cubic-bezier(.2,.8,.2,1)' }} />
      </svg>
      <span className={`absolute inset-0 flex items-center justify-center font-bold tabular-nums tracking-tight ${scoreTone(pct)}`}
        style={{ fontSize: Math.round(size * 0.25) }}>
        {pct === null ? '—' : <>{pct}<span className="font-semibold opacity-70" style={{ fontSize: Math.round(size * 0.13) }}>%</span></>}
      </span>
    </div>
  )
}

/** How the answers went, as one bar: right, partly, wrong, no answer. */
function Spread({ questions }: { questions: ExamQuestion[] }) {
  const parts = (['right', 'partly', 'wrong', 'no_answer'] as Verdict[])
    .map((v) => ({ v, n: questions.filter((q) => q.verdict === v).length })).filter((x) => x.n > 0)
  const total = parts.reduce((n, x) => n + x.n, 0)
  if (!total) return null
  return (
    <div className="mt-4">
      <div className="flex h-2 gap-0.5 overflow-hidden rounded-full">
        {parts.map((x) => (
          <div key={x.v} className={`${VERDICT[x.v].dot} first:rounded-l-full last:rounded-r-full`} style={{ width: `${(x.n / total) * 100}%` }} />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {parts.map((x) => (
          <span key={x.v} className="flex items-center gap-1.5 text-xs text-zinc-400">
            <span className={`h-2 w-2 rounded-full ${VERDICT[x.v].dot}`} />{VERDICT[x.v].label}
            <span className="tabular-nums text-zinc-500">{x.n}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

function Tile({ label, hint, score }: { label: string; hint: string; score: Score }) {
  const none = !score.questions
  return (
    <div className={`min-w-0 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 ${none ? 'opacity-60' : ''}`}>
      <p className="truncate text-[11px] font-semibold uppercase tracking-wide text-zinc-500">{label}</p>
      <p className={`mt-1.5 text-2xl font-semibold tabular-nums tracking-tight ${none ? 'text-zinc-500' : scoreTone(score.pct)}`}>
        {none ? '—' : `${score.pct}%`}
      </p>
      <div className="mt-2 h-1 overflow-hidden rounded-full bg-zinc-800">
        <div className={`h-full rounded-full ${scoreBar(score.pct)}`} style={{ width: `${none ? 0 : score.pct ?? 0}%` }} />
      </div>
      <p className="mt-2 text-[11px] text-zinc-400">{none ? 'none asked' : `${score.right} of ${score.questions} right`}</p>
      <p className="truncate text-[11px] text-zinc-500" title={hint}>{hint}</p>
    </div>
  )
}

function DocScores({ docs }: { docs: (Score & { doc: string; title: string })[] }) {
  return (
    <section className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
      <h5 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">By document</h5>
      <ul className="flex flex-col gap-2.5">
        {docs.map((d) => (
          <li key={d.doc} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 sm:grid-cols-[minmax(0,14rem)_minmax(0,1fr)_auto]">
            <span className="flex min-w-0 items-center gap-1.5 text-sm text-zinc-200">
              <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" /><span className="truncate">{d.title}</span>
            </span>
            <div className="order-last col-span-2 h-1.5 overflow-hidden rounded-full bg-zinc-800 sm:order-none sm:col-span-1">
              <div className={`h-full rounded-full ${scoreBar(d.pct)}`} style={{ width: `${d.pct ?? 0}%` }} />
            </div>
            <span className={`text-right text-sm font-semibold tabular-nums ${scoreTone(d.pct)}`}>
              {d.pct ?? '—'}{d.pct !== null && '%'}<span className="ml-1 text-xs font-normal text-zinc-500">of {d.questions}</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}

function Segment({ on, onClick, label, count }: { on: boolean; onClick: () => void; label: string; count: number }) {
  return (
    <button role="tab" aria-selected={on} onClick={onClick}
      className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${on
        ? 'bg-zinc-900 text-zinc-100 shadow-sm ring-1 ring-zinc-700' : 'text-zinc-400 hover:text-zinc-200'}`}>
      {label}<span className={`tabular-nums ${on ? 'text-zinc-400' : 'text-zinc-500'}`}>{count}</span>
    </button>
  )
}

function QuestionCard({ n, q, open, onToggle, onTeach, onRead }: {
  n: number
  q: ExamQuestion
  open: boolean
  onToggle: () => void
  onTeach: () => Promise<void>
  onRead?: () => void
}) {
  const v = q.verdict ? VERDICT[q.verdict] : null
  // Exams from before titles were tidied keep a PDF's emphasis ("**22**").
  const where = [q.doc_title, ...q.path, q.title].map((t) => t.replace(/\*\*|__/g, '').trim()).join(' › ')
  return (
    <li className={`overflow-hidden rounded-2xl border transition-colors ${open
      ? 'border-zinc-700 bg-zinc-900/60 shadow-lg shadow-black/5' : 'border-zinc-800 hover:border-zinc-700'}`}>
      <button onClick={onToggle} aria-expanded={open} className="flex w-full items-start gap-3 p-4 text-left">
        <NumberBadge n={n} verdict={q.verdict} />
        <span className="min-w-0 flex-1">
          <span className={`block text-sm font-medium leading-snug text-zinc-100 ${open ? '' : 'line-clamp-2'}`}>{q.question}</span>
          <span className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px]">
            <KindTag kind={q.kind} />
            {v && <span className={`rounded-md px-1.5 py-0.5 font-medium ${v.pill}`}>{v.label}</span>}
            {q.before && q.verdict && q.before !== q.verdict && (
              <span className="rounded-md bg-zinc-800 px-1.5 py-0.5 text-zinc-400">was {VERDICT[q.before].label.toLowerCase()}</span>
            )}
            {q.cause && <span className="text-zinc-400">{CAUSE[q.cause].label}</span>}
            {q.made_up && q.cause !== 'made_up' && <span className="text-red-600 dark:text-red-400">made things up</span>}
            {q.taught && <span className="inline-flex items-center gap-1 text-emerald-700 dark:text-emerald-400"><GraduationCap className="h-3 w-3" />taught</span>}
            <span className="inline-flex min-w-0 max-w-full items-center gap-1 text-zinc-500">
              <BookOpen className="h-3 w-3 shrink-0" /><span className="truncate">{where}</span>
            </span>
          </span>
        </span>
        <ChevronDown className={`mt-1 h-4 w-4 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && <QuestionDetail q={q} where={where} onTeach={onTeach} onRead={onRead} />}
    </li>
  )
}

const FOUND: Record<string, string> = {
  documents: 'elsewhere in its documents', lessons: 'in what it was taught', looked_up: 'in what it looked up',
}

// The details the grader found outside the question's passage, each checked against the rest of
// its documents: found elsewhere, or made up.
function Details({ details }: { details: NonNullable<ExamQuestion['details']> }) {
  return (
    <div className="mt-3 border-t border-zinc-800 pt-2.5">
      <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">Details beyond this passage</p>
      <ul className="flex flex-col gap-1.5">
        {details.map((d) => (
          <li key={d.text} className="flex items-start gap-2 text-xs">
            <span className={`mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full ${d.found
              ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400' : 'bg-red-500/15 text-red-600 dark:text-red-400'}`}>
              {d.found ? <Check className="h-2.5 w-2.5" /> : <X className="h-2.5 w-2.5" />}
            </span>
            <span className="text-zinc-300">
              {d.text}
              <span className="text-zinc-500"> — {d.found ? FOUND[d.found] ?? 'found' : 'found nowhere: made up'}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function QuestionDetail({ q, where, onTeach, onRead }: {
  q: ExamQuestion; where: string; onTeach: () => Promise<void>; onRead?: () => void
}) {
  const [teaching, setTeaching] = useState(false)
  const [teachErr, setTeachErr] = useState('')
  const missing = new Set(q.missing ?? [])
  const teach = async () => {
    setTeaching(true)
    setTeachErr('')
    try {
      await onTeach()
    } catch (e) {
      setTeachErr(message(e))
    } finally {
      setTeaching(false)
    }
  }
  const wrong = wentWrong(q)
  return (
    <div className="fd-rise flex flex-col gap-3 border-t border-zinc-800 p-4">
      <div className="grid gap-3 lg:grid-cols-2">
        <Panel icon={MessageSquareText} title="Its answer" aside={q.seconds !== undefined ? `${q.seconds} s` : ''}>
          {q.answer ? <Answer text={q.answer} max={320} /> : <p className="text-sm text-zinc-500">{q.error || 'No answer.'}</p>}
        </Panel>
        <Panel icon={BookOpen} title="What its documents say" tone="source">
          <p className="text-sm leading-relaxed text-zinc-200">{q.expected}</p>
          {q.points.length > 0 && (
            <ul className="mt-3 flex flex-col gap-1.5">
              {q.points.map((p) => {
                const miss = missing.has(p)
                return (
                  <li key={p} className="flex items-start gap-2 text-xs">
                    <span className={`mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full ${miss
                      ? 'bg-red-500/15 text-red-600 dark:text-red-400' : 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'}`}>
                      {miss ? <X className="h-2.5 w-2.5" /> : <Check className="h-2.5 w-2.5" />}
                    </span>
                    <span className={miss ? 'text-zinc-400' : 'text-zinc-300'}>{p}{miss && <span className="text-zinc-500"> — missing</span>}</span>
                  </li>
                )
              })}
            </ul>
          )}
          <p className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-zinc-800 pt-2.5 text-[11px] text-zinc-500">
            <span className="min-w-0 truncate">{where}</span>
            <span className={`shrink-0 rounded-md px-1.5 py-0.5 ${q.read === false ? 'bg-amber-500/10 text-amber-700 dark:text-amber-400' : 'bg-zinc-800 text-zinc-400'}`}>
              {q.read === true ? 'read in its setup' : q.read === false ? 'never read — only searchable' : 'read before reads were tracked'}
            </span>
          </p>
        </Panel>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel icon={Search} title="What it looked up" aside={q.lookups?.length ? plural(q.lookups.length, 'lookup') : ''}>
          <Lookups lookups={q.lookups ?? []} />
        </Panel>
        {(q.why || q.cause) && (
          <Panel icon={Lightbulb} title={q.verdict === 'right' ? 'How it was graded' : 'Why it went wrong'} tone={wrong ? 'wrong' : 'right'}>
            {q.why && <p className="text-sm leading-relaxed text-zinc-200">{q.why}</p>}
            {q.cause && <p className="mt-1.5 text-xs text-zinc-500">{CAUSE[q.cause].explain}</p>}
            {(q.details?.length ?? 0) > 0 && <Details details={q.details ?? []} />}
            {wrong && (
              <div className="mt-3 flex flex-wrap items-center gap-2">
                {q.taught ? (
                  <span className="flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400">
                    <Check className="h-3.5 w-3.5" />
                    Taught — a lesson from its next message; ask the same again to check
                  </span>
                ) : (
                  <button onClick={() => void teach()} disabled={teaching}
                    title="The answer becomes a lesson it's reminded of when a message is about it"
                    className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm shadow-violet-600/30 hover:bg-violet-500 disabled:opacity-60">
                    {teaching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <GraduationCap className="h-3.5 w-3.5" />} Teach it this
                  </button>
                )}
                {onRead && (
                  <button onClick={onRead}
                    title="Read this document's unread parts for rules and facts, in Way of working"
                    className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800">
                    <BookOpen className="h-3.5 w-3.5" /> Read this part
                  </button>
                )}
                {teachErr && <span className="text-xs text-red-500">{teachErr}</span>}
              </div>
            )}
          </Panel>
        )}
      </div>
    </div>
  )
}

const PANEL_TONE = {
  plain: 'border-zinc-800 bg-zinc-950/30',
  source: 'border-sky-500/20 bg-sky-500/[0.04]',
  right: 'border-emerald-500/20 bg-emerald-500/[0.04]',
  wrong: 'border-amber-500/25 bg-amber-500/[0.05]',
}

function Panel({ icon: Icon, title, aside = '', tone = 'plain', children }: {
  icon: typeof Clock; title: string; aside?: string; tone?: keyof typeof PANEL_TONE; children: ReactNode
}) {
  return (
    <section className={`min-w-0 rounded-xl border p-3.5 ${PANEL_TONE[tone]}`}>
      <h6 className="mb-2.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
        <Icon className="h-3.5 w-3.5" />{title}
        {aside && <span className="ml-auto font-normal normal-case tracking-normal tabular-nums">{aside}</span>}
      </h6>
      {children}
    </section>
  )
}

/** A lookup as a person reads it: what it searched for, or what it opened — and its icon. */
function lookupText(c: Lookup): { icon: typeof Clock; text: string } {
  const a = c.args
  const str = (k: string) => (typeof a[k] === 'string' ? (a[k] as string) : '')
  if (c.tool === 'knowledge') {
    const action = str('action')
    if (action === 'search') return { icon: Search, text: `Searched its documents for “${str('query')}”${str('doc') ? ` in ${str('doc')}` : ''}` }
    if (action === 'section') return { icon: FileText, text: `Opened ${str('doc')} › ${str('section')}` }
    if (action === 'outline') return { icon: ListTree, text: `Looked at the outline of ${str('doc')}` }
    if (action === 'list') return { icon: List, text: 'Listed its documents' }
  }
  const what = str('query') || str('pattern') || str('path') || str('action')
  return { icon: Wrench, text: `${c.tool}${what ? ` — ${what}` : ''}` }
}

function Lookups({ lookups }: { lookups: Lookup[] }) {
  if (!lookups.length) return <p className="text-sm text-zinc-500">Nothing — it answered without looking anything up.</p>
  return (
    <ol className="relative flex flex-col gap-2.5 before:absolute before:bottom-2 before:left-[9px] before:top-2 before:w-px before:bg-zinc-700">
      {lookups.map((c, i) => {
        const { icon: Icon, text } = lookupText(c)
        return (
          <li key={i} className="relative flex items-start gap-2.5 text-xs">
            <span className={`relative flex h-5 w-5 shrink-0 items-center justify-center rounded-full ring-1 ${c.ok
              ? 'bg-zinc-900 text-zinc-400 ring-zinc-700' : 'bg-zinc-900 text-red-500 ring-red-500/40'}`}>
              <Icon className="h-2.5 w-2.5" />
            </span>
            <span className={`min-w-0 pt-0.5 [overflow-wrap:anywhere] ${c.ok ? 'text-zinc-300' : 'text-zinc-500'}`}>
              {text}{!c.ok && <span className="ml-1.5 rounded bg-red-500/10 px-1 text-[10px] text-red-600 dark:text-red-400">failed</span>}
            </span>
          </li>
        )
      })}
    </ol>
  )
}
