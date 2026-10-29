import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertTriangle, ArrowLeft, BookOpen, Check, ChevronDown, ChevronRight, CircleDashed, GraduationCap, Lightbulb,
  Loader2, MessageSquareText, Play, RotateCcw, Search, Square, Trash2, X,
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
//   start    — what it is examined on, how many questions (Quick / Standard
//              / Deep) and what that costs, an optional focus, earlier exams;
//   running  — writing, asking, grading, live, the answers as they come;
//   results  — the score, by kind, by document and by whether the setup read
//              that part; each question with its answer, the answer its
//              document gives, what it looked up, why it went wrong — and the
//              fix: teach it the answer, or read the part it was never read.

const POLL_MS = 2000

const SIZES: { id: ExamSize; label: string; hint: string }[] = [
  { id: 'quick', label: 'Quick', hint: 'A first look' },
  { id: 'standard', label: 'Standard', hint: 'A fair picture' },
  { id: 'deep', label: 'Deep', hint: 'Into every corner' },
]

const VERDICT: Record<Verdict, { label: string; tone: string; dot: string }> = {
  right: { label: 'Right', tone: 'text-emerald-700 dark:text-emerald-400', dot: 'bg-emerald-500' },
  partly: { label: 'Partly right', tone: 'text-amber-700 dark:text-amber-400', dot: 'bg-amber-500' },
  wrong: { label: 'Wrong', tone: 'text-red-700 dark:text-red-400', dot: 'bg-red-500' },
  no_answer: { label: 'No answer', tone: 'text-zinc-400', dot: 'bg-zinc-500' },
}

const CAUSES: Cause[] = ['didnt_look', 'wrong_place', 'got_wrong', 'made_up']
const CAUSE: Record<Cause, { label: string; explain: string }> = {
  didnt_look: { label: "Didn't look", explain: 'It answered from memory, without looking in its documents.' },
  wrong_place: { label: 'Looked elsewhere', explain: 'It searched its documents but never reached this section.' },
  got_wrong: { label: 'Found it, got it wrong', explain: 'It reached the right section and still answered wrong.' },
  made_up: { label: 'Made things up', explain: "It stated details that aren't in your documents." },
}

const PHASES: { id: Exam['phase']; label: string }[] = [
  { id: 'writing', label: 'Writing questions' },
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
  const [overview, setOverview] = useState<ExamOverview | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const [exam, setExam] = useState<Exam | null>(null)
  const [view, setView] = useState<'start' | 'exam'>('start')
  const [size, setSize] = useState<ExamSize>('quick')
  const [focus, setFocus] = useState('')
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
  const start = () => act(() => agentExamApi.start(agent.slug, { size, focus: focus.trim() }))
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
  const taught = (qid: string, later: boolean) => setExam((e) => (e && e.questions
    ? { ...e, questions: e.questions.map((q) => (q.id === qid ? { ...q, taught: true, taught_later: later } : q)) } : e))

  const running = exam?.state === 'running'
  const subtitle = overview
    ? `${plural(overview.docs, 'document')} · answered by ${overview.model || 'its model'}`
    : 'Does it really know its documents?'

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label={`${agent.name}'s onboarding exam`}
        className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-900 shadow-2xl wide:h-[min(90vh,56rem)] wide:max-w-5xl wide:rounded-2xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
          {view === 'start' && exam ? (
            <button onClick={() => setView('exam')} aria-label="Back to the exam"
              className="-ml-2 rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
              <ArrowLeft className="h-5 w-5" />
            </button>
          ) : (
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
              <GraduationCap className="h-4 w-4" />
            </span>
          )}
          <div className="min-w-0">
            <h3 className="truncate text-base font-semibold text-zinc-100">{agent.name}'s onboarding exam</h3>
            <p className="truncate text-xs text-zinc-500">{subtitle}</p>
          </div>
          <div className="ml-auto flex shrink-0 items-center gap-1">
            {view === 'exam' && !running && overview && (
              <button onClick={() => { setErr(''); setView('start') }}
                className="hidden items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 sm:flex">
                <Play className="h-3.5 w-3.5" /> New exam
              </button>
            )}
            <button onClick={onClose} aria-label="Close" title="Close"
              className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
              <X className="h-5 w-5" />
            </button>
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
                busy={busy} err={err} onStart={start} onOpen={open} onForget={forget} />
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
                const r = await agentExamApi.teach(agent.slug, exam.id, q.id)
                taught(q.id, r.waiting)
              }}
              onRead={onReadDocs}
            />
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}

function Centered({ children }: { children: ReactNode }) {
  return <div className="flex h-full min-h-[16rem] flex-col items-center justify-center gap-3 p-6 text-center">{children}</div>
}

// ── Start ─────────────────────────────────────────────────────────────

function StartPanel({ overview, size, setSize, focus, setFocus, busy, err, onStart, onOpen, onForget }: {
  overview: ExamOverview
  size: ExamSize
  setSize: (s: ExamSize) => void
  focus: string
  setFocus: (f: string) => void
  busy: boolean
  err: string
  onStart: () => void
  onOpen: (id: string) => void
  onForget: (id: string) => void
}) {
  const nothing = overview.docs === 0 || overview.sections === 0
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
          <div>
            <div className="mb-2 flex items-baseline justify-between gap-2">
              <h4 className="text-sm font-semibold text-zinc-200">How many questions</h4>
              <span className="text-xs text-zinc-500">
                from {plural(overview.docs, 'document')} · {plural(overview.sections, 'section')} to ask about
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2" role="radiogroup" aria-label="Exam size">
              {SIZES.map((s) => {
                const n = Math.min(overview.sizes[s.id], overview.sections)
                const on = size === s.id
                return (
                  <button key={s.id} role="radio" aria-checked={on} onClick={() => setSize(s.id)}
                    className={`rounded-xl border p-3 text-left transition-colors ${on
                      ? 'border-violet-500 bg-violet-500/10 ring-1 ring-violet-500/40'
                      : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-800/50'}`}>
                    <span className="block text-sm font-semibold text-zinc-100">{s.label}</span>
                    <span className="mt-0.5 block text-2xl font-semibold tabular-nums text-zinc-100">{n}</span>
                    <span className="block text-[11px] text-zinc-500">questions · {s.hint}</span>
                    <span className="mt-2 block text-[11px] text-zinc-500">about {kTokens(overview.cost[s.id])} tokens</span>
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
            <button onClick={onStart} disabled={busy}
              className="flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-500 disabled:opacity-60">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />} Start the exam
            </button>
          </div>
        </>
      )}

      {overview.exams.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-zinc-200">Earlier exams</h4>
          <ul className="divide-y divide-zinc-800 overflow-hidden rounded-xl border border-zinc-800">
            {overview.exams.map((e) => (
              <li key={e.id} className="group flex items-center gap-3 px-3 py-2.5 hover:bg-zinc-800/40">
                <button onClick={() => onOpen(e.id)} className="flex min-w-0 flex-1 items-center gap-3 text-left">
                  <span className={`w-12 shrink-0 text-right text-lg font-semibold tabular-nums ${scoreTone(e.summary?.pct ?? null)}`}>
                    {e.summary?.pct != null ? `${e.summary.pct}%` : '—'}
                  </span>
                  <span className="min-w-0">
                    <span className="block truncate text-sm text-zinc-200">
                      {sizeLabel(e.size)}{e.again ? ' · asked again' : ''}{e.focus ? ` · ${e.focus}` : ''}
                    </span>
                    <span className="block truncate text-xs text-zinc-500">
                      {when(e.started)} · {e.model}{e.state !== 'done' ? ` · ${e.state === 'error' ? 'stopped by an error' : 'stopped'}` : ''}
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

// ── Running ───────────────────────────────────────────────────────────

function RunningPanel({ exam, busy, onStop }: { exam: Exam; busy: boolean; onStop: () => void }) {
  const at = PHASES.findIndex((p) => p.id === exam.phase)
  const pct = exam.steps ? Math.round((exam.step / exam.steps) * 100) : 0
  const asked = (exam.questions ?? []).filter((q) => q.answer !== undefined || q.error)
  const waiting = exam.waiting
  const note = !waiting ? ''
    : waiting.why === 'pacing' ? 'Pacing the calls, to keep well within the model’s limits…'
      : waiting.why === 'busy' ? `${waiting.model || 'The model'} is busy — waiting and trying again…`
        : waiting.why === 'waking' ? `Waking ${waiting.name || 'the agent'}…` : ''
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-5 p-4 wide:p-6">
      <div className="rounded-xl border border-zinc-800 p-4">
        <ol className="mb-4 flex flex-wrap items-center gap-2 text-xs" aria-label="Exam steps">
          {PHASES.map((p, i) => (
            <li key={p.id} className="flex items-center gap-2">
              {i > 0 && <span className="h-px w-5 bg-zinc-700" />}
              <span className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 ${i === at
                ? 'bg-violet-500/15 font-medium text-violet-700 dark:text-violet-200' : i < at ? 'text-zinc-400' : 'text-zinc-600'}`}>
                {i < at ? <Check className="h-3 w-3 text-emerald-500" /> : i === at ? <Loader2 className="h-3 w-3 animate-spin" /> : <CircleDashed className="h-3 w-3" />}
                {p.label}
              </span>
            </li>
          ))}
        </ol>
        <div className="flex items-baseline justify-between gap-3">
          <p className="text-sm text-zinc-200">
            {exam.phase === 'writing' ? 'Writing questions from its documents'
              : exam.phase === 'asking' ? `Asking question ${Math.min(exam.step + 1, exam.steps)} of ${exam.steps}`
                : 'Checking its answers against the documents'}
          </p>
          <span className="text-xs tabular-nums text-zinc-500">{took(exam)} · {kTokens(exam.tokens)} tokens</span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-800">
          <div className="h-full rounded-full bg-violet-500 transition-all" style={{ width: `${Math.max(3, pct)}%` }} />
        </div>
        {note && <p className="mt-2 text-xs text-zinc-500">{note}</p>}
        <div className="mt-4 flex items-center justify-between gap-3">
          <p className="text-xs text-zinc-500">You can close this — the exam goes on, and the bell tells you when it's done.</p>
          <button onClick={onStop} disabled={busy}
            className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-60">
            <Square className="h-3 w-3" /> Stop
          </button>
        </div>
      </div>

      {asked.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-zinc-200">Answered so far</h4>
          <ul className="flex flex-col gap-2">
            {[...asked].reverse().map((q) => (
              <li key={q.id} className="rounded-xl border border-zinc-800 p-3">
                <p className="text-sm text-zinc-200">{q.question}</p>
                <p className="mt-1 line-clamp-2 text-xs text-zinc-500">
                  {q.error ? q.error : q.answer}
                </p>
                <p className="mt-1.5 flex items-center gap-1.5 text-[11px] text-zinc-600">
                  {q.lookups && q.lookups.length > 0
                    ? <><Search className="h-3 w-3" /> looked up {plural(q.lookups.length, 'thing')}</>
                    : <>answered without looking anything up</>}
                  {q.seconds !== undefined && <> · {q.seconds} s</>}
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// ── Results ───────────────────────────────────────────────────────────

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
  const graded = questions.filter((q) => q.verdict)
  const s = exam.summary
  const unfinished = exam.state !== 'done'
  const shown = questions.filter((q) => (
    filter === 'all' ? true : filter === 'wrong' ? wentWrong(q) : filter === 'right' ? q.verdict === 'right' : q.cause === filter
  ))
  // The documents with unread parts it got wrong: one click reads them.
  const toRead = [...new Set(questions.filter((q) => wentWrong(q) && q.read === false && q.doc_name).map((q) => q.doc_name))]

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-5 p-4 wide:p-6">
      {unfinished && (
        <div className="flex flex-col gap-3 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 sm:flex-row sm:items-center">
          <AlertTriangle className="h-5 w-5 shrink-0 text-amber-500" />
          <p className="flex-1 text-sm text-zinc-300">
            {exam.state === 'error' ? exam.error || 'The exam stopped.' : 'The exam was stopped.'}
            {questions.length > 0 && ` ${graded.length} of ${questions.length} questions are graded.`}
          </p>
          {questions.length > 0 && (
            <button onClick={onResume} disabled={busy}
              className="flex shrink-0 items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-60">
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />} Continue
            </button>
          )}
        </div>
      )}
      {err && <p className="flex items-start gap-1.5 text-sm text-red-500"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{err}</p>}

      {s && s.questions > 0 && (
        <div className="flex flex-col gap-4 rounded-xl border border-zinc-800 p-4 sm:flex-row sm:items-center">
          <ScoreRing pct={s.pct} />
          <div className="min-w-0 flex-1">
            <p className="text-lg font-semibold text-zinc-100">
              {s.right} of {s.questions} right
              {s.questions - s.right > 0 && (
                <span className="text-sm font-normal text-zinc-500">
                  {' '}· {questions.filter((q) => q.verdict === 'partly').length} partly
                </span>
              )}
            </p>
            <p className="mt-0.5 text-xs text-zinc-500">
              {sizeLabel(exam.size)} exam{exam.again ? ', asked again' : ''}{exam.focus ? ` · focus: ${exam.focus}` : ''} · {exam.model} · {when(exam.started)} · {took(exam)} · {kTokens(exam.tokens)} tokens
            </p>
            {s.changed && s.changed.compared > 0 && (
              <p className="mt-1 text-xs text-zinc-400">
                Since last time: <span className="text-emerald-600 dark:text-emerald-400">{s.changed.better} better</span>
                {' · '}<span className="text-red-600 dark:text-red-400">{s.changed.worse} worse</span>
                {' · '}{s.changed.compared - s.changed.better - s.changed.worse} the same
                {s.questions < 20 && <span className="text-zinc-500"> — with few questions, some change is chance: a model doesn't answer the same way twice</span>}
              </p>
            )}
            {exam.notes.length > 0 && <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">{exam.notes.join(' ')}</p>}
            <div className="mt-3 flex flex-wrap gap-2">
              <button onClick={onAgain} disabled={busy}
                title="The same questions again, with what it knows and the model it has now"
                className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800 disabled:opacity-60">
                {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />} Ask the same again
              </button>
              <button onClick={onNew}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800 sm:hidden">
                <Play className="h-3.5 w-3.5" /> New exam
              </button>
              {toRead.length > 0 && onRead && (
                <button onClick={() => onRead(toRead)}
                  title="Its rules and facts are taken from the parts it was never read"
                  className="flex items-center gap-1.5 rounded-lg border border-violet-500/40 px-3 py-1.5 text-xs font-medium text-violet-700 hover:bg-violet-500/10 dark:text-violet-300">
                  <BookOpen className="h-3.5 w-3.5" /> Read the parts it missed
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {s && s.questions > 0 && (
        <div className="grid min-w-0 grid-cols-1 gap-3 sm:grid-cols-2">
          <Breakdown title="By kind of question" rows={[
            { label: 'Questions', hint: 'asked straight out', score: s.kinds.ask },
            { label: 'Situations', hint: 'a rule decides what to do', score: s.kinds.situation },
          ]} />
          <Breakdown title="By what the setup read" rows={[
            { label: 'Parts it was read', hint: 'rules and facts taken from them', score: s.read.read },
            { label: 'Parts never read', hint: 'only searchable', score: s.read.unread },
          ]} />
          {s.docs.length > 1 && (
            <div className="min-w-0 sm:col-span-2">
              <Breakdown title="By document" rows={s.docs.map((d) => ({ label: d.title, hint: '', score: d }))} />
            </div>
          )}
        </div>
      )}

      {questions.length > 0 && (
        <div>
          <div className="mb-3 flex flex-wrap items-center gap-1.5" role="tablist" aria-label="Show">
            <FilterChip on={filter === 'all'} onClick={() => setFilter('all')} label="All" count={questions.length} />
            <FilterChip on={filter === 'wrong'} onClick={() => setFilter('wrong')} label="Not right" count={questions.filter(wentWrong).length} />
            <FilterChip on={filter === 'right'} onClick={() => setFilter('right')} label="Right" count={questions.filter((q) => q.verdict === 'right').length} />
            {s && CAUSES.filter((c) => s.causes[c] > 0).map((c) => (
              <FilterChip key={c} on={filter === c} onClick={() => setFilter(c)} label={CAUSE[c].label} count={s.causes[c]} subtle />
            ))}
          </div>
          {shown.length === 0 ? (
            <p className="rounded-xl border border-zinc-800 p-6 text-center text-sm text-zinc-500">Nothing here.</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {shown.map((q) => (
                <QuestionCard key={q.id} q={q} open={openQ === q.id} onToggle={() => setOpenQ(openQ === q.id ? null : q.id)}
                  onTeach={() => onTeach(q)} onRead={q.read === false && q.doc_name && onRead ? () => onRead([q.doc_name]) : undefined} />
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

function ScoreRing({ pct }: { pct: number | null }) {
  const r = 34
  const c = 2 * Math.PI * r
  const tone = pct === null ? 'stroke-zinc-600' : pct >= 80 ? 'stroke-emerald-500' : pct >= 50 ? 'stroke-amber-500' : 'stroke-red-500'
  return (
    <div className="relative h-24 w-24 shrink-0 self-center">
      <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90" aria-hidden>
        <circle cx="40" cy="40" r={r} className="fill-none stroke-zinc-800" strokeWidth="7" />
        <circle cx="40" cy="40" r={r} className={`fill-none ${tone}`} strokeWidth="7" strokeLinecap="round"
          strokeDasharray={`${(c * (pct ?? 0)) / 100} ${c}`} />
      </svg>
      <span className={`absolute inset-0 flex items-center justify-center text-2xl font-bold tabular-nums ${scoreTone(pct)}`}>
        {pct === null ? '—' : `${pct}%`}
      </span>
    </div>
  )
}

function Breakdown({ title, rows }: { title: string; rows: { label: string; hint: string; score: Score }[] }) {
  return (
    <div className="min-w-0 rounded-xl border border-zinc-800 p-3">
      <h5 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">{title}</h5>
      <ul className="flex flex-col gap-2">
        {rows.map((r) => (
          <li key={r.label}>
            <div className="flex items-baseline justify-between gap-2">
              <span className="min-w-0 truncate text-sm text-zinc-200">
                {r.label}{r.hint && <span className="text-xs text-zinc-500"> · {r.hint}</span>}
              </span>
              <span className={`shrink-0 text-sm font-semibold tabular-nums ${scoreTone(r.score.pct)}`}>
                {r.score.questions ? `${r.score.pct}%` : '—'}
                <span className="ml-1 text-xs font-normal text-zinc-500">{r.score.questions ? `of ${r.score.questions}` : 'none asked'}</span>
              </span>
            </div>
            <div className="mt-1 h-1 overflow-hidden rounded-full bg-zinc-800">
              <div className={`h-full rounded-full ${scoreBar(r.score.pct)}`} style={{ width: `${r.score.pct ?? 0}%` }} />
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

function FilterChip({ on, onClick, label, count, subtle = false }: {
  on: boolean; onClick: () => void; label: string; count: number; subtle?: boolean
}) {
  return (
    <button role="tab" aria-selected={on} onClick={onClick}
      className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors ${on
        ? 'border-violet-500 bg-violet-500/15 text-violet-700 dark:text-violet-200'
        : subtle ? 'border-zinc-800 text-zinc-500 hover:text-zinc-300' : 'border-zinc-700 text-zinc-300 hover:bg-zinc-800'}`}>
      {label}<span className="tabular-nums text-zinc-500">{count}</span>
    </button>
  )
}

function QuestionCard({ q, open, onToggle, onTeach, onRead }: {
  q: ExamQuestion
  open: boolean
  onToggle: () => void
  onTeach: () => Promise<void>
  onRead?: () => void
}) {
  const v = q.verdict ? VERDICT[q.verdict] : null
  const where = [q.doc_title, ...q.path, q.title].join(' › ')
  return (
    <li className={`overflow-hidden rounded-xl border ${open ? 'border-zinc-700' : 'border-zinc-800'}`}>
      <button onClick={onToggle} aria-expanded={open} className="flex w-full items-start gap-3 p-3 text-left hover:bg-zinc-800/30">
        <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${v ? v.dot : 'bg-zinc-700'}`} title={v?.label ?? 'Not graded'} />
        <span className="min-w-0 flex-1">
          <span className={`block text-sm text-zinc-100 ${open ? '' : 'line-clamp-2'}`}>{q.question}</span>
          <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-zinc-500">
            <span className={`rounded px-1.5 py-0.5 ${q.kind === 'situation' ? 'bg-sky-500/10 text-sky-700 dark:text-sky-300' : 'bg-zinc-800 text-zinc-400'}`}>
              {q.kind === 'situation' ? 'Situation' : 'Question'}
            </span>
            {v && <span className={v.tone}>{v.label}</span>}
            {q.before && q.verdict && q.before !== q.verdict && (
              <span className="text-zinc-500">(was {VERDICT[q.before].label.toLowerCase()})</span>
            )}
            {q.cause && <span className="text-zinc-400">· {CAUSE[q.cause].label}</span>}
            {q.made_up && q.cause !== 'made_up' && <span className="text-red-600 dark:text-red-400">· made things up</span>}
            <span className="min-w-0 truncate">· {where}</span>
          </span>
        </span>
        {open ? <ChevronDown className="mt-0.5 h-4 w-4 shrink-0 text-zinc-500" /> : <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-zinc-500" />}
      </button>
      {open && <QuestionDetail q={q} where={where} onTeach={onTeach} onRead={onRead} />}
    </li>
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
  return (
    <div className="grid gap-4 border-t border-zinc-800 bg-zinc-950/40 p-4 lg:grid-cols-2">
      <Section icon={<MessageSquareText className="h-3.5 w-3.5" />} title={`Its answer${q.seconds !== undefined ? ` · ${q.seconds} s` : ''}`}>
        {q.answer ? (
          <div className="fd-file-markdown max-h-72 overflow-y-auto text-sm">
            <Markdown remarkPlugins={[remarkGfm]} components={MD}>{q.answer}</Markdown>
          </div>
        ) : <p className="text-sm text-zinc-500">{q.error || 'No answer.'}</p>}
      </Section>

      <Section icon={<BookOpen className="h-3.5 w-3.5" />} title="What its documents say">
        <p className="text-sm text-zinc-300">{q.expected}</p>
        <ul className="mt-2 flex flex-col gap-1">
          {q.points.map((p) => {
            const miss = missing.has(p)
            return (
              <li key={p} className="flex items-start gap-1.5 text-xs">
                {miss ? <X className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-500" /> : <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-500" />}
                <span className={miss ? 'text-zinc-400' : 'text-zinc-300'}>{p}</span>
              </li>
            )
          })}
        </ul>
        <p className="mt-2 text-[11px] text-zinc-500">
          From {where} · {q.read === true ? 'read in its setup' : q.read === false ? 'never read — only searchable' : 'read before reads were tracked'}
        </p>
      </Section>

      <Section icon={<Search className="h-3.5 w-3.5" />} title="What it looked up">
        <Lookups lookups={q.lookups ?? []} />
      </Section>

      {(q.why || q.cause) && (
        <Section icon={<Lightbulb className="h-3.5 w-3.5" />} title={q.verdict === 'right' ? 'Graded' : 'Why it went wrong'}>
          {q.why && <p className="text-sm text-zinc-300">{q.why}</p>}
          {q.cause && <p className="mt-1.5 text-xs text-zinc-500">{CAUSE[q.cause].explain}</p>}
          {wentWrong(q) && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {q.taught ? (
                <span className="flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400">
                  <Check className="h-3.5 w-3.5" />
                  {q.taught_later ? 'Taught — it gets it when it next starts' : 'Taught — ask the same again to check'}
                </span>
              ) : (
                <button onClick={() => void teach()} disabled={teaching}
                  title="The answer becomes a fact it's reminded of when a message is about it"
                  className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-60">
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
        </Section>
      )}
    </div>
  )
}

function Section({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <section className="min-w-0">
      <h6 className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">{icon}{title}</h6>
      {children}
    </section>
  )
}

/** A lookup as a person reads it: what it searched for, or what it opened. */
function lookupText(c: Lookup): string {
  const a = c.args
  const str = (k: string) => (typeof a[k] === 'string' ? (a[k] as string) : '')
  if (c.tool === 'knowledge') {
    const action = str('action')
    if (action === 'search') return `Searched its documents for “${str('query')}”${str('doc') ? ` in ${str('doc')}` : ''}`
    if (action === 'section') return `Opened ${str('doc')} › ${str('section')}`
    if (action === 'outline') return `Looked at the outline of ${str('doc')}`
    if (action === 'list') return 'Listed its documents'
  }
  const what = str('query') || str('pattern') || str('path') || str('action')
  return `${c.tool}${what ? ` — ${what}` : ''}`
}

function Lookups({ lookups }: { lookups: Lookup[] }) {
  if (!lookups.length) return <p className="text-sm text-zinc-500">Nothing — it answered without looking anything up.</p>
  return (
    <ol className="flex flex-col gap-1.5">
      {lookups.map((c, i) => (
        <li key={i} className="flex items-start gap-1.5 text-xs">
          <span className="mt-0.5 w-4 shrink-0 text-right tabular-nums text-zinc-600">{i + 1}.</span>
          <span className={c.ok ? 'text-zinc-300' : 'text-zinc-500 line-through decoration-zinc-600'}>{lookupText(c)}</span>
        </li>
      ))}
    </ol>
  )
}
