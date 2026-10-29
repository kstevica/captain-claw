import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  AlertTriangle, BookMarked, BookOpen, Check, CheckCircle2, FilePlus2, GraduationCap, Hand, Lightbulb, Loader2,
  Pencil, Pin, PinOff, Plus, RotateCcw, ScrollText, Search, Sparkles, Trash2, X,
} from 'lucide-react'
import { agentExamApi, type ExamOverview } from '../../services/agentExam'
import { agentKnowledgeApi, type KnowledgeOverview } from '../../services/agentKnowledge'
import { agentLessonsApi, type Lesson, type LessonKind, type LessonsChanged, type LessonsView } from '../../services/agentLessons'
import { useChatStore } from '../../stores/chatStore'
import { useProcessStore } from '../../stores/processStore'
import { KnowledgeBrowser } from '../library/KnowledgeBrowser'
import { AgentExamPanel, ScoreRing } from './AgentExamDialog'
import { stripOwnerBlock } from './agentSetup'

// ── An agent's onboarding, in one place ──────────────────────────────
//
// The Onboarding item in the kiosk's agent menu: what its owner gave it and
// taught it, and how well it knows it.
//
//   overview  — its last exam, how much of its material was read, its
//               lessons, and what to do next;
//   lessons   — every rule, thing to ask first and fact it was taught, with
//               where it came from and who taught it: add one, reword one,
//               keep it always in mind or bring it up when it matters, or
//               take it away (/fd/agent-lessons) — its instructions follow;
//   materials — its documents (the Knowledge browser), and more to give it;
//   exams     — its onboarding exams (AgentExamPanel).
//
// Giving it documents, or reading more of them, is Way of working: this
// dialog hands over to it (onSetup).

export type OnboardingTab = 'overview' | 'lessons' | 'materials' | 'exams'

const TABS: { id: OnboardingTab; label: string; icon: typeof BookOpen }[] = [
  { id: 'overview', label: 'Overview', icon: Sparkles },
  { id: 'lessons', label: 'Lessons', icon: ScrollText },
  { id: 'materials', label: 'Materials', icon: BookMarked },
  { id: 'exams', label: 'Exams', icon: GraduationCap },
]

const KIND: Record<LessonKind, { label: string; badge: string; icon: typeof BookOpen }> = {
  rule: { label: 'Rule', badge: 'bg-violet-500/10 text-violet-700 dark:text-violet-300', icon: ScrollText },
  ask_first: { label: 'Ask first', badge: 'bg-amber-500/10 text-amber-700 dark:text-amber-300', icon: Hand },
  fact: { label: 'Fact', badge: 'bg-sky-500/10 text-sky-700 dark:text-sky-300', icon: Lightbulb },
}
const TAUGHT: Record<string, string> = { setup: 'from a setup', exam: 'taught after an exam', you: 'added by you' }

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))
const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`
const ago = (epoch: number) => {
  const s = Math.max(0, Date.now() / 1000 - epoch)
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} days ago`
}

/** What a lesson change did to its instructions, mirrored into the chat's copy and sent to it. */
function syncInstructions(slug: string, res: LessonsChanged) {
  const procs = useProcessStore.getState()
  const had = procs.getFleetInstructions(slug)
  const next = res.stored ? res.instructions : [stripOwnerBlock(had), res.block].filter(Boolean).join('\n\n')
  if (next !== had) {
    void procs.setFleetInstructions(slug, next).then(() => useChatStore.getState().resendFleetIdentity(`proc-${slug}`))
  }
}

export function OnboardingDialog({ agent, initialTab = 'overview', phone = false, onClose, onSetup }: {
  agent: { slug: string; name: string }
  initialTab?: OnboardingTab
  phone?: boolean
  onClose: () => void
  /** Open its Way of working — to give it documents, or (with their setup names) to read ones it keeps. */
  onSetup: (readDocs?: string[]) => void
}) {
  const [tab, setTab] = useState<OnboardingTab>(initialTab)
  const [knowledge, setKnowledge] = useState<KnowledgeOverview | null>(null)
  const [lessons, setLessons] = useState<LessonsView | null>(null)
  const [exams, setExams] = useState<ExamOverview | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const [tick, setTick] = useState(0)

  // What the overview and the header say — again each time the overview is opened.
  useEffect(() => {
    if (tab !== 'overview' && knowledge && lessons && exams) return
    let live = true
    Promise.all([agentKnowledgeApi.overview(agent.slug), agentLessonsApi.list(agent.slug), agentExamApi.overview(agent.slug)])
      .then(([k, l, e]) => { if (live) { setKnowledge(k); setLessons(l); setExams(e); setLoadErr('') } })
      .catch((e) => { if (live) setLoadErr(message(e)) })
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent.slug, tab, tick])

  const last = exams ? (exams.running ?? exams.exams.find((e) => e.state === 'done') ?? null) : null
  const subtitle = !knowledge || !lessons ? 'What it was given and taught, and how well it knows it'
    : [plural(knowledge.totals.documents, 'document'),
      plural(lessons.lessons.length, 'lesson'),
      last?.summary?.pct != null ? `last exam ${last.summary.pct}%` : 'no exam yet'].join(' · ')

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label={`${agent.name}'s onboarding`}
        className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-900 shadow-2xl wide:h-[min(92vh,60rem)] wide:max-w-6xl wide:rounded-2xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center gap-3 px-4 pt-3 wide:px-5">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
            <GraduationCap className="h-[18px] w-[18px]" />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-base font-semibold text-zinc-100">{agent.name}'s onboarding</h3>
            <p className="truncate text-xs text-zinc-500">{subtitle}</p>
          </div>
          <button onClick={onClose} aria-label="Close" title="Close" className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-5 w-5" />
          </button>
        </div>
        <div role="tablist" aria-label="Onboarding" className="flex shrink-0 gap-1 overflow-x-auto border-b border-zinc-800 px-3 pt-2 wide:px-4">
          {TABS.map((t) => {
            const on = tab === t.id
            const count = t.id === 'lessons' ? lessons?.lessons.length : t.id === 'materials' ? knowledge?.totals.documents
              : t.id === 'exams' ? exams?.exams.length : undefined
            return (
              <button key={t.id} role="tab" aria-selected={on} onClick={() => setTab(t.id)}
                className={`-mb-px flex shrink-0 items-center gap-1.5 border-b-2 px-3 py-2 text-sm transition-colors ${on
                  ? 'border-violet-500 font-medium text-zinc-100' : 'border-transparent text-zinc-500 hover:text-zinc-300'}`}>
                <t.icon className="h-4 w-4" />{t.label}
                {count ? <span className="rounded-full bg-zinc-800 px-1.5 text-[10px] tabular-nums text-zinc-400">{count}</span> : null}
              </button>
            )
          })}
        </div>

        <div className="min-h-0 flex-1 overflow-hidden">
          {tab === 'overview' && (
            <div className="h-full overflow-y-auto">
              {loadErr ? (
                <Centered>
                  <AlertTriangle className="h-6 w-6 text-amber-500" />
                  <p className="max-w-sm text-sm text-zinc-400">{loadErr}</p>
                  <button onClick={() => setTick((n) => n + 1)}
                    className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800">Try again</button>
                </Centered>
              ) : !knowledge || !lessons || !exams ? (
                <Centered><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></Centered>
              ) : (
                <Overview name={agent.name} knowledge={knowledge} lessons={lessons} exams={exams}
                  onTab={setTab} onSetup={onSetup} />
              )}
            </div>
          )}
          {tab === 'lessons' && (
            <div className="h-full overflow-y-auto">
              <LessonsPanel slug={agent.slug} name={agent.name} initial={lessons} onChanged={setLessons} />
            </div>
          )}
          {tab === 'materials' && (
            <div className="flex h-full flex-col">
              <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-2 wide:px-5">
                <p className="min-w-0 flex-1 truncate text-xs text-zinc-500">
                  The documents it keeps, whole: it searches them when it answers. Reading one takes rules and facts from it.
                </p>
                <button onClick={() => onSetup()}
                  className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800">
                  <FilePlus2 className="h-3.5 w-3.5" /> Add documents
                </button>
              </div>
              <div className="min-h-0 flex-1">
                <KnowledgeBrowser agent={agent} phone={phone} embedded onClose={onClose} onSetup={() => onSetup()} />
              </div>
            </div>
          )}
          {tab === 'exams' && <AgentExamPanel agent={agent} onReadDocs={onSetup} />}
        </div>
      </div>
    </div>,
    document.body,
  )
}

function Centered({ children }: { children: ReactNode }) {
  return <div className="flex h-full min-h-[16rem] flex-col items-center justify-center gap-3 p-6 text-center">{children}</div>
}

// ── Overview ──────────────────────────────────────────────────────────

interface Step { key: string; icon: typeof BookOpen; text: string; action: string; go: () => void }

function Overview({ name, knowledge, lessons, exams, onTab, onSetup }: {
  name: string
  knowledge: KnowledgeOverview
  lessons: LessonsView
  exams: ExamOverview
  onTab: (t: OnboardingTab) => void
  onSetup: (readDocs?: string[]) => void
}) {
  const docs = knowledge.documents
  const unread = docs.filter((d) => d.from_setup && (d.read.state === 'none' || d.read.state === 'partial'))
  const running = exams.running
  const last = exams.exams.find((e) => e.state === 'done') ?? null
  const s = last?.summary ?? null
  const wrong = s ? s.questions - s.right : 0
  const lastTaught = Math.max(0, ...lessons.lessons.map((x) => x.added || 0))
  const { totals } = knowledge
  const readPct = totals.tracked ? Math.round((totals.read / Math.max(1, totals.sections)) * 100) : null
  const rules = lessons.counts.rule + lessons.counts.ask_first

  const steps: Step[] = []
  if (!docs.length) {
    steps.push({ key: 'docs', icon: FilePlus2, text: `Give ${name} the documents it works from — policies, price lists, procedures.`,
      action: 'Add documents', go: () => onSetup() })
  }
  if (unread.length) {
    steps.push({ key: 'read', icon: BookOpen, action: 'Read them',
      text: `${plural(unread.length, 'document')} ${unread.length === 1 ? 'is' : 'are'} only kept, not read for rules and facts: ${unread.slice(0, 3).map((d) => d.title).join(', ')}${unread.length > 3 ? '…' : ''}.`,
      go: () => onSetup(unread.map((d) => d.doc)) })
  }
  if (docs.length && !last && !running) {
    steps.push({ key: 'exam', icon: GraduationCap, text: 'Run a first exam, to see how well it knows its documents.',
      action: 'Start an exam', go: () => onTab('exams') })
  }
  if (last && wrong > 0) {
    steps.push({ key: 'wrong', icon: Lightbulb, action: 'See what it missed', go: () => onTab('exams'),
      text: `${plural(wrong, 'answer')} in its last exam ${wrong === 1 ? "wasn't" : "weren't"} right — teach it the answer, or read the part it missed.` })
  }
  if (last && last.finished && lastTaught > last.finished) {
    steps.push({ key: 'again', icon: RotateCcw, action: 'Ask the same again', go: () => onTab('exams'),
      text: 'It was taught more since its last exam: ask the same questions again to see what changed.' })
  }
  if (rules > 0 && lessons.counts.always === 0) {
    steps.push({ key: 'always', icon: Pin, action: 'Open lessons', go: () => onTab('lessons'),
      text: 'None of its rules is always in mind. Mark those that apply to every conversation — language, tone, what never to do.' })
  }

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-5 p-4 wide:p-6">
      <div className="grid gap-3 md:grid-cols-3">
        <Card title="Exam" onOpen={() => onTab('exams')} open={last || running ? 'Open exams' : 'Start an exam'}>
          {running ? (
            <p className="flex items-center gap-2 text-sm text-zinc-300"><Loader2 className="h-4 w-4 animate-spin text-violet-500" />An exam is under way…</p>
          ) : s ? (
            <div className="flex items-center gap-3">
              <div className="scale-75 -m-3"><ScoreRing pct={s.pct} /></div>
              <div className="min-w-0">
                <p className="text-sm font-medium text-zinc-100">{s.right} of {s.questions} right</p>
                <p className="text-xs text-zinc-500">{last?.model} · {last ? ago(last.finished || last.started) : ''}</p>
                {s.changed && s.changed.compared > 0 && (
                  <p className="text-xs text-zinc-500">
                    since before: <span className="text-emerald-600 dark:text-emerald-400">{s.changed.better} better</span>,{' '}
                    <span className="text-red-600 dark:text-red-400">{s.changed.worse} worse</span>
                  </p>
                )}
              </div>
            </div>
          ) : (
            <p className="text-sm text-zinc-500">No exam yet. One asks it questions from its documents, as your users would.</p>
          )}
        </Card>
        <Card title="Materials" onOpen={() => onTab('materials')} open="Open materials">
          <p className="text-sm font-medium text-zinc-100">{plural(totals.documents, 'document')}</p>
          {readPct !== null ? (
            <>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-800">
                <div className="h-full rounded-full bg-sky-500" style={{ width: `${Math.max(2, readPct)}%` }} />
              </div>
              <p className="mt-1.5 text-xs text-zinc-500">
                {totals.read.toLocaleString()} of {plural(totals.sections, 'section')} read for rules and facts — all of it searchable
              </p>
            </>
          ) : <p className="text-xs text-zinc-500">{docs.length ? 'All of it searchable.' : 'None yet.'}</p>}
        </Card>
        <Card title="Lessons" onOpen={() => onTab('lessons')} open="Open lessons">
          <p className="text-sm font-medium text-zinc-100">{plural(lessons.lessons.length, 'lesson')}</p>
          <p className="mt-1 text-xs text-zinc-500">
            {[lessons.counts.rule && plural(lessons.counts.rule, 'rule'), lessons.counts.ask_first && `${lessons.counts.ask_first} to ask first`,
              lessons.counts.fact && plural(lessons.counts.fact, 'fact')].filter(Boolean).join(' · ') || 'None yet.'}
          </p>
          {lessons.lessons.length > 0 && (
            <p className="mt-1 text-xs text-zinc-500">
              <Pin className="mr-1 inline h-3 w-3" />{lessons.counts.always} always in mind; the rest come up when they matter
            </p>
          )}
        </Card>
      </div>

      <div>
        <h4 className="mb-2 text-sm font-semibold text-zinc-200">What to do next</h4>
        {steps.length === 0 ? (
          <p className="flex items-center gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4 text-sm text-zinc-300">
            <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-500" />
            Nothing waiting. Ask the same exam again after you give it more, to see that it still knows it all.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {steps.map((st) => (
              <li key={st.key} className="flex flex-col gap-3 rounded-xl border border-zinc-800 p-3 sm:flex-row sm:items-center">
                <st.icon className="hidden h-5 w-5 shrink-0 text-violet-500 sm:block dark:text-violet-400" />
                <p className="min-w-0 flex-1 text-sm text-zinc-300">{st.text}</p>
                <button onClick={st.go}
                  className="shrink-0 self-start rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800 sm:self-auto">
                  {st.action}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

function Card({ title, open, onOpen, children }: { title: string; open: string; onOpen: () => void; children: ReactNode }) {
  return (
    <section className="flex min-w-0 flex-col rounded-xl border border-zinc-800 p-4">
      <h5 className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-500">{title}</h5>
      <div className="min-w-0 flex-1">{children}</div>
      <button onClick={onOpen} className="mt-3 self-start text-xs font-medium text-violet-600 hover:underline dark:text-violet-400">{open} →</button>
    </section>
  )
}

// ── Lessons ───────────────────────────────────────────────────────────

type Filter = 'all' | 'always' | LessonKind

function LessonsPanel({ slug, name, initial, onChanged }: {
  slug: string
  name: string
  initial: LessonsView | null
  onChanged: (v: LessonsView) => void
}) {
  const [data, setData] = useState<LessonsView | null>(initial)
  const [err, setErr] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [query, setQuery] = useState('')
  const [adding, setAdding] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    agentLessonsApi.list(slug).then((v) => { if (live) { setData(v); onChanged(v) } }).catch((e) => { if (live) setErr(message(e)) })
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slug])

  const change = async (key: string, fn: () => Promise<LessonsChanged>): Promise<boolean> => {
    setBusy(key)
    setErr('')
    try {
      const res = await fn()
      syncInstructions(slug, res)
      setData(res)
      onChanged(res)
      return true
    } catch (e) {
      setErr(message(e))
      return false
    } finally {
      setBusy(null)
    }
  }

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    return (data?.lessons ?? []).filter((x) => (
      (filter === 'all' || (filter === 'always' ? x.always : x.kind === filter))
      && (!q || x.text.toLowerCase().includes(q) || x.source.doc.toLowerCase().includes(q) || x.source.section.toLowerCase().includes(q))
    ))
  }, [data, filter, query])

  if (!data) {
    return err ? <Centered><p className="text-sm text-red-500">{err}</p></Centered>
      : <Centered><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></Centered>
  }
  const missing = new Set(data.not_in_block)
  const chips: { id: Filter; label: string; n: number }[] = [
    { id: 'all', label: 'All', n: data.lessons.length },
    { id: 'always', label: 'Always in mind', n: data.counts.always },
    { id: 'rule', label: 'Rules', n: data.counts.rule },
    { id: 'ask_first', label: 'Ask first', n: data.counts.ask_first },
    { id: 'fact', label: 'Facts', n: data.counts.fact },
  ]

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4 p-4 wide:p-6">
      <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4 text-sm leading-relaxed text-zinc-300">
        What you and its setups taught {name}. <Pin className="inline h-3.5 w-3.5 text-violet-500" /> <b className="font-medium">Always in mind</b>{' '}
        ones are in its instructions, for every conversation; the rest reach it with the messages they bear on — found by what a
        message says and what it means. A change counts from its next message.
      </div>

      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <div className="flex min-w-0 flex-1 items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950 px-2.5">
          <Search className="h-4 w-4 shrink-0 text-zinc-500" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Find a lesson, or its document…"
            className="min-w-0 flex-1 bg-transparent py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none" />
        </div>
        <button onClick={() => setAdding((v) => !v)}
          className="flex shrink-0 items-center justify-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-violet-500">
          <Plus className="h-4 w-4" /> Add a lesson
        </button>
      </div>

      <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Show">
        {chips.map((c) => (
          <button key={c.id} role="tab" aria-selected={filter === c.id} onClick={() => setFilter(c.id)}
            className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors ${filter === c.id
              ? 'border-violet-500 bg-violet-500/15 text-violet-700 dark:text-violet-200' : 'border-zinc-700 text-zinc-300 hover:bg-zinc-800'}`}>
            {c.label}<span className="tabular-nums text-zinc-500">{c.n}</span>
          </button>
        ))}
      </div>

      {err && <p className="flex items-start gap-1.5 text-sm text-red-500"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{err}</p>}

      {adding && (
        <AddLesson busy={busy === 'add'} onCancel={() => setAdding(false)}
          onAdd={async (kind, text, always) => {
            if (await change('add', () => agentLessonsApi.add(slug, { kind, text, always }))) setAdding(false)
          }} />
      )}

      {data.lessons.length === 0 ? (
        <p className="rounded-xl border border-zinc-800 p-6 text-center text-sm text-zinc-500">
          No lessons yet. Way of working reads your instructions and documents into them; you can also add one here.
        </p>
      ) : shown.length === 0 ? (
        <p className="rounded-xl border border-zinc-800 p-6 text-center text-sm text-zinc-500">Nothing here.</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {shown.map((x) => (
            <LessonRow key={x.id} lesson={x} busy={busy === x.id} noRoom={missing.has(x.id)}
              onAlways={(always) => void change(x.id, () => agentLessonsApi.change(slug, x.id, { always }))}
              onSave={(text) => change(x.id, () => agentLessonsApi.change(slug, x.id, { text }))}
              onRemove={() => void change(x.id, () => agentLessonsApi.remove(slug, x.id))} />
          ))}
        </ul>
      )}
    </div>
  )
}

function AddLesson({ busy, onAdd, onCancel }: {
  busy: boolean
  onAdd: (kind: LessonKind, text: string, always: boolean) => void
  onCancel: () => void
}) {
  const [kind, setKind] = useState<LessonKind>('rule')
  const [text, setText] = useState('')
  const [always, setAlways] = useState(false)
  const hint = kind === 'rule' ? 'e.g. Quotes for business customers are valid for 30 days.'
    : kind === 'ask_first' ? 'e.g. Ask me before promising a delivery date.' : 'e.g. Our warehouse closes at 15:00 on Fridays.'
  return (
    <div className="rounded-xl border border-violet-500/40 bg-zinc-900 p-3">
      <div className="mb-2 flex flex-wrap gap-1" role="radiogroup" aria-label="Kind of lesson">
        {(Object.keys(KIND) as LessonKind[]).map((k) => (
          <button key={k} role="radio" aria-checked={kind === k} onClick={() => setKind(k)}
            className={`rounded-md px-2.5 py-1 text-xs ${kind === k ? KIND[k].badge + ' font-medium' : 'text-zinc-400 hover:bg-zinc-800'}`}>
            {KIND[k].label}
          </button>
        ))}
      </div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} placeholder={hint} autoFocus maxLength={400}
        className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-violet-500 focus:outline-none" />
      <div className="mt-2 flex flex-wrap items-center gap-2">
        {kind !== 'fact' && (
          <label className="flex items-center gap-1.5 text-xs text-zinc-400">
            <input type="checkbox" checked={always} onChange={(e) => setAlways(e.target.checked)} className="h-3.5 w-3.5 accent-violet-600" />
            Always in mind — it applies to every conversation
          </label>
        )}
        <div className="ml-auto flex gap-2">
          <button onClick={onCancel} className="rounded-lg px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">Cancel</button>
          <button onClick={() => onAdd(kind, text, kind !== 'fact' && always)} disabled={busy || !text.trim()}
            className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-50">
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />} Teach it
          </button>
        </div>
      </div>
    </div>
  )
}

function LessonRow({ lesson: x, busy, noRoom, onAlways, onSave, onRemove }: {
  lesson: Lesson
  busy: boolean
  noRoom: boolean
  onAlways: (always: boolean) => void
  onSave: (text: string) => Promise<boolean>
  onRemove: () => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(x.text)
  const [confirm, setConfirm] = useState(false)
  const k = KIND[x.kind]
  const where = x.source.doc ? `${x.source.doc}${x.source.section ? ` — ${x.source.section}` : ''}` : ''
  return (
    <li className={`rounded-xl border px-3 py-2.5 ${x.always ? 'border-violet-500/30 bg-violet-500/[0.04]' : 'border-zinc-800'}`}>
      <div className="flex items-start gap-2.5">
        <span className={`mt-0.5 inline-flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${k.badge}`}>
          <k.icon className="h-3 w-3" />{k.label}
        </span>
        <div className="min-w-0 flex-1">
          {editing ? (
            <div>
              <textarea value={draft} onChange={(e) => setDraft(e.target.value)} rows={2} autoFocus maxLength={400}
                className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-100 focus:border-violet-500 focus:outline-none" />
              <div className="mt-1.5 flex gap-2">
                <button onClick={async () => { if (await onSave(draft)) setEditing(false) }} disabled={busy || !draft.trim()}
                  className="flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-50">
                  {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />} Save
                </button>
                <button onClick={() => { setDraft(x.text); setEditing(false) }} className="rounded-md px-2.5 py-1 text-xs text-zinc-400 hover:bg-zinc-800">Cancel</button>
              </div>
            </div>
          ) : (
            <p className="text-sm text-zinc-200">{x.text}</p>
          )}
          <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-zinc-500">
            {where && <span className="inline-flex min-w-0 items-center gap-1"><BookOpen className="h-3 w-3 shrink-0" /><span className="truncate">{where}</span></span>}
            <span>{TAUGHT[x.taught_by] ?? x.taught_by}{x.added ? ` · ${ago(x.added)}` : ''}</span>
            {noRoom && <span className="text-amber-600 dark:text-amber-400">no room in its instructions — it comes up when it matters</span>}
          </p>
        </div>
        {!editing && (
          <div className="flex shrink-0 items-center gap-0.5">
            {x.kind !== 'fact' && (
              <button onClick={() => onAlways(!x.always)} disabled={busy} aria-pressed={x.always}
                title={x.always ? 'Always in mind — click to bring it up only when it matters' : 'When it matters — click to keep it always in mind'}
                className={`rounded-md p-1.5 transition-colors disabled:opacity-50 ${x.always
                  ? 'text-violet-600 hover:bg-violet-500/10 dark:text-violet-300' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'}`}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : x.always ? <Pin className="h-4 w-4" /> : <PinOff className="h-4 w-4" />}
              </button>
            )}
            <button onClick={() => { setDraft(x.text); setEditing(true) }} disabled={busy} title="Reword it" aria-label="Reword it"
              className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-50">
              <Pencil className="h-4 w-4" />
            </button>
            {confirm ? (
              <span className="flex items-center gap-1 pl-1 text-xs">
                <button onClick={() => { setConfirm(false); onRemove() }} className="rounded-md bg-red-600 px-2 py-1 font-medium text-white hover:bg-red-500">Remove</button>
                <button onClick={() => setConfirm(false)} className="rounded-md px-2 py-1 text-zinc-400 hover:bg-zinc-800">Keep</button>
              </span>
            ) : (
              <button onClick={() => setConfirm(true)} disabled={busy} title="Take it away" aria-label="Take it away"
                className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-red-500 disabled:opacity-50">
                <Trash2 className="h-4 w-4" />
              </button>
            )}
          </div>
        )}
      </div>
    </li>
  )
}
