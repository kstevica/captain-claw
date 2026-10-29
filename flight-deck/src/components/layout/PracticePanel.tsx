import { useEffect, useState } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertTriangle, BookOpen, Check, ChevronDown, ChevronRight, Loader2, MessageSquareText, Play, RotateCcw, Search,
  Sparkles, ThumbsUp, Trash2, Wrench,
} from 'lucide-react'
import { agentPracticeApi, type PracticeAttempt, type PracticeSuggestion, type PracticeTask } from '../../services/agentPractice'
import { LessonDraftForm } from './LessonDraft'

// ── Practice: try it on real tasks, and correct it ───────────────────
//
// The Onboarding page's Practice tab (/fd/agent-practice/{slug}). The owner
// gives the agent a task the way a user would — or takes one written from its
// documents, with the answer they give — and it answers on its private exam
// lane: nothing is sent, nothing kept. Good, or "Fix it": what it should have
// done becomes a lesson (LessonDraftForm), and it tries again.

const POLL_MS = 2000

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))
const MD: Components = {
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer noopener" className="text-violet-600 underline underline-offset-2 dark:text-violet-400">{children}</a>
  ),
  img: ({ alt }) => (alt ? <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[11px] text-zinc-500">[picture: {alt}]</span> : null),
}

export function PracticePanel({ slug, name }: { slug: string; name: string }) {
  const [tasks, setTasks] = useState<PracticeTask[] | null>(null)
  const [err, setErr] = useState('')
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [ideas, setIdeas] = useState<PracticeSuggestion[] | null>(null)
  const [thinking, setThinking] = useState(false)

  useEffect(() => {
    let live = true
    agentPracticeApi.list(slug).then((r) => { if (live) setTasks(r.tasks) }).catch((e) => { if (live) setErr(message(e)) })
    return () => { live = false }
  }, [slug])

  // Follow the answers under way.
  const answering = (tasks ?? []).some((t) => t.attempts.some((a) => a.state === 'answering'))
  useEffect(() => {
    if (!answering) return
    let live = true
    const timer = setTimeout(() => {
      agentPracticeApi.list(slug).then((r) => { if (live) setTasks(r.tasks) }).catch(() => undefined)
    }, POLL_MS)
    return () => { live = false; clearTimeout(timer) }
  }, [slug, answering, tasks])

  const put = (t: PracticeTask) => setTasks((all) => [t, ...(all ?? []).filter((x) => x.id !== t.id)])
  const replace = (t: PracticeTask) => setTasks((all) => (all ?? []).map((x) => (x.id === t.id ? t : x)))

  const add = async (body: { task: string; expected?: string; source?: PracticeSuggestion['source'] }) => {
    setBusy(true)
    setErr('')
    try {
      put(await agentPracticeApi.add(slug, body))
      return true
    } catch (e) {
      setErr(message(e))
      return false
    } finally {
      setBusy(false)
    }
  }
  const suggest = async () => {
    setThinking(true)
    setErr('')
    try {
      setIdeas((await agentPracticeApi.suggest(slug)).tasks)
    } catch (e) {
      setErr(message(e))
    } finally {
      setThinking(false)
    }
  }

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4 p-4 wide:p-6">
      <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4 text-sm leading-relaxed text-zinc-300">
        Try {name} on a task, the way your users would give it one. It answers in private — nothing is sent, and nothing
        is kept in its memory. When it gets it wrong, say what it should have done: that becomes a lesson, and it can try again.
      </div>

      <div className="rounded-xl border border-zinc-800 p-3">
        <textarea value={draft} onChange={(e) => setDraft(e.target.value)} rows={3} maxLength={2000}
          placeholder={'e.g. A customer writes: "My parcel arrived damaged — can I get my money back?" What do you reply?'}
          className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-violet-500 focus:outline-none" />
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button onClick={() => void suggest()} disabled={thinking}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800 disabled:opacity-60">
            {thinking ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />} Suggest tasks from its documents
          </button>
          <button onClick={async () => { if (await add({ task: draft })) setDraft('') }} disabled={busy || !draft.trim()}
            className="ml-auto flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-50">
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />} Try it
          </button>
        </div>
      </div>

      {ideas && ideas.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-zinc-200">Tasks from its documents</h4>
          <ul className="grid gap-2 md:grid-cols-3">
            {ideas.map((s) => (
              <li key={s.task} className="flex flex-col rounded-xl border border-zinc-800 p-3">
                <p className="flex-1 text-sm text-zinc-200">{s.task}</p>
                {s.source.doc_title && (
                  <p className="mt-2 truncate text-[11px] text-zinc-500">
                    <BookOpen className="mr-1 inline h-3 w-3" />{[s.source.doc_title, s.source.title].filter(Boolean).join(' › ')}
                  </p>
                )}
                <button onClick={async () => { if (await add(s)) setIdeas((xs) => (xs ?? []).filter((x) => x !== s)) }} disabled={busy}
                  className="mt-2 self-start rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-50">
                  Try this
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      {ideas && ideas.length === 0 && <p className="text-sm text-zinc-500">No tasks came out of its documents this time — try again.</p>}

      {err && <p className="flex items-start gap-1.5 text-sm text-red-500"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{err}</p>}

      {!tasks ? (
        <p className="flex justify-center py-6"><Loader2 className="h-5 w-5 animate-spin text-zinc-500" /></p>
      ) : tasks.length === 0 ? (
        <p className="rounded-xl border border-zinc-800 p-6 text-center text-sm text-zinc-500">No practice yet.</p>
      ) : (
        <ul className="flex flex-col gap-3">
          {tasks.map((t) => (
            <TaskCard key={t.id} slug={slug} task={t} onChange={replace}
              onForget={async () => {
                await agentPracticeApi.forget(slug, t.id).catch(() => undefined)
                setTasks((all) => (all ?? []).filter((x) => x.id !== t.id))
              }}
              onError={setErr} />
          ))}
        </ul>
      )}
    </div>
  )
}

function TaskCard({ slug, task: t, onChange, onForget, onError }: {
  slug: string
  task: PracticeTask
  onChange: (t: PracticeTask) => void
  onForget: () => void
  onError: (e: string) => void
}) {
  const last = t.attempts[t.attempts.length - 1]
  const earlier = t.attempts.slice(0, -1)
  const [showEarlier, setShowEarlier] = useState(false)
  const [showExpected, setShowExpected] = useState(false)
  const [busy, setBusy] = useState(false)

  const act = async (fn: () => Promise<PracticeTask>) => {
    setBusy(true)
    try {
      onChange(await fn())
    } catch (e) {
      onError(message(e))
    } finally {
      setBusy(false)
    }
  }
  const again = () => act(() => agentPracticeApi.again(slug, t.id))

  return (
    <li className="rounded-xl border border-zinc-800">
      <div className="flex items-start gap-2 border-b border-zinc-800 p-3">
        <MessageSquareText className="mt-0.5 h-4 w-4 shrink-0 text-zinc-500" />
        <div className="min-w-0 flex-1">
          <p className="text-sm text-zinc-100">{t.task}</p>
          {t.source.doc_title && (
            <p className="mt-1 truncate text-[11px] text-zinc-500">
              <BookOpen className="mr-1 inline h-3 w-3" />from {[t.source.doc_title, t.source.title].filter(Boolean).join(' › ')}
            </p>
          )}
        </div>
        <button onClick={onForget} title="Forget this task" aria-label="Forget this task"
          className="rounded-md p-1.5 text-zinc-600 hover:bg-zinc-800 hover:text-red-500">
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="p-3">
        {last && <Attempt slug={slug} task={t} attempt={last} busy={busy} onAgain={() => void again()} onChange={onChange} onError={onError} />}
        {t.expected && (
          <div className="mt-3">
            <button onClick={() => setShowExpected((v) => !v)} className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300">
              {showExpected ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />} What its documents say
            </button>
            {showExpected && <p className="mt-1 rounded-lg bg-zinc-950/60 p-2.5 text-sm text-zinc-300">{t.expected}</p>}
          </div>
        )}
        {earlier.length > 0 && (
          <div className="mt-3">
            <button onClick={() => setShowEarlier((v) => !v)} className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300">
              {showEarlier ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              {earlier.length === 1 ? 'Its earlier try' : `Its ${earlier.length} earlier tries`}
            </button>
            {showEarlier && (
              <ol className="mt-2 flex flex-col gap-2">
                {earlier.map((a, i) => (
                  <li key={a.id} className="rounded-lg border border-zinc-800/70 p-2.5">
                    <p className="mb-1 text-[11px] text-zinc-500">
                      Try {i + 1} · {a.verdict === 'good' ? 'good' : a.verdict === 'fix' ? 'fixed with a lesson' : 'not marked'}
                    </p>
                    <div className="fd-file-markdown max-h-40 overflow-y-auto text-xs">
                      <Markdown remarkPlugins={[remarkGfm]} components={MD}>{a.answer || a.error || ''}</Markdown>
                    </div>
                    {a.note && <p className="mt-1 text-xs text-amber-700 dark:text-amber-400">You said: {a.note}</p>}
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
      </div>
    </li>
  )
}

function Attempt({ slug, task, attempt: a, busy, onAgain, onChange, onError }: {
  slug: string
  task: PracticeTask
  attempt: PracticeAttempt
  busy: boolean
  onAgain: () => void
  onChange: (t: PracticeTask) => void
  onError: (e: string) => void
}) {
  const [fixing, setFixing] = useState(false)
  const [note, setNote] = useState('')
  const [drafting, setDrafting] = useState(false)
  const mark = async (verdict: '' | 'good' | 'fix', extra: { note?: string; lesson?: string } = {}) => {
    try {
      onChange(await agentPracticeApi.mark(slug, task.id, { attempt: a.id, verdict, ...extra }))
    } catch (e) {
      onError(message(e))
    }
  }

  if (a.state === 'answering') {
    return <p className="flex items-center gap-2 text-sm text-zinc-400"><Loader2 className="h-4 w-4 animate-spin text-violet-500" /> Answering…</p>
  }
  if (a.state === 'error') {
    return (
      <div className="flex flex-wrap items-center gap-3">
        <p className="flex-1 text-sm text-red-500">{a.error || 'It couldn’t answer.'}</p>
        <button onClick={onAgain} disabled={busy}
          className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-50">
          <RotateCcw className="h-3.5 w-3.5" /> Try again
        </button>
      </div>
    )
  }
  return (
    <div>
      <div className="fd-file-markdown max-h-80 overflow-y-auto text-sm">
        <Markdown remarkPlugins={[remarkGfm]} components={MD}>{a.answer || ''}</Markdown>
      </div>
      <p className="mt-2 flex flex-wrap items-center gap-x-3 text-[11px] text-zinc-500">
        <span className="inline-flex items-center gap-1"><Search className="h-3 w-3" />
          {a.lookups && a.lookups.length ? `looked up ${a.lookups.length} thing${a.lookups.length === 1 ? '' : 's'}` : 'looked nothing up'}
        </span>
        {a.seconds != null && <span>{a.seconds} s</span>}
        {task.attempts.length > 1 && <span>try {task.attempts.length}</span>}
      </p>

      <div className="mt-3">
        {a.verdict === 'good' ? (
          <p className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400">
            <Check className="h-4 w-4" /> Good
            <button onClick={() => void mark('')} className="text-xs text-zinc-500 hover:text-zinc-300">undo</button>
          </p>
        ) : a.verdict === 'fix' ? (
          <div className="flex flex-wrap items-center gap-3">
            <p className="min-w-0 flex-1 text-sm text-zinc-300">
              <Wrench className="mr-1.5 inline h-3.5 w-3.5 text-amber-500" />
              {a.lesson ? 'Taught it a lesson for this.' : 'Marked to fix.'}{a.note ? ` You said: ${a.note}` : ''}
            </p>
            <button onClick={onAgain} disabled={busy}
              className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-50">
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />} Try again
            </button>
          </div>
        ) : fixing ? (
          <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3">
            {drafting ? (
              <LessonDraftForm slug={slug} said={note} reply={a.answer} task={task.task} taughtBy="practice"
                onCancel={() => setDrafting(false)}
                onDone={(lesson) => { void mark('fix', { note, lesson }); setFixing(false); setDrafting(false) }} />
            ) : (
              <>
                <label className="mb-1.5 block text-xs font-medium text-zinc-300">What should it have done?</label>
                <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2} autoFocus maxLength={2000}
                  placeholder="e.g. A damaged parcel doesn't need to come back — offer the refund straight away."
                  className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-violet-500 focus:outline-none" />
                <div className="mt-2 flex flex-wrap justify-end gap-2">
                  <button onClick={() => setFixing(false)} className="rounded-lg px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">Cancel</button>
                  <button onClick={() => void mark('fix', { note })} disabled={!note.trim()}
                    className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-800 disabled:opacity-50">
                    Just mark it
                  </button>
                  <button onClick={() => setDrafting(true)} disabled={!note.trim()}
                    className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-50">
                    <Sparkles className="h-3.5 w-3.5" /> Make it a lesson
                  </button>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="flex flex-wrap gap-2">
            <button onClick={() => void mark('good')}
              className="flex items-center gap-1.5 rounded-lg border border-emerald-500/40 px-3 py-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-500/10 dark:text-emerald-300">
              <ThumbsUp className="h-3.5 w-3.5" /> Good
            </button>
            <button onClick={() => setFixing(true)}
              className="flex items-center gap-1.5 rounded-lg border border-amber-500/40 px-3 py-1.5 text-xs font-medium text-amber-700 hover:bg-amber-500/10 dark:text-amber-300">
              <Wrench className="h-3.5 w-3.5" /> Fix it
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
