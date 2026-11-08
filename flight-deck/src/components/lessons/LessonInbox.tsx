import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  AlertTriangle, Check, CheckCheck, GraduationCap, Hand, Lightbulb, Loader2, Pencil, Pin, ScrollText, Undo2, X,
} from 'lucide-react'
import {
  agentLessonsApi, lessonProposalsApi, type AgentProposals, type LessonKind, type LessonSuggestion,
} from '../../services/agentLessons'
import { useLessonInbox } from '../../stores/lessonInboxStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { syncInstructions } from '../layout/lessonSync'

// ── Lessons to approve ───────────────────────────────────────────────
//
// What the owner's agents noticed in their chats ("too long", "always CC
// Marko") and drafted as lessons, and what members of a shared agent
// suggested — every agent's, in one place, top-down:
//
//   how many wait, and from which agents (a chip each, to jump to it)
//   → each agent's lessons: the lesson, the owner's words it came from,
//     Approve · Edit · Dismiss (and Approve all, per agent and overall)
//   → Just learned: what was approved lately, each with Undo.
//
// Nothing is learned until the owner approves it; an approved lesson
// counts from the agent's next message (lesson_proposals.py on the deck).
// One dialog for the app (LessonInboxHost, in App), opened by the badge
// beside the bell and by the bell's lesson rows.

const KIND: Record<LessonKind, { label: string; badge: string; icon: typeof ScrollText; hint: string }> = {
  rule: { label: 'Rule', badge: 'bg-violet-500/10 text-violet-700 dark:text-violet-300', icon: ScrollText,
    hint: 'Something to do, or never to do' },
  ask_first: { label: 'Ask first', badge: 'bg-amber-500/10 text-amber-700 dark:text-amber-300', icon: Hand,
    hint: 'When to stop and ask you before acting' },
  fact: { label: 'Fact', badge: 'bg-sky-500/10 text-sky-700 dark:text-sky-300', icon: Lightbulb,
    hint: 'Something true it should know' },
}
const KINDS: LessonKind[] = ['rule', 'ask_first', 'fact']
const CHANNEL: Record<string, string> = {
  web: 'your chat', whatsapp: 'WhatsApp', telegram: 'Telegram', device: 'a device', app: 'the app', voice: 'a voice chat',
  slack: 'Slack', discord: 'Discord',
}
// An agent's colour, picked by its name — the same as on the Line.
const HUES = [
  'bg-violet-500/15 text-violet-700 dark:text-violet-300',
  'bg-sky-500/15 text-sky-700 dark:text-sky-300',
  'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  'bg-rose-500/15 text-rose-700 dark:text-rose-300',
  'bg-teal-500/15 text-teal-700 dark:text-teal-300',
  'bg-indigo-500/15 text-indigo-700 dark:text-indigo-300',
]
const hueOf = (name: string) => HUES[[...name].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7) % HUES.length]
function initials(name: string): string {
  const words = name.trim().split(/[\s_-]+/).filter(Boolean)
  if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase()
  return name.trim().slice(0, 2).toUpperCase() || '?'
}

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))
const ago = (epoch: number) => {
  const s = Math.max(0, Date.now() / 1000 - epoch)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  const d = Math.round(s / 86400)
  return d === 1 ? 'yesterday' : `${d} days ago`
}
const lessonsWord = (n: number) => `${n} ${n === 1 ? 'lesson' : 'lessons'}`

function AgentMark({ name, size = 'md' }: { name: string; size?: 'sm' | 'md' }) {
  const box = size === 'sm' ? 'h-5 w-5 rounded-md text-[9px]' : 'h-8 w-8 rounded-lg text-[11px]'
  return <span className={`flex shrink-0 items-center justify-center font-semibold ${box} ${hueOf(name)}`}>{initials(name)}</span>
}

function KindPill({ kind }: { kind: LessonKind }) {
  const k = KIND[kind]
  return (
    <span className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium ${k.badge}`} title={k.hint}>
      <k.icon className="h-3 w-3" />{k.label}
    </span>
  )
}

/** The words a lesson came from: what the owner wrote (the quote the agent copied, else their message). */
function Evidence({ p }: { p: LessonSuggestion }) {
  const who = p.origin === 'auto' ? 'You wrote' : `${p.by_name || 'A member'} wrote`
  const words = p.origin === 'auto' ? (p.quote || p.said) : p.said
  if (!words) return null
  return (
    <div className="mt-2.5">
      <blockquote className="border-l-2 border-zinc-700 pl-2.5 text-[13px] leading-snug text-zinc-400">
        <span className="mr-1 text-[11px] font-medium not-italic text-zinc-500">{who}</span>
        <span className="italic">“{words.length > 280 ? `${words.slice(0, 280)}…` : words}”</span>
      </blockquote>
      {(p.reply || (p.origin === 'auto' && p.quote && p.said !== p.quote)) && (
        <details className="group mt-1.5">
          <summary className="cursor-pointer list-none text-[11px] text-zinc-500 hover:text-zinc-300">
            <span className="group-open:hidden">Show the conversation</span>
            <span className="hidden group-open:inline">Hide the conversation</span>
          </summary>
          <div className="mt-1.5 flex flex-col gap-1.5 rounded-lg bg-zinc-950/60 p-2.5 text-xs leading-relaxed">
            {p.reply && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">Its answer</p>
                <p className="line-clamp-6 whitespace-pre-wrap text-zinc-400">{p.reply}</p>
              </div>
            )}
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">{who}</p>
              <p className="whitespace-pre-wrap text-zinc-300">{p.said}</p>
            </div>
          </div>
        </details>
      )}
    </div>
  )
}

/** One lesson waiting for its owner: approve it as it is, edit it first, or dismiss it. A member's suggestion is
 *  drafted (by the agent's model) when its owner writes it as a lesson. */
export function ProposalCard({ slug, p, onSettled }: {
  slug: string
  p: LessonSuggestion
  onSettled: (how: 'approved' | 'dismissed', p: LessonSuggestion) => void
}) {
  const drafted = !!p.text && !!p.kind
  const [editing, setEditing] = useState(false)
  const [kind, setKind] = useState<LessonKind>((p.kind || 'rule') as LessonKind)
  const [text, setText] = useState(p.text)
  const [always, setAlways] = useState(p.always)
  const [busy, setBusy] = useState<'approve' | 'dismiss' | 'draft' | null>(null)
  const [err, setErr] = useState('')
  const [note, setNote] = useState('')

  const approve = async (edits?: { kind: LessonKind; text: string; always: boolean }) => {
    setBusy('approve')
    setErr('')
    try {
      const res = await lessonProposalsApi.approve(slug, p.id, edits)
      if (res.changed) syncInstructions(slug, res.changed)
      onSettled('approved', res.proposal)
    } catch (e) {
      setErr(message(e))
      setBusy(null)
    }
  }
  const dismiss = async () => {
    setBusy('dismiss')
    setErr('')
    try {
      const res = await lessonProposalsApi.dismiss(slug, p.id)
      onSettled('dismissed', res.proposal)
    } catch (e) {
      setErr(message(e))
      setBusy(null)
    }
  }
  const draft = async () => {
    setBusy('draft')
    setErr('')
    try {
      const d = await agentLessonsApi.draft(slug, { said: p.said, reply: p.reply })
      setKind(d.kind)
      setText(d.text)
      setAlways(d.always)
      setNote(d.text ? '' : 'Their words don’t teach a lesson on their own — write one, or dismiss it.')
      setEditing(true)
    } catch (e) {
      setErr(message(e))
    } finally {
      setBusy(null)
    }
  }

  const cleaned = text.trim()
  const shownKind = (p.kind || 'rule') as LessonKind
  return (
    <li className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4 shadow-sm">
      <div className="flex flex-wrap items-center gap-1.5">
        {drafted && !editing && <KindPill kind={shownKind} />}
        {drafted && !editing && p.always && (
          <span className="inline-flex items-center gap-1 rounded-md bg-zinc-800 px-1.5 py-0.5 text-[11px] text-zinc-300"
            title="In its instructions: it keeps this in mind in every conversation">
            <Pin className="h-3 w-3" />Always in mind
          </span>
        )}
        {!drafted && !editing && (
          <span className="inline-flex items-center rounded-md bg-amber-500/10 px-1.5 py-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-300">
            Suggestion
          </span>
        )}
        <span className="ml-auto text-[11px] text-zinc-500">
          {p.origin === 'auto' ? `Noticed in ${CHANNEL[p.channel] ?? 'a chat'}` : `Suggested by ${p.by_name || 'a member'}`} · {ago(p.created)}
        </span>
      </div>

      {editing ? (
        <div className="mt-3 flex flex-col gap-2.5">
          <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Kind of lesson">
            {KINDS.map((k) => (
              <button key={k} role="radio" aria-checked={kind === k} onClick={() => { setKind(k); if (k === 'fact') setAlways(false) }}
                aria-label={KIND[k].label} title={KIND[k].hint}
                className={`flex items-center gap-1 rounded-lg border px-2.5 py-1 text-xs transition-colors ${kind === k
                  ? 'border-violet-500 bg-violet-500/15 text-violet-700 dark:text-violet-200' : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`}>
                {(() => { const I = KIND[k].icon; return <I className="h-3.5 w-3.5" /> })()}{KIND[k].label}
              </button>
            ))}
          </div>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} autoFocus maxLength={400}
            placeholder="One sentence it can follow on its own, e.g. Keep answers to three short paragraphs."
            className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm leading-snug text-zinc-100 placeholder:text-zinc-600 focus:border-violet-500 focus:outline-none" />
          <label className={`flex items-start gap-2 text-xs ${kind === 'fact' ? 'text-zinc-600' : 'text-zinc-300'}`}>
            <input type="checkbox" checked={always} disabled={kind === 'fact'} onChange={(e) => setAlways(e.target.checked)}
              className="mt-0.5 accent-violet-600" />
            <span><b className="font-medium">Always in mind</b> — keep it in mind in every conversation (tone, language, length, format).
              Otherwise it comes up when a message is about it.</span>
          </label>
          {note && <p className="text-xs text-amber-600 dark:text-amber-400">{note}</p>}
        </div>
      ) : drafted ? (
        <p className="mt-2.5 text-[15px] leading-snug text-zinc-100">{p.text}</p>
      ) : null}

      {!editing && <Evidence p={p} />}

      {err && <p className="mt-2 flex items-start gap-1.5 text-xs text-red-500"><AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" />{err}</p>}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {editing ? (
          <>
            <button onClick={() => void approve({ kind, text: cleaned, always: always && kind !== 'fact' })} disabled={!cleaned || !!busy}
              className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-1.5 text-sm font-semibold text-white hover:bg-violet-500 disabled:opacity-50">
              {busy === 'approve' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />} Approve
            </button>
            <button onClick={() => { setEditing(false); setNote(''); setKind(shownKind); setText(p.text); setAlways(p.always) }} disabled={!!busy}
              className="rounded-lg px-3 py-1.5 text-sm text-zinc-400 hover:bg-zinc-800 disabled:opacity-50">
              Cancel
            </button>
          </>
        ) : drafted ? (
          <>
            <button onClick={() => void approve()} disabled={!!busy}
              className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-1.5 text-sm font-semibold text-white hover:bg-violet-500 disabled:opacity-50">
              {busy === 'approve' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />} Approve
            </button>
            <button onClick={() => setEditing(true)} disabled={!!busy}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800 disabled:opacity-50">
              <Pencil className="h-3.5 w-3.5" /> Edit
            </button>
          </>
        ) : (
          <button onClick={() => void draft()} disabled={!!busy}
            className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-1.5 text-sm font-semibold text-white hover:bg-violet-500 disabled:opacity-50">
            {busy === 'draft' ? <Loader2 className="h-4 w-4 animate-spin" /> : <GraduationCap className="h-4 w-4" />} Write it as a lesson
          </button>
        )}
        <button onClick={() => void dismiss()} disabled={!!busy}
          className="ml-auto flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-sm text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-50">
          {busy === 'dismiss' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <X className="h-3.5 w-3.5" />} Dismiss
        </button>
      </div>
    </li>
  )
}

/** A lesson approved lately, with Undo. */
function LearnedRow({ agent, p, onUndone }: { agent: AgentProposals; p: LessonSuggestion; onUndone: () => void }) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const undo = async () => {
    setBusy(true)
    setErr('')
    try {
      const res = await lessonProposalsApi.undo(agent.slug, p.id)
      if (res.changed) syncInstructions(agent.slug, res.changed)
      onUndone()
    } catch (e) {
      setErr(message(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <li className="flex items-start gap-2.5 px-3 py-2.5">
      <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
        <Check className="h-3 w-3" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm leading-snug text-zinc-200">{p.text}</p>
        <p className="mt-0.5 flex items-center gap-1.5 text-[11px] text-zinc-500">
          <AgentMark name={agent.name} size="sm" />{agent.name} · {KIND[(p.kind || 'rule') as LessonKind].label}
          {p.always ? ' · every chat' : ''} · {ago(p.settled)}
        </p>
        {err && <p className="mt-1 text-xs text-red-500">{err}</p>}
      </div>
      <button onClick={() => void undo()} disabled={busy} title="Take this lesson away again; it waits for you once more"
        className="flex shrink-0 items-center gap-1 rounded-lg px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
        {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Undo2 className="h-3.5 w-3.5" />} Undo
      </button>
    </li>
  )
}

const LEARNED_SHOWN = 5

export function LessonInboxDialog() {
  const data = useLessonInbox((s) => s.data)
  const focus = useLessonInbox((s) => s.focus)
  // Opened from one agent's menu: its lessons only, until "See every agent's".
  const scopeName = useLessonInbox((s) => s.scopeName)
  const widen = useLessonInbox((s) => s.widen)
  const hide = useLessonInbox((s) => s.hide)
  const refresh = useLessonInbox((s) => s.refresh)
  // Settled here and not refreshed yet: hidden right away, so the list never jumps back.
  const [settled, setSettled] = useState<Set<string>>(new Set())
  const [toast, setToast] = useState<{ text: string; key: number } | null>(null)
  // A new key each time, so the same words shown twice still show again.
  const say = (text: string) => setToast((t) => ({ text, key: (t?.key ?? 0) + 1 }))
  const [allBusy, setAllBusy] = useState(false)
  const [confirmAll, setConfirmAll] = useState(false)
  const [allLearned, setAllLearned] = useState(false)
  const sections = useRef<Record<string, HTMLElement | null>>({})

  const every = useMemo(() => (data?.agents ?? []).map((a) => ({
    ...a, waiting: a.waiting.filter((p) => !settled.has(`${a.slug}/${p.id}`)),
  })), [data, settled])
  const agents = useMemo(() => (scopeName ? every.filter((a) => a.slug === focus) : every), [every, scopeName, focus])
  // Waiting on the other agents, while only one agent's show.
  const elsewhere = scopeName ? every.filter((a) => a.slug !== focus).reduce((n, a) => n + a.waiting.length, 0) : 0
  const withWaiting = agents.filter((a) => a.waiting.length)
  const waiting = withWaiting.reduce((n, a) => n + a.waiting.length, 0)
  const learned = useMemo(() => agents.flatMap((a) => a.learned.map((p) => ({ agent: a, p })))
    .sort((x, y) => (y.p.settled || 0) - (x.p.settled || 0)), [agents])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') hide() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [hide])

  // Opened from an agent's bell: bring its lessons into view.
  useEffect(() => {
    if (focus && data) sections.current[focus]?.scrollIntoView({ block: 'start', behavior: 'smooth' })
  }, [focus, data])

  useEffect(() => {
    if (!toast) return
    const t = window.setTimeout(() => setToast(null), 5000)
    return () => window.clearTimeout(t)
  }, [toast])

  const settle = (slug: string, name: string, how: 'approved' | 'dismissed', p: LessonSuggestion) => {
    setSettled((s) => new Set(s).add(`${slug}/${p.id}`))
    say(how === 'approved' ? `${name} learned it — from its next message.` : 'Dismissed.')
    void refresh()
  }

  const approveAll = async (list: AgentProposals[]) => {
    setAllBusy(true)
    setConfirmAll(false)
    let done = 0
    let failed = 0
    for (const a of list) {
      for (const p of a.waiting) {
        if (!p.text || !p.kind) continue           // a member's suggestion is written first
        try {
          const res = await lessonProposalsApi.approve(a.slug, p.id)
          if (res.changed) syncInstructions(a.slug, res.changed)
          setSettled((s) => new Set(s).add(`${a.slug}/${p.id}`))
          done += 1
        } catch {
          failed += 1
        }
      }
    }
    setAllBusy(false)
    say(`Approved ${lessonsWord(done)}${failed ? ` · ${failed} couldn’t be` : ''}. Undo any under Just learned.`)
    void refresh()
  }
  const ready = (list: AgentProposals[]) => list.reduce((n, a) => n + a.waiting.filter((p) => p.text && p.kind).length, 0)
  const readyAll = ready(withWaiting)

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={hide}>
      <div role="dialog" aria-label="Lessons to approve"
        className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-900 shadow-2xl wide:h-[min(90vh,56rem)] wide:max-w-3xl wide:rounded-2xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}>

        {/* Title */}
        <div className="flex shrink-0 items-center gap-3 px-4 pt-3 wide:px-5">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
            <GraduationCap className="h-[18px] w-[18px]" />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-base font-semibold text-zinc-100">
              Lessons to approve{scopeName ? <span className="font-normal text-zinc-500"> · {scopeName}</span> : null}
            </h3>
            <p className="truncate text-xs text-zinc-500">
              {scopeName ? `What ${scopeName} noticed in your chats.` : 'What your agents noticed in your chats.'} Nothing is learned
              until you approve it.
            </p>
          </div>
          <button onClick={hide} aria-label="Close" title="Close" className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* The top: how many, from whom, approve all */}
        {data && waiting > 0 && (
          <div className="mx-4 mt-3 shrink-0 rounded-2xl border border-violet-500/20 bg-violet-500/[0.06] p-3.5 wide:mx-5">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-semibold tabular-nums leading-none text-zinc-100">{waiting}</span>
                <span className="text-sm text-zinc-400">
                  {waiting === 1 ? 'waits' : 'wait'} for you{withWaiting.length > 1 ? ` · ${withWaiting.length} agents` : ''}
                </span>
              </div>
              {readyAll > 1 && (
                <div className="ml-auto flex items-center gap-2">
                  {confirmAll ? (
                    <>
                      <span className="text-xs text-zinc-400">Approve {lessonsWord(readyAll)} as written?</span>
                      <button onClick={() => void approveAll(withWaiting)}
                        className="rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-violet-500">Yes, approve</button>
                      <button onClick={() => setConfirmAll(false)}
                        className="rounded-lg px-2.5 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">No</button>
                    </>
                  ) : (
                    <button onClick={() => setConfirmAll(true)} disabled={allBusy}
                      className="flex items-center gap-1.5 rounded-lg border border-violet-500/40 px-3 py-1.5 text-xs font-semibold text-violet-700 hover:bg-violet-500/10 disabled:opacity-50 dark:text-violet-200">
                      {allBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CheckCheck className="h-3.5 w-3.5" />} Approve all {readyAll}
                    </button>
                  )}
                </div>
              )}
            </div>
            {withWaiting.length > 1 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {withWaiting.map((a) => (
                  <button key={a.slug} onClick={() => sections.current[a.slug]?.scrollIntoView({ block: 'start', behavior: 'smooth' })}
                    className="flex items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-900/60 py-0.5 pl-0.5 pr-2.5 text-xs text-zinc-300 hover:bg-zinc-800">
                    <AgentMark name={a.name} size="sm" />{a.name}
                    <span className="tabular-nums text-zinc-500">{a.waiting.length}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-6 pt-4 wide:px-5">
          {!data ? (
            <Centered><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></Centered>
          ) : waiting === 0 ? (
            <div className="mx-auto flex max-w-md flex-col items-center gap-2 px-4 pb-6 pt-10 text-center">
              <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                <CheckCheck className="h-6 w-6" />
              </span>
              <p className="text-base font-semibold text-zinc-100">
                {scopeName ? `Nothing from ${scopeName} waits for you` : 'You’re all caught up'}
              </p>
              <p className="text-sm leading-relaxed text-zinc-500">
                When you correct {scopeName || 'an agent'} in a chat — “too long”, “always CC Marko”, “ask me before booking” —
                it suggests a lesson here. Approve it, and it follows it from then on.
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-6">
              {withWaiting.map((a) => {
                const n = ready([a])
                return (
                  <section key={a.slug} ref={(el) => { sections.current[a.slug] = el }} className="scroll-mt-2">
                    <div className="mb-2 flex items-center gap-2.5">
                      <AgentMark name={a.name} />
                      <div className="min-w-0 flex-1">
                        <h4 className="truncate text-sm font-semibold text-zinc-100">{a.name}</h4>
                        <p className="text-[11px] text-zinc-500">{lessonsWord(a.waiting.length)} to approve</p>
                      </div>
                      {n > 1 && withWaiting.length > 1 && (
                        <button onClick={() => void approveAll([a])} disabled={allBusy}
                          className="flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs text-violet-700 hover:bg-violet-500/10 disabled:opacity-50 dark:text-violet-300">
                          <CheckCheck className="h-3.5 w-3.5" /> Approve {n}
                        </button>
                      )}
                    </div>
                    <ul className="flex flex-col gap-2.5">
                      {a.waiting.map((p) => (
                        <ProposalCard key={p.id} slug={a.slug} p={p} onSettled={(how, q) => settle(a.slug, a.name, how, q)} />
                      ))}
                    </ul>
                  </section>
                )
              })}
            </div>
          )}

          {scopeName && elsewhere > 0 && (
            <div className="mt-6 flex justify-center">
              <button onClick={widen}
                className="rounded-lg border border-zinc-700 px-3.5 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800">
                See every agent’s · {lessonsWord(elsewhere)} more
              </button>
            </div>
          )}

          {data && learned.length > 0 && (
            <section className="mt-8">
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">Just learned</h4>
              <ul className="divide-y divide-zinc-800 overflow-hidden rounded-xl border border-zinc-800">
                {(allLearned ? learned : learned.slice(0, LEARNED_SHOWN)).map(({ agent, p }) => (
                  <LearnedRow key={`${agent.slug}/${p.id}`} agent={agent} p={p} onUndone={() => {
                    setSettled((s) => { const next = new Set(s); next.delete(`${agent.slug}/${p.id}`); return next })
                    say('Taken away — it waits for you again.')
                    void refresh()
                  }} />
                ))}
              </ul>
              {learned.length > LEARNED_SHOWN && (
                <button onClick={() => setAllLearned((v) => !v)} className="mt-2 text-xs text-zinc-500 hover:text-zinc-300">
                  {allLearned ? 'Show fewer' : `Show all ${learned.length}`}
                </button>
              )}
            </section>
          )}
        </div>

        {toast && (
          <div key={toast.key} role="status"
            className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center px-4">
            <span className="rounded-full bg-zinc-100 px-4 py-2 text-sm font-medium text-zinc-900 shadow-lg dark:bg-zinc-800 dark:text-zinc-100">
              {toast.text}
            </span>
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}

function Centered({ children }: { children: ReactNode }) {
  return <div className="flex h-full min-h-[12rem] flex-col items-center justify-center gap-3 p-6 text-center">{children}</div>
}

/** Beside the bell: how many lessons wait, when any do — a click opens them. */
export function LessonInboxBadge() {
  const waiting = useLessonInbox((s) => s.waiting)
  const show = useLessonInbox((s) => s.show)
  if (!waiting) return null
  const label = `${lessonsWord(waiting)} to approve`
  // The bell's size and shape, so it fits wherever the bell sits (a narrow kiosk column too).
  return (
    <button onClick={() => show()} title={label} aria-label={label}
      className="relative rounded p-1.5 text-violet-600 transition-colors hover:bg-zinc-800 dark:text-violet-400">
      <GraduationCap className="h-4 w-4" />
      <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-violet-600 px-0.5 text-[9px] font-bold tabular-nums text-white">
        {waiting > 9 ? '9+' : waiting}
      </span>
    </button>
  )
}

/** Mounted once (App): keeps the count fresh — on start, every minute while the page is in view, and when the bell
 *  gets something new — and shows the dialog when it's opened. */
export function LessonInboxHost() {
  const open = useLessonInbox((s) => s.open)
  const refresh = useLessonInbox((s) => s.refresh)
  const unread = useNotificationStore((s) => s.unreadCount)

  useEffect(() => {
    void refresh()
    const tick = () => { if (document.visibilityState === 'visible') void refresh() }
    const t = window.setInterval(tick, 60_000)
    document.addEventListener('visibilitychange', tick)
    return () => { window.clearInterval(t); document.removeEventListener('visibilitychange', tick) }
  }, [refresh])

  // A new bell may be a new lesson.
  useEffect(() => { if (unread) void refresh() }, [unread, refresh])

  return open ? <LessonInboxDialog /> : null
}
