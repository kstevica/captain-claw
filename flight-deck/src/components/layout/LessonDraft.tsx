import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Check, GraduationCap, Hand, Lightbulb, Loader2, ScrollText, X } from 'lucide-react'
import { agentLessonsApi, type LessonKind } from '../../services/agentLessons'
import { syncInstructions } from './lessonSync'

// ── A lesson from the owner's words ──────────────────────────────────
//
// What they said — a correction in a chat, what a practice answer should
// have been — drafted into one lesson by the agent's model
// (/fd/agent-lessons/{slug}/draft), for them to check and teach: a rule,
// something to ask first, or a fact; always in mind or when it matters.

const KINDS: { id: LessonKind; label: string; icon: typeof ScrollText; tone: string }[] = [
  { id: 'rule', label: 'Rule', icon: ScrollText, tone: 'bg-violet-500/10 text-violet-700 dark:text-violet-300' },
  { id: 'ask_first', label: 'Ask first', icon: Hand, tone: 'bg-amber-500/10 text-amber-700 dark:text-amber-300' },
  { id: 'fact', label: 'Fact', icon: Lightbulb, tone: 'bg-sky-500/10 text-sky-700 dark:text-sky-300' },
]

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))

export function LessonDraftForm({ slug, said, reply = '', task = '', taughtBy, onDone, onCancel }: {
  slug: string
  /** The owner's words. */
  said: string
  /** What the agent had answered, and the practice task it answered. */
  reply?: string
  task?: string
  taughtBy: 'chat' | 'practice'
  /** Taught: the new lesson's id. */
  onDone: (lessonId: string) => void
  onCancel: () => void
}) {
  const [drafting, setDrafting] = useState(true)
  const [kind, setKind] = useState<LessonKind>('rule')
  const [text, setText] = useState('')
  const [always, setAlways] = useState(false)
  const [nothing, setNothing] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  useEffect(() => {
    let live = true
    agentLessonsApi.draft(slug, { said, reply, task })
      .then((d) => {
        if (!live) return
        setKind(d.kind)
        setText(d.text)
        setAlways(d.always)
        setNothing(d.drafted && !d.text)
      })
      .catch(() => { if (live) setText(said) })
      .finally(() => { if (live) setDrafting(false) })
    return () => { live = false }
  }, [slug, said, reply, task])

  const teach = async () => {
    setBusy(true)
    setErr('')
    try {
      const res = await agentLessonsApi.add(slug, { kind, text, always: kind !== 'fact' && always, taught_by: taughtBy })
      syncInstructions(slug, res)
      const flat = text.trim().split(/\s+/).join(' ').toLowerCase()
      onDone(res.lessons.find((x) => x.text.toLowerCase() === flat)?.id ?? '')
    } catch (e) {
      setErr(message(e))
    } finally {
      setBusy(false)
    }
  }

  if (drafting) {
    return (
      <p className="flex items-center gap-2 py-2 text-sm text-zinc-400">
        <Loader2 className="h-4 w-4 animate-spin text-violet-500" /> Drafting a lesson from what you said…
      </p>
    )
  }
  return (
    <div className="flex flex-col gap-2">
      {nothing && (
        <p className="text-xs text-zinc-500">That doesn't read like something for it to learn. Write the lesson yourself, or cancel.</p>
      )}
      <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Kind of lesson">
        {KINDS.map((k) => (
          <button key={k.id} role="radio" aria-checked={kind === k.id} onClick={() => setKind(k.id)}
            className={`flex items-center gap-1 rounded-md px-2.5 py-1 text-xs ${kind === k.id ? `${k.tone} font-medium` : 'text-zinc-400 hover:bg-zinc-800'}`}>
            <k.icon className="h-3 w-3" />{k.label}
          </button>
        ))}
      </div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} maxLength={400} autoFocus
        placeholder="The lesson, as it should follow it from now on"
        className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-violet-500 focus:outline-none" />
      <div className="flex flex-wrap items-center gap-2">
        {kind !== 'fact' && (
          <label className="flex items-center gap-1.5 text-xs text-zinc-400">
            <input type="checkbox" checked={always} onChange={(e) => setAlways(e.target.checked)} className="h-3.5 w-3.5 accent-violet-600" />
            Always in mind — it applies to every conversation
          </label>
        )}
        <div className="ml-auto flex gap-2">
          <button onClick={onCancel} className="rounded-lg px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">Cancel</button>
          <button onClick={() => void teach()} disabled={busy || !text.trim()}
            className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-50">
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />} Teach it
          </button>
        </div>
      </div>
      {err && <p className="text-xs text-red-500">{err}</p>}
    </div>
  )
}

/** "Save as a lesson" from a chat: the owner's message (and the answer it corrects) as a lesson. */
export function SaveLessonDialog({ agent, said, reply, onClose }: {
  agent: { slug: string; name: string }
  said: string
  reply?: string
  onClose: () => void
}) {
  const [saved, setSaved] = useState(false)
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 p-0 sm:items-center sm:p-4" onClick={onClose}>
      <div role="dialog" aria-label="Save as a lesson" onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg rounded-t-2xl border border-zinc-800 bg-zinc-900 p-4 shadow-2xl sm:rounded-2xl">
        <div className="mb-3 flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
            <GraduationCap className="h-4 w-4" />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-semibold text-zinc-100">Save as a lesson</h3>
            <p className="truncate text-xs text-zinc-500">{agent.name} follows it from its next message — in every conversation</p>
          </div>
          <button onClick={onClose} aria-label="Close" className="rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>
        <blockquote className="mb-3 line-clamp-3 border-l-2 border-violet-500/50 pl-3 text-xs text-zinc-400">{said}</blockquote>
        {saved ? (
          <div className="flex items-center justify-between gap-3">
            <p className="flex items-center gap-1.5 text-sm text-emerald-600 dark:text-emerald-400">
              <Check className="h-4 w-4" /> Taught — it's in its lessons now.
            </p>
            <button onClick={onClose} className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-800">Done</button>
          </div>
        ) : (
          <LessonDraftForm slug={agent.slug} said={said} reply={reply} taughtBy="chat" onDone={() => setSaved(true)} onCancel={onClose} />
        )}
      </div>
    </div>,
    document.body,
  )
}
