import { useEffect, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'
import { Bell, Check, ChevronDown, Loader2, Mail, MessageSquare, Sparkles } from 'lucide-react'
import { resolveProposal } from '../../services/notifications'
import { taskboardApi, type HandOverAgent, type NextStepKind, type ProposalEdit, type TaskProposal } from '../../services/taskboardApi'
import { ChannelIcon, KeyHint } from './taskboardParts'
import { TASK_KEYS, channelLabel, proposalLabel, recipientText } from './taskboardText'

// The board's next steps (v2): what it would do next on a task — an email
// draft, or a reply on the chat the task came in on — waiting for your
// Approve, and yours to change first: who it goes to (only the task's own
// people and chats; picking an email address makes it a draft, a chat a
// reply there), the subject, the words. "Suggest next step" asks for one
// now. Nothing leaves without your Approve, and email is only ever a draft:
// Approve puts it in your drafts, and nothing sends it.

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))
const SEND_KEY = /Mac|iPhone|iPad/.test(navigator.userAgent) ? '⌘↵' : 'Ctrl↵'

// What the server takes (taskboard_nextsteps: _TEXT_MAX, _SUBJECT_MAX).
const TEXT_MAX = 2000
const SUBJECT_MAX = 200

interface Draft { to: string; subject: string; text: string }

const draftOf = (p: TaskProposal): Draft => ({
  to: p.payload?.to || '', subject: p.payload?.subject || '', text: p.payload?.text || '',
})
const sameDraft = (a: Draft, b: Draft) => a.to === b.to && a.subject === b.subject && a.text === b.text

/** Escape in a field leaves the field, not the board. */
const leaveOnEscape = (e: ReactKeyboardEvent<HTMLElement>) => {
  if (e.key !== 'Escape') return
  e.stopPropagation()
  e.currentTarget.blur()
}

/** A next step the board proposed, waiting for you: read it, change it,
 *  approve or reject it. Approve with unsaved changes saves them first. */
export function NextStepCard({ proposal, phone, onAnswered, onSaved }: {
  proposal: TaskProposal
  phone: boolean
  /** Approved or rejected: the task moves on. */
  onAnswered: () => Promise<void>
  /** A change was saved: the task reads again. */
  onSaved: () => void
}) {
  const incoming = draftOf(proposal)
  const [base, setBase] = useState<Draft>(incoming)
  const [draft, setDraft] = useState<Draft>(incoming)
  const [busy, setBusy] = useState<'save' | 'approve' | 'reject' | null>(null)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const dirty = !sameDraft(draft, base)

  // A fresh copy from the server (a change saved, a new proposal) is taken
  // in — unless you are changing this one.
  const seenKey = `${proposal.id}\u0000${incoming.to}\u0000${incoming.subject}\u0000${incoming.text}`
  const [seen, setSeen] = useState(seenKey)
  if (seen !== seenKey) {
    setSeen(seenKey)
    if (!dirty) { setBase(incoming); setDraft(incoming) }
  }

  const options = proposal.payload?.to_options ?? []
  const editable = !!proposal.editable && proposal.status === 'awaiting_approval'
  // Who it goes to is one of the task's own people, never anyone else —
  // and what it is follows them: an email address makes it a draft, a chat
  // a reply on that chat.
  const toKnown = !options.length || options.some((o) => o.address === draft.to)
  const recipient = options.find((o) => o.address === draft.to)
  const toLine = recipient ? recipientText(recipient) : draft.to
  const reminder = proposal.kind === 'nudge'
  const mail = recipient ? recipient.kind === 'email' : proposal.kind === 'mail.draft'
  const channel = recipient ? (recipient.kind === 'email' ? 'email' : recipient.kind) : proposal.payload?.channel || ''
  const label = proposalLabel({ kind: reminder ? 'nudge' : mail ? 'mail.draft' : 'reply.on_origin', payload: { channel } })
  // Words it can't go without (the server refuses an empty one).
  const blank = editable && !draft.text.trim()

  const patch = (): ProposalEdit => {
    const out: ProposalEdit = {}
    if (draft.to !== base.to) out.to = draft.to
    if (mail && draft.subject !== base.subject) out.subject = draft.subject
    if (draft.text !== base.text) out.text = draft.text
    return out
  }
  const edit = (next: Partial<Draft>) => {
    setDraft((cur) => ({ ...cur, ...next }))
    setSaved(false)
    setError('')
  }
  const save = async () => {
    if (!dirty || !toKnown || blank) return
    setBusy('save')
    setError('')
    try {
      await taskboardApi.editProposal(proposal.id, patch())
      setBase(draft)
      setSaved(true)
      onSaved()
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(null)
    }
  }
  const answer = async (approve: boolean) => {
    setBusy(approve ? 'approve' : 'reject')
    setError('')
    try {
      if (approve && dirty) {
        await taskboardApi.editProposal(proposal.id, patch())
        setBase(draft)
      }
      await resolveProposal(proposal.id, approve)
      await onAnswered()
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(null)
    }
  }

  const KindIcon = mail ? Mail : reminder ? Bell : MessageSquare
  const field = `w-full rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 text-zinc-200 placeholder-zinc-500 focus:border-teal-500/60 focus:outline-none disabled:opacity-60 ${
    phone ? 'h-11 text-base' : 'py-1.5 text-sm'}`
  const fieldLabel = 'w-14 shrink-0 text-[11px] font-medium text-zinc-500'
  const btn = `flex items-center justify-center gap-1 rounded-lg font-medium disabled:opacity-50 ${phone ? 'h-10 px-3.5 text-sm' : 'px-2.5 py-1.5 text-xs'}`

  return (
    <div data-next-step className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2.5">
      <div className="flex items-center gap-1.5 text-xs font-medium text-amber-800 dark:text-amber-300">
        {channel && !mail && !reminder ? <ChannelIcon channel={channel} className="h-3.5 w-3.5" /> : <KindIcon className="h-3.5 w-3.5 shrink-0" />}
        <span className="min-w-0 flex-1 truncate">Needs you: next step · {label}</span>
      </div>
      {proposal.title && <p className="mt-1 break-words text-sm text-zinc-300">{proposal.title}</p>}

      <div className="mt-2.5 flex flex-col gap-2">
        {!reminder && (options.length > 1 && editable ? (
          <label className="flex items-center gap-2">
            <span className={fieldLabel}>To</span>
            <select
              value={toKnown ? draft.to : ''}
              onChange={(e) => edit({ to: e.target.value })}
              onKeyDown={leaveOnEscape}
              disabled={busy !== null}
              aria-label="Who it goes to"
              className={`${field} min-w-0`}
            >
              {!toKnown && <option value="" disabled>Pick who it goes to</option>}
              {options.map((o) => <option key={`${o.kind}:${o.address}`} value={o.address}>{recipientText(o)}</option>)}
            </select>
          </label>
        ) : toLine ? (
          <div className="flex min-w-0 items-center gap-2 text-xs">
            <span className={fieldLabel}>To</span>
            <span className="min-w-0 truncate text-zinc-300" title={toLine}>{toLine}</span>
          </div>
        ) : null)}
        {mail && (editable ? (
          <label className="flex items-center gap-2">
            <span className={fieldLabel}>Subject</span>
            <input
              value={draft.subject}
              maxLength={SUBJECT_MAX}
              onChange={(e) => edit({ subject: e.target.value })}
              onKeyDown={leaveOnEscape}
              disabled={busy !== null}
              aria-label="Subject"
              className={`${field} min-w-0`}
            />
          </label>
        ) : draft.subject ? (
          <div className="flex min-w-0 items-center gap-2 text-xs">
            <span className={fieldLabel}>Subject</span>
            <span className="min-w-0 truncate text-zinc-300">{draft.subject}</span>
          </div>
        ) : null)}
        {editable ? (
          <textarea
            value={draft.text}
            maxLength={TEXT_MAX}
            rows={phone ? 7 : 6}
            onChange={(e) => edit({ text: e.target.value })}
            onKeyDown={leaveOnEscape}
            disabled={busy !== null}
            aria-label={mail ? 'The email' : reminder ? 'The reminder' : 'The message'}
            className={`w-full resize-y rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-2 leading-relaxed text-zinc-200 focus:border-teal-500/60 focus:outline-none disabled:opacity-60 ${
              phone ? 'text-base' : 'text-sm'}`}
          />
        ) : draft.text ? (
          <p className="whitespace-pre-wrap break-words rounded-lg border border-zinc-800 bg-zinc-900/60 px-2.5 py-2 text-sm leading-relaxed text-zinc-300">
            {draft.text}
          </p>
        ) : null}
      </div>

      <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">
        {mail
          ? 'Email is only ever a draft: Approve puts it in your drafts, and Captain never sends it — you do.'
          : reminder
            ? 'Nothing goes out until you approve. Reject drops it.'
            : toLine
              ? `Nothing goes out until you approve — then it is sent to ${toLine}.`
              : `Nothing goes out until you approve${channel ? ` — then it is sent on ${channelLabel(channel)}` : ''}.`}
      </p>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <button
          onClick={() => answer(true)}
          disabled={busy !== null || !toKnown || blank}
          className={`${btn} bg-emerald-600 text-white hover:bg-emerald-500`}
          title={mail ? 'Save it as a draft in your mailbox' : 'Send it'}
        >
          {busy === 'approve' && <Loader2 className="h-3 w-3 animate-spin" />}{dirty ? 'Save & approve' : 'Approve'}
        </button>
        <button onClick={() => answer(false)} disabled={busy !== null} className={`${btn} bg-zinc-800 text-zinc-300 hover:bg-zinc-700`}>
          {busy === 'reject' && <Loader2 className="h-3 w-3 animate-spin" />}Reject
        </button>
        {editable && dirty && (
          <>
            <button onClick={save} disabled={busy !== null || !toKnown || blank} className={`${btn} border border-zinc-800 text-zinc-300 hover:bg-zinc-800/60`}>
              {busy === 'save' && <Loader2 className="h-3 w-3 animate-spin" />}Save changes
            </button>
            <button onClick={() => edit(base)} disabled={busy !== null} className={`${btn} text-zinc-400 hover:bg-zinc-800/60`}>
              Undo changes
            </button>
          </>
        )}
        {saved && !dirty && (
          <span className="flex items-center gap-1 text-[11px] text-emerald-700 dark:text-emerald-400"><Check className="h-3 w-3" />Saved</span>
        )}
      </div>
      {error && <p className="mt-1.5 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

const KINDS: { id: NextStepKind; label: string; detail: string }[] = [
  { id: 'auto', label: 'Whatever fits best', detail: 'The board picks from what the task needs' },
  { id: 'reply_with_result', label: 'Reply with the result', detail: 'Where the task came in' },
  { id: 'ask_clarifying', label: 'Ask a question back', detail: 'What the agent needs to go on' },
  { id: 'draft_email', label: 'An email draft', detail: 'It stays a draft — never sent' },
  { id: 'remind', label: 'A reminder', detail: 'To whoever owes the next move' },
]

/** "Suggest next step": asks the board for one now (drafted on the agent's
 *  own model) — it then waits for your Approve, like any other; or the
 *  board says why it made none. */
export function SuggestNextStep({ taskId, phone, kinds, covered, askRef, onProposed }: {
  taskId: string
  phone: boolean
  /** Filled with what the button does, for its key. */
  askRef: { current: (() => void) | null }
  /** What the server can make of it: a kind nobody can be reached for isn't offered. */
  kinds: NextStepKind[]
  /** Something opened over the task: Escape is that one's. */
  covered: boolean
  /** One was made: the task reads again (its card shows). */
  onProposed: () => void
}) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState<NextStepKind | null>(null)
  const [reason, setReason] = useState('')
  const [error, setError] = useState('')

  // Escape closes the list of kinds, not the board (it marks the key used).
  useEffect(() => {
    if (!open || covered) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); setOpen(false) } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, covered])

  const ask = async (kind: NextStepKind) => {
    setBusy(kind)
    setOpen(false)
    setReason('')
    setError('')
    try {
      const res = await taskboardApi.propose(taskId, kind)
      if (res.proposal) onProposed()
      else setReason(res.reason || 'Nothing fits this task right now.')
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(null)
    }
  }

  useEffect(() => {
    askRef.current = busy === null ? () => { void ask('auto') } : null
    return () => { askRef.current = null }
  })

  const base = `flex items-center gap-1.5 border border-zinc-800 font-medium text-teal-600 transition-colors hover:bg-teal-500/10 disabled:opacity-60 dark:text-teal-400 ${
    phone ? 'h-11 text-sm' : 'py-1.5 text-xs'}`
  return (
    <section className="mt-4">
      <div className="flex items-stretch">
        <button onClick={() => ask('auto')} disabled={busy !== null} className={`${base} rounded-l-lg px-3`}
          title="Ask for a next step for this task — it waits for your Approve">
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
          {busy ? 'Thinking of a next step…' : 'Suggest next step'}
          {!busy && <KeyHint k={TASK_KEYS.nextStep} />}
        </button>
        <button
          onClick={() => setOpen((o) => !o)}
          disabled={busy !== null}
          aria-expanded={open}
          aria-label="Pick what kind of next step"
          title="Pick what kind of next step"
          className={`${base} -ml-px rounded-r-lg px-2`}
        >
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>
      {open && (
        <ul role="menu" aria-label="Kinds of next step" className="mt-1.5 overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950">
          {KINDS.filter((k) => kinds.includes(k.id)).map((k) => (
            <li key={k.id}>
              <button role="menuitem" onClick={() => ask(k.id)}
                className={`flex w-full items-baseline gap-2 px-3 text-left transition-colors hover:bg-zinc-800/60 ${phone ? 'py-3' : 'py-2'}`}>
                <span className="text-sm text-zinc-200">{k.label}</span>
                <span className="min-w-0 truncate text-[11px] text-zinc-500">{k.detail}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {reason && <p className="mt-1.5 text-xs leading-relaxed text-zinc-400"><span className="font-medium text-zinc-300">No next step:</span> {reason}</p>}
      {error && <p className="mt-1.5 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
      <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-500">
        Nothing leaves without your Approve — and an email only ever becomes a draft, never sent.
      </p>
    </section>
  )
}

/** "Next move → Agent": which agent and what it should do. Nothing goes to it
 *  until Hand over; what was written stays if this is closed and opened again. */
export function HandOverForm({ taskAgent, phone, words, onWords, onHandOver, onCancel, count = 1, children }: {
  /** The task's agent now: the one picked first, when it can take it. */
  taskAgent: string
  phone: boolean
  words: string
  onWords: (words: string) => void
  onHandOver: (agent: string, words: string) => Promise<void>
  onCancel: () => void
  /** Several tasks at once: one set of words for all, one at a time. */
  count?: number
  /** What goes along, above the words. */
  children?: ReactNode
}) {
  const many = count > 1
  const [agents, setAgents] = useState<HandOverAgent[] | null>(null)
  const [agent, setAgent] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let gone = false
    taskboardApi.agents()
      .then((res) => {
        if (gone) return
        setAgents(res.agents)
        setAgent(res.agents.some((a) => a.slug === taskAgent) ? taskAgent : res.agents[0]?.slug ?? '')
      })
      .catch((e) => { if (!gone) { setAgents([]); setError(errText(e)) } })
    return () => { gone = true }
  }, [taskAgent])

  const ready = !!agent && !!words.trim() && !busy
  const send = async () => {
    if (!ready) return
    setBusy(true)
    setError('')
    try {
      await onHandOver(agent, words.trim())
    } catch (e) {
      setError(errText(e))
      setBusy(false)
    }
  }
  const field = `w-full rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-zinc-200 placeholder-zinc-500 focus:border-teal-500/60 focus:outline-none disabled:opacity-50 ${
    phone ? 'text-base' : 'text-sm'}`

  return (
    <div className="col-span-2 flex flex-col gap-2">
      <div className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">{many ? `Hand ${count} tasks to an agent` : 'Hand to an agent'}</div>
      {agents === null ? (
        <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />
      ) : agents.length === 0 ? (
        <p className="text-xs leading-relaxed text-zinc-400">{error || 'None of your agents is on. Power one on, then hand this over.'}</p>
      ) : (
        <>
          <select value={agent} onChange={(e) => setAgent(e.target.value)} disabled={busy} aria-label="Which agent" className={field}>
            {agents.map((a) => <option key={a.slug} value={a.slug}>{a.name}{a.asleep ? ' (asleep, it will wake)' : ''}</option>)}
          </select>
          {children}
          <textarea
            autoFocus={!phone}
            value={words}
            onChange={(e) => onWords(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); void send() } }}
            disabled={busy}
            rows={4}
            maxLength={4000}
            aria-label="What the agent should do"
            placeholder={many ? 'What should it do on each? For example: thank them and politely decline.' : 'What should it do? Add anything it needs to know.'}
            className={`${field} resize-y`}
          />
          <p className="text-[11px] leading-relaxed text-zinc-500">
            {many
              ? 'These words go with every task. This is your own request: it starts now, on one task at a time, each with its own details, and it will write email or messages if that is what you ask. You can stop the rest at any point.'
              : 'This is your own request: it starts now, and it will write email or messages if that is what you ask. It also gets this task\'s details.'}
          </p>
          {error && <p className="text-[11px] text-red-600 dark:text-red-400">{error}</p>}
        </>
      )}
      <div className="grid grid-cols-2 gap-2">
        <button onClick={() => void send()} disabled={!ready}
          className="flex items-center justify-center gap-1.5 rounded-lg bg-teal-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-teal-500 disabled:opacity-60">
          {busy && <Loader2 className="h-4 w-4 animate-spin" />}{many ? `Hand over ${count}` : 'Hand over'}<KeyHint k={SEND_KEY} onPrimary />
        </button>
        <button onClick={onCancel} disabled={busy}
          className="flex items-center justify-center gap-1.5 rounded-lg border border-zinc-800 px-3 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:opacity-50">
          Cancel<KeyHint k="Esc" />
        </button>
      </div>
    </div>
  )
}
