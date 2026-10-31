import { useEffect, useState } from 'react'
import { Loader2, Mail, Send } from 'lucide-react'
import { adminHeaders } from './deckConfig'
import { PROVIDER_LABEL, PROVIDERS, type MailOptions, type Provider } from './InvitesAdmin'

// Users → a user → Ask for feedback: an email asking how the deck is working
// for them. The admin reads and edits the draft, and it goes from their own
// connected Gmail or Outlook, so the reply reaches them directly
// (captain_claw/flight_deck/feedback_request_routes.py).

interface Draft {
  to: string
  subject: string
  body: string
  asked_at: string
  mail: MailOptions
}

interface SendResult {
  ok: boolean
  email: { from: string; error: string }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: adminHeaders(), credentials: 'include', ...init })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`)
  }
  return res.json()
}

export function FeedbackRequestPanel({ userId, onSent, onClose }: {
  userId: string
  onSent: (from: string) => void
  onClose: () => void
}) {
  const [draft, setDraft] = useState<Draft | null>(null)
  const [subject, setSubject] = useState('')
  const [body, setBody] = useState('')
  const [via, setVia] = useState<Provider | ''>('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const url = `/fd/admin/users/${userId}/feedback-request`

  useEffect(() => {
    let live = true
    call<Draft>(url)
      .then((d) => {
        if (!live) return
        setDraft(d)
        setSubject(d.subject)
        setBody(d.body)
        setVia(PROVIDERS.find((p) => d.mail[p].available) ?? '')
      })
      .catch((e) => { if (live) setError(e instanceof Error ? e.message : String(e)) })
    return () => { live = false }
  }, [url])

  const send = async () => {
    if (!via) return
    setSending(true); setError('')
    try {
      const r = await call<SendResult>(url, {
        method: 'POST', body: JSON.stringify({ subject, body, send_via: via }),
      })
      if (r.ok) { onSent(r.email.from); return }
      setError(r.email.error || 'The email did not go out.')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
    setSending(false)
  }

  const input = 'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none'
  const canSend = !!via && !sending && subject.trim() !== '' && body.trim() !== ''

  return (
    <div className="rounded-lg border border-violet-800/40 bg-zinc-900/50 p-3 space-y-2">
      <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-300">
        <Mail className="h-3.5 w-3.5 text-violet-400" />
        Ask for feedback
      </div>
      {!draft && !error && <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />}
      {draft && (
        <>
          <p className="text-xs text-zinc-500">
            To {draft.to}. Their reply comes straight to your mailbox.
            {draft.asked_at && ` You last asked them on ${new Date(draft.asked_at).toLocaleDateString()}.`}
          </p>
          {PROVIDERS.some((p) => draft.mail[p].available) ? (
            <label className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
              Send from
              <select
                value={via}
                onChange={(e) => setVia(e.target.value as Provider)}
                className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              >
                {PROVIDERS.filter((p) => draft.mail[p].available).map((p) => (
                  <option key={p} value={p}>
                    {PROVIDER_LABEL[p]}{draft.mail[p].from ? ` · ${draft.mail[p].from}` : ''}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <div className="space-y-1 text-xs text-amber-500/80">
              {PROVIDERS.map((p) => <p key={p}>{draft.mail[p].reason}</p>)}
            </div>
          )}
          <input
            type="text"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            maxLength={200}
            aria-label="Subject"
            className={input}
          />
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            rows={13}
            maxLength={5000}
            aria-label="Message"
            className={input}
          />
        </>
      )}
      {error && <p className="text-xs text-red-400">{error}</p>}
      <div className="flex items-center gap-2 pt-1">
        {draft && (
          <button
            onClick={send}
            disabled={!canSend}
            className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
          >
            <Send className="h-3 w-3" />
            {sending ? 'Sending…' : 'Send email'}
          </button>
        )}
        <button
          onClick={onClose}
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}
