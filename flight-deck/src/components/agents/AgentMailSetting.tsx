import { useEffect, useState } from 'react'
import { IdCard, Loader2, Mail } from 'lucide-react'
import { SwitchTrack } from '../common/SwitchTrack'
import { KioskAgentIdentityDialog } from '../layout/KioskAgentIdentityDialog'
import { colleaguesApi, type ColleagueAgent, type ColleaguesView, type GmailReplies } from '../../services/colleaguesApi'
import { useNotificationStore } from '../../stores/notificationStore'

// ── Replies in your Gmail ─────────────────────────────────────────────
//
// Email that came to the agent in your Gmail: its reply is a draft you send
// (off), or it sends the reply itself — only to you, or to any sender whose
// address Gmail verified. Always the one reply, to that sender, on that email,
// within Connections → Google → Email sending (on/off, allowlist, daily limit).

export const GMAIL_REPLY_CHOICES: { id: GmailReplies; label: string; hint: string }[] = [
  { id: 'off', label: 'Drafts', hint: 'It saves its reply as a draft; you send it.' },
  { id: 'me', label: 'Send to you', hint: 'When you email it, it sends its reply to you. Anyone else gets a draft.' },
  { id: 'verified', label: 'Send to anyone', hint: 'It sends its reply to whoever wrote, when Gmail verified their address. Their reply can carry what it found — keep an allowlist in Connections → Google → Email sending.' },
]

export function GmailReplyChoice({ value, onChange, disabled = false, compact = false }: {
  value: GmailReplies
  onChange: (v: GmailReplies) => void
  disabled?: boolean
  compact?: boolean
}) {
  const current = GMAIL_REPLY_CHOICES.find((c) => c.id === value) ?? GMAIL_REPLY_CHOICES[0]
  return (
    <div>
      <div className={`font-medium text-zinc-300 ${compact ? 'text-[11px]' : 'text-xs'}`}>Replies in your Gmail</div>
      <div role="radiogroup" aria-label="Replies in your Gmail" className="mt-1.5 grid grid-cols-3 rounded-lg bg-zinc-900 p-0.5">
        {GMAIL_REPLY_CHOICES.map((c) => (
          <button
            key={c.id} type="button" role="radio" aria-checked={value === c.id} disabled={disabled}
            onClick={() => onChange(c.id)}
            className={`rounded-md px-2 py-1.5 text-xs font-medium transition-colors disabled:cursor-wait ${
              value === c.id ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}
          >
            {c.label}
          </button>
        ))}
      </div>
      <p className={`mt-1 leading-relaxed text-zinc-500 ${compact ? 'text-[11px]' : 'text-xs'}`}>{current.hint}</p>
    </div>
  )
}

// ── One agent's email setting ─────────────────────────────────────────
//
// The switch for whether this agent picks up the email meant for it — its own
// inbox, its aliases, and mail in your Gmail, Outlook or IMAP inboxes that
// names it in the subject or addresses it in the body. On for every agent
// unless switched off here (or in ⋯ → Identity & email, which also sets its
// own inbox, aliases and the names it answers to). Saves at once; no restart.
// Flight Deck: colleagues.py (the setting), agent_mail.py (the watcher).

export function AgentMailSetting({ slug, agentName, compact = false }: {
  /** The process agent's slug. */
  slug: string
  agentName: string
  /** Smaller text, for a dialog's side column. */
  compact?: boolean
}) {
  const [view, setView] = useState<ColleaguesView | null>(null)
  const [me, setMe] = useState<ColleagueAgent | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [identityOpen, setIdentityOpen] = useState(false)
  const [rev, setRev] = useState(0)

  useEffect(() => {
    let live = true
    colleaguesApi.list().then((v) => {
      if (!live) return
      setView(v)
      setMe(v.agents.find((a) => a.slug === slug) || null)
    }).catch((e) => live && setError(String(e?.message || e)))
    return () => { live = false }
  }, [slug, rev])

  const save = async (patch: { pick_up_mail?: boolean; gmail_replies?: GmailReplies }) => {
    if (!me || saving) return
    const before = me
    setSaving(true)
    setError('')
    setMe({ ...me, ...patch })
    try {
      const { agent } = await colleaguesApi.save(slug, patch)
      setMe(agent)
    } catch (e) {
      setMe(before)
      const msg = String((e as Error)?.message || e)
      setError(msg)
      useNotificationStore.getState().add('error', 'Email setting not saved', msg)
    } finally {
      setSaving(false)
    }
  }
  const toggle = () => save({ pick_up_mail: !me?.pick_up_mail })

  if (error && !me) return <p className="text-[11px] text-red-400">{error}</p>
  if (!view || !me) {
    return view ? null : <div className="flex items-center text-xs text-zinc-500"><Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />Loading…</div>
  }

  const sources = [view.mail.gmail && 'Gmail', view.mail.outlook && 'Outlook', view.mail.imap && 'your IMAP inboxes']
    .filter(Boolean) as string[]
  const names = me.names.join(', ')
  const where = [me.inbox_email && `its own inbox ${me.inbox_email}`, ...me.addresses].filter(Boolean).join(', ')
  const text = compact ? 'text-[11px]' : 'text-xs'

  return (
    <div>
      <button
        type="button" role="switch" aria-checked={me.pick_up_mail} onClick={toggle} disabled={saving}
        aria-label={`Pick up email meant for ${agentName}`}
        className="flex w-full items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-3 text-left hover:border-zinc-700 disabled:cursor-wait"
      >
        <span className="mt-0.5"><SwitchTrack on={me.pick_up_mail} /></span>
        <span className="min-w-0">
          <span className="flex items-center gap-1.5 text-sm font-medium text-zinc-100">
            <Mail className="h-3.5 w-3.5 text-rose-400" />Pick up email meant for {agentName}
            {saving && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
          </span>
          <span className={`mt-1 block leading-relaxed text-zinc-500 ${text}`}>
            {me.pick_up_mail
              ? `It starts on email ${where ? `sent to ${where}, and ` : ''}in ${sources.length ? sources.join(', ') : 'your mail'} that names “${names}” in the subject or addresses it in the body. Each one is a task on your Taskboard.`
              : 'Off: email that names it stays where it is.'}
            {me.pick_up_mail && !sources.length && !me.inbox_email && ' Connect Gmail, Outlook or an IMAP inbox under Connections first.'}
          </span>
          <span className={`mt-1 block leading-relaxed text-zinc-600 ${text}`}>
            {me.pick_up_mail && view.mail.gmail
              ? 'Replies from its own inbox go out only if that inbox may send; in Gmail, as set below; anywhere else they wait as drafts for you.'
              : 'Replies from its own inbox go out only if that inbox may send; anywhere else they wait as drafts for you.'}
          </span>
        </span>
      </button>
      {me.pick_up_mail && view.mail.gmail && (
        <div className="mt-2.5 px-1">
          <GmailReplyChoice value={me.gmail_replies} onChange={(v) => void save({ gmail_replies: v })} disabled={saving} compact={compact} />
        </div>
      )}
      <button
        type="button" onClick={() => setIdentityOpen(true)}
        className={`mt-1.5 flex items-center gap-1.5 rounded-md px-1 py-0.5 text-violet-300 hover:text-violet-200 ${text}`}
      >
        <IdCard className="h-3.5 w-3.5" />Its own inbox, aliases and names…
      </button>
      {error && <p className="mt-1 text-[11px] text-red-400">{error}</p>}
      {identityOpen && (
        <KioskAgentIdentityDialog slug={slug} agentName={agentName}
          onClose={() => { setIdentityOpen(false); setRev((n) => n + 1) }} />
      )}
    </div>
  )
}
