import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { Smartphone, Loader2, ChevronDown, ChevronRight, Check, AlertCircle, RefreshCw } from 'lucide-react'
import {
  getMyWhatsAppWeb,
  linkWhatsAppWeb,
  whatsAppWebPairCode,
  reconnectWhatsAppWeb,
  cancelWhatsAppWeb,
  unlinkWhatsAppWeb,
  deleteWhatsAppWebKept,
  getWhatsAppWebAdmin,
  setWhatsAppWebAllowed,
  setWhatsAppWebRetention,
  getWhatsAppWebRecent,
  getWhatsAppWebSending,
  setWhatsAppWebSending,
  getWhatsAppWebAutoreplyActivity,
  type WhatsAppWebAutoreplyEntry,
  type MyWhatsAppWeb,
  type WhatsAppWebSending,
  type WhatsAppWebSendingChanges,
  type WhatsAppWebMessage,
  type WhatsAppWebAdmin,
  type WhatsAppWebState,
} from '../../services/whatsappWeb'

// WhatsApp Web (captain_claw/flight_deck/whatsapp_web/): the user's own
// WhatsApp account linked as a device — scan a QR, or type a pairing code on
// the phone — held by Flight Deck's WhatsApp Web service. Off until an admin
// allows it on the deck; every new link needs the user's consent, because this
// is an unofficial client. The history the phone sends after linking and every
// message since are kept (encrypted) for the user's agents, who read them with
// the whatsapp_web tool. Agents send only when the user turns Sending on, only to
// the chats on their whitelist, within a daily limit, signed; nothing is ever
// marked read or shown as "online" in the user's name. In whitelisted chats the
// user marks for auto-reply, the agent they pick answers by itself when it's
// mentioned (captain_claw/flight_deck/whatsapp_web_autoreply.py). When a link ends
// its messages stay, because numbers get banned: disconnecting asks, a ban or a
// removal on the phone keeps them, and only the user deletes them.

type Tone = 'on' | 'wait' | 'warn' | 'off'

const STATE: Record<WhatsAppWebState, { text: string; tone: Tone }> = {
  off: { text: 'Not linked', tone: 'off' },
  pairing: { text: 'Waiting for your phone', tone: 'wait' },
  connecting: { text: 'Connecting', tone: 'wait' },
  connected: { text: 'Linked', tone: 'on' },
  disconnected: { text: 'Reconnecting', tone: 'wait' },
  expired: { text: 'Code expired', tone: 'off' },
  failed: { text: 'Couldn’t link', tone: 'warn' },
  logged_out: { text: 'Unlinked', tone: 'warn' },
  stream_replaced: { text: 'Taken over elsewhere', tone: 'warn' },
  temporary_ban: { text: 'Temporarily banned', tone: 'warn' },
  outdated: { text: 'Needs an update', tone: 'warn' },
}

const PILL: Record<Tone, string> = {
  on: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-500 border-emerald-500/30',
  wait: 'bg-sky-500/15 text-sky-600 dark:text-sky-400 border-sky-500/30',
  warn: 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30',
  off: 'bg-zinc-800 text-zinc-400 border-zinc-700',
}

const BOX = 'rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2.5 text-xs text-zinc-300'
const BUTTON = 'flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs disabled:opacity-50'
const PRIMARY = `${BUTTON} bg-violet-600 text-white hover:bg-violet-500`
const SECONDARY = `${BUTTON} bg-zinc-800 text-zinc-200 hover:bg-zinc-700`

// States the user can start a fresh link from (after consenting again).
const STARTABLE: WhatsAppWebState[] = ['off', 'expired', 'failed', 'logged_out']
// States where the card asks the service every couple of seconds: the QR
// rotates and the state moves on by itself.
const LIVE: WhatsAppWebState[] = ['pairing', 'connecting', 'disconnected']

function when(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function day(ts: number | null | undefined): string {
  if (!ts) return ''
  try { return new Date(ts * 1000).toLocaleDateString() } catch { return '' }
}

function stamp(ts: number): string {
  try { return new Date(ts * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }) } catch { return '' }
}

const KIND_LABEL: Record<string, string> = {
  image: 'image', video: 'video', audio: 'audio', voice: 'voice note', document: 'document', sticker: 'sticker',
  video_note: 'video note', location: 'location', contact: 'contact', contacts: 'contacts', poll: 'poll', event: 'event',
}

const SEND_STATUS: Record<string, string> = { sent: 'sent', unknown: 'not confirmed', failed: 'failed', sending: 'sending' }

const AGENT_REASON: Record<string, string> = {
  agent_no_token: 'no web token', agent_no_port: 'no port', agent_managed: 'a worker', agent_not_found: 'not found',
}

const triggerList = (text: string) => text.split(',').map((t) => t.trim().replace(/\s+/g, ' ')).filter(Boolean)

function Sending() {
  const [open, setOpen] = useState(false)
  const [view, setView] = useState<WhatsAppWebSending | null>(null)
  const [draft, setDraft] = useState<WhatsAppWebSendingChanges>({})
  const [auto, setAuto] = useState<{ chats: string[]; agent: string; triggers: string }>({ chats: [], agent: '', triggers: '' })
  const [filter, setFilter] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const apply = (v: WhatsAppWebSending) => {
    setView(v)
    setDraft({ allowed: v.allowed ?? [], daily_limit: v.daily_limit, signature: v.signature, signature_on: v.signature_on })
    setAuto({ chats: v.autoreply?.chats ?? [], agent: v.autoreply?.agent ?? '', triggers: (v.autoreply?.triggers ?? []).join(', ') })
  }
  const load = useCallback(async () => {
    try { apply(await getWhatsAppWebSending()); setError('') } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
  }, [])
  useEffect(() => { if (open) load() }, [open, load])

  const save = async (changes: WhatsAppWebSendingChanges, done?: string) => {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const v = await setWhatsAppWebSending(changes)
      apply(v)
      const dropped = v.dropped?.length ? ` ${v.dropped.length} left out: there's no conversation with them yet.` : ''
      setNotice((done ?? 'Saved.') + dropped)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const allowed = new Set(draft.allowed ?? [])
  const answered = new Set(auto.chats.filter((j) => allowed.has(j)))
  const toggleChat = (jid: string) => {
    const next = new Set(allowed)
    if (next.has(jid)) next.delete(jid); else next.add(jid)
    setDraft((d) => ({ ...d, allowed: [...next] }))
  }
  const toggleAuto = (jid: string) => {
    const next = new Set(answered)
    if (next.has(jid)) next.delete(jid); else next.add(jid)
    setAuto((a) => ({ ...a, chats: [...next] }))
  }
  const autoChanges = { chats: [...answered], agent: auto.agent, triggers: triggerList(auto.triggers) }
  const autoView = view?.autoreply
  const answerer = autoView?.agents.find((a) => a.ref === auto.agent)
  // Whitelisted chats first, auto-reply ones on top — by what's saved, so a row
  // doesn't jump away while it's being ticked.
  const saved = new Set(view?.allowed ?? [])
  const savedAuto = new Set(view?.autoreply?.chats ?? [])
  const rank = (jid: string) => (savedAuto.has(jid) ? 0 : saved.has(jid) ? 1 : 2)
  const chats = (view?.chats ?? []).filter((c) => {
    const q = filter.trim().toLowerCase()
    return !q || c.title.toLowerCase().includes(q) || (c.number || '').includes(q.replace(/\s/g, ''))
  }).sort((a, b) => rank(a.jid) - rank(b.jid))
  const dirty = !!view && (
    JSON.stringify([...(view.allowed ?? [])].sort()) !== JSON.stringify([...allowed].sort())
    || draft.daily_limit !== view.daily_limit || draft.signature !== view.signature || draft.signature_on !== view.signature_on
    || JSON.stringify([...(autoView?.chats ?? [])].sort()) !== JSON.stringify([...answered].sort())
    || auto.agent !== (autoView?.agent ?? '')
    || JSON.stringify(autoChanges.triggers) !== JSON.stringify(autoView?.triggers ?? []))

  return (
    <div className={`${BOX} space-y-2`}>
      <button onClick={() => setOpen((v) => !v)} className="flex w-full items-center gap-2 text-left">
        <ChevronRight className={`h-3 w-3 transition-transform ${open ? 'rotate-90' : ''}`} />
        <span className="font-medium text-zinc-100">Sending</span>
        {view?.available && (
          <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border ${PILL[view.enabled && !view.kill_switch ? 'on' : 'off']}`}>
            {view.kill_switch ? 'switched off on this deck' : view.enabled ? `on · ${allowed.size} chat${allowed.size === 1 ? '' : 's'}` : 'off'}
          </span>
        )}
      </button>
      {open && view && !view.available && <div className="text-zinc-500">Sending needs a linked WhatsApp.</div>}
      {open && view?.available && (
        <div className="space-y-3">
          <p className="text-zinc-400">
            When it’s on, your agents send text messages without asking you first — but only to the chats
            you tick here, at most {view.daily_limit} a day, never to someone you haven’t talked with, and
            each one ends with your signature. You get a notification for every message they send.
          </p>
          {view.kill_switch && (
            <div className="flex items-start gap-1.5 text-amber-600 dark:text-amber-400">
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-px" />
              An admin switched sending off for this deck (FD_WHATSAPP_WEB_SEND=off).
            </div>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => {
                if (!view.enabled && !confirm('Let your agents send WhatsApp messages from your account to the chats you whitelist, without asking each time? WhatsApp bans some accounts that automate sending; keep the list short.')) return
                save({ enabled: !view.enabled }, view.enabled ? 'Sending is off.' : 'Sending is on.')
              }}
              disabled={busy || view.kill_switch}
              className={view.enabled ? SECONDARY : PRIMARY}
            >
              {busy && <Loader2 className="h-3 w-3 animate-spin" />}
              {view.enabled ? 'Turn sending off' : 'Turn sending on'}
            </button>
            <span className="text-zinc-500">{view.sent_today ?? 0} sent in the last 24 hours</span>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="font-medium text-zinc-200">Whitelist</span>
              <span className="text-zinc-500">{allowed.size} of {view.chats?.length ?? 0} chats</span>
              <input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Find a chat"
                className="ml-auto w-40 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              />
            </div>
            <div className="max-h-56 overflow-y-auto rounded-md border border-zinc-800 bg-zinc-950 divide-y divide-zinc-800/60">
              {chats.length === 0 && <div className="px-3 py-2 text-zinc-500">No chats {filter ? 'match' : 'kept yet'}.</div>}
              {chats.slice(0, 300).map((c) => (
                <div key={c.jid} className="flex items-center gap-2 px-3 py-1.5 hover:bg-zinc-900/60">
                  <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2">
                    <input type="checkbox" checked={allowed.has(c.jid)} onChange={() => toggleChat(c.jid)} className="h-3.5 w-3.5 accent-violet-600" />
                    <span className="truncate text-zinc-200">{c.title}</span>
                  </label>
                  {allowed.has(c.jid) && (
                    <label className="flex shrink-0 cursor-pointer items-center gap-1 text-zinc-400" title="The agent answers here by itself when it's mentioned">
                      <input type="checkbox" checked={answered.has(c.jid)} onChange={() => toggleAuto(c.jid)} className="h-3 w-3 accent-violet-600" />
                      auto-reply
                    </label>
                  )}
                  <span className="shrink-0 text-zinc-500">{c.kind === 'group' ? 'group' : c.number}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <label className="flex items-center gap-1.5">
              At most
              <input
                type="number" min={1} max={200} value={draft.daily_limit ?? 30}
                onChange={(e) => setDraft((d) => ({ ...d, daily_limit: Number(e.target.value) || 1 }))}
                className="w-16 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              />
              a day
            </label>
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={draft.signature_on ?? true}
                onChange={(e) => setDraft((d) => ({ ...d, signature_on: e.target.checked }))} className="h-3.5 w-3.5 accent-violet-600" />
              Sign every message:
            </label>
            <input
              value={draft.signature ?? ''}
              onChange={(e) => setDraft((d) => ({ ...d, signature: e.target.value }))}
              maxLength={120}
              disabled={!(draft.signature_on ?? true)}
              className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none disabled:opacity-50"
            />
          </div>
          {autoView && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="font-medium text-zinc-200">Auto-replies</span>
                <span className="text-zinc-500">{answered.size} chat{answered.size === 1 ? '' : 's'}</span>
              </div>
              <p className="text-zinc-400">
                In the chats marked auto-reply, {answerer?.name ?? 'the agent you pick'} answers by itself when someone
                mentions it by name or a trigger word, or replies to one of its messages (you can call it in the same
                way). For 15 minutes after that it
                joins in when it has something useful to add, and steps back as soon as you write there yourself. It sees
                only that chat and can search the web — never your mail, files, calendar, memory or other chats. Its
                messages count toward the daily limit and carry your signature.
              </p>
              {autoView.kill_switch && (
                <div className="flex items-start gap-1.5 text-amber-600 dark:text-amber-400">
                  <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-px" />
                  An admin switched auto-replies off for this deck (FD_WHATSAPP_WEB_AUTOREPLY=off).
                </div>
              )}
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                <label className="flex items-center gap-1.5">
                  Answers as
                  <select
                    value={auto.agent}
                    onChange={(e) => setAuto((a) => ({ ...a, agent: e.target.value }))}
                    className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                  >
                    <option value="">— pick an agent —</option>
                    {autoView.agents.map((a) => (
                      <option key={a.ref} value={a.ref} disabled={!a.eligible}>
                        {a.name}{!a.eligible ? ` (${AGENT_REASON[a.reason] ?? 'can’t answer'})` : !a.running ? ' (stopped)' : ''}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex min-w-0 flex-1 items-center gap-1.5">
                  <span className="shrink-0">Also answers to</span>
                  <input
                    value={auto.triggers}
                    onChange={(e) => setAuto((a) => ({ ...a, triggers: e.target.value }))}
                    placeholder="words, separated by commas"
                    className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                  />
                </label>
              </div>
              {answered.size > 0 && !auto.agent && <div className="text-amber-600 dark:text-amber-400">Pick the agent that answers.</div>}
              {answered.size > 0 && answerer && !answerer.running && (
                <div className="text-amber-600 dark:text-amber-400">{answerer.name} is stopped: it won’t answer until it runs.</div>
              )}
            </div>
          )}
          <div className="flex items-center gap-2">
            <button onClick={() => save(autoView ? { ...draft, autoreply: autoChanges } : draft)} disabled={busy || !dirty} className={PRIMARY}>
              {busy && <Loader2 className="h-3 w-3 animate-spin" />} Save
            </button>
            {notice && <span className="text-emerald-600 dark:text-emerald-500">{notice}</span>}
          </div>

          {(view.sends?.length ?? 0) > 0 && (
            <div className="space-y-1">
              <div className="font-medium text-zinc-200">Sent by your agents</div>
              <div className="max-h-48 overflow-y-auto rounded-md border border-zinc-800 bg-zinc-950 divide-y divide-zinc-800/60">
                {view.sends!.map((m, i) => (
                  <div key={`${m.ts}-${i}`} className="px-3 py-1.5">
                    <div className="text-zinc-500">
                      {stamp(m.ts)} · {m.chat} · {m.agent || 'agent'} ·{' '}
                      <span className={m.status === 'sent' ? '' : 'text-amber-600 dark:text-amber-400'}>{SEND_STATUS[m.status] ?? m.status}</span>
                    </div>
                    <div className="text-zinc-300 whitespace-pre-wrap break-words">{m.text}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      {error && <div className="text-red-600 dark:text-red-400">{error}</div>}
    </div>
  )
}

// What each auto-reply outcome means, for the activity list.
function outcomeLabel(outcome: string): { text: string; tone: Tone } {
  if (outcome === 'sent') return { text: 'answered', tone: 'on' }
  if (outcome === 'skip:silent') return { text: 'stayed silent', tone: 'off' }
  if (outcome === 'skip:not addressed') return { text: 'not for it', tone: 'off' }
  if (outcome === 'skip:cap') return { text: 'hourly limit', tone: 'warn' }
  if (outcome === 'skip:no context') return { text: 'no chat history', tone: 'warn' }
  if (outcome === 'skip:search refused') return { text: 'search refused', tone: 'warn' }
  if (outcome.startsWith('skip:refused')) return { text: 'send refused', tone: 'warn' }
  if (outcome === 'off:switch') return { text: 'switched off', tone: 'warn' }
  if (outcome === 'off:agent_stopped') return { text: 'agent stopped', tone: 'warn' }
  if (outcome.startsWith('off:')) return { text: 'no agent to answer', tone: 'warn' }
  return { text: 'couldn’t answer', tone: 'warn' }
}

const STEP_LABEL: Record<string, string> = {
  reply: 'Answer', silent: 'Stayed silent', search: 'Searched the web for', skip: 'Unusable answer', error: 'No answer',
}

function ActivityEntry({ e }: { e: WhatsAppWebAutoreplyEntry }) {
  const [open, setOpen] = useState(false)
  const label = outcomeLabel(e.outcome)
  const steps = e.steps ?? []
  const why = [...steps].reverse().find((s) => s.why)?.why
  const final = steps[steps.length - 1]
  const how = e.mode === 'mention'
    ? (e.called?.length ? `called by ${e.called.map((w) => `“${w}”`).join(', ')}` : 'a reply to its message')
    : e.mode === 'follow_up' ? 'following up (not addressed)' : ''
  return (
    <div className="px-3 py-2 space-y-1">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-zinc-500">
        <span>{stamp(e.ts)} · {e.chat}</span>
        <span className={`inline-flex items-center text-[11px] px-1.5 py-px rounded-full border ${PILL[label.tone]}`}>{label.text}</span>
        {how && <span>{how}</span>}
        {steps.length > 0 && (
          <button onClick={() => setOpen((v) => !v)} className="ml-auto flex items-center gap-0.5 text-violet-500 hover:underline">
            <ChevronRight className={`h-3 w-3 transition-transform ${open ? 'rotate-90' : ''}`} /> details
          </button>
        )}
      </div>
      {(e.messages ?? []).map((m) => (
        <div key={m.id} className="text-zinc-300 break-words">
          <span className="text-zinc-500">{m.from_owner ? 'You' : m.sender || 'someone'}: </span>{m.text || <span className="italic text-zinc-500">no text</span>}
        </div>
      ))}
      {why && <div className="italic text-zinc-400">Why: {why}</div>}
      {final?.kind === 'reply' && (
        <div className="border-l-2 border-emerald-500/50 pl-2 text-zinc-200 whitespace-pre-wrap break-words">{final.text}</div>
      )}
      {final?.kind === 'silent' && final.text && <div className="text-zinc-400">Its reason: {final.text}</div>}
      {e.error && <div className="text-amber-600 dark:text-amber-400">{e.error}</div>}
      {open && (
        <div className="space-y-2 rounded-md border border-zinc-800 bg-zinc-900/40 px-2.5 py-2">
          {steps.map((s, i) => (
            <div key={i} className="space-y-1">
              <div className="text-zinc-400">
                <span className="font-medium text-zinc-300">{i + 1}. {STEP_LABEL[s.kind] ?? s.kind}</span>
                {s.kind === 'search' && <span>: “{s.text}”</span>}
                {(s.kind === 'skip' || s.kind === 'error') && <span>: {s.text}</span>}
                {(s.model || s.tokens) ? <span className="text-zinc-500"> · {[s.model, s.tokens ? `${s.tokens} tokens` : ''].filter(Boolean).join(' · ')}</span> : null}
              </div>
              {s.reasoning && (
                <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap break-words rounded bg-zinc-950 px-2 py-1.5 font-mono text-[11px] text-zinc-400">{s.reasoning}</pre>
              )}
              {s.kind !== 'search' && s.why && <div className="text-zinc-400">Why: {s.why}</div>}
              {(s.results ?? []).length > 0 && (
                <ul className="list-disc pl-5 text-zinc-500">{s.results!.map((r, j) => <li key={j}>{r}</li>)}</ul>
              )}
            </div>
          ))}
          {!steps.some((s) => s.reasoning) && (
            <div className="text-zinc-500">This model didn’t return its reasoning: only its one-line “why” is shown.</div>
          )}
          {e.seconds !== undefined && <div className="text-zinc-500">{e.agent ? `${e.agent} · ` : ''}{e.seconds} s in all</div>}
        </div>
      )}
    </div>
  )
}

function AutoreplyActivity() {
  const [open, setOpen] = useState(false)
  const [ignored, setIgnored] = useState(false)
  const [entries, setEntries] = useState<WhatsAppWebAutoreplyEntry[] | null>(null)
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const load = useCallback(async () => {
    setBusy(true)
    try {
      const r = await getWhatsAppWebAutoreplyActivity(ignored)
      setEntries(r.entries)
      setNote(r.stale_service ? 'The WhatsApp Web service is older than Flight Deck: an admin can restart it by turning WhatsApp Web off and on for the deck.' : '')
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [ignored])
  useEffect(() => { if (open) load() }, [open, load])
  return (
    <div className={`${BOX} space-y-2`}>
      <div className="flex items-center gap-2">
        <button onClick={() => setOpen((v) => !v)} className="flex items-center gap-2 text-left">
          <ChevronRight className={`h-3 w-3 transition-transform ${open ? 'rotate-90' : ''}`} />
          <span className="font-medium text-zinc-100">Auto-reply activity</span>
        </button>
        {open && (
          <>
            <label className="ml-auto flex items-center gap-1.5 text-zinc-400">
              <input type="checkbox" checked={ignored} onChange={(e) => setIgnored(e.target.checked)} className="h-3 w-3 accent-violet-600" />
              messages not for it
            </label>
            <button onClick={load} disabled={busy} className="text-zinc-400 hover:text-zinc-200 disabled:opacity-50" title="Refresh">
              <RefreshCw className={`h-3.5 w-3.5 ${busy ? 'animate-spin' : ''}`} />
            </button>
          </>
        )}
      </div>
      {open && (
        <>
          <p className="text-zinc-500">
            Every time a message in an auto-reply chat could call your agent: what it decided, why, what it searched
            and what it said. Only you see this.
          </p>
          {note && <div className="text-amber-600 dark:text-amber-400">{note}</div>}
          {entries && entries.length === 0 && !note && <div className="text-zinc-500">Nothing yet.</div>}
          {entries && entries.length > 0 && (
            <div className="max-h-[28rem] overflow-y-auto rounded-md border border-zinc-800 bg-zinc-950 divide-y divide-zinc-800/60">
              {entries.map((e) => <ActivityEntry key={e.id} e={e} />)}
            </div>
          )}
        </>
      )}
    </div>
  )
}

function AdminSwitch({ onChange }: { onChange: () => void }) {
  const [admin, setAdmin] = useState<WhatsAppWebAdmin | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => { getWhatsAppWebAdmin().then(setAdmin).catch((e) => setError(String(e?.message ?? e))) }, [])

  if (!admin) return error ? <div className="text-xs text-red-600 dark:text-red-400">{error}</div> : null
  const toggle = async () => {
    if (!admin.allowed && !confirm(
      'Allow WhatsApp Web on this deck? People can then link their own WhatsApp accounts. ' +
      'It is an unofficial client: it goes against WhatsApp\'s terms and accounts do get banned.')) return
    if (admin.allowed && !confirm('Turn WhatsApp Web off? Every link on this deck is paused (kept on disk) until it\'s on again.')) return
    setBusy(true)
    setError('')
    try {
      setAdmin(await setWhatsAppWebAllowed(!admin.allowed))
      onChange()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  const links = admin.service ? Object.values(admin.service.links).reduce((a, b) => a + b, 0) : 0
  return (
    <div className={`${BOX} space-y-1.5`}>
      <div className="flex items-center gap-2">
        <div className="font-medium text-zinc-100">For everyone on this deck</div>
        <button onClick={toggle} disabled={busy || admin.kill_switch} className={`ml-auto ${admin.allowed ? SECONDARY : PRIMARY}`}>
          {busy && <Loader2 className="h-3 w-3 animate-spin" />}
          {admin.allowed ? 'Turn off' : 'Allow WhatsApp Web'}
        </button>
      </div>
      <div className="text-zinc-400">
        {admin.kill_switch
          ? 'Switched off for this deck by FD_WHATSAPP_WEB=off.'
          : !admin.installed
            ? <>Not installed here: <span className="font-mono text-zinc-200">pip install &apos;captain-claw[whatsapp-web]&apos;</span>, then restart Flight Deck.</>
            : admin.allowed
              ? `Allowed. ${admin.service ? `Service running (neonize ${admin.service.neonize || '?'}), ${links} link${links === 1 ? '' : 's'}.` : 'The service starts when someone links.'}`
              : 'Off. Nobody on this deck can link WhatsApp until you allow it.'}
      </div>
      {admin.service && !admin.service.available && (
        <div className="flex items-start gap-1.5 text-amber-600 dark:text-amber-400">
          <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-px" />
          {admin.service.reason}
        </div>
      )}
      {error && <div className="text-red-600 dark:text-red-400">{error}</div>}
    </div>
  )
}

export default function WhatsAppWebConnection() {
  const [view, setView] = useState<MyWhatsAppWeb | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  const [consent, setConsent] = useState(false)
  const [phone, setPhone] = useState('')
  const [byPhone, setByPhone] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [recent, setRecent] = useState<WhatsAppWebMessage[] | null>(null)
  // Disconnecting or switching number with messages kept: the keep-or-delete question is open.
  const [leaving, setLeaving] = useState<'disconnect' | 'switch' | null>(null)

  const load = useCallback(async () => {
    try {
      setView(await getMyWhatsAppWeb())
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => { load() }, [load])

  const state: WhatsAppWebState = view?.link.state ?? 'off'
  const live = LIVE.includes(state)
  useEffect(() => {
    if (!live) return
    const t = window.setInterval(() => { getMyWhatsAppWeb().then(setView).catch(() => {}) }, 2000)
    return () => window.clearInterval(t)
  }, [live])

  const run = async (key: string, fn: () => Promise<MyWhatsAppWeb>) => {
    setBusy(key)
    setError('')
    setNotice('')
    try {
      setView(await fn())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const link = view?.link
  const label = STATE[state] ?? STATE.off
  const spin = (key: string) => busy === key && <Loader2 className="h-3 w-3 animate-spin" />
  const messagesKept = link?.stats?.messages ?? 0
  const leave = (mode: 'disconnect' | 'switch', keepHistory: boolean) => {
    setLeaving(null)
    run('unlink', async () => {
      const v = await unlinkWhatsAppWeb(keepHistory)
      setConsent(false)
      setByPhone(false)
      setNotice([
        v.removed_from_phone
          ? 'Disconnected. “Captain Claw” is gone from your phone’s Linked devices.'
          : 'Disconnected here. Also remove “Captain Claw” on your phone: WhatsApp → Settings → Linked devices.',
        messagesKept > 0 && (keepHistory ? 'The history stays here.' : 'This number’s history is deleted.'),
        mode === 'switch' && 'Link the other number below.',
      ].filter(Boolean).join(' '))
      return v
    })
  }
  // No messages known, nothing to ask. Still "keep": with the service down the card
  // can't count them, and the service drops a store with nothing in it anyway.
  const askToLeave = (mode: 'disconnect' | 'switch') => {
    if (messagesKept > 0) setLeaving(mode)
    else if (confirm('Disconnect WhatsApp? The device is removed from your phone’s Linked devices.')) leave(mode, true)
  }
  const disconnect = (
    <>
      <button onClick={() => askToLeave('switch')} disabled={!!busy} className={SECONDARY}>
        Switch number
      </button>
      <button onClick={() => askToLeave('disconnect')} disabled={!!busy} className={`${SECONDARY} hover:text-red-400`}>
        {spin('unlink')} Disconnect
      </button>
    </>
  )

  let body: ReactNode = null
  if (view && !view.available) {
    body = (
      <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2.5 text-xs text-zinc-300">
        {view.reason}
      </div>
    )
  } else if (view && link && STARTABLE.includes(state)) {
    body = (
      <>
        {state !== 'off' && link.detail && (
          <div className="flex items-start gap-1.5 text-xs text-amber-600 dark:text-amber-400">
            <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-px" />
            {link.detail}
          </div>
        )}
        <div className={`${BOX} space-y-2.5`}>
          <label className="flex items-start gap-2">
            <input
              type="checkbox"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
              className="mt-0.5 h-3.5 w-3.5 shrink-0 accent-violet-600"
            />
            <span>
              I understand that this links my WhatsApp to Flight Deck as a device, the way WhatsApp Web
              does, through an <span className="text-zinc-100">unofficial client</span>: that goes against
              WhatsApp’s terms, and WhatsApp bans some accounts that do it. Flight Deck keeps a copy of my
              chats — mine and other people’s messages — encrypted on this deck, my agents read them, and
              what they read goes to the AI model they use. When I disconnect, I choose whether what’s kept
              stays or goes; if the link ends any other way (removed on the phone, a ban), it stays until I
              delete it.
            </span>
          </label>
          <div className="flex flex-wrap gap-2">
            <button onClick={() => run('link', linkWhatsAppWeb)} disabled={!!busy || !consent} className={PRIMARY}>
              {spin('link')} {state === 'off' ? 'Link WhatsApp' : 'Link again'}
            </button>
            {state !== 'off' && (
              <button onClick={() => run('cancel', cancelWhatsAppWeb)} disabled={!!busy} className={SECONDARY}>
                {spin('cancel')} Dismiss
              </button>
            )}
          </div>
        </div>
      </>
    )
  } else if (view && link && state === 'pairing') {
    body = (
      <div className={`${BOX} space-y-3`}>
        {!byPhone ? (
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="h-56 w-56 shrink-0 rounded-md bg-white p-2 flex items-center justify-center">
              {link.qr_svg
                ? <img src={link.qr_svg} alt="WhatsApp pairing QR code" className="h-full w-full [image-rendering:pixelated]" />
                : <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />}
            </div>
            <ol className="list-decimal pl-4 space-y-1 text-zinc-400">
              <li>Open WhatsApp on your phone.</li>
              <li>Go to <span className="text-zinc-200">Settings → Linked devices → Link a device</span>.</li>
              <li>Point the phone at this code. It changes every 20 seconds or so; keep this page open.</li>
            </ol>
          </div>
        ) : (
          <div className="space-y-2">
            <form
              onSubmit={(e) => { e.preventDefault(); run('code', () => whatsAppWebPairCode(phone)) }}
              className="flex flex-wrap items-center gap-2"
            >
              <input
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                placeholder="+385 97 123 4567"
                inputMode="tel"
                autoComplete="tel"
                className="w-48 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              />
              <button type="submit" disabled={!!busy || !phone.trim()} className={PRIMARY}>
                {spin('code')} Get code
              </button>
            </form>
            {link.pair_code && (
              <>
                <div className="font-mono text-2xl tracking-[0.2em] text-zinc-100">{link.pair_code}</div>
                <div className="text-zinc-400">
                  On your phone: <span className="text-zinc-200">Settings → Linked devices → Link a device →
                  Link with phone number instead</span>, then type this code.
                </div>
              </>
            )}
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => setByPhone((v) => !v)} className="text-violet-500 hover:underline">
            {byPhone ? 'Scan a QR code instead' : 'Link with a phone number instead'}
          </button>
          {link.pairing_expires_at && (
            <span className="text-zinc-500">Expires {when(link.pairing_expires_at)}</span>
          )}
          <button onClick={() => run('cancel', cancelWhatsAppWeb)} disabled={!!busy} className={`ml-auto ${SECONDARY}`}>
            {spin('cancel')} Cancel
          </button>
        </div>
      </div>
    )
  } else if (view && link) {
    const sync = link.sync
    body = (
      <>
        <div className={`${BOX} space-y-1.5`}>
          <div className="flex items-center gap-2">
            <span className="text-base font-medium text-zinc-100">{link.number || link.jid || 'Your account'}</span>
            {state === 'connecting' && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
          </div>
          {link.linked_at && <div className="text-zinc-400">Linked {when(link.linked_at)}.</div>}
          {link.detail && (
            <div className="flex items-start gap-1.5 text-amber-600 dark:text-amber-400">
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-px" />
              {link.detail}
            </div>
          )}
          {sync && sync.chunks > 0 && (
            <div className="text-zinc-400">
              History from your phone: {sync.messages.toLocaleString()} messages in {sync.chunks} part
              {sync.chunks === 1 ? '' : 's'}{sync.progress ? ` (${sync.progress}%)` : ''}.
            </div>
          )}
          {link.stats && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-zinc-400">
              <span>
                {link.stats.messages.toLocaleString()} message{link.stats.messages === 1 ? '' : 's'} kept in{' '}
                {link.stats.chats} chat{link.stats.chats === 1 ? '' : 's'}
                {link.stats.oldest ? `, since ${day(link.stats.oldest)}` : ''}
              </span>
              <label className="flex items-center gap-1.5">
                Keep
                <select
                  value={link.retention_days ?? ''}
                  disabled={!!busy}
                  onChange={(e) => {
                    const days = e.target.value ? Number(e.target.value) : null
                    if (days && !confirm(`Keep messages for ${days === 365 ? 'a year' : `${days} days`}? Older ones are deleted now.`)) return
                    run('retention', () => setWhatsAppWebRetention(days))
                  }}
                  className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none disabled:opacity-50"
                >
                  <option value="">everything</option>
                  {(link.retention_choices ?? [7, 30, 90, 365]).map((d) => (
                    <option key={d} value={d}>{d === 365 ? '1 year' : `${d} days`}</option>
                  ))}
                </select>
              </label>
            </div>
          )}
        </div>
        {link.linked_at && state !== 'logged_out' && <Sending />}
        {link.linked_at && state !== 'logged_out' && <AutoreplyActivity />}
        {link.stats && link.stats.messages > 0 && (
          <div>
            <button
              onClick={async () => {
                if (recent) { setRecent(null); return }
                try { setRecent((await getWhatsAppWebRecent()).messages) } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
              }}
              className="flex items-center gap-1 text-xs text-violet-500 hover:underline"
            >
              <ChevronRight className={`h-3 w-3 transition-transform ${recent ? 'rotate-90' : ''}`} />
              What your agents see
            </button>
            {recent && (
              <div className="mt-2 max-h-72 overflow-y-auto rounded-md border border-zinc-800 bg-zinc-950 divide-y divide-zinc-800/60">
                {recent.map((m) => (
                  <div key={`${m.chat_jid}:${m.id}`} className="px-3 py-1.5 text-xs">
                    <div className="text-zinc-500">{stamp(m.ts)} · {m.chat} · {m.from_me ? 'you' : m.sender}</div>
                    <div className="text-zinc-300 whitespace-pre-wrap break-words">
                      {m.deleted ? <span className="italic text-zinc-500">deleted</span> : (
                        <>
                          {m.kind !== 'text' && <span className="text-zinc-500">[{KIND_LABEL[m.kind] ?? m.kind}{m.media?.file_name ? `: ${m.media.file_name}` : ''}] </span>}
                          {m.text}
                          {m.edited && <span className="text-zinc-500"> (edited)</span>}
                          {m.reactions.length > 0 && <span className="text-zinc-500"> · {m.reactions.join(', ')}</span>}
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        <div className="flex flex-wrap gap-2">
          {(state === 'stream_replaced' || (!link.running && link.has_session && state !== 'outdated')) && (
            <button onClick={() => run('reconnect', reconnectWhatsAppWeb)} disabled={!!busy} className={PRIMARY}>
              {busy === 'reconnect' ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />} Reconnect
            </button>
          )}
          {disconnect}
        </div>
        {leaving && (
          <div className={`${BOX} space-y-2`}>
            <div className="text-sm font-medium text-zinc-100">
              {leaving === 'switch' ? 'Switch to another number' : 'Disconnect WhatsApp'}
            </div>
            <div className="text-zinc-300">
              “Captain Claw” leaves the Linked devices of {link.number || 'your phone'}. Keep the history from
              this number ({messagesKept.toLocaleString()} message{messagesKept === 1 ? '' : 's'})?
            </div>
            <div className="flex flex-wrap gap-2">
              <button onClick={() => leave(leaving, true)} className={PRIMARY}>Keep the history</button>
              <button onClick={() => leave(leaving, false)} className={`${SECONDARY} hover:text-red-400`}>Delete it</button>
              <button onClick={() => setLeaving(null)} className={SECONDARY}>Cancel</button>
            </div>
            <div className="text-zinc-500">
              If you keep it, your agents can still read it, nothing is sent from it, and you can delete it here
              at any time.
            </div>
          </div>
        )}
      </>
    )
  }

  const kept = view?.kept ?? []

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
      <button
        type="button"
        onClick={() => setCollapsed((v) => !v)}
        className={
          'w-full flex items-center gap-3 px-5 py-4 text-left hover:bg-zinc-900/80 transition-colors ' +
          (collapsed ? '' : 'border-b border-zinc-800')
        }
        aria-expanded={!collapsed}
      >
        <div className="h-10 w-10 rounded-md bg-gradient-to-br from-emerald-500/20 to-green-500/20 border border-zinc-800 flex items-center justify-center">
          <Smartphone className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">WhatsApp Web — your own account</h3>
            {view?.available && (
              <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border ${PILL[label.tone]}`}>
                {label.tone === 'on' && <Check className="h-3 w-3" />}
                {label.text}
              </span>
            )}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            Link your WhatsApp as a device, the way WhatsApp Web does, so your agents can read it like your email.
          </div>
        </div>
        {!view && !error && <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />}
        <ChevronDown className={'h-4 w-4 text-zinc-500 transition-transform ' + (collapsed ? '-rotate-90' : '')} />
      </button>

      {!collapsed && (
        <div className="px-5 py-4 space-y-3">
          {view?.can_admin && <AdminSwitch onChange={load} />}
          {body}
          {kept.length > 0 && (
            <div className={`${BOX} space-y-2`}>
              <div className="text-sm font-medium text-zinc-100">History kept from earlier links</div>
              {kept.map((k) => (
                <div key={k.id} className="flex flex-wrap items-center gap-2">
                  <span className="flex-1 min-w-0">
                    <span className="text-zinc-100">{k.number || 'An earlier link'}</span>
                    {' — '}{k.stats.messages.toLocaleString()} message{k.stats.messages === 1 ? '' : 's'} in{' '}
                    {k.stats.chats} chat{k.stats.chats === 1 ? '' : 's'}
                    {k.stats.oldest ? `, ${day(k.stats.oldest)} to ${day(k.stats.newest)}` : ''}
                    {k.retention_days ? `. Older than ${k.retention_days} days goes, as you chose for that link` : ''}.
                  </span>
                  <button
                    onClick={() => {
                      if (confirm(`Delete the ${k.stats.messages.toLocaleString()} messages kept from ${k.number || 'this link'}? This can’t be undone.`)) {
                        run(`kept:${k.id}`, () => deleteWhatsAppWebKept(k.id))
                      }
                    }}
                    disabled={!!busy}
                    className={`${SECONDARY} hover:text-red-400`}
                  >
                    {spin(`kept:${k.id}`)} Delete
                  </button>
                </div>
              ))}
              {view?.available && (
                <div className="text-zinc-500">Your agents can still read these. Nothing is sent from them.</div>
              )}
            </div>
          )}
          {notice && <div className="text-xs text-emerald-600 dark:text-emerald-500">{notice}</div>}
          {error && (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
              {error}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
