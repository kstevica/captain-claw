import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { Send, Loader2, Check, AlertCircle, ChevronDown, ExternalLink, Clock, RefreshCw, Users, X } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { openConsole } from '../../stores/consoleStore'
import {
  getMyTelegram,
  requestTelegram,
  disconnectTelegram,
  removeTelegramGroup,
  type MyTelegram,
} from '../../services/sharedIntegrations'
import TelegramInbox from './TelegramInbox'

// The deck's shared Telegram bot (captain_claw/flight_deck/telegram_shared.py):
// one bot an admin set up for everyone. A user picks which of their own agents
// answers them there and asks for access; once an admin approves and the user
// opened the bot through their one-time link, Flight Deck relays their chat to
// that agent. The conversation itself lives in the agent, like any other chat.
//
// Agents with a Telegram bot of their own keep it (agent settings); this card
// is only the shared one.
export default function TelegramConnection({ onStatus }: { onStatus?: (s: MyTelegram | null) => void }) {
  // Not in the desktop app's users' kiosk (kioskView): the kiosk user is no admin.
  const isAdmin = useAuthStore((s) => s.user?.role === 'admin' && !s.kioskView)
  const [me, setMe] = useState<MyTelegram | null>(null)
  const [agent, setAgent] = useState('')
  const [collapsed, setCollapsed] = useState(false)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const apply = useCallback((data: MyTelegram) => {
    setMe(data)
    onStatus?.(data)
    setAgent((cur) => {
      if (data.link?.agent_slug && data.link.status !== 'rejected') return data.link.agent_slug
      if (cur && data.agents.some((a) => a.slug === cur)) return cur
      return data.agents[0]?.slug ?? ''
    })
  }, [onStatus])

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      apply(await getMyTelegram())
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [apply])

  useEffect(() => { refresh() }, [refresh])

  // While waiting on the admin or on the user opening the bot, look again now
  // and then so the card moves on by itself.
  const waiting = !!me?.link && me.link.status !== 'rejected' && (me.link.status === 'pending' || !me.link.linked)
  useEffect(() => {
    if (!waiting) return
    const t = window.setInterval(() => { getMyTelegram().then(apply).catch(() => {}) }, 8000)
    return () => window.clearInterval(t)
  }, [waiting, apply])

  const run = async (fn: () => Promise<MyTelegram>) => {
    setBusy(true)
    setError('')
    try {
      apply(await fn())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const link = me?.link ?? null
  const active = link && link.status !== 'rejected' ? link : null
  const pill = 'inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border'
  const field =
    'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none disabled:opacity-50'
  const label = 'block text-[11px] font-medium uppercase tracking-wider text-zinc-500 mb-1'

  let status: ReactNode = null
  if (me && !me.available) {
    status = <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Not set up</span>
  } else if (active?.status === 'approved' && active.linked) {
    status = (
      <span className={`${pill} bg-emerald-500/15 text-emerald-600 dark:text-emerald-500 border-emerald-500/30`}>
        <Check className="h-3 w-3" /> Connected
      </span>
    )
  } else if (active?.status === 'approved') {
    status = (
      <span className={`${pill} bg-amber-500/15 text-amber-600 dark:text-amber-500 border-amber-500/30`}>
        Approved — open the bot
      </span>
    )
  } else if (active?.status === 'pending') {
    status = (
      <span className={`${pill} bg-amber-500/15 text-amber-600 dark:text-amber-500 border-amber-500/30`}>
        <Clock className="h-3 w-3" /> Waiting for approval
      </span>
    )
  } else if (link?.status === 'rejected') {
    status = (
      <span className={`${pill} bg-red-500/15 text-red-500 border-red-500/30`}>
        <AlertCircle className="h-3 w-3" /> Not approved
      </span>
    )
  } else if (me) {
    status = <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Not connected</span>
  }

  const bot = me?.bot_username ? `@${me.bot_username}` : 'the team bot'
  const agentChanged = !!active && agent !== '' && agent !== active.agent_slug

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
        <div className="h-10 w-10 rounded-md bg-gradient-to-br from-sky-500/20 to-blue-500/20 border border-zinc-800 flex items-center justify-center">
          <Send className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">Telegram — team bot</h3>
            {status}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            Chat with one of your agents from Telegram, through the bot your admin set up for this deck.
          </div>
        </div>
        {loading && <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />}
        <ChevronDown
          className={'h-4 w-4 text-zinc-500 transition-transform ' + (collapsed ? '-rotate-90' : '')}
        />
      </button>

      {!collapsed && me && (
        <div className="px-5 py-4 space-y-4">
          {!me.auth_enabled && (
            <div className="rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-xs text-zinc-400">
              The team bot needs Flight Deck sign-in, which is off on this deck. An agent can still
              use a Telegram bot of its own (its settings → Telegram).
            </div>
          )}

          {me.auth_enabled && !me.available && (
            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2.5 text-xs text-zinc-300">
              <div className="font-medium text-amber-600 mb-0.5">No team Telegram bot yet</div>
              <div className="text-zinc-400">
                {isAdmin ? (
                  <>
                    Create a bot with @BotFather and add its token in{' '}
                    <button
                      type="button"
                      onClick={() => openConsole('connections')}
                      className="text-violet-500 hover:underline"
                    >
                      Admin console → Connections → Telegram
                    </button>
                    . Everyone on this deck can then ask to use it.
                  </>
                ) : (
                  'Ask an admin to set up the team bot. Once they have, you can connect your Telegram here.'
                )}
              </div>
            </div>
          )}

          {me.available && (
            <>
              {link?.status === 'rejected' && (
                <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
                  An admin didn't approve your last request. You can ask again.
                </div>
              )}

              {active?.status === 'pending' && (
                <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                  Your request is with the admins. {active.linked
                    ? `Your Telegram account (${active.tg_username ? '@' + active.tg_username : active.tg_first_name}) is already linked — the bot will tell you when you're approved.`
                    : 'You can link your Telegram account now, so you get a message there once you\'re approved.'}
                </div>
              )}

              {active && !active.linked && active.deep_link && (
                <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-3 space-y-2">
                  <div className="text-xs text-zinc-300">
                    Open {bot} through this link and press <span className="font-medium text-zinc-100">Start</span> —
                    it ties your Telegram account to your Flight Deck account. The link works once.
                  </div>
                  <a
                    href={active.deep_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 rounded-md bg-sky-600 px-3 py-1.5 text-xs text-white hover:bg-sky-500"
                  >
                    <ExternalLink className="h-3 w-3" /> Open {bot} in Telegram
                  </a>
                </div>
              )}

              {active?.status === 'approved' && active.linked && (
                <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2.5 text-xs text-zinc-300 space-y-1">
                  <div>
                    Telegram account{' '}
                    <span className="font-medium text-zinc-100">
                      {active.tg_username ? `@${active.tg_username}` : active.tg_first_name || 'linked'}
                    </span>{' '}
                    talks to <span className="font-medium text-zinc-100">{active.agent_name}</span> on {bot}.
                  </div>
                  {active.agent_online === false && (
                    <div className="text-amber-600 dark:text-amber-400">
                      {active.agent_name} isn't running — start it to get answers. Messages wait a few hours.
                    </div>
                  )}
                  {active.agent_online === null && (
                    <div className="text-zinc-500">
                      {active.agent_name} hasn't checked in with the team bot yet. If it was created before
                      this feature, restart it once.
                    </div>
                  )}
                </div>
              )}

              {/* Groups: bots can't use a group's invite link, so the bot's own
                  add-to-group link brings the agent in (telegram_agent_groups.py). */}
              {active?.status === 'approved' && active.linked && me.group_link && (
                <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-3 space-y-2">
                  <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-200">
                    <Users className="h-3.5 w-3.5" /> Telegram groups
                  </div>
                  <div className="text-xs text-zinc-400">
                    Bots can't join through a group's invite link — someone in the group adds them. Open
                    this link, pick the group, and {active.agent_name} joins it. There it answers when
                    someone mentions it or replies to it, and only uses the web — nothing of yours.
                  </div>
                  <a
                    href={me.group_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 rounded-md bg-sky-600 px-3 py-1.5 text-xs text-white hover:bg-sky-500"
                  >
                    <ExternalLink className="h-3 w-3" /> Add {active.agent_name} to a group
                  </a>
                  {(me.groups ?? []).length > 0 && (
                    <ul className="divide-y divide-zinc-800 rounded-md border border-zinc-800">
                      {(me.groups ?? []).map((g) => (
                        <li key={g.chat_id} className="flex items-center gap-2 px-2.5 py-1.5 text-xs text-zinc-300">
                          <span className="flex-1 min-w-0 truncate">{g.title}</span>
                          <button
                            type="button"
                            onClick={() => {
                              if (confirm(`Take ${active.agent_name} out of “${g.title}”?`)) {
                                run(() => removeTelegramGroup(g.chat_id))
                              }
                            }}
                            disabled={busy}
                            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-zinc-500 hover:text-red-500 disabled:opacity-40"
                            title="Remove from this group"
                          >
                            <X className="h-3 w-3" /> Remove
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {me.agents.length === 0 ? (
                <div className="text-xs text-zinc-500">
                  You have no agents yet. Create one first, then pick it here.
                </div>
              ) : (
                <div>
                  <label className={label}>Agent that answers you on Telegram</label>
                  <select
                    value={agent}
                    onChange={(e) => setAgent(e.target.value)}
                    disabled={busy}
                    className={field}
                  >
                    {me.agents.map((a) => (
                      <option key={a.slug} value={a.slug}>
                        {a.name}{a.status !== 'running' ? ' (stopped)' : ''}
                      </option>
                    ))}
                    {active && !me.agents.some((a) => a.slug === active.agent_slug) && (
                      <option value={active.agent_slug}>{active.agent_name} (removed)</option>
                    )}
                  </select>
                  <p className="mt-1 text-[11px] text-zinc-500">
                    Only your own agents. Your chats and files stay in that agent; Flight Deck just
                    passes the messages between it and Telegram.
                  </p>
                </div>
              )}

              <div className="flex flex-wrap items-center gap-2">
                {!active && (
                  <button
                    onClick={() => run(() => requestTelegram(agent))}
                    disabled={busy || !agent}
                    className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
                  >
                    {busy && <Loader2 className="h-3 w-3 animate-spin" />}
                    {/* Approved at once — unless an admin turned you down before. */}
                    {me?.auto_approve && link?.status !== 'rejected' ? 'Connect' : 'Request access'}
                  </button>
                )}
                {active && (
                  <button
                    onClick={() => run(() => requestTelegram(agent))}
                    disabled={busy || !agentChanged}
                    className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
                  >
                    Switch agent
                  </button>
                )}
                {link && (
                  <button
                    onClick={() => {
                      const msg = active?.status === 'pending'
                        ? 'Withdraw your request for the team Telegram bot?'
                        : 'Disconnect Telegram? The bot stops passing your messages on, your Telegram inbox (if on) is deleted, and asking again needs a new approval.'
                      if (confirm(msg)) run(disconnectTelegram)
                    }}
                    disabled={busy}
                    className="rounded-md bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-40"
                  >
                    {active?.status === 'pending' ? 'Withdraw request' : link.status === 'rejected' ? 'Clear' : 'Disconnect'}
                  </button>
                )}
                <button
                  onClick={refresh}
                  disabled={loading}
                  className="ml-auto flex items-center gap-1 rounded-md px-2 py-1.5 text-xs text-zinc-500 hover:text-zinc-300"
                  title="Refresh"
                >
                  <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
                </button>
              </div>

              <TelegramInbox linkKey={`${link?.status ?? ''}:${link?.linked ? 1 : 0}`} />
            </>
          )}

          {error && (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
              {error}
            </div>
          )}
        </div>
      )}
      {!collapsed && !me && error && (
        <div className="px-5 py-4 text-xs text-red-600 dark:text-red-400">{error}</div>
      )}
    </div>
  )
}
