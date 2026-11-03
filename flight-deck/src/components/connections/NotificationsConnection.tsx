import { useEffect, useState, type ComponentType, type ReactNode } from 'react'
import { isAppShell } from '../../services/desktopShell'
import { Bell, Loader2, Check, AlertCircle, ChevronDown, MessageCircle, Send, Watch, Bot } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'
import { useUIStore } from '../../stores/uiStore'

// Mirrors captain_claw/flight_deck/autonomy_routes.py (/fd/autonomy/notify)
type ChannelId = 'whatsapp' | 'telegram' | 'devices'
// How much of the Line a channel gets (flight_deck/line.py levels).
type Level = 'everything' | 'needs_you' | 'off'

interface Channel {
  on: boolean
  ready: boolean
  detail: string
  connected?: number
  /** WhatsApp and Telegram: how much of the Line. */
  level?: Level
}

interface LineAgent {
  slug: string
  name: string
  level: Level
  options: { slug: string; name: string }[]
}

interface NotifyState {
  autonomy_active: boolean
  waiting: number
  channels: Record<ChannelId, Channel>
  agent?: LineAgent
}

const LEVELS: { id: Level; label: string }[] = [
  { id: 'off', label: 'Off' },
  { id: 'needs_you', label: 'Needs you' },
  { id: 'everything', label: 'Everything' },
]

// The WhatsApp card switches the same key: each tells the other to reload.
export const NOTIFY_CHANGED = 'fd:notify-changed'

function _headers(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

async function api(url: string, init: RequestInit = {}): Promise<Response> {
  const build = (): RequestInit => ({
    ...init,
    headers: { ..._headers(), ...((init.headers as Record<string, string>) || {}) },
    credentials: 'include',
  })
  let res = await fetch(url, build())
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (await refreshAccessToken()) res = await fetch(url, build())
  }
  return res
}

async function _detail(res: Response): Promise<string> {
  try {
    const d = await res.json()
    return typeof d.detail === 'string' ? d.detail : JSON.stringify(d)
  } catch {
    return res.statusText || `HTTP ${res.status}`
  }
}

const CHANNELS: { id: ChannelId; label: string; what: string; icon: ComponentType<{ className?: string }> }[] = [
  { id: 'whatsapp', label: 'WhatsApp', what: 'the item, to your number', icon: MessageCircle },
  { id: 'telegram', label: 'Telegram', what: "the item, to your chat with the deck's bot", icon: Send },
  { id: 'devices', label: 'Devices', what: 'a beep, a banner and how many proposals wait', icon: Watch },
]

// Where your Line — what the deck and your agents said on their own — reaches
// you besides Flight Deck: each phone at a level, devices, and one agent's chat.
// `kiosk`: shown in the kiosk's Connections, which has no Autonomous Work page.
// `open`: starts open (the Line's own Channels button).
export default function NotificationsConnection({ kiosk = false, open = false }: { kiosk?: boolean; open?: boolean }) {
  const setView = useUIStore((s) => s.setView)
  const [state, setState] = useState<NotifyState | null>(null)
  const [collapsed, setCollapsed] = useState(!open)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState<ChannelId | 'agent' | null>(null)
  const [testing, setTesting] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    try {
      const res = await api('/fd/autonomy/notify')
      if (!res.ok) throw new Error(await _detail(res))
      setState(await res.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
    const reload = () => { load() }
    window.addEventListener(NOTIFY_CHANGED, reload)
    return () => window.removeEventListener(NOTIFY_CHANGED, reload)
  }, [])

  const save = async (id: ChannelId | 'agent', body: Record<string, unknown>) => {
    setSaving(id); setError(null); setNotice(null)
    try {
      const res = await api('/fd/autonomy/notify', { method: 'PUT', body: JSON.stringify(body) })
      if (!res.ok) throw new Error(await _detail(res))
      setState(await res.json())
      window.dispatchEvent(new Event(NOTIFY_CHANGED))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(null)
    }
  }

  const test = async () => {
    setTesting(true); setError(null); setNotice(null)
    try {
      const res = await api('/fd/autonomy/notify/test', { method: 'POST' })
      if (!res.ok) throw new Error(await _detail(res))
      const { sent } = await res.json() as { sent: { whatsapp: [number, number] | null; telegram: boolean | null; devices: number | null } }
      const parts: string[] = []
      if (sent.whatsapp) {
        parts.push(sent.whatsapp[1] ? `WhatsApp ${sent.whatsapp[0]}/${sent.whatsapp[1]}` : 'WhatsApp has no number to send to')
      }
      if (sent.telegram !== null) parts.push(sent.telegram ? 'Telegram sent' : 'Telegram not sent')
      if (sent.devices !== null) parts.push(`${sent.devices} device${sent.devices === 1 ? '' : 's'} beeped`)
      setNotice(parts.length ? `Test: ${parts.join(' · ')}.` : 'Every channel is off — nothing to test.')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setTesting(false)
    }
  }

  const pill = 'inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border'
  const select = 'rounded-md border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-[11px] text-zinc-200 focus:border-violet-500/60 focus:outline-none disabled:opacity-50'
  const onCount = state
    ? CHANNELS.filter((c) => state.channels[c.id].on).length + (state.agent?.slug ? 1 : 0)
    : 0

  let status: ReactNode
  if (!state && error && !loading) {
    status = (
      <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-500 border-red-500/30`}>
        <AlertCircle className="h-3 w-3" /> Unavailable
      </span>
    )
  } else if (loading && !state) {
    status = (
      <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>
        <Loader2 className="h-3 w-3 animate-spin" /> Checking
      </span>
    )
  } else if (state && onCount === 0) {
    status = <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Off</span>
  } else if (state) {
    status = (
      <span className={`${pill} bg-emerald-500/15 text-emerald-600 dark:text-emerald-500 border-emerald-500/30`}>
        <Check className="h-3 w-3" /> {onCount} on
      </span>
    )
  }

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
        <div className="h-10 w-10 rounded-md bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 border border-zinc-800 flex items-center justify-center">
          <Bell className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">Notifications</h3>
            {status}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            Where your Line — results, finished runs, proposals, nudges and what waits for you —
            reaches you besides Flight Deck.
          </div>
        </div>
        <ChevronDown
          className={'h-4 w-4 text-zinc-500 transition-transform ' + (collapsed ? '' : 'rotate-180')}
        />
      </button>

      {!collapsed && state && (
        <div className="px-5 py-4 space-y-4">
          {!state.autonomy_active && (
            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
              Autonomous Work isn't running for you: no nudges or proposals. Everything else on
              your Line still comes.
              {/* The desktop app and the kiosk have no Autonomous Work page to send you to. */}
              {!kiosk && !isAppShell() && (
                <>
                  {' '}
                  <button
                    type="button"
                    onClick={() => setView('autonomous-work')}
                    className="underline hover:no-underline"
                  >
                    Turn it on in Autonomous Work
                  </button>
                  .
                </>
              )}
            </div>
          )}

          <div className="divide-y divide-zinc-800 rounded-md border border-zinc-800">
            {CHANNELS.map(({ id, label, what, icon: Icon }) => {
              const ch = state.channels[id]
              // WhatsApp and Telegram take a level; devices are on or off.
              const leveled = id !== 'devices' && ch.level !== undefined
              return (
                <label key={id} className="flex items-start gap-3 px-3 py-2.5 cursor-pointer">
                  {!leveled && (
                    <input
                      type="checkbox"
                      checked={ch.on}
                      disabled={saving !== null}
                      onChange={(e) => save(id, { [id]: e.target.checked })}
                      className="mt-0.5 h-3.5 w-3.5 accent-violet-600"
                    />
                  )}
                  <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 text-xs text-zinc-200">
                      {label}
                      {saving === id && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
                      {leveled && (
                        <select
                          value={ch.level}
                          disabled={saving !== null}
                          onChange={(e) => save(id, { [id]: e.target.value })}
                          className={`ml-auto ${select}`}
                          aria-label={`${label}: how much of the Line`}
                        >
                          {LEVELS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                        </select>
                      )}
                    </div>
                    <div className="text-[11px] text-zinc-500">{what}</div>
                    <div
                      className={
                        'text-[11px] ' +
                        (ch.on && !ch.ready ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-500')
                      }
                    >
                      {ch.detail}
                    </div>
                  </div>
                </label>
              )
            })}
            {state.agent && (
              <div className="flex items-start gap-3 px-3 py-2.5">
                <Bot className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-200">
                    One agent's chat
                    {saving === 'agent' && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
                    <span className="ml-auto flex items-center gap-1.5">
                      <select
                        value={state.agent.slug}
                        disabled={saving !== null}
                        onChange={(e) => save('agent', { agent: e.target.value })}
                        className={`max-w-[10rem] ${select}`}
                        aria-label="The agent whose chat gets the Line"
                      >
                        <option value="">None</option>
                        {state.agent.options.map((a) => <option key={a.slug} value={a.slug}>{a.name}</option>)}
                      </select>
                      {state.agent.slug && (
                        <select
                          value={state.agent.level}
                          disabled={saving !== null}
                          onChange={(e) => save('agent', { agent_level: e.target.value })}
                          className={select}
                          aria-label="How much of the Line the agent gets"
                        >
                          {LEVELS.filter((l) => l.id !== 'off').map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                        </select>
                      )}
                    </span>
                  </div>
                  <div className="text-[11px] text-zinc-500">
                    a card in its chat it keeps as context — ask it what happened; it never answers on its own
                  </div>
                  {state.agent.slug && (
                    <div className="text-[11px] text-zinc-500">
                      {state.agent.name}: only while it's running — an agent asleep reads the Line when it wakes
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          <p className="text-[11px] text-zinc-500">
            Needs you: what waits for you or failed, and nudges. Everything: all of it but fleet
            changes, which stay on the Line. In quiet hours phones get only nudges. A run that
            already answered on WhatsApp or Telegram, or a scheduled task's own delivery, isn't sent
            twice. What needs you is also in the bell. Devices: a sleeping one beeps when it next
            wakes. The WhatsApp number is set on the WhatsApp tab.
          </p>

          <div className="flex items-center gap-3">
            <button
              onClick={test}
              disabled={testing || onCount === 0}
              className="flex items-center gap-1.5 rounded-md bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-40"
            >
              {testing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Bell className="h-3 w-3" />}
              Send test
            </button>
            {state.waiting > 0 && (
              <span className="text-xs text-zinc-400">
                {state.waiting} proposal{state.waiting === 1 ? '' : 's'} waiting for you
              </span>
            )}
          </div>

          {notice && (
            <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-300">
              {notice}
            </div>
          )}
          {error && (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
              {error}
            </div>
          )}
        </div>
      )}
      {!collapsed && !state && error && (
        <div className="px-5 py-4 text-xs text-red-600 dark:text-red-400">{error}</div>
      )}
    </div>
  )
}
