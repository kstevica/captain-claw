import { useEffect, useState, type ReactNode } from 'react'
import { MessageCircle, Loader2, Check, AlertCircle, ChevronDown, Send } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'
import { useUIStore } from '../../stores/uiStore'

// Mirrors captain_claw/flight_deck/autonomy_routes.py (/fd/autonomy/whatsapp)
interface TestResult {
  sent: number
  total: number
  results: { to: string; ok: boolean; error: string }[]
}

interface NudgeState {
  bridge_configured: boolean
  auth_enabled: boolean
  autonomy_enabled: boolean
  autonomy_active: boolean
  nudge_to_whatsapp: boolean
  notify_waid: string
  recipients: string[]
  issue: string
}

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

// Where Autonomous Work's proactive nudges reach this user on WhatsApp.
export default function WhatsAppNudgeConnection() {
  const setView = useUIStore((s) => s.setView)
  const [state, setState] = useState<NudgeState | null>(null)
  const [number, setNumber] = useState('')
  const [enabled, setEnabled] = useState(true)
  const [collapsed, setCollapsed] = useState(true)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const apply = (d: NudgeState) => {
    setState(d)
    setNumber(d.notify_waid)
    setEnabled(d.nudge_to_whatsapp)
  }

  const load = async () => {
    setLoading(true)
    try {
      const res = await api('/fd/autonomy/whatsapp')
      if (!res.ok) throw new Error(await _detail(res))
      apply(await res.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const save = async () => {
    if (!state) return
    setSaving(true); setError(null); setNotice(null)
    // Only what changed — switching nudges off must work even if the saved
    // number has since left the allowlist.
    const body: Record<string, unknown> = {}
    if (number.trim() !== state.notify_waid) body.notify_waid = number
    if (enabled !== state.nudge_to_whatsapp) body.nudge_to_whatsapp = enabled
    try {
      const res = await api('/fd/autonomy/whatsapp', { method: 'PUT', body: JSON.stringify(body) })
      if (!res.ok) throw new Error(await _detail(res))
      apply(await res.json())
      setNotice('Saved.')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const test = async () => {
    setTesting(true); setError(null); setNotice(null)
    try {
      const res = await api('/fd/autonomy/whatsapp/test', { method: 'POST' })
      if (!res.ok) throw new Error(await _detail(res))
      const r: TestResult = await res.json()
      if (r.sent === r.total) {
        setNotice(
          `WhatsApp accepted the test for ${r.sent} number${r.sent === 1 ? '' : 's'}. If it doesn't ` +
            'arrive, send the bot any message first — WhatsApp only lets it write to you within ' +
            '24 hours of your last message.',
        )
      } else {
        setError(r.results.filter((x) => !x.ok).map((x) => `+${x.to}: ${x.error}`).join(' · '))
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setTesting(false)
    }
  }

  const dirty = !!state && (number.trim() !== state.notify_waid || enabled !== state.nudge_to_whatsapp)
  const deliverable = !!state && state.bridge_configured && state.nudge_to_whatsapp && state.recipients.length > 0

  const pill = 'inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border'
  const field =
    'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none disabled:opacity-50'
  const label = 'block text-[11px] font-medium uppercase tracking-wider text-zinc-500 mb-1'

  let status: ReactNode
  if (!state && error && !loading) {
    status = (
      <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-500 border-red-500/30`}>
        <AlertCircle className="h-3 w-3" /> Unavailable
      </span>
    )
  } else if (loading || !state) {
    status = (
      <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>
        <Loader2 className="h-3 w-3 animate-spin" /> Checking
      </span>
    )
  } else if (deliverable && state.autonomy_active) {
    status = (
      <span className={`${pill} bg-emerald-500/15 text-emerald-600 dark:text-emerald-500 border-emerald-500/30`}>
        <Check className="h-3 w-3" /> Active
      </span>
    )
  } else if (deliverable) {
    // Delivery is set up; Autonomous Work isn't running (the note below says so).
    status = (
      <span className={`${pill} bg-zinc-800 text-zinc-300 border-zinc-700`}>
        <Check className="h-3 w-3" /> Ready
      </span>
    )
  } else if (!state.bridge_configured) {
    status = <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>WhatsApp not set up</span>
  } else if (!state.nudge_to_whatsapp) {
    status = <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Off</span>
  } else if (state.notify_waid) {
    // A saved number that the allowlist no longer includes.
    status = (
      <span className={`${pill} bg-amber-500/15 text-amber-600 dark:text-amber-500 border-amber-500/30`}>
        <AlertCircle className="h-3 w-3" /> Number not allowed
      </span>
    )
  } else {
    status = (
      <span className={`${pill} bg-amber-500/15 text-amber-600 dark:text-amber-500 border-amber-500/30`}>
        <AlertCircle className="h-3 w-3" /> No number
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
        <div className="h-10 w-10 rounded-md bg-gradient-to-br from-emerald-500/20 to-green-500/20 border border-zinc-800 flex items-center justify-center">
          <MessageCircle className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">WhatsApp nudges</h3>
            {status}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            Autonomous Work's reminders and follow-up nudges, delivered to your own WhatsApp
            as well as the agent's chat.
          </div>
        </div>
        <ChevronDown
          className={'h-4 w-4 text-zinc-500 transition-transform ' + (collapsed ? '' : 'rotate-180')}
        />
      </button>

      {!collapsed && state && (
        <div className="px-5 py-4 space-y-4">
          {!state.bridge_configured && (
            <div className="rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-xs text-zinc-400">
              The WhatsApp bridge isn't set up on this deck. An admin has to configure it
              (WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_ALLOWED_WAIDS) before
              nudges can go to WhatsApp.
            </div>
          )}
          {state.bridge_configured && !state.autonomy_active && (
            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
              Autonomous Work isn't running for you, so no nudges are sent yet.{' '}
              <button
                type="button"
                onClick={() => setView('autonomous-work')}
                className="underline hover:no-underline"
              >
                Turn it on in Autonomous Work
              </button>
              .
            </div>
          )}

          <label className="flex items-center gap-2 text-xs text-zinc-300">
            <input
              type="checkbox"
              checked={enabled}
              disabled={!state.bridge_configured}
              onChange={(e) => setEnabled(e.target.checked)}
              className="h-3.5 w-3.5 accent-violet-600"
            />
            Send my nudges to WhatsApp
          </label>

          <div>
            <label className={label}>Your WhatsApp number</label>
            <input
              value={number}
              disabled={!state.bridge_configured}
              onChange={(e) => setNumber(e.target.value)}
              placeholder="385911234567"
              className={`${field} font-mono`}
            />
            <p className="mt-1 text-[11px] text-zinc-500">
              International format without the +, e.g. 385911234567. Separate several numbers with
              commas (up to 3). Each must be on this deck's WhatsApp allowlist.
              {!state.auth_enabled && ' Leave empty to use every allowlisted number.'}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={save}
              disabled={saving || !dirty || !state.bridge_configured}
              className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
            >
              {saving ? 'Saving…' : 'Save'}
            </button>
            <button
              onClick={test}
              disabled={testing || dirty || !state.bridge_configured || state.recipients.length === 0}
              title={dirty ? 'Save first' : undefined}
              className="flex items-center gap-1.5 rounded-md bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-40"
            >
              {testing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Send className="h-3 w-3" />}
              Send test
            </button>
          </div>

          {state.bridge_configured && (
            <div className="text-xs text-zinc-500">
              {state.recipients.length > 0
                ? <>Nudges go to: <span className="font-mono text-zinc-300">{state.recipients.map((w) => `+${w}`).join(', ')}</span>
                    {!state.nudge_to_whatsapp && ' (paused — the switch above is off)'}</>
                : state.issue && <>Not delivered to WhatsApp: {state.issue}.</>}
            </div>
          )}

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
