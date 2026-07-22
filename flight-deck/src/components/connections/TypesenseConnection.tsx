import { useEffect, useState } from 'react'
import {
  Database,
  Loader2,
  Check,
  AlertCircle,
  ChevronDown,
  Server,
} from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'

// Mirrors captain_claw/flight_deck/deep_memory_routes.py
interface Connection {
  enabled: boolean
  host: string
  port: number
  protocol: string
  collection_name: string
  api_key: string
  has_api_key: boolean
  configured: boolean
}

interface Probe {
  ok: boolean
  error?: string
  base_url?: string
  collections?: number
  collection_exists?: boolean
  documents?: number
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

const EMPTY: Connection = {
  enabled: false, host: 'localhost', port: 8108, protocol: 'http',
  collection_name: 'captain_claw_deep_memory', api_key: '', has_api_key: false,
  configured: false,
}

export default function TypesenseConnection() {
  const [conn, setConn] = useState<Connection>(EMPTY)
  const [draft, setDraft] = useState<Connection>(EMPTY)
  const [collapsed, setCollapsed] = useState(true)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [probe, setProbe] = useState<Probe | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    try {
      const res = await api('/fd/deep-memory/connection')
      if (!res.ok) throw new Error(await res.text())
      const d: Connection = await res.json()
      setConn(d); setDraft(d)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const save = async () => {
    setSaving(true); setError(null); setProbe(null)
    try {
      const res = await api('/fd/deep-memory/connection', {
        method: 'PUT',
        body: JSON.stringify({
          enabled: draft.enabled,
          host: draft.host.trim(),
          port: Number(draft.port) || 8108,
          protocol: draft.protocol,
          // An untouched field still holds the mask; the server reads that as
          // "keep the existing key" so saving never wipes it.
          api_key: draft.api_key,
          collection_name: draft.collection_name.trim(),
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const r = await res.json()
      setProbe(r.ok === false || r.error ? { ok: false, error: r.error } : { ...r, ok: true })
      await load()
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  const test = async () => {
    setTesting(true); setError(null)
    try {
      const res = await api('/fd/deep-memory/connection/test', { method: 'POST' })
      setProbe(await res.json())
    } catch (e) {
      setError(String(e))
    } finally {
      setTesting(false)
    }
  }

  const pill = 'inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border'
  const field =
    'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none'
  const label = 'block text-[11px] font-medium uppercase tracking-wider text-zinc-500 mb-1'

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
        <div className="h-10 w-10 rounded-md bg-gradient-to-br from-sky-500/20 to-indigo-500/20 border border-zinc-800 flex items-center justify-center">
          <Database className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">Typesense (Deep Memory)</h3>
            {loading ? (
              <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>
                <Loader2 className="h-3 w-3 animate-spin" /> Checking
              </span>
            ) : conn.configured ? (
              <span className={`${pill} bg-emerald-500/15 text-emerald-600 dark:text-emerald-500 border-emerald-500/30`}>
                <Check className="h-3 w-3" /> Configured
              </span>
            ) : (
              <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-500 border-red-500/30`}>
                <AlertCircle className="h-3 w-3" /> Not configured
              </span>
            )}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            The long-term archive behind Deep Memory. Flight Deck holds the connection —
            agents reach it through Flight Deck and never see these credentials.
          </div>
        </div>
        <ChevronDown
          className={'h-4 w-4 text-zinc-500 transition-transform ' + (collapsed ? '' : 'rotate-180')}
        />
      </button>

      {!collapsed && (
        <div className="px-5 py-4 space-y-4">
          <label className="flex items-center gap-2 text-xs text-zinc-300">
            <input
              type="checkbox"
              checked={draft.enabled}
              onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })}
              className="h-3.5 w-3.5 accent-violet-600"
            />
            Enable deep memory
          </label>

          <div className="grid grid-cols-12 gap-3">
            <div className="col-span-3">
              <label className={label}>Protocol</label>
              <select
                value={draft.protocol}
                onChange={(e) => setDraft({ ...draft, protocol: e.target.value })}
                className={field}
              >
                <option value="http">http</option>
                <option value="https">https</option>
              </select>
            </div>
            <div className="col-span-6">
              <label className={label}>Host</label>
              <input
                value={draft.host}
                onChange={(e) => setDraft({ ...draft, host: e.target.value })}
                placeholder="localhost"
                className={field}
              />
            </div>
            <div className="col-span-3">
              <label className={label}>Port</label>
              <input
                type="number"
                value={draft.port}
                onChange={(e) => setDraft({ ...draft, port: Number(e.target.value) })}
                placeholder="8108"
                className={field}
              />
            </div>
            <div className="col-span-6">
              <label className={label}>API key</label>
              <input
                type="password"
                value={draft.api_key}
                onChange={(e) => setDraft({ ...draft, api_key: e.target.value })}
                placeholder={conn.has_api_key ? 'unchanged' : 'admin API key'}
                className={`${field} font-mono`}
              />
            </div>
            <div className="col-span-6">
              <label className={label}>Collection</label>
              <input
                value={draft.collection_name}
                onChange={(e) => setDraft({ ...draft, collection_name: e.target.value })}
                placeholder="captain_claw_deep_memory"
                className={`${field} font-mono`}
              />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={save}
              disabled={saving}
              className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
            >
              {saving ? 'Saving…' : 'Save'}
            </button>
            <button
              onClick={test}
              disabled={testing || !conn.configured}
              className="flex items-center gap-1.5 rounded-md bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-40"
            >
              {testing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Server className="h-3 w-3" />}
              Test connection
            </button>
          </div>

          {probe && (
            <div
              className={
                'rounded-md border px-3 py-2 text-xs ' +
                (probe.ok
                  ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                  : 'border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300')
              }
            >
              {probe.ok ? (
                <>
                  Reached <span className="font-mono">{probe.base_url}</span> — {probe.collections}{' '}
                  collection{probe.collections === 1 ? '' : 's'}.{' '}
                  {probe.collection_exists
                    ? `"${draft.collection_name}" holds ${probe.documents} document(s).`
                    : `"${draft.collection_name}" doesn't exist yet — it's created on first index.`}
                </>
              ) : (
                probe.error
              )}
            </div>
          )}

          {error && (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
              {error}
            </div>
          )}

          <p className="text-[11px] leading-relaxed text-zinc-600">
            Needs a running Typesense server. Chunking, embedding and relevance tuning stay in{' '}
            <span className="font-mono">config.yaml</span> under{' '}
            <span className="font-mono">deep_memory</span>; only the connection lives here.
          </p>
        </div>
      )}
    </div>
  )
}
