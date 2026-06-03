import { useState, useEffect } from 'react'
import { Target, Loader2, AlertTriangle, RefreshCw, X, Check, Ban, Clock, OctagonX, User, Bot } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'

interface Intention {
  id: string
  origin: string
  title: string
  body?: string
  why?: string
  category?: string
  risk?: string
  approval_mode?: string
  status: string
  repeat?: string | null
  tags?: string[]
  created_at?: string
}

interface Decision {
  id: string
  intention_id: string
  kind: string
  prompt_text: string
  options?: string[]
  created_at?: string
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { ...(init?.headers as Record<string, string> | undefined) }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`

  let res = await fetch(`/fd${path}`, { ...init, headers, credentials: 'include' })
  if (res.status === 401 && authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) {
      const t2 = useAuthStore.getState().token
      const h2: Record<string, string> = { ...(init?.headers as Record<string, string> | undefined) }
      if (t2) h2['Authorization'] = `Bearer ${t2}`
      res = await fetch(`/fd${path}`, { ...init, headers: h2, credentials: 'include' })
    }
  }
  if (!res.ok) {
    const b = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(b.error || b.detail || `${res.status}`)
  }
  return res.json()
}

interface IntentionsPanelProps {
  host: string
  port: number
  auth?: string
  agentName: string
  onClose: () => void
}

export function IntentionsPanel({ host, port, auth, agentName, onClose }: IntentionsPanelProps) {
  const [intentions, setIntentions] = useState<Intention[]>([])
  const [decisions, setDecisions] = useState<Decision[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const [tagFilter, setTagFilter] = useState<string | null>(null)

  const allTags = Array.from(new Set(intentions.flatMap((i) => i.tags || []))).sort()
  const shownIntentions = tagFilter
    ? intentions.filter((i) => (i.tags || []).includes(tagFilter))
    : intentions

  const tokenQs = auth ? `?token=${encodeURIComponent(auth)}` : ''

  const refresh = async () => {
    setLoading(true)
    setError('')
    try {
      const [d, i] = await Promise.all([
        fdFetch<{ decisions: Decision[] }>(`/agent-intentions-decisions/${host}/${port}${tokenQs}`),
        fdFetch<{ intentions: Intention[] }>(`/agent-intentions/${host}/${port}${tokenQs}`),
      ])
      setDecisions(d.decisions || [])
      setIntentions(i.intentions || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { refresh() }, [host, port])

  const resolve = async (decisionId: string, verdict: string) => {
    setBusy(decisionId)
    setError('')
    try {
      await fdFetch(`/agent-intentions-decision/${host}/${port}/${decisionId}/resolve${tokenQs}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verdict, via: 'flight_deck' }),
      })
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const originIcon = (o: string) => (o === 'user'
    ? <User className="h-3.5 w-3.5 text-sky-400/80 shrink-0" />
    : <Bot className="h-3.5 w-3.5 text-violet-400/80 shrink-0" />)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="flex flex-col rounded-xl border border-zinc-700/50 bg-zinc-900 shadow-2xl"
        style={{ width: '70vw', maxWidth: '760px', height: '80vh' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800 shrink-0">
          <div className="flex items-center gap-2.5">
            <Target className="h-4 w-4 text-amber-400" />
            <span className="text-sm font-medium text-zinc-200">Intentions — {agentName}</span>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={refresh} className="text-zinc-500 hover:text-zinc-300 transition-colors" title="Refresh">
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
            <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 transition-colors">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
              <span className="ml-2 text-sm text-zinc-500">Loading…</span>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-full">
              <AlertTriangle className="h-5 w-5 text-red-400 mr-2" />
              <span className="text-sm text-red-400">{error}</span>
            </div>
          ) : (
            <div className="p-4 space-y-5">
              {/* Pending decisions */}
              <section>
                <h3 className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500 mb-2">
                  Pending decisions {decisions.length > 0 && <span className="text-amber-400">({decisions.length})</span>}
                </h3>
                {decisions.length === 0 ? (
                  <p className="text-xs text-zinc-600">Nothing awaiting your call.</p>
                ) : (
                  <div className="space-y-2">
                    {decisions.map((d) => (
                      <div key={d.id} className="rounded-lg border border-amber-500/20 bg-amber-500/[0.06] p-3">
                        <div className="text-sm text-zinc-200">{d.prompt_text}</div>
                        <div className="mt-2.5 flex items-center gap-2">
                          {d.kind === 'announce_undo' ? (
                            <button
                              disabled={busy === d.id}
                              onClick={() => resolve(d.id, 'stop')}
                              className="flex items-center gap-1 rounded-md bg-red-500/15 px-2.5 py-1 text-xs font-medium text-red-300 hover:bg-red-500/25 disabled:opacity-40"
                            >
                              <OctagonX className="h-3.5 w-3.5" /> Stop
                            </button>
                          ) : (
                            <>
                              <button
                                disabled={busy === d.id}
                                onClick={() => resolve(d.id, 'yes')}
                                className="flex items-center gap-1 rounded-md bg-emerald-500/15 px-2.5 py-1 text-xs font-medium text-emerald-300 hover:bg-emerald-500/25 disabled:opacity-40"
                              >
                                <Check className="h-3.5 w-3.5" /> Approve
                              </button>
                              <button
                                disabled={busy === d.id}
                                onClick={() => resolve(d.id, 'no')}
                                className="flex items-center gap-1 rounded-md bg-zinc-700/50 px-2.5 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-700 disabled:opacity-40"
                              >
                                <Ban className="h-3.5 w-3.5" /> Decline
                              </button>
                              <button
                                disabled={busy === d.id}
                                onClick={() => resolve(d.id, 'later')}
                                className="flex items-center gap-1 rounded-md bg-zinc-700/50 px-2.5 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-700 disabled:opacity-40"
                              >
                                <Clock className="h-3.5 w-3.5" /> Later
                              </button>
                            </>
                          )}
                          {busy === d.id && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              {/* Open intentions */}
              <section>
                <h3 className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500 mb-2">
                  Active intentions {shownIntentions.length > 0 && <span className="text-zinc-400">({shownIntentions.length}{tagFilter ? ` of ${intentions.length}` : ''})</span>}
                </h3>
                {/* Tag filter */}
                {allTags.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1.5 mb-2">
                    <button
                      onClick={() => setTagFilter(null)}
                      className={`rounded-full px-2 py-0.5 text-[10px] transition-colors ${tagFilter === null ? 'bg-amber-500/20 text-amber-300' : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'}`}
                    >
                      all
                    </button>
                    {allTags.map((t) => (
                      <button
                        key={t}
                        onClick={() => setTagFilter(tagFilter === t ? null : t)}
                        className={`rounded-full px-2 py-0.5 text-[10px] transition-colors ${tagFilter === t ? 'bg-amber-500/20 text-amber-300' : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'}`}
                      >
                        #{t}
                      </button>
                    ))}
                  </div>
                )}
                {shownIntentions.length === 0 ? (
                  <p className="text-xs text-zinc-600">{tagFilter ? `No intentions tagged #${tagFilter}.` : 'No active intentions.'}</p>
                ) : (
                  <div className="divide-y divide-zinc-800/70 rounded-lg border border-zinc-800">
                    {shownIntentions.map((it) => (
                      <div key={it.id} className="flex items-start gap-2.5 px-3 py-2.5">
                        {originIcon(it.origin)}
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-zinc-200 truncate">{it.title}</span>
                            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400 shrink-0">{it.status}</span>
                            {it.repeat && (
                              <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500 shrink-0">{it.repeat}</span>
                            )}
                          </div>
                          {it.why && <div className="text-[11px] text-zinc-500 mt-0.5">Why: {it.why}</div>}
                          {(it.tags && it.tags.length > 0) && (
                            <div className="flex flex-wrap gap-1 mt-1">
                              {it.tags.map((t) => (
                                <button
                                  key={t}
                                  onClick={() => setTagFilter(t)}
                                  className="rounded-full bg-zinc-800/80 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:text-amber-300"
                                >
                                  #{t}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
