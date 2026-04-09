import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  X,
  Inbox,
  Loader2,
  AlertCircle,
  Check,
  Ban,
  ArrowRightLeft,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'

/**
 * Review staged-conflict insights on a target agent.
 *
 * When a memory bundle is imported with `stage_conflicts=true`, incoming
 * decision/preference/workflow insights that would otherwise hit the
 * silent dedup path are instead routed to the agent's pending-review
 * queue. This modal lists them, shows the conflicting live insight
 * side-by-side where applicable, and lets a human approve or reject
 * each one.
 */

export interface PendingInsightsTarget {
  host: string
  port: number
  authToken: string
  label: string
}

interface ConflictSnapshot {
  id?: string
  content?: string
  category?: string
  importance?: number
  tags?: string
  created_at?: string
}

interface PendingInsight {
  id: string
  content: string
  category: string
  entity_key?: string | null
  importance: number
  source_tool?: string | null
  source_session?: string | null
  created_at: string
  expires_at?: string | null
  tags?: string | null
  source_label?: string | null
  conflict_reason?: string | null
  conflicts_with?: string | null
  conflict_snapshot?: ConflictSnapshot | null
}

interface Props {
  target: PendingInsightsTarget
  onClose: () => void
}

function fdAuthHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = {}
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

const CATEGORY_COLORS: Record<string, string> = {
  decision: 'text-amber-300 bg-amber-500/10 border-amber-500/30',
  preference: 'text-violet-300 bg-violet-500/10 border-violet-500/30',
  workflow: 'text-sky-300 bg-sky-500/10 border-sky-500/30',
}

export function PendingInsightsModal({ target, onClose }: Props) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [items, setItems] = useState<PendingInsight[]>([])
  const [busyId, setBusyId] = useState<string | null>(null)
  const [categoryFilter, setCategoryFilter] = useState<string>('')

  const base = `/fd/agent-insights/${encodeURIComponent(target.host)}/${target.port}`

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams()
      if (target.authToken) params.set('token', target.authToken)
      if (categoryFilter) params.set('category', categoryFilter)
      params.set('limit', '200')
      const res = await fetch(`${base}/pending?${params}`, {
        headers: fdAuthHeaders(),
        credentials: 'include',
      })
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text().catch(() => '')}`)
      const data = await res.json()
      setItems(Array.isArray(data.items) ? data.items : [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [base, target.authToken, categoryFilter])

  useEffect(() => {
    void load()
  }, [load])

  const approve = useCallback(
    async (id: string, supersede: boolean) => {
      setBusyId(id)
      try {
        const params = new URLSearchParams()
        if (target.authToken) params.set('token', target.authToken)
        params.set('supersede', supersede ? '1' : '0')
        const res = await fetch(`${base}/pending/${encodeURIComponent(id)}/approve?${params}`, {
          method: 'POST',
          headers: fdAuthHeaders(),
          credentials: 'include',
        })
        if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text().catch(() => '')}`)
        setItems((prev) => prev.filter((i) => i.id !== id))
      } catch (e) {
        alert(`Approve failed: ${e instanceof Error ? e.message : String(e)}`)
      } finally {
        setBusyId(null)
      }
    },
    [base, target.authToken]
  )

  const reject = useCallback(
    async (id: string) => {
      setBusyId(id)
      try {
        const params = new URLSearchParams()
        if (target.authToken) params.set('token', target.authToken)
        const res = await fetch(`${base}/pending/${encodeURIComponent(id)}/reject?${params}`, {
          method: 'POST',
          headers: fdAuthHeaders(),
          credentials: 'include',
        })
        if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text().catch(() => '')}`)
        setItems((prev) => prev.filter((i) => i.id !== id))
      } catch (e) {
        alert(`Reject failed: ${e instanceof Error ? e.message : String(e)}`)
      } finally {
        setBusyId(null)
      }
    },
    [base, target.authToken]
  )

  const groups = useMemo(() => {
    const by: Record<string, PendingInsight[]> = {}
    for (const it of items) {
      const key = it.category || 'other'
      by[key] = by[key] || []
      by[key].push(it)
    }
    return Object.entries(by).sort(([a], [b]) => a.localeCompare(b))
  }, [items])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="flex max-h-[85vh] w-[820px] flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div className="flex items-center gap-2">
            <Inbox className="h-4 w-4 text-amber-400" />
            <h2 className="text-sm font-semibold">Pending Insights — {target.label}</h2>
            {items.length > 0 && (
              <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[10px] font-medium text-amber-300">
                {items.length}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-[11px] text-zinc-300 focus:border-zinc-700 focus:outline-none"
            >
              <option value="">All categories</option>
              <option value="decision">decision</option>
              <option value="preference">preference</option>
              <option value="workflow">workflow</option>
            </select>
            <button
              onClick={() => void load()}
              className="rounded-md border border-zinc-800 px-2 py-1 text-[11px] text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
            >
              Refresh
            </button>
            <button
              onClick={onClose}
              className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          <div className="text-xs text-zinc-500">
            These insights were staged because they conflict with an existing entry in the{' '}
            <code className="text-zinc-400">decision</code>,{' '}
            <code className="text-zinc-400">preference</code>, or{' '}
            <code className="text-zinc-400">workflow</code> categories. Approve to promote
            them (optionally replacing the live one); reject to drop.
          </div>

          {loading && (
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading pending insights…
            </div>
          )}

          {error && !loading && (
            <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <div>
                <div className="font-semibold">Failed</div>
                <div className="text-red-300/80">{error}</div>
              </div>
            </div>
          )}

          {!loading && !error && items.length === 0 && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 text-xs text-zinc-500">
              Nothing pending. Either no recent import used{' '}
              <span className="text-zinc-300">stage conflicts</span>, or all staged items were
              already reviewed.
            </div>
          )}

          {!loading && groups.map(([cat, rows]) => (
            <div key={cat} className="space-y-2">
              <div className="flex items-center gap-2">
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${
                    CATEGORY_COLORS[cat] ?? 'text-zinc-300 bg-zinc-800 border-zinc-700'
                  }`}
                >
                  {cat}
                </span>
                <span className="text-[10px] text-zinc-600">{rows.length} pending</span>
              </div>
              {rows.map((it) => {
                const live = it.conflict_snapshot
                const busy = busyId === it.id
                return (
                  <div
                    key={it.id}
                    className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3"
                  >
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="mb-1 text-[10px] uppercase tracking-wide text-zinc-500">
                          Incoming
                        </div>
                        <div className="text-[12px] text-zinc-200">{it.content}</div>
                        <div className="mt-1 text-[10px] text-zinc-500">
                          importance {it.importance}
                          {it.source_label ? ` · from ${it.source_label}` : ''}
                          {it.tags ? ` · ${it.tags}` : ''}
                        </div>
                      </div>
                      <div>
                        <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-zinc-500">
                          <ArrowRightLeft className="h-3 w-3" />
                          Current
                        </div>
                        {live && live.content ? (
                          <>
                            <div className="text-[12px] text-zinc-400">{live.content}</div>
                            <div className="mt-1 text-[10px] text-zinc-500">
                              importance {live.importance ?? '?'}
                              {live.created_at ? ` · ${live.created_at}` : ''}
                            </div>
                          </>
                        ) : (
                          <div className="text-[11px] italic text-zinc-600">
                            Conflict live insight no longer available (may have been deleted).
                          </div>
                        )}
                      </div>
                    </div>
                    {it.conflict_reason && (
                      <div className="mt-2 text-[10px] text-zinc-600">
                        reason: {it.conflict_reason}
                      </div>
                    )}
                    <div className="mt-3 flex items-center justify-end gap-2">
                      <button
                        onClick={() => void reject(it.id)}
                        disabled={busy}
                        className="flex items-center gap-1 rounded-md border border-zinc-800 px-2.5 py-1 text-[11px] font-medium text-zinc-400 hover:bg-red-500/10 hover:text-red-300 disabled:opacity-40"
                      >
                        <Ban className="h-3 w-3" />
                        Reject
                      </button>
                      {live && live.id ? (
                        <button
                          onClick={() => void approve(it.id, false)}
                          disabled={busy}
                          className="flex items-center gap-1 rounded-md border border-zinc-800 px-2.5 py-1 text-[11px] font-medium text-zinc-300 hover:bg-emerald-500/10 hover:text-emerald-300 disabled:opacity-40"
                          title="Keep both — add incoming alongside the current insight"
                        >
                          <Check className="h-3 w-3" />
                          Keep both
                        </button>
                      ) : null}
                      <button
                        onClick={() => void approve(it.id, true)}
                        disabled={busy}
                        className="flex items-center gap-1 rounded-md bg-emerald-600/20 px-2.5 py-1 text-[11px] font-medium text-emerald-300 hover:bg-emerald-600/30 disabled:opacity-40"
                        title={
                          live && live.id
                            ? 'Promote incoming and delete current'
                            : 'Promote incoming to live insights'
                        }
                      >
                        {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
                        {live && live.id ? 'Replace current' : 'Approve'}
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
