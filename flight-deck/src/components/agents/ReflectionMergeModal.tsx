import { useCallback, useEffect, useMemo, useState } from 'react'
import { X, Brain, Sparkles, Loader2, AlertCircle, ChevronRight } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'

/**
 * Personality-preserving reflection merge picker.
 *
 * Lists every staged `reflections/imported/<label>/` source on the target
 * agent, lets the user pick one, and kicks off the LLM merge flow on the
 * agent side. The merged reflection becomes the new active personality —
 * the imported file stays in place so the user can re-run or pick another.
 */

export interface ReflectionMergeTarget {
  host: string
  port: number
  authToken: string
  label: string
}

interface ImportedReflection {
  filename: string
  path: string
  timestamp: string
  summary: string
  topics_reviewed: string[]
}

interface ImportedSource {
  label: string
  count: number
  latest_mtime: number
  reflections: ImportedReflection[]
}

interface MergeResultReflection {
  timestamp: string
  summary: string
  topics_reviewed: string[]
}

interface Props {
  target: ReflectionMergeTarget
  onClose: () => void
}

function fdAuthHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = {}
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

export function ReflectionMergeModal({ target, onClose }: Props) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sources, setSources] = useState<ImportedSource[]>([])
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [merging, setMerging] = useState(false)
  const [mergeResult, setMergeResult] = useState<MergeResultReflection | null>(null)

  const base = `/fd/agent-memory/${encodeURIComponent(target.host)}/${target.port}`

  const loadSources = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams()
      if (target.authToken) params.set('token', target.authToken)
      const res = await fetch(`${base}/reflections/imported?${params}`, {
        headers: fdAuthHeaders(),
        credentials: 'include',
      })
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text().catch(() => '')}`)
      const data = await res.json()
      setSources(Array.isArray(data.sources) ? data.sources : [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [base, target.authToken])

  useEffect(() => {
    void loadSources()
  }, [loadSources])

  const selected = useMemo(() => {
    if (!selectedPath) return null
    for (const s of sources) {
      for (const r of s.reflections) {
        if (`${s.label}/${r.filename}` === selectedPath) {
          return { source: s, reflection: r }
        }
      }
    }
    return null
  }, [selectedPath, sources])

  const handleMerge = useCallback(async () => {
    if (!selected) return
    const confirmed = confirm(
      `Merge reflection '${selected.reflection.filename}' from '${selected.source.label}' into the active personality of '${target.label}'?\n\n` +
        `This runs the agent's LLM merge flow. The result will REPLACE the active reflection. The imported file will stay in place for re-runs.`
    )
    if (!confirmed) return

    setMerging(true)
    setError(null)
    try {
      const params = new URLSearchParams()
      if (target.authToken) params.set('token', target.authToken)
      const res = await fetch(`${base}/reflections/merge?${params}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...fdAuthHeaders() },
        credentials: 'include',
        body: JSON.stringify({
          label: selected.source.label,
          filename: selected.reflection.filename,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text().catch(() => '')}`)
      const data = await res.json()
      setMergeResult(data.reflection as MergeResultReflection)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setMerging(false)
    }
  }, [base, selected, target.authToken, target.label])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="flex max-h-[85vh] w-[720px] flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-semibold">Merge Reflection into {target.label}</h2>
          </div>
          <button
            onClick={onClose}
            className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          <div className="text-xs text-zinc-500">
            Imported reflections are staged under{' '}
            <code className="text-zinc-400">reflections/imported/&lt;label&gt;/</code>.
            Pick one to merge — the agent runs a LLM synthesis that{' '}
            <span className="text-violet-300">preserves the current personality</span> while
            absorbing new knowledge and directives.
          </div>

          {loading && (
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading imported reflections...
            </div>
          )}

          {error && !loading && (
            <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <div>
                <div className="font-semibold">Failed</div>
                <div className="text-red-300/80">{error}</div>
                <button
                  onClick={() => void loadSources()}
                  className="mt-2 text-red-200 underline"
                >
                  Retry
                </button>
              </div>
            </div>
          )}

          {!loading && !error && sources.length === 0 && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 text-xs text-zinc-500">
              No imported reflections staged yet. Import a memory bundle from another agent
              first — use the <span className="text-zinc-300">Import Memory</span> action in
              the Actions menu.
            </div>
          )}

          {!loading && sources.length > 0 && !mergeResult && (
            <div className="space-y-3">
              {sources.map((s) => (
                <div key={s.label} className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="text-xs font-semibold text-zinc-200">{s.label}</div>
                    <div className="text-[10px] text-zinc-500">{s.count} reflection(s)</div>
                  </div>
                  <div className="space-y-1.5">
                    {s.reflections.map((r) => {
                      const path = `${s.label}/${r.filename}`
                      const isSelected = selectedPath === path
                      return (
                        <button
                          key={path}
                          onClick={() => setSelectedPath(path)}
                          className={`w-full rounded-md border px-3 py-2 text-left transition-colors ${
                            isSelected
                              ? 'border-violet-500/50 bg-violet-500/10'
                              : 'border-zinc-800 bg-zinc-950/50 hover:border-zinc-700 hover:bg-zinc-900'
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="text-[11px] font-medium text-zinc-300 truncate">
                              {r.filename}
                            </div>
                            <div className="text-[10px] text-zinc-500 shrink-0">
                              {r.timestamp}
                            </div>
                          </div>
                          <div className="mt-1 text-[11px] text-zinc-500 line-clamp-2">
                            {r.summary.slice(0, 240)}
                            {r.summary.length > 240 ? '…' : ''}
                          </div>
                          {r.topics_reviewed.length > 0 && (
                            <div className="mt-1 text-[10px] text-zinc-600">
                              topics: {r.topics_reviewed.join(', ')}
                            </div>
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}

          {mergeResult && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs text-emerald-300">
                <Sparkles className="h-4 w-4" />
                <div>
                  <div className="font-semibold">Merge complete</div>
                  <div className="text-emerald-300/80">
                    New active reflection saved at {mergeResult.timestamp}
                  </div>
                </div>
              </div>
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
                <div className="mb-1 text-[10px] uppercase tracking-wide text-zinc-500">
                  Merged reflection
                </div>
                <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap text-[11px] leading-relaxed text-zinc-300">
                  {mergeResult.summary}
                </pre>
              </div>
            </div>
          )}
        </div>

        {/* Footer actions */}
        <div className="flex items-center justify-between border-t border-zinc-800 px-5 py-3">
          <div className="text-[10px] text-zinc-600">
            {selected
              ? `Selected: ${selected.source.label}/${selected.reflection.filename}`
              : 'Pick an imported reflection to merge.'}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="rounded-md border border-zinc-800 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
            >
              {mergeResult ? 'Done' : 'Cancel'}
            </button>
            {!mergeResult && (
              <button
                onClick={() => void handleMerge()}
                disabled={!selected || merging}
                className="flex items-center gap-1.5 rounded-md bg-violet-600/20 px-3 py-1.5 text-xs font-medium text-violet-300 hover:bg-violet-600/30 disabled:opacity-40"
              >
                {merging ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Merging via LLM…
                  </>
                ) : (
                  <>
                    Merge into active
                    <ChevronRight className="h-3 w-3" />
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
