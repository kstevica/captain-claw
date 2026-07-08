import { useMemo, useState } from 'react'
import { Brain, Check, Search, Users, Network } from 'lucide-react'
import type { BasnaSession } from '../../stores/basnaStore'

function isVatraConfig(config: string): boolean {
  try { return JSON.parse(config || '{}')?.mode === 'vatra' } catch { return false }
}

function fmtDate(s: string): string {
  const d = new Date(s)
  if (isNaN(d.getTime())) return ''
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

/**
 * Multi-select of FINISHED prior runs whose knowledge (final report + gaps/blind
 * spots, optionally the shared board) seeds a new run. Shared by the prepare form
 * and the creation wizard.
 */
export function KnowledgePicker({
  sessions,
  selectedIds,
  onToggle,
  includeBoard,
  onIncludeBoard,
}: {
  sessions: BasnaSession[]
  selectedIds: string[]
  onToggle: (id: string) => void
  includeBoard: boolean
  onIncludeBoard: (v: boolean) => void
}) {
  const [query, setQuery] = useState('')
  const finished = useMemo(
    () => sessions.filter((s) => s.status === 'done' && (s.truth || '').trim()),
    [sessions],
  )
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    return finished.filter((s) => !q || (s.title || s.intent || '').toLowerCase().includes(q))
  }, [finished, query])

  if (finished.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-zinc-800 px-3 py-4 text-center text-[11px] text-zinc-500">
        No finished runs yet — once a Basna/Vatra run completes, its report + gaps can seed future runs.
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {finished.length > 5 && (
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-600" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search prior runs…"
            className="w-full rounded-md border border-zinc-700 bg-zinc-950/60 py-1 pl-7 pr-2 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
          />
        </div>
      )}
      <ul className="max-h-52 space-y-1 overflow-y-auto pr-0.5">
        {shown.map((s) => {
          const sel = selectedIds.includes(s.id)
          const vatra = isVatraConfig(s.config)
          const Icon = vatra ? Users : Network
          return (
            <li key={s.id}>
              <button
                type="button"
                onClick={() => onToggle(s.id)}
                className={`flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-left transition-colors ${
                  sel
                    ? 'border-violet-500/50 bg-violet-500/10'
                    : 'border-zinc-800 bg-zinc-900/50 hover:border-zinc-700'
                }`}
              >
                <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border ${
                  sel ? 'border-violet-500 bg-violet-600 text-white' : 'border-zinc-600'
                }`}>
                  {sel && <Check className="h-3 w-3" />}
                </span>
                <Icon className={`h-3.5 w-3.5 shrink-0 ${vatra ? 'text-violet-400' : 'text-sky-500 dark:text-sky-400'}`} />
                <span className={`min-w-0 flex-1 truncate text-[11px] ${sel ? 'text-violet-700 dark:text-violet-200' : 'text-zinc-300'}`}>
                  {s.title || s.intent || 'Untitled run'}
                </span>
                {s.domain && <span className="shrink-0 text-[10px] text-zinc-500">{s.domain}</span>}
                <span className="shrink-0 text-[10px] text-zinc-600">{fmtDate(s.created_at)}</span>
              </button>
            </li>
          )
        })}
        {shown.length === 0 && (
          <li className="px-2 py-3 text-center text-[11px] text-zinc-500">No runs match.</li>
        )}
      </ul>
      <label className="flex cursor-pointer items-center gap-1.5 text-[11px] text-zinc-400">
        <input
          type="checkbox"
          checked={includeBoard}
          onChange={(e) => onIncludeBoard(e.target.checked)}
          className="h-3.5 w-3.5 rounded border-zinc-700 bg-zinc-950/60 accent-violet-600"
        />
        <Brain className="h-3.5 w-3.5 text-violet-400" />
        Include the team board notes (richer, but more tokens)
      </label>
    </div>
  )
}
