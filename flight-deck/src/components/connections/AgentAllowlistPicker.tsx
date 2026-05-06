import { useEffect, useState } from 'react'
import { Loader2, Users, RefreshCw, Check, X as XIcon, AlertCircle } from 'lucide-react'

interface FleetAgent {
  name: string
  slug: string
  kind: string
  host: string
  port: number
  status: string
  description: string
}

interface AgentAllowlistPickerProps {
  /** Currently selected agent slugs. Empty array = "every agent in the fleet". */
  value: string[]
  onChange: (next: string[]) => void
  /** Optional class name on the outer wrapper. */
  className?: string
}

/**
 * Multi-select picker that lists Flight Deck's fleet (docker + process agents)
 * and lets the user toggle which slugs are on the MCP server's allowlist.
 *
 * Empty selection ("Allow all") collapses to the Phase 1 behaviour: every
 * agent in the fleet can see/list/call the server.
 */
export function AgentAllowlistPicker({
  value,
  onChange,
  className,
}: AgentAllowlistPickerProps) {
  const [fleet, setFleet] = useState<FleetAgent[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Selection might contain slugs no longer present in the fleet (renamed
  // agents, agents removed since the server was saved). We surface them as
  // "ghost" entries so the user can keep or drop them.
  const knownSlugs = new Set(fleet.map((a) => a.slug))
  const ghosts = value.filter((s) => !knownSlugs.has(s))

  async function refresh() {
    setLoading(true)
    setError(null)
    try {
      const resp = await fetch('/fd/fleet', { credentials: 'include' })
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
      const data: FleetAgent[] = await resp.json()
      // Stable sort: running first, then by name.
      data.sort((a, b) => {
        if (a.status === b.status) return a.name.localeCompare(b.name)
        if (a.status === 'running') return -1
        if (b.status === 'running') return 1
        return 0
      })
      setFleet(data)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    // Manual refresh only — fleet shouldn't churn while the form is open.
  }, [])

  function toggle(slug: string) {
    if (value.includes(slug)) {
      onChange(value.filter((s) => s !== slug))
    } else {
      onChange([...value, slug])
    }
  }

  function selectAll() {
    onChange(fleet.map((a) => a.slug))
  }
  function clearAll() {
    onChange([])
  }

  const selectedCount = value.length
  const fleetAllowed = selectedCount === 0

  return (
    <div className={className}>
      <div className="flex items-center gap-2 mb-2">
        <Users className="h-3.5 w-3.5 text-zinc-400" />
        <span className="text-xs font-medium text-zinc-300">
          Allowed agents
        </span>
        <span
          className={
            'text-[10px] px-2 py-0.5 rounded-full border ' +
            (fleetAllowed
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              : 'bg-blue-500/10 text-blue-400 border-blue-500/30')
          }
        >
          {fleetAllowed
            ? 'Allow all (fleet-wide)'
            : `${selectedCount} agent${selectedCount === 1 ? '' : 's'}`}
        </span>
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={refresh}
            disabled={loading}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-700 hover:bg-zinc-800 px-2 py-1 text-[11px] text-zinc-300 disabled:opacity-50"
            title="Reload fleet"
          >
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCw className="h-3 w-3" />
            )}
          </button>
          <button
            type="button"
            onClick={selectAll}
            disabled={loading || fleet.length === 0}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-700 hover:bg-zinc-800 px-2 py-1 text-[11px] text-zinc-300 disabled:opacity-50"
          >
            All
          </button>
          <button
            type="button"
            onClick={clearAll}
            disabled={selectedCount === 0}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-700 hover:bg-zinc-800 px-2 py-1 text-[11px] text-zinc-300 disabled:opacity-50"
          >
            Clear
          </button>
        </div>
      </div>

      <p className="text-[11px] text-zinc-500 mb-2">
        Pick the agents that may see and call this server. Leave empty to allow
        every agent in the fleet.
      </p>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-2.5 py-1.5 text-[11px] text-red-400 mb-2">
          <AlertCircle className="h-3 w-3 shrink-0 mt-0.5" />
          <span className="break-all">Failed to load fleet: {error}</span>
        </div>
      )}

      <div className="rounded-md border border-zinc-800 bg-zinc-950/40 max-h-48 overflow-y-auto divide-y divide-zinc-800/60">
        {loading && fleet.length === 0 && (
          <div className="flex items-center gap-2 px-3 py-2 text-xs text-zinc-500">
            <Loader2 className="h-3 w-3 animate-spin" /> Loading fleet…
          </div>
        )}
        {!loading && fleet.length === 0 && !error && (
          <div className="px-3 py-2 text-xs text-zinc-500">
            No agents found in the fleet yet.
          </div>
        )}
        {fleet.map((a) => {
          const checked = value.includes(a.slug)
          const isRunning = a.status === 'running'
          return (
            <label
              key={`${a.kind}-${a.slug}`}
              className="flex items-center gap-2 px-2.5 py-1.5 text-xs hover:bg-zinc-900/60 cursor-pointer"
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggle(a.slug)}
                className="h-3.5 w-3.5 accent-violet-500"
              />
              <span className="text-zinc-100 font-medium truncate max-w-[12rem]">
                {a.name}
              </span>
              <span className="text-[10px] rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">
                {a.kind}
              </span>
              <span
                className={
                  'text-[10px] rounded px-1.5 py-0.5 ' +
                  (isRunning
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : 'bg-zinc-800 text-zinc-500')
                }
              >
                {a.status}
              </span>
              <span className="ml-auto text-[10px] font-mono text-zinc-500 truncate max-w-[10rem]">
                {a.slug}
              </span>
            </label>
          )
        })}
        {ghosts.length > 0 && (
          <div className="px-2.5 py-1.5 bg-amber-500/5 border-t border-amber-500/20">
            <div className="text-[10px] text-amber-400 mb-1">
              Selected slugs not in current fleet:
            </div>
            <div className="flex flex-wrap gap-1">
              {ghosts.map((slug) => (
                <span
                  key={slug}
                  className="inline-flex items-center gap-1 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] text-amber-300"
                >
                  <Check className="h-2.5 w-2.5" />
                  <span className="font-mono">{slug}</span>
                  <button
                    type="button"
                    onClick={() => toggle(slug)}
                    className="ml-0.5 text-amber-300/70 hover:text-amber-200"
                    title="Remove from allowlist"
                  >
                    <XIcon className="h-2.5 w-2.5" />
                  </button>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
