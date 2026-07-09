import { useEffect, useState, useCallback, useRef } from 'react'
import {
  Activity,
  RefreshCw,
  ChevronRight,
  ChevronDown,
  Loader2,
  Square,
  Cpu,
  HardDrive,
  Server,
  Globe,
  Boxes,
  Terminal,
  Search,
  Play,
  AlertTriangle,
} from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import {
  getSystemProcesses,
  stopSystemProcess,
  type SystemProcessNode,
  type SystemProcessResponse,
} from '../services/system'

const REFRESH_MS = 4000

// ── Formatting helpers ──

function fmtMem(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${Math.round(mb)} MB`
}

const KIND_META: Record<string, { label: string; cls: string; Icon: typeof Server }> = {
  agent: { label: 'agent', cls: 'bg-violet-500/15 text-violet-300 border-violet-500/30', Icon: Boxes },
  'hosted-app': { label: 'hosted', cls: 'bg-sky-500/15 text-sky-300 border-sky-500/30', Icon: Globe },
  'code-app': { label: 'code', cls: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30', Icon: Terminal },
  'flight-deck': { label: 'system', cls: 'bg-amber-500/15 text-amber-300 border-amber-500/30', Icon: Server },
  child: { label: 'child', cls: 'bg-zinc-700/40 text-zinc-400 border-zinc-600/40', Icon: Terminal },
}

/** Keep a node if it — or any descendant — matches the filter query. */
function nodeMatches(n: SystemProcessNode, q: string): boolean {
  if (!q) return true
  const needle = q.toLowerCase()
  if (
    n.name.toLowerCase().includes(needle) ||
    n.command.toLowerCase().includes(needle) ||
    n.label.toLowerCase().includes(needle) ||
    String(n.pid).includes(needle)
  )
    return true
  return n.children.some((c) => nodeMatches(c, needle))
}

// ── Tree row ──

function ProcessRow({
  node,
  depth,
  canStop,
  busyPid,
  onStop,
  filter,
}: {
  node: SystemProcessNode
  depth: number
  canStop: boolean
  busyPid: number | null
  onStop: (node: SystemProcessNode, tree: boolean) => void
  filter: string
}) {
  const [expanded, setExpanded] = useState(depth === 0)
  const hasChildren = node.children.length > 0
  const meta = KIND_META[node.kind] || KIND_META.child
  const busy = busyPid === node.pid
  const isServer = node.kind === 'flight-deck' && node.is_root

  if (!nodeMatches(node, filter)) return null

  const cpu = node.is_root ? node.agg_cpu : node.cpu
  const mem = node.is_root ? node.agg_mem_mb : node.rss_mb

  return (
    <div>
      <div
        className="group flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-zinc-800/40"
        style={{ paddingLeft: `${8 + depth * 18}px` }}
      >
        {/* expand toggle */}
        {hasChildren ? (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="shrink-0 text-zinc-500 hover:text-zinc-300"
          >
            {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </button>
        ) : (
          <span className="w-3.5 shrink-0" />
        )}

        {/* kind badge */}
        <span
          className={`inline-flex shrink-0 items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-medium ${meta.cls}`}
        >
          <meta.Icon className="h-3 w-3" />
          {meta.label}
        </span>

        {/* name / command */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm text-zinc-200">{node.label}</span>
            {node.is_root && node.detail && (
              <span className="truncate text-xs text-zinc-500">{node.detail}</span>
            )}
            {node.is_root && node.descendant_count > 0 && (
              <span className="shrink-0 text-[10px] text-zinc-600">
                +{node.descendant_count} child{node.descendant_count === 1 ? '' : 'ren'}
              </span>
            )}
          </div>
          <div className="truncate font-mono text-[11px] text-zinc-600">{node.command}</div>
        </div>

        {/* owner (admin) */}
        {node.owner_email && (
          <span className="hidden shrink-0 text-xs text-zinc-500 sm:inline">{node.owner_email}</span>
        )}

        {/* metrics */}
        <span className="w-16 shrink-0 text-right font-mono text-xs text-zinc-400" title="CPU">
          {cpu.toFixed(1)}%
        </span>
        <span className="w-20 shrink-0 text-right font-mono text-xs text-zinc-400" title="Memory (RSS)">
          {fmtMem(mem)}
        </span>
        <span className="hidden w-20 shrink-0 text-right font-mono text-xs text-zinc-600 md:inline" title="Elapsed">
          {node.elapsed}
        </span>
        <span className="hidden w-16 shrink-0 text-right font-mono text-[11px] text-zinc-600 lg:inline" title="PID">
          {node.pid}
        </span>

        {/* actions */}
        <div className="flex w-24 shrink-0 items-center justify-end gap-1">
          {canStop && !isServer ? (
            <>
              <button
                onClick={() => onStop(node, false)}
                disabled={busy}
                title="Stop this process"
                className="rounded border border-zinc-700 p-1 text-zinc-400 hover:border-red-800/60 hover:bg-red-950/40 hover:text-red-400 disabled:opacity-40"
              >
                {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Square className="h-3.5 w-3.5" />}
              </button>
              {node.is_root && node.descendant_count > 0 && (
                <button
                  onClick={() => onStop(node, true)}
                  disabled={busy}
                  title="Stop this process and all its children"
                  className="rounded border border-zinc-700 px-1.5 py-1 text-[10px] text-zinc-400 hover:border-red-800/60 hover:bg-red-950/40 hover:text-red-400 disabled:opacity-40"
                >
                  tree
                </button>
              )}
            </>
          ) : (
            isServer && <span className="text-[10px] text-zinc-600">protected</span>
          )}
        </div>
      </div>

      {expanded &&
        node.children.map((child) => (
          <ProcessRow
            key={child.pid}
            node={child}
            depth={depth + 1}
            canStop={canStop}
            busyPid={busyPid}
            onStop={onStop}
            filter={filter}
          />
        ))}
    </div>
  )
}

// ── Vital tile ──

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-3">
      <div className="text-lg font-semibold text-zinc-100">{value}</div>
      <div className="text-xs text-zinc-500">{label}</div>
      {sub && <div className="mt-0.5 text-[11px] text-zinc-600">{sub}</div>}
    </div>
  )
}

// ── Page ──

export function SystemPage() {
  const currentUser = useAuthStore((s) => s.user)
  const [data, setData] = useState<SystemProcessResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busyPid, setBusyPid] = useState<number | null>(null)
  const [auto, setAuto] = useState(true)
  const [onlyMine, setOnlyMine] = useState(false)
  const [filter, setFilter] = useState('')
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)

  const load = useCallback(async (spinner = false) => {
    if (spinner) setLoading(true)
    try {
      const res = await getSystemProcesses()
      setData(res)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load processes')
    } finally {
      if (spinner) setLoading(false)
    }
  }, [])

  useEffect(() => {
    load(true)
  }, [load])

  useEffect(() => {
    if (timer.current) clearInterval(timer.current)
    if (auto && !busyPid) {
      timer.current = setInterval(() => load(false), REFRESH_MS)
    }
    return () => {
      if (timer.current) clearInterval(timer.current)
    }
  }, [auto, busyPid, load])

  const handleStop = async (node: SystemProcessNode, tree: boolean) => {
    const what = tree
      ? `Stop "${node.label}" (pid ${node.pid}) and its ${node.descendant_count} child process(es)?`
      : `Stop "${node.label}" (pid ${node.pid})?`
    if (!confirm(what)) return
    setBusyPid(node.pid)
    try {
      await stopSystemProcess(node.pid, tree)
      await load(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to stop process')
    } finally {
      setBusyPid(null)
    }
  }

  const host = data?.host
  const summary = data?.summary
  const isAdmin = data?.is_admin

  let trees = data?.trees ?? []
  if (onlyMine && currentUser) {
    trees = trees.filter((t) => t.owner === currentUser.id)
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-5 p-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="h-5 w-5 text-violet-400" />
            <div>
              <h1 className="text-lg font-semibold text-zinc-100">System Processes</h1>
              <p className="text-xs text-zinc-500">
                {isAdmin
                  ? 'All agent processes and their children, across every user.'
                  : 'Your agent processes and any child processes they spawned.'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setAuto((v) => !v)}
              className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors ${
                auto
                  ? 'border-emerald-500/50 text-emerald-600 hover:bg-emerald-500/10'
                  : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'
              }`}
              title={auto ? 'Auto-refresh on — click to pause' : 'Auto-refresh off — click to resume'}
            >
              {auto ? (
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-500 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
                </span>
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
              {auto ? 'Live' : 'Paused'}
            </button>
            <button
              onClick={() => load(true)}
              disabled={loading}
              className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-2 rounded-md border border-red-900/50 bg-red-950/20 px-4 py-2 text-sm text-red-400">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {data && !data.available && (
          <div className="rounded-md border border-amber-900/50 bg-amber-950/20 px-4 py-2 text-sm text-amber-400">
            Process enumeration is unavailable on this host (couldn't read the process table).
          </div>
        )}

        {/* Host vitals */}
        {host && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-3">
              <div className="flex items-center gap-1.5 text-lg font-semibold text-zinc-100">
                <Cpu className="h-4 w-4 text-zinc-500" />
                {host.load_avg ? host.load_avg[0].toFixed(2) : '—'}
              </div>
              <div className="text-xs text-zinc-500">Load avg (1m)</div>
              <div className="mt-0.5 text-[11px] text-zinc-600">
                {host.cpu_count ? `${host.cpu_count} cores` : ''}
                {host.load_avg ? ` · ${host.load_avg[1].toFixed(2)} / ${host.load_avg[2].toFixed(2)}` : ''}
              </div>
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-3">
              <div className="text-lg font-semibold text-zinc-100">
                {host.mem_percent != null ? `${host.mem_percent}%` : '—'}
              </div>
              <div className="text-xs text-zinc-500">Memory used</div>
              <div className="mt-0.5 text-[11px] text-zinc-600">
                {host.mem_used_mb != null && host.mem_total_mb != null
                  ? `${fmtMem(host.mem_used_mb)} / ${fmtMem(host.mem_total_mb)}`
                  : ''}
              </div>
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-3">
              <div className="flex items-center gap-1.5 text-lg font-semibold text-zinc-100">
                <HardDrive className="h-4 w-4 text-zinc-500" />
                {host.disk_free_gb != null ? `${host.disk_free_gb} GB` : '—'}
              </div>
              <div className="text-xs text-zinc-500">Disk free</div>
              <div className="mt-0.5 text-[11px] text-zinc-600">
                {host.disk_total_gb != null ? `of ${host.disk_total_gb} GB` : ''}
              </div>
            </div>
            {summary && (
              <Tile
                label={isAdmin ? 'Memory · all agents/processes' : 'Memory · your agents/processes'}
                value={fmtMem(summary.total_mem_mb)}
                sub={`${summary.roots} proc · ${summary.children} child · ${summary.total_cpu.toFixed(0)}% CPU`}
              />
            )}
          </div>
        )}

        {/* Per-user memory breakdown (admin) */}
        {isAdmin && data && data.by_user && data.by_user.length > 1 && (
          <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/50">
            <div className="border-b border-zinc-800 px-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
              Memory by user
            </div>
            {(() => {
              const maxMem = Math.max(...data.by_user.map((u) => u.mem_mb), 1)
              return data.by_user.map((u) => (
                <div
                  key={u.owner_email}
                  className="relative flex items-center gap-3 border-b border-zinc-800/50 px-4 py-2 text-sm last:border-0"
                >
                  <div
                    className="absolute inset-y-0 left-0 bg-violet-500/10"
                    style={{ width: `${(u.mem_mb / maxMem) * 100}%` }}
                  />
                  <span className="relative min-w-0 flex-1 truncate text-zinc-300">{u.owner_email}</span>
                  <span className="relative w-24 shrink-0 text-right text-xs text-zinc-500">{u.procs} proc</span>
                  <span className="relative w-16 shrink-0 text-right font-mono text-xs text-zinc-400">
                    {u.cpu.toFixed(1)}%
                  </span>
                  <span className="relative w-24 shrink-0 text-right font-mono text-sm text-zinc-200">
                    {fmtMem(u.mem_mb)}
                  </span>
                </div>
              ))
            })()}
          </div>
        )}

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-600" />
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter by name, command or pid…"
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 py-1.5 pl-8 pr-2.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
            />
          </div>
          {isAdmin && currentUser && (
            <label className="flex cursor-pointer items-center gap-2 text-xs text-zinc-400">
              <input
                type="checkbox"
                checked={onlyMine}
                onChange={(e) => setOnlyMine(e.target.checked)}
                className="accent-violet-600"
              />
              Only mine
            </label>
          )}
          {summary && summary.stopped > 0 && (
            <span className="text-xs text-zinc-600">{summary.stopped} stopped agent(s)</span>
          )}
        </div>

        {/* Tree */}
        {loading && !data ? (
          <div className="flex justify-center py-16">
            <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
          </div>
        ) : trees.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-800 py-16 text-center text-sm text-zinc-500">
            No running processes{onlyMine ? ' for you' : ''}.
          </div>
        ) : (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/30">
            {/* column header */}
            <div className="flex items-center gap-2 border-b border-zinc-800 px-2 py-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-600">
              <span className="w-3.5 shrink-0" />
              <span className="flex-1">Process</span>
              <span className="w-16 shrink-0 text-right">CPU</span>
              <span className="w-20 shrink-0 text-right">Memory</span>
              <span className="hidden w-20 shrink-0 text-right md:inline">Uptime</span>
              <span className="hidden w-16 shrink-0 text-right lg:inline">PID</span>
              <span className="w-24 shrink-0 text-right">Actions</span>
            </div>
            <div className="py-1">
              {trees.map((tree) => (
                <ProcessRow
                  key={tree.pid}
                  node={tree}
                  depth={0}
                  canStop
                  busyPid={busyPid}
                  onStop={handleStop}
                  filter={filter}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
