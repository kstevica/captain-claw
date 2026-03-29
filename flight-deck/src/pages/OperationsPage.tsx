import { useEffect, useState, useMemo, useCallback } from 'react'
import {
  BarChart3,
  Zap,
  AlertTriangle,
  RefreshCw,
  Box,
  Settings2 as Cog,
  Circle,
  ArrowUpDown,
  TrendingUp,
  Cpu,
  HardDrive,
  Timer,
  Activity,
  Loader2,
} from 'lucide-react'
import { useContainerStore } from '../stores/containerStore'
import { useLocalAgentStore } from '../stores/localAgentStore'
import { useChatStore } from '../stores/chatStore'

// ── Types ──

interface UsageTotals {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  cache_creation_input_tokens: number
  cache_read_input_tokens: number
  input_bytes: number
  output_bytes: number
  total_calls: number
  avg_latency_ms: number
  error_count: number
  byok_calls: number
}

interface UsageRecord {
  id: string
  model: string
  provider: string
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  latency_ms: number
  error: number
  created_at: string
  task_name: string
  interaction: string
  cache_creation_input_tokens: number
  cache_read_input_tokens: number
}

interface AgentUsage {
  agentId: string
  agentName: string
  kind: 'docker' | 'local'
  status: string
  host: string
  port: number
  auth: string
  totals: UsageTotals | null
  records: UsageRecord[]
  loading: boolean
  error: string | null
  lastFetched: number | null
}

type Period = 'last_hour' | 'today' | 'yesterday' | 'this_week' | 'this_month' | 'all'

const PERIODS: { id: Period; label: string }[] = [
  { id: 'last_hour', label: 'Last Hour' },
  { id: 'today', label: 'Today' },
  { id: 'yesterday', label: 'Yesterday' },
  { id: 'this_week', label: 'This Week' },
  { id: 'this_month', label: 'This Month' },
  { id: 'all', label: 'All Time' },
]

// ── Helpers ──

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

function formatBytes(b: number): string {
  if (b >= 1_073_741_824) return `${(b / 1_073_741_824).toFixed(1)} GB`
  if (b >= 1_048_576) return `${(b / 1_048_576).toFixed(1)} MB`
  if (b >= 1_024) return `${(b / 1_024).toFixed(1)} KB`
  return `${b} B`
}

function formatMs(ms: number): string {
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(1)}m`
  if (ms >= 1_000) return `${(ms / 1_000).toFixed(1)}s`
  return `${ms}ms`
}

function estimateCost(totals: UsageTotals): number {
  // Rough Claude pricing: $3/MTok input, $15/MTok output, cache read $0.30/MTok, cache write $3.75/MTok
  const inputCost = (totals.prompt_tokens / 1_000_000) * 3
  const outputCost = (totals.completion_tokens / 1_000_000) * 15
  const cacheReadCost = (totals.cache_read_input_tokens / 1_000_000) * 0.30
  const cacheWriteCost = (totals.cache_creation_input_tokens / 1_000_000) * 3.75
  return inputCost + outputCost + cacheReadCost + cacheWriteCost
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  if (diff < 0) return 'just now'
  const s = Math.floor(diff / 1000)
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

// ── Fetch usage ──

async function fetchAgentUsage(host: string, port: number, auth: string, period: Period): Promise<{ totals: UsageTotals; records: UsageRecord[] }> {
  const params = new URLSearchParams({ period })
  if (auth) params.set('token', auth)
  const resp = await fetch(`/fd/agent-usage/${host}/${port}?${params}`)
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  const data = await resp.json()
  return { totals: data.totals, records: data.records || [] }
}

// ── Component ──

export function OperationsPage() {
  const containers = useContainerStore((s) => s.containers)
  const localAgents = useLocalAgentStore((s) => s.agents)
  const chatSessions = useChatStore((s) => s.sessions)

  const [period, setPeriod] = useState<Period>('today')
  const [agentUsages, setAgentUsages] = useState<Map<string, AgentUsage>>(new Map())
  const [refreshing, setRefreshing] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [sortBy, setSortBy] = useState<'tokens' | 'calls' | 'cost' | 'errors'>('tokens')

  // Build agent list
  const agents = useMemo(() => {
    const list: { id: string; name: string; kind: 'docker' | 'local'; status: string; host: string; port: number; auth: string }[] = []
    for (const c of containers) {
      if (c.web_port) {
        list.push({ id: c.id, name: c.agent_name || c.name, kind: 'docker', status: c.status, host: 'localhost', port: c.web_port, auth: c.web_auth || '' })
      }
    }
    for (const a of localAgents) {
      list.push({ id: a.id, name: a.name, kind: 'local', status: a.status === 'online' ? 'running' : a.status, host: a.host, port: a.port, auth: a.authToken || '' })
    }
    return list
  }, [containers, localAgents])

  // Fetch all usage
  const fetchAll = useCallback(async () => {
    setRefreshing(true)
    const updated = new Map(agentUsages)

    await Promise.all(agents.map(async (agent) => {
      const existing = updated.get(agent.id) || {
        agentId: agent.id, agentName: agent.name, kind: agent.kind, status: agent.status,
        host: agent.host, port: agent.port, auth: agent.auth,
        totals: null, records: [], loading: true, error: null, lastFetched: null,
      }
      updated.set(agent.id, { ...existing, loading: true, error: null })

      try {
        const data = await fetchAgentUsage(agent.host, agent.port, agent.auth, period)
        updated.set(agent.id, {
          ...existing, totals: data.totals, records: data.records,
          loading: false, error: null, lastFetched: Date.now(),
        })
      } catch (err) {
        updated.set(agent.id, {
          ...existing, loading: false, error: (err as Error).message, lastFetched: null,
        })
      }
    }))

    setAgentUsages(new Map(updated))
    setRefreshing(false)
  }, [agents, period]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { fetchAll() }, [period]) // eslint-disable-line react-hooks/exhaustive-deps

  // Aggregate totals
  const globalTotals = useMemo(() => {
    const t: UsageTotals = {
      prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
      cache_creation_input_tokens: 0, cache_read_input_tokens: 0,
      input_bytes: 0, output_bytes: 0, total_calls: 0, avg_latency_ms: 0,
      error_count: 0, byok_calls: 0,
    }
    let latencySum = 0
    let count = 0
    for (const u of agentUsages.values()) {
      if (!u.totals) continue
      t.prompt_tokens += u.totals.prompt_tokens
      t.completion_tokens += u.totals.completion_tokens
      t.total_tokens += u.totals.total_tokens
      t.cache_creation_input_tokens += u.totals.cache_creation_input_tokens
      t.cache_read_input_tokens += u.totals.cache_read_input_tokens
      t.input_bytes += u.totals.input_bytes
      t.output_bytes += u.totals.output_bytes
      t.total_calls += u.totals.total_calls
      t.error_count += u.totals.error_count
      t.byok_calls += u.totals.byok_calls
      if (u.totals.avg_latency_ms > 0) {
        latencySum += u.totals.avg_latency_ms * u.totals.total_calls
        count += u.totals.total_calls
      }
    }
    t.avg_latency_ms = count > 0 ? Math.round(latencySum / count) : 0
    return t
  }, [agentUsages])

  const totalCost = estimateCost(globalTotals)

  // Sorted agent list
  const sortedAgents = useMemo(() => {
    const list = agents.map((a) => ({ ...a, usage: agentUsages.get(a.id) }))
    list.sort((a, b) => {
      const ua = a.usage?.totals
      const ub = b.usage?.totals
      if (!ua && !ub) return 0
      if (!ua) return 1
      if (!ub) return -1
      switch (sortBy) {
        case 'tokens': return ub.total_tokens - ua.total_tokens
        case 'calls': return ub.total_calls - ua.total_calls
        case 'cost': return estimateCost(ub) - estimateCost(ua)
        case 'errors': return ub.error_count - ua.error_count
      }
    })
    return list
  }, [agents, agentUsages, sortBy])

  // Model breakdown across all agents
  const modelBreakdown = useMemo(() => {
    const models = new Map<string, { calls: number; tokens: number; errors: number }>()
    for (const u of agentUsages.values()) {
      for (const r of u.records) {
        const key = r.model || 'unknown'
        const existing = models.get(key) || { calls: 0, tokens: 0, errors: 0 }
        existing.calls++
        existing.tokens += r.total_tokens
        if (r.error) existing.errors++
        models.set(key, existing)
      }
    }
    return [...models.entries()].sort((a, b) => b[1].tokens - a[1].tokens)
  }, [agentUsages])

  // Selected agent detail
  const selectedUsage = selectedAgent ? agentUsages.get(selectedAgent) : null

  return (
    <div className="flex h-full">
      <div className="flex-1 overflow-y-auto p-6">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-violet-400" />
              Operations
            </h1>
            <p className="text-sm text-zinc-500">Token usage, costs, and performance across all agents</p>
          </div>
          <div className="flex items-center gap-2">
            {/* Period selector */}
            <div className="flex rounded-lg border border-zinc-800 overflow-hidden">
              {PERIODS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => setPeriod(p.id)}
                  className={`px-2.5 py-1 text-[11px] font-medium transition-colors ${
                    period === p.id ? 'bg-violet-600/20 text-violet-400' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <button
              onClick={fetchAll}
              disabled={refreshing}
              className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 transition-colors disabled:opacity-40"
              title="Refresh"
            >
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Global summary cards */}
        <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <SummaryCard icon={Zap} label="Total Tokens" value={formatTokens(globalTotals.total_tokens)} sub={`${formatTokens(globalTotals.prompt_tokens)} in / ${formatTokens(globalTotals.completion_tokens)} out`} color="text-blue-400" />
          <SummaryCard icon={TrendingUp} label="Est. Cost" value={`$${totalCost.toFixed(2)}`} sub={`${globalTotals.byok_calls} BYOK calls`} color="text-emerald-400" />
          <SummaryCard icon={Activity} label="API Calls" value={String(globalTotals.total_calls)} sub={`${globalTotals.error_count} errors`} color="text-violet-400" />
          <SummaryCard icon={Timer} label="Avg Latency" value={formatMs(globalTotals.avg_latency_ms)} sub="per call" color="text-amber-400" />
          <SummaryCard icon={HardDrive} label="Data" value={formatBytes(globalTotals.input_bytes + globalTotals.output_bytes)} sub={`${formatBytes(globalTotals.input_bytes)} in / ${formatBytes(globalTotals.output_bytes)} out`} color="text-cyan-400" />
          <SummaryCard icon={Cpu} label="Cache" value={formatTokens(globalTotals.cache_read_input_tokens)} sub={`${formatTokens(globalTotals.cache_creation_input_tokens)} created`} color="text-pink-400" />
        </div>

        {/* Token distribution bar */}
        {sortedAgents.some((a) => a.usage?.totals?.total_tokens) && (
          <div className="mb-6">
            <h2 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-500">Token Distribution</h2>
            <div className="flex h-6 overflow-hidden rounded-lg bg-zinc-900">
              {sortedAgents.filter((a) => a.usage?.totals?.total_tokens).map((a, i) => {
                const pct = globalTotals.total_tokens > 0
                  ? ((a.usage!.totals!.total_tokens / globalTotals.total_tokens) * 100)
                  : 0
                const colors = ['bg-violet-500', 'bg-blue-500', 'bg-emerald-500', 'bg-amber-500', 'bg-pink-500', 'bg-cyan-500', 'bg-red-500', 'bg-indigo-500']
                return (
                  <div
                    key={a.id}
                    className={`${colors[i % colors.length]} flex items-center justify-center transition-all cursor-pointer hover:brightness-110`}
                    style={{ width: `${Math.max(pct, 1)}%` }}
                    title={`${a.name}: ${formatTokens(a.usage!.totals!.total_tokens)} (${pct.toFixed(1)}%)`}
                    onClick={() => setSelectedAgent(selectedAgent === a.id ? null : a.id)}
                  >
                    {pct > 8 && <span className="text-[10px] font-medium text-white truncate px-1">{a.name}</span>}
                  </div>
                )
              })}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          {/* Per-agent usage table */}
          <div>
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-xs font-medium uppercase tracking-wider text-zinc-500">Per-Agent Usage</h2>
              <div className="flex gap-0.5">
                {(['tokens', 'calls', 'cost', 'errors'] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSortBy(s)}
                    className={`flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[10px] font-medium capitalize transition-colors ${
                      sortBy === s ? 'bg-violet-600/20 text-violet-400' : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    {s}
                    {sortBy === s && <ArrowUpDown className="h-2.5 w-2.5" />}
                  </button>
                ))}
              </div>
            </div>
            <div className="rounded-xl border border-zinc-800 overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-zinc-800 bg-zinc-900/50">
                    <th className="px-3 py-2 text-left font-medium text-zinc-500">Agent</th>
                    <th className="px-3 py-2 text-right font-medium text-zinc-500">Tokens</th>
                    <th className="px-3 py-2 text-right font-medium text-zinc-500">Calls</th>
                    <th className="px-3 py-2 text-right font-medium text-zinc-500">Cost</th>
                    <th className="px-3 py-2 text-right font-medium text-zinc-500">Latency</th>
                    <th className="px-3 py-2 text-right font-medium text-zinc-500">Errors</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60">
                  {sortedAgents.map((a) => {
                    const u = a.usage
                    const t = u?.totals
                    const isSelected = selectedAgent === a.id
                    return (
                      <tr
                        key={a.id}
                        onClick={() => setSelectedAgent(isSelected ? null : a.id)}
                        className={`cursor-pointer transition-colors ${isSelected ? 'bg-violet-600/10' : 'hover:bg-zinc-900/40'}`}
                      >
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-1.5">
                            {a.kind === 'docker' ? <Box className="h-3 w-3 text-blue-400/70" /> : <Cog className="h-3 w-3 text-amber-400/70" />}
                            <span className="font-medium text-zinc-200">{a.name}</span>
                            {u?.loading && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
                          </div>
                          {u?.error && <span className="text-[10px] text-red-400">{u.error}</span>}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-zinc-300">{t ? formatTokens(t.total_tokens) : '—'}</td>
                        <td className="px-3 py-2 text-right font-mono text-zinc-300">{t ? t.total_calls : '—'}</td>
                        <td className="px-3 py-2 text-right font-mono text-emerald-400">{t ? `$${estimateCost(t).toFixed(2)}` : '—'}</td>
                        <td className="px-3 py-2 text-right font-mono text-zinc-400">{t?.avg_latency_ms ? formatMs(t.avg_latency_ms) : '—'}</td>
                        <td className={`px-3 py-2 text-right font-mono ${t?.error_count ? 'text-red-400' : 'text-zinc-500'}`}>{t ? t.error_count : '—'}</td>
                      </tr>
                    )
                  })}
                  {sortedAgents.length === 0 && (
                    <tr><td colSpan={6} className="px-3 py-6 text-center text-zinc-600">No agents with web ports available</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model breakdown */}
          <div>
            <h2 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-500">Model Breakdown</h2>
            <div className="rounded-xl border border-zinc-800 overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-zinc-800 bg-zinc-900/50">
                    <th className="px-3 py-2 text-left font-medium text-zinc-500">Model</th>
                    <th className="px-3 py-2 text-right font-medium text-zinc-500">Calls</th>
                    <th className="px-3 py-2 text-right font-medium text-zinc-500">Tokens</th>
                    <th className="px-3 py-2 text-right font-medium text-zinc-500">Errors</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60">
                  {modelBreakdown.map(([model, stats]) => (
                    <tr key={model} className="hover:bg-zinc-900/40">
                      <td className="px-3 py-2 font-mono text-zinc-200">{model}</td>
                      <td className="px-3 py-2 text-right font-mono text-zinc-300">{stats.calls}</td>
                      <td className="px-3 py-2 text-right font-mono text-zinc-300">{formatTokens(stats.tokens)}</td>
                      <td className={`px-3 py-2 text-right font-mono ${stats.errors ? 'text-red-400' : 'text-zinc-500'}`}>{stats.errors}</td>
                    </tr>
                  ))}
                  {modelBreakdown.length === 0 && (
                    <tr><td colSpan={4} className="px-3 py-6 text-center text-zinc-600">No data yet</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Selected agent detail */}
        {selectedUsage && selectedUsage.totals && (
          <div className="mt-6">
            <h2 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
              {selectedUsage.agentName} — Recent Calls
            </h2>
            <div className="rounded-xl border border-zinc-800 overflow-hidden">
              <div className="max-h-[400px] overflow-y-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0">
                    <tr className="border-b border-zinc-800 bg-zinc-900">
                      <th className="px-3 py-1.5 text-left font-medium text-zinc-500">Time</th>
                      <th className="px-3 py-1.5 text-left font-medium text-zinc-500">Model</th>
                      <th className="px-3 py-1.5 text-left font-medium text-zinc-500">Task</th>
                      <th className="px-3 py-1.5 text-right font-medium text-zinc-500">In</th>
                      <th className="px-3 py-1.5 text-right font-medium text-zinc-500">Out</th>
                      <th className="px-3 py-1.5 text-right font-medium text-zinc-500">Cache</th>
                      <th className="px-3 py-1.5 text-right font-medium text-zinc-500">Latency</th>
                      <th className="px-3 py-1.5 text-right font-medium text-zinc-500">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/40">
                    {selectedUsage.records.slice(0, 100).map((r) => (
                      <tr key={r.id} className="hover:bg-zinc-900/30">
                        <td className="px-3 py-1.5 text-zinc-500" title={r.created_at}>{relativeTime(r.created_at)}</td>
                        <td className="px-3 py-1.5 font-mono text-zinc-300 truncate max-w-[150px]">{r.model}</td>
                        <td className="px-3 py-1.5 text-zinc-400 truncate max-w-[120px]">{r.task_name || r.interaction || '—'}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-zinc-400">{formatTokens(r.prompt_tokens)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-zinc-400">{formatTokens(r.completion_tokens)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-zinc-500">{r.cache_read_input_tokens ? formatTokens(r.cache_read_input_tokens) : '—'}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-zinc-400">{formatMs(r.latency_ms)}</td>
                        <td className="px-3 py-1.5 text-right">
                          {r.error ? (
                            <AlertTriangle className="inline h-3 w-3 text-red-400" />
                          ) : (
                            <Circle className="inline h-2 w-2 fill-emerald-400 text-emerald-400" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Agent health overview */}
        <div className="mt-6 mb-6">
          <h2 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-500">Agent Health</h2>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {sortedAgents.map((a) => {
              const u = a.usage
              const t = u?.totals
              const session = chatSessions.get(a.id)
              const isRunning = /running|online/i.test(a.status)
              const errorRate = t && t.total_calls > 0 ? (t.error_count / t.total_calls) * 100 : 0
              const health = !isRunning ? 'offline' : errorRate > 10 ? 'degraded' : errorRate > 0 ? 'warning' : 'healthy'
              const healthColor = { healthy: 'border-emerald-500/30', warning: 'border-amber-500/30', degraded: 'border-red-500/30', offline: 'border-zinc-700' }[health]
              const healthDot = { healthy: 'bg-emerald-400', warning: 'bg-amber-400', degraded: 'bg-red-400', offline: 'bg-zinc-600' }[health]

              return (
                <div key={a.id} className={`rounded-xl border ${healthColor} bg-zinc-900/30 p-3`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-1.5">
                      {a.kind === 'docker' ? <Box className="h-3 w-3 text-blue-400/70" /> : <Cog className="h-3 w-3 text-amber-400/70" />}
                      <span className="text-xs font-medium text-zinc-200">{a.name}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className={`h-2 w-2 rounded-full ${healthDot}`} />
                      <span className="text-[10px] text-zinc-500 capitalize">{health}</span>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-[10px]">
                    <div>
                      <span className="text-zinc-600">Calls</span>
                      <div className="font-mono text-zinc-300">{t?.total_calls ?? '—'}</div>
                    </div>
                    <div>
                      <span className="text-zinc-600">Tokens</span>
                      <div className="font-mono text-zinc-300">{t ? formatTokens(t.total_tokens) : '—'}</div>
                    </div>
                    <div>
                      <span className="text-zinc-600">Errors</span>
                      <div className={`font-mono ${t?.error_count ? 'text-red-400' : 'text-zinc-500'}`}>
                        {t ? `${t.error_count} (${errorRate.toFixed(0)}%)` : '—'}
                      </div>
                    </div>
                    <div>
                      <span className="text-zinc-600">Latency</span>
                      <div className="font-mono text-zinc-300">{t?.avg_latency_ms ? formatMs(t.avg_latency_ms) : '—'}</div>
                    </div>
                    <div>
                      <span className="text-zinc-600">Chat</span>
                      <div className={session?.connected ? 'text-emerald-400' : 'text-zinc-600'}>
                        {session?.connected ? 'Connected' : 'Off'}
                      </div>
                    </div>
                    <div>
                      <span className="text-zinc-600">Cost</span>
                      <div className="font-mono text-emerald-400">{t ? `$${estimateCost(t).toFixed(2)}` : '—'}</div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Summary Card ──

function SummaryCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
}: {
  icon: typeof Zap
  label: string
  value: string
  sub: string
  color: string
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-3">
      <div className="flex items-center gap-1.5 mb-1">
        <Icon className={`h-3.5 w-3.5 ${color}`} />
        <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">{label}</span>
      </div>
      <div className="text-lg font-semibold text-zinc-100">{value}</div>
      <div className="text-[10px] text-zinc-500">{sub}</div>
    </div>
  )
}
