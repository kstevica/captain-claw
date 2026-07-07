import { useState } from 'react'
import { DollarSign } from 'lucide-react'
import type { RunCost } from '../stores/basnaStore'

// Shared across Basna, Vatra (BasnaPage) and Code (CodePage). The human hourly
// wage is a single global preference persisted here, so every surface compares
// against the same number without any store plumbing.
const _WAGE_LS = 'fd.wagePerHour'

function fmtUsd(v: number | null | undefined): string {
  if (v == null) return '—'
  if (v >= 1) return `$${v.toFixed(2)}`
  if (v >= 0.01) return `$${v.toFixed(3)}`
  return `$${v.toFixed(5)}`
}
function fmtDur(s: number | null | undefined): string {
  const n = Math.round(s || 0)
  if (n < 60) return `${n}s`
  if (n < 3600) return `${Math.floor(n / 60)}m ${n % 60}s`
  return `${Math.floor(n / 3600)}h ${Math.floor((n % 3600) / 60)}m`
}
function fmtTok(n?: number): string {
  const v = n || 0
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`
  return String(v)
}

// The terminal `cost` progress event carries the run's cost block; fall back to a
// caller-supplied value (e.g. the execute response). Null until a run finishes.
export function deriveCost(
  progress: { stage: string; cost?: RunCost }[],
  fallback?: RunCost | null,
): RunCost | null {
  for (let i = progress.length - 1; i >= 0; i--) {
    if (progress[i].stage === 'cost' && progress[i].cost) return progress[i].cost as RunCost
  }
  return fallback ?? null
}

function loadWage(): number {
  try {
    const raw = typeof localStorage !== 'undefined' && localStorage.getItem(_WAGE_LS)
    const n = raw ? Number(raw) : NaN
    return Number.isFinite(n) && n >= 0 ? n : 0
  } catch { return 0 }
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="flex flex-col">
      <span className="text-[9px] uppercase tracking-wide text-zinc-500">{label}</span>
      <span className={`text-sm font-semibold tabular-nums ${accent ? 'text-emerald-700 dark:text-emerald-300' : 'text-zinc-800 dark:text-zinc-100'}`}>{value}</span>
    </div>
  )
}

/**
 * Run cost: dollars + effective $/hour, made directly comparable to a human wage.
 * $/hour is spend ÷ wall-clock; entering an hourly wage shows how many times
 * cheaper (or dearer) running the system is per hour. Shown once a run finishes.
 */
export function CostCard({ cost }: { cost: RunCost }) {
  const [wage, setWage] = useState<number>(loadWage)
  const [open, setOpen] = useState(false)
  const onWage = (n: number) => {
    const v = Number.isFinite(n) && n >= 0 ? n : 0
    setWage(v)
    try { localStorage.setItem(_WAGE_LS, String(v)) } catch { /* ignore */ }
  }
  const tok = cost.tokens
  const totalTok = (tok.prompt_tokens || 0) + (tok.completion_tokens || 0)
  // Token split: fresh input (incl. cache writes) · reused from cache · generated output.
  const inputTok = (tok.prompt_tokens || 0) + (tok.cache_creation_input_tokens || 0)
  const cachedTok = tok.cache_read_input_tokens || 0
  const outputTok = tok.completion_tokens || 0
  // Time: real wall-clock vs total agent-time (Σ of every model call — larger when
  // agents run in parallel); their ratio is the effective parallelism.
  const wall = cost.elapsed_seconds
  const agentT = cost.agent_seconds
  const parallel = wall && agentT && wall > 0 ? agentT / wall : null
  const hourly = cost.hourly_usd
  const ratio = wage > 0 && hourly && hourly > 0 ? wage / hourly : null
  return (
    <div className="rounded-lg border border-emerald-300/60 bg-emerald-50/60 p-4 dark:border-emerald-800/40 dark:bg-emerald-900/10">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <div className="flex items-center gap-1.5">
          <DollarSign className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          <span className="text-xs font-semibold uppercase tracking-wide text-emerald-700 dark:text-emerald-300">Run cost</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="text-2xl font-bold tabular-nums text-zinc-900 dark:text-zinc-50">{cost.priced ? fmtUsd(cost.usd) : '—'}</span>
          {!cost.priced && <span className="text-[10px] text-zinc-500">unpriced model</span>}
        </div>
        <Stat label="wall-clock" value={fmtDur(wall)} />
        {agentT != null && <Stat label="agent-time" value={fmtDur(agentT)} />}
        {parallel != null && parallel > 1.05 && <Stat label="parallel" value={`${parallel.toFixed(1)}×`} />}
        {hourly != null && <Stat label="per hour" value={`${fmtUsd(hourly)}/hr`} accent />}
      </div>
      {/* Token split — input (fresh) · cached (reused) · output (generated). */}
      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-2">
        <Stat label="input" value={fmtTok(inputTok)} />
        {cachedTok > 0 && <Stat label="cached" value={fmtTok(cachedTok)} />}
        <Stat label="output" value={fmtTok(outputTok)} />
        <Stat label="total tokens" value={fmtTok(totalTok)} />
      </div>
      {/* Wage comparison — the whole point: is this cheaper than a person? */}
      <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-emerald-300/30 pt-2.5 dark:border-emerald-800/30">
        <span className="text-[11px] text-zinc-500">vs. a human at</span>
        <span className="text-zinc-500">$</span>
        <input
          type="number" min={0} step={5} value={wage || ''} placeholder="0"
          onChange={(e) => onWage(Math.max(0, Number(e.target.value)))}
          className="w-16 rounded border border-zinc-300 bg-white/70 px-2 py-0.5 text-xs text-zinc-800 focus:border-emerald-500 focus:outline-none dark:border-zinc-700 dark:bg-zinc-950/60 dark:text-zinc-200"
        />
        <span className="text-[11px] text-zinc-500">/hr</span>
        {ratio != null && (
          <span className="text-xs font-medium text-emerald-700 dark:text-emerald-300">
            {ratio >= 1
              ? `→ ${ratio >= 10 ? ratio.toFixed(0) : ratio.toFixed(1)}× cheaper per hour`
              : `→ ${(1 / ratio).toFixed(1)}× dearer per hour`}
          </span>
        )}
        {Object.keys(cost.per_model || {}).length > 0 && (
          <button onClick={() => setOpen((o) => !o)} className="ml-auto text-[11px] text-zinc-500 hover:text-zinc-300">
            {open ? 'Hide models' : 'By model'}
          </button>
        )}
      </div>
      {open && (
        <div className="mt-2 space-y-1">
          {Object.entries(cost.per_model).sort((a, b) => (b[1].usd || 0) - (a[1].usd || 0)).map(([m, pm]) => (
            <div key={m} className="flex items-center gap-2 text-[11px]">
              <span className="truncate font-mono text-zinc-600 dark:text-zinc-400">{m || '?'}</span>
              {pm.calls > 1 && <span className="text-[9px] text-zinc-500">×{pm.calls}</span>}
              <span className="ml-auto tabular-nums text-zinc-500">{fmtTok((pm.prompt_tokens || 0) + (pm.completion_tokens || 0))} tok</span>
              <span className="w-16 text-right font-medium tabular-nums text-zinc-800 dark:text-zinc-200">{pm.priced ? fmtUsd(pm.usd) : '—'}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
