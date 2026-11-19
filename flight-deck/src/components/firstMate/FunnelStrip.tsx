import { Filter } from 'lucide-react'
import type { FunnelCounts, FunnelGroup } from '../../services/firstMate'

// A list funnel run's strip in its panel: per group, how many items it looked at and
// where each went (ruled out by code, after reading, by the judging call, by a check;
// maybe; kept; added by the finishing step), and how many the group has against what
// it needs.

const PARTS: { key: keyof FunnelGroup; label: string; cls: string }[] = [
  { key: 'kept', label: 'kept', cls: 'bg-emerald-500' },
  { key: 'agent', label: 'added by the finishing step', cls: 'bg-teal-400' },
  { key: 'maybe', label: 'came close', cls: 'bg-amber-400' },
  { key: 'check', label: 'failed a check', cls: 'bg-rose-400' },
  { key: 'agent-check', label: 'finishing step’s, failed a check', cls: 'bg-rose-300' },
  { key: 'judge', label: 'judged out', cls: 'bg-zinc-400' },
  { key: 'read', label: 'ruled out after reading', cls: 'bg-zinc-500' },
  { key: 'filter', label: 'ruled out by code', cls: 'bg-zinc-600' },
  { key: 'full', label: 'group already full', cls: 'bg-zinc-700' },
  { key: 'duplicate', label: 'duplicates', cls: 'bg-zinc-700' },
]

function Row({ name, g }: { name: string; g: FunnelGroup }) {
  const total = Math.max(1, PARTS.reduce((n, p) => n + (Number(g[p.key]) || 0), 0))
  const have = g.on_list
  const full = g.quota > 0 && have >= g.quota
  return (
    <li className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_auto] items-center gap-3 @2xl:grid-cols-[minmax(0,13rem)_minmax(0,1fr)_auto]">
      <span className="truncate text-xs text-zinc-300" title={name}>{name}</span>
      <span className="flex h-2 overflow-hidden rounded-full bg-zinc-500/15" role="img"
        aria-label={`${name}: ${g.looked} looked at, ${have} on the list`}>
        {PARTS.map((p) => {
          const n = Number(g[p.key]) || 0
          return n ? <span key={p.key} title={`${n} ${p.label}`} className={`h-full ${p.cls}`} style={{ width: `${(n / total) * 100}%` }} /> : null
        })}
      </span>
      <span className="whitespace-nowrap text-right text-[11px] tabular-nums text-zinc-500">
        <span className={full ? 'font-semibold text-emerald-600 dark:text-emerald-400' : 'font-semibold text-zinc-200'}>{have}</span>
        {g.quota > 0 && <> of {g.quota}</>} on the list · {g.looked} looked at
      </span>
    </li>
  )
}

export function FunnelStrip({ counts }: { counts: FunnelCounts }) {
  const groups = Object.entries(counts.groups).filter(([, g]) => g.looked > 0 || g.quota > 0)
  if (!groups.length) return null
  const used = PARTS.filter((p) => groups.some(([, g]) => (Number(g[p.key]) || 0) > 0))
  const totals = Object.fromEntries(used.map((p) => [p.key, groups.reduce((n, [, g]) => n + (Number(g[p.key]) || 0), 0)]))
  return (
    <div className="rounded-xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] px-3 py-2.5">
      <p className="mb-2 flex items-center gap-1.5 text-[11px] font-medium text-zinc-400">
        <Filter className="h-3 w-3 text-violet-600 dark:text-violet-300" /> List funnel
        <span className="font-normal text-zinc-600">· {counts.searches} {counts.searches === 1 ? 'search' : 'searches'}</span>
      </p>
      <ul className="space-y-1.5">
        {groups.map(([name, g]) => <Row key={name || '-'} name={name || 'The list'} g={g} />)}
      </ul>
      {used.length > 0 && (
        <p className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-zinc-500">
          {used.map((p) => (
            <span key={p.key} className="inline-flex items-center gap-1">
              <span className={`h-1.5 w-1.5 rounded-full ${p.cls}`} />{totals[p.key]} {p.label}
            </span>
          ))}
        </p>
      )}
    </div>
  )
}
