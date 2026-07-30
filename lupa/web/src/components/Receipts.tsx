import { useEffect, useState } from 'react'
import { BadgeCheck, CircleAlert, Coins } from 'lucide-react'
import { api } from '../api'
import { useVocab } from '../stores'

interface Receipts {
  status: string
  verdict: string
  blocking: { rounds?: number; verdict?: string } | null
  metrics: Record<string, number | string | boolean>
  consistency: Record<string, number | boolean> | null
  gaps: { severity?: string; text?: string; [k: string]: unknown }[]
  facts: { key: string; value?: string; unit?: string; status?: string; updated_by?: string }[]
  conflicts: { key?: string; value?: string; existing?: string; by?: string }[]
  contract: { constraints?: { id?: string; text?: string; severity?: string; status?: string }[] } | null
  cost: { usd?: number | null; elapsed_seconds?: number | null; hourly_usd?: number | null
          tokens?: Record<string, number> } | null
  roi: { analyst_hourly_usd?: number; analyst_label?: string }
}

const FACT_COLORS: Record<string, string> = {
  verified: 'text-emerald-400 border-emerald-700/60',
  derived: 'text-sky-400 border-sky-700/60',
  estimated: 'text-amber-400 border-amber-700/60',
  assumed: 'text-amber-400 border-amber-700/60',
  to_be_completed: 'text-red-400 border-red-700/60',
}

function Chip({ children, tone = '' }: { children: React.ReactNode; tone?: string }) {
  return (
    <span className={`inline-block text-[11px] px-1.5 py-0.5 rounded border ${
      tone || 'text-[var(--lp-text-dim)] border-[var(--lp-border)]'}`}>
      {children}
    </span>
  )
}

function Tile({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-[var(--lp-text-dim)]">{label}</div>
      <div className={`text-sm font-semibold mt-0.5 ${warn ? 'text-amber-400' : ''}`}>{value}</div>
    </div>
  )
}

export default function ReceiptsPanel({ sid }: { sid: string }) {
  const v = useVocab()
  const [rec, setRec] = useState<Receipts | null>(null)

  useEffect(() => {
    setRec(null)
    void api<Receipts>(`/api/commissions/${sid}/receipts`).then(setRec).catch(() => {})
  }, [sid])

  if (!rec) return null
  const m = rec.metrics ?? {}
  const num = (k: string) => Number(m[k] ?? 0)
  const pass = (rec.verdict || '').toLowerCase() === 'pass'
  const hasMetrics = Object.keys(m).length > 0
  const cost = rec.cost
  const analystRate = rec.roi?.analyst_hourly_usd
  const multiple = cost?.hourly_usd && analystRate
    ? Math.round(analystRate / cost.hourly_usd) : null

  return (
    <div className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] p-5 space-y-4">
      <div className="flex items-center gap-2">
        {pass
          ? <BadgeCheck size={17} className="text-emerald-400" />
          : <CircleAlert size={17} className="text-amber-400" />}
        <span className="font-semibold">{v('receipts_title', 'Receipts')}</span>
        {rec.verdict && (
          <Chip tone={pass ? 'text-emerald-400 border-emerald-700/60'
                          : 'text-amber-400 border-amber-700/60'}>
            {rec.verdict}
          </Chip>
        )}
        <span className="text-xs text-[var(--lp-text-dim)]">{v('receipts_hint', '')}</span>
      </div>

      {hasMetrics && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <Tile label="Claims"
                value={`${num('claims_confirmed')}/${num('claims_checked')} confirmed`}
                warn={num('claims_refuted') > 0} />
          <Tile label="Consistency"
                value={`${num('consistency_critical')} critical · ${num('consistency_major')} major`}
                warn={num('consistency_critical') > 0} />
          <Tile label="Contract"
                value={`${num('contract_checked')} checked · ${num('contract_failed_critical') + num('contract_failed_major')} failed`}
                warn={num('contract_failed_critical') > 0} />
          <Tile label="Gaps"
                value={`${num('gaps_major')} major · ${num('gaps_minor')} minor`}
                warn={num('gaps_major') > 0} />
        </div>
      )}

      {cost && (cost.usd != null || cost.elapsed_seconds != null) && (
        <div className="flex items-center gap-2 flex-wrap text-sm rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2.5">
          <Coins size={15} style={{ color: 'var(--lp-accent)' }} />
          <span className="font-semibold">{v('cost_title', 'Cost')}:</span>
          {cost.usd != null && <span>${cost.usd.toFixed(2)}</span>}
          {cost.elapsed_seconds != null && (
            <span className="text-[var(--lp-text-dim)]">
              · {Math.round(cost.elapsed_seconds / 60)} min
            </span>
          )}
          {cost.hourly_usd != null && (
            <span className="text-[var(--lp-text-dim)]">
              · effective ${cost.hourly_usd.toFixed(2)}/hr
            </span>
          )}
          {multiple != null && multiple > 1 && (
            <Chip tone="text-emerald-400 border-emerald-700/60">
              ≈{multiple}× below {rec.roi.analyst_label ?? 'an analyst'} at ${analystRate}/hr
            </Chip>
          )}
        </div>
      )}

      {rec.facts.length > 0 && (
        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-[var(--lp-text-dim)] mb-1.5">
            {v('facts_title', 'Facts ledger')}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <tbody>
                {rec.facts.map((f) => (
                  <tr key={f.key} className="border-t border-[var(--lp-border)]">
                    <td className="py-1.5 pr-3 font-mono text-xs text-[var(--lp-text-dim)]">{f.key}</td>
                    <td className="py-1.5 pr-3 whitespace-nowrap">{f.value} {f.unit}</td>
                    <td className="py-1.5 pr-3">
                      <Chip tone={FACT_COLORS[f.status ?? ''] ?? ''}>{f.status}</Chip>
                    </td>
                    <td className="py-1.5 text-xs text-[var(--lp-text-dim)]">{f.updated_by}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {rec.conflicts.length > 0 && (
            <div className="mt-2 space-y-1">
              {rec.conflicts.map((c, i) => (
                <div key={i} className="text-xs text-amber-400">
                  conflict on <span className="font-mono">{c.key}</span>: “{c.value}”
                  (by {c.by ?? '?'}) vs recorded “{c.existing}”
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {rec.contract?.constraints && rec.contract.constraints.length > 0 && (
        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-[var(--lp-text-dim)] mb-1.5">
            Contract
          </div>
          <ul className="space-y-1">
            {rec.contract.constraints.map((c, i) => (
              <li key={c.id ?? i} className="text-sm flex items-center gap-2">
                <Chip tone={c.severity === 'critical'
                  ? 'text-red-400 border-red-700/60' : 'text-amber-400 border-amber-700/60'}>
                  {c.severity}
                </Chip>
                <span>{c.text}</span>
                {c.status && (
                  <Chip tone={c.status === 'pass'
                    ? 'text-emerald-400 border-emerald-700/60'
                    : c.status === 'unclear'
                      ? 'text-amber-400 border-amber-700/60'
                      : 'text-red-400 border-red-700/60'}>
                    {c.status}
                  </Chip>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {rec.gaps.length > 0 && (
        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-[var(--lp-text-dim)] mb-1.5">
            Open gaps
          </div>
          <ul className="space-y-1">
            {rec.gaps.map((g, i) => (
              <li key={i} className="text-sm flex items-center gap-2">
                <Chip tone={g.severity === 'major'
                  ? 'text-amber-400 border-amber-700/60' : ''}>
                  {g.severity ?? 'minor'}
                </Chip>
                <span>{String(g.text ?? g.description ?? '')}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
