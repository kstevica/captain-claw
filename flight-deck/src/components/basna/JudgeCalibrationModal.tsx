import { useEffect, useState } from 'react'
import { Download, Loader2, Scale, X } from 'lucide-react'
import {
  apiExportLabelPairs, apiLabelAgreement,
  type LabelAgreement, type LabelAgreementGroup, type LabelAgreementSummary,
} from '../../stores/basnaStore'
import { Button } from '../console/parts'
import { checkCls, iconBtnCls, segBtnCls, segFrameCls, smallInputCls } from './lookCls'

// ── Judge calibration ───────────────────────────────────────────────────────
// How far the automatic judge's success/fail verdict on each agent run agrees
// with the user's own thumbs votes (Cohen's kappa), plus the eval-set export of
// those judge/human pairs.

// Landis & Koch bands — the usual reading of a kappa value.
function kappaBand(k: number): string {
  if (k < 0) return 'worse than chance'
  if (k <= 0.2) return 'slight'
  if (k <= 0.4) return 'fair'
  if (k <= 0.6) return 'moderate'
  if (k <= 0.8) return 'substantial'
  return 'almost perfect'
}

const pct = (v: number | null) => (v === null ? '—' : `${Math.round(v * 100)}%`)
const kap = (v: number | null) => (v === null ? '—' : v.toFixed(2))

type GroupKey = 'by_mode' | 'by_domain' | 'by_archetype'
const GROUPS: { id: GroupKey; label: string; field: 'mode' | 'domain' | 'archetype_id' }[] = [
  { id: 'by_mode', label: 'Mode', field: 'mode' },
  { id: 'by_domain', label: 'Domain', field: 'domain' },
  { id: 'by_archetype', label: 'Archetype', field: 'archetype_id' },
]

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
      <div className="text-xs font-medium text-zinc-500">{label}</div>
      <div className="mt-1.5 text-2xl font-semibold tabular-nums tracking-tight text-zinc-100">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-zinc-500">{hint}</div>}
    </div>
  )
}

function Confusion({ a }: { a: LabelAgreement }) {
  const c = a.confusion
  const cell = (n: number, agree: boolean) => (
    <td className={`px-3 py-1.5 text-center font-mono ${agree ? 'text-emerald-700 dark:text-emerald-300' : 'text-rose-700 dark:text-rose-300'}`}>{n}</td>
  )
  return (
    <table className="text-xs">
      <thead>
        <tr className="text-zinc-500">
          <th />
          <th className="px-3 py-1 font-medium">You: success</th>
          <th className="px-3 py-1 font-medium">You: fail</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th className="px-2 py-1.5 text-left font-medium text-zinc-500">Judge: success</th>
          {cell(c.both_success, true)}
          {cell(c.judge_success_human_fail, false)}
        </tr>
        <tr>
          <th className="px-2 py-1.5 text-left font-medium text-zinc-500">Judge: fail</th>
          {cell(c.judge_fail_human_success, false)}
          {cell(c.both_fail, true)}
        </tr>
      </tbody>
    </table>
  )
}

/** `mode`: the page is locked to Basna or Vatra — no breakdown by mode here. */
export function JudgeCalibrationModal({ onClose, mode }: { onClose: () => void; mode?: 'basna' | 'vatra' }) {
  const [since, setSince] = useState('')
  const [summary, setSummary] = useState<LabelAgreementSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [group, setGroup] = useState<GroupKey>('by_domain')
  const [includeUnpaired, setIncludeUnpaired] = useState(false)
  const [includeText, setIncludeText] = useState(false)
  const [exporting, setExporting] = useState<'jsonl' | 'csv' | null>(null)

  useEffect(() => {
    let live = true
    setLoading(true)
    setError(null)
    apiLabelAgreement(since)
      .then((s) => { if (live) setSummary(s) })
      .catch((e) => { if (live) setError(String(e.message || e)) })
      .finally(() => { if (live) setLoading(false) })
    return () => { live = false }
  }, [since])

  const onExport = async (format: 'jsonl' | 'csv') => {
    setExporting(format)
    setError(null)
    try {
      await apiExportLabelPairs({ format, since, includeUnpaired, includeText })
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setExporting(null)
    }
  }

  const o = summary?.overall
  const groups = mode ? GROUPS.filter((x) => x.id !== 'by_mode') : GROUPS
  const g = GROUPS.find((x) => x.id === group)!
  const rows: LabelAgreementGroup[] = summary ? summary[group] : []

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Judge calibration"
        className="flex max-h-[92dvh] w-full flex-col rounded-t-2xl border border-zinc-800 bg-zinc-950 shadow-2xl wide:max-h-[90vh] wide:max-w-2xl wide:rounded-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-5 py-4">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
            <Scale className="h-[18px] w-[18px]" />
          </span>
          <h2 className="min-w-0 flex-1 truncate text-base font-semibold text-zinc-100">Judge calibration</h2>
          <button onClick={onClose} aria-label="Close" title="Close" className={iconBtnCls}>
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-5 overflow-auto px-5 py-5 pb-[max(1.25rem,env(safe-area-inset-bottom))]">
          <p className="text-sm leading-relaxed text-zinc-400">
            How often the automatic judge's success/fail verdict on an agent's contribution matches your
            thumbs vote. Vote on runs in a run's <span className="text-zinc-300">Agents</span> tab — agreeing
            votes count too.
          </p>

          <label className="flex flex-wrap items-center gap-2 text-sm text-zinc-400">
            Runs since
            <input
              type="date"
              value={since}
              onChange={(e) => setSince(e.target.value)}
              className={smallInputCls}
            />
            {since && (
              <button onClick={() => setSince('')} className="rounded-lg px-2 py-1 text-xs font-medium text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">clear</button>
            )}
            <span className="basis-full text-xs leading-relaxed text-zinc-500">
              Set this to the day your deck started keeping judge labels separately: a run voted on before that
              and voted on again since may carry your old vote in place of the judge's.
            </span>
          </label>

          {loading && !summary && (
            <div className="flex items-center gap-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…</div>
          )}

          {o && (
            <>
              <div className="grid grid-cols-1 gap-2.5 wide:grid-cols-3">
                <Stat label="Pairs" value={String(o.n)}
                  hint={summary!.unpaired ? `+${summary!.unpaired} vote(s) the judge left unscored` : undefined} />
                <Stat label="Agreement" value={pct(o.agreement)}
                  hint={o.expected_agreement !== null ? `chance: ${pct(o.expected_agreement)}` : undefined} />
                <Stat label="Cohen's κ" value={kap(o.kappa)}
                  hint={o.kappa !== null ? kappaBand(o.kappa) : o.n ? 'undefined — every label is the same' : undefined} />
              </div>
              {o.n === 0 && (
                <p className="text-xs text-zinc-500">No judge/human pairs yet — vote on some agent runs first.</p>
              )}
              {o.n > 0 && o.n < 30 && (
                <p className="text-[11px] text-amber-700 dark:text-amber-300">
                  Fewer than 30 pairs — treat κ as a rough signal.
                </p>
              )}
              {o.n > 0 && (
                <div className="flex flex-wrap items-start gap-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
                  <Confusion a={o} />
                  <div className="min-w-0 flex-1">
                    <div className={`mb-2.5 ${segFrameCls}`}>
                      {groups.map((x) => (
                        <button
                          key={x.id}
                          onClick={() => setGroup(x.id)}
                          aria-pressed={group === x.id}
                          className={segBtnCls(group === x.id)}
                        >
                          {x.label}
                        </button>
                      ))}
                    </div>
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-left text-zinc-500">
                          <th className="py-1 pr-2 font-medium">{g.label}</th>
                          <th className="py-1 pr-2 text-right font-medium">Pairs</th>
                          <th className="py-1 pr-2 text-right font-medium">Agree</th>
                          <th className="py-1 text-right font-medium">κ</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((r) => (
                          <tr key={r[g.field]} className="border-t border-zinc-800 text-zinc-300">
                            <td className="max-w-[12rem] truncate py-1 pr-2" title={r[g.field]}>{r[g.field] || '—'}</td>
                            <td className="py-1 pr-2 text-right font-mono">{r.n}</td>
                            <td className="py-1 pr-2 text-right font-mono">{pct(r.agreement)}</td>
                            <td className="py-1 text-right font-mono">{kap(r.kappa)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}

          <div className="space-y-3 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
            <div className="text-[13px] font-semibold text-zinc-200">Export eval set</div>
            <label className="flex items-start gap-2.5 text-sm text-zinc-400">
              <input type="checkbox" checked={includeUnpaired} onChange={(e) => setIncludeUnpaired(e.target.checked)} className={`${checkCls} mt-0.5`} />
              Include votes the judge left unscored
            </label>
            <label className="flex items-start gap-2.5 text-sm text-zinc-400">
              <input type="checkbox" checked={includeText} onChange={(e) => setIncludeText(e.target.checked)} className={`${checkCls} mt-0.5`} />
              Include text — task, compiled answer and each agent's output (larger; may hold sensitive content)
            </label>
            <div className="flex gap-2 pt-1">
              {(['jsonl', 'csv'] as const).map((f) => (
                <Button
                  key={f}
                  onClick={() => onExport(f)}
                  disabled={exporting !== null}
                >
                  {exporting === f ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                  Export {f.toUpperCase()}
                </Button>
              ))}
            </div>
          </div>

          {error && <p className="rounded-xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-sm text-red-700 dark:text-red-300">{error}</p>}
        </div>
      </div>
    </div>
  )
}
