import { Loader2 } from 'lucide-react'
import { tokensText, type RunCap } from '../../services/firstMate'

// The factory's cap per run: how many tokens one run may use before it stops,
// or no cap (-1). A setting on the factory, like its share: changing it needs
// no approval, and the next run (or Continue) uses it.

const NO_CAP = -1

function capText(n: number): string {
  return n < 0 ? 'No cap' : tokensText(n)
}

const STEPS = [1_000_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000, 50_000_000]

export function CapPicker({ cap, own, busy, onPick, id }: {
  cap: RunCap
  /** The factory's own cap (0 = none set, so `cap.inherited` applies). */
  own: number
  busy: boolean
  onPick: (tokens: number) => void
  id?: string
}) {
  const steps = own > 0 && !STEPS.includes(own) ? [...STEPS, own].sort((a, b) => a - b) : STEPS
  return (
    <span className="inline-flex items-center gap-2">
      <select id={id} value={own} disabled={busy} onChange={(e) => onPick(Number(e.target.value))}
        aria-label="Tokens one run may use"
        className="h-8 rounded-lg border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] px-2 text-[13px] text-zinc-200 focus:outline-none focus:ring-2 focus:ring-violet-500/40 disabled:opacity-60">
        <option value={0}>
          {capText(cap.inherited)} ({cap.inherited_from === 'charter' ? 'its charter' : 'your default'})
        </option>
        {steps.map((n) => <option key={n} value={n}>{tokensText(n)}</option>)}
        <option value={NO_CAP}>No cap</option>
      </select>
      {busy && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
    </span>
  )
}
