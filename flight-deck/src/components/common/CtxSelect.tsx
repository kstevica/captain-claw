// Context-size dropdown — the one way to pick input/output token windows
// across every panel (Library tiers, Spawner, Basna route plans).
// A saved value outside the menu (legacy 200k/400k tiers, hand-typed
// numbers) is kept as a "(current)" entry so opening the form never
// silently rewrites a working config.
import { fmtCtxTokens } from '../../services/tierConfig'

const DEFAULT_CLS =
  'w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm tabular-nums text-zinc-200 focus:border-violet-500/50 focus:outline-none'

export function CtxSelect({
  options,
  value,
  onChange,
  zeroLabel,
  className,
}: {
  options: number[]
  value: number
  onChange: (n: number) => void
  // When set, 0 is a legal choice with this label (e.g. "Tier default").
  zeroLabel?: string
  className?: string
}) {
  const zeroIsOption = zeroLabel !== undefined
  const needsCurrent = !options.includes(value) && !(value === 0 && zeroIsOption)
  return (
    <select
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className={className || DEFAULT_CLS}
    >
      {zeroIsOption && <option value={0}>{zeroLabel}</option>}
      {needsCurrent && (
        <option value={value}>
          {value === 0 ? 'unset (0)' : `${fmtCtxTokens(value)} (current)`}
        </option>
      )}
      {options.map((n) => (
        <option key={n} value={n}>{fmtCtxTokens(n)}</option>
      ))}
    </select>
  )
}
