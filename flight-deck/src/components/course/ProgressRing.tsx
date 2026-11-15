import { Check, Pause, X } from 'lucide-react'
import type { GoalStatus } from '../../services/course'

// How far along a goal is, as a small ring: violet while it moves, green once
// it's there, grey and paused when it rests. With no measure and no estimate
// yet it's a dotted track. A finished goal shows how it ended instead — a
// check when done, a cross when dropped.
export function ProgressRing({ value, status = 'active', size = 28, stroke = 3, label }: {
  value: number | null
  status?: GoalStatus
  size?: number
  stroke?: number
  /** Read out instead of the percentage. */
  label?: string
}) {
  const box = { width: size, height: size }
  if (status === 'achieved') {
    return (
      <span style={box} className="flex shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400" aria-label="Done">
        <Check className="h-[52%] w-[52%]" strokeWidth={2.5} />
      </span>
    )
  }
  if (status === 'dropped' || status === 'declined' || status === 'superseded') {
    return (
      <span style={box} className="flex shrink-0 items-center justify-center rounded-full bg-zinc-500/10 text-zinc-500" aria-label="Dropped">
        <X className="h-[48%] w-[48%]" strokeWidth={2.25} />
      </span>
    )
  }
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const v = value == null ? 0 : Math.max(0, Math.min(1, value))
  const pct = Math.round(v * 100)
  const tone = status === 'paused' ? 'stroke-zinc-500' : v >= 1 ? 'stroke-emerald-500' : 'stroke-violet-500'
  return (
    <span
      style={box}
      className="relative flex shrink-0 items-center justify-center"
      role="img"
      aria-label={label ?? (value == null ? 'No progress estimate yet' : `${pct}% along`)}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90" aria-hidden>
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke}
          className="stroke-zinc-800"
          strokeDasharray={value == null ? '1.5 3' : undefined}
          strokeLinecap={value == null ? 'round' : undefined}
        />
        {v > 0 && (
          <circle
            cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke} strokeLinecap="round"
            strokeDasharray={c} strokeDashoffset={c * (1 - v)}
            className={`${tone} transition-[stroke-dashoffset] duration-500`}
          />
        )}
      </svg>
      {status === 'paused' && <Pause className="absolute h-[38%] w-[38%] text-zinc-500" strokeWidth={2.5} aria-hidden />}
    </span>
  )
}
