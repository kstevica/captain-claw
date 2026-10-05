import { useId } from 'react'
import { charCount } from '../../services/profile'

// A labelled textarea with a live character counter against a server cap. Over
// the cap the counter and border turn red; the caller keeps Save disabled.
export function CappedTextarea({
  label, hint, value, onChange, cap, rows = 5, placeholder, disabled,
}: {
  label: string
  hint?: string
  value: string
  onChange: (v: string) => void
  cap: number
  rows?: number
  placeholder?: string
  disabled?: boolean
}) {
  const id = useId()
  const n = charCount(value)
  const over = n > cap
  const near = !over && n > cap * 0.9
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
          {label}
        </label>
        <span
          className={`text-[11px] tabular-nums ${
            over ? 'text-red-600 dark:text-red-400' : near ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-600'
          }`}
        >
          {n.toLocaleString()} / {cap.toLocaleString()}
        </span>
      </div>
      <textarea
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={rows}
        placeholder={placeholder}
        disabled={disabled}
        className={`w-full resize-y rounded-md border bg-zinc-950 px-2.5 py-2 text-sm leading-relaxed text-zinc-200 placeholder-zinc-600 focus:outline-none disabled:opacity-50 ${
          over ? 'border-red-500/60 focus:border-red-500/70' : 'border-zinc-700 focus:border-violet-500/50'
        }`}
      />
      {over ? (
        <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">
          {(n - cap).toLocaleString()} character{n - cap === 1 ? '' : 's'} over the limit.
        </p>
      ) : hint ? (
        <p className="mt-1 text-[11px] text-zinc-500">{hint}</p>
      ) : null}
    </div>
  )
}
