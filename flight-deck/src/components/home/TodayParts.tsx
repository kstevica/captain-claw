import { useId, useState, type ReactNode } from 'react'
import { AlertCircle, Check, ChevronDown, Loader2, RotateCw } from 'lucide-react'

// ── The Today home's building blocks ──
//
// One calm column: a section is a bold heading with a quiet line under it,
// then its items; buttons are pills (the main one dark on light, light on
// dark). The zinc palette flips with the theme (index.css), so zinc classes
// read right in both; colours other than zinc carry their `dark:` twin.

export function Section({ title, count, hint, children, delay = 0 }: {
  title: string
  count?: number
  hint?: string
  children: ReactNode
  delay?: number
}) {
  const id = useId()
  return (
    <section aria-labelledby={id} className="fd-rise mt-12" style={{ animationDelay: `${delay}ms` }}>
      <div className="mb-4">
        <h2 id={id} className="flex items-baseline gap-2 text-lg font-semibold tracking-tight text-zinc-100">
          {title}
          {count !== undefined && count > 0 && (
            <span className="text-sm font-medium tabular-nums text-zinc-500">{count}</span>
          )}
        </h2>
        {hint && <p className="mt-0.5 text-sm text-zinc-500">{hint}</p>}
      </div>
      {children}
    </section>
  )
}

/** What a section says with nothing in it: never a blank panel. */
export function Calm({ icon, children }: { icon?: ReactNode; children: ReactNode }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-dashed border-zinc-800 px-5 py-4 text-sm text-zinc-400">
      {icon ?? <Check className="h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" />}
      <span>{children}</span>
    </div>
  )
}

/** A section still loading its first answer. */
export function Loading({ rows = 2 }: { rows?: number }) {
  return (
    <div className="space-y-2.5" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="fd-shimmer h-16 rounded-2xl" />
      ))}
    </div>
  )
}

/** A source that didn't answer: said plainly, with another go. */
export function Trouble({ what, error, onRetry }: { what: string; error: string; onRetry: () => void }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-zinc-800 px-5 py-4 text-sm">
      <AlertCircle className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
      <span className="min-w-0 flex-1 text-zinc-400" title={error}>Couldn’t load {what} just now.</span>
      <button onClick={onRetry}
        className="flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800/70 hover:text-zinc-100">
        <RotateCw className="h-3.5 w-3.5" />Try again
      </button>
    </div>
  )
}

export type PillTone = 'primary' | 'quiet'

const PILL: Record<PillTone, string> = {
  primary: 'bg-zinc-100 text-zinc-900 hover:bg-zinc-200',
  quiet: 'text-zinc-400 hover:bg-zinc-800/70 hover:text-zinc-100',
}

export function Pill({ tone = 'quiet', busy, disabled, onClick, children, title }: {
  tone?: PillTone
  busy?: boolean
  disabled?: boolean
  onClick: () => void
  children: ReactNode
  title?: string
}) {
  return (
    <button type="button" onClick={onClick} disabled={disabled || busy} title={title}
      className={`inline-flex items-center gap-1.5 rounded-full px-4 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/50 disabled:cursor-default disabled:opacity-50 ${PILL[tone]}`}>
      {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  )
}

/** A card for one thing waiting on the person. */
export function WaitingCard({ who, when, headline, title, children, actions, error }: {
  who: string
  when: string
  /** The plain sentence: what is asked of them. */
  headline: string
  title?: string
  children?: ReactNode
  actions: ReactNode
  error?: string
}) {
  return (
    <li className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5 shadow-sm shadow-black/[0.03]">
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500 dark:bg-amber-400" />
        <span className="truncate font-medium text-zinc-400">{who}</span>
        {when && <><span aria-hidden>·</span><span className="shrink-0">{when}</span></>}
      </div>
      <p className="mt-2 text-[15px] font-medium leading-snug text-zinc-100">{headline}</p>
      {title && <p className="mt-1 text-sm leading-relaxed text-zinc-400">{title}</p>}
      {children}
      <div className="mt-4 flex flex-wrap items-center gap-1.5">{actions}</div>
      {error && <p className="mt-2 text-xs text-red-600 dark:text-red-400">{error}</p>}
    </li>
  )
}

/** The words that would go out, in sight before anything is approved. */
export function Draft({ text, to }: { text: string; to?: string }) {
  return (
    <div className="mt-3 rounded-xl bg-zinc-950/70 px-4 py-3">
      {to && <div className="mb-1.5 truncate text-xs text-zinc-500">To {to}</div>}
      <p className="max-h-48 overflow-y-auto whitespace-pre-wrap break-words text-sm leading-relaxed text-zinc-300">{text}</p>
    </div>
  )
}

/** One row in a quiet list: a mark, what it is, a line under it, and an
 *  optional "See what happened" that opens the detail in place. */
export function Row({ mark, title, sub, aside, detail }: {
  mark: ReactNode
  title: string
  sub: string
  aside?: string
  detail?: ReactNode
}) {
  const [open, setOpen] = useState(false)
  const id = useId()
  return (
    <li className="px-5 py-3.5">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center">{mark}</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-3">
            <span className="min-w-0 flex-1 truncate text-[15px] text-zinc-100" title={title}>{title}</span>
            {aside && <span className="shrink-0 text-xs text-zinc-500">{aside}</span>}
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-zinc-500">
            <span className="min-w-0 truncate">{sub}</span>
            {detail && (
              <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open} aria-controls={id}
                className="flex shrink-0 items-center gap-0.5 text-sm font-medium text-zinc-400 transition-colors hover:text-zinc-100">
                See what happened
                <ChevronDown className={`h-3.5 w-3.5 transition-transform motion-reduce:transition-none ${open ? 'rotate-180' : ''}`} />
              </button>
            )}
          </div>
          {detail && open && <div id={id} className="mt-3">{detail}</div>}
        </div>
      </div>
    </li>
  )
}

/** A quiet list's frame: one rounded card, rows split by hairlines. */
export function List({ children }: { children: ReactNode }) {
  return (
    <ul className="divide-y divide-zinc-800/70 overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/40">
      {children}
    </ul>
  )
}
