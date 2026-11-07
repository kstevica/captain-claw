import { useEffect, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Loader2, X, type LucideIcon } from 'lucide-react'

// The Admin console's building blocks: one look for every page — a roomy
// header, cards, stat tiles, pills, a side sheet for editing, and the form
// controls. Dark-first zinc classes (the light theme remaps zinc); coloured
// text carries its own light/dark pair.

export const inputCls =
  'w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/15 disabled:opacity-60'

export const labelCls = 'mb-1.5 block text-xs font-medium text-zinc-400'

// ── Page ──

/** A console page: centred and at most 5xl wide, or ``full`` the width of
 *  whatever holds it (a kiosk dialog). */
export function Page({ children, full = false }: { children: ReactNode; full?: boolean }) {
  return <div className={`w-full space-y-6 px-4 py-6 wide:px-8 wide:py-8 ${full ? '' : 'mx-auto max-w-5xl'}`}>{children}</div>
}

export function PageHeader({ icon: Icon, title, description, actions }: {
  icon: LucideIcon
  title: string
  description: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
      <div className="flex min-w-0 items-start gap-4">
        <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
          <Icon className="h-6 w-6" />
        </span>
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">{title}</h1>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-zinc-500">{description}</p>
        </div>
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

// ── Cards ──

export function Card({ title, hint, icon: Icon, action, children, className = '', bodyClassName = 'p-5' }: {
  title?: ReactNode
  hint?: ReactNode
  icon?: LucideIcon
  action?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}) {
  return (
    <section className={`rounded-2xl border border-zinc-800 bg-zinc-900/40 ${className}`}>
      {(title || action) && (
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-zinc-800/80 px-5 py-4">
          <div className="flex min-w-0 items-start gap-3">
            {Icon && (
              <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800/80 text-zinc-300">
                <Icon className="h-4 w-4" />
              </span>
            )}
            <div className="min-w-0">
              {title && <h2 className="text-[15px] font-semibold text-zinc-100">{title}</h2>}
              {hint && <p className="mt-0.5 text-xs leading-relaxed text-zinc-500">{hint}</p>}
            </div>
          </div>
          {action && <div className="flex shrink-0 flex-wrap items-center gap-2">{action}</div>}
        </div>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  )
}

/** A setting: what it is on the left, its control on the right. */
export function SettingRow({ title, hint, children }: { title: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-3 py-4 first:pt-0 last:pb-0 wide:flex-row wide:items-center wide:justify-between wide:gap-6">
      <div className="min-w-0">
        <div className="text-sm font-medium text-zinc-200">{title}</div>
        {hint && <div className="mt-0.5 text-xs leading-relaxed text-zinc-500">{hint}</div>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  )
}

export function StatTile({ icon: Icon, label, value, sub, tone = 'zinc', onClick, loading }: {
  icon: LucideIcon
  label: string
  value: ReactNode
  sub?: ReactNode
  tone?: Tone
  onClick?: () => void
  loading?: boolean
}) {
  const body = (
    <>
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-zinc-500">{label}</span>
        <span className={`flex h-7 w-7 items-center justify-center rounded-lg ${TONE_SOFT[tone]}`}>
          <Icon className="h-3.5 w-3.5" />
        </span>
      </div>
      <div className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-zinc-100">
        {loading ? <Loader2 className="h-5 w-5 animate-spin text-zinc-600" /> : value}
      </div>
      {sub && <div className="mt-0.5 truncate text-xs text-zinc-500">{sub}</div>}
    </>
  )
  const cls = 'rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 text-left'
  return onClick ? (
    <button onClick={onClick} className={`${cls} transition-colors hover:border-violet-500/40 hover:bg-zinc-900/70`}>{body}</button>
  ) : (
    <div className={cls}>{body}</div>
  )
}

// ── Pills ──

export type Tone = 'zinc' | 'violet' | 'emerald' | 'amber' | 'red' | 'sky' | 'cyan'

const TONE_SOFT: Record<Tone, string> = {
  zinc: 'bg-zinc-800 text-zinc-400',
  violet: 'bg-violet-500/10 text-violet-700 dark:text-violet-300',
  emerald: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
  amber: 'bg-amber-500/10 text-amber-700 dark:text-amber-300',
  red: 'bg-red-500/10 text-red-700 dark:text-red-300',
  sky: 'bg-sky-500/10 text-sky-700 dark:text-sky-300',
  cyan: 'bg-cyan-500/10 text-cyan-700 dark:text-cyan-300',
}

export function Pill({ tone = 'zinc', icon: Icon, children, title }: {
  tone?: Tone
  icon?: LucideIcon
  children: ReactNode
  title?: string
}) {
  return (
    <span title={title} className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${TONE_SOFT[tone]}`}>
      {Icon && <Icon className="h-3 w-3" />}
      {children}
    </span>
  )
}

export function Dot({ tone }: { tone: 'on' | 'pending' | 'off' }) {
  return (
    <span className={`h-2 w-2 shrink-0 rounded-full ${
      tone === 'on' ? 'bg-emerald-500' : tone === 'pending' ? 'bg-amber-500' : 'bg-zinc-600'
    }`} />
  )
}

// ── Buttons ──

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

const VARIANT: Record<Variant, string> = {
  primary: 'bg-violet-600 text-white hover:bg-violet-500 shadow-sm shadow-violet-900/20',
  secondary: 'border border-zinc-700 bg-zinc-900/60 text-zinc-200 hover:border-zinc-600 hover:bg-zinc-800',
  ghost: 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200',
  danger: 'border border-zinc-700 text-zinc-400 hover:border-red-500/40 hover:bg-red-500/10 hover:text-red-600 dark:hover:text-red-400',
}

export function Button({ variant = 'secondary', icon: Icon, busy, children, className = '', ...rest }: {
  variant?: Variant
  icon?: LucideIcon
  busy?: boolean
  children?: ReactNode
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      {...rest}
      disabled={rest.disabled || busy}
      className={`inline-flex items-center justify-center gap-1.5 rounded-xl px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${VARIANT[variant]} ${className}`}
    >
      {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : Icon && <Icon className="h-4 w-4" />}
      {children}
    </button>
  )
}

// ── Choices ──

/** A row of mutually exclusive choices (a radio group). */
export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T
  options: { value: T; label: ReactNode; count?: number }[]
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex max-w-full flex-wrap gap-1 rounded-xl border border-zinc-800 bg-zinc-950/60 p-1">
      {options.map((o) => {
        const on = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(o.value)}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              on ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-500 hover:text-zinc-200'
            }`}
          >
            {o.label}
            {o.count !== undefined && (
              <span className={`rounded-full px-1.5 text-[11px] tabular-nums ${on ? 'bg-zinc-700 text-zinc-200' : 'bg-zinc-800/80 text-zinc-500'}`}>
                {o.count}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}

/** Chips to pick one value from a list (context windows, tiers). */
export function ChipPicker<T extends string | number>({ value, options, onChange, format }: {
  value: T
  options: T[]
  onChange: (v: T) => void
  format?: (v: T) => string
}) {
  // A saved value outside the list stays pickable, so opening the editor
  // never silently rewrites a working config.
  const all = options.includes(value) ? options : [value, ...options]
  return (
    <div className="flex flex-wrap gap-1.5">
      {all.map((o) => (
        <button
          key={String(o)}
          type="button"
          onClick={() => onChange(o)}
          className={`rounded-lg border px-2.5 py-1 text-xs font-medium tabular-nums transition-colors ${
            o === value
              ? 'border-violet-500/50 bg-violet-500/15 text-violet-700 dark:text-violet-200'
              : 'border-zinc-800 bg-zinc-950 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200'
          }`}
        >
          {format ? format(o) : String(o)}
        </button>
      ))}
    </div>
  )
}

export function Switch({ on, busy, label, onChange }: { on: boolean; busy?: boolean; label?: string; onChange: (on: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      disabled={busy}
      onClick={() => onChange(!on)}
      className={`relative h-5 w-9 shrink-0 rounded-full transition-colors disabled:opacity-60 ${on ? 'bg-violet-600' : 'bg-zinc-700'}`}
    >
      <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${on ? 'left-[18px]' : 'left-0.5'}`} />
    </button>
  )
}

// ── Messages ──

export function Notice({ tone = 'zinc', children }: { tone?: 'zinc' | 'emerald' | 'amber' | 'red'; children: ReactNode }) {
  const cls = {
    zinc: 'border-zinc-800 bg-zinc-900/60 text-zinc-400',
    emerald: 'border-emerald-500/30 bg-emerald-500/5 text-emerald-700 dark:text-emerald-300',
    amber: 'border-amber-500/30 bg-amber-500/5 text-amber-700 dark:text-amber-300',
    red: 'border-red-500/30 bg-red-500/5 text-red-700 dark:text-red-300',
  }[tone]
  return <div className={`rounded-xl border px-4 py-3 text-sm leading-relaxed ${cls}`}>{children}</div>
}

export function EmptyState({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center rounded-2xl border border-dashed border-zinc-800 px-6 py-12 text-center">
      <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-zinc-900 text-zinc-500">
        <Icon className="h-5 w-5" />
      </span>
      <div className="mt-3 text-sm font-medium text-zinc-300">{title}</div>
      {children && <div className="mt-1 max-w-sm text-xs leading-relaxed text-zinc-500">{children}</div>}
    </div>
  )
}

// ── Side sheet ──

/** An editor that slides in from the right (from the bottom on a phone). */
export function Sheet({ title, subtitle, icon: Icon, onClose, footer, children, wide = false }: {
  title: ReactNode
  subtitle?: ReactNode
  icon?: LucideIcon
  onClose: () => void
  footer?: ReactNode
  children: ReactNode
  wide?: boolean
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-end justify-end bg-black/50 wide:items-stretch" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        className={`flex max-h-[92dvh] w-full flex-col rounded-t-2xl border border-zinc-800 bg-zinc-950 shadow-2xl wide:max-h-none wide:rounded-none wide:border-y-0 wide:border-r-0 ${
          wide ? 'wide:max-w-2xl' : 'wide:max-w-lg'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3 border-b border-zinc-800 px-5 py-4">
          {Icon && (
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
              <Icon className="h-[18px] w-[18px]" />
            </span>
          )}
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-base font-semibold text-zinc-100">{title}</h2>
            {subtitle && <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p>}
          </div>
          <button onClick={onClose} aria-label="Close" title="Close" className="rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-5 py-5">{children}</div>
        {footer && (
          <div className="flex flex-wrap items-center justify-end gap-2 border-t border-zinc-800 px-5 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}

/** A centred dialog for short choices; ``wide`` for one read at chat size
 *  (a question for the person); ``large`` for writing (most of the screen, and
 *  its body can fill the height). ``keepOpen`` ignores Escape and a click
 *  outside, so typed work isn't lost by accident. */
export function Dialog({ title, icon: Icon, onClose, footer, children, large = false, wide = false, keepOpen = false }: {
  title: ReactNode
  icon?: LucideIcon
  onClose: () => void
  footer?: ReactNode
  children: ReactNode
  large?: boolean
  wide?: boolean
  keepOpen?: boolean
}) {
  useEffect(() => {
    if (keepOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, keepOpen])

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 wide:items-center wide:p-4" onClick={keepOpen ? undefined : onClose}>
      <div
        role="dialog"
        aria-modal="true"
        className={`flex w-full flex-col rounded-t-2xl border border-zinc-800 bg-zinc-950 shadow-2xl wide:rounded-2xl ${
          large ? 'h-[94dvh] wide:h-[90dvh] wide:max-w-6xl' : wide ? 'max-h-[92dvh] max-w-2xl' : 'max-h-[92dvh] max-w-lg'}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-zinc-800 px-5 py-4">
          {Icon && (
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
              <Icon className="h-[18px] w-[18px]" />
            </span>
          )}
          <h2 className="min-w-0 flex-1 truncate text-base font-semibold text-zinc-100">{title}</h2>
          <button onClick={onClose} aria-label="Close" title="Close" className="rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className={`min-h-0 flex-1 overflow-y-auto px-5 py-5 ${large ? 'flex flex-col' : ''}`}>{children}</div>
        {footer && (
          <div className="flex flex-wrap items-center justify-end gap-2 border-t border-zinc-800 px-5 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}

export function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <div>
      <label className={labelCls}>{label}</label>
      {children}
      {hint && <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-500">{hint}</p>}
    </div>
  )
}
