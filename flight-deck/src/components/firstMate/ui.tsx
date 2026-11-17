import { Check, Plus, X } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

// Small pieces the First Mate page and drawer share, in the Course look:
// soft section headings, pills, a switch, an empty band, a toast.

export function Section({ icon: Icon, tone = 'accent', title, count, hint, action, children }: {
  icon: LucideIcon
  tone?: 'accent' | 'amber'
  title: string
  count?: number
  hint?: string
  action?: React.ReactNode
  children: React.ReactNode
}) {
  const tile = tone === 'amber'
    ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
    : 'bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)]'
  return (
    <section>
      <div className="mb-3 flex items-center gap-3">
        <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-[0.7rem] ${tile}`}>
          <Icon className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="flex items-center gap-2 text-[15px] font-semibold tracking-tight text-zinc-100">
            {title}
            {count != null && count > 0 && <span className="fd-count fd-count-hot">{count}</span>}
          </h3>
          {hint && <p className="text-xs text-zinc-500">{hint}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}

const PILL = {
  red: 'bg-red-500/10 text-red-700 dark:text-red-300',
  amber: 'bg-amber-500/12 text-amber-800 dark:text-amber-300',
  green: 'bg-emerald-500/12 text-emerald-700 dark:text-emerald-300',
  violet: 'bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)]',
  zinc: 'bg-zinc-500/10 text-zinc-500',
} as const

export type PillTone = keyof typeof PILL

export function Pill({ tone, title, children }: { tone: PillTone; title?: string; children: React.ReactNode }) {
  return <span title={title} className={`inline-flex items-center rounded-full px-2 py-px text-[11px] font-medium ${PILL[tone]}`}>{children}</span>
}

export function Switch({ on, disabled, label, onChange }: { on: boolean; disabled?: boolean; label: string; onChange: (on: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!on)}
      className={`relative h-6 w-11 shrink-0 rounded-full transition-colors disabled:opacity-60 ${on ? 'bg-violet-600' : 'bg-zinc-700'}`}
    >
      <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${on ? 'left-[1.375rem]' : 'left-0.5'}`} />
    </button>
  )
}

export function EmptyBand({ text, onAdd }: { text: string; onAdd?: () => void }) {
  const cls = 'flex w-full items-center gap-3 rounded-[var(--fd-radius-card)] border border-dashed border-zinc-800 px-4 py-4 text-left text-sm text-zinc-500'
  if (!onAdd) return <div className={cls}>{text}</div>
  return (
    <button onClick={onAdd} className={`${cls} group transition-colors hover:border-violet-500/40 hover:bg-[var(--fd-hover)] hover:text-zinc-200`}>
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)] transition-transform group-hover:scale-105">
        <Plus className="h-4 w-4" />
      </span>
      {text}
    </button>
  )
}

export function ErrorBox({ text, onDismiss }: { text: string; onDismiss?: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-2xl border border-red-500/25 bg-red-500/[0.07] px-4 py-3 text-sm text-red-700 dark:text-red-300">
      <span className="min-w-0 flex-1">{text}</span>
      {onDismiss && (
        <button onClick={onDismiss} aria-label="Dismiss" className="rounded-full p-0.5 opacity-70 hover:opacity-100">
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  )
}

export function Toast({ text, onDismiss }: { text: string; onDismiss: () => void }) {
  return (
    <div role="status" aria-live="polite" className="pointer-events-none absolute inset-x-0 bottom-5 z-20 flex justify-center px-4">
      <div className="fd-rise pointer-events-auto flex max-w-xl items-center gap-2.5 rounded-2xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] py-2 pl-2.5 pr-2 text-sm text-zinc-200 shadow-[0_18px_40px_-16px_rgba(0,0,0,0.55)]">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
          <Check className="h-3.5 w-3.5" strokeWidth={2.5} />
        </span>
        <span className="line-clamp-2 min-w-0 flex-1">{text}</span>
        <button onClick={onDismiss} aria-label="Dismiss"
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}

export const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))
