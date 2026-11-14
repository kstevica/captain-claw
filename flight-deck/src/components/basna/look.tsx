import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import { SQUARE_TONE, type LookTone } from './lookCls'

// ── The Basna / Vatra page family's own look helpers ────────────────────────
// Built on the console parts (cards, buttons, Segmented): the tinted icon
// squares of the kiosk menu, a calm section label, and a few compact control
// classes for the dense run views. Dark-first zinc; coloured text carries its
// light/dark pair.

const SQUARE_SIZE = {
  sm: { box: 'h-8 w-8 rounded-lg', icon: 'h-4 w-4' },
  md: { box: 'h-10 w-10 rounded-xl', icon: 'h-5 w-5' },
  lg: { box: 'h-12 w-12 rounded-2xl', icon: 'h-6 w-6' },
} as const

export function IconSquare({ icon: Icon, tone = 'violet', size = 'md' }: {
  icon: LucideIcon; tone?: LookTone; size?: keyof typeof SQUARE_SIZE
}) {
  const s = SQUARE_SIZE[size]
  return (
    <span className={`flex shrink-0 items-center justify-center ring-1 ${s.box} ${SQUARE_TONE[tone]}`}>
      <Icon className={s.icon} />
    </span>
  )
}

/** A small heading over a group of controls. */
export function SectionLabel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`text-[11px] font-semibold uppercase tracking-wider text-zinc-500 ${className}`}>{children}</div>
}

/** An inner panel's title: icon, name, an optional count. */
export function PanelTitle({ icon: Icon, children, count, iconCls = 'text-zinc-500' }: {
  icon?: LucideIcon; children: ReactNode; count?: number; iconCls?: string
}) {
  return (
    <span className="flex min-w-0 items-center gap-2">
      {Icon && <Icon className={`h-4 w-4 shrink-0 ${iconCls}`} />}
      <span className="truncate text-[13px] font-semibold text-zinc-200">{children}</span>
      {typeof count === 'number' && count > 0 && (
        <span className="rounded-full bg-zinc-800 px-1.5 text-[11px] font-medium tabular-nums text-zinc-400">{count}</span>
      )}
    </span>
  )
}

/** The page's header: the mode's tinted square, its name and a line on it,
 *  the actions on the right of the name (they stay on its row on a phone). */
export function TeamHeader({ icon, tone, title, description, actions, compact = false }: {
  icon: LucideIcon; tone: LookTone; title: string; description: ReactNode; actions?: ReactNode
  /** On a phone, no description: the title row only (inside a project). */
  compact?: boolean
}) {
  return (
    <div className={`flex gap-3.5 wide:items-start wide:gap-4 ${compact ? 'items-center' : 'items-start'}`}>
      <IconSquare icon={icon} tone={tone} size="lg" />
      <div className="min-w-0 flex-1">
        <div className="flex items-start justify-between gap-3">
          <h1 className={`min-w-0 truncate text-xl font-semibold tracking-tight text-zinc-100 wide:pt-0.5 wide:text-2xl ${compact ? 'pt-0.5' : 'pt-1.5'}`}>{title}</h1>
          {actions && <div className="-mr-1 flex shrink-0 flex-wrap items-center justify-end gap-1.5 wide:gap-2">{actions}</div>}
        </div>
        <p className={`mt-1 max-w-2xl text-sm leading-relaxed text-zinc-500 ${compact ? 'hidden wide:block' : ''}`}>{description}</p>
      </div>
    </div>
  )
}
