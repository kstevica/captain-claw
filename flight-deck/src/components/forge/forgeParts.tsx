import { useEffect, useState, type ReactNode } from 'react'
import { AlertTriangle, ChevronDown, type LucideIcon } from 'lucide-react'
import type { Archetype } from '../../services/tierConfig'
import { archetypeLook } from '../layout/archetypeLook'

// ── What Forge agents' steps share ──
//
// The Admin console's look (console/parts.tsx), as Crew has it: soft tinted
// squares, cards, a violet accent, and what a person wants up front with the
// detail behind a click.

/** An agent's archetype as a tinted square: the same one the New agent gallery shows. */
export function AgentBadge({ archetype, archetypes, size = 'md' }: {
  archetype: string; archetypes: Archetype[]; size?: 'sm' | 'md' | 'lg'
}) {
  const look = archetypeLook(archetype, archetypes.find((a) => a.id === archetype)?.family || '')
  const Icon = look.icon
  const box = size === 'lg' ? 'h-11 w-11 rounded-2xl' : size === 'sm' ? 'h-7 w-7 rounded-lg' : 'h-9 w-9 rounded-xl'
  const glyph = size === 'lg' ? 'h-5 w-5' : size === 'sm' ? 'h-3.5 w-3.5' : 'h-[18px] w-[18px]'
  return (
    <span className={`flex shrink-0 items-center justify-center ${box} ${look.bg} ${look.text}`}>
      <Icon className={glyph} />
    </span>
  )
}

/** A short eyebrow over a heading. */
export function Eyebrow({ icon: Icon, tone = 'violet', children }: { icon?: LucideIcon; tone?: 'violet' | 'emerald' | 'amber'; children: ReactNode }) {
  const color = { violet: 'text-violet-700 dark:text-violet-300', emerald: 'text-emerald-700 dark:text-emerald-300', amber: 'text-amber-700 dark:text-amber-300' }[tone]
  return (
    <p className={`flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider ${color}`}>
      {Icon && <Icon className="h-3.5 w-3.5" />}
      {children}
    </p>
  )
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p role="alert" className="flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/5 px-3 py-2.5 text-sm leading-relaxed text-red-700 dark:text-red-300">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> <span className="min-w-0">{children}</span>
    </p>
  )
}

/** One of several choices that can each be on or off. */
export function Chip({ on, disabled, onClick, children }: { on: boolean; disabled?: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={on}
      className={`rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
        on ? 'border-violet-500/50 bg-violet-500/15 text-violet-700 dark:text-violet-200'
          : 'border-zinc-800 bg-zinc-950 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200'
      }`}
    >
      {children}
    </button>
  )
}

/** Detail behind a click. */
export function Fold({ label, open, onToggle, children }: { label: ReactNode; open: boolean; onToggle: () => void; children: ReactNode }) {
  return (
    <div>
      <button type="button" onClick={onToggle} aria-expanded={open} className="inline-flex items-center gap-1 text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-300">
        <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? '' : '-rotate-90'}`} /> {label}
      </button>
      {open && <div className="fd-rise mt-2">{children}</div>}
    </div>
  )
}

/** How long something has been going: "45s", "2m 05s". */
export function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [])
  const secs = Math.max(0, Math.round(now / 1000 - since))
  const text = secs < 60 ? `${secs}s` : `${Math.floor(secs / 60)}m ${String(secs % 60).padStart(2, '0')}s`
  return <span className="tabular-nums">{text}</span>
}
