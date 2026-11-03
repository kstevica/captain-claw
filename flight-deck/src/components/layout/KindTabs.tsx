import { LayoutGrid } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { KINDS, type ArchetypeKind } from './archetypeLook'

// The gallery's first split: All, then Company, Generalists and Specialists, each in
// its own colour whether chosen or not. The New agent picker, the Library and the
// Admin console's Archetypes page all filter with it.

export type KindFilter = ArchetypeKind | 'all'

export function KindTabs({ value, onChange, counts, hints, compact }: {
  value: KindFilter
  onChange: (next: KindFilter) => void
  /** How many of each kind there are: a kind with none isn't offered. */
  counts: Partial<Record<ArchetypeKind, number>>
  /** Say under each name what the kind is for, where the room allows. */
  hints?: boolean
  /** A phone: smaller, and the row scrolls sideways. */
  compact?: boolean
}) {
  const kinds = KINDS.filter((k) => (counts[k.id] || 0) > 0)
  const total = kinds.reduce((n, k) => n + (counts[k.id] || 0), 0)
  const row = (
    <div
      role="tablist"
      aria-label="Kind of agent"
      // A tab never cuts its name short: when they don't fit, the row scrolls.
      className={compact
        ? '-mx-4 flex gap-1.5 overflow-x-auto px-4 py-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden'
        : `flex overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden ${hints ? 'gap-2' : 'gap-1.5'}`}
    >
      <Tab
        active={value === 'all'} onClick={() => onChange('all')} compact={compact} hints={hints}
        icon={LayoutGrid} label="All" count={total} hint="Every archetype on this deck."
        tile="bg-violet-500/15 text-violet-600 ring-violet-500/25 dark:text-violet-300" activeFrame="border-violet-500/50 bg-violet-500/[0.08]"
        grow={false} squeeze
      />
      {kinds.map((k) => (
        <Tab
          key={k.id} active={value === k.id} onClick={() => onChange(k.id)} compact={compact} hints={hints}
          icon={k.icon} label={k.label} count={counts[k.id] || 0} hint={k.hint}
          tile={`${k.bg} ${k.text} ${k.ring}`} activeFrame={k.active}
        />
      ))}
    </div>
  )
  // Where the row is tight, All gives up its name first: its grid icon says it.
  return compact ? row : <div className="@container min-w-0">{row}</div>
}

function Tab({ active, onClick, icon: Icon, label, count, hint, tile, activeFrame, compact, hints, grow = true, squeeze }: {
  active: boolean
  onClick: () => void
  icon: LucideIcon
  label: string
  count: number
  hint: string
  tile: string
  activeFrame: string
  compact?: boolean
  hints?: boolean
  grow?: boolean
  /** Only its icon shows when the row is tight. */
  squeeze?: boolean
}) {
  const frame = active ? activeFrame : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-900/70'
  if (compact) {
    return (
      <button role="tab" aria-selected={active} onClick={onClick} className={`flex shrink-0 items-center gap-1.5 rounded-full border py-1 pl-1 pr-3 text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${frame}`}>
        <span className={`flex h-6 w-6 items-center justify-center rounded-full ring-1 ${tile}`}><Icon className="h-3.5 w-3.5" /></span>
        <span className={active ? 'font-semibold text-zinc-100' : 'font-medium text-zinc-300'}>{label}</span>
        <span className="text-[10px] tabular-nums text-zinc-500">{count}</span>
      </button>
    )
  }
  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={onClick}
      title={hints ? undefined : hint}
      className={`flex shrink-0 items-center rounded-xl border text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${
        hints ? 'gap-2.5 px-2.5 py-2' : 'gap-2 px-2 py-1.5'} ${grow ? `flex-auto ${hints ? 'basis-0' : ''}` : ''} ${frame}`}
    >
      <span className={`flex shrink-0 items-center justify-center rounded-lg ring-1 ${hints ? 'h-9 w-9' : 'h-6 w-6'} ${tile}`}>
        <Icon className={hints ? 'h-[18px] w-[18px]' : 'h-3.5 w-3.5'} />
      </span>
      <span className={`min-w-0 flex-1 ${squeeze ? '@max-[35rem]:sr-only' : ''}`}>
        <span className="flex items-baseline gap-1.5 whitespace-nowrap">
          <span className={`text-sm ${active ? 'font-semibold text-zinc-100' : 'font-medium text-zinc-300'}`}>{label}</span>
          <span className="text-[11px] tabular-nums text-zinc-500">{count}</span>
        </span>
        {hints && <span className="line-clamp-2 block text-[11px] leading-snug text-zinc-500">{hint}</span>}
      </span>
    </button>
  )
}
