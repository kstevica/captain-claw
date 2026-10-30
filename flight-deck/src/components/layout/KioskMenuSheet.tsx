import { useEffect, useRef, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ChevronRight, X, type LucideIcon } from 'lucide-react'

// ── A kiosk menu, as a modal ──
//
// What the agent "⋯" menu and Settings share: who or what it's about at the
// top, what can be done as tiles — each with its icon and a word on what it
// opens — and the one thing to be careful with kept apart at the bottom. A card
// in the middle of a larger screen, a sheet from the bottom on a phone. Arrow
// keys move between the tiles; Esc closes it.

export type MenuTone = 'violet' | 'fuchsia' | 'sky' | 'cyan' | 'teal' | 'emerald' | 'amber' | 'zinc'

const TONE: Record<MenuTone, string> = {
  violet: 'bg-violet-500/15 text-violet-600 ring-violet-500/20 dark:text-violet-300',
  fuchsia: 'bg-fuchsia-500/15 text-fuchsia-600 ring-fuchsia-500/20 dark:text-fuchsia-300',
  sky: 'bg-sky-500/15 text-sky-600 ring-sky-500/20 dark:text-sky-300',
  cyan: 'bg-cyan-500/15 text-cyan-600 ring-cyan-500/20 dark:text-cyan-300',
  // The Taskboard's own colour (its tile, its header).
  teal: 'bg-teal-500/15 text-teal-600 ring-teal-500/20 dark:text-teal-300',
  emerald: 'bg-emerald-500/15 text-emerald-600 ring-emerald-500/20 dark:text-emerald-300',
  amber: 'bg-amber-500/15 text-amber-600 ring-amber-500/20 dark:text-amber-300',
  zinc: 'bg-zinc-500/15 text-zinc-600 ring-zinc-500/20 dark:text-zinc-300',
}

export interface MenuItem {
  icon: LucideIcon
  label: string
  /** A word more under the label, or why it's off. */
  hint?: string
  onSelect: () => void
  danger?: boolean
  disabled?: boolean
  /** Its icon's colour in the menu. */
  tone?: MenuTone
  /** Turned on and off right here: a switch in place of the arrow. */
  on?: boolean
  /** It happens at once, nothing opens: no arrow. */
  direct?: boolean
  /** Under way: its icon spins. */
  busy?: boolean
}

export interface MenuSection {
  /** A small heading over its tiles. */
  title?: string
  items: MenuItem[]
  /** Small tiles of one line, the hint only a tooltip: the quick things. */
  compact?: boolean
}

const TILE = 'group relative flex min-w-0 border border-zinc-800 bg-zinc-950/40 text-left transition-all hover:-translate-y-px hover:border-zinc-700 hover:bg-zinc-800/60 hover:shadow-lg hover:shadow-black/10 focus-visible:border-violet-500 focus-visible:bg-violet-500/[0.07] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:translate-y-0 disabled:hover:border-zinc-800 disabled:hover:bg-zinc-950/40 disabled:hover:shadow-none'

const GRID: Record<number, string> = { 1: 'grid-cols-1', 2: 'grid-cols-2', 3: 'grid-cols-3' }

/** Up or down: the nearest tile in the next row that way (the rows aren't all the same length). */
function tileInNextRow(tiles: HTMLElement[], from: HTMLElement, way: number) {
  const box = from.getBoundingClientRect()
  const middle = box.left + box.width / 2
  let best: HTMLElement | undefined
  let bestScore = Infinity
  for (const tile of tiles) {
    const b = tile.getBoundingClientRect()
    const rows = (b.top - box.top) * way
    if (rows < 1) continue
    const score = rows * 1000 + Math.abs(b.left + b.width / 2 - middle)
    if (score < bestScore) {
      best = tile
      bestScore = score
    }
  }
  return best
}

function CompactTile({ item, wholeRow }: { item: MenuItem; wholeRow: boolean }) {
  const { icon: Icon, label, hint, onSelect, disabled, tone = 'zinc', on, busy } = item
  return (
    <button
      data-tile
      disabled={disabled}
      onClick={onSelect}
      title={hint}
      {...(on === undefined ? {} : { role: 'switch', 'aria-checked': on })}
      className={`${TILE} items-center gap-2.5 rounded-xl p-2.5 ${wholeRow ? 'col-span-full' : ''}`}
    >
      <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ring-1 ${TONE[tone]}`}>
        <Icon className={`h-4 w-4 ${busy ? 'animate-spin' : ''}`} />
      </span>
      <span className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-200">{label}</span>
      {on !== undefined && (
        <span className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${on ? 'bg-violet-500' : 'bg-zinc-700'}`} aria-hidden>
          <span className={`absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${on ? 'translate-x-4' : ''}`} />
        </span>
      )}
    </button>
  )
}

function Tile({ item, across, wholeRow }: { item: MenuItem; across: boolean; wholeRow: boolean }) {
  const { icon: Icon, label, hint, onSelect, disabled, tone = 'violet', busy } = item
  return (
    <button
      data-tile
      disabled={disabled}
      onClick={onSelect}
      title={hint}
      className={`${TILE} rounded-2xl p-3.5 ${across ? 'items-center gap-3.5' : 'flex-col'} ${wholeRow ? 'col-span-full' : ''}`}
    >
      <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ring-1 ${TONE[tone]}`}>
        <Icon className={`h-5 w-5 ${busy ? 'animate-spin' : ''}`} />
      </span>
      <span className={`min-w-0 ${across ? 'flex-1' : 'mt-3'}`}>
        <span className="block text-sm font-semibold leading-snug text-zinc-100">{label}</span>
        {hint && <span className="mt-0.5 line-clamp-2 text-balance text-xs leading-snug text-zinc-500">{hint}</span>}
      </span>
      <ChevronRight className={`h-4 w-4 shrink-0 text-zinc-600 transition-all group-hover:translate-x-0.5 group-hover:text-zinc-400 group-disabled:opacity-0 ${
        across ? '' : 'absolute right-3 top-3.5'}`} />
    </button>
  )
}

export function MenuSheet({ label, picture, status, title, subtitle, sections, apart = [], columns = 2, phone = false, onClose }: {
  /** What a screen reader calls it. */
  label: string
  /** Inside the coloured square at the top: initials or an icon. */
  picture: ReactNode
  /** A state as a dot, on the corner of the picture. */
  status?: ReactNode
  title: string
  subtitle?: string
  sections: MenuSection[]
  /** Kept apart at the bottom, in red. */
  apart?: MenuItem[]
  /** Tiles to a row (three need the wider card). */
  columns?: 2 | 3
  /** Always a sheet from the bottom: a phone's own row opened it. */
  phone?: boolean
  onClose: () => void
}) {
  const panelRef = useRef<HTMLDivElement>(null)
  const tiles = () => [...(panelRef.current?.querySelectorAll<HTMLButtonElement>('[data-tile]:not(:disabled)') ?? [])]

  // The first tile that can be used takes the focus.
  useEffect(() => {
    tiles()[0]?.focus()
  }, [])

  // Arrows move the focus — across a row, and up and down a column.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose()
        return
      }
      const across = { ArrowRight: 1, ArrowLeft: -1 }[e.key]
      const down = { ArrowDown: 1, ArrowUp: -1 }[e.key]
      if (!across && !down) return
      const all = tiles()
      const at = all.indexOf(document.activeElement as HTMLButtonElement)
      if (at < 0) return
      e.preventDefault()
      const next = across ? all[Math.max(0, Math.min(all.length - 1, at + across))] : tileInNextRow(all, all[at], down ?? 0)
      next?.focus()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  return createPortal(
    <div
      className={`fd-fade fixed inset-0 z-50 flex justify-center bg-black/50 backdrop-blur-[2px] ${phone ? 'items-end' : 'items-end wide:items-center wide:p-4'}`}
      onClick={onClose}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={label}
        onClick={(e) => e.stopPropagation()}
        className={`${phone ? 'fd-sheet' : 'fd-sheet fd-card-wide'} relative flex max-h-[88dvh] w-full flex-col wide:max-h-[92dvh] overflow-hidden border border-zinc-800 bg-zinc-900 pb-[env(safe-area-inset-bottom)] shadow-2xl shadow-black/40 ${
          phone ? 'rounded-t-3xl' : `rounded-t-3xl wide:rounded-3xl wide:pb-0 ${columns === 3 ? 'wide:max-w-2xl' : 'wide:max-w-lg'}`}`}
      >
        <div className={`mx-auto mt-2.5 h-1 w-10 shrink-0 rounded-full bg-zinc-700 ${phone ? '' : 'wide:hidden'}`} aria-hidden />
        <div className="flex shrink-0 items-center gap-3.5 px-5 pb-4 pt-4 wide:pt-5">
          <span className="relative flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-sm font-semibold text-white shadow-lg shadow-violet-500/25">
            {picture}
            {status && <span className="absolute -bottom-1 -right-1 flex rounded-full bg-zinc-900 p-[3px]">{status}</span>}
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-base font-semibold text-zinc-100">{title}</h2>
            {subtitle && <p className="truncate text-xs text-zinc-500">{subtitle}</p>}
          </div>
          <button onClick={onClose} aria-label="Close" title="Close"
            className="-mr-1.5 rounded-full p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="min-h-0 space-y-3 overflow-y-auto overscroll-contain px-4 pb-4 wide:px-5 wide:pb-5">
          {sections.filter((s) => s.items.length > 0).map(({ title: heading, items, compact }, n) => {
            // Two to a row is too narrow for a small tile's label and switch.
            const cols = compact && columns < 3 ? 1 : Math.min(columns, items.length)
            return (
              <section key={heading ?? n}>
                {heading && <h3 className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">{heading}</h3>}
                <div className={`grid gap-2.5 ${GRID[cols]}`}>
                  {items.map((item, i) => {
                    // An odd one out takes the whole last row, laid out across it;
                    // so is every tile of a row with room to spare.
                    const alone = items.length % cols === 1 && i === items.length - 1
                    if (compact) return <CompactTile key={item.label} item={item} wholeRow={alone} />
                    return <Tile key={item.label} item={item} across={alone || cols < columns} wholeRow={alone} />
                  })}
                </div>
              </section>
            )
          })}

          {apart.length > 0 && (
            <div className="border-t border-zinc-800 pt-3">
              {apart.map(({ icon: Icon, label: name, hint, onSelect, disabled, direct, busy }) => (
                <button
                  key={name}
                  data-tile
                  disabled={disabled}
                  onClick={onSelect}
                  className="group flex w-full items-center gap-3.5 rounded-2xl p-3 text-left transition-colors hover:bg-red-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500/40 disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-transparent"
                >
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-red-500/10 text-red-600 ring-1 ring-red-500/20 dark:text-red-400">
                    <Icon className={`h-5 w-5 ${busy ? 'animate-spin' : ''}`} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-semibold text-red-600 dark:text-red-400">{name}</span>
                    {hint && <span className="block truncate text-xs text-zinc-500">{hint}</span>}
                  </span>
                  {!direct && <ChevronRight className="h-4 w-4 shrink-0 text-red-400/50 transition-transform group-hover:translate-x-0.5" />}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}
