import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import { Filter, X } from 'lucide-react'
import {
  OPS, describeFilter, filterReady, opInfo, type ColumnFilter, type FilterKind, type FilterOp,
} from './columnFilters'

// The column filters' pieces on screen: the funnel in a column's header, the
// filter it opens, and the chips above the rows saying what is filtered.

// ── The funnel in a column's header ──

export function FilterButton({ label, active, onOpen }: {
  label: string; active: boolean; onOpen: (anchor: DOMRect) => void
}) {
  return (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); onOpen(e.currentTarget.getBoundingClientRect()) }}
      className={`rounded p-0.5 transition-colors ${active
        ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
        : 'text-zinc-600 hover:bg-zinc-800 hover:text-zinc-200'}`}
      title={active ? `Change the filter on ${label}` : `Filter ${label}`}
      aria-label={active ? `Change the filter on ${label}` : `Filter ${label}`}
      aria-pressed={active}
    >
      <Filter className="h-3 w-3" />
    </button>
  )
}

// ── A column's filter, under its funnel ──

const POP_W = 288

export function FilterPopover({ anchor, label, fk, current, onApply, onClose }: {
  anchor: DOMRect
  label: string
  fk: FilterKind
  /** The column's filter now, if it has one. */
  current?: ColumnFilter
  /** The new filter, or null to take it off. */
  onApply: (f: Omit<ColumnFilter, 'col'> | null) => void
  onClose: () => void
}) {
  const [op, setOp] = useState<FilterOp>(current?.op ?? OPS[fk][0].op)
  const [value, setValue] = useState(current?.value ?? '')
  const [value2, setValue2] = useState(current?.value2 ?? '')
  const box = useRef<HTMLDivElement>(null)

  // Under the funnel, kept inside the window (above it when there's no room
  // below), placed once its height is known — before it is painted.
  useLayoutEffect(() => {
    const el = box.current
    if (!el) return
    const h = el.offsetHeight
    const below = anchor.bottom + 6
    el.style.left = `${Math.max(8, Math.min(anchor.left - 8, window.innerWidth - POP_W - 8))}px`
    el.style.top = `${below + h > window.innerHeight - 8 ? Math.max(8, anchor.top - h - 6) : below}px`
  }, [anchor, op])

  // A click elsewhere, or the window changing size, closes it.
  useEffect(() => {
    const down = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) onClose() }
    window.addEventListener('mousedown', down, true)
    window.addEventListener('resize', onClose)
    return () => { window.removeEventListener('mousedown', down, true); window.removeEventListener('resize', onClose) }
  }, [onClose])

  const info = opInfo(fk, op)
  const draft = { op, value: info.args ? value : undefined, value2: info.args === 2 ? value2 : undefined }
  const ready = filterReady({ col: '', ...draft }, fk)
  const apply = () => { if (ready) onApply(draft) }
  const keys = (e: ReactKeyboardEvent) => {
    if (e.key === 'Escape') { e.stopPropagation(); e.preventDefault(); onClose() }
    if (e.key === 'Enter' && !e.nativeEvent.isComposing && (e.target as HTMLElement).tagName === 'INPUT') { e.preventDefault(); apply() }
  }
  const inputType = fk === 'date' ? 'date' : 'text'
  const input = 'w-full min-w-0 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-[13px] text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500/70 focus:outline-none'

  return createPortal(
    <div
      ref={box}
      data-cell-editor
      tabIndex={-1}
      role="dialog"
      aria-label={`Filter ${label}`}
      onKeyDown={keys}
      onMouseDown={(e) => e.stopPropagation()}
      className="fixed z-[200] rounded-xl border border-zinc-700 bg-zinc-900 p-3 shadow-2xl"
      style={{ left: anchor.left, top: anchor.bottom + 6, width: POP_W }}
    >
      <div className="mb-2 flex items-center gap-2">
        <Filter className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
        <span className="min-w-0 flex-1 truncate text-xs font-semibold text-zinc-200" title={label}>{label}</span>
        <button type="button" onClick={onClose} className="rounded p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" aria-label="Close">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <select
        value={op}
        onChange={(e) => setOp(e.target.value as FilterOp)}
        autoFocus={opInfo(fk, op).args === 0}
        className={`${input} mb-2`}
        aria-label="How to filter"
      >
        {OPS[fk].map((o) => <option key={o.op} value={o.op}>{o.label}</option>)}
      </select>
      {info.args >= 1 && (
        <div className="flex items-center gap-1.5">
          <input
            type={inputType}
            inputMode={fk === 'number' ? 'decimal' : undefined}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={fk === 'number' ? (info.args === 2 ? 'From' : 'A number') : info.args === 2 ? 'From' : 'Value'}
            autoFocus
            className={input}
            aria-label={info.args === 2 ? 'From' : 'Value'}
          />
          {info.args === 2 && (
            <>
              <span className="shrink-0 text-[11px] text-zinc-500">and</span>
              <input
                type={inputType}
                inputMode={fk === 'number' ? 'decimal' : undefined}
                value={value2}
                onChange={(e) => setValue2(e.target.value)}
                placeholder="To"
                className={input}
                aria-label="To"
              />
            </>
          )}
        </div>
      )}
      <div className="mt-3 flex items-center gap-2">
        {current && (
          <button type="button" onClick={() => onApply(null)} className="rounded-md px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100">
            Remove filter
          </button>
        )}
        <button
          type="button"
          onClick={apply}
          disabled={!ready}
          className="ml-auto rounded-md bg-emerald-600 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-40"
        >
          Filter
        </button>
      </div>
    </div>,
    document.body,
  )
}

// ── What is filtered, above the rows ──

export function FilterChips({ filters, labelOf, kindOf, onEdit, onRemove, onClear }: {
  filters: ColumnFilter[]
  labelOf: (col: string) => string
  kindOf: (col: string) => FilterKind
  onEdit: (col: string, anchor: DOMRect) => void
  onRemove: (col: string) => void
  onClear: () => void
}) {
  if (!filters.length) return null
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-1.5 border-b border-zinc-800 px-3 py-1.5">
      <Filter className="h-3 w-3 shrink-0 text-zinc-500" />
      {filters.map((f) => (
        <span key={f.col} className="inline-flex max-w-full items-center overflow-hidden rounded-full bg-emerald-500/10 text-[11px] text-emerald-800 ring-1 ring-emerald-500/25 dark:text-emerald-200">
          <button
            type="button"
            onClick={(e) => onEdit(f.col, e.currentTarget.getBoundingClientRect())}
            className="truncate py-0.5 pl-2.5 pr-1 text-left hover:underline"
            title="Change this filter"
          >
            <span className="font-medium">{labelOf(f.col)}</span> {describeFilter(f, kindOf(f.col))}
          </button>
          <button
            type="button"
            onClick={() => onRemove(f.col)}
            className="mr-1 rounded-full p-0.5 hover:bg-emerald-500/20"
            aria-label={`Remove the filter on ${labelOf(f.col)}`}
            title="Remove this filter"
          >
            <X className="h-3 w-3" />
          </button>
        </span>
      ))}
      {filters.length > 1 && (
        <button type="button" onClick={onClear} className="ml-1 text-[11px] text-zinc-500 hover:text-zinc-200 hover:underline">
          Clear all
        </button>
      )}
    </div>
  )
}
