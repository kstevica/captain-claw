import { useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import { Check, Filter, Loader2, Search, X } from 'lucide-react'
import {
  CONDITIONS, datePresets, describeFilter, filterReady, opInfo, toNumber,
  type ColumnFilter, type FilterKind, type FilterOp, type ValueStats,
} from './columnFilters'

// The column filters' pieces on screen: the funnel in a column's header, the
// filter it opens, and the chips above the rows saying what is filtered.
//
// A column's filter has up to three ways in, as tabs: Values (tick the ones to
// keep — or to hide — from the column's own, most common first, searchable),
// Range (numbers or days between two ends, either open; quick picks: top 10,
// above average, last 30 days …) and Condition (the free-form ones: contains,
// > 10, before a day, repeats, empty …). Yes/no columns are three buttons.

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

const POP_W = 340

type Tab = 'values' | 'range' | 'condition'
type Stats = { state: 'loading' } | { state: 'ready'; data: ValueStats } | { state: 'error'; why: string }

const TABS: Record<FilterKind, Tab[]> = {
  text: ['values', 'condition'],
  number: ['range', 'values', 'condition'],
  date: ['range', 'values', 'condition'],
  boolean: ['condition'],
}
const TAB_LABEL: Record<Tab, string> = { values: 'Values', range: 'Range', condition: 'Condition' }

function startTab(fk: FilterKind, op?: FilterOp): Tab {
  if (op === 'in' || op === 'not_in') return 'values'
  if (op === 'range' || op === 'between') return fk === 'text' ? 'condition' : 'range'
  if (op) return 'condition'
  return TABS[fk][0]
}

const inputCls = 'w-full min-w-0 rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-[13px] text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500/70 focus:outline-none'
const chipCls = 'rounded-full border border-zinc-700 px-2.5 py-1 text-[11px] text-zinc-300 transition-colors hover:border-emerald-500/60 hover:bg-emerald-500/10 hover:text-emerald-700 dark:hover:text-emerald-300'

export function FilterPopover({ anchor, label, fk, current, loadStats, onApply, onClose }: {
  anchor: DOMRect
  label: string
  fk: FilterKind
  /** The column's filter now, if it has one. */
  current?: ColumnFilter
  /** What the column holds (among the rows the other filters leave), `search` narrowing its values. */
  loadStats: (search: string) => Promise<ValueStats>
  /** The new filter, or null to take it off. */
  onApply: (f: Omit<ColumnFilter, 'col'> | null) => void
  onClose: () => void
}) {
  const [tab, setTab] = useState<Tab>(() => startTab(fk, current?.op))
  const box = useRef<HTMLDivElement>(null)

  // Values
  const picking = current?.op === 'in' || current?.op === 'not_in'
  const [picked, setPicked] = useState<Set<string>>(() => new Set(picking ? current?.values ?? [] : []))
  const [mode, setMode] = useState<'in' | 'not_in'>(current?.op === 'not_in' ? 'not_in' : 'in')
  const [search, setSearch] = useState('')
  const [order, setOrder] = useState<'count' | 'value'>(fk === 'text' ? 'count' : 'value')
  const [stats, setStats] = useState<Stats>({ state: 'loading' })
  // The whole column's, for the range's ends (a search narrows `stats`).
  const [bounds, setBounds] = useState<{ min: ValueStats['min']; max: ValueStats['max'] } | null>(null)

  // Range
  const ranged = current?.op === 'range' || current?.op === 'between'
  const [lo, setLo] = useState(ranged ? current?.value ?? '' : '')
  const [hi, setHi] = useState(ranged ? current?.value2 ?? '' : '')

  // Condition
  const conditional = !!current && !picking && !(ranged && fk !== 'text')
  const [op, setOp] = useState<FilterOp>(conditional && current ? current.op : CONDITIONS[fk][0].op)
  const [value, setValue] = useState(conditional ? current?.value ?? '' : '')
  const [value2, setValue2] = useState(conditional ? current?.value2 ?? '' : '')

  // What the column holds: at once (the range shows its ends), again as a search narrows it.
  // The newest `loadStats` (the caller makes it anew each render), kept without re-running the load.
  const load = useRef(loadStats)
  useEffect(() => { load.current = loadStats })
  useEffect(() => {
    if (fk === 'boolean') return
    let live = true
    const t = setTimeout(() => {
      load.current(search)
        .then((data) => {
          if (!live) return
          setStats({ state: 'ready', data })
          if (!search) setBounds({ min: data.min, max: data.max })
        })
        .catch((e) => { if (live) setStats({ state: 'error', why: e instanceof Error ? e.message : String(e) }) })
    }, search ? 220 : 0)
    return () => { live = false; clearTimeout(t) }
  }, [search, fk])

  // Under the funnel, kept inside the window (above it when there's no room
  // below), placed once its height is known — before it is painted.
  useLayoutEffect(() => {
    const el = box.current
    if (!el) return
    const h = el.offsetHeight
    const room = window.innerHeight - 8
    const below = anchor.bottom + 6
    const above = anchor.top - h - 6
    // Below the funnel when it fits, else above it when that fits, else as
    // low as the window lets it sit.
    const top = below + h <= room ? below : above >= 8 ? above : Math.max(8, room - h)
    el.style.left = `${Math.max(8, Math.min(anchor.left - 8, window.innerWidth - POP_W - 8))}px`
    el.style.top = `${top}px`
  }, [anchor, tab, op, stats.state])

  // A click elsewhere, or the window changing size, closes it.
  useEffect(() => {
    const down = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) onClose() }
    window.addEventListener('mousedown', down, true)
    window.addEventListener('resize', onClose)
    return () => { window.removeEventListener('mousedown', down, true); window.removeEventListener('resize', onClose) }
  }, [onClose])

  const info = opInfo(fk, op)
  const draft: Omit<ColumnFilter, 'col'> | null =
    tab === 'values' ? (picked.size ? { op: mode, values: [...picked] } : null)
      : tab === 'range' ? { op: 'range', value: lo.trim() || undefined, value2: hi.trim() || undefined }
        : {
          op,
          value: info.args === 0 ? undefined : value,
          value2: info.args === 2 ? value2 : undefined,
        }
  const ready = !!draft && filterReady({ col: '', ...draft }, fk)
  const apply = () => { if (ready && draft) onApply(draft) }
  const keys = (e: ReactKeyboardEvent) => {
    if (e.key === 'Escape') { e.stopPropagation(); e.preventDefault(); onClose() }
    if (e.key === 'Enter' && !e.nativeEvent.isComposing && (e.target as HTMLElement).tagName === 'INPUT'
      && (e.target as HTMLInputElement).type !== 'checkbox') { e.preventDefault(); apply() }
  }

  const tabs = TABS[fk]
  return createPortal(
    <div
      ref={box}
      data-cell-editor
      tabIndex={-1}
      role="dialog"
      aria-label={`Filter ${label}`}
      onKeyDown={keys}
      onMouseDown={(e) => e.stopPropagation()}
      className="fixed z-[200] flex max-h-[calc(100vh-16px)] flex-col overflow-hidden rounded-2xl border border-zinc-700/80 bg-zinc-900 shadow-2xl outline-none"
      style={{ left: anchor.left, top: anchor.bottom + 6, width: POP_W }}
    >
      <div className="flex items-center gap-2 px-3.5 pb-2 pt-3">
        <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-emerald-500/10">
          <Filter className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
        </span>
        <span className="min-w-0 flex-1 truncate text-[13px] font-semibold text-zinc-100" title={label}>{label}</span>
        <button type="button" onClick={onClose} className="rounded-md p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" aria-label="Close">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      {tabs.length > 1 && (
        <div role="tablist" aria-label="How to filter" className="mx-3.5 mb-2.5 flex rounded-lg bg-zinc-800/70 p-0.5">
          {tabs.map((t) => (
            <button key={t} type="button" role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
              className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors ${tab === t
                ? 'bg-zinc-950 text-zinc-100 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'}`}>
              {TAB_LABEL[t]}
            </button>
          ))}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-3.5 pb-1">
        {fk === 'boolean' ? (
          <BooleanPick current={current?.op} onPick={(o) => onApply({ op: o })} />
        ) : tab === 'values' ? (
          <ValuesTab
            fk={fk} stats={stats} search={search} onSearch={setSearch} picked={picked} setPicked={setPicked}
            mode={mode} setMode={setMode} order={order} setOrder={setOrder}
          />
        ) : tab === 'range' ? (
          <RangeTab fk={fk} bounds={bounds} lo={lo} hi={hi} setLo={setLo} setHi={setHi} onQuick={onApply} />
        ) : (
          <ConditionTab fk={fk} op={op} setOp={setOp} value={value} setValue={setValue} value2={value2} setValue2={setValue2} />
        )}
      </div>

      {fk !== 'boolean' && (
        <div className="flex items-center gap-2 border-t border-zinc-800 px-3.5 py-2.5">
          {current && (
            <button type="button" onClick={() => onApply(null)} className="rounded-md px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100">
              Remove filter
            </button>
          )}
          <button
            type="button"
            onClick={apply}
            disabled={!ready}
            className="ml-auto rounded-lg bg-emerald-600 px-3.5 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-emerald-500 disabled:opacity-40"
          >
            Filter
          </button>
        </div>
      )}
    </div>,
    document.body,
  )
}

// ── Values: tick the ones to keep, or to hide ──

function ValuesTab({ fk, stats, search, onSearch, picked, setPicked, mode, setMode, order, setOrder }: {
  fk: FilterKind
  stats: Stats
  search: string
  onSearch: (v: string) => void
  picked: Set<string>
  setPicked: (s: Set<string>) => void
  mode: 'in' | 'not_in'
  setMode: (m: 'in' | 'not_in') => void
  order: 'count' | 'value'
  setOrder: (o: 'count' | 'value') => void
}) {
  const data = stats.state === 'ready' ? stats.data : null
  const list = useMemo(() => {
    const vs = data ? [...data.values] : []
    if (order === 'value') {
      vs.sort((a, b) => {
        if (fk === 'number') {
          const x = toNumber(a.value)
          const y = toNumber(b.value)
          if (!Number.isNaN(x) && !Number.isNaN(y)) return x - y
        }
        return a.value.localeCompare(b.value, undefined, { numeric: true, sensitivity: 'base' })
      })
    }
    return vs
  }, [data, order, fk])
  const showEmpty = !!data && data.empty > 0 && !search.trim()
  const toggle = (v: string) => {
    const next = new Set(picked)
    if (next.has(v)) next.delete(v)
    else next.add(v)
    setPicked(next)
  }
  const allShown = () => {
    const next = new Set(picked)
    for (const v of list) next.add(v.value)
    if (showEmpty) next.add('')
    setPicked(next)
  }
  const seg = (on: boolean) => `rounded-md px-2 py-0.5 text-[11px] font-medium transition-colors ${on
    ? 'bg-zinc-950 text-zinc-100 shadow-sm' : 'text-zinc-400 hover:text-zinc-200'}`
  const link = 'text-[11px] text-emerald-700 hover:underline disabled:opacity-40 disabled:no-underline dark:text-emerald-400'

  return (
    <div className="space-y-2">
      <div className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
        <input value={search} onChange={(e) => onSearch(e.target.value)} placeholder="Search the values…" autoFocus
          className={`${inputCls} pl-8`} aria-label="Search the values" />
      </div>
      <div className="flex items-center gap-2">
        <div className="flex rounded-lg bg-zinc-800/70 p-0.5" role="radiogroup" aria-label="Keep or hide the ticked values">
          <button type="button" role="radio" aria-checked={mode === 'in'} onClick={() => setMode('in')} className={seg(mode === 'in')}>Show only</button>
          <button type="button" role="radio" aria-checked={mode === 'not_in'} onClick={() => setMode('not_in')} className={seg(mode === 'not_in')}>Hide</button>
        </div>
        <button type="button" onClick={() => setOrder(order === 'count' ? 'value' : 'count')}
          className="ml-auto text-[11px] text-zinc-500 hover:text-zinc-200" title="Change the order">
          {order === 'count' ? 'Most common first' : fk === 'number' ? 'Smallest first' : fk === 'date' ? 'Earliest first' : 'A–Z'}
        </button>
      </div>
      <div className="flex items-center gap-3">
        <button type="button" onClick={allShown} disabled={!list.length && !showEmpty} className={link}>Tick all shown</button>
        <button type="button" onClick={() => setPicked(new Set())} disabled={!picked.size} className={link}>Clear</button>
        <span className="ml-auto text-[11px] tabular-nums text-zinc-500">{picked.size ? `${picked.size} ticked` : 'None ticked'}</span>
      </div>
      <div className="max-h-56 overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-950/40 py-1" role="listbox" aria-multiselectable aria-label="Values">
        {stats.state === 'loading' ? (
          <div className="flex justify-center py-6"><Loader2 className="h-4 w-4 animate-spin text-zinc-500" /></div>
        ) : stats.state === 'error' ? (
          <p className="px-3 py-4 text-center text-xs text-red-600 dark:text-red-400">{stats.why}</p>
        ) : !list.length && !showEmpty ? (
          <p className="px-3 py-4 text-center text-xs text-zinc-500">{search.trim() ? 'No value matches.' : 'Nothing in this column.'}</p>
        ) : (
          <>
            {showEmpty && <ValueRow text="(empty)" muted count={data?.empty ?? 0} on={picked.has('')} onToggle={() => toggle('')} />}
            {list.map((v) => <ValueRow key={v.value} text={v.value} count={v.count} on={picked.has(v.value)} onToggle={() => toggle(v.value)} />)}
          </>
        )}
      </div>
      {data && data.distinct > list.length && (
        <p className="text-[11px] leading-snug text-zinc-500">
          The {list.length.toLocaleString()} most common of {data.distinct.toLocaleString()} — search to find the others.
        </p>
      )}
    </div>
  )
}

function ValueRow({ text, count, on, muted = false, onToggle }: {
  text: string; count: number; on: boolean; muted?: boolean; onToggle: () => void
}) {
  return (
    <button type="button" role="option" aria-selected={on} onClick={onToggle}
      className="flex w-full items-center gap-2.5 px-2.5 py-1.5 text-left transition-colors hover:bg-zinc-800/60">
      <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors ${on
        ? 'border-emerald-600 bg-emerald-600 text-white' : 'border-zinc-600'}`}>
        {on && <Check className="h-3 w-3" strokeWidth={3} />}
      </span>
      <span className={`min-w-0 flex-1 truncate text-[13px] ${muted ? 'italic text-zinc-500' : 'text-zinc-200'}`} title={text}>{text}</span>
      <span className="shrink-0 text-[11px] tabular-nums text-zinc-500">{count.toLocaleString()}</span>
    </button>
  )
}

// ── Range: between two ends, either open ──

/** A step that moves a slider in ~200 even stops across the span. */
function stepFor(min: number, max: number): number {
  const span = max - min
  if (Number.isInteger(min) && Number.isInteger(max) && span >= 50) return 1
  if (span <= 0) return 1
  return 10 ** Math.floor(Math.log10(span / 200))
}

function fmtStep(n: number, step: number): string {
  const decimals = Math.max(0, -Math.floor(Math.log10(step)))
  return String(Number(n.toFixed(decimals)))
}

function RangeTab({ fk, bounds, lo, hi, setLo, setHi, onQuick }: {
  fk: FilterKind
  bounds: { min: ValueStats['min']; max: ValueStats['max'] } | null
  lo: string
  hi: string
  setLo: (v: string) => void
  setHi: (v: string) => void
  onQuick: (f: Omit<ColumnFilter, 'col'>) => void
}) {
  if (fk === 'date') {
    const presets = datePresets()
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <label className="min-w-0 flex-1">
            <span className="mb-1 block text-[11px] text-zinc-500">From</span>
            <input type="date" value={lo} onChange={(e) => setLo(e.target.value)} className={`${inputCls} dark:[color-scheme:dark]`} aria-label="From" />
          </label>
          <label className="min-w-0 flex-1">
            <span className="mb-1 block text-[11px] text-zinc-500">To</span>
            <input type="date" value={hi} onChange={(e) => setHi(e.target.value)} className={`${inputCls} dark:[color-scheme:dark]`} aria-label="To" />
          </label>
        </div>
        {bounds?.min != null && (
          <p className="text-[11px] text-zinc-500">Earliest {String(bounds.min)} · latest {String(bounds.max)}</p>
        )}
        <div>
          <p className="mb-1.5 text-[11px] font-medium text-zinc-400">Quick picks</p>
          <div className="flex flex-wrap gap-1.5">
            {presets.map((p) => (
              <button key={p.label} type="button" className={chipCls} onClick={() => onQuick({ op: 'range', value: p.from, value2: p.to })}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    )
  }
  const min = typeof bounds?.min === 'number' ? bounds.min : null
  const max = typeof bounds?.max === 'number' ? bounds.max : null
  const slider = min !== null && max !== null && max > min
  const step = slider ? stepFor(min, max) : 1
  const loN = lo.trim() ? toNumber(lo) : NaN
  const hiN = hi.trim() ? toNumber(hi) : NaN
  const loAt = slider ? Math.min(max, Math.max(min, Number.isNaN(loN) ? min : loN)) : 0
  const hiAt = slider ? Math.max(min, Math.min(max, Number.isNaN(hiN) ? max : hiN)) : 0
  const pct = (v: number) => (slider ? ((v - min) / (max - min)) * 100 : 0)
  return (
    <div className="space-y-3">
      <div className="flex items-end gap-2">
        <label className="min-w-0 flex-1">
          <span className="mb-1 block text-[11px] text-zinc-500">At least</span>
          <input type="text" inputMode="decimal" value={lo} onChange={(e) => setLo(e.target.value)} autoFocus
            placeholder={min !== null ? `${min}` : 'Any'} className={`${inputCls} tabular-nums`} aria-label="At least" />
        </label>
        <span className="pb-2 text-zinc-600">–</span>
        <label className="min-w-0 flex-1">
          <span className="mb-1 block text-[11px] text-zinc-500">At most</span>
          <input type="text" inputMode="decimal" value={hi} onChange={(e) => setHi(e.target.value)}
            placeholder={max !== null ? `${max}` : 'Any'} className={`${inputCls} tabular-nums`} aria-label="At most" />
        </label>
      </div>
      {slider && (
        <div className="fd-dual-range relative h-6">
          <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-zinc-800" />
          <div className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-emerald-500"
            style={{ left: `${pct(loAt)}%`, right: `${100 - pct(hiAt)}%` }} />
          <input type="range" min={min} max={max} step={step} value={loAt} aria-label="At least"
            onChange={(e) => { const v = Math.min(Number(e.target.value), hiAt); setLo(v <= min ? '' : fmtStep(v, step)) }} />
          <input type="range" min={min} max={max} step={step} value={hiAt} aria-label="At most"
            onChange={(e) => { const v = Math.max(Number(e.target.value), loAt); setHi(v >= max ? '' : fmtStep(v, step)) }} />
        </div>
      )}
      {min !== null && (
        <p className="text-[11px] text-zinc-500">The column runs from {min.toLocaleString()} to {(max ?? min).toLocaleString()}; an empty end is open.</p>
      )}
      <div>
        <p className="mb-1.5 text-[11px] font-medium text-zinc-400">Quick picks</p>
        <div className="flex flex-wrap gap-1.5">
          <button type="button" className={chipCls} onClick={() => onQuick({ op: 'top', value: '10' })}>Top 10</button>
          <button type="button" className={chipCls} onClick={() => onQuick({ op: 'bottom', value: '10' })}>Bottom 10</button>
          <button type="button" className={chipCls} onClick={() => onQuick({ op: 'above_avg' })}>Above average</button>
          <button type="button" className={chipCls} onClick={() => onQuick({ op: 'below_avg' })}>Below average</button>
        </div>
      </div>
    </div>
  )
}

// ── Condition: the free-form ones ──

function ConditionTab({ fk, op, setOp, value, setValue, value2, setValue2 }: {
  fk: FilterKind
  op: FilterOp
  setOp: (o: FilterOp) => void
  value: string
  setValue: (v: string) => void
  value2: string
  setValue2: (v: string) => void
}) {
  const info = opInfo(fk, op)
  const type = fk === 'date' ? 'date' : 'text'
  const numeric = fk === 'number' || info.args === 'count'
  return (
    <div className="space-y-2">
      <select value={op} onChange={(e) => setOp(e.target.value as FilterOp)} autoFocus={info.args === 0}
        className={inputCls} aria-label="Condition">
        {CONDITIONS[fk].map((o) => <option key={o.op} value={o.op}>{o.label}</option>)}
      </select>
      {(info.args === 1 || info.args === 2 || info.args === 'count') && (
        <div className="flex items-center gap-1.5">
          <input
            type={info.args === 'count' ? 'text' : type}
            inputMode={numeric ? (info.args === 'count' ? 'numeric' : 'decimal') : undefined}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={info.args === 'count' ? 'How many, e.g. 10' : fk === 'number' ? 'A number' : 'Value'}
            autoFocus
            className={`${inputCls} ${fk === 'date' ? 'dark:[color-scheme:dark]' : ''} ${numeric ? 'tabular-nums' : ''}`}
            aria-label={info.args === 'count' ? 'How many' : 'Value'}
          />
          {info.args === 2 && (
            <>
              <span className="shrink-0 text-[11px] text-zinc-500">and</span>
              <input type={type} value={value2} onChange={(e) => setValue2(e.target.value)} placeholder="To" className={inputCls} aria-label="To" />
            </>
          )}
        </div>
      )}
      {(op === 'dupes' || op === 'unique') && (
        <p className="text-[11px] leading-snug text-zinc-500">Across the whole column, ignoring case and surrounding spaces — handy for finding doubles.</p>
      )}
      {(op === 'top' || op === 'bottom' || op === 'above_avg' || op === 'below_avg') && (
        <p className="text-[11px] leading-snug text-zinc-500">Measured across the whole column.</p>
      )}
    </div>
  )
}

// ── Yes / no ──

function BooleanPick({ current, onPick }: { current?: FilterOp; onPick: (op: FilterOp) => void }) {
  const opts: [FilterOp, string][] = [['yes', 'Yes'], ['no', 'No'], ['empty', 'Empty']]
  return (
    <div className="flex gap-1.5 pb-2.5" role="radiogroup" aria-label="Keep the rows that are">
      {opts.map(([o, text]) => (
        <button key={o} type="button" role="radio" aria-checked={current === o} onClick={() => onPick(o)}
          className={`flex-1 rounded-lg border px-3 py-2 text-[13px] font-medium transition-colors ${current === o
            ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
            : 'border-zinc-700 text-zinc-300 hover:bg-zinc-800'}`}>
          {text}
        </button>
      ))}
    </div>
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
            title={f.values && f.values.length > 2 ? f.values.map((v) => v || '(empty)').join(', ') : 'Change this filter'}
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
