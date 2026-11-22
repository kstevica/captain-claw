import { useState } from 'react'
import type { ColumnKind } from '../../services/dataLibraryApi'

// ── A filter on each column ──
//
// The table views (the Data library's rows, a CSV or Excel file as a table)
// take a filter on any column, and as many columns as you like, every filter
// narrowing the rows further (all of them must hold). A filter keeps values
// picked from the column's own ("is one of" / "isn't one of"), a range of
// numbers or days (either end open), or a condition: contains, > 10, before a
// day, the top or bottom N, above or below the average, values that repeat
// or don't, empty or not. A column's funnel in its header opens its filter;
// what is set shows as chips above the rows (ColumnFilterUI.tsx). Here: what
// a filter is, how a cell is matched, what a column holds for the picker,
// and the state a table view keeps. The Data library's rows are filtered by
// Flight Deck the same way (data_library._filter_where).

export type FilterKind = 'text' | 'number' | 'date' | 'boolean'

export type FilterOp =
  | 'contains' | 'not_contains' | 'eq' | 'ne' | 'starts' | 'ends'
  | 'gt' | 'gte' | 'lt' | 'lte' | 'between' | 'range'
  | 'in' | 'not_in' | 'top' | 'bottom' | 'above_avg' | 'below_avg' | 'dupes' | 'unique'
  | 'empty' | 'not_empty' | 'yes' | 'no'

export interface ColumnFilter {
  /** The column: its name (Data library) or its place (a file's table). */
  col: string
  op: FilterOp
  /** The value; a range's low end; how many for top / bottom. */
  value?: string
  /** A range's high end. */
  value2?: string
  /** "is one of" / "isn't one of": the picked values ('' is an empty cell). */
  values?: string[]
}

/** What an operator takes: nothing, a value, two, or picked values. */
type Args = 0 | 1 | 2 | 'count' | 'values'
interface OpInfo { op: FilterOp; label: string; args: Args }

const COMMON_END: OpInfo[] = [
  { op: 'dupes', label: 'appears more than once', args: 0 },
  { op: 'unique', label: 'appears only once', args: 0 },
  { op: 'not_empty', label: 'has a value', args: 0 },
  { op: 'empty', label: 'is empty', args: 0 },
]

/** The conditions each kind of column offers (the Condition tab). */
export const CONDITIONS: Record<FilterKind, OpInfo[]> = {
  text: [
    { op: 'contains', label: 'contains', args: 1 },
    { op: 'not_contains', label: "doesn't contain", args: 1 },
    { op: 'eq', label: 'is', args: 1 },
    { op: 'ne', label: "isn't", args: 1 },
    { op: 'starts', label: 'starts with', args: 1 },
    { op: 'ends', label: 'ends with', args: 1 },
    ...COMMON_END,
  ],
  number: [
    { op: 'eq', label: '=', args: 1 },
    { op: 'ne', label: '≠', args: 1 },
    { op: 'gt', label: '>', args: 1 },
    { op: 'gte', label: '≥', args: 1 },
    { op: 'lt', label: '<', args: 1 },
    { op: 'lte', label: '≤', args: 1 },
    { op: 'top', label: 'top', args: 'count' },
    { op: 'bottom', label: 'bottom', args: 'count' },
    { op: 'above_avg', label: 'above average', args: 0 },
    { op: 'below_avg', label: 'below average', args: 0 },
    ...COMMON_END,
  ],
  date: [
    { op: 'eq', label: 'on', args: 1 },
    { op: 'lt', label: 'before', args: 1 },
    { op: 'gt', label: 'after', args: 1 },
    ...COMMON_END,
  ],
  boolean: [
    { op: 'yes', label: 'is yes', args: 0 },
    { op: 'no', label: 'is no', args: 0 },
    { op: 'empty', label: 'is empty', args: 0 },
  ],
}

const OTHER_OPS: OpInfo[] = [
  { op: 'in', label: 'is one of', args: 'values' },
  { op: 'not_in', label: "isn't one of", args: 'values' },
  { op: 'range', label: 'range', args: 2 },
  { op: 'between', label: 'between', args: 2 },
]

export function opInfo(fk: FilterKind, op: FilterOp): OpInfo {
  return CONDITIONS[fk].find((o) => o.op === op) ?? OTHER_OPS.find((o) => o.op === op) ?? CONDITIONS[fk][0]
}

/** How a column is filtered, from what it holds (and its stored type). */
export function filterKindOf(kind: ColumnKind | undefined, type = ''): FilterKind {
  if (type === 'boolean' || kind === 'boolean') return 'boolean'
  if (kind === 'number' || type === 'integer' || type === 'real') return 'number'
  if (kind === 'date' || type === 'date' || type === 'datetime') return 'date'
  return 'text'
}

const DAY_RE = /^\d{4}-\d{2}-\d{2}$/

/** A number as a person may have written it: "1 234,5", "12%", "-3.2". */
export function toNumber(s: string): number {
  const t = s.trim().replace(/\s/g, '').replace(/%$/, '')
  if (!t) return NaN
  // "1,234.5" → thousands commas; "1234,5" → a decimal comma.
  const n = /^-?\d{1,3}(,\d{3})+(\.\d+)?$/.test(t) ? t.replace(/,/g, '') : t.replace(',', '.')
  return /^-?\d*\.?\d+(e-?\d+)?$/i.test(n) ? Number(n) : NaN
}

/** A filter is whole: it has what its operator needs (a number for a
 *  number, a day for a date, a count for top / bottom, picked values). */
export function filterReady(f: ColumnFilter, fk: FilterKind): boolean {
  const { args } = opInfo(fk, f.op)
  const ok = (v?: string) => {
    const t = (v ?? '').trim()
    if (!t) return false
    if (fk === 'number') return !Number.isNaN(toNumber(t))
    if (fk === 'date') return DAY_RE.test(t)
    return true
  }
  if (args === 'values') return !!f.values?.length
  if (args === 'count') return /^\d+$/.test((f.value ?? '').trim()) && Number(f.value) >= 1
  if (f.op === 'range') {
    const a = (f.value ?? '').trim()
    const b = (f.value2 ?? '').trim()
    return (!!a || !!b) && (!a || ok(a)) && (!b || ok(b))
  }
  return (args === 0 || ok(f.value)) && (args !== 2 || ok(f.value2))
}

const shown = (v: string) => (v === '' ? '(empty)' : v)

/** "contains “north”", "is Norway, Sweden +2", "10 – 50", "top 10", "is empty". */
export function describeFilter(f: ColumnFilter, fk: FilterKind): string {
  const { label, args } = opInfo(fk, f.op)
  const a = (f.value ?? '').trim()
  const b = (f.value2 ?? '').trim()
  if (args === 'values') {
    const vs = f.values ?? []
    const head = vs.slice(0, 2).map(shown).join(', ')
    const more = vs.length > 2 ? ` +${vs.length - 2}` : ''
    return `${f.op === 'in' ? 'is' : "isn't"} ${head}${more}`
  }
  if (f.op === 'range') {
    if (a && b) return `${a} – ${b}`
    if (fk === 'date') return a ? `from ${a}` : `until ${b}`
    return a ? `≥ ${a}` : `≤ ${b}`
  }
  if (args === 'count') return `${label} ${a}`
  const v = (s: string) => (fk === 'text' ? `“${s}”` : s)
  if (args === 0) return label
  if (args === 2) return `${label} ${v(a)} and ${v(b)}`
  return `${label} ${v(a)}`
}

// ── Matching, for tables held in the page (a file's) ──

const YES = new Set(['1', 'true', 'yes', 'y'])
const NO = new Set(['0', 'false', 'no', 'n'])

const cellText = (raw: unknown) => (raw === null || raw === undefined ? '' : String(raw).trim())

/** A test for one cell. `column` is every cell of the filter's column (the
 *  whole table): top / bottom, the average and repeats are about all of it. */
export function makeMatcher(f: ColumnFilter, fk: FilterKind, column: unknown[]): (raw: unknown) => boolean {
  const a = (f.value ?? '').trim()
  const b = (f.value2 ?? '').trim()
  switch (f.op) {
    case 'empty': return (raw) => cellText(raw) === ''
    case 'not_empty': return (raw) => cellText(raw) !== ''
    case 'in':
    case 'not_in': {
      const picked = new Set((f.values ?? []).map((v) => v.trim().toLowerCase()))
      const keep = f.op === 'in'
      return (raw) => picked.has(cellText(raw).toLowerCase()) === keep
    }
    case 'dupes':
    case 'unique': {
      const counts = new Map<string, number>()
      for (const c of column) {
        const k = cellText(c).toLowerCase()
        if (k) counts.set(k, (counts.get(k) ?? 0) + 1)
      }
      const many = f.op === 'dupes'
      return (raw) => {
        const k = cellText(raw).toLowerCase()
        return !!k && ((counts.get(k) ?? 0) > 1) === many
      }
    }
    case 'top':
    case 'bottom': {
      const nums = column.map((c) => toNumber(cellText(c))).filter((n) => !Number.isNaN(n))
      nums.sort((x, y) => (f.op === 'top' ? y - x : x - y))
      const k = Math.max(1, Number(a) || 1)
      const edge = nums.length ? nums[Math.min(k, nums.length) - 1] : NaN
      return (raw) => {
        const n = toNumber(cellText(raw))
        return !Number.isNaN(n) && (f.op === 'top' ? n >= edge : n <= edge)
      }
    }
    case 'above_avg':
    case 'below_avg': {
      const nums = column.map((c) => toNumber(cellText(c))).filter((n) => !Number.isNaN(n))
      const avg = nums.reduce((s, n) => s + n, 0) / (nums.length || 1)
      return (raw) => {
        const n = toNumber(cellText(raw))
        return !Number.isNaN(n) && nums.length > 0 && (f.op === 'above_avg' ? n > avg : n < avg)
      }
    }
    default: return (raw) => matchesFilter(f, raw, fk, a, b)
  }
}

function matchesFilter(f: ColumnFilter, raw: unknown, fk: FilterKind, a: string, b: string): boolean {
  const s = cellText(raw)
  if (fk === 'boolean') {
    const low = s.toLowerCase()
    return f.op === 'yes' ? YES.has(low) : f.op === 'no' ? NO.has(low) : true
  }
  // "isn't" and "doesn't contain" keep the empty cells, as a spreadsheet's do.
  if (s === '') return f.op === 'ne' || f.op === 'not_contains'
  if (fk === 'number') {
    const n = toNumber(s)
    const x = toNumber(a)
    if (f.op === 'ne') return Number.isNaN(n) || n !== x
    if (Number.isNaN(n)) return false
    switch (f.op) {
      case 'eq': return n === x
      case 'gt': return n > x
      case 'gte': return n >= x
      case 'lt': return n < x
      case 'lte': return n <= x
      case 'range': {
        let lo = a ? x : -Infinity
        let hi = b ? toNumber(b) : Infinity
        if (lo > hi) [lo, hi] = [hi, lo]
        return n >= lo && n <= hi
      }
      case 'between': { const y = toNumber(b); return n >= Math.min(x, y) && n <= Math.max(x, y) }
      default: return true
    }
  }
  if (fk === 'date') {
    const day = s.slice(0, 10)
    switch (f.op) {
      case 'eq': return day === a
      case 'lt': return day < a
      case 'gt': return day > a
      case 'range': {
        const [lo, hi] = a && b && a > b ? [b, a] : [a, b]
        return (!lo || day >= lo) && (!hi || day <= hi)
      }
      case 'between': { const [lo, hi] = a <= b ? [a, b] : [b, a]; return day >= lo && day <= hi }
      default: return true
    }
  }
  const low = s.toLowerCase()
  const q = a.toLowerCase()
  switch (f.op) {
    case 'contains': return low.includes(q)
    case 'not_contains': return !low.includes(q)
    case 'eq': return low === q
    case 'ne': return low !== q
    case 'starts': return low.startsWith(q)
    case 'ends': return low.endsWith(q)
    default: return true
  }
}

// ── What a column holds, for its filter to pick from ──

export interface ValueStats {
  /** Each value once (the most common first), with how many rows hold it. */
  values: { value: string; count: number }[]
  /** How many distinct values (beyond the ones listed, too). */
  distinct: number
  /** How many rows leave it empty. */
  empty: number
  /** How many rows were looked at. */
  total: number
  /** A number or date column's smallest and largest. */
  min: number | string | null
  max: number | string | null
}

/** A column's values among rows held in the page, the way the Data library's
 *  values route counts them: without case or surrounding space, the most
 *  common first, `search` narrowing them. */
export function localValueStats(cells: unknown[], fk: FilterKind, search = '', limit = 200): ValueStats {
  const groups = new Map<string, { value: string; count: number }>()
  let empty = 0
  let min: number | string | null = null
  let max: number | string | null = null
  for (const c of cells) {
    const s = cellText(c)
    if (!s) { empty++; continue }
    const k = s.toLowerCase()
    const g = groups.get(k)
    if (g) g.count++
    else groups.set(k, { value: s, count: 1 })
    if (fk === 'number') {
      const n = toNumber(s)
      if (!Number.isNaN(n)) {
        if (min === null || n < (min as number)) min = n
        if (max === null || n > (max as number)) max = n
      }
    } else if (fk === 'date' && DAY_RE.test(s.slice(0, 10))) {
      const d = s.slice(0, 10)
      if (min === null || d < (min as string)) min = d
      if (max === null || d > (max as string)) max = d
    }
  }
  const q = search.trim().toLowerCase()
  const all = [...groups.entries()].filter(([k]) => !q || k.includes(q)).map(([, g]) => g)
  all.sort((x, y) => y.count - x.count || x.value.localeCompare(y.value, undefined, { numeric: true }))
  return { values: all.slice(0, limit), distinct: all.length, empty, total: cells.length, min, max }
}

// ── Days for a date range ──

const iso = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`

/** Ready-made ranges of days, from today. */
export function datePresets(today = new Date()): { label: string; from: string; to: string }[] {
  const y = today.getFullYear()
  const m = today.getMonth()
  const ago = (n: number) => { const d = new Date(today); d.setDate(d.getDate() - n); return iso(d) }
  return [
    { label: 'Today', from: iso(today), to: iso(today) },
    { label: 'Last 7 days', from: ago(6), to: iso(today) },
    { label: 'Last 30 days', from: ago(29), to: iso(today) },
    { label: 'This month', from: iso(new Date(y, m, 1)), to: iso(new Date(y, m + 1, 0)) },
    { label: 'Last month', from: iso(new Date(y, m - 1, 1)), to: iso(new Date(y, m, 0)) },
    { label: 'This year', from: `${y}-01-01`, to: `${y}-12-31` },
    { label: 'Last year', from: `${y - 1}-01-01`, to: `${y - 1}-12-31` },
  ]
}

/** The column filters a table view keeps: set, change, take off, and which
 *  column's filter is open (and where). */
export function useColumnFilters() {
  const [filters, setFilters] = useState<ColumnFilter[]>([])
  const [menu, setMenu] = useState<{ col: string; anchor: DOMRect } | null>(null)
  const apply = (col: string, f: Omit<ColumnFilter, 'col'> | null) => {
    setFilters((cur) => {
      if (!f) return cur.filter((x) => x.col !== col)
      // A column's filter keeps its place among the chips when it changes.
      return cur.some((x) => x.col === col) ? cur.map((x) => (x.col === col ? { col, ...f } : x)) : [...cur, { col, ...f }]
    })
    setMenu(null)
  }
  const remove = (col: string) => setFilters((cur) => cur.filter((x) => x.col !== col))
  const clear = () => { setFilters([]); setMenu(null) }
  const of = (col: string) => filters.find((x) => x.col === col)
  return { filters, menu, setMenu, apply, remove, clear, of, setFilters }
}
