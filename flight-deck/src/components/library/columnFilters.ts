import { useState } from 'react'
import type { ColumnKind } from '../../services/dataLibraryApi'

// ── A filter on each column ──
//
// The table views (the Data library's rows, a CSV or Excel file as a table)
// take a filter on any column — "contains", "> 10", "before a date", "is
// empty" — and as many columns as you like, every filter narrowing the rows
// further (all of them must hold). A column's funnel in its header opens its
// filter; what is set shows as chips above the rows, each one a click away
// from being changed or taken off (ColumnFilterUI.tsx). Here: what a filter
// is, how a cell is matched, and the state a table view keeps.

export type FilterKind = 'text' | 'number' | 'date' | 'boolean'

export type FilterOp =
  | 'contains' | 'not_contains' | 'eq' | 'ne' | 'starts'
  | 'gt' | 'gte' | 'lt' | 'lte' | 'between'
  | 'empty' | 'not_empty' | 'yes' | 'no'

export interface ColumnFilter {
  /** The column: its name (Data library) or its place (a file's table). */
  col: string
  op: FilterOp
  value?: string
  value2?: string
}

interface OpInfo { op: FilterOp; label: string; args: 0 | 1 | 2 }

export const OPS: Record<FilterKind, OpInfo[]> = {
  text: [
    { op: 'contains', label: 'contains', args: 1 },
    { op: 'not_contains', label: "doesn't contain", args: 1 },
    { op: 'eq', label: 'is', args: 1 },
    { op: 'ne', label: "isn't", args: 1 },
    { op: 'starts', label: 'starts with', args: 1 },
    { op: 'not_empty', label: 'has a value', args: 0 },
    { op: 'empty', label: 'is empty', args: 0 },
  ],
  number: [
    { op: 'eq', label: '=', args: 1 },
    { op: 'ne', label: '≠', args: 1 },
    { op: 'gt', label: '>', args: 1 },
    { op: 'gte', label: '≥', args: 1 },
    { op: 'lt', label: '<', args: 1 },
    { op: 'lte', label: '≤', args: 1 },
    { op: 'between', label: 'between', args: 2 },
    { op: 'not_empty', label: 'has a value', args: 0 },
    { op: 'empty', label: 'is empty', args: 0 },
  ],
  date: [
    { op: 'eq', label: 'on', args: 1 },
    { op: 'lt', label: 'before', args: 1 },
    { op: 'gt', label: 'after', args: 1 },
    { op: 'between', label: 'between', args: 2 },
    { op: 'not_empty', label: 'has a value', args: 0 },
    { op: 'empty', label: 'is empty', args: 0 },
  ],
  boolean: [
    { op: 'yes', label: 'is yes', args: 0 },
    { op: 'no', label: 'is no', args: 0 },
    { op: 'empty', label: 'is empty', args: 0 },
  ],
}

/** How a column is filtered, from what it holds (and its stored type). */
export function filterKindOf(kind: ColumnKind | undefined, type = ''): FilterKind {
  if (type === 'boolean' || kind === 'boolean') return 'boolean'
  if (kind === 'number' || type === 'integer' || type === 'real') return 'number'
  if (kind === 'date' || type === 'date' || type === 'datetime') return 'date'
  return 'text'
}

export const opInfo = (fk: FilterKind, op: FilterOp) => OPS[fk].find((o) => o.op === op) ?? OPS[fk][0]

const DAY_RE = /^\d{4}-\d{2}-\d{2}$/

/** A filter is whole: it has the values its operator needs (a number for a
 *  number, a day for a date). */
export function filterReady(f: ColumnFilter, fk: FilterKind): boolean {
  const { args } = opInfo(fk, f.op)
  const ok = (v?: string) => {
    const t = (v ?? '').trim()
    if (!t) return false
    if (fk === 'number') return !Number.isNaN(toNumber(t))
    if (fk === 'date') return DAY_RE.test(t)
    return true
  }
  return (args < 1 || ok(f.value)) && (args < 2 || ok(f.value2))
}

/** "contains “north”", "between 1 and 5", "is empty". */
export function describeFilter(f: ColumnFilter, fk: FilterKind): string {
  const { label, args } = opInfo(fk, f.op)
  const v = (s = '') => (fk === 'text' ? `“${s}”` : s)
  if (args === 0) return label
  if (args === 2) return `${label} ${v(f.value)} and ${v(f.value2)}`
  return `${label} ${v(f.value)}`
}

// ── Matching, for tables held in the page (a file's) ──

const YES = new Set(['1', 'true', 'yes', 'y'])
const NO = new Set(['0', 'false', 'no', 'n'])

/** A number as a person may have written it: "1 234,5", "12%", "-3.2". */
function toNumber(s: string): number {
  const t = s.trim().replace(/\s/g, '').replace(/%$/, '')
  if (!t) return NaN
  // "1,234.5" → thousands commas; "1234,5" → a decimal comma.
  const n = /^-?\d{1,3}(,\d{3})+(\.\d+)?$/.test(t) ? t.replace(/,/g, '') : t.replace(',', '.')
  return /^-?\d*\.?\d+(e-?\d+)?$/i.test(n) ? Number(n) : NaN
}

/** Whether one cell passes the filter. Text compares without case; a date
 *  by its day (YYYY-MM-DD). */
export function matchesFilter(f: ColumnFilter, raw: unknown, fk: FilterKind): boolean {
  const s = raw === null || raw === undefined ? '' : String(raw).trim()
  if (f.op === 'empty') return s === ''
  if (f.op === 'not_empty') return s !== ''
  if (fk === 'boolean') {
    const low = s.toLowerCase()
    return f.op === 'yes' ? YES.has(low) : f.op === 'no' ? NO.has(low) : true
  }
  // "isn't" and "doesn't contain" keep the empty cells, as a spreadsheet's do.
  if (s === '') return f.op === 'ne' || f.op === 'not_contains'
  const a = (f.value ?? '').trim()
  const b = (f.value2 ?? '').trim()
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
    default: return true
  }
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
