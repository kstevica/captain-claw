// Datastore tab: in-memory caches and value formatting (read-only browsing).
//
// Caches (module memory, dropped when the signed-in user changes):
//  - the table list per agent: the tables screen renders it at once on Back,
//    and the rows / record screens take column types and row counts from it
//    (listTables runs one COUNT(*) per table on the agent — never poll it);
//  - a few pages of rows and single records, so Back from a record and
//    stepping through records on the page just seen cost nothing over the
//    glasses' ~500 Kbps link;
//  - the page offset per agent+table and the last record viewed, so Back from
//    a record lands on the page (and the row) that holds it.
//
// Formatting turns raw cells into short plain strings by column type
// (text | integer | real | boolean 0/1 | date | datetime | json). Everything is
// rendered as React text by the screens — cells can hold untrusted member data.

import { useAuthStore } from '../../stores/authStore'
import { listTables, queryRows, type HudAgent, type HudColumn, type HudRows, type HudTable } from '../api'

/** Rows per page on the rows screen (newest first). */
export const PAGE = 8

/** A table list younger than this is shown without refetching. */
export const TABLES_TTL_MS = 30_000
/** A page / record younger than this is shown without refetching. */
export const ROWS_TTL_MS = 30_000
/** Pages / records older than this are not shown at all (indices drift as rows are added). */
const ROWS_MAX_AGE_MS = 5 * 60_000

const MAX_PAGES = 12
const MAX_RECORDS = 24

/** Bumped by clearDataCache so a request started before it never writes back. */
let generation = 0

export function isFresh(entry: { at: number } | null | undefined, ttl: number, now = Date.now()): boolean {
  return !!entry && now - entry.at < ttl
}

function tableKey(agentId: string, table: string): string {
  return `${agentId}\u0000${table}`
}

// ── Table lists ──

export interface TablesEntry {
  tables: HudTable[]
  /** Epoch ms of the load. */
  at: number
}

const tableLists = new Map<string, TablesEntry>()
const tablesInflight = new Map<string, Promise<HudTable[]>>()

function updatedMs(t: HudTable): number | null {
  if (!t.updatedAt) return null
  const ms = Date.parse(t.updatedAt)
  return Number.isFinite(ms) ? ms : null
}

/** Updated most recently first (unknown last), then by name. */
function sortTables(tables: HudTable[]): HudTable[] {
  return tables
    .map((t) => ({ t, ms: updatedMs(t) }))
    .sort((a, b) => {
      if (a.ms !== b.ms) {
        if (a.ms === null) return 1
        if (b.ms === null) return -1
        return b.ms - a.ms
      }
      return a.t.name.localeCompare(b.t.name)
    })
    .map((x) => x.t)
}

export function getCachedTables(agentId: string): TablesEntry | null {
  return tableLists.get(agentId) ?? null
}

export function findCachedTable(agentId: string, name: string): HudTable | null {
  return tableLists.get(agentId)?.tables.find((t) => t.name === name) ?? null
}

/** Fetch the agent's tables (sorted) and cache them — single flight per agent. */
export function loadTables(agent: HudAgent): Promise<HudTable[]> {
  const running = tablesInflight.get(agent.id)
  if (running) return running
  const gen = generation
  const p = listTables(agent)
    .then((raw) => {
      const tables = sortTables(raw)
      if (gen === generation) tableLists.set(agent.id, { tables, at: Date.now() })
      return tables
    })
    .finally(() => {
      if (tablesInflight.get(agent.id) === p) tablesInflight.delete(agent.id)
    })
  tablesInflight.set(agent.id, p)
  return p
}

// ── Pages of rows (newest first) ──

export interface PageEntry {
  data: HudRows
  at: number
}

/** Insertion order doubles as LRU order. */
const pages = new Map<string, PageEntry>()
const pageInflight = new Map<string, Promise<HudRows>>()

function pageKey(agentId: string, table: string, offset: number): string {
  return `${tableKey(agentId, table)}\u0000${offset}`
}

function remember<V>(map: Map<string, V>, key: string, value: V, max: number): void {
  map.delete(key)
  map.set(key, value)
  while (map.size > max) {
    const oldest = map.keys().next()
    if (oldest.done) break
    map.delete(oldest.value)
  }
}

/** A cached page (possibly stale — check isFresh), or null. */
export function getCachedPage(agentId: string, table: string, offset: number): PageEntry | null {
  const e = pages.get(pageKey(agentId, table, offset))
  return e && Date.now() - e.at < ROWS_MAX_AGE_MS ? e : null
}

/** One page of PAGE rows at `offset`, newest first (single flight per page). */
export function loadPage(agent: HudAgent, table: string, offset: number): Promise<HudRows> {
  const key = pageKey(agent.id, table, offset)
  const running = pageInflight.get(key)
  if (running) return running
  const gen = generation
  const p = queryRows(agent, table, { limit: PAGE, offset, orderBy: '_id', orderDir: 'desc' })
    .then((data) => {
      if (gen === generation) remember(pages, key, { data, at: Date.now() }, MAX_PAGES)
      return data
    })
    .finally(() => {
      if (pageInflight.get(key) === p) pageInflight.delete(key)
    })
  pageInflight.set(key, p)
  return p
}

// ── Single records ──

export type Cells = Record<string, unknown>

export interface RecordEntry {
  /** null: no record at this index (rows were deleted). */
  row: Cells | null
  total: number
  /** Column names in the agent's order (`_id` first). */
  columns: string[]
  at: number
}

const records = new Map<string, RecordEntry>()
const recordInflight = new Map<string, Promise<RecordEntry>>()

function recordKey(agentId: string, table: string, index: number): string {
  return `${tableKey(agentId, table)}\u0000#${index}`
}

/** The record at `index` from a cached page or an earlier single fetch — the
 *  newer of the two — or null. Possibly stale (check isFresh). */
export function getCachedRecord(agentId: string, table: string, index: number): RecordEntry | null {
  const offset = Math.floor(index / PAGE) * PAGE
  const page = getCachedPage(agentId, table, offset)
  let fromPage: RecordEntry | null = null
  if (page) {
    const row = page.data.rows[index - offset]
    if (row) fromPage = { row, total: page.data.total, columns: page.data.columns, at: page.at }
  }
  let single = records.get(recordKey(agentId, table, index)) ?? null
  if (single && Date.now() - single.at >= ROWS_MAX_AGE_MS) single = null
  if (fromPage && single) return single.at > fromPage.at ? single : fromPage
  return fromPage ?? single
}

/** Fetch the one record at `index` (newest first) and cache it. */
export function loadRecord(agent: HudAgent, table: string, index: number): Promise<RecordEntry> {
  const key = recordKey(agent.id, table, index)
  const running = recordInflight.get(key)
  if (running) return running
  const gen = generation
  const p = queryRows(agent, table, { limit: 1, offset: index, orderBy: '_id', orderDir: 'desc' })
    .then((data) => {
      const entry: RecordEntry = { row: data.rows[0] ?? null, total: data.total, columns: data.columns, at: Date.now() }
      if (gen === generation) remember(records, key, entry, MAX_RECORDS)
      return entry
    })
    .finally(() => {
      if (recordInflight.get(key) === p) recordInflight.delete(key)
    })
  recordInflight.set(key, p)
  return p
}

// ── Where the wearer was (page offset + last record, per agent+table) ──

const offsets = new Map<string, number>()
const lastRecords = new Map<string, number>()

export function getPageOffset(agentId: string, table: string): number {
  return offsets.get(tableKey(agentId, table)) ?? 0
}

export function setPageOffset(agentId: string, table: string, offset: number): void {
  const o = Math.max(0, Math.floor(offset / PAGE) * PAGE)
  offsets.set(tableKey(agentId, table), o)
}

/** Index of the record last opened in this table (rows screen focuses its row). */
export function getLastRecord(agentId: string, table: string): number | null {
  return lastRecords.get(tableKey(agentId, table)) ?? null
}

export function setLastRecord(agentId: string, table: string, index: number): void {
  lastRecords.set(tableKey(agentId, table), index)
}

export function clearDataCache(): void {
  generation++
  tableLists.clear()
  tablesInflight.clear()
  pages.clear()
  pageInflight.clear()
  records.clear()
  recordInflight.clear()
  offsets.clear()
  lastRecords.clear()
}

useAuthStore.subscribe((s, prev) => {
  if (s.user?.id !== prev.user?.id || (prev.isAuthenticated && !s.isAuthenticated)) clearDataCache()
})

// ── Columns ──

/** Never shown as fields: the key and the creator bookkeeping. */
export const SKIP_FIELDS: ReadonlySet<string> = new Set(['_id', '_creator', '_created_by', '_created_by_name'])

export const EMPTY = '—'

const TITLE_RE = /^(name|title|subject|label|headline|summary|topic|question)$/i
const SHORT_TYPES = new Set(['integer', 'real', 'date', 'datetime', 'boolean'])
/** Text this short (one line) can stand in a row's meta line. */
const SHORT_TEXT = 40

export function isEmptyValue(v: unknown): boolean {
  return v === null || v === undefined || (typeof v === 'string' && v.trim() === '')
}

/** Column type from a JS value when the table metadata doesn't say. */
function typeOfValue(v: unknown): string {
  if (typeof v === 'boolean') return 'boolean'
  if (typeof v === 'number') return Number.isInteger(v) ? 'integer' : 'real'
  if (v !== null && typeof v === 'object') return 'json'
  return 'text'
}

/** Column name → declared type (empty map without metadata). */
export function columnTypes(table: HudTable | null): Map<string, string> {
  const m = new Map<string, string>()
  for (const c of table?.columns ?? []) m.set(c.name, (c.type || '').toLowerCase())
  return m
}

/** The declared type of a cell, else one inferred from its JS value. */
export function cellType(types: Map<string, string>, name: string, value: unknown): string {
  return types.get(name) || typeOfValue(value)
}

/**
 * The user columns of a page in display order, with types: the table's own
 * columns (by position) first, then anything else the rows carry. Types
 * missing from the metadata are inferred from the first non-empty cell.
 */
export function pageColumns(table: HudTable | null, data: HudRows): HudColumn[] {
  const out: HudColumn[] = []
  const seen = new Set<string>()
  const infer = (name: string): string => {
    for (const r of data.rows) {
      if (!isEmptyValue(r[name])) return typeOfValue(r[name])
    }
    return 'text'
  }
  const add = (name: string, type?: string) => {
    if (SKIP_FIELDS.has(name) || seen.has(name)) return
    seen.add(name)
    out.push({ name, type: (type || '').toLowerCase() || infer(name) })
  }
  const present = new Set(data.columns)
  for (const c of table?.columns ?? []) {
    if (present.size === 0 || present.has(c.name)) add(c.name, c.type)
  }
  for (const n of data.columns) add(n)
  return out
}

/** The column a row is named by: a text column called name/title/…, else
 *  the first text column, else none (rows are then named '#<_id>'). */
export function pickTitleColumn(cols: HudColumn[]): string | null {
  const text = cols.filter((c) => c.type === 'text')
  return (text.find((c) => TITLE_RE.test(c.name)) ?? text[0])?.name ?? null
}

function isShortText(v: unknown): boolean {
  if (isEmptyValue(v)) return true
  return typeof v === 'string' && v.length <= SHORT_TEXT && !v.includes('\n')
}

/** Up to two columns for a row's meta line: short typed columns first
 *  (integer, real, date, datetime, boolean), then text that is short on every
 *  row of the page. Never json or long text; columns empty on the whole page
 *  are skipped. */
export function pickMetaColumns(cols: HudColumn[], rows: Cells[], titleCol: string | null): string[] {
  const rest = cols.filter((c) => c.name !== titleCol)
  const hasValue = (c: HudColumn) => rows.some((r) => !isEmptyValue(r[c.name]))
  const typed = rest.filter((c) => SHORT_TYPES.has(c.type) && hasValue(c))
  const shortText = rest.filter((c) => c.type === 'text' && hasValue(c) && rows.every((r) => isShortText(r[c.name])))
  return [...typed, ...shortText].slice(0, 2).map((c) => c.name)
}

export function rowId(row: Cells): string {
  const id = row._id
  return typeof id === 'number' || typeof id === 'string' ? String(id) : '?'
}

/** A row's one-line title, or null when the title cell is empty. */
export function rowTitle(row: Cells, titleCol: string | null): string | null {
  if (!titleCol) return null
  const v = row[titleCol]
  if (isEmptyValue(v)) return null
  return oneLine(typeof v === 'string' ? v : formatValue(v, typeOfValue(v)), 120)
}

/** 'col: value · col: value' for the meta columns that have a value. */
export function rowMeta(row: Cells, metaCols: string[], types: Map<string, string>, now: number): string {
  const parts: string[] = []
  for (const name of metaCols) {
    const v = row[name]
    if (isEmptyValue(v)) continue
    parts.push(`${name}: ${formatShort(v, cellType(types, name, v), now)}`)
  }
  return parts.join(' · ')
}

/** Field names of a record in display order (table order, then the rest). */
export function recordFields(table: HudTable | null, columns: string[], row: Cells): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  const add = (n: string) => {
    if (SKIP_FIELDS.has(n) || seen.has(n)) return
    seen.add(n)
    out.push(n)
  }
  for (const c of table?.columns ?? []) {
    if (Object.prototype.hasOwnProperty.call(row, c.name)) add(c.name)
  }
  for (const n of columns) add(n)
  for (const n of Object.keys(row)) add(n)
  return out
}

/** "Added by …" name when a workspace member (not the owner, not me) created the row. */
export function memberCreator(row: Cells): string | null {
  const c = row._creator
  if (!c || typeof c !== 'object') return null
  const o = c as { kind?: unknown; name?: unknown }
  if (o.kind !== 'member') return null
  return typeof o.name === 'string' && o.name.trim() ? o.name.trim() : 'a member'
}

// ── Values ──

const JSON_MAX = 2000
/** Longer text is clipped (a 600 px display, a slow CPU, D-pad reading). */
const TEXT_MAX = 10_000
/** Don't JSON.parse giant strings on the glasses' CPU. */
const JSON_PARSE_MAX = 200_000

function oneLine(s: string, max: number): string {
  const t = s.replace(/\s+/g, ' ').trim()
  return t.length > max ? t.slice(0, max - 1).trimEnd() + '…' : t
}

function clip(s: string, max: number): string {
  if (s.length <= max) return s
  const more = s.length - max
  return `${s.slice(0, max).trimEnd()}…\n(${more.toLocaleString()} more character${more === 1 ? '' : 's'})`
}

function formatBool(v: unknown): string {
  if (v === true || v === 1 || v === '1' || v === 'true') return 'Yes'
  if (v === false || v === 0 || v === '0' || v === 'false') return 'No'
  return String(v)
}

/** Reals with at most 6 decimals (tiny / huge ones with 6 significant digits). */
export function formatReal(v: unknown): string {
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN
  if (!Number.isFinite(n)) return String(v)
  if (Number.isInteger(n)) return String(n)
  const abs = Math.abs(n)
  if (abs >= 1e-4 && abs < 1e15) return String(parseFloat(n.toFixed(6)))
  return String(parseFloat(n.toPrecision(6)))
}

const DATE_ONLY = /^(\d{4})-(\d{2})-(\d{2})$/
const DATE_TIME = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}/

/** ISO date / datetime → Date (a bare date is local midnight, not UTC). */
function parseDateish(v: unknown): { d: Date; time: boolean } | null {
  if (typeof v !== 'string') return null
  const s = v.trim()
  const m = DATE_ONLY.exec(s)
  if (m) {
    const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]))
    return Number.isNaN(d.getTime()) ? null : { d, time: false }
  }
  if (!DATE_TIME.test(s)) return null
  const d = new Date(s.replace(' ', 'T'))
  return Number.isNaN(d.getTime()) ? null : { d, time: true }
}

const formatters = new Map<string, Intl.DateTimeFormat>()

function dtf(key: string, opts: Intl.DateTimeFormatOptions): Intl.DateTimeFormat {
  let f = formatters.get(key)
  if (!f) {
    f = new Intl.DateTimeFormat(undefined, opts)
    formatters.set(key, f)
  }
  return f
}

function formatDate(v: unknown, short: boolean, now: number): string {
  const p = parseDateish(v)
  if (!p) return String(v)
  const sameYear = short && p.d.getFullYear() === new Date(now).getFullYear()
  if (!p.time) {
    return sameYear
      ? dtf('d-md', { month: 'short', day: 'numeric' }).format(p.d)
      : dtf('d-ymd', { year: 'numeric', month: 'short', day: 'numeric' }).format(p.d)
  }
  return sameYear
    ? dtf('t-md', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(p.d)
    : dtf('t-ymd', { year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(p.d)
}

function formatJson(v: unknown): string {
  let parsed = v
  if (typeof v === 'string') {
    if (v.length > JSON_PARSE_MAX) return clip(v, JSON_MAX)
    try { parsed = JSON.parse(v) } catch { return clip(v, JSON_MAX) }
  }
  let s: string
  try { s = JSON.stringify(parsed, null, 2) ?? String(parsed) } catch { s = String(v) }
  return clip(s, JSON_MAX)
}

/** A cell as shown on the record screen (multi-line allowed). */
export function formatValue(v: unknown, type: string, now = 0): string {
  if (isEmptyValue(v)) return EMPTY
  switch (type) {
    case 'boolean': return formatBool(v)
    case 'date':
    case 'datetime': return formatDate(v, false, now)
    case 'real': return formatReal(v)
    case 'integer': return typeof v === 'object' ? formatJson(v) : String(v)
    case 'json': return formatJson(v)
  }
  if (typeof v === 'boolean') return formatBool(v)
  if (typeof v === 'number') return formatReal(v)
  if (typeof v === 'object') return formatJson(v)
  return clip(String(v), TEXT_MAX)
}

/** A cell squeezed onto a row's meta line. */
export function formatShort(v: unknown, type: string, now: number): string {
  if (isEmptyValue(v)) return EMPTY
  if (type === 'date' || type === 'datetime') return formatDate(v, true, now)
  if (type === 'json' || (v !== null && typeof v === 'object')) return oneLine(formatJson(v), SHORT_TEXT)
  return oneLine(formatValue(v, type, now), SHORT_TEXT)
}

/** Value blocks that hold structured text get a monospace face. */
export function isStructured(v: unknown, type: string): boolean {
  return type === 'json' || (v !== null && typeof v === 'object')
}

/** '1 row' / '12 rows'. */
export function plural(n: number, word: string): string {
  return `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`
}
