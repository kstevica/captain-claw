// Datastore tab: in-memory caches and value formatting (read-only browsing).
//
// Caches (module memory, dropped when the signed-in user changes):
//  - the table list per agent: the tables screen renders it at once on Back,
//    and the rows / record screens take column types and row counts from it
//    (listTables runs one COUNT(*) per table on the agent — never poll it);
//  - a few pages of rows, so Back from a record and stepping through records
//    cost nothing over the glasses' ~500 Kbps link. Records are read from
//    these pages too (a step past one loads the next page, not one row);
//  - the page offset per agent+table and the last record viewed, so Back from
//    a record lands on the page (and the row) that holds it.
//
// Rows are paged by position (newest first), and positions move whenever the
// agent adds or deletes rows. So a page whose row count differs from a newer
// read is dropped, and records are followed by _id: a refresh finds the record
// again by _id, and a step goes to the next older / newer _id, not to
// "position ± 1" (rankOf).
//
// Formatting turns raw cells into short plain strings by column type
// (text | integer | real | boolean 0/1 | date | datetime | json). Everything is
// rendered as React text by the screens — cells can hold untrusted member data.
// useFocusRecovery (at the end) is the screens' shared focus safety net.

import { useEffect, useLayoutEffect, useRef } from 'react'
import type { RefObject } from 'react'
import { useAuthStore } from '../../stores/authStore'
import { listTables, queryRows, type HudAgent, type HudColumn, type HudRows, type HudTable } from '../api'
import { focusInitial } from '../focus'

/** Rows per page on the rows screen (newest first). */
export const PAGE = 8

/** A table list younger than this is shown without refetching. */
export const TABLES_TTL_MS = 30_000
/** A page / record younger than this is shown without refetching. */
export const ROWS_TTL_MS = 30_000
/** Pages older than this are not shown at all (they're refetched instead). */
const ROWS_MAX_AGE_MS = 5 * 60_000

const MAX_PAGES = 12

/** Bumped by clearDataCache so a request started before it never writes back. */
let generation = 0

export function isFresh(entry: { at: number } | null | undefined, ttl: number, now = Date.now()): boolean {
  return !!entry && now - entry.at < ttl
}

function tableKey(agentId: string, table: string): string {
  return `${agentId}\u0000${table}`
}

/** Offset of the page that holds row `index`. */
export function pageStart(index: number): number {
  return Math.max(0, Math.floor(index / PAGE) * PAGE)
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

/** Fetch the agent's tables (sorted) and cache them — single flight per agent.
 *  A table whose row count or update time changed loses its cached pages. */
export function loadTables(agent: HudAgent): Promise<HudTable[]> {
  const running = tablesInflight.get(agent.id)
  if (running) return running
  const gen = generation
  const p = listTables(agent)
    .then((raw) => {
      const tables = sortTables(raw)
      if (gen === generation) {
        const prev = tableLists.get(agent.id)
        for (const t of tables) {
          const was = prev?.tables.find((x) => x.name === t.name)
          const read = totals.get(tableKey(agent.id, t.name))
          if ((was && was.updatedAt !== t.updatedAt) || (read !== undefined && read !== t.rowCount)) {
            dropPages(agent.id, t.name)
          }
        }
        tableLists.set(agent.id, { tables, at: Date.now() })
      }
      return tables
    })
    .finally(() => {
      if (tablesInflight.get(agent.id) === p) tablesInflight.delete(agent.id)
    })
  tablesInflight.set(agent.id, p)
  return p
}

/** A page just told the table's row count: keep the cached list in step, so
 *  the tables screen and the rows screen agree. */
function syncRowCount(agentId: string, table: string, total: number): void {
  const list = tableLists.get(agentId)
  if (!list?.tables.some((t) => t.name === table && t.rowCount !== total)) return
  tableLists.set(agentId, { ...list, tables: list.tables.map((t) => (t.name === table ? { ...t, rowCount: total } : t)) })
}

// ── Pages of rows (newest first) ──

export interface PageEntry {
  data: HudRows
  at: number
}

/** Insertion order doubles as LRU order. */
const pages = new Map<string, PageEntry>()
const pageInflight = new Map<string, Promise<HudRows>>()
/** The row count each table's cached pages were read under. */
const totals = new Map<string, number>()

function pageKey(agentId: string, table: string, offset: number): string {
  return `${tableKey(agentId, table)}\u0000${offset}`
}

/** Forget a table's cached pages: rows were added or deleted since, so the
 *  positions in them no longer hold. */
function dropPages(agentId: string, table: string): void {
  const prefix = `${tableKey(agentId, table)}\u0000`
  for (const k of Array.from(pages.keys())) {
    if (k.startsWith(prefix)) pages.delete(k)
  }
  totals.delete(tableKey(agentId, table))
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
      if (gen === generation) {
        const tk = tableKey(agent.id, table)
        const read = totals.get(tk)
        if (read !== undefined && read !== data.total) dropPages(agent.id, table)
        totals.set(tk, data.total)
        syncRowCount(agent.id, table, data.total)
        // A page without rows is kept only for an empty table. Past the end (a
        // deep link, rows deleted) the rows screen moves on to the last page;
        // and no rows where the count says there are some is a read that raced
        // a write (the agent runs SELECT and COUNT(*) separately).
        if (data.rows.length > 0 || (offset === 0 && data.total === 0)) remember(pages, key, { data, at: Date.now() }, MAX_PAGES)
      }
      return data
    })
    .finally(() => {
      if (pageInflight.get(key) === p) pageInflight.delete(key)
    })
  pageInflight.set(key, p)
  return p
}

/** A page that is cached and fresh, else loaded. */
async function freshPage(agent: HudAgent, table: string, offset: number): Promise<PageEntry> {
  const cached = getCachedPage(agent.id, table, offset)
  if (isFresh(cached, ROWS_TTL_MS)) return cached!
  const data = await loadPage(agent, table, offset)
  return { data, at: Date.now() }
}

// ── Records ──

export type Cells = Record<string, unknown>

export interface RecordEntry {
  /** null: no such record (deleted, or past the end). */
  row: Cells | null
  /** Its position, newest first. */
  index: number
  total: number
  /** Column names in the agent's order (`_id` first). */
  columns: string[]
  at: number
}

/** A row's _id as a number, or null. */
export function idOf(row: Cells | null | undefined): number | null {
  const v = row?._id
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN
  return Number.isFinite(n) ? n : null
}

function entryAt(page: PageEntry, offset: number, index: number): RecordEntry {
  const { data } = page
  return { row: data.rows[index - offset] ?? null, index, total: data.total, columns: data.columns, at: page.at }
}

/**
 * From cached pages: the record at `index` — or, given `id`, the record with
 * that _id wherever it now sits (the page holding `index` is tried first).
 * Null when not cached. Possibly stale (check isFresh).
 */
export function getCachedRecord(agentId: string, table: string, index: number, id: number | null = null): RecordEntry | null {
  const home = pageStart(index)
  const page = getCachedPage(agentId, table, home)
  if (id === null) {
    const e = page ? entryAt(page, home, index) : null
    return e?.row ? e : null
  }
  const find = (p: PageEntry, offset: number) => {
    const k = p.data.rows.findIndex((r) => idOf(r) === id)
    return k >= 0 ? entryAt(p, offset, offset + k) : null
  }
  if (page) {
    const e = find(page, home)
    if (e) return e
  }
  const prefix = `${tableKey(agentId, table)}\u0000`
  for (const [k, p] of pages) {
    if (!k.startsWith(prefix) || Date.now() - p.at >= ROWS_MAX_AGE_MS) continue
    const offset = Number(k.slice(prefix.length))
    if (offset === home) continue
    const e = find(p, offset)
    if (e) return e
  }
  return null
}

/**
 * From cached pages: the record next to `from` — dir 1 the next older one
 * (Right), -1 the next newer one (Left). 'none' when there is none that way;
 * null when the cache can't tell (loadNeighbour).
 */
export function getCachedNeighbour(agentId: string, table: string, from: RecordEntry, dir: 1 | -1): RecordEntry | 'none' | null {
  const id = idOf(from.row)
  const at = id === null ? from : getCachedRecord(agentId, table, from.index, id)
  if (!at) return null
  const t = at.index + dir
  // Newer rows may have come since that page was read; never older ones.
  if (t < 0) return isFresh(at, ROWS_TTL_MS) ? 'none' : null
  if (t >= at.total) return 'none'
  const next = getCachedRecord(agentId, table, t)
  if (!next?.row || id === null) return next
  const nid = idOf(next.row)
  return nid !== null && (dir > 0 ? nid < id : nid > id) ? next : null
}

const CHANGING = 'The table is changing — try again.'
const LOCATE_PAGES = 6

/**
 * Where _id `id` sits now, newest first: the number of rows with a larger
 * _id — its index, or where it would be if it was deleted. Starts at the page
 * holding `guess` (`total0` rows then) and jumps by the change in the row
 * count, then narrows page by page. `reload`: read that first page afresh.
 */
async function rankOf(
  agent: HudAgent, table: string, id: number, guess: number, total0: number, reload: boolean,
): Promise<{ rank: number; total: number }> {
  let lo = 0
  let hi = Number.POSITIVE_INFINITY
  let at = Math.max(0, guess)
  for (let n = 0; n < LOCATE_PAGES; n++) {
    const offset = pageStart(at)
    const data = n === 0 && reload ? await loadPage(agent, table, offset) : (await freshPage(agent, table, offset)).data
    const rows = data.rows
    let k = 0 // rows of this page newer than `id` (they come first)
    while (k < rows.length && (idOf(rows[k]) ?? Number.NEGATIVE_INFINITY) > id) k++
    if (k > 0 && k < rows.length) return { rank: offset + k, total: data.total }
    hi = Math.min(hi, data.total)
    if (rows.length > 0 && k === rows.length) lo = Math.max(lo, offset + k) // all newer: it is further on
    else hi = Math.min(hi, offset) // none newer, or past the end: further back
    if (lo === hi) return { rank: lo, total: data.total }
    if (lo > hi) break // read across a change
    at = Math.min(Math.max(guess + (total0 > 0 ? data.total - total0 : 0), lo), hi - 1)
  }
  throw new Error(CHANGING)
}

async function rowAt(agent: HudAgent, table: string, index: number): Promise<RecordEntry> {
  const offset = pageStart(index)
  return entryAt(await freshPage(agent, table, offset), offset, index)
}

/**
 * Load the record with _id `id`, last seen at `index` when the table had
 * `total` rows (row null: it was deleted). Without an id (a deep link), the
 * record at `index`. Reads whole pages, shared with the rows screen.
 */
export async function loadRecord(agent: HudAgent, table: string, index: number, id: number | null, total: number): Promise<RecordEntry> {
  if (id === null) {
    const offset = pageStart(index)
    const data = await loadPage(agent, table, offset)
    return entryAt({ data, at: Date.now() }, offset, index)
  }
  const found = await rankOf(agent, table, id, index, total, true)
  if (found.rank >= found.total) return { row: null, index: found.rank, total: found.total, columns: [], at: Date.now() }
  const e = await rowAt(agent, table, found.rank)
  return idOf(e.row) === id ? e : { ...e, row: null }
}

/** The record next to `from` (see getCachedNeighbour), loaded; null when
 *  there is none that way. */
export async function loadNeighbour(agent: HudAgent, table: string, from: RecordEntry, dir: 1 | -1): Promise<RecordEntry | null> {
  const id = idOf(from.row)
  if (id === null) { // nothing to go by but the position
    if (from.index + dir < 0) return null
    const e = await rowAt(agent, table, from.index + dir)
    return e.row ? e : null
  }
  const { rank, total } = await rankOf(agent, table, id, from.index + dir, from.total, false)
  // `from` itself, if still there, sits at `rank`.
  let t = dir > 0 ? rank : rank - 1
  if (t < 0 || t >= total) return null
  let e = await rowAt(agent, table, t)
  if (dir > 0 && idOf(e.row) === id) {
    t++
    if (t >= e.total) return null
    e = await rowAt(agent, table, t)
  }
  if (!e.row) return null
  const nid = idOf(e.row)
  if (nid === null || (dir > 0 ? nid >= id : nid <= id)) throw new Error(CHANGING)
  return e
}

/** Warm the page across the edge `index` sits on, so the step over it is instant. */
export function prefetchNear(agent: HudAgent, table: string, index: number, total: number): void {
  const t = index % PAGE === PAGE - 1 ? index + 1 : index % PAGE === 0 ? index - 1 : -1
  if (t < 0 || t >= total || isFresh(getCachedPage(agent.id, table, pageStart(t)), ROWS_TTL_MS)) return
  loadPage(agent, table, pageStart(t)).catch(() => { /* the step loads it */ })
}

// ── Where the wearer was (page offset + last record, per agent+table) ──

export interface LastRecord {
  index: number
  /** Its _id: the record screen goes by it (positions move). */
  id: number | null
}

const offsets = new Map<string, number>()
const lastRecords = new Map<string, LastRecord>()

export function getPageOffset(agentId: string, table: string): number {
  return offsets.get(tableKey(agentId, table)) ?? 0
}

export function setPageOffset(agentId: string, table: string, offset: number): void {
  offsets.set(tableKey(agentId, table), pageStart(offset))
}

/** The record last opened in this table (the rows screen focuses its row). */
export function getLastRecord(agentId: string, table: string): LastRecord | null {
  return lastRecords.get(tableKey(agentId, table)) ?? null
}

export function setLastRecord(agentId: string, table: string, index: number, id: number | null): void {
  lastRecords.set(tableKey(agentId, table), { index, id })
}

/** A table opened afresh (from the tables list) starts at its newest rows;
 *  only Back from a record returns to the remembered page and row. */
export function resetTablePosition(agentId: string, table: string): void {
  offsets.delete(tableKey(agentId, table))
  lastRecords.delete(tableKey(agentId, table))
}

export function clearDataCache(): void {
  generation++
  tableLists.clear()
  tablesInflight.clear()
  pages.clear()
  pageInflight.clear()
  totals.clear()
  offsets.clear()
  lastRecords.clear()
}

useAuthStore.subscribe((s, prev) => {
  if (s.user?.id !== prev.user?.id || (prev.isAuthenticated && !s.isAuthenticated)) clearDataCache()
})

// ── Focus ──

/**
 * A refresh or page change removed the focused row / button while nothing
 * else could take focus, so it fell to <body> and a pinch would hit nothing.
 * Once the screen has content again, focus its landing row / button or its
 * first stop — before paint. Only when this screen's focused element was
 * removed: never before the first placement (useAutoFocus, or the row Back
 * restores), and never against a plain blur.
 */
export function useFocusRecovery(scrollRef: RefObject<HTMLElement | null>, ready: boolean): void {
  const last = useRef<Node | null>(null)
  useEffect(() => {
    const onIn = (e: FocusEvent) => {
      const t = e.target as Node
      last.current = scrollRef.current?.contains(t) ? t : null
    }
    document.addEventListener('focusin', onIn)
    return () => document.removeEventListener('focusin', onIn)
  }, [scrollRef])
  useLayoutEffect(() => {
    if (!ready || !last.current || last.current.isConnected) return
    const a = document.activeElement
    if (a && a !== document.body) return
    focusInitial({ force: true })
  })
}

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

const DATE_PART = /^(\d{4})-(\d{2})-(\d{2})(?=$|[T ])/
const DATE_TIME = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}/

/** Local midnight of a YYYY-MM-DD match; null for a day that doesn't exist
 *  (02-30, 13-01 — `new Date` would roll them over). */
function calendarDay(m: RegExpExecArray): Date | null {
  const y = Number(m[1]), mo = Number(m[2]) - 1, day = Number(m[3])
  const d = new Date(0)
  d.setFullYear(y, mo, day) // not new Date(y, …): years below 100 stay as written
  d.setHours(0, 0, 0, 0)
  return d.getFullYear() === y && d.getMonth() === mo && d.getDate() === day ? d : null
}

/**
 * ISO date / datetime → Date. A bare date — and anything in a `date` column,
 * where agents often write '2026-10-10T00:00:00Z' — is a calendar day: its
 * date part, never moved by a time zone and shown without a time. Other
 * datetimes are instants (an explicit offset is converted to local time).
 * Null (shown raw) for anything else, impossible days included.
 */
function parseDateish(v: unknown, dateCol: boolean): { d: Date; time: boolean } | null {
  if (typeof v !== 'string') return null
  const s = v.trim()
  const m = DATE_PART.exec(s)
  if (!m || (s.length > 10 && !DATE_TIME.test(s))) return null
  const day = calendarDay(m)
  if (!day) return null
  if (dateCol || s.length === 10) return { d: day, time: false }
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

function formatDate(v: unknown, dateCol: boolean, short: boolean, now: number): string {
  const p = parseDateish(v, dateCol)
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
    case 'datetime': return formatDate(v, type === 'date', false, now)
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
  if (type === 'date' || type === 'datetime') return formatDate(v, type === 'date', true, now)
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
