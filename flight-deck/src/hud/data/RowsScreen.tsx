// Rows of one datastore table, PAGE at a time, newest first. A pinch opens a
// record; Left / Right (or the ‹ Prev / Next › chips) change page; Refresh
// re-reads the page (the agent may be writing to the table).
//
// The page offset lives in dataCache, so Back from a record — even one the
// wearer stepped to across a page boundary — lands on the page that holds it,
// with focus on that record's row. Opening the table from the tables list
// starts at the newest page instead (TablesScreen resets the position).

import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import type { HudAgent, HudRows, HudTable } from '../api'
import { focusEl, focusables } from '../focus'
import { useAutoFocus, useNow } from '../hooks'
import { navigate, takeRestoreFocusKey } from '../router'
import { Btn, Row, ScreenFrame, StateView } from '../ui'
import {
  columnTypes, findCachedTable, getCachedPage, getLastRecord, getPageOffset, idOf, isFresh, loadPage, loadTables,
  PAGE, pageColumns, pageStart, pickMetaColumns, pickTitleColumn, plural, ROWS_TTL_MS, rowId, rowMeta, rowTitle,
  setLastRecord, setPageOffset, useFocusRecovery,
} from './dataCache'
import './data.css'

interface PageState {
  /** Offset of the page asked for. */
  offset: number
  /** The page on screen — still the previous one while `offset` loads. */
  shown: { offset: number; data: HudRows } | null
  error: string | null
  /** The page at `offset` must be (re)loaded — the loader runs while true. */
  pending: boolean
  /** Re-reads of a page that came back empty although the count said it has rows. */
  attempt: number
}

function message(e: unknown): string {
  return e instanceof Error && e.message ? e.message : 'Could not load rows.'
}

function initialState(agentId: string, table: string): PageState {
  const offset = getPageOffset(agentId, table)
  const cached = getCachedPage(agentId, table, offset)
  return {
    offset,
    shown: cached ? { offset, data: cached.data } : null,
    error: null,
    pending: !isFresh(cached, ROWS_TTL_MS),
    attempt: 0,
  }
}

export function RowsScreen({ agent, table }: { agent: HudAgent; table: string }) {
  const [st, setSt] = useState<PageState>(() => initialState(agent.id, table))
  const [meta, setMeta] = useState<HudTable | null>(() => findCachedTable(agent.id, table))
  const [landing] = useState(() => getLastRecord(agent.id, table))
  const scrollRef = useRef<HTMLDivElement>(null)
  /** Set when the wearer changes page: focus the new page's first row. */
  const focusFirst = useRef(false)
  const now = useNow(60_000)

  // Load the requested page while one is pending (a fresh cached copy is
  // shown without it). Responses for a page the wearer already left are ignored.
  useEffect(() => {
    if (!st.pending) return
    const offset = st.offset
    const attempt = st.attempt
    let alive = true
    loadPage(agent, table, offset).then(
      (data) => {
        if (!alive) return
        if (data.rows.length === 0 && data.total > 0) {
          const last = pageStart(data.total - 1)
          if (last < offset) {
            // Rows were deleted since this page was remembered: show the last page.
            setPageOffset(agent.id, table, last)
            setSt((s) => (s.offset === offset ? { ...s, offset: last, pending: true, attempt: 0 } : s))
            return
          }
          if (attempt === 0) {
            // The count raced the read (rows were written in between): read once more.
            setSt((s) => (s.offset === offset ? { ...s, attempt: 1 } : s))
            return
          }
        }
        setSt((s) => (s.offset === offset ? { ...s, shown: { offset, data }, error: null, pending: false } : s))
      },
      (e: unknown) => {
        if (!alive) return
        setSt((s) => (s.offset === offset ? { ...s, error: message(e), pending: false } : s))
      },
    )
    return () => { alive = false }
  }, [agent, table, st.offset, st.pending, st.attempt])

  // Column types (title / meta choice, value formatting) come from the table
  // list; fetch it once if the wearer deep-linked here. Without it, types are
  // inferred from the values.
  useEffect(() => {
    if (meta) return
    let alive = true
    loadTables(agent).then(
      (tables) => { if (alive) setMeta(tables.find((t) => t.name === table) ?? null) },
      () => { /* inferred types are good enough */ },
    )
    return () => { alive = false }
  }, [agent, table, meta])

  const goPage = (next: number) => {
    const total = st.shown?.data.total ?? 0
    if (next < 0 || next === st.offset || next >= total) return
    setPageOffset(agent.id, table, next)
    focusFirst.current = true
    const cached = getCachedPage(agent.id, table, next)
    setSt((s) => ({
      ...s,
      offset: next,
      shown: cached ? { offset: next, data: cached.data } : s.shown,
      error: null,
      pending: !isFresh(cached, ROWS_TTL_MS),
      attempt: 0,
    }))
  }

  const retry = () => {
    if (st.pending) return
    setSt((s) => ({ ...s, error: null, pending: true, attempt: 0 }))
  }

  const onListKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    if (e.altKey || e.ctrlKey || e.metaKey) return
    e.preventDefault()
    goPage(st.offset + (e.key === 'ArrowRight' ? PAGE : -PAGE))
  }

  const shown = st.shown
  const showError = st.error !== null && (!shown || shown.offset !== st.offset)
  const ready = showError || shown !== null

  // Back from a record: its row (data-autofocus below) beats the row the
  // record was first opened from, which the router would otherwise restore.
  // Found by _id: rows may have moved since.
  const landingRow = shown && landing
    ? shown.data.rows.findIndex((r, i) => (landing.id !== null ? idOf(r) === landing.id : shown.offset + i === landing.index))
    : -1
  useEffect(() => {
    if (ready && landingRow >= 0) takeRestoreFocusKey()
  }, [ready, landingRow])
  useAutoFocus(ready)

  // After a page change, put focus on the new page's first row (or Retry) as
  // soon as that page is on screen, even while it refreshes — before paint, so
  // the ring never blinks off while the old rows unmount.
  useLayoutEffect(() => {
    if (!focusFirst.current || (st.pending && st.shown?.offset !== st.offset)) return
    focusFirst.current = false
    const first = scrollRef.current ? focusables(scrollRef.current)[0] : null
    if (first) focusEl(first)
  }, [st.pending, st.shown, st.error, st.offset])
  useFocusRecovery(scrollRef, ready)

  let body
  if (showError) {
    body = <StateView kind="error" message={st.error ?? undefined} onRetry={retry} />
  } else if (!shown) {
    body = <StateView kind="loading" message="Loading rows…" />
  } else if (shown.data.rows.length === 0) {
    body = (
      <StateView kind="empty"
        message={st.pending ? 'Checking for rows…' : shown.data.total > 0 ? 'The table changed while loading.' : 'This table is empty.'}
        action={{ label: 'Refresh', onActivate: retry }} />
    )
  } else {
    const { data } = shown
    const total = Math.max(data.total, shown.offset + data.rows.length)
    const cols = pageColumns(meta, data)
    const types = columnTypes(meta)
    const titleCol = pickTitleColumn(cols)
    const metaCols = pickMetaColumns(cols, data.rows, titleCol)
    const end = Math.min(st.offset + PAGE, total)
    body = (
      <>
        <div className="hud-note hud-data-note">
          {plural(total, 'record')} · newest first
          {total > PAGE ? ' · ◂ ▸ pages' : ''}
          {st.error ? <span className="hud-data-stale"> · couldn't refresh</span> : null}
        </div>
        <div className={`hud-data-list${st.pending ? ' hud-data-list--pending' : ''}`} onKeyDown={onListKey}>
          {data.rows.map((row, i) => {
            const index = shown.offset + i
            const id = rowId(row)
            const title = rowTitle(row, titleCol)
            // Keyed by _id, so a row that survives a refresh keeps its node (and focus).
            const key = idOf(row) !== null ? id : `@${index}`
            return (
              <Row
                key={key}
                fk={`row-${key}`}
                title={title ?? `#${id}`}
                meta={rowMeta(row, metaCols, types, now) || undefined}
                badge={title !== null ? `#${id}` : undefined}
                autoFocus={i === landingRow}
                onActivate={() => {
                  setLastRecord(agent.id, table, index, idOf(row))
                  navigate({ v: 'record', a: agent.id, tb: table, i: index })
                }}
              />
            )
          })}
        </div>
        {total > PAGE ? (
          <div className="hud-data-pager">
            <Btn variant="chip" fk="page-prev" disabled={st.offset <= 0} onActivate={() => goPage(st.offset - PAGE)}>
              ‹ Prev
            </Btn>
            <div className="hud-data-range" aria-live="polite">
              {st.pending ? 'Loading…' : `${st.offset + 1}–${end} of ${total.toLocaleString()}`}
            </div>
            <Btn variant="chip" fk="page-next" disabled={st.offset + PAGE >= total} onActivate={() => goPage(st.offset + PAGE)}>
              Next ›
            </Btn>
          </div>
        ) : null}
        <div className="hud-actions">
          <Btn variant="chip" fk="rows-refresh" onActivate={retry}>
            {st.pending ? 'Refreshing…' : 'Refresh'}
          </Btn>
        </div>
      </>
    )
  }

  return (
    <ScreenFrame title={table} subtitle={agent.name} scrollRef={scrollRef}>
      {body}
    </ScreenFrame>
  )
}
