// One datastore record, one field per reading block. Left / Right step to the
// previous / next record (newest first, the rows screen's order) in place, so
// Back still goes to the rows screen — on the page holding this record.
//
// The route holds the record's position, but positions move as the agent adds
// or deletes rows. So the screen follows the record's _id (left in dataCache
// by the rows screen or the step that led here): a refresh finds it again by
// _id, a step goes to the next older / newer _id, and a note says when the
// table changed meanwhile. Steps read whole pages (shared with the rows
// screen), and swipes made while a record loads are queued, not dropped.

import { useEffect, useEffectEvent, useRef, useState } from 'react'
import type { HudAgent, HudTable } from '../api'
import { focusEl } from '../focus'
import { useAutoFocus } from '../hooks'
import { navigate, navigateUp, routeKey, type Route } from '../router'
import { ScreenFrame, StateView } from '../ui'
import {
  cellType, columnTypes, EMPTY, findCachedTable, formatValue, getCachedNeighbour, getCachedRecord, getLastRecord,
  idOf, isEmptyValue, isFresh, isStructured, loadNeighbour, loadRecord, loadTables, memberCreator, PAGE, plural,
  prefetchNear, recordFields, ROWS_TTL_MS, rowId, setLastRecord, setPageOffset, useFocusRecovery, type RecordEntry,
} from './dataCache'
import './data.css'

const CHANGED = 'The table changed while you browsed.'
const DELETED = 'This record has been deleted.'
const STEP_FAILED = "Couldn't load the next record."

interface RecordState {
  /** The record's _id (null: a deep link until it loads, or a row without one). */
  id: number | null
  entry: RecordEntry | null
  error: string | null
  /** The record must be (re)loaded — the loader runs while true. */
  pending: boolean
  /** One line in the head block: deleted, the table changed, a step failed. */
  note: string | null
  /** A step is loading the record it leads to. */
  stepping: boolean
}

/** The note the next record screen opens with (set by the step leading there). */
let handoff: { key: string; note: string } | null = null

function message(e: unknown): string {
  return e instanceof Error && e.message ? e.message : 'Could not load this record.'
}

export function RecordScreen({ agent, table, index }: { agent: HudAgent; table: string; index: number }) {
  const [st, setSt] = useState<RecordState>(() => {
    const last = getLastRecord(agent.id, table)
    const id = last && last.index === index ? last.id : null
    const entry = getCachedRecord(agent.id, table, index, id)
    const key = routeKey({ v: 'record', a: agent.id, tb: table, i: index })
    const note = handoff?.key === key ? handoff.note : null
    return { id, entry, error: null, pending: !isFresh(entry, ROWS_TTL_MS), note, stepping: false }
  })
  const [meta, setMeta] = useState<HudTable | null>(() => findCachedTable(agent.id, table))
  const scrollRef = useRef<HTMLDivElement>(null)
  /** Swipes not carried out yet: + older (Right), − newer (Left). */
  const queued = useRef(0)
  /** walk() is running. */
  const busy = useRef(false)
  const mounted = useRef(true)

  useEffect(() => {
    handoff = null
    mounted.current = true
    return () => { mounted.current = false }
  }, [])

  // Back lands on the page holding this record, focused on its row.
  const at = st.entry?.row ? st.entry.index : index
  useEffect(() => {
    setPageOffset(agent.id, table, at)
    setLastRecord(agent.id, table, at, st.id)
  }, [agent.id, table, at, st.id])

  // (Re)load: by _id when known, so rows added or deleted meanwhile never
  // swap the record on screen for another one.
  const seenAt = st.entry?.index ?? index
  const seenTotal = st.entry?.total ?? 0
  useEffect(() => {
    if (!st.pending) return
    let live = true
    loadRecord(agent, table, seenAt, st.id, seenTotal).then(
      (e) => {
        if (!live) return
        setSt((s) => {
          if (e.row) {
            const moved = !!s.entry?.row && (e.index !== s.entry.index || e.total !== s.entry.total)
            return { ...s, id: idOf(e.row), entry: e, error: null, pending: false, note: moved ? CHANGED : s.note }
          }
          // Deleted while on screen: keep the copy (steps still work from it).
          if (s.entry?.row) return { ...s, error: null, pending: false, note: DELETED }
          return { ...s, entry: e, error: null, pending: false }
        })
      },
      (err: unknown) => {
        if (!live) return
        queued.current = 0
        setSt((s) => ({ ...s, error: message(err), pending: false }))
      },
    )
    return () => { live = false }
  }, [agent, table, seenAt, seenTotal, st.id, st.pending])

  // Column types for formatting (deep link: nothing cached yet).
  useEffect(() => {
    if (meta) return
    let alive = true
    loadTables(agent).then(
      (tables) => { if (alive) setMeta(tables.find((t) => t.name === table) ?? null) },
      () => { /* fall back to JS types */ },
    )
    return () => { alive = false }
  }, [agent, table, meta])

  // On the first / last row of a page, load the page beyond it now.
  useEffect(() => {
    if (!st.pending && st.entry?.row) prefetchNear(agent, table, st.entry.index, st.entry.total)
  }, [agent, table, st.pending, st.entry])

  const retry = () => {
    if (st.pending) return
    setSt((s) => ({ ...s, error: null, pending: true }))
  }

  /** Show `n`, reached by stepping (`changed`: the table changed on the way). */
  const open = (n: RecordEntry, changed: boolean) => {
    const nid = idOf(n.row)
    setLastRecord(agent.id, table, n.index, nid)
    if (n.index === index) {
      // Rows moved, and it now sits at this screen's own position: show it here.
      setSt({ id: nid, entry: n, error: null, pending: !isFresh(n, ROWS_TTL_MS), note: changed ? CHANGED : null, stepping: false })
      const head = scrollRef.current?.querySelector<HTMLElement>('.hud-data-head')
      if (head) focusEl(head)
      return
    }
    const r: Route = { v: 'record', a: agent.id, tb: table, i: n.index }
    handoff = changed ? { key: routeKey(r), note: CHANGED } : null
    navigate(r, { replace: true })
  }

  /** Carry out the queued swipes from `start`, one record at a time (from
   *  cached pages at once, else loaded), then show where they lead. Swipes
   *  made meanwhile join the queue. */
  const walk = async (start: RecordEntry) => {
    busy.current = true
    let cur = start
    let changed = false
    let failed = false
    try {
      while (queued.current !== 0) {
        const dir = queued.current > 0 ? 1 : -1
        let next = getCachedNeighbour(agent.id, table, cur, dir)
        if (next === null) {
          setSt((s) => (s.stepping ? s : { ...s, stepping: true }))
          next = await loadNeighbour(agent, table, cur, dir)
          if (!mounted.current) return
        }
        if (next === 'none' || next === null) break // the first / last record
        queued.current -= dir
        if (next.index !== cur.index + dir || next.total !== cur.total) changed = true
        cur = next
      }
    } catch {
      failed = true
    } finally {
      busy.current = false
      queued.current = 0
    }
    if (!mounted.current) return
    if (cur !== start) open(cur, changed)
    else setSt((s) => ({ ...s, stepping: false, note: failed ? STEP_FAILED : s.note }))
  }

  const step = (dir: 1 | -1) => {
    queued.current = Math.max(-PAGE, Math.min(PAGE, queued.current + dir))
    if (!busy.current && st.entry?.row) void walk(st.entry)
  }

  // Swipes made while the record loaded: carry them out now that it's here.
  const drain = useEffectEvent(() => {
    if (busy.current || queued.current === 0) return
    if (st.entry?.row) void walk(st.entry)
    else if (st.entry || st.error) queued.current = 0 // nothing to step from
  })
  useEffect(() => { drain() }, [st.entry, st.error])

  // Left / Right step — also while the record is still loading (nothing has
  // focus then), so a quick double swipe is never lost.
  const onArrow = useEffectEvent((e: globalThis.KeyboardEvent) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    if (e.altKey || e.ctrlKey || e.metaKey || e.defaultPrevented) return
    if (st.entry ? !st.entry.row : st.error !== null) return // gone / failed: arrows move focus
    e.preventDefault()
    step(e.key === 'ArrowRight' ? 1 : -1)
  })
  useEffect(() => {
    const h = (e: globalThis.KeyboardEvent) => onArrow(e)
    document.addEventListener('keydown', h, true)
    return () => document.removeEventListener('keydown', h, true)
  }, [])

  const entry = st.entry
  const total = entry?.total ?? 0
  const pos = (entry?.row ? entry.index : index) + 1

  const ready = entry !== null || st.error !== null
  useAutoFocus(ready)
  useFocusRecovery(scrollRef, ready)

  let body
  if (!entry) {
    body = st.error
      ? <StateView kind="error" message={st.error} onRetry={retry} />
      : <StateView kind="loading" message="Loading record…" />
  } else if (!entry.row) {
    body = (
      <StateView
        kind="empty"
        message={total === 0
          ? 'This table is empty.'
          : `${st.id !== null ? `Record #${st.id} has been deleted` : `Record ${pos.toLocaleString()} is gone`} — the table now has ${plural(total, 'record')}.`}
        action={{ label: 'Back to list', onActivate: () => navigateUp({ v: 'rows', a: agent.id, tb: table }) }}
      />
    )
  } else {
    const row = entry.row
    const types = columnTypes(meta)
    const creator = memberCreator(row)
    const fields = recordFields(meta, entry.columns, row)
    body = (
      <div className="hud-data-record">
        <div className="hud-block hud-data-head" tabIndex={0}>
          <div className="hud-data-id">#{rowId(row)}</div>
          <div className="hud-data-pos">Record {pos.toLocaleString()} of {total.toLocaleString()}</div>
          {creator ? <div className="hud-data-creator">Added by {creator}</div> : null}
          {total > 1 ? <div className="hud-data-hint">{st.stepping ? 'Loading…' : '◂ ▸ previous / next record'}</div> : null}
          {st.note ? <div className="hud-data-stale">{st.note}</div> : null}
          {st.error ? <div className="hud-data-stale">Couldn't refresh</div> : null}
        </div>
        {fields.length === 0 ? <div className="hud-note">No fields.</div> : null}
        {fields.map((name) => {
          const v = row[name]
          const type = cellType(types, name, v)
          const empty = isEmptyValue(v)
          const cls = empty ? ' hud-data-val--empty' : isStructured(v, type) ? ' hud-data-val--code' : ''
          return (
            <div key={name} className="hud-block hud-data-field" tabIndex={0}>
              <div className="hud-data-key">{name}</div>
              <div className={`hud-data-val${cls}`}>{empty ? EMPTY : formatValue(v, type)}</div>
            </div>
          )
        })}
      </div>
    )
  }

  const subtitle = entry?.row
    ? `${agent.name} · ${pos.toLocaleString()} of ${total.toLocaleString()}`
    : entry ? agent.name : `${agent.name} · ${pos.toLocaleString()} of …`
  return (
    <ScreenFrame title={table} subtitle={subtitle} scrollRef={scrollRef}>
      {body}
    </ScreenFrame>
  )
}
