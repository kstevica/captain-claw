// One datastore record, one field per reading block. Left / Right step to the
// previous / next record (newest first, the rows screen's order) in place, so
// Back still goes to the rows screen — on the page holding this record.

import { useEffect, useState } from 'react'
import type { KeyboardEvent } from 'react'
import type { HudAgent, HudTable } from '../api'
import { useAutoFocus } from '../hooks'
import { navigate } from '../router'
import { ScreenFrame, StateView } from '../ui'
import {
  cellType, columnTypes, EMPTY, findCachedTable, formatValue, getCachedRecord, isEmptyValue, isFresh,
  isStructured, loadRecord, loadTables, memberCreator, PAGE, plural, recordFields, ROWS_TTL_MS, rowId, setLastRecord,
  setPageOffset, type RecordEntry,
} from './dataCache'
import './data.css'

interface RecordState {
  entry: RecordEntry | null
  error: string | null
  /** The record must be (re)loaded — the loader runs while true. */
  pending: boolean
}

function message(e: unknown): string {
  return e instanceof Error && e.message ? e.message : 'Could not load this record.'
}

export function RecordScreen({ agent, table, index }: { agent: HudAgent; table: string; index: number }) {
  const [st, setSt] = useState<RecordState>(() => {
    const entry = getCachedRecord(agent.id, table, index)
    return { entry, error: null, pending: !isFresh(entry, ROWS_TTL_MS) }
  })
  const [meta, setMeta] = useState<HudTable | null>(() => findCachedTable(agent.id, table))

  // Back lands on the page holding this record, focused on its row.
  useEffect(() => {
    setPageOffset(agent.id, table, Math.floor(index / PAGE) * PAGE)
    setLastRecord(agent.id, table, index)
  }, [agent.id, table, index])

  useEffect(() => {
    if (!st.pending) return
    let alive = true
    loadRecord(agent, table, index).then(
      (entry) => { if (alive) setSt({ entry, error: null, pending: false }) },
      (e: unknown) => { if (alive) setSt((s) => ({ ...s, error: message(e), pending: false })) },
    )
    return () => { alive = false }
  }, [agent, table, index, st.pending])

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

  const retry = () => {
    if (st.pending) return
    setSt((s) => ({ ...s, error: null, pending: true }))
  }

  const entry = st.entry
  const total = entry?.total ?? 0

  const step = (delta: -1 | 1) => {
    const i = index + delta
    if (i < 0 || i > total - 1) return
    navigate({ v: 'record', a: agent.id, tb: table, i }, { replace: true })
  }

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    if (e.altKey || e.ctrlKey || e.metaKey) return
    e.preventDefault()
    step(e.key === 'ArrowRight' ? 1 : -1)
  }

  const ready = entry !== null || st.error !== null
  useAutoFocus(ready)

  let body
  if (!entry) {
    body = st.error
      ? <StateView kind="error" message={st.error} onRetry={retry} />
      : <StateView kind="loading" message="Loading record…" />
  } else if (!entry.row) {
    body = (
      <StateView
        kind="empty"
        message={total > 0
          ? `Record ${(index + 1).toLocaleString()} is gone — the table now has ${plural(total, 'record')}.`
          : 'This table is empty.'}
        action={{ label: 'Back to list', onActivate: () => navigate({ v: 'rows', a: agent.id, tb: table }, { replace: true }) }}
      />
    )
  } else {
    const row = entry.row
    const types = columnTypes(meta)
    const creator = memberCreator(row)
    const fields = recordFields(meta, entry.columns, row)
    body = (
      <div className="hud-data-record" onKeyDown={onKey}>
        <div className="hud-block hud-data-head" tabIndex={0}>
          <div className="hud-data-id">#{rowId(row)}</div>
          <div className="hud-data-pos">Record {(index + 1).toLocaleString()} of {total.toLocaleString()}</div>
          {creator ? <div className="hud-data-creator">Added by {creator}</div> : null}
          {total > 1 ? <div className="hud-data-hint">◂ ▸ previous / next record</div> : null}
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
    ? `${agent.name} · ${(index + 1).toLocaleString()} of ${total.toLocaleString()}`
    : entry ? agent.name : `${agent.name} · ${(index + 1).toLocaleString()} of …`
  return (
    <ScreenFrame title={table} subtitle={subtitle}>
      {body}
    </ScreenFrame>
  )
}
