// Data tab: the agent's datastore tables, most recently updated first.
// A pinch opens a table's rows. Read-only — no SQL, edits or export here.
//
// The list comes from dataCache (shown at once when cached; refetched when
// older than TABLES_TTL_MS) so Back from a table is instant and focus returns
// to the row the wearer opened.

import { useEffect, useState } from 'react'
import type { HudAgent, HudTable } from '../api'
import { relTime } from '../format'
import { useAutoFocus, useNow } from '../hooks'
import { navigate } from '../router'
import { Btn, Row, ScreenFrame, StateView } from '../ui'
import { getCachedTables, isFresh, loadTables, plural, TABLES_TTL_MS } from './dataCache'
import './data.css'

interface ListState {
  tables: HudTable[] | null
  /** Load error with nothing to show. */
  error: string | null
  /** A (re)load is running. */
  pending: boolean
  /** A refresh failed while a list is shown. */
  stale: string | null
}

function message(e: unknown): string {
  return e instanceof Error && e.message ? e.message : 'Could not load tables.'
}

function tableMeta(t: HudTable, now: number): string {
  const parts = [plural(t.rowCount, 'row'), plural(t.columns.length, 'column')]
  const ms = t.updatedAt ? Date.parse(t.updatedAt) : NaN
  const ago = Number.isFinite(ms) ? relTime(ms, now) : ''
  if (ago) parts.push(`updated ${ago}`)
  return parts.join(' · ')
}

export function TablesScreen({ agent }: { agent: HudAgent }) {
  const [st, setSt] = useState<ListState>(() => {
    const cached = getCachedTables(agent.id)
    return { tables: cached?.tables ?? null, error: null, pending: !cached, stale: null }
  })
  const now = useNow(60_000)

  // First load, or a refresh of a list older than the TTL.
  useEffect(() => {
    if (isFresh(getCachedTables(agent.id), TABLES_TTL_MS)) return
    let alive = true
    loadTables(agent).then(
      (tables) => { if (alive) setSt({ tables, error: null, pending: false, stale: null }) },
      (e: unknown) => {
        if (!alive) return
        setSt((s) => (s.tables
          ? { ...s, pending: false, stale: message(e) }
          : { tables: null, error: message(e), pending: false, stale: null }))
      },
    )
    return () => { alive = false }
  }, [agent])

  const reload = () => {
    if (st.pending) return
    setSt((s) => ({ ...s, error: null, pending: true }))
    loadTables(agent).then(
      (tables) => setSt({ tables, error: null, pending: false, stale: null }),
      (e: unknown) => setSt((s) => (s.tables
        ? { ...s, pending: false, stale: message(e) }
        : { tables: null, error: message(e), pending: false, stale: null })),
    )
  }

  const ready = st.tables !== null || st.error !== null
  useAutoFocus(ready)

  let body
  if (st.tables === null) {
    body = st.error
      ? <StateView kind="error" message={st.error} onRetry={reload} />
      : <StateView kind="loading" message="Loading tables…" />
  } else if (st.tables.length === 0) {
    body = (
      <StateView kind="empty" message={st.pending ? 'Checking for tables…' : 'No tables yet.'}
        action={{ label: 'Refresh', onActivate: reload }} />
    )
  } else {
    body = (
      <>
        <div className="hud-note hud-data-note">
          {plural(st.tables.length, 'table')}
          {st.stale ? <span className="hud-data-stale"> · couldn't refresh</span> : null}
        </div>
        <div className="hud-data-list">
          {st.tables.map((t) => (
            <Row
              key={t.name}
              fk={`table-${t.name}`}
              title={t.name}
              meta={tableMeta(t, now)}
              onActivate={() => navigate({ v: 'rows', a: agent.id, tb: t.name })}
            />
          ))}
        </div>
        <div className="hud-actions">
          <Btn variant="chip" fk="tables-refresh" onActivate={reload}>
            {st.pending ? 'Refreshing…' : 'Refresh'}
          </Btn>
        </div>
      </>
    )
  }

  return (
    <ScreenFrame title={agent.name} agent={agent} tab="data">
      {body}
    </ScreenFrame>
  )
}
