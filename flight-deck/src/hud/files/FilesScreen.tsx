// Files tab: the agent's markdown files, newest first. A pinch opens one.
//
// The list comes from filesCache (shown at once when cached; refetched in the
// background when older than LIST_TTL_MS) so Back from a file is instant and
// focus returns to the row the wearer opened.

import { useEffect, useState } from 'react'
import { MAX_FILE_BYTES, type HudAgent, type HudFile } from '../api'
import { formatSize, relTime } from '../format'
import { useAutoFocus, useNow } from '../hooks'
import { navigate } from '../router'
import { Btn, Row, ScreenFrame, StateView } from '../ui'
import { getCachedFiles, isFresh, loadFiles } from './filesCache'
import './files.css'

/** Rows rendered at most (the glasses' CPU is ~12× slower than a laptop). */
const MAX_ROWS = 150

interface ListState {
  files: HudFile[] | null
  /** Load error with nothing to show. */
  error: string | null
  /** A (re)load is running. */
  pending: boolean
  /** A background refresh failed while a list is shown. */
  stale: string | null
}

function message(e: unknown): string {
  return e instanceof Error && e.message ? e.message : 'Could not load files.'
}

function fileMeta(f: HudFile, now: number): string {
  const parts = [relTime(f.modified, now), formatSize(f.size)].filter(Boolean)
  if (!f.trusted) parts.push(`by ${f.creator || 'a member'}`)
  return parts.join(' · ')
}

export function FilesScreen({ agent }: { agent: HudAgent }) {
  const [st, setSt] = useState<ListState>(() => {
    const cached = getCachedFiles(agent.id)
    return { files: cached?.files ?? null, error: null, pending: !cached, stale: null }
  })
  const now = useNow(60_000)

  // First load, or a background refresh of a stale cached list.
  useEffect(() => {
    if (isFresh(getCachedFiles(agent.id))) return
    let alive = true
    loadFiles(agent).then(
      (files) => { if (alive) setSt({ files, error: null, pending: false, stale: null }) },
      (e: unknown) => {
        if (!alive) return
        setSt((s) => (s.files
          ? { ...s, pending: false, stale: message(e) }
          : { files: null, error: message(e), pending: false, stale: null }))
      },
    )
    return () => { alive = false }
  }, [agent])

  const reload = () => {
    if (st.pending) return
    setSt((s) => ({ ...s, error: null, pending: true }))
    loadFiles(agent).then(
      (files) => setSt({ files, error: null, pending: false, stale: null }),
      (e: unknown) => setSt((s) => (s.files
        ? { ...s, pending: false, stale: message(e) }
        : { files: null, error: message(e), pending: false, stale: null })),
    )
  }

  const ready = st.files !== null || st.error !== null
  useAutoFocus(ready)

  let body
  if (st.files === null) {
    body = st.error
      ? <StateView kind="error" message={st.error} onRetry={reload} />
      : <StateView kind="loading" message="Loading files…" />
  } else if (st.files.length === 0) {
    body = (
      <StateView kind="empty" message={st.pending ? 'Checking for files…' : 'No markdown files yet.'}
        action={{ label: 'Refresh', onActivate: reload }} />
    )
  } else {
    const shown = st.files.slice(0, MAX_ROWS)
    const count = st.files.length
    body = (
      <>
        <div className="hud-note hud-files-count">
          {count === 1 ? '1 markdown file' : `${count} markdown files`}
          {count > shown.length ? ` · newest ${shown.length} shown` : ''}
          {st.stale ? <span className="hud-files-stale"> · couldn't refresh</span> : null}
        </div>
        {shown.map((f) => (
          <Row
            key={f.key}
            fk={`file-${f.key}`}
            title={f.name}
            meta={fileMeta(f, now)}
            dim={f.size > MAX_FILE_BYTES}
            onActivate={() => navigate({ v: 'file', a: agent.id, p: f.key, n: f.name })}
          />
        ))}
        <div className="hud-actions">
          <Btn variant="chip" fk="files-refresh" onActivate={reload}>
            {st.pending ? 'Refreshing…' : 'Refresh'}
          </Btn>
        </div>
      </>
    )
  }

  return (
    <ScreenFrame title={agent.name} agent={agent} tab="files">
      {body}
    </ScreenFrame>
  )
}
