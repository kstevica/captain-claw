// One markdown file, read on the glasses.
//
// A first info block (path, age, size, who wrote it), then the document as
// focusable reading blocks (HudMarkdown blocks). Up/Down walk the blocks (a
// tall block is read page by page by the focus engine); Left/Right turn a
// whole page and put focus on the first block that starts on it, so the next
// Up/Down continues from what is on screen.

import { useEffect, useMemo, useRef, useState } from 'react'
import { HudApiError, readFileText, type HudAgent, type HudFile } from '../api'
import { FOCUSABLE_SELECTOR, focusEl } from '../focus'
import { formatSize, relTime } from '../format'
import { useAutoFocus, useNow } from '../hooks'
import { splitMarkdown } from '../markdown/chunks'
import { HudMarkdown } from '../markdown/HudMarkdown'
import { ScreenFrame, StateView } from '../ui'
import { cachedText, findCachedFile, loadFiles, rememberText } from './filesCache'
import './files.css'

/** Render at most this much text (parsing + layout on the glasses' slow CPU). */
const RENDER_LIMIT = 200_000

type IdleHandle = number
const onIdle: (cb: () => void) => IdleHandle = typeof window.requestIdleCallback === 'function'
  ? (cb) => window.requestIdleCallback(cb, { timeout: 400 })
  : (cb) => window.setTimeout(cb, 16)
const cancelIdle: (h: IdleHandle) => void = typeof window.cancelIdleCallback === 'function'
  ? (h) => window.cancelIdleCallback(h)
  : (h) => window.clearTimeout(h)

interface FileState {
  meta: HudFile | null
  text: string | null
  error: string | null
  /** Retry makes sense (not for "too large"). */
  retry: boolean
}

function describeError(e: unknown): { error: string; retry: boolean } {
  if (e instanceof HudApiError) {
    if (e.status === 413) return { error: 'Too large to show on the glasses (over 1 MB)', retry: false }
    if (e.status === 404) return { error: 'File not found. It may have been moved or deleted.', retry: true }
    if (e.message) return { error: e.message, retry: true }
  }
  return { error: e instanceof Error && e.message ? e.message : 'Could not open the file.', retry: true }
}

/** First RENDER_LIMIT chars, cut at a line break when one is near. */
function clip(text: string): { body: string; clipped: boolean } {
  if (text.length <= RENDER_LIMIT) return { body: text, clipped: false }
  const nl = text.lastIndexOf('\n', RENDER_LIMIT)
  return { body: text.slice(0, nl > RENDER_LIMIT - 10_000 ? nl : RENDER_LIMIT), clipped: true }
}

/**
 * Scroll one page (80 % of the scroller) and focus the first stop whose top
 * is on the new page — or, inside a block taller than the page, that block.
 * At either end: focus the first / last stop. focusEl()'s ensureVisible would
 * scroll that block to its own top (or pin first/last stops to the ends), so
 * the page position is put back right after (same task: nothing is painted
 * in between).
 */
function turnPage(scroller: HTMLElement, dir: 1 | -1): void {
  const before = scroller.scrollTop
  scroller.scrollTop = before + dir * Math.max(40, Math.round(scroller.clientHeight * 0.8))
  const after = scroller.scrollTop
  const stops = Array.from(scroller.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
  if (!stops.length) return
  let target: HTMLElement | undefined
  if (after === before) {
    target = dir > 0 ? stops[stops.length - 1] : stops[0]
  } else {
    const s = scroller.getBoundingClientRect()
    const top = s.top - 2
    const bottom = s.bottom - 32
    target = stops.find((b) => { const r = b.getBoundingClientRect(); return r.top >= top && r.top < bottom })
      ?? stops.find((b) => { const r = b.getBoundingClientRect(); return r.top < top && r.bottom > s.top + 32 })
  }
  if (!target) return
  if (target !== document.activeElement) focusEl(target)
  if (after !== before) scroller.scrollTop = after
}

export function FileScreen({ agent, fileKey, name }: { agent: HudAgent; fileKey: string; name: string }) {
  const [st, setSt] = useState<FileState>(() => {
    const meta = findCachedFile(agent.id, fileKey)
    const text = meta ? cachedText(agent.id, fileKey, meta.modified) : null
    return { meta, text, error: null, retry: false }
  })
  const [attempt, setAttempt] = useState(0)
  const scrollRef = useRef<HTMLDivElement>(null)
  const now = useNow(60_000)

  useEffect(() => {
    const known = findCachedFile(agent.id, fileKey)
    if (known && cachedText(agent.id, fileKey, known.modified) !== null) return // shown from cache
    let alive = true
    ;(async () => {
      let meta = known
      if (!meta) {
        // Deep link / reload: one list fetch for size + trust (optional).
        try { await loadFiles(agent) } catch { /* read the file anyway */ }
        meta = findCachedFile(agent.id, fileKey)
      }
      try {
        const text = await readFileText(agent, fileKey, meta?.size || undefined)
        if (meta) rememberText(agent.id, fileKey, meta.modified, text)
        if (alive) setSt({ meta, text, error: null, retry: false })
      } catch (e) {
        if (alive) setSt({ meta, text: null, ...describeError(e) })
      }
    })()
    return () => { alive = false }
  }, [agent, fileKey, attempt])

  const loaded = st.text !== null
  const shown = useMemo(() => (st.text === null ? null : clip(st.text)), [st.text])
  const pieces = useMemo(() => (shown ? splitMarkdown(shown.body) : []), [shown])
  const [rendered, setRendered] = useState(1)

  // Progressive rendering: one more piece while the reader is within two
  // screens of the end of what is rendered (checked when idle, after each
  // piece lands and on every scroll).
  useEffect(() => {
    const scroller = scrollRef.current
    if (!scroller || rendered >= pieces.length) return
    let handle: IdleHandle | 0 = 0
    const grow = () => {
      handle = 0
      const left = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight
      if (left < scroller.clientHeight * 2) setRendered((n) => Math.min(pieces.length, n + 1))
    }
    const schedule = () => { if (!handle) handle = onIdle(grow) }
    scroller.addEventListener('scroll', schedule, { passive: true })
    schedule()
    return () => {
      scroller.removeEventListener('scroll', schedule)
      if (handle) cancelIdle(handle)
    }
  }, [rendered, pieces.length])

  // Left/Right page turning. Capture phase: runs before the focus engine
  // (which would otherwise treat Left/Right as a focus move).
  useEffect(() => {
    if (!loaded) return
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
      const scroller = scrollRef.current
      if (!scroller || !scroller.isConnected) return
      e.preventDefault()
      turnPage(scroller, e.key === 'ArrowRight' ? 1 : -1)
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [loaded])

  useAutoFocus(loaded || st.error !== null)

  const retry = () => {
    setSt((s) => ({ ...s, error: null }))
    setAttempt((n) => n + 1)
  }

  let body
  if (st.error !== null) {
    body = <StateView kind="error" message={st.error} onRetry={st.retry ? retry : undefined} />
  } else if (st.text === null || !shown) {
    body = <StateView kind="loading" message="Opening…" />
  } else {
    const meta = st.meta
    const size = meta?.size || st.text.length
    const facts = [meta ? relTime(meta.modified, now) : '', formatSize(size)].filter(Boolean).join(' · ')
    body = (
      <>
        <div className="hud-block hud-file-info" tabIndex={0}>
          <div className="hud-file-path">{meta?.logical || name}</div>
          {facts ? <div className="hud-file-facts">{facts}</div> : null}
          {meta && !meta.trusted ? (
            <div className="hud-file-creator">Created by {meta.creator || 'a member'}</div>
          ) : null}
          {shown.clipped ? (
            <div className="hud-file-note">Long file: showing the first 200 KB.</div>
          ) : null}
          {/\S/.test(st.text) ? null : <div className="hud-file-note">This file is empty.</div>}
        </div>
        {pieces.slice(0, rendered).map((piece, i) => (
          <HudMarkdown key={i} text={piece} blocks className="hud-file-piece" />
        ))}
        {rendered < pieces.length ? <div className="hud-note hud-file-more">More below…</div> : null}
      </>
    )
  }

  return (
    <ScreenFrame title={name} subtitle={agent.name} scrollRef={scrollRef} className="hud-file-screen">
      {body}
    </ScreenFrame>
  )
}
