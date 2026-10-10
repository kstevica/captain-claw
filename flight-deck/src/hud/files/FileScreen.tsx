// One markdown file, read on the glasses.
//
// A first info block (path, age, size, who wrote it), then the document as
// focusable reading blocks (HudMarkdown blocks). Up/Down walk the blocks (a
// tall block is read page by page by the focus engine); Left/Right turn a
// whole page and put focus on the first block that starts on it, so the next
// Up/Down continues from what is on screen.
//
// A cached copy is shown at once; unless the file list was fetched moments
// ago, the list is re-checked in the background and a newer version replaces
// the copy (with a note in the info block).

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { HudApiError, type HudAgent, type HudFile } from '../api'
import { FOCUSABLE_SELECTOR, focusEl } from '../focus'
import { formatSize, relTime } from '../format'
import { useAutoFocus, useNow } from '../hooks'
import { splitMarkdown } from '../markdown/chunks'
import { HudMarkdown } from '../markdown/HudMarkdown'
import { htmlText } from '../markdown/rehypeHud'
import { ScreenFrame, StateView } from '../ui'
import { cachedText, findCachedFile, getCachedFiles, isFresh, loadFiles, loadText } from './filesCache'
import './files.css'

/** Render at most this much text (parsing + layout on the glasses' slow CPU). */
const RENDER_LIMIT = 200_000
/** A cached text is trusted without a re-check only when the list it was
 *  matched against is this young. */
const JUST_LISTED_MS = 5_000
const TIMED_OUT = 'Timed out downloading the file (slow connection?). Try again.'

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
  /** What the background re-check did to the copy shown. */
  notice: string | null
}

function describeError(e: unknown): { error: string; retry: boolean } {
  if (e instanceof HudApiError) {
    if (e.status === 413) return { error: 'Too large to show on the glasses (over 1 MB)', retry: false }
    if (e.status === 404) return { error: 'File not found. It may have been moved or deleted.', retry: true }
    if (e.status === 0 && e.message === 'Timed out') return { error: TIMED_OUT, retry: true }
    if (e.message) return { error: e.message, retry: true }
  }
  // A download cut off by the request timeout ("The user aborted a request.").
  const name = (e as { name?: unknown } | null)?.name
  if (name === 'TimeoutError' || name === 'AbortError') return { error: TIMED_OUT, retry: true }
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
    return { meta, text, error: null, retry: false, notice: null }
  })
  const [attempt, setAttempt] = useState(0)
  const [rendered, setRendered] = useState(1)
  const scrollRef = useRef<HTMLDivElement>(null)
  const infoRef = useRef<HTMLDivElement>(null)
  /** The text was replaced while the wearer was reading further down. */
  const backToTop = useRef(false)
  const now = useNow(60_000)

  useEffect(() => {
    let alive = true
    ;(async () => {
      let meta = findCachedFile(agent.id, fileKey)
      const listedAt = getCachedFiles(agent.id)?.at ?? 0
      // The first attempt rendered this straight from the cache.
      const cached = attempt === 0 && meta ? cachedText(agent.id, fileKey, meta.modified) : null
      let listFailed = false
      if (cached === null) {
        if (!meta) {
          // Deep link / reload: one list fetch for size + trust (optional).
          try { await loadFiles(agent) } catch { listFailed = true /* read the file anyway */ }
          if (!alive) return
          meta = findCachedFile(agent.id, fileKey)
        }
        try {
          const text = await loadText(agent, fileKey, meta)
          if (alive) setSt({ meta, text, error: null, retry: false, notice: null })
        } catch (e) {
          if (alive) setSt({ meta, text: null, notice: null, ...describeError(e) })
          return
        }
      }
      // Background re-check of the list: a cached copy may be outdated (the
      // list it matched can be old); a fresh read may carry old labels, or
      // none when the list could not be loaded ("Author unknown").
      const recheck = cached !== null
        ? Date.now() - listedAt >= JUST_LISTED_MS
        : meta ? !isFresh(getCachedFiles(agent.id)) : listFailed
      if (!alive || !recheck) return
      let files: HudFile[]
      try {
        files = await loadFiles(agent)
      } catch {
        // Keep what is shown; a cached copy may be outdated.
        if (alive && cached !== null) setSt((s) => ({ ...s, notice: "Saved copy: couldn't check for a newer version." }))
        return
      }
      const m = files.find((f) => f.key === fileKey)
      if (!alive || !m) return
      if (cached === null || !meta || (m.modified === meta.modified && m.size === meta.size)) {
        setSt((s) => ({ ...s, meta: m }))
        return
      }
      // Changed since the cached copy: show the new version.
      let text: string
      try {
        text = await loadText(agent, fileKey, m)
      } catch {
        if (alive) setSt((s) => ({ ...s, notice: 'Changed since this copy; the new version could not be loaded.' }))
        return
      }
      if (!alive) return
      const scroller = scrollRef.current
      const active = document.activeElement
      backToTop.current = !!scroller && !!active && scroller.contains(active) && active !== infoRef.current
      setRendered(1)
      setSt({ meta: m, text, error: null, retry: false, notice: 'Updated to the latest version.' })
    })()
    return () => { alive = false }
  }, [agent, fileKey, attempt])

  // A replaced text while reading further down: back to the top (the info
  // block says why); the old position means nothing in the new version.
  useLayoutEffect(() => {
    if (!backToTop.current) return
    backToTop.current = false
    if (infoRef.current) focusEl(infoRef.current)
    if (scrollRef.current) scrollRef.current.scrollTop = 0
  }, [st.text])

  const loaded = st.text !== null
  const shown = useMemo(() => (st.text === null ? null : clip(st.text.replace(/\r\n?/g, '\n'))), [st.text])
  const pieces = useMemo(() => (shown ? splitMarkdown(shown.body) : []), [shown])
  /** Nothing the glasses can show: an empty file, or one of HTML only. */
  const blank = useMemo(() => {
    if (!shown) return null
    if (!/\S/.test(shown.body)) return 'This file is empty.'
    if (/^\s*</.test(shown.body) && !htmlText(shown.body).length) {
      return "Nothing to show: this file only holds HTML the glasses don't display."
    }
    return null
  }, [shown])

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
    setRendered(1)
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
        <div className="hud-block hud-file-info" tabIndex={0} ref={infoRef}>
          <div className="hud-file-path">{meta?.logical || name}</div>
          {facts ? <div className="hud-file-facts">{facts}</div> : null}
          {/* Fail closed: unknown authorship is not the owner's. */}
          {!meta || !meta.trusted ? (
            <div className="hud-file-creator">
              {meta ? `Created by ${meta.creator || 'a member'}` : "Author unknown (couldn't check the file list)"}
            </div>
          ) : null}
          {st.notice ? <div className="hud-file-note">{st.notice}</div> : null}
          {shown.clipped ? (
            <div className="hud-file-note">Long file: showing the first 200 KB.</div>
          ) : null}
          {blank ? <div className="hud-file-note">{blank}</div> : null}
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
