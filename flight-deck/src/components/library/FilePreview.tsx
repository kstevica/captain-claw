import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Eye, FileQuestion, Loader2, ShieldAlert } from 'lucide-react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { formatSize } from '../../services/fileTransfer'
import { htmlFrame, untrustedUrlTransform } from '../../utils/untrustedContent'
import { isSheetExt, readXlsx } from '../../utils/xlsx'

// ── A small preview of the selected file ──
//
// Under the file's details in the library: the picture, a player, the first
// page of a PDF, or the start of its text. It comes straight from Flight
// Deck's copy of the file (the library's view route), so it shows whether the
// agent runs or not; each version of a file has its own URL, so an edited
// file never shows a cached copy of the last one. A PDF's first page is a
// picture Flight Deck renders (browsers show PDFs inline only with a viewer
// plugin, which the desktop app and phones lack). Markdown is rendered (tables
// too) and a page shows as a small picture of itself, under the viewer's
// rules for files someone else may have written: no scripts and nothing
// remote from those. Text is read only as far as the preview needs; a very
// large picture, and a file that may hold secrets (.env), wait for a click.

/** How much of a file the preview reads: a page whole (up to a point, or it
 *  renders broken), the start of markdown, a screenful of anything else. */
const READ_BYTES: Record<string, number> = { html: 1024 * 1024, markdown: 64 * 1024 }
const TEXT_BYTES = 16 * 1024
const TEXT_LINES = 60
/** A page is laid out this wide, then scaled down into the preview. */
const PAGE_WIDTH = 1024
const PAGE_PREVIEW_HEIGHT = 288
/** Bigger than this, a picture loads only when asked. */
const AUTO_IMAGE_BYTES = 15 * 1024 * 1024
/** Files the library never reads, for the same reason: they may hold secrets. */
const SECRET_EXTS = new Set(['.env'])
const TABLE_ROWS = 8
const TABLE_COLS = 6
/** A workbook is read whole for its preview: bigger than this, Open reads it. */
const SHEET_PREVIEW_BYTES = 8 * 1024 * 1024

/** The groups FilePreview reads as text (getFileTypeGroup's names). */
const TEXT_GROUPS = new Set(['text', 'code', 'data', 'markdown', 'html'])

type TextState =
  | { state: 'loading' }
  | { state: 'ready'; text: string; cut: boolean }
  | { state: 'binary' }
  | { state: 'error' }

// The start of a text file, kept for the files looked at this session (keyed
// by id and change time, so an edited file is read again).
const textCache = new Map<string, { text: string; cut: boolean } | 'binary'>()
const TEXT_CACHE_MAX = 40

type SheetState =
  | { state: 'loading' }
  | { state: 'ready'; name: string; rows: string[][]; cut: boolean; sheets: number }
  | { state: 'error'; why: string }

// The first rows of a workbook's first sheet, kept the same way.
const sheetCache = new Map<string, SheetState>()

async function readStart(url: string, limit: number, signal: AbortSignal): Promise<{ text: string; cut: boolean } | 'binary'> {
  const res = await fetch(url, { signal, credentials: 'include', cache: 'no-cache' })
  if (!res.ok || !res.body) throw new Error(`${res.status}`)
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let got = 0
  let cut = false
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    got += value.length
    // More than the limit: the file goes on past what the preview shows.
    if (got > limit) { cut = true; void reader.cancel(); break }
  }
  const bytes = new Uint8Array(Math.min(got, limit))
  let at = 0
  for (const c of chunks) {
    const take = Math.min(c.length, bytes.length - at)
    bytes.set(c.subarray(0, take), at)
    at += take
    if (at >= bytes.length) break
  }
  // A NUL in the first bytes: not text after all.
  if (bytes.subarray(0, 4096).includes(0)) return 'binary'
  let text = new TextDecoder('utf-8', { fatal: false }).decode(bytes)
  // Cut mid-file: drop the partial last line (and a broken character with it).
  if (cut) {
    const nl = text.lastIndexOf('\n')
    if (nl > 0) text = text.slice(0, nl)
  }
  return { text, cut }
}

const SEPARATORS = [';', '\t', ',']

/** The separator a .csv uses (Excel in many European locales writes ;, with
 *  decimal commas): over its first lines, the one found the same number of
 *  times on most lines, outside quotes; a tie goes to ; and tab before the
 *  comma. A title line with none, or a header with a comma in a name, doesn't
 *  mislead it. */
function sniffSeparator(text: string): string {
  const lines: Record<string, number>[] = []
  let counts: Record<string, number> = { ';': 0, '\t': 0, ',': 0 }
  let quoted = false
  for (const ch of text) {
    if (ch === '"') quoted = !quoted
    else if (!quoted && (ch === '\n' || ch === '\r')) {
      if (Object.values(counts).some((n) => n > 0)) lines.push(counts)
      counts = { ';': 0, '\t': 0, ',': 0 }
      if (lines.length >= 10) break
    } else if (!quoted && ch in counts) counts[ch]++
  }
  if (Object.values(counts).some((n) => n > 0)) lines.push(counts)
  let best = ','
  let bestScore = 0
  for (const sep of SEPARATORS) {
    // How many lines agree on this separator's most common non-zero count.
    const tally = new Map<number, number>()
    for (const l of lines) if (l[sep] > 0) tally.set(l[sep], (tally.get(l[sep]) ?? 0) + 1)
    const score = Math.max(0, ...tally.values())
    if (score > bestScore) { best = sep; bestScore = score }
  }
  return best
}

/** The first rows of a CSV/TSV (quotes, doubled quotes and line breaks in
 *  quotes understood), and whether more follow. */
function parseDelimited(text: string, sep: string): { rows: string[][]; more: boolean } {
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  let i = 0
  for (; i < text.length && rows.length <= TABLE_ROWS; i++) {
    const ch = text[i]
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++ }
      else if (ch === '"') quoted = false
      else cell += ch
    } else if (ch === '"' && cell === '') quoted = true
    else if (ch === sep) { row.push(cell); cell = '' }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++
      row.push(cell); cell = ''
      if (row.some((c) => c !== '')) rows.push(row)
      row = []
    } else cell += ch
  }
  if (i >= text.length && (cell !== '' || row.length)) { row.push(cell); if (row.some((c) => c !== '')) rows.push(row) }
  // Anything but blank lines after the last row read: more of the file follows.
  const more = i < text.length && text.slice(i).trim() !== ''
  return { rows, more }
}

function shownText(text: string, ext: string, cut: boolean): string {
  // Whole JSON reads better laid out.
  if (!cut && ext === '.json') {
    try { return JSON.stringify(JSON.parse(text), null, 2) } catch { /* not valid JSON: as it is */ }
  }
  return text
}

/** A callback ref, and a value measured from the element it is set on:
 *  once when the element mounts (a browser may hold resize observations back
 *  while the page is in the background), then whenever it or its children
 *  change size. */
function useMeasured<T>(measure: (el: HTMLElement) => T, initial: T): [(el: HTMLElement | null) => void, T] {
  const [value, setValue] = useState(initial)
  const ro = useRef<ResizeObserver | null>(null)
  const attach = useCallback((el: HTMLElement | null) => {
    ro.current?.disconnect()
    ro.current = null
    if (!el) return
    const check = () => setValue(measure(el))
    check()
    if (typeof ResizeObserver === 'undefined') return
    ro.current = new ResizeObserver(check)
    ro.current.observe(el)
    // The content can grow inside a box of the same size (images in markdown).
    for (const child of Array.from(el.children)) ro.current.observe(child)
    // `measure` is a module-level function: the ref stays the same.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return [attach, value]
}

// In the preview a link is only text (a click there would take the whole app
// away), and an image shows only from a web or data address: a relative one
// resolves against Flight Deck's page, not the file's folder.
const MD_PREVIEW: Components = {
  a: ({ children }) => <span className="text-violet-600 underline underline-offset-2 dark:text-violet-400">{children}</span>,
  img: ({ src, alt }) => (typeof src === 'string' && /^(https?:|data:image\/)/i.test(src) ? <img src={src} alt={alt ?? ''} /> : null),
}

/** Whether the element's content runs past its height (it is clipped). */
const overflowsOf = (el: HTMLElement) => el.scrollHeight > el.clientHeight + 1
const widthOf = (el: HTMLElement) => el.clientWidth

const FADE = '[mask-image:linear-gradient(to_bottom,black_75%,transparent)]'

export function FilePreview({ group, extension, size, url, freshUrl, thumbUrl, cacheKey, untrusted, phone, viewerOpen, canOpen, onOpen }: {
  /** getFileTypeGroup of the file. */
  group: string
  extension: string
  size: number
  /** The file inline (the library's view route), for this version of it. */
  url: string
  /** The same address made now, with the current sign-in token. */
  freshUrl: () => string
  /** A PDF's first page as a picture, for this version of it. */
  thumbUrl: string
  /** Unique per file version: id + change time. */
  cacheKey: string
  /** Someone else may have written it: a page runs no scripts and loads
   *  nothing remote, markdown shows no remote images. */
  untrusted: boolean
  phone: boolean
  /** A file is open in the viewer over the library: a playing preview pauses. */
  viewerOpen: boolean
  canOpen: boolean
  onOpen: () => void
}) {
  const ext = extension.toLowerCase()
  const isText = TEXT_GROUPS.has(group)
  const secret = SECRET_EXTS.has(ext)
  const big = group === 'image' && size > AUTO_IMAGE_BYTES
  const [asked, setAsked] = useState(false)
  const [failed, setFailed] = useState(false)
  const [text, setText] = useState<TextState>(() => {
    const hit = isText ? textCache.get(cacheKey) : undefined
    return hit === undefined ? { state: 'loading' } : hit === 'binary' ? { state: 'binary' } : { state: 'ready', ...hit }
  })
  const reading = isText && (!secret || asked)
  const workbook = isSheetExt(ext)
  const [sheet, setSheet] = useState<SheetState>(() => sheetCache.get(cacheKey) ?? { state: 'loading' })
  const sheetTooBig = workbook && size > SHEET_PREVIEW_BYTES

  useEffect(() => {
    if (!workbook || sheetTooBig || sheetCache.has(cacheKey)) return
    const ctl = new AbortController()
    fetch(freshUrl(), { signal: ctl.signal, credentials: 'include', cache: 'no-cache' })
      .then(async (res) => {
        if (!res.ok) throw new Error(`${res.status}`)
        const wb = await readXlsx(await res.arrayBuffer(), { maxRows: TABLE_ROWS + 1, maxSheets: 1 })
        const first = wb.sheets[0]
        return { state: 'ready', name: first.name, rows: first.rows, cut: first.cut, sheets: wb.total } as SheetState
      })
      .catch((e) => ({ state: 'error', why: e instanceof Error ? e.message : '' }) as SheetState)
      .then((r) => {
        if (ctl.signal.aborted) return
        sheetCache.set(cacheKey, r)
        if (sheetCache.size > TEXT_CACHE_MAX) sheetCache.delete(sheetCache.keys().next().value as string)
        setSheet(r)
      })
    return () => ctl.abort()
    // freshUrl is made anew on every render; it only matters when this runs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workbook, sheetTooBig, cacheKey])

  // The URL changes with every token refresh; the version (cacheKey) doesn't.
  useEffect(() => {
    if (!reading || textCache.has(cacheKey)) return
    const ctl = new AbortController()
    readStart(freshUrl(), READ_BYTES[group] ?? TEXT_BYTES, ctl.signal)
      .then((r) => {
        textCache.set(cacheKey, r)
        if (textCache.size > TEXT_CACHE_MAX) textCache.delete(textCache.keys().next().value as string)
        setText(r === 'binary' ? { state: 'binary' } : { state: 'ready', ...r })
      })
      .catch(() => { if (!ctl.signal.aborted) setText({ state: 'error' }) })
    return () => ctl.abort()
    // freshUrl is made anew on every render; it only matters when this runs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reading, cacheKey, group])

  // A player keeps the address it started with: a token refresh changes the
  // URL, and a new src would restart it. When it fails, it tries once more
  // with a fresh address if the token has changed since (a long recording);
  // otherwise — or if that fails too — the browser can't play it. A source
  // that loads earns the next expiry its own retry.
  const media = useRef<HTMLMediaElement | null>(null)
  const [mediaSrc, setMediaSrc] = useState(url)
  const [retried, setRetried] = useState(false)
  const mediaFailed = () => {
    const next = freshUrl()
    if (retried || next === mediaSrc) { setFailed(true); return }
    setRetried(true)
    const at = media.current?.currentTime ?? 0
    setMediaSrc(next)
    // Pick up where it was once the new source is ready.
    media.current?.addEventListener('loadedmetadata', () => { if (media.current && at) media.current.currentTime = at }, { once: true })
  }
  const mediaLoaded = () => { if (retried) setRetried(false) }
  useEffect(() => { if (viewerOpen) media.current?.pause() }, [viewerOpen])
  const open = () => { media.current?.pause(); onOpen() }

  const [textBox, overflows] = useMeasured(overflowsOf, false)

  let body: ReactNode
  let caption = ''
  if (failed) {
    body = group === 'pdf'
      ? <Empty text="No preview of this PDF here." hint={canOpen ? 'Open reads it.' : undefined} />
      : <Empty text={`This ${group === 'image' ? 'picture' : 'file'} can't be previewed here.`} />
  } else if ((big || secret) && !asked) {
    body = (
      <button
        type="button"
        onClick={() => setAsked(true)}
        className="flex w-full flex-col items-center gap-1.5 px-4 py-6 text-xs text-zinc-400 transition-colors hover:bg-zinc-800/50 hover:text-zinc-200"
      >
        {secret ? <ShieldAlert className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
        {secret ? 'It may hold passwords or keys. Show its contents' : `Show a preview (${formatSize(size)})`}
      </button>
    )
  } else if (group === 'image') {
    body = (
      <button type="button" onClick={canOpen ? open : undefined} disabled={!canOpen} aria-label={canOpen ? 'Open the picture' : undefined} className="flex w-full items-center justify-center bg-zinc-950/60 p-2 disabled:cursor-default" title={canOpen ? 'Open' : undefined}>
        <img
          src={url}
          alt=""
          loading="lazy"
          decoding="async"
          onError={() => setFailed(true)}
          className={`max-w-full rounded object-contain ${phone ? 'max-h-56' : 'max-h-64'}`}
        />
      </button>
    )
  } else if (group === 'video') {
    body = (
      <div className="bg-black">
        <video ref={(el) => { media.current = el }} src={mediaSrc} controls playsInline preload="metadata" onError={mediaFailed} onLoadedMetadata={mediaLoaded} className={`w-full ${phone ? 'max-h-56' : 'max-h-64'}`} />
      </div>
    )
  } else if (group === 'audio') {
    body = (
      <div className="px-3 py-4">
        <audio ref={(el) => { media.current = el }} src={mediaSrc} controls preload="metadata" onError={mediaFailed} onLoadedMetadata={mediaLoaded} className="w-full" />
      </div>
    )
  } else if (group === 'pdf') {
    body = (
      <button type="button" onClick={canOpen ? open : undefined} disabled={!canOpen} aria-label={canOpen ? 'Open the PDF' : undefined} className="flex w-full items-center justify-center bg-zinc-950/60 p-2 disabled:cursor-default" title={canOpen ? 'Open' : undefined}>
        {/* A small picture: loaded at once (a lazy one waits to be scrolled
            into view, and the preview sits under the file's details). */}
        <img
          src={thumbUrl}
          alt=""
          decoding="async"
          onError={() => setFailed(true)}
          className={`max-w-full rounded bg-white object-contain shadow-sm ${phone ? 'max-h-56' : 'max-h-72'}`}
        />
      </button>
    )
    caption = 'The first page'
  } else if (workbook) {
    if (sheetTooBig) {
      body = <Empty text="This workbook is too big to preview." hint={canOpen ? 'Open shows it as a table.' : undefined} />
    } else if (sheet.state === 'loading') {
      body = <div className="flex items-center justify-center py-8"><Loader2 className="h-4 w-4 animate-spin text-zinc-500" /></div>
    } else if (sheet.state === 'error') {
      body = <Empty text="Couldn't read the workbook for a preview." hint={sheet.why || undefined} />
    } else if (!sheet.rows.length) {
      body = <Empty text="Its first sheet is empty." hint={sheet.sheets > 1 && canOpen ? 'Open shows the other sheets.' : undefined} />
    } else {
      const cols = Math.max(0, ...sheet.rows.map((r) => r.length))
      body = <TablePreview rows={sheet.rows} />
      caption = [
        sheet.cut ? `The first ${Math.max(0, sheet.rows.length - 1)} rows` : '',
        cols > TABLE_COLS ? `${TABLE_COLS} of ${cols} columns` : '',
        sheet.sheets > 1 ? `sheet “${sheet.name}” of ${sheet.sheets}` : '',
      ].filter(Boolean).join(' · ')
    }
  } else if (isText) {
    if (text.state === 'loading') {
      body = <div className="flex items-center justify-center py-8"><Loader2 className="h-4 w-4 animate-spin text-zinc-500" /></div>
    } else if (text.state === 'error') {
      body = <Empty text="Couldn't read the file for a preview." />
    } else if (text.state === 'binary') {
      body = <Empty text="It isn't text, so there's nothing to preview." />
    } else if (!text.text.trim()) {
      body = <Empty text="The file is empty." />
    } else if (group === 'markdown') {
      body = (
        <div ref={textBox} className={`fd-file-markdown fd-md-preview overflow-hidden px-3 py-2 ${overflows ? FADE : ''} ${phone ? 'max-h-72' : 'max-h-80'}`}>
          {untrusted
            ? <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={MD_PREVIEW}>{text.text}</Markdown>
            : <Markdown remarkPlugins={[remarkGfm]} components={MD_PREVIEW}>{text.text}</Markdown>}
        </div>
      )
      caption = [overflows || text.cut ? 'The start of it' : '', untrusted ? 'remote images off' : ''].filter(Boolean).join(' · ')
    } else if (group === 'html' && text.cut) {
      // Cut short, a page renders broken or blank (an inline script library
      // on one long line, say): the viewer reads it whole.
      body = <Empty text="This page is too big to preview." hint={canOpen ? 'Open shows it.' : undefined} />
    } else if (group === 'html') {
      body = <PagePreview html={text.text} untrusted={untrusted} paused={viewerOpen} canOpen={canOpen} onOpen={open} />
      caption = untrusted ? 'Scripts and remote content off' : ''
    } else if (ext === '.csv' || ext === '.tsv') {
      const { rows, more } = parseDelimited(text.text, ext === '.tsv' ? '\t' : sniffSeparator(text.text))
      const shown = rows.slice(0, TABLE_ROWS + 1)
      const cols = Math.max(0, ...rows.map((r) => r.length))
      body = <TablePreview rows={shown} />
      caption = [
        more || text.cut || rows.length > shown.length ? `The first ${Math.max(0, shown.length - 1)} rows` : '',
        cols > TABLE_COLS ? `${TABLE_COLS} of ${cols} columns` : '',
      ].filter(Boolean).join(' · ')
    } else {
      const all = shownText(text.text, ext, text.cut).split('\n')
      body = (
        <div ref={textBox} className={`max-h-64 overflow-hidden ${overflows ? FADE : ''}`}>
          <pre className="whitespace-pre-wrap break-words px-3 py-2.5 font-mono text-[11px] leading-relaxed text-zinc-300">
            {all.slice(0, TEXT_LINES).join('\n')}
          </pre>
        </div>
      )
      if (text.cut || overflows || all.length > TEXT_LINES) caption = 'The start of the file'
    }
  } else {
    body = <Empty text="No preview for this kind of file." hint={canOpen ? undefined : 'Download it to open it on your computer.'} />
  }

  return (
    <section className="mt-5" aria-label="Preview">
      <div className="mb-1.5 flex items-baseline gap-2">
        <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">Preview</span>
        {caption && <span className="text-[11px] text-zinc-600">{caption.charAt(0).toUpperCase() + caption.slice(1)}</span>}
        {canOpen && ((isText && text.state === 'ready') || (group === 'pdf' && !failed) || (workbook && (sheet.state === 'ready' || sheetTooBig))) && (
          <button type="button" onClick={open} className="ml-auto text-[11px] text-violet-600 hover:underline dark:text-violet-400">
            Open
          </button>
        )}
      </div>
      <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/60">{body}</div>
    </section>
  )
}

/** A page as a small picture of itself: laid out PAGE_WIDTH wide in a
 *  sandboxed frame (never same-origin; someone else's runs no scripts and
 *  loads nothing), scaled to the preview's width. A click opens it. While the
 *  viewer shows the page (`paused`), this copy is taken down, so its scripts
 *  don't run twice. */
function PagePreview({ html, untrusted, paused, canOpen, onOpen }: {
  html: string; untrusted: boolean; paused: boolean; canOpen: boolean; onOpen: () => void
}) {
  const [box, width] = useMeasured(widthOf, 0)
  const scale = width ? Math.min(1, width / PAGE_WIDTH) : 0
  const frame = htmlFrame(html, untrusted)
  return (
    <div ref={box} className="relative overflow-hidden bg-white" style={{ height: PAGE_PREVIEW_HEIGHT }}>
      {scale > 0 && !paused && (
        <iframe
          srcDoc={frame.srcDoc}
          sandbox={frame.sandbox}
          allow="autoplay 'none'"
          title="Page preview"
          tabIndex={-1}
          aria-hidden
          className="pointer-events-none absolute left-0 top-0 origin-top-left border-0"
          style={{ width: PAGE_WIDTH, height: PAGE_PREVIEW_HEIGHT / scale, transform: `scale(${scale})` }}
        />
      )}
      {canOpen && (
        <button type="button" onClick={onOpen} aria-label="Open the page" className="absolute inset-0 cursor-pointer" />
      )}
    </div>
  )
}

function Empty({ text, hint }: { text: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center gap-1.5 px-4 py-6 text-center text-xs text-zinc-500">
      <FileQuestion className="h-5 w-5 text-zinc-600" />
      <span>{text}</span>
      {hint && <span className="text-zinc-600">{hint}</span>}
    </div>
  )
}

function TablePreview({ rows }: { rows: string[][] }) {
  const [head, ...rest] = rows
  if (!head) return <Empty text="The file is empty." />
  const cols = Math.min(TABLE_COLS, Math.max(...rows.map((r) => r.length)))
  const cell = 'max-w-[9rem] truncate px-2 py-1 text-left'
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px]">
        <thead className="bg-zinc-800/60 text-zinc-300">
          <tr>{Array.from({ length: cols }, (_, i) => <th key={i} className={`${cell} font-medium`} title={head[i]}>{head[i] ?? ''}</th>)}</tr>
        </thead>
        <tbody className="text-zinc-400">
          {rest.map((r, ri) => (
            <tr key={ri} className="border-t border-zinc-800/70">
              {Array.from({ length: cols }, (_, i) => <td key={i} className={cell} title={r[i]}>{r[i] ?? ''}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
