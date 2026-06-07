import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Prism from 'prismjs'
// Language grammars (markup/css/clike/javascript ship in the core bundle).
import 'prismjs/components/prism-json'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-yaml'
import 'prismjs/components/prism-markdown'
import 'prismjs/components/prism-typescript'
import './prism-flow'   // registers Prism.languages.flow
import { X, ChevronUp, ChevronDown } from 'lucide-react'
import './code-editor.css'

const LANG_BY_EXT: Record<string, string> = {
  '.md': 'markdown', '.markdown': 'markdown',
  '.html': 'markup', '.htm': 'markup', '.xml': 'markup', '.svg': 'markup',
  '.css': 'css',
  '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
  '.ts': 'typescript', '.tsx': 'typescript',
  '.json': 'json',
  '.py': 'python',
  '.sh': 'bash', '.bash': 'bash',
  '.yml': 'yaml', '.yaml': 'yaml',
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

interface CodeEditorProps {
  value: string
  onChange: (value: string) => void
  extension?: string
  /** Explicit Prism grammar id (e.g. 'flow'); overrides extension detection */
  language?: string
  /** Stable key (the file's physical path / flow id) for remembering the cursor */
  storageKey: string
}

export function CodeEditor({ value, onChange, extension, language, storageKey }: CodeEditorProps) {
  const taRef = useRef<HTMLTextAreaElement>(null)
  const preRef = useRef<HTMLPreElement>(null)
  const gutterRef = useRef<HTMLDivElement>(null)
  const cursorKey = `fdEditCursor:${storageKey}`

  const lang = language || LANG_BY_EXT[(extension || '').toLowerCase()] || ''

  const lineCount = useMemo(() => {
    let n = 1
    for (let i = 0; i < value.length; i++) if (value.charCodeAt(i) === 10) n++
    return n
  }, [value])

  const highlighted = useMemo(() => {
    const grammar = lang ? Prism.languages[lang] : undefined
    let html = grammar ? Prism.highlight(value, grammar, lang) : escapeHtml(value)
    // A trailing newline collapses in the <pre> unless we pad it, which would
    // misalign the last line against the textarea.
    if (value.endsWith('\n')) html += '\n'
    return html
  }, [value, lang])

  const syncScroll = useCallback(() => {
    const ta = taRef.current, pre = preRef.current, gut = gutterRef.current
    if (ta && pre) { pre.scrollTop = ta.scrollTop; pre.scrollLeft = ta.scrollLeft }
    if (ta && gut) { gut.scrollTop = ta.scrollTop }
  }, [])

  // Keep the highlight layer aligned after every value change re-render.
  useEffect(() => { syncScroll() }, [highlighted, syncScroll])

  // ── Cursor memory: restore on open, persist on move ────────────────
  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    const raw = sessionStorage.getItem(cursorKey)
    const saved = raw ? Math.min(parseInt(raw, 10) || 0, value.length) : 0
    // Restore to the BEGINNING of the saved row (column 0), not the exact column.
    const pos = value.lastIndexOf('\n', saved - 1) + 1
    requestAnimationFrame(() => {
      try {
        ta.focus()
        ta.setSelectionRange(pos, pos)
        scrollOffsetIntoView(pos, pos)
      } catch { /* ignore */ }
    })
    // Only when the file (storageKey) changes — not on every keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursorKey])

  const persistCursor = useCallback(() => {
    const ta = taRef.current
    if (ta) {
      try { sessionStorage.setItem(cursorKey, String(ta.selectionStart)) } catch { /* ignore */ }
    }
  }, [cursorKey])

  // Scroll a character offset into view (textarea doesn't do this for
  // programmatic selections). No-wrap ⇒ visual line == logical line.
  const scrollOffsetIntoView = useCallback((start: number, end: number) => {
    const ta = taRef.current
    if (!ta) return
    const before = value.slice(0, start)
    const line = (before.match(/\n/g) || []).length
    const cs = getComputedStyle(ta)
    let lh = parseFloat(cs.lineHeight)
    if (!Number.isFinite(lh)) lh = parseFloat(cs.fontSize) * 1.6
    const targetTop = Math.max(0, (line - 3) * lh)
    if (ta.scrollTop > targetTop || targetTop > ta.scrollTop + ta.clientHeight - lh * 4) {
      ta.scrollTop = targetTop
    }
    // Horizontal: keep the match column roughly centered.
    const col = end - (before.lastIndexOf('\n') + 1)
    let chW = 7.0
    try {
      const c = document.createElement('canvas').getContext('2d')
      if (c) { c.font = `${cs.fontSize} ${cs.fontFamily}`; chW = c.measureText('0').width || chW }
    } catch { /* ignore */ }
    const x = col * chW
    if (x < ta.scrollLeft || x > ta.scrollLeft + ta.clientWidth - chW * 4) {
      ta.scrollLeft = Math.max(0, x - ta.clientWidth / 2)
    }
    syncScroll()
  }, [value, syncScroll])

  // ── Find ───────────────────────────────────────────────────────────
  const [findOpen, setFindOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [matches, setMatches] = useState<number[]>([])
  const [active, setActive] = useState(0)
  const findInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!query) { setMatches([]); setActive(0); return }
    const q = query.toLowerCase()
    const hay = value.toLowerCase()
    const out: number[] = []
    let i = hay.indexOf(q)
    while (i !== -1 && out.length < 5000) {
      out.push(i)
      i = hay.indexOf(q, i + Math.max(1, q.length))
    }
    setMatches(out)
    setActive(0)
  }, [query, value])

  const jumpTo = useCallback((idx: number) => {
    const ta = taRef.current
    if (!ta || matches.length === 0) return
    const n = ((idx % matches.length) + matches.length) % matches.length
    setActive(n)
    const start = matches[n]
    const end = start + query.length
    ta.focus()
    ta.setSelectionRange(start, end)
    scrollOffsetIntoView(start, end)
  }, [matches, query, scrollOffsetIntoView])

  const openFind = useCallback(() => {
    const ta = taRef.current
    setFindOpen(true)
    if (ta && ta.selectionStart !== ta.selectionEnd) {
      setQuery(value.slice(ta.selectionStart, ta.selectionEnd))
    }
    requestAnimationFrame(() => findInputRef.current?.select())
  }, [value])

  const closeFind = useCallback(() => {
    setFindOpen(false)
    taRef.current?.focus()
  }, [])

  const onEditorKeyDown = useCallback((e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'f') {
      e.preventDefault()
      e.stopPropagation()
      openFind()
    }
  }, [openFind])

  const onFindKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') { e.preventDefault(); jumpTo(active + (e.shiftKey ? -1 : 1)) }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeFind() }
  }, [active, jumpTo, closeFind])

  return (
    <div className="cc-code-editor">
      {findOpen && (
        <div className="cc-find" onClick={(e) => e.stopPropagation()}>
          <input
            ref={findInputRef}
            value={query}
            placeholder="Find"
            spellCheck={false}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onFindKeyDown}
          />
          <span className="cc-count">{matches.length ? `${active + 1}/${matches.length}` : (query ? '0/0' : '')}</span>
          <button disabled={!matches.length} onClick={() => jumpTo(active - 1)} title="Previous (Shift+Enter)">
            <ChevronUp className="h-3.5 w-3.5" />
          </button>
          <button disabled={!matches.length} onClick={() => jumpTo(active + 1)} title="Next (Enter)">
            <ChevronDown className="h-3.5 w-3.5" />
          </button>
          <button onClick={closeFind} title="Close (Esc)">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
      <div className="cc-gutter" ref={gutterRef} aria-hidden="true">
        <div className="cc-gutter-inner">
          {Array.from({ length: lineCount }, (_, i) => (
            <div className="cc-gutter-line" key={i}>{i + 1}</div>
          ))}
        </div>
      </div>
      <div className="cc-code-wrap">
        <pre ref={preRef} aria-hidden="true">
          <code className={lang ? `language-${lang}` : undefined} dangerouslySetInnerHTML={{ __html: highlighted }} />
        </pre>
        <textarea
          ref={taRef}
          value={value}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          wrap="off"
          onChange={(e) => onChange(e.target.value)}
          onScroll={syncScroll}
          onKeyDown={onEditorKeyDown}
          onKeyUp={persistCursor}
          onClick={persistCursor}
          onSelect={persistCursor}
          placeholder="Empty file"
        />
      </div>
    </div>
  )
}
