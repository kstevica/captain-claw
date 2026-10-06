import { useState, useEffect, useCallback, useRef } from 'react'
import {
  X, Download, Loader2, AlertCircle, Maximize2, Minimize2,
  ChevronLeft, ChevronRight, Copy, Check, Pencil, Save,
} from 'lucide-react'
import Markdown, { defaultUrlTransform } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { AgentFile } from '../../services/fileTransfer'
import { getViewUrl, getDownloadUrl, formatSize, getFileTypeGroup, saveFileContent } from '../../services/fileTransfer'
import { isMemberCreated } from '../../utils/sharedWorkspace'
import { CodeEditor } from './CodeEditor'

// File groups whose text content can be edited in place.
const EDITABLE_GROUPS = new Set(['markdown', 'code', 'data', 'text', 'html'])

// Markdown somebody else wrote (another member's file, or a member's file in
// the owner's panels) must not make the viewer's browser fetch remote images —
// a tracking pixel would tell its author who opened it, and when. An image
// src that can reach another host (any scheme, `//`, or the `/\` spelling
// browsers read as `//`) is dropped; links still render. Browsers drop tab/LF/CR
// anywhere and every leading C0 control or space (`<\x01//host>` is a valid
// markdown destination and loads from `host`), so the check does too.
function untrustedUrlTransform(url: string, key: string): string {
  // eslint-disable-next-line no-control-regex
  if (key === 'src' && /^([a-z][a-z0-9+.-]*:|[\\/]{2})/i.test(url.replace(/[\t\n\r]/g, '').replace(/^[\x00-\x20\s]+/, ''))) return ''
  return defaultUrlTransform(url)
}

// HTML somebody else wrote renders inert: no scripts (an empty sandbox), and a
// CSP that lets it load nothing from anywhere — no remote image, stylesheet or
// font that would tell its author who opened it. Inline styles and data:
// images still show. The viewer's own (or, in the owner's panels, the owner's)
// HTML keeps running scripts — never same-origin.
const INERT_HTML_CSP = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">`

function htmlFrame(content: string, untrusted: boolean): { srcDoc: string; sandbox: string } {
  return untrusted
    ? { srcDoc: INERT_HTML_CSP + content, sandbox: '' }
    : { srcDoc: content, sandbox: 'allow-scripts' }
}

interface FileViewerProps {
  file: AgentFile
  /** The owner's agent endpoint (owner routes). Not needed with `urls`. */
  host?: string
  port?: number
  auth?: string
  /** Ready-made view/download URLs (a member's shared-agent routes), used
   *  instead of building owner URLs from host/port/auth. */
  urls?: { view: string; download: string }
  /** No Edit/Save (and `startInEdit` is ignored) — a member's panel. */
  readOnly?: boolean
  onClose: () => void
  /** Open straight into edit mode (e.g. the file-list Edit button) */
  startInEdit?: boolean
  /** Navigate to adjacent files */
  onPrev?: () => void
  onNext?: () => void
  hasPrev?: boolean
  hasNext?: boolean
}

export function FileViewer({ file, host = '', port = 0, auth = '', urls, readOnly, startInEdit, onClose, onPrev, onNext, hasPrev, hasNext }: FileViewerProps) {
  const [content, setContent] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [maximized, setMaximized] = useState(false)
  const [copied, setCopied] = useState(false)
  // Edit mode
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [savedTick, setSavedTick] = useState(false)

  const group = getFileTypeGroup(file)
  const editable = !readOnly && content !== null && EDITABLE_GROUPS.has(group)
  const dirty = editing && draft !== content
  // Consume startInEdit once (on the file it was opened for) — file nav inside
  // the viewer shouldn't re-trigger edit mode.
  const autoEditRef = useRef(!!startInEdit && !readOnly)
  const viewUrl = urls ? urls.view : getViewUrl(host, port, file.physical, auth)
  const downloadUrl = urls ? urls.download : getDownloadUrl(host, port, file.physical, auth)
  const sharedRoutes = !!urls
  // Another file (Prev/Next): drop the last one's content in this same render.
  // The fetch effect below runs only after a commit, so for one render the old
  // text would show under the new file's trust — someone else's HTML running,
  // or their markdown's remote images loading, as if it were the viewer's own.
  const [contentFor, setContentFor] = useState(viewUrl)
  if (contentFor !== viewUrl) {
    setContentFor(viewUrl)
    setContent(null)
    setLoading(true)
  }
  // Someone else's file: in a member's panel anything not theirs, in the
  // owner's panels what a member added. Its markdown loads no remote images.
  const untrusted = file.source === 'shared' ? file.created_by?.kind !== 'me' : isMemberCreated(file.created_by)

  // Fetch text content for text-based files
  useEffect(() => {
    setLoading(true)
    setError('')
    setContent(null)
    setCopied(false)
    setEditing(false)
    setSaveError('')

    if (group === 'image' || group === 'audio' || group === 'video') {
      // Binary media render straight from the URL — no text fetch (fetching a
      // binary as text just yields garbage in the fallback code view).
      setLoading(false)
      return
    }

    fetch(viewUrl)
      .then(async (resp) => {
        if (!resp.ok) {
          // A member's routes explain a refusal (e.g. the agent needs a restart).
          const detail = sharedRoutes
            ? await resp.json().then((b) => (typeof b?.detail === 'string' ? b.detail : ''), () => '')
            : ''
          throw new Error(detail || `Failed to load: ${resp.status}`)
        }
        const text = await resp.text()
        setContent(text)
        // Opened via the Edit button → drop straight into edit mode (once).
        if (autoEditRef.current && EDITABLE_GROUPS.has(group)) {
          autoEditRef.current = false
          setDraft(text)
          setEditing(true)
        }
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [file.physical, viewUrl, group, sharedRoutes])

  const startEdit = () => { setDraft(content ?? ''); setSaveError(''); setEditing(true) }
  const cancelEdit = () => { setEditing(false); setSaveError('') }

  const handleSave = useCallback(async () => {
    if (saving || readOnly) return
    setSaving(true)
    setSaveError('')
    try {
      await saveFileContent(host, port, auth, file.physical, draft)
      setContent(draft)        // preview now reflects the saved content
      setEditing(false)
      setSavedTick(true)
      setTimeout(() => setSavedTick(false), 2000)
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }, [saving, readOnly, host, port, auth, file.physical, draft])

  // Keyboard: while editing, Esc cancels and ⌘/Ctrl+S saves (arrows type, not
  // navigate). Otherwise Esc closes and arrows move between files. Esc is
  // marked handled, so a dialog hosting the viewer (a member's Files overlay)
  // knows not to close as well.
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (editing) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') { e.preventDefault(); handleSave() }
      else if (e.key === 'Escape') { e.preventDefault(); cancelEdit() }
      return
    }
    if (e.key === 'Escape') { e.preventDefault(); onClose() }
    if (e.key === 'ArrowLeft' && onPrev && hasPrev) onPrev()
    if (e.key === 'ArrowRight' && onNext && hasNext) onNext()
  }, [editing, handleSave, onClose, onPrev, onNext, hasPrev, hasNext])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const handleCopy = async () => {
    if (!content) return
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch { /* ignore */ }
  }

  const sizeClass = maximized
    ? 'w-[95vw] max-h-[95vh]'
    : 'w-[900px] max-h-[85vh]'

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className={`flex flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl transition-all duration-200 ${sizeClass}`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2.5 shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <h3 className="text-sm font-semibold truncate">{file.filename}</h3>
            <span className="text-[11px] text-zinc-500 font-mono shrink-0">{file.extension}</span>
            <span className="text-[11px] text-zinc-600 shrink-0">{formatSize(file.size)}</span>
          </div>
          <div className="flex items-center gap-0.5 shrink-0">
            {/* Prev / Next */}
            {(hasPrev || hasNext) && (
              <>
                <button
                  onClick={onPrev}
                  disabled={!hasPrev || editing}
                  className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-25 disabled:hover:bg-transparent"
                  title="Previous file (Left arrow)"
                >
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <button
                  onClick={onNext}
                  disabled={!hasNext || editing}
                  className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-25 disabled:hover:bg-transparent"
                  title="Next file (Right arrow)"
                >
                  <ChevronRight className="h-4 w-4" />
                </button>
                <div className="w-px h-4 bg-zinc-800 mx-1" />
              </>
            )}
            {/* Edit / Save / Cancel */}
            {editing ? (
              <>
                <button
                  onClick={handleSave}
                  disabled={saving || !dirty}
                  className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-emerald-300 hover:bg-emerald-600/20 disabled:opacity-40 disabled:hover:bg-transparent"
                  title="Save (⌘/Ctrl+S)"
                >
                  {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                  {dirty ? 'Save' : 'Saved'}
                </button>
                <button
                  onClick={cancelEdit}
                  className="rounded px-2 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                  title="Cancel (Esc)"
                >
                  Cancel
                </button>
                <div className="w-px h-4 bg-zinc-800 mx-1" />
              </>
            ) : editable ? (
              <>
                <button
                  onClick={startEdit}
                  className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                  title="Edit file"
                >
                  {savedTick ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Pencil className="h-3.5 w-3.5" />}
                  {savedTick ? 'Saved' : 'Edit'}
                </button>
                <div className="w-px h-4 bg-zinc-800 mx-1" />
              </>
            ) : null}
            {/* Copy (text content only) */}
            {content !== null && (
              <button
                onClick={handleCopy}
                className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
                title="Copy content"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            )}
            {/* Download */}
            <button
              onClick={() => {
                const a = document.createElement('a')
                a.href = downloadUrl
                a.download = file.filename
                document.body.appendChild(a)
                a.click()
                document.body.removeChild(a)
              }}
              className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
              title="Download"
            >
              <Download className="h-3.5 w-3.5" />
            </button>
            {/* Maximize */}
            <button
              onClick={() => setMaximized(!maximized)}
              className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
              title={maximized ? 'Restore' : 'Maximize'}
            >
              {maximized ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
            </button>
            {/* Close */}
            <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Close (Esc)">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto">
          {editing ? (
            <div className="flex flex-col" style={{ height: maximized ? 'calc(95vh - 52px)' : 'calc(85vh - 52px)' }}>
              {saveError && (
                <div className="flex items-center gap-2 px-4 py-2 text-xs text-red-300 bg-red-500/10 border-b border-red-500/20 shrink-0">
                  <AlertCircle className="h-3.5 w-3.5 shrink-0" /> {saveError}
                </div>
              )}
              <div className="flex-1 min-h-0">
                <CodeEditor
                  value={draft}
                  onChange={setDraft}
                  extension={file.extension}
                  storageKey={file.physical}
                />
              </div>
            </div>
          ) : (<>
          {loading && (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 px-6 py-8 text-sm text-red-400">
              <AlertCircle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          )}

          {!loading && !error && group === 'image' && (
            <div className="flex items-center justify-center p-6 bg-zinc-900/50 min-h-[300px]">
              <img
                src={viewUrl}
                alt={file.filename}
                className="max-w-full max-h-[70vh] object-contain rounded-lg"
                style={{ imageRendering: file.extension === '.svg' ? 'auto' : undefined }}
              />
            </div>
          )}

          {!loading && !error && group === 'video' && (
            <div className="flex items-center justify-center bg-black p-4 min-h-[300px]">
              <video src={viewUrl} controls className="max-w-full max-h-[75vh] rounded-lg" />
            </div>
          )}

          {!loading && !error && group === 'audio' && (
            <div className="flex flex-col items-center justify-center gap-4 p-10 min-h-[200px]">
              <span className="truncate text-sm text-zinc-400">{file.filename}</span>
              <audio src={viewUrl} controls className="w-full max-w-lg" />
            </div>
          )}

          {!loading && !error && content !== null && group === 'html' && (
            <div className="bg-white min-h-[300px]">
              <iframe
                srcDoc={htmlFrame(content, untrusted).srcDoc}
                title={file.filename}
                className="w-full border-0"
                style={{ height: maximized ? 'calc(95vh - 52px)' : 'calc(85vh - 52px)' }}
                // Never same-origin, for any file: a srcdoc frame would
                // otherwise run in Flight Deck's origin, with its storage and
                // the viewer's token — and a shared agent's saved/ folder holds
                // files other people wrote. Theirs get no scripts at all.
                sandbox={htmlFrame(content, untrusted).sandbox}
              />
            </div>
          )}

          {!loading && !error && content !== null && group === 'markdown' && (
            <div className="fd-file-markdown p-6">
              {untrusted ? (
                <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform}>{content}</Markdown>
              ) : (
                <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
              )}
            </div>
          )}

          {!loading && !error && content !== null && group === 'data' && file.extension === '.json' && (
            <pre className="p-6 text-xs font-mono text-zinc-300 leading-relaxed whitespace-pre-wrap break-words">
              {(() => { try { return JSON.stringify(JSON.parse(content), null, 2) } catch { return content } })()}
            </pre>
          )}

          {!loading && !error && content !== null && !['html', 'markdown', 'image', 'audio', 'video'].includes(group) && !(group === 'data' && file.extension === '.json') && (
            <div className="relative">
              {/* Line numbers + code */}
              <pre className="p-6 text-xs font-mono leading-relaxed overflow-x-auto">
                {content.split('\n').map((line, i) => (
                  <div key={i} className="flex">
                    <span className="w-10 shrink-0 text-right pr-4 text-zinc-700 select-none">{i + 1}</span>
                    <span className="text-zinc-300 whitespace-pre-wrap break-all">{line}</span>
                  </div>
                ))}
              </pre>
            </div>
          )}
          </>)}
        </div>
      </div>
    </div>
  )
}
