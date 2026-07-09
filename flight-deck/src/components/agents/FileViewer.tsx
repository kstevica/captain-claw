import { useState, useEffect, useCallback, useRef } from 'react'
import {
  X, Download, Loader2, AlertCircle, Maximize2, Minimize2,
  ChevronLeft, ChevronRight, Copy, Check, Pencil, Save,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { AgentFile } from '../../services/fileTransfer'
import { getViewUrl, getDownloadUrl, formatSize, getFileTypeGroup, saveFileContent } from '../../services/fileTransfer'
import { CodeEditor } from './CodeEditor'

// File groups whose text content can be edited in place.
const EDITABLE_GROUPS = new Set(['markdown', 'code', 'data', 'text', 'html'])

interface FileViewerProps {
  file: AgentFile
  host: string
  port: number
  auth: string
  onClose: () => void
  /** Open straight into edit mode (e.g. the file-list Edit button) */
  startInEdit?: boolean
  /** Navigate to adjacent files */
  onPrev?: () => void
  onNext?: () => void
  hasPrev?: boolean
  hasNext?: boolean
}

export function FileViewer({ file, host, port, auth, startInEdit, onClose, onPrev, onNext, hasPrev, hasNext }: FileViewerProps) {
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
  const editable = content !== null && EDITABLE_GROUPS.has(group)
  const dirty = editing && draft !== content
  // Consume startInEdit once (on the file it was opened for) — file nav inside
  // the viewer shouldn't re-trigger edit mode.
  const autoEditRef = useRef(!!startInEdit)
  const viewUrl = getViewUrl(host, port, file.physical, auth)
  const downloadUrl = getDownloadUrl(host, port, file.physical, auth)

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
        if (!resp.ok) throw new Error(`Failed to load: ${resp.status}`)
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
  }, [file.physical, viewUrl, group])

  const startEdit = () => { setDraft(content ?? ''); setSaveError(''); setEditing(true) }
  const cancelEdit = () => { setEditing(false); setSaveError('') }

  const handleSave = useCallback(async () => {
    if (saving) return
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
  }, [saving, host, port, auth, file.physical, draft])

  // Keyboard: while editing, Esc cancels and ⌘/Ctrl+S saves (arrows type, not
  // navigate). Otherwise Esc closes and arrows move between files.
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (editing) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') { e.preventDefault(); handleSave() }
      else if (e.key === 'Escape') { e.preventDefault(); cancelEdit() }
      return
    }
    if (e.key === 'Escape') onClose()
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
                srcDoc={content}
                title={file.filename}
                className="w-full border-0"
                style={{ height: maximized ? 'calc(95vh - 52px)' : 'calc(85vh - 52px)' }}
                sandbox="allow-scripts allow-same-origin"
              />
            </div>
          )}

          {!loading && !error && content !== null && group === 'markdown' && (
            <div className="fd-file-markdown p-6">
              <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
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
