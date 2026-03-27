import { useState, useEffect, useCallback } from 'react'
import {
  X, Download, Loader2, AlertCircle, Maximize2, Minimize2,
  ChevronLeft, ChevronRight, Copy, Check,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { AgentFile } from '../../services/fileTransfer'
import { getViewUrl, getDownloadUrl, formatSize, getFileTypeGroup } from '../../services/fileTransfer'

interface FileViewerProps {
  file: AgentFile
  host: string
  port: number
  auth: string
  onClose: () => void
  /** Navigate to adjacent files */
  onPrev?: () => void
  onNext?: () => void
  hasPrev?: boolean
  hasNext?: boolean
}

export function FileViewer({ file, host, port, auth, onClose, onPrev, onNext, hasPrev, hasNext }: FileViewerProps) {
  const [content, setContent] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [maximized, setMaximized] = useState(false)
  const [copied, setCopied] = useState(false)

  const group = getFileTypeGroup(file)
  const viewUrl = getViewUrl(host, port, file.physical, auth)
  const downloadUrl = getDownloadUrl(host, port, file.physical, auth)

  // Fetch text content for text-based files
  useEffect(() => {
    setLoading(true)
    setError('')
    setContent(null)
    setCopied(false)

    if (group === 'image') {
      // Images don't need text fetch
      setLoading(false)
      return
    }

    fetch(viewUrl)
      .then(async (resp) => {
        if (!resp.ok) throw new Error(`Failed to load: ${resp.status}`)
        const text = await resp.text()
        setContent(text)
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [file.physical, viewUrl, group])

  // Keyboard navigation
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
    if (e.key === 'ArrowLeft' && onPrev && hasPrev) onPrev()
    if (e.key === 'ArrowRight' && onNext && hasNext) onNext()
  }, [onClose, onPrev, onNext, hasPrev, hasNext])

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
                  disabled={!hasPrev}
                  className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-25 disabled:hover:bg-transparent"
                  title="Previous file (Left arrow)"
                >
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <button
                  onClick={onNext}
                  disabled={!hasNext}
                  className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-25 disabled:hover:bg-transparent"
                  title="Next file (Right arrow)"
                >
                  <ChevronRight className="h-4 w-4" />
                </button>
                <div className="w-px h-4 bg-zinc-800 mx-1" />
              </>
            )}
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
            <div className="p-6 prose prose-invert prose-sm max-w-none prose-headings:text-zinc-200 prose-p:text-zinc-300 prose-a:text-violet-400 prose-code:text-emerald-300 prose-code:bg-zinc-800 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-zinc-900 prose-pre:border prose-pre:border-zinc-800 prose-strong:text-zinc-200 prose-table:text-zinc-300 prose-th:text-zinc-200 prose-td:border-zinc-800 prose-th:border-zinc-800 prose-hr:border-zinc-800">
              <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
            </div>
          )}

          {!loading && !error && content !== null && group === 'data' && file.extension === '.json' && (
            <pre className="p-6 text-xs font-mono text-zinc-300 leading-relaxed whitespace-pre-wrap break-words">
              {(() => { try { return JSON.stringify(JSON.parse(content), null, 2) } catch { return content } })()}
            </pre>
          )}

          {!loading && !error && content !== null && !['html', 'markdown', 'image'].includes(group) && !(group === 'data' && file.extension === '.json') && (
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
        </div>
      </div>
    </div>
  )
}
