import { useCallback, useEffect, useState } from 'react'
import {
  X, Download, Loader2, AlertCircle, Maximize2, Minimize2,
  ChevronLeft, ChevronRight, Copy, Check, Pencil, Save,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { AgentFile } from '../../services/fileTransfer'
import { formatSize, getFileTypeGroup } from '../../services/fileTransfer'
import { CodeEditor } from '../agents/CodeEditor'
import { useVFSStore, extOf } from '../../stores/vfsStore'

// Mirrors components/agents/FileViewer EDITABLE_GROUPS.
const EDITABLE_GROUPS = new Set(['markdown', 'code', 'data', 'text', 'html'])

interface Props {
  onClose: () => void
  onPrev?: () => void
  onNext?: () => void
  hasPrev?: boolean
  hasNext?: boolean
}

export function VFSFileViewer({ onClose, onPrev, onNext, hasPrev, hasNext }: Props) {
  const s = useVFSStore()
  const f = s.file
  const [maximized, setMaximized] = useState(false)
  const [copied, setCopied] = useState(false)

  const ext = f ? extOf(f.name) : ''
  // getFileTypeGroup only reads .extension — feed it a minimal AgentFile.
  const group = getFileTypeGroup({ extension: ext, filename: f?.name || '' } as AgentFile)
  const editable = !!f && !f.binary && !f.truncated && f.text !== undefined && EDITABLE_GROUPS.has(group)
  const dirty = s.editing && s.draft !== (f?.text ?? '')

  const handleSave = useCallback(() => { s.saveFile() }, [s])

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (s.editing) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') { e.preventDefault(); handleSave() }
      else if (e.key === 'Escape') { e.preventDefault(); useVFSStore.setState({ editing: false }) }
      return
    }
    if (e.key === 'Escape') onClose()
    if (e.key === 'ArrowLeft' && onPrev && hasPrev) onPrev()
    if (e.key === 'ArrowRight' && onNext && hasNext) onNext()
  }, [s.editing, handleSave, onClose, onPrev, onNext, hasPrev, hasNext])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  if (!f) return null

  const handleCopy = async () => {
    if (!f.text) return
    try {
      await navigator.clipboard.writeText(f.text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch { /* ignore */ }
  }

  // Maximized is a fixed tall box (95vh) so the viewer fills the screen even
  // for short files; the restored size stays compact (height hugs content).
  const sizeClass = maximized ? 'h-[95vh] w-[95vw]' : 'w-[900px] max-h-[85vh]'
  const frameH = maximized ? 'calc(95vh - 52px)' : 'calc(85vh - 52px)'

  const downloadEntry = { name: f.name, type: 'file' as const, path: f.path, project: f.project, size: f.size, mtime: 0 }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className={`flex flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl transition-all duration-200 ${sizeClass}`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2.5 shrink-0">
          <div className="flex min-w-0 items-center gap-2">
            <h3 className="truncate text-sm font-semibold">{f.name}</h3>
            <span className="shrink-0 font-mono text-[11px] text-zinc-500">{ext}</span>
            <span className="shrink-0 text-[11px] text-zinc-600">{formatSize(f.size)}</span>
            <span className="shrink-0 truncate text-[11px] text-violet-400/80">vfs:{f.project}/{f.path}</span>
          </div>
          <div className="flex shrink-0 items-center gap-0.5">
            {(hasPrev || hasNext) && (
              <>
                <button onClick={onPrev} disabled={!hasPrev || s.editing}
                  className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-25 disabled:hover:bg-transparent"
                  title="Previous file (Left arrow)"><ChevronLeft className="h-4 w-4" /></button>
                <button onClick={onNext} disabled={!hasNext || s.editing}
                  className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-25 disabled:hover:bg-transparent"
                  title="Next file (Right arrow)"><ChevronRight className="h-4 w-4" /></button>
                <div className="mx-1 h-4 w-px bg-zinc-800" />
              </>
            )}
            {s.editing ? (
              <>
                <button onClick={handleSave} disabled={s.saving || !dirty}
                  className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-emerald-300 hover:bg-emerald-600/20 disabled:opacity-40 disabled:hover:bg-transparent"
                  title="Save (⌘/Ctrl+S)">
                  {s.saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                  {dirty ? 'Save' : 'Saved'}
                </button>
                <button onClick={() => useVFSStore.setState({ editing: false })}
                  className="rounded px-2 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200" title="Cancel (Esc)">Cancel</button>
                <div className="mx-1 h-4 w-px bg-zinc-800" />
              </>
            ) : editable ? (
              <>
                <button onClick={s.startEdit}
                  className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200" title="Edit file">
                  <Pencil className="h-3.5 w-3.5" /> Edit
                </button>
                <div className="mx-1 h-4 w-px bg-zinc-800" />
              </>
            ) : null}
            {!f.binary && f.text !== '' && (
              <button onClick={handleCopy} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Copy content">
                {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            )}
            <button onClick={() => s.download(downloadEntry)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Download">
              <Download className="h-3.5 w-3.5" />
            </button>
            <button onClick={() => setMaximized(!maximized)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title={maximized ? 'Restore' : 'Maximize'}>
              {maximized ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
            </button>
            <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Close (Esc)">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto">
          {s.editing ? (
            <div className="flex flex-col" style={{ height: frameH }}>
              {s.fileError && (
                <div className="flex shrink-0 items-center gap-2 border-b border-red-500/20 bg-red-500/10 px-4 py-2 text-xs text-red-300">
                  <AlertCircle className="h-3.5 w-3.5 shrink-0" /> {s.fileError}
                </div>
              )}
              <div className="min-h-0 flex-1">
                <CodeEditor value={s.draft} onChange={s.setDraft} extension={ext} storageKey={`vfs:${f.project}/${f.path}`} />
              </div>
            </div>
          ) : (<>
            {s.fileLoading && (
              <div className="flex items-center justify-center py-20"><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></div>
            )}
            {s.fileError && (
              <div className="flex items-center gap-2 px-6 py-8 text-sm text-red-400"><AlertCircle className="h-4 w-4 shrink-0" />{s.fileError}</div>
            )}

            {!s.fileLoading && !s.fileError && group === 'image' && s.blobUrl && (
              <div className="flex min-h-[300px] items-center justify-center bg-zinc-900/50 p-6">
                <img src={s.blobUrl} alt={f.name} className="max-h-[70vh] max-w-full rounded-lg object-contain" />
              </div>
            )}

            {!s.fileLoading && !s.fileError && f.truncated && (
              <div className="px-6 py-16 text-center text-sm text-zinc-500">
                File too large to preview ({formatSize(f.size)}) —{' '}
                <button onClick={() => s.download(downloadEntry)} className="text-violet-400 hover:underline">download</button>.
              </div>
            )}

            {!s.fileLoading && !s.fileError && f.binary && group !== 'image' && (
              <div className="px-6 py-16 text-center text-sm text-zinc-500">
                Binary file —{' '}
                <button onClick={() => s.download(downloadEntry)} className="text-violet-400 hover:underline">download</button> to open.
              </div>
            )}

            {!s.fileLoading && !s.fileError && !f.binary && !f.truncated && group === 'html' && (
              <div className="min-h-[300px] bg-white">
                <iframe srcDoc={f.text} title={f.name} className="w-full border-0" style={{ height: frameH }} sandbox="allow-scripts allow-same-origin" />
              </div>
            )}

            {!s.fileLoading && !s.fileError && !f.binary && !f.truncated && group === 'markdown' && (
              <div className="fd-file-markdown p-6"><Markdown remarkPlugins={[remarkGfm]}>{f.text}</Markdown></div>
            )}

            {!s.fileLoading && !s.fileError && !f.binary && !f.truncated && group === 'data' && ext === '.json' && (
              <pre className="whitespace-pre-wrap break-words p-6 font-mono text-xs leading-relaxed text-zinc-300">
                {(() => { try { return JSON.stringify(JSON.parse(f.text), null, 2) } catch { return f.text } })()}
              </pre>
            )}

            {!s.fileLoading && !s.fileError && !f.binary && !f.truncated && !['html', 'markdown', 'image'].includes(group) && !(group === 'data' && ext === '.json') && (
              <pre className="overflow-x-auto p-6 font-mono text-xs leading-relaxed">
                {f.text.split('\n').map((line, i) => (
                  <div key={i} className="flex">
                    <span className="w-10 shrink-0 select-none pr-4 text-right text-zinc-700">{i + 1}</span>
                    <span className="whitespace-pre-wrap break-all text-zinc-300">{line}</span>
                  </div>
                ))}
              </pre>
            )}
          </>)}
        </div>
      </div>
    </div>
  )
}
