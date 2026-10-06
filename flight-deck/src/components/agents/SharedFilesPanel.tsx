import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import {
  FileText, Loader2, AlertCircle, FolderOpen, RefreshCw, Download, Eye,
  Search, Image, FileCode, FileSpreadsheet, Film, Music, Archive, Trash2, Upload,
} from 'lucide-react'
import { formatSize, getFileTypeGroup, isViewable } from '../../services/fileTransfer'
import {
  listSharedFiles, sharedFileUrl, uploadSharedFile, deleteSharedFile,
  type SharedFile, type SharedFilesResponse,
} from '../../services/sharedWorkspace'
import {
  MEMBER_FILES_NOTE, NO_SHARED_FILES, TRUNCATED_NOTE, UPLOAD_LABEL, UPLOAD_TITLE,
  deleteConfirmText, toViewerFile, uploadError,
} from '../../utils/sharedWorkspace'
import { FileViewer } from './FileViewer'
import { CreatorBadge } from './CreatorBadge'

// A shared agent's saved/ folder as a MEMBER sees it: every file in it, the
// owner's and every member's, each with who created it. They can view and
// download any of them, upload into their own folder, and delete what they
// added — nothing else (no pin, edit, deck or transfer). Everything goes
// through Flight Deck's member routes by the agent's ref; the agent decides
// what a member may delete (`can_delete` only decides which button shows).

const TYPE_ICONS: Record<string, typeof FileText> = {
  image: Image, video: Film, audio: Music, code: FileCode,
  data: FileSpreadsheet, archive: Archive,
}
const TYPE_COLORS: Record<string, string> = {
  image: 'text-blue-400', video: 'text-pink-400', audio: 'text-amber-400',
  code: 'text-emerald-400', data: 'text-cyan-400', archive: 'text-orange-400',
  html: 'text-orange-300', markdown: 'text-violet-400', pdf: 'text-red-400',
}

function SharedFileIcon({ file }: { file: SharedFile }) {
  const group = getFileTypeGroup(toViewerFile(file))
  const Icon = TYPE_ICONS[group] || FileText
  return <Icon className={`h-3.5 w-3.5 shrink-0 ${TYPE_COLORS[group] || 'text-zinc-500'}`} />
}

function IconBtn({ onClick, title, Icon, hoverClass }: {
  onClick: () => void
  title: string
  Icon: typeof FileText
  hoverClass?: string
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`rounded p-0.5 text-zinc-500 transition-colors hover:bg-zinc-800 ${hoverClass || 'hover:text-zinc-300'}`}
    >
      <Icon className="h-3.5 w-3.5" />
    </button>
  )
}

export function SharedFilesPanel({ agentRef, agentName, ownerName }: {
  agentRef: string
  agentName: string
  ownerName: string
}) {
  const [files, setFiles] = useState<SharedFile[]>([])
  const [truncated, setTruncated] = useState(false)
  const [upload, setUpload] = useState<SharedFilesResponse['upload'] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [viewing, setViewing] = useState<SharedFile | null>(null)
  const [uploading, setUploading] = useState(false)
  const [actionError, setActionError] = useState('')
  const [deleting, setDeleting] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Loaded on mount, on refresh and after an upload or delete — no polling.
  const loadFiles = useCallback(async () => {
    setError('')
    try {
      const r = await listSharedFiles(agentRef)
      setFiles(Array.isArray(r?.files) ? r.files : [])
      setTruncated(r?.truncated === true)
      setUpload(r?.upload && Array.isArray(r.upload.extensions) ? r.upload : null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [agentRef])
  const reload = useCallback(() => { setLoading(true); void loadFiles() }, [loadFiles])
  useEffect(() => { void loadFiles() }, [loadFiles])

  const shown = useMemo(() => {
    let r = [...files]
    if (search) {
      const q = search.toLowerCase()
      r = r.filter((f) =>
        f.filename.toLowerCase().includes(q) ||
        f.path.toLowerCase().includes(q) ||
        f.extension.toLowerCase().includes(q))
    }
    r.sort((a, b) => b.modified - a.modified)
    return r
  }, [files, search])

  const handlePick = async (file: File | undefined) => {
    if (!file || !upload) return
    setActionError('')
    const why = uploadError(file, upload)
    if (why) { setActionError(why); return }
    setUploading(true)
    try {
      await uploadSharedFile(agentRef, file, 'A')
      await loadFiles()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setUploading(false)
    }
  }

  const handleView = (f: SharedFile) => {
    if (getFileTypeGroup(toViewerFile(f)) === 'pdf') {
      // The browser's own PDF viewer, in a tab of its own.
      window.open(sharedFileUrl(agentRef, f.id, 'view'), '_blank', 'noopener')
    } else {
      setViewing(f)
    }
  }
  const handleDownload = (f: SharedFile) => {
    const a = document.createElement('a')
    a.href = sharedFileUrl(agentRef, f.id, 'download')
    a.download = f.filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }
  const handleDelete = async (f: SharedFile) => {
    if (!confirm(deleteConfirmText(f.filename))) return
    setActionError('')
    setDeleting(f.id)
    try {
      await deleteSharedFile(agentRef, f.id)
      if (viewing && viewing.id === f.id) setViewing(null)
      await loadFiles()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setDeleting(null)
    }
  }

  // Prev/next for the viewer, over the currently-visible viewables.
  const viewable = shown.filter((f) => isViewable(toViewerFile(f)))
  const viewIdx = viewing ? viewable.findIndex((f) => f.id === viewing.id) : -1

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2" title={`Files saved on ${agentName || 'this agent'}`}>
            <FolderOpen className="h-3.5 w-3.5 text-violet-400" />
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-300">Files</span>
            {files.length > 0 && (
              <span className="rounded-full bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-medium text-violet-700 dark:text-violet-300">{files.length}</span>
            )}
          </div>
          <div className="flex items-center gap-0.5">
            <button
              onClick={() => inputRef.current?.click()}
              disabled={!upload || uploading}
              title={UPLOAD_TITLE}
              className="flex items-center gap-1 rounded px-1.5 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40 disabled:hover:bg-transparent"
            >
              {uploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
              {UPLOAD_LABEL}
            </button>
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              accept={upload ? upload.extensions.join(',') : undefined}
              onChange={(e) => {
                const file = e.target.files?.[0]
                e.target.value = ''   // picking the same file again still fires
                handlePick(file)
              }}
            />
            <button
              onClick={reload}
              className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
              title="Refresh"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
        <p className="mt-1 text-[11px] leading-snug text-zinc-500">{MEMBER_FILES_NOTE}</p>
        {actionError && (
          <div className="mt-1 flex items-start gap-1.5 text-[11px] text-red-600 dark:text-red-400">
            <AlertCircle className="mt-px h-3.5 w-3.5 shrink-0" />{actionError}
          </div>
        )}
      </div>

      {/* Search */}
      <div className="border-b border-zinc-800/50 px-2 py-1.5">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-600" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search files…"
            className="w-full rounded-md border border-zinc-700 bg-zinc-900 py-1 pl-7 pr-2 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
          />
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto px-1 py-1">
        {loading && (
          <div className="flex justify-center py-6"><Loader2 className="h-4 w-4 animate-spin text-zinc-500" /></div>
        )}
        {error && !loading && (
          <div className="flex items-start gap-1.5 px-2 py-3 text-[11px] text-red-600 dark:text-red-400">
            <AlertCircle className="h-3.5 w-3.5 shrink-0" />{error}
          </div>
        )}
        {!loading && !error && truncated && (
          <p className="px-2 pb-1 pt-0.5 text-[10px] text-zinc-500">{TRUNCATED_NOTE}</p>
        )}
        {!loading && !error && shown.length === 0 && (
          <p className="px-2 py-6 text-center text-[11px] text-zinc-500">
            {files.length === 0 ? NO_SHARED_FILES : 'No files match your search.'}
          </p>
        )}
        {!loading && !error && shown.length > 0 && (
          <ul className="flex flex-col gap-0.5">
            {shown.map((f) => (
              <li key={f.id} className="group rounded-md px-1.5 py-1 hover:bg-zinc-900/60">
                <div className="flex items-center gap-1.5">
                  <SharedFileIcon file={f} />
                  <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-300" title={f.path}>
                    {f.filename}
                  </span>
                  <span className="shrink-0 text-[10px] text-zinc-600">{formatSize(f.size)}</span>
                </div>
                <div className="mt-0.5 flex items-center gap-1">
                  <CreatorBadge mode="member" creator={f.created_by} ownerName={ownerName} />
                  <div className="ml-auto flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
                    {isViewable(toViewerFile(f)) && <IconBtn onClick={() => handleView(f)} title="View" Icon={Eye} />}
                    <IconBtn onClick={() => handleDownload(f)} title="Download" Icon={Download} />
                    {f.can_delete && (
                      deleting === f.id
                        ? <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />
                        : <IconBtn onClick={() => handleDelete(f)} title="Delete" Icon={Trash2} hoverClass="hover:text-red-600 dark:hover:text-red-400" />
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Viewer: read-only, over the member routes (never an owner URL). */}
      {viewing && (
        <FileViewer
          file={toViewerFile(viewing)}
          urls={{
            view: sharedFileUrl(agentRef, viewing.id, 'view'),
            download: sharedFileUrl(agentRef, viewing.id, 'download'),
          }}
          readOnly
          onClose={() => setViewing(null)}
          hasPrev={viewIdx > 0}
          hasNext={viewIdx >= 0 && viewIdx < viewable.length - 1}
          onPrev={() => { if (viewIdx > 0) setViewing(viewable[viewIdx - 1]) }}
          onNext={() => { if (viewIdx >= 0 && viewIdx < viewable.length - 1) setViewing(viewable[viewIdx + 1]) }}
        />
      )}
    </div>
  )
}
