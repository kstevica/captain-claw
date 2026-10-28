import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, Download, Eye, Files, LibraryBig, Loader2 } from 'lucide-react'
import type { ChatMessage } from '../../services/agentChat'
import type { AgentConn } from '../../hooks/useFileAttachments'
import {
  formatSize, getDownloadUrl, getFileTypeGroup, getViewUrl, isViewable, listAgentFiles,
  type AgentFile,
} from '../../services/fileTransfer'
import { listSharedFiles, sharedFileUrl } from '../../services/sharedWorkspace'
import { useFileLibraryStore } from '../../stores/fileLibraryStore'
import { toViewerFile } from '../../utils/sharedWorkspace'
import { belowSaved, resolveChatFiles, scanChatFiles, type ChatFileEntry } from '../../utils/chatFiles'
import { FileIcon } from './AgentFilesPanel'
import { FileViewer } from './FileViewer'

type Entry = ChatFileEntry<AgentFile>

const RECENT = 5

/** The agent's files, each with its path below saved/. A member of a shared
 *  agent lists them through Flight Deck's member routes. */
async function listEntries(conn: AgentConn | null, sharedRef?: string): Promise<Entry[]> {
  if (sharedRef) {
    const res = await listSharedFiles(sharedRef)
    return res.files.map((f) => ({ rel: f.path, name: f.filename, modified: f.modified, file: toViewerFile(f) }))
  }
  if (!conn) return []
  const files = await listAgentFiles(conn.host, conn.port, conn.auth)
  return files.flatMap((f) => {
    const rel = f.exists ? belowSaved(f.physical) : null
    return rel ? [{ rel, name: f.filename, modified: f.modified, file: f }] : []
  })
}

function ago(sec: number): string {
  const s = Math.max(0, Date.now() / 1000 - sec)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return new Date(sec * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}

function RowButton({ label, Icon, iconOnly, onClick }: {
  label: string
  Icon: typeof Eye
  /** Phone: the icon alone, the label as its name. */
  iconOnly?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      title={label}
      aria-label={iconOnly ? label : undefined}
      className={`flex h-8 shrink-0 items-center gap-1.5 rounded-lg border border-zinc-700 text-sm text-zinc-300 transition-colors hover:bg-zinc-800 hover:text-zinc-100 ${
        iconOnly ? 'w-8 justify-center' : 'px-2.5'
      }`}
    >
      <Icon className="h-4 w-4 text-zinc-500" />
      {!iconOnly && <span>{label}</span>}
    </button>
  )
}

function download(url: string, filename: string) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

// The composer's Files button (simple layout): the last files the agent made
// in this chat — recognised from the chat (utils/chatFiles.ts) and checked
// against the agent's file list — to open or download. Hidden until there is
// one.
export function ChatFilesMenu({ messages, conn, sharedRef, librarySlug, busy, compact }: {
  messages: ChatMessage[]
  /** The agent's endpoint (owner routes). */
  conn: AgentConn | null
  /** An agent shared with you: its files come by the member routes. */
  sharedRef?: string
  /** The agent's slug when the File library holds its files (your own
   *  process agent): each file gets Show in library. */
  librarySlug?: string
  /** A turn is running: its files are listed once it ends. */
  busy: boolean
  /** Phone: the button shows its icon and count only. */
  compact?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [viewing, setViewing] = useState<Entry | null>(null)
  const ref = useRef<HTMLDivElement>(null)

  const source = sharedRef ? `shared:${sharedRef}` : conn ? `${conn.host}:${conn.port}` : ''
  const [listed, setListed] = useState<{ source: string; entries: Entry[] } | null>(null)
  const [loading, setLoading] = useState(false)
  const scan = useMemo(() => scanChatFiles(messages), [messages])
  const files = useMemo(
    () => (listed && listed.source === source ? resolveChatFiles(scan, listed.entries, RECENT) : []),
    [scan, listed, source],
  )

  const load = useCallback(async () => {
    if (!source) return
    const src = source
    setLoading(true)
    try {
      const entries = await listEntries(conn, sharedRef)
      setListed({ source: src, entries })
    } catch {
      // Keep the last list: the agent may be restarting.
    } finally {
      setLoading(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  // The chat named a file it hadn't before: list the agent's files again —
  // after the turn, when they are on disk.
  const fetched = useRef('')
  useEffect(() => {
    if (!source || !scan.sig || busy) return
    const key = `${source}\n${scan.sig}`
    if (fetched.current === key) return
    const t = setTimeout(() => { fetched.current = key; void load() }, 400)
    return () => clearTimeout(t)
  }, [source, scan.sig, busy, load])

  useEffect(() => {
    if (!open) return
    void load()
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const urls = (f: AgentFile) => sharedRef
    ? { view: sharedFileUrl(sharedRef, f.physical, 'view'), download: sharedFileUrl(sharedRef, f.physical, 'download') }
    : conn
      ? { view: getViewUrl(conn.host, conn.port, f.physical, conn.auth), download: getDownloadUrl(conn.host, conn.port, f.physical, conn.auth) }
      : null

  const openFile = (e: Entry) => {
    const u = urls(e.file)
    if (!u) return
    setOpen(false)
    // PDFs in the browser's own viewer, as the Files panels do.
    if (getFileTypeGroup(e.file) === 'pdf') window.open(u.view, '_blank', 'noopener')
    else if (isViewable(e.file)) setViewing(e)
    else download(u.download, e.name)
  }

  const showInLibrary = (agent: string, e: Entry) => {
    setOpen(false)
    useFileLibraryStore.getState().showFile({ agent, relPath: `saved/${e.rel}`, filename: e.name })
  }

  const viewable = files.filter((e) => isViewable(e.file) && getFileTypeGroup(e.file) !== 'pdf')
  const viewIdx = viewing ? viewable.findIndex((e) => e.rel === viewing.rel) : -1

  if (files.length === 0 && !viewing) return null
  return (
    // The menu opens from the composer box's left edge (its positioned
    // ancestor), as the skills menu does: the rows want the width.
    <div className="shrink-0" ref={ref}>
      {files.length > 0 && (
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          aria-haspopup="menu"
          title="Files the agent made in this chat"
          className={`flex h-9 items-center gap-1.5 rounded-lg border px-2.5 text-sm transition-colors ${
            open
              ? 'border-violet-500/40 bg-violet-500/10 text-zinc-100'
              : 'border-zinc-700 text-zinc-300 hover:bg-zinc-800'
          }`}
        >
          <Files className="h-4 w-4 text-zinc-500" />
          {!compact && <span>Files</span>}
          <span className="text-xs tabular-nums text-zinc-500">{files.length}</span>
          <ChevronDown className={`h-3.5 w-3.5 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      )}
      {open && files.length > 0 && (
        <div
          role="menu"
          aria-label="Recent files"
          className="absolute bottom-full left-0 z-40 mb-2 w-full max-w-xl overflow-hidden rounded-2xl border border-zinc-700 bg-zinc-900 shadow-2xl"
        >
          <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2.5">
            <div className="min-w-0 flex-1">
              <div className="text-sm font-medium text-zinc-200">Recent files</div>
              <div className="text-xs text-zinc-500">The last files the agent made in this chat</div>
            </div>
            {loading && <Loader2 className="h-4 w-4 shrink-0 animate-spin text-zinc-500" />}
          </div>
          <div className="p-1.5">
            {files.map((e) => (
              <div key={e.rel} className="flex items-center gap-3 rounded-xl px-2.5 py-2" title={e.rel}>
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800">
                  <FileIcon file={e.file} />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[15px] font-medium text-zinc-100">{e.name}</span>
                  <span className="block truncate text-sm text-zinc-400">{formatSize(e.file.size)} · {ago(e.modified)}</span>
                </span>
                <RowButton
                  label={isViewable(e.file) ? 'Open' : 'Download'}
                  Icon={isViewable(e.file) ? Eye : Download}
                  iconOnly={compact}
                  onClick={() => openFile(e)}
                />
                {librarySlug && (
                  <RowButton
                    label="Show in library"
                    Icon={LibraryBig}
                    iconOnly={compact}
                    onClick={() => showInLibrary(librarySlug, e)}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      {viewing && (sharedRef ? (
        <FileViewer
          file={viewing.file}
          urls={urls(viewing.file) ?? undefined}
          readOnly
          onClose={() => setViewing(null)}
          hasPrev={viewIdx > 0}
          hasNext={viewIdx >= 0 && viewIdx < viewable.length - 1}
          onPrev={() => { if (viewIdx > 0) setViewing(viewable[viewIdx - 1]) }}
          onNext={() => { if (viewIdx >= 0 && viewIdx < viewable.length - 1) setViewing(viewable[viewIdx + 1]) }}
        />
      ) : conn && (
        <FileViewer
          file={viewing.file}
          host={conn.host}
          port={conn.port}
          auth={conn.auth}
          onClose={() => setViewing(null)}
          hasPrev={viewIdx > 0}
          hasNext={viewIdx >= 0 && viewIdx < viewable.length - 1}
          onPrev={() => { if (viewIdx > 0) setViewing(viewable[viewIdx - 1]) }}
          onNext={() => { if (viewIdx >= 0 && viewIdx < viewable.length - 1) setViewing(viewable[viewIdx + 1]) }}
        />
      ))}
    </div>
  )
}
