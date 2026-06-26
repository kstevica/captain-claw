import { useEffect, useState } from 'react'
import {
  FolderTree,
  Folder,
  FileText,
  ChevronRight,
  RefreshCw,
  Download,
  Trash2,
  FolderPlus,
  X,
  ArrowLeft,
  HardDrive,
} from 'lucide-react'
import { useVFSStore, type VFSEntry } from '../../stores/vfsStore'
import { VFSFileViewer } from './VFSFileViewer'

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

export function VFSBrowser() {
  const s = useVFSStore()
  const [creating, setCreating] = useState(false)
  const [folderName, setFolderName] = useState('')

  useEffect(() => {
    if (!s.project) s.loadProjects()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Project list view ──────────────────────────────────────────────
  if (!s.project) {
    return (
      <div className="flex h-full flex-col">
        <div className="flex h-12 items-center justify-between border-b border-zinc-800 px-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
            <FolderTree className="h-4 w-4 text-violet-400" /> Shared VFS
            <span className="text-xs font-normal text-zinc-500">cross-agent filesystem</span>
          </div>
          <button
            onClick={() => s.loadProjects()}
            className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${s.loading ? 'animate-spin' : ''}`} /> Refresh
          </button>
        </div>
        <div className="flex-1 overflow-auto p-4">
          {s.error && <div className="mb-3 rounded bg-red-950/50 px-3 py-2 text-xs text-red-300">{s.error}</div>}
          {s.projects.length === 0 && !s.loading && (
            <div className="mt-16 text-center text-sm text-zinc-500">
              <HardDrive className="mx-auto mb-3 h-8 w-8 text-zinc-700" />
              No projects yet. Agents create them by writing{' '}
              <code className="rounded bg-zinc-800 px-1 text-zinc-300">vfs:&lt;project&gt;/&lt;file&gt;</code>.
            </div>
          )}
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {s.projects.map((p) => (
              <div
                key={p.name}
                className="group flex flex-col gap-1 rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 hover:border-violet-700/60"
              >
                <button onClick={() => s.openProject(p.name)} className="flex items-center gap-2 text-left">
                  <Folder className="h-4 w-4 shrink-0 text-violet-400" />
                  <span className="truncate text-sm font-medium text-zinc-100">{p.name}</span>
                </button>
                <div className="flex items-center justify-between text-[11px] text-zinc-500">
                  <span>
                    {p.files} file{p.files !== 1 ? 's' : ''} · {fmtBytes(p.bytes)}
                  </span>
                  <button
                    onClick={() => {
                      if (confirm(`Delete project "${p.name}" and all its files?`)) s.deleteProject(p.name)
                    }}
                    className="opacity-0 transition-opacity hover:text-red-400 group-hover:opacity-100"
                    title="Delete project"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  // ── In-project browser ─────────────────────────────────────────────
  const segs = s.path ? s.path.split('/') : []
  const parentPath = segs.slice(0, -1).join('/')

  return (
    <div className="flex h-full flex-col">
      {/* breadcrumb + actions */}
      <div className="flex h-12 items-center justify-between border-b border-zinc-800 px-4">
        <div className="flex min-w-0 items-center gap-1 text-sm">
          <button onClick={s.closeProject} className="mr-1 rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200" title="All projects">
            <ArrowLeft className="h-4 w-4" />
          </button>
          <button onClick={() => s.browse('')} className="flex items-center gap-1 font-medium text-violet-300 hover:text-violet-200">
            <FolderTree className="h-4 w-4" /> {s.project}
          </button>
          {segs.map((seg, i) => {
            const sub = segs.slice(0, i + 1).join('/')
            return (
              <span key={sub} className="flex min-w-0 items-center gap-1">
                <ChevronRight className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
                <button onClick={() => s.browse(sub)} className="truncate text-zinc-300 hover:text-zinc-100">
                  {seg}
                </button>
              </span>
            )
          })}
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => setCreating(true)} className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
            <FolderPlus className="h-3.5 w-3.5" /> New folder
          </button>
          <button onClick={() => s.refresh()} className="rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200" title="Refresh">
            <RefreshCw className={`h-3.5 w-3.5 ${s.loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {creating && (
        <div className="flex items-center gap-2 border-b border-zinc-800 bg-zinc-900/60 px-4 py-2">
          <input
            autoFocus
            value={folderName}
            onChange={(e) => setFolderName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                s.newFolder(folderName)
                setFolderName('')
                setCreating(false)
              } else if (e.key === 'Escape') {
                setCreating(false)
                setFolderName('')
              }
            }}
            placeholder="folder name"
            className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 outline-none focus:border-violet-600"
          />
          <button
            onClick={() => {
              s.newFolder(folderName)
              setFolderName('')
              setCreating(false)
            }}
            className="rounded bg-violet-600 px-2 py-1 text-xs text-white hover:bg-violet-500"
          >
            Create
          </button>
          <button onClick={() => { setCreating(false); setFolderName('') }} className="rounded p-1 text-zinc-400 hover:text-zinc-200">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        {/* entries */}
        <div className="flex-1 overflow-auto">
          {s.error && <div className="m-3 rounded bg-red-950/50 px-3 py-2 text-xs text-red-300">{s.error}</div>}
          {s.path && (
            <button
              onClick={() => s.browse(parentPath)}
              className="flex w-full items-center gap-2 border-b border-zinc-900 px-4 py-2 text-left text-xs text-zinc-500 hover:bg-zinc-900"
            >
              <ArrowLeft className="h-3.5 w-3.5" /> ..
            </button>
          )}
          {s.entries.length === 0 && !s.loading && (
            <div className="mt-12 text-center text-sm text-zinc-600">Empty directory</div>
          )}
          {s.entries.map((e) => (
            <EntryRow key={e.path} entry={e} active={s.file?.path === e.path} />
          ))}
        </div>
      </div>

      {/* Rich file viewer modal — same renderers as the agent-card FileViewer */}
      {s.file && <FileViewerHost />}
    </div>
  )
}

// Hosts the modal viewer and wires prev/next navigation across the files in
// the current directory listing.
function FileViewerHost() {
  const s = useVFSStore()
  const files = s.entries.filter((e) => e.type === 'file')
  const idx = files.findIndex((e) => e.path === s.file?.path)
  const hasPrev = idx > 0
  const hasNext = idx >= 0 && idx < files.length - 1
  return (
    <VFSFileViewer
      onClose={s.closeFile}
      onPrev={hasPrev ? () => s.openFile(files[idx - 1]) : undefined}
      onNext={hasNext ? () => s.openFile(files[idx + 1]) : undefined}
      hasPrev={hasPrev}
      hasNext={hasNext}
    />
  )
}

function EntryRow({ entry, active }: { entry: VFSEntry; active: boolean }) {
  const s = useVFSStore()
  const Icon = entry.type === 'dir' ? Folder : FileText
  return (
    <div
      className={`group flex items-center gap-2 border-b border-zinc-900 px-4 py-2 text-sm ${
        active ? 'bg-violet-950/30' : 'hover:bg-zinc-900'
      }`}
    >
      <button onClick={() => s.openFile(entry)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
        <Icon className={`h-4 w-4 shrink-0 ${entry.type === 'dir' ? 'text-violet-400' : 'text-zinc-400'}`} />
        <span className="truncate text-zinc-200">{entry.name}</span>
        {entry.type === 'file' && <span className="shrink-0 text-[11px] text-zinc-600">{fmtBytes(entry.size)}</span>}
      </button>
      <div className="flex shrink-0 items-center gap-1 opacity-0 group-hover:opacity-100">
        {entry.type === 'file' && (
          <button onClick={() => s.download(entry)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Download">
            <Download className="h-3.5 w-3.5" />
          </button>
        )}
        <button
          onClick={() => {
            if (confirm(`Delete ${entry.type} "${entry.name}"?`)) s.deleteEntry(entry)
          }}
          className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-red-400"
          title="Delete"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}

