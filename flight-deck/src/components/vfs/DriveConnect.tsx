import { useEffect, useState } from 'react'
import { Cloud, Folder, ArrowLeft, X, Loader2, Sparkles, ChevronRight } from 'lucide-react'
import { useVFSStore, type DriveFolder } from '../../stores/vfsStore'

/** Connect a Google Drive folder as a read-only VFS mount.
 *
 * Browses Drive folders (folders only — this is a place to pick a mount root),
 * lets the user name the mount and choose whether to clone to Markdown, then
 * mounts. Nothing is mirrored unless clonemd is on; otherwise the tree is
 * placeholders fetched on demand.
 */
export function DriveConnect({ onClose, onDone }: { onClose: () => void; onDone: (name: string) => void }) {
  const s = useVFSStore()
  const [stack, setStack] = useState<{ id: string; name: string }[]>([{ id: 'root', name: 'My Drive' }])
  const [folders, setFolders] = useState<DriveFolder[]>([])
  const [loading, setLoading] = useState(false)
  const [truncated, setTruncated] = useState(false)
  const [err, setErr] = useState('')
  const [name, setName] = useState('')
  const [clonemd, setClonemd] = useState(false)
  const [mounting, setMounting] = useState(false)

  const here = stack[stack.length - 1]

  const load = async (folderId: string) => {
    setLoading(true); setErr('')
    try {
      const d = await s.browseDrive(folderId)
      setFolders(d.folders)
      setTruncated(d.truncated)
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Could not reach Google Drive')
      setFolders([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load('root') }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const enter = (f: DriveFolder) => {
    setStack((st) => [...st, f])
    // Default the mount name to the folder you step into.
    setName(f.name.replace(/[^a-zA-Z0-9._-]/g, '-'))
    load(f.id)
  }
  const up = () => {
    if (stack.length < 2) return
    const st = stack.slice(0, -1)
    setStack(st)
    load(st[st.length - 1].id)
  }

  const mount = async () => {
    const n = name.trim()
    if (!n) return
    setMounting(true); setErr('')
    try {
      await s.mountDrive(n, here.id, clonemd)
      onDone(n)
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'mount failed')
      setMounting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-8" onClick={onClose}>
      <div className="flex max-h-full w-full max-w-2xl flex-col rounded-lg border border-zinc-800 bg-zinc-950"
           onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <Cloud className="h-4 w-4 text-blue-500 dark:text-blue-400" />
          <span className="text-sm font-semibold text-zinc-200">Connect Google Drive</span>
          <button onClick={onClose} className="ml-auto text-zinc-500 hover:text-zinc-200"><X className="h-4 w-4" /></button>
        </div>

        {/* breadcrumb */}
        <div className="flex items-center gap-1 border-b border-zinc-800 px-4 py-2 text-xs text-zinc-400">
          {stack.length > 1 && (
            <button onClick={up} className="mr-1 rounded p-0.5 hover:bg-zinc-800 hover:text-zinc-200" title="Up">
              <ArrowLeft className="h-3.5 w-3.5" />
            </button>
          )}
          {stack.map((f, i) => (
            <span key={f.id} className="flex items-center gap-1">
              {i > 0 && <ChevronRight className="h-3 w-3 text-zinc-600" />}
              <span className={i === stack.length - 1 ? 'text-zinc-200' : ''}>{f.name}</span>
            </span>
          ))}
        </div>

        {/* folder list */}
        <div className="min-h-0 flex-1 overflow-y-auto p-2" style={{ minHeight: 220 }}>
          {loading && (
            <div className="flex items-center justify-center gap-2 py-10 text-xs text-zinc-500">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading…
            </div>
          )}
          {!loading && err && (
            <div className="m-2 rounded border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-600 dark:text-red-300">
              {err}
            </div>
          )}
          {!loading && !err && folders.length === 0 && (
            <div className="py-10 text-center text-xs text-zinc-500">No sub-folders here. Mount this folder itself below.</div>
          )}
          {!loading && folders.map((f) => (
            <div key={f.id} className="flex items-center gap-2 rounded px-2 py-1.5 hover:bg-zinc-900">
              <Folder className="h-4 w-4 shrink-0 text-blue-500 dark:text-blue-400" />
              <button onClick={() => enter(f)} className="min-w-0 flex-1 truncate text-left text-sm text-zinc-200">
                {f.name}
              </button>
              <button onClick={() => enter(f)} className="rounded p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Open">
                <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
          {truncated && (
            <div className="px-2 py-1 text-[11px] text-amber-600 dark:text-amber-400">
              Showing the first 500 folders here.
            </div>
          )}
        </div>

        {/* mount controls */}
        <div className="border-t border-zinc-800 px-4 py-3">
          <div className="mb-2 text-[11px] text-zinc-500">
            Mounts <span className="text-zinc-300">{here.name}</span> as a read-only folder.
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="vfs name"
              className="w-40 rounded bg-zinc-900 px-2 py-1 text-xs text-zinc-100 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600"
            />
            <label className="flex items-center gap-1.5 text-xs text-zinc-400" title="Convert files to Markdown on disk so they're searchable and indexable. Off = on-demand placeholders.">
              <input type="checkbox" checked={clonemd} onChange={(e) => setClonemd(e.target.checked)} className="h-3.5 w-3.5 accent-blue-600" />
              <Sparkles className="h-3 w-3" /> clone to Markdown
            </label>
            <button
              onClick={mount}
              disabled={!name.trim() || mounting}
              className="ml-auto flex items-center gap-1.5 rounded bg-blue-600/90 px-3 py-1 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-40"
            >
              {mounting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Cloud className="h-3.5 w-3.5" />}
              Mount folder
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
