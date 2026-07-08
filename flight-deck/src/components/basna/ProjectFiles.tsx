import { useEffect, useRef, useState } from 'react'
import {
  Download, Eye, FileText, Loader2, Paperclip, Trash2, Network,
  Folder, ChevronDown, ChevronRight, Users,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import type { VFSEntry } from '../../stores/vfsStore'
import { fdGet, qp, walkFiles } from './RunArtifacts'
import { FileModal, isViewable, viewModeForFile, type ViewMode } from './shared'

// One file/datastore source in a project: the project's own upload folder plus
// each run's own write folder, so the project view can show everything with its
// owner. Deduped by folder (continuation chains share the root's folder).
export interface ProjectSource { label: string; folder: string; owner: 'project' | 'run' }

// Aggregated file browser for a project: the project's shared folder (uploads)
// plus every run's folder, grouped by owner, each file view/download-able.
export function ProjectFiles({ sources, uploadFolder }: {
  sources: ProjectSource[]; uploadFolder: string
}) {
  const [byFolder, setByFolder] = useState<Record<string, VFSEntry[]>>({})
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const [modal, setModal] = useState<{ title: string; content: string; mode: ViewMode } | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderKey = sources.map((s) => s.folder).join('|')

  const loadAll = async () => {
    const pairs = await Promise.all(
      sources.map(async (s) => [s.folder, await walkFiles(s.folder).catch(() => [])] as const))
    setByFolder(Object.fromEntries(pairs))
    setLoading(false)
  }
  useEffect(() => {
    setLoading(true)
    // Project section open by default; run sections collapsed to keep it tidy.
    setOpen(Object.fromEntries(sources.map((s) => [s.folder, s.owner === 'project'])))
    void loadAll()
  }, [folderKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const upload = async (list: FileList) => {
    if (!uploadFolder || list.length === 0) return
    setUploading(true)
    try {
      const form = new FormData()
      form.append('project', uploadFolder)
      form.append('path', '')
      for (const f of Array.from(list)) form.append('files', f)
      const { token, authEnabled } = useAuthStore.getState()
      const headers: Record<string, string> = {}
      if (authEnabled && token) headers.Authorization = `Bearer ${token}`
      await fetch('/fd/vfs/upload', { method: 'POST', headers, body: form, credentials: 'include' })
      await loadAll()
    } finally {
      setUploading(false)
    }
  }

  const view = async (folder: string, e: VFSEntry) => {
    const res = await fdGet(`/vfs/read?${qp({ project: folder, path: e.path })}`)
    if (!res.ok) return
    const data = await res.json().catch(() => ({ text: '' }))
    setModal({ title: e.name, content: data.text || '', mode: viewModeForFile(e.name) })
  }
  const download = async (folder: string, e: VFSEntry) => {
    const res = await fdGet(`/vfs/download?${qp({ project: folder, path: e.path })}`)
    if (!res.ok) return
    const url = URL.createObjectURL(await res.blob())
    const a = document.createElement('a'); a.href = url; a.download = e.name; a.click(); URL.revokeObjectURL(url)
  }
  const remove = async (folder: string, e: VFSEntry) => {
    if (!window.confirm(`Delete "${e.name}" from the project folder?`)) return
    const { token, authEnabled } = useAuthStore.getState()
    const headers: Record<string, string> = {}
    if (authEnabled && token) headers.Authorization = `Bearer ${token}`
    await fetch(`/fd/vfs/entry?${qp({ project: folder, path: e.path, recursive: 'true' })}`, { method: 'DELETE', headers, credentials: 'include' }).catch(() => {})
    await loadAll()
  }

  const totalFiles = Object.values(byFolder).reduce((n, fs) => n + fs.length, 0)

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-3 flex items-center gap-2">
        <FileText className="h-3.5 w-3.5 text-zinc-500" />
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Files</span>
        {!loading && <span className="text-[10px] tabular-nums text-zinc-600">{totalFiles} across {sources.length} source(s)</span>}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading || !uploadFolder}
          className="ml-auto flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
        >
          {uploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Paperclip className="h-3.5 w-3.5" />} Upload to project
        </button>
        <input
          ref={fileInputRef} type="file" multiple className="hidden"
          onChange={(e) => { if (e.target.files) void upload(e.target.files); e.target.value = '' }}
        />
      </div>

      {loading ? (
        <div className="flex items-center gap-2 text-[11px] text-zinc-600"><Loader2 className="h-3 w-3 animate-spin" /> loading…</div>
      ) : sources.length === 0 ? (
        <p className="text-[11px] text-zinc-600">No files yet.</p>
      ) : (
        <div className="space-y-2">
          {sources.map((s) => {
            const files = byFolder[s.folder] || []
            const isOpen = !!open[s.folder]
            return (
              <div key={s.folder} className="rounded-lg border border-zinc-800 bg-zinc-900/40">
                <button
                  onClick={() => setOpen((o) => ({ ...o, [s.folder]: !o[s.folder] }))}
                  className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left"
                >
                  {isOpen ? <ChevronDown className="h-3 w-3 shrink-0 text-zinc-600" /> : <ChevronRight className="h-3 w-3 shrink-0 text-zinc-600" />}
                  {s.owner === 'project'
                    ? <Folder className="h-3.5 w-3.5 shrink-0 text-sky-500/80" />
                    : <Users className="h-3.5 w-3.5 shrink-0 text-violet-500/80" />}
                  <span className="truncate text-xs font-medium text-zinc-200" title={s.label}>{s.label}</span>
                  <span className={`shrink-0 rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase ${
                    s.owner === 'project' ? 'bg-sky-500/15 text-sky-700 dark:text-sky-300' : 'bg-violet-500/15 text-violet-700 dark:text-violet-300'}`}>
                    {s.owner === 'project' ? 'project' : 'run'}
                  </span>
                  <span className="ml-auto flex shrink-0 items-center gap-2 text-[10px] text-zinc-600">
                    <span className="tabular-nums">{files.length} files</span>
                    <span className="hidden items-center gap-1 sm:flex"><Network className="h-3 w-3" />{s.folder}</span>
                  </span>
                </button>
                {isOpen && (
                  files.length === 0
                    ? <p className="px-3 pb-2 text-[11px] text-zinc-600">No files.</p>
                    : (
                      <div className="space-y-0.5 px-2 pb-2">
                        {files.map((e) => (
                          <div key={e.path} className="flex items-center gap-1.5 rounded px-1 py-0.5 text-xs hover:bg-zinc-800/50">
                            <FileText className="h-3 w-3 shrink-0 text-zinc-500" />
                            <span className="truncate text-zinc-300" title={e.path}>{e.path}</span>
                            <span className="shrink-0 text-[10px] text-zinc-600">{Math.max(1, Math.round(e.size / 1024))}kb</span>
                            <div className="ml-auto flex shrink-0 items-center gap-0.5">
                              {isViewable(e.name) && (
                                <button onClick={() => view(s.folder, e)} title="View" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"><Eye className="h-3.5 w-3.5" /></button>
                              )}
                              <button onClick={() => download(s.folder, e)} title="Download" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"><Download className="h-3.5 w-3.5" /></button>
                              {s.owner === 'project' && (
                                <button onClick={() => remove(s.folder, e)} title="Delete" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-rose-400"><Trash2 className="h-3.5 w-3.5" /></button>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )
                )}
              </div>
            )
          })}
        </div>
      )}
      {modal && <FileModal title={modal.title} content={modal.content} mode={modal.mode} onClose={() => setModal(null)} />}
    </div>
  )
}
