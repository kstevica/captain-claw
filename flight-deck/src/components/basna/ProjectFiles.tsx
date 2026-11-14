import { useEffect, useRef, useState } from 'react'
import {
  Download, Eye, FileText, Loader2, Paperclip, Trash2, HardDrive,
  Folder, FolderClock, ChevronDown,
} from 'lucide-react'
import { Button, Card, EmptyState, Pill } from '../console/parts'
import { dangerIconBtnCls, iconBtnCls } from './lookCls'
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
    <Card
      title="Files"
      icon={FileText}
      hint={loading ? 'The project folder and every run’s own folder.' : `${totalFiles} across ${sources.length} source(s) — the project folder and every run’s own folder.`}
      action={
        <>
          <Button icon={uploading ? undefined : Paperclip} busy={uploading} disabled={!uploadFolder} onClick={() => fileInputRef.current?.click()}>
            Upload to project
          </Button>
          <input
            ref={fileInputRef} type="file" multiple className="hidden"
            onChange={(e) => { if (e.target.files) void upload(e.target.files); e.target.value = '' }}
          />
        </>
      }
    >
      {loading ? (
        <div className="flex items-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</div>
      ) : sources.length === 0 ? (
        <EmptyState icon={FileText} title="No files yet">Upload files to share them with every run in this project.</EmptyState>
      ) : (
        <div className="space-y-2.5">
          {sources.map((s) => {
            const files = byFolder[s.folder] || []
            const isOpen = !!open[s.folder]
            return (
              <div key={s.folder} className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950/40">
                <button
                  onClick={() => setOpen((o) => ({ ...o, [s.folder]: !o[s.folder] }))}
                  aria-expanded={isOpen}
                  className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left transition-colors hover:bg-zinc-800/40"
                >
                  {s.owner === 'project'
                    ? <Folder className="h-4 w-4 shrink-0 text-zinc-400" />
                    : <FolderClock className="h-4 w-4 shrink-0 text-zinc-500" />}
                  <span className="min-w-0 truncate text-sm font-medium text-zinc-200" title={s.label}>{s.label}</span>
                  <Pill tone={s.owner === 'project' ? 'violet' : 'zinc'}>{s.owner === 'project' ? 'Project' : 'Run'}</Pill>
                  <span className="ml-auto flex shrink-0 items-center gap-3 text-[11px] text-zinc-500">
                    <span className="hidden items-center gap-1 font-mono @2xl:flex"><HardDrive className="h-3 w-3" />{s.folder}</span>
                    <span className="tabular-nums">{files.length} {files.length === 1 ? 'file' : 'files'}</span>
                    <ChevronDown className={`h-4 w-4 text-zinc-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                  </span>
                </button>
                {isOpen && (
                  files.length === 0
                    ? <p className="border-t border-zinc-800/80 px-3.5 py-3 text-xs text-zinc-500">No files.</p>
                    : (
                      <div className="divide-y divide-zinc-800/60 border-t border-zinc-800/80">
                        {files.map((e) => (
                          <div key={e.path} className="flex items-center gap-2.5 px-3.5 py-1.5 text-sm hover:bg-zinc-800/30">
                            <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                            <span className="min-w-0 truncate text-zinc-300" title={e.path}>{e.path}</span>
                            <span className="shrink-0 text-[11px] tabular-nums text-zinc-500">{Math.max(1, Math.round(e.size / 1024))}kb</span>
                            <div className="ml-auto flex shrink-0 items-center gap-0.5">
                              {isViewable(e.name) && (
                                <button onClick={() => view(s.folder, e)} title="View" aria-label={`View ${e.name}`} className={iconBtnCls}><Eye className="h-3.5 w-3.5" /></button>
                              )}
                              <button onClick={() => download(s.folder, e)} title="Download" aria-label={`Download ${e.name}`} className={iconBtnCls}><Download className="h-3.5 w-3.5" /></button>
                              {s.owner === 'project' && (
                                <button onClick={() => remove(s.folder, e)} title="Delete" aria-label={`Delete ${e.name}`} className={dangerIconBtnCls}><Trash2 className="h-3.5 w-3.5" /></button>
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
    </Card>
  )
}
