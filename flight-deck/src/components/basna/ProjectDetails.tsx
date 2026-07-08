import { useEffect, useRef, useState } from 'react'
import {
  Check, Download, Eye, FileText, FolderSearch, Loader2, Paperclip, Trash2, Network,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import type { BasnaProject } from '../../stores/basnaProjectStore'
import type { VFSEntry } from '../../stores/vfsStore'
import { fdGet, qp, walkFiles } from './RunArtifacts'
import { FileModal, isViewable, viewModeForFile, type ViewMode } from './shared'

// Details tab for a project: edit the theme (name / description / instructions)
// that seeds every run, and manage the project's VFS folder — the files agents
// read (auto-added read-only as a reference folder to each run).
export function ProjectDetails({ project, saving, onSave }: {
  project: BasnaProject
  saving?: boolean
  onSave: (fields: { name: string; description: string; instructions: string }) => void
}) {
  const [name, setName] = useState(project.name)
  const [description, setDescription] = useState(project.description)
  const [instructions, setInstructions] = useState(project.instructions)
  useEffect(() => {
    setName(project.name); setDescription(project.description); setInstructions(project.instructions)
  }, [project.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const dirty = name !== project.name || description !== project.description || instructions !== project.instructions

  const [files, setFiles] = useState<VFSEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [modal, setModal] = useState<{ title: string; content: string; mode: ViewMode } | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folder = project.vfs_folder

  const loadFiles = async () => {
    if (!folder) { setLoading(false); return }
    const fs = await walkFiles(folder).catch(() => [])
    setFiles(fs); setLoading(false)
  }
  useEffect(() => { setLoading(true); void loadFiles() }, [folder]) // eslint-disable-line react-hooks/exhaustive-deps

  const upload = async (list: FileList) => {
    if (!folder || list.length === 0) return
    setUploading(true)
    try {
      const form = new FormData()
      form.append('project', folder)
      form.append('path', '')
      for (const f of Array.from(list)) form.append('files', f)
      const { token, authEnabled } = useAuthStore.getState()
      const headers: Record<string, string> = {}
      if (authEnabled && token) headers.Authorization = `Bearer ${token}`
      await fetch('/fd/vfs/upload', { method: 'POST', headers, body: form, credentials: 'include' })
      await loadFiles()
    } finally {
      setUploading(false)
    }
  }

  const view = async (e: VFSEntry) => {
    const res = await fdGet(`/vfs/read?${qp({ project: folder, path: e.path })}`)
    if (!res.ok) return
    const data = await res.json().catch(() => ({ text: '' }))
    setModal({ title: e.name, content: data.text || '', mode: viewModeForFile(e.name) })
  }
  const download = async (e: VFSEntry) => {
    const res = await fdGet(`/vfs/download?${qp({ project: folder, path: e.path })}`)
    if (!res.ok) return
    const url = URL.createObjectURL(await res.blob())
    const a = document.createElement('a'); a.href = url; a.download = e.name; a.click(); URL.revokeObjectURL(url)
  }
  const remove = async (e: VFSEntry) => {
    if (!window.confirm(`Delete "${e.name}" from the project folder?`)) return
    const { token, authEnabled } = useAuthStore.getState()
    const headers: Record<string, string> = {}
    if (authEnabled && token) headers.Authorization = `Bearer ${token}`
    await fetch(`/fd/vfs/entry?${qp({ project: folder, path: e.path, recursive: 'true' })}`, { method: 'DELETE', headers, credentials: 'include' }).catch(() => {})
    await loadFiles()
  }

  const isUnfiled = !folder

  return (
    <div className="space-y-5">
      {/* Theme — description + instructions injected into every run. */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
        <div className="mb-3 flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Project theme</span>
          <span className="text-[11px] text-zinc-600">description + instructions are sent to every run</span>
          {dirty && !isUnfiled && (
            <button
              onClick={() => onSave({ name: name.trim() || project.name, description, instructions })}
              disabled={saving}
              className="ml-auto flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
            >
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
              Save changes
            </button>
          )}
        </div>
        {isUnfiled ? (
          <p className="text-xs text-zinc-500">
            Unfiled is a bucket for runs that don't belong to a project — it has no theme or shared folder.
            Create a project to bundle runs with a shared theme and files.
          </p>
        ) : (
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-[11px] font-medium text-zinc-400">Name</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-sky-600 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-[11px] font-medium text-zinc-400">Description <span className="font-normal text-zinc-600">— the shared theme</span></label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
                placeholder="What this project is about — prepended to every run."
                className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-[11px] font-medium text-zinc-400">Additional instructions <span className="font-normal text-zinc-600">— extra guidance for every run</span></label>
              <textarea
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                rows={5}
                placeholder="Constraints, format, focus areas — appended to each run's task."
                className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
              />
            </div>
          </div>
        )}
      </div>

      {/* Files — the project's VFS folder, added read-only as a reference to each run. */}
      {!isUnfiled && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
          <div className="mb-3 flex items-center gap-2">
            <FolderSearch className="h-3.5 w-3.5 text-emerald-500/80" />
            <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Project files</span>
            <span className="flex items-center gap-1 text-[10px] text-zinc-600"><Network className="h-3 w-3" />{folder}</span>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="ml-auto flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
            >
              {uploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Paperclip className="h-3.5 w-3.5" />} Upload
            </button>
            <input
              ref={fileInputRef} type="file" multiple className="hidden"
              onChange={(e) => { if (e.target.files) void upload(e.target.files); e.target.value = '' }}
            />
          </div>
          <p className="mb-2 text-[11px] text-zinc-600">
            Agents in this project read these files (read-only) before web search.
          </p>
          {loading ? (
            <div className="flex items-center gap-2 text-[11px] text-zinc-600"><Loader2 className="h-3 w-3 animate-spin" /> loading…</div>
          ) : files.length === 0 ? (
            <p className="text-[11px] text-zinc-600">No files yet — upload reference material for the team.</p>
          ) : (
            <div className="space-y-0.5">
              {files.map((e) => (
                <div key={e.path} className="flex items-center gap-1.5 rounded px-1 py-0.5 text-xs hover:bg-zinc-800/50">
                  <FileText className="h-3 w-3 shrink-0 text-zinc-500" />
                  <span className="truncate text-zinc-300" title={e.path}>{e.path}</span>
                  <span className="shrink-0 text-[10px] text-zinc-600">{Math.max(1, Math.round(e.size / 1024))}kb</span>
                  <div className="ml-auto flex shrink-0 items-center gap-0.5">
                    {isViewable(e.name) && (
                      <button onClick={() => view(e)} title="View" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"><Eye className="h-3.5 w-3.5" /></button>
                    )}
                    <button onClick={() => download(e)} title="Download" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"><Download className="h-3.5 w-3.5" /></button>
                    <button onClick={() => remove(e)} title="Delete" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-rose-400"><Trash2 className="h-3.5 w-3.5" /></button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {modal && <FileModal title={modal.title} content={modal.content} mode={modal.mode} onClose={() => setModal(null)} />}
    </div>
  )
}
