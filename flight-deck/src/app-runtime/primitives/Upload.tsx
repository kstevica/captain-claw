import { useEffect, useRef, useState } from 'react'
import type { AgentManifest, ActionDef } from '../types'
import { listFiles, uploadFile, deleteFile, fileUrl, type FileMeta } from '../files'
import { ActionButton } from './ActionButton'

interface Props {
  manifest: AgentManifest
  accept?: string
  multiple?: boolean
  actionIds?: string[]
}

// Drag-and-drop upload surface. Shows a dropzone, a thumbnail/file grid,
// and any actions declared in `sources`. Actions that have a `file`-typed
// input get the selected file_id pre-filled into the first such input.
export function Upload({ manifest, accept, multiple, actionIds = [] }: Props) {
  const agentId = manifest.agent.id
  const [files, setFiles] = useState<FileMeta[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [drag, setDrag] = useState(false)
  const inputRef = useRef<HTMLInputElement | null>(null)

  const refresh = async () => setFiles(await listFiles(agentId))

  useEffect(() => { refresh() }, [agentId])  // eslint-disable-line react-hooks/exhaustive-deps

  const handleFiles = async (chosen: FileList | File[]) => {
    setError(null)
    setBusy(true)
    try {
      for (const f of Array.from(chosen)) {
        const meta = await uploadFile(agentId, f)
        setFiles((prev) => [meta, ...prev])
        if (!selected) setSelected(meta.file_id)
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setBusy(false)
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDrag(false)
    if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files)
  }

  const onRemove = async (id: string) => {
    if (!confirm('Delete this file?')) return
    await deleteFile(agentId, id)
    setFiles((prev) => prev.filter((f) => f.file_id !== id))
    if (selected === id) setSelected(null)
  }

  const actions = actionIds
    .map((id) => manifest.actions[id])
    .filter((a): a is ActionDef => Boolean(a))

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors ${
          drag
            ? 'border-violet-500 bg-violet-500/10'
            : 'border-zinc-700 bg-zinc-900/30 hover:border-zinc-600'
        }`}
      >
        <div className="text-sm font-medium text-zinc-200">
          {busy ? 'Uploading…' : 'Drop files here or click to choose'}
        </div>
        {accept && (
          <div className="mt-1 text-[11px] text-zinc-500">accepts: {accept}</div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          className="hidden"
          onChange={(e) => { if (e.target.files) handleFiles(e.target.files); e.target.value = '' }}
        />
      </div>

      {error && <div className="rounded bg-red-950/50 px-3 py-2 text-xs text-red-300">{error}</div>}

      {files.length === 0 ? (
        <p className="text-center text-xs text-zinc-600">No files yet.</p>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {files.map((f) => (
            <FileCard
              key={f.file_id}
              meta={f}
              agentId={agentId}
              selected={selected === f.file_id}
              onSelect={() => setSelected(f.file_id)}
              onRemove={() => onRemove(f.file_id)}
            />
          ))}
        </div>
      )}

      {actions.length > 0 && selected && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
          <div className="mb-2 text-[11px] uppercase tracking-wide text-zinc-500">
            Actions on selected file
          </div>
          <div className="flex flex-wrap gap-2">
            {actions.map((a) => {
              const fileArg = firstFileInput(a)
              const prefill = fileArg ? { [fileArg]: selected } : {}
              return (
                <ActionButton
                  key={a.id}
                  manifest={manifest}
                  action={a}
                  prefill={prefill}
                />
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function firstFileInput(action: ActionDef): string | null {
  for (const [name, def] of Object.entries(action.inputs)) {
    if (def.type === 'file') return name
  }
  return null
}

interface FileCardProps {
  meta: FileMeta
  agentId: string
  selected: boolean
  onSelect: () => void
  onRemove: () => void
}

function FileCard({ meta, agentId, selected, onSelect, onRemove }: FileCardProps) {
  const isImage = meta.mime.startsWith('image/')
  return (
    <div
      onClick={onSelect}
      className={`group relative cursor-pointer overflow-hidden rounded-lg border bg-zinc-900/50 transition-colors ${
        selected
          ? 'border-violet-500 ring-1 ring-violet-500/50'
          : 'border-zinc-800 hover:border-zinc-700'
      }`}
    >
      {isImage ? (
        <img
          src={fileUrl(agentId, meta.file_id)}
          alt={meta.filename}
          className="aspect-square w-full object-cover"
        />
      ) : (
        <div className="flex aspect-square w-full items-center justify-center bg-zinc-900 text-[10px] uppercase tracking-wider text-zinc-500">
          {meta.mime.split('/')[1] ?? 'file'}
        </div>
      )}
      <div className="px-2 py-1.5">
        <div className="truncate text-xs text-zinc-200">{meta.filename}</div>
        <div className="text-[10px] text-zinc-500">{formatSize(meta.size)}</div>
      </div>
      <button
        onClick={(e) => { e.stopPropagation(); onRemove() }}
        className="absolute right-1 top-1 hidden rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-zinc-200 group-hover:block hover:bg-red-600"
      >
        ✕
      </button>
    </div>
  )
}

function formatSize(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}
