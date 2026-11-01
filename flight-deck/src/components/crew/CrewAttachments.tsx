import { useCallback, useEffect, useRef, useState, type ChangeEvent, type ClipboardEvent, type DragEvent } from 'react'
import { AlertTriangle, FileText, Loader2, Paperclip, X } from 'lucide-react'
import { crewApi } from '../../services/crewApi'
import { errorOf } from './crewParts'

// ── Files a person hands a worker ──
//
// Pasted, dropped or picked, of any kind. Each goes up as soon as it's added, so sending
// the task or the message only names what's already there.

export interface AttachedFile {
  key: string
  name: string
  size: number
  id?: string
  error?: string
  preview?: string
}

const sizeText = (size: number) => (size >= 1_000_000 ? `${(size / 1_000_000).toFixed(1)} MB` : `${Math.max(1, Math.round(size / 1000))} KB`)

/** A screenshot on the clipboard has no name of its own: "image.png" every time. */
function named(file: File, index: number): File {
  if (file.name && file.name !== 'image.png') return file
  const ext = (file.type.split('/')[1] || 'png').split('+')[0]
  return new File([file], `pasted-${Date.now()}-${index + 1}.${ext}`, { type: file.type })
}

/** A clipboard's own word on what's in it. Copied from a document, rich text comes with a
 *  picture of itself, and it's the text that's wanted: only that has both these types.
 *  A picture copied from a page has HTML and no text, a file from the desktop the opposite. */
export function pastedFiles(data: DataTransfer | null): File[] {
  if (!data || (data.types.includes('text/html') && data.types.includes('text/plain'))) return []
  const files = Array.from(data.files ?? [])
  if (files.length) return files
  // Some browsers only put them here.
  return Array.from(data.items ?? []).filter((i) => i.kind === 'file').map((i) => i.getAsFile()).filter((f): f is File => !!f)
}

/** `enabled`: there's a box here for what's pasted anywhere on the page outside a field. */
export function useAttachments(enabled = true) {
  const [files, setFiles] = useState<AttachedFile[]>([])
  const live = useRef<AttachedFile[]>([])
  live.current = files
  // What's on screen is let go of with the page, and what wasn't sent is the deck's to clear.
  useEffect(() => () => {
    for (const f of live.current) if (f.preview) URL.revokeObjectURL(f.preview)
  }, [])

  const add = useCallback((list: File[]) => {
    list.forEach((raw, index) => {
      const file = named(raw, index)
      const key = `${file.name}-${crypto.randomUUID()}`
      const preview = file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined
      setFiles((old) => [...old, { key, name: file.name, size: file.size, preview }])
      crewApi.upload(file)
        .then((made) => setFiles((old) => old.map((f) => (f.key === key ? { ...f, id: made.id, name: made.name } : f))))
        .catch((e) => setFiles((old) => old.map((f) => (f.key === key ? { ...f, error: errorOf(e) } : f))))
    })
  }, [])

  const remove = useCallback((key: string) => {
    setFiles((old) => {
      const gone = old.find((f) => f.key === key)
      if (gone?.preview) URL.revokeObjectURL(gone.preview)
      if (gone?.id) crewApi.dropUpload(gone.id).catch(() => {})
      return old.filter((f) => f.key !== key)
    })
  }, [])

  /** They went with what was sent: the deck has them now. */
  const clear = useCallback(() => {
    setFiles((old) => {
      for (const f of old) if (f.preview) URL.revokeObjectURL(f.preview)
      return []
    })
  }, [])

  const onPaste = useCallback((ev: ClipboardEvent) => {
    const list = pastedFiles(ev.clipboardData)
    if (!list.length) return
    ev.preventDefault()
    add(list)
  }, [add])

  // A screenshot pasted straight after looking at the page, with no box clicked first.
  useEffect(() => {
    if (!enabled) return
    const anywhere = (ev: globalThis.ClipboardEvent) => {
      if ((ev.target as Element | null)?.closest?.('input, textarea, [contenteditable="true"]')) return
      const list = pastedFiles(ev.clipboardData)
      if (!list.length) return
      ev.preventDefault()
      add(list)
    }
    document.addEventListener('paste', anywhere)
    return () => document.removeEventListener('paste', anywhere)
  }, [enabled, add])

  const onDragOver = useCallback((ev: DragEvent) => {
    if (ev.dataTransfer?.types.includes('Files')) ev.preventDefault()
  }, [])

  const onDrop = useCallback((ev: DragEvent) => {
    const list = Array.from(ev.dataTransfer?.files ?? [])
    if (!list.length) return
    ev.preventDefault()
    add(list)
  }, [add])

  const onPick = useCallback((ev: ChangeEvent<HTMLInputElement>) => {
    add(Array.from(ev.target.files ?? []))
    ev.target.value = ''
  }, [add])

  return {
    files, add, remove, clear, onPaste, onDragOver, onDrop, onPick,
    ids: files.filter((f) => f.id).map((f) => f.id as string),
    // A file that didn't go up holds the message back: it's meant to go with it, so it's not left out quietly.
    ready: files.every((f) => f.id && !f.error),
  }
}

export function AttachButton({ onPick, label = 'Attach files', compact = false }: {
  onPick: (ev: ChangeEvent<HTMLInputElement>) => void; label?: string; compact?: boolean
}) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <>
      <input ref={input} type="file" multiple onChange={onPick} className="hidden" aria-hidden="true" tabIndex={-1} />
      <button
        type="button" onClick={() => input.current?.click()} title={compact ? label : undefined} aria-label={label}
        className={compact
          ? 'flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200'
          : 'inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200'}
      >
        <Paperclip className={compact ? 'h-4 w-4' : 'h-3.5 w-3.5'} />{!compact && label}
      </button>
    </>
  )
}

export function AttachmentList({ files, onRemove }: { files: AttachedFile[]; onRemove: (key: string) => void }) {
  if (!files.length) return null
  return (
    <ul className="flex max-h-36 flex-wrap gap-2 overflow-y-auto" aria-label="Attached files">
      {files.map((f) => (
        <li key={f.key} title={f.error || f.name} className={`flex max-w-full items-center gap-2 rounded-xl border py-1 pl-1 pr-1.5 text-xs ${f.error ? 'border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-300' : 'border-zinc-800 bg-zinc-900/60 text-zinc-300'}`}>
          {f.preview && !f.error
            ? <img src={f.preview} alt="" className="h-8 w-8 shrink-0 rounded-lg object-cover" />
            : (
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800/70 text-zinc-500">
                {f.error ? <AlertTriangle className="h-4 w-4 text-red-500" /> : <FileText className="h-4 w-4" />}
              </span>
            )}
          <span className="min-w-0">
            <span className="block max-w-[14rem] truncate font-medium">{f.name}</span>
            <span className="block max-w-[14rem] truncate text-[11px] opacity-70">{f.error || sizeText(f.size)}</span>
          </span>
          {!f.id && !f.error && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-zinc-500" />}
          <button type="button" onClick={() => onRemove(f.key)} aria-label={`Remove ${f.name}`} className="rounded-md p-0.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-3.5 w-3.5" />
          </button>
        </li>
      ))}
    </ul>
  )
}
