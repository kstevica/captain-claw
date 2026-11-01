import { useEffect, useMemo, useState } from 'react'
import { ChevronRight, Folder, FolderPlus, Loader2, Lock, X } from 'lucide-react'
import { Segmented } from '../console/parts'
import { crewApi, type CrewFolderChoice, type CrewTask, type CrewVfsFolder } from '../../services/crewApi'
import { vfsApi } from '../../services/vfsApi'
import type { VFSEntry } from '../../stores/vfsStore'
import { ErrorLine, errorOf, fieldClass, primaryBtn, quietBtn } from './crewParts'

// ── Folders of a person's VFS that a worker can use ──
//
// Any number, each to read (the worker gets its own copy) or to read and write (it works
// in the real folder). Flight Deck decides what's offered: no Google Drive or OneDrive.

const sameFolder = (a: CrewFolderChoice, b: CrewFolderChoice) =>
  a.project === b.project && (a.owner || '') === (b.owner || '') && (a.path || '') === (b.path || '')

/** *a* holds *b*: ticking a folder takes in what's inside it. */
const holds = (a: CrewFolderChoice, b: CrewFolderChoice) =>
  a.project === b.project && (a.owner || '') === (b.owner || '') && !!(b.path || '') && (!a.path || (b.path || '').startsWith(`${a.path}/`))

const labelOf = (f: CrewFolderChoice) => `${f.project}${f.path ? `/${f.path}` : ''}`

const sizeOf = (bytes: number) => {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`
  if (bytes >= 1_000_000) return `${Math.round(bytes / 1_000_000)} MB`
  return `${Math.max(1, Math.round(bytes / 1000))} KB`
}

function Branch({ root, path, name, depth, value, onToggle }: {
  root: CrewVfsFolder; path: string; name: string; depth: number; value: CrewFolderChoice[]; onToggle: (f: CrewFolderChoice) => void
}) {
  const [open, setOpen] = useState(false)
  const [inside, setInside] = useState<VFSEntry[] | null>(null)
  const [error, setError] = useState('')
  const self: CrewFolderChoice = { project: root.project, owner: root.owner, path, write: false }
  const ticked = value.some((f) => sameFolder(f, self))
  const within = value.some((f) => holds(f, self))
  const expand = () => {
    setOpen((was) => !was)
    if (inside || error) return
    vfsApi.list({ project: root.project, owner: root.owner }, path)
      .then((entries) => setInside(entries.filter((e) => e.type === 'dir' && !e.name.startsWith('.'))))
      .catch((e) => setError(errorOf(e)))
  }
  const about = depth === 0
    ? [root.title, root.kind === 'shared' ? `shared by ${root.owner_name || 'someone'}` : root.kind === 'link' ? 'linked from this computer' : '',
        `${root.files} file${root.files === 1 ? '' : 's'}`, root.files ? sizeOf(root.bytes) : '', root.can_write ? '' : 'read only'].filter(Boolean).join(' · ')
    : ''
  return (
    <li>
      <div className="flex items-center gap-1 rounded-lg py-0.5 pr-2 transition-colors hover:bg-zinc-800/50" style={{ paddingLeft: `${depth * 18 + 4}px` }}>
        <button type="button" onClick={expand} aria-label={`${open ? 'Hide' : 'Show'} what's in ${name}`} aria-expanded={open} className="rounded-md p-1 text-zinc-500 transition-colors hover:text-zinc-200">
          <ChevronRight className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-90' : ''}`} />
        </button>
        <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 py-1 text-sm text-zinc-200">
          <input type="checkbox" checked={ticked || within} disabled={within} onChange={() => onToggle(self)} className="h-4 w-4 shrink-0 accent-violet-600" />
          <Folder className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
          <span className="min-w-0 truncate">{name}</span>
          {about && <span className="min-w-0 truncate text-xs text-zinc-500">{about}</span>}
        </label>
      </div>
      {open && (
        <div>
          {error && <p className="py-1 text-xs text-red-700 dark:text-red-300" style={{ paddingLeft: `${(depth + 1) * 18 + 8}px` }}>{error}</p>}
          {!inside && !error && <Loader2 className="my-1 h-3.5 w-3.5 animate-spin text-zinc-500" style={{ marginLeft: `${(depth + 1) * 18 + 8}px` }} />}
          {inside?.length === 0 && <p className="py-1 text-xs text-zinc-500" style={{ paddingLeft: `${(depth + 1) * 18 + 8}px` }}>No folders inside.</p>}
          {!!inside?.length && (
            <ul>{inside.map((e) => <Branch key={e.path} root={root} path={e.path} name={e.name} depth={depth + 1} value={value} onToggle={onToggle} />)}</ul>
          )}
        </div>
      )}
    </li>
  )
}

/** The chosen folders, each with what the worker may do in it, and the tree to choose from. */
export function FolderPicker({ value, onChange, startOpen = false }: {
  value: CrewFolderChoice[]; onChange: (next: CrewFolderChoice[]) => void; startOpen?: boolean
}) {
  const [open, setOpen] = useState(startOpen)
  const [roots, setRoots] = useState<CrewVfsFolder[] | null>(null)
  const [filter, setFilter] = useState('')
  const [error, setError] = useState('')
  // Also for what's chosen already: what it can be given, read or write, is Flight Deck's to say.
  const needed = open || value.length > 0
  useEffect(() => {
    if (!needed || roots) return
    crewApi.vfsFolders().then(setRoots).catch((e) => setError(errorOf(e)))
  }, [needed, roots])
  const shown = useMemo(() => {
    const words = filter.trim().toLowerCase()
    return (roots ?? []).filter((r) => !words || `${r.project} ${r.title} ${r.owner_name}`.toLowerCase().includes(words))
  }, [roots, filter])
  const toggle = (f: CrewFolderChoice) => {
    onChange(value.some((v) => sameFolder(v, f)) ? value.filter((v) => !sameFolder(v, f)) : [...value.filter((v) => !holds(f, v)), f])
  }
  const readOnly = (f: CrewFolderChoice) => roots?.find((r) => r.project === f.project && r.owner === (f.owner || ''))?.can_write === false
  const setWrite = (f: CrewFolderChoice, write: boolean) => onChange(value.map((v) => (sameFolder(v, f) ? { ...v, write } : v)))

  return (
    <div className="space-y-2">
      {value.length > 0 && (
        <ul className="space-y-1.5" aria-label="Folders it can use">
          {value.map((f) => (
            <li key={labelOf(f) + (f.owner || '')} className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-xl border border-zinc-800 bg-zinc-900/60 py-1.5 pl-3 pr-1.5">
              <span className="flex min-w-0 flex-1 basis-40 items-center gap-2 text-sm text-zinc-200" title={labelOf(f)}>
                <Folder className="h-3.5 w-3.5 shrink-0 text-violet-500" />
                <span className="min-w-0 truncate">{labelOf(f)}</span>
              </span>
              {readOnly(f)
                ? <span className="inline-flex items-center gap-1 px-2 text-xs text-zinc-500"><Lock className="h-3 w-3" /> Read only</span>
                : <div className="order-last sm:order-none"><Segmented label={`What it can do in ${labelOf(f)}`} value={f.write ? 'write' : 'read'} onChange={(mode) => setWrite(f, mode === 'write')} options={[{ value: 'read' as const, label: 'Read' }, { value: 'write' as const, label: 'Read and write' }]} /></div>}
              <button type="button" onClick={() => toggle(f)} aria-label={`Remove ${labelOf(f)}`} className="rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                <X className="h-3.5 w-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
      {value.length > 0 && (
        <p className="text-xs leading-relaxed text-zinc-500">
          With <strong className="font-medium text-zinc-400">Read</strong> it gets its own copy, so your files stay as they are.
          With <strong className="font-medium text-zinc-400">Read and write</strong> it works in your real folder: what it changes happens at once, with no branch and no undo.
        </p>
      )}
      <button type="button" onClick={() => setOpen((was) => !was)} aria-expanded={open} className={quietBtn}>
        <FolderPlus className="h-4 w-4" /> {open ? 'Hide your folders' : value.length ? 'Choose more folders' : 'Choose folders'}
      </button>
      {open && (
        <div className="space-y-2 rounded-xl border border-zinc-800 bg-zinc-900/40 p-2">
          {(roots?.length ?? 0) > 6 && <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Find a folder" aria-label="Find a folder" className={fieldClass} />}
          <ErrorLine text={error} />
          {!roots && !error && <Loader2 className="m-2 h-4 w-4 animate-spin text-zinc-500" />}
          {roots?.length === 0 && <p className="p-2 text-xs text-zinc-500">There are no folders in your files yet.</p>}
          {!!shown.length && (
            <ul className="max-h-64 overflow-y-auto">
              {shown.map((r) => <Branch key={`${r.owner}/${r.project}`} root={r} path="" name={r.project} depth={0} value={value} onToggle={toggle} />)}
            </ul>
          )}
          <p className="px-2 pb-1 text-[11px] text-zinc-500">Google Drive and OneDrive folders can't be given to a worker yet.</p>
        </div>
      )}
    </div>
  )
}

/** On a task's page: what it has, and changing that for its next turn. */
export function TaskFolders({ task, canChange, onChanged }: { task: CrewTask; canChange: boolean; onChanged: () => void }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState<CrewFolderChoice[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const has = task.vfs ?? []
  if (!canChange && !has.length) return null
  const save = async () => {
    setBusy(true); setError('')
    try {
      await crewApi.setFolders(task.id, draft)
      setEditing(false)
      onChanged()
    } catch (e) {
      setError(errorOf(e))
    }
    setBusy(false)
  }
  if (editing) {
    return (
      <div className="space-y-2 rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
        <p className="text-xs font-medium text-zinc-300">Folders it can use, from its next turn</p>
        <FolderPicker value={draft} onChange={setDraft} startOpen={!draft.length} />
        <ErrorLine text={error} />
        <div className="flex gap-2">
          <button type="button" disabled={busy} onClick={save} className={primaryBtn}>{busy && <Loader2 className="h-4 w-4 animate-spin" />} Save</button>
          <button type="button" disabled={busy} onClick={() => setEditing(false)} className={quietBtn}>Cancel</button>
        </div>
      </div>
    )
  }
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-zinc-500">
      <Folder className="h-3.5 w-3.5 shrink-0" />
      {has.length
        ? has.map((f) => <span key={labelOf(f) + (f.owner || '')} title={f.write ? 'Read and write: its real folder' : 'Read: its own copy'} className="rounded-lg bg-zinc-800/70 px-2 py-0.5 text-zinc-300">{labelOf(f)} · {f.write ? 'read and write' : 'read'}</span>)
        : <span>No folders from your files</span>}
      {canChange && (
        <button type="button" onClick={() => { setDraft(has.map((f) => ({ ...f }))); setError(''); setEditing(true) }} className="font-medium text-violet-700 hover:underline dark:text-violet-300">
          {has.length ? 'Change' : 'Give it folders'}
        </button>
      )}
    </div>
  )
}
