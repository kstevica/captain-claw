import { useState } from 'react'
import { FolderPlus, Folder, Loader2, Plus, Trash2, Network, Inbox, X } from 'lucide-react'
import { UNFILED_ID, type BasnaProject } from '../../stores/basnaProjectStore'
import { timeAgo } from './shared'

// Landing screen for Basna's project-first navigation: pick a project (or the
// Unfiled bucket of legacy runs), or create a new one. Selecting opens the
// project's Details / Run view.
export function ProjectPicker({
  projects, counts, busy, onSelect, onCreate, onDelete,
}: {
  projects: BasnaProject[]
  counts: Record<string, number>
  busy?: boolean
  onSelect: (p: BasnaProject) => void
  onCreate: (name: string, description: string, instructions: string) => Promise<void>
  onDelete: (p: BasnaProject) => void
}) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [instructions, setInstructions] = useState('')
  const [creating, setCreating] = useState(false)

  const submit = async () => {
    if (!name.trim()) return
    setCreating(true)
    try {
      await onCreate(name.trim(), description.trim(), instructions.trim())
      setName(''); setDescription(''); setInstructions(''); setOpen(false)
    } finally {
      setCreating(false)
    }
  }

  const unfiledCount = counts[UNFILED_ID] || 0

  return (
    <div className="mx-auto w-[92%] max-w-[1100px] py-8">
      <div className="mb-5 flex items-center gap-3">
        <div>
          <h2 className="text-lg font-semibold text-zinc-100">Projects</h2>
          <p className="text-xs text-zinc-500">
            Bundle runs under one theme and shared files. Every run in a project is seeded with its
            description, instructions, and folder — but stays independent.
          </p>
        </div>
        <button
          onClick={() => setOpen((o) => !o)}
          className="ml-auto flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-2 text-xs font-medium text-white hover:bg-sky-500"
        >
          <FolderPlus className="h-3.5 w-3.5" /> New project
        </button>
      </div>

      {open && (
        <div className="mb-5 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <div className="mb-2 flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">New project</span>
            <button onClick={() => setOpen(false)} className="ml-auto rounded p-1 text-zinc-500 hover:text-zinc-300"><X className="h-3.5 w-3.5" /></button>
          </div>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
            placeholder="Project name — e.g. Q3 competitor scan"
            className="mb-2 w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
          />
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={2}
            placeholder="Description — the theme shared by every run (sent to each run)"
            className="mb-2 w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
          />
          <textarea
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            rows={3}
            placeholder="Additional instructions — extra guidance prepended to every run (optional)"
            className="mb-2 w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
          />
          <div className="flex items-center gap-2">
            <button
              onClick={submit}
              disabled={!name.trim() || creating}
              className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
            >
              {creating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
              Create project
            </button>
            <span className="text-[11px] text-zinc-600">A VFS folder is created for the project's files.</span>
          </div>
        </div>
      )}

      {busy && projects.length === 0 ? (
        <div className="flex items-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> loading projects…</div>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {/* Unfiled — always present when there are legacy/ungrouped runs. */}
          {unfiledCount > 0 && (
            <button
              onClick={() => onSelect({ id: UNFILED_ID, user_id: '', name: 'Unfiled', description: '', instructions: '', vfs_folder: '', created_at: '', updated_at: '' })}
              className="flex flex-col items-start rounded-xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-colors hover:border-zinc-700 hover:bg-zinc-800/40"
            >
              <div className="mb-2 flex w-full items-center gap-2">
                <Inbox className="h-4 w-4 text-zinc-500" />
                <span className="font-medium text-zinc-200">Unfiled</span>
                <span className="ml-auto rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] tabular-nums text-zinc-400">{unfiledCount}</span>
              </div>
              <p className="text-xs text-zinc-500">Runs not in any project.</p>
            </button>
          )}

          {projects.map((p) => (
            <div
              key={p.id}
              onClick={() => onSelect(p)}
              className="group relative flex cursor-pointer flex-col items-start rounded-xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-colors hover:border-sky-700/60 hover:bg-zinc-800/40"
            >
              <div className="mb-2 flex w-full items-center gap-2">
                <Folder className="h-4 w-4 text-sky-500/80" />
                <span className="truncate font-medium text-zinc-200">{p.name}</span>
                <span className="ml-auto flex items-center gap-1">
                  <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] tabular-nums text-zinc-400" title="runs">
                    {counts[p.id] || 0}
                  </span>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDelete(p) }}
                    title="Delete project"
                    className="rounded p-1 text-zinc-600 opacity-0 transition-opacity hover:text-rose-400 group-hover:opacity-100"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </span>
              </div>
              {p.description
                ? <p className="line-clamp-2 text-xs text-zinc-500">{p.description}</p>
                : <p className="text-xs italic text-zinc-600">No description</p>}
              <div className="mt-auto flex w-full items-center gap-2 pt-3 text-[10px] text-zinc-600">
                <Network className="h-3 w-3" />
                <span className="truncate">{p.vfs_folder}</span>
                <span className="ml-auto tabular-nums">{timeAgo(p.updated_at)}</span>
              </div>
            </div>
          ))}

          {projects.length === 0 && unfiledCount === 0 && (
            <div className="col-span-full rounded-xl border border-dashed border-zinc-800 p-8 text-center text-sm text-zinc-600">
              No projects yet. Create one to start bundling runs.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
