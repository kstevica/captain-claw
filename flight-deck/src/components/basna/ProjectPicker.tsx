import { useState } from 'react'
import { FolderPlus, Folder, FolderOpen, Loader2, Plus, Trash2, Inbox, X, HardDrive } from 'lucide-react'
import { UNFILED_ID, type BasnaProject } from '../../stores/basnaProjectStore'
import { Button, Card, Field, Pill, inputCls } from '../console/parts'
import { IconSquare, SectionLabel } from './look'
import { timeAgo } from './shared'

// Landing screen for Basna's project-first navigation: pick a project (or the
// Unfiled bucket of legacy runs), or create a new one. Selecting opens the
// project's Details / Run view.
export function ProjectPicker({
  projects, counts, busy, accent = 'sky', onSelect, onCreate, onDelete,
}: {
  projects: BasnaProject[]
  counts: Record<string, number>
  busy?: boolean
  /** The folder icons' colour: the mode the page is in. */
  accent?: 'sky' | 'violet'
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
  const empty = projects.length === 0 && unfiledCount === 0
  const hover = accent === 'sky' ? 'hover:border-sky-500/40' : 'hover:border-violet-500/50'
  const card = `group relative flex min-w-0 cursor-pointer flex-col rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-all hover:-translate-y-px hover:bg-zinc-900/70 hover:shadow-lg hover:shadow-black/10 ${hover}`

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <SectionLabel>Projects</SectionLabel>
            {projects.length > 0 && <span className="text-[11px] tabular-nums text-zinc-600">{projects.length}</span>}
          </div>
          <p className="mt-1 max-w-2xl text-xs leading-relaxed text-zinc-500">
            Bundle runs under one theme and shared files. Every run in a project is seeded with its
            description, instructions, and folder — but stays independent.
          </p>
        </div>
        {!(empty && !open) && (
          <Button variant={open ? 'secondary' : 'primary'} icon={open ? X : FolderPlus} onClick={() => setOpen((o) => !o)}>
            {open ? 'Cancel' : 'New project'}
          </Button>
        )}
      </div>

      {open && (
        <Card title="New project" icon={FolderPlus} hint="A VFS folder is created for the project's files.">
          <div className="space-y-4">
            <Field label="Name">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoFocus
                placeholder="e.g. Q3 competitor scan"
                className={inputCls}
              />
            </Field>
            <Field label="Description" hint="The theme shared by every run (sent to each run).">
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                placeholder="What this project is about"
                className={`${inputCls} resize-y`}
              />
            </Field>
            <Field label="Additional instructions (optional)" hint="Extra guidance prepended to every run.">
              <textarea
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                rows={3}
                placeholder="Constraints, format, focus areas"
                className={`${inputCls} resize-y`}
              />
            </Field>
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <Button variant="primary" icon={Plus} busy={creating} disabled={!name.trim()} onClick={submit}>
                Create project
              </Button>
            </div>
          </div>
        </Card>
      )}

      {busy && projects.length === 0 ? (
        <div className="flex items-center gap-2 py-6 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading projects…</div>
      ) : empty ? (
        !open && (
          <div className="flex flex-col items-center rounded-2xl border border-dashed border-zinc-800 px-6 py-14 text-center">
            <IconSquare icon={FolderOpen} tone={accent} size="lg" />
            <div className="mt-4 text-sm font-semibold text-zinc-200">No projects yet</div>
            <p className="mt-1 max-w-sm text-xs leading-relaxed text-zinc-500">
              A project keeps related runs together, with one theme and one shared folder.
            </p>
            <Button variant="primary" icon={FolderPlus} onClick={() => setOpen(true)} className="mt-5">
              New project
            </Button>
          </div>
        )
      ) : (
        <div className="grid grid-cols-1 gap-3 @2xl:grid-cols-2 @5xl:grid-cols-3">
          {/* Unfiled — always present when there are legacy/ungrouped runs. */}
          {unfiledCount > 0 && (
            <button
              onClick={() => onSelect({ id: UNFILED_ID, user_id: '', name: 'Unfiled', description: '', instructions: '', vfs_folder: '', created_at: '', updated_at: '' })}
              className={card}
            >
              <div className="flex w-full items-start gap-3">
                <IconSquare icon={Inbox} tone="zinc" />
                <div className="min-w-0 flex-1 pt-0.5">
                  <div className="truncate text-sm font-semibold text-zinc-100">Unfiled</div>
                  <p className="mt-0.5 text-xs leading-snug text-zinc-500">Runs not in any project.</p>
                </div>
                <Pill>{unfiledCount} {unfiledCount === 1 ? 'run' : 'runs'}</Pill>
              </div>
            </button>
          )}

          {projects.map((p) => {
            const n = counts[p.id] || 0
            return (
              <div key={p.id} onClick={() => onSelect(p)} className={card}>
                <div className="flex w-full items-start gap-3">
                  <IconSquare icon={Folder} tone={accent} />
                  <div className="min-w-0 flex-1 pt-0.5">
                    <div className="truncate text-sm font-semibold text-zinc-100" title={p.name}>{p.name}</div>
                    {p.description
                      ? <p className="mt-0.5 line-clamp-2 text-xs leading-snug text-zinc-500">{p.description}</p>
                      : <p className="mt-0.5 text-xs italic text-zinc-600">No description</p>}
                  </div>
                  <Pill tone={n > 0 ? accent : 'zinc'} title="runs">{n} {n === 1 ? 'run' : 'runs'}</Pill>
                </div>
                <div className="mt-auto w-full pt-3.5">
                <div className="flex w-full items-center gap-2 border-t border-zinc-800/70 pt-3 text-[11px] text-zinc-500">
                  <HardDrive className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
                  <span className="min-w-0 truncate font-mono">{p.vfs_folder}</span>
                  <span className="ml-auto shrink-0 tabular-nums">{timeAgo(p.updated_at)}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDelete(p) }}
                    title="Delete project"
                    aria-label={`Delete project ${p.name}`}
                    className="-my-1 -mr-1 shrink-0 rounded-lg p-1.5 text-zinc-500 transition-all hover:bg-red-500/10 hover:text-red-600 focus-visible:opacity-100 wide:opacity-0 wide:group-hover:opacity-100 dark:hover:text-red-400"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
