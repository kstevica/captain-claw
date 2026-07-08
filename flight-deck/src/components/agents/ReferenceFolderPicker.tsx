import { Check } from 'lucide-react'
import type { VfsProject } from '../../stores/basnaStore'

/**
 * Multi-select of read-only VFS folders agents consult BEFORE web-searching.
 * Shared by the prepare form and the creation wizard. Linked folders are excluded.
 */
export function ReferenceFolderPicker({
  projects,
  selected,
  onToggle,
}: {
  projects: VfsProject[]
  selected: string[]
  onToggle: (name: string) => void
}) {
  const folders = projects.filter((p) => p.kind !== 'link')
  return (
    <div>
      <p className="mb-2 text-[10px] text-zinc-500">
        Agents check these first (glob + read), then web-search only what's missing. Folders from selected prior-knowledge runs are included automatically.
      </p>
      {folders.length === 0 ? (
        <p className="text-[11px] text-zinc-500">No VFS folders yet.</p>
      ) : (
        <ul className="max-h-44 space-y-1 overflow-y-auto pr-0.5">
          {folders.map((p) => {
            const sel = selected.includes(p.name)
            return (
              <li key={p.name}>
                <button
                  onClick={() => onToggle(p.name)}
                  className={`flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-left transition-colors ${
                    sel ? 'border-emerald-500/50 bg-emerald-500/10' : 'border-zinc-800 bg-zinc-900/50 hover:border-zinc-700'
                  }`}
                >
                  <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border ${sel ? 'border-emerald-500 bg-emerald-600 text-white' : 'border-zinc-600'}`}>
                    {sel && <Check className="h-3 w-3" />}
                  </span>
                  <span className={`min-w-0 flex-1 truncate text-[11px] ${sel ? 'text-emerald-700 dark:text-emerald-200' : 'text-zinc-300'}`}>{p.name}</span>
                  {p.files > 0 && <span className="shrink-0 text-[10px] text-zinc-600">{p.files} file{p.files !== 1 ? 's' : ''}</span>}
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
