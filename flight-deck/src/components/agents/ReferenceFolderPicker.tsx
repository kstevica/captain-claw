import { useState } from 'react'
import { Check, Search } from 'lucide-react'
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
  const [query, setQuery] = useState('')
  const folders = projects.filter((p) => p.kind !== 'link')
  const q = query.trim().toLowerCase()
  const shown = q
    ? folders.filter((p) => p.name.toLowerCase().includes(q) || (p.title || '').toLowerCase().includes(q))
    : folders
  return (
    <div>
      <p className="mb-2 text-[10px] text-zinc-500">
        Agents check these first (glob + read), then web-search only what's missing. Folders from selected prior-knowledge runs are included automatically.
      </p>
      {folders.length === 0 ? (
        <p className="text-[11px] text-zinc-500">No VFS folders yet.</p>
      ) : (
        <>
          {folders.length > 6 && (
            <div className="relative mb-1.5">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-600" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter folders…"
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 py-1 pl-7 pr-2 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              />
            </div>
          )}
          <ul className="max-h-44 space-y-1 overflow-y-auto pr-0.5">
            {shown.map((p) => {
              const sel = selected.includes(p.name)
              const label = p.title || p.name
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
                    <span className="min-w-0 flex-1">
                      <span className={`block truncate text-[11px] ${sel ? 'text-emerald-700 dark:text-emerald-200' : 'text-zinc-300'}`}>{label}</span>
                      {p.title && p.title !== p.name && (
                        <span className="block truncate font-mono text-[9px] text-zinc-600">{p.name}</span>
                      )}
                    </span>
                    {p.files > 0 && <span className="shrink-0 text-[10px] text-zinc-600">{p.files} file{p.files !== 1 ? 's' : ''}</span>}
                  </button>
                </li>
              )
            })}
            {shown.length === 0 && (
              <li className="px-1 py-2 text-center text-[11px] text-zinc-600">No folders match.</li>
            )}
          </ul>
        </>
      )}
    </div>
  )
}
