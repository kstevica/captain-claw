import { useEffect, useState } from 'react'
import { Database, Table2, Eye, Loader2, Folder, ChevronDown, ChevronRight, Users } from 'lucide-react'
import { DatastoreBrowser } from '../agents/DatastoreBrowser'
import { fdGet } from './RunArtifacts'
import type { ProjectSource } from './ProjectFiles'

interface TableRow { name: string; row_count: number }

// Aggregated datastore view for a project: the project's shared folder plus every
// run's folder, grouped by owner. View opens the datastore browser on that folder.
export function ProjectDatastore({ sources }: { sources: ProjectSource[] }) {
  const [byFolder, setByFolder] = useState<Record<string, TableRow[]>>({})
  const [loading, setLoading] = useState(true)
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const [ds, setDs] = useState<{ folder: string; label: string; table?: string } | null>(null)
  const folderKey = sources.map((s) => s.folder).join('|')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setOpen(Object.fromEntries(sources.map((s) => [s.folder, s.owner === 'project'])))
    ;(async () => {
      const pairs = await Promise.all(sources.map(async (s) => {
        const res = await fdGet(`/vfs/datastore/${encodeURIComponent(s.folder)}/tables?_=1`)
        const data = res.ok ? await res.json().catch(() => []) : []
        return [s.folder, Array.isArray(data) ? data : []] as const
      }))
      if (!cancelled) { setByFolder(Object.fromEntries(pairs)); setLoading(false) }
    })()
    return () => { cancelled = true }
  }, [folderKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const total = Object.values(byFolder).reduce((n, t) => n + t.length, 0)
  // Only sources that actually have tables — a project of empty run stores is noise.
  const withTables = sources.filter((s) => (byFolder[s.folder] || []).length > 0)

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-3 flex items-center gap-2">
        <Database className="h-3.5 w-3.5 text-emerald-500/80" />
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Datastore</span>
        {!loading && <span className="text-[10px] tabular-nums text-zinc-600">{total} table(s) across {withTables.length} source(s)</span>}
      </div>

      {loading ? (
        <div className="flex items-center gap-2 text-[11px] text-zinc-600"><Loader2 className="h-3 w-3 animate-spin" /> loading…</div>
      ) : withTables.length === 0 ? (
        <p className="text-[11px] text-zinc-600">No datastore tables in this project's runs yet.</p>
      ) : (
        <div className="space-y-2">
          {withTables.map((s) => {
            const tables = byFolder[s.folder] || []
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
                    <span className="tabular-nums">{tables.length} tables</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); setDs({ folder: s.folder, label: s.label }) }}
                      title="Open datastore"
                      className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                    >
                      <Database className="h-3.5 w-3.5" />
                    </button>
                  </span>
                </button>
                {isOpen && (
                  <div className="space-y-0.5 px-2 pb-2">
                    {tables.map((t) => (
                      <div key={t.name} className="flex items-center gap-1.5 rounded px-1 py-0.5 text-xs hover:bg-zinc-800/50">
                        <Table2 className="h-3 w-3 shrink-0 text-emerald-500/80" />
                        <span className="truncate font-mono text-zinc-300" title={t.name}>{t.name}</span>
                        <span className="shrink-0 text-[10px] tabular-nums text-zinc-600">{t.row_count} rows</span>
                        <button
                          onClick={() => setDs({ folder: s.folder, label: s.label, table: t.name })}
                          title="View table"
                          className="ml-auto shrink-0 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                        >
                          <Eye className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
      {ds && (
        <DatastoreBrowser
          vfsProject={ds.folder}
          initialTable={ds.table}
          title={`Datastore — ${ds.label}`}
          onClose={() => setDs(null)}
        />
      )}
    </div>
  )
}
