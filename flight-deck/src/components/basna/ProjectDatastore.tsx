import { useEffect, useState } from 'react'
import { Database, Table2, Eye, Loader2, Folder, FolderClock, ChevronDown } from 'lucide-react'
import { Card, EmptyState, Pill } from '../console/parts'
import { iconBtnCls } from './lookCls'
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
    <Card
      title="Datastore"
      icon={Database}
      hint={loading ? 'Tables in the project folder and every run’s folder.' : `${total} table(s) across ${withTables.length} source(s).`}
    >
      {loading ? (
        <div className="flex items-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</div>
      ) : withTables.length === 0 ? (
        <EmptyState icon={Database} title="No tables yet">No datastore tables in this project's runs yet.</EmptyState>
      ) : (
        <div className="space-y-2.5">
          {withTables.map((s) => {
            const tables = byFolder[s.folder] || []
            const isOpen = !!open[s.folder]
            return (
              <div key={s.folder} className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950/40">
                <div className="flex items-center gap-1 pr-2 transition-colors hover:bg-zinc-800/40">
                  <button
                    onClick={() => setOpen((o) => ({ ...o, [s.folder]: !o[s.folder] }))}
                    aria-expanded={isOpen}
                    className="flex min-w-0 flex-1 items-center gap-2.5 py-2.5 pl-3.5 text-left"
                  >
                    {s.owner === 'project'
                      ? <Folder className="h-4 w-4 shrink-0 text-zinc-400" />
                      : <FolderClock className="h-4 w-4 shrink-0 text-zinc-500" />}
                    <span className="min-w-0 truncate text-sm font-medium text-zinc-200" title={s.label}>{s.label}</span>
                    <Pill tone={s.owner === 'project' ? 'violet' : 'zinc'}>{s.owner === 'project' ? 'Project' : 'Run'}</Pill>
                    <span className="ml-auto flex shrink-0 items-center gap-3 text-[11px] text-zinc-500">
                      <span className="tabular-nums">{tables.length} {tables.length === 1 ? 'table' : 'tables'}</span>
                      <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                    </span>
                  </button>
                  <button
                    onClick={() => setDs({ folder: s.folder, label: s.label })}
                    title="Open datastore"
                    aria-label="Open datastore"
                    className={iconBtnCls}
                  >
                    <Database className="h-3.5 w-3.5" />
                  </button>
                </div>
                {isOpen && (
                  <div className="divide-y divide-zinc-800/60 border-t border-zinc-800/80">
                    {tables.map((t) => (
                      <div key={t.name} className="flex items-center gap-2.5 px-3.5 py-1.5 text-sm hover:bg-zinc-800/30">
                        <Table2 className="h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                        <span className="min-w-0 truncate font-mono text-[13px] text-zinc-300" title={t.name}>{t.name}</span>
                        <span className="shrink-0 text-[11px] tabular-nums text-zinc-500">{t.row_count} rows</span>
                        <button
                          onClick={() => setDs({ folder: s.folder, label: s.label, table: t.name })}
                          title="View table"
                          aria-label={`View table ${t.name}`}
                          className={`ml-auto shrink-0 ${iconBtnCls}`}
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
    </Card>
  )
}
