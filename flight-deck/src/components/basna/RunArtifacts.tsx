import { useEffect, useState } from 'react'
import {
  FileText, Database, Eye, Download, Loader2, ChevronDown, Table2, type LucideIcon,
} from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'
import type { VFSEntry } from '../../stores/vfsStore'
import { DatastoreBrowser } from '../agents/DatastoreBrowser'
import { FileModal, isViewable, viewModeForFile, type ViewMode } from './shared'
import { PanelTitle } from './look'
import { iconBtnCls, panelCls } from './lookCls'

// ── Run artifacts: the files the team wrote to the run's VFS folder and the
// datastore tables it built. Reused live (in the run rail) and after the run
// (as report tabs). View-only — no edit/delete here. ────────────────────────

export async function fdGet(path: string): Promise<Response> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = {}
  if (authEnabled && token) headers.Authorization = `Bearer ${token}`
  let res = await fetch(`/fd${path}`, { headers, credentials: 'include' })
  if (res.status === 401 && authEnabled && await refreshAccessToken()) {
    const t2 = useAuthStore.getState().token
    res = await fetch(`/fd${path}`, { headers: t2 ? { Authorization: `Bearer ${t2}` } : {}, credentials: 'include' })
  }
  return res
}

export const qp = (o: Record<string, string>) => new URLSearchParams(o).toString()

// Walk the run folder, gathering files (skipping hidden dirs like .datastore /
// .history / .code that hold machinery, not deliverables).
export async function walkFiles(project: string, dir = '', depth = 0, acc: VFSEntry[] = []): Promise<VFSEntry[]> {
  if (depth > 3) return acc
  const res = await fdGet(`/vfs/list?${qp({ project, path: dir })}`)
  if (!res.ok) return acc
  const data = await res.json().catch(() => ({ entries: [] }))
  for (const e of (data.entries || []) as VFSEntry[]) {
    if (e.type === 'dir') {
      if (e.name.startsWith('.')) continue
      await walkFiles(project, e.path, depth + 1, acc)
    } else {
      acc.push(e)
    }
  }
  return acc
}

interface TableRow { name: string; row_count: number }

function fmtKb(bytes: number): string {
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + 'mb'
  return Math.max(1, Math.round(bytes / 1024)) + 'kb'
}

// Shared collapsible shell so the rail panels match ProgressFeed / AgentRoster.
function Panel({ icon, iconCls, label, count, right, variant, children }: {
  icon: LucideIcon; iconCls?: string; label: string; count?: number; right?: React.ReactNode
  variant: 'rail' | 'tab'; children: React.ReactNode
}) {
  const [open, setOpen] = useState(true)
  if (variant === 'tab') {
    return <div>{children}</div>
  }
  return (
    <div className={`${panelCls} p-4`}>
      <div className="flex items-center gap-1">
        <button onClick={() => setOpen((o) => !o)} aria-expanded={open} className="flex min-w-0 flex-1 items-center gap-2 text-left">
          <PanelTitle icon={icon} iconCls={iconCls} count={count}>{label}</PanelTitle>
          <ChevronDown className={`ml-auto h-4 w-4 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
        {right}
      </div>
      {open && <div className="mt-3">{children}</div>}
    </div>
  )
}

export function RunFilesPanel({ project, live, variant = 'rail' }: {
  project: string; live?: boolean; variant?: 'rail' | 'tab'
}) {
  const [files, setFiles] = useState<VFSEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [modal, setModal] = useState<{ title: string; content: string; mode: ViewMode } | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const fs = await walkFiles(project).catch(() => [])
      if (!cancelled) { setFiles(fs); setLoading(false) }
    }
    load()
    if (!live) return () => { cancelled = true }
    const t = setInterval(load, 5000)
    return () => { cancelled = true; clearInterval(t) }
  }, [project, live])

  const view = async (e: VFSEntry) => {
    const res = await fdGet(`/vfs/read?${qp({ project, path: e.path })}`)
    if (!res.ok) return
    const data = await res.json().catch(() => ({ text: '' }))
    setModal({ title: e.name, content: data.text || '', mode: viewModeForFile(e.name) })
  }
  const download = async (e: VFSEntry) => {
    const res = await fdGet(`/vfs/download?${qp({ project, path: e.path })}`)
    if (!res.ok) return
    const url = URL.createObjectURL(await res.blob())
    const a = document.createElement('a'); a.href = url; a.download = e.name; a.click(); URL.revokeObjectURL(url)
  }

  const body = (
    loading && files.length === 0
      ? <div className="flex items-center gap-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…</div>
      : files.length === 0
        ? <p className="text-xs text-zinc-500">No files written yet.</p>
        : (
          <div className={`-mx-1.5 space-y-0.5 overflow-auto ${variant === 'rail' ? 'max-h-56' : 'max-h-[60vh]'}`}>
            {files.map((e) => (
              <div key={e.path} className="flex items-center gap-2 rounded-lg px-1.5 py-0.5 text-[13px] hover:bg-zinc-800/50">
                <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                <span className="min-w-0 truncate text-zinc-300" title={e.path}>{e.path}</span>
                <span className="shrink-0 text-[11px] tabular-nums text-zinc-500">{fmtKb(e.size)}</span>
                <div className="ml-auto flex shrink-0 items-center gap-0.5">
                  {isViewable(e.name) ? (
                    <button onClick={() => view(e)} title="View" aria-label={`View ${e.name}`} className={iconBtnCls}>
                      <Eye className="h-3.5 w-3.5" />
                    </button>
                  ) : (
                    <button onClick={() => download(e)} title="Download" aria-label={`Download ${e.name}`} className={iconBtnCls}>
                      <Download className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )
  )

  return (
    <>
      <Panel
        variant={variant}
        icon={FileText}
        label="Files"
        count={files.length}
      >
        {body}
      </Panel>
      {modal && <FileModal title={modal.title} content={modal.content} mode={modal.mode} onClose={() => setModal(null)} />}
    </>
  )
}

export function RunDatastorePanel({ project, live, variant = 'rail', hideWhenEmpty }: {
  project: string; live?: boolean; variant?: 'rail' | 'tab'; hideWhenEmpty?: boolean
}) {
  const [tables, setTables] = useState<TableRow[]>([])
  const [loading, setLoading] = useState(true)
  const [dsOpen, setDsOpen] = useState(false)
  const [dsTable, setDsTable] = useState<string | undefined>(undefined)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const res = await fdGet(`/vfs/datastore/${encodeURIComponent(project)}/tables?_=1`)
      const data = res.ok ? await res.json().catch(() => []) : []
      if (!cancelled) { setTables(Array.isArray(data) ? data : []); setLoading(false) }
    }
    load()
    if (!live) return () => { cancelled = true }
    const t = setInterval(load, 5000)
    return () => { cancelled = true; clearInterval(t) }
  }, [project, live])

  const openDs = (table?: string) => { setDsTable(table); setDsOpen(true) }

  if (hideWhenEmpty && !loading && tables.length === 0) return null

  const body = (
    loading && tables.length === 0
      ? <div className="flex items-center gap-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…</div>
      : tables.length === 0
        ? <p className="text-xs text-zinc-500">No datastore tables.</p>
        : (
          <div className={`-mx-1.5 space-y-0.5 overflow-auto ${variant === 'rail' ? 'max-h-56' : 'max-h-[60vh]'}`}>
            {tables.map((t) => (
              <div key={t.name} className="flex items-center gap-2 rounded-lg px-1.5 py-0.5 text-[13px] hover:bg-zinc-800/50">
                <Table2 className="h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                <span className="min-w-0 truncate font-mono text-zinc-300" title={t.name}>{t.name}</span>
                <span className="shrink-0 text-[11px] tabular-nums text-zinc-500">{t.row_count} rows</span>
                <button onClick={() => openDs(t.name)} title="View table" aria-label={`View table ${t.name}`} className={`ml-auto shrink-0 ${iconBtnCls}`}>
                  <Eye className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        )
  )

  return (
    <>
      <Panel
        variant={variant}
        icon={Database}
        iconCls="text-emerald-600 dark:text-emerald-400"
        label="Datastore"
        count={tables.length}
        right={variant === 'rail' && tables.length > 0 ? (
          <button onClick={() => openDs(undefined)} title="Open datastore" aria-label="Open datastore" className={`shrink-0 ${iconBtnCls}`}>
            <Database className="h-3.5 w-3.5" />
          </button>
        ) : undefined}
      >
        {body}
      </Panel>
      {dsOpen && (
        <DatastoreBrowser
          vfsProject={project}
          initialTable={dsTable}
          title={`Datastore — ${project}`}
          onClose={() => setDsOpen(false)}
        />
      )}
    </>
  )
}
