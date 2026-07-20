import { useState, useEffect, useMemo } from 'react'
import { Database, Table2, Loader2, AlertCircle, RefreshCw, Search } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'
import { useAgentEndpoint } from './AgentFilesPanel'
import { DatastoreBrowser } from './DatastoreBrowser'

interface TableInfo {
  name: string
  columns: { name: string; type: string; position: number }[]
  row_count: number
  created_at: string
  updated_at: string
}

async function fdFetch<T>(path: string): Promise<T> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = {}
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`

  let res = await fetch(`/fd${path}`, { headers, credentials: 'include' })
  if (res.status === 401 && authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) {
      const h2: Record<string, string> = {}
      const t2 = useAuthStore.getState().token
      if (t2) h2['Authorization'] = `Bearer ${t2}`
      res = await fetch(`/fd${path}`, { headers: h2, credentials: 'include' })
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

// Compact, sidebar-sized datastore view for a single agent: the table list
// only. Clicking a table opens the full DatastoreBrowser modal deep-linked to
// it, so rows/paging/export stay in the one component that already does them.
export function AgentDatastorePanel({ containerId }: { containerId: string }) {
  const agent = useAgentEndpoint(containerId)
  const [tables, setTables] = useState<TableInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [openTable, setOpenTable] = useState<string | null>(null)
  const [showBrowser, setShowBrowser] = useState(false)

  const host = agent?.host
  const port = agent?.port
  const auth = agent?.auth

  const loadTables = async () => {
    if (!host || !port) return
    setLoading(true)
    setError('')
    try {
      const qs = auth ? `&token=${encodeURIComponent(auth)}` : ''
      const data = await fdFetch<TableInfo[]>(`/agent-datastore/${host}/${port}/tables?_=1${qs}`)
      setTables(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { loadTables() }, [host, port])

  const shown = useMemo(() => {
    const q = search.trim().toLowerCase()
    const r = q
      ? tables.filter((t) =>
          t.name.toLowerCase().includes(q) ||
          t.columns.some((c) => c.name.toLowerCase().includes(q)))
      : [...tables]
    r.sort((a, b) => a.name.localeCompare(b.name))
    return r
  }, [tables, search])

  if (!agent) {
    return (
      <div className="flex h-full items-center justify-center px-3 text-center text-[11px] text-zinc-500">
        Agent not connected — datastore unavailable.
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center gap-2">
          <Database className="h-3.5 w-3.5 text-emerald-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-zinc-300">Datastore</span>
          {tables.length > 0 && (
            <span className="rounded-full bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300">{tables.length}</span>
          )}
        </div>
        <button
          onClick={loadTables}
          className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
          title="Refresh"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Search */}
      {tables.length > 5 && (
        <div className="border-b border-zinc-800/50 px-2 py-1.5">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-600" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search tables…"
              className="w-full rounded-md border border-zinc-700 bg-zinc-900 py-1 pl-7 pr-2 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-emerald-500/60 focus:outline-none"
            />
          </div>
        </div>
      )}

      {/* List */}
      <div className="flex-1 overflow-y-auto px-1 py-1">
        {loading && (
          <div className="flex justify-center py-6"><Loader2 className="h-4 w-4 animate-spin text-zinc-500" /></div>
        )}
        {error && !loading && (
          <div className="flex items-start gap-1.5 px-2 py-3 text-[11px] text-red-400">
            <AlertCircle className="h-3.5 w-3.5 shrink-0" />{error}
          </div>
        )}
        {!loading && !error && shown.length === 0 && (
          <p className="px-2 py-6 text-center text-[11px] text-zinc-500">
            {tables.length === 0 ? 'No tables yet.' : 'No tables match your search.'}
          </p>
        )}
        {!loading && !error && shown.length > 0 && (
          <ul className="flex flex-col gap-0.5">
            {shown.map((t) => (
              <li key={t.name}>
                <button
                  onClick={() => { setOpenTable(t.name); setShowBrowser(true) }}
                  title={`${t.columns.length} columns · ${t.row_count} rows`}
                  className="flex w-full items-center gap-1.5 rounded-md px-1.5 py-1 text-left hover:bg-zinc-900/60"
                >
                  <Table2 className="h-3.5 w-3.5 shrink-0 text-emerald-400/80" />
                  <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-300">{t.name}</span>
                  <span className="shrink-0 text-[10px] text-zinc-600">
                    {t.row_count.toLocaleString()}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Full browser modal, deep-linked to the clicked table */}
      {showBrowser && (
        <DatastoreBrowser
          host={agent.host}
          port={agent.port}
          auth={agent.auth}
          agentName={agent.name}
          initialTable={openTable || undefined}
          onClose={() => { setShowBrowser(false); setOpenTable(null); loadTables() }}
        />
      )}
    </div>
  )
}
