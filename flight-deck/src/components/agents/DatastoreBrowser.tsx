import { useState, useEffect } from 'react'
import { Database, Table2, ChevronRight, Loader2, AlertTriangle, RefreshCw, ChevronLeft, X } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'

interface TableInfo {
  name: string
  columns: { name: string; type: string; position: number }[]
  row_count: number
  created_at: string
  updated_at: string
}

interface QueryResult {
  columns: string[]
  rows: Record<string, any>[]
  total: number
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

interface DatastoreBrowserProps {
  host: string
  port: number
  auth?: string
  agentName: string
  onClose: () => void
}

export function DatastoreBrowser({ host, port, auth, agentName, onClose }: DatastoreBrowserProps) {
  const [tables, setTables] = useState<TableInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const [selectedTable, setSelectedTable] = useState<string | null>(null)
  const [rows, setRows] = useState<QueryResult | null>(null)
  const [rowsLoading, setRowsLoading] = useState(false)
  const [page, setPage] = useState(0)
  const pageSize = 50

  const tokenQs = auth ? `&token=${encodeURIComponent(auth)}` : ''

  const fetchTables = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await fdFetch<TableInfo[]>(`/agent-datastore/${host}/${port}/tables?_=1${tokenQs}`)
      setTables(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchTables() }, [host, port])

  const fetchRows = async (tableName: string, pageNum: number) => {
    setRowsLoading(true)
    setError('')
    try {
      const data = await fdFetch<QueryResult>(
        `/agent-datastore/${host}/${port}/tables/${encodeURIComponent(tableName)}/rows?limit=${pageSize}&offset=${pageNum * pageSize}${tokenQs}`
      )
      setRows(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRowsLoading(false)
    }
  }

  const openTable = (name: string) => {
    setSelectedTable(name)
    setPage(0)
    fetchRows(name, 0)
  }

  const changePage = (newPage: number) => {
    if (!selectedTable) return
    setPage(newPage)
    fetchRows(selectedTable, newPage)
  }

  const selectedTableInfo = tables.find((t) => t.name === selectedTable)
  const totalPages = rows ? Math.ceil(rows.total / pageSize) : 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="flex flex-col rounded-xl border border-zinc-700/50 bg-zinc-900 shadow-2xl"
        style={{ width: '80vw', height: '80vh' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800 shrink-0">
          <div className="flex items-center gap-2.5">
            <Database className="h-4 w-4 text-emerald-400" />
            <span className="text-sm font-medium text-zinc-200">
              Datastore — {agentName}
              {selectedTable && (
                <span className="text-zinc-500"> / {selectedTable}</span>
              )}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => { if (selectedTable) fetchRows(selectedTable, page); else fetchTables() }} className="text-zinc-500 hover:text-zinc-300 transition-colors" title="Refresh">
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
            <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 transition-colors">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {loading ? (
            <div className="flex items-center justify-center flex-1">
              <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
              <span className="ml-2 text-sm text-zinc-500">Loading tables...</span>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center flex-1">
              <AlertTriangle className="h-5 w-5 text-red-400 mr-2" />
              <span className="text-sm text-red-400">{error}</span>
            </div>
          ) : !selectedTable ? (
            /* ── Table List ── */
            <div className="flex-1 overflow-auto">
              {tables.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-zinc-500">
                  <Database className="h-8 w-8 mb-2 opacity-40" />
                  <p className="text-sm">No tables yet</p>
                  <p className="text-xs mt-1">Tables will appear here when the agent creates them</p>
                </div>
              ) : (
                <div className="divide-y divide-zinc-800">
                  {tables.map((table) => (
                    <button
                      key={table.name}
                      onClick={() => openTable(table.name)}
                      className="flex items-center gap-3 w-full px-5 py-3 text-left hover:bg-zinc-800/50 transition-colors"
                    >
                      <Table2 className="h-4 w-4 text-emerald-400/70 shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-zinc-200">{table.name}</div>
                        <div className="text-[11px] text-zinc-500 mt-0.5">
                          {table.columns.length} columns · {table.row_count} rows
                          {table.columns.length > 0 && (
                            <span className="ml-2 text-zinc-600">
                              ({table.columns.map((c) => `${c.name}: ${c.type}`).join(', ')})
                            </span>
                          )}
                        </div>
                      </div>
                      <ChevronRight className="h-4 w-4 text-zinc-600 shrink-0" />
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            /* ── Table Rows View ── */
            <>
              {/* Back button + table info */}
              <div className="flex items-center gap-2 px-4 py-2 border-b border-zinc-800 bg-zinc-950/30 shrink-0">
                <button
                  onClick={() => { setSelectedTable(null); setRows(null); setError('') }}
                  className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200"
                >
                  <ChevronLeft className="h-3 w-3" /> Tables
                </button>
                <span className="text-xs text-zinc-600">|</span>
                <span className="text-xs font-medium text-zinc-300">{selectedTable}</span>
                {selectedTableInfo && (
                  <span className="text-[10px] text-zinc-500">
                    {selectedTableInfo.row_count} rows · {selectedTableInfo.columns.length} columns
                  </span>
                )}
                {rowsLoading && <Loader2 className="h-3 w-3 animate-spin text-zinc-500 ml-auto" />}
              </div>

              {/* Data table */}
              <div className="flex-1 overflow-auto">
                {rows && rows.columns.length > 0 && rows.rows.length > 0 ? (
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-zinc-900 z-10">
                      <tr>
                        {rows.columns.map((col) => (
                          <th key={col} className="px-3 py-2 text-left font-medium text-zinc-400 border-b border-zinc-800 whitespace-nowrap">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800/50">
                      {rows.rows.map((row, i) => (
                        <tr key={i} className="hover:bg-zinc-800/30">
                          {rows.columns.map((col) => {
                            const cell = row[col]
                            return (
                              <td key={col} className="px-3 py-1.5 text-zinc-300 whitespace-nowrap max-w-[300px] truncate font-mono">
                                {cell === null || cell === undefined ? <span className="text-zinc-600 italic">null</span> : String(cell)}
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : rows && rows.rows.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-zinc-500 text-sm">
                    No rows in this table
                  </div>
                ) : !rowsLoading ? (
                  <div className="flex items-center justify-center h-full text-zinc-500 text-sm">
                    Loading...
                  </div>
                ) : null}
              </div>

              {/* Pagination */}
              {rows && totalPages > 1 && (
                <div className="flex items-center justify-between px-4 py-2 border-t border-zinc-800 bg-zinc-950/30 shrink-0">
                  <span className="text-[11px] text-zinc-500">
                    Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, rows.total)} of {rows.total}
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => changePage(page - 1)}
                      disabled={page === 0}
                      className="rounded px-2 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800 disabled:opacity-30"
                    >
                      Prev
                    </button>
                    <span className="text-[11px] text-zinc-500 px-2">{page + 1} / {totalPages}</span>
                    <button
                      onClick={() => changePage(page + 1)}
                      disabled={page >= totalPages - 1}
                      className="rounded px-2 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800 disabled:opacity-30"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
