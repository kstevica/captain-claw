import { useEffect, useState, useCallback } from 'react'
import type { AgentManifest, FeedDef, EntityDef } from '../types'
import { callTool, extractRows } from '../mcp'
import { useAppRuntime, applyTemplates } from '../store'

interface Props {
  manifest: AgentManifest
  feed: FeedDef
  filter?: Record<string, string>   // template-able
}

export function EntityList({ manifest, feed, filter }: Props) {
  const [rows, setRows] = useState<Record<string, unknown>[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const selectedEntity = useAppRuntime((s) => s.selectedEntity)
  const selectEntity = useAppRuntime((s) => s.selectEntity)

  const entity: EntityDef | undefined = manifest.entities[feed.returns]

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    const staticArgs = feed.arguments ?? {}
    const filterArgs = applyTemplates(filter, selectedEntity)
    const args = { ...staticArgs, ...filterArgs }
    const res = await callTool(manifest.agent.id, manifest.agent.mcp_server, feed.mcp_tool, args)
    if (!res.ok) {
      setError(res.error ?? 'unknown error')
      setRows([])
    } else {
      setRows(extractRows(res.result))
    }
    setLoading(false)
  }, [manifest.agent.mcp_server, feed.mcp_tool, feed.arguments, filter, selectedEntity])

  useEffect(() => {
    load()
    if (!feed.refresh_seconds) return
    const t = setInterval(load, feed.refresh_seconds * 1000)
    return () => clearInterval(t)
  }, [load, feed.refresh_seconds])

  const titleField = entity
    ? Object.entries(entity.fields).find(([, f]) => f.title)?.[0] ?? 'name'
    : 'name'
  const primaryField = entity
    ? Object.entries(entity.fields).find(([, f]) => f.primary)?.[0] ?? 'id'
    : 'id'

  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-900/40">
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
        <h3 className="text-sm font-semibold text-zinc-200">{feed.label}</h3>
        <div className="flex items-center gap-2">
          {loading && <span className="text-[10px] text-zinc-500">loading…</span>}
          <button
            onClick={load}
            className="rounded px-2 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
          >
            refresh
          </button>
        </div>
      </header>
      <div className="divide-y divide-zinc-800/60">
        {error && (
          <div className="px-4 py-3 text-xs text-red-400">{prettyFeedError(error)}</div>
        )}
        {!error && rows.length === 0 && !loading && (
          <div className="px-4 py-6 text-center text-xs text-zinc-500">No items</div>
        )}
        {rows.map((row, i) => {
          const id = String(row[primaryField] ?? row.id ?? i)
          const title = String(row[titleField] ?? row.name ?? row.title ?? id)
          return (
            <button
              key={id + ':' + i}
              onClick={() => selectEntity(feed.returns, id, row)}
              className="flex w-full items-center justify-between px-4 py-2 text-left hover:bg-zinc-800/50"
            >
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm text-zinc-200">{title}</div>
                <SubtitleFields row={row} entity={entity} titleField={titleField} primaryField={primaryField} />
              </div>
              <span className="text-zinc-600">›</span>
            </button>
          )
        })}
      </div>
    </section>
  )
}

function prettyFeedError(raw: string): string {
  // The MCP proxy reports unknown servers as `404: {"detail":"No MCP server named 'x'"}`.
  // Strip the HTTP framing so the user sees the human message.
  const m = raw.match(/^\d+:\s*({.*})$/s)
  if (m) {
    try {
      const j = JSON.parse(m[1]) as { detail?: string; error?: string; message?: string }
      return j.detail || j.message || j.error || raw
    } catch { /* fall through */ }
  }
  return raw
}

function SubtitleFields({
  row,
  entity,
  titleField,
  primaryField,
}: {
  row: Record<string, unknown>
  entity: EntityDef | undefined
  titleField: string
  primaryField: string
}) {
  if (!entity) return null
  const others = Object.keys(entity.fields)
    .filter((f) => f !== titleField && f !== primaryField)
    .slice(0, 3)
    .map((f) => ({ key: f, value: row[f] }))
    .filter((p) => p.value != null && p.value !== '')
  if (others.length === 0) return null
  return (
    <div className="mt-0.5 flex gap-3 text-[11px] text-zinc-500">
      {others.map((p) => (
        <span key={p.key}>
          <span className="text-zinc-600">{p.key}:</span> {String(p.value)}
        </span>
      ))}
    </div>
  )
}
