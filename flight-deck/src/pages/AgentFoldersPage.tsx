import { useEffect, useMemo, useState } from 'react'
import {
  HardDrive,
  RefreshCw,
  Trash2,
  Download,
  Eye,
  ChevronRight,
  ChevronDown,
  FileText,
  CheckCircle2,
  CircleSlash,
  Radio,
  Search,
  X,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useAgentFsStore, type AgentFolder } from '../stores/agentFsStore'
import { getFileTypeGroup, type AgentFile } from '../services/fileTransfer'

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

function fmtTime(ts?: number): string {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  const diff = Date.now() - d.getTime()
  if (diff < 60_000) return 'just now'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`
  if (diff < 7 * 86_400_000) return `${Math.floor(diff / 86_400_000)}d ago`
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

const fmtFull = (ts?: number) => (ts ? new Date(ts * 1000).toLocaleString([], { hour12: false }) : '')

const extOf = (name: string): string => {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i).toLowerCase() : ''
}

type Status = 'running' | 'desktop' | 'orphaned'
const statusOf = (f: AgentFolder): Status => (f.running ? 'running' : f.registered ? 'desktop' : 'orphaned')

export function AgentFoldersPage() {
  const s = useAgentFsStore()
  const [query, setQuery] = useState('')
  const [active, setActive] = useState<Set<Status>>(new Set())

  useEffect(() => {
    s.load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const toggle = (st: Status) =>
    setActive((prev) => {
      const next = new Set(prev)
      next.has(st) ? next.delete(st) : next.add(st)
      return next
    })

  const totalBytes = s.folders.reduce((a, f) => a + f.bytes, 0)
  const orphans = s.folders.filter((f) => !f.registered)
  const orphanBytes = orphans.reduce((a, f) => a + f.bytes, 0)
  const running = s.folders.filter((f) => f.running).length
  const onDesktop = s.folders.filter((f) => f.registered && !f.running).length

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return s.folders.filter((f) => {
      if (active.size && !active.has(statusOf(f))) return false
      if (q && !f.name.toLowerCase().includes(q)) return false
      return true
    })
  }, [s.folders, active, query])

  const FilterChip = ({ st, label, count, cls }: { st: Status; label: string; count: number; cls: string }) => {
    const on = active.has(st)
    return (
      <button
        onClick={() => toggle(st)}
        className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors ${
          on ? cls : 'border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200'
        }`}
      >
        {label} <span className="opacity-70">{count}</span>
      </button>
    )
  }

  return (
    <div className="flex h-full flex-col">
      {/* header */}
      <div className="flex h-12 items-center justify-between border-b border-zinc-800 px-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
          <HardDrive className="h-4 w-4 text-amber-600 dark:text-amber-400" /> Agent Folders
          <span className="text-xs font-normal text-zinc-500">fd-data cleanup</span>
        </div>
        <button
          onClick={() => s.load()}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${s.loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>

      {/* summary bar */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 border-b border-zinc-800 bg-zinc-900/40 px-4 py-2 text-xs text-zinc-400">
        <span><span className="font-semibold text-zinc-200">{s.folders.length}</span> folders</span>
        <span><span className="font-semibold text-zinc-200">{fmtBytes(totalBytes)}</span> total</span>
        <span className={orphans.length ? 'text-amber-700 dark:text-amber-300' : ''}>
          <span className="font-semibold">{orphans.length}</span> orphaned
          {orphans.length > 0 && <> · {fmtBytes(orphanBytes)} reclaimable</>}
        </span>
        {orphans.length > 0 && (
          <button
            onClick={() => {
              if (
                confirm(
                  `Delete all ${orphans.length} orphaned folder(s) (${fmtBytes(orphanBytes)})?\n\n` +
                    orphans.map((o) => `· ${o.name}`).join('\n'),
                )
              ) {
                ;(async () => {
                  for (const o of orphans) await s.deleteFolder(o.name)
                })()
              }
            }}
            className="ml-auto flex items-center gap-1 rounded border border-red-300 bg-red-50 px-2 py-1 text-[11px] text-red-700 hover:bg-red-100 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300 dark:hover:bg-red-900/50"
          >
            <Trash2 className="h-3.5 w-3.5" /> Remove all orphans
          </button>
        )}
      </div>

      {/* filter bar */}
      <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-4 py-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by name…"
            className="w-56 rounded-md border border-zinc-700 bg-zinc-950 py-1 pl-7 pr-7 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
          {query && (
            <button onClick={() => setQuery('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-zinc-500 hover:text-zinc-300">
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <FilterChip st="running" label="Running" count={running} cls="border-emerald-500/50 bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300" />
        <FilterChip st="desktop" label="On desktop" count={onDesktop} cls="border-sky-500/50 bg-sky-50 text-sky-700 dark:bg-sky-500/10 dark:text-sky-300" />
        <FilterChip st="orphaned" label="Orphaned" count={orphans.length} cls="border-amber-500/50 bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300" />
        {(active.size > 0 || query) && (
          <button
            onClick={() => { setActive(new Set()); setQuery('') }}
            className="text-[11px] text-zinc-500 hover:text-zinc-300"
          >
            Clear
          </button>
        )}
        <span className="ml-auto text-[11px] text-zinc-500">
          {filtered.length === s.folders.length ? `${s.folders.length}` : `${filtered.length} of ${s.folders.length}`} shown
        </span>
      </div>

      <div className="flex-1 overflow-auto p-3">
        {s.error && <div className="mb-3 rounded bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">{s.error}</div>}
        {filtered.length === 0 && !s.loading && (
          <div className="mt-16 text-center text-sm text-zinc-500">
            <HardDrive className="mx-auto mb-3 h-8 w-8 text-zinc-700" />
            {s.folders.length === 0 ? 'No agent folders found.' : 'No folders match the current filters.'}
          </div>
        )}
        <div className="flex flex-col gap-1.5">
          {filtered.map((f) => (
            <FolderRow key={f.name} folder={f} />
          ))}
        </div>
      </div>

      {s.preview && <PreviewModal />}
    </div>
  )
}

function FolderRow({ folder: f }: { folder: AgentFolder }) {
  const s = useAgentFsStore()
  const open = s.expanded === f.name
  const orphaned = !f.registered

  return (
    <div
      className={`rounded-lg border bg-zinc-900/50 ${
        orphaned ? 'border-amber-300 dark:border-amber-900/50' : 'border-zinc-800'
      }`}
    >
      <div className="flex items-center gap-2 px-3 py-2">
        <button onClick={() => s.toggleFolder(f.name)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
          {open ? (
            <ChevronDown className="h-4 w-4 shrink-0 text-zinc-500" />
          ) : (
            <ChevronRight className="h-4 w-4 shrink-0 text-zinc-500" />
          )}
          <span className="truncate font-mono text-sm text-zinc-100">{f.name}</span>
          {/* presence badge */}
          {f.running ? (
            <span className="flex shrink-0 items-center gap-1 rounded border border-emerald-500/50 bg-emerald-50 px-1.5 py-0.5 text-[9px] font-medium uppercase text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300">
              <Radio className="h-2.5 w-2.5" /> Running
            </span>
          ) : f.registered ? (
            <span className="flex shrink-0 items-center gap-1 rounded border border-sky-500/50 bg-sky-50 px-1.5 py-0.5 text-[9px] font-medium uppercase text-sky-700 dark:bg-sky-500/10 dark:text-sky-300">
              <CheckCircle2 className="h-2.5 w-2.5" /> On desktop
            </span>
          ) : (
            <span className="flex shrink-0 items-center gap-1 rounded border border-amber-500/50 bg-amber-50 px-1.5 py-0.5 text-[9px] font-medium uppercase text-amber-700 dark:bg-amber-500/10 dark:text-amber-300">
              <CircleSlash className="h-2.5 w-2.5" /> Orphaned
            </span>
          )}
        </button>
        <div className="flex shrink-0 items-center gap-3 text-[11px] text-zinc-500">
          <span className="font-medium text-zinc-300">{fmtBytes(f.bytes)}</span>
          <span>{f.files} file{f.files !== 1 ? 's' : ''}</span>
          <span className="hidden sm:inline">{f.workspace_files} in ws</span>
          {f.mtime ? <span title={fmtFull(f.mtime)}>{fmtTime(f.mtime)}</span> : null}
        </div>
        <button
          onClick={() => {
            if (f.running) {
              alert('Agent is still running. Stop it from the Agent Desktop first.')
              return
            }
            if (confirm(`Delete folder "${f.name}" (${fmtBytes(f.bytes)})? This cannot be undone.`)) {
              s.deleteFolder(f.name)
            }
          }}
          className="shrink-0 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-red-600 disabled:opacity-30 dark:hover:text-red-400"
          disabled={f.running}
          title={f.running ? 'Stop the agent first' : 'Delete folder'}
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>

      {open && (
        <div className="border-t border-zinc-800/80 px-3 py-2">
          {s.filesLoading && <div className="py-2 text-center text-xs text-zinc-600">Loading…</div>}
          {!s.filesLoading && s.expandedFiles.length === 0 && (
            <div className="py-2 text-center text-xs text-zinc-600">No files in workspace</div>
          )}
          {s.expandedFiles.map((file) => (
            <div key={file.path} className="group flex items-center gap-2 rounded px-1 py-1 text-sm hover:bg-zinc-800/50">
              <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
              <span className="min-w-0 flex-1 truncate font-mono text-xs text-zinc-300" title={file.path}>
                {file.path}
              </span>
              <span className="shrink-0 text-[11px] text-zinc-600">{fmtBytes(file.size)}</span>
              <div className="flex shrink-0 items-center gap-0.5 opacity-0 group-hover:opacity-100">
                <button
                  onClick={() => s.openFile(f.name, file)}
                  className="rounded p-1 text-zinc-500 hover:bg-zinc-700 hover:text-zinc-200"
                  title="View"
                >
                  <Eye className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={() => s.download(f.name, file)}
                  className="rounded p-1 text-zinc-500 hover:bg-zinc-700 hover:text-zinc-200"
                  title="Download"
                >
                  <Download className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function PreviewModal() {
  const s = useAgentFsStore()
  const p = s.preview!
  const ext = extOf(p.name)
  const group = getFileTypeGroup({ extension: ext, filename: p.name } as AgentFile)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={s.closeFile}>
      <div
        className="flex max-h-[88vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg border border-zinc-700 bg-zinc-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2.5">
          <div className="flex min-w-0 items-center gap-2">
            <span className="truncate font-mono text-sm text-zinc-200" title={p.path}>{p.name}</span>
            <span className="shrink-0 font-mono text-[11px] text-zinc-500">{ext}</span>
            <span className="shrink-0 text-[11px] text-zinc-600">{fmtBytes(p.size)}</span>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <button
              onClick={() => s.download(p.folder, { path: p.path, name: p.name, size: p.size, mtime: 0, ext, is_text: !p.binary })}
              className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
              title="Download"
            >
              <Download className="h-3.5 w-3.5" />
            </button>
            <button onClick={s.closeFile} className="rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200" title="Close">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-auto">
          {s.previewLoading && <div className="py-20 text-center text-sm text-zinc-500">Loading…</div>}
          {s.previewError && <div className="m-4 rounded bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">{s.previewError}</div>}

          {!s.previewLoading && !s.previewError && (
            <>
              {/* image */}
              {s.blobUrl && (
                <div className="flex min-h-[300px] items-center justify-center bg-zinc-900/50 p-6">
                  <img src={s.blobUrl} alt={p.name} className="max-h-[72vh] max-w-full rounded-lg object-contain" />
                </div>
              )}

              {/* too large / binary */}
              {!s.blobUrl && p.truncated && (
                <div className="px-6 py-16 text-center text-sm text-zinc-500">File too large to preview — download it instead.</div>
              )}
              {!s.blobUrl && !p.truncated && p.binary && group !== 'image' && (
                <div className="px-6 py-16 text-center text-sm text-zinc-500">Binary file — download it to view.</div>
              )}

              {/* HTML — render in a sandboxed iframe */}
              {!s.blobUrl && !p.truncated && !p.binary && group === 'html' && (
                <div className="min-h-[300px] bg-white">
                  <iframe srcDoc={p.text} title={p.name} className="w-full border-0" style={{ height: '72vh' }} sandbox="allow-scripts allow-same-origin" />
                </div>
              )}

              {/* Markdown — GFM (tables, etc.) */}
              {!s.blobUrl && !p.truncated && !p.binary && group === 'markdown' && (
                <div className="fd-file-markdown p-6"><Markdown remarkPlugins={[remarkGfm]}>{p.text}</Markdown></div>
              )}

              {/* JSON pretty-print */}
              {!s.blobUrl && !p.truncated && !p.binary && group === 'data' && ext === '.json' && (
                <pre className="whitespace-pre-wrap break-words p-6 font-mono text-xs leading-relaxed text-zinc-300">
                  {(() => { try { return JSON.stringify(JSON.parse(p.text), null, 2) } catch { return p.text } })()}
                </pre>
              )}

              {/* everything else — plain text */}
              {!s.blobUrl && !p.truncated && !p.binary && group !== 'html' && group !== 'markdown' && group !== 'image' && !(group === 'data' && ext === '.json') && (
                <pre className="whitespace-pre-wrap break-words p-6 font-mono text-xs leading-relaxed text-zinc-300">{p.text}</pre>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
