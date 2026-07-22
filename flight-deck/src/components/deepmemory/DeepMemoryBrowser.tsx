import { useEffect, useMemo, useState } from 'react'
import {
  Database,
  Search,
  RefreshCw,
  FolderTree,
  Trash2,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  X,
  FileText,
  Loader2,
  Info,
} from 'lucide-react'
import { useDeepMemoryStore, type DMHit } from '../../stores/deepMemoryStore'
import { useVFSStore } from '../../stores/vfsStore'

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

// Same palette the VFS panel uses, so a folder reads the same in both places.
const KIND_BADGE: Record<string, string> = {
  basna: 'border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300',
  vatra: 'border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-300',
  council: 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300',
  link: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
}

const SOURCE_BADGE: Record<string, string> = {
  vfs: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
  agent: 'border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300',
  manual: 'border-zinc-600 bg-zinc-800 text-zinc-300',
}

/** Relevance is an absolute 0..1 score (the better of keyword coverage and
 *  cosine similarity), so the bar is meaningful across queries — unlike a
 *  positional rank score, where the top hit is always full. */
function ScoreBar({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(1, score)) * 100
  const tone = score >= 0.5 ? 'bg-emerald-500' : score >= 0.25 ? 'bg-amber-500' : 'bg-zinc-500'
  return (
    <span className="flex items-center gap-1.5" title={`relevance ${score.toFixed(3)}`}>
      <span className="h-1 w-10 overflow-hidden rounded-full bg-zinc-800">
        <span className={`block h-full ${tone}`} style={{ width: `${pct}%` }} />
      </span>
      <span className="font-mono text-[10px] text-zinc-500">{score.toFixed(2)}</span>
    </span>
  )
}

/** Vector health. The original bug was a collection declaring 1536-dim vectors
 *  while the provider emitted 256 — every vector silently discarded, hybrid
 *  search off, and nothing in the UI to show it. This panel exists to make
 *  that state impossible to miss. */
function HealthPanel() {
  const status = useDeepMemoryStore((s) => s.status)
  const claimUnowned = useDeepMemoryStore((s) => s.claimUnowned)
  if (!status) return null

  if (!status.enabled) {
    return (
      <div className="flex items-start gap-2 border-b border-zinc-800 bg-amber-500/5 px-4 py-3">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
        <div className="text-xs text-zinc-400">
          <div className="font-medium text-zinc-200">No Typesense connection</div>
          Set one up in{' '}
          <span className="font-medium text-zinc-300">Connections → Typesense (Deep Memory)</span>.
          Flight Deck holds the connection; agents reach the archive through Flight Deck and
          never need one of their own.
        </div>
      </div>
    )
  }

  const coll = status.embedding_dims ?? 0
  const prov = status.provider_dims ?? 0
  const mismatch = !!(coll && prov && coll !== prov)
  const noVectors = status.vectors_disabled || (!!prov && !coll)
  const bad = mismatch || noVectors

  return (
    <div
      className={`flex items-center gap-4 border-b border-zinc-800 px-4 py-2 text-xs ${
        bad ? 'bg-red-500/5' : 'bg-zinc-900/60'
      }`}
    >
      {bad ? (
        <AlertTriangle className="h-4 w-4 shrink-0 text-red-600 dark:text-red-400" />
      ) : (
        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
      )}
      <span className="text-zinc-400">
        collection <span className="font-mono text-zinc-200">{status.collection}</span>
      </span>
      <span className="text-zinc-500">·</span>
      <span className="text-zinc-400">
        vectors{' '}
        <span className={`font-mono ${mismatch ? 'text-red-600 dark:text-red-300' : 'text-zinc-200'}`}>
          {coll || '—'}
        </span>
        <span className="text-zinc-600"> / provider </span>
        <span className={`font-mono ${mismatch ? 'text-red-600 dark:text-red-300' : 'text-zinc-200'}`}>
          {prov || '—'}
        </span>
      </span>
      {bad && (
        <span className="text-red-600 dark:text-red-300">
          {mismatch
            ? 'width mismatch — vectors are being discarded, search is keyword-only'
            : 'vector search disabled — search is keyword-only'}
        </span>
      )}
      {!bad && <span className="text-emerald-700 dark:text-emerald-400/80">hybrid search active</span>}
      {status.error && <span className="truncate text-red-600 dark:text-red-300">{status.error}</span>}
      {!!status.unowned && (
        // Documents from before tenancy existed: still in the collection, but
        // owner-scoped search can never return them. Say so and offer the fix,
        // rather than auto-assigning them — on a multi-user Flight Deck,
        // guessing wrong hands one user another's archive.
        <span className="ml-auto flex items-center gap-2 text-amber-700 dark:text-amber-400">
          {status.unowned} document{status.unowned === 1 ? '' : 's'} with no owner — not searchable
          <button
            onClick={claimUnowned}
            className="rounded bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-200 hover:bg-zinc-700"
            title="Assign these to your account so they appear in search"
          >
            Claim
          </button>
        </span>
      )}
    </div>
  )
}

function Hit({ hit }: { hit: DMHit }) {
  const isVfs = hit.reference.startsWith('vfs:')
  const body = hit.summary || hit.snippet
  return (
    <div className="border-b border-zinc-800/60 px-4 py-3 hover:bg-zinc-900/40">
      <div className="mb-1 flex items-center gap-2">
        <span
          className={`rounded border px-1.5 py-0.5 text-[10px] ${
            SOURCE_BADGE[hit.source] || SOURCE_BADGE.manual
          }`}
        >
          {hit.source || 'doc'}
        </span>
        {isVfs ? (
          <FolderTree className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
        ) : (
          <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
        )}
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-zinc-200" title={hit.reference}>
          {hit.reference}
          {hit.start_line ? <span className="text-zinc-500">:{hit.start_line}</span> : null}
        </span>
        <ScoreBar score={hit.score} />
        <span className="shrink-0 text-[10px] text-zinc-500">{fmtTime(hit.updated_at)}</span>
      </div>
      <p className="line-clamp-3 whitespace-pre-wrap text-xs leading-relaxed text-zinc-400">{body}</p>
    </div>
  )
}

export function DeepMemoryBrowser() {
  const {
    status, projects, query, results, searched, loading, busy, error, notice,
    loadStatus, loadProjects, setQuery, search, clearResults,
    toggleIndexing, indexProject, dropProject, dismiss,
  } = useDeepMemoryStore()
  const vfsProjects = useVFSStore((s) => s.projects)
  const loadVfsProjects = useVFSStore((s) => s.loadProjects)
  const [summarize, setSummarize] = useState(false)
  const [confirmDrop, setConfirmDrop] = useState<string | null>(null)

  useEffect(() => {
    loadStatus()
    loadProjects()
    loadVfsProjects()
  }, [loadStatus, loadProjects, loadVfsProjects])

  // Every VFS folder is a candidate, annotated with its opt-in state — the
  // registry alone only knows about folders someone already toggled.
  const rows = useMemo(() => {
    const byName = new Map(vfsProjects.map((p) => [p.name, p]))
    const names = new Set<string>([
      ...vfsProjects.filter((p) => !p.shared).map((p) => p.name),
      ...Object.keys(projects),
    ])
    return [...names]
      .map((name) => {
        const p = byName.get(name)
        return {
          name,
          // Run folders are named after a session id (`vatra-0a66336d`), which
          // says nothing about what the run was. The VFS listing already
          // carries the run's human title — prefer it, and keep the folder name
          // as the secondary line so the two are still connectable.
          title: p?.title || '',
          kind: p?.kind || '',
          enabled: !!projects[name]?.enabled,
          files: p?.files ?? 0,
        }
      })
      .sort((a, b) =>
        (a.title || a.name).localeCompare(b.title || b.name, undefined, { sensitivity: 'base' }),
      )
  }, [vfsProjects, projects])

  const enabledCount = rows.filter((r) => r.enabled).length

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-12 items-center justify-between border-b border-zinc-800 px-4">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
          <Database className="h-4 w-4" />
          Deep Memory
          {/* "set to auto-index" rather than "indexed": this counts the opt-in
              registry, which persists even while deep memory is switched off —
              "2 folders indexed" would then claim content that isn't there. */}
          <span className="text-xs font-normal text-zinc-500">
            {enabledCount} folder{enabledCount === 1 ? '' : 's'} set to auto-index
          </span>
        </h2>
        <button
          onClick={() => { loadStatus(); loadProjects(); loadVfsProjects() }}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </div>

      <HealthPanel />

      {(error || notice) && (
        <div
          className={`flex items-start gap-2 border-b border-zinc-800 px-4 py-2 text-xs ${
            error ? 'bg-red-500/5 text-red-700 dark:text-red-300' : 'bg-zinc-900/60 text-zinc-300'
          }`}
        >
          {error ? (
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          ) : (
            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />
          )}
          <span className="min-w-0 flex-1 break-words">{error || notice}</span>
          <button onClick={dismiss} className="shrink-0 rounded p-0.5 text-zinc-500 hover:text-zinc-200">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      <div className="flex items-center gap-2 border-b border-zinc-800 bg-zinc-900/60 px-4 py-2">
        <Search className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && search()}
          placeholder="Search the archive — keyword and meaning"
          disabled={!status?.enabled}
          className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 outline-none focus:border-violet-600 disabled:opacity-50"
        />
        <button
          onClick={search}
          disabled={!status?.enabled || !query.trim()}
          className="rounded bg-zinc-800 px-2 py-1 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-40"
        >
          Search
        </button>
        {searched && (
          <button onClick={clearResults} className="rounded p-1 text-zinc-400 hover:text-zinc-200">
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      <div className="flex min-h-0 flex-1">
        {/* ── Folders ─────────────────────────────────────────── */}
        <div className="flex w-80 shrink-0 flex-col border-r border-zinc-800">
          <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
            <span className="text-xs font-medium text-zinc-300">VFS folders</span>
            <label
              className="flex cursor-pointer items-center gap-1 text-[10px] text-zinc-500 hover:text-zinc-300"
              title="Summaries cost one LLM call per ~1400 characters — a 100KB file is ~70 calls. Off by default."
            >
              <input
                type="checkbox"
                checked={summarize}
                onChange={(e) => setSummarize(e.target.checked)}
                className="h-3 w-3 accent-violet-600"
              />
              <Sparkles className="h-3 w-3" />
              summarise
            </label>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto">
            {rows.length === 0 && (
              <p className="px-4 py-6 text-center text-xs text-zinc-500">
                No VFS folders yet. Create one in the VFS panel.
              </p>
            )}
            {rows.map((r) => (
              <div key={r.name} className="border-b border-zinc-800/60 px-4 py-2.5">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={r.enabled}
                    disabled={!status?.enabled}
                    onChange={(e) => toggleIndexing(r.name, e.target.checked)}
                    className="h-3.5 w-3.5 shrink-0 accent-violet-600 disabled:opacity-40"
                    title="Keep this folder indexed automatically"
                  />
                  <span
                    className="min-w-0 flex-1 truncate text-xs text-zinc-200"
                    title={r.title ? `${r.title} — ${r.name}` : r.name}
                  >
                    {r.title || r.name}
                  </span>
                  {r.kind && KIND_BADGE[r.kind] && (
                    <span className={`shrink-0 rounded border px-1 py-0.5 text-[9px] ${KIND_BADGE[r.kind]}`}>
                      {r.kind}
                    </span>
                  )}
                  {busy === r.name && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-zinc-400" />}
                </div>
                {r.title && (
                  <div className="truncate pl-5.5 font-mono text-[10px] text-zinc-600" title={r.name}>
                    {r.name}
                  </div>
                )}
                <div className="mt-1.5 flex items-center gap-2 pl-5.5">
                  <button
                    onClick={() => indexProject(r.name, summarize)}
                    disabled={!status?.enabled || !!busy}
                    className="rounded bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-200 hover:bg-zinc-700 disabled:opacity-40"
                  >
                    Index now
                  </button>
                  {confirmDrop === r.name ? (
                    <>
                      <button
                        onClick={() => { dropProject(r.name); setConfirmDrop(null) }}
                        className="rounded bg-red-600/80 px-2 py-0.5 text-[10px] text-white hover:bg-red-600"
                      >
                        Confirm
                      </button>
                      <button
                        onClick={() => setConfirmDrop(null)}
                        className="rounded px-1 py-0.5 text-[10px] text-zinc-400 hover:text-zinc-200"
                      >
                        Cancel
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => setConfirmDrop(r.name)}
                      disabled={!status?.enabled || !!busy}
                      className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800 hover:text-red-600 dark:hover:text-red-300 disabled:opacity-40"
                      title="Remove this folder's content from the archive (files are not touched)"
                    >
                      <Trash2 className="h-3 w-3" />
                      Remove
                    </button>
                  )}
                  {r.files > 0 && <span className="ml-auto text-[10px] text-zinc-600">{r.files} files</span>}
                </div>
              </div>
            ))}
          </div>

          <p className="border-t border-zinc-800 px-4 py-2 text-[10px] leading-relaxed text-zinc-600">
            Indexed folders stay fresh on their own — edits re-index, deletes unlink, and
            unchanged files are skipped.
          </p>
        </div>

        {/* ── Results ─────────────────────────────────────────── */}
        <div className="min-h-0 flex-1 overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center gap-2 py-10 text-xs text-zinc-500">
              <Loader2 className="h-4 w-4 animate-spin" />
              Searching…
            </div>
          )}
          {!loading && !searched && (
            <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
              <Database className="h-8 w-8 text-zinc-700" />
              <p className="text-xs text-zinc-500">
                Search the long-term archive.
                <br />
                Agents search the same index through their <span className="font-mono">typesense</span> tool.
              </p>
            </div>
          )}
          {!loading && searched && results.length === 0 && (
            <p className="px-4 py-10 text-center text-xs text-zinc-500">
              Nothing matched. Only hits above the relevance floor are returned.
            </p>
          )}
          {!loading &&
            results.map((h) => <Hit key={`${h.reference}:${h.chunk_index}`} hit={h} />)}
        </div>
      </div>
    </div>
  )
}
