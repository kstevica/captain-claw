import { useEffect, useMemo, useRef, useState } from 'react'
import {
  FolderTree,
  Folder,
  FileText,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Search,
  RefreshCw,
  Download,
  Trash2,
  FolderPlus,
  Link2,
  Lock,
  X,
  ArrowLeft,
  HardDrive,
  Upload,
  Database,
  Share2,
  Users,
} from 'lucide-react'
import { useVFSStore, type VFSEntry, type VFSProject } from '../../stores/vfsStore'
import { VFSFileViewer } from './VFSFileViewer'
import { DatastoreBrowser } from '../agents/DatastoreBrowser'
import { ShareModal } from '../common/ShareModal'
import { leaveShare } from '../../services/shares'

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

// Compact relative timestamp from a unix-seconds mtime.
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

function fmtFull(ts?: number): string {
  return ts ? new Date(ts * 1000).toLocaleString([], { hour12: false }) : ''
}

const KIND_BADGE: Record<string, string> = {
  basna: 'border-sky-500/40 bg-sky-500/10 text-sky-300',
  vatra: 'border-violet-500/40 bg-violet-500/10 text-violet-300',
  council: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
  link: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
}

// Project-list filter chips. A "run folder" is one a multi-agent run auto-created
// (kind basna/vatra/council); a "project" is a plain user/agent folder (no kind).
type KindFilter = 'all' | 'project' | 'basna' | 'vatra' | 'council' | 'link'
type SortBy = 'recent' | 'name' | 'size'
const CHIPS: { key: KindFilter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'project', label: 'Projects' },
  { key: 'vatra', label: 'Vatra' },
  { key: 'basna', label: 'Basna' },
  { key: 'council', label: 'Council' },
  { key: 'link', label: 'Linked' },
]

const _isRun = (p: VFSProject) => p.kind === 'basna' || p.kind === 'vatra' || p.kind === 'council'
// A raw run folder = an agent run with no human title (just its hash id). These are
// folded away by default so named projects and titled runs lead the list.
const _isRawRun = (p: VFSProject) => _isRun(p) && !p.title
const _matchesKind = (p: VFSProject, k: KindFilter) =>
  k === 'all' ? true : k === 'project' ? !p.kind : p.kind === k

export function VFSBrowser() {
  const s = useVFSStore()
  const [creating, setCreating] = useState(false)
  const [folderName, setFolderName] = useState('')
  const [creatingProject, setCreatingProject] = useState(false)
  const [projectName, setProjectName] = useState('')
  const [linking, setLinking] = useState(false)
  const [linkName, setLinkName] = useState('')
  const [linkPath, setLinkPath] = useState('')
  const [linkMode, setLinkMode] = useState('rw')
  const [linkErr, setLinkErr] = useState('')

  // Shared-datastore viewer: the project whose vfs:<project>/.datastore is open.
  const [dsProject, setDsProject] = useState<string | null>(null)
  const [shareProject, setShareProject] = useState<string | null>(null)
  // Project-list controls: search, kind filter, sort, and folded run folders.
  const [query, setQuery] = useState('')
  const [kindFilter, setKindFilter] = useState<KindFilter>('all')
  const [sortBy, setSortBy] = useState<SortBy>(() => {
    try { return (localStorage.getItem('vfs.sortBy') as SortBy) || 'recent' } catch { return 'recent' }
  })
  const [showRuns, setShowRuns] = useState(false)
  const onSort = (v: SortBy) => {
    setSortBy(v)
    try { localStorage.setItem('vfs.sortBy', v) } catch { /* ignore */ }
  }

  const onAddLink = async () => {
    setLinkErr('')
    try {
      await s.addLink(linkName.trim(), linkPath.trim(), linkMode)
      setLinking(false); setLinkName(''); setLinkPath('')
    } catch (e) {
      setLinkErr(e instanceof Error ? e.message : 'link failed')
    }
  }

  const onCreateProject = async () => {
    if (!projectName.trim()) return
    await s.newProject(projectName)
    setProjectName(''); setCreatingProject(false)
  }

  // File upload into the current directory (in-project view).
  const uploadInputRef = useRef<HTMLInputElement | null>(null)
  const [dragging, setDragging] = useState(false)

  // Server-side folder picker (FD runs on the user's machine).
  const [fsOpen, setFsOpen] = useState(false)
  const [fsPath, setFsPath] = useState('')
  const [fsParent, setFsParent] = useState('')
  const [fsDirs, setFsDirs] = useState<{ name: string; hidden: boolean; is_git: boolean }[]>([])
  const [showHidden, setShowHidden] = useState(false)

  const fsGo = async (path: string) => {
    const r = await s.browseFs(path)
    setFsPath(r.path); setFsParent(r.parent); setFsDirs(r.dirs)
  }
  const openBrowse = async () => { setFsOpen(true); await fsGo(linkPath.trim()) }
  const useFolder = () => {
    setLinkPath(fsPath)
    if (!linkName.trim()) setLinkName((fsPath.split('/').pop() || '').replace(/[^a-zA-Z0-9._-]/g, '-'))
    setFsOpen(false)
  }

  useEffect(() => {
    if (!s.project) s.loadProjects()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Kind counts for the chips (from the full list, so they don't jitter while filtering).
  const counts = useMemo(() => {
    const c: Record<KindFilter, number> = { all: s.projects.length, project: 0, basna: 0, vatra: 0, council: 0, link: 0 }
    for (const p of s.projects) {
      if (!p.kind) c.project++
      else if (p.kind === 'basna' || p.kind === 'vatra' || p.kind === 'council' || p.kind === 'link') c[p.kind]++
    }
    return c
  }, [s.projects])

  // Search + kind filter + sort. Default sort is most-recent so fresh runs lead.
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const list = s.projects.filter((p) =>
      _matchesKind(p, kindFilter)
      && (!q || p.name.toLowerCase().includes(q) || (p.title || '').toLowerCase().includes(q)))
    return list.sort((a, b) => {
      if (sortBy === 'name') return (a.title || a.name).localeCompare(b.title || b.name)
      if (sortBy === 'size') return b.bytes - a.bytes
      return (b.mtime || 0) - (a.mtime || 0)
    })
  }, [s.projects, query, kindFilter, sortBy])

  // ── Project list view ──────────────────────────────────────────────
  if (!s.project) {
    // Fold the raw (untitled) run folders unless the user is narrowing or expands them.
    const noNarrow = kindFilter === 'all' && !query.trim()
    const rawRuns = noNarrow ? filtered.filter(_isRawRun) : []
    const shown = noNarrow && !showRuns ? filtered.filter((p) => !_isRawRun(p)) : filtered
    return (
      <div className="flex h-full flex-col">
        <div className="flex h-12 items-center justify-between border-b border-zinc-800 px-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
            <FolderTree className="h-4 w-4 text-violet-400" /> Shared VFS
            <span className="text-xs font-normal text-zinc-500">cross-agent filesystem</span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => { setCreatingProject((v) => !v); setLinking(false) }}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <FolderPlus className="h-3.5 w-3.5" /> New project
            </button>
            <button
              onClick={() => { setLinking((v) => !v); setCreatingProject(false) }}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <Link2 className="h-3.5 w-3.5" /> Link folder
            </button>
            <button
              onClick={() => s.loadProjects()}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${s.loading ? 'animate-spin' : ''}`} /> Refresh
            </button>
          </div>
        </div>
        {creatingProject && (
          <div className="flex items-center gap-2 border-b border-zinc-800 bg-zinc-900/60 px-4 py-2">
            <input
              autoFocus
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') onCreateProject()
                else if (e.key === 'Escape') { setCreatingProject(false); setProjectName('') }
              }}
              placeholder="project name"
              className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 outline-none focus:border-violet-600"
            />
            <button
              onClick={onCreateProject}
              disabled={!projectName.trim()}
              className="rounded bg-violet-600 px-2 py-1 text-xs text-white hover:bg-violet-500 disabled:opacity-40"
            >
              Create
            </button>
            <button onClick={() => { setCreatingProject(false); setProjectName('') }} className="rounded p-1 text-zinc-400 hover:text-zinc-200">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
        {linking && (
          <div className="border-b border-zinc-800 bg-zinc-950/60 px-4 py-3">
            <div className="mb-2 text-xs font-medium text-zinc-300">Link an existing local folder into the VFS</div>
            <div className="flex flex-wrap items-center gap-2">
              <input
                value={linkName}
                onChange={(e) => setLinkName(e.target.value)}
                placeholder="vfs name (e.g. my-repo)"
                className="w-40 rounded bg-zinc-900 px-2 py-1 text-xs text-zinc-100 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600"
              />
              <input
                value={linkPath}
                onChange={(e) => setLinkPath(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && onAddLink()}
                placeholder="/absolute/path/to/folder"
                className="min-w-0 flex-1 rounded bg-zinc-900 px-2 py-1 font-mono text-xs text-zinc-100 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600"
              />
              <button
                onClick={openBrowse}
                className="flex items-center gap-1 rounded bg-zinc-800 px-2 py-1 text-xs text-zinc-200 hover:bg-zinc-700"
                title="Browse folders on this machine"
              >
                <FolderTree className="h-3.5 w-3.5" /> Browse
              </button>
              <select
                value={linkMode}
                onChange={(e) => setLinkMode(e.target.value)}
                className="rounded bg-zinc-900 px-2 py-1 text-xs text-zinc-200 outline-none ring-1 ring-zinc-800"
                title="Read-write lets agents modify the folder; read-only is browse-only"
              >
                <option value="rw">read-write</option>
                <option value="ro">read-only</option>
              </select>
              <button
                onClick={onAddLink}
                disabled={!linkName.trim() || !linkPath.trim()}
                className="rounded bg-emerald-600/80 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-600 disabled:opacity-40"
              >
                Link
              </button>
            </div>
            {linkErr && <div className="mt-2 text-xs text-red-400">{linkErr}</div>}
          </div>
        )}
        {fsOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-8" onClick={() => setFsOpen(false)}>
            <div className="flex max-h-full w-full max-w-2xl flex-col rounded-lg border border-zinc-800 bg-zinc-950" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2">
                <FolderTree className="h-4 w-4 text-violet-400" />
                <span className="truncate font-mono text-xs text-zinc-300">{fsPath || '~'}</span>
                <label className="ml-auto flex items-center gap-1 text-[11px] text-zinc-500">
                  <input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} /> hidden
                </label>
                <button onClick={() => setFsOpen(false)} className="text-zinc-500 hover:text-zinc-200"><X className="h-4 w-4" /></button>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto p-2">
                {fsParent && (
                  <button onClick={() => fsGo(fsParent)} className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm text-zinc-400 hover:bg-zinc-900">
                    <ArrowLeft className="h-4 w-4" /> ..
                  </button>
                )}
                {fsDirs.filter((d) => showHidden || !d.hidden).map((d) => (
                  <button
                    key={d.name}
                    onClick={() => fsGo(`${fsPath}/${d.name}`)}
                    className={`flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm hover:bg-zinc-900 ${d.hidden ? 'text-zinc-500' : 'text-zinc-200'}`}
                  >
                    <Folder className="h-4 w-4 shrink-0 text-violet-400" />
                    <span className="truncate">{d.name}</span>
                    {d.is_git && <span className="ml-auto shrink-0 rounded border border-emerald-500/40 bg-emerald-500/10 px-1 text-[9px] uppercase text-emerald-300">git</span>}
                  </button>
                ))}
                {fsDirs.length === 0 && <div className="px-2 py-4 text-xs text-zinc-500">No sub-folders.</div>}
              </div>
              <div className="flex items-center justify-between gap-2 border-t border-zinc-800 px-4 py-2">
                <span className="truncate font-mono text-[11px] text-zinc-500">{fsPath}</span>
                <button onClick={useFolder} className="shrink-0 rounded bg-emerald-600/80 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-600">
                  Use this folder
                </button>
              </div>
            </div>
          </div>
        )}
        {s.projects.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 bg-zinc-950/40 px-4 py-2">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search projects…"
                className="w-52 rounded bg-zinc-900 py-1 pl-7 pr-2 text-xs text-zinc-100 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600"
              />
            </div>
            <div className="flex flex-wrap items-center gap-1">
              {CHIPS.filter((c) => c.key === 'all' || counts[c.key] > 0).map((c) => (
                <button
                  key={c.key}
                  onClick={() => setKindFilter(c.key)}
                  className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors ${
                    kindFilter === c.key
                      ? 'bg-violet-600 text-white'
                      : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
                  }`}
                >
                  {c.label}
                  <span className={`ml-1 tabular-nums ${kindFilter === c.key ? 'text-violet-200' : 'text-zinc-600'}`}>
                    {counts[c.key]}
                  </span>
                </button>
              ))}
            </div>
            <select
              value={sortBy}
              onChange={(e) => onSort(e.target.value as SortBy)}
              className="ml-auto rounded bg-zinc-900 px-2 py-1 text-xs text-zinc-300 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600"
              title="Sort projects"
            >
              <option value="recent">Recent</option>
              <option value="name">Name</option>
              <option value="size">Size</option>
            </select>
          </div>
        )}
        <div className="flex-1 overflow-auto p-4">
          {s.error && <div className="mb-3 rounded bg-red-950/50 px-3 py-2 text-xs text-red-300">{s.error}</div>}
          {s.projects.length === 0 && !s.loading && (
            <div className="mt-16 text-center text-sm text-zinc-500">
              <HardDrive className="mx-auto mb-3 h-8 w-8 text-zinc-700" />
              No projects yet. Agents create them by writing{' '}
              <code className="rounded bg-zinc-800 px-1 text-zinc-300">vfs:&lt;project&gt;/&lt;file&gt;</code>.
            </div>
          )}
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {shown.map((p) => (
              <div
                key={p.name}
                className="group flex flex-col gap-1 rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 hover:border-violet-700/60"
              >
                <button onClick={() => s.openProject(p.name, p.owner_id, p.permission)} className="flex items-center gap-2 text-left">
                  <Folder className="h-4 w-4 shrink-0 text-violet-400" />
                  <span className="truncate text-sm font-medium text-zinc-100">{p.title || p.name}</span>
                  {p.kind && p.kind !== 'shared' && (
                    <span className={`shrink-0 rounded border px-1.5 py-0.5 text-[9px] font-medium uppercase ${KIND_BADGE[p.kind] || 'border-zinc-700 text-zinc-400'}`}>
                      {p.kind}
                    </span>
                  )}
                  {p.shared && (
                    <span className="flex shrink-0 items-center gap-0.5 rounded border border-sky-500/25 bg-sky-500/15 px-1 py-0.5 text-[9px] font-medium text-sky-300"
                      title={`Shared by ${p.owner_name || p.owner_email || 'another user'} · ${p.permission}`}>
                      <Users className="h-2.5 w-2.5" />{p.permission === 'edit' ? 'edit' : 'view'}
                    </span>
                  )}
                  {p.mode === 'ro' && <Lock className="h-3 w-3 shrink-0 text-zinc-500" aria-label="read-only" />}
                </button>
                {p.kind === 'link'
                  ? <span className={`truncate font-mono text-[10px] ${p.missing ? 'text-red-400' : 'text-zinc-600'}`} title={p.link_path}>
                      {p.missing ? '⚠ missing: ' : '↪ '}{p.link_path}
                    </span>
                  : p.title && <span className="truncate font-mono text-[10px] text-zinc-600">{p.name}</span>}
                <div className="flex items-center justify-between text-[11px] text-zinc-500">
                  <span title={fmtFull(p.mtime)}>
                    {p.files} file{p.files !== 1 ? 's' : ''} · {fmtBytes(p.bytes)}
                    {p.mtime ? ` · ${fmtTime(p.mtime)}` : ''}
                  </span>
                  <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                    {p.kind !== 'link' && (
                      <button
                        onClick={() => setDsProject(p.name)}
                        className="hover:text-emerald-300"
                        title="Shared datastore — browse the folder's tables"
                      >
                        <Database className="h-3.5 w-3.5" />
                      </button>
                    )}
                    <button
                      onClick={() => s.downloadProject(p.name)}
                      className="hover:text-violet-300"
                      title="Download folder as .zip"
                    >
                      <Download className="h-3.5 w-3.5" />
                    </button>
                    {p.shared ? (
                      <button
                        onClick={async () => {
                          if (confirm(`Remove shared folder "${p.name}" from your view?`)) {
                            await leaveShare('vfs', p.name, p.owner_id || '')
                            s.loadProjects()
                          }
                        }}
                        className="hover:text-red-400"
                        title="Remove from my shared folders"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    ) : (
                      <>
                        {p.kind !== 'link' && (
                          <button
                            onClick={() => setShareProject(p.name)}
                            className="hover:text-sky-300"
                            title="Share with other users"
                          >
                            <Share2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                        <button
                          onClick={() => {
                            const msg = p.kind === 'link'
                              ? `Unlink "${p.name}"? Your real files at ${p.link_path} are NOT deleted.`
                              : `Delete project "${p.name}" and all its files?`
                            if (confirm(msg)) s.deleteProject(p.name)
                          }}
                          className="hover:text-red-400"
                          title={p.kind === 'link' ? 'Unlink (keeps real files)' : 'Delete project'}
                        >
                          {p.kind === 'link' ? <Link2 className="h-3.5 w-3.5" /> : <Trash2 className="h-3.5 w-3.5" />}
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
          {shown.length === 0 && rawRuns.length === 0 && s.projects.length > 0 && (
            <div className="mt-12 text-center text-sm text-zinc-600">No projects match.</div>
          )}
          {rawRuns.length > 0 && (
            <button
              onClick={() => setShowRuns((v) => !v)}
              className="mt-3 flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-zinc-800 py-2 text-xs text-zinc-500 hover:border-zinc-700 hover:text-zinc-300"
            >
              {showRuns
                ? <><ChevronUp className="h-3.5 w-3.5" /> Hide agent run folders</>
                : <><ChevronDown className="h-3.5 w-3.5" /> Show {rawRuns.length} agent run folder{rawRuns.length !== 1 ? 's' : ''}</>}
            </button>
          )}
        </div>
        {dsProject && (
          <DatastoreBrowser vfsProject={dsProject} title={`Datastore — ${dsProject}`} onClose={() => setDsProject(null)} />
        )}
      </div>
    )
  }

  // ── In-project browser ─────────────────────────────────────────────
  const segs = s.path ? s.path.split('/') : []
  const parentPath = segs.slice(0, -1).join('/')

  return (
    <div className="flex h-full flex-col">
      {/* breadcrumb + actions */}
      <div className="flex h-12 items-center justify-between border-b border-zinc-800 px-4">
        <div className="flex min-w-0 items-center gap-1 text-sm">
          <button onClick={s.closeProject} className="mr-1 rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200" title="All projects">
            <ArrowLeft className="h-4 w-4" />
          </button>
          <button onClick={() => s.browse('')} className="flex items-center gap-1 font-medium text-violet-300 hover:text-violet-200">
            <FolderTree className="h-4 w-4" /> {s.project}
          </button>
          {segs.map((seg, i) => {
            const sub = segs.slice(0, i + 1).join('/')
            return (
              <span key={sub} className="flex min-w-0 items-center gap-1">
                <ChevronRight className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
                <button onClick={() => s.browse(sub)} className="truncate text-zinc-300 hover:text-zinc-100">
                  {seg}
                </button>
              </span>
            )
          })}
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => setCreating(true)} className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
            <FolderPlus className="h-3.5 w-3.5" /> New folder
          </button>
          <button onClick={() => uploadInputRef.current?.click()} className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
            <Upload className="h-3.5 w-3.5" /> Upload
          </button>
          <input
            ref={uploadInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => { if (e.target.files?.length) s.uploadFiles(e.target.files); e.target.value = '' }}
          />
          {s.project && (
            <button onClick={() => setDsProject(s.project)} className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-emerald-300" title="Shared datastore — browse this folder's tables">
              <Database className="h-3.5 w-3.5" /> Datastore
            </button>
          )}
          <button onClick={() => s.refresh()} className="rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200" title="Refresh">
            <RefreshCw className={`h-3.5 w-3.5 ${s.loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>
      {dsProject && (
        <DatastoreBrowser vfsProject={dsProject} title={`Datastore — ${dsProject}`} onClose={() => setDsProject(null)} />
      )}
      {shareProject && (
        <ShareModal
          resourceType="vfs"
          resourceId={shareProject}
          resourceName={shareProject}
          onClose={() => setShareProject(null)}
        />
      )}

      {creating && (
        <div className="flex items-center gap-2 border-b border-zinc-800 bg-zinc-900/60 px-4 py-2">
          <input
            autoFocus
            value={folderName}
            onChange={(e) => setFolderName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                s.newFolder(folderName)
                setFolderName('')
                setCreating(false)
              } else if (e.key === 'Escape') {
                setCreating(false)
                setFolderName('')
              }
            }}
            placeholder="folder name"
            className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 outline-none focus:border-violet-600"
          />
          <button
            onClick={() => {
              s.newFolder(folderName)
              setFolderName('')
              setCreating(false)
            }}
            className="rounded bg-violet-600 px-2 py-1 text-xs text-white hover:bg-violet-500"
          >
            Create
          </button>
          <button onClick={() => { setCreating(false); setFolderName('') }} className="rounded p-1 text-zinc-400 hover:text-zinc-200">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      <div
        className="relative flex min-h-0 flex-1"
        onDragOver={(e) => { e.preventDefault(); if (!dragging) setDragging(true) }}
        onDragLeave={(e) => { if (e.currentTarget === e.target) setDragging(false) }}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (e.dataTransfer.files?.length) s.uploadFiles(e.dataTransfer.files)
        }}
      >
        {dragging && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center border-2 border-dashed border-violet-500 bg-violet-500/10">
            <div className="flex items-center gap-2 text-sm font-medium text-violet-200">
              <Upload className="h-4 w-4" /> Drop files to upload{s.path ? ` to ${s.path}` : ''}
            </div>
          </div>
        )}
        {/* entries */}
        <div className="flex-1 overflow-auto">
          {s.error && <div className="m-3 rounded bg-red-950/50 px-3 py-2 text-xs text-red-300">{s.error}</div>}
          {s.path && (
            <button
              onClick={() => s.browse(parentPath)}
              className="flex w-full items-center gap-2 border-b border-zinc-900 px-4 py-2 text-left text-xs text-zinc-500 hover:bg-zinc-900"
            >
              <ArrowLeft className="h-3.5 w-3.5" /> ..
            </button>
          )}
          {s.entries.length === 0 && !s.loading && (
            <div className="mt-12 text-center text-sm text-zinc-600">Empty directory</div>
          )}
          {s.entries.map((e) => (
            <EntryRow key={e.path} entry={e} active={s.file?.path === e.path} />
          ))}
        </div>
      </div>

      {/* Rich file viewer modal — same renderers as the agent-card FileViewer */}
      {s.file && <FileViewerHost />}
    </div>
  )
}

// Hosts the modal viewer and wires prev/next navigation across the files in
// the current directory listing.
function FileViewerHost() {
  const s = useVFSStore()
  const files = s.entries.filter((e) => e.type === 'file')
  const idx = files.findIndex((e) => e.path === s.file?.path)
  const hasPrev = idx > 0
  const hasNext = idx >= 0 && idx < files.length - 1
  return (
    <VFSFileViewer
      onClose={s.closeFile}
      onPrev={hasPrev ? () => s.openFile(files[idx - 1]) : undefined}
      onNext={hasNext ? () => s.openFile(files[idx + 1]) : undefined}
      hasPrev={hasPrev}
      hasNext={hasNext}
    />
  )
}

function EntryRow({ entry, active }: { entry: VFSEntry; active: boolean }) {
  const s = useVFSStore()
  const Icon = entry.type === 'dir' ? Folder : FileText
  return (
    <div
      className={`group flex items-center gap-2 border-b border-zinc-900 px-4 py-2 text-sm ${
        active ? 'bg-violet-950/30' : 'hover:bg-zinc-900'
      }`}
    >
      <button onClick={() => s.openFile(entry)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
        <Icon className={`h-4 w-4 shrink-0 ${entry.type === 'dir' ? 'text-violet-400' : 'text-zinc-400'}`} />
        <span className="truncate text-zinc-200">{entry.name}</span>
        {entry.type === 'file' && <span className="shrink-0 text-[11px] text-zinc-600">{fmtBytes(entry.size)}</span>}
      </button>
      {/* metadata: who wrote it + when (always visible) */}
      <div className="flex shrink-0 items-center gap-2 text-[11px] text-zinc-600">
        {entry.author && (
          <span className="max-w-[14rem] truncate text-zinc-500" title={`Last written by ${entry.author}`}>
            ✎ {entry.author}
          </span>
        )}
        {entry.mtime ? <span title={fmtFull(entry.mtime)}>{fmtTime(entry.mtime)}</span> : null}
      </div>
      <div className="flex shrink-0 items-center gap-1 opacity-0 group-hover:opacity-100">
        {entry.type === 'file' && (
          <button onClick={() => s.download(entry)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Download">
            <Download className="h-3.5 w-3.5" />
          </button>
        )}
        <button
          onClick={() => {
            if (confirm(`Delete ${entry.type} "${entry.name}"?`)) s.deleteEntry(entry)
          }}
          className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-red-400"
          title="Delete"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}

