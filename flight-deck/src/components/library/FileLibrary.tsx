import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  AlertTriangle,
  ArrowLeft,
  Braces,
  Download,
  Eye,
  File,
  FileArchive,
  FolderOpen,
  FileCode,
  FileSpreadsheet,
  FileText,
  Film,
  Globe,
  Image,
  LibraryBig,
  Loader2,
  MessageSquare,
  MonitorPlay,
  Pencil,
  Presentation,
  RotateCw,
  Search,
  Settings2,
  Share2,
  SlidersHorizontal,
  Sparkles,
  X,
} from 'lucide-react'
import { useChatStore } from '../../stores/chatStore'
import { useNotificationStore } from '../../stores/notificationStore'
import type { LibraryFocus } from '../../stores/fileLibraryStore'
import { useProcessStore } from '../../stores/processStore'
import { FileViewer } from '../agents/FileViewer'
import { FileActionSheet } from '../agents/FileActionSheet'
import { PublicLinkDialog } from '../agents/PublicLinkDialog'
import { CreatorBadge } from '../agents/CreatorBadge'
import { useAgentEndpoint } from '../agents/AgentFilesPanel'
import type { AgentEndpoint, AgentFile } from '../../services/fileTransfer'
import {
  formatSize, getDeckUrl, getFileTypeGroup, isEditable, isViewable, listAgentFiles, processEndpoint,
} from '../../services/fileTransfer'
import { usePublicLinksEnabled } from '../../services/publicLinks'
import { isMemberCreated, type Creator } from '../../utils/sharedWorkspace'
import { fmtRelative, fmtWhen } from '../../services/schedulerApi'
import {
  fileLibraryApi,
  type LibraryItem,
  type LibraryPage,
  type LibrarySort,
  type LibraryState,
} from '../../services/fileLibraryApi'
import { FacetGroup, FullscreenButton, Highlight, ScanStatus, libraryIconBtn } from './libraryParts'
import { FilePreview } from './FilePreview'
import { isUntrustedFile } from '../../utils/untrustedContent'

// ── File library ────────────────────────────────────────────────────
//
// Every file your agents made, in one place: Flight Deck reads each new or
// changed file, writes a title and a short summary, sorts it into a category
// and tags it (a picture is looked at, a recording or video transcribed).
// Search finds a file by any of that or by its text, a picture's description
// or what's said in a recording; filters narrow it to a category, an agent
// or a kind of file. Opening the library
// looks for new files in the background; the list fills in as they're read.

const PAGE = 60

const CATEGORY_LABEL: Record<string, string> = {
  report: 'Report', research: 'Research', notes: 'Notes', plan: 'Plan', data: 'Data & lists',
  finance: 'Finance', legal: 'Legal', correspondence: 'Correspondence', presentation: 'Presentation',
  code: 'Code', media: 'Images & media', other: 'Other', pending: 'Not summarised yet',
}

const CATEGORY_TONE: Record<string, string> = {
  report: 'bg-violet-500/10 text-violet-700 dark:text-violet-300',
  research: 'bg-sky-500/10 text-sky-700 dark:text-sky-300',
  notes: 'bg-amber-500/10 text-amber-700 dark:text-amber-300',
  plan: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
  data: 'bg-cyan-500/10 text-cyan-700 dark:text-cyan-300',
  finance: 'bg-lime-500/10 text-lime-700 dark:text-lime-300',
  legal: 'bg-rose-500/10 text-rose-700 dark:text-rose-300',
  correspondence: 'bg-indigo-500/10 text-indigo-700 dark:text-indigo-300',
  presentation: 'bg-orange-500/10 text-orange-700 dark:text-orange-300',
  code: 'bg-teal-500/10 text-teal-700 dark:text-teal-300',
  media: 'bg-pink-500/10 text-pink-700 dark:text-pink-300',
  other: 'bg-zinc-800 text-zinc-400',
  pending: 'bg-zinc-800 text-zinc-500',
}

const CATEGORY_DOT: Record<string, string> = {
  report: 'bg-violet-500', research: 'bg-sky-500', notes: 'bg-amber-500', plan: 'bg-emerald-500',
  data: 'bg-cyan-500', finance: 'bg-lime-500', legal: 'bg-rose-500', correspondence: 'bg-indigo-500',
  presentation: 'bg-orange-500', code: 'bg-teal-500', media: 'bg-pink-500', other: 'bg-zinc-500', pending: 'bg-zinc-600',
}

const KIND_ICON: Record<string, typeof FileText> = {
  document: FileText, spreadsheet: FileSpreadsheet, presentation: Presentation, web: Globe,
  data: Braces, code: FileCode, image: Image, media: Film, archive: FileArchive, config: Settings2,
}
const KIND_TONE: Record<string, string> = {
  document: 'text-violet-500 dark:text-violet-400', spreadsheet: 'text-cyan-600 dark:text-cyan-400',
  presentation: 'text-orange-600 dark:text-orange-400', web: 'text-orange-500 dark:text-orange-300',
  data: 'text-sky-600 dark:text-sky-400', code: 'text-emerald-600 dark:text-emerald-400',
  image: 'text-blue-600 dark:text-blue-400', media: 'text-pink-600 dark:text-pink-400',
}

const SORTS: { id: LibrarySort; label: string }[] = [
  { id: 'recent', label: 'Newest first' },
  { id: 'name', label: 'Name' },
  { id: 'size', label: 'Largest first' },
  { id: 'agent', label: 'By agent' },
]

const iconBtn = libraryIconBtn

function categoryOf(item: LibraryItem): string {
  return item.category || 'pending'
}

/** The library's file as the agent's Files panel names it, so the shared
 *  viewer, public links and decks take it as they take one from there.
 *  `creator` is who wrote it; undefined when nobody could say. */
function toAgentFile(item: LibraryItem, creator?: Creator | null): AgentFile {
  return {
    logical: item.logical,
    physical: item.physical,
    filename: item.filename,
    extension: item.extension,
    exists: true,
    size: item.size,
    modified: item.modified,
    mime_type: item.mime_type,
    is_text: ['document', 'data', 'code', 'web', 'config'].includes(item.kind),
    source: 'workspace',
    created_by: creator,
  }
}

/** What the viewer opens. A page or markdown nobody could vouch for (its agent
 *  is off, or didn't list it) opens as someone else's would — no scripts, no
 *  remote images — since a member may have written it. FileViewer trusts a
 *  'shared' file only when it says it's the viewer's own. */
function toViewerFile(item: LibraryItem, creator: Creator | null | undefined): AgentFile {
  const file = toAgentFile(item, creator)
  const risky = ['html', 'markdown'].includes(getFileTypeGroup(file))
  return creator === undefined && risky ? { ...file, source: 'shared' } : file
}

// Whether this browser can hand a file to the phone's share sheet.
const canShareFiles = typeof navigator !== 'undefined' && typeof navigator.share === 'function'
  && typeof navigator.canShare === 'function'

/** The agent a library file belongs to, while it runs. Editing, presenting and
 *  public links go through the agent itself; a stopped one's files are still
 *  on disk (the library serves those), but there's nobody to ask. */
function useRunningAgent(slug: string): AgentEndpoint | null {
  const agent = useAgentEndpoint(`proc-${slug}`)
  const running = useProcessStore((s) => s.processes.some((p) => p.slug === slug && p.status === 'running'))
  return running ? agent : null
}

/** Who wrote each file. A member of a shared agent writes into its saved/
 *  folder, and that file must open as someone else's and carry their badge.
 *  The library doesn't record it; a running agent's own file list does
 *  (created_by). Each agent's list is fetched while the library is open, and
 *  a file is matched to it by its path on disk. It is fetched again when the
 *  agent couldn't say (it was still starting: only Flight Deck's own disk
 *  scan came back) — every few seconds for a minute, and when a chat to it
 *  gets through — when it runs again after being off, and when a file shows
 *  up that it didn't list then. */
const CREATOR_TRIES = 20
const CREATOR_RETRY_MS = 3000

type CreatorAsk =
  | { state: 'asking' }
  /** Its word, and every file it covered (listed, or in the library then). */
  | { state: 'done'; seen: Set<string> }
  /** Came back without its word `n` times, the last at `at`. */
  | { state: 'missed'; n: number; at: number }

function useFileCreators(items: LibraryItem[]): (item: LibraryItem) => Creator | null | undefined {
  const processes = useProcessStore((s) => s.processes)
  // Which agents' chats are through: one getting through is up.
  const connected = useChatStore((s) => {
    const ids = new Set<string>()
    for (const x of s.sessions.values()) if (x.connected) ids.add(x.containerId)
    return [...ids].sort().join(' ')
  })
  const [lists, setLists] = useState<Record<string, Map<string, Creator | null>>>({})
  const asked = useRef(new Map<string, CreatorAsk>())
  const [retry, setRetry] = useState(0)
  const timers = useRef<ReturnType<typeof setTimeout>[]>([])
  useEffect(() => () => { timers.current.forEach(clearTimeout) }, [])
  // A chat got through (or dropped): an agent that couldn't say is asked
  // again, its tries afresh.
  useEffect(() => {
    for (const [slug, a] of asked.current) if (a.state === 'missed') asked.current.delete(slug)
  }, [connected])
  useEffect(() => {
    const bySlug = new Map<string, string[]>()
    for (const i of items) {
      const paths = bySlug.get(i.agent_slug)
      if (paths) paths.push(i.physical)
      else bySlug.set(i.agent_slug, [i.physical])
    }
    for (const [slug, paths] of bySlug) {
      const was = asked.current.get(slug)
      if (was?.state === 'asking') continue
      const proc = processes.find((p) => p.slug === slug)
      const agent = proc?.status === 'running' ? processEndpoint(proc) : null
      // Off: asked afresh once it runs again.
      if (!agent) { asked.current.delete(slug); continue }
      if (was?.state === 'done' && paths.every((p) => was.seen.has(p))) continue
      const misses = was?.state === 'missed' ? was.n : 0
      if (was?.state === 'missed' && (misses >= CREATOR_TRIES || Date.now() - was.at < CREATOR_RETRY_MS)) continue
      asked.current.set(slug, { state: 'asking' })
      const miss = () => {
        asked.current.set(slug, { state: 'missed', n: misses + 1, at: Date.now() })
        if (misses + 1 < CREATOR_TRIES) timers.current.push(setTimeout(() => setRetry((n) => n + 1), CREATOR_RETRY_MS))
      }
      listAgentFiles(agent.host, agent.port, agent.auth)
        .then((files) => {
          // Only what the agent itself said: Flight Deck's own disk scan (all
          // that comes back while the agent can't be reached) names no author.
          if (!files.some((f) => f.created_by !== undefined)) { miss(); return }
          const known = new Map<string, Creator | null>()
          for (const f of files) if (f.created_by !== undefined) known.set(f.physical, f.created_by)
          asked.current.set(slug, { state: 'done', seen: new Set([...paths, ...files.map((f) => f.physical)]) })
          setLists((cur) => ({ ...cur, [slug]: known }))
        })
        .catch(miss)
    }
  }, [items, processes, retry, connected])
  return useCallback((item: LibraryItem) => lists[item.agent_slug]?.get(item.physical), [lists])
}

export function FileLibrary({ phone = false, initialAgent, initialFile, onClose, onOpenAgent, onOpenLive, liveReady }: {
  phone?: boolean
  /** Open filtered to this agent's files. */
  initialAgent?: string | null
  /** Open showing this file: searched for by name, selected once listed (a
   *  file made a moment ago is listed after the scan opening the library
   *  starts). */
  initialFile?: LibraryFocus | null
  onClose: () => void
  /** Chat with the agent a file belongs to. */
  onOpenAgent?: (agentSlug: string) => void
  /** Everything one agent keeps, read live from it (the kiosk's Files & data):
   *  also what the library doesn't list — a type it doesn't read, a file in
   *  a VFS or linked folder. */
  onOpenLive?: (agentSlug: string) => void
  /** Whether that agent can be opened live now — not while it is still
   *  starting up. Without it: while it runs. */
  liveReady?: (agentSlug: string) => boolean
}) {
  const notify = useNotificationStore((s) => s.add)
  const [state, setState] = useState<LibraryState | null>(null)
  const [query, setQuery] = useState(initialFile?.filename ?? '')
  const [debounced, setDebounced] = useState('')
  // The file asked for, until a load lists it.
  const wanted = useRef(initialFile ?? null)
  const [agent, setAgent] = useState(initialAgent || '')
  const [category, setCategory] = useState('')
  const [kind, setKind] = useState('')
  const [sort, setSort] = useState<LibrarySort>('recent')
  const [page, setPage] = useState<LibraryPage | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<LibraryItem | null>(null)
  const [viewing, setViewing] = useState<LibraryItem | null>(null)
  // Edit and Public link go through the file's agent, captured when tapped.
  const [editing, setEditing] = useState<{ item: LibraryItem; agent: AgentEndpoint } | null>(null)
  const [linking, setLinking] = useState<{ item: LibraryItem; agent: AgentEndpoint } | null>(null)
  const [sharing, setSharing] = useState<LibraryItem | null>(null)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const reqId = useRef(0)
  const shown = useRef(PAGE)

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), query ? 250 : 0)
    return () => clearTimeout(t)
  }, [query])

  // The kiosk can point the open library at another agent: follow it, and
  // let go of a file that belongs to a different one.
  useEffect(() => {
    setAgent(initialAgent || '')
    setSelected((cur) => (cur && initialAgent && cur.agent_slug !== initialAgent ? null : cur))
  }, [initialAgent])

  const load = useCallback(async (quiet = false) => {
    const id = ++reqId.current
    if (!quiet) { setLoading(true); shown.current = PAGE }
    try {
      const res = await fileLibraryApi.files({
        q: debounced, agent, category, kind, sort, limit: Math.max(PAGE, shown.current), offset: 0,
      })
      if (id !== reqId.current) return
      setPage(res)
      setError('')
      // The open file follows its fresh copy (a summary that just arrived).
      const want = wanted.current
      const hit = want && res.items.find((i) => i.agent_slug === want.agent && i.rel_path === want.relPath)
      if (hit) { wanted.current = null; setSelected(hit) }
      else setSelected((cur) => (cur ? res.items.find((i) => i.id === cur.id) ?? cur : cur))
    } catch (e) {
      if (id === reqId.current && !quiet) setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (id === reqId.current) setLoading(false)
    }
  }, [debounced, agent, category, kind, sort])

  useEffect(() => { load() }, [load])

  const loadMore = async () => {
    if (!page) return
    setLoadingMore(true)
    try {
      const res = await fileLibraryApi.files({
        q: debounced, agent, category, kind, sort, limit: PAGE, offset: page.items.length,
      })
      setPage((cur) => cur && {
        ...res,
        items: [...cur.items, ...res.items.filter((i) => !cur.items.some((c) => c.id === i.id))],
      })
      shown.current = page.items.length + res.items.length
    } catch (e) {
      notify('error', 'Could not load more files', e instanceof Error ? e.message : String(e))
    } finally {
      setLoadingMore(false)
    }
  }

  // Opening the library looks for new and changed files.
  useEffect(() => {
    fileLibraryApi.scan().then(setState).catch(() => fileLibraryApi.status().then(setState).catch(() => {}))
  }, [])

  // While a scan runs: its progress often, the list as summaries land.
  const running = !!state?.job.running
  const loadRef = useRef(load)
  loadRef.current = load
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => {
      fileLibraryApi.status().then(setState).catch(() => {})
      loadRef.current(true)
    }, 2500)
    return () => clearInterval(t)
  }, [running])
  // …and once more when it finishes.
  const wasRunning = useRef(false)
  useEffect(() => {
    if (wasRunning.current && !running) loadRef.current(true)
    wasRunning.current = running
  }, [running])

  const rescan = async () => {
    try { setState(await fileLibraryApi.scan(true)) } catch (e) {
      notify('error', 'Could not scan for files', e instanceof Error ? e.message : String(e))
    }
  }

  const resummarize = async (item: LibraryItem) => {
    const fresh = await fileLibraryApi.resummarize(item.id)
    setSelected((cur) => (cur?.id === fresh.id ? fresh : cur))
    setPage((cur) => cur && { ...cur, items: cur.items.map((i) => (i.id === fresh.id ? fresh : i)) })
    fileLibraryApi.status().then(setState).catch(() => {})
  }

  const openFile = (item: LibraryItem) => {
    if (getFileTypeGroup(toAgentFile(item)) === 'pdf') {
      window.open(fileLibraryApi.viewUrl(item.id), '_blank')
      return
    }
    setViewing(item)
  }
  const download = (item: LibraryItem) => {
    const a = document.createElement('a')
    a.href = fileLibraryApi.downloadUrl(item.id)
    a.download = item.filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }
  // An edit may have changed the file: look again, so its summary catches up.
  const closeEditor = () => {
    setEditing(null)
    fileLibraryApi.scan().then((s) => { setState(s); load(true) }).catch(() => {})
  }

  // Escape: a viewer, dialog or sheet closes itself; then the open file
  // (phone), then the library.
  const overlayRef = useRef(false)
  overlayRef.current = !!(viewing || editing || linking || sharing)
  const selectedRef = useRef(selected)
  selectedRef.current = selected
  const filtersRef = useRef(filtersOpen)
  filtersRef.current = filtersOpen
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || overlayRef.current) return
      if (filtersRef.current) { setFiltersOpen(false); return }
      if (phone && selectedRef.current) { setSelected(null); return }
      onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, phone])

  const items = page?.items ?? []
  const facets = page?.facets
  const terms = useMemo(() => debounced.toLowerCase().split(/\s+/).filter(Boolean), [debounced])
  const filtered = !!(debounced || agent || category || kind)
  // Only one agent's files (how an agent's menu opens it): an empty list
  // means that agent has no files yet, not that a search missed.
  const agentOnly = !!agent && !debounced && !category && !kind
  const clearFilters = () => { setQuery(''); setAgent(''); setCategory(''); setKind('') }
  // One agent's files, while it runs (and is up): the way to everything it keeps.
  const agentRunning = useProcessStore((s) => !!agent && s.processes.some((p) => p.slug === agent && p.status === 'running'))
  const openLive = onOpenLive && agent && agentRunning && (liveReady?.(agent) ?? true) ? () => onOpenLive(agent) : undefined
  const viewable = items.filter((i) => isViewable(toAgentFile(i)) && getFileTypeGroup(toAgentFile(i)) !== 'pdf')
  const viewIdx = viewing ? viewable.findIndex((i) => i.id === viewing.id) : -1
  const creatorOf = useFileCreators(items)

  const header = (
    <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/15 text-violet-600 dark:text-violet-400">
        <LibraryBig className="h-[18px] w-[18px]" />
      </span>
      <div className="min-w-0 flex-1">
        <h2 className="truncate text-base font-semibold text-zinc-100">File library</h2>
        {!phone && <p className="truncate text-xs text-zinc-500">Every file your agents made — summarised, sorted and searchable.</p>}
      </div>
      {!phone && <ScanStatus state={state} noun="file" />}
      <button onClick={rescan} disabled={running} className={iconBtn} title="Look for new files" aria-label="Look for new files">
        <RotateCw className={`h-4 w-4 ${running ? 'animate-spin' : ''}`} />
      </button>
      <FullscreenButton />
      <button onClick={onClose} className={iconBtn} title="Close" aria-label="Close">
        <X className="h-4 w-4" />
      </button>
    </div>
  )

  const toolbar = (
    <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2.5 wide:px-4">
      <div className="relative min-w-0 flex-1">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={phone ? 'Search files…' : 'Search summaries, tags, the text inside and what\'s said in recordings…'}
          className="w-full rounded-lg border border-zinc-800 bg-zinc-900 py-2 pl-8 pr-8 text-sm text-zinc-200 placeholder-zinc-500 focus:border-violet-500/60 focus:outline-none"
        />
        {query && (
          <button onClick={() => setQuery('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-zinc-500 hover:text-zinc-200" aria-label="Clear search">
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
      {phone ? (
        <button
          onClick={() => setFiltersOpen(true)}
          className={`relative flex h-9 shrink-0 items-center gap-1.5 rounded-lg border px-3 text-sm ${
            agent || category || kind ? 'border-violet-500/50 text-violet-700 dark:text-violet-300' : 'border-zinc-800 text-zinc-400'}`}
        >
          <SlidersHorizontal className="h-4 w-4" />Filters
        </button>
      ) : (
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as LibrarySort)}
          aria-label="Sort"
          className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-2 text-sm text-zinc-300 focus:border-violet-500/60 focus:outline-none"
        >
          {SORTS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
      )}
    </div>
  )

  const filters = (
    <Filters
      facets={facets}
      agent={agent} category={category} kind={kind}
      onAgent={setAgent} onCategory={setCategory} onKind={setKind}
    />
  )

  const list = (
    <ResultList
      items={items}
      total={page?.total ?? 0}
      state={state}
      loading={loading}
      error={error}
      filtered={filtered}
      agentOnly={agentOnly}
      terms={terms}
      selectedId={selected?.id ?? null}
      loadingMore={loadingMore}
      creatorOf={creatorOf}
      onSelect={setSelected}
      onMore={loadMore}
      onClear={clearFilters}
      onRetry={() => load()}
      onLive={openLive}
    />
  )

  const detail = selected && (
    <FileDetail
      key={selected.id}
      item={selected}
      creator={creatorOf(selected)}
      terms={terms}
      phone={phone}
      viewerOpen={!!(viewing || editing)}
      onClose={() => setSelected(null)}
      onOpen={() => openFile(selected)}
      onEdit={(a) => setEditing({ item: selected, agent: a })}
      onDownload={() => download(selected)}
      onShare={() => setSharing(selected)}
      onPublicLink={(a) => setLinking({ item: selected, agent: a })}
      onResummarize={() => resummarize(selected)}
      onTag={(t) => setQuery(t)}
      onAgent={() => setAgent(selected.agent_slug)}
      onOpenAgent={onOpenAgent && selected.agent_running ? () => onOpenAgent(selected.agent_slug) : undefined}
    />
  )

  // The library's own routes open any file, its agent running or not; an
  // edit is read and saved through the agent, like in its Files panel.
  const viewer = viewing ? (
    <FileViewer
      file={toViewerFile(viewing, creatorOf(viewing))}
      urls={{ view: fileLibraryApi.viewUrl(viewing.id), download: fileLibraryApi.downloadUrl(viewing.id) }}
      readOnly
      onClose={() => setViewing(null)}
      hasPrev={viewIdx > 0}
      hasNext={viewIdx >= 0 && viewIdx < viewable.length - 1}
      onPrev={() => { if (viewIdx > 0) setViewing(viewable[viewIdx - 1]) }}
      onNext={() => { if (viewIdx >= 0 && viewIdx < viewable.length - 1) setViewing(viewable[viewIdx + 1]) }}
    />
  ) : editing ? (
    <FileViewer
      file={toViewerFile(editing.item, creatorOf(editing.item))}
      host={editing.agent.host}
      port={editing.agent.port}
      auth={editing.agent.auth}
      startInEdit
      onClose={closeEditor}
    />
  ) : null

  const dialogs = (
    <>
      {linking && (
        <PublicLinkDialog
          agent={linking.agent}
          file={toAgentFile(linking.item, creatorOf(linking.item))}
          onClose={() => setLinking(null)}
        />
      )}
      {/* Phone: hand the file itself to the share sheet (Messages, Mail…). */}
      {sharing && (
        <FileActionSheet
          filename={sharing.filename}
          size={sharing.size}
          mime={sharing.mime_type}
          icon={<KindIcon kind={sharing.kind} />}
          detail={<CreatorBadge mode="owner" creator={creatorOf(sharing)} />}
          downloadUrl={fileLibraryApi.downloadUrl(sharing.id)}
          onDownload={() => download(sharing)}
          onClose={() => setSharing(null)}
        />
      )}
    </>
  )

  if (phone) {
    return (
      <div className="relative flex h-full flex-col bg-zinc-950">
        {header}
        <div className="border-b border-zinc-800 px-3 py-2"><ScanStatus state={state} noun="file" /></div>
        {toolbar}
        <div className="min-h-0 flex-1">{list}</div>
        {selected && <div className="absolute inset-0 z-10 flex flex-col bg-zinc-950">{detail}</div>}
        {filtersOpen && (
          <div className="absolute inset-0 z-20 flex flex-col bg-zinc-950">
            <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
              <h3 className="flex-1 text-base font-semibold text-zinc-100">Filters</h3>
              <button onClick={() => setFiltersOpen(false)} className={iconBtn} aria-label="Close filters"><X className="h-5 w-5" /></button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto">
              <div className="px-4 pt-3">
                <label className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Sort</label>
                <select
                  value={sort}
                  onChange={(e) => setSort(e.target.value as LibrarySort)}
                  className="h-11 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 text-base text-zinc-200"
                >
                  {SORTS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
                </select>
              </div>
              {filters}
            </div>
            <div className="flex gap-2 border-t border-zinc-800 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
              <button onClick={() => { setAgent(''); setCategory(''); setKind('') }} className="h-11 flex-1 rounded-xl border border-zinc-800 text-sm font-medium text-zinc-300">
                Clear
              </button>
              <button onClick={() => setFiltersOpen(false)} className="h-11 flex-1 rounded-xl bg-violet-600 text-sm font-medium text-white">
                Show {page?.total ?? 0} file{page?.total === 1 ? '' : 's'}
              </button>
            </div>
          </div>
        )}
        {viewer}
        {dialogs}
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col bg-zinc-950">
      {header}
      {toolbar}
      <div className="flex min-h-0 flex-1">
        <aside className="w-56 shrink-0 overflow-y-auto border-r border-zinc-800 bg-zinc-900/40">{filters}</aside>
        <section className="min-w-0 flex-1">{list}</section>
        {selected && <aside className="w-[380px] shrink-0 border-l border-zinc-800 bg-zinc-900/30">{detail}</aside>}
      </div>
      {viewer}
      {dialogs}
    </div>
  )
}

// ── Filters ──

function Filters({ facets, agent, category, kind, onAgent, onCategory, onKind }: {
  facets: LibraryPage['facets'] | undefined
  agent: string; category: string; kind: string
  onAgent: (v: string) => void; onCategory: (v: string) => void; onKind: (v: string) => void
}) {
  if (!facets) return <div className="flex justify-center py-8 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>
  return (
    <div className="flex flex-col gap-4 px-2 py-3">
      <FacetGroup
        title="Categories" allLabel="All files" facets={facets.categories} value={category} onChange={onCategory}
        labelOf={(id) => CATEGORY_LABEL[id] || id}
        dot={(id) => <span className={`h-2 w-2 shrink-0 rounded-full ${CATEGORY_DOT[id] || CATEGORY_DOT.other}`} />}
      />
      <FacetGroup title="Agents" allLabel="All agents" facets={facets.agents} value={agent} onChange={onAgent} />
      {facets.kinds.length > 1 && (
        <FacetGroup
          title="Type" allLabel="Any type" facets={facets.kinds} value={kind} onChange={onKind}
          dot={(id) => { const Icon = KIND_ICON[id] || File; return <Icon className={`h-3.5 w-3.5 shrink-0 ${KIND_TONE[id] || 'text-zinc-500'}`} /> }}
        />
      )}
    </div>
  )
}

// ── Results ──

/** One agent's files, read live from it: what the library may leave out. */
function LiveLink({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      title="Its files and tables, read live from the agent — also those the library doesn't list"
      className="flex shrink-0 items-center gap-1 text-[11px] font-medium text-violet-600 hover:underline dark:text-violet-400"
    >
      <FolderOpen className="h-3.5 w-3.5" />Everything it keeps
    </button>
  )
}

function ResultList({
  items, total, state, loading, error, filtered, agentOnly, terms, selectedId, loadingMore, creatorOf, onSelect, onMore, onClear, onRetry, onLive,
}: {
  items: LibraryItem[]; total: number; state: LibraryState | null; loading: boolean; error: string
  filtered: boolean; agentOnly: boolean; terms: string[]; selectedId: string | null; loadingMore: boolean
  creatorOf: (i: LibraryItem) => Creator | null | undefined
  onSelect: (i: LibraryItem) => void; onMore: () => void; onClear: () => void; onRetry: () => void
  /** The agent filtered to, live (it runs, in the kiosk). */
  onLive?: () => void
}) {
  if (loading && items.length === 0) {
    return <div className="flex h-full justify-center pt-16 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
  }
  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-zinc-500">
        <AlertTriangle className="h-5 w-5 text-red-500" />{error}
        <button onClick={onRetry} className="text-sm font-medium text-violet-600 hover:underline dark:text-violet-400">Try again</button>
      </div>
    )
  }
  if (items.length === 0) {
    const scanning = !!state?.job.running
    return (
      <div className="flex h-full flex-col items-center justify-center px-6 text-center">
        <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-violet-500/15 text-violet-600 dark:text-violet-400">
          {scanning ? <Loader2 className="h-7 w-7 animate-spin" /> : <LibraryBig className="h-7 w-7" />}
        </span>
        <h3 className="mt-4 text-base font-semibold text-zinc-100">
          {filtered && !agentOnly
            ? 'Nothing matches'
            : scanning ? 'Gathering your files…' : agentOnly ? 'No files yet' : state?.agents === 0 ? 'No agents yet' : 'No files yet'}
        </h3>
        <p className="mt-1.5 max-w-sm text-sm leading-relaxed text-zinc-400">
          {filtered && !agentOnly
            ? 'Try other words, or clear the filters.'
            : scanning
              ? agentOnly
                ? 'This agent\'s files show up here as soon as they\'re found and read.'
                : 'Your agents\' files show up here as soon as they\'re found and read.'
              : agentOnly
                ? 'This agent hasn\'t saved any files yet. When it saves reports, notes, spreadsheets or anything else, each one shows up here with a short summary.'
                : state?.agents === 0
                  ? 'Create an agent and the reports, notes and data it makes will collect here.'
                  : 'When your agents save reports, notes, spreadsheets or anything else, each one shows up here with a short summary.'}
        </p>
        {filtered && (
          <button onClick={onClear} className="mt-4 text-sm font-medium text-violet-600 hover:underline dark:text-violet-400">
            {agentOnly ? 'Show every agent\'s files' : 'Clear search and filters'}
          </button>
        )}
        {onLive && <div className="mt-3"><LiveLink onClick={onLive} /></div>}
      </div>
    )
  }
  return (
    <div className="h-full overflow-y-auto px-3 py-3 wide:px-4">
      <div className="mb-2 flex items-center justify-between gap-2 px-1 text-[11px] text-zinc-500">
        <span className="min-w-0 truncate">
          {total} file{total === 1 ? '' : 's'}{terms.length ? <> for “{terms.join(' ')}”</> : null}
        </span>
        {onLive && <LiveLink onClick={onLive} />}
      </div>
      <ul className="flex flex-col gap-1.5">
        {items.map((i) => (
          <li key={i.id}>
            <ResultRow item={i} creator={creatorOf(i)} terms={terms} active={i.id === selectedId} onClick={() => onSelect(i)} />
          </li>
        ))}
      </ul>
      {items.length < total && (
        <div className="flex justify-center py-4">
          <button
            onClick={onMore}
            disabled={loadingMore}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-800 px-4 py-2 text-sm text-zinc-300 hover:bg-zinc-800/60 disabled:opacity-60"
          >
            {loadingMore && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            Show more ({total - items.length} left)
          </button>
        </div>
      )}
    </div>
  )
}

function KindIcon({ kind, className = 'h-4 w-4' }: { kind: string; className?: string }) {
  const Icon = KIND_ICON[kind] || File
  return <Icon className={`${className} ${KIND_TONE[kind] || 'text-zinc-500'}`} />
}

function CategoryChip({ id }: { id: string }) {
  return (
    <span className={`shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-medium ${CATEGORY_TONE[id] || CATEGORY_TONE.other}`}>
      {CATEGORY_LABEL[id] || id}
    </span>
  )
}

function ResultRow({ item, creator, terms, active, onClick }: {
  item: LibraryItem; creator: Creator | null | undefined; terms: string[]; active: boolean; onClick: () => void
}) {
  const pending = item.status === 'pending' && !item.summary
  return (
    <button
      onClick={onClick}
      className={`flex w-full gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors ${
        active ? 'border-violet-500/40 bg-violet-500/5' : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-900/80'}`}
    >
      <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800/80">
        <KindIcon kind={item.kind} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2">
          <span className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-100">
            <Highlight text={item.title || item.filename} terms={terms} />
          </span>
          <CreatorBadge mode="owner" creator={creator} />
          <CategoryChip id={categoryOf(item)} />
        </span>
        <span className="mt-0.5 block truncate text-[11px] text-zinc-500">
          {item.title && item.title !== item.filename ? <><Highlight text={item.filename} terms={terms} /> · </> : null}
          {item.agent_name} · {fmtRelative(item.modified)} · {formatSize(item.size)}
        </span>
        {pending ? (
          <span className="mt-1 flex items-center gap-1.5 text-xs text-zinc-500">
            <Loader2 className="h-3 w-3 animate-spin" />Waiting for its summary…
          </span>
        ) : item.status === 'error' && !item.summary ? (
          <span className="mt-1 flex items-center gap-1.5 text-xs text-red-600 dark:text-red-400">
            <AlertTriangle className="h-3 w-3" />Couldn't summarise it
          </span>
        ) : item.summary ? (
          <span className="mt-1 line-clamp-2 block text-xs leading-relaxed text-zinc-400">
            <Highlight text={item.summary} terms={terms} />
          </span>
        ) : null}
        {item.snippet && (
          <span className="mt-1 line-clamp-2 block border-l-2 border-amber-500/40 pl-2 text-[11px] italic leading-relaxed text-zinc-500">
            <Highlight text={item.snippet} terms={terms} />
          </span>
        )}
        {item.tags.length > 0 && (
          <span className="mt-1.5 flex flex-wrap gap-1">
            {item.tags.map((t) => (
              <span key={t} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
                <Highlight text={t} terms={terms} />
              </span>
            ))}
          </span>
        )}
      </span>
    </button>
  )
}

// ── One file ──

function FileDetail({
  item, creator, terms, phone, viewerOpen, onClose, onOpen, onEdit, onDownload, onShare, onPublicLink, onResummarize, onTag, onAgent, onOpenAgent,
}: {
  item: LibraryItem; creator: Creator | null | undefined; terms: string[]; phone: boolean
  /** A file is open in the viewer over the library (its preview pauses). */
  viewerOpen: boolean
  onClose: () => void; onOpen: () => void; onEdit: (agent: AgentEndpoint) => void
  onDownload: () => void; onShare: () => void; onPublicLink: (agent: AgentEndpoint) => void
  onResummarize: () => Promise<void>; onTag: (t: string) => void; onAgent: () => void
  onOpenAgent?: () => void
}) {
  const notify = useNotificationStore((s) => s.add)
  const [busy, setBusy] = useState(false)
  const agent = useRunningAgent(item.agent_slug)
  const publicLinks = usePublicLinksEnabled()
  const file = toAgentFile(item, creator)
  const canView = isViewable(file)
  // What the Files panel offers on a phone: no editing, no decks.
  const canEdit = !phone && isEditable(file)
  // A member's HTML never runs as a deck (Flight Deck's /deck/view refuses it too).
  const canDeck = !phone && getFileTypeGroup(file) === 'html' && !isMemberCreated(creator)
  const offTitle = `Start ${item.agent_name} first — this goes through the agent`
  // Said under the buttons too: a phone has no hover for their titles.
  const viaAgent = [canEdit && 'edit', canDeck && 'present', publicLinks && 'make a public link']
    .filter((a): a is string => !!a)
  const offNote = viaAgent.length > 1
    ? `${viaAgent.slice(0, -1).join(', ')} or ${viaAgent[viaAgent.length - 1]}`
    : viaAgent[0]
  const present = () => {
    if (agent) window.open(getDeckUrl(agent.host, agent.port, item.physical, agent.auth), '_blank')
  }
  const redo = async () => {
    setBusy(true)
    try { await onResummarize() } catch (e) {
      notify('error', 'Could not summarise the file', e instanceof Error ? e.message : String(e))
    } finally { setBusy(false) }
  }

  const meta: [string, ReactNode][] = [
    ['Agent', (
      <button onClick={onAgent} className="truncate text-left text-violet-600 hover:underline dark:text-violet-400" title="Show only this agent's files">
        {item.agent_name}
      </button>
    )],
    ['Location', <span className="break-all font-mono text-[11px]">{item.rel_path}</span>],
    ['Size', formatSize(item.size)],
    ['Changed', fmtWhen(item.modified)],
  ]
  if (item.summarized_at && item.status !== 'pending') {
    meta.push(['Summarised', `${fmtWhen(item.summarized_at)}${item.model ? ` · ${item.model.split('/').pop()}` : ''}`])
  }

  const btn = 'flex items-center justify-center gap-1.5 rounded-lg border border-zinc-800 px-3 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:opacity-50'

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2.5">
        {phone ? (
          <button onClick={onClose} className="flex flex-1 items-center gap-1.5 text-sm text-violet-600 dark:text-violet-400">
            <ArrowLeft className="h-4 w-4" />All files
          </button>
        ) : (
          <>
            <span className="flex-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">File</span>
            <button onClick={onClose} className={iconBtn} title="Close" aria-label="Close file"><X className="h-4 w-4" /></button>
          </>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <div className="flex items-start gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-zinc-800/80">
            <KindIcon kind={item.kind} className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="break-words text-base font-semibold leading-snug text-zinc-100">
              <Highlight text={item.title || item.filename} terms={terms} />
            </h3>
            <div className="mt-0.5 truncate text-xs text-zinc-500" title={item.filename}>{item.filename}</div>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <CategoryChip id={categoryOf(item)} />
          <span className="rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">{item.extension.replace('.', '').toUpperCase() || 'FILE'}</span>
          <CreatorBadge mode="owner" creator={creator} />
        </div>

        <div className="mt-4">
          {item.status === 'pending' && !item.summary ? (
            <p className="flex items-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" />Waiting for its summary…</p>
          ) : item.status === 'error' ? (
            <div className="rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2.5 text-sm text-red-700 dark:text-red-300">
              <div className="flex items-center gap-1.5 font-medium"><AlertTriangle className="h-4 w-4" />Couldn't summarise it</div>
              {item.error && <div className="mt-1 break-words text-xs opacity-80">{item.error}</div>}
            </div>
          ) : null}
          {item.summary && (
            <p className={`text-sm leading-relaxed ${item.status === 'skipped' ? 'text-zinc-500' : 'text-zinc-300'} ${item.status === 'error' ? 'mt-3' : ''}`}>
              <Highlight text={item.summary} terms={terms} />
            </p>
          )}
          {item.status === 'pending' && item.summary && (
            <p className="mt-2 flex items-center gap-1.5 text-[11px] text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" />Changed since — a new summary is on its way.</p>
          )}
        </div>

        {item.snippet && (
          <div className="mt-3 border-l-2 border-amber-500/40 pl-3 text-xs italic leading-relaxed text-zinc-500">
            <Highlight text={item.snippet} terms={terms} />
          </div>
        )}

        {item.tags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-1.5">
            {item.tags.map((t) => (
              <button
                key={t}
                onClick={() => onTag(t)}
                title={`Search for “${t}”`}
                className="rounded-md bg-zinc-800 px-2 py-1 text-[11px] text-zinc-300 transition-colors hover:bg-violet-500/15 hover:text-violet-700 dark:hover:text-violet-300"
              >
                #{t}
              </button>
            ))}
          </div>
        )}

        <dl className="mt-5 grid grid-cols-[84px_1fr] gap-x-3 gap-y-2 text-xs">
          {meta.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-zinc-500">{k}</dt>
              <dd className="min-w-0 text-zinc-300">{v}</dd>
            </div>
          ))}
        </dl>

        {/* Keyed by the file's version: a new file (or a changed one) starts
            its preview afresh. */}
        <FilePreview
          key={`${item.id}:${item.modified}`}
          group={getFileTypeGroup(file)}
          extension={item.extension}
          size={item.size}
          url={fileLibraryApi.viewUrl(item.id, item.modified)}
          freshUrl={() => fileLibraryApi.viewUrl(item.id, item.modified)}
          thumbUrl={fileLibraryApi.thumbUrl(item.id, item.modified)}
          cacheKey={`${item.id}:${item.modified}`}
          untrusted={isUntrustedFile(toViewerFile(item, creator))}
          phone={phone}
          viewerOpen={viewerOpen}
          canOpen={canView}
          onOpen={onOpen}
        />
      </div>

      <div className="grid grid-cols-2 gap-2 border-t border-zinc-800 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:pb-3">
        {canView && (
          <button onClick={onOpen} className="flex items-center justify-center gap-1.5 rounded-lg bg-violet-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500">
            <Eye className="h-4 w-4" />Open
          </button>
        )}
        <button onClick={onDownload} className={btn}><Download className="h-4 w-4" />Download</button>
        {canEdit && (
          <button onClick={() => agent && onEdit(agent)} disabled={!agent} className={btn} title={agent ? 'Edit the file' : offTitle}>
            <Pencil className="h-4 w-4" />Edit
          </button>
        )}
        {canDeck && (
          <button onClick={present} disabled={!agent} className={btn} title={agent ? 'Present it as a slide deck in a new tab' : offTitle}>
            <MonitorPlay className="h-4 w-4" />Present as deck
          </button>
        )}
        {phone && canShareFiles && (
          <button onClick={onShare} className={btn}><Share2 className="h-4 w-4" />Share</button>
        )}
        {publicLinks && (
          <button onClick={() => agent && onPublicLink(agent)} disabled={!agent} className={btn} title={agent ? 'Share — public link' : offTitle}>
            <Globe className="h-4 w-4" />Public link
          </button>
        )}
        {onOpenAgent && (
          <button onClick={onOpenAgent} className={btn} title={`Chat with ${item.agent_name}`}>
            <MessageSquare className="h-4 w-4" />Agent chat
          </button>
        )}
        <button onClick={redo} disabled={busy || item.status === 'pending'} className={btn} title="Read the file again and write a fresh summary">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}Summarise again
        </button>
        {offNote && !agent && (
          <p className="col-span-2 text-center text-[11px] text-zinc-500">
            Start {item.agent_name} to {offNote} — that goes through the agent.
          </p>
        )}
      </div>
    </div>
  )
}
