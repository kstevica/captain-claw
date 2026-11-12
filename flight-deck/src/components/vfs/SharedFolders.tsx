import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  AlertTriangle,
  ArrowLeft,
  Braces,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Clock,
  Cloud,
  CloudUpload,
  Cloudy,
  Copy,
  Database,
  Download,
  Eye,
  EyeOff,
  File,
  FileArchive,
  FileCode,
  FileSpreadsheet,
  FileText,
  Film,
  Folder,
  FolderInput,
  FolderOpen,
  FolderPlus,
  Folders,
  Globe,
  HardDrive,
  Image,
  Info,
  Link2,
  Loader2,
  Lock,
  Music,
  Pencil,
  Presentation,
  RotateCw,
  Search,
  Share2,
  ShieldCheck,
  Sparkles,
  TextCursorInput,
  Trash2,
  Upload,
  Users,
  Workflow,
  X,
} from 'lucide-react'
import { useNotificationStore } from '../../stores/notificationStore'
import { useVFSStore, extOf, type VFSEntry, type VFSProject } from '../../stores/vfsStore'
import { vfsApi, type FolderHit, type FolderRef, type FolderSearch } from '../../services/vfsApi'
import { formatSize, getFileTypeGroup, isEditable, isViewable, type AgentFile } from '../../services/fileTransfer'
import { fmtRelative, fmtWhen } from '../../services/schedulerApi'
import { leaveShare } from '../../services/shares'
import { FileViewer } from '../agents/FileViewer'
import { FileActionSheet } from '../agents/FileActionSheet'
import { DatastoreBrowser } from '../agents/DatastoreBrowser'
import { ShareModal } from '../common/ShareModal'
import { AgentMenu, type AgentMenuItem } from '../layout/KioskAgentDialogs'
import { AnonymizeHereDialog, type AnonTarget } from '../anonymize/AnonymizeHereDialog'
import { anonymizable } from '../anonymize/anonFormat'
import { FullscreenButton, Highlight, libraryIconBtn } from '../library/libraryParts'
import { FilePreview } from '../library/FilePreview'
import { isUntrustedFile } from '../../utils/untrustedContent'
import { DriveConnect } from './DriveConnect'
import { OneDriveConnect } from './OneDriveConnect'

// ── Shared folders ──────────────────────────────────────────────────
//
// The kiosk's "Folders": the folders every agent of the user reads and
// writes as vfs:<folder>/<path> — the ones they made, the ones an agent run
// made for its team, folders linked from this computer, Google Drive and
// OneDrive folders// and the ones other people shared. Open one and walk it like a file
// browser: preview a file beside the list, open, edit, rename or download
// it, drop files in. Search finds a file by name in every folder at once,
// and Recent files shows what the agents wrote last.

type Group = 'mine' | 'runs' | 'shared' | 'link' | 'gdrive' | 'onedrive'
type Nav = '' | 'recent' | Group
type Sort = 'recent' | 'name' | 'size'

const GROUPS: { id: Group; label: string; one: string; icon: typeof Folder; tile: string }[] = [
  { id: 'mine', label: 'My folders', one: 'Folder', icon: Folder, tile: 'bg-sky-500/15 text-sky-600 dark:text-sky-400' },
  { id: 'runs', label: 'Agent runs', one: 'Agent run', icon: Workflow, tile: 'bg-violet-500/15 text-violet-600 dark:text-violet-400' },
  { id: 'shared', label: 'Shared with me', one: 'Shared with you', icon: Users, tile: 'bg-amber-500/15 text-amber-600 dark:text-amber-400' },
  { id: 'link', label: 'Linked folders', one: 'Linked folder', icon: Link2, tile: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400' },
  { id: 'gdrive', label: 'Google Drive', one: 'Google Drive', icon: Cloud, tile: 'bg-blue-500/15 text-blue-600 dark:text-blue-400' },
  { id: 'onedrive', label: 'OneDrive', one: 'OneDrive', icon: Cloudy, tile: 'bg-indigo-500/15 text-indigo-600 dark:text-indigo-400' },
]
const GROUP = Object.fromEntries(GROUPS.map((g) => [g.id, g])) as Record<Group, (typeof GROUPS)[number]>
const RUN_LABEL: Record<string, string> = { vatra: 'Vatra', basna: 'Basna', council: 'Council' }

const SORTS: { id: Sort; label: string }[] = [
  { id: 'recent', label: 'Newest first' },
  { id: 'name', label: 'Name' },
  { id: 'size', label: 'Largest first' },
]
const SORT_KEY = 'fd:folders-sort'

const iconBtn = libraryIconBtn
const btn = 'flex items-center justify-center gap-1.5 rounded-lg border border-zinc-800 px-3 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:opacity-50'
const primaryBtn = 'flex items-center justify-center gap-1.5 rounded-lg bg-violet-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50'

// Whether this browser can hand a file to the phone's share sheet.
const canShareFiles = typeof navigator !== 'undefined' && typeof navigator.share === 'function'
  && typeof navigator.canShare === 'function'

/** A file in a list: an entry of the open folder, a search hit or a recent file. */
interface FileItem { ref: FolderRef; path: string; name: string; size: number; mtime: number; author?: string }

type Dialog =
  | { kind: 'name'; title: string; initial: string; action: string; hint?: (name: string) => string; onSubmit: (name: string) => Promise<void> }
  | { kind: 'confirm'; title: string; body: ReactNode; action: string; check?: string; onConfirm: (checked: boolean) => Promise<void> }
  | { kind: 'share'; folder: VFSProject }
  | { kind: 'tables'; folder: VFSProject }
  | { kind: 'drive' }
  | { kind: 'onedrive' }
  | { kind: 'link' }
  | { kind: 'anonymize'; target: AnonTarget }

const refKey = (r: FolderRef) => `${r.owner}\0${r.project}`
const itemKey = (i: FileItem) => `${refKey(i.ref)}\0${i.path}`
const refOf = (p: VFSProject): FolderRef => ({ project: p.name, owner: p.shared ? p.owner_id || '' : '' })
const sameRef = (a: FolderRef, b: FolderRef) => a.project === b.project && a.owner === b.owner
const msg = (e: unknown) => (e instanceof Error ? e.message : String(e))
const parentOf = (path: string) => path.split('/').slice(0, -1).join('/')
// A file written a moment ago (or stamped a hair ahead of this clock) is in the past.
const ago = (t: number) => fmtRelative(Math.min(t, Date.now() / 1000 - 1))
const joinPath = (dir: string, name: string) => (dir ? `${dir}/${name}` : name)

function groupOf(p: VFSProject | undefined): Group {
  if (!p) return 'mine'
  if (p.shared) return 'shared'
  if (p.kind === 'link') return 'link'
  if (p.kind === 'gdrive') return 'gdrive'
  if (p.kind === 'onedrive') return 'onedrive'
  if (p.kind && RUN_LABEL[p.kind]) return 'runs'
  return 'mine'
}

const titleOf = (p: VFSProject) => p.title || p.name

/** A folder mounted from Google Drive or OneDrive: read-only, its files on
 *  disk stand-ins until read (or Markdown copies). */
const isMount = (p: VFSProject) => p.kind === 'gdrive' || p.kind === 'onedrive'
const cloudOf = (p: VFSProject) => (p.kind === 'onedrive' ? 'OneDrive' : 'Google Drive')

// The explanation at the top of the folder lists, until the viewer hides it.
const ABOUT_KEY = 'fd:folders-about-hidden'

/** Files can be added, changed and removed: the user's own folder, a linked
 *  one that isn't read-only, one shared with them to edit. Drive is read-only. */
function writable(p: VFSProject | undefined): boolean {
  if (!p) return false
  if (p.shared) return p.permission === 'edit'
  if (p.kind === 'gdrive' || p.kind === 'onedrive') return false
  if (p.kind === 'link') return p.mode !== 'ro' && !p.missing
  return true
}

/** What a folder's name becomes on disk (vfs.safe_name): its agents' address. */
function folderName(name: string): string {
  return name.trim().replace(/[^\p{L}\p{N}._-]/gu, '-').replace(/^[-._]+|[-._]+$/g, '')
}

/** Where a folder comes from, in a line under its name. */
function originOf(p: VFSProject): string {
  if (p.shared) return `From ${p.owner_name || p.owner_email || 'someone'} · ${p.permission === 'edit' ? 'you can edit' : 'view only'}`
  if (p.kind === 'link') return p.missing ? `Missing — ${p.link_path}` : p.link_path || ''
  if (p.kind === 'gdrive') return p.drive?.source_path ? `Drive · ${p.drive.source_path.replace(/\//g, ' / ')}` : 'Google Drive'
  if (p.kind === 'onedrive') return p.drive?.source_path ? `OneDrive · ${p.drive.source_path.replace(/\//g, ' / ')}` : 'OneDrive'
  if (p.title) return p.name
  return ''
}

function readSort(): Sort {
  try {
    const v = localStorage.getItem(SORT_KEY)
    return v === 'name' || v === 'size' ? v : 'recent'
  } catch { return 'recent' }
}

function sortBy<T>(list: T[], sort: Sort, name: (x: T) => string, size: (x: T) => number, time: (x: T) => number): T[] {
  return [...list].sort((a, b) => sort === 'name' ? name(a).localeCompare(name(b), undefined, { numeric: true, sensitivity: 'base' })
    : sort === 'size' ? size(b) - size(a) : time(b) - time(a))
}

const asAgentFile = (name: string) => ({ extension: extOf(name), filename: name }) as AgentFile

function kindOf(name: string): { icon: typeof File; tone: string } {
  const group = getFileTypeGroup(asAgentFile(name))
  const ext = extOf(name)
  switch (group) {
    case 'image': return { icon: Image, tone: 'text-blue-600 dark:text-blue-400' }
    case 'video': return { icon: Film, tone: 'text-pink-600 dark:text-pink-400' }
    case 'audio': return { icon: Music, tone: 'text-pink-600 dark:text-pink-400' }
    case 'pdf': return { icon: FileText, tone: 'text-red-600 dark:text-red-400' }
    case 'html': return { icon: Globe, tone: 'text-orange-600 dark:text-orange-400' }
    case 'markdown': return { icon: FileText, tone: 'text-violet-600 dark:text-violet-400' }
    case 'data': return ext === '.csv' || ext === '.tsv'
      ? { icon: FileSpreadsheet, tone: 'text-cyan-600 dark:text-cyan-400' }
      : { icon: Braces, tone: 'text-sky-600 dark:text-sky-400' }
    case 'code': return { icon: FileCode, tone: 'text-emerald-600 dark:text-emerald-400' }
    case 'archive': return { icon: FileArchive, tone: 'text-amber-600 dark:text-amber-400' }
    case 'document':
      if (['.xls', '.xlsx', '.ods'].includes(ext)) return { icon: FileSpreadsheet, tone: 'text-cyan-600 dark:text-cyan-400' }
      if (['.ppt', '.pptx'].includes(ext)) return { icon: Presentation, tone: 'text-orange-600 dark:text-orange-400' }
      return { icon: FileText, tone: 'text-violet-600 dark:text-violet-400' }
    case 'text': return { icon: FileText, tone: 'text-zinc-400' }
    default: return { icon: File, tone: 'text-zinc-500' }
  }
}

function KindIcon({ name, className = 'h-4 w-4' }: { name: string; className?: string }) {
  const { icon: Icon, tone } = kindOf(name)
  return <Icon className={`${className} ${tone}`} />
}

function FolderTile({ folder, size = 'md' }: { folder: VFSProject | undefined; size?: 'sm' | 'md' | 'lg' }) {
  const g = GROUP[groupOf(folder)]
  const Icon = g.icon
  const box = size === 'lg' ? 'h-11 w-11 rounded-xl' : size === 'md' ? 'h-9 w-9 rounded-xl' : 'h-7 w-7 rounded-lg'
  const icon = size === 'lg' ? 'h-5 w-5' : size === 'md' ? 'h-[18px] w-[18px]' : 'h-3.5 w-3.5'
  return <span className={`flex shrink-0 items-center justify-center ${box} ${g.tile}`}><Icon className={icon} /></span>
}

/** A file for the shared viewer. Someone else's folder (shared, or Google
 *  Drive) opens as someone else's file would: no scripts, nothing remote. */
function toAgentFile(item: FileItem, folder: VFSProject | undefined): AgentFile {
  const others = !!folder && (folder.shared || isMount(folder))
  const group = getFileTypeGroup(asAgentFile(item.name))
  return {
    logical: `vfs:${item.ref.project}/${item.path}`,
    physical: itemKey(item),
    filename: item.name,
    extension: extOf(item.name),
    exists: true,
    size: item.size,
    modified: item.mtime,
    mime_type: '',
    is_text: ['markdown', 'code', 'data', 'text', 'html'].includes(group),
    source: others ? 'shared' : 'workspace',
  }
}

export function SharedFolders({ phone = false, canLink = false, onClose, onConnect }: {
  phone?: boolean
  /** May link a folder from this computer (an admin, or a deck without sign-in). */
  canLink?: boolean
  onClose: () => void
  /** Connections on that account's tab — Drive and OneDrive folders want it. */
  onConnect?: (service: 'google' | 'microsoft') => void
}) {
  const notify = useNotificationStore((s) => s.add)
  const [projects, setProjects] = useState<VFSProject[] | null>(null)
  const [projectsError, setProjectsError] = useState('')
  const [nav, setNav] = useState<Nav>('')
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [sort, setSortState] = useState<Sort>(readSort)
  const [showRuns, setShowRuns] = useState(false)
  // The open folder, and where in it.
  const [open, setOpen] = useState<{ ref: FolderRef; path: string } | null>(null)
  const [entries, setEntries] = useState<VFSEntry[] | null>(null)
  const [entriesError, setEntriesError] = useState('')
  const [showHidden, setShowHidden] = useState(false)
  // Search results, or (Recent files) the newest files.
  const [hits, setHits] = useState<FolderSearch | null>(null)
  const [hitsLoading, setHitsLoading] = useState(false)
  const [hitsError, setHitsError] = useState('')
  const [selected, setSelected] = useState<FileItem | null>(null)
  const [viewing, setViewing] = useState<{ item: FileItem; edit: boolean } | null>(null)
  const [sharingFile, setSharingFile] = useState<FileItem | null>(null)
  const [dialog, setDialog] = useState<Dialog | null>(null)
  const [uploading, setUploading] = useState(0)
  const [dragging, setDragging] = useState(false)
  const uploadInput = useRef<HTMLInputElement | null>(null)
  const [aboutHidden, setAboutHidden] = useState(() => {
    try { return localStorage.getItem(ABOUT_KEY) === '1' } catch { return false }
  })
  const showAbout = (show: boolean) => {
    setAboutHidden(!show)
    try { if (show) localStorage.removeItem(ABOUT_KEY); else localStorage.setItem(ABOUT_KEY, '1') } catch { /* private window */ }
  }

  const setSort = (s: Sort) => {
    setSortState(s)
    try { localStorage.setItem(SORT_KEY, s) } catch { /* private window */ }
  }

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), query ? 250 : 0)
    return () => clearTimeout(t)
  }, [query])

  // ── Loading ──

  const loadProjects = useCallback(async (quiet = false) => {
    try {
      setProjects(await vfsApi.projects())
      setProjectsError('')
    } catch (e) {
      if (!quiet) setProjectsError(msg(e))
    }
  }, [])
  useEffect(() => { loadProjects() }, [loadProjects])

  const entriesReq = useRef(0)
  const loadEntries = useCallback(async (quiet = false) => {
    if (!open) return
    const id = ++entriesReq.current
    if (!quiet) { setEntries(null); setEntriesError('') }
    try {
      const list = await vfsApi.list(open.ref, open.path)
      if (id !== entriesReq.current) return
      setEntries(list)
      setEntriesError('')
    } catch (e) {
      if (id === entriesReq.current && !quiet) setEntriesError(msg(e))
    }
  }, [open])
  useEffect(() => { loadEntries() }, [loadEntries])

  // In a folder the search keeps to it; anywhere else it looks in every
  // folder. Recent files is the search with no words.
  const scope = open?.ref
  const scopeKey = scope ? refKey(scope) : ''
  const wantHits = !!debounced || (nav === 'recent' && !open)
  const hitsReq = useRef(0)
  const loadHits = useCallback(async (quiet = false) => {
    const id = ++hitsReq.current
    if (!wantHits) { setHits(null); setHitsLoading(false); return }
    if (!quiet) { setHitsLoading(true); setHitsError('') }
    try {
      const r = await vfsApi.search(debounced, scope)
      if (id !== hitsReq.current) return
      setHits(r)
      setHitsError('')
    } catch (e) {
      if (id === hitsReq.current && !quiet) setHitsError(msg(e))
    } finally {
      if (id === hitsReq.current) setHitsLoading(false)
    }
    // `scope` is followed by its key: a new path in the same folder isn't a new search.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wantHits, debounced, scopeKey])
  useEffect(() => { loadHits() }, [loadHits])

  // Agents keep writing while the dialog is open: what it shows follows,
  // quietly (a search's results stay as they came). The open folder's
  // listing is cheap and goes first; Recent files and the folder list walk
  // every folder, so they follow more slowly.
  const tick = useRef<(n: number) => void>(() => {})
  tick.current = (n) => {
    if (open) loadEntries(true)
    if (wantHits && !debounced && n % 2 === 0) loadHits(true)
    if (n % 3 === 0) loadProjects(true)
  }
  useEffect(() => {
    let n = 0
    const t = setInterval(() => { n++; if (document.visibilityState === 'visible') tick.current(n) }, 10000)
    return () => clearInterval(t)
  }, [])

  const refreshAll = () => { loadProjects(); if (open) loadEntries(); if (wantHits) loadHits() }

  // ── Derived ──

  const byKey = useMemo(() => new Map((projects ?? []).map((p) => [refKey(refOf(p)), p])), [projects])
  const folderOf = useCallback((ref: FolderRef) => byKey.get(refKey(ref)), [byKey])
  const openFolder = open ? folderOf(open.ref) : undefined
  const canWrite = writable(openFolder)
  const terms = useMemo(() => debounced.toLowerCase().split(/\s+/).filter(Boolean), [debounced])

  const counts = useMemo(() => {
    const c: Record<Group, number> = { mine: 0, runs: 0, shared: 0, link: 0, gdrive: 0, onedrive: 0 }
    for (const p of projects ?? []) c[groupOf(p)]++
    return c
  }, [projects])
  const totals = useMemo(() => (projects ?? []).reduce((t, p) => ({ files: t.files + p.files, bytes: t.bytes + p.bytes }), { files: 0, bytes: 0 }), [projects])

  const folderMatches = useCallback((p: VFSProject) => !terms.length || terms.every((t) =>
    [p.name, p.title, p.link_path, p.drive?.source_path, p.owner_name, p.owner_email].some((f) => (f || '').toLowerCase().includes(t))), [terms])

  const groupFolders = useMemo(() => {
    const inGroup = (projects ?? []).filter((p) => (nav === '' || nav === 'recent' || groupOf(p) === nav) && folderMatches(p))
    return sortBy(inGroup, sort, titleOf, (p) => p.bytes, (p) => p.mtime || 0)
  }, [projects, nav, folderMatches, sort])
  // Folders an agent run made and never named fold away under All folders.
  const rawRun = (p: VFSProject) => groupOf(p) === 'runs' && !p.title
  const folding = nav === '' && !terms.length
  const rawRuns = folding ? groupFolders.filter(rawRun) : []
  const shownFolders = folding && !showRuns ? groupFolders.filter((p) => !rawRun(p)) : groupFolders

  // The files the search found — in one group of folders when one is picked.
  const hitItems = useMemo<FileItem[]>(() => {
    const list = (hits?.files ?? []).filter((h) => open || nav === '' || nav === 'recent'
      || groupOf(byKey.get(refKey(h))) === nav)
    return list.map((h: FolderHit) => ({ ref: { project: h.project, owner: h.owner }, path: h.path, name: h.name, size: h.size, mtime: h.mtime, author: h.author }))
  }, [hits, open, nav, byKey])

  const visibleEntries = useMemo(() => {
    const list = (entries ?? []).filter((e) => showHidden || !e.name.startsWith('.'))
    const dirs = sortBy(list.filter((e) => e.type === 'dir'), sort === 'size' ? 'name' : sort, (e) => e.name, (e) => e.size, (e) => e.mtime)
    const files = sortBy(list.filter((e) => e.type === 'file'), sort, (e) => e.name, (e) => e.size, (e) => e.mtime)
    return [...dirs, ...files]
  }, [entries, showHidden, sort])
  const hiddenCount = (entries ?? []).filter((e) => e.name.startsWith('.')).length
  // One of the user's folders whose agents keep a shared datastore
  // (vfs:<folder>/.datastore) — its routes read the user's own folders only.
  const hasTables = !!open && !open.ref.owner && !open.path && !!entries?.some((e) => e.type === 'dir' && e.name === '.datastore')

  const entryItem = useCallback((e: VFSEntry): FileItem | null => (open && e.type === 'file'
    ? { ref: open.ref, path: e.path, name: e.name, size: e.size, mtime: e.mtime, author: e.author }
    : null), [open])

  // What the viewer steps through: the files of the list in front.
  const listed = useMemo<FileItem[]>(() => (open && !debounced
    ? visibleEntries.map(entryItem).filter((i): i is FileItem => !!i)
    : hitItems), [open, debounced, visibleEntries, entryItem, hitItems])

  // The selected file follows its fresh copy (an agent changed it).
  useEffect(() => {
    setSelected((cur) => {
      if (!cur) return cur
      const fresh = listed.find((i) => itemKey(i) === itemKey(cur))
      return fresh && (fresh.mtime !== cur.mtime || fresh.size !== cur.size || fresh.author !== cur.author) ? fresh : cur
    })
  }, [listed])

  // ── Moving around ──

  const openAt = (ref: FolderRef, path = '', select?: FileItem) => {
    setOpen({ ref, path })
    setSelected(select ?? null)
    setQuery('')
    setDebounced('')
  }
  const goPath = (path: string) => { if (open) { setOpen({ ref: open.ref, path }); setSelected(null) } }
  // Where Back from a folder goes: the list it was opened from — Recent
  // files, or its own group — else all folders (a new folder made from
  // another group's list, a copy of a shared one).
  const backTo: Nav = nav === 'recent' || (nav !== '' && groupOf(openFolder) === nav) ? nav : ''
  const closeFolder = () => {
    if (open) setNav(backTo)
    setOpen(null); setEntries(null); setSelected(null); setQuery(''); setDebounced('')
  }
  const pickNav = (n: Nav) => { closeFolder(); setNav(n) }

  // ── Files ──

  const viewable = (i: FileItem) => isViewable(asAgentFile(i.name))
  const openFile = (item: FileItem) => {
    if (getFileTypeGroup(asAgentFile(item.name)) === 'pdf') {
      window.open(vfsApi.viewUrl(item.ref, item.path, item.mtime), '_blank')
      return
    }
    setViewing({ item, edit: false })
  }
  const download = (href: string, name: string) => {
    const a = document.createElement('a')
    a.href = href
    a.download = name
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }
  const afterChange = () => { loadEntries(true); loadProjects(true); if (wantHits) loadHits(true) }

  const upload = async (files: File[]) => {
    if (!open || !canWrite || files.length === 0) return
    setUploading(files.length)
    try {
      await vfsApi.upload(open.ref, open.path, files)
      notify('success', files.length === 1 ? `Added “${files[0].name}”` : `Added ${files.length} files`, openFolder ? `In ${titleOf(openFolder)}` : '')
    } catch (e) {
      notify('error', 'Could not add the files', msg(e))
    } finally {
      setUploading(0)
      afterChange()
    }
  }

  const copyAddress = async (address: string) => {
    try {
      await navigator.clipboard.writeText(address)
      notify('success', 'Address copied', `Paste ${address} into a chat and the agent opens it.`)
    } catch {
      notify('warning', 'Could not copy', address)
    }
  }

  const saveToDrive = async (item: FileItem) => {
    try {
      await vfsApi.exportToDrive(item.ref, item.path)
      notify('success', 'Saved to Google Drive', `“${item.name}” is in My Drive.`)
    } catch (e) {
      notify('error', 'Could not save to Google Drive', msg(e))
    }
  }

  const renameItem = (ref: FolderRef, path: string, name: string, isDir: boolean) => setDialog({
    kind: 'name', title: isDir ? 'Rename folder' : 'Rename file', initial: name, action: 'Rename',
    onSubmit: async (next) => {
      if (next === name) return
      const to = joinPath(parentOf(path), next)
      await vfsApi.rename(ref, path, to)
      if (selected && sameRef(selected.ref, ref) && selected.path === path) setSelected({ ...selected, path: to, name: next })
      afterChange()
    },
  })

  const deleteItem = (ref: FolderRef, path: string, name: string, isDir: boolean) => setDialog({
    kind: 'confirm', title: isDir ? 'Delete this folder?' : 'Delete this file?', action: 'Delete',
    body: isDir
      ? <>“{name}” and everything in it will be gone — for your agents too.</>
      : <>“{name}” will be gone — for your agents too.</>,
    onConfirm: async () => {
      await vfsApi.remove(ref, path)
      if (selected && sameRef(selected.ref, ref) && (selected.path === path || selected.path.startsWith(`${path}/`))) setSelected(null)
      afterChange()
    },
  })

  const newSubfolder = () => {
    if (!open) return
    const at = open
    setDialog({
      kind: 'name', title: 'New folder', initial: '', action: 'Create',
      onSubmit: async (name) => {
        await vfsApi.mkdir(at.ref, joinPath(at.path, name))
        afterChange()
      },
    })
  }

  // ── Folders ──

  const newFolder = () => setDialog({
    kind: 'name', title: 'New folder', initial: '', action: 'Create',
    hint: (name) => (folderName(name) ? `Your agents reach it as vfs:${folderName(name)}/` : 'Letters, numbers, dots and dashes'),
    onSubmit: async (name) => {
      const clean = folderName(name)
      if (!clean) throw new Error('Give it a name with a letter or a number in it.')
      if ((projects ?? []).some((p) => !p.shared && p.name.toLowerCase() === clean.toLowerCase())) {
        throw new Error(`You already have a folder called “${clean}”.`)
      }
      await vfsApi.mkdir({ project: clean, owner: '' }, '')
      await loadProjects(true)
      openAt({ project: clean, owner: '' })
    },
  })

  const removeFolder = (p: VFSProject) => {
    const ref = refOf(p)
    const done = () => {
      if (open && sameRef(open.ref, ref)) closeFolder()
      loadProjects(true)
      if (wantHits) loadHits(true)
    }
    const g = groupOf(p)
    if (g === 'shared') {
      setDialog({
        kind: 'confirm', title: 'Remove this folder?', action: 'Remove',
        body: <>“{titleOf(p)}” leaves your folders. {p.owner_name || p.owner_email || 'Its owner'} still has it — ask them to share it again to get it back.</>,
        onConfirm: async () => { await leaveShare('vfs', p.name, p.owner_id || ''); done() },
      })
    } else if (g === 'gdrive' || g === 'onedrive') {
      const cloned = p.drive?.cloned || 0
      setDialog({
        kind: 'confirm', title: `Disconnect this ${g === 'onedrive' ? 'OneDrive' : 'Drive'} folder?`, action: 'Disconnect',
        body: <>“{titleOf(p)}” leaves your folders. Nothing in {cloudOf(p)} changes.</>,
        check: cloned ? `Also delete the ${cloned} file${cloned === 1 ? '' : 's'} turned into Markdown here` : undefined,
        onConfirm: async (alsoCloned) => { await vfsApi.driveUnmount(p.name, !alsoCloned); done() },
      })
    } else if (g === 'link') {
      setDialog({
        kind: 'confirm', title: 'Unlink this folder?', action: 'Unlink',
        body: <>“{titleOf(p)}” leaves your folders. The folder on this computer{p.link_path ? <> (<span className="break-all font-mono text-xs">{p.link_path}</span>)</> : null} and its files stay as they are.</>,
        onConfirm: async () => { await vfsApi.removeFolder(p.name); done() },
      })
    } else {
      setDialog({
        kind: 'confirm', title: 'Delete this folder?', action: 'Delete folder',
        body: p.files
          ? <>“{titleOf(p)}” and its {p.files} file{p.files === 1 ? '' : 's'} will be gone — for your agents too.</>
          : <>“{titleOf(p)}” will be gone — for your agents too.</>,
        onConfirm: async () => { await vfsApi.removeFolder(p.name); done() },
      })
    }
  }

  const copyShared = async (p: VFSProject) => {
    try {
      const r = await vfsApi.copyShared(refOf(p))
      await loadProjects(true)
      notify('success', 'Copied to your folders', `Your agents can use it as vfs:${r.project}/.`)
      openAt({ project: r.project, owner: '' })
    } catch (e) {
      notify('error', 'Could not copy the folder', msg(e))
    }
  }

  const [driveBusy, setDriveBusy] = useState('')
  const driveAction = async (p: VFSProject, what: 'refresh' | 'clone') => {
    setDriveBusy(p.name)
    try {
      if (what === 'refresh') {
        const r = await vfsApi.driveRefresh(p.name)
        notify('success', `Up to date with ${cloudOf(p)}`, `${r.files} file${r.files === 1 ? '' : 's'}${r.cloned ? `, ${r.cloned} turned into Markdown` : ''}${r.truncated ? ' (capped)' : ''}.`)
      } else {
        const on = !p.drive?.clonemd
        const r = await vfsApi.driveClone(p.name, on)
        notify('success', on ? 'Turning files into Markdown' : 'Markdown copies off',
          on ? `${r.cloned ?? 0} file${r.cloned === 1 ? '' : 's'} so far — agents read them faster.` : 'The ones already made stay.')
      }
    } catch (e) {
      notify('error', cloudOf(p), msg(e))
    } finally {
      setDriveBusy('')
      afterChange()
    }
  }

  const folderMenu = (p: VFSProject, inside: boolean): AgentMenuItem[] => {
    const g = groupOf(p)
    const items: AgentMenuItem[] = []
    if (!inside) items.push({ icon: FolderOpen, label: 'Open', onSelect: () => openAt(refOf(p)) })
    // A Drive or OneDrive folder's files on disk are stand-ins until read: no .zip.
    if (p.files && !isMount(p)) {
      items.push({ icon: Download, label: 'Download as .zip', hint: `${p.files} file${p.files === 1 ? '' : 's'} · ${formatSize(p.bytes)}`, onSelect: () => download(vfsApi.zipUrl(refOf(p)), `${p.name}.zip`) })
    }
    if (g === 'mine' || g === 'runs') {
      items.push({ icon: Share2, label: 'Share with people', hint: 'They can open it here, to view or edit', onSelect: () => setDialog({ kind: 'share', folder: p }) })
    }
    if (g === 'shared') {
      items.push({ icon: FolderInput, label: 'Copy to my folders', hint: 'So your agents can work with it', onSelect: () => copyShared(p) })
    }
    if (isMount(p)) {
      items.push({ icon: RotateCw, label: `Update from ${cloudOf(p)}`, disabled: driveBusy === p.name, onSelect: () => driveAction(p, 'refresh') })
      items.push({
        icon: Sparkles, label: p.drive?.clonemd ? 'Stop making Markdown copies' : 'Make Markdown copies',
        hint: p.drive?.clonemd ? `New files are read from ${cloudOf(p)} when asked` : 'Agents read the files faster', disabled: driveBusy === p.name,
        onSelect: () => driveAction(p, 'clone'),
      })
    }
    {
      const sub = inside && open && sameRef(open.ref, refOf(p)) ? open.path : ''
      items.push({
        icon: ShieldCheck, label: sub ? 'Anonymize this folder' : 'Anonymize the whole folder',
        hint: 'An anonymized copy beside it — the folder isn\'t changed',
        onSelect: () => setDialog({ kind: 'anonymize', target: {
          kind: 'vfs-folder', project: p.name, path: sub, owner: refOf(p).owner,
          name: sub ? sub.split('/').pop() || sub : titleOf(p),
        } }),
      })
    }
    // A mount's hidden entries are only its bookkeeping (manifest, cache).
    if (inside && hiddenCount && !isMount(p)) {
      items.push({ icon: showHidden ? EyeOff : Eye, label: showHidden ? 'Hide hidden files' : `Show hidden files (${hiddenCount})`, onSelect: () => setShowHidden((v) => !v) })
    }
    items.push({
      icon: g === 'link' ? Link2 : g === 'gdrive' ? Cloud : g === 'onedrive' ? Cloudy : g === 'shared' ? X : Trash2,
      label: g === 'link' ? 'Unlink' : isMount(p) ? 'Disconnect' : g === 'shared' ? 'Remove from my folders' : 'Delete folder',
      danger: true,
      onSelect: () => removeFolder(p),
    })
    return items
  }

  // ── Keys ──

  // Escape: a viewer, dialog or menu closes itself first; then the open file
  // on a phone, then the dialog.
  const overlayRef = useRef(false)
  overlayRef.current = !!(viewing || dialog || sharingFile)
  const selectedRef = useRef(selected)
  selectedRef.current = selected
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || overlayRef.current || e.defaultPrevented) return
      if (phone && selectedRef.current) { setSelected(null); return }
      onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, phone])

  // ── Pieces ──

  const header = (
    <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-sky-500/15 text-sky-600 dark:text-sky-400">
        <Folders className="h-[18px] w-[18px]" />
      </span>
      <div className="min-w-0 flex-1">
        <h2 className="truncate text-base font-semibold text-zinc-100">Shared folders</h2>
        {!phone && <p className="truncate text-xs text-zinc-500">Where your agents keep their work together — open, search and add files.</p>}
      </div>
      {!phone && projects && projects.length > 0 && (
        <span className="hidden shrink-0 text-xs text-zinc-500 sm:block">
          {projects.length} folder{projects.length === 1 ? '' : 's'} · {totals.files} file{totals.files === 1 ? '' : 's'} · {formatSize(totals.bytes)}
        </span>
      )}
      <button
        onClick={() => showAbout(aboutHidden)}
        className={`${iconBtn} ${aboutHidden ? '' : 'text-sky-600 dark:text-sky-400'}`}
        title="What are shared folders?"
        aria-label="What are shared folders?"
        aria-pressed={!aboutHidden}
      >
        <Info className="h-4 w-4" />
      </button>
      <button onClick={refreshAll} className={iconBtn} title="Refresh" aria-label="Refresh">
        <RotateCw className={`h-4 w-4 ${hitsLoading || (open && !entries && !entriesError) ? 'animate-spin' : ''}`} />
      </button>
      <FullscreenButton />
      <button onClick={onClose} className={iconBtn} title="Close" aria-label="Close">
        <X className="h-4 w-4" />
      </button>
    </div>
  )

  const searchPlaceholder = open
    ? `Search in ${openFolder ? titleOf(openFolder) : open.ref.project}…`
    : phone ? 'Search folders and files…' : 'Search every folder by file or folder name…'

  const toolbar = (
    <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2.5 wide:px-4">
      <div className="relative min-w-0 flex-1">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={searchPlaceholder}
          className="w-full rounded-lg border border-zinc-800 bg-zinc-900 py-2 pl-8 pr-8 text-sm text-zinc-200 placeholder-zinc-500 focus:border-violet-500/60 focus:outline-none"
        />
        {query && (
          <button onClick={() => setQuery('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-zinc-500 hover:text-zinc-200" aria-label="Clear search">
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
      {phone ? (
        <AgentMenu
          agentName="Shared folders"
          phone
          items={[
            { icon: FolderPlus, label: 'New folder', onSelect: newFolder },
            { icon: Cloud, label: 'Connect a Google Drive folder', onSelect: () => { useVFSStore.getState().loadProjects(); setDialog({ kind: 'drive' }) } },
            { icon: Cloudy, label: 'Connect a OneDrive folder', onSelect: () => setDialog({ kind: 'onedrive' }) },
            ...(canLink ? [{ icon: Link2, label: 'Link a folder', hint: 'From the computer this runs on', onSelect: () => setDialog({ kind: 'link' }) }] : []),
          ]}
        />
      ) : (
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as Sort)}
          aria-label="Sort"
          className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-2 text-sm text-zinc-300 focus:border-violet-500/60 focus:outline-none"
        >
          {SORTS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
      )}
    </div>
  )

  const navRow = (id: Nav, label: string, count: number | null, icon: ReactNode) => {
    const active = nav === id && !open
    const inFolder = !!open && (id === '' ? false : id === groupOf(openFolder) && nav !== 'recent')
    return (
      <li key={id || '_all'}>
        <button
          onClick={() => pickNav(id)}
          className={`flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] transition-colors ${
            active ? 'bg-zinc-800 font-medium text-zinc-100' : inFolder ? 'text-zinc-200 hover:bg-zinc-800/50' : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'}`}
        >
          {icon}
          <span className="min-w-0 flex-1 truncate">{label}</span>
          {count !== null && <span className="shrink-0 text-[11px] tabular-nums text-zinc-500">{count}</span>}
        </button>
      </li>
    )
  }

  const navList = (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
        <ul className="flex flex-col gap-0.5">
          {navRow('recent', 'Recent files', null, <Clock className="h-3.5 w-3.5 shrink-0 text-zinc-500" />)}
        </ul>
        <div className="mt-4 px-2.5 pb-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Folders</div>
        <ul className="flex flex-col gap-0.5">
          {navRow('', 'All folders', projects?.length ?? 0, <Folders className="h-3.5 w-3.5 shrink-0 text-zinc-500" />)}
          {GROUPS.filter((g) => counts[g.id] > 0 || nav === g.id).map((g) => {
            const Icon = g.icon
            return navRow(g.id, g.label, counts[g.id], <Icon className={`h-3.5 w-3.5 shrink-0 ${g.tile.split(' ').filter((c) => c.includes('text-')).join(' ')}`} />)
          })}
        </ul>
      </div>
      <div className="flex flex-col gap-1 border-t border-zinc-800 p-2">
        <button onClick={newFolder} className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[13px] font-medium text-violet-600 transition-colors hover:bg-violet-500/10 dark:text-violet-400">
          <FolderPlus className="h-4 w-4 shrink-0" />New folder
        </button>
        <button
          onClick={() => { useVFSStore.getState().loadProjects(); setDialog({ kind: 'drive' }) }}
          className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[13px] text-zinc-400 transition-colors hover:bg-zinc-800/50 hover:text-zinc-200"
        >
          <Cloud className="h-4 w-4 shrink-0 text-blue-600 dark:text-blue-400" />Connect Google Drive
        </button>
        <button
          onClick={() => setDialog({ kind: 'onedrive' })}
          className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[13px] text-zinc-400 transition-colors hover:bg-zinc-800/50 hover:text-zinc-200"
        >
          <Cloudy className="h-4 w-4 shrink-0 text-indigo-600 dark:text-indigo-400" />Connect OneDrive
        </button>
        {canLink && (
          <button onClick={() => setDialog({ kind: 'link' })} className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[13px] text-zinc-400 transition-colors hover:bg-zinc-800/50 hover:text-zinc-200">
            <Link2 className="h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" />Link a folder
          </button>
        )}
      </div>
    </div>
  )

  // A phone: the same choices as a row of chips.
  const navChips = (
    <div className="flex gap-1.5 overflow-x-auto border-b border-zinc-800 px-3 py-2 [scrollbar-width:none]">
      {([['recent', 'Recent'], ['', 'All'], ...GROUPS.filter((g) => counts[g.id] > 0).map((g) => [g.id, g.label])] as [Nav, string][]).map(([id, label]) => (
        <button
          key={id || '_all'}
          onClick={() => pickNav(id)}
          className={`shrink-0 rounded-full border px-3 py-1.5 text-[13px] transition-colors ${
            nav === id && !open ? 'border-violet-500/50 bg-violet-500/10 font-medium text-violet-700 dark:text-violet-300' : 'border-zinc-800 text-zinc-400'}`}
        >
          {label}{id !== 'recent' && <span className="ml-1 text-[11px] tabular-nums opacity-70">{id === '' ? projects?.length ?? 0 : counts[id as Group]}</span>}
        </button>
      ))}
    </div>
  )

  // ── The middle ──

  const fileRow = (item: FileItem, opts: { showFolder: boolean; showDir: boolean }) => {
    const folder = folderOf(item.ref)
    const active = !!selected && itemKey(selected) === itemKey(item)
    const dir = parentOf(item.path)
    return (
      <li key={itemKey(item)}>
        <button
          onClick={() => setSelected(item)}
          onDoubleClick={() => { if (viewable(item)) openFile(item) }}
          className={`flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors ${
            active ? 'border-violet-500/40 bg-violet-500/5' : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-900/80'}`}
        >
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800/80"><KindIcon name={item.name} /></span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-sm font-medium text-zinc-100"><Highlight text={item.name} terms={terms} /></span>
            <span className="mt-0.5 flex min-w-0 items-center gap-1.5 text-[11px] text-zinc-500">
              {opts.showFolder && folder && (
                <span className="flex min-w-0 shrink items-center gap-1 truncate">
                  <FolderTile folder={folder} size="sm" />
                  <span className="truncate text-zinc-400">{titleOf(folder)}</span>
                </span>
              )}
              {opts.showDir && dir && <span className="min-w-0 truncate font-mono"><Highlight text={`${dir}/`} terms={terms} /></span>}
              {(opts.showFolder || (opts.showDir && dir)) && <span className="shrink-0">·</span>}
              <span className="shrink-0">{ago(item.mtime)}</span>
              <span className="shrink-0">·</span>
              <span className="shrink-0">{formatSize(item.size)}</span>
            </span>
          </span>
          {item.author && !phone && (
            <span className="max-w-[9rem] shrink-0 truncate rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400" title={`Last written by ${item.author}`}>
              {item.author}
            </span>
          )}
        </button>
      </li>
    )
  }

  // What these folders are, above the lists — until the viewer hides it.
  const about = !aboutHidden && (
    <div className="mb-3 flex gap-3 rounded-xl border border-sky-500/25 bg-sky-500/5 px-3.5 py-3">
      <Info className="mt-0.5 h-4 w-4 shrink-0 text-sky-600 dark:text-sky-400" />
      <div className="min-w-0 flex-1 text-[13px] leading-relaxed text-zinc-400">
        <p>
          <span className="font-semibold text-zinc-100">A file system all your agents share.</span>{' '}
          These folders are a virtual file system: any of your agents can open, read and write them by name
          (<span className="font-mono text-xs text-zinc-300">vfs:folder/file</span>), so what one agent saves, another can pick up.
        </p>
        <p className="mt-1">
          They belong to you, not to an agent: <span className="text-zinc-200">the files stay here even after the agent that made them is removed.</span>{' '}
          Folders you connect from Google Drive or OneDrive show up here too, read-only.
        </p>
      </div>
      <button onClick={() => showAbout(false)} className="h-fit shrink-0 rounded-lg p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200" title="Hide" aria-label="Hide this explanation">
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  )

  const spinner = <div className="flex h-full justify-center pt-16 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
  const failure = (text: string, retry: () => void) => (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-zinc-500">
      <AlertTriangle className="h-5 w-5 text-red-500" />{text}
      <button onClick={retry} className="text-sm font-medium text-violet-600 hover:underline dark:text-violet-400">Try again</button>
    </div>
  )
  const emptyState = (icon: ReactNode, title: string, text: ReactNode, actions?: ReactNode) => (
    <div className="flex h-full flex-col items-center justify-center px-6 py-10 text-center">
      <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-sky-500/15 text-sky-600 dark:text-sky-400">{icon}</span>
      <h3 className="mt-4 text-base font-semibold text-zinc-100">{title}</h3>
      <p className="mt-1.5 max-w-sm text-sm leading-relaxed text-zinc-400">{text}</p>
      {actions && <div className="mt-4 flex flex-wrap justify-center gap-2">{actions}</div>}
    </div>
  )

  // `under`: below the folders that matched — an empty result is a line, not a page.
  const hitList = (showFolder: boolean, under = false) => {
    if (hitsError) return failure(hitsError, () => loadHits())
    if (!hits) return under ? <div className="flex justify-center py-6 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div> : spinner
    if (hitItems.length === 0 && under) return <p className="px-1 text-sm text-zinc-500">No file names match.</p>
    if (hitItems.length === 0) {
      return debounced
        ? emptyState(<Search className="h-7 w-7" />, 'Nothing matches', 'No file or folder name has those words. Try fewer or other words.',
          <button onClick={() => setQuery('')} className="text-sm font-medium text-violet-600 hover:underline dark:text-violet-400">Clear search</button>)
        : emptyState(<Clock className="h-7 w-7" />, 'No files yet', 'What your agents write into shared folders shows up here, newest first.')
    }
    return (
      <>
        <div className="mb-2 flex items-center justify-between gap-2 px-1 text-[11px] text-zinc-500">
          <span className="min-w-0 truncate">
            {debounced
              ? <>{hits.total} file{hits.total === 1 ? '' : 's'} for “{debounced}”{hits.total > hitItems.length ? ` · the newest ${hitItems.length}` : ''}</>
              : 'Newest first'}
          </span>
          {hits.truncated && <span className="shrink-0" title="Your folders hold more files than one search looks through">Some folders not searched</span>}
        </div>
        <ul className="flex flex-col gap-1.5">{hitItems.map((i) => fileRow(i, { showFolder, showDir: true }))}</ul>
      </>
    )
  }

  const folderCard = (p: VFSProject) => {
    const g = groupOf(p)
    const origin = originOf(p)
    return (
      <li key={refKey(refOf(p))} className="group relative">
        <button
          onClick={() => openAt(refOf(p))}
          className="flex h-full w-full flex-col gap-2.5 rounded-xl border border-zinc-800 bg-zinc-900/40 p-3.5 text-left transition-colors hover:border-zinc-700 hover:bg-zinc-900/80"
        >
          <span className="flex min-w-0 items-start gap-3 pr-6">
            <FolderTile folder={p} />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-medium text-zinc-100"><Highlight text={titleOf(p)} terms={terms} /></span>
              <span className={`mt-0.5 block truncate text-[11px] ${p.missing ? 'text-red-600 dark:text-red-400' : 'text-zinc-500'} ${!origin || g === 'link' || (g === 'runs' && p.title) ? 'font-mono' : ''}`}>
                {origin ? <Highlight text={origin} terms={terms} /> : `vfs:${p.name}/`}
              </span>
            </span>
          </span>
          <span className="mt-auto flex flex-wrap items-center gap-1.5 text-[11px] text-zinc-500">
            <span>{p.files} file{p.files === 1 ? '' : 's'} · {formatSize(p.bytes)}{p.mtime ? ` · ${ago(p.mtime)}` : ''}</span>
            {g === 'runs' && p.kind && <span className="rounded-md bg-violet-500/10 px-1.5 py-0.5 text-[10px] font-medium text-violet-700 dark:text-violet-300">{RUN_LABEL[p.kind]}</span>}
            {!writable(p) && !p.missing && (
              <span className="flex items-center gap-1 rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400"><Lock className="h-2.5 w-2.5" />Read-only</span>
            )}
            {isMount(p) && p.drive?.clonemd && <span className="rounded-md bg-emerald-500/10 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:text-emerald-300">Markdown copies</span>}
          </span>
        </button>
        <div className="absolute right-2 top-2.5">
          <AgentMenu agentName={titleOf(p)} items={folderMenu(p, false)} />
        </div>
      </li>
    )
  }

  const foldersView = () => {
    if (projectsError && !projects) return failure(projectsError, () => loadProjects())
    if (!projects) return spinner
    if (projects.length === 0) {
      return (
        <div className="flex h-full flex-col overflow-y-auto px-3 py-3 wide:px-4">
          {about}
          <div className="min-h-0 flex-1">
            {emptyState(<Folders className="h-7 w-7" />, 'No folders yet',
              'When your agents save work to a shared folder it shows up here. Make one yourself, or connect a Google Drive or OneDrive folder.',
              <>
                <button onClick={newFolder} className={primaryBtn}><FolderPlus className="h-4 w-4" />New folder</button>
                <button onClick={() => { useVFSStore.getState().loadProjects(); setDialog({ kind: 'drive' }) }} className={btn}><Cloud className="h-4 w-4" />Google Drive</button>
                <button onClick={() => setDialog({ kind: 'onedrive' })} className={btn}><Cloudy className="h-4 w-4" />OneDrive</button>
              </>)}
          </div>
        </div>
      )
    }
    const noFolders = shownFolders.length === 0 && rawRuns.length === 0
    return (
      <div className="@container h-full overflow-y-auto px-3 py-3 wide:px-4">
        {about}
        {debounced && <div className="mb-2 px-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Folders</div>}
        {noFolders && !debounced && (
          <p className="px-1 py-6 text-center text-sm text-zinc-500">No folders here.</p>
        )}
        {noFolders && debounced && <p className="mb-3 px-1 text-sm text-zinc-500">No folder names match.</p>}
        <ul className="grid grid-cols-1 gap-2 @md:grid-cols-2 @3xl:grid-cols-3">
          {shownFolders.map(folderCard)}
        </ul>
        {rawRuns.length > 0 && (
          <button
            onClick={() => setShowRuns((v) => !v)}
            className="mt-2 flex w-full items-center justify-center gap-1.5 rounded-xl border border-dashed border-zinc-800 py-2 text-xs text-zinc-500 hover:border-zinc-700 hover:text-zinc-300"
          >
            {showRuns
              ? <><ChevronUp className="h-3.5 w-3.5" />Hide unnamed agent runs</>
              : <><ChevronDown className="h-3.5 w-3.5" />{rawRuns.length} more from agent runs, unnamed</>}
          </button>
        )}
        {debounced && (
          <div className="mt-5">
            <div className="mb-2 px-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Files</div>
            {hitList(true, shownFolders.length > 0)}
          </div>
        )}
      </div>
    )
  }

  const recentView = () => (
    <div className="flex h-full flex-col overflow-y-auto px-3 py-3 wide:px-4">
      {about}
      <div className="min-h-0 flex-1">{hitList(true)}</div>
    </div>
  )

  const crumbs = open && (() => {
    const segs = open.path ? open.path.split('/') : []
    return (
      <nav className="flex min-w-0 items-center gap-1 text-[13px]" aria-label="Where you are">
        <button onClick={closeFolder} className="shrink-0 text-zinc-500 hover:text-zinc-200">{backTo === 'recent' ? 'Recent files' : backTo ? GROUP[backTo].label : 'Folders'}</button>
        <ChevronRight className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
        <button onClick={() => goPath('')} className={`min-w-0 truncate ${segs.length ? 'text-zinc-400 hover:text-zinc-200' : 'font-medium text-zinc-100'}`}>
          {openFolder ? titleOf(openFolder) : open.ref.project}
        </button>
        {segs.map((seg, i) => {
          const sub = segs.slice(0, i + 1).join('/')
          const last = i === segs.length - 1
          return (
            <span key={sub} className="flex min-w-0 items-center gap-1">
              <ChevronRight className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
              <button onClick={() => goPath(sub)} className={`min-w-0 truncate ${last ? 'font-medium text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}>{seg}</button>
            </span>
          )
        })}
      </nav>
    )
  })()

  const folderView = () => {
    if (!open) return null
    const p = openFolder
    const origin = p ? originOf(p) : ''
    return (
      <div
        className="relative flex h-full flex-col"
        onDragOver={(e) => { if (canWrite && e.dataTransfer.types.includes('Files')) { e.preventDefault(); if (!dragging) setDragging(true) } }}
        onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false) }}
        onDrop={(e) => {
          if (!canWrite) return
          e.preventDefault()
          setDragging(false)
          if (e.dataTransfer.files.length) upload(Array.from(e.dataTransfer.files))
        }}
      >
        <div className="border-b border-zinc-800 px-3 py-3 wide:px-4">
          <div className="flex items-center gap-2">
            <button
              onClick={() => (open.path ? goPath(parentOf(open.path)) : closeFolder())}
              className="shrink-0 rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
              title={open.path ? 'Up one folder' : 'All folders'}
              aria-label={open.path ? 'Up one folder' : 'All folders'}
            >
              <ArrowLeft className="h-4 w-4" />
            </button>
            {crumbs}
            {p && <span className="ml-auto shrink-0"><AgentMenu agentName={titleOf(p)} items={folderMenu(p, true)} shown /></span>}
          </div>
          {!open.path && p && (
            <div className="mt-3 flex items-start gap-3">
              <FolderTile folder={p} size="lg" />
              <div className="min-w-0 flex-1">
                <h3 className="truncate text-base font-semibold text-zinc-100">{titleOf(p)}</h3>
                <p className="mt-0.5 truncate text-xs text-zinc-500">
                  {GROUP[groupOf(p)].one}{p.kind && RUN_LABEL[p.kind] ? ` · ${RUN_LABEL[p.kind]}` : ''} · {p.files} file{p.files === 1 ? '' : 's'} · {formatSize(p.bytes)}{p.mtime ? ` · changed ${ago(p.mtime)}` : ''}
                </p>
                {origin && <p className={`mt-0.5 truncate text-[11px] ${p.missing ? 'text-red-600 dark:text-red-400' : 'text-zinc-500'} ${groupOf(p) === 'link' ? 'font-mono' : ''}`} title={origin}>{origin}</p>}
                {!p.shared && <p className="mt-1 truncate font-mono text-[11px] text-violet-600/80 dark:text-violet-400/80">vfs:{p.name}/</p>}
              </div>
            </div>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            {canWrite && (
              <>
                <button onClick={() => uploadInput.current?.click()} disabled={uploading > 0} className={`${primaryBtn} py-1.5`}>
                  {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                  {uploading ? `Adding ${uploading}…` : 'Add files'}
                </button>
                <button onClick={newSubfolder} className={`${btn} py-1.5`}><FolderPlus className="h-4 w-4" />New folder</button>
              </>
            )}
            {hasTables && p && (
              <button onClick={() => setDialog({ kind: 'tables', folder: p })} className={`${btn} py-1.5`} title="The tables this folder's agents keep together">
                <Database className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />Tables
              </button>
            )}
            {p && groupOf(p) === 'shared' && (
              <button onClick={() => copyShared(p)} className={`${btn} py-1.5`} title="A copy of your own, so your agents can work with it">
                <FolderInput className="h-4 w-4" />Copy to my folders
              </button>
            )}
            {p && !canWrite && groupOf(p) !== 'shared' && (
              <span className="flex items-center gap-1 px-1 text-[11px] text-zinc-500"><Lock className="h-3 w-3" />Read-only</span>
            )}
          </div>
          <input
            ref={uploadInput}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => { if (e.target.files?.length) upload(Array.from(e.target.files)); e.target.value = '' }}
          />
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3 wide:px-4">
          {debounced ? hitList(false)
            : entriesError ? failure(entriesError, () => loadEntries())
              : !entries ? spinner
                : visibleEntries.length === 0
                  ? emptyState(<FolderOpen className="h-7 w-7" />, 'Nothing here yet',
                    canWrite ? <>Drop files here, or tell an agent to save into <span className="font-mono text-xs">vfs:{open.ref.project}/{open.path ? `${open.path}/` : ''}</span>.</> : 'This folder is empty.')
                  : (
                    <ul className="flex flex-col gap-1.5">
                      {visibleEntries.map((e) => (e.type === 'dir' ? (
                        <li key={e.path} className="group relative">
                          <button
                            onClick={() => goPath(e.path)}
                            className="flex w-full items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-2.5 pr-11 text-left transition-colors hover:border-zinc-700 hover:bg-zinc-900/80"
                          >
                            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-sky-500/10"><Folder className="h-4 w-4 text-sky-600 dark:text-sky-400" /></span>
                            <span className="min-w-0 flex-1">
                              <span className="block truncate text-sm font-medium text-zinc-100">{e.name}</span>
                              <span className="mt-0.5 block text-[11px] text-zinc-500">Folder{e.mtime ? ` · ${ago(e.mtime)}` : ''}</span>
                            </span>
                          </button>
                          <div className="absolute right-2.5 top-1/2 -translate-y-1/2">
                            <AgentMenu agentName={e.name} items={[
                              { icon: ShieldCheck, label: 'Anonymize', hint: 'An anonymized copy beside it',
                                onSelect: () => setDialog({ kind: 'anonymize', target: {
                                  kind: 'vfs-folder', project: open.ref.project, path: e.path, owner: open.ref.owner, name: e.name,
                                } }) },
                              ...(canWrite ? [
                                { icon: TextCursorInput, label: 'Rename', onSelect: () => renameItem(open.ref, e.path, e.name, true) },
                                { icon: Trash2, label: 'Delete', danger: true, onSelect: () => deleteItem(open.ref, e.path, e.name, true) },
                              ] : []),
                            ]} />
                          </div>
                        </li>
                      ) : fileRow(entryItem(e)!, { showFolder: false, showDir: false })))}
                    </ul>
                  )}
        </div>

        {dragging && (
          <div className="pointer-events-none absolute inset-2 z-10 flex items-center justify-center rounded-2xl border-2 border-dashed border-violet-500/70 bg-violet-500/10 backdrop-blur-[1px]">
            <div className="flex items-center gap-2 rounded-xl bg-zinc-950/90 px-4 py-2.5 text-sm font-medium text-violet-700 shadow-lg dark:text-violet-300">
              <Upload className="h-4 w-4" />Drop to add to {open.path ? open.path.split('/').pop() : p ? titleOf(p) : open.ref.project}
            </div>
          </div>
        )}
      </div>
    )
  }

  const main = open ? folderView() : nav === 'recent' ? (debounced ? foldersView() : recentView()) : foldersView()

  // ── The selected file ──

  const selFolder = selected ? folderOf(selected.ref) : undefined
  const selWritable = writable(selFolder)
  const detail = selected && (
    <FileDetail
      key={itemKey(selected)}
      item={selected}
      folder={selFolder}
      phone={phone}
      terms={terms}
      writable={selWritable}
      viewerOpen={!!viewing}
      inFolder={!!open && sameRef(open.ref, selected.ref) && open.path === parentOf(selected.path)}
      onClose={() => setSelected(null)}
      onOpen={() => openFile(selected)}
      onEdit={() => setViewing({ item: selected, edit: true })}
      onDownload={() => download(vfsApi.downloadUrl(selected.ref, selected.path), selected.name)}
      onShare={() => setSharingFile(selected)}
      onCopyAddress={(a) => copyAddress(a)}
      onSaveToDrive={() => saveToDrive(selected)}
      onAnonymize={anonymizable(selected.name) ? () => setDialog({ kind: 'anonymize', target: {
        kind: 'vfs-file', project: selected.ref.project, path: selected.path, owner: selected.ref.owner, name: selected.name,
      } }) : undefined}
      onRename={() => renameItem(selected.ref, selected.path, selected.name, false)}
      onDelete={() => deleteItem(selected.ref, selected.path, selected.name, false)}
      onShowInFolder={() => openAt(selected.ref, parentOf(selected.path), selected)}
    />
  )

  // Prev/Next step through the list's files the viewer opens (a PDF opens in a tab).
  const viewList = listed.filter((i) => viewable(i) && getFileTypeGroup(asAgentFile(i.name)) !== 'pdf')
  const viewPos = viewing ? viewList.findIndex((i) => itemKey(i) === itemKey(viewing.item)) : -1
  const viewingFolder = viewing ? folderOf(viewing.item.ref) : undefined
  const viewer = viewing && (
    <FileViewer
      key={viewing.edit ? `edit:${itemKey(viewing.item)}` : 'view'}
      file={toAgentFile(viewing.item, viewingFolder)}
      urls={{
        view: vfsApi.viewUrl(viewing.item.ref, viewing.item.path, viewing.item.mtime),
        download: vfsApi.downloadUrl(viewing.item.ref, viewing.item.path),
      }}
      readOnly={!writable(viewingFolder)}
      onSave={async (text) => {
        await vfsApi.write(viewing.item.ref, viewing.item.path, text)
        const size = new TextEncoder().encode(text).length
        setViewing((v) => v && { ...v, item: { ...v.item, size } })
        afterChange()
      }}
      startInEdit={viewing.edit}
      onClose={() => setViewing(null)}
      hasPrev={!viewing.edit && viewPos > 0}
      hasNext={!viewing.edit && viewPos >= 0 && viewPos < viewList.length - 1}
      onPrev={() => { if (viewPos > 0) setViewing({ item: viewList[viewPos - 1], edit: false }) }}
      onNext={() => { if (viewPos >= 0 && viewPos < viewList.length - 1) setViewing({ item: viewList[viewPos + 1], edit: false }) }}
    />
  )

  const dialogs = (
    <>
      {dialog?.kind === 'name' && <NameDialog {...dialog} onClose={() => setDialog(null)} />}
      {dialog?.kind === 'anonymize' && (
        <AnonymizeHereDialog target={dialog.target} onClose={() => setDialog(null)} onDone={afterChange} />
      )}
      {dialog?.kind === 'confirm' && <ConfirmDialog {...dialog} onClose={() => setDialog(null)} />}
      {dialog?.kind === 'share' && (
        <ShareModal
          resourceType="vfs"
          resourceId={dialog.folder.name}
          resourceName={titleOf(dialog.folder)}
          note="They open it here, in their own Shared folders. To let their agents use it, they copy it to their folders."
          onClose={() => setDialog(null)}
        />
      )}
      {dialog?.kind === 'tables' && (
        <DatastoreBrowser vfsProject={dialog.folder.name} title={`Tables — ${titleOf(dialog.folder)}`} onClose={() => setDialog(null)} />
      )}
      {dialog?.kind === 'drive' && (
        <DriveConnect
          onClose={() => setDialog(null)}
          onConnectGoogle={onConnect ? () => onConnect('google') : undefined}
          onDone={async (name) => {
            setDialog(null)
            await loadProjects(true)
            notify('success', 'Google Drive folder connected', `Your agents can read it as vfs:${name}/.`)
            openAt({ project: name, owner: '' })
          }}
        />
      )}
      {dialog?.kind === 'onedrive' && (
        <OneDriveConnect
          taken={new Set((projects ?? []).filter((p) => !p.shared).map((p) => p.name))}
          onClose={() => setDialog(null)}
          onConnectMicrosoft={onConnect ? () => onConnect('microsoft') : undefined}
          onDone={async (name) => {
            setDialog(null)
            await loadProjects(true)
            notify('success', 'OneDrive folder connected', `Your agents can read it as vfs:${name}/.`)
            openAt({ project: name, owner: '' })
          }}
        />
      )}
      {dialog?.kind === 'link' && (
        <LinkFolderDialog
          taken={new Set((projects ?? []).filter((p) => !p.shared).map((p) => p.name.toLowerCase()))}
          onClose={() => setDialog(null)}
          onLinked={async (name) => {
            setDialog(null)
            await loadProjects(true)
            notify('success', 'Folder linked', `Your agents can use it as vfs:${name}/.`)
            openAt({ project: name, owner: '' })
          }}
        />
      )}
      {sharingFile && (
        <FileActionSheet
          filename={sharingFile.name}
          size={sharingFile.size}
          icon={<KindIcon name={sharingFile.name} />}
          downloadUrl={vfsApi.downloadUrl(sharingFile.ref, sharingFile.path)}
          onDownload={() => download(vfsApi.downloadUrl(sharingFile.ref, sharingFile.path), sharingFile.name)}
          onClose={() => setSharingFile(null)}
        />
      )}
    </>
  )

  if (phone) {
    return (
      <div className="relative flex h-full flex-col bg-zinc-950">
        {header}
        {toolbar}
        {!open && navChips}
        <div className="min-h-0 flex-1">{main}</div>
        {selected && <div className="absolute inset-0 z-10 flex flex-col bg-zinc-950">{detail}</div>}
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
        <aside className="w-56 shrink-0 border-r border-zinc-800 bg-zinc-900/40">{navList}</aside>
        <section className="min-w-0 flex-1">{main}</section>
        {selected && <aside className="w-[360px] shrink-0 border-l border-zinc-800 bg-zinc-900/30">{detail}</aside>}
      </div>
      {viewer}
      {dialogs}
    </div>
  )
}

// ── One file ──

function FileDetail({
  item, folder, phone, terms, writable: canWrite, viewerOpen, inFolder,
  onClose, onOpen, onEdit, onDownload, onShare, onCopyAddress, onSaveToDrive, onAnonymize, onRename, onDelete, onShowInFolder,
}: {
  item: FileItem
  folder: VFSProject | undefined
  phone: boolean
  terms: string[]
  writable: boolean
  /** The viewer is open over the dialog (a playing preview pauses). */
  viewerOpen: boolean
  /** The open folder is the one the file is in. */
  inFolder: boolean
  onClose: () => void; onOpen: () => void; onEdit: () => void; onDownload: () => void; onShare: () => void
  onCopyAddress: (address: string) => void; onSaveToDrive: () => void; onAnonymize?: () => void; onRename: () => void; onDelete: () => void
  onShowInFolder: () => void
}) {
  const file = toAgentFile(item, folder)
  const group = getFileTypeGroup(file)
  const canView = isViewable(file)
  const canEdit = canWrite && !phone && isEditable(file)
  // How the user's own agents reach it; someone else's folder they can't.
  const address = folder && !folder.shared ? `vfs:${item.ref.project}/${item.path}` : ''
  const dir = parentOf(item.path)

  const meta: [string, ReactNode][] = []
  if (folder) {
    meta.push(['Folder', (
      <button onClick={onShowInFolder} disabled={inFolder} className="flex min-w-0 items-center gap-1.5 text-left text-violet-600 hover:underline disabled:text-zinc-300 disabled:no-underline dark:text-violet-400" title={inFolder ? undefined : 'Show it in its folder'}>
        <span className="truncate">{titleOf(folder)}{dir ? ` / ${dir}` : ''}</span>
      </button>
    )])
  }
  if (item.author) meta.push(['Written by', item.author])
  meta.push(['Size', formatSize(item.size)])
  meta.push(['Changed', fmtWhen(item.mtime)])

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2.5">
        {phone ? (
          <button onClick={onClose} className="flex flex-1 items-center gap-1.5 text-sm text-violet-600 dark:text-violet-400">
            <ArrowLeft className="h-4 w-4" />Back
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
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-zinc-800/80"><KindIcon name={item.name} className="h-5 w-5" /></span>
          <div className="min-w-0 flex-1">
            <h3 className="break-words text-base font-semibold leading-snug text-zinc-100"><Highlight text={item.name} terms={terms} /></h3>
            <div className="mt-1 flex flex-wrap items-center gap-1.5">
              <span className="rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">{extOf(item.name).replace('.', '').toUpperCase() || 'FILE'}</span>
              {!canWrite && <span className="flex items-center gap-1 rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400"><Lock className="h-2.5 w-2.5" />Read-only</span>}
            </div>
          </div>
        </div>

        {address && (
          <button
            onClick={() => onCopyAddress(address)}
            title="Copy — paste it into a chat and the agent opens this file"
            className="group mt-4 flex w-full items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-left transition-colors hover:border-violet-500/40"
          >
            <HardDrive className="h-3.5 w-3.5 shrink-0 text-violet-600 dark:text-violet-400" />
            <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-zinc-300">{address}</span>
            <Copy className="h-3.5 w-3.5 shrink-0 text-zinc-500 group-hover:text-zinc-200" />
          </button>
        )}

        <dl className="mt-4 grid grid-cols-[84px_1fr] gap-x-3 gap-y-2 text-xs">
          {meta.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-zinc-500">{k}</dt>
              <dd className="min-w-0 text-zinc-300">{v}</dd>
            </div>
          ))}
        </dl>

        <FilePreview
          key={`${itemKey(item)}:${item.mtime}`}
          group={group}
          extension={extOf(item.name)}
          size={item.size}
          url={vfsApi.viewUrl(item.ref, item.path, item.mtime)}
          freshUrl={() => vfsApi.viewUrl(item.ref, item.path, item.mtime)}
          thumbUrl={vfsApi.thumbUrl(item.ref, item.path, item.mtime)}
          cacheKey={`${itemKey(item)}:${item.mtime}`}
          untrusted={isUntrustedFile(file)}
          phone={phone}
          viewerOpen={viewerOpen}
          canOpen={canView}
          onOpen={onOpen}
        />
      </div>

      <div className="grid grid-cols-2 gap-2 border-t border-zinc-800 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:pb-3">
        {canView && <button onClick={onOpen} className={primaryBtn}><Eye className="h-4 w-4" />Open</button>}
        <button onClick={onDownload} className={btn}><Download className="h-4 w-4" />Download</button>
        {canEdit && <button onClick={onEdit} className={btn}><Pencil className="h-4 w-4" />Edit</button>}
        {phone && canShareFiles && <button onClick={onShare} className={btn}><Share2 className="h-4 w-4" />Share</button>}
        {folder?.kind !== 'gdrive' && <button onClick={onSaveToDrive} className={btn} title="A copy into your Google Drive (My Drive)"><CloudUpload className="h-4 w-4" />To Drive</button>}
        {onAnonymize && <button onClick={onAnonymize} className={btn} title="An anonymized copy beside it"><ShieldCheck className="h-4 w-4" />Anonymize</button>}
        {canWrite && <button onClick={onRename} className={btn}><TextCursorInput className="h-4 w-4" />Rename</button>}
        {canWrite && (
          <button onClick={onDelete} className={`${btn} hover:border-red-500/40 hover:bg-red-500/10 hover:text-red-600 dark:hover:text-red-400`}>
            <Trash2 className="h-4 w-4" />Delete
          </button>
        )}
      </div>
    </div>
  )
}

// ── Small dialogs ──

function SmallDialog({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  // Esc closes this one only (the folders dialog waits while it's open).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose() } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="fixed inset-0 z-[60] flex items-end justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label={title}
        className="w-full max-w-md rounded-t-2xl border border-zinc-800 bg-zinc-950 pb-[env(safe-area-inset-bottom)] shadow-2xl wide:rounded-2xl wide:pb-0"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3.5">
          <h3 className="text-[15px] font-semibold text-zinc-100">{title}</h3>
          <button onClick={onClose} className={iconBtn} aria-label="Close"><X className="h-4 w-4" /></button>
        </div>
        {children}
      </div>
    </div>
  )
}

function NameDialog({ title, initial, action, hint, onSubmit, onClose }: {
  title: string; initial: string; action: string
  hint?: (name: string) => string
  onSubmit: (name: string) => Promise<void>
  onClose: () => void
}) {
  const [name, setName] = useState(initial)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const input = useRef<HTMLInputElement | null>(null)
  useEffect(() => {
    const el = input.current
    if (!el) return
    el.focus()
    // A file's name without its extension, ready to type over.
    const dot = initial.lastIndexOf('.')
    el.setSelectionRange(0, dot > 0 ? dot : initial.length)
  }, [initial])
  const value = name.trim()
  const bad = value.includes('/') || value.includes('\\') ? 'A name can\'t have a slash in it.'
    : value === '.' || value === '..' ? 'Pick another name.' : ''
  const submit = async () => {
    if (!value || bad || busy) return
    setBusy(true)
    setError('')
    try {
      await onSubmit(value)
      onClose()
    } catch (e) {
      setError(msg(e))
      setBusy(false)
    }
  }
  return (
    <SmallDialog title={title} onClose={onClose}>
      <div className="px-5 py-4">
        <input
          ref={input}
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') submit() }}
          placeholder="Name"
          className="h-11 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 text-base text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none wide:h-10 wide:text-sm"
        />
        <p className={`mt-2 min-h-[1rem] text-xs ${error || bad ? 'text-red-600 dark:text-red-400' : 'text-zinc-500'}`}>
          {error || bad || (hint ? <span className="font-mono">{hint(value)}</span> : '')}
        </p>
      </div>
      <div className="flex justify-end gap-2 border-t border-zinc-800 px-5 py-3">
        <button onClick={onClose} className={btn}>Cancel</button>
        <button onClick={submit} disabled={!value || !!bad || busy} className={primaryBtn}>
          {busy && <Loader2 className="h-4 w-4 animate-spin" />}{action}
        </button>
      </div>
    </SmallDialog>
  )
}

function ConfirmDialog({ title, body, action, check, onConfirm, onClose }: {
  title: string; body: ReactNode; action: string; check?: string
  onConfirm: (checked: boolean) => Promise<void>
  onClose: () => void
}) {
  const [checked, setChecked] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const go = async () => {
    setBusy(true)
    setError('')
    try {
      await onConfirm(checked)
      onClose()
    } catch (e) {
      setError(msg(e))
      setBusy(false)
    }
  }
  return (
    <SmallDialog title={title} onClose={onClose}>
      <div className="px-5 py-4 text-sm leading-relaxed text-zinc-300">
        {body}
        {check && (
          <label className="mt-3 flex items-start gap-2 text-[13px] text-zinc-400">
            <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} className="mt-0.5 accent-red-500" />
            {check}
          </label>
        )}
        {error && <p className="mt-3 text-xs text-red-600 dark:text-red-400">{error}</p>}
      </div>
      <div className="flex justify-end gap-2 border-t border-zinc-800 px-5 py-3">
        <button onClick={onClose} className={btn}>Cancel</button>
        <button
          onClick={go}
          disabled={busy}
          className="flex items-center justify-center gap-1.5 rounded-lg bg-red-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-red-500 disabled:opacity-50"
        >
          {busy && <Loader2 className="h-4 w-4 animate-spin" />}{action}
        </button>
      </div>
    </SmallDialog>
  )
}

/** A folder on this computer, linked in by its path: its agents reach it as
 *  vfs:<name>/ (the files stay where they are). */
function LinkFolderDialog({ taken, onClose, onLinked }: {
  taken: Set<string>
  onClose: () => void
  onLinked: (name: string) => void
}) {
  const [path, setPath] = useState('')
  const [name, setName] = useState('')
  const [named, setNamed] = useState(false)
  const [mode, setMode] = useState<'rw' | 'ro'>('ro')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [browsing, setBrowsing] = useState<{ path: string; parent: string; dirs: { name: string; hidden: boolean; is_git: boolean }[] } | null>(null)

  const go = async (p: string) => {
    try { setBrowsing(await vfsApi.browseFs(p)) } catch (e) { setError(msg(e)) }
  }
  const pick = (p: string) => {
    setPath(p)
    if (!named) setName(folderName(p.split('/').filter(Boolean).pop() || ''))
    setBrowsing(null)
  }
  const clean = folderName(name)
  const clash = clean && taken.has(clean.toLowerCase()) ? `You already have a folder called “${clean}”.` : ''
  const submit = async () => {
    if (!clean || !path.trim() || clash || busy) return
    setBusy(true)
    setError('')
    try {
      await vfsApi.addLink(clean, path.trim(), mode)
      onLinked(clean)
    } catch (e) {
      setError(msg(e))
      setBusy(false)
    }
  }

  return (
    <SmallDialog title="Link a folder" onClose={onClose}>
      <div className="flex flex-col gap-3 px-5 py-4">
        {!browsing && (
          <p className="text-[13px] leading-relaxed text-zinc-400">
            A folder on the computer this runs on. Its files stay where they are — your agents reach them by the name you give it.
          </p>
        )}
        {browsing ? (
          <div className="flex max-h-[50vh] flex-col overflow-hidden rounded-xl border border-zinc-800">
            <div className="flex items-center gap-2 border-b border-zinc-800 bg-zinc-900/60 px-3 py-2">
              <button onClick={() => browsing.parent && go(browsing.parent)} disabled={!browsing.parent} className="rounded p-1 text-zinc-500 hover:text-zinc-200 disabled:opacity-40" aria-label="Up">
                <ArrowLeft className="h-4 w-4" />
              </button>
              <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-zinc-400" title={browsing.path}>{browsing.path}</span>
            </div>
            <ul className="min-h-0 flex-1 overflow-y-auto p-1">
              {browsing.dirs.filter((d) => !d.hidden).map((d) => (
                <li key={d.name}>
                  <button onClick={() => go(`${browsing.path}/${d.name}`)} className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm text-zinc-200 hover:bg-zinc-800/60">
                    <Folder className="h-4 w-4 shrink-0 text-sky-600 dark:text-sky-400" />
                    <span className="min-w-0 flex-1 truncate">{d.name}</span>
                    {d.is_git && <span className="shrink-0 rounded bg-emerald-500/10 px-1 text-[9px] font-medium uppercase text-emerald-700 dark:text-emerald-300">git</span>}
                  </button>
                </li>
              ))}
              {browsing.dirs.filter((d) => !d.hidden).length === 0 && <li className="px-2 py-3 text-xs text-zinc-500">No folders inside.</li>}
            </ul>
            <div className="flex justify-end gap-2 border-t border-zinc-800 px-3 py-2">
              <button onClick={() => setBrowsing(null)} className={`${btn} py-1.5`}>Back</button>
              <button onClick={() => pick(browsing.path)} className={`${primaryBtn} py-1.5`}>Use this folder</button>
            </div>
          </div>
        ) : (
          <>
            <label className="block">
              <span className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Folder</span>
              <span className="flex gap-2">
                <input
                  value={path}
                  onChange={(e) => setPath(e.target.value)}
                  placeholder="/Users/you/Documents/Reports"
                  className="h-10 min-w-0 flex-1 rounded-xl border border-zinc-800 bg-zinc-900 px-3 font-mono text-xs text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
                />
                <button onClick={() => go(path.trim())} className={btn}><FolderOpen className="h-4 w-4" />Browse</button>
              </span>
            </label>
            <label className="block">
              <span className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Name</span>
              <input
                value={name}
                onChange={(e) => { setName(e.target.value); setNamed(true) }}
                placeholder="reports"
                className="h-10 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 text-sm text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
              />
              <span className={`mt-1 block text-xs ${clash ? 'text-red-600 dark:text-red-400' : 'font-mono text-zinc-500'}`}>
                {clash || (clean ? `Your agents reach it as vfs:${clean}/` : ' ')}
              </span>
            </label>
            <div className="grid grid-cols-2 gap-2">
              {([['ro', 'Read only', 'Agents read, never change'], ['rw', 'Can change', 'Agents may add and edit files']] as const).map(([id, label, text]) => (
                <button
                  key={id}
                  onClick={() => setMode(id)}
                  className={`rounded-xl border px-3 py-2 text-left transition-colors ${mode === id ? 'border-violet-500/50 bg-violet-500/10' : 'border-zinc-800 hover:bg-zinc-800/40'}`}
                >
                  <span className="block text-sm font-medium text-zinc-100">{label}</span>
                  <span className="block text-[11px] text-zinc-500">{text}</span>
                </button>
              ))}
            </div>
            {error && <p className="text-xs text-red-600 dark:text-red-400">{error}</p>}
          </>
        )}
      </div>
      {!browsing && (
        <div className="flex justify-end gap-2 border-t border-zinc-800 px-5 py-3">
          <button onClick={onClose} className={btn}>Cancel</button>
          <button onClick={submit} disabled={!clean || !path.trim() || !!clash || busy} className={primaryBtn}>
            {busy && <Loader2 className="h-4 w-4 animate-spin" />}Link folder
          </button>
        </div>
      )}
    </SmallDialog>
  )
}
