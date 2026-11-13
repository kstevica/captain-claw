import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  Activity, AlertTriangle, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, Download, EyeOff, File, FileImage,
  FileSpreadsheet, FileText, FileUp, Folder, FolderLock, FolderOpen, FolderTree, Inbox, List, ListTodo, Loader2, Play,
  Presentation, RotateCcw, ScanText, Send, ShieldCheck, Square, Tag, Trash2, X,
} from 'lucide-react'
import {
  roomsApi, type FileState, type IntakeReport, type Room, type RoomAlarm, type RoomFile, type RoomRead, type Seed,
} from '../../services/roomsApi'
import { anonymizeApi, type VaultEntry } from '../../services/anonymizeApi'
import { useAuthStore } from '../../stores/authStore'
import { Button, Dialog, EmptyState, Field, Notice, Pill, Segmented, inputCls } from '../console/parts'
import { FullscreenButton } from '../library/libraryParts'
import { AlarmsNotice, ExportCard, ReadsCard } from './RoomsAccess'
import { InboxSection, SectionHead, TasksSection } from './RoomsInbox'
import { RoomFileViewer } from './RoomFileViewer'
import { IconBtn, SearchField, TokenChip } from './roomParts'
import { STATE_LABEL, STATE_TONE, clock, count, duration, errorText, fileSize, runEta } from './roomText'

// One data room: a left rail with what the room has (its files, as folders),
// what you tell it (names to hide, roles) and what Claude did with it (its
// deliveries, your tasks for it, what it read); the chosen part on the right.

type Section = 'files' | 'add' | 'names' | 'roles' | 'inbox' | 'tasks' | 'activity' | 'export'

const FILE_TONE: Record<FileState, 'zinc' | 'emerald' | 'red' | 'amber' | 'sky'> = {
  new: 'sky', done: 'emerald', blocked: 'red', failed: 'amber', unread: 'zinc',
}
const FILE_LABEL: Record<FileState, string> = {
  new: 'Not run yet', done: 'Clean', blocked: 'Blocked', failed: 'Failed', unread: "Can't be read",
}
const PHASE_LABEL: Record<string, string> = {
  queued: 'Waiting for another room to finish', anonymizing: 'Anonymizing files', naming: 'Naming folders',
  sweeping: 'Final sweep', checking: 'Checking for leftovers', done: 'Finished',
}
const SEED_FIELDS: { key: keyof Seed; label: string; hint: string }[] = [
  { key: 'orgs', label: 'Companies', hint: 'The target, the buyer, customers, funds — with their legal forms' },
  { key: 'people', label: 'People', hint: 'Founders, managers, shareholders, advisers' },
  { key: 'products', label: 'Products and brands', hint: "The company's own products, apps and services" },
  { key: 'domains', label: 'Web domains', hint: 'e.g. example.hr — they give a company away in links' },
  { key: 'ids', label: 'Registry and ID numbers', hint: 'Company registry numbers, tax numbers' },
  { key: 'other', label: 'Anything else', hint: 'Short names, codenames, addresses, project names' },
]

export function RoomView({ id, onBack, onClose }: {
  id: string
  phone?: boolean
  onBack: () => void
  onClose?: () => void
}) {
  const [room, setRoom] = useState<Room | null>(null)
  const [files, setFiles] = useState<RoomFile[]>([])
  const [entries, setEntries] = useState<VaultEntry[] | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState('')
  const [section, setSection] = useState<Section | null>(null)
  const [viewing, setViewing] = useState<RoomFile | null>(null)
  const [delivery, setDelivery] = useState<number | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [reads, setReads] = useState<RoomRead[]>([])
  const [alarms, setAlarms] = useState<RoomAlarm[]>([])

  const reload = useCallback(async () => {
    try {
      const [r, f] = await Promise.all([roomsApi.get(id), roomsApi.files(id)])
      setRoom(r)
      setFiles(f.files)
    } catch (e) {
      setError(errorText(e))
    }
    roomsApi.reads(id).then((x) => { setReads(x.reads); setAlarms(x.alarms) }).catch(() => {})
  }, [id])
  useEffect(() => { void reload() }, [reload])

  const vaultId = room?.vault_id
  const roomUpdated = room?.updated_at
  useEffect(() => {
    if (!vaultId) return
    anonymizeApi.vault(vaultId).then((r) => setEntries(r.entries)).catch(() => setEntries([]))
  }, [vaultId, roomUpdated])

  const running = !!room && (room.job.running || room.state === 'running')
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => { void reload() }, 2000)
    return () => clearInterval(t)
  }, [running, reload])

  // Escape leaves the room — a dialog open on top handles its own first.
  useEffect(() => {
    if (confirmDelete) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') (onClose || onBack)() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [confirmDelete, onBack, onClose])

  const act = async (label: string, work: () => Promise<unknown>) => {
    setBusy(label); setError(''); setNotice('')
    try {
      await work()
      await reload()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy('')
    }
  }

  if (!room) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        {error ? <Notice tone="red">{error}</Notice> : <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />}
      </div>
    )
  }

  const c = room.counts
  const canRelease = !running && c.done > 0
  const pending = c.done - c.released
  const unseenAlarms = alarms.filter((a) => !a.seen).length
  const current: Section = section || (c.files === 0 ? 'add' : 'files')
  const openDelivery = (seq: number | null) => { setDelivery(seq); if (seq !== null) setSection('inbox') }

  const nav: { group: string; items: { key: Section; icon: typeof Inbox; label: string; badge?: ReactNode }[] }[] = [
    { group: 'Room', items: [
      { key: 'files', icon: FolderTree, label: 'Files', badge: c.blocked > 0 ? <Badge tone="red">{c.blocked}</Badge> : <Badge>{c.files}</Badge> },
      { key: 'add', icon: FileUp, label: 'Add files' },
      { key: 'names', icon: EyeOff, label: 'Names to hide',
        badge: Object.values(room.seed).reduce((n, v) => n + (v?.length || 0), 0) || undefined },
      { key: 'roles', icon: Tag, label: 'Roles', badge: Object.keys(room.roles).length || undefined },
    ] },
    { group: 'Claude', items: [
      { key: 'inbox', icon: Inbox, label: 'From Claude',
        badge: room.inbox.unseen > 0 ? <Badge tone="violet">{room.inbox.unseen} new</Badge> : room.inbox.total || undefined },
      { key: 'tasks', icon: ListTodo, label: 'Tasks for Claude',
        badge: room.tasks?.open ? <Badge tone="amber">{room.tasks.open}</Badge> : room.tasks?.total || undefined },
      { key: 'activity', icon: Activity, label: "Claude's activity",
        badge: unseenAlarms ? <Badge tone="red">!</Badge> : reads.length || undefined },
      { key: 'export', icon: Download, label: 'Download for Claude' },
    ] },
  ]

  return (
    <div className="@container flex h-full min-h-0 flex-col">
      {/* the room's bar */}
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-zinc-800 px-3 py-2.5 wide:px-4">
        <button type="button" onClick={onBack}
          className="inline-flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100">
          <ChevronLeft className="h-4 w-4" /> Data rooms
        </button>
        <span className="hidden h-5 w-px bg-zinc-800 sm:block" />
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
          <FolderLock className="h-[18px] w-[18px]" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-2">
            <h2 className="truncate text-base font-semibold text-zinc-100">{room.codename}</h2>
            <Pill tone={STATE_TONE[room.state]}>{running ? 'Running' : STATE_LABEL[room.state]}</Pill>
          </div>
          <div className="truncate text-xs text-zinc-500">
            {room.name && <>{room.name} · </>}{count(c.files, 'file')}{c.released > 0 && <> · {c.released} released to Claude</>}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {running ? (
            <Button icon={Square} busy={busy === 'stop'} onClick={() => act('stop', () => roomsApi.stop(id))}>
              {busy === 'stop' ? 'Stopping…' : 'Stop'}
            </Button>
          ) : (
            <Button variant="primary" icon={Play} busy={busy === 'run'}
              disabled={c.files === 0 || (c.new === 0 && c.blocked === 0 && c.done === 0)}
              title={c.new === 0 && (c.done > 0 || c.blocked > 0)
                ? 'Go over every file again with the names this room knows now (and the names to hide), then check it for real names. A released file that changes is taken back until you release it again.'
                : undefined}
              onClick={() => act('run', () => roomsApi.run(id))}>
              {c.new > 0 ? (c.done > 0 ? 'Run new files' : 'Run') : 'Check again'}
            </Button>
          )}
          <Button icon={Send} busy={busy === 'release'} disabled={!canRelease || pending <= 0}
            onClick={() => act('release', async () => {
              const r = await roomsApi.release(id)
              setNotice(`${count(r.released_now, 'file')} released. Claude reads them through the Captain Data Rooms connector, if you gave it this room; blocked and failed files stay here.`)
            })}>
            Release{pending > 0 ? ` ${pending}` : ''}
          </Button>
          <Button variant="ghost" icon={Trash2} onClick={() => setConfirmDelete(true)} aria-label="Delete room" title="Delete room" />
          <FullscreenButton />
          {onClose && (
            <button type="button" onClick={onClose} aria-label="Close" title="Close"
              className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </header>
      {(running || (room.job.phase && room.job.phase !== 'done')) && <RunStrip room={room} files={files} />}

      {/* phones: the sections as a strip */}
      <div className="flex gap-1 overflow-x-auto border-b border-zinc-800 px-2 py-2 md:hidden">
        {nav.flatMap((g) => g.items).map((it) => (
          <button key={it.key} type="button" onClick={() => setSection(it.key)}
            className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium ${
              current === it.key ? 'bg-violet-500/15 text-violet-700 dark:text-violet-300' : 'text-zinc-400 hover:bg-zinc-800'}`}>
            <it.icon className="h-3.5 w-3.5" />{it.label}
          </button>
        ))}
      </div>

      <div className="flex min-h-0 flex-1">
        <nav className="hidden w-60 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/20 md:flex">
          <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-3">
            {nav.map((g) => (
              <div key={g.group}>
                <div className="px-2.5 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-500">{g.group}</div>
                <div className="space-y-0.5">
                  {g.items.map((it) => (
                    <NavItem key={it.key} icon={it.icon} label={it.label} badge={it.badge} active={current === it.key}
                      onClick={() => setSection(it.key)} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <Health room={room} />
        </nav>

        <main className="min-h-0 min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-6xl space-y-4 px-4 py-5 wide:px-7 wide:py-6">
            {error && <Notice tone="red">{error}</Notice>}
            {notice && <Notice tone="emerald">{notice}</Notice>}
            <AlarmsNotice roomId={id} alarms={alarms} onSeen={() => void reload()} />
            {!running && room.job.phase === 'done' && room.job.message && (
              <Notice tone="amber">
                <div className="flex items-center gap-2">
                  <Square className="h-3.5 w-3.5 shrink-0" />
                  <span>{room.job.message}</span>
                </div>
              </Notice>
            )}
            {current === 'files' && (
              <FilesSection room={room} files={files} running={running} busy={busy} onOpen={setViewing}
                onRetry={() => act('retry', () => roomsApi.run(id, true))}
                onDownload={(f) => act('dl', () => roomsApi.downloadClean(id, f))}
                onRemove={(f) => act('rm', () => roomsApi.removeFile(id, f.id))}
                onAdd={() => setSection('add')} />
            )}
            {current === 'add' && <AddFilesSection room={room} disabled={running} onDone={() => void reload()} onError={setError} />}
            {current === 'names' && <NamesSection room={room} disabled={running} onSaved={setRoom} onError={setError} />}
            {current === 'roles' && <RolesSection room={room} entries={entries} onSaved={setRoom} />}
            {current === 'inbox' && (
              <InboxSection roomId={id} onChanged={() => void reload()} openSeq={delivery} onOpen={openDelivery} />
            )}
            {current === 'tasks' && <TasksSection roomId={id} onChanged={() => void reload()} onOpenDelivery={openDelivery} />}
            {current === 'activity' && (
              <div className="space-y-4">
                <SectionHead icon={Activity} title="Claude's activity"
                  hint="Every call through the connector: which file and page Claude read, what it delivered — never the text it got. An alarm means a real value showed up in what Claude sent." />
                {reads.length === 0 && unseenAlarms === 0
                  ? <EmptyState icon={Activity} title="Nothing yet">Claude hasn't used this room.</EmptyState>
                  : <ReadsCard reads={reads} headless />}
              </div>
            )}
            {current === 'export' && (
              <div className="space-y-4">
                <SectionHead icon={Download} title="Download for Claude"
                  hint="The released files as a ZIP for a Claude Project, when you'd rather not use the connector." />
                {c.released > 0
                  ? <ExportCard roomId={id} released={c.released} onDone={() => void reload()} headless />
                  : <EmptyState icon={Download} title="Nothing released yet">Release files first: only released files go in the ZIP.</EmptyState>}
              </div>
            )}
          </div>
        </main>
      </div>

      {viewing && <RoomFileViewer roomId={id} file={viewing} entries={entries} onClose={() => setViewing(null)} />}
      {confirmDelete && <DeleteDialog room={room} onClose={() => setConfirmDelete(false)} onDeleted={onBack} />}
    </div>
  )
}

function Badge({ children, tone = 'zinc' }: { children: ReactNode; tone?: 'zinc' | 'red' | 'violet' | 'amber' }) {
  const cls = {
    zinc: 'bg-zinc-800 text-zinc-400',
    red: 'bg-red-500/15 text-red-700 dark:text-red-300',
    violet: 'bg-violet-500/15 text-violet-700 dark:text-violet-300',
    amber: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  }[tone]
  return <span className={`rounded-full px-1.5 py-px text-[10px] font-semibold tabular-nums ${cls}`}>{children}</span>
}

function NavItem({ icon: Icon, label, badge, active, onClick }: {
  icon: typeof Inbox; label: string; badge?: ReactNode; active: boolean; onClick: () => void
}) {
  return (
    <button type="button" onClick={onClick} aria-current={active ? 'page' : undefined}
      className={`group flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors ${
        active ? 'bg-violet-500/10 font-medium text-violet-700 dark:text-violet-200' : 'text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-100'}`}>
      <Icon className={`h-4 w-4 shrink-0 ${active ? 'text-violet-600 dark:text-violet-300' : 'text-zinc-500 group-hover:text-zinc-300'}`} />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {badge !== undefined && (typeof badge === 'number' ? <Badge>{badge}</Badge> : badge)}
    </button>
  )
}

/** What the room can count on: the trusted model and OCR. */
function Health({ room }: { room: Room }) {
  const tone = (ok: boolean) => `mt-0.5 h-3.5 w-3.5 shrink-0 ${ok ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}`
  const row = (icon: ReactNode, label: string, detail: string) => {
    return (
      <div className="flex items-start gap-2" title={detail}>
        {icon}
        <div className="min-w-0">
          <div className="text-[11px] font-medium text-zinc-300">{label}</div>
          <div className="truncate text-[11px] text-zinc-500">{detail}</div>
        </div>
      </div>
    )
  }
  return (
    <div className="space-y-2 border-t border-zinc-800 p-3">
      {row(<ShieldCheck className={tone(!!room.trusted_model)} />, 'Trusted model', room.trusted_model || 'None — code alone looks for names')}
      {row(<ScanText className={tone(room.ocr)} />, 'OCR', room.ocr ? 'Scans and pictures are read' : 'Not installed — scans only removed')}
    </div>
  )
}

const STEP_LABEL: Record<string, string> = {
  reading: 'Reading the file', model: 'Finding names', replacing: 'Replacing names',
}

/** Where a run stands: the phase, the file it's on (size, step, the trusted
 *  model's chunk n of N), and how long the file and the room still take. */
function RunStrip({ room, files }: { room: Room; files: RoomFile[] }) {
  const job = room.job
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(t)
  }, [])
  const total = job.total || 0
  const done = (job.done || 0) + (job.failed || 0)
  const anonymizing = job.phase === 'anonymizing'
  const eta = runEta(job, now)
  const overall = anonymizing && (job.bytes_total || 0) > 0
    ? Math.min(100, (((job.bytes_done || 0) + (job.current_size || 0) * eta.fileShare) / job.bytes_total!) * 100)
    : total ? (done / total) * 100 : 0
  const file = job.current ? files.find((f) => f.id === job.current) : undefined
  const index = job.file_index
  const count = anonymizing ? total : job.file_count
  const chunks = job.chunks || 0
  const filePct = job.step === 'model' && chunks ? ((job.chunk || 0) / chunks) * 100 : job.step === 'replacing' ? 100 : 0
  const since = job.file_started_at ? now - job.file_started_at : 0

  return (
    <div className="space-y-1.5 border-b border-zinc-800 bg-violet-500/[0.04] px-4 py-2.5 text-xs">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-500" />
        <span className="font-medium text-zinc-100">{PHASE_LABEL[job.phase || ''] || job.phase || 'Starting'}</span>
        {anonymizing && total > 0 && <span className="text-zinc-500">{done} of {total} done</span>}
        {job.phase === 'sweeping' && job.sweep_round && <span className="text-zinc-500">round {job.sweep_round}</span>}
        {(anonymizing || job.phase === 'sweeping' || job.phase === 'checking') && (
          <span className="h-1.5 w-48 overflow-hidden rounded-full bg-zinc-800">
            <span className="block h-full rounded-full bg-violet-500 transition-all duration-700" style={{ width: `${overall}%` }} />
          </span>
        )}
        {anonymizing && eta.run !== null && <span className="text-zinc-400">{duration(eta.run)} left for the room</span>}
        {(job.failed || 0) > 0 && <Pill tone="amber">{job.failed} failed</Pill>}
        {room.trusted_model && anonymizing && (
          <span className="ml-auto hidden text-zinc-500 xl:inline">You can close this; it keeps going.</span>
        )}
      </div>
      {file && (
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 pl-[26px] text-zinc-500">
          {index && count ? <span className="tabular-nums">File {index} of {count}</span> : null}
          <span className="max-w-[28rem] truncate font-medium text-zinc-300" title={file.path}>{file.path.split('/').pop()}</span>
          <span>{fileSize(file.size)}{file.info?.rows ? ` · ${file.info.rows.toLocaleString()} rows` : ''}</span>
          {anonymizing && job.step && (
            <>
              <span className="text-zinc-600">·</span>
              <span className="text-zinc-300">{STEP_LABEL[job.step] || job.step}</span>
              {job.step === 'model' && chunks > 0 && (
                <>
                  <span className="tabular-nums">chunk {Math.min((job.chunk || 0) + 1, chunks)} of {chunks}</span>
                  <span className="h-1 w-24 overflow-hidden rounded-full bg-zinc-800">
                    <span className="block h-full rounded-full bg-sky-500 transition-all duration-700" style={{ width: `${filePct}%` }} />
                  </span>
                </>
              )}
              {eta.file !== null
                ? <span>{duration(eta.file)} left on this file</span>
                : job.step === 'model' && chunks > 0 && <span>estimating…</span>}
              {since >= 1 && <span className="tabular-nums text-zinc-500">· running {clock(since)}</span>}
            </>
          )}
        </div>
      )}
    </div>
  )
}

// ── files, as folders ──────────────────────────────────────────────

type FileFilter = 'all' | 'blocked' | 'failed' | 'warnings' | 'unread'

interface Folder { name: string; path: string; folders: Map<string, Folder>; files: RoomFile[]; size: number; problems: number }

function tree(files: RoomFile[]): Folder {
  const root: Folder = { name: '', path: '', folders: new Map(), files: [], size: 0, problems: 0 }
  for (const f of files) {
    const parts = f.path.split('/')
    let node = root
    node.size++
    if (f.state === 'blocked' || f.state === 'failed') node.problems++
    for (const p of parts.slice(0, -1)) {
      let next = node.folders.get(p)
      if (!next) {
        next = { name: p, path: node.path ? `${node.path}/${p}` : p, folders: new Map(), files: [], size: 0, problems: 0 }
        node.folders.set(p, next)
      }
      node = next
      node.size++
      if (f.state === 'blocked' || f.state === 'failed') node.problems++
    }
    node.files.push(f)
  }
  return root
}

function FileIcon({ f }: { f: RoomFile }) {
  const ext = f.path.split('.').pop()?.toLowerCase() || ''
  const cls = 'h-4 w-4 shrink-0'
  if (['xlsx', 'xls', 'csv', 'tsv'].includes(ext)) return <FileSpreadsheet className={`${cls} text-emerald-600 dark:text-emerald-400`} />
  if (['pptx', 'ppt'].includes(ext)) return <Presentation className={`${cls} text-orange-600 dark:text-orange-400`} />
  if (['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'webp', 'gif'].includes(ext)) return <FileImage className={`${cls} text-fuchsia-600 dark:text-fuchsia-400`} />
  if (ext === 'pdf') return <FileText className={`${cls} text-red-600 dark:text-red-400`} />
  if (['docx', 'txt', 'md', 'markdown', 'html', 'htm', 'json'].includes(ext)) return <FileText className={`${cls} text-sky-600 dark:text-sky-400`} />
  return <File className={`${cls} text-zinc-500`} />
}

function FilesSection({ room, files, running, busy, onOpen, onRetry, onDownload, onRemove, onAdd }: {
  room: Room
  files: RoomFile[]
  running: boolean
  busy: string
  onOpen: (f: RoomFile) => void
  onRetry: () => void
  onDownload: (f: RoomFile) => void
  onRemove: (f: RoomFile) => void
  onAdd: () => void
}) {
  const c = room.counts
  const [filter, setFilter] = useState<FileFilter>('all')
  const [q, setQ] = useState('')
  const [asTree, setAsTree] = useState(true)
  const [closed, setClosed] = useState<Set<string>>(new Set())

  const shown = useMemo(() => files.filter((f) => {
    if (q.trim() && !`${f.path} ${f.clean_path}`.toLowerCase().includes(q.trim().toLowerCase())) return false
    if (filter === 'all') return true
    if (filter === 'warnings') return f.state === 'done' && f.warnings.length > 0
    return f.state === filter
  }), [files, filter, q])
  const root = useMemo(() => tree(shown), [shown])
  const hasFolders = root.folders.size > 0

  const toggle = (p: string) => {
    const next = new Set(closed)
    if (next.has(p)) next.delete(p); else next.add(p)
    setClosed(next)
  }

  const row = (f: RoomFile, depth: number) => (
    <FileRow key={f.id} f={f} depth={depth} flat={!asTree || !hasFolders} running={running}
      onOpen={() => onOpen(f)} onDownload={() => onDownload(f)} onRemove={() => onRemove(f)} />
  )
  const folder = (node: Folder, depth: number): ReactNode[] => {
    const out: ReactNode[] = []
    for (const sub of [...node.folders.values()].sort((a, b) => a.name.localeCompare(b.name))) {
      const open = !closed.has(sub.path) || q.trim() !== ''
      out.push(
        <li key={`d:${sub.path}`}>
          <button type="button" onClick={() => toggle(sub.path)}
            className="flex w-full items-center gap-2 py-2 pr-4 text-left text-sm transition-colors hover:bg-zinc-800/30"
            style={{ paddingLeft: 16 + depth * 20 }}>
            {open ? <ChevronDown className="h-3.5 w-3.5 text-zinc-500" /> : <ChevronRight className="h-3.5 w-3.5 text-zinc-500" />}
            {open ? <FolderOpen className="h-4 w-4 text-amber-600 dark:text-amber-400" /> : <Folder className="h-4 w-4 text-amber-600 dark:text-amber-400" />}
            <span className="min-w-0 truncate font-medium text-zinc-200">{sub.name}</span>
            <span className="text-xs text-zinc-500">{sub.size}</span>
            {sub.problems > 0 && <AlertTriangle className="h-3.5 w-3.5 text-red-500" />}
          </button>
        </li>,
      )
      if (open) out.push(...folder(sub, depth + 1))
    }
    for (const f of [...node.files].sort((a, b) => a.path.localeCompare(b.path))) out.push(row(f, depth))
    return out
  }

  return (
    <div className="space-y-4">
      <SectionHead icon={FolderTree} title="Files"
        hint="Open a file to see the original next to what Claude reads. Blocked files still hold a real value and are never released."
        actions={c.failed > 0 && !running && <Button icon={RotateCcw} busy={busy === 'retry'} onClick={onRetry}>Retry failed</Button>} />
      {files.length === 0 ? (
        <EmptyState icon={FolderTree} title="No files yet">
          <button type="button" onClick={onAdd} className="text-violet-600 hover:underline dark:text-violet-400">Add a ZIP or a folder</button> to start.
        </EmptyState>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/30">
          <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-3 py-2.5">
            <SearchField value={q} onChange={setQ} placeholder="Find a file" className="min-w-[12rem] flex-1" />
            <Segmented label="Show" value={filter} onChange={setFilter} options={[
              { value: 'all', label: 'All', count: c.files },
              ...(c.blocked ? [{ value: 'blocked' as const, label: 'Blocked', count: c.blocked }] : []),
              ...(c.failed ? [{ value: 'failed' as const, label: 'Failed', count: c.failed }] : []),
              ...(c.warnings ? [{ value: 'warnings' as const, label: 'Warnings', count: c.warnings }] : []),
              ...(c.unread ? [{ value: 'unread' as const, label: "Can't read", count: c.unread }] : []),
            ]} />
            {hasFolders && (
              <div className="flex items-center rounded-xl border border-zinc-800 p-0.5">
                <IconBtn label="Folders" active={asTree} onClick={() => setAsTree(true)}><FolderTree className="h-4 w-4" /></IconBtn>
                <IconBtn label="One list" active={!asTree} onClick={() => setAsTree(false)}><List className="h-4 w-4" /></IconBtn>
              </div>
            )}
          </div>
          {shown.length === 0 ? (
            <div className="px-5 py-10 text-center text-sm text-zinc-500">No file matches.</div>
          ) : (
            <ul className="divide-y divide-zinc-800/60">
              {asTree && hasFolders ? folder(root, 0) : [...shown].sort((a, b) => a.path.localeCompare(b.path)).map((f) => row(f, 0))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

function FileRow({ f, depth, flat, running, onOpen, onDownload, onRemove }: {
  f: RoomFile
  depth: number
  flat: boolean
  running: boolean
  onOpen: () => void
  onDownload: () => void
  onRemove: () => void
}) {
  const replaced = Object.values(f.entities).reduce((a, b) => a + b, 0)
  const name = flat ? f.path : f.path.split('/').pop() || f.path
  const claudeName = f.clean_path ? (flat ? f.clean_path : f.clean_path.split('/').pop()) : ''
  return (
    <li className="group">
      <div className="flex items-center gap-3 py-2.5 pr-3 transition-colors hover:bg-zinc-800/30" style={{ paddingLeft: 16 + depth * 20 + (flat ? 0 : 20) }}>
        <FileIcon f={f} />
        <button type="button" onClick={onOpen} className="min-w-0 flex-1 text-left">
          <span className="block truncate text-sm text-zinc-100 group-hover:text-violet-700 dark:group-hover:text-violet-300" title={f.path}>{name}</span>
          {claudeName && claudeName !== name && (
            <span className="block truncate text-[11px] text-zinc-500" title={f.clean_path}>Claude sees {claudeName}</span>
          )}
          {f.error && <span className={`block truncate text-[11px] ${f.state === 'blocked' ? 'text-red-600 dark:text-red-400' : 'text-amber-700 dark:text-amber-300'}`}>{f.error}</span>}
        </button>
        <div className="flex shrink-0 items-center gap-1.5">
          <span className="hidden whitespace-nowrap text-right text-[11px] tabular-nums text-zinc-500 md:inline">
            {fileSize(f.size)}
            {f.info?.rows !== undefined && <> · {count(f.info.rows, 'row')}</>}
            {(f.info?.sheets || 0) > 1 && <> · {f.info!.sheets} sheets</>}
          </span>
          {replaced > 0 && <span className="hidden text-[11px] tabular-nums text-zinc-500 lg:inline">· {replaced} hidden</span>}
          {f.warnings.length > 0 && <span title={f.warnings.join('\n')}><AlertTriangle className="h-3.5 w-3.5 text-amber-500" /></span>}
          {f.released
            ? <Pill tone="emerald" icon={CheckCircle2}>Released</Pill>
            : <Pill tone={FILE_TONE[f.state]}>{FILE_LABEL[f.state]}</Pill>}
          <div className="flex items-center opacity-60 transition-opacity group-hover:opacity-100">
            {(f.state === 'done' || f.state === 'blocked') && (
              <IconBtn label="Download the anonymized copy" onClick={onDownload}><Download className="h-4 w-4" /></IconBtn>
            )}
            {!running && <IconBtn label="Remove from the room" onClick={onRemove}><Trash2 className="h-4 w-4" /></IconBtn>}
          </div>
        </div>
      </div>
    </li>
  )
}

// ── adding files ───────────────────────────────────────────────────

function AddFilesSection({ room, disabled, onDone, onError }: {
  room: Room
  disabled: boolean
  onDone: () => void
  onError: (e: string) => void
}) {
  const isAdmin = useAuthStore((s) => s.user?.role === 'admin' || s.authEnabled === false)
  const [zip, setZip] = useState<File | null>(null)
  const [over, setOver] = useState(false)
  const [folder, setFolder] = useState(room.source.kind === 'local' ? room.source.path || '' : '')
  const [project, setProject] = useState(room.source.kind === 'vfs' ? room.source.project || '' : '')
  const [subpath, setSubpath] = useState(room.source.kind === 'vfs' ? room.source.path || '' : '')
  const [busy, setBusy] = useState('')
  const [report, setReport] = useState<IntakeReport | null>(null)

  const take = async (label: string, work: () => Promise<{ report: IntakeReport }>) => {
    setBusy(label); onError('')
    try {
      setReport((await work()).report)
      onDone()
    } catch (e) {
      onError(errorText(e))
    } finally {
      setBusy('')
    }
  }
  const upload = () => {
    if (!zip) return
    if (zip.size > room.max_zip_mb * 1024 * 1024) {
      onError(`The ZIP is over ${room.max_zip_mb} MB — use a folder on the deck's machine instead.`)
      return
    }
    void take('zip', async () => { const r = await roomsApi.uploadZip(room.id, zip); setZip(null); return r })
  }
  const link = () => take('folder', () => roomsApi.linkFolder(room.id,
    isAdmin && folder.trim() ? { path: folder.trim() } : { project: project.trim(), path: subpath.trim() }))

  return (
    <div className="space-y-4">
      <SectionHead icon={FileUp} title="Add files"
        hint={`A ZIP up to ${room.max_zip_mb} MB, or a folder on the deck's machine. Word, Excel, PowerPoint, PDF (scans too), text and pictures; each file up to ${room.max_file_mb} MB. Adding again replaces files that changed. Then run the room.`} />
      {report && <IntakeNotice report={report} onClose={() => setReport(null)} />}
      <div className="grid gap-4 @4xl:grid-cols-2">
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
          <div className="mb-3 text-sm font-medium text-zinc-200">A ZIP</div>
          <label
            onDragOver={(e) => { e.preventDefault(); setOver(true) }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) setZip(f) }}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
              over ? 'border-violet-500/60 bg-violet-500/5' : 'border-zinc-800 hover:border-zinc-600'}`}
          >
            <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
              <FileUp className="h-5 w-5" />
            </span>
            <span className="text-sm text-zinc-300">
              {zip ? <>{zip.name} <span className="text-zinc-500">· {(zip.size / 1024 / 1024).toFixed(1)} MB</span></> : 'Drop a ZIP here, or click to choose one'}
            </span>
            <input type="file" accept=".zip,application/zip" className="sr-only" disabled={disabled}
              onChange={(e) => setZip(e.target.files?.[0] || null)} />
          </label>
          <div className="mt-3 flex justify-end">
            <Button variant="primary" icon={FileUp} busy={busy === 'zip'} disabled={!zip || disabled} onClick={upload}>Add the ZIP</Button>
          </div>
        </div>
        <div className="space-y-3 rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
          <div className="text-sm font-medium text-zinc-200">A folder</div>
          {isAdmin && (
            <Field label="On the deck's machine" hint="Read where it is, never changed. Its full path, starting with /.">
              <input className={inputCls} value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="/srv/vdr/project-falcon" />
            </Field>
          )}
          <div className="grid gap-3 @lg:grid-cols-2">
            <Field label={isAdmin ? '…or a folder in Files' : 'A folder in Files'}>
              <input className={inputCls} value={project} onChange={(e) => setProject(e.target.value)} placeholder="Folder name" />
            </Field>
            <Field label="Path inside it (optional)">
              <input className={inputCls} value={subpath} onChange={(e) => setSubpath(e.target.value)} placeholder="Due diligence" />
            </Field>
          </div>
          <div className="flex justify-end">
            <Button icon={FolderOpen} busy={busy === 'folder'} disabled={disabled || !(folder.trim() && isAdmin) && !project.trim()}
              onClick={() => void link()}>
              Read the folder
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function IntakeNotice({ report, onClose }: { report: IntakeReport; onClose: () => void }) {
  return (
    <Notice tone={report.unread_count || report.skipped_count ? 'amber' : 'emerald'}>
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1 space-y-1">
          <div>
            {count(report.added, 'file')} added{report.replaced ? `, ${report.replaced} replaced` : ''}
            {report.unchanged ? `, ${report.unchanged} unchanged` : ''}{report.nested_zips ? `, ${count(report.nested_zips, 'ZIP')} inside unpacked` : ''}. Run the room to anonymize them.
          </div>
          {report.unread_count > 0 && (
            <div>{count(report.unread_count, 'file')} can't be read: {report.unread.slice(0, 5).map((u) => u.path).join(', ')}{report.unread_count > 5 ? ' …' : ''}</div>
          )}
          {report.skipped_count > 0 && <div>{count(report.skipped_count, 'entry', 'entries')} left out (junk or unsafe paths).</div>}
        </div>
        <button type="button" onClick={onClose} className="opacity-70 hover:opacity-100" aria-label="Dismiss"><X className="h-4 w-4" /></button>
      </div>
    </Notice>
  )
}

// ── names to hide ──────────────────────────────────────────────────

function NamesSection({ room, disabled, onSaved, onError }: {
  room: Room
  disabled: boolean
  onSaved: (r: Room) => void
  onError: (e: string) => void
}) {
  const toText = (s: Seed) => Object.fromEntries(SEED_FIELDS.map((f) => [f.key, (s[f.key] || []).join('\n')])) as Record<keyof Seed, string>
  const [text, setText] = useState(() => toText(room.seed))
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)
  const dirty = SEED_FIELDS.some((f) => text[f.key] !== (room.seed[f.key] || []).join('\n'))

  const save = async () => {
    setBusy(true); onError(''); setSaved(false)
    try {
      const seed = Object.fromEntries(SEED_FIELDS.map((f) => [f.key, text[f.key].split('\n').map((x) => x.trim()).filter(Boolean)])) as Seed
      onSaved(await roomsApi.update(room.id, { seed }))
      setSaved(true)
    } catch (e) {
      onError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-4">
      <SectionHead icon={EyeOff} title="Names to hide"
        hint="Names you already know are hidden first, wherever they appear — including short names a document never spells out in full. One per line. Only you see this list."
        actions={<Button variant={dirty ? 'primary' : 'secondary'} busy={busy} disabled={disabled || (!dirty && !saved)} onClick={save}>
          {saved && !dirty ? 'Saved' : 'Save'}
        </Button>} />
      {saved && !dirty && room.counts.done > 0 && (
        <Notice tone="emerald">Saved. Click Check again at the top to apply the list to the files already done.</Notice>
      )}
      <div className="grid gap-4 @2xl:grid-cols-2 @5xl:grid-cols-3">
        {SEED_FIELDS.map((f) => (
          <div key={f.key} className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-3">
            <Field label={f.label} hint={f.hint}>
              <textarea className={`${inputCls} min-h-[96px] font-mono text-xs`} value={text[f.key]}
                onChange={(e) => { setSaved(false); setText({ ...text, [f.key]: e.target.value }) }} />
            </Field>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── roles ──────────────────────────────────────────────────────────

function RolesSection({ room, entries, onSaved }: { room: Room; entries: VaultEntry[] | null; onSaved: (r: Room) => void }) {
  const [roles, setRoles] = useState<Record<string, string>>(room.roles)
  const [q, setQ] = useState('')
  const [only, setOnly] = useState<'all' | 'set'>('all')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const dirty = JSON.stringify(roles) !== JSON.stringify(room.roles)

  const shown = (entries || []).filter((e) => ['PERSON', 'ORG', 'DETAIL', 'ADDRESS'].includes(e.kind))
    .filter((e) => only === 'all' || (roles[e.token] || '').trim())
    .filter((e) => !q.trim() || `${e.token} ${e.original} ${roles[e.token] || ''}`.toLowerCase().includes(q.trim().toLowerCase()))

  const save = async () => {
    setBusy(true); setError(''); setSaved(false)
    try {
      onSaved(await roomsApi.update(room.id, { roles: Object.fromEntries(Object.entries(roles).filter(([, v]) => v.trim())) }))
      setSaved(true)
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-4">
      <SectionHead icon={Tag} title="Roles"
        hint={'Say what each party is — "Target", "Buyer", "CFO of the target" — never who. Claude sees the role next to the token and reasons much better with it. The real names here are for you only.'}
        actions={<Button variant={dirty ? 'primary' : 'secondary'} busy={busy} disabled={!dirty} onClick={save}>{saved && !dirty ? 'Saved' : 'Save roles'}</Button>} />
      {error && <Notice tone="amber">{error}</Notice>}
      {room.counts.done === 0 ? (
        <EmptyState icon={Tag} title="No tokens yet">Run the room first: the tokens come from its files.</EmptyState>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/30">
          <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-3 py-2.5">
            <SearchField value={q} onChange={setQ} placeholder="Find a name, token or role" className="min-w-[12rem] flex-1" />
            <Segmented label="Show" value={only} onChange={setOnly} options={[
              { value: 'all', label: 'All' }, { value: 'set', label: 'With a role', count: Object.values(roles).filter((v) => v.trim()).length },
            ]} />
          </div>
          {entries === null ? <div className="px-5 py-8 text-sm text-zinc-500">Loading…</div> : (
            <ul className="divide-y divide-zinc-800/60">
              {shown.map((e) => (
                <li key={e.token} className="grid items-center gap-2 px-4 py-2.5 @2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                  <div className="flex min-w-0 items-center gap-2 text-sm">
                    <TokenChip token={e.token} />
                    <span className="truncate text-zinc-200" title={e.original}>{e.original}</span>
                  </div>
                  <input className={inputCls} placeholder="Role — what, not who" maxLength={120} value={roles[e.token] || ''}
                    onChange={(ev) => { setSaved(false); setRoles({ ...roles, [e.token]: ev.target.value }) }} />
                </li>
              ))}
              {shown.length === 0 && <li className="px-4 py-8 text-center text-sm text-zinc-500">Nothing matches.</li>}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

function DeleteDialog({ room, onClose, onDeleted }: { room: Room; onClose: () => void; onDeleted: () => void }) {
  const [vault, setVault] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const remove = async () => {
    setBusy(true); setError('')
    try {
      await roomsApi.remove(room.id, vault)
      onDeleted()
    } catch (e) {
      setError(errorText(e))
      setBusy(false)
    }
  }
  return (
    <Dialog title={`Delete ${room.codename}?`} icon={Trash2} onClose={onClose} footer={
      <div className="flex justify-end gap-2">
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="danger" busy={busy} onClick={remove}>Delete</Button>
      </div>
    }>
      {error && <Notice tone="red">{error}</Notice>}
      <p className="text-sm text-zinc-400">
        The room's originals, anonymized copies and text are deleted from this deck. A folder on the deck's machine is
        never touched. This can't be undone.
      </p>
      <label className="mt-4 flex items-start gap-2 text-sm text-zinc-300">
        <input type="checkbox" checked={vault} onChange={(e) => setVault(e.target.checked)} className="mt-0.5" />
        <span>Delete its vault too — the real names behind the tokens. Anything Claude made from this room can't be put back afterwards.</span>
      </label>
    </Dialog>
  )
}
