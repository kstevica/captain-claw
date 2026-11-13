import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle, ChevronLeft, Download, Eye, FileUp, FolderLock, FolderOpen, Play, Plus, RotateCcw, Send,
  Square, Tag, Trash2, X,
} from 'lucide-react'
import {
  roomsApi, type FileState, type FileText, type IntakeReport, type Room, type RoomFile, type RoomSummary, type Seed,
} from '../../services/roomsApi'
import { anonymizeApi, type VaultEntry } from '../../services/anonymizeApi'
import { useAuthStore } from '../../stores/authStore'
import { Button, Card, Dialog, EmptyState, Field, Notice, Page, PageHeader, Pill, Segmented, inputCls } from '../console/parts'
import { TokenText } from '../anonymize/anonParts'
import { kindLabel, kindTone, when } from '../anonymize/anonFormat'
import { AlarmsNotice, ClaudeAccessCard, ExportCard, ReadsCard } from './RoomsAccess'
import { InboxCard } from './RoomsInbox'
import type { RoomAlarm, RoomRead } from '../../services/roomsApi'

// Data rooms: a virtual data room — a ZIP or a folder of files — anonymized
// into one vault. The person checks what came out, gives the parties roles,
// and releases the room; then Claude reads it through the Captain connector
// and never sees a real name. Results Claude sends back are restored here.

const STATE_LABEL: Record<Room['state'], string> = {
  setup: 'Setting up', running: 'Running', review: 'Ready to check', released: 'Released',
}
const STATE_TONE: Record<Room['state'], 'zinc' | 'sky' | 'amber' | 'emerald'> = {
  setup: 'zinc', running: 'sky', review: 'amber', released: 'emerald',
}
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
  { key: 'other', label: 'Anything else', hint: 'Codenames, addresses, project names' },
]

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function count(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`
}

export function RoomsPage({ phone = false, onClose }: { phone?: boolean; onClose?: () => void }) {
  const [rooms, setRooms] = useState<RoomSummary[] | null>(null)
  const [openId, setOpenId] = useState<string | null>(null)
  const [codename, setCodename] = useState('')
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const reload = useCallback(async () => {
    try {
      setRooms((await roomsApi.list()).rooms)
    } catch (e) {
      setError(errorText(e))
    }
  }, [])

  useEffect(() => { void reload() }, [reload])

  useEffect(() => {
    if (!onClose || openId) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, openId])

  if (openId) {
    return <RoomView id={openId} phone={phone} onBack={() => { setOpenId(null); void reload() }} onClose={onClose} />
  }

  const create = async () => {
    setBusy(true); setError('')
    try {
      const room = await roomsApi.create(codename.trim(), name.trim())
      setCodename(''); setName('')
      setOpenId(room.id)
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="@container h-full overflow-y-auto">
      <Page full={!!onClose}>
        <PageHeader
          icon={FolderLock}
          title="Data rooms"
          description="Put a data room's files here — a ZIP or a folder. Captain anonymizes all of them into one vault; you check the result and release it. Claude then reads it through the Captain connector with every name replaced by a token, and what Claude sends back is restored here, for you alone."
          actions={onClose && <Button variant="ghost" icon={X} onClick={onClose} aria-label="Close" />}
        />
        {error && <Notice tone="red">{error}</Notice>}
        <Card title="New data room" icon={Plus} hint="The codename is what Claude sees — don't use a real name there.">
          <div className="grid gap-4 @lg:grid-cols-2">
            <Field label="Codename (Claude sees this)">
              <input className={inputCls} value={codename} onChange={(e) => setCodename(e.target.value)}
                placeholder="Project Falcon" maxLength={80} />
            </Field>
            <Field label="Your name for it (only you see this)">
              <input className={inputCls} value={name} onChange={(e) => setName(e.target.value)}
                placeholder="The deal's real name" maxLength={200} />
            </Field>
          </div>
          <div className="mt-4">
            <Button variant="primary" icon={Plus} busy={busy} disabled={!codename.trim()} onClick={create}>
              Create data room
            </Button>
          </div>
        </Card>
        {rooms && rooms.length > 0 && <ClaudeAccessCard rooms={rooms} />}
        {rooms === null ? null : rooms.length === 0 ? (
          <EmptyState icon={FolderLock} title="No data rooms yet">
            Create one above, then add a ZIP or a folder.
          </EmptyState>
        ) : (
          <div className="grid gap-3 @2xl:grid-cols-2">
            {rooms.map((r) => (
              <button key={r.id} type="button" onClick={() => setOpenId(r.id)}
                className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-colors hover:border-zinc-600">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate font-medium text-zinc-100">{r.codename}</div>
                    {r.name && <div className="truncate text-sm text-zinc-500">{r.name}</div>}
                  </div>
                  <Pill tone={STATE_TONE[r.state]}>{r.job.running ? 'Running' : STATE_LABEL[r.state]}</Pill>
                </div>
                <div className="mt-3 flex flex-wrap gap-1.5 text-xs text-zinc-500">
                  <span>{count(r.counts.files, 'file')}</span>
                  {r.counts.released > 0 && <Pill tone="emerald">{r.counts.released} released</Pill>}
                  {r.counts.blocked > 0 && <Pill tone="red">{r.counts.blocked} blocked</Pill>}
                  {r.counts.failed > 0 && <Pill tone="amber">{r.counts.failed} failed</Pill>}
                  {r.inbox.unseen > 0 && <Pill tone="sky">{r.inbox.unseen} new from Claude</Pill>}
                  <span className="ml-auto">{when(r.updated_at)}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </Page>
    </div>
  )
}

// ── One room ────────────────────────────────────────────────────────

type FileFilter = 'all' | 'blocked' | 'failed' | 'warnings' | 'unread'

function RoomView({ id, phone, onBack, onClose }: {
  id: string
  phone: boolean
  onBack: () => void
  onClose?: () => void
}) {
  const [room, setRoom] = useState<Room | null>(null)
  const [files, setFiles] = useState<RoomFile[]>([])
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState<string>('')
  const [report, setReport] = useState<IntakeReport | null>(null)
  const [compare, setCompare] = useState<RoomFile | null>(null)
  const [rolesOpen, setRolesOpen] = useState(false)
  const [filter, setFilter] = useState<FileFilter>('all')
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [reads, setReads] = useState<RoomRead[]>([])
  const [alarms, setAlarms] = useState<RoomAlarm[]>([])
  const [inboxOpen, setInboxOpen] = useState(false)

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

  const running = !!room && (room.job.running || room.state === 'running')
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => { void reload() }, 2000)
    return () => clearInterval(t)
  }, [running, reload])

  useEffect(() => {
    if (compare || rolesOpen || confirmDelete || inboxOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') (onClose || onBack)() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [compare, rolesOpen, confirmDelete, inboxOpen, onBack, onClose])

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

  const shown = useMemo(() => files.filter((f) => {
    if (filter === 'all') return true
    if (filter === 'warnings') return f.state === 'done' && f.warnings.length > 0
    return f.state === filter
  }), [files, filter])

  if (!room) {
    return (
      <div className="h-full overflow-y-auto">
        <Page full={!!onClose}>{error ? <Notice tone="red">{error}</Notice> : <div className="text-sm text-zinc-500">Loading…</div>}</Page>
      </div>
    )
  }

  const c = room.counts
  const job = room.job
  const canRelease = !running && c.done > 0
  const pending = c.done - c.released

  return (
    <div className="@container h-full overflow-y-auto">
      <Page full={!!onClose}>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="ghost" icon={ChevronLeft} onClick={onBack}>Data rooms</Button>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            {onClose && <Button variant="ghost" icon={X} onClick={onClose} aria-label="Close" />}
          </div>
        </div>
        <PageHeader
          icon={FolderLock}
          title={room.codename}
          description={<>
            {room.name && <span className="text-zinc-400">{room.name} · </span>}
            {STATE_LABEL[room.state]} · {count(c.files, 'file')}
            {c.released > 0 && <> · {c.released} released to Claude</>}
          </>}
          actions={<>
            {running ? (
              <Button icon={Square} busy={busy === 'stop'} onClick={() => act('stop', () => roomsApi.stop(id))}>Stop</Button>
            ) : (
              <Button variant="primary" icon={Play} busy={busy === 'run'} disabled={c.files === 0 || (c.new === 0 && c.blocked === 0)}
                title={c.blocked > 0 && c.new === 0 ? 'Run the sweep and the check again over the blocked files' : undefined}
                onClick={() => act('run', () => roomsApi.run(id))}>
                {c.done > 0 ? 'Run new files' : 'Run'}
              </Button>
            )}
            <Button icon={Send} busy={busy === 'release'} disabled={!canRelease || pending <= 0}
              onClick={() => act('release', async () => {
                const r = await roomsApi.release(id)
                setNotice(`${count(r.released_now, 'file')} released. Claude reads them through the Captain Data Rooms connector, if you gave it this room; blocked and failed files stay here.`)
              })}>
              Release{pending > 0 ? ` ${pending}` : ''}
            </Button>
            <Button variant="danger" icon={Trash2} onClick={() => setConfirmDelete(true)} aria-label="Delete room" />
          </>}
        />
        {error && <Notice tone="red">{error}</Notice>}
        {notice && <Notice tone="emerald">{notice}</Notice>}
        <AlarmsNotice roomId={id} alarms={alarms} onSeen={() => void reload()} />
        {room.inbox.total > 0 && <InboxCard roomId={id} onChanged={() => void reload()} onDialog={setInboxOpen} />}
        {!room.trusted_model && (
          <Notice tone="amber">
            No trusted model is set up, so code alone looks for private data: emails, IBANs, ID numbers, labelled
            names and your list below — but it misses most names and companies in running text. An admin sets one in
            Anonymize → Trusted model.
          </Notice>
        )}
        {!room.ocr && (
          <Notice tone="amber">No OCR engine is installed: scanned pages and pictures can't be read, only removed.</Notice>
        )}

        <FilesIn room={room} disabled={running} onReport={(r) => { setReport(r); void reload() }} onError={setError} />
        {report && <ReportView report={report} onClose={() => setReport(null)} />}

        <SeedCard room={room} disabled={running} onSaved={setRoom} onError={setError} />

        {(running || job.phase) && <RunCard room={room} />}

        <Card title="Check" icon={Eye}
          hint="Open a file to see the original next to what Claude will read. Blocked files still hold a real value and are never released."
          bodyClassName="p-0">
          <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800/80 px-5 py-3">
            <Segmented label="Show" value={filter} onChange={setFilter} options={[
              { value: 'all', label: 'All', count: c.files },
              { value: 'blocked', label: 'Blocked', count: c.blocked },
              { value: 'failed', label: 'Failed', count: c.failed },
              { value: 'warnings', label: 'Warnings', count: c.warnings },
              { value: 'unread', label: "Can't read", count: c.unread },
            ]} />
            {c.failed > 0 && !running && (
              <Button icon={RotateCcw} busy={busy === 'retry'} onClick={() => act('retry', () => roomsApi.run(id, true))}>
                Retry failed
              </Button>
            )}
          </div>
          {shown.length === 0 ? (
            <div className="px-5 py-8 text-center text-sm text-zinc-500">{files.length ? 'Nothing here.' : 'Add a ZIP or a folder above.'}</div>
          ) : (
            <ul className="divide-y divide-zinc-800/80">
              {shown.map((f) => (
                <FileRow key={f.id} f={f} phone={phone} onOpen={() => setCompare(f)}
                  onDownload={() => act('dl', () => roomsApi.downloadClean(id, f))}
                  onRemove={running ? undefined : () => act('rm', () => roomsApi.removeFile(id, f.id))} />
              ))}
            </ul>
          )}
        </Card>

        <Card title="Roles" icon={Tag}
          hint="Tell Claude what each party is — the target, the buyer, the CFO — without saying who. Claude reasons much better with roles."
          action={<Button icon={Tag} disabled={c.done === 0} onClick={() => setRolesOpen(true)}>Edit roles</Button>}>
          {Object.keys(room.roles).length === 0 ? (
            <p className="text-sm text-zinc-500">No roles yet.{c.done === 0 ? ' Run the room first: the tokens come from its files.' : ''}</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {Object.entries(room.roles).map(([token, role]) => (
                <span key={token} className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-800 px-2 py-1 text-sm">
                  <TokenChip token={token} /> <span className="text-zinc-300">{role}</span>
                </span>
              ))}
            </div>
          )}
        </Card>

        {room.inbox.total === 0 && c.released > 0 && <InboxCard roomId={id} onChanged={() => void reload()} onDialog={setInboxOpen} />}
        {c.released > 0 && <ExportCard roomId={id} released={c.released} onDone={() => void reload()} />}
        {(c.released > 0 || reads.length > 0) && <ReadsCard reads={reads} />}

        {compare && <CompareDialog roomId={id} f={compare} onClose={() => setCompare(null)} />}
        {rolesOpen && (
          <RolesDialog room={room} onClose={() => setRolesOpen(false)} onSaved={setRoom} />
        )}
        {confirmDelete && (
          <DeleteDialog room={room} onClose={() => setConfirmDelete(false)} onDeleted={onBack} />
        )}
      </Page>
    </div>
  )
}

function TokenChip({ token }: { token: string }) {
  const kind = token.replace(/^⟦|⟧$/g, '').replace(/_\d+(:[a-z0-9]+)?$/, '')
  return <span className={`rounded px-1 py-px font-mono text-[0.85em] ${kindTone(kind)}`} title={kindLabel(kind)}>{token}</span>
}

function FilesIn({ room, disabled, onReport, onError }: {
  room: Room
  disabled: boolean
  onReport: (r: IntakeReport) => void
  onError: (e: string) => void
}) {
  const isAdmin = useAuthStore((s) => s.user?.role === 'admin' || s.authEnabled === false)
  const [zip, setZip] = useState<File | null>(null)
  const [over, setOver] = useState(false)
  const [folder, setFolder] = useState(room.source.kind === 'local' ? room.source.path || '' : '')
  const [project, setProject] = useState(room.source.kind === 'vfs' ? room.source.project || '' : '')
  const [subpath, setSubpath] = useState(room.source.kind === 'vfs' ? room.source.path || '' : '')
  const [busy, setBusy] = useState('')

  const upload = async () => {
    if (!zip) return
    if (zip.size > room.max_zip_mb * 1024 * 1024) {
      onError(`The ZIP is over ${room.max_zip_mb} MB — use a folder on the deck's machine instead.`)
      return
    }
    setBusy('zip'); onError('')
    try {
      onReport((await roomsApi.uploadZip(room.id, zip)).report)
      setZip(null)
    } catch (e) {
      onError(errorText(e))
    } finally {
      setBusy('')
    }
  }

  const link = async () => {
    setBusy('folder'); onError('')
    try {
      const body = isAdmin && folder.trim() ? { path: folder.trim() } : { project: project.trim(), path: subpath.trim() }
      onReport((await roomsApi.linkFolder(room.id, body)).report)
    } catch (e) {
      onError(errorText(e))
    } finally {
      setBusy('')
    }
  }

  return (
    <Card title="Files" icon={FileUp}
      hint={`A ZIP up to ${room.max_zip_mb} MB, or a folder on the deck's machine. Word, Excel, PowerPoint, PDF (scans too), text and pictures; each file up to ${room.max_file_mb} MB. Adding again replaces changed files.`}>
      <div className="grid gap-5 @3xl:grid-cols-2">
        <div>
          <label
            onDragOver={(e) => { e.preventDefault(); setOver(true) }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) setZip(f) }}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-6 py-8 text-center transition-colors ${
              over ? 'border-violet-500/60 bg-violet-500/5' : 'border-zinc-800 hover:border-zinc-600'}`}
          >
            <FileUp className="h-6 w-6 text-zinc-500" />
            <span className="text-sm text-zinc-300">
              {zip ? <>{zip.name} <span className="text-zinc-500">· {(zip.size / 1024 / 1024).toFixed(1)} MB</span></> : 'Drop a ZIP here, or click to choose one'}
            </span>
            <input type="file" accept=".zip,application/zip" className="sr-only" disabled={disabled}
              onChange={(e) => setZip(e.target.files?.[0] || null)} />
          </label>
          <div className="mt-3">
            <Button variant="primary" icon={FileUp} busy={busy === 'zip'} disabled={!zip || disabled} onClick={upload}>Add the ZIP</Button>
          </div>
        </div>
        <div className="space-y-3">
          {isAdmin && (
            <Field label="A folder on the deck's machine" hint="Read where it is, never changed. Its full path, starting with /.">
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
          <Button icon={FolderOpen} busy={busy === 'folder'} disabled={disabled || !(folder.trim() && isAdmin) && !project.trim()}
            onClick={link}>
            Read the folder
          </Button>
        </div>
      </div>
    </Card>
  )
}

function ReportView({ report, onClose }: { report: IntakeReport; onClose: () => void }) {
  return (
    <Notice tone={report.unread_count || report.skipped_count ? 'amber' : 'emerald'}>
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1 space-y-1">
          <div>
            {count(report.added, 'file')} added{report.replaced ? `, ${report.replaced} replaced` : ''}
            {report.unchanged ? `, ${report.unchanged} unchanged` : ''}{report.nested_zips ? `, ${count(report.nested_zips, 'ZIP')} inside unpacked` : ''}.
          </div>
          {report.unread_count > 0 && (
            <div>{count(report.unread_count, 'file')} can't be read: {report.unread.slice(0, 5).map((u) => u.path).join(', ')}{report.unread_count > 5 ? ' …' : ''}</div>
          )}
          {report.skipped_count > 0 && <div>{count(report.skipped_count, 'entry', 'entries')} left out (junk or unsafe paths).</div>}
        </div>
        <button type="button" onClick={onClose} className="text-zinc-500 hover:text-zinc-300" aria-label="Dismiss"><X className="h-4 w-4" /></button>
      </div>
    </Notice>
  )
}

function SeedCard({ room, disabled, onSaved, onError }: {
  room: Room
  disabled: boolean
  onSaved: (r: Room) => void
  onError: (e: string) => void
}) {
  const toText = (s: Seed) => Object.fromEntries(SEED_FIELDS.map((f) => [f.key, (s[f.key] || []).join('\n')])) as Record<keyof Seed, string>
  const [text, setText] = useState(() => toText(room.seed))
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)

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
    <Card title="What to look for" icon={AlertTriangle}
      hint="Names you already know: they are anonymized first, wherever they appear. One per line. This helps more than anything else. Only you see this list."
      action={<Button busy={busy} disabled={disabled} onClick={save}>{saved ? 'Saved' : 'Save'}</Button>}>
      <div className="grid gap-4 @2xl:grid-cols-2 @5xl:grid-cols-3">
        {SEED_FIELDS.map((f) => (
          <Field key={f.key} label={f.label} hint={f.hint}>
            <textarea className={`${inputCls} min-h-[84px] font-mono text-xs`} value={text[f.key]}
              onChange={(e) => { setSaved(false); setText({ ...text, [f.key]: e.target.value }) }} />
          </Field>
        ))}
      </div>
    </Card>
  )
}

function RunCard({ room }: { room: Room }) {
  const job = room.job
  const total = job.total || 0
  const done = (job.done || 0) + (job.failed || 0)
  const pct = total ? Math.round((done / total) * 100) : job.phase === 'done' ? 100 : 0
  return (
    <Card title="Run" icon={Play}>
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="font-medium text-zinc-200">{PHASE_LABEL[job.phase || ''] || job.phase}</span>
          {job.phase === 'anonymizing' && total > 0 && <span className="text-zinc-500">{done} of {total}</span>}
          {job.phase === 'done' && total > 0 && <span className="text-zinc-500">{count(job.done || 0, 'file')} anonymized</span>}
          {job.phase === 'sweeping' && job.sweep_round && <span className="text-zinc-500">round {job.sweep_round}</span>}
          {(job.failed || 0) > 0 && <Pill tone="amber">{job.failed} failed</Pill>}
          {(job.sweep_changed || 0) > 0 && <Pill tone="sky">{count(job.sweep_changed || 0, 'file')} fixed by the sweep</Pill>}
          {(job.blocked || 0) > 0 && <Pill tone="red">{job.blocked} blocked</Pill>}
        </div>
        {job.phase !== 'done' && (
          <div className="h-2 overflow-hidden rounded-full bg-zinc-800">
            <div className="h-full rounded-full bg-violet-500 transition-all" style={{ width: `${pct}%` }} />
          </div>
        )}
        {job.message && <p className="text-sm text-zinc-400">{job.message}</p>}
        {job.phase === 'anonymizing' && room.trusted_model && (
          <p className="text-xs text-zinc-500">The trusted model ({room.trusted_model}) reads every file — a large room takes a while. You can close this; it keeps going.</p>
        )}
      </div>
    </Card>
  )
}

function FileRow({ f, phone, onOpen, onDownload, onRemove }: {
  f: RoomFile
  phone: boolean
  onOpen: () => void
  onDownload: () => void
  onRemove?: () => void
}) {
  const found = Object.values(f.entities).reduce((a, b) => a + b, 0)
  return (
    <li className="flex flex-wrap items-start gap-3 px-5 py-3">
      <div className="min-w-0 flex-1">
        <button type="button" onClick={onOpen} className="block max-w-full truncate text-left text-sm font-medium text-zinc-200 hover:underline" title={f.path}>
          {f.path}
        </button>
        {f.clean_path && (
          <div className="mt-0.5 truncate text-xs text-zinc-500" title={f.clean_path}>Claude sees: {f.clean_path}</div>
        )}
        {f.error && <div className={`mt-1 text-xs ${f.state === 'blocked' ? 'text-red-600 dark:text-red-400' : 'text-amber-700 dark:text-amber-300'}`}>{f.error}</div>}
        {f.warnings.length > 0 && !phone && (
          <ul className="mt-1 space-y-0.5 text-xs text-zinc-500">
            {f.warnings.slice(0, 3).map((w, i) => <li key={i}>· {w}</li>)}
          </ul>
        )}
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-1.5">
        <Pill tone={FILE_TONE[f.state]}>{FILE_LABEL[f.state]}</Pill>
        {f.released && <Pill tone="emerald" icon={Send}>Released</Pill>}
        {found > 0 && <Pill>{found} replaced</Pill>}
        <Button variant="ghost" icon={Eye} onClick={onOpen} aria-label="Compare" />
        {(f.state === 'done' || f.state === 'blocked') && <Button variant="ghost" icon={Download} onClick={onDownload} aria-label="Download the anonymized copy" />}
        {onRemove && <Button variant="ghost" icon={Trash2} onClick={onRemove} aria-label="Remove from the room" />}
      </div>
    </li>
  )
}

function CompareDialog({ roomId, f, onClose }: { roomId: string; f: RoomFile; onClose: () => void }) {
  const [text, setText] = useState<FileText | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    roomsApi.text(roomId, f.id, true).then(setText).catch((e) => setError(errorText(e)))
  }, [roomId, f.id])
  return (
    <Dialog title={f.path} icon={Eye} onClose={onClose} wide large>
      {error && <Notice tone="red">{error}</Notice>}
      {f.error && <Notice tone={f.state === 'blocked' ? 'red' : 'amber'}>{f.error}</Notice>}
      {f.warnings.length > 0 && (
        <Notice tone="amber"><ul className="space-y-0.5">{f.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></Notice>
      )}
      {!text ? <div className="text-sm text-zinc-500">Loading…</div> : (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="min-w-0">
            <div className="mb-2 text-xs font-medium text-zinc-400">Original (only you see this)</div>
            <div className="max-h-[60vh] overflow-auto whitespace-pre-wrap break-words rounded-xl border border-zinc-800 bg-zinc-950 p-4 text-sm text-zinc-300">
              {text.original || <span className="text-zinc-600">(no text)</span>}
            </div>
          </div>
          <div className="min-w-0">
            <div className="mb-2 text-xs font-medium text-zinc-400">What Claude reads{text.clean_path ? ` — ${text.clean_path}` : ''}</div>
            {text.clean
              ? <TokenText text={text.clean} className="max-h-[60vh] overflow-auto" />
              : <div className="rounded-xl border border-zinc-800 p-4 text-sm text-zinc-500">Not run yet.</div>}
          </div>
        </div>
      )}
    </Dialog>
  )
}

function RolesDialog({ room, onClose, onSaved }: { room: Room; onClose: () => void; onSaved: (r: Room) => void }) {
  const [entries, setEntries] = useState<VaultEntry[] | null>(null)
  const [roles, setRoles] = useState<Record<string, string>>(room.roles)
  const [query, setQuery] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    anonymizeApi.vault(room.vault_id).then((r) => setEntries(r.entries)).catch((e) => setError(errorText(e)))
  }, [room.vault_id])

  const shown = (entries || []).filter((e) => ['PERSON', 'ORG', 'DETAIL', 'ADDRESS'].includes(e.kind))
    .filter((e) => !query.trim() || `${e.token} ${e.original} ${roles[e.token] || ''}`.toLowerCase().includes(query.trim().toLowerCase()))

  const save = async () => {
    setBusy(true); setError('')
    try {
      onSaved(await roomsApi.update(room.id, { roles }))
      onClose()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog title="Roles" icon={Tag} onClose={onClose} wide large footer={
      <div className="flex justify-end gap-2">
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="primary" busy={busy} onClick={save}>Save roles</Button>
      </div>
    }>
      <p className="mb-3 text-sm text-zinc-500">
        Say what each one is — "Target", "Buyer", "CFO of the target", "largest customer" — never who. Claude sees the
        role next to the token. The real names are shown here for you only.
      </p>
      {error && <Notice tone="amber">{error}</Notice>}
      <input className={`${inputCls} mb-3`} placeholder="Find a name or token" value={query} onChange={(e) => setQuery(e.target.value)} />
      {entries === null ? <div className="text-sm text-zinc-500">Loading…</div> : (
        <ul className="max-h-[55vh] divide-y divide-zinc-800/80 overflow-y-auto rounded-xl border border-zinc-800">
          {shown.map((e) => (
            <li key={e.token} className="grid items-center gap-2 px-3 py-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <div className="min-w-0 text-sm">
                <TokenChip token={e.token} /> <span className="text-zinc-300">{e.original}</span>
              </div>
              <input className={inputCls} placeholder="Role (what, not who)" maxLength={120} value={roles[e.token] || ''}
                onChange={(ev) => setRoles({ ...roles, [e.token]: ev.target.value })} />
            </li>
          ))}
          {shown.length === 0 && <li className="px-3 py-6 text-center text-sm text-zinc-500">No people or companies found yet.</li>}
        </ul>
      )}
    </Dialog>
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
