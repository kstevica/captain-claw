import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, Archive, CheckCircle2, ChevronRight, CircleDashed, Clock, Code2, Compass, Download, Eye, FolderArchive, Loader2, MessageSquareWarning, RefreshCw, Table2, Trophy, Wrench } from 'lucide-react'
import { getFactoryResults, durationText, whenText, RUN_LABELS, type ResultFile, type RunResults, type RunTiming } from '../../services/firstMate'
import { vfsApi, type FolderRef } from '../../services/vfsApi'
import { getFileTypeGroup, isViewable, type AgentFile } from '../../services/fileTransfer'
import { sizeText } from '../../utils/format'
import { isSheetExt } from '../../utils/xlsx'
import { FileGlyph } from '../context/AddData'
import { FileViewer } from '../agents/FileViewer'
import { CsvTable } from '../library/CsvTable'
import { LibraryPick } from '../library/LibraryPick'
import { Pill, errText, type PillTone } from './ui'

// A factory's results: each run's files, the result first (what its checks name, or a
// file called a result), then the rest at the top of its folder, then the working files
// (subfolders, scripts, temporary files). A table opens as a table; any file can go in
// the File library, which holds only what you add.

const RUN_TONE: Record<string, PillTone> = { queued: 'zinc', running: 'violet', done: 'green', failed: 'red', cancelled: 'zinc' }
const extOf = (name: string) => (name.includes('.') ? name.slice(name.lastIndexOf('.')).toLowerCase() : '')
/** A CSV, TSV or Excel workbook: opens as a table. */
const isTable = (name: string) => ['.csv', '.tsv'].includes(extOf(name)) || isSheetExt(extOf(name))
const rowsText = (n: number) => `${n.toLocaleString()} row${n === 1 ? '' : 's'}`

function asFile(project: string, f: ResultFile): AgentFile {
  const group = getFileTypeGroup({ extension: extOf(f.name), filename: f.name } as AgentFile)
  return {
    logical: `vfs:${project}/${f.path}`, physical: `${project}/${f.path}`, filename: f.name, extension: extOf(f.name),
    exists: true, size: f.size, modified: f.mtime, mime_type: '',
    is_text: ['markdown', 'code', 'data', 'text', 'html'].includes(group), source: 'workspace',
  }
}

function agoText(mtime: number): string {
  const s = Math.max(0, Date.now() / 1000 - mtime)
  return s < 90 ? 'just now' : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129_600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86_400)} d ago`
}

/** "Took 23 h 4 min · agents worked 9 h 12 min · waiting for you 11 h 30 min", as chips. */
export function TimingChips({ timing, going = false, className = '' }: { timing?: RunTiming | null; going?: boolean; className?: string }) {
  if (!timing || !timing.total_s) return null
  const chip = 'inline-flex items-center gap-1 rounded-full bg-zinc-800/70 px-2 py-0.5 text-[11px] text-zinc-400'
  const sideBySide = timing.agent_s > timing.working_s * 1.2
  return (
    <span className={`flex flex-wrap items-center gap-1.5 ${className}`}>
      <span className={chip} title="From when it started to when it ended, everything included">
        <Clock className="h-3 w-3" />{going ? 'So far' : 'Took'} {durationText(timing.total_s)}
      </span>
      <span className={chip} title={sideBySide ? `${durationText(timing.agent_s)} of agent time in all: steps ran side by side` : 'While at least one agent was working on a step'}>
        Agents worked {durationText(timing.working_s)}{sideBySide ? ` (${durationText(timing.agent_s)} side by side)` : ''}
      </span>
      {timing.waiting_s >= 60 && (
        <span className={`${chip} !bg-amber-500/10 !text-amber-700 dark:!text-amber-300`} title="Stopped, waiting for you to press Continue">
          Waiting for you {durationText(timing.waiting_s)}
        </span>
      )}
      {timing.other_s >= 60 && (
        <span className={chip} title="Captain planning and checking the work, and any restart of the deck">
          Planning, checks, restarts {durationText(timing.other_s)}
        </span>
      )}
    </span>
  )
}

export function FactoryResults({ factoryId, onNotice }: { factoryId: string; onNotice: (text: string) => void }) {
  const [runs, setRuns] = useState<RunResults[] | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const [viewing, setViewing] = useState<{ project: string; file: ResultFile } | null>(null)
  const [table, setTable] = useState<{ project: string; file: ResultFile } | null>(null)

  const load = useCallback(async () => {
    setBusy(true)
    try {
      setRuns(await getFactoryResults(factoryId))
      setError('')
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(false)
    }
  }, [factoryId])
  useEffect(() => { void load() }, [load])

  const openFile = (project: string, f: ResultFile) => {
    const ref: FolderRef = { project, owner: '' }
    if (isTable(f.name)) setTable({ project, file: f })
    else if (getFileTypeGroup(asFile(project, f)) === 'pdf') window.open(vfsApi.viewUrl(ref, f.path, f.mtime), '_blank', 'noopener')
    else setViewing({ project, file: f })
  }
  const canOpen = (project: string, f: ResultFile) => isTable(f.name) || isViewable(asFile(project, f))

  if (!runs) {
    return error ? <p className="text-xs text-red-700 dark:text-red-300">{error}</p>
      : <p className="flex items-center gap-1.5 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" />Gathering its results…</p>
  }
  if (!runs.length) {
    return (
      <div className="flex flex-col items-center gap-2 rounded-2xl border border-dashed border-zinc-800 px-6 py-12 text-center">
        <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-zinc-800/70 text-zinc-500"><Trophy className="h-5 w-5" /></span>
        <p className="text-sm font-medium text-zinc-300">No results yet</p>
        <p className="max-w-sm text-xs text-zinc-500">What its runs make shows up here, the result first.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <p className="text-xs text-zinc-500">The result first; a table opens as a table. Only what you add goes in the File library.</p>
        <button onClick={() => void load()} disabled={busy} title="Look again" aria-label="Look again"
          className="ml-auto flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
          <RefreshCw className={`h-3.5 w-3.5 ${busy ? 'animate-spin' : ''}`} />
        </button>
      </div>
      {error && <p role="alert" className="text-xs text-red-700 dark:text-red-300">{error}</p>}
      {runs.map((r, i) => {
        const expanded = open[r.run_id] ?? i === 0
        const ref: FolderRef = { project: r.project, owner: '' }
        const going = r.status === 'running' || r.status === 'queued'
        return (
          <section key={r.run_id} className={`overflow-hidden rounded-2xl border bg-[var(--fd-surface-raised)] ${i === 0 ? 'border-violet-500/30' : 'border-[var(--fd-line)]'}`}>
            <div className="flex flex-wrap items-center gap-2 px-4 py-3">
              <button onClick={() => setOpen((o) => ({ ...o, [r.run_id]: !expanded }))} aria-expanded={expanded}
                aria-label={expanded ? 'Fold this run' : 'Open this run'}
                className="-ml-1 flex h-7 w-7 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                <ChevronRight className={`h-4 w-4 transition-transform ${expanded ? 'rotate-90' : ''}`} />
              </button>
              <Pill tone={RUN_TONE[r.status] ?? 'zinc'}>{RUN_LABELS[r.status] ?? r.status}</Pill>
              {r.kind === 'rehearsal' && <Pill tone="zinc">Rehearsal</Pill>}
              <span className="text-xs text-zinc-500">{whenText(r.created_at)} · {r.total} file{r.total === 1 ? '' : 's'} · {sizeText(r.bytes)}</span>
              <a href={vfsApi.zipUrl(ref)} className="ml-auto inline-flex h-7 items-center gap-1 rounded-full px-2.5 text-[11px] font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100"
                title="Every file of this run, as one zip">
                <FolderArchive className="h-3.5 w-3.5" />Download all
              </a>
              <TimingChips timing={r.timing} going={going} className="w-full pl-8" />
            </div>
            {!expanded && r.main[0] && (
              <button onClick={() => openFile(r.project, r.main[0])} className="flex w-full items-center gap-2 border-t border-[var(--fd-line)] px-4 py-2 text-left text-xs text-zinc-400 hover:bg-zinc-900/40">
                <FileGlyph name={r.main[0].name} className="h-3.5 w-3.5" />
                <span className="truncate font-medium text-zinc-300">{r.main[0].name}</span>
                {r.main[0].rows !== null && <span>· {rowsText(r.main[0].rows)}</span>}
              </button>
            )}
            {expanded && (
              <div className="space-y-3 border-t border-[var(--fd-line)] px-4 py-3">
                <Checked r={r} />
                {r.code ? (
                  <div className="flex flex-wrap items-center gap-3 rounded-xl bg-violet-500/[0.06] p-3 ring-1 ring-violet-500/20">
                    <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--fd-surface-raised)] shadow-sm"><Code2 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" /></span>
                    <div className="min-w-[10rem] flex-1">
                      <p className="text-sm font-semibold text-zinc-100">A code project</p>
                      <p className="text-[11px] text-zinc-500">{r.total} files · {sizeText(r.bytes)}</p>
                    </div>
                    <a href={vfsApi.zipUrl(ref)} className="inline-flex h-8 items-center gap-1.5 rounded-full bg-violet-600 px-3 text-xs font-semibold text-white transition-colors hover:bg-violet-500">
                      <FolderArchive className="h-3.5 w-3.5" />Download it
                    </a>
                    <LibraryPick folder={r.project} onError={onNotice} />
                  </div>
                ) : null}
                {r.code && <FileGroup title="Its files" icon={Code2} project={r.project} files={r.main} onOpen={openFile} canOpen={canOpen} onNotice={onNotice} />}
                <ul className={`space-y-2 ${r.code ? 'hidden' : ''}`}>
                  {r.main.map((f) => (
                    <li key={f.path} className="flex flex-wrap items-center gap-3 rounded-xl bg-violet-500/[0.06] p-3 ring-1 ring-violet-500/20">
                      <button onClick={() => openFile(r.project, f)} disabled={!canOpen(r.project, f)}
                        className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[var(--fd-surface-raised)] shadow-sm transition-transform enabled:hover:scale-105">
                        <FileGlyph name={f.name} className="h-5 w-5" />
                      </button>
                      <div className="min-w-[10rem] flex-1">
                        <p className="break-all text-sm font-semibold text-zinc-100" title={f.path}>{f.name}</p>
                        <p className="text-[11px] text-zinc-500">
                          {[f.rows !== null ? rowsText(f.rows) : '', sizeText(f.size), agoText(f.mtime)].filter(Boolean).join(' · ')}
                        </p>
                      </div>
                      <div className="flex flex-wrap items-center gap-1.5">
                        {canOpen(r.project, f) && (
                          <button onClick={() => openFile(r.project, f)}
                            className="inline-flex h-8 items-center gap-1.5 rounded-full bg-violet-600 px-3 text-xs font-semibold text-white transition-colors hover:bg-violet-500">
                            {isTable(f.name) ? <Table2 className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}{isTable(f.name) ? 'Open table' : 'Open'}
                          </button>
                        )}
                        <a href={vfsApi.downloadUrl(ref, f.path)} className="inline-flex h-8 items-center gap-1.5 rounded-full border border-[var(--fd-line)] px-3 text-xs font-medium text-zinc-300 transition-colors hover:text-zinc-100">
                          <Download className="h-3.5 w-3.5" />Download
                        </a>
                        <LibraryPick folder={r.project} path={f.path} onError={onNotice} />
                      </div>
                    </li>
                  ))}
                </ul>
                <FileGroup title="Came close" hint="Items that nearly made the list, with why each didn't: not on the list" icon={CircleDashed}
                  project={r.project} files={r.close ?? []} onOpen={openFile} canOpen={canOpen} onNotice={onNotice} startOpen />
                <FileGroup title="Other files" icon={Archive} project={r.project} files={r.other} onOpen={openFile} canOpen={canOpen} onNotice={onNotice} />
                <FileGroup title="Working files" hint="Scripts, temporary files and subfolders its workers used" icon={Wrench}
                  project={r.project} files={r.work} onOpen={openFile} canOpen={canOpen} onNotice={onNotice} />
              </div>
            )}
          </section>
        )
      })}
      {viewing && (
        <div data-above-drawer className="contents">
          <FileViewer file={asFile(viewing.project, viewing.file)} readOnly
            urls={{ view: vfsApi.viewUrl({ project: viewing.project, owner: '' }, viewing.file.path, viewing.file.mtime),
              download: vfsApi.downloadUrl({ project: viewing.project, owner: '' }, viewing.file.path) }}
            onClose={() => setViewing(null)} />
        </div>
      )}
      {table && (
        <CsvTable fileRef={{ project: table.project, owner: '' }} path={table.file.path} name={table.file.name}
          extension={extOf(table.file.name)} onClose={() => setTable(null)} onSaved={() => void load()} />
      )}
    </div>
  )
}

function FileGroup({ title, hint, icon: Icon, project, files, onOpen, canOpen, onNotice, startOpen = false }: {
  title: string
  hint?: string
  startOpen?: boolean
  icon: typeof Archive
  project: string
  files: ResultFile[]
  onOpen: (project: string, f: ResultFile) => void
  canOpen: (project: string, f: ResultFile) => boolean
  onNotice: (text: string) => void
}) {
  const [open, setOpen] = useState(startOpen)
  if (!files.length) return null
  const ref: FolderRef = { project, owner: '' }
  const btn = 'flex h-8 w-8 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200'
  return (
    <div>
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open} title={hint}
        className="flex w-full items-center gap-2 rounded-lg px-1 py-1 text-left text-xs font-medium text-zinc-400 transition-colors hover:text-zinc-200">
        <ChevronRight className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-90' : ''}`} />
        <Icon className="h-3.5 w-3.5" />{title} · {files.length}
      </button>
      {open && (
        <ul className="mt-1 divide-y divide-[var(--fd-line)] overflow-hidden rounded-xl border border-[var(--fd-line)]">
          {files.map((f) => (
            <li key={f.path} className="flex items-center gap-2.5 px-3 py-1.5">
              <FileGlyph name={f.name} />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13px] text-zinc-200" title={f.path}>{f.name}</span>
                <span className="block truncate text-[11px] text-zinc-500">
                  {[f.path.includes('/') ? f.path.slice(0, f.path.lastIndexOf('/')) : '', f.rows !== null ? rowsText(f.rows) : '', sizeText(f.size), agoText(f.mtime)].filter(Boolean).join(' · ')}
                </span>
              </span>
              {canOpen(project, f) && <button onClick={() => onOpen(project, f)} className={btn} title="Open" aria-label={`Open ${f.name}`}><Eye className="h-3.5 w-3.5" /></button>}
              <a href={vfsApi.downloadUrl(ref, f.path)} className={btn} title="Download" aria-label={`Download ${f.name}`}><Download className="h-3.5 w-3.5" /></a>
              <LibraryPick folder={project} path={f.path} compact onError={onNotice} />
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** What Captain found reading the files, the done-checks it ran that failed, and a reviewer who said no. */
function Checked({ r }: { r: RunResults }) {
  const flags = r.checked?.flags ?? []
  const failed = (r.checks ?? []).filter((c) => !c.passed)
  const fixes = r.corrections ?? []
  if (!r.checked && !failed.length && !r.dissent && !fixes.length) return null
  const clean = !flags.length && !failed.length && !r.dissent
  return (
    <div className={`rounded-xl px-3 py-2.5 text-xs ${clean ? 'bg-emerald-500/[0.07] text-emerald-800 dark:text-emerald-300' : 'bg-amber-500/[0.08] text-amber-900 dark:text-amber-200'}`}>
      <p className="flex items-center gap-1.5 font-semibold">
        {clean ? <CheckCircle2 className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
        {clean ? 'Captain read the result: nothing stood out' : 'Worth a look before you use it'}
      </p>
      {(failed.length > 0 || flags.length > 0) && (
        <ul className="mt-1.5 space-y-0.5 pl-5">
          {failed.map((c) => <li key={`c:${c.text}`} className="list-disc">Failed check: {c.text}</li>)}
          {flags.map((f) => <li key={f} className="list-disc">{f}</li>)}
        </ul>
      )}
      {r.dissent && (
        <p className="mt-1.5 flex items-start gap-1.5"><MessageSquareWarning className="mt-0.5 h-3.5 w-3.5 shrink-0" />{r.dissent}</p>
      )}
      {fixes.length > 0 && (
        <details className="mt-1.5">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 font-medium">
            <Compass className="h-3.5 w-3.5" />Captain corrected its course {fixes.length} time{fixes.length === 1 ? '' : 's'} during the run
          </summary>
          <ul className="mt-1 space-y-0.5 pl-5">{fixes.map((c) => <li key={c} className="list-disc">{c}</li>)}</ul>
        </details>
      )}
    </div>
  )
}
