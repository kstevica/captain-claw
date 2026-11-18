import { useState } from 'react'
import {
  Braces, BookOpen, Download, Eye, File, FileCode, FileSpreadsheet, FileText, Globe, Image, Loader2, MousePointerClick,
  NotebookPen, Share2, Table2, type LucideIcon,
} from 'lucide-react'
import { FileViewer } from '../agents/FileViewer'
import { FilePreview } from '../library/FilePreview'
import { CsvTable } from '../library/CsvTable'
import { RunDatastorePanel } from '../basna/RunArtifacts'
import { getFileTypeGroup, isViewable, type AgentFile } from '../../services/fileTransfer'
import { vfsApi, type FolderHit } from '../../services/vfsApi'
import { isUntrustedFile } from '../../utils/untrustedContent'
import { LibraryPick } from '../library/LibraryPick'
import { NOTE_DIRS, type RunFolder } from './useRunFolder'

// A run's folder the way the Library shows files, on two of the run's tabs: Files (what
// it made, then the data tables it kept) and Notes (the workers' own: where each step
// got to, what every step shares, what they kept as they read). Pick one to preview it
// beside the list; open it in the Library's viewer (text can be changed and saved
// back). A CSV or TSV opens as a table to sort, search and edit.

function extOf(name: string): string {
  const i = name.lastIndexOf('.')
  return i > 0 ? name.slice(i).toLowerCase() : ''
}

const asAgentFile = (name: string) => ({ extension: extOf(name), filename: name }) as AgentFile
const isTable = (name: string) => ['.csv', '.tsv'].includes(extOf(name))

function toAgentFile(f: FolderHit): AgentFile {
  const group = getFileTypeGroup(asAgentFile(f.name))
  return {
    logical: `vfs:${f.project}/${f.path}`, physical: `${f.project}/${f.path}`, filename: f.name, extension: extOf(f.name),
    exists: true, size: f.size, modified: f.mtime, mime_type: '',
    is_text: ['markdown', 'code', 'data', 'text', 'html'].includes(group), source: 'workspace',
  }
}

function Icon({ name }: { name: string }) {
  const group = getFileTypeGroup(asAgentFile(name))
  const cls = 'h-4 w-4 shrink-0'
  if (isTable(name)) return <FileSpreadsheet className={`${cls} text-cyan-600 dark:text-cyan-400`} />
  if (group === 'image') return <Image className={`${cls} text-blue-600 dark:text-blue-400`} />
  if (group === 'markdown' || group === 'pdf') return <FileText className={`${cls} text-violet-600 dark:text-violet-400`} />
  if (group === 'html') return <Globe className={`${cls} text-orange-600 dark:text-orange-400`} />
  if (group === 'data') return <Braces className={`${cls} text-sky-600 dark:text-sky-400`} />
  if (group === 'code') return <FileCode className={`${cls} text-emerald-600 dark:text-emerald-400`} />
  return <File className={`${cls} text-zinc-500`} />
}

function sizeText(n: number): string {
  return n >= 1_048_576 ? `${(n / 1_048_576).toFixed(1)} MB` : n >= 1024 ? `${Math.round(n / 1024)} KB` : `${n} B`
}

function ago(mtime: number): string {
  const s = Math.max(0, Date.now() / 1000 - mtime)
  return s < 90 ? 'just now' : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129_600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86_400)} d ago`
}

interface Group { key: string; label?: string; hint?: string; icon?: LucideIcon; files: FolderHit[]; prefix?: string }

/** The list, grouped, with the picked file's preview beside it when there's room. */
function FolderPane({ folder, groups, empty, canPick }: {
  folder: RunFolder
  groups: Group[]
  empty: string
  /** Results may go in the File library; the workers' notes don't. */
  canPick: boolean
}) {
  const { ref, files, error, setError, load } = folder
  const [picked, setPicked] = useState('')
  const [viewing, setViewing] = useState<{ file: FolderHit; edit: boolean } | null>(null)
  const [table, setTable] = useState<FolderHit | null>(null)

  if (error && !files) return <p className="px-1 text-xs text-red-600 dark:text-red-300">{error}</p>
  if (!files) return <p className="flex items-center gap-1.5 px-1 text-xs text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" /> Listing its files…</p>

  const listed = groups.flatMap((g) => g.files)
  const current = listed.find((f) => f.path === picked) ?? null

  /** A table opens as one unless `asText`; a PDF in a tab; the rest in the viewer. */
  const open = (f: FolderHit, asText = false) => {
    const group = getFileTypeGroup(asAgentFile(f.name))
    if (isTable(f.name) && !asText) setTable(f)
    else if (group === 'pdf') window.open(vfsApi.viewUrl(ref, f.path, f.mtime), '_blank', 'noopener')
    else setViewing({ file: f, edit: false })
  }

  const row = (f: FolderHit, prefix = '') => {
    const rest = f.path.slice(prefix.length)
    const dir = rest.includes('/') ? rest.slice(0, rest.lastIndexOf('/')) : ''
    return (
      <li key={f.path}>
        <button onClick={() => setPicked(f.path === picked ? '' : f.path)} onDoubleClick={() => open(f)}
          aria-pressed={f.path === picked} title={`${f.path} · double-click to open`}
          className={`flex w-full items-center gap-2.5 rounded-xl px-2.5 py-2 text-left transition-colors ${
            f.path === picked ? 'bg-violet-500/10 ring-1 ring-violet-500/25' : 'hover:bg-[var(--fd-hover)]'}`}>
          <Icon name={f.name} />
          <span className="min-w-0 flex-1">
            <span className="block truncate text-[13px] text-zinc-200">{f.name}</span>
            {dir && <span className="block truncate text-[11px] text-zinc-500">{dir}</span>}
          </span>
          <span className="shrink-0 text-right text-[11px] tabular-nums leading-tight text-zinc-500">{sizeText(f.size)}<br />{ago(f.mtime)}</span>
        </button>
      </li>
    )
  }

  const canView = !!current && isViewable(asAgentFile(current.name))
  return (
    <div className="grid gap-3 @4xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] @4xl:items-start">
      <div className="max-h-[32rem] space-y-3 overflow-y-auto pr-1">
        {listed.length === 0 ? <p className="px-1 text-xs text-zinc-500">{empty}</p> : groups.filter((g) => g.files.length).map((g) => (
          <section key={g.key}>
            {g.label && (
              <p className="mb-1 flex items-center gap-1.5 px-1 text-[11px] font-semibold text-zinc-400">
                {g.icon && <g.icon className="h-3 w-3" />}{g.label}
                <span className="font-normal text-zinc-500">· {g.files.length}{g.hint ? ` · ${g.hint}` : ''}</span>
              </p>
            )}
            <ul className="space-y-0.5">{g.files.map((f) => row(f, g.prefix))}</ul>
          </section>
        ))}
      </div>
      {current ? (
        <div className="rounded-xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] p-3 @4xl:sticky @4xl:top-0">
          <FilePreview key={`${current.path}:${current.mtime}`} group={getFileTypeGroup(asAgentFile(current.name))}
            extension={extOf(current.name)} size={current.size}
            url={vfsApi.viewUrl(ref, current.path, current.mtime)}
            freshUrl={() => vfsApi.viewUrl(ref, current.path, current.mtime)}
            thumbUrl={vfsApi.thumbUrl(ref, current.path, current.mtime)}
            cacheKey={`${ref.project}/${current.path}:${current.mtime}`}
            untrusted={isUntrustedFile(toAgentFile(current))} phone={false}
            viewerOpen={!!viewing || !!table} canOpen={canView} onOpen={() => open(current)} />
          <div className="mt-2 flex flex-wrap gap-2">
            {isTable(current.name) && (
              <button onClick={() => setTable(current)} className="inline-flex h-8 items-center gap-1.5 rounded-full bg-violet-600 px-3 text-xs font-semibold text-white hover:bg-violet-500">
                <Table2 className="h-3.5 w-3.5" />Open as a table
              </button>
            )}
            {canView && (
              <button onClick={() => open(current, isTable(current.name))} className="fd-btn-quiet inline-flex h-8 items-center gap-1.5 px-3 text-xs">
                <Eye className="h-3.5 w-3.5" />{isTable(current.name) ? 'Open as text' : 'Open'}
              </button>
            )}
            <a href={vfsApi.downloadUrl(ref, current.path)} className="fd-btn-quiet inline-flex h-8 items-center gap-1.5 px-3 text-xs">
              <Download className="h-3.5 w-3.5" />Download
            </a>
            {canPick && <LibraryPick folder={ref.project} path={current.path} onError={setError} />}
          </div>
        </div>
      ) : listed.length > 0 && (
        <div className="hidden h-40 flex-col items-center justify-center gap-1.5 rounded-xl border border-dashed border-[var(--fd-line)] text-xs text-zinc-500 @4xl:flex">
          <MousePointerClick className="h-4 w-4" />Pick a file to see it here; double-click opens it.
        </div>
      )}
      {viewing && (
        <div data-above-drawer className="contents">
        <FileViewer key={viewing.edit ? `edit:${viewing.file.path}` : `view:${viewing.file.path}`}
          file={toAgentFile(viewing.file)}
          urls={{ view: vfsApi.viewUrl(ref, viewing.file.path, viewing.file.mtime), download: vfsApi.downloadUrl(ref, viewing.file.path) }}
          onSave={async (text) => { await vfsApi.write(ref, viewing.file.path, text); void load() }}
          startInEdit={viewing.edit} onClose={() => setViewing(null)} />
        </div>
      )}
      {table && (
        <CsvTable fileRef={ref} path={table.path} name={table.name} extension={extOf(table.name)}
          onClose={() => setTable(null)} onSaved={() => void load()} />
      )}
    </div>
  )
}

/** The run's Files tab: what it made, newest first, then the data tables it kept. */
export function RunFiles({ folder, live }: { folder: RunFolder; live: boolean }) {
  return (
    <div className="space-y-4">
      <FolderPane folder={folder} canPick empty="No results in its folder yet."
        groups={[{ key: 'results', files: folder.results }]} />
      <RunDatastorePanel project={folder.ref.project} live={live} variant="tab" hideWhenEmpty />
    </div>
  )
}

const byName = (a: FolderHit, b: FolderHit) => a.path.localeCompare(b.path, undefined, { numeric: true })

/** The run's Notes tab: the workers' own, by kind. Progress notes in step order. */
export function RunNotes({ folder }: { folder: RunFolder }) {
  const of = (dir: string) => folder.notes.filter((f) => f.path.startsWith(dir))
  return (
    <FolderPane folder={folder} canPick={false} empty="No notes yet: each step's worker keeps one as it goes."
      groups={[
        { key: 'progress', label: 'Progress notes', hint: 'where each step got to', icon: NotebookPen,
          files: of(NOTE_DIRS.progress).sort(byName), prefix: NOTE_DIRS.progress },
        { key: 'shared', label: 'Shared by every step', hint: 'conventions, corrections, tracking', icon: Share2,
          files: of(NOTE_DIRS.shared), prefix: NOTE_DIRS.shared },
        { key: 'notes', label: 'Research notes', hint: 'what workers kept as they read', icon: BookOpen,
          files: of(NOTE_DIRS.notes), prefix: NOTE_DIRS.notes },
      ]} />
  )
}
