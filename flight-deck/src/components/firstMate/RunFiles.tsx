import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Braces, Download, Eye, File, FileCode, FileSpreadsheet, FileText, Globe, Image, Loader2, NotebookPen, Table2,
} from 'lucide-react'
import { FileViewer } from '../agents/FileViewer'
import { FilePreview } from '../library/FilePreview'
import { CsvTable } from '../library/CsvTable'
import { RunDatastorePanel } from '../basna/RunArtifacts'
import { getFileTypeGroup, isViewable, type AgentFile } from '../../services/fileTransfer'
import { vfsApi, type FolderHit, type FolderRef } from '../../services/vfsApi'
import { isUntrustedFile } from '../../utils/untrustedContent'
import { LibraryPick } from '../library/LibraryPick'

// A run's folder the way the Library shows files: the newest first, a preview of
// the one picked, and the Library's viewer to open it (text files can be changed
// and saved back). A CSV or TSV opens as a table to sort, search and edit. The
// workers' own notes (progress/, notes/) are listed apart from the results, and
// the data tables the run made follow.

const WORK_DIRS = ['progress/', 'notes/']
// The extra data it was given is listed on its own (Extra data), not as its files.
const GIVEN = 'extra/'
const POLL_MS = 10_000

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

export function RunFiles({ project, live }: { project: string; live: boolean }) {
  const ref: FolderRef = useMemo(() => ({ project, owner: '' }), [project])
  const [files, setFiles] = useState<FolderHit[] | null>(null)
  const [error, setError] = useState('')
  const [picked, setPicked] = useState('')
  const [viewing, setViewing] = useState<{ file: FolderHit; edit: boolean } | null>(null)
  const [table, setTable] = useState<FolderHit | null>(null)

  const load = useCallback(async () => {
    try {
      const got = await vfsApi.search('', ref, 500)
      setFiles(got.files)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't list its files")
    }
  }, [ref])

  useEffect(() => {
    void load()
    if (!live) return
    const t = setInterval(load, POLL_MS)
    return () => clearInterval(t)
  }, [load, live])

  const results = (files ?? []).filter((f) => !f.path.startsWith(GIVEN) && !WORK_DIRS.some((d) => f.path.startsWith(d)))
  const notes = (files ?? []).filter((f) => WORK_DIRS.some((d) => f.path.startsWith(d)))
  const current = (files ?? []).find((f) => f.path === picked) ?? null

  /** A table opens as one unless `asText`; a PDF in a tab; the rest in the viewer. */
  const open = (f: FolderHit, asText = false) => {
    const group = getFileTypeGroup(asAgentFile(f.name))
    if (isTable(f.name) && !asText) setTable(f)
    else if (group === 'pdf') window.open(vfsApi.viewUrl(ref, f.path, f.mtime), '_blank', 'noopener')
    else setViewing({ file: f, edit: false })
  }

  const row = (f: FolderHit) => (
    <li key={f.path}>
      <button onClick={() => setPicked(f.path === picked ? '' : f.path)} onDoubleClick={() => open(f)}
        aria-pressed={f.path === picked}
        className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors ${
          f.path === picked ? 'bg-violet-500/10' : 'hover:bg-[var(--fd-hover)]'}`}>
        <Icon name={f.name} />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] text-zinc-200" title={f.path}>{f.name}</span>
          {f.path.includes('/') && <span className="block truncate text-[11px] text-zinc-500">{f.path.slice(0, f.path.lastIndexOf('/'))}</span>}
        </span>
        <span className="shrink-0 text-right text-[11px] tabular-nums text-zinc-500">{sizeText(f.size)}<br />{ago(f.mtime)}</span>
      </button>
    </li>
  )

  if (error && !files) return <p className="px-1 text-xs text-red-600 dark:text-red-300">{error}</p>
  if (!files) return <p className="flex items-center gap-1.5 px-1 text-xs text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" /> Listing its files…</p>

  const canView = !!current && isViewable(asAgentFile(current.name))
  return (
    <div className="space-y-3">
      <div className={current ? 'grid gap-3 @4xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] @4xl:items-start' : ''}>
      {results.length === 0 ? <p className="px-1 text-xs text-zinc-500">No results in its folder yet.</p> : (
        <ul className="max-h-72 space-y-0.5 overflow-y-auto pr-1 @4xl:max-h-[28rem]">{results.map(row)}</ul>
      )}
      {current && (
        <div className="rounded-xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] p-3 @4xl:sticky @4xl:top-0">
          <FilePreview key={`${current.path}:${current.mtime}`} group={getFileTypeGroup(asAgentFile(current.name))}
            extension={extOf(current.name)} size={current.size}
            url={vfsApi.viewUrl(ref, current.path, current.mtime)}
            freshUrl={() => vfsApi.viewUrl(ref, current.path, current.mtime)}
            thumbUrl={vfsApi.thumbUrl(ref, current.path, current.mtime)}
            cacheKey={`${project}/${current.path}:${current.mtime}`}
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
            {!WORK_DIRS.some((d) => current.path.startsWith(d)) && <LibraryPick folder={project} path={current.path} onError={setError} />}
          </div>
        </div>
      )}
      </div>
      {notes.length > 0 && (
        <div>
          <p className="mb-1 flex items-center gap-1.5 px-1 text-[11px] font-medium text-zinc-500"><NotebookPen className="h-3 w-3" />Its working notes</p>
          <ul className="grid max-h-56 gap-0.5 overflow-y-auto pr-1 @4xl:grid-cols-2">{notes.map(row)}</ul>
        </div>
      )}
      <RunDatastorePanel project={project} live={live} variant="tab" hideWhenEmpty />
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
