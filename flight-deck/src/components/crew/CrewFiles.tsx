import { useCallback, useEffect, useState } from 'react'
import { ChevronRight, Download, ExternalLink, FileCode2, FileText, FolderOpen, Globe, Link2, Loader2, Play } from 'lucide-react'
import { crewApi, type CrewFile, type CrewFiles as Files, type CrewLink, type CrewTask } from '../../services/crewApi'
import { hostingApi } from '../../services/hostingApi'
import { CopyChip, CrewMarkdown, ErrorLine, LIVE, errorOf, fullUrl, hasPage, primaryBtn, quietBtn } from './crewParts'

// ── What it made ──
//
// A finished task's files sit in the owner's VFS, where their agents read them too.
// From here a person can look through them and turn a page into a link.

function Preview({ taskId, path }: { taskId: string; path: string }) {
  const [file, setFile] = useState<CrewFile | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    setFile(null); setError('')
    crewApi.file(taskId, path).then(setFile).catch((e) => setError(errorOf(e)))
  }, [taskId, path])
  if (error) return <ErrorLine text={error} />
  if (!file) return <div className="flex justify-center p-6 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>
  if (file.binary) return <p className="p-4 text-sm text-zinc-500">This one can't be shown here.</p>
  return (
    <>
      {file.truncated && <p className="px-3 pt-2 text-xs text-zinc-500">Only its start is shown: it's a long file.</p>}
      {/\.(md|markdown)$/i.test(path)
        ? <div className="p-4"><CrewMarkdown text={file.text} /></div>
        : <pre className="overflow-auto p-3 font-mono text-[11.5px] leading-[1.6] text-zinc-300">{file.text}</pre>}
    </>
  )
}

function LinkCard({ link, busy, onStart }: { link: CrewLink; busy: boolean; onStart: () => void }) {
  if (link.stopped) {
    return (
      <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-zinc-800 text-zinc-400"><Globe className="h-[18px] w-[18px]" /></span>
        <p className="min-w-0 flex-1 basis-48 text-sm text-zinc-400">This page is stopped. It keeps its address, and comes back when you start it.</p>
        <button disabled={busy} onClick={onStart} className={primaryBtn}><Play className="h-4 w-4" /> Start</button>
      </div>
    )
  }
  return (
    <div className="space-y-2 rounded-2xl border border-violet-500/30 bg-violet-500/[0.06] p-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/15 text-violet-600 dark:text-violet-300"><Globe className="h-[18px] w-[18px]" /></span>
        <span className="min-w-0 flex-1 basis-48"><CopyChip text={fullUrl(link.url)} /></span>
        <a href={link.url} target="_blank" rel="noreferrer" className={quietBtn}><ExternalLink className="h-4 w-4" /> Open</a>
      </div>
      <p className="text-xs text-zinc-500">Anyone with this address can open it. It follows the task's files, without the repository's history. Stop or remove it under Live sites.</p>
    </div>
  )
}

export function CrewFiles({ task }: { task: CrewTask }) {
  const [files, setFiles] = useState<Files | null>(null)
  const [path, setPath] = useState('')
  const [link, setLink] = useState<CrewLink | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const live = LIVE.includes(task.status)

  const load = useCallback(async (whole = false) => {
    setBusy(true); setError('')
    try {
      const got = await crewApi.files(task.id, whole)
      setFiles(got)
      setLink(got.link ?? null)
    } catch (e) {
      setError(errorOf(e))
    }
    setBusy(false)
  }, [task.id])

  useEffect(() => {
    if (!live) load()
  }, [live, load, task.results?.head])

  if (live) return <p className="p-5 text-sm text-zinc-500">Its files come here when it finishes.</p>

  const makeLink = async () => {
    setBusy(true); setError('')
    try {
      setLink(await crewApi.link(task.id))
      await load()
    } catch (e) {
      setError(errorOf(e))
    }
    setBusy(false)
  }

  const startLink = async () => {
    if (!link) return
    setBusy(true); setError('')
    try {
      await hostingApi.start(link.name)
      await load()
    } catch (e) {
      setError(errorOf(e))
    }
    setBusy(false)
  }

  const names = files?.files ?? []
  const changed = new Set(files?.changed ?? [])
  const canLink = hasPage(files ?? undefined)

  return (
    <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-5">
      <ErrorLine text={error} />
      {!files && !error && <div className="flex justify-center p-6 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>}
      {files && (
        <>
          {link && <LinkCard link={link} busy={busy} onStart={startLink} />}
          <div className="space-y-2 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
            {files.files_path ? (
              <p className="flex items-center gap-2 text-sm text-zinc-200">
                <FolderOpen className="h-4 w-4 shrink-0 text-zinc-500" />
                {files.count} file{files.count === 1 ? '' : 's'}, {files.whole ? 'the whole project' : 'only what it changed'}
              </p>
            ) : (
              <p className="text-sm text-zinc-400">{task.kind === 'scout' ? 'It only looked, so it left a report and no files.' : 'It left no files to copy.'}</p>
            )}
            {files.landed && <p className="text-xs text-zinc-500">It's merged into the project.</p>}
            {(files.files_path || files.report || (files.landed && files.project_path)) && (
              <details className="group/where text-xs">
                <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-zinc-500 transition-colors hover:text-zinc-300 [&::-webkit-details-marker]:hidden">
                  <ChevronRight className="h-3.5 w-3.5 transition-transform group-open/where:rotate-90" /> Where they are, for your agents
                </summary>
                <div className="mt-2 space-y-1.5">
                  {files.files_path && <p className="flex flex-wrap items-center gap-2 text-zinc-500">Its files <CopyChip text={files.files_path} /></p>}
                  {files.report && <p className="flex flex-wrap items-center gap-2 text-zinc-500">Its whole message <CopyChip text={files.report} /></p>}
                  {files.landed && files.project_path && <p className="flex flex-wrap items-center gap-2 text-zinc-500">The project <CopyChip text={files.project_path} /></p>}
                </div>
              </details>
            )}
            {!!files.uncommitted && <p className="text-xs text-amber-700 dark:text-amber-300">{files.uncommitted} uncommitted change{files.uncommitted === 1 ? ' isn’t' : 's aren’t'} in this copy: only commits are.</p>}
            {!!files.skipped && <p className="text-xs text-amber-700 dark:text-amber-300">{files.skipped} file{files.skipped === 1 ? ' was' : 's were'} too big to copy.</p>}
            {task.status !== 'closed' && (
              <div className="flex flex-wrap gap-1.5 pt-0.5">
                {!link && canLink && <button disabled={busy} onClick={makeLink} className={primaryBtn}><Link2 className="h-3.5 w-3.5" /> Put this page online</button>}
                {!files.whole && <button disabled={busy} onClick={() => load(true)} className={quietBtn}><Download className="h-3.5 w-3.5" /> Copy the whole project</button>}
                {busy && <Loader2 className="my-auto h-4 w-4 animate-spin text-zinc-500" />}
              </div>
            )}
          </div>
          {names.length > 0 && (
            <div className="grid gap-3 lg:grid-cols-[minmax(0,15rem)_minmax(0,1fr)]">
              <ul className="max-h-[26rem] space-y-px overflow-y-auto rounded-2xl border border-zinc-800 bg-zinc-900/40 p-1.5">
                {names.map((name) => {
                  const cut = name.lastIndexOf('/') + 1
                  return (
                    <li key={name}>
                      <button
                        type="button" onClick={() => setPath(name)} title={name}
                        className={`flex w-full items-center gap-2 rounded-xl px-2.5 py-1.5 text-left text-xs transition-colors ${path === name ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-300 hover:bg-zinc-800/60'}`}
                      >
                        {/\.(md|markdown|txt)$/i.test(name) ? <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" /> : <FileCode2 className="h-3.5 w-3.5 shrink-0 text-zinc-500" />}
                        <span className="min-w-0 truncate"><span className="text-zinc-500">{name.slice(0, cut)}</span>{name.slice(cut)}</span>
                        {changed.has(name) && <span className="ml-auto h-1.5 w-1.5 shrink-0 rounded-full bg-violet-500" title="It changed this one" />}
                      </button>
                    </li>
                  )
                })}
              </ul>
              <div className="min-h-32 overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/40">
                {path ? <Preview taskId={task.id} path={path} /> : <p className="p-4 text-sm text-zinc-500">Choose a file to look at it.</p>}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
