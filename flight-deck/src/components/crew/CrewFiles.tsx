import { useCallback, useEffect, useState } from 'react'
import { Download, ExternalLink, FileCode2, FileText, FolderOpen, Link2, Loader2 } from 'lucide-react'
import { crewApi, type CrewFile, type CrewFiles as Files, type CrewLink, type CrewTask } from '../../services/crewApi'
import { CopyChip, CrewMarkdown, ErrorLine, LIVE, errorOf, primaryBtn, quietBtn } from './crewParts'

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

function LinkCard({ link }: { link: CrewLink }) {
  return (
    <div className="space-y-1.5 rounded-xl border border-teal-500/30 bg-teal-500/5 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Link2 className="h-3.5 w-3.5 text-teal-700 dark:text-teal-300" />
        <span className="min-w-0 flex-1"><CopyChip text={link.url} /></span>
        <a href={link.url} target="_blank" rel="noreferrer" className={quietBtn}><ExternalLink className="h-3.5 w-3.5" /> Open</a>
      </div>
      <p className="text-xs text-zinc-500">Anyone with this link can open it. It follows the task's files, without the repository's history.</p>
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
      if (got.link) setLink(got.link)
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

  const names = files?.files ?? []
  const changed = new Set(files?.changed ?? [])
  const canLink = files?.whole ? !!files.publish?.subdir : names.some((n) => n.endsWith('index.html'))

  return (
    <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
      <ErrorLine text={error} />
      {!files && !error && <div className="flex justify-center p-6 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>}
      {files && (
        <>
          {link && <LinkCard link={link} />}
          <div className="space-y-2 rounded-xl border border-zinc-800 p-3">
            {files.files_path ? (
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5">
                <FolderOpen className="h-3.5 w-3.5 text-zinc-500" />
                <span className="min-w-0"><CopyChip text={files.files_path} /></span>
                <span className="text-xs text-zinc-500">{files.count} file{files.count === 1 ? '' : 's'}, {files.whole ? 'the whole project' : 'only what it changed'}</span>
              </div>
            ) : (
              <p className="text-sm text-zinc-400">{task.kind === 'scout' ? 'A scout changes nothing, so it left a report and no files.' : 'It left no files to copy.'}</p>
            )}
            {files.landed && files.project_path && (
              <p className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">Merged. The project itself is at <CopyChip text={files.project_path} /></p>
            )}
            {files.report && <p className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">Its whole message is at <CopyChip text={files.report} /></p>}
            {!!files.uncommitted && <p className="text-xs text-amber-700 dark:text-amber-300">{files.uncommitted} uncommitted change{files.uncommitted === 1 ? ' isn’t' : 's aren’t'} in this copy: only commits are.</p>}
            {!!files.skipped && <p className="text-xs text-amber-700 dark:text-amber-300">{files.skipped} file{files.skipped === 1 ? ' was' : 's were'} too big to copy.</p>}
            {task.status !== 'closed' && (
              <div className="flex flex-wrap gap-1.5 pt-0.5">
                {!link && canLink && <button disabled={busy} onClick={makeLink} className={primaryBtn}><Link2 className="h-3.5 w-3.5" /> Get a link to this page</button>}
                {!files.whole && <button disabled={busy} onClick={() => load(true)} className={quietBtn}><Download className="h-3.5 w-3.5" /> Copy the whole project</button>}
                {busy && <Loader2 className="my-auto h-4 w-4 animate-spin text-zinc-500" />}
              </div>
            )}
          </div>
          {names.length > 0 && (
            <div className="grid gap-3 lg:grid-cols-[minmax(0,15rem)_minmax(0,1fr)]">
              <ul className="max-h-[26rem] space-y-px overflow-y-auto rounded-xl border border-zinc-800 p-1">
                {names.map((name) => {
                  const cut = name.lastIndexOf('/') + 1
                  return (
                    <li key={name}>
                      <button
                        type="button" onClick={() => setPath(name)} title={name}
                        className={`flex w-full items-center gap-2 rounded-lg px-2 py-1 text-left text-xs transition-colors ${path === name ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-300 hover:bg-zinc-800/60'}`}
                      >
                        {/\.(md|markdown|txt)$/i.test(name) ? <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" /> : <FileCode2 className="h-3.5 w-3.5 shrink-0 text-zinc-500" />}
                        <span className="min-w-0 truncate"><span className="text-zinc-500">{name.slice(0, cut)}</span>{name.slice(cut)}</span>
                        {changed.has(name) && <span className="ml-auto h-1.5 w-1.5 shrink-0 rounded-full bg-teal-500" title="It changed this one" />}
                      </button>
                    </li>
                  )
                })}
              </ul>
              <div className="min-h-32 overflow-hidden rounded-xl border border-zinc-800">
                {path ? <Preview taskId={task.id} path={path} /> : <p className="p-4 text-sm text-zinc-500">Choose a file to look at it.</p>}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
