import { useCallback, useEffect, useState } from 'react'
import { ShieldCheck, Square } from 'lucide-react'
import {
  anonymizeApi, type FolderJob, type HereResult, type LibraryResult, type Mode, type RunOptions, type Vault,
} from '../../services/anonymizeApi'
import { Button, Dialog, Field, Notice, Pill, Segmented, inputCls } from '../console/parts'
import { kindLabel } from './anonFormat'

// Anonymize a file or a folder where it is — a VFS file, a whole VFS folder,
// or a file in the Library. The copy goes next to the original
// ("Contract.anonymized.docx", "Clients_anonymized/"); from a read-only place
// it goes to the person's own "Anonymized" folder. A folder runs on the deck
// as a job this dialog follows (closing it doesn't stop the job).

export type AnonTarget =
  | { kind: 'vfs-file'; project: string; path: string; owner: string; name: string }
  | { kind: 'vfs-folder'; project: string; path: string; owner: string; name: string }
  | { kind: 'library'; id: string; name: string; agent: string }

const lines = (s: string) => s.split('\n').map((l) => l.trim()).filter(Boolean)
const msg = (e: unknown) => (e instanceof Error ? e.message : String(e))
const isJob = (j: FolderJob | { running: false }): j is FolderJob => 'id' in j

function defaultVault(t: AnonTarget): string {
  if (t.kind === 'library') return t.agent
  if (t.kind === 'vfs-folder' && t.path) return t.path.split('/').pop() || t.project
  return t.project
}

function where(p: { project: string; path: string }): string {
  return [p.project, p.path].filter(Boolean).join('/')
}

function Summary({ summary }: { summary: Record<string, number> }) {
  const kinds = Object.entries(summary).filter(([, n]) => n > 0)
  if (!kinds.length) return <p className="text-sm text-zinc-500">Nothing private was found.</p>
  return (
    <div className="flex flex-wrap gap-1.5">
      {kinds.map(([k, n]) => <Pill key={k}>{kindLabel(k)} · {n}</Pill>)}
    </div>
  )
}

export function AnonymizeHereDialog({ target, onClose, onDone }: {
  target: AnonTarget
  onClose: () => void
  /** After a copy was written (to refresh a listing). */
  onDone?: () => void
}) {
  const [vaults, setVaults] = useState<Vault[]>([])
  const [mode, setMode] = useState<Mode>('reversible')
  const [vault, setVault] = useState('')
  const [coarser, setCoarser] = useState(false)
  const [keep, setKeep] = useState('')
  const [more, setMore] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<HereResult | LibraryResult | null>(null)
  const [job, setJob] = useState<FolderJob | null>(null)
  const folder = target.kind === 'vfs-folder'

  useEffect(() => {
    anonymizeApi.vaults().then((r) => setVaults(r.vaults)).catch(() => { /* the field still works */ })
    // A folder job already running (from before) is shown instead of the form.
    if (folder) anonymizeApi.vfsJob().then((j) => { if (isJob(j) && j.running) setJob(j) }).catch(() => {})
  }, [folder])

  // Follow a running job.
  const running = !!job?.running
  useEffect(() => {
    if (!running) return
    const t = setInterval(async () => {
      try {
        const j = await anonymizeApi.vfsJob()
        if (isJob(j)) {
          setJob(j)
          if (!j.running) onDone?.()
        }
      } catch { /* try again on the next tick */ }
    }, 1500)
    return () => clearInterval(t)
  }, [running, onDone])

  const opts = (): RunOptions => ({
    mode,
    vault: mode === 'reversible' ? (vault.trim() || defaultVault(target)) : undefined,
    keep: lines(keep),
    generalize: mode === 'one_way' && coarser,
  })

  const run = async () => {
    setBusy(true); setError('')
    try {
      if (target.kind === 'vfs-file') setResult(await anonymizeApi.vfsFile(target.project, target.path, target.owner, opts()))
      else if (target.kind === 'library') setResult(await anonymizeApi.libraryFile(target.id, opts()))
      else setJob(await anonymizeApi.vfsFolder(target.project, target.path, target.owner, opts()))
      if (target.kind !== 'vfs-folder') onDone?.()
    } catch (e) {
      setError(msg(e))
    } finally {
      setBusy(false)
    }
  }

  const stop = useCallback(async () => {
    try { setJob(await anonymizeApi.vfsJobCancel()) } catch (e) { setError(msg(e)) }
  }, [])

  const title = `Anonymize ${target.name}`
  const showForm = !result && !job

  return (
    <Dialog
      title={title}
      icon={ShieldCheck}
      onClose={onClose}
      footer={showForm ? (
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" icon={ShieldCheck} busy={busy} onClick={run}>Anonymize</Button>
        </>
      ) : job?.running ? (
        <>
          <Button icon={Square} onClick={stop} disabled={job.cancelled}>{job.cancelled ? 'Stopping…' : 'Stop'}</Button>
          <Button onClick={onClose}>Close (it keeps going)</Button>
        </>
      ) : <Button variant="primary" onClick={onClose}>Done</Button>}
    >
      {showForm && (
        <div className="space-y-4">
          <p className="text-sm text-zinc-400">
            {folder
              ? 'Every file in it that can be anonymized gets an anonymized copy, in a new folder beside this one. Pictures and other kinds are left out. Names in the folder and file names become tokens too.'
              : 'An anonymized copy is saved beside the file. The file itself isn\'t changed.'}
            {' '}If you can't write here (Drive, a read-only or view-only folder), the copy goes to your own “Anonymized” folder.
          </p>
          <Field label="What happens to the real values" hint={mode === 'reversible'
            ? 'Kept in a vault, so you can put them back later. The same person gets the same token in every file of that vault.'
            : 'Nothing is kept: the tokens can never be turned back.'}>
            <Segmented label="What happens to the real values" value={mode} onChange={setMode}
              options={[{ value: 'reversible', label: 'In a vault' }, { value: 'one_way', label: 'Keep nothing' }]} />
          </Field>
          {mode === 'one_way' && (
            <label className="flex items-start gap-2.5 text-sm text-zinc-300">
              <input type="checkbox" checked={coarser} onChange={(e) => setCoarser(e.target.checked)} className="mt-0.5" />
              <span>Make them coarser: date of birth → year, address → city, ages and amounts → ranges</span>
            </label>
          )}
          {mode === 'reversible' && (
            <Field label="Vault" hint="A new name makes a new vault.">
              <input list="anon-here-vaults" value={vault} onChange={(e) => setVault(e.target.value)}
                placeholder={defaultVault(target)} className={inputCls} />
              <datalist id="anon-here-vaults">
                {vaults.map((v) => <option key={v.id} value={v.name} />)}
              </datalist>
            </Field>
          )}
          {more ? (
            <Field label="Leave these as they are" hint="One per line — your own company, a product name.">
              <textarea value={keep} onChange={(e) => setKeep(e.target.value)} rows={3} className={inputCls} />
            </Field>
          ) : (
            <button onClick={() => setMore(true)} className="text-xs text-zinc-500 underline-offset-2 hover:text-zinc-300 hover:underline">More</button>
          )}
          {error && <Notice tone="red">{error}</Notice>}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <Notice tone="emerald">
            {'beside' in result
              ? result.beside
                ? <>Saved beside it as <span className="font-mono">{result.written.path}</span>.</>
                : <>You can't write in that folder, so it's in your own: <span className="font-mono">{where(result.written)}</span>.</>
              : <>Saved beside it as <span className="font-mono">{result.written.rel_path}</span>.</>}
            {result.vault ? ` Real values are in your vault “${result.vault.name}”.` : ' Nothing was kept.'}
          </Notice>
          <Summary summary={result.summary} />
          {result.warnings.map((w) => <Notice key={w} tone="amber">{w}</Notice>)}
        </div>
      )}

      {job && <JobView job={job} />}
    </Dialog>
  )
}

function JobView({ job }: { job: FolderJob }) {
  const pct = job.total ? Math.round((job.done + job.failed_count) * 100 / job.total) : 100
  const dest = job.dest.project ? { project: job.dest.project, path: job.dest.path || '' } : null
  return (
    <div className="space-y-4">
      {job.running ? (
        <>
          <div className="h-2 overflow-hidden rounded-full bg-zinc-800">
            <div className="h-full bg-violet-500 transition-all" style={{ width: `${pct}%` }} />
          </div>
          <p className="text-sm text-zinc-300">
            {job.phase === 'placing' ? 'Saving the copies…' : `${job.done + job.failed_count} of ${job.total} files`}
            {job.current && <span className="block truncate font-mono text-xs text-zinc-500">{job.current}</span>}
          </p>
        </>
      ) : dest ? (
        <Notice tone="emerald">
          {job.done} {job.done === 1 ? 'file' : 'files'} anonymized into <span className="font-mono">{where(dest)}</span>
          {job.beside ? '.' : ' (in your own folder — you can\'t write beside the original).'}
          {job.vault ? ` Real values are in your vault “${job.vault}”.` : ' Nothing was kept.'}
        </Notice>
      ) : (
        <Notice tone="amber">{job.message || 'Nothing was anonymized.'}</Notice>
      )}
      {job.message && dest && <Notice tone="amber">{job.message}</Notice>}
      {!job.running && <Summary summary={job.entities} />}
      {job.failed_count > 0 && (
        <details className="text-sm text-zinc-400">
          <summary className="cursor-pointer text-red-600 dark:text-red-400">{job.failed_count} couldn't be anonymized (no copy made)</summary>
          <ul className="mt-2 space-y-1">
            {job.failed.map((f) => <li key={f.path}><span className="font-mono text-xs">{f.path}</span> — {f.error}</li>)}
          </ul>
        </details>
      )}
      {job.skipped_count > 0 && (
        <details className="text-sm text-zinc-400">
          <summary className="cursor-pointer">{job.skipped_count} left out</summary>
          <ul className="mt-2 space-y-1">
            {job.skipped.map((f) => <li key={f.path}><span className="font-mono text-xs">{f.path}</span> — {f.reason}</li>)}
          </ul>
        </details>
      )}
    </div>
  )
}
