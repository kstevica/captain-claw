import { useCallback, useEffect, useMemo, useState } from 'react'
import { vfsApi, type FolderHit, type FolderRef } from '../../services/vfsApi'

// One run's folder, read once for its row and shared by its Files and Notes tabs (and
// their counts). Read again every few seconds while the run is going and one of those
// tabs is open.

const POLL_MS = 10_000

/** The workers' own: progress notes, research notes, what every step shares
 *  (conventions, corrections, tracking). The extra data it was given is its own tab. */
export const NOTE_DIRS = { progress: 'progress/', notes: 'notes/', shared: 'shared/' } as const
const GIVEN = 'extra/'

const isNote = (f: FolderHit) => Object.values(NOTE_DIRS).some((d) => f.path.startsWith(d))

/** ``project`` '' until the run's progress says where its folder is: nothing is read. */
export function useRunFolder(project: string, poll: boolean) {
  const ref: FolderRef = useMemo(() => ({ project, owner: '' }), [project])
  const [files, setFiles] = useState<FolderHit[] | null>(null)
  const [error, setError] = useState('')

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
    if (!project) return  // not known yet
    let stop = false
    const run = () => { if (!stop) void load() }
    const first = setTimeout(run, 0)
    const t = poll ? setInterval(run, POLL_MS) : undefined
    return () => { stop = true; clearTimeout(first); if (t) clearInterval(t) }
  }, [load, poll, project])

  const all = files ?? []
  return {
    ref,
    files,
    error,
    setError,
    load,
    results: all.filter((f) => !f.path.startsWith(GIVEN) && !isNote(f)),
    notes: all.filter(isNote),
  }
}

export type RunFolder = ReturnType<typeof useRunFolder>
