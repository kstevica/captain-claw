import { useEffect, useState } from 'react'
import { Check, LibraryBig, Loader2, Plus } from 'lucide-react'
import { useLibraryPicks } from '../../stores/libraryPicksStore'

// "Add to Library" for a file of a shared folder (or the whole folder, `path` ''):
// the File library holds only what the person added, so a run's batches, scripts and
// notes don't bury what matters. Pressed again, it takes it out (the file stays where
// it is). A file whose whole folder is in shows that, and can't be taken out alone.

export function LibraryPick({ folder, path = '', compact = false, className = '', onError, onChange }: {
  folder: string
  path?: string
  /** An icon button, for a row of actions. */
  compact?: boolean
  className?: string
  onError?: (text: string) => void
  /** After it went in (true) or came out (false). */
  onChange?: (inIt: boolean) => void
}) {
  const load = useLibraryPicks((s) => s.load)
  const add = useLibraryPicks((s) => s.add)
  const drop = useLibraryPicks((s) => s.drop)
  const inIt = useLibraryPicks((s) => s.has(folder, path))
  const byFolder = useLibraryPicks((s) => s.has(folder, path) && !s.own(folder, path))
  const [busy, setBusy] = useState(false)
  useEffect(() => { void load().catch(() => {}) }, [load])
  const whole = !path
  const label = inIt
    ? (byFolder ? 'In the Library with its folder' : whole ? 'Folder in the Library' : 'In the Library')
    : whole ? 'Add folder to Library' : 'Add to Library'
  const title = byFolder
    ? 'Its whole folder is in the File library'
    : inIt ? 'In the File library, summarised and searchable. Press to take it out (the file stays where it is).'
      : whole ? "Every file of this folder into the File library, summarised and searchable"
        : 'Into the File library, summarised and searchable by what is inside'

  const toggle = async () => {
    if (byFolder || busy) return
    setBusy(true)
    try {
      if (inIt) await drop(folder, path)
      else await add(folder, path)
      onChange?.(!inIt)
    } catch (e) {
      onError?.(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const icon = busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
    : inIt ? <Check className="h-3.5 w-3.5" /> : compact ? <LibraryBig className="h-3.5 w-3.5" /> : <Plus className="h-3.5 w-3.5" />

  if (compact) {
    return (
      <button type="button" onClick={() => void toggle()} disabled={byFolder} aria-pressed={inIt} title={label + ' — ' + title} aria-label={label}
        className={`flex h-8 w-8 items-center justify-center rounded-full transition-colors ${inIt
          ? 'bg-violet-500/12 text-violet-700 dark:text-violet-300' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'} ${className}`}>
        {icon}
      </button>
    )
  }
  return (
    <button type="button" onClick={() => void toggle()} disabled={byFolder} aria-pressed={inIt} title={title}
      className={`inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-colors ${inIt
        ? 'border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-300'
        : 'border-[var(--fd-line)] text-zinc-300 hover:border-violet-500/40 hover:text-zinc-100'} ${className}`}>
      {icon}{label}
    </button>
  )
}
