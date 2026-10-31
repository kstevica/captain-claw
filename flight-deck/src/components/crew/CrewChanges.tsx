import { useMemo, useState } from 'react'
import { ChevronRight, GitCommitHorizontal } from 'lucide-react'
import type { CrewEvidence } from '../../services/crewApi'

// ── What it changed ──
//
// What git says, a file at a time. Big diffs start folded.

interface FileDiff { name: string; added: number; removed: number; lines: string[] }

const OPEN_LINES = 400

function parseDiff(text: string): FileDiff[] {
  const files: FileDiff[] = []
  let file: FileDiff | null = null
  for (const line of text.split('\n')) {
    if (line.startsWith('diff --git ')) {
      const name = / b\/(.+)$/.exec(line)?.[1] ?? line.slice(11)
      file = { name, added: 0, removed: 0, lines: [] }
      files.push(file)
    } else if (!file) {
      continue
    } else if (line.startsWith('+++ ') || line.startsWith('--- ') || line.startsWith('index ')) {
      continue
    } else {
      if (line.startsWith('+')) file.added += 1
      else if (line.startsWith('-')) file.removed += 1
      file.lines.push(line)
    }
  }
  return files
}

function lineTone(line: string): string {
  if (line.startsWith('+')) return 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
  if (line.startsWith('-')) return 'bg-red-500/10 text-red-700 dark:text-red-300'
  if (line.startsWith('@@')) return 'bg-sky-500/5 text-sky-700 dark:text-sky-400'
  return 'text-zinc-500'
}

function FileCard({ file, startOpen }: { file: FileDiff; startOpen: boolean }) {
  const [open, setOpen] = useState(startOpen)
  return (
    <div className="overflow-hidden rounded-xl border border-zinc-800">
      <button type="button" onClick={() => setOpen((v) => !v)} className="flex w-full items-center gap-2 bg-zinc-800/40 px-3 py-2 text-left transition-colors hover:bg-zinc-800/70">
        <ChevronRight className={`h-3.5 w-3.5 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-90' : ''}`} />
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-zinc-200">{file.name}</span>
        {file.added > 0 && <span className="font-mono text-[11px] text-emerald-700 dark:text-emerald-400">+{file.added}</span>}
        {file.removed > 0 && <span className="font-mono text-[11px] text-red-700 dark:text-red-400">−{file.removed}</span>}
      </button>
      {open && (
        <div className="overflow-x-auto py-1 font-mono text-[11.5px] leading-[1.6]">
          <div className="w-max min-w-full">
            {file.lines.map((line, i) => <div key={i} className={`whitespace-pre px-3 ${lineTone(line)}`}>{line || ' '}</div>)}
          </div>
        </div>
      )}
    </div>
  )
}

function DiffFiles({ text }: { text: string }) {
  const files = useMemo(() => parseDiff(text), [text])
  const total = files.reduce((n, f) => n + f.lines.length, 0)
  return <>{files.map((f) => <FileCard key={f.name} file={f} startOpen={total <= OPEN_LINES} />)}</>
}

export function CrewChanges({ diff, evidence }: {
  diff: { committed: string; uncommitted: string; note?: string } | null; evidence: CrewEvidence
}) {
  const log = evidence.log ?? []
  return (
    <div className="min-h-0 flex-1 space-y-2.5 overflow-y-auto p-4">
      {log.length > 0 && (
        <ul className="space-y-1 rounded-xl border border-zinc-800 p-2.5">
          {log.map((entry) => {
            const [sha, ...words] = entry.split(' ')
            return (
              <li key={entry} className="flex items-start gap-2 text-xs">
                <GitCommitHorizontal className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />
                <code className="shrink-0 text-[11px] text-zinc-500">{sha}</code>
                <span className="min-w-0 break-words text-zinc-300">{words.join(' ')}</span>
              </li>
            )
          })}
        </ul>
      )}
      {!diff && <p className="text-sm text-zinc-500">Reading…</p>}
      {diff?.note && <p className="text-xs text-zinc-500">{diff.note}</p>}
      {diff && !diff.committed && !diff.uncommitted && !diff.note && <p className="text-sm text-zinc-500">No changes.</p>}
      {diff?.committed && <DiffFiles text={diff.committed} />}
      {diff?.uncommitted && (
        <>
          <p className="pt-1 text-xs font-medium text-amber-700 dark:text-amber-300">Not committed yet</p>
          <DiffFiles text={diff.uncommitted} />
        </>
      )}
    </div>
  )
}
