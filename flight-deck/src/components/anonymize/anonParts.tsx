import { useState, type ReactNode } from 'react'
import { Check, Copy, Download, FileUp } from 'lucide-react'
import { TOKEN_RE } from '../../services/anonymizeApi'
import { Button } from '../console/parts'
import { kindLabel, kindTone } from './anonFormat'

// Shared pieces of the Anonymize page: tokens shown as coloured chips, a copy
// button, and the drop zone for files.

/** Text with every token as a chip; the rest as written. */
export function TokenText({ text, className = '' }: { text: string; className?: string }) {
  const parts: ReactNode[] = []
  let last = 0
  for (const m of text.matchAll(TOKEN_RE)) {
    const at = m.index ?? 0
    if (at > last) parts.push(text.slice(last, at))
    parts.push(
      <span key={at} className={`mx-px rounded px-1 py-px font-mono text-[0.85em] ${kindTone(m[1])}`} title={kindLabel(m[1])}>
        {m[0]}
      </span>,
    )
    last = at + m[0].length
  }
  if (last < text.length) parts.push(text.slice(last))
  return (
    <div className={`whitespace-pre-wrap break-words rounded-xl border border-zinc-800 bg-zinc-950 p-4 text-sm leading-relaxed text-zinc-200 ${className}`}>
      {parts.length ? parts : <span className="text-zinc-600">(empty)</span>}
    </div>
  )
}

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false)
  return (
    <Button
      icon={done ? Check : Copy}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text)
          setDone(true)
          setTimeout(() => setDone(false), 1500)
        } catch { /* the browser refused; nothing to do */ }
      }}
    >
      {done ? 'Copied' : label}
    </Button>
  )
}

export function DownloadButton({ onClick, children = 'Download' }: { onClick: () => void; children?: ReactNode }) {
  return <Button icon={Download} onClick={onClick}>{children}</Button>
}

/** Drop a file here, or pick one. */
export function FileDrop({ file, accept, maxMb, onFile }: {
  file: File | null
  accept: string[]
  maxMb: number
  onFile: (f: File | null) => void
}) {
  const [over, setOver] = useState(false)
  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const f = e.dataTransfer.files?.[0]
        if (f) onFile(f)
      }}
      className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
        over ? 'border-violet-500/60 bg-violet-500/5' : 'border-zinc-800 hover:border-zinc-600'
      }`}
    >
      <FileUp className="h-6 w-6 text-zinc-500" />
      {file ? (
        <span className="text-sm font-medium text-zinc-200">{file.name} <span className="text-zinc-500">· {(file.size / 1024).toFixed(0)} KB</span></span>
      ) : (
        <span className="text-sm text-zinc-300">Drop a file here, or click to choose one</span>
      )}
      <span className="text-xs text-zinc-500">
        Word, Excel and PowerPoint come back as the same kind of file. Text, CSV and JSON too; PDF and web pages as text. Up to {maxMb} MB.
      </span>
      <input
        type="file"
        className="sr-only"
        accept={accept.join(',')}
        onChange={(e) => onFile(e.target.files?.[0] || null)}
      />
    </label>
  )
}
