import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  AlertTriangle, ArrowLeft, Bot, Check, ChevronRight, ClipboardPaste, Cloud, Database, FileText, Folder,
  FolderSymlink, Link2, Loader2, MessagesSquare, Plus, Search, Share2, Trash2, UploadCloud, X, type LucideIcon,
} from 'lucide-react'
import {
  contextApi,
  type AgentFile, type ContextAgent, type Conversation, type ConversationMessage, type ConversationSession, type Staged,
} from '../../services/contextApi'
import { vfsApi, type FolderHit } from '../../services/vfsApi'
import type { VFSProject } from '../../stores/vfsStore'
import { pastedFiles } from '../crew/CrewAttachments'
import { sizeText } from '../../utils/format'

// ── Add data ──
//
// One panel for everything a person can hand their work: files from this computer,
// something pasted, an agent's files, files from Flight Deck's folders, a stretch of a
// conversation with an agent. Each is kept by the deck as soon as it's chosen and waits in
// the tray at the bottom; the panel's button hands the tray to whatever it's for (a
// factory, a run, a Crew task). Closed without that, what's in the tray is let go of.

export type DataSource = 'upload' | 'paste' | 'links' | 'agents' | 'deck' | 'conversation'

const SOURCES: { id: DataSource; label: string; hint: string; icon: LucideIcon }[] = [
  { id: 'upload', label: 'This computer', hint: 'Drop or browse files', icon: UploadCloud },
  { id: 'paste', label: 'Paste', hint: 'Notes, a list, a table', icon: ClipboardPaste },
  { id: 'links', label: 'Links', hint: 'Web pages to work from', icon: Link2 },
  { id: 'agents', label: "Agents' files", hint: 'What your agents saved', icon: Bot },
  { id: 'deck', label: 'Flight Deck', hint: 'Your folders and shared ones', icon: Folder },
  { id: 'conversation', label: 'A conversation', hint: 'What you said with an agent', icon: MessagesSquare },
]

interface TrayItem {
  key: string
  name: string
  size: number
  origin: string
  id?: string
  error?: string
  preview?: string
  staged?: Staged
}

const errorOf = (e: unknown) => (e instanceof Error ? e.message : String(e))

function agoText(seconds: number): string {
  const s = Math.max(0, Date.now() / 1000 - seconds)
  return s < 90 ? 'just now' : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129_600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86_400)} d ago`
}

function when(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const sameYear = d.getFullYear() === new Date().getFullYear()
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', ...(sameYear ? {} : { year: 'numeric' }), hour: '2-digit', minute: '2-digit' })
}

/** A screenshot on the clipboard has no name of its own: "image.png" every time. */
function named(file: File, index: number): File {
  if (file.name && file.name !== 'image.png') return file
  const ext = (file.type.split('/')[1] || 'png').split('+')[0]
  return new File([file], `pasted-${Date.now()}-${index + 1}.${ext}`, { type: file.type })
}

const PRIMARY = 'inline-flex h-9 items-center justify-center gap-1.5 rounded-full bg-violet-600 px-4 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50'
const QUIET = 'inline-flex h-9 items-center justify-center gap-1.5 rounded-full border border-[var(--fd-line)] px-4 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800 hover:text-zinc-100 disabled:opacity-50'
const FIELD = 'w-full rounded-xl border border-[var(--fd-line)] bg-zinc-950 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/15'

export function AddDataPanel({ title, subtitle, action, onDone, onClose, start = 'upload' }: {
  title: string
  subtitle?: ReactNode
  /** The button's words, said with how many: "Add to the factory" → "Add 3 to the factory". */
  action: (count: number) => string
  /** Hands over what's ready. Throws to keep the panel open with the reason. */
  onDone: (items: Staged[]) => Promise<void> | void
  onClose: () => void
  start?: DataSource
}) {
  const [source, setSource] = useState<DataSource>(start)
  const [tray, setTray] = useState<TrayItem[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [flash, setFlash] = useState('')
  const live = useRef<TrayItem[]>([])
  useEffect(() => { live.current = tray }, [tray])
  const handed = useRef(false)

  // Whatever wasn't handed over is let go of with the panel.
  useEffect(() => () => {
    for (const item of live.current) {
      if (item.preview) URL.revokeObjectURL(item.preview)
      if (!handed.current && item.id) contextApi.drop(item.id).catch(() => {})
    }
  }, [])

  const added = useCallback((list: Staged[]) => {
    if (!list.length) return
    setTray((old) => [...old, ...list.map((s) => ({ key: `${s.id}-${crypto.randomUUID()}`, id: s.id, name: s.name, size: s.size, origin: s.origin || 'Uploaded', staged: s }))])
    setFlash(`${list.length === 1 ? list[0].name : `${list.length} files`} added below`)
  }, [])

  useEffect(() => {
    if (!flash) return
    const t = setTimeout(() => setFlash(''), 2400)
    return () => clearTimeout(t)
  }, [flash])

  const upload = useCallback((files: File[]) => {
    files.forEach((raw, index) => {
      const file = named(raw, index)
      const key = `${file.name}-${crypto.randomUUID()}`
      const preview = file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined
      setTray((old) => [...old, { key, name: file.name, size: file.size, origin: 'Uploaded', preview }])
      contextApi.upload(file)
        .then((made) => setTray((old) => old.map((t) => (t.key === key ? { ...t, id: made.id, name: made.name, staged: { ...made, origin: 'Uploaded' } } : t))))
        .catch((e) => setTray((old) => old.map((t) => (t.key === key ? { ...t, error: errorOf(e) } : t))))
    })
  }, [])

  const remove = (key: string) => setTray((old) => {
    const gone = old.find((t) => t.key === key)
    if (gone?.preview) URL.revokeObjectURL(gone.preview)
    if (gone?.id) contextApi.drop(gone.id).catch(() => {})
    return old.filter((t) => t.key !== key)
  })

  // Escape closes this panel only, not what it was opened from.
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key !== 'Escape') return
      ev.stopPropagation()
      onClose()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  // A screenshot pasted anywhere in the panel but a field goes up as a file.
  useEffect(() => {
    const onPaste = (ev: ClipboardEvent) => {
      if ((ev.target as Element | null)?.closest?.('input, textarea, [contenteditable="true"]')) return
      const list = pastedFiles(ev.clipboardData)
      if (!list.length) return
      ev.preventDefault()
      upload(list)
    }
    document.addEventListener('paste', onPaste)
    return () => document.removeEventListener('paste', onPaste)
  }, [upload])

  const pending = tray.some((t) => !t.id && !t.error)
  const ready = tray.filter((t) => t.id && !t.error && t.staged).map((t) => t.staged as Staged)
  const failed = tray.filter((t) => t.error).length

  const done = async () => {
    setBusy(true); setError('')
    try {
      await onDone(ready)
      handed.current = true
      onClose()
    } catch (e) {
      setError(errorOf(e))
      setBusy(false)
    }
  }

  return createPortal(
    <div data-above-drawer className="fixed inset-0 z-[70] flex items-stretch justify-center bg-black/60 backdrop-blur-sm wide:items-center wide:p-6"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div role="dialog" aria-modal="true" aria-label={title}
        onDragOver={(e) => { if (e.dataTransfer?.types.includes('Files')) { e.preventDefault(); setDragging(true) } }}
        onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false) }}
        onDrop={(e) => {
          const list = Array.from(e.dataTransfer?.files ?? [])
          setDragging(false)
          if (!list.length) return
          e.preventDefault()
          upload(list)
        }}
        className="@container relative flex h-full w-full flex-col overflow-hidden bg-[var(--fd-surface,#09090b)] shadow-2xl wide:h-[min(46rem,92dvh)] wide:max-w-5xl wide:rounded-[22px] wide:border wide:border-[var(--fd-line)]">
        {/* header */}
        <div className="flex items-start gap-3 border-b border-[var(--fd-line)] px-5 py-4">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-md shadow-violet-500/20">
            <Database className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-base font-semibold text-zinc-100">{title}</h2>
            {subtitle && <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p>}
          </div>
          <button onClick={onClose} aria-label="Close" title="Close (Esc)" className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex min-h-0 flex-1 flex-col @3xl:flex-row">
          {/* sources: a rail on a wide panel, pills on a narrow one */}
          <nav aria-label="Where from" className="shrink-0 border-b border-[var(--fd-line)] @3xl:w-60 @3xl:border-b-0 @3xl:border-r">
            <div className="flex gap-1.5 overflow-x-auto px-4 py-3 @3xl:flex-col @3xl:gap-1 @3xl:overflow-visible @3xl:p-3">
              {SOURCES.map((s) => {
                const on = s.id === source
                const Icon = s.icon
                return (
                  <button key={s.id} onClick={() => setSource(s.id)} aria-current={on ? 'true' : undefined}
                    className={`group flex shrink-0 items-center gap-3 rounded-full px-3 py-2 text-left transition-all @3xl:rounded-xl @3xl:py-2.5 ${on
                      ? 'bg-violet-500/12 text-zinc-100 ring-1 ring-violet-500/30'
                      : 'text-zinc-400 hover:bg-zinc-800/70 hover:text-zinc-200'}`}>
                    <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg transition-colors ${on ? 'bg-violet-600 text-white' : 'bg-zinc-800/80 text-zinc-400 group-hover:text-zinc-200'}`}>
                      <Icon className="h-3.5 w-3.5" />
                    </span>
                    <span className="min-w-0">
                      <span className="block whitespace-nowrap text-[13px] font-medium">{s.label}</span>
                      <span className="hidden truncate text-[11px] text-zinc-500 @3xl:block">{s.hint}</span>
                    </span>
                  </button>
                )
              })}
            </div>
          </nav>

          {/* the chosen source */}
          <div className="relative min-h-0 min-w-0 flex-1 overflow-y-auto px-5 py-5">
            {source === 'upload' && <UploadSource onFiles={upload} />}
            {source === 'paste' && <PasteSource onAdded={added} onFiles={upload} />}
            {source === 'links' && <LinksSource onAdded={added} />}
            {source === 'agents' && <AgentsSource onAdded={added} />}
            {source === 'deck' && <DeckSource onAdded={added} />}
            {source === 'conversation' && <ConversationSource onAdded={added} />}
          </div>
        </div>

        {/* the tray, and handing it over */}
        <div className="border-t border-[var(--fd-line)] bg-[var(--fd-surface-raised)]/60 px-5 py-3">
          <div className="mb-2 flex items-center gap-2 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            Ready to add
            {tray.length > 0 && <span className="rounded-full bg-violet-500/15 px-1.5 py-px text-[10px] text-violet-700 dark:text-violet-300">{tray.length}</span>}
            {flash && <span role="status" className="ml-auto inline-flex items-center gap-1 normal-case tracking-normal text-emerald-600 dark:text-emerald-400"><Check className="h-3 w-3" />{flash}</span>}
          </div>
          {tray.length ? (
            <ul className="flex max-h-28 flex-wrap gap-2 overflow-y-auto pb-1" aria-label="Ready to add">
              {tray.map((t) => <TrayChip key={t.key} item={t} onRemove={() => remove(t.key)} />)}
            </ul>
          ) : (
            <p className="text-xs text-zinc-500">Nothing yet. Pick from a source above, drop files anywhere here, or paste a screenshot.</p>
          )}
          {error && <p role="alert" className="mt-2 text-xs text-red-700 dark:text-red-300">{error}</p>}
          <div className="mt-3 flex flex-wrap items-center justify-end gap-2">
            {failed > 0 && <span className="mr-auto text-xs text-red-700 dark:text-red-300">{failed} didn't go up: take {failed === 1 ? 'it' : 'them'} off or try again.</span>}
            <button onClick={onClose} className={QUIET}>Cancel</button>
            <button onClick={() => void done()} disabled={busy || pending || !ready.length} className={PRIMARY}>
              {busy || pending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
              {pending ? 'Uploading…' : action(ready.length)}
            </button>
          </div>
        </div>

        {dragging && (
          <div className="pointer-events-none absolute inset-3 z-10 flex flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-violet-500/70 bg-violet-500/10 text-violet-700 backdrop-blur-[2px] dark:text-violet-200">
            <UploadCloud className="h-10 w-10" />
            <p className="text-sm font-semibold">Drop to add</p>
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}

function TrayChip({ item, onRemove }: { item: TrayItem; onRemove: () => void }) {
  return (
    <li title={item.error || `${item.name} · ${item.origin}`}
      className={`fd-fade flex max-w-full items-center gap-2 rounded-xl border py-1 pl-1 pr-1.5 text-xs ${item.error
        ? 'border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-300'
        : 'border-[var(--fd-line)] bg-zinc-900/70 text-zinc-300'}`}>
      {item.preview && !item.error
        ? <img src={item.preview} alt="" className="h-8 w-8 shrink-0 rounded-lg object-cover" />
        : (
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800/80">
            {item.error ? <AlertTriangle className="h-4 w-4 text-red-500" /> : <FileGlyph name={item.name} />}
          </span>
        )}
      <span className="min-w-0">
        <span className="block max-w-[13rem] truncate font-medium text-zinc-200">{item.name}</span>
        <span className="block max-w-[13rem] truncate text-[11px] text-zinc-500">{item.error || `${item.origin} · ${sizeText(item.size)}`}</span>
      </span>
      {!item.id && !item.error && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-zinc-500" />}
      <button type="button" onClick={onRemove} aria-label={`Take ${item.name} off`} className="rounded-md p-0.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
        <X className="h-3.5 w-3.5" />
      </button>
    </li>
  )
}

/** A file's icon, coloured by kind. */
export function FileGlyph({ name, className = 'h-4 w-4' }: { name: string; className?: string }) {
  const ext = name.includes('.') ? name.slice(name.lastIndexOf('.') + 1).toLowerCase() : ''
  const tone = ['csv', 'tsv', 'xlsx', 'xls'].includes(ext) ? 'text-emerald-600 dark:text-emerald-400'
    : ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'].includes(ext) ? 'text-sky-600 dark:text-sky-400'
      : ['md', 'txt', 'doc', 'docx', 'pdf'].includes(ext) ? 'text-violet-600 dark:text-violet-400'
        : ['json', 'yaml', 'yml', 'xml', 'html'].includes(ext) ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-500'
  return <FileText className={`${className} shrink-0 ${tone}`} />
}

// ── Sources ──

function SourceHead({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="mb-4">
      <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
      <p className="mt-0.5 text-xs text-zinc-500">{hint}</p>
    </div>
  )
}

function UploadSource({ onFiles }: { onFiles: (files: File[]) => void }) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <div className="flex h-full flex-col">
      <SourceHead title="Files from this computer" hint="Any kind: documents, spreadsheets, pictures, exports. Up to 50 MB each." />
      <button type="button" onClick={() => input.current?.click()}
        className="group flex min-h-[14rem] flex-1 flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-zinc-700 bg-zinc-900/30 p-8 text-center transition-all hover:border-violet-500/60 hover:bg-violet-500/[0.05]">
        <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-violet-500/10 text-violet-600 transition-transform group-hover:-translate-y-0.5 dark:text-violet-300">
          <UploadCloud className="h-7 w-7" />
        </span>
        <span className="text-sm font-semibold text-zinc-200">Drop files here, or <span className="text-violet-700 underline-offset-2 group-hover:underline dark:text-violet-300">browse</span></span>
        <span className="text-xs text-zinc-500">A screenshot can be pasted anywhere in this panel.</span>
      </button>
      <input ref={input} type="file" multiple className="hidden" aria-hidden="true" tabIndex={-1}
        onChange={(e) => { onFiles(Array.from(e.target.files ?? [])); e.target.value = '' }} />
    </div>
  )
}

function PasteSource({ onAdded, onFiles }: { onAdded: (list: Staged[]) => void; onFiles: (files: File[]) => void }) {
  const [text, setText] = useState('')
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const add = async () => {
    setBusy(true); setError('')
    try {
      onAdded([await contextApi.paste(text, name)])
      setText(''); setName('')
    } catch (e) {
      setError(errorOf(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="flex h-full flex-col">
      <SourceHead title="Paste" hint="Notes, a brief, a list, a table copied from a sheet. It's kept as a file of its own." />
      <label className="sr-only" htmlFor="ad-paste">What to add</label>
      <textarea id="ad-paste" value={text} onChange={(e) => setText(e.target.value)} autoFocus
        onPaste={(e) => { const files = pastedFiles(e.clipboardData); if (files.length) { e.preventDefault(); onFiles(files) } }}
        onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && text.trim()) void add() }}
        placeholder="Paste or type here…"
        className={`${FIELD} min-h-[12rem] flex-1 resize-none font-mono text-[13px] leading-relaxed`} />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <label className="sr-only" htmlFor="ad-paste-name">Name</label>
        <input id="ad-paste-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Name it (optional)" maxLength={120}
          className={`${FIELD} h-9 min-w-0 flex-1 py-0 @lg:max-w-xs`} />
        <span className="text-[11px] tabular-nums text-zinc-500">{text.length.toLocaleString()} characters</span>
        <button onClick={() => void add()} disabled={busy || !text.trim()} className={`${PRIMARY} ml-auto`}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ClipboardPaste className="h-4 w-4" />}Add it
        </button>
      </div>
      {error && <p role="alert" className="mt-2 text-xs text-red-700 dark:text-red-300">{error}</p>}
    </div>
  )
}

// ── Links: as many as needed, kept as one list ──

const URL_RE = /^https?:\/\/[^\s/?#]+\.[^\s/?#]+\S*$/i
const FIND_URLS = /https?:\/\/[^\s<>"'`)\]]+/gi
const LINKS_MAX = 500

interface LinkRow { id: string; url: string; title: string }
let linkSeq = 0
const linkRow = (url = '', title = ''): LinkRow => ({ id: `l${++linkSeq}`, url, title })

/** Every web address in some text, without trailing punctuation, each once. */
function urlsIn(text: string): string[] {
  const found = (text.match(FIND_URLS) ?? []).map((u) => u.replace(/[.,;:!?]+$/, ''))
  return [...new Set(found)]
}

function LinksSource({ onAdded }: { onAdded: (list: Staged[]) => void }) {
  const [rows, setRows] = useState<LinkRow[]>([linkRow()])
  const [bulk, setBulk] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const filled = rows.filter((r) => r.url.trim())
  const bad = filled.filter((r) => !URL_RE.test(r.url.trim()))
  const ready = filled.length > 0 && bad.length === 0

  const set = (id: string, change: Partial<LinkRow>) => setRows((old) => old.map((r) => (r.id === id ? { ...r, ...change } : r)))
  const take = (urls: string[]) => setRows((old) => {
    const have = new Set(old.map((r) => r.url.trim().toLowerCase()).filter(Boolean))
    const fresh = urls.filter((u) => !have.has(u.toLowerCase())).map((u) => linkRow(u))
    const kept = old.filter((r) => r.url.trim() || r.title.trim())
    return [...kept, ...fresh].slice(0, LINKS_MAX).concat(kept.length + fresh.length ? [] : [linkRow()])
  })

  const add = async () => {
    setBusy(true); setError('')
    try {
      onAdded([await contextApi.links(filled.map((r) => ({ url: r.url.trim(), title: r.title.trim() })), name)])
      setRows([linkRow()]); setName('')
    } catch (e) {
      setError(errorOf(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <SourceHead title="Links" hint="Web pages to work from, as many as you need: they're kept as one list that the workers open as the work needs them." />
      {bulk === null ? (
        <>
          <ol className="space-y-2" aria-label="Links">
            {rows.map((r, i) => {
              const wrong = !!r.url.trim() && !URL_RE.test(r.url.trim())
              return (
                <li key={r.id} className="flex flex-wrap items-start gap-2 @lg:flex-nowrap">
                  <span className="mt-2 w-5 shrink-0 text-right text-[11px] font-semibold tabular-nums text-zinc-500">{i + 1}</span>
                  <div className="min-w-0 flex-1">
                    <input value={r.url} placeholder="https://…" aria-label={`Link ${i + 1}`} inputMode="url" autoFocus={i === rows.length - 1 && i > 0}
                      onChange={(e) => set(r.id, { url: e.target.value })}
                      onPaste={(e) => {
                        const found = urlsIn(e.clipboardData.getData('text'))
                        if (found.length > 1) { e.preventDefault(); set(r.id, { url: found[0] }); take(found.slice(1)) }
                      }}
                      onKeyDown={(e) => { if (e.key === 'Enter' && r.url.trim()) { e.preventDefault(); setRows((old) => [...old, linkRow()]) } }}
                      className={`${FIELD} h-9 py-0 font-mono text-[12.5px] ${wrong ? '!border-red-500/60' : ''}`} />
                    {wrong && <p className="mt-0.5 text-[11px] text-red-700 dark:text-red-300">Not a web address: it starts with http:// or https://</p>}
                  </div>
                  <input value={r.title} placeholder="What it is (optional)" aria-label={`What link ${i + 1} is`}
                    onChange={(e) => set(r.id, { title: e.target.value })} className={`${FIELD} h-9 w-full py-0 @lg:w-56`} />
                  <button type="button" onClick={() => setRows((old) => (old.length > 1 ? old.filter((x) => x.id !== r.id) : [linkRow()]))}
                    aria-label={`Remove link ${i + 1}`} title="Remove it"
                    className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-red-500/10 hover:text-red-600 dark:hover:text-red-300">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </li>
              )
            })}
          </ol>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button type="button" onClick={() => setRows((old) => [...old, linkRow()])} disabled={rows.length >= LINKS_MAX}
              className="inline-flex h-8 items-center gap-1.5 rounded-full border border-dashed border-[var(--fd-line)] px-3 text-xs font-medium text-zinc-300 transition-colors hover:border-violet-500/40 hover:text-zinc-100">
              <Plus className="h-3.5 w-3.5" />Add a link
            </button>
            <button type="button" onClick={() => setBulk('')}
              className="inline-flex h-8 items-center gap-1.5 rounded-full border border-[var(--fd-line)] px-3 text-xs font-medium text-zinc-300 transition-colors hover:border-violet-500/40 hover:text-zinc-100">
              <ClipboardPaste className="h-3.5 w-3.5" />Paste many
            </button>
          </div>
        </>
      ) : (
        <div>
          <label className="sr-only" htmlFor="ad-links-bulk">Paste links</label>
          <textarea id="ad-links-bulk" value={bulk} onChange={(e) => setBulk(e.target.value)} autoFocus
            placeholder="Paste links: one per line, or any text with web addresses in it (an email, a document, a list)."
            className={`${FIELD} min-h-[12rem] resize-y font-mono text-[12.5px] leading-relaxed`} />
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span className="text-[11px] text-zinc-500">{urlsIn(bulk).length} link{urlsIn(bulk).length === 1 ? '' : 's'} found</span>
            <button type="button" onClick={() => setBulk(null)} className={`${QUIET} ml-auto h-8 px-3 text-xs`}>Back</button>
            <button type="button" disabled={!urlsIn(bulk).length} onClick={() => { take(urlsIn(bulk)); setBulk(null) }} className={`${PRIMARY} h-8 px-3 text-xs`}>
              <Plus className="h-3.5 w-3.5" />Take them
            </button>
          </div>
        </div>
      )}
      {bulk === null && (
        <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-[var(--fd-line)] pt-3">
          <label className="sr-only" htmlFor="ad-links-name">Name the list</label>
          <input id="ad-links-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Name the list (optional)" maxLength={120}
            className={`${FIELD} h-9 min-w-0 flex-1 py-0 @lg:max-w-xs`} />
          <span className="text-[11px] tabular-nums text-zinc-500">
            {filled.length} link{filled.length === 1 ? '' : 's'}{bad.length ? ` · ${bad.length} to fix` : ''}
          </span>
          <button onClick={() => void add()} disabled={busy || !ready} className={`${PRIMARY} ml-auto`}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Link2 className="h-4 w-4" />}
            {filled.length > 1 ? `Add ${filled.length} links` : 'Add the link'}
          </button>
        </div>
      )}
      {error && <p role="alert" className="mt-2 text-xs text-red-700 dark:text-red-300">{error}</p>}
    </div>
  )
}

function useChecks() {
  const [picked, setPicked] = useState<string[]>([])
  const toggle = (key: string) => setPicked((old) => (old.includes(key) ? old.filter((k) => k !== key) : [...old, key]))
  return { picked, toggle, clear: () => setPicked([]), set: setPicked }
}

function AgentsSource({ onAdded }: { onAdded: (list: Staged[]) => void }) {
  const [agents, setAgents] = useState<ContextAgent[] | null>(null)
  const [agent, setAgent] = useState('')
  const [files, setFiles] = useState<Record<string, { files: AgentFile[]; more: boolean } | undefined>>({})
  const [filter, setFilter] = useState('')
  const checks = useChecks()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    contextApi.agents().then((list) => { setAgents(list); if (list[0]) setAgent(list[0].id) }).catch((e) => setError(errorOf(e)))
  }, [])
  useEffect(() => {
    if (!agent || files[agent]) return
    contextApi.agentFiles(agent).then((got) => setFiles((old) => ({ ...old, [agent]: got }))).catch((e) => setError(errorOf(e)))
  }, [agent, files])

  const here = files[agent]
  const shown = useMemo(() => {
    const words = filter.trim().toLowerCase()
    const list = here?.files ?? []
    return words ? list.filter((f) => f.path.toLowerCase().includes(words)) : list
  }, [here, filter])

  const add = async () => {
    setBusy(true); setError('')
    try {
      onAdded(await contextApi.attachAgentFiles(agent, checks.picked))
      checks.clear()
    } catch (e) {
      setError(errorOf(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <SourceHead title="Files your agents saved" hint="Copies: the agent keeps its own. Only files in its folder are here, and never ones that hold keys." />
      {error && <ErrorText text={error} />}
      {agents === null && !error && <Loading />}
      {agents?.length === 0 && <Empty icon={Bot} title="You have no agents here">Files can come from agents of your own on this deck.</Empty>}
      {!!agents?.length && (
        <>
          <div className="mb-3 flex gap-1.5 overflow-x-auto pb-1" role="tablist" aria-label="Agent">
            {agents.map((a) => (
              <button key={a.id} role="tab" aria-selected={a.id === agent} onClick={() => { setAgent(a.id); setFilter(''); checks.clear() }}
                className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${a.id === agent
                  ? 'border-violet-500/50 bg-violet-500/10 text-zinc-100' : 'border-[var(--fd-line)] text-zinc-400 hover:text-zinc-200'}`}>
                <Bot className="h-3.5 w-3.5" />{a.name}
              </button>
            ))}
          </div>
          {!here && !error && <Loading />}
          {here && (here.files.length ? (
            <FilePicker
              rows={shown.map((f) => ({ key: f.path, name: f.name, folder: f.path.includes('/') ? f.path.slice(0, f.path.lastIndexOf('/')) : '', size: f.size, mtime: f.modified }))}
              filter={filter} onFilter={setFilter} checks={checks} busy={busy} onAdd={() => void add()}
              note={here.more ? `Only the newest ${here.files.length} are listed. Find an older one by its name.` : ''} />
          ) : <Empty icon={FileText} title="No files yet">This agent hasn't saved anything in its folder.</Empty>)}
        </>
      )}
    </div>
  )
}

function folderIcon(p: VFSProject): LucideIcon {
  if (p.shared) return Share2
  if (p.kind === 'gdrive' || p.drive) return Cloud
  if (p.kind === 'link' || p.link_path) return FolderSymlink
  return Folder
}

function DeckSource({ onAdded }: { onAdded: (list: Staged[]) => void }) {
  const [folders, setFolders] = useState<VFSProject[] | null>(null)
  const [open, setOpen] = useState<VFSProject | null>(null)
  const [files, setFiles] = useState<{ files: FolderHit[]; truncated: boolean } | null>(null)
  const [find, setFind] = useState('')
  const [filter, setFilter] = useState('')
  const checks = useChecks()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    vfsApi.projects().then(setFolders).catch((e) => setError(errorOf(e)))
  }, [])

  const owner = (p: VFSProject) => (p.shared ? p.owner_id || '' : '')
  const openFolder = (p: VFSProject) => {
    setOpen(p); setFiles(null); setFilter(''); checks.clear(); setError('')
    vfsApi.search('', { project: p.name, owner: owner(p) }, 500)
      .then((got) => setFiles({ files: got.files, truncated: got.truncated }))
      .catch((e) => setError(errorOf(e)))
  }

  const shownFolders = useMemo(() => {
    const words = find.trim().toLowerCase()
    const list = (folders ?? []).filter((p) => !p.missing)
    return words ? list.filter((p) => `${p.name} ${p.title ?? ''} ${p.owner_name ?? ''}`.toLowerCase().includes(words)) : list
  }, [folders, find])

  const shown = useMemo(() => {
    const words = filter.trim().toLowerCase()
    const list = files?.files ?? []
    return words ? list.filter((f) => f.path.toLowerCase().includes(words)) : list
  }, [files, filter])

  const add = async () => {
    if (!open) return
    setBusy(true); setError('')
    try {
      onAdded(await contextApi.deckFiles(open.name, owner(open), checks.picked))
      checks.clear()
    } catch (e) {
      setError(errorOf(e))
    } finally {
      setBusy(false)
    }
  }

  if (open) {
    return (
      <div>
        <BackLine onBack={() => { setOpen(null); setError('') }} label="All folders" title={open.title || open.name} />
        {error && <ErrorText text={error} />}
        {files === null && !error && <Loading />}
        {files && (files.files.length ? (
          <FilePicker
            rows={shown.map((f) => ({ key: f.path, name: f.name, folder: f.path.includes('/') ? f.path.slice(0, f.path.lastIndexOf('/')) : '', size: f.size, mtime: f.mtime }))}
            filter={filter} onFilter={setFilter} checks={checks} busy={busy} onAdd={() => void add()}
            note={files.truncated ? `Only the newest ${files.files.length} are listed. Find an older one by its name.` : ''} />
        ) : <Empty icon={Folder} title="This folder is empty" />)}
      </div>
    )
  }
  return (
    <div>
      <SourceHead title="Files from Flight Deck" hint="Your folders, linked and mounted ones, and folders others shared with you. Copies go; the folder keeps its own." />
      {error && <ErrorText text={error} />}
      {folders === null && !error && <Loading />}
      {folders?.length === 0 && <Empty icon={Folder} title="No folders yet">Folders your agents and you make in Flight Deck show up here.</Empty>}
      {!!folders?.length && (
        <>
          {folders.length > 6 && <SearchBox value={find} onChange={setFind} placeholder="Find a folder" />}
          <ul className="grid gap-2 @lg:grid-cols-2" aria-label="Folders">
            {shownFolders.map((p) => {
              const Icon = folderIcon(p)
              return (
                <li key={`${p.owner_id ?? ''}/${p.name}`}>
                  <button onClick={() => openFolder(p)}
                    className="group flex w-full items-center gap-3 rounded-xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] px-3 py-2.5 text-left transition-all hover:-translate-y-px hover:border-violet-500/40 hover:shadow-sm">
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-zinc-800/80 text-zinc-400 group-hover:text-violet-600 dark:group-hover:text-violet-300">
                      <Icon className="h-4 w-4" />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium text-zinc-200">{p.title || p.name}</span>
                      <span className="block truncate text-[11px] text-zinc-500">
                        {[p.shared && `from ${p.owner_name || p.owner_email || 'someone'}`, `${p.files.toLocaleString()} file${p.files === 1 ? '' : 's'}`, p.bytes ? sizeText(p.bytes) : '']
                          .filter(Boolean).join(' · ')}
                      </span>
                    </span>
                    <ChevronRight className="h-4 w-4 shrink-0 text-zinc-600 transition-transform group-hover:translate-x-0.5" />
                  </button>
                </li>
              )
            })}
          </ul>
          {!shownFolders.length && <p className="py-6 text-center text-xs text-zinc-500">Nothing matches.</p>}
        </>
      )}
    </div>
  )
}

const laneLabel = (c: Pick<Conversation, 'agent' | 'lane'>) => (c.lane && c.lane !== 'A' ? `${c.agent} · chat ${c.lane}` : c.agent)

function ConversationSource({ onAdded }: { onAdded: (list: Staged[]) => void }) {
  const [chats, setChats] = useState<{ list: Conversation[]; kept: boolean } | null>(null)
  const [chat, setChat] = useState<Conversation | null>(null)
  const [sessions, setSessions] = useState<ConversationSession[] | null>(null)
  const [session, setSession] = useState<ConversationSession | null>(null)
  const [msgs, setMsgs] = useState<{ messages: ConversationMessage[]; more: boolean } | null>(null)
  // The range, as indexes into msgs: all of it to start with; a tap starts a new one, the next ends it.
  const [range, setRange] = useState<{ a: number; b: number; ending: boolean }>({ a: 0, b: 0, ending: false })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    contextApi.conversations().then((r) => setChats({ list: r.conversations, kept: r.kept })).catch((e) => setError(errorOf(e)))
  }, [])

  const openSession = (c: Conversation, s: ConversationSession) => {
    setSession(s); setMsgs(null); setError('')
    contextApi.messages(c.id, s.id).then((got) => {
      setMsgs(got)
      setRange({ a: 0, b: Math.max(0, got.messages.length - 1), ending: false })
    }).catch((e) => setError(errorOf(e)))
  }
  const openChat = (c: Conversation) => {
    setChat(c); setSessions(null); setSession(null); setMsgs(null); setError('')
    contextApi.sessions(c.id).then((list) => {
      setSessions(list)
      if (list.length === 1) openSession(c, list[0])
    }).catch((e) => setError(errorOf(e)))
  }
  const back = () => {
    setError('')
    if (session && sessions && sessions.length > 1) { setSession(null); setMsgs(null) } else { setChat(null); setSession(null); setSessions(null); setMsgs(null) }
  }

  const list = msgs?.messages ?? []
  const lo = Math.min(range.a, range.b)
  const hi = Math.max(range.a, range.b)
  const tap = (i: number) => setRange((r) => (r.ending ? { a: r.a, b: i, ending: false } : { a: i, b: i, ending: true }))

  const add = async () => {
    if (!chat || !session || !list.length) return
    setBusy(true); setError('')
    try {
      onAdded([await contextApi.conversation(chat.id, session.id, list[lo].id, list[hi].id)])
      setChat(null); setSession(null); setSessions(null); setMsgs(null)
    } catch (e) {
      setError(errorOf(e))
    } finally {
      setBusy(false)
    }
  }

  if (!chat) {
    return (
      <div>
        <SourceHead title="A conversation with an agent" hint="Your messages and the agent's final answers, written out as a file. Its steps and tool calls are left out." />
        {error && <ErrorText text={error} />}
        {chats === null ? (!error && <Loading />)
          : !chats.kept ? <Empty icon={MessagesSquare} title="Conversations aren't kept here">Flight Deck keeps your chats only when sign-in is on.</Empty>
            : !chats.list.length ? <Empty icon={MessagesSquare} title="No conversations yet">Chat with one of your agents, and it can be handed on from here.</Empty>
              : (
                <Rows label="Conversations">
                  {chats.list.map((c) => (
                    <Row key={c.id} onClick={() => openChat(c)} icon={Bot} title={laneLabel(c)} hint={c.updated_at ? `Last message ${when(c.updated_at)}` : ''} />
                  ))}
                </Rows>
              )}
      </div>
    )
  }
  if (!session) {
    return (
      <div>
        <BackLine onBack={back} label="All conversations" title={laneLabel(chat)} />
        {error && <ErrorText text={error} />}
        {sessions === null ? (!error && <Loading />) : !sessions.length ? <Empty icon={MessagesSquare} title="Nothing said yet" /> : (
          <Rows label="Sessions">
            {sessions.map((s) => (
              <Row key={s.id} onClick={() => openSession(chat, s)} icon={MessagesSquare} title={s.title}
                hint={[s.started_at && when(s.started_at), s.count !== null && `${s.count} messages`].filter(Boolean).join(' · ')} />
            ))}
          </Rows>
        )}
      </div>
    )
  }
  return (
    <div>
      <BackLine onBack={back} label={sessions && sessions.length > 1 ? 'Sessions' : 'All conversations'} title={`${laneLabel(chat)} · ${session.title}`} />
      {error && <ErrorText text={error} />}
      {msgs === null ? (!error && <Loading />) : !list.length ? (
        <Empty icon={MessagesSquare} title="Nothing to hand on here">This session has no messages of yours or final answers.</Empty>
      ) : (
        <>
          <div className="sticky -top-5 z-[1] -mx-5 mb-3 flex flex-wrap items-center gap-2 border-b border-[var(--fd-line)] bg-[var(--fd-surface,#09090b)]/95 px-5 py-2.5 backdrop-blur">
            <span aria-live="polite" className="text-xs text-zinc-500">
              {range.ending ? 'Now tap the last message.' : `${hi - lo + 1} of ${list.length} chosen. Tap the first message, then the last, to choose fewer.`}
            </span>
            {(lo > 0 || hi < list.length - 1 || range.ending) && (
              <button type="button" onClick={() => setRange({ a: 0, b: list.length - 1, ending: false })} className="text-xs font-medium text-violet-700 hover:underline dark:text-violet-300">All of it</button>
            )}
            <button onClick={() => void add()} disabled={busy || range.ending} className={`${PRIMARY} ml-auto h-8 px-3 text-xs`}>
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <MessagesSquare className="h-3.5 w-3.5" />}
              Add {hi - lo + 1} message{hi === lo ? '' : 's'}
            </button>
          </div>
          <ol className="space-y-1.5" aria-label="Messages">
            {list.map((m, i) => {
              const on = i >= lo && i <= hi
              const edge = i === lo || i === hi
              return (
                <li key={m.id}>
                  <button type="button" onClick={() => tap(i)} aria-pressed={on}
                    className={`w-full rounded-xl border px-3 py-2 text-left transition-all ${on
                      ? `${edge ? 'border-violet-500/60' : 'border-violet-500/25'} bg-violet-500/[0.07]`
                      : 'border-[var(--fd-line)] opacity-55 hover:opacity-100'}`}>
                    <span className="flex items-center gap-2 text-[11px] text-zinc-500">
                      <span className={`font-semibold ${m.role === 'user' ? 'text-zinc-300' : 'text-violet-700 dark:text-violet-300'}`}>{m.role === 'user' ? 'You' : chat.agent}</span>
                      <span>{when(m.at)}</span>
                      {on && edge && <Check className="ml-auto h-3.5 w-3.5 text-violet-600 dark:text-violet-400" />}
                    </span>
                    <span className="mt-0.5 line-clamp-3 whitespace-pre-line break-words text-sm text-zinc-200">{m.text}{m.chars > m.text.length ? '…' : ''}</span>
                  </button>
                </li>
              )
            })}
          </ol>
          {msgs.more && <p className="mt-2 text-xs text-zinc-500">Only the first {list.length} messages of this session are listed.</p>}
        </>
      )}
    </div>
  )
}

// ── Small parts ──

interface PickRow { key: string; name: string; folder: string; size: number; mtime: number }

function FilePicker({ rows, filter, onFilter, checks, busy, onAdd, note }: {
  rows: PickRow[]
  filter: string
  onFilter: (v: string) => void
  checks: ReturnType<typeof useChecks>
  busy: boolean
  onAdd: () => void
  note: string
}) {
  const n = checks.picked.length
  const allShown = rows.length > 0 && rows.every((r) => checks.picked.includes(r.key))
  return (
    <div>
      <div className="sticky -top-5 z-[1] -mx-5 mb-3 flex flex-wrap items-center gap-2 border-b border-[var(--fd-line)] bg-[var(--fd-surface,#09090b)]/95 px-5 py-2.5 backdrop-blur">
        <div className="min-w-0 flex-1"><SearchBox value={filter} onChange={onFilter} placeholder="Find by name or folder" inline /></div>
        <button onClick={() => checks.set(allShown ? checks.picked.filter((k) => !rows.some((r) => r.key === k)) : [...new Set([...checks.picked, ...rows.slice(0, 30).map((r) => r.key)])])}
          disabled={!rows.length} className="text-xs font-medium text-violet-700 hover:underline disabled:opacity-40 dark:text-violet-300">
          {allShown ? 'Clear' : 'Choose all'}
        </button>
        <button onClick={onAdd} disabled={busy || !n} className={`${PRIMARY} h-8 px-3 text-xs`}>
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
          {n ? `Add ${n} file${n === 1 ? '' : 's'}` : 'Add'}
        </button>
      </div>
      <ul className="divide-y divide-[var(--fd-line)] overflow-hidden rounded-xl border border-[var(--fd-line)]" aria-label="Files">
        {rows.map((f) => {
          const on = checks.picked.includes(f.key)
          return (
            <li key={f.key}>
              <button type="button" role="checkbox" aria-checked={on} onClick={() => checks.toggle(f.key)}
                className={`flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors ${on ? 'bg-violet-500/10' : 'hover:bg-zinc-900/60'}`}>
                <Tick on={on} />
                <FileGlyph name={f.name} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-zinc-200">{f.name}</span>
                  {f.folder && <span className="block truncate text-[11px] text-zinc-500">{f.folder}</span>}
                </span>
                <span className="shrink-0 text-right text-[11px] tabular-nums text-zinc-500">{sizeText(f.size)}<br />{agoText(f.mtime)}</span>
              </button>
            </li>
          )
        })}
        {!rows.length && <li className="px-3 py-6 text-center text-xs text-zinc-500">Nothing matches.</li>}
      </ul>
      {note && <p className="mt-2 text-xs text-zinc-500">{note}</p>}
    </div>
  )
}

function SearchBox({ value, onChange, placeholder, inline = false }: { value: string; onChange: (v: string) => void; placeholder: string; inline?: boolean }) {
  return (
    <div className={`relative ${inline ? '' : 'mb-3'}`}>
      <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
      <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} aria-label={placeholder}
        className={`${FIELD} h-8 py-0 !pl-9 text-[13px]`} />
    </div>
  )
}

function Loading() {
  return <div className="flex justify-center py-12 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
}

function ErrorText({ text }: { text: string }) {
  return <p role="alert" className="mb-3 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">{text}</p>
}

function Empty({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-2xl border border-dashed border-zinc-800 px-6 py-12 text-center">
      <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-zinc-800/70 text-zinc-500"><Icon className="h-5 w-5" /></span>
      <p className="text-sm font-medium text-zinc-300">{title}</p>
      {children && <p className="max-w-sm text-xs text-zinc-500">{children}</p>}
    </div>
  )
}

function Tick({ on }: { on: boolean }) {
  return (
    <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors ${on ? 'border-violet-500 bg-violet-600 text-white' : 'border-zinc-600'}`}>
      {on && <Check className="h-3 w-3" />}
    </span>
  )
}

function Rows({ label, children }: { label: string; children: ReactNode }) {
  return <ul aria-label={label} className="divide-y divide-[var(--fd-line)] overflow-hidden rounded-xl border border-[var(--fd-line)]">{children}</ul>
}

function Row({ title, hint, icon: Icon, onClick }: { title: string; hint?: string; icon: LucideIcon; onClick: () => void }) {
  return (
    <li>
      <button type="button" onClick={onClick} className="group flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors hover:bg-zinc-900/60">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800/80 text-zinc-400"><Icon className="h-4 w-4" /></span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm text-zinc-200">{title}</span>
          {hint && <span className="block truncate text-[11px] text-zinc-500">{hint}</span>}
        </span>
        <ChevronRight className="h-4 w-4 shrink-0 text-zinc-600 transition-transform group-hover:translate-x-0.5" />
      </button>
    </li>
  )
}

function BackLine({ onBack, label, title }: { onBack: () => void; label: string; title: string }) {
  return (
    <div className="mb-4 flex min-w-0 items-center gap-2">
      <button type="button" onClick={onBack} className="inline-flex shrink-0 items-center gap-1 rounded-full border border-[var(--fd-line)] px-2.5 py-1 text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
        <ArrowLeft className="h-3.5 w-3.5" />{label}
      </button>
      <span className="min-w-0 truncate text-sm font-semibold text-zinc-200">{title}</span>
    </div>
  )
}
