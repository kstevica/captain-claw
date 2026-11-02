import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, ClipboardEvent, DragEvent } from 'react'
import {
  AlertTriangle, ArrowRight, CalendarClock, CheckCircle2, ChevronRight, ClipboardCheck, File as FileIcon, FileSpreadsheet,
  FileText, Hourglass, Image as ImageIcon, ListChecks, Loader2, Mail, MessageSquareText, Paperclip, Presentation, Sparkles,
  Telescope, Trash2, Upload, X, type LucideIcon,
} from 'lucide-react'
import { forgeApi, type ForgeLimits, type ForgePlanRow } from '../../services/agentForge'
import { fmtRelative } from '../../services/schedulerApi'
import { useIsMobile } from '../../hooks/useMediaQuery'
import { ErrorNote, TeamSize } from './forgeParts'
import { errorOf, fmtSize } from './forgeUtil'

// Step one: a blank page is the hardest part, so it starts with a sentence to
// finish, a few ideas to take, and the plan you left waiting if there is one.

const MAX_FILES = 10
const MAX_FILE_BYTES = 25 * 1024 * 1024

const STARTERS: { icon: LucideIcon; label: string; text: string }[] = [
  {
    icon: FileSpreadsheet, label: 'Quarterly fund reports',
    text: "Every quarter we go through our funds' reports, work out what they really invested in, and fill in a standard Excel for Risk. It takes days, and most of it is finding and copying numbers.",
  },
  {
    icon: Telescope, label: 'Watch our competitors',
    text: 'Keep an eye on our competitors: new products, prices, hires and news. Every Monday morning, tell me what changed and why it matters to us.',
  },
  {
    icon: Mail, label: 'Client email',
    text: 'Go through incoming client email, sort what needs an answer, and draft the replies for me to approve. Nothing goes out without me.',
  },
  {
    icon: ListChecks, label: 'Meeting follow-ups',
    text: 'After each meeting, turn the notes into action items with owners and dates, and remind people before they are due.',
  },
]

const HOW: { icon: LucideIcon; title: string; text: string }[] = [
  { icon: MessageSquareText, title: 'You describe it', text: 'In your own words, with your documents.' },
  { icon: ClipboardCheck, title: 'You check the plan', text: 'Who does what. Nothing exists until you approve.' },
  { icon: CalendarClock, title: 'You choose the rest', text: 'What runs by itself, and what software to build.' },
]

function iconOf(name: string): LucideIcon {
  if (/\.(xlsx?|csv|tsv)$/i.test(name)) return FileSpreadsheet
  if (/\.(pptx?|key)$/i.test(name)) return Presentation
  if (/\.(png|jpe?g|gif|webp|heic|tiff?)$/i.test(name)) return ImageIcon
  if (/\.(pdf|docx?|txt|md|rtf)$/i.test(name)) return FileText
  return FileIcon
}

const MOD = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘' : 'Ctrl'

export function ForgeAsk({ onStarted }: { onStarted: (id: string) => void }) {
  const [request, setRequest] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [recent, setRecent] = useState<ForgePlanRow[]>([])
  const [limits, setLimits] = useState<ForgeLimits | null>(null)
  const [cap, setCap] = useState(0)
  const [dragging, setDragging] = useState(false)
  const depth = useRef(0)
  const picker = useRef<HTMLInputElement>(null)
  const box = useRef<HTMLTextAreaElement>(null)
  const { isMobile } = useIsMobile()

  useEffect(() => { forgeApi.list().then(setRecent).catch(() => {}) }, [])
  useEffect(() => { forgeApi.limits().then(setLimits).catch(() => {}) }, [])

  const add = (list: File[]) => {
    if (!list.length) return
    const tooBig = list.find((f) => f.size > MAX_FILE_BYTES)
    if (tooBig) { setErr(`${tooBig.name} is over 25 MB, so it can't come along. Try a smaller copy.`); return }
    setErr('')
    setFiles((old) => {
      const names = new Set(old.map((f) => f.name))
      const next = [...old, ...list.filter((f) => !names.has(f.name))]
      if (next.length > MAX_FILES) setErr(`You can attach up to ${MAX_FILES} files.`)
      return next.slice(0, MAX_FILES)
    })
  }
  const onPaste = (e: ClipboardEvent) => {
    const list = Array.from(e.clipboardData?.files ?? [])
    if (list.length) { e.preventDefault(); add(list) }
  }
  const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes('Files')
  const onDrop = (e: DragEvent) => {
    depth.current = 0
    setDragging(false)
    const list = Array.from(e.dataTransfer?.files ?? [])
    if (list.length) { e.preventDefault(); add(list) }
  }
  const onPick = (e: ChangeEvent<HTMLInputElement>) => { add(Array.from(e.target.files ?? [])); e.target.value = '' }

  const take = (text: string) => {
    setRequest(text)
    box.current?.focus()
  }
  const forget = async (id: string) => {
    try {
      await forgeApi.forget(id)
      setRecent((old) => old.filter((r) => r.id !== id))
    } catch (e) {
      setErr(errorOf(e))
    }
  }
  const submit = async () => {
    if (!request.trim() || busy) return
    setBusy(true)
    setErr('')
    try {
      onStarted((await forgeApi.start(request.trim(), files, cap)).id)
    } catch (e) {
      setErr(errorOf(e))
      setBusy(false)
    }
  }

  const waiting = recent.find((r) => r.status === 'awaiting_plan')
  const others = recent.filter((r) => r !== waiting).slice(0, 5)

  return (
    <div
      className="relative min-h-0 flex-1 overflow-y-auto"
      onDragEnter={(e) => { if (hasFiles(e)) { depth.current += 1; setDragging(true) } }}
      onDragLeave={() => { depth.current = Math.max(0, depth.current - 1); if (!depth.current) setDragging(false) }}
      onDragOver={(e) => { if (hasFiles(e)) e.preventDefault() }}
      onDrop={onDrop}
    >
      {dragging && (
        <div className="fd-fade pointer-events-none absolute inset-3 z-10 flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-violet-500/60 bg-zinc-950/90 text-center">
          <Upload className="h-7 w-7 text-violet-500" />
          <p className="mt-2 text-sm font-medium text-zinc-100">Drop to attach</p>
          <p className="mt-0.5 text-xs text-zinc-500">Captain reads them before planning.</p>
        </div>
      )}
      <div className="mx-auto w-full max-w-3xl space-y-5 px-4 py-6 wide:py-10">
        {waiting && (
          <button onClick={() => onStarted(waiting.id)} className="fd-rise group flex w-full items-center gap-3 rounded-2xl border border-amber-500/30 bg-amber-500/5 p-3 text-left transition-colors hover:border-amber-500/50">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-amber-500/15 text-amber-700 dark:text-amber-300"><Hourglass className="h-[18px] w-[18px]" /></span>
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-medium text-zinc-100">A team plan is waiting for you</span>
              <span className="block truncate text-xs text-zinc-500">{waiting.title}</span>
            </span>
            <ChevronRight className="h-4 w-4 shrink-0 text-zinc-500 transition-transform group-hover:translate-x-0.5" />
          </button>
        )}

        <div className="fd-rise">
          <h2 className="text-2xl font-semibold tracking-tight text-zinc-100 wide:text-3xl">What should your agents take care of?</h2>
          <p className="mt-1.5 text-sm leading-relaxed text-zinc-500">Say it the way you'd tell a new colleague. Captain works out who you need.</p>
        </div>

        <div className="fd-rise rounded-2xl border border-zinc-700 bg-zinc-900/40 shadow-sm transition-colors focus-within:border-violet-500/60 focus-within:ring-2 focus-within:ring-violet-500/15">
          <textarea
            ref={box}
            autoFocus={!isMobile}
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            onPaste={onPaste}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); void submit() } }}
            disabled={busy}
            aria-label="What do you need?"
            placeholder="Every quarter we go through our funds' reports and fill in a standard Excel for Risk. It takes days…"
            className="block max-h-[45vh] min-h-[8.5rem] w-full resize-none bg-transparent px-4 pt-4 pb-2 text-base leading-relaxed text-zinc-100 placeholder-zinc-600 [field-sizing:content] focus:outline-none disabled:opacity-60 wide:text-[15px]"
          />
          {files.length > 0 && (
            <ul className="flex flex-wrap gap-1.5 px-3 pb-2" aria-label="Attached files">
              {files.map((f) => {
                const Icon = iconOf(f.name)
                return (
                  <li key={f.name} className="fd-rise flex max-w-full items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-950 py-1 pl-2 pr-1 text-xs">
                    <Icon className="h-3.5 w-3.5 shrink-0 text-violet-600 dark:text-violet-400" />
                    <span className="truncate text-zinc-300">{f.name}</span>
                    <span className="shrink-0 tabular-nums text-zinc-600">{fmtSize(f.size)}</span>
                    <button onClick={() => setFiles((old) => old.filter((x) => x !== f))} disabled={busy} aria-label={`Remove ${f.name}`} className="shrink-0 rounded p-0.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                      <X className="h-3 w-3" />
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
          <div className="flex items-center gap-2 border-t border-zinc-800/80 p-2">
            <button type="button" onClick={() => picker.current?.click()} disabled={busy} className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
              <Paperclip className="h-4 w-4" /> Attach
            </button>
            <input ref={picker} type="file" multiple hidden onChange={onPick} />
            <span className="hidden min-w-0 truncate text-xs text-zinc-600 wide:block">Reports, templates, procedures. Or drop them here.</span>
            <button onClick={submit} disabled={!request.trim() || busy} className="ml-auto inline-flex shrink-0 items-center gap-1.5 rounded-xl bg-violet-600 px-3.5 py-2 text-sm font-medium text-white shadow-sm shadow-violet-900/20 transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
              {busy ? 'Starting…' : 'Plan my team'}
              {!busy && !isMobile && <kbd className="ml-1 rounded bg-white/15 px-1 font-sans text-[10px] font-medium">{MOD}↵</kbd>}
            </button>
          </div>
        </div>

        {limits && limits.seats > 0 && (
          <div className="fd-rise flex flex-wrap items-center gap-x-3 gap-y-1.5">
            <TeamSize value={cap} largest={Math.min(limits.seats, limits.max_team)} suggested={limits.default_team} disabled={busy} onChange={setCap} />
            <p className="min-w-0 flex-1 basis-56 text-xs leading-relaxed text-zinc-500">
              A limit, not a target: Captain combines work that belongs together. For one agent per item, say so above.
              {limits.seats < limits.max_team && ` Your plan has room for ${limits.seats} more.`}
            </p>
          </div>
        )}

        {err && <ErrorNote>{err}</ErrorNote>}

        {!request.trim() && (
          <div className="fd-rise">
            <p className="mb-2 text-xs font-medium text-zinc-500">Or start from one of these</p>
            <div className="flex flex-wrap gap-2">
              {STARTERS.map((s) => {
                const Icon = s.icon
                return (
                  <button key={s.label} type="button" onClick={() => take(s.text)} className="inline-flex items-center gap-1.5 rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-1.5 text-sm text-zinc-300 transition-colors hover:border-violet-500/40 hover:bg-zinc-900/70">
                    <Icon className="h-3.5 w-3.5 text-zinc-500" /> {s.label}
                  </button>
                )
              })}
            </div>
          </div>
        )}

        <ol className="fd-rise grid gap-2 wide:grid-cols-3" aria-label="How it works">
          {HOW.map((h, i) => {
            const Icon = h.icon
            return (
              <li key={h.title} className="flex items-start gap-3 rounded-2xl border border-zinc-800/70 p-3">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800/80 text-zinc-300"><Icon className="h-4 w-4" /></span>
                <span className="min-w-0">
                  <span className="block text-sm font-medium text-zinc-200"><span className="text-zinc-600">{i + 1}.</span> {h.title}</span>
                  <span className="block text-xs leading-relaxed text-zinc-500">{h.text}</span>
                </span>
              </li>
            )
          })}
        </ol>

        {others.length > 0 && (
          <section aria-label="Earlier plans">
            <p className="mb-2 text-xs font-medium text-zinc-500">Earlier plans</p>
            <ul className="divide-y divide-zinc-800/70 rounded-2xl border border-zinc-800 bg-zinc-900/40">
              {others.map((r) => <PlanRow key={r.id} row={r} onOpen={() => onStarted(r.id)} onForget={() => forget(r.id)} />)}
            </ul>
          </section>
        )}
      </div>
    </div>
  )
}

function PlanRow({ row, onOpen, onForget }: { row: ForgePlanRow; onOpen: () => void; onForget: () => void }) {
  const stopped = row.status === 'error' || row.stalled
  const working = !stopped && row.status !== 'ready'
  const Icon = stopped ? AlertTriangle : working ? Loader2 : CheckCircle2
  const tone = stopped ? 'bg-red-500/15 text-red-700 dark:text-red-300' : working ? 'bg-sky-500/15 text-sky-600 dark:text-sky-300' : 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
  const label = stopped ? 'Stopped' : working ? 'Working' : `${row.created} agent${row.created === 1 ? '' : 's'}`
  const when = Date.parse(row.updated_at)
  return (
    <li className="group flex items-center gap-1 pr-2">
      <button onClick={onOpen} className="flex min-w-0 flex-1 items-center gap-3 px-3 py-2.5 text-left">
        <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${tone}`}><Icon className={`h-4 w-4 ${working ? 'animate-spin' : ''}`} /></span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-zinc-200">{row.title || 'Untitled'}</span>
          <span className="block truncate text-xs text-zinc-500">{label}{Number.isNaN(when) ? '' : ` · ${fmtRelative(when / 1000)}`}</span>
        </span>
        <ArrowRight className="h-4 w-4 shrink-0 text-zinc-600 transition-transform group-hover:translate-x-0.5 group-hover:text-zinc-300" />
      </button>
      <button onClick={onForget} aria-label={`Forget the plan: ${row.title || 'Untitled'}`} title="Forget this plan. Any agents, tasks and apps it made stay." className="shrink-0 rounded-lg p-1.5 text-zinc-600 transition-colors hover:bg-zinc-800 hover:text-zinc-300">
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </li>
  )
}
