import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertTriangle, Bot, Check, CheckCircle2, CircleDashed, Copy, Download, Eye, FileSpreadsheet, FileText, FileUp,
  Inbox, Languages, ListTodo, Loader2, MessageSquarePlus, PanelRightClose, PanelRightOpen, Presentation, Send,
  Trash2, X, XCircle,
} from 'lucide-react'
import {
  roomsApi, type Delivery, type DeliverySummary, type GrammarReport, type RoomTask,
} from '../../services/roomsApi'
import { untrustedUrlTransform } from '../../utils/untrustedContent'
import { Button, EmptyState, Notice, Pill, Segmented } from '../console/parts'
import { Marked, RoomDialog, SearchField, TokenChip } from './roomParts'
import { count, errorText, findAll } from './roomText'

// What Claude sent back to a room, with the real names put back here for the
// owner alone, and the tasks the owner leaves for Claude. Claude's markdown is
// someone else's writing, and once restored it holds real names: no pictures
// and no live links (a picture Claude planted would fetch an address with the
// names in it; a link would send them on with one careless click).

function when(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

function InertLink({ href, children }: { href?: string; children?: ReactNode }) {
  return <span className="underline decoration-dotted" title={href ? `Link (not opened): ${href}` : undefined}>{children}</span>
}

function InertPicture({ alt }: { alt?: string }) {
  return <span className="text-zinc-500">[picture{alt ? `: ${alt}` : ''}]</span>
}

const KIND_LABEL: Record<string, string> = { workbook: 'Excel', deck: 'PowerPoint', document: 'Word' }
const EXT_LABEL: Record<string, string> = {
  xlsx: 'Excel', pptx: 'PowerPoint', docx: 'Word', md: 'Markdown', markdown: 'Markdown', csv: 'CSV', tsv: 'TSV',
  txt: 'Text', json: 'JSON',
}
const DROP_ACCEPT = '.docx,.xlsx,.pptx,.md,.markdown,.txt,.csv,.tsv,.json'

function kindLabelOf(r: { kind?: string; file?: string }): string {
  if (r.kind === 'file') return EXT_LABEL[r.file || ''] || (r.file || 'File').toUpperCase()
  return KIND_LABEL[r.kind || ''] || ''
}

function kindIcon(r: { kind?: string; file?: string }): typeof FileText {
  const label = kindLabelOf(r)
  return label === 'Excel' || label === 'CSV' || label === 'TSV' ? FileSpreadsheet : label === 'PowerPoint' ? Presentation : FileText
}

function KindIcon({ r, className }: { r: { kind?: string; file?: string }; className?: string }) {
  const label = kindLabelOf(r)
  if (label === 'Excel' || label === 'CSV' || label === 'TSV') return <FileSpreadsheet className={className} />
  if (label === 'PowerPoint') return <Presentation className={className} />
  return <FileText className={className} />
}

const ASK_PHRASE = 'Check my Captain data room tasks'

// ── the Inbox ──────────────────────────────────────────────────────

export function InboxSection({ roomId, onChanged, openSeq, onOpen }: {
  roomId: string
  onChanged: () => void
  /** The delivery shown in the reader (owned by the room view, so tasks can open one too). */
  openSeq: number | null
  onOpen: (seq: number | null) => void
}) {
  const [items, setItems] = useState<DeliverySummary[] | null>(null)
  const [error, setError] = useState('')
  const [tick, setTick] = useState(0)
  const [dropping, setDropping] = useState(false)
  const [q, setQ] = useState('')

  useEffect(() => {
    roomsApi.deliveries(roomId).then((r) => setItems(r.deliveries)).catch((e) => setError(errorText(e)))
  }, [roomId, tick, openSeq])

  const waiting = !!items?.some((d) => d.report.grammar?.state === 'waiting')
  useEffect(() => {
    if (!waiting || openSeq !== null) return
    const t = setTimeout(() => setTick((n) => n + 1), 3000)
    return () => clearTimeout(t)
  }, [waiting, openSeq, items])

  const drop = async (file: File | undefined) => {
    if (!file) return
    setDropping(true); setError('')
    try {
      const r = await roomsApi.dropFile(roomId, file)
      setTick((n) => n + 1); onChanged()
      onOpen(r.seq)
    } catch (e) {
      setError(errorText(e))
    } finally {
      setDropping(false)
    }
  }

  const shown = (items || []).filter((d) => !q.trim() || d.title.toLowerCase().includes(q.trim().toLowerCase()))

  return (
    <div className="space-y-4">
      <SectionHead icon={Inbox} title="From Claude"
        hint="What Claude delivered, with the real names put back here — for you alone. Open one to read it, search it, or ask Claude to take it further."
        actions={
          <label className={`inline-flex cursor-pointer items-center gap-1.5 rounded-xl border border-zinc-700 bg-zinc-900/60 px-3.5 py-2 text-sm font-medium text-zinc-200 transition-colors hover:border-zinc-600 hover:bg-zinc-800 ${dropping ? 'pointer-events-none opacity-50' : ''}`}>
            {dropping ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileUp className="h-4 w-4" />}
            {dropping ? 'Putting the names back…' : 'Add a file Claude made'}
            <input type="file" className="sr-only" accept={DROP_ACCEPT} disabled={dropping}
              onChange={(e) => { void drop(e.target.files?.[0]); e.target.value = '' }} />
          </label>
        } />
      {error && <Notice tone="red">{error}</Notice>}
      {items && items.length > 3 && <SearchField value={q} onChange={setQ} placeholder="Find a delivery by its title" />}
      {items === null ? (
        <div className="text-sm text-zinc-500">Loading…</div>
      ) : items.length === 0 ? (
        <EmptyState icon={Inbox} title="Nothing from Claude yet">
          Ask Claude to deliver its result with the Captain Data Rooms connector, or add a file it made.
        </EmptyState>
      ) : (
        <ul className="space-y-2">
          {shown.map((d) => <DeliveryRow key={d.seq} d={d} onOpen={() => onOpen(d.seq)} />)}
          {shown.length === 0 && <li className="py-6 text-center text-sm text-zinc-500">No delivery has that in its title.</li>}
        </ul>
      )}
      {openSeq !== null && (
        <DeliveryViewer roomId={roomId} seq={openSeq} onClose={() => { onOpen(null); onChanged() }}
          onDeleted={() => { onOpen(null); setTick((n) => n + 1); onChanged() }} onOpenOther={onOpen} />
      )}
    </div>
  )
}

function DeliveryRow({ d, onOpen }: { d: DeliverySummary; onOpen: () => void }) {
  const r = d.report
  return (
    <li>
      <button type="button" onClick={onOpen}
        className="group flex w-full items-start gap-3 rounded-2xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 text-left transition-all hover:border-zinc-600 hover:bg-zinc-900/70">
        <span className={`mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${r.kind ? 'bg-sky-500/10 text-sky-700 dark:text-sky-300' : 'bg-violet-500/10 text-violet-600 dark:text-violet-300'}`}>
          <KindIcon r={r} className="h-[18px] w-[18px]" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex min-w-0 items-center gap-2">
            {!d.seen && <span className="h-2 w-2 shrink-0 rounded-full bg-violet-500" aria-label="new" />}
            <span className="truncate font-medium text-zinc-100">{d.title}</span>
          </span>
          <span className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-zinc-500">
            <span className="font-mono">#{d.seq}</span>
            <span>· {when(d.at)} · {d.app}</span>
            {r.kind && <Pill tone="sky">{kindLabelOf(r)}</Pill>}
            {r.task && <Pill tone="violet" icon={ListTodo}>answers task {r.task}</Pill>}
            {r.grammar?.state === 'waiting' && <Pill tone="violet" icon={Loader2}>case endings…</Pill>}
            {r.no_tokens && <Pill tone="amber">no tokens</Pill>}
            {r.real_values.length > 0 && <Pill tone="red" icon={AlertTriangle}>real values</Pill>}
            {r.not_in_room.length > 0 && <Pill tone="amber">{r.not_in_room.length} unknown</Pill>}
          </span>
        </span>
        <span className="hidden shrink-0 self-center text-xs text-zinc-500 sm:block">{r.restored} put back</span>
      </button>
    </li>
  )
}

// ── one delivery, to read ──────────────────────────────────────────

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

type HNode = { type: string; value?: string; tagName?: string; properties?: Record<string, unknown>; children?: HNode[] }

/** Wrap what matches *query* in the rendered markdown in <mark data-mi>. */
function rehypeMarks(query: string) {
  return () => (tree: HNode) => {
    const q = query.trim()
    if (q.length < 2) return
    const re = new RegExp(esc(q), 'giu')
    let n = 0
    const visit = (node: HNode) => {
      if (!node.children) return
      const out: HNode[] = []
      for (const child of node.children) {
        if (child.type !== 'text' || !child.value) {
          visit(child)
          out.push(child)
          continue
        }
        const v = child.value
        let last = 0
        for (const m of v.matchAll(re)) {
          const i = m.index ?? 0
          if (!m[0]) continue
          if (i > last) out.push({ type: 'text', value: v.slice(last, i) })
          out.push({ type: 'element', tagName: 'mark', properties: { dataMi: String(n++) }, children: [{ type: 'text', value: m[0] }] })
          last = i + m[0].length
        }
        if (last === 0) out.push(child)
        else if (last < v.length) out.push({ type: 'text', value: v.slice(last) })
      }
      node.children = out
    }
    visit(tree)
  }
}

const MD_MARK = 'rounded-[3px] bg-yellow-200 text-yellow-950 shadow-[0_0_0_1px_rgba(234,179,8,0.45)] dark:bg-yellow-400/25 dark:text-yellow-50'
const MD_MARK_NOW = 'rounded-[3px] bg-amber-300 text-amber-950 shadow-[0_0_0_2px_rgba(245,158,11,0.9)] dark:bg-amber-400/60 dark:text-white'

/** Which match (its data-mi) is the current one, for the marks in rendered markdown. */
const CurrentMark = createContext(-1)

function MarkEl({ children, ...p }: { children?: ReactNode }) {
  const cur = useContext(CurrentMark)
  const i = Number((p as Record<string, unknown>)['data-mi'])
  const now = i === cur
  return <mark data-mi={String(i)} data-current={now ? '' : undefined} className={now ? MD_MARK_NOW : MD_MARK}>{children}</mark>
}

// One object, so the marks aren't remounted on every render.
const MD_COMPONENTS = { a: InertLink, img: InertPicture, mark: MarkEl }
const REMARK = [remarkGfm]

function DeliveryViewer({ roomId, seq, onClose, onDeleted, onOpenOther }: {
  roomId: string
  seq: number
  onClose: () => void
  onDeleted: () => void
  onOpenOther: (seq: number) => void
}) {
  const [d, setD] = useState<Delivery | null>(null)
  const [view, setView] = useState<'names' | 'tokens'>('names')
  const [names, setNames] = useState<'fixed' | 'plain'>('fixed')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [confirm, setConfirm] = useState(false)
  const [tick, setTick] = useState(0)
  const [q, setQ] = useState('')
  const [at, setAt] = useState(0)
  const [total, setTotal] = useState(0)
  const [panel, setPanel] = useState(() => typeof window !== 'undefined' && window.innerWidth >= 1100)
  const [tasks, setTasks] = useState<RoomTask[]>([])
  const body = useRef<HTMLDivElement>(null)

  useEffect(() => {
    roomsApi.delivery(roomId, seq).then(setD).catch((e) => setError(errorText(e)))
  }, [roomId, seq, tick])
  const loadTasks = useCallback(() => {
    roomsApi.tasks(roomId).then((r) => setTasks(r.tasks.filter((t) => t.about === seq || t.delivery === seq))).catch(() => {})
  }, [roomId, seq])
  useEffect(() => { loadTasks() }, [loadTasks])

  const waiting = d?.report.grammar?.state === 'waiting'
  useEffect(() => {
    if (!waiting) return
    const t = setTimeout(() => setTick((n) => n + 1), 2500)
    return () => clearTimeout(t)
  }, [waiting, d])

  const hasFixed = !!d && (d.body_fixed !== null || d.title_fixed !== null)
  const plain = !hasFixed || names === 'plain'
  const title = d ? (view === 'tokens' ? d.title_tokens : plain ? d.title : d.title_fixed || d.title) : ''
  const text = d ? (view === 'tokens' ? d.body_tokens : plain ? d.body : d.body_fixed || d.body) : ''
  const canFix = !!d && d.report.restored > 0 && !d.report.no_tokens && !waiting &&
    (!d.report.grammar || d.report.grammar.state === 'failed' || d.report.grammar.state === 'unavailable')

  const plugins = useMemo(() => [rehypeMarks(q)], [q])
  const tokenRanges = useMemo(() => (view === 'tokens' && q.trim().length >= 2 ? findAll(text, [q.trim()]) : []), [view, q, text])
  const cur = total ? Math.min(at, total - 1) : -1

  // The current match: marked brighter and brought into view; the count kept.
  useEffect(() => {
    const marks = body.current ? [...body.current.querySelectorAll('mark')] : []
    const el = (body.current?.querySelector('mark[data-current]') || undefined) as HTMLElement | undefined
    const box = body.current
    if (el && box) {
      const a = box.getBoundingClientRect(), b = el.getBoundingClientRect()
      box.scrollTo({ top: box.scrollTop + (b.top - a.top) - a.height / 3, behavior: 'smooth' })
    }
    void Promise.resolve().then(() => setTotal(marks.length))
  }, [q, cur, text, view])

  const act = async (label: string, work: () => Promise<unknown>) => {
    setBusy(label); setError('')
    try { await work() } catch (e) { setError(errorText(e)) } finally { setBusy('') }
  }
  const step = (dir: 1 | -1) => { if (total) setAt((a) => (Math.min(a, total - 1) + dir + total) % total) }
  const r = d?.report
  const openTasks = tasks.filter((t) => t.about === seq && (t.state === 'waiting' || t.state === 'taken')).length

  return (
    <RoomDialog icon={r ? kindIcon(r) : FileText} onClose={onClose} scroll={false}
      title={d ? title : `#${seq}`}
      subtitle={d && r && <>
        <span className="font-mono">#{d.seq}</span>
        <span>{when(d.at)} · {d.app}</span>
        {r.kind && <Pill tone="sky">{kindLabelOf(r)}</Pill>}
        {r.task && <Pill tone="violet" icon={ListTodo}>answers task {r.task}</Pill>}
        <span>{r.characters.toLocaleString()} characters{r.parts > 1 ? ` in ${r.parts} parts` : ''} · {r.restored} put back</span>
      </>}
      actions={d && (confirm ? (
        <>
          <span className="text-sm text-zinc-400">Delete it? This can't be undone.</span>
          <Button onClick={() => setConfirm(false)}>Keep</Button>
          <Button variant="danger" busy={busy === 'delete'}
            onClick={() => act('delete', async () => { await roomsApi.deleteDelivery(roomId, d.seq); onDeleted() })}>Delete</Button>
        </>
      ) : (
        <>
          {d.report.kind ? (
            <Button variant="primary" icon={Download} busy={busy === 'file'}
              onClick={() => act('file', () => roomsApi.downloadDelivery(roomId, d.seq, 'file', title, d.report.file, plain))}>
              {kindLabelOf(d.report)}
            </Button>
          ) : (
            <>
              <Button icon={Download} busy={busy === 'md'}
                onClick={() => act('md', () => roomsApi.downloadDelivery(roomId, d.seq, 'md', title, undefined, plain))}>Markdown</Button>
              <Button variant="primary" icon={Download} busy={busy === 'docx'}
                onClick={() => act('docx', () => roomsApi.downloadDelivery(roomId, d.seq, 'docx', title, undefined, plain))}>Word</Button>
            </>
          )}
          <Button variant="ghost" icon={Trash2} onClick={() => setConfirm(true)} aria-label="Delete" title="Delete" />
          <Button icon={panel ? PanelRightClose : PanelRightOpen} onClick={() => setPanel(!panel)}>
            Ask Claude{openTasks ? ` · ${openTasks}` : ''}
          </Button>
        </>
      ))}>
      <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 bg-zinc-900/30 px-4 py-2.5">
        <Segmented label="Show" value={view} onChange={(v) => { setView(v); setAt(0) }} options={[
          { value: 'names', label: <span className="inline-flex items-center gap-1.5"><Eye className="h-3.5 w-3.5" />Real names</span> },
          { value: 'tokens', label: <span className="inline-flex items-center gap-1.5"><Bot className="h-3.5 w-3.5" />As Claude wrote it</span> },
        ]} />
        {view === 'names' && hasFixed && (
          <Segmented label="Names" value={names} onChange={setNames} options={[
            { value: 'fixed', label: 'In their case' }, { value: 'plain', label: 'As stored' },
          ]} />
        )}
        <SearchField value={q} onChange={(v) => { setQ(v); setAt(0) }} placeholder="Search this delivery…"
          total={q.trim().length >= 2 ? total : undefined} at={cur} onStep={step} className="min-w-[14rem] flex-1" />
      </div>
      <div className="relative flex min-h-0 flex-1">
        <div ref={body} className="min-h-0 min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-3xl space-y-3 px-5 py-6 wide:px-8">
            {error && <Notice tone="red">{error}</Notice>}
            {!d ? <div className="text-sm text-zinc-500">Loading…</div> : (
              <>
                <GrammarLine grammar={d.report.grammar} hasFixed={hasFixed} canFix={canFix} busy={busy === 'grammar'}
                  onFix={() => act('grammar', async () => { await roomsApi.fixGrammar(roomId, d.seq); setTick((n) => n + 1) })} />
                {r!.real_values.length > 0 && (
                  <Notice tone="red">
                    <div className="flex flex-wrap items-center gap-1">
                      Claude wrote the real value of {r!.real_values.slice(0, 10).map((t) => <TokenChip key={t} token={t} />)} itself — it
                      was told it some other way, or guessed it. It wasn't told it was right.
                    </div>
                  </Notice>
                )}
                {r!.no_tokens && (
                  <Notice tone="amber">
                    No tokens were found in this file, so nothing could be put back. Claude probably wrote a codename or
                    descriptions instead of the tokens — ask it to keep the tokens (⟦ORG_7⟧ …) in files it makes.
                  </Notice>
                )}
                {r!.not_in_room.length > 0 && (
                  <Notice tone="amber">
                    <div className="flex flex-wrap items-center gap-1">
                      {r!.not_in_room.slice(0, 10).map((t) => <TokenChip key={t} token={t} />)}
                      {r!.not_in_room.length === 1 ? "isn't" : "aren't"} in this room, so Claude made {r!.not_in_room.length === 1 ? 'it' : 'them'} up;
                      {r!.not_in_room.length === 1 ? ' it stays' : ' they stay'} as written.
                    </div>
                  </Notice>
                )}
                {r!.kind && (
                  <div className="flex items-start gap-2 rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 text-sm text-zinc-400">
                    <KindIcon r={r!} className="mt-0.5 h-4 w-4 shrink-0 text-sky-600 dark:text-sky-400" />
                    <div>
                      {r!.kind === 'file'
                        ? `A ${kindLabelOf(r!)} file Claude made itself${r!.source === 'owner' ? ', added by you' : ''}, with the real names put back inside.`
                        : `A ${kindLabelOf(r!)} file, built here from what Claude sent, with the real names put back inside.`}
                      {' '}Below is its text; the button above downloads the file.
                      {r!.notes && r!.notes.length > 0 && (
                        <ul className="mt-1 space-y-0.5 text-xs opacity-80">{r!.notes.slice(0, 8).map((n, i) => <li key={i}>· {n}</li>)}</ul>
                      )}
                    </div>
                  </div>
                )}
                {view === 'tokens' ? (
                  <div className="whitespace-pre-wrap break-words rounded-2xl border border-zinc-800 bg-zinc-900/30 p-5 text-[14px] leading-relaxed text-zinc-200">
                    <Marked text={text} ranges={tokenRanges} current={cur} tokens />
                  </div>
                ) : d.format === 'markdown' ? (
                  <article className="fd-file-markdown text-[15px] leading-relaxed">
                    <CurrentMark.Provider value={cur}>
                      <Markdown remarkPlugins={REMARK} rehypePlugins={plugins} urlTransform={untrustedUrlTransform}
                        components={MD_COMPONENTS}>
                        {text}
                      </Markdown>
                    </CurrentMark.Provider>
                  </article>
                ) : (
                  <div className="whitespace-pre-wrap break-words text-[15px] leading-relaxed text-zinc-200">
                    <Marked text={text} ranges={q.trim().length >= 2 ? findAll(text, [q.trim()]) : []} current={cur} />
                  </div>
                )}
              </>
            )}
          </div>
        </div>
        {panel && d && (
          <aside className="flex w-full shrink-0 flex-col border-l border-zinc-800 bg-zinc-900/30 md:w-[380px] max-md:absolute max-md:inset-0 max-md:z-10 max-md:bg-zinc-950">
            <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
              <MessageSquarePlus className="h-4 w-4 text-violet-500" />
              <span className="text-sm font-semibold text-zinc-100">Ask Claude about this</span>
              <button type="button" onClick={() => setPanel(false)} className="ml-auto rounded-lg p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 md:hidden" aria-label="Close">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
              <TaskComposer roomId={roomId} about={seq} onQueued={loadTasks} compact />
              {tasks.length > 0 && (
                <div className="space-y-2">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Tasks about this</div>
                  {tasks.map((t) => <TaskRow key={t.seq} roomId={roomId} t={t} here={seq} onChanged={loadTasks} onOpen={onOpenOther} />)}
                </div>
              )}
            </div>
          </aside>
        )}
      </div>
    </RoomDialog>
  )
}

/** Where the names' case endings stand, and the button to fix them. */
function GrammarLine({ grammar, hasFixed, canFix, busy, onFix }: {
  grammar?: GrammarReport
  hasFixed: boolean
  canFix: boolean
  busy: boolean
  onFix: () => void
}) {
  const fix = canFix && <Button icon={Languages} busy={busy} onClick={onFix}>Fix case endings</Button>
  if (grammar?.state === 'waiting') {
    return (
      <div className="flex items-center gap-2 text-sm text-zinc-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Putting the names into the case each sentence needs…
      </div>
    )
  }
  if (grammar?.state === 'done' && hasFixed) {
    const n = grammar.changed || 0
    return (
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <Languages className="h-3.5 w-3.5" />
        {n} name{n === 1 ? '' : 's'} put into the case {n === 1 ? 'its' : 'their'} sentence needs ({grammar.model}).
        {grammar.refused ? ` ${grammar.refused} suggested form${grammar.refused === 1 ? '' : 's'} didn't pass the check.` : ''}
      </div>
    )
  }
  if (grammar?.state === 'failed') {
    return (
      <Notice tone="amber">
        <div className="flex flex-wrap items-center gap-2">
          <span className="min-w-0 flex-1">The trusted model couldn't fix the case endings ({grammar.error || 'error'}).</span>{fix}
        </div>
      </Notice>
    )
  }
  if (grammar?.state === 'unavailable') {
    return <div className="text-xs text-zinc-500">Names read as stored — no trusted model is set up to fix case endings.</div>
  }
  return fix ? <div className="flex justify-end">{fix}</div> : null
}

// ── tasks for Claude ───────────────────────────────────────────────

/** Write a task; see exactly what Claude will read (anonymized); queue it. */
export function TaskComposer({ roomId, about, onQueued, compact = false }: {
  roomId: string
  about?: number | null
  onQueued: () => void
  compact?: boolean
}) {
  const [text, setText] = useState('')
  const [preview, setPreview] = useState<{ tokens: string; real: string[] } | null>(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [done, setDone] = useState(false)

  const look = async () => {
    setBusy('preview'); setError('')
    try {
      const r = await roomsApi.previewTask(roomId, text)
      setPreview({ tokens: r.prompt_tokens, real: r.still_real })
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy('')
    }
  }
  const queue = async () => {
    setBusy('queue'); setError('')
    try {
      await roomsApi.createTask(roomId, text, about)
      setText(''); setPreview(null); setDone(true)
      onQueued()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy('')
    }
  }

  return (
    <div className="space-y-3">
      <textarea value={text} rows={compact ? 4 : 3} maxLength={8000}
        onChange={(e) => { setText(e.target.value); setPreview(null); setDone(false) }}
        placeholder={about ? 'What should Claude do next with this? Real names are fine — Claude only sees tokens.'
          : 'What should Claude do in this room? Real names are fine — Claude only sees tokens.'}
        className="w-full resize-y rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2.5 text-sm leading-relaxed text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/15" />
      {error && <Notice tone="red">{error}</Notice>}
      {preview && (
        <div className="rounded-xl border border-violet-500/25 bg-violet-500/[0.04] p-3">
          <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-violet-700 dark:text-violet-300">
            <Bot className="h-3.5 w-3.5" /> Claude will read
          </div>
          <div className="whitespace-pre-wrap break-words text-sm leading-relaxed text-zinc-200">
            <Marked text={preview.tokens} ranges={[]} tokens />
          </div>
          {preview.real.length > 0 && (
            <div className="mt-2 text-xs text-red-700 dark:text-red-300">A real name is still in it — reword it or add the name to Names to hide.</div>
          )}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {!preview ? (
          <Button icon={Eye} busy={busy === 'preview'} disabled={!text.trim()} onClick={look}>See what Claude reads</Button>
        ) : (
          <Button variant="primary" icon={Send} busy={busy === 'queue'} disabled={preview.real.length > 0} onClick={queue}>
            Leave it for Claude
          </Button>
        )}
        {preview && <Button variant="ghost" onClick={() => setPreview(null)}>Edit</Button>}
        {done && <span className="inline-flex items-center gap-1 text-xs text-emerald-700 dark:text-emerald-300"><Check className="h-3.5 w-3.5" />Waiting for Claude</span>}
      </div>
      <AskHint />
    </div>
  )
}

function AskHint() {
  const [copied, setCopied] = useState(false)
  return (
    <div className="flex items-start gap-2 rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2.5 text-xs leading-relaxed text-zinc-500">
      <Bot className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <div className="min-w-0 flex-1">
        Claude picks tasks up when you ask it. In Claude, with the Captain Data Rooms connector on, say:
        <button type="button" onClick={async () => {
          try { await navigator.clipboard.writeText(ASK_PHRASE); setCopied(true); setTimeout(() => setCopied(false), 1500) } catch { /* refused */ }
        }} className="mt-1.5 flex w-full items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-left font-medium text-zinc-200 hover:border-zinc-600">
          <span className="min-w-0 flex-1 truncate">“{ASK_PHRASE}”</span>
          {copied ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
      </div>
    </div>
  )
}

const TASK_STATE: Record<RoomTask['state'], { label: string; tone: 'amber' | 'sky' | 'emerald' | 'zinc'; icon: typeof CheckCircle2 }> = {
  waiting: { label: 'Waiting for Claude', tone: 'amber', icon: CircleDashed },
  taken: { label: 'Claude has it', tone: 'sky', icon: Loader2 },
  done: { label: 'Answered', tone: 'emerald', icon: CheckCircle2 },
  cancelled: { label: 'Cancelled', tone: 'zinc', icon: XCircle },
}

function TaskRow({ roomId, t, here, onChanged, onOpen }: {
  roomId: string
  t: RoomTask
  here?: number
  onChanged: () => void
  onOpen: (seq: number) => void
}) {
  const [show, setShow] = useState(false)
  const [busy, setBusy] = useState(false)
  const s = TASK_STATE[t.state]
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs text-zinc-500">Task {t.seq}</span>
        <Pill tone={s.tone} icon={s.icon}>{s.label}</Pill>
        <span className="ml-auto text-[11px] text-zinc-500">{when(t.created_at)}</span>
      </div>
      <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-relaxed text-zinc-200">{t.prompt}</p>
      {show && (
        <div className="mt-2 rounded-lg bg-violet-500/[0.05] p-2.5 text-sm leading-relaxed text-zinc-300">
          <Marked text={t.prompt_tokens} ranges={[]} tokens />
        </div>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <button type="button" onClick={() => setShow(!show)} className="text-zinc-500 hover:text-zinc-200">
          {show ? 'Hide what Claude reads' : 'What Claude reads'}
        </button>
        {t.about && t.about !== here && (
          <button type="button" onClick={() => onOpen(t.about!)} className="text-zinc-500 hover:text-zinc-200">About #{t.about}</button>
        )}
        {t.delivery && t.delivery !== here && (
          <button type="button" onClick={() => onOpen(t.delivery!)} className="font-medium text-emerald-700 hover:underline dark:text-emerald-300">
            Open the answer #{t.delivery}
          </button>
        )}
        {(t.state === 'waiting' || t.state === 'taken') && (
          <button type="button" disabled={busy} onClick={async () => {
            setBusy(true)
            try { await roomsApi.cancelTask(roomId, t.seq); onChanged() } finally { setBusy(false) }
          }} className="ml-auto text-zinc-500 hover:text-red-500">Cancel</button>
        )}
      </div>
    </div>
  )
}

export function TasksSection({ roomId, onChanged, onOpenDelivery }: {
  roomId: string
  onChanged: () => void
  onOpenDelivery: (seq: number) => void
}) {
  const [tasks, setTasks] = useState<RoomTask[] | null>(null)
  const [error, setError] = useState('')
  const load = useCallback(() => {
    roomsApi.tasks(roomId).then((r) => setTasks(r.tasks)).catch((e) => setError(errorText(e)))
  }, [roomId])
  useEffect(() => { load() }, [load])
  const open = (tasks || []).filter((t) => t.state === 'waiting' || t.state === 'taken')
  const rest = (tasks || []).filter((t) => t.state === 'done' || t.state === 'cancelled')
  return (
    <div className="space-y-5">
      <SectionHead icon={ListTodo} title="Tasks for Claude"
        hint="Leave Claude work in this room. Your words are anonymized with the room's names first — you see what Claude reads before it waits for Claude. Answers land in From Claude." />
      {error && <Notice tone="red">{error}</Notice>}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
        <TaskComposer roomId={roomId} onQueued={() => { load(); onChanged() }} />
      </div>
      {tasks && tasks.length > 0 && (
        <div className="grid gap-5 @3xl:grid-cols-2">
          <div className="space-y-2">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Open · {open.length}</div>
            {open.length === 0 && <div className="rounded-xl border border-dashed border-zinc-800 px-3 py-5 text-center text-xs text-zinc-500">Nothing waiting.</div>}
            {open.map((t) => <TaskRow key={t.seq} roomId={roomId} t={t} onChanged={() => { load(); onChanged() }} onOpen={onOpenDelivery} />)}
          </div>
          <div className="space-y-2">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Done · {count(rest.length, 'task')}</div>
            {rest.map((t) => <TaskRow key={t.seq} roomId={roomId} t={t} onChanged={load} onOpen={onOpenDelivery} />)}
          </div>
        </div>
      )}
    </div>
  )
}

// ── a section's head ───────────────────────────────────────────────

export function SectionHead({ icon: Icon, title, hint, actions }: {
  icon: typeof Inbox
  title: ReactNode
  hint?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="flex flex-wrap items-start gap-3">
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
        <Icon className="h-5 w-5" />
      </span>
      <div className="min-w-0 flex-1">
        <h3 className="text-lg font-semibold tracking-tight text-zinc-100">{title}</h3>
        {hint && <p className="mt-0.5 max-w-3xl text-sm leading-relaxed text-zinc-500">{hint}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}
