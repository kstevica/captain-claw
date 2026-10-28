import { useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'
import {
  AlertTriangle,
  ArrowDownWideNarrow,
  Bookmark,
  Brain,
  CalendarClock,
  Check,
  CircleCheck,
  Clock,
  FileText,
  FolderKanban,
  Heart,
  Lightbulb,
  Loader2,
  MessageSquareQuote,
  Pencil,
  Plus,
  Search,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  UserRound,
  Workflow,
  X,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import {
  addInsight, editInsight, forgetInsight, listInsights,
  type AgentAddress, type Insight, type InsightCategory, type InsightFields,
} from '../../services/agentInsights'

// ── Insights: what a kiosk agent remembers between conversations ──────
//
// The second tab of "Instructions & insights": the agent's own insights (its
// /api/insights, through Flight Deck) as cards — what it is, how important,
// where it came from (a document the owner's setup gave it, the owner, a
// conversation) — searchable, filterable by kind, and the owner can teach
// it something, correct it, or make it forget.

const KINDS: Record<InsightCategory, { label: string; one: string; icon: LucideIcon; tone: string; dot: string }> = {
  fact: { label: 'Facts', one: 'Fact', icon: Lightbulb, tone: 'bg-violet-500/10 text-violet-600 dark:text-violet-300', dot: 'bg-violet-500' },
  preference: { label: 'About you', one: 'About you', icon: Heart, tone: 'bg-pink-500/10 text-pink-600 dark:text-pink-300', dot: 'bg-pink-500' },
  feedback: { label: 'How you like it', one: 'How you like it', icon: MessageSquareQuote, tone: 'bg-amber-500/10 text-amber-700 dark:text-amber-300', dot: 'bg-amber-500' },
  decision: { label: 'Decisions', one: 'Decision', icon: CircleCheck, tone: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300', dot: 'bg-emerald-500' },
  contact: { label: 'People', one: 'Person', icon: UserRound, tone: 'bg-sky-500/10 text-sky-700 dark:text-sky-300', dot: 'bg-sky-500' },
  deadline: { label: 'Deadlines', one: 'Deadline', icon: CalendarClock, tone: 'bg-rose-500/10 text-rose-700 dark:text-rose-300', dot: 'bg-rose-500' },
  project: { label: 'Projects', one: 'Project', icon: FolderKanban, tone: 'bg-indigo-500/10 text-indigo-700 dark:text-indigo-300', dot: 'bg-indigo-500' },
  workflow: { label: 'Ways of working', one: 'Way of working', icon: Workflow, tone: 'bg-teal-500/10 text-teal-700 dark:text-teal-300', dot: 'bg-teal-500' },
  reference: { label: 'Where things are', one: 'Where it is', icon: Bookmark, tone: 'bg-zinc-500/10 text-zinc-300', dot: 'bg-zinc-500' },
}
const KIND_ORDER = Object.keys(KINDS) as InsightCategory[]
const kindOf = (c: string) => KINDS[c as InsightCategory] ?? KINDS.fact

/** How it came to know it. */
function originOf(i: Insight): string {
  if ((i.tags || '').split(',').some((t) => t.trim() === 'owner-setup') || i.source_tool === 'owner_setup') return 'From your setup'
  switch (i.source_tool) {
    case 'web_ui': case 'slash_command': return 'Added by you'
    case 'memory_import': return 'Imported'
    case null: case undefined: case '': case 'periodic': return 'Learned in conversation'
    default: return 'It noted this'
  }
}

function when(iso: string): string {
  const t = Date.parse(iso)
  if (Number.isNaN(t)) return ''
  const days = Math.floor((Date.now() - t) / 86_400_000)
  if (days <= 0) return 'today'
  if (days === 1) return 'yesterday'
  if (days < 30) return `${days} days ago`
  return new Date(t).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: days > 300 ? 'numeric' : undefined })
}

/** Importance as five bars: 8+ goes with every reply. */
function Importance({ value }: { value: number }) {
  const lit = Math.max(1, Math.round(value / 2))
  return (
    <span className="flex items-center gap-0.5" title={`Importance ${value}/10${value >= 8 ? ' — it keeps this in mind in every conversation' : ''}`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <span key={n} className={`h-2.5 w-1 rounded-full ${n <= lit ? (value >= 8 ? 'bg-violet-500' : 'bg-zinc-400') : 'bg-zinc-700'}`} />
      ))}
    </span>
  )
}

type Sort = 'newest' | 'important'

export function KioskInsightsPanel({ address, agentName, onTotal }: {
  /** The running agent; null when it's off. */
  address: AgentAddress | null
  agentName: string
  /** How many it remembers in all, for the tab's badge. */
  onTotal: (n: number) => void
}) {
  const [items, setItems] = useState<Insight[] | null>(null)
  const [total, setTotal] = useState<number | null>(null)
  const [query, setQuery] = useState('')
  const [kind, setKind] = useState<InsightCategory | ''>('')
  const [sort, setSort] = useState<Sort>('newest')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [adding, setAdding] = useState(false)
  const [notice, setNotice] = useState('')
  const seq = useRef(0)

  useEffect(() => {
    if (!address) return
    const n = ++seq.current
    const t = setTimeout(() => {
      setLoading(true)
      listInsights(address, query)
        .then((r) => {
          if (n !== seq.current) return
          setItems(r.items); setError('')
          setTotal(r.total)
        })
        .catch((e) => { if (n === seq.current) setError(e instanceof Error ? e.message : String(e)) })
        .finally(() => { if (n === seq.current) setLoading(false) })
    }, query.trim() ? 300 : 0)
    return () => clearTimeout(t)
  }, [address, query])
  useEffect(() => { if (total !== null) onTotal(total) }, [total, onTotal])

  const counts = useMemo(() => {
    const c = new Map<InsightCategory, number>()
    for (const i of items ?? []) c.set(i.category, (c.get(i.category) ?? 0) + 1)
    return c
  }, [items])
  const shown = useMemo(() => {
    const xs = (items ?? []).filter((i) => !kind || i.category === kind)
    return sort === 'important' ? [...xs].sort((a, b) => b.importance - a.importance) : xs
  }, [items, kind, sort])
  const fromDocs = (items ?? []).filter((i) => i.source_file).length

  if (!address) {
    return (
      <Empty icon={Brain} title={`${agentName} is off`}>
        Power it on to see what it remembers.
      </Empty>
    )
  }

  const update = (got: Insight) => setItems((xs) => (xs ?? []).map((x) => (x.id === got.id ? { ...x, ...got } : x)))
  const remove = (id: string) => {
    setItems((xs) => (xs ?? []).filter((x) => x.id !== id))
    setTotal((t) => (t === null ? t : Math.max(0, t - 1)))
  }
  const added = (got: Insight | null) => {
    setAdding(false)
    if (!got) { setNotice(`${agentName} already knew that.`); return }
    setItems((xs) => [got, ...(xs ?? [])])
    setNotice(`${agentName} will remember that.`)
    setTotal((t) => (t === null ? t : t + 1))
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 space-y-3 border-b border-zinc-800 px-5 pb-3 pt-4">
        <p className="text-xs leading-relaxed text-zinc-500">
          What {agentName} remembers between conversations: facts, people, decisions and how you like things done.
          The most important come up in every conversation; the rest when they're relevant.
        </p>
        <div className="flex items-center gap-2">
          <label className="flex h-9 min-w-0 flex-1 items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-950 px-3 focus-within:border-violet-500/60">
            <Search className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search what it remembers"
              aria-label="Search insights"
              className="min-w-0 flex-1 bg-transparent text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none"
            />
            {loading && items !== null && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-zinc-500" />}
            {query && !loading && (
              <button onClick={() => setQuery('')} aria-label="Clear search" className="rounded p-0.5 text-zinc-500 hover:text-zinc-300">
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </label>
          <button
            onClick={() => setSort((s) => (s === 'newest' ? 'important' : 'newest'))}
            title={sort === 'newest' ? 'Newest first — show the most important first' : 'Most important first — show the newest first'}
            className="flex h-9 shrink-0 items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
          >
            {sort === 'newest' ? <Clock className="h-3.5 w-3.5" /> : <ArrowDownWideNarrow className="h-3.5 w-3.5" />}
            <span className="hidden wide:inline">{sort === 'newest' ? 'Newest' : 'Most important'}</span>
          </button>
          <button
            onClick={() => { setAdding(true); setNotice('') }}
            disabled={adding}
            className="flex h-9 shrink-0 items-center gap-1.5 rounded-lg bg-violet-600 px-3 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
          >
            <Plus className="h-4 w-4" />
            <span className="hidden wide:inline">Teach it something</span>
          </button>
        </div>
        {(items?.length ?? 0) > 0 && (
          <div className="-mx-1 flex items-center gap-1.5 overflow-x-auto px-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            role="tablist" aria-label="Kinds">
            <Chip on={kind === ''} onClick={() => setKind('')}>All <Count n={items?.length ?? 0} /></Chip>
            {KIND_ORDER.filter((k) => counts.get(k)).map((k) => (
              <Chip key={k} on={kind === k} onClick={() => setKind(kind === k ? '' : k)}>
                <span className={`h-1.5 w-1.5 rounded-full ${KINDS[k].dot}`} />
                {KINDS[k].label} <Count n={counts.get(k) ?? 0} />
              </Chip>
            ))}
            {fromDocs > 0 && (
              <span className="ml-auto hidden shrink-0 items-center gap-1 pl-2 text-[11px] text-zinc-500 wide:flex">
                <FileText className="h-3 w-3" /> {fromDocs} from your documents
              </span>
            )}
          </div>
        )}
      </div>

      <div className="min-h-0 flex-1 space-y-2.5 overflow-y-auto px-5 py-4">
        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> Couldn't read its insights: {error}
          </div>
        )}
        {notice && (
          <div className="flex items-center gap-2 rounded-lg border border-emerald-500/25 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-300">
            <Check className="h-3.5 w-3.5 shrink-0" /> {notice}
            <button onClick={() => setNotice('')} aria-label="Dismiss" className="ml-auto rounded p-0.5 opacity-70 hover:opacity-100">
              <X className="h-3 w-3" />
            </button>
          </div>
        )}
        {adding && <InsightEditor address={address} agentName={agentName} onDone={added} onCancel={() => setAdding(false)} />}
        {items === null && !error ? (
          [0, 1, 2].map((n) => (
            <div key={n} className="animate-pulse rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <div className="flex gap-3">
                <div className="h-8 w-8 rounded-lg bg-zinc-800" />
                <div className="flex-1 space-y-2">
                  <div className="h-3 w-4/5 rounded bg-zinc-800" />
                  <div className="h-3 w-2/5 rounded bg-zinc-800/70" />
                </div>
              </div>
            </div>
          ))
        ) : shown.length === 0 && !adding && !error ? (
          query.trim() || kind ? (
            <Empty icon={Search} title="Nothing matches">
              {query.trim() ? <>Nothing it remembers mentions "{query.trim()}".</> : 'Nothing of this kind yet.'}
            </Empty>
          ) : (
            <Empty icon={Brain} title="Nothing remembered yet">
              {agentName} saves facts, people, decisions and your preferences as it works with you — and the facts
              its way of working gives it. You can also teach it something yourself.
            </Empty>
          )
        ) : (
          shown.map((i) => (
            <InsightCard key={i.id} insight={i} address={address} agentName={agentName} onSaved={update} onForgotten={remove} />
          ))
        )}
      </div>
    </div>
  )
}

function Chip({ on, onClick, children }: { on: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      role="tab"
      aria-selected={on}
      onClick={onClick}
      className={`flex h-7 shrink-0 items-center gap-1.5 rounded-full border px-2.5 text-xs transition-colors ${
        on ? 'border-violet-500/50 bg-violet-500/15 text-zinc-100' : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
      }`}
    >
      {children}
    </button>
  )
}

function Count({ n }: { n: number }) {
  return <span className="tabular-nums text-zinc-500">{n}</span>
}

function Empty({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children: ReactNode }) {
  return (
    <div className="flex h-full min-h-[14rem] flex-col items-center justify-center px-8 text-center">
      <span className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-900 text-zinc-500">
        <Icon className="h-6 w-6" />
      </span>
      <p className="text-sm font-medium text-zinc-300">{title}</p>
      <p className="mt-1 max-w-sm text-xs leading-relaxed text-zinc-500">{children}</p>
    </div>
  )
}

function InsightCard({ insight: i, address, agentName, onSaved, onForgotten }: {
  insight: Insight
  address: AgentAddress
  agentName: string
  onSaved: (i: Insight) => void
  onForgotten: (id: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const k = kindOf(i.category)
  const Icon = k.icon

  const forget = async () => {
    setBusy(true); setError('')
    try {
      await forgetInsight(address, i.id)
      onForgotten(i.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  if (editing) {
    return (
      <InsightEditor
        address={address} agentName={agentName} insight={i}
        onDone={(got) => { if (got) onSaved(got); setEditing(false) }}
        onCancel={() => setEditing(false)}
      />
    )
  }
  return (
    <article className="group rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 transition-colors hover:border-zinc-700">
      <div className="flex items-start gap-3">
        <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${k.tone}`} title={k.one}>
          <Icon className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-zinc-100">{i.content}</p>
          {(i.why || i.how_to_apply) && (
            <div className="mt-1.5 space-y-0.5 border-l-2 border-zinc-800 pl-2.5 text-xs leading-relaxed text-zinc-500">
              {i.why && <p><span className="text-zinc-400">Why:</span> {i.why}</p>}
              {i.how_to_apply && <p><span className="text-zinc-400">When:</span> {i.how_to_apply}</p>}
            </div>
          )}
          <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[11px] text-zinc-500">
            <span className="font-medium text-zinc-400">{k.one}</span>
            <Importance value={i.importance} />
            {i.category === 'feedback' && i.polarity && (
              <span className={`flex items-center gap-1 ${i.polarity === 'positive' ? 'text-emerald-700 dark:text-emerald-400' : 'text-rose-700 dark:text-rose-400'}`}>
                {i.polarity === 'positive' ? <ThumbsUp className="h-3 w-3" /> : <ThumbsDown className="h-3 w-3" />}
                {i.polarity === 'positive' ? 'Keep doing' : 'Stop doing'}
              </span>
            )}
            {i.source_file && (
              <span
                className="flex max-w-[16rem] items-center gap-1 rounded-md bg-sky-500/10 px-1.5 py-0.5 text-sky-700 dark:text-sky-300"
                title={`From ${i.source_file}${i.source_section ? ` — ${i.source_section}` : ''}`}
              >
                <FileText className="h-3 w-3 shrink-0" />
                <span className="truncate">{i.source_file}{i.source_section ? ` · ${i.source_section}` : ''}</span>
              </span>
            )}
            <span>{originOf(i)}{when(i.updated_at || i.created_at) ? ` · ${when(i.updated_at || i.created_at)}` : ''}</span>
          </div>
        </div>
        {!confirming && (
          <div className="flex shrink-0 items-center gap-0.5 transition-opacity wide:opacity-0 wide:focus-within:opacity-100 wide:group-hover:opacity-100">
            <button onClick={() => setEditing(true)} title="Correct it" aria-label="Edit"
              className="rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
              <Pencil className="h-3.5 w-3.5" />
            </button>
            <button onClick={() => setConfirming(true)} title="Make it forget this" aria-label="Forget"
              className="rounded-lg p-1.5 text-zinc-500 hover:bg-rose-500/10 hover:text-rose-600 dark:hover:text-rose-400">
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </div>
      {confirming && (
        <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-rose-500/25 bg-rose-500/5 px-3 py-2">
          <span className="min-w-0 flex-1 text-xs text-zinc-300">{agentName} forgets this for good.</span>
          <button onClick={() => setConfirming(false)} disabled={busy}
            className="rounded-md px-2.5 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
            Keep it
          </button>
          <button onClick={forget} disabled={busy}
            className="flex items-center gap-1 rounded-md bg-rose-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-rose-500 disabled:opacity-50">
            {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />} Forget
          </button>
        </div>
      )}
      {error && <p className="mt-2 text-xs text-red-700 dark:text-red-300">Couldn't do that: {error}</p>}
    </article>
  )
}

/** A new insight, or a correction of one. */
function InsightEditor({ address, agentName, insight, onDone, onCancel }: {
  address: AgentAddress
  agentName: string
  insight?: Insight
  onDone: (got: Insight | null) => void
  onCancel: () => void
}) {
  const [content, setContent] = useState(insight?.content ?? '')
  const [category, setCategory] = useState<InsightCategory>(insight?.category ?? 'fact')
  const [importance, setImportance] = useState(insight?.importance ?? 6)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const ready = content.trim().length > 0 && (!insight || content.trim() !== insight.content
    || category !== insight.category || importance !== insight.importance)

  const save = async () => {
    if (!ready || busy) return
    setBusy(true); setError('')
    const fields: InsightFields = { content: content.trim(), category, importance }
    try {
      onDone(insight ? await editInsight(address, insight.id, fields) : await addInsight(address, fields))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }
  const onKey = (e: ReactKeyboardEvent) => {
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); onCancel() }
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); void save() }
  }

  return (
    <div className="rounded-xl border border-violet-500/40 bg-violet-500/5 p-4" onKeyDown={onKey}>
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        autoFocus
        rows={3}
        maxLength={2000}
        placeholder={`What should ${agentName} remember? e.g. "Our fiscal year starts in April."`}
        className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm leading-relaxed text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
      />
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-3">
        <label className="flex items-center gap-2 text-xs text-zinc-400">
          Kind
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value as InsightCategory)}
            className="h-8 rounded-lg border border-zinc-700 bg-zinc-950 px-2 text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none"
          >
            {KIND_ORDER.map((k) => <option key={k} value={k}>{KINDS[k].one}</option>)}
          </select>
        </label>
        <label className="flex min-w-[12rem] flex-1 items-center gap-2 text-xs text-zinc-400">
          Importance
          <input type="range" min={1} max={10} value={importance} onChange={(e) => setImportance(Number(e.target.value))}
            className="min-w-0 flex-1 accent-violet-500" aria-valuetext={`${importance} of 10`} />
          <span className="w-10 shrink-0 tabular-nums text-zinc-300">{importance}/10</span>
        </label>
      </div>
      <p className="mt-1.5 text-[11px] text-zinc-500">
        {importance >= 8 ? `${agentName} keeps this in mind in every conversation.` : 'It comes up when a conversation touches it.'}
      </p>
      {error && <p className="mt-2 text-xs text-red-700 dark:text-red-300">Couldn't save it: {error}</p>}
      <div className="mt-3 flex items-center justify-end gap-2">
        <button onClick={onCancel} disabled={busy}
          className="rounded-lg px-3 py-1.5 text-sm font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
          Cancel
        </button>
        <button onClick={save} disabled={!ready || busy}
          className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-1.5 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50">
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
          {insight ? 'Save' : 'Remember it'}
        </button>
      </div>
    </div>
  )
}
