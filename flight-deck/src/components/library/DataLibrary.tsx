import {
  useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState,
  type KeyboardEvent as ReactKeyboardEvent, type MouseEvent as ReactMouseEvent, type ReactNode,
} from 'react'
import {
  AlertTriangle,
  ArrowLeft,
  AtSign,
  Braces,
  CalendarDays,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronsUpDown,
  Copy,
  Database,
  Download,
  Hash,
  Lightbulb,
  Link2,
  Loader2,
  MessageSquare,
  Pencil,
  Plus,
  RotateCw,
  Search,
  SlidersHorizontal,
  Sparkles,
  Table2,
  ToggleLeft,
  Trash2,
  Type,
  X,
} from 'lucide-react'
import { useNotificationStore } from '../../stores/notificationStore'
import { fmtRelative, fmtWhen } from '../../services/schedulerApi'
import { apiStatus, type LibraryState } from '../../services/fileLibraryApi'
import {
  dataLibraryApi,
  type ColumnKind,
  type ColumnProfile,
  type DataPage,
  type DataRow,
  type DataSort,
  type DataTable,
  type ExportFormat,
  type RowsPage,
} from '../../services/dataLibraryApi'
import { FacetGroup, FullscreenButton, Highlight, ScanStatus, libraryIconBtn as iconBtn } from './libraryParts'

// ── Data library ────────────────────────────────────────────────────
//
// Every table your agents keep in their datastores, in one place: Flight
// Deck profiles each new or changed table (how full each column is, ranges,
// the most common values) and writes a title, a summary, key facts and a
// line on every column. Open a table to read it: the summary on top, then
// the rows themselves — live, searchable, sortable, one record at a time if
// you like, and changed by hand (a cell, a new row, a deleted one) — or each
// column's profile.

const PAGE = 60
const ROWS_PAGE = 50

const CATEGORY_LABEL: Record<string, string> = {
  people: 'People & contacts', companies: 'Companies & organisations', leads: 'Leads & deals',
  finance: 'Finance', events: 'Events & schedules', products: 'Products & inventory',
  tasks: 'Tasks & projects', research: 'Research data', content: 'Content & media',
  metrics: 'Metrics & logs', other: 'Other', pending: 'Not summarised yet',
}

const CATEGORY_TONE: Record<string, string> = {
  people: 'bg-sky-500/10 text-sky-700 dark:text-sky-300',
  companies: 'bg-indigo-500/10 text-indigo-700 dark:text-indigo-300',
  leads: 'bg-amber-500/10 text-amber-700 dark:text-amber-300',
  finance: 'bg-lime-500/10 text-lime-700 dark:text-lime-300',
  events: 'bg-violet-500/10 text-violet-700 dark:text-violet-300',
  products: 'bg-orange-500/10 text-orange-700 dark:text-orange-300',
  tasks: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
  research: 'bg-cyan-500/10 text-cyan-700 dark:text-cyan-300',
  content: 'bg-pink-500/10 text-pink-700 dark:text-pink-300',
  metrics: 'bg-teal-500/10 text-teal-700 dark:text-teal-300',
  other: 'bg-zinc-800 text-zinc-400',
  pending: 'bg-zinc-800 text-zinc-500',
}

const CATEGORY_DOT: Record<string, string> = {
  people: 'bg-sky-500', companies: 'bg-indigo-500', leads: 'bg-amber-500', finance: 'bg-lime-500',
  events: 'bg-violet-500', products: 'bg-orange-500', tasks: 'bg-emerald-500', research: 'bg-cyan-500',
  content: 'bg-pink-500', metrics: 'bg-teal-500', other: 'bg-zinc-500', pending: 'bg-zinc-600',
}

const KIND_ICON: Record<ColumnKind, typeof Type> = {
  text: Type, number: Hash, url: Link2, email: AtSign, date: CalendarDays, boolean: ToggleLeft, json: Braces,
}

const SORTS: { id: DataSort; label: string }[] = [
  { id: 'recent', label: 'Recently changed' },
  { id: 'name', label: 'Name' },
  { id: 'rows', label: 'Most rows' },
  { id: 'agent', label: 'By agent' },
]

const URL_RE = /^https?:\/\//i
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/
const ISO_RE = /^\d{4}-\d{2}-\d{2}(?:[T ](\d{2}):(\d{2}))?/

const numberFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 })

function fmtNum(v: unknown): string {
  return typeof v === 'number' && Number.isFinite(v) ? numberFormat.format(v) : String(v ?? '')
}

/** A date or date-time as people read it; anything else as it is. */
function fmtDate(v: unknown): string {
  const s = String(v ?? '')
  const m = ISO_RE.exec(s)
  if (!m) return s
  const d = new Date(m[1] !== undefined ? s.replace(' ', 'T') : `${s.slice(0, 10)}T00:00:00`)
  if (Number.isNaN(d.getTime())) return s
  const date = d.toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' })
  return m[1] !== undefined ? `${date}, ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : date
}

function isoEpoch(iso: string): number | null {
  const t = Date.parse(iso)
  return Number.isFinite(t) ? t / 1000 : null
}

/** "company_slug" → "Company slug". */
function humanize(name: string): string {
  const s = name.replace(/[_-]+/g, ' ').trim()
  return s ? s[0].toUpperCase() + s.slice(1) : name
}

function prettyUrl(s: string): string {
  try {
    const u = new URL(s)
    const path = u.pathname === '/' ? '' : u.pathname
    return `${u.hostname.replace(/^www\./, '')}${path}${u.search ? '?…' : ''}`
  } catch {
    return s
  }
}

function categoryOf(t: DataTable): string {
  return t.category || 'pending'
}

function CategoryChip({ id }: { id: string }) {
  return (
    <span className={`shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-medium ${CATEGORY_TONE[id] || CATEGORY_TONE.other}`}>
      {CATEGORY_LABEL[id] || id}
    </span>
  )
}

function KindIcon({ kind, className = 'h-3.5 w-3.5' }: { kind: ColumnKind | undefined; className?: string }) {
  const Icon = KIND_ICON[kind || 'text'] || Type
  return <Icon className={`${className} shrink-0 text-zinc-500`} />
}

export function DataLibrary({ phone = false, initialAgent, onClose, onOpenAgent }: {
  phone?: boolean
  /** Open filtered to this agent's tables. */
  initialAgent?: string | null
  onClose: () => void
  /** Chat with the agent a table belongs to. */
  onOpenAgent?: (agentSlug: string) => void
}) {
  const notify = useNotificationStore((s) => s.add)
  const [state, setState] = useState<LibraryState | null>(null)
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [agent, setAgent] = useState(initialAgent || '')
  const [category, setCategory] = useState('')
  const [sort, setSort] = useState<DataSort>('recent')
  const [page, setPage] = useState<DataPage | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')
  const [opened, setOpened] = useState<DataTable | null>(null)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const reqId = useRef(0)
  const shown = useRef(PAGE)

  // Opened again for another agent while it is open (from that agent's
  // menu): the filter follows, and another agent's open table gives way to
  // the list.
  useEffect(() => {
    const a = initialAgent || ''
    setAgent(a)
    if (a) setOpened((cur) => (cur && cur.agent_slug !== a ? null : cur))
  }, [initialAgent])

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), query ? 250 : 0)
    return () => clearTimeout(t)
  }, [query])

  const load = useCallback(async (quiet = false) => {
    const id = ++reqId.current
    if (!quiet) { setLoading(true); shown.current = PAGE }
    try {
      const res = await dataLibraryApi.tables({
        q: debounced, agent, category, sort, limit: Math.max(PAGE, shown.current), offset: 0,
      })
      if (id !== reqId.current) return
      setPage(res)
      setError('')
      // The open table follows its fresh copy (a summary that just arrived).
      setOpened((cur) => (cur ? res.items.find((i) => i.id === cur.id) ?? cur : cur))
    } catch (e) {
      if (id === reqId.current && !quiet) setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (id === reqId.current) setLoading(false)
    }
  }, [debounced, agent, category, sort])

  useEffect(() => { load() }, [load])

  const loadMore = async () => {
    if (!page) return
    setLoadingMore(true)
    try {
      const res = await dataLibraryApi.tables({ q: debounced, agent, category, sort, limit: PAGE, offset: page.items.length })
      setPage((cur) => cur && { ...res, items: [...cur.items, ...res.items.filter((i) => !cur.items.some((c) => c.id === i.id))] })
      shown.current = page.items.length + res.items.length
    } catch (e) {
      notify('error', 'Could not load more tables', e instanceof Error ? e.message : String(e))
    } finally {
      setLoadingMore(false)
    }
  }

  // Opening the library looks for new and changed tables.
  useEffect(() => {
    dataLibraryApi.scan().then(setState).catch(() => dataLibraryApi.status().then(setState).catch(() => {}))
  }, [])

  // While a scan runs: its progress and the list, every few seconds.
  const running = !!state?.job.running
  const loadRef = useRef(load)
  loadRef.current = load
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => {
      dataLibraryApi.status().then(setState).catch(() => {})
      loadRef.current(true)
    }, 2500)
    return () => clearInterval(t)
  }, [running])
  const wasRunning = useRef(false)
  useEffect(() => {
    if (wasRunning.current && !running) loadRef.current(true)
    wasRunning.current = running
  }, [running])

  const rescan = async () => {
    try { setState(await dataLibraryApi.scan(true)) } catch (e) {
      notify('error', 'Could not scan for tables', e instanceof Error ? e.message : String(e))
    }
  }

  // Escape: a record closes itself (TableView); then the filters, the open
  // table, the library.
  const openedRef = useRef(opened)
  openedRef.current = opened
  const filtersRef = useRef(filtersOpen)
  filtersRef.current = filtersOpen
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (filtersRef.current) { setFiltersOpen(false); return }
      if (openedRef.current) { setOpened(null); return }
      onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const items = page?.items ?? []
  const facets = page?.facets
  const terms = useMemo(() => debounced.toLowerCase().split(/\s+/).filter(Boolean), [debounced])
  const filtered = !!(debounced || agent || category)
  // Only one agent's tables (how an agent's menu opens it): an empty list
  // means that agent has no data yet, not that a search missed.
  const agentOnly = !!agent && !debounced && !category
  const clearFilters = () => { setQuery(''); setAgent(''); setCategory('') }

  if (opened) {
    return (
      <TableView
        key={opened.id}
        item={opened}
        terms={terms}
        phone={phone}
        onBack={() => setOpened(null)}
        onClose={onClose}
        onChanged={(fresh) => {
          setOpened(fresh)
          setPage((cur) => cur && { ...cur, items: cur.items.map((i) => (i.id === fresh.id ? { ...i, ...fresh } : i)) })
          dataLibraryApi.status().then(setState).catch(() => {})
        }}
        onRescan={async () => {
          // The scan's own answer: while it runs the list keeps reloading,
          // and the open table follows its fresh copy when it ends.
          try { setState(await dataLibraryApi.scan()) } catch { /* the rows reload all the same */ }
        }}
        onOpenAgent={onOpenAgent && opened.agent_running ? () => onOpenAgent(opened.agent_slug) : undefined}
      />
    )
  }

  const header = (
    <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
        <Database className="h-[18px] w-[18px]" />
      </span>
      <div className="min-w-0 flex-1">
        <h2 className="truncate text-base font-semibold text-zinc-100">Data library</h2>
        {!phone && <p className="truncate text-xs text-zinc-500">Every table your agents keep — summarised, profiled and easy to read.</p>}
      </div>
      {!phone && <ScanStatus state={state} noun="table" tone="emerald" />}
      <button onClick={rescan} disabled={running} className={iconBtn} title="Look for new tables" aria-label="Look for new tables">
        <RotateCw className={`h-4 w-4 ${running ? 'animate-spin' : ''}`} />
      </button>
      <FullscreenButton />
      <button onClick={onClose} className={iconBtn} title="Close" aria-label="Close">
        <X className="h-4 w-4" />
      </button>
    </div>
  )

  const sortSelect = (cls: string) => (
    <select value={sort} onChange={(e) => setSort(e.target.value as DataSort)} aria-label="Sort" className={cls}>
      {SORTS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
    </select>
  )

  const toolbar = (
    <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2.5 wide:px-4">
      <SearchBox
        value={query}
        onChange={setQuery}
        placeholder={phone ? 'Search tables…' : 'Search tables, columns and the data inside…'}
      />
      {phone ? (
        <button
          onClick={() => setFiltersOpen(true)}
          className={`flex h-9 shrink-0 items-center gap-1.5 rounded-lg border px-3 text-sm ${
            agent || category ? 'border-emerald-500/50 text-emerald-700 dark:text-emerald-300' : 'border-zinc-800 text-zinc-400'}`}
        >
          <SlidersHorizontal className="h-4 w-4" />Filters
        </button>
      ) : sortSelect('shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-2 text-sm text-zinc-300 focus:border-emerald-500/60 focus:outline-none')}
    </div>
  )

  const filters = !facets
    ? <div className="flex justify-center py-8 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>
    : (
      <div className="flex flex-col gap-4 px-2 py-3">
        <FacetGroup
          title="Categories" allLabel="All tables" facets={facets.categories} value={category} onChange={setCategory}
          labelOf={(id) => CATEGORY_LABEL[id] || id}
          dot={(id) => <span className={`h-2 w-2 shrink-0 rounded-full ${CATEGORY_DOT[id] || CATEGORY_DOT.other}`} />}
        />
        <FacetGroup title="Agents" allLabel="All agents" facets={facets.agents} value={agent} onChange={setAgent} />
      </div>
    )

  const list = (
    <TableList
      items={items}
      total={page?.total ?? 0}
      rowsTotal={page?.rows_total ?? 0}
      state={state}
      loading={loading}
      error={error}
      filtered={filtered}
      agentOnly={agentOnly}
      terms={terms}
      phone={phone}
      loadingMore={loadingMore}
      onOpen={setOpened}
      onMore={loadMore}
      onClear={clearFilters}
      onRetry={() => load()}
    />
  )

  if (phone) {
    return (
      <div className="relative flex h-full flex-col bg-zinc-950">
        {header}
        <div className="border-b border-zinc-800 px-3 py-2"><ScanStatus state={state} noun="table" tone="emerald" /></div>
        {toolbar}
        <div className="min-h-0 flex-1">{list}</div>
        {filtersOpen && (
          <div className="absolute inset-0 z-20 flex flex-col bg-zinc-950">
            <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
              <h3 className="flex-1 text-base font-semibold text-zinc-100">Filters</h3>
              <button onClick={() => setFiltersOpen(false)} className={iconBtn} aria-label="Close filters"><X className="h-5 w-5" /></button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto">
              <div className="px-4 pt-3">
                <label className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Sort</label>
                {sortSelect('h-11 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 text-base text-zinc-200')}
              </div>
              {filters}
            </div>
            <div className="flex gap-2 border-t border-zinc-800 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
              <button onClick={() => { setAgent(''); setCategory('') }} className="h-11 flex-1 rounded-xl border border-zinc-800 text-sm font-medium text-zinc-300">
                Clear
              </button>
              <button onClick={() => setFiltersOpen(false)} className="h-11 flex-1 rounded-xl bg-emerald-600 text-sm font-medium text-white">
                Show {page?.total ?? 0} table{page?.total === 1 ? '' : 's'}
              </button>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col bg-zinc-950">
      {header}
      {toolbar}
      <div className="flex min-h-0 flex-1">
        <aside className="w-56 shrink-0 overflow-y-auto border-r border-zinc-800 bg-zinc-900/40">{filters}</aside>
        <section className="min-w-0 flex-1">{list}</section>
      </div>
    </div>
  )
}

function SearchBox({ value, onChange, placeholder, small = false }: {
  value: string; onChange: (v: string) => void; placeholder: string; small?: boolean
}) {
  return (
    <div className="relative min-w-0 flex-1">
      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={`w-full rounded-lg border border-zinc-800 bg-zinc-900 pl-8 pr-8 text-zinc-200 placeholder-zinc-500 focus:border-emerald-500/60 focus:outline-none ${
          small ? 'py-1.5 text-[13px]' : 'py-2 text-sm'}`}
      />
      {value && (
        <button onClick={() => onChange('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-zinc-500 hover:text-zinc-200" aria-label="Clear search">
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  )
}

// ── The list of tables ──

function TableList({
  items, total, rowsTotal, state, loading, error, filtered, agentOnly, terms, phone, loadingMore, onOpen, onMore, onClear, onRetry,
}: {
  items: DataTable[]; total: number; rowsTotal: number; state: LibraryState | null; loading: boolean; error: string
  filtered: boolean; agentOnly: boolean; terms: string[]; phone: boolean; loadingMore: boolean
  onOpen: (t: DataTable) => void; onMore: () => void; onClear: () => void; onRetry: () => void
}) {
  if (loading && items.length === 0) {
    return <div className="flex h-full justify-center pt-16 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
  }
  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-zinc-500">
        <AlertTriangle className="h-5 w-5 text-red-500" />{error}
        <button onClick={onRetry} className="text-sm font-medium text-emerald-700 hover:underline dark:text-emerald-400">Try again</button>
      </div>
    )
  }
  if (items.length === 0) {
    const scanning = !!state?.job.running
    return (
      <div className="flex h-full flex-col items-center justify-center px-6 text-center">
        <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
          {scanning ? <Loader2 className="h-7 w-7 animate-spin" /> : <Database className="h-7 w-7" />}
        </span>
        <h3 className="mt-4 text-base font-semibold text-zinc-100">
          {filtered && !agentOnly
            ? 'Nothing matches'
            : scanning ? 'Gathering your tables…' : agentOnly ? 'No tables yet' : state?.agents === 0 ? 'No agents yet' : 'No tables yet'}
        </h3>
        <p className="mt-1.5 max-w-sm text-sm leading-relaxed text-zinc-400">
          {filtered && !agentOnly
            ? 'Try other words, or clear the filters.'
            : scanning
              ? 'Your agents\' tables show up here as soon as they\'re found and read.'
              : agentOnly
                ? 'This agent hasn\'t kept any data yet. When it collects some — contacts, companies, prices, results — it shows up here, summarised.'
                : 'When your agents collect data — contacts, companies, prices, results — it lands in their datastore and shows up here, summarised.'}
        </p>
        {filtered && (
          <button onClick={onClear} className="mt-4 text-sm font-medium text-emerald-700 hover:underline dark:text-emerald-400">
            {agentOnly ? 'Show every agent\'s tables' : 'Clear search and filters'}
          </button>
        )}
      </div>
    )
  }
  return (
    <div className="h-full overflow-y-auto px-3 py-3 wide:px-4">
      <div className="mb-2 px-1 text-[11px] text-zinc-500">
        {total} table{total === 1 ? '' : 's'} · {fmtNum(rowsTotal)} row{rowsTotal === 1 ? '' : 's'}
        {terms.length ? <> for “{terms.join(' ')}”</> : null}
      </div>
      <ul className={`grid gap-2.5 ${phone ? 'grid-cols-1' : 'grid-cols-2'}`}>
        {items.map((t) => (
          <li key={t.id}><TableCard t={t} terms={terms} onClick={() => onOpen(t)} /></li>
        ))}
      </ul>
      {items.length < total && (
        <div className="flex justify-center py-4">
          <button
            onClick={onMore}
            disabled={loadingMore}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-800 px-4 py-2 text-sm text-zinc-300 hover:bg-zinc-800/60 disabled:opacity-60"
          >
            {loadingMore && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            Show more ({total - items.length} left)
          </button>
        </div>
      )}
    </div>
  )
}

function TableCard({ t, terms, onClick }: { t: DataTable; terms: string[]; onClick: () => void }) {
  const changed = isoEpoch(t.updated_at)
  const cols = t.columns.slice(0, 6)
  return (
    <button
      onClick={onClick}
      className="flex h-full w-full flex-col gap-2 rounded-xl border border-zinc-800 bg-zinc-900/40 p-3.5 text-left transition-colors hover:border-emerald-500/40 hover:bg-zinc-900/80"
    >
      <span className="flex w-full items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
          <Table2 className="h-[18px] w-[18px]" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-semibold text-zinc-100">
            <Highlight text={t.title || humanize(t.table_name)} terms={terms} />
          </span>
          <span className="block truncate text-[11px] text-zinc-500">
            <span className="font-mono"><Highlight text={t.table_name} terms={terms} /></span> · {t.agent_name}
          </span>
        </span>
      </span>
      <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-zinc-400">
        <CategoryChip id={categoryOf(t)} />
        <span><span className="font-semibold tabular-nums text-zinc-200">{fmtNum(t.row_count)}</span> row{t.row_count === 1 ? '' : 's'}</span>
        <span><span className="font-semibold tabular-nums text-zinc-200">{t.columns.length}</span> column{t.columns.length === 1 ? '' : 's'}</span>
        {changed && <span>changed {fmtRelative(changed)}</span>}
      </span>
      {t.status === 'pending' && !t.summary ? (
        <span className="flex items-center gap-1.5 text-xs text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" />Reading the table…</span>
      ) : t.status === 'error' && !t.summary ? (
        <span className="flex items-center gap-1.5 text-xs text-red-600 dark:text-red-400"><AlertTriangle className="h-3 w-3" />Couldn't summarise it</span>
      ) : t.summary ? (
        <span className="line-clamp-3 text-xs leading-relaxed text-zinc-400"><Highlight text={t.summary} terms={terms} /></span>
      ) : null}
      {t.snippet && (
        <span className="line-clamp-2 border-l-2 border-amber-500/40 pl-2 text-[11px] italic leading-relaxed text-zinc-500">
          <Highlight text={t.snippet} terms={terms} />
        </span>
      )}
      {cols.length > 0 && (
        <span className="mt-auto flex flex-wrap gap-1 pt-0.5">
          {cols.map((c) => (
            <span key={c.name} className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
              <Highlight text={c.name} terms={terms} />
            </span>
          ))}
          {t.columns.length > cols.length && (
            <span className="rounded px-1 py-0.5 text-[10px] text-zinc-500">+{t.columns.length - cols.length}</span>
          )}
        </span>
      )}
    </button>
  )
}

// ── One table ──

type Tab = 'summary' | 'data' | 'columns'

function TableView({ item, terms, phone, onBack, onClose, onChanged, onRescan, onOpenAgent }: {
  item: DataTable; terms: string[]; phone: boolean
  onBack: () => void; onClose: () => void
  onChanged: (fresh: DataTable) => void
  /** Have the library look for changed tables again. */
  onRescan: () => Promise<void>
  onOpenAgent?: () => void
}) {
  const notify = useNotificationStore((s) => s.add)
  const [detail, setDetail] = useState<DataTable>(item)
  const [tab, setTab] = useState<Tab>(phone ? 'summary' : 'data')
  const [summaryOpen, setSummaryOpen] = useState(true)
  const [busy, setBusy] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  // Bumped by Refresh: the grid reloads its page even when nothing it knows
  // of has changed.
  const [nonce, setNonce] = useState(0)

  // The column profile comes with the table itself — again when a new
  // summary or a change arrives.
  useEffect(() => {
    let live = true
    dataLibraryApi.table(item.id).then((d) => { if (live) setDetail(d) }).catch(() => {})
    return () => { live = false }
  }, [item.id, item.status, item.summarized_at, item.row_count, item.updated_at])

  const t = { ...detail, ...item, profile: detail.profile }
  const kinds = useMemo(() => {
    const m: Record<string, ColumnKind> = {}
    for (const p of t.profile ?? []) m[p.name] = p.kind
    return m
  }, [t.profile])

  const redo = async () => {
    setBusy(true)
    try {
      const fresh = await dataLibraryApi.resummarize(item.id)
      setDetail(fresh)
      onChanged(fresh)
    } catch (e) {
      notify('error', 'Could not summarise the table', e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  // The rows are read live, so the grid's page reloads straight away. The
  // row count, summary and column profile are the library's from its last
  // look at the table, so it looks again first; a change it finds arrives
  // through the list like any other.
  const refresh = async () => {
    setRefreshing(true)
    setNonce((n) => n + 1)
    try {
      await onRescan()
      const fresh = await dataLibraryApi.table(item.id)
      setDetail(fresh)
      onChanged(fresh)
    } catch (e) {
      notify('error', 'Could not refresh the table', e instanceof Error ? e.message : String(e))
    } finally {
      setRefreshing(false)
    }
  }

  const btn = 'flex items-center justify-center gap-1.5 rounded-lg border border-zinc-800 px-3 py-1.5 text-sm text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:opacity-50'
  const actions = (
    <>
      <ExportMenu item={item} className={btn} phone={phone} />
      <button onClick={redo} disabled={busy || t.status === 'pending'} className={btn} title="Read the table again and write a fresh summary">
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}{phone ? 'Summarise' : 'Summarise again'}
      </button>
      {onOpenAgent && (
        <button onClick={onOpenAgent} className={btn} title={`Chat with ${t.agent_name}`}>
          <MessageSquare className="h-4 w-4" />{phone ? 'Chat' : 'Agent chat'}
        </button>
      )}
    </>
  )

  const top = (
    <div className="flex items-center gap-3 border-b border-zinc-800 px-3 py-2.5 wide:px-4">
      <button onClick={onBack} className={`${iconBtn} shrink-0`} title="All tables" aria-label="All tables">
        <ArrowLeft className="h-4 w-4" />
      </button>
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
        <Table2 className="h-[18px] w-[18px]" />
      </span>
      <div className="min-w-0 flex-1">
        <h2 className="truncate text-base font-semibold text-zinc-100">{t.title || humanize(t.table_name)}</h2>
        <p className="truncate text-xs text-zinc-500"><span className="font-mono">{t.table_name}</span> · {t.agent_name}</p>
      </div>
      {!phone && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      <button onClick={refresh} disabled={refreshing} className={`${iconBtn} shrink-0`} title="Refresh — the table as it is now" aria-label="Refresh">
        <RotateCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
      </button>
      <FullscreenButton className={`${iconBtn} shrink-0`} />
      <button onClick={onClose} className={`${iconBtn} shrink-0`} title="Close" aria-label="Close">
        <X className="h-4 w-4" />
      </button>
    </div>
  )

  const tabBtn = (id: Tab, label: string, count?: number) => (
    <button
      key={id}
      role="tab"
      aria-selected={tab === id}
      onClick={() => setTab(id)}
      className={`relative px-3 py-2.5 text-sm font-medium transition-colors ${
        tab === id ? 'text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}
    >
      {label}{count !== undefined && <span className="ml-1.5 text-[11px] tabular-nums text-zinc-500">{fmtNum(count)}</span>}
      {tab === id && <span className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-emerald-500" />}
    </button>
  )

  const grid = (
    <DataGrid
      tableId={item.id}
      kinds={kinds}
      notes={t.column_notes}
      phone={phone}
      version={`${item.row_count}:${item.updated_at}:${nonce}`}
      onTableChanged={onChanged}
    />
  )
  const columns = (
    <div className="h-full overflow-y-auto">
      <ColumnProfiles profile={t.profile} notes={t.column_notes} rows={t.row_count} phone={phone} />
    </div>
  )

  if (phone) {
    return (
      <div className="flex h-full flex-col bg-zinc-950">
        {top}
        <div role="tablist" className="flex shrink-0 border-b border-zinc-800 px-2">
          {tabBtn('summary', 'Summary')}
          {tabBtn('data', 'Data', t.row_count)}
          {tabBtn('columns', 'Columns', t.columns.length)}
        </div>
        <div className="min-h-0 flex-1">
          {tab === 'summary' ? (
            <div className="h-full overflow-y-auto px-4 py-4">
              <SummaryPanel t={t} terms={terms} />
              <div className="mt-5 grid grid-cols-2 gap-2">{actions}</div>
            </div>
          ) : tab === 'data' ? grid : columns}
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col bg-zinc-950">
      {top}
      {summaryOpen && (
        <div className="max-h-[42%] shrink-0 overflow-y-auto border-b border-zinc-800 px-5 py-4">
          <SummaryPanel t={t} terms={terms} />
        </div>
      )}
      <div role="tablist" className="flex shrink-0 items-center border-b border-zinc-800 px-3">
        {tabBtn('data', 'Data', t.row_count)}
        {tabBtn('columns', 'Columns', t.columns.length)}
        <button
          onClick={() => setSummaryOpen((v) => !v)}
          className="ml-auto flex items-center gap-1 rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
        >
          {summaryOpen ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          {summaryOpen ? 'Hide summary' : 'Show summary'}
        </button>
      </div>
      <div className="min-h-0 flex-1">{tab === 'columns' ? columns : grid}</div>
    </div>
  )
}

const EXPORTS: { format: ExportFormat; label: string; hint: string }[] = [
  { format: 'csv', label: 'CSV', hint: 'Opens in Excel, Numbers or Google Sheets' },
  { format: 'json', label: 'JSON', hint: 'Every row as an object, for code and other tools' },
  { format: 'xlsx', label: 'Excel (XLSX)', hint: 'A workbook, ready for Excel' },
]

/** Download the whole table as CSV, JSON or an Excel workbook. */
function ExportMenu({ item, className, phone }: { item: DataTable; className: string; phone: boolean }) {
  const notify = useNotificationStore((s) => s.add)
  const [open, setOpen] = useState(false)
  const [saving, setSaving] = useState(false)

  // The menu closes on Escape before the table does.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.stopPropagation()
      setOpen(false)
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [open])

  const save = async (format: ExportFormat) => {
    setOpen(false)
    setSaving(true)
    try {
      await dataLibraryApi.download(item.id, format, item.table_name)
    } catch (e) {
      notify('error', 'Could not export the table', e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        disabled={saving}
        className={`${className} w-full`}
        title="Download the whole table"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}Export
        <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-20" onClick={() => setOpen(false)} />
          {/* On a phone the buttons sit at the end of the summary: the menu
              opens upwards so it is never below the fold. */}
          <div
            role="menu"
            className={`absolute z-30 min-w-[240px] rounded-lg border border-zinc-800 bg-zinc-900 py-1 shadow-xl ${
              phone ? 'bottom-full left-0 mb-1' : 'right-0 top-full mt-1'}`}
          >
            {EXPORTS.map((x) => (
              <button
                key={x.format}
                role="menuitem"
                onClick={() => save(x.format)}
                className="block w-full px-3 py-2 text-left transition-colors hover:bg-zinc-800"
              >
                <span className="block text-sm text-zinc-200">{x.label}</span>
                <span className="block text-[11px] text-zinc-500">{x.hint}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// ── Summary: what the table is, at a glance ──

function SummaryPanel({ t, terms }: { t: DataTable; terms: string[] }) {
  const changed = isoEpoch(t.updated_at)
  const fill = t.profile && t.profile.length && t.row_count
    ? Math.round((t.profile.reduce((n, p) => n + p.filled, 0) / (t.profile.length * t.row_count)) * 100)
    : null
  const stats: [string, string, string?][] = [
    ['Rows', fmtNum(t.row_count)],
    ['Columns', String(t.columns.length)],
    ['Changed', changed ? fmtRelative(changed) : '—', changed ? fmtWhen(changed) : undefined],
    ['Filled in', fill === null ? '—' : `${fill}%`, 'How many of the table\'s cells hold a value'],
  ]
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-2 wide:grid-cols-4">
        {stats.map(([label, value, title]) => (
          <div key={label} className="rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2" title={title}>
            <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">{label}</div>
            <div className="mt-0.5 text-lg font-semibold tabular-nums text-zinc-100">{value}</div>
          </div>
        ))}
      </div>

      <div>
        <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
          <CategoryChip id={categoryOf(t)} />
          {t.tags.map((tag) => (
            <span key={tag} className="rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">#{tag}</span>
          ))}
        </div>
        {t.status === 'pending' && !t.summary ? (
          <p className="flex items-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" />Reading the table — its summary is on its way.</p>
        ) : t.status === 'error' ? (
          <div className="rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2.5 text-sm text-red-700 dark:text-red-300">
            <div className="flex items-center gap-1.5 font-medium"><AlertTriangle className="h-4 w-4" />Couldn't summarise it</div>
            {t.error && <div className="mt-1 break-words text-xs opacity-80">{t.error}</div>}
          </div>
        ) : null}
        {t.summary && (
          <p className={`text-sm leading-relaxed ${t.status === 'skipped' ? 'text-zinc-500' : 'text-zinc-300'} ${t.status === 'error' ? 'mt-3' : ''}`}>
            <Highlight text={t.summary} terms={terms} />
          </p>
        )}
        {t.status === 'pending' && t.summary && (
          <p className="mt-1.5 flex items-center gap-1.5 text-[11px] text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" />Changed since — a new summary is on its way.</p>
        )}
      </div>

      {t.insights.length > 0 && (
        <div>
          <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            <Lightbulb className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400" />Key facts
          </div>
          <ul className="flex flex-col gap-1">
            {t.insights.map((x, i) => (
              <li key={i} className="flex gap-2 text-sm leading-snug text-zinc-300">
                <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" />
                <span><Highlight text={x} terms={terms} /></span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {t.summarized_at && t.status !== 'pending' && t.model && (
        <div className="text-[11px] text-zinc-500">Summarised {fmtWhen(t.summarized_at)} · {t.model.split('/').pop()}</div>
      )}
    </div>
  )
}

// ── Data: the rows, live — and changed by hand ──
//
// Double-click a cell to change it in place, or open a row and click a
// value; "Add row" adds one, the row's bin deletes it. Every change goes
// straight to the agent's datastore (the agent's own rules apply: a
// protected row or cell stays as it is), and carries what you saw in the
// cell — one the agent changed meanwhile is never overwritten: the page
// reloads with what is there now.

type Col = { name: string; type: string }
/** Where the open record panel is: one row (by id, so an edit that moves it
 *  keeps it open), a new row being typed in, or nothing. */
type Panel = { kind: 'row'; id: number } | { kind: 'new' } | null
type Editing = { id: number; col: string; where: 'grid' | 'panel' } | null

function DataGrid({ tableId, kinds, notes, phone, version, onTableChanged }: {
  tableId: string
  kinds: Record<string, ColumnKind>
  notes: Record<string, string>
  phone: boolean
  /** Changes when the table does: the page reloads. */
  version: string
  /** A hand edit changed the table: its fresh row count and last change. */
  onTableChanged: (fresh: DataTable) => void
}) {
  const notify = useNotificationStore((s) => s.add)
  const [q, setQ] = useState('')
  const [dq, setDq] = useState('')
  const [sort, setSort] = useState<{ col: string; dir: 'asc' | 'desc' } | null>(null)
  const [offset, setOffset] = useState(0)
  const [data, setData] = useState<RowsPage | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [panel, setPanel] = useState<Panel>(null)
  // The open row as last known: it stays open when an edit moves it off the page.
  const [openCopy, setOpenCopy] = useState<DataRow | null>(null)
  const [editing, setEditing] = useState<Editing>(null)
  const [reload, setReload] = useState(0)
  const keepPanel = useRef(false)
  const reqId = useRef(0)

  useEffect(() => {
    const t = setTimeout(() => { setDq(q.trim()); setOffset(0) }, q ? 250 : 0)
    return () => clearTimeout(t)
  }, [q])

  useEffect(() => {
    const id = ++reqId.current
    setLoading(true)
    dataLibraryApi.rows(tableId, { limit: ROWS_PAGE, offset, q: dq, sort: sort?.col, dir: sort?.dir })
      .then((res) => {
        if (id !== reqId.current) return
        // The last row of the last page was deleted: the page before.
        if (!res.rows.length && offset > 0 && res.total > 0) {
          setOffset(Math.max(0, Math.floor((res.total - 1) / ROWS_PAGE) * ROWS_PAGE))
          return
        }
        setData(res)
        setError('')
      })
      .catch((e) => { if (id === reqId.current) setError(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (id === reqId.current) setLoading(false) })
  }, [tableId, dq, sort, offset, version, reload])

  // Escape: an editor cancels itself; otherwise the open record (or the new
  // row) closes before anything else does.
  useEffect(() => {
    if (panel === null) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if ((e.target as Element | null)?.closest?.('[data-cell-editor]')) return
      e.stopPropagation()
      setPanel(null)
      setEditing(null)
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [panel])
  useEffect(() => {
    if (keepPanel.current) { keepPanel.current = false; return }
    setPanel(null)
    setEditing(null)
  }, [dq, sort, offset])

  const cols = data?.columns ?? []
  const rows = data?.rows ?? []
  const total = data?.total ?? 0
  const terms = useMemo(() => dq.toLowerCase().split(/\s+/).filter(Boolean), [dq])
  const toggleSort = (col: string) => {
    setOffset(0)
    setSort((cur) => (!cur || cur.col !== col ? { col, dir: 'asc' } : cur.dir === 'asc' ? { col, dir: 'desc' } : null))
  }
  const from = total ? offset + 1 : 0
  const to = Math.min(offset + ROWS_PAGE, total)
  const pagerBtn = 'rounded-md p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100 disabled:opacity-30 disabled:hover:bg-transparent'

  const openId = panel?.kind === 'row' ? panel.id : null
  const openIndex = openId === null ? -1 : rows.findIndex((r) => r.__rowid === openId)
  const openRow = openIndex >= 0 ? rows[openIndex] : openCopy && openCopy.__rowid === openId ? openCopy : null
  const openRowAt = (i: number) => {
    const r = rows[i]
    if (!r) return
    setPanel({ kind: 'row', id: r.__rowid })
    setOpenCopy(r)
    setEditing((cur) => (cur && cur.id === r.__rowid ? cur : null))
  }

  // A click opens the row a moment later, so a double-click changes the cell
  // instead: opening the record narrows the grid, and the second click would
  // land on another cell. On a phone a tap opens the row at once (its record
  // is where a value is changed).
  const clickTimer = useRef<number | null>(null)
  useEffect(() => () => { if (clickTimer.current !== null) clearTimeout(clickTimer.current) }, [])
  const rowClick = (e: ReactMouseEvent, i: number) => {
    if (clickTimer.current !== null) { clearTimeout(clickTimer.current); clickTimer.current = null }
    const r = rows[i]
    const col = (e.target as Element | null)?.closest?.('td[data-col]')?.getAttribute('data-col')
    if (phone || !r) { openRowAt(i); return }
    if (e.detail >= 2) {
      if (!col) { openRowAt(i); return }
      window.getSelection()?.removeAllRanges()
      // An open record follows to this row (it's open already: nothing moves).
      if (panel?.kind === 'row') { setPanel({ kind: 'row', id: r.__rowid }); setOpenCopy(r) }
      setEditing({ id: r.__rowid, col, where: 'grid' })
      return
    }
    clickTimer.current = window.setTimeout(() => { clickTimer.current = null; openRowAt(i) }, 220)
  }

  const failed = (e: unknown) => e instanceof Error ? e.message : String(e)

  /** Save one cell; '' when saved (or when the page reloads instead), else
   *  why not, for the editor to show. */
  const saveCell = async (row: DataRow, col: string, draft: string): Promise<string> => {
    try {
      const res = await dataLibraryApi.editRow(tableId, row.__rowid, { [col]: draft }, { [col]: row[col] ?? null })
      const fresh = res.row
      if (fresh) {
        setData((d) => d && { ...d, rows: d.rows.map((r) => (r.__rowid === fresh.__rowid ? fresh : r)) })
        setOpenCopy((c) => (c && c.__rowid === fresh.__rowid ? fresh : c))
      }
      onTableChanged(res.table)
      return ''
    } catch (e) {
      const status = apiStatus(e)
      const gone = status === 404
      if (gone || (status === 409 && /changed since/i.test(failed(e)))) {
        // Not this edit's to fix: the row as it is now.
        notify('error', gone ? 'The row is gone' : 'The row changed meanwhile', failed(e))
        setEditing(null)
        if (gone && openId === row.__rowid) setPanel(null)
        setReload((n) => n + 1)
        return ''
      }
      return failed(e)
    }
  }

  const addRow = async (values: Record<string, string>): Promise<string> => {
    try {
      const res = await dataLibraryApi.addRow(tableId, values)
      onTableChanged(res.table)
      if (res.row) {
        setOpenCopy(res.row)
        setPanel({ kind: 'row', id: res.row.__rowid })
        // The rows read oldest first: the new one is on the last page.
        if (!dq && !sort) {
          const last = Math.max(0, Math.floor((res.table.row_count - 1) / ROWS_PAGE) * ROWS_PAGE)
          if (last !== offset) { keepPanel.current = true; setOffset(last) }
        }
      } else {
        setPanel(null)
      }
      setReload((n) => n + 1)
      return ''
    } catch (e) {
      return failed(e)
    }
  }

  const deleteRow = async (row: DataRow): Promise<string> => {
    try {
      const res = await dataLibraryApi.deleteRow(tableId, row.__rowid)
      setPanel(null)
      setEditing(null)
      setOpenCopy(null)
      setData((d) => d && { ...d, rows: d.rows.filter((r) => r.__rowid !== row.__rowid), total: Math.max(0, d.total - 1) })
      onTableChanged(res.table)
      return ''
    } catch (e) {
      if (apiStatus(e) !== 404) return failed(e)
      notify('error', 'The row is gone', failed(e))
      setPanel(null)
      setReload((n) => n + 1)
      return ''
    }
  }

  const newRow = () => {
    setEditing(null)
    setPanel({ kind: 'new' })
  }

  const record = panel?.kind === 'new' ? (
    <NewRowPanel columns={cols} kinds={kinds} notes={notes} phone={phone} onAdd={addRow} onClose={() => setPanel(null)} />
  ) : openRow ? (
    <RecordPanel
      key={openRow.__rowid}
      row={openRow}
      index={openIndex >= 0 ? offset + openIndex + 1 : null}
      columns={cols}
      kinds={kinds}
      notes={notes}
      terms={terms}
      phone={phone}
      hasPrev={openIndex > 0}
      hasNext={openIndex >= 0 && openIndex < rows.length - 1}
      onPrev={() => openRowAt(openIndex - 1)}
      onNext={() => openRowAt(openIndex + 1)}
      onClose={() => { setPanel(null); setEditing(null) }}
      editingCol={editing?.where === 'panel' && editing.id === openRow.__rowid ? editing.col : null}
      onEdit={(col) => setEditing({ id: openRow.__rowid, col, where: 'panel' })}
      onSave={(col, draft) => saveCell(openRow, col, draft)}
      onDoneEditing={() => setEditing(null)}
      onDelete={() => deleteRow(openRow)}
    />
  ) : null

  return (
    <div className="relative flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-3 py-2">
        <div className="max-w-sm flex-1"><SearchBox value={q} onChange={setQ} placeholder="Search the rows…" small /></div>
        <button
          onClick={newRow}
          disabled={!cols.length}
          className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-800 px-2.5 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:opacity-40"
          title="Add a row by hand"
          aria-label="Add row"
        >
          <Plus className="h-3.5 w-3.5" />{!phone && 'Add row'}
        </button>
        <span className="ml-auto shrink-0 text-xs tabular-nums text-zinc-500">
          {loading && <Loader2 className="mr-1.5 inline h-3 w-3 animate-spin" />}
          {total ? `${fmtNum(from)}–${fmtNum(to)} of ${fmtNum(total)}${dq ? ' matching' : ''}` : dq ? 'No matching rows' : ''}
        </span>
        <button onClick={() => setOffset((o) => Math.max(0, o - ROWS_PAGE))} disabled={offset === 0} className={pagerBtn} aria-label="Previous page" title="Previous page">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <button onClick={() => setOffset((o) => o + ROWS_PAGE)} disabled={to >= total} className={pagerBtn} aria-label="Next page" title="Next page">
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
      <div className="flex min-h-0 flex-1">
        <div className="min-w-0 flex-1 overflow-auto">
          {error ? (
            <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-sm text-zinc-500">
              <AlertTriangle className="h-5 w-5 text-red-500" />{error}
            </div>
          ) : !data && loading ? (
            <div className="flex h-full justify-center pt-12 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
          ) : rows.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-zinc-500">
              {dq ? 'No row matches that search.' : 'This table has no rows yet.'}
              {!dq && cols.length > 0 && (
                <button onClick={newRow} className="flex items-center gap-1.5 rounded-lg border border-zinc-800 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800/60">
                  <Plus className="h-4 w-4" />Add the first row
                </button>
              )}
            </div>
          ) : (
            <table className="w-max min-w-full border-separate border-spacing-0 text-[13px]">
              <thead className="sticky top-0 z-10">
                <tr>
                  <th className="sticky left-0 z-20 border-b border-zinc-800 bg-zinc-900 px-3 py-2 text-right text-[11px] font-medium text-zinc-500">#</th>
                  {cols.map((c) => {
                    const sorted = sort?.col === c.name ? sort.dir : null
                    const numeric = kinds[c.name] === 'number'
                    return (
                      <th
                        key={c.name}
                        onClick={() => toggleSort(c.name)}
                        title={`${c.name}${notes[c.name] ? ` — ${notes[c.name]}` : ''} · click to sort`}
                        className="cursor-pointer select-none border-b border-zinc-800 bg-zinc-900 px-3 py-2 font-medium text-zinc-300 hover:text-zinc-100"
                      >
                        <span className={`flex items-center gap-1.5 whitespace-nowrap ${numeric ? 'justify-end' : ''}`}>
                          <KindIcon kind={kinds[c.name]} />
                          {humanize(c.name)}
                          {sorted === 'asc' ? <ChevronUp className="h-3.5 w-3.5 text-emerald-500" />
                            : sorted === 'desc' ? <ChevronDown className="h-3.5 w-3.5 text-emerald-500" />
                              : <ChevronsUpDown className="h-3 w-3 text-zinc-600" />}
                        </span>
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody className={loading ? 'opacity-60' : ''}>
                {rows.map((r, i) => (
                  <tr
                    key={r.__rowid}
                    onClick={(e) => rowClick(e, i)}
                    className={`cursor-pointer transition-colors ${openId === r.__rowid ? 'bg-emerald-500/10' : 'hover:bg-zinc-900/70'}`}
                  >
                    <td className="sticky left-0 border-b border-zinc-800/60 bg-zinc-950 px-3 py-1.5 text-right text-[11px] tabular-nums text-zinc-600">
                      {offset + i + 1}
                    </td>
                    {cols.map((c) => {
                      const numeric = typeof r[c.name] === 'number' && c.type !== 'boolean'
                      const here = editing?.where === 'grid' && editing.id === r.__rowid && editing.col === c.name
                      const cell = <Cell value={r[c.name]} kind={kinds[c.name]} type={c.type} terms={terms} />
                      return (
                        <td
                          key={c.name}
                          data-col={c.name}
                          className={`relative max-w-[320px] border-b border-zinc-800/60 px-3 py-1.5 text-zinc-300 ${
                            here ? 'whitespace-nowrap' : 'truncate'} ${numeric ? 'text-right tabular-nums' : ''}`}
                          title={here ? undefined : `${r[c.name] == null ? '' : `${String(r[c.name])}\n`}Double-click to change`}
                        >
                          {here ? (
                            <>
                              {/* The cell keeps its size under the editor. */}
                              <span className="invisible block truncate">{cell}</span>
                              <div
                                onClick={(e) => e.stopPropagation()}
                                className="absolute left-0 top-0 z-30 w-[max(100%,300px)] whitespace-normal rounded-lg border border-zinc-700 bg-zinc-900 p-2 text-left shadow-xl"
                              >
                                <CellEditor
                                  col={c}
                                  value={r[c.name]}
                                  compact
                                  onSave={(draft) => saveCell(r, c.name, draft)}
                                  onDone={() => setEditing(null)}
                                />
                              </div>
                            </>
                          ) : cell}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        {record && !phone && <div className="w-[360px] shrink-0 border-l border-zinc-800 bg-zinc-900/30">{record}</div>}
      </div>
      {record && phone && <div className="absolute inset-0 z-20 bg-zinc-950">{record}</div>}
    </div>
  )
}

/** One value as it reads best: links and emails clickable, numbers and dates
 *  formatted, booleans as ticks, empty as a dash. */
function Cell({ value, kind, type, terms, full = false }: {
  value: unknown; kind?: ColumnKind; type: string; terms: string[]; full?: boolean
}) {
  if (value === null || value === undefined || value === '') return <span className="text-zinc-600">—</span>
  if (type === 'boolean' || kind === 'boolean') {
    const yes = value === 1 || value === true || value === '1' || String(value).toLowerCase() === 'true'
    return yes
      ? <Check className="inline h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" aria-label="yes" />
      : <X className="inline h-3.5 w-3.5 text-zinc-500" aria-label="no" />
  }
  if (typeof value === 'number') return <>{fmtNum(value)}</>
  const s = String(value)
  if (URL_RE.test(s.trim()) && !/\s/.test(s.trim())) {
    return (
      <a
        href={s.trim()}
        target="_blank"
        rel="noopener noreferrer"
        onClick={(e) => e.stopPropagation()}
        className="text-sky-700 hover:underline dark:text-sky-400"
      >
        {full ? s : prettyUrl(s.trim())}
      </a>
    )
  }
  if (EMAIL_RE.test(s.trim())) {
    return (
      <a href={`mailto:${s.trim()}`} onClick={(e) => e.stopPropagation()} className="text-sky-700 hover:underline dark:text-sky-400">
        <Highlight text={s} terms={terms} />
      </a>
    )
  }
  if (kind === 'date' || type === 'date' || type === 'datetime') return <>{fmtDate(s)}</>
  if (full && (type === 'json' || kind === 'json')) {
    try {
      return <pre className="whitespace-pre-wrap break-words font-mono text-xs">{JSON.stringify(JSON.parse(s), null, 2)}</pre>
    } catch { /* not JSON after all */ }
  }
  return <span className={full ? 'whitespace-pre-wrap break-words' : ''}><Highlight text={s} terms={terms} /></span>
}

// ── Editing a value ──

const DATE_ONLY_RE = /^\d{4}-\d{2}-\d{2}$/
const editInput = 'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500/70 focus:outline-none'

/** A stored value as an editor holds it: plain text and numbers, yes / no. */
function toDraft(v: unknown, type: string): string {
  if (v === null || v === undefined) return ''
  if (type === 'boolean') {
    const s = String(v).trim().toLowerCase()
    if (v === true || ['1', 'true', 'yes'].includes(s)) return 'yes'
    if (v === false || ['0', 'false', 'no'].includes(s)) return 'no'
  }
  return String(v)
}

function editHint(type: string): string {
  if (type === 'boolean') return 'Esc cancels'
  if (type === 'json') return '⌘/Ctrl+Enter saves · Esc cancels'
  if (['integer', 'real', 'date', 'datetime'].includes(type)) return 'Enter saves · Esc cancels'
  return 'Enter saves · Shift+Enter new line · Esc cancels'
}

/** The input for one column's type. Enter saves (⌘/Ctrl+Enter in JSON,
 *  where Enter is a new line); empty is no value. */
function FieldInput({ col, value, onChange, onSubmit, onPick, autoFocus = false }: {
  col: Col
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  /** Yes / No / Empty picked: save it at once (one cell), when given. */
  onPick?: (v: string) => void
  autoFocus?: boolean
}) {
  const area = useRef<HTMLTextAreaElement>(null)
  // A text box grows with what it holds, up to a point.
  useLayoutEffect(() => {
    const el = area.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(260, el.scrollHeight + 2)}px`
  }, [value])
  const caretToEnd = (e: { currentTarget: HTMLInputElement | HTMLTextAreaElement }) => {
    const el = e.currentTarget
    if (el.type === 'date') return
    const n = el.value.length
    try { el.setSelectionRange(n, n) } catch { /* not a text input */ }
  }
  const enter = (e: ReactKeyboardEvent) => {
    if (e.key !== 'Enter' || e.nativeEvent.isComposing) return
    e.preventDefault()
    onSubmit()
  }
  const label = humanize(col.name)

  if (col.type === 'boolean') {
    const opts: [string, string][] = [['yes', 'Yes'], ['no', 'No'], ['', 'Empty']]
    const known = opts.some(([v]) => v === value)
    return (
      <div className="flex gap-1" role="radiogroup" aria-label={label}>
        {opts.map(([v, text], i) => (
          <button
            key={text}
            type="button"
            role="radio"
            aria-checked={value === v}
            autoFocus={autoFocus && (value === v || (!known && i === 0))}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => (onPick ? onPick(v) : onChange(v))}
            className={`rounded-md border px-2.5 py-1 text-xs transition-colors ${value === v
              ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
              : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`}
          >
            {text}
          </button>
        ))}
      </div>
    )
  }
  if (col.type === 'integer' || col.type === 'real') {
    return (
      <input
        type="text"
        inputMode={col.type === 'integer' ? 'numeric' : 'decimal'}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={enter}
        onFocus={caretToEnd}
        autoFocus={autoFocus}
        placeholder={col.type === 'integer' ? 'A whole number' : 'A number'}
        aria-label={label}
        className={`${editInput} tabular-nums`}
      />
    )
  }
  if (col.type === 'date' && (value === '' || DATE_ONLY_RE.test(value))) {
    return (
      <input
        type="date"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={enter}
        autoFocus={autoFocus}
        aria-label={label}
        className={`${editInput} dark:[color-scheme:dark]`}
      />
    )
  }
  if (col.type === 'date' || col.type === 'datetime') {
    return (
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={enter}
        onFocus={caretToEnd}
        autoFocus={autoFocus}
        placeholder={col.type === 'date' ? '2026-11-11' : '2026-11-11T14:30'}
        aria-label={label}
        className={`${editInput} tabular-nums`}
      />
    )
  }
  const json = col.type === 'json'
  return (
    <textarea
      ref={area}
      rows={1}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={(e) => {
        if (e.key !== 'Enter' || e.nativeEvent.isComposing) return
        if (json ? (e.metaKey || e.ctrlKey) : !e.shiftKey) {
          e.preventDefault()
          onSubmit()
        }
      }}
      onFocus={caretToEnd}
      autoFocus={autoFocus}
      spellCheck={!json}
      aria-label={label}
      className={`${editInput} block resize-none leading-snug ${json ? 'font-mono text-xs' : ''}`}
    />
  )
}

/** One cell being changed: saves on Enter, on Save or when you click away
 *  (only when it changed); Esc or Cancel leaves it as it was. A value that
 *  doesn't fit stays open with the reason. */
function CellEditor({ col, value, compact = false, onSave, onDone }: {
  col: Col
  value: unknown
  compact?: boolean
  /** Resolves to why it couldn't be saved, or '' when it's done. */
  onSave: (draft: string) => Promise<string>
  onDone: () => void
}) {
  const initial = toDraft(value, col.type)
  const [draft, setDraft] = useState(initial)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const busy = useRef(false)
  const root = useRef<HTMLDivElement>(null)
  // The editor, its buttons and a reason it couldn't save are in sight —
  // in the grid it floats over the cell, which can sit at the edge.
  useEffect(() => {
    const el = compact ? root.current?.parentElement : root.current
    el?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [compact, error])

  const commit = async (v: string = draft) => {
    if (busy.current) return
    if (v === initial) { onDone(); return }
    busy.current = true
    setSaving(true)
    setError('')
    const why = await onSave(v)
    busy.current = false
    setSaving(false)
    if (why) setError(why)
    else onDone()
  }

  const small = 'rounded-md px-2.5 py-1 text-xs font-medium transition-colors disabled:opacity-50'
  return (
    <div
      ref={root}
      data-cell-editor
      onKeyDown={(e) => {
        if (e.key !== 'Escape') return
        e.stopPropagation()
        onDone()
      }}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) void commit()
      }}
    >
      <FieldInput
        col={col}
        value={draft}
        onChange={(v) => { setDraft(v); setError('') }}
        onSubmit={() => void commit()}
        onPick={(v) => { setDraft(v); void commit(v) }}
        autoFocus
      />
      {error && <p className="mt-1.5 text-[11px] leading-snug text-red-600 dark:text-red-400">{error}</p>}
      <div className="mt-1.5 flex items-center gap-1.5">
        {col.type !== 'boolean' && (
          <button
            type="button"
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => void commit()}
            disabled={saving}
            className={`${small} flex items-center gap-1 bg-emerald-600 text-white hover:bg-emerald-500`}
          >
            {saving && <Loader2 className="h-3 w-3 animate-spin" />}Save
          </button>
        )}
        <button
          type="button"
          onMouseDown={(e) => e.preventDefault()}
          onClick={onDone}
          disabled={saving}
          className={`${small} text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200`}
        >
          Cancel
        </button>
        {col.type === 'boolean' && saving && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
        {!compact && <span className="ml-auto truncate text-[10px] text-zinc-500">{editHint(col.type)}</span>}
      </div>
    </div>
  )
}

// ── One record ──

function RecordPanel({
  row, index, columns, kinds, notes, terms, phone, hasPrev, hasNext, onPrev, onNext, onClose,
  editingCol, onEdit, onSave, onDoneEditing, onDelete,
}: {
  row: DataRow
  /** Its place in the table as shown; null when an edit moved it off this page. */
  index: number | null
  columns: Col[]
  kinds: Record<string, ColumnKind>; notes: Record<string, string>; terms: string[]; phone: boolean
  hasPrev: boolean; hasNext: boolean; onPrev: () => void; onNext: () => void; onClose: () => void
  /** The column being changed here, if any. */
  editingCol: string | null
  onEdit: (col: string) => void
  onSave: (col: string, draft: string) => Promise<string>
  onDoneEditing: () => void
  /** Resolves to why it couldn't be deleted, or '' when it's gone. */
  onDelete: () => Promise<string>
}) {
  const [copied, setCopied] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')
  const copy = async () => {
    const obj: Record<string, unknown> = {}
    for (const c of columns) obj[c.name] = row[c.name]
    try {
      await navigator.clipboard.writeText(JSON.stringify(obj, null, 2))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard refused */ }
  }
  const remove = async () => {
    setDeleting(true)
    setDeleteError('')
    const why = await onDelete()
    setDeleting(false)
    if (why) setDeleteError(why)
  }
  const small = 'rounded-md p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100 disabled:opacity-30'
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-1 border-b border-zinc-800 px-3 py-2">
        <span className="flex-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
          {index === null ? 'Row' : `Row ${fmtNum(index)}`}
        </span>
        <button onClick={onPrev} disabled={!hasPrev} className={small} title="Previous row" aria-label="Previous row"><ChevronUp className="h-4 w-4" /></button>
        <button onClick={onNext} disabled={!hasNext} className={small} title="Next row" aria-label="Next row"><ChevronDown className="h-4 w-4" /></button>
        <button onClick={copy} className={small} title="Copy the row as JSON" aria-label="Copy the row">
          {copied ? <Check className="h-4 w-4 text-emerald-500" /> : <Copy className="h-4 w-4" />}
        </button>
        <button
          onClick={() => { setConfirming(true); setDeleteError('') }}
          className={`${small} hover:text-red-600 dark:hover:text-red-400`}
          title="Delete this row"
          aria-label="Delete this row"
        >
          <Trash2 className="h-4 w-4" />
        </button>
        <button onClick={onClose} className={small} title="Close" aria-label="Close row"><X className={phone ? 'h-5 w-5' : 'h-4 w-4'} /></button>
      </div>
      {confirming && (
        <div className="border-b border-red-500/30 bg-red-500/5 px-3 py-2.5">
          <div className="flex items-center gap-2">
            <span className="flex-1 text-sm text-red-700 dark:text-red-300">Delete this row? It can't be undone.</span>
            <button
              onClick={remove}
              disabled={deleting}
              className="flex items-center gap-1 rounded-md bg-red-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-red-500 disabled:opacity-50"
            >
              {deleting && <Loader2 className="h-3 w-3 animate-spin" />}Delete
            </button>
            <button
              onClick={() => setConfirming(false)}
              disabled={deleting}
              className="rounded-md px-2.5 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              Cancel
            </button>
          </div>
          {deleteError && <p className="mt-1.5 text-[11px] leading-snug text-red-600 dark:text-red-400">{deleteError}</p>}
        </div>
      )}
      <dl className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {columns.map((c) => {
          const editing = editingCol === c.name
          const label = humanize(c.name)
          return (
            <div key={c.name} className="group border-b border-zinc-800/60 py-2.5 last:border-0">
              <dt className="text-[11px] text-zinc-500">
                <span className="flex items-center gap-1.5">
                  <KindIcon kind={kinds[c.name]} className="h-3 w-3" />
                  <span className="font-medium text-zinc-400">{label}</span>
                  {!editing && (
                    <button
                      onClick={() => onEdit(c.name)}
                      className={`ml-auto rounded p-0.5 text-zinc-500 transition-opacity hover:bg-zinc-800 hover:text-zinc-200 focus:opacity-100 ${
                        phone ? '' : 'opacity-0 group-hover:opacity-100'}`}
                      title={`Change ${label}`}
                      aria-label={`Change ${label}`}
                    >
                      <Pencil className="h-3 w-3" />
                    </button>
                  )}
                </span>
                {notes[c.name] && <span className="mt-0.5 line-clamp-2 block leading-snug">{notes[c.name]}</span>}
              </dt>
              <dd className="mt-1 break-words text-sm text-zinc-200">
                {editing ? (
                  <CellEditor col={c} value={row[c.name]} onSave={(draft) => onSave(c.name, draft)} onDone={onDoneEditing} />
                ) : (
                  <div
                    onClick={() => {
                      // Selecting text to copy it isn't asking to change it.
                      if (window.getSelection()?.toString()) return
                      onEdit(c.name)
                    }}
                    className="-mx-1 cursor-text rounded px-1 py-0.5 transition-colors hover:bg-zinc-800/50"
                    title="Click to change"
                  >
                    <Cell value={row[c.name]} kind={kinds[c.name]} type={c.type} terms={terms} full />
                  </div>
                )}
              </dd>
            </div>
          )
        })}
      </dl>
      {!phone && (
        <div className="border-t border-zinc-800 px-4 py-2 text-[10px] leading-snug text-zinc-500">
          Click a value to change it — changes go straight to the agent's datastore.
        </div>
      )}
    </div>
  )
}

/** A new row typed in by hand: every column, empty to start with. */
function NewRowPanel({ columns, kinds, notes, phone, onAdd, onClose }: {
  columns: Col[]
  kinds: Record<string, ColumnKind>
  notes: Record<string, string>
  phone: boolean
  /** Resolves to why it couldn't be added, or '' when it's in. */
  onAdd: (values: Record<string, string>) => Promise<string>
  onClose: () => void
}) {
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const submit = async () => {
    if (saving) return
    setSaving(true)
    setError('')
    const why = await onAdd(Object.fromEntries(Object.entries(draft).filter(([, v]) => v !== '')))
    setSaving(false)
    if (why) setError(why)
  }
  return (
    <div
      data-cell-editor
      className="flex h-full flex-col"
      onKeyDown={(e) => {
        if (e.key !== 'Escape') return
        e.stopPropagation()
        onClose()
      }}
    >
      <div className="flex items-center gap-1 border-b border-zinc-800 px-3 py-2">
        <span className="flex-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">New row</span>
        <button
          onClick={onClose}
          className="rounded-md p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100"
          title="Close"
          aria-label="Close new row"
        >
          <X className={phone ? 'h-5 w-5' : 'h-4 w-4'} />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {columns.map((c, i) => (
          <div key={c.name} className="border-b border-zinc-800/60 py-2.5 last:border-0">
            <span className="flex items-center gap-1.5 text-[11px]">
              <KindIcon kind={kinds[c.name]} className="h-3 w-3" />
              <span className="font-medium text-zinc-400">{humanize(c.name)}</span>
            </span>
            {notes[c.name] && <span className="mt-0.5 line-clamp-2 block text-[11px] leading-snug text-zinc-500">{notes[c.name]}</span>}
            <div className="mt-1.5">
              <FieldInput
                col={c}
                value={draft[c.name] ?? ''}
                onChange={(v) => setDraft((d) => ({ ...d, [c.name]: v }))}
                onSubmit={() => void submit()}
                autoFocus={i === 0}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="border-t border-zinc-800 px-4 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
        {error && <p className="mb-2 text-[11px] leading-snug text-red-600 dark:text-red-400">{error}</p>}
        <div className="flex items-center gap-2">
          <button
            onClick={() => void submit()}
            disabled={saving}
            className="flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
          >
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}Add row
          </button>
          <button onClick={onClose} disabled={saving} className="rounded-lg px-3 py-1.5 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
            Cancel
          </button>
          {!phone && <span className="ml-auto text-[10px] text-zinc-500">Empty fields stay empty</span>}
        </div>
      </div>
    </div>
  )
}

// ── Columns: each column's profile ──

function ColumnProfiles({ profile, notes, rows, phone }: {
  profile?: ColumnProfile[]; notes: Record<string, string>; rows: number; phone: boolean
}) {
  if (!profile) return <div className="flex justify-center pt-12 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
  if (profile.length === 0) return <div className="px-6 pt-12 text-center text-sm text-zinc-500">No column profile yet.</div>
  return (
    <div className={`grid gap-3 p-4 ${phone ? 'grid-cols-1' : 'grid-cols-2 xl:grid-cols-3'}`}>
      {profile.map((p) => <ColumnCard key={p.name} p={p} note={notes[p.name]} rows={rows} />)}
    </div>
  )
}

function Bar({ value, max, tone = 'bg-emerald-500' }: { value: number; max: number; tone?: string }) {
  const pct = max > 0 ? Math.max(2, Math.round((value / max) * 100)) : 0
  return (
    <span className="block h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
      <span className={`block h-full rounded-full ${tone}`} style={{ width: `${pct}%` }} />
    </span>
  )
}

function ColumnCard({ p, note, rows }: { p: ColumnProfile; note?: string; rows: number }) {
  const fillPct = rows ? Math.round((p.filled / rows) * 100) : 0
  const stat = (label: string, value: ReactNode) => (
    <div className="min-w-0 rounded-md bg-zinc-800/50 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div className="truncate text-[13px] font-medium tabular-nums text-zinc-200">{value}</div>
    </div>
  )
  const kindLabel: Record<ColumnKind, string> = {
    text: 'Text', number: 'Number', url: 'Links', email: 'Email addresses', date: 'Dates', boolean: 'Yes / no', json: 'JSON',
  }
  return (
    <div className="flex flex-col gap-2.5 rounded-xl border border-zinc-800 bg-zinc-900/40 p-3.5">
      <div className="flex items-start gap-2">
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-zinc-800/80">
          <KindIcon kind={p.kind} className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold text-zinc-100">{humanize(p.name)}</div>
          <div className="truncate font-mono text-[10px] text-zinc-500">{p.name}</div>
        </div>
        <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">{kindLabel[p.kind] || p.type}</span>
      </div>
      {note && <p className="text-xs leading-snug text-zinc-400">{note}</p>}
      <div>
        <div className="mb-1 flex justify-between text-[11px] text-zinc-500">
          <span>{fmtNum(p.filled)} of {fmtNum(rows)} filled</span>
          <span className={`tabular-nums ${fillPct < 50 ? 'text-amber-700 dark:text-amber-400' : ''}`}>{fillPct}%</span>
        </div>
        <Bar value={p.filled} max={rows} tone={fillPct < 50 ? 'bg-amber-500' : 'bg-emerald-500'} />
        {!!p.placeholders && (
          <div className="mt-1 text-[11px] text-amber-700 dark:text-amber-400">
            {fmtNum(p.placeholders)} say “unknown”, “TBD” or the like
          </div>
        )}
        <div className="mt-1 text-[11px] text-zinc-500">
          {p.distinct === p.filled && p.filled > 1 && !p.placeholders ? 'Every value is different' : `${fmtNum(p.distinct)} different value${p.distinct === 1 ? '' : 's'}`}
        </div>
      </div>
      {p.kind === 'number' && (
        <div className="grid grid-cols-4 gap-1.5">
          {stat('Min', fmtNum(p.min))}
          {stat('Avg', fmtNum(p.avg))}
          {stat('Max', fmtNum(p.max))}
          {stat('Sum', fmtNum(p.sum))}
        </div>
      )}
      {p.kind === 'date' && p.min != null && (
        <div className="grid grid-cols-2 gap-1.5">
          {stat('From', fmtDate(p.min))}
          {stat('To', fmtDate(p.max))}
        </div>
      )}
      {p.kind === 'boolean' && (
        <div className="grid grid-cols-2 gap-1.5">
          {stat('Yes', fmtNum(p.true ?? 0))}
          {stat('No', fmtNum(p.false ?? 0))}
        </div>
      )}
      {p.top && p.top.length > 0 && (
        <div>
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">Most common</div>
          <ul className="flex flex-col gap-1">
            {p.top.slice(0, 5).map((v) => (
              <li key={v.value} className="grid grid-cols-[minmax(0,1fr)_64px_auto] items-center gap-2 text-xs">
                <span className="truncate text-zinc-300" title={v.value}>{p.kind === 'date' ? fmtDate(v.value) : v.value}</span>
                <Bar value={v.count} max={p.top![0].count} tone="bg-sky-500" />
                <span className="w-8 text-right tabular-nums text-zinc-500">{fmtNum(v.count)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
