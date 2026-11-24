import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ComponentType, ReactNode } from 'react'
import {
  Briefcase, ChevronDown, ChevronRight, Clapperboard, Coins, Eye, Flag, Globe, Home, Lock, MapPin, OctagonX, PenLine,
  Plane, Play, Receipt, RotateCw, Search, SearchX, ShoppingBag, Sparkles, Star, X, Zap,
} from 'lucide-react'
import { listBrowsers, type ConnectedBrowser } from '../../services/browserClip'
import {
  flagOf, myRegion, playbooksApi,
  type Playbook, type PlaybookFront, type PlaybookInput, type PlaybookSummary,
} from '../../services/playbooksApi'

// ── Playbooks: the store in Flight Deck ─────────────────────────────
//
// The ready-made ways to get things done on the web that Captain for Chrome's
// panel offers (flight_deck/browser_playbooks.py, /fd/playbooks), in the
// Agent store's calm marketplace look: a big heading, category pills and a
// search, Popular as large cards, then each category as two columns of rows.
// A playbook opens as a sheet — what it does, where it runs, a short form —
// and Run sends the person's own message to the agent they pick, in its chat
// (the deck words it; SimpleLayout opens the chat and sends it).
//
// Both themes: zinc is the app's themed surface scale (html.light remaps it);
// only accents outside it carry a dark: variant.

/** An agent a playbook can go to (the layout's agents that can chat). */
export interface PlaybookAgent { id: string; name: string; reachable: boolean; asleep?: boolean }

interface Look { icon: ComponentType<{ className?: string }>; tile: string; wash: string }

const LOOKS: Record<string, Look> = {
  travel: { icon: Plane, tile: 'bg-sky-500/15 text-sky-600 ring-sky-500/25 dark:text-sky-300', wash: 'from-sky-500/[0.10]' },
  local: { icon: MapPin, tile: 'bg-emerald-500/15 text-emerald-600 ring-emerald-500/25 dark:text-emerald-300', wash: 'from-emerald-500/[0.10]' },
  shopping: { icon: ShoppingBag, tile: 'bg-pink-500/15 text-pink-600 ring-pink-500/25 dark:text-pink-300', wash: 'from-pink-500/[0.10]' },
  accounts: { icon: Receipt, tile: 'bg-amber-500/15 text-amber-700 ring-amber-500/25 dark:text-amber-300', wash: 'from-amber-500/[0.10]' },
  work: { icon: Briefcase, tile: 'bg-indigo-500/15 text-indigo-600 ring-indigo-500/25 dark:text-indigo-300', wash: 'from-indigo-500/[0.10]' },
  research: { icon: Search, tile: 'bg-cyan-500/15 text-cyan-700 ring-cyan-500/25 dark:text-cyan-300', wash: 'from-cyan-500/[0.10]' },
  money: { icon: Coins, tile: 'bg-lime-500/15 text-lime-700 ring-lime-500/25 dark:text-lime-300', wash: 'from-lime-500/[0.10]' },
  home: { icon: Home, tile: 'bg-orange-500/15 text-orange-600 ring-orange-500/25 dark:text-orange-300', wash: 'from-orange-500/[0.10]' },
  media: { icon: Clapperboard, tile: 'bg-fuchsia-500/15 text-fuchsia-600 ring-fuchsia-500/25 dark:text-fuchsia-300', wash: 'from-fuchsia-500/[0.10]' },
  regional: { icon: Flag, tile: 'bg-rose-500/15 text-rose-600 ring-rose-500/25 dark:text-rose-300', wash: 'from-rose-500/[0.10]' },
}
const PLAIN: Look = { icon: Zap, tile: 'bg-violet-500/15 text-violet-600 ring-violet-500/25 dark:text-violet-300', wash: 'from-violet-500/[0.10]' }
const lookOf = (category: string) => LOOKS[category] || PLAIN

const WHERE: Record<string, string> = { any: 'Any browser', prefer_chrome: 'Best in your Chrome', chrome: 'Your Chrome, signed in' }
const HERE = /^\s*(here|me|my location|current location|where i am|home)\s*[.!]?\s*$/i
const SECTION_PREVIEW = 6
const siteName = (site: string) => site.replace(/^www\./, '').replace(/\/+$/, '')

type ChromeState = 'unknown' | 'none' | 'set_up' | 'lent'

export function PlaybookStorePanel({ onClose, agents, preferAgentId, onRun, onConnections }: {
  onClose: () => void
  agents: PlaybookAgent[]
  /** The agent whose chat is open, if any: the one a run goes to unless they pick another. */
  preferAgentId?: string | null
  /** Run it: the agent's chat opens and this message goes there as theirs. */
  onRun: (agentId: string, content: string) => void
  /** Connections → Browser (Captain for Chrome isn't set up). */
  onConnections?: () => void
}) {
  const [front, setFront] = useState<PlaybookFront | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState<string | null>(null)
  const [items, setItems] = useState<PlaybookSummary[] | null>(null)
  const [listErr, setListErr] = useState('')
  const [openId, setOpenId] = useState<string | null>(null)
  const [chrome, setChrome] = useState<ChromeState>('unknown')
  const searchRef = useRef<HTMLInputElement>(null)
  const topRef = useRef<HTMLDivElement>(null)
  const seq = useRef(0)
  const region = useMemo(() => myRegion(), [])

  const load = useCallback(() => {
    setLoadErr('')
    playbooksApi.front(region).then(setFront).catch((e) => setLoadErr(e instanceof Error ? e.message : String(e)))
  }, [region])
  useEffect(() => { load() }, [load])
  useEffect(() => {
    listBrowsers()
      .then((rows) => {
        const browsers = rows.filter((r) => (r.kind || 'browser') === 'browser') as (ConnectedBrowser & { agents_can_use?: boolean })[]
        setChrome(browsers.some((b) => b.agents_can_use) ? 'lent' : browsers.length ? 'set_up' : 'none')
      })
      .catch(() => setChrome('unknown'))
  }, [])

  // A search or a category: the deck's list (typed searches wait a beat).
  const q = query.trim()
  useEffect(() => {
    if (!q && !category) { setItems(null); return }
    const mine = ++seq.current
    const t = setTimeout(() => {
      playbooksApi.search({ q, category: category || '', region, limit: 60 })
        .then((list) => { if (mine === seq.current) { setItems(list); setListErr('') } })
        .catch((e) => { if (mine === seq.current) setListErr(e instanceof Error ? e.message : String(e)) })
    }, q ? 200 : 0)
    return () => clearTimeout(t)
  }, [q, category, region])

  const choose = (key: string | null) => {
    setCategory(key)
    if (key) setItems(null)
    topRef.current?.scrollIntoView({ block: 'start' })
  }
  const typed = (text: string) => {
    setQuery(text)
    // A new search looks everywhere (Mine stays Mine).
    if (text.trim() && category && category !== 'mine') setCategory(null)
  }

  // Escape: the sheet first, then the store.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || openId) return
      if (query) { setQuery(''); return }
      onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [openId, query, onClose])

  // Kept or taken off the shelf: every list that shows it says so.
  const toggleMine = async (p: PlaybookSummary) => {
    const on = !p.mine
    try { await playbooksApi.shelf(p.id, on) } catch { return }
    const flip = (x: PlaybookSummary) => (x.id === p.id ? { ...x, mine: on } : x)
    setItems((cur) => (cur ? cur.map(flip) : cur))
    setFront((cur) => cur && {
      ...cur,
      popular: cur.popular.map(flip),
      top: Object.fromEntries(Object.entries(cur.top).map(([k, v]) => [k, v.map(flip)])),
      shelf: on ? [{ ...p, mine: true }, ...cur.shelf.filter((x) => x.id !== p.id)] : cur.shelf.filter((x) => x.id !== p.id),
    })
  }

  const cats = front?.categories || []
  const current = category === 'mine' ? { label: 'My playbooks' } : cats.find((c) => c.id === category)
  const browsing = !q && !category

  return (
    <div className="relative min-h-0 flex-1 overflow-y-auto overscroll-contain bg-zinc-950 text-zinc-100">
      <div className="mx-auto w-full max-w-5xl px-4 pb-16 pt-4 wide:px-8">
        <div className="flex items-center justify-end">
          <button onClick={onClose} aria-label="Close" title="Close"
            className="-mr-2 rounded-full p-2 text-zinc-500 transition-colors hover:bg-zinc-900 hover:text-zinc-200">
            <X className="h-5 w-5" />
          </button>
        </div>

        <header ref={topRef} className="scroll-mt-4 pt-1">
          <h1 className="text-4xl font-bold tracking-tight text-zinc-100 wide:text-5xl">Playbooks</h1>
          <p className="mt-2 max-w-2xl text-base text-zinc-500">
            {front ? `${front.total} ready-made ways` : 'Ready-made ways'} to get things done on the web — travel times, parcels,
            invoices, prices, bookings. Pick one, fill in a line or two, and an agent does it.
          </p>
        </header>

        {/* Categories and search */}
        <div className="sticky top-0 z-10 -mx-4 mt-8 flex flex-col gap-3 bg-zinc-950/90 px-4 py-3 backdrop-blur wide:-mx-8 wide:flex-row wide:items-center wide:px-8">
          <div role="tablist" aria-label="Categories"
            className="-mx-4 flex min-w-0 flex-1 gap-2 overflow-x-auto px-4 [mask-image:linear-gradient(to_right,black_calc(100%-2rem),transparent)] [scrollbar-width:none] wide:mx-0 wide:px-0 [&::-webkit-scrollbar]:hidden">
            <Pill active={!category} onClick={() => choose(null)}>All</Pill>
            {!!front?.shelf.length && (
              <Pill active={category === 'mine'} onClick={() => choose('mine')}>
                <Star className="h-3.5 w-3.5" /> Mine <span className="tabular-nums opacity-60">{front.shelf.length}</span>
              </Pill>
            )}
            {cats.map((c) => (
              <Pill key={c.id} active={category === c.id} onClick={() => choose(c.id)}>{c.label}</Pill>
            ))}
          </div>
          <div className="relative wide:w-80 wide:shrink-0">
            <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
            <input
              ref={searchRef}
              value={query}
              onChange={(e) => typed(e.target.value)}
              placeholder="What do you want done?"
              aria-label="Search playbooks"
              className="w-full rounded-full border border-zinc-800 bg-zinc-900/60 py-2.5 pl-10 pr-9 text-base text-zinc-100 placeholder-zinc-500 transition-colors focus:border-violet-500/50 focus:outline-none focus:ring-2 focus:ring-violet-500/15 wide:py-2 wide:text-sm"
            />
            {query && (
              <button onClick={() => { setQuery(''); searchRef.current?.focus() }} aria-label="Clear search"
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>

        <main className="mt-6 space-y-12">
          {loadErr && (
            <Empty icon={RotateCw} title="The playbooks couldn't load">
              <span className="block">{loadErr}</span>
              <button onClick={load} className="mt-3 font-medium text-violet-700 hover:underline dark:text-violet-300">Try again</button>
            </Empty>
          )}

          {!loadErr && !front && (
            <div role="status" aria-label="Loading playbooks" className="space-y-10">
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-5">
                {Array.from({ length: 5 }, (_, i) => <div key={i} className="fd-shimmer h-52 rounded-3xl" />)}
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                {Array.from({ length: 6 }, (_, i) => <div key={i} className="fd-shimmer h-16 rounded-2xl" />)}
              </div>
            </div>
          )}

          {front && browsing && (
            <>
              <section aria-labelledby="pb-popular">
                <h2 id="pb-popular" className="mb-4 flex items-center gap-2 text-xl font-bold tracking-tight text-zinc-100">
                  <Sparkles className="h-5 w-5 text-violet-600 dark:text-violet-300" /> Popular
                </h2>
                {/* A phone swipes through them; wider screens lay them out. */}
                <ul className="-mx-4 flex snap-x snap-mandatory scroll-px-4 gap-3 overflow-x-auto px-4 pb-1 [scrollbar-width:none] sm:mx-0 sm:grid sm:grid-cols-2 sm:gap-4 sm:overflow-visible sm:px-0 sm:pb-0 lg:grid-cols-3 xl:grid-cols-5 [&::-webkit-scrollbar]:hidden">
                  {front.popular.slice(0, 10).map((p) => (
                    <li key={p.id} className="w-60 shrink-0 snap-start sm:w-auto"><Card p={p} onOpen={() => setOpenId(p.id)} onMine={() => toggleMine(p)} /></li>
                  ))}
                </ul>
              </section>
              {front.shelf.length > 0 && (
                <Section label="My playbooks" icon={Star} count={front.shelf.length} rows={front.shelf}
                  more={front.shelf.length > SECTION_PREVIEW} onSeeAll={() => choose('mine')}
                  onOpen={setOpenId} onMine={toggleMine} />
              )}
              {cats.map((c) => {
                const rows = front.top[c.id] || []
                if (!rows.length) return null
                return (
                  <Section key={c.id} label={c.label} icon={lookOf(c.id).icon} count={c.count} rows={rows}
                    more={c.count > rows.length} onSeeAll={() => choose(c.id)} onOpen={setOpenId} onMine={toggleMine} />
                )
              })}
            </>
          )}

          {front && !browsing && (
            listErr ? <Empty icon={RotateCw} title="That didn't load">{listErr}</Empty>
              : items === null ? (
                <div className="grid gap-3 md:grid-cols-2">
                  {Array.from({ length: 6 }, (_, i) => <div key={i} className="fd-shimmer h-16 rounded-2xl" />)}
                </div>
              ) : items.length === 0 ? (
                <Empty icon={SearchX} title={q ? `No playbook fits “${q}”${current ? ` in ${current.label}` : ''}` : 'Nothing here yet'}>
                  {category === 'mine' && !q ? 'Star a playbook and it waits here — and first in the chips Captain for Chrome shows.'
                    : 'Ask one of your agents anyway — it works the site step by step. Or record it once in Captain for Chrome.'}
                </Empty>
              ) : category === 'regional' && !q ? (
                Object.entries(groupBy(items, (p) => p.region)).map(([cc, rows]) => (
                  <Section key={cc} label={`${flagOf(cc)} ${cc.toUpperCase()}`} count={rows.length} rows={rows} all
                    onOpen={setOpenId} onMine={toggleMine} />
                ))
              ) : (
                <Section label={q ? `Best for “${q}”` : current?.label || ''} count={items.length} rows={items} all
                  icon={category && category !== 'mine' ? lookOf(category).icon : category === 'mine' ? Star : undefined}
                  onOpen={setOpenId} onMine={toggleMine} />
              )
          )}
        </main>
      </div>

      {openId && (
        <PlaybookSheet
          id={openId}
          agents={agents}
          preferAgentId={preferAgentId}
          chrome={chrome}
          onClose={() => setOpenId(null)}
          onMine={(p) => toggleMine(p)}
          onConnections={onConnections}
          onRun={(agentId, content) => { setOpenId(null); onRun(agentId, content) }}
        />
      )}
    </div>
  )
}

function groupBy<T>(list: T[], key: (x: T) => string): Record<string, T[]> {
  const out: Record<string, T[]> = {}
  for (const x of list) (out[key(x)] = out[key(x)] || []).push(x)
  return out
}

// ── Pieces ───────────────────────────────────────────────────────────

function Pill({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button role="tab" aria-selected={active} onClick={onClick}
      className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-4 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${
        active ? 'border-zinc-100 bg-zinc-100 font-medium text-zinc-950'
          : 'border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:bg-zinc-900 hover:text-zinc-200'}`}>
      {children}
    </button>
  )
}

function Tile({ p, size }: { p: { category: string }; size: 'lg' | 'md' | 'xl' }) {
  const look = lookOf(p.category)
  const Icon = look.icon
  const box = size === 'xl' ? 'h-16 w-16 rounded-2xl' : size === 'lg' ? 'h-14 w-14 rounded-2xl' : 'h-11 w-11 rounded-xl'
  return (
    <span className={`flex shrink-0 items-center justify-center ring-1 ${look.tile} ${box}`}>
      <Icon className={size === 'md' ? 'h-5 w-5' : 'h-7 w-7'} />
    </span>
  )
}

function Meta({ p, stacked = false }: { p: PlaybookSummary; stacked?: boolean }) {
  if (stacked) {
    // A narrow card: the site on a line of its own, what it does under it.
    return (
      <div className="min-w-0 text-xs text-zinc-500">
        <p className="truncate">{p.site ? siteName(p.site) : 'a site you name'}{p.region.length === 2 ? ` · ${flagOf(p.region)} ${p.region.toUpperCase()}` : ''}</p>
        <p className="truncate">
          {p.kind === 'act' ? <span className="text-amber-700 dark:text-amber-300">Fills it in</span> : 'Looks it up'}
          {p.where === 'chrome' ? ' · your Chrome' : ''}
        </p>
      </div>
    )
  }
  return (
    <p className="flex min-w-0 items-center gap-1.5 truncate text-xs text-zinc-500">
      <span className="truncate">{p.site ? siteName(p.site) : 'a site you name'}</span>
      <span aria-hidden>·</span>
      {p.kind === 'act'
        ? <span className="shrink-0 text-amber-700 dark:text-amber-300">Fills it in</span>
        : <span className="shrink-0">Looks it up</span>}
      {p.where === 'chrome' && <><span aria-hidden>·</span><span className="shrink-0">your Chrome</span></>}
      {p.region.length === 2 && <><span aria-hidden>·</span><span className="shrink-0">{flagOf(p.region)} {p.region.toUpperCase()}</span></>}
    </p>
  )
}

function StarButton({ on, onClick, big = false }: { on: boolean; onClick: () => void; big?: boolean }) {
  return (
    <button onClick={(e) => { e.stopPropagation(); onClick() }} aria-pressed={on}
      aria-label={on ? 'Take it off My playbooks' : 'Add to My playbooks'} title={on ? 'In My playbooks — take it off' : 'Add to My playbooks'}
      className={`inline-flex shrink-0 items-center justify-center gap-1.5 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${
        big ? 'px-3.5 py-1.5 text-sm font-medium' : 'h-9 w-9'} ${
        on ? 'bg-violet-500/15 text-violet-700 hover:bg-violet-500/25 dark:text-violet-300'
          : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`}>
      <Star className={`h-4 w-4 ${on ? 'fill-current' : ''}`} />
      {big && (on ? 'In My playbooks' : 'Add to mine')}
    </button>
  )
}

function Card({ p, onOpen, onMine }: { p: PlaybookSummary; onOpen: () => void; onMine: () => void }) {
  const look = lookOf(p.category)
  return (
    <article onClick={onOpen} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') onOpen() }}
      className={`group flex h-full cursor-pointer flex-col rounded-3xl border border-zinc-800 bg-gradient-to-b ${look.wash} to-transparent p-5 transition-colors hover:border-zinc-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40`}>
      <div className="flex items-start justify-between">
        <Tile p={p} size="lg" />
        <StarButton on={p.mine} onClick={onMine} />
      </div>
      <h3 className="mt-4 text-base font-semibold leading-snug text-zinc-100">{p.title}</h3>
      <Meta p={p} stacked />
      {p.hint && <p className="mt-2 line-clamp-2 text-sm italic leading-relaxed text-zinc-400">“{p.hint}”</p>}
      <div className="mt-auto pt-4">
        <span className="inline-flex w-full items-center justify-center gap-1 rounded-full bg-zinc-800 px-4 py-1.5 text-sm font-medium text-zinc-100 transition-colors group-hover:bg-zinc-700">
          Use <ChevronRight className="h-3.5 w-3.5" />
        </span>
      </div>
    </article>
  )
}

function Row({ p, onOpen, onMine }: { p: PlaybookSummary; onOpen: () => void; onMine: () => void }) {
  return (
    <li onClick={onOpen} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') onOpen() }}
      className="flex cursor-pointer items-center gap-4 rounded-2xl px-3 py-3 transition-colors hover:bg-zinc-900/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40">
      <Tile p={p} size="md" />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-semibold text-zinc-100">{p.title}</p>
        <Meta p={p} />
        {p.hint && <p className="mt-0.5 truncate text-sm text-zinc-400" title={p.hint}>“{p.hint}”</p>}
      </div>
      <StarButton on={p.mine} onClick={onMine} />
    </li>
  )
}

function Section({ label, icon: Icon, count, rows, more = false, all = false, onSeeAll, onOpen, onMine }: {
  label: string
  icon?: ComponentType<{ className?: string }>
  count: number
  rows: PlaybookSummary[]
  /** Browsing: more than these rows — "See all". */
  more?: boolean
  /** A whole list (a category, a search): every row. */
  all?: boolean
  onSeeAll?: () => void
  onOpen: (id: string) => void
  onMine: (p: PlaybookSummary) => void
}) {
  const shown = all ? rows : rows.slice(0, SECTION_PREVIEW)
  return (
    <section aria-label={label}>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h2 className="flex items-center gap-2 text-xl font-bold tracking-tight text-zinc-100">
          {Icon && <Icon className="h-5 w-5 self-center text-zinc-500" />}
          {label} <span className="ml-1 text-sm font-medium tabular-nums text-zinc-500">{count}</span>
        </h2>
        {more && onSeeAll && (
          <button onClick={onSeeAll} className="inline-flex shrink-0 items-center gap-0.5 text-sm text-zinc-500 transition-colors hover:text-zinc-200">
            See all <ChevronRight className="h-4 w-4" />
          </button>
        )}
      </div>
      <ul className="-mx-3 grid gap-x-6 gap-y-1 md:grid-cols-2">
        {shown.map((p) => <Row key={p.id} p={p} onOpen={() => onOpen(p.id)} onMine={() => onMine(p)} />)}
      </ul>
    </section>
  )
}

function Empty({ icon: Icon, title, children }: { icon: ComponentType<{ className?: string }>; title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col items-center rounded-3xl border border-dashed border-zinc-800 px-6 py-16 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-900 text-zinc-500">
        <Icon className="h-6 w-6" />
      </span>
      <h2 className="mt-4 text-base font-semibold text-zinc-200">{title}</h2>
      <p className="mt-1 max-w-sm text-sm leading-relaxed text-zinc-500">{children}</p>
    </div>
  )
}

// ── One playbook: what it does, its form, Run ────────────────────────

const fieldCls = 'w-full rounded-xl border border-zinc-800 bg-zinc-900/60 px-3.5 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 transition-colors focus:border-violet-500/50 focus:outline-none focus:ring-2 focus:ring-violet-500/15'

function PlaybookSheet({ id, agents, preferAgentId, chrome, onClose, onMine, onConnections, onRun }: {
  id: string
  agents: PlaybookAgent[]
  preferAgentId?: string | null
  chrome: ChromeState
  onClose: () => void
  onMine: (p: PlaybookSummary) => void
  onConnections?: () => void
  onRun: (agentId: string, content: string) => void
}) {
  const [book, setBook] = useState<Playbook | null>(null)
  const [err, setErr] = useState('')
  const [values, setValues] = useState<Record<string, string | boolean>>({})
  const [note, setNote] = useState('')
  const [moreOpen, setMoreOpen] = useState(false)
  const [howOpen, setHowOpen] = useState(false)
  const [sending, setSending] = useState(false)
  const usable = agents.filter((a) => a.reachable || a.asleep)
  const [agentId, setAgentId] = useState(() =>
    (preferAgentId && usable.some((a) => a.id === preferAgentId) ? preferAgentId : usable[0]?.id) || '')

  useEffect(() => {
    playbooksApi.one(id).then(setBook).catch((e) => setErr(e instanceof Error ? e.message : String(e)))
  }, [id])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  const main = book?.inputs.filter((i) => i.required && i.default === undefined) || []
  const more = book?.inputs.filter((i) => !main.includes(i)) || []
  const missing = main.filter((i) => i.ask === 'before' && !String(values[i.name] ?? '').trim())
  const here = (book?.inputs || []).filter((i) => i.kind === 'place' && HERE.test(String(values[i.name] ?? 'x')))
  const outOfRange = (book?.inputs || []).filter((i) => rangeProblem(i, values[i.name]))
  const agent = usable.find((a) => a.id === agentId)
  const blocked = missing.length > 0 || here.length > 0 || outOfRange.length > 0
  const needsChrome = book?.where === 'chrome' || book?.where === 'prefer_chrome'

  const run = async () => {
    if (!book || !agent || blocked || sending) return
    setSending(true)
    setErr('')
    const inputs: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(values)) if (v !== '' && v !== undefined) inputs[k] = v
    try {
      const { content } = await playbooksApi.compose(book.id, inputs, note)
      onRun(agent.id, content)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      setSending(false)
    }
  }

  const runLabel = sending ? 'Starting…' : !agent ? 'No agent is up — start one first'
    : missing.length ? `Fill in ${missing.map((i) => i.label).slice(0, 2).join(' and ')}`
      : here.length ? `Give ${here[0].label} as a town or address`
        : outOfRange.length ? `${outOfRange[0].label} is ${rangeProblem(outOfRange[0], values[outOfRange[0].name])}`
          : `Run in ${agent.name}'s chat`

  return createPortal(
    <div className="fd-fade fixed inset-0 z-[60] flex items-stretch justify-center bg-black/50 backdrop-blur-[2px] wide:items-center wide:p-4" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-label={book?.title || 'Playbook'}
        className="fd-sheet flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 shadow-2xl wide:h-auto wide:max-h-[min(90vh,52rem)] wide:max-w-2xl wide:rounded-3xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}>
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-5 pb-6 pt-4 wide:px-7">
          <div className="flex items-center justify-end gap-1">
            {book && <StarButton on={book.mine} big onClick={() => { onMine(book); setBook({ ...book, mine: !book.mine }) }} />}
            <button onClick={onClose} aria-label="Close" className="-mr-2 rounded-full p-2 text-zinc-500 transition-colors hover:bg-zinc-900 hover:text-zinc-200">
              <X className="h-5 w-5" />
            </button>
          </div>

          {!book ? (
            err ? <Empty icon={RotateCw} title="That playbook didn't load">{err}</Empty>
              : <div className="space-y-4 pt-2"><div className="fd-shimmer h-16 w-16 rounded-2xl" /><div className="fd-shimmer h-6 w-2/3 rounded-lg" /><div className="fd-shimmer h-40 rounded-2xl" /></div>
          ) : (
            <>
              <div className="flex items-start gap-4 pt-1">
                <Tile p={book} size="xl" />
                <div className="min-w-0 flex-1">
                  <h2 className="text-2xl font-bold leading-tight tracking-tight text-zinc-100">{book.title}</h2>
                  <p className="mt-1 truncate text-sm text-zinc-500">
                    {book.sites.length ? [...new Set(book.sites.map(siteName))].slice(0, 3).join(', ') : 'Any site you name'}
                  </p>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                {book.kind === 'act'
                  ? <Badge tone="amber" icon={PenLine}>Fills it in for you</Badge>
                  : <Badge tone="emerald" icon={Eye}>Looks it up</Badge>}
                <Badge icon={needsChrome ? Lock : Globe}>{WHERE[book.where] || book.where}</Badge>
                {book.region.length === 2 && <Badge>{flagOf(book.region)} {book.region.toUpperCase()}</Badge>}
                {book.runs > 0 && <Badge>Ran {book.runs}×</Badge>}
              </div>

              {book.says.length > 0 && (
                <div className="mt-4 space-y-1">
                  {book.says.slice(0, 2).map((s) => (
                    <p key={s} className="border-l-2 border-zinc-700 pl-3 text-sm italic text-zinc-400">“{s}”</p>
                  ))}
                </div>
              )}

              {book.kind === 'act' && (
                <div className="mt-4 flex gap-3 rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-zinc-200">
                  <OctagonX className="mt-0.5 h-4 w-4 shrink-0 text-amber-700 dark:text-amber-300" />
                  <p>It fills everything in and stops before <b>“{book.stop}”</b> — you look it over and press it yourself.</p>
                </div>
              )}

              {needsChrome && chrome !== 'lent' && chrome !== 'unknown' && (
                <div className="mt-4 flex gap-3 rounded-2xl border border-zinc-800 bg-zinc-900/60 px-4 py-3 text-sm text-zinc-300">
                  <Globe className="mt-0.5 h-4 w-4 shrink-0 text-zinc-500" />
                  <div>
                    {chrome === 'none'
                      ? <>{book.where === 'chrome' ? 'This one runs in your own Chrome, signed in as you' : 'This site often turns robots away — it goes best in your own Chrome'}. Set up <b>Captain for Chrome</b> and turn on “Agents can use this browser”.</>
                      : <>Captain for Chrome is set up — turn on <b>“Agents can use this browser”</b> in its settings so your agent can work there.</>}
                    {chrome === 'none' && onConnections && (
                      <button onClick={onConnections} className="mt-2 block font-medium text-violet-700 hover:underline dark:text-violet-300">Set up Captain for Chrome</button>
                    )}
                  </div>
                </div>
              )}

              <div className="mt-6 space-y-4">
                {main.map((i) => <Field key={i.name} i={i} value={values[i.name]} set={(v) => setValues((c) => ({ ...c, [i.name]: v }))} />)}
                {more.length > 0 && (main.length > 0 ? (
                  <div>
                    <button onClick={() => setMoreOpen((v) => !v)} aria-expanded={moreOpen}
                      className="inline-flex items-center gap-1.5 text-sm text-zinc-400 transition-colors hover:text-zinc-200">
                      <ChevronDown className={`h-4 w-4 transition-transform ${moreOpen ? 'rotate-180' : ''}`} />
                      More options <span className="text-zinc-500">— {more.length}, or let {agent?.name || 'the agent'} decide</span>
                    </button>
                    {moreOpen && <div className="mt-4 space-y-4">{more.map((i) => <Field key={i.name} i={i} value={values[i.name]} set={(v) => setValues((c) => ({ ...c, [i.name]: v }))} />)}</div>}
                  </div>
                ) : more.map((i) => <Field key={i.name} i={i} value={values[i.name]} set={(v) => setValues((c) => ({ ...c, [i.name]: v }))} />))}

                <div>
                  <label htmlFor="pb-agent" className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-zinc-500">Who does it</label>
                  <select id="pb-agent" value={agentId} onChange={(e) => setAgentId(e.target.value)} className={fieldCls} disabled={!usable.length}>
                    {usable.length === 0 && <option value="">No agent is up</option>}
                    {usable.map((a) => <option key={a.id} value={a.id}>{a.name}{a.asleep ? ' (asleep — it wakes up)' : ''}</option>)}
                  </select>
                </div>
                <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2}
                  placeholder="Anything else it should know? (optional)" className={`${fieldCls} resize-y`} />
              </div>

              <div className="mt-5 rounded-2xl border border-zinc-800">
                <button onClick={() => setHowOpen((v) => !v)} aria-expanded={howOpen}
                  className="flex w-full items-center gap-2 px-4 py-3 text-left text-sm">
                  <ChevronDown className={`h-4 w-4 text-zinc-500 transition-transform ${howOpen ? 'rotate-180' : ''}`} />
                  <span className="font-semibold text-zinc-200">How it goes</span>
                  <span className="text-zinc-500">— {book.steps.length ? `${book.steps.length} step${book.steps.length === 1 ? '' : 's'}, then ` : ''}it reads {book.read.length} thing{book.read.length === 1 ? '' : 's'}</span>
                </button>
                {howOpen && (
                  <div className="border-t border-zinc-800 px-4 py-3">
                    {book.steps.length > 0 && (
                      <ol className="space-y-2">
                        {book.steps.map((s, n) => (
                          <li key={n} className="flex gap-2.5 text-sm text-zinc-400">
                            <span className="mt-0.5 flex h-5 min-w-5 items-center justify-center rounded-full bg-zinc-800 px-1 text-[11px] font-semibold tabular-nums text-zinc-400">{n + 1}</span>
                            <span>{s.replace(/^\d+\.\s*/, '')}</span>
                          </li>
                        ))}
                      </ol>
                    )}
                    {book.read.length > 0 && (
                      <div className={`space-y-1 ${book.steps.length ? 'mt-3 border-t border-zinc-800 pt-3' : ''}`}>
                        {book.read.map((r) => (
                          <p key={r.name} className="text-sm text-zinc-400"><b className="font-semibold text-zinc-200">{r.name.replace(/_/g, ' ')}</b> — {r.about}</p>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
              {err && <p className="mt-4 rounded-xl bg-red-500/10 px-4 py-2.5 text-sm text-red-700 dark:text-red-300">{err}</p>}
            </>
          )}
        </div>
        {book && (
          <div className="border-t border-zinc-800 bg-zinc-950 px-5 py-4 wide:px-7">
            <button onClick={run} disabled={!agent || blocked || sending}
              className="inline-flex w-full items-center justify-center gap-2 rounded-full bg-violet-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-violet-500 disabled:cursor-default disabled:opacity-50 disabled:hover:bg-violet-600">
              <Play className="h-4 w-4" /> {runLabel}
            </button>
            <p className="mt-2 text-center text-xs text-zinc-500">
              {agent ? `It goes to ${agent.name} as your message — you follow it in the chat.` : 'Start an agent, then come back.'}
            </p>
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}

function Badge({ tone, icon: Icon, children }: { tone?: 'amber' | 'emerald'; icon?: ComponentType<{ className?: string }>; children: ReactNode }) {
  const cls = tone === 'amber' ? 'border-amber-500/30 text-amber-700 dark:text-amber-300'
    : tone === 'emerald' ? 'border-emerald-500/30 text-emerald-700 dark:text-emerald-300'
      : 'border-zinc-800 text-zinc-400'
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${cls}`}>
      {Icon && <Icon className="h-3.5 w-3.5" />}{children}
    </span>
  )
}

// What's wrong with a number the person typed ("at least 0"), or '': the
// deck and the agent's tool check it again.
function rangeProblem(i: PlaybookInput, raw: unknown): string {
  const text = String(raw ?? '').trim()
  if (i.kind !== 'number' || !text) return ''
  const n = Number(text.replace(',', '.'))
  if (!Number.isFinite(n)) return 'a number'
  const out = (i.min !== undefined && n < i.min) || (i.max !== undefined && n > i.max)
  if (!out) return ''
  return i.min !== undefined && i.max !== undefined ? `${i.min} to ${i.max}`
    : i.min !== undefined ? `at least ${i.min}` : `at most ${i.max}`
}

function Field({ i, value, set }: { i: PlaybookInput; value: string | boolean | undefined; set: (v: string | boolean) => void }) {
  const optional = !i.required || i.default !== undefined
  const dflt = i.default !== undefined && i.default !== '' && i.default !== false ? String(i.default) : ''
  const label = (
    <span className="mb-1.5 flex items-baseline gap-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
      {i.label}{optional && <span className="font-normal normal-case tracking-normal text-zinc-500/80">optional</span>}
    </span>
  )
  const help = i.hint ? <p className="mt-1.5 text-xs text-zinc-500">{i.hint}</p> : null
  if (i.kind === 'choice') {
    const cur = String(value ?? i.default ?? '')
    const options = i.options || []
    return (
      <div>
        {label}
        {options.length <= 5 ? (
          <div role="group" aria-label={i.label} className="flex flex-wrap gap-1.5">
            {options.map((o) => (
              <button key={o} type="button" onClick={() => set(o)} aria-pressed={o === cur}
                className={`rounded-xl border px-3.5 py-1.5 text-sm transition-colors ${o === cur
                  ? 'border-violet-500/60 bg-violet-500/10 font-medium text-zinc-100'
                  : 'border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'}`}>
                {o.replace(/_/g, ' ')}
              </button>
            ))}
          </div>
        ) : (
          <select value={cur} onChange={(e) => set(e.target.value)} className={fieldCls}>
            {options.map((o) => <option key={o} value={o}>{o.replace(/_/g, ' ')}</option>)}
          </select>
        )}
        {help}
      </div>
    )
  }
  if (i.kind === 'flag') {
    const on = Boolean(value ?? i.default ?? false)
    return (
      <div>
        <label className="flex cursor-pointer items-center gap-3 text-sm text-zinc-200">
          <button type="button" role="switch" aria-checked={on} onClick={() => set(!on)}
            className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${on ? 'bg-violet-600' : 'bg-zinc-700'}`}>
            <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${on ? 'translate-x-4' : 'translate-x-0.5'}`} />
          </button>
          {i.label}
        </label>
        {help}
      </div>
    )
  }
  if (i.kind === 'file') {
    return <div>{label}<p className="text-sm text-zinc-500">{dflt || 'Your agent picks the file.'}</p></div>
  }
  const type = ({ date: 'date', time: 'time', number: 'number', url: 'url' } as Record<string, string>)[i.kind] || 'text'
  const v = typeof value === 'string' ? value : type === 'date' && /^\d{4}-\d{2}-\d{2}$/.test(dflt) ? dflt : type === 'time' && /^\d{2}:\d{2}$/.test(dflt) ? dflt : ''
  const Icon = i.kind === 'place' ? MapPin : i.kind === 'url' ? Globe : null
  const here = i.kind === 'place' && HERE.test(String(value ?? 'x'))
  const bad = rangeProblem(i, v)
  return (
    <label className="block">
      {label}
      <span className="relative block">
        {Icon && <Icon className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />}
        <input type={type} value={v} onChange={(e) => set(e.target.value)} autoComplete="off"
          placeholder={i.kind === 'place' ? 'Town or address' : i.kind === 'url' ? 'https://…' : dflt ? `Default: ${dflt}` : ''}
          {...(type === 'number' ? {
            inputMode: 'decimal' as const, step: 'any', min: i.min, max: i.max, 'aria-invalid': bad ? true : undefined,
            // A focused number box counts up or down under the wheel: scrolling
            // the sheet over it once turned 7 nights into -51. Let go instead.
            onWheel: (e: React.WheelEvent<HTMLInputElement>) => e.currentTarget.blur(),
          } : {})}
          className={`${fieldCls} ${Icon ? 'pl-10' : ''}`} />
      </span>
      {bad ? <p className="mt-1.5 text-xs text-red-600 dark:text-red-400">{i.label} is {bad}.</p>
        : here ? <p className="mt-1.5 text-xs text-amber-700 dark:text-amber-300">Type the town or address — a link can't carry where you are.</p>
        : help || (type === 'date' && dflt && !v ? <p className="mt-1.5 text-xs text-zinc-500">Leave it for {dflt}.</p> : null)}
    </label>
  )
}
