import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { ArrowLeft, Bot, Check, ChevronRight, Moon, Plus, RotateCw, Search, SearchX, Sun, X } from 'lucide-react'
import { AgentSetupDialog } from '../layout/AgentSetupDialog'
import type { SetupRequest } from '../layout/ArchetypeSpawnDialog'
import { archetypeLook, rememberTeamLooks } from '../layout/archetypeLook'
import { listArchetypes } from '../../services/archetypes'
import type { Archetype } from '../../services/tierConfig'
import { useProcessStore } from '../../stores/processStore'
import { useThemeStore } from '../../stores/themeStore'
import { AddAgentDialog } from './AddAgentDialog'
import { creatorOf, featured, matchesSearch, storeSections, type StoreSection } from './storeModel'

// ── Agent store ──────────────────────────────────────────────────────
//
// Ready-made agents to browse and add to a team, in a calm marketplace look:
// a big heading, category pills and a search, a Featured row of large cards,
// then each category as two columns of rows with an Add button. The agents
// are the deck's archetype catalog (GET /fd/archetypes — the built-in ones,
// what the admin published, the person's own and those shared with them);
// a category is a family on its shelf. Add creates the agent for real
// (AddAgentDialog → POST /fd/spawn-process, as the New agent picker does).
//
// Both themes: the zinc utilities are the app's themed surface colours —
// html.light remaps the zinc scale (index.css), so `bg-zinc-950` is the page
// in either theme — and only the accents outside it (violet, emerald, sky)
// carry a dark: variant, as everywhere else in the app. A light-first zinc
// class (`text-zinc-900 dark:text-zinc-100`) would turn pale in light.

/** Rows a category shows under "All" before its "See all". */
const SECTION_PREVIEW = 6

export function AgentStorePanel() {
  const [catalog, setCatalog] = useState<Archetype[] | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState<string | null>(null)
  const [adding, setAdding] = useState<Archetype | null>(null)
  const [setupFor, setSetupFor] = useState<SetupRequest | null>(null)
  // Archetypes added from this visit: their button says so.
  const [added, setAdded] = useState<Set<string>>(() => new Set())
  const searchRef = useRef<HTMLInputElement>(null)
  const topRef = useRef<HTMLDivElement>(null)
  const processes = useProcessStore((s) => s.processes)
  const theme = useThemeStore((s) => s.theme)
  const toggleTheme = useThemeStore((s) => s.toggle)

  const fetchCatalog = useCallback(() => {
    listArchetypes(undefined, true)
      .then((reg) => {
        const registry = { ...reg, archetypes: reg.archetypes || [] }
        rememberTeamLooks(registry)
        setCatalog(registry.archetypes)
      })
      .catch((e) => setLoadErr(e instanceof Error ? e.message : String(e)))
  }, [])
  useEffect(() => { fetchCatalog() }, [fetchCatalog])
  const retry = () => { setLoadErr(''); setCatalog(null); fetchCatalog() }
  // The agents this person already has: a new one's default name avoids theirs.
  useEffect(() => { void useProcessStore.getState().fetchProcesses() }, [])
  const takenSlugs = useMemo(() => new Set(processes.map((p) => p.slug)), [processes])

  const sections = useMemo(() => storeSections(catalog || []), [catalog])
  const q = query.trim()
  const shown = useMemo(() => sections
    .filter((s) => !category || s.key === category)
    .map((s) => ({ ...s, agents: s.agents.filter((a) => matchesSearch(a, q)) }))
    .filter((s) => s.agents.length > 0), [sections, category, q])
  const browsing = !category && !q
  const picks = useMemo(() => (browsing ? featured(catalog || []) : []), [browsing, catalog])
  const current = sections.find((s) => s.key === category)
  // A search that finds nothing in this category may find something in another.
  const elsewhere = useMemo(() => (q && category && !shown.length
    ? sections.reduce((n, s) => n + s.agents.filter((a) => matchesSearch(a, q)).length, 0) : 0), [q, category, shown, sections])

  const choose = (key: string | null) => {
    setCategory(key)
    topRef.current?.scrollIntoView({ block: 'start' })
  }

  const onAdded = (archetypeId: string, setup?: SetupRequest) => {
    setAdded((prev) => new Set(prev).add(archetypeId))
    if (setup) {
      setAdding(null)
      setSetupFor(setup)
    }
  }

  return (
    <div className="h-full min-h-screen overflow-y-auto bg-zinc-950 text-zinc-100">
      <div className="mx-auto w-full max-w-5xl px-4 pb-16 pt-4 wide:px-8">
        {/* Top bar: the way back, and the theme — this page has no sidebar of its own. */}
        <div className="flex items-center justify-between">
          <a href="/" className="-ml-2 inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm text-zinc-500 transition-colors hover:bg-zinc-900 hover:text-zinc-200">
            <ArrowLeft className="h-4 w-4" /> Flight Deck
          </a>
          <button onClick={toggleTheme} aria-label={theme === 'dark' ? 'Light theme' : 'Dark theme'} title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
            className="rounded-full p-2 text-zinc-500 transition-colors hover:bg-zinc-900 hover:text-zinc-200">
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>

        <header ref={topRef} className="scroll-mt-4 pt-8 wide:pt-12">
          <h1 className="text-4xl font-bold tracking-tight text-zinc-100 wide:text-5xl">Agent store</h1>
          <p className="mt-2 max-w-xl text-base text-zinc-500">
            Ready-made agents for the work your business does. Add one and it joins your team.
          </p>
        </header>

        {/* Categories and search */}
        <div className="sticky top-0 z-10 -mx-4 mt-8 flex flex-col gap-3 bg-zinc-950/90 px-4 py-3 backdrop-blur wide:-mx-8 wide:flex-row wide:items-center wide:px-8">
          <div role="tablist" aria-label="Categories"
            className="-mx-4 flex min-w-0 flex-1 gap-2 overflow-x-auto px-4 [mask-image:linear-gradient(to_right,black_calc(100%-2rem),transparent)] [scrollbar-width:none] wide:mx-0 wide:px-0 [&::-webkit-scrollbar]:hidden">
            <Pill active={!category} onClick={() => choose(null)}>All</Pill>
            {sections.map((s) => (
              <Pill key={s.key} active={category === s.key} onClick={() => choose(s.key)}>{s.label}</Pill>
            ))}
          </div>
          <div className="relative wide:w-72 wide:shrink-0">
            <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
            <input
              ref={searchRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search agents"
              aria-label="Search agents"
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
            <Empty icon={RotateCw} title="The store couldn't load its agents">
              <span className="block">{loadErr}</span>
              <button onClick={retry} className="mt-3 font-medium text-violet-700 hover:underline dark:text-violet-300">Try again</button>
            </Empty>
          )}

          {!loadErr && catalog === null && (
            <div role="status" aria-label="Loading agents" className="space-y-10">
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                {Array.from({ length: 4 }, (_, i) => <div key={i} className="fd-shimmer h-56 rounded-3xl" />)}
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                {Array.from({ length: 6 }, (_, i) => <div key={i} className="fd-shimmer h-20 rounded-2xl" />)}
              </div>
            </div>
          )}

          {catalog && catalog.length === 0 && (
            <Empty icon={Bot} title="No agents here yet">
              Your Flight Deck has no ready-made agents to offer. An admin can add some under Archetypes, and they show up here.
            </Empty>
          )}

          {catalog && catalog.length > 0 && shown.length === 0 && (
            <Empty icon={SearchX} title={`No agents match “${q}”${current ? ` in ${current.label}` : ''}`}>
              Try another word, or {elsewhere > 0
                ? <button onClick={() => choose(null)} className="font-medium text-violet-700 hover:underline dark:text-violet-300">see {elsewhere} match{elsewhere === 1 ? '' : 'es'} in all categories</button>
                : <button onClick={() => { setQuery(''); choose(null) }} className="font-medium text-violet-700 hover:underline dark:text-violet-300">show every agent</button>}.
            </Empty>
          )}

          {picks.length > 0 && (
            <section aria-labelledby="store-featured">
              <h2 id="store-featured" className="mb-4 text-xl font-bold tracking-tight text-zinc-100">Featured</h2>
              <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {picks.map((a) => (
                  <li key={a.id}><FeaturedCard a={a} added={added.has(a.id)} onAdd={() => setAdding(a)} /></li>
                ))}
              </ul>
            </section>
          )}

          {shown.map((s) => <Section key={s.key} section={s} preview={browsing} added={added} onAdd={setAdding} onSeeAll={() => choose(s.key)} />)}
        </main>
      </div>

      {adding && (
        <AddAgentDialog
          archetype={adding}
          takenSlugs={takenSlugs}
          onClose={() => setAdding(null)}
          onAdded={(_slug, _name, setup) => onAdded(adding.id, setup)}
        />
      )}
      {setupFor && <AgentSetupDialog agent={setupFor} onClose={() => setSetupFor(null)} />}
    </div>
  )
}

// ── Pieces ───────────────────────────────────────────────────────────

function Pill({ active, onClick, children }: { active: boolean; onClick: () => void; children: string }) {
  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`shrink-0 whitespace-nowrap rounded-full border px-4 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${
        active
          ? 'border-zinc-100 bg-zinc-100 font-medium text-zinc-950'
          : 'border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:bg-zinc-900 hover:text-zinc-200'
      }`}
    >
      {children}
    </button>
  )
}

function Avatar({ a, size }: { a: Archetype; size: 'lg' | 'md' }) {
  const look = archetypeLook(a.id, a.family || 'Other')
  const Icon = look.icon
  return (
    <span className={`flex shrink-0 items-center justify-center ring-1 ${look.bg} ${look.text} ${look.ring} ${
      size === 'lg' ? 'h-16 w-16 rounded-2xl' : 'h-12 w-12 rounded-xl'
    }`}>
      <Icon className={size === 'lg' ? 'h-8 w-8' : 'h-6 w-6'} />
    </span>
  )
}

function AddButton({ name, added, onAdd, wide = false }: { name: string; added: boolean; onAdd: () => void; wide?: boolean }) {
  return (
    <button
      onClick={onAdd}
      aria-label={added ? `Add another ${name}` : `Add ${name}`}
      title={added ? 'Added — add another' : `Add ${name} to your team`}
      className={`inline-flex shrink-0 items-center justify-center gap-1 rounded-full px-4 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${
        added
          ? 'bg-emerald-500/15 text-emerald-700 hover:bg-emerald-500/25 dark:text-emerald-300'
          : 'bg-zinc-800 text-zinc-100 hover:bg-zinc-700'
      } ${wide ? 'w-full' : ''}`}
    >
      {added ? <Check className="h-3.5 w-3.5" /> : <Plus className="h-3.5 w-3.5" />}
      {added ? 'Added' : 'Add'}
    </button>
  )
}

function FeaturedCard({ a, added, onAdd }: { a: Archetype; added: boolean; onAdd: () => void }) {
  const look = archetypeLook(a.id, a.family || 'Other')
  const name = a.role || a.id
  return (
    <article className={`flex h-full flex-col rounded-3xl border border-zinc-800 bg-gradient-to-b ${look.wash} to-transparent p-5 transition-colors hover:border-zinc-700`}>
      <Avatar a={a} size="lg" />
      <h3 className="mt-4 text-base font-semibold leading-snug text-zinc-100">{name}</h3>
      <p className="text-xs text-zinc-500">{creatorOf(a)}</p>
      {a.description && <p className="mt-2 line-clamp-2 text-sm leading-relaxed text-zinc-400">{a.description}</p>}
      <div className="mt-auto pt-4"><AddButton name={name} added={added} onAdd={onAdd} wide /></div>
    </article>
  )
}

function Row({ a, added, onAdd }: { a: Archetype; added: boolean; onAdd: () => void }) {
  const name = a.role || a.id
  return (
    <li className="flex items-center gap-4 rounded-2xl px-3 py-3 transition-colors hover:bg-zinc-900/60">
      <Avatar a={a} size="md" />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-semibold text-zinc-100">{name}</p>
        <p className="truncate text-xs text-zinc-500">{creatorOf(a)}</p>
        {a.description && <p className="mt-0.5 truncate text-sm text-zinc-400" title={a.description}>{a.description}</p>}
      </div>
      <AddButton name={name} added={added} onAdd={onAdd} />
    </li>
  )
}

function Section({ section: s, preview, added, onAdd, onSeeAll }: {
  section: StoreSection
  /** Browsing everything: a few rows, then "See all". */
  preview: boolean
  added: Set<string>
  onAdd: (a: Archetype) => void
  onSeeAll: () => void
}) {
  const rows = preview ? s.agents.slice(0, SECTION_PREVIEW) : s.agents
  const more = s.agents.length - rows.length
  return (
    <section aria-label={s.label}>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h2 className="text-xl font-bold tracking-tight text-zinc-100">
          {s.label} <span className="ml-1 text-sm font-medium tabular-nums text-zinc-500">{s.agents.length}</span>
        </h2>
        {more > 0 && (
          <button onClick={onSeeAll} className="inline-flex shrink-0 items-center gap-0.5 text-sm text-zinc-500 transition-colors hover:text-zinc-200">
            See all <ChevronRight className="h-4 w-4" />
          </button>
        )}
      </div>
      <ul className="-mx-3 grid gap-x-6 gap-y-1 md:grid-cols-2">
        {rows.map((a) => <Row key={a.id} a={a} added={added.has(a.id)} onAdd={() => onAdd(a)} />)}
      </ul>
    </section>
  )
}

function Empty({ icon: Icon, title, children }: { icon: typeof Bot; title: string; children: ReactNode }) {
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
