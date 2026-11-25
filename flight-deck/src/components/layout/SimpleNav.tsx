import { Fragment, useState, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import { NAV_GROUPS, type NavEntry, type NavGroupId, type NavTarget } from './navDestinations'

// ── Simple layout: the outcome nav ───────────────────────────────────
//
// The destinations of navDestinations.ts as rows under their group's heading
// (or icons on the folded rail, a thin line between groups). A heading folds
// its group, remembered in this browser: the one in front stays, and what
// wants the person is counted on the heading. A group can carry a row that
// isn't a destination (`extra`: the Browser group's Captain for Chrome).
// The styling is the shell's (.fd-nav-item in index.css).

const FOLDED_KEY = 'fd:nav-folded'

function readFolded(): NavGroupId[] {
  try {
    const raw = JSON.parse(localStorage.getItem(FOLDED_KEY) || '[]')
    return Array.isArray(raw) ? raw.filter((x): x is NavGroupId => NAV_GROUPS.some((g) => g.id === x)) : []
  } catch { return [] }
}

function useFolded(): [NavGroupId[], (id: NavGroupId) => void] {
  const [folded, setFolded] = useState<NavGroupId[]>(readFolded)
  const toggle = (id: NavGroupId) => setFolded((was) => {
    const next = was.includes(id) ? was.filter((x) => x !== id) : [...was, id]
    try { localStorage.setItem(FOLDED_KEY, JSON.stringify(next)) } catch { /* storage blocked: this visit only */ }
    return next
  })
  return [folded, toggle]
}

const countText = (n?: number) => (n && n > 0 ? (n > 99 ? '99+' : String(n)) : '')

function Row({ e, current }: { e: NavEntry; current: NavTarget | null }) {
  const Icon = e.icon
  const count = countText(e.count)
  const label = count && e.hot ? `${e.label} — ${e.hint}` : e.label
  return (
    <button onClick={e.onSelect} title={e.hint} aria-label={label}
      aria-current={current === e.id ? 'page' : undefined} className="fd-nav-item">
      <Icon className="fd-nav-icon" />
      <span className="min-w-0 flex-1 truncate">{e.label}</span>
      {e.dot && <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-[var(--fd-accent)]" title="Running now" />}
      {count && <span className={`fd-count ${e.hot ? 'fd-count-hot' : ''}`}>{count}</span>}
    </button>
  )
}

function RailButton({ e, current }: { e: NavEntry; current: NavTarget | null }) {
  const Icon = e.icon
  const count = countText(e.count)
  const label = count && e.hot ? `${e.label} — ${e.hint}` : e.label
  return (
    <button onClick={e.onSelect} title={`${e.label} — ${e.hint}`} aria-label={label}
      aria-current={current === e.id ? 'page' : undefined} className="fd-nav-item fd-nav-rail">
      <Icon className="fd-nav-icon" />
      {e.dot ? <span className="absolute right-2 top-2 h-2 w-2 animate-pulse rounded-full bg-[var(--fd-accent)]" />
        : count && e.hot ? <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-[var(--fd-accent)]" /> : null}
    </button>
  )
}

/** The nav: groups of rows, or (`rail`) groups of icons. Buttons in a <nav>,
 *  never list items — the agent list below owns the column's <li>s. */
export function SimpleNav({ entries, current, rail = false, extra = {} }: {
  entries: NavEntry[]
  current: NavTarget | null
  rail?: boolean
  /** A row of a group's own, after its destinations (the rail's version on the rail). */
  extra?: Partial<Record<NavGroupId, ReactNode>>
}) {
  const [folded, toggle] = useFolded()
  const groups = NAV_GROUPS
    .map((g) => ({ ...g, entries: entries.filter((e) => e.group === g.id), extra: extra[g.id] }))
    .filter((g) => g.entries.length > 0 || g.extra)

  if (rail) {
    return (
      <nav aria-label="Main" className="flex flex-col items-center gap-1">
        {groups.map((g, i) => (
          <Fragment key={g.id}>
            {i > 0 && <span aria-hidden className="my-1 h-px w-6 bg-[var(--fd-line)]" />}
            {g.entries.map((e) => <RailButton key={e.id} e={e} current={current} />)}
            {g.extra}
          </Fragment>
        ))}
      </nav>
    )
  }

  return (
    <nav aria-label="Main" className="flex flex-col">
      {groups.map((g, i) => {
        const isFolded = g.fold && folded.includes(g.id)
        // Folded: the one in front stays, and what wants the person is counted.
        const shown = isFolded ? g.entries.filter((e) => e.id === current) : g.entries
        const waiting = isFolded ? g.entries.filter((e) => e.hot && e.id !== current).reduce((n, e) => n + (e.count || 0), 0) : 0
        const running = isFolded && g.entries.some((e) => e.dot && e.id !== current)
        const headingId = `fd-nav-${g.id}`
        return (
          <div key={g.id} role="group" aria-labelledby={headingId} className={i > 0 ? 'mt-2.5' : ''}>
            <div className="group/heading flex h-6 items-center gap-1.5 px-3">
              {g.fold ? (
                <button id={headingId} onClick={() => toggle(g.id)} aria-expanded={!isFolded}
                  title={isFolded ? `Show ${g.label}` : `Fold ${g.label}`}
                  className="fd-section-label flex min-w-0 items-center gap-1 rounded transition-colors hover:text-[var(--fd-ink)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40">
                  <span className="truncate">{g.label}</span>
                  <ChevronDown className={`h-3 w-3 shrink-0 transition-[transform,opacity] motion-reduce:transition-none ${
                    isFolded ? '-rotate-90 opacity-100' : 'opacity-0 group-hover/heading:opacity-100 group-focus-within/heading:opacity-100 pointer-coarse:opacity-100'}`} />
                </button>
              ) : (
                <span id={headingId} className="fd-section-label truncate">{g.label}</span>
              )}
              <span className="ml-auto flex items-center gap-1.5">
                {running && <span className="h-2 w-2 animate-pulse rounded-full bg-[var(--fd-accent)]" title="Running now" />}
                {waiting > 0 && <span className="fd-count fd-count-hot" title="Waiting for you">{countText(waiting)}</span>}
              </span>
            </div>
            {shown.map((e) => <Row key={e.id} e={e} current={current} />)}
            {g.extra}
          </div>
        )
      })}
    </nav>
  )
}
