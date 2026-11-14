import type { NavEntry, NavTarget } from './navDestinations'

// ── Simple layout: the outcome nav ───────────────────────────────────
//
// The destinations of navDestinations.ts as rows (or icons on the folded rail).
// The styling is the shell's (.fd-nav-item in index.css).

/** The nav: a full row per destination, or (`rail`) an icon each. Buttons in a
 *  <nav>, never list items — the agent list below owns the column's <li>s. */
export function SimpleNav({ entries, current, rail = false }: { entries: NavEntry[]; current: NavTarget | null; rail?: boolean }) {
  return (
    <nav aria-label="Main" className={rail ? 'flex flex-col items-center gap-1' : 'flex flex-col'}>
      {entries.map((e) => {
        const Icon = e.icon
        const count = e.count && e.count > 0 ? (e.count > 99 ? '99+' : String(e.count)) : ''
        const label = count && e.hot ? `${e.label} — ${e.hint}` : e.label
        if (rail) {
          return (
            <button key={e.id} onClick={e.onSelect} title={`${e.label} — ${e.hint}`} aria-label={label}
              aria-current={current === e.id ? 'page' : undefined} className="fd-nav-item fd-nav-rail">
              <Icon className="fd-nav-icon" />
              {e.dot ? <span className="absolute right-2 top-2 h-2 w-2 animate-pulse rounded-full bg-[var(--fd-accent)]" />
                : count && e.hot ? <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-[var(--fd-accent)]" /> : null}
            </button>
          )
        }
        return (
          <button key={e.id} onClick={e.onSelect} title={e.hint} aria-label={label}
            aria-current={current === e.id ? 'page' : undefined} className="fd-nav-item">
            <Icon className="fd-nav-icon" />
            <span className="min-w-0 flex-1 truncate">{e.label}</span>
            {e.dot && <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-[var(--fd-accent)]" title="Running now" />}
            {count && <span className={`fd-count ${e.hot ? 'fd-count-hot' : ''}`}>{count}</span>}
          </button>
        )
      })}
    </nav>
  )
}
