import { archetypeCategory, shelfOf, shelves } from '../layout/archetypeLook'
import type { Archetype, ArchetypeCategory } from '../../services/tierConfig'

// ── The Agent store's view of the archetype catalog ──────────────────
//
// The store shows the same catalog the New agent picker does (GET
// /fd/archetypes), sorted the way that gallery sorts it: the deck's own shelves
// first, then the built-in ones, families alphabetically on each shelf.

/** One category of the store: a family on its shelf. The company's "Legal" isn't the built-in one. */
export interface StoreSection {
  key: string
  /** What the pill and the heading say. */
  label: string
  shelf: ArchetypeCategory
  family: string
  agents: Archetype[]
}

export function storeSections(list: Archetype[]): StoreSection[] {
  const order = shelves().map((sh) => sh.id)
  const byKey = new Map<string, StoreSection>()
  for (const a of list) {
    const shelf = archetypeCategory(a)
    const family = a.family || 'Other'
    const key = `${shelf}/${family}`
    const section = byKey.get(key)
    if (section) section.agents.push(a)
    else byKey.set(key, { key, label: family, shelf, family, agents: [a] })
  }
  const sections = [...byKey.values()].sort((x, y) =>
    order.indexOf(x.shelf) - order.indexOf(y.shelf) || x.family.localeCompare(y.family))
  // A family on two shelves says which one, so its two pills never read alike.
  const seen = new Map<string, number>()
  for (const s of sections) seen.set(s.family, (seen.get(s.family) || 0) + 1)
  for (const s of sections) {
    if ((seen.get(s.family) || 0) > 1) s.label = `${s.family} · ${shelfOf(s.shelf).label}`
  }
  return sections
}

/** The Featured row: the first few archetypes, in the catalog's own order. */
export function featured(list: Archetype[], count = 4): Archetype[] {
  return list.slice(0, count)
}

/** Tools that work in someone's mail or calendar (as the New agent picker lists them). */
const MAIL_TOOLS = new Set(['google_mail', 'google_calendar', 'microsoft_mail', 'microsoft_calendar'])

/** It does its work in someone's mail or calendar: it needs a connected account. */
export function worksInMail(a: Archetype): boolean {
  return (a.tools || []).some((t) => MAIL_TOOLS.has(t))
}

/** Who made it, as the store credits it. */
export function creatorOf(a: Archetype): string {
  switch (a.source) {
    case 'team': return 'Your team'
    case 'user': return 'You'
    case 'shared': return a.shared_owner_name || a.shared_owner_email || 'Shared with you'
    default: return 'Captain Claw'
  }
}

/** The search: its name or what it does. */
export function matchesSearch(a: Archetype, query: string): boolean {
  const q = query.trim().toLowerCase()
  return !q || (a.role || a.id).toLowerCase().includes(q) || (a.description || '').toLowerCase().includes(q)
}

/** Mirrors the backend `_slug` (as the New agent picker does): a process agent's name is its directory. */
export function agentSlug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/^-+|-+$/g, '') || 'cc-agent'
}

/** The archetype's name, or "Name 2", "Name 3"… when an agent already has it. */
export function uniqueName(base: string, taken: Set<string>): string {
  if (!taken.has(agentSlug(base))) return base
  for (let n = 2; ; n++) {
    const candidate = `${base} ${n}`
    if (!taken.has(agentSlug(candidate))) return candidate
  }
}
