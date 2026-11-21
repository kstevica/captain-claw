import type { Archetype } from '../../services/tierConfig'

// ── The store's Featured row ─────────────────────────────────────────
//
// Five agents most people start with. Four were chosen 2026-11-15 from the catalog and
// from what agents on real decks were made of (Data Analyst, Deep Researcher
// and Inbox Manager among the most created): research, mail and calendar,
// numbers, and writing — the everyday work of a small firm built on
// documents and client email. Coding agents were made often too, but by
// people testing the deck; the store's first row is for everyone else.
// The fifth, the Computer User (2026-11-21), works in the person's own browser
// and computer like a person would, towards a goal.
// An id the catalog doesn't offer is skipped, and the row fills up from the
// catalog's own order, so it is never short.

export const FEATURED_IDS = ['deep-researcher', 'inbox-manager', 'data-analyst', 'editor-writer', 'computer-user'] as const

export function featured(list: Archetype[], count = FEATURED_IDS.length): Archetype[] {
  const byId = new Map(list.map((a) => [a.id, a]))
  const picks: Archetype[] = []
  for (const id of FEATURED_IDS) {
    const a = byId.get(id)
    if (a) picks.push(a)
  }
  for (const a of list) {
    if (picks.length >= count) break
    if (!picks.includes(a)) picks.push(a)
  }
  return picks.slice(0, count)
}
