// Web playbooks — the store in Flight Deck (captain_claw/flight_deck/browser_playbooks.py,
// /fd/playbooks). The same few hundred ready-made ways to get things done on
// the web that Captain for Chrome's panel offers: a front page, search,
// categories, the person's shelf ("My playbooks"), one playbook in full. A
// run is the person's own message in an agent's chat: the deck words it
// (/compose) and Flight Deck sends it there.

import { api } from './fileLibraryApi'

export type PlaybookKind = 'look' | 'act'
export type PlaybookWhere = 'any' | 'prefer_chrome' | 'chrome'
export type InputKind = 'text' | 'place' | 'date' | 'time' | 'number' | 'choice' | 'url' | 'file' | 'flag'

/** A playbook as a row or a card. */
export interface PlaybookSummary {
  id: string
  title: string
  category: string
  kind: PlaybookKind
  where: PlaybookWhere
  /** global, eu or a two-letter country code. */
  region: string
  /** Its first site ("google.com/maps"); '' for one on a site the person names. */
  site: string
  /** How people ask for it. */
  hint: string
  /** An act playbook's last button, never pressed. */
  stop: string
  /** How many values it needs from the person. */
  needs: number
  mine: boolean
  runs: number
}

export interface PlaybookInput {
  name: string
  label: string
  kind: InputKind
  required: boolean
  ask: 'before' | 'during' | 'agent'
  default?: string | number | boolean
  options?: string[]
  hint?: string
  // A number's range (min is 0 unless the playbook says otherwise).
  min?: number
  max?: number
}

export interface Playbook extends PlaybookSummary {
  says: string[]
  sites: string[]
  inputs: PlaybookInput[]
  /** Its steps in words, numbered. */
  steps: string[]
  read: { name: string; about: string }[]
  answer: string
  open: string
  named_site: boolean
}

export interface PlaybookCategory { id: string; label: string; icon: string; count: number }

export interface PlaybookFront {
  categories: PlaybookCategory[]
  total: number
  shelf: PlaybookSummary[]
  popular: PlaybookSummary[]
  top: Record<string, PlaybookSummary[]>
  regions: string[]
}

/** The person's country from the browser's language ("hr-HR" → "hr"), or ''. */
export function myRegion(): string {
  const m = /^[a-z]{2,3}-([A-Za-z]{2})\b/.exec(navigator.language || '')
  return m ? m[1].toLowerCase() : ''
}

/** 🇭🇷 for "hr", 🇪🇺 for "eu". */
export function flagOf(cc: string): string {
  return /^[a-z]{2}$/i.test(cc || '')
    ? String.fromCodePoint(...[...cc.toUpperCase()].map((c) => 0x1F1E6 + c.charCodeAt(0) - 65))
    : ''
}

export const playbooksApi = {
  front: (region = myRegion()) => api<PlaybookFront>(`/fd/playbooks?region=${encodeURIComponent(region)}&per=6`),
  search: (o: { q?: string; category?: string; region?: string; limit?: number }) => {
    const qs = new URLSearchParams({ q: o.q || '', category: o.category || '', region: o.region ?? myRegion(), limit: String(o.limit ?? 40) })
    return api<{ items: PlaybookSummary[] }>(`/fd/playbooks/search?${qs}`).then((r) => r.items)
  },
  one: (id: string) => api<Playbook>(`/fd/playbooks/${encodeURIComponent(id)}`),
  shelf: (id: string, on: boolean) =>
    api<{ ok: boolean; mine: boolean; shelf: string[] }>(`/fd/playbooks/${encodeURIComponent(id)}/shelf`, {
      method: 'POST', body: JSON.stringify({ on }) }),
  /** The message that asks an agent to run it with these values. */
  compose: (id: string, inputs: Record<string, unknown>, note = '') =>
    api<{ content: string; link: string }>(`/fd/playbooks/${encodeURIComponent(id)}/compose`, {
      method: 'POST', body: JSON.stringify({ inputs, note }) }),
}
