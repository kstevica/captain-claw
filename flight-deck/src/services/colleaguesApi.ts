import { api } from './fileLibraryApi'

// Colleagues: each agent's identity (role, what it does, the names it answers
// to, its own inbox and aliases, whether it picks up email) and which of its
// colleagues it may hand work to. Flight Deck: colleagues.py.

export interface ColleagueAgent {
  slug: string
  name: string
  role: string
  about: string
  /** What it answers to in email (its name when none was set). */
  names: string[]
  names_set: boolean
  /** Aliases that land in your inbox, e.g. you+researcher@gmail.com. */
  addresses: string[]
  /** The IMAP inbox that is its own ('' for none). */
  inbox: string
  inbox_email: string
  /** Its own inbox's address, else its first alias. */
  email: string
  pick_up_mail: boolean
  hands_to: 'all' | string[]
  asleep: boolean
  on?: boolean
}

export interface ColleagueInbox {
  id: string
  email: string
  label: string
  can_send: boolean
  sending_enabled: boolean
  agents: string[]
}

export interface ColleaguesView {
  agents: ColleagueAgent[]
  inboxes: ColleagueInbox[]
  /** Which of your mail can be watched. */
  mail: { gmail: boolean; outlook: boolean; imap: boolean }
}

export interface IdentityPatch {
  role?: string
  about?: string
  names?: string[]
  addresses?: string[]
  inbox?: string
  pick_up_mail?: boolean
  hands_to?: 'all' | string[]
}

const BASE = '/fd/colleagues'

export const colleaguesApi = {
  list: () => api<ColleaguesView>(BASE),
  save: (slug: string, patch: IdentityPatch) =>
    api<{ agent: ColleagueAgent }>(`${BASE}/${encodeURIComponent(slug)}`, { method: 'PUT', body: JSON.stringify(patch) }),
}
