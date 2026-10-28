/** The documents an agent keeps in its knowledge/ folder — what its owner's
 *  setup gave it — to browse, through Flight Deck (/fd/agent-knowledge/{slug};
 *  it reads the folder itself, so the agent may be off). Read-only. */

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import type { ReadState } from './agentPreset'

export interface KnowledgeDoc {
  /** Its file in knowledge/ — how the other calls name it. */
  id: string
  /** The name it was given under ("handbook.pdf"). */
  doc: string
  title: string
  summary: string
  /** How it is used ("Reference", "Rules to follow"…; "" — the model decided). */
  use: string
  parts: number
  bytes: number
  /** Its section map's file, when it is long enough for one. */
  map: string
  /** Given by a setup (else a file put in the folder some other way). */
  from_setup: boolean
  read: ReadState
  /** Datastore tables its spreadsheet became. */
  tables: string[]
}

export interface KnowledgeOverview {
  agent: string
  documents: KnowledgeDoc[]
  /** The glossary and the owner's rules, when there are. */
  extras: { id: string; title: string }[]
  totals: { documents: number; bytes: number; sections: number; read: number; tracked: number }
}

export interface KnowledgeSection {
  i: number
  level: number
  title: string
  /** Its parents' titles. */
  path: string[]
  chars: number
  /** The model read it — null for a heading with nothing of its own, or a document from before reads were kept. */
  read: boolean | null
}

export interface KnowledgeDocDetail extends KnowledgeDoc {
  chars: number
  sections: KnowledgeSection[]
}

export interface KnowledgeSectionText {
  id: string
  title: string
  i: number
  level: number
  section: string
  path: string[]
  /** This page of its own text (its heading to its first subsection), Markdown. */
  text: string
  page: number
  pages: number
  chars: number
  /** Where it starts in the agent's files. */
  file: string
  line: number
  children: { i: number; title: string }[]
  prev: { i: number; title: string } | null
  next: { i: number; title: string } | null
  read: boolean | null
}

export interface KnowledgeHit {
  file: string
  first: number
  last: number
  text: string
  /** "Document › Chapter › Section". */
  where: string
  /** The section to open: its document and index… */
  id?: string
  title?: string
  i?: number
  section?: string
  path?: string[]
  /** …or the glossary / the owner's rules. */
  extra?: string
}

async function get<T>(path: string): Promise<T> {
  const headers = (): Record<string, string> => {
    const { token, authEnabled } = useAuthStore.getState()
    return authEnabled && token ? { Authorization: `Bearer ${token}` } : {}
  }
  const go = () => fetch(path, { headers: headers(), credentials: 'include', cache: 'no-store' })
  let res = await go()
  if (res.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    res = await go()
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : `${res.status}`)
  }
  return res.json()
}

const base = (slug: string) => `/fd/agent-knowledge/${encodeURIComponent(slug)}`
const qs = (q: Record<string, string | number>) =>
  new URLSearchParams(Object.entries(q).map(([k, v]) => [k, String(v)])).toString()

export const agentKnowledgeApi = {
  overview: (slug: string) => get<KnowledgeOverview>(base(slug)),
  doc: (slug: string, id: string) => get<KnowledgeDocDetail>(`${base(slug)}/doc?${qs({ id })}`),
  section: (slug: string, id: string, i: number, page = 1) =>
    get<KnowledgeSectionText>(`${base(slug)}/section?${qs({ id, i, page })}`),
  extra: (slug: string, id: string) => get<{ id: string; title: string; text: string }>(`${base(slug)}/extra?${qs({ id })}`),
  search: (slug: string, q: string, id = '') =>
    get<{ results: KnowledgeHit[] }>(`${base(slug)}/search?${qs(id ? { q, id } : { q })}`),
}
