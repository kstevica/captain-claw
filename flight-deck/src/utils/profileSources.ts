// The Profile page's Personal tab, the part that isn't a component: the
// details draft (name, links, company) against what the server holds, link
// labels by host, and the document kinds in words. Mirrors
// captain_claw/flight_deck/profile_sources.py where it decides the same thing.

import type { ProfileFields, ProfileLink, ProfileSources, ResearchSource } from '../services/profile'

/** What the person edits on the Personal tab, apart from the three texts. */
export interface DetailsDraft {
  name: string
  links: ProfileLink[]
  company_name: string
  company_links: ProfileLink[]
}

/** The kiosk opens its Profile dialog on this window event (the bell sends
 *  it: a finished research run). */
export const OPEN_PROFILE_EVENT = 'fd:open-profile'

export const EMPTY_DETAILS: DetailsDraft = { name: '', links: [], company_name: '', company_links: [] }

export function detailsOf(s: ProfileSources): DetailsDraft {
  return { name: s.name, links: s.links.map((l) => ({ ...l })), company_name: s.company_name, company_links: s.company_links.map((l) => ({ ...l })) }
}

// A link's label when the person gives none, by host (the server does the
// same on save; this fills the box as they type).
const KNOWN_HOSTS: [string, string][] = [
  ['linkedin.com', 'LinkedIn'], ['github.com', 'GitHub'], ['x.com', 'X'], ['twitter.com', 'X'],
  ['instagram.com', 'Instagram'], ['facebook.com', 'Facebook'], ['youtube.com', 'YouTube'],
  ['medium.com', 'Medium'], ['substack.com', 'Substack'], ['threads.net', 'Threads'],
  ['bsky.app', 'Bluesky'], ['mastodon.social', 'Mastodon'], ['tiktok.com', 'TikTok'],
  ['crunchbase.com', 'Crunchbase'], ['scholar.google.com', 'Google Scholar'],
  ['researchgate.net', 'ResearchGate'], ['orcid.org', 'ORCID'], ['dribbble.com', 'Dribbble'],
  ['behance.net', 'Behance'], ['stackoverflow.com', 'Stack Overflow'], ['gitlab.com', 'GitLab'],
]

export function hostOf(url: string): string {
  const text = url.trim()
  if (!text) return ''
  try {
    const host = new URL(text.includes('://') ? text : `https://${text}`).hostname.toLowerCase()
    return host.startsWith('www.') ? host.slice(4) : host
  } catch {
    return ''
  }
}

export function labelForUrl(url: string): string {
  const host = hostOf(url)
  if (!host) return ''
  for (const [suffix, name] of KNOWN_HOSTS) {
    if (host === suffix || host.endsWith(`.${suffix}`)) return name
  }
  return host.slice(0, 40)
}

/** The links worth sending: rows with an address, spaces trimmed. */
export function keptLinks(links: ProfileLink[]): ProfileLink[] {
  return links
    .map((l) => ({ url: l.url.trim(), label: l.label.trim() }))
    .filter((l) => l.url)
}

function sameLinks(a: ProfileLink[], b: ProfileLink[]): boolean {
  const x = keptLinks(a)
  const y = keptLinks(b)
  return x.length === y.length && x.every((l, i) => l.url === y[i].url && l.label === y[i].label)
}

/** Which of the details changed against what the server holds — the PUT body
 *  (the name goes to the account, not here). */
export function detailsPatch(draft: DetailsDraft, saved: ProfileSources): {
  links?: ProfileLink[]; company_name?: string; company_links?: ProfileLink[]
} {
  const patch: { links?: ProfileLink[]; company_name?: string; company_links?: ProfileLink[] } = {}
  if (!sameLinks(draft.links, saved.links)) patch.links = keptLinks(draft.links)
  if (draft.company_name.trim() !== saved.company_name) patch.company_name = draft.company_name.trim()
  if (!sameLinks(draft.company_links, saved.company_links)) patch.company_links = keptLinks(draft.company_links)
  return patch
}

/** Whether the name changed (only where the account can be renamed). */
export function nameChanged(draft: DetailsDraft, saved: ProfileSources): boolean {
  return saved.can_rename && draft.name.trim() !== saved.name && draft.name.trim() !== ''
}

export function detailsDirty(draft: DetailsDraft, saved: ProfileSources | null): boolean {
  if (!saved) return false
  return nameChanged(draft, saved) || Object.keys(detailsPatch(draft, saved)).length > 0
}

/** Nothing for the researcher to read from. */
export function sourcesEmpty(draft: DetailsDraft, saved: ProfileSources | null): boolean {
  return keptLinks(draft.links).length === 0 && !draft.company_name.trim()
    && keptLinks(draft.company_links).length === 0 && !(saved?.documents.length)
}

/** ".pdf, .docx, …" as words for the upload hint. */
export function docKindsLabel(kinds: string[]): string {
  const words: Record<string, string> = {
    '.pdf': 'PDF', '.docx': 'Word', '.xlsx': 'Excel', '.pptx': 'PowerPoint', '.txt': 'text',
    '.md': 'Markdown', '.csv': 'CSV', '.html': 'HTML', '.htm': '',
  }
  const out: string[] = []
  for (const k of kinds) {
    const w = words[k] ?? k.replace(/^\./, '').toUpperCase()
    if (w && !out.includes(w)) out.push(w)
  }
  return out.join(', ')
}

/** The drafts that differ from what is in the text boxes now. */
export function newDrafts(drafts: ProfileFields | null, current: ProfileFields): Partial<ProfileFields> {
  const out: Partial<ProfileFields> = {}
  if (!drafts) return out
  for (const k of ['about_me', 'company', 'instructions'] as const) {
    if (drafts[k].trim() && drafts[k] !== current[k]) out[k] = drafts[k]
  }
  return out
}

/** "3 sources read, 1 couldn't be" for the finished run's line. */
export function sourcesSummary(sources: ResearchSource[]): string {
  const read = sources.filter((s) => s.ok && s.kind !== 'search').length
  const failed = sources.filter((s) => !s.ok).length
  const searches = sources.filter((s) => s.kind === 'search').length
  const parts = [`${read} source${read === 1 ? '' : 's'} read`]
  if (searches) parts.push(`${searches} web search${searches === 1 ? '' : 'es'}`)
  if (failed) parts.push(`${failed} couldn't be read`)
  return parts.join(', ')
}

export function dateLabel(unixSeconds: number): string {
  if (!unixSeconds) return ''
  try {
    return new Date(unixSeconds * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
  } catch {
    return ''
  }
}

export function timeLabel(unixSeconds: number | null): string {
  if (!unixSeconds) return ''
  try {
    return new Date(unixSeconds * 1000).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}
