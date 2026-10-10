// Text helpers for the HUD chat: cleaning agent / user text for display, a
// plain-text version of a reply for speech, and markdown-file references in
// replies (rendered as chips that open the file viewer).
//
// Everything here is pure (no DOM, no store) so it can run on every message
// without surprises.

import { sanitizeAgentContent } from '../../utils/sanitizeAgentContent'
import type { HudFile } from '../api'
import { truncate } from '../format'
import { splitMarkdown } from '../markdown/chunks'

// ── Display cleaning ──

/**
 * JS twin of msg_origin.SURFACE_BLOCK_RE: the glasses rules block we prepend
 * to the first message of a connection. The agent strips it from stored
 * history, but live echoes (and older replays) can still carry it.
 */
export const SURFACE_BLOCK_RE = /\[SYSTEM CONTEXT — do not echo, quote, or acknowledge this block in your reply\.\][\s\S]*?USER MESSAGE:\n/

const SURFACE_BLOCK_ALL = new RegExp(SURFACE_BLOCK_RE.source, 'g')

/** Rating prompt the agent appends ("---\n💡 If this worked well… rate good / rate bad"). */
const RATING_RE = /\n---\n(?:\u{1F4A1}|\u{1F514}).*(?:rate good|rate bad).*$/isu
const SUGGESTIONS_RE = /\nSUGGESTED NEXT STEPS[\s\S]*$/i

/** Same as ChatPanel's stripSuggestions: drop the trailing next-steps / rating blocks. */
export function stripSuggestions(text: string): string {
  return text.replace(RATING_RE, '').replace(SUGGESTIONS_RE, '').trimEnd()
}

/** User text as the wearer wrote it (rules block removed). */
export function cleanUserText(raw: unknown): string {
  return (typeof raw === 'string' ? raw : '').replace(SURFACE_BLOCK_ALL, '').trim()
}

/** Assistant text ready for <HudMarkdown>: thinking / memory echoes and the
 *  trailing suggestions + rating blocks removed. */
export function cleanAssistantText(raw: unknown): string {
  const text = typeof raw === 'string' ? raw : ''
  if (!text) return ''
  return stripSuggestions(sanitizeAgentContent(text)).trim()
}

/**
 * Longest reply kept for display. Markdown parse + layout costs ~7 µs per
 * character on a laptop and the glasses' CPU is ~12× slower — paid when the
 * reply arrives and again whenever the Chat tab mounts — so a desktop-length
 * answer would freeze the D-pad for seconds.
 */
export const MAX_REPLY_CHARS = 6_000

/** `md` cut at a block boundary near `max` characters, with a note saying how
 *  much is left for the dashboard. Short replies come back unchanged. */
export function capReply(md: string, max = MAX_REPLY_CHARS): string {
  // splitMarkdown keeps anything up to 1.5 × target whole.
  if (md.length <= max * 1.5) return md
  let head = splitMarkdown(md, max)[0]
  if (head.length > max * 1.5) {
    // No safe block boundary (one huge table, list or code block): cut at a
    // line, closing a code fence left open so the note stays prose.
    const cut = head.lastIndexOf('\n', max)
    head = head.slice(0, cut > max * 0.5 ? cut : max)
    const fences = head.match(/^ {0,3}(?:`{3,}|~{3,})/gm)
    if (fences && fences.length % 2 === 1) head += `\n${fences[fences.length - 1].trim()}`
  }
  const more = Math.max(1, Math.round((md.length - head.length) / 1024))
  return `${head.trimEnd()}\n\n*… ${more} KB more — open this chat on Flight Deck.*`
}

/** Short labels for user rows the agent's machinery wrote (msg_origin.py). */
const SYNTHETIC_LABELS: Record<string, string> = {
  cron: 'Scheduled job',
  autonomy: 'Autonomous work',
  flow: 'Flow',
  peer: 'Task from another agent',
  delegated_result: 'Result from another agent',
  mcp_task: 'MCP task',
  life_tick: 'Life tick',
  worker_task: 'Worker task',
  automated: 'Automated turn',
  fleet_notice: 'Fleet update',
  notification: 'Notification',
}

/** A user row the wearer did not write, as one short system line. */
export function syntheticRowText(origin: string, text: string): string {
  const label = SYNTHETIC_LABELS[origin] ?? 'Automated turn'
  const body = text.replace(/^\[Automated turn[^\]\n]*\]\s*/i, '').replace(/\s+/g, ' ').trim()
  return body ? `${label}: ${truncate(body, 120)}` : label
}

const IDLE_RE = /^(ready|idle|done|completed)$/i

export function isIdleStatus(text: string): boolean {
  return IDLE_RE.test(text.trim())
}

/** "thinking" → "Thinking…", "Using web_search..." → "Using web_search…". */
export function prettyStatus(text: string): string {
  let t = text.trim().replace(/\.{3}$/, '…')
  if (/^thinking$/i.test(t)) t = 'Thinking…'
  if (t) t = t[0].toUpperCase() + t.slice(1)
  return t.length > 90 ? t.slice(0, 89).trimEnd() + '…' : t
}

// ── Misc ──

/** Epoch ms from an agent timestamp (ISO string, unix seconds or ms). */
export function toMs(ts: unknown): number {
  if (typeof ts === 'number' && Number.isFinite(ts) && ts > 0) return ts > 1e12 ? ts : ts * 1000
  if (typeof ts === 'string' && ts) {
    const n = Number(ts)
    if (Number.isFinite(n) && n > 0) return n > 1e12 ? n : n * 1000
    const p = Date.parse(ts)
    if (Number.isFinite(p)) return p
  }
  return Date.now()
}

/** Random id (crypto.randomUUID needs a secure context; LAN http has none). */
export function newId(): string {
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  } catch { /* insecure context */ }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

// ── Speech ──

function shortUrl(url: string): string {
  const m = /^(?:https?:\/\/)?(?:www\.)?([^/\s?#]+)/i.exec(url)
  return m ? m[1] : 'a link'
}

/**
 * A reply as something worth hearing: code blocks dropped, tables read as
 * comma-separated cells, links read by their label, URLs by their host,
 * markdown markers removed, capped near `max` characters at a sentence or
 * word boundary.
 */
export function speakableText(md: string, max = 600): string {
  let s = md.replace(/\r\n?/g, '\n')
  // Fenced code: not worth reading aloud.
  s = s.replace(/^[ \t]*(```|~~~)[^\n]*\n[\s\S]*?(?:^[ \t]*\1[^\n]*$|(?![\s\S]))/gm, ' ')
  // Tables: drop separator rows, read cells.
  s = s.replace(/^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$/gm, '')
  s = s.replace(/^[ \t]*\|(.*)\|?[ \t]*$/gm, (_, row: string) =>
    row.split('|').map((c) => c.trim()).filter(Boolean).join(', ') + '.')
  // Images and links → their text; bare URLs → host.
  s = s.replace(/!\[([^\]\n]*)\]\([^)\n]*\)/g, '$1')
  s = s.replace(/\[([^\]\n]+)\]\([^)\n]*\)/g, '$1')
  s = s.replace(/<(https?:\/\/[^>\s]+)>/g, (_, u: string) => shortUrl(u))
  s = s.replace(/\bhttps?:\/\/[^\s)>\]]+/g, (u) => shortUrl(u))
  // HTML tags (rendered as text on screen; noise when spoken).
  s = s.replace(/<\/?[a-z][^>\n]*>/gi, ' ')
  // Block markers: headings, quotes, list bullets, rules.
  s = s.replace(/^[ \t]*#{1,6}[ \t]+/gm, '')
  s = s.replace(/^[ \t]*>[ \t]?/gm, '')
  s = s.replace(/^[ \t]*(?:[-*+]|\d+[.)])[ \t]+(?:\[[ xX]\][ \t]+)?/gm, '')
  s = s.replace(/^[ \t]*(?:[-*_][ \t]*){3,}$/gm, '')
  // Inline markers.
  s = s.replace(/`+([^`\n]+)`+/g, '$1')
  s = s.replace(/(\*{1,3}|_{2,3}|~~)(\S(?:[^\n]*?\S)?)\1/g, '$2')
  s = s.replace(/(^|[^\w])_(\S(?:[^\n_]*?\S)?)_(?!\w)/g, '$1$2')
  // Lines → sentences.
  s = s
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => (/[.!?:;…,]$/.test(l) ? l : l + '.'))
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim()
  if (s.length <= max) return s
  const cut = s.slice(0, max)
  const sentence = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('! '), cut.lastIndexOf('? '))
  if (sentence > max * 0.5) return cut.slice(0, sentence + 1)
  const word = cut.lastIndexOf(' ')
  return (word > max * 0.5 ? cut.slice(0, word) : cut).trimEnd() + '…'
}

/** Split speech into short utterances (long ones stall on some engines). */
export function speechChunks(text: string, size = 200): string[] {
  const parts = text.split(/(?<=[.!?…])\s+/)
  const out: string[] = []
  let cur = ''
  for (const p of parts) {
    if (!p) continue
    if (cur && cur.length + 1 + p.length > size) { out.push(cur); cur = '' }
    if (p.length > size) {
      // A run-on sentence: split at word boundaries.
      let rest = p
      while (rest.length > size) {
        const at = rest.lastIndexOf(' ', size)
        const n = at > size * 0.4 ? at : size
        out.push(rest.slice(0, n).trim())
        rest = rest.slice(n).trim()
      }
      cur = rest
    } else {
      cur = cur ? `${cur} ${p}` : p
    }
  }
  if (cur) out.push(cur)
  return out
}

// ── Markdown file references in replies ──

const MD_HINT_RE = /\.(?:md|markdown)\b/i
const CODE_SPAN_MD_RE = /`([^`\n]{1,300}?\.(?:md|markdown))`/gi
const LINK_MD_RE = /\]\(\s*<?([^()<>\n]{1,300}?\.(?:md|markdown))>?(?:\s+"[^"\n]*")?\s*\)/gi
const BARE_MD_RE = /(?:file:\/\/)?[^\s`'"<>()[\]{}|*,;:=!?]{1,300}?\.(?:md|markdown)(?![\w/-])/gi

function normalizeRef(raw: string): string | null {
  let r = raw.trim().replace(/\\/g, '/')
  r = r.replace(/^file:\/\//i, '')
  // Web links (with or without a scheme) are not agent files.
  if (/:\/\//.test(r) || /^(?:https?:|www\.|\/\/)/i.test(r) || /^[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}\//i.test(r)) return null
  r = r.replace(/^\.\//, '')
  try { if (r.includes('%')) r = decodeURIComponent(r) } catch { /* keep as is */ }
  const base = r.split('/').pop() || ''
  const stem = base.replace(/\.(?:md|markdown)$/i, '')
  if (!stem || /^\.+$/.test(stem)) return null
  return r
}

function isPathSuffix(long: string, short: string): boolean {
  if (long === short) return true
  if (!long.endsWith(short)) return false
  const ch = long[long.length - short.length - 1]
  return ch === '/' || ch === ' '
}

/**
 * Markdown files a reply mentions (link targets, `code spans`, bare paths),
 * in order of appearance, de-duplicated (a bare "report.md" and
 * "saved/report.md" are one file), at most `max`.
 */
export function findFileRefs(text: string, max = 3): string[] {
  if (!text || !MD_HINT_RE.test(text)) return []
  const found: { at: number; ref: string }[] = []
  let offset = 0
  for (const line of text.split('\n')) {
    if (MD_HINT_RE.test(line) && line.length <= 4000) {
      for (const re of [CODE_SPAN_MD_RE, LINK_MD_RE]) {
        re.lastIndex = 0
        for (let m = re.exec(line); m; m = re.exec(line)) found.push({ at: offset + m.index, ref: m[1] })
      }
      BARE_MD_RE.lastIndex = 0
      for (let m = BARE_MD_RE.exec(line); m; m = BARE_MD_RE.exec(line)) found.push({ at: offset + m.index, ref: m[0] })
    }
    offset += line.length + 1
  }
  found.sort((a, b) => a.at - b.at)
  const out: string[] = []
  for (const f of found) {
    const ref = normalizeRef(f.ref)
    if (!ref) continue
    const lc = ref.toLowerCase()
    const i = out.findIndex((o) => {
      const lo = o.toLowerCase()
      return isPathSuffix(lo, lc) || isPathSuffix(lc, lo)
    })
    if (i < 0) out.push(ref)
    else if (ref.length > out[i].length) out[i] = ref
  }
  return out.slice(0, max)
}

/** File name shown on a chip. */
export function refName(ref: string): string {
  return ref.split('/').pop() || ref
}

function norm(p: string): string {
  return p.replace(/\\/g, '/').replace(/^\.\//, '').toLowerCase()
}

/**
 * The listed file a reference points at: exact path, then a path-suffix match
 * either way (replies often show absolute host paths or workspace-relative
 * ones), then the newest file with the same name (`files` is newest first).
 */
export function matchFileRef(files: HudFile[], ref: string): HudFile | null {
  const r = norm(ref)
  const base = r.split('/').pop() || r
  return (
    files.find((f) => norm(f.logical) === r || norm(f.key) === r)
    ?? files.find((f) => {
      const l = norm(f.logical)
      const k = norm(f.key)
      return l.endsWith('/' + r) || r.endsWith('/' + l) || k.endsWith('/' + r)
    })
    ?? files.find((f) => f.name.toLowerCase() === base)
    ?? null
  )
}
