import type { ChatMessage } from '../services/agentChat'

// The files an agent made in a chat, recognised from the chat itself — what
// the composer's Files menu lists. A file counts when the agent's reply names
// it, or a file-making tool reports making it (write, docx, image_gen, speech).
// What the user attached never counts, nor anything in a working or input
// folder of saved/ — the line the chat bridges draw for deliverables
// (captain_claw/turn_deliverables.py), plus tmp/ scratch. A mention is only a
// name: it counts once it matches a file in the agent's own file list.

/** saved/ folders that never hold a file made for the user. */
const WORKING_DIRS = new Set(['scripts', 'tools', 'skills', 'downloads', 'tmp'])

/** Tools whose `Path:` line is a file made for the user. A browser screenshot
 *  or a screen capture is the agent looking, not making. */
const PATH_TOOLS = new Set(['image_gen', 'pocket_tts', 'audio', 'termux'])

const WRITTEN_RE = /^Written \d+ chars \(\d+ lines?\) to (.+?)(?: \(requested: .*\))?$/
const DOCX_SAVED_RE = /^Saved(?: in place)?: (.+?)(?: \(the original .* is unchanged\))?$/
const PATH_LINE_RE = /^Path: (.+?)(?: \(requested: .*\))?$/
const ATTACHED_RE = /^\[Attached (?:image: (.+?)|file: .*? → (.+?))\]$/
// A path or file name in a reply: no spaces, quotes, brackets or markdown
// marks, ending in an extension. Names with spaces are found by the file list.
const NAME_RE = /[^\s`'"()<>[\]{}*|,;]+\.[A-Za-z0-9]{1,8}(?![\w/-])/g

/** One message's mentions, newest message first. */
interface Source {
  tokens: { token: string; at: number }[]
  /** A reply's text, for names a token can't hold (spaces). */
  text?: string
}

export interface ChatFileScan {
  sources: Source[]
  /** Below-saved/ paths of the files the user attached. */
  attached: Set<string>
  /** Changes when the chat names another file — a reason to list files again. */
  sig: string
}

export interface ChatFileEntry<F> {
  /** Path below the agent's saved/ folder ("showcase/<session>/report.md"). */
  rel: string
  name: string
  modified: number
  file: F
}

/** Where a path sits below the agent's saved/ folder, or null when it names
 *  nothing there (a bare name, or a path relative to somewhere else). */
export function belowSaved(path: string): string | null {
  const p = path.replace(/\\/g, '/').replace(/^file:\/\//i, '')
  const ws = p.indexOf('/workspace/saved/')
  if (ws >= 0) return p.slice(ws + '/workspace/saved/'.length)
  const rel = p.replace(/^\.\//, '')
  if (rel.startsWith('saved/')) return rel.slice('saved/'.length)
  const i = p.indexOf('/saved/')
  return i >= 0 ? p.slice(i + '/saved/'.length) : null
}

/** A file made for the user: not hidden, not in a working or input folder. */
export function madeForUser(rel: string): boolean {
  const parts = rel.split('/')
  if (parts.some((s) => s.startsWith('.'))) return false
  return parts.length < 2 || !WORKING_DIRS.has(parts[0].normalize('NFC').toLowerCase())
}

function lineMatch(output: string | undefined, re: RegExp): string[] {
  if (!output) return []
  const out: string[] = []
  for (const line of output.split('\n')) {
    const m = line.trim().match(re)
    if (m) out.push(m[1])
  }
  return out
}

function toolPaths(m: ChatMessage): string[] {
  // A peer's or a child's call comes as "<agent>:<tool>".
  const tool = (m.tool_name || '').slice((m.tool_name || '').lastIndexOf(':') + 1)
  if (tool === 'write') return lineMatch(m.tool_output, WRITTEN_RE).slice(0, 1)
  if (tool === 'docx') return lineMatch(m.tool_output, DOCX_SAVED_RE)
  if (PATH_TOOLS.has(tool)) return lineMatch(m.tool_output, PATH_LINE_RE)
  return []
}

function replyTokens(text: string): { token: string; at: number }[] {
  const out: { token: string; at: number }[] = []
  for (const m of text.matchAll(NAME_RE)) {
    const token = m[0]
    if (token.includes('://') || /^(?:www\.|mailto:)/i.test(token)) continue
    out.push({ token, at: m.index ?? 0 })
  }
  return out
}

/** What the chat says about files. Pure, and cheap enough to run on every
 *  new message. */
export function scanChatFiles(messages: ChatMessage[]): ChatFileScan {
  const sources: Source[] = []
  const attached = new Set<string>()
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m.role === 'tool') {
      const paths = toolPaths(m)
      if (paths.length) sources.push({ tokens: paths.map((token, at) => ({ token, at })) })
    } else if (m.role === 'assistant' && m.content) {
      const tokens = replyTokens(m.content)
      sources.push({ tokens, text: m.content })
    } else if (m.role === 'user' && m.content) {
      for (const line of m.content.split('\n')) {
        const a = line.trim().match(ATTACHED_RE)
        const path = a && (a[1] || a[2])
        if (path) attached.add(belowSaved(path) ?? path)
      }
    }
  }
  const sig = sources.flatMap((s) => s.tokens.map((t) => t.token)).join('\n')
  return { sources, attached, sig }
}

function newest<F>(entries: ChatFileEntry<F>[]): ChatFileEntry<F> | null {
  let best: ChatFileEntry<F> | null = null
  for (const e of entries) if (!best || e.modified > best.modified) best = e
  return best
}

/** The file a mention names: by its path below saved/ when it has one, else
 *  by how its path ends. Several files fit (a bare name) → the newest. */
function matchToken<F>(token: string, byName: Map<string, ChatFileEntry<F>[]>): ChatFileEntry<F> | null {
  const anchored = belowSaved(token)
  // Any other absolute path is outside saved/.
  if (anchored === null && /^(?:\/|~|[A-Za-z]:[\\/])/.test(token)) return null
  const want = (anchored ?? token).replace(/^\.\//, '')
  const cands = byName.get(want.slice(want.lastIndexOf('/') + 1))
  if (!cands) return null
  // Below saved/: that very file (a folder above the workspace may be called
  // "saved" too). Else a relative path matches by its end, a bare name any.
  if (anchored !== null) return newest(cands.filter((e) => e.rel === want || e.rel.endsWith('/saved/' + want)))
  return newest(want.includes('/') ? cands.filter((e) => e.rel === want || e.rel.endsWith('/' + want)) : cands)
}

/** The last `limit` files the chat made, newest first — from the agent's
 *  file list, in the order the chat last named them. */
export function resolveChatFiles<F>(scan: ChatFileScan, entries: ChatFileEntry<F>[], limit = 5): ChatFileEntry<F>[] {
  const usable = entries.filter((e) => madeForUser(e.rel) && !scan.attached.has(e.rel))
  const byName = new Map<string, ChatFileEntry<F>[]>()
  for (const e of usable) {
    const list = byName.get(e.name)
    if (list) list.push(e)
    else byName.set(e.name, [e])
  }
  const spaced = [...byName.keys()].filter((n) => /\s/.test(n))
  const out: ChatFileEntry<F>[] = []
  const seen = new Set<string>()
  for (const src of scan.sources) {
    const found: { at: number; e: ChatFileEntry<F> }[] = []
    for (const { token, at } of src.tokens) {
      const e = matchToken(token, byName)
      if (e) found.push({ at, e })
    }
    if (src.text) {
      for (const name of spaced) {
        const at = src.text.indexOf(name)
        const e = at >= 0 ? newest(byName.get(name)!) : null
        if (e) found.push({ at, e })
      }
    }
    found.sort((a, b) => a.at - b.at)
    for (const { e } of found) {
      if (seen.has(e.rel)) continue
      seen.add(e.rel)
      out.push(e)
      if (out.length >= limit) return out
    }
  }
  return out
}
