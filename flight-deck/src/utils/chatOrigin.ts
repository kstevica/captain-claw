import type { ChatMessage } from '../services/agentChat'

// What the chat shows of messages a person didn't write, and of paths on the
// agent's server.
//
// The runtime writes some messages into the agent's session with role "user"
// (captain_claw/msg_origin.py): a guard's corrective ("[system] STOP — you
// have already written this file…"), a fleet notice, a relayed result ("This
// is the ANSWER you were waiting for. Relay it…"), an automated turn's
// opener. The agent tells them apart by `origin` (replays, notification
// frames; the log tags rows an earlier client stored as the user's). None of
// them is the user's own bubble: plumbing is not shown at all, the rest is a
// muted note.

/** Origins the user never sees: talk between the runtime and the model. */
const HIDDEN = new Set(['corrective', 'fleet_notice'])

/** How a user-role message with this origin is shown: `hide`, as a `note`,
 *  or `null` — it is the person's own (no origin, or `human`). */
export function injectedAs(origin: unknown): 'hide' | 'note' | null {
  const o = typeof origin === 'string' ? origin.trim() : ''
  if (!o || o === 'human') return null
  return HIDDEN.has(o) ? 'hide' : 'note'
}

/** A notification frame's origin when the agent (an older build) didn't
 *  send one: its text's shape, as msg_origin reads it. */
export function notificationOrigin(content: string): string {
  const t = content.trimStart()
  if (/^\[Flight Deck\] Agent '[^']*' has \w+ the fleet/.test(t)) return 'fleet_notice'
  if (/^\[Delegated result from /.test(t)) return 'delegated_result'
  if (/^\[(?:Basna|Vatra|coding session) '.*?' (?:finished successfully|ran into an error)\]/.test(t)) return 'delegated_result'
  return 'notification'
}

/** A user-role message the runtime wrote, as the chat keeps it: a system
 *  note marked with its origin. */
export function asNote(msg: ChatMessage, origin: string): ChatMessage {
  return { ...msg, role: 'system', note: origin, via: undefined, sourceLabel: undefined }
}

const NOTE_LABELS: Record<string, string> = {
  delegated_result: 'Result handed to the agent',
  notification: 'Notice for the agent',
  cron: 'Scheduled task started',
  autonomy: 'Automatic step',
  flow: 'Flow step',
  peer: 'Message from another agent',
  life_tick: 'Automatic check',
  worker_task: 'Task for the agent',
  mcp_task: 'Task from a connected app',
  automated: 'Automatic message',
}

/** The note's one line: what it is ("Result from Researcher") and the start
 *  of its text without the bracketed header. */
export function noteSummary(origin: string, content: string): { label: string; preview: string } {
  let text = content.trim()
  let label = NOTE_LABELS[origin] || 'Message for the agent'
  const delegated = /^\[Delegated result from ([^\]]+)\]\s*/.exec(text)
  if (delegated) {
    label = `Result from ${delegated[1].trim()}`
    text = text.slice(delegated[0].length)
    // The instruction to the model, before the result itself.
    const body = text.indexOf('\n\n')
    if (body >= 0) text = text.slice(body + 2)
  } else {
    const run = /^\[((?:Basna|Vatra|coding session) '.*?') ([^\]]+)\]\s*/.exec(text)
    if (run) {
      label = `${run[1]} ${run[2]}`
      text = text.slice(run[0].length)
      const body = text.indexOf('\n\n')
      if (body >= 0) text = text.slice(body + 2)
    } else {
      text = text.replace(/^\[[^\]\n]{1,160}\]\s*/, '')
    }
  }
  const preview = text.replace(/\s+/g, ' ').trim().slice(0, 160)
  return { label, preview }
}

// ── Narration rows stored before their flag was kept ──

/** Log rows of between-step narration stored before `narration` went into
 *  their metadata come back as bare system rows (`legacy`), shown as boxes of their
 *  own: "Starting to collect…" under a finished answer, as if the job had
 *  started again. A bare system row inside a turn's activity — the next
 *  turn row after it is a tool call, or it follows a tool call and the
 *  turn's answer comes next — is narration again, and folds into the
 *  turn's collapsed activity. */
export function markLegacyNarration(messages: ChatMessage[]): ChatMessage[] {
  const bare = (m: ChatMessage) => m.role === 'system' && !!m.legacy
  const out = messages.slice()
  for (let i = 0; i < out.length; i++) {
    if (!bare(out[i])) continue
    let j = i
    while (j < out.length && bare(out[j])) j++
    const next = out[j]
    let k = i - 1
    while (k >= 0 && out[k].role === 'system' && !out[k].topic) k--
    const prev = k >= 0 ? out[k] : undefined
    const inActivity = next?.role === 'tool'
      || (prev?.role === 'tool' && (next?.role === 'assistant' || next === undefined))
    if (inActivity) for (let x = i; x < j; x++) out[x] = { ...out[x], narration: true }
    i = j - 1
  }
  return out
}

// ── Server paths in replies ──

// An absolute path on the agent's server (/home/…/workspace/saved/…), or one
// below its saved/ folder (saved/showcase/<session>/report.html): no spaces,
// quotes, brackets or markdown marks, ending in a file name with an
// extension. Not a link's target (after "(") or part of a URL.
const SERVER_PATH_RE = /(?<![\w(/:.-])(?:file:\/\/)?(?:(?:~|\/(?:home|Users|root|srv|opt|var|data|mnt|tmp|private|app))\/[^\s`'"()<>[\]{}|,;*]*|(?:\.\/)?(?:workspace\/)?saved\/[^\s`'"()<>[\]{}|,;*]*)\/([^\s`'"()<>[\]{}|,;*/]+\.[A-Za-z0-9]{1,8})(?![\w/])/g

/** The reply with each server path written as its file name: the file's
 *  card under the reply opens it, and the person has no use for where it
 *  lies on the server. Fenced code blocks are left as written. */
export function tidyServerPaths(markdown: string): string {
  if (!markdown || !/(?:\/|saved\/)/.test(markdown)) return markdown
  const parts = markdown.split(/(```[\s\S]*?(?:```|$))/)
  return parts.map((part, i) => (i % 2 === 1 ? part : part.replace(SERVER_PATH_RE, (_m, name: string) => name))).join('')
}
