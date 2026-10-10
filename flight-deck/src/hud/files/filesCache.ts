// In-memory cache for the Files tab.
//
//  - The markdown file list per agent: the file screen looks up size + trust
//    without another request, and Back to the list renders instantly (so the
//    focus-restore lands on the row the wearer left).
//  - The last few file texts: re-opening a file costs nothing over the
//    glasses' ~500 Kbps link. A text is reused only while the listed
//    `modified` time still matches.
//
// Everything is dropped when the signed-in user changes or signs out.

import { useAuthStore } from '../../stores/authStore'
import { listMarkdownFiles, type HudAgent, type HudFile } from '../api'

/** A list younger than this is shown without refetching. */
export const LIST_TTL_MS = 60_000

export interface FilesEntry {
  files: HudFile[]
  /** Epoch ms of the load. */
  at: number
}

const lists = new Map<string, FilesEntry>()
const inflight = new Map<string, Promise<HudFile[]>>()
/** Bumped by clearFilesCache so a request started before it never writes back. */
let generation = 0

export function getCachedFiles(agentId: string): FilesEntry | null {
  return lists.get(agentId) ?? null
}

export function isFresh(entry: FilesEntry | null, now = Date.now()): boolean {
  return !!entry && now - entry.at < LIST_TTL_MS
}

export function findCachedFile(agentId: string, key: string): HudFile | null {
  return lists.get(agentId)?.files.find((f) => f.key === key) ?? null
}

/** Fetch the agent's markdown files and cache them (single flight per agent). */
export function loadFiles(agent: HudAgent): Promise<HudFile[]> {
  const running = inflight.get(agent.id)
  if (running) return running
  const gen = generation
  const p = listMarkdownFiles(agent)
    .then((files) => {
      if (gen === generation) lists.set(agent.id, { files, at: Date.now() })
      return files
    })
    .finally(() => {
      if (inflight.get(agent.id) === p) inflight.delete(agent.id)
    })
  inflight.set(agent.id, p)
  return p
}

// ── File texts (small LRU) ──

const TEXT_MAX_ENTRIES = 5
const TEXT_MAX_CHARS = 2_000_000

interface TextEntry { id: string; modified: number; text: string }

let texts: TextEntry[] = []

function textId(agentId: string, key: string): string {
  return `${agentId}\u0000${key}`
}

/** A cached text of this file version, or null. Pure (safe during render). */
export function cachedText(agentId: string, key: string, modified: number): string | null {
  if (!modified) return null
  const id = textId(agentId, key)
  const hit = texts.find((t) => t.id === id)
  return hit && hit.modified === modified ? hit.text : null
}

export function rememberText(agentId: string, key: string, modified: number, text: string): void {
  if (!modified || text.length > TEXT_MAX_CHARS) return
  const id = textId(agentId, key)
  texts = texts.filter((t) => t.id !== id)
  texts.push({ id, modified, text })
  let total = texts.reduce((n, t) => n + t.text.length, 0)
  while (texts.length > TEXT_MAX_ENTRIES || total > TEXT_MAX_CHARS) {
    const old = texts.shift()
    if (!old) break
    total -= old.text.length
  }
}

export function clearFilesCache(): void {
  generation++
  lists.clear()
  inflight.clear()
  texts = []
}

useAuthStore.subscribe((s, prev) => {
  if (s.user?.id !== prev.user?.id || (prev.isAuthenticated && !s.isAuthenticated)) clearFilesCache()
})
