/** Agent bundles — one agent in a .captain file, to back it up, restore it, or
 *  hand it to another Flight Deck (/fd/agent-bundle/*). An admin acting for a
 *  user passes `asUser` (X-FD-Act-As), as on the other agent routes. */

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import { ACT_AS_HEADER } from './docker'

export type BundleCategory = 'identity' | 'onboarding' | 'memory' | 'data' | 'conversations' | 'files' | 'schedules'

export interface BundleSummary {
  agent: { name: string; description: string }
  /** Unix seconds. */
  made: number
  captain_version: string
  made_by: string
  preset: string
  categories: BundleCategory[]
  /** Bytes, unpacked. */
  size: number
  counts: Record<string, number>
  notes: string[]
  secrets: boolean
  /** What it was connected to on the deck it came from — by name; none of it travels. */
  connections: Connection[]
  /** A named package's version: an agent holding an earlier one can be updated from it. */
  package: Package | null
}

export interface Package { id: string; name: string; version: number }

export interface Connection { kind: string; label: string }

export type Preset = 'template' | 'trained' | 'full'

/** What the export carries beyond the categories — the review's choices. */
export interface ExportOptions {
  preset: Preset
  categories?: BundleCategory[]
  /** Insight categories that go; `null`: all of them. */
  insight_categories?: string[] | null
  exclude_insights?: string[]
  /** Its contacts; unset: only in a full backup. */
  contacts?: boolean
  /** A full backup with a passphrase carries its keys, sealed. */
  passphrase?: string
  /** Name it to send updates later: this export is the package's next version. */
  package?: string
}

export interface ReviewItem { id: string; content: string; importance: number }
export interface ReviewCategory { id: string; label: string; count: number; default: boolean; items: ReviewItem[] }
export interface AgentReview { insights: ReviewCategory[]; contacts: number }
export interface ModelTier { tier: string; provider: string; model: string }

export interface ExportResult {
  id: string
  slug: string
  file_name: string
  /** Bytes, the file. */
  size: number
  made: number
  manifest: BundleSummary
}

export interface UploadResult extends BundleSummary {
  upload_id: string
  suggested_name: string
  model: { provider: string; model: string }
  /** The tier it ran on where it was made ("" when its model was picked by hand). */
  model_tier: string
  /** Its keys are in the file, sealed with a passphrase. */
  has_secrets: boolean
  /** Your agents holding an earlier version of its package. */
  updates: { slug: string; name: string; from_version: number }[]
}

export type MergeMode = 'append' | 'reset' | 'update'

export interface ImportOptions {
  name: string
  model?: { mode: 'keep' } | { mode: 'tier'; tier: string }
  passphrase?: string
  /** Into one of your agents instead of a new one: add to what it has, or replace it. */
  target?: { kind: 'new' } | { kind: 'existing'; slug: string; mode: MergeMode }
}

/** What an import into an existing agent did. */
export interface IntoResult {
  ok: boolean
  slug: string
  name: string
  mode: MergeMode
  into: true
  /** insights, intentions, playbooks, sessions…: added, and already there. */
  counts: Record<string, { added: number; skipped: number }>
  chat_messages: number
  schedules: number
  connections: Connection[]
  restarted: boolean
  notes: string[]
  /** The backup taken just before — kept a week, to undo. */
  safety_backup: { id: string; file_name: string; size: number } | null
  undone?: boolean
  package?: Package | null
  /** An update: the earlier version's items taken out, and those kept because they were changed here. */
  taken_out?: { gone?: number; kept?: number }
  from_version?: number
}

export interface ImportResult {
  ok: boolean
  slug: string
  name: string
  web_auth: string
  fleet_instructions: string
  chat_messages: number
  schedules: number
  connections: Connection[]
  keys_restored: boolean
  needs_key: boolean
  notes: string[]
}

async function send(path: string, init: RequestInit = {}, asUser?: string): Promise<Response> {
  const headers = (): Record<string, string> => {
    const { token, authEnabled } = useAuthStore.getState()
    const h: Record<string, string> = typeof init.body === 'string' ? { 'Content-Type': 'application/json' } : {}
    if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
    if (asUser) h[ACT_AS_HEADER] = asUser
    return h
  }
  const go = () => fetch(path, { ...init, headers: headers(), credentials: 'include', cache: 'no-store' })
  let res = await go()
  if (res.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    res = await go()
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : body.error || `${res.status}`)
  }
  return res
}

const json = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) })

/** What the agent learned, per category, and how many contacts it keeps — to look over before an export. */
export async function reviewAgent(slug: string, asUser?: string): Promise<AgentReview> {
  return (await send(`/fd/agent-bundle/review/${encodeURIComponent(slug)}`, {}, asUser)).json()
}

/** Make a bundle of one of your agents (it stays on the deck for a day, or until downloaded and deleted). */
export async function exportAgent(slug: string, options: ExportOptions, asUser?: string): Promise<ExportResult> {
  return (await send(`/fd/agent-bundle/export/${encodeURIComponent(slug)}`, json(options), asUser)).json()
}

/** The packages an agent was exported as, with their last version. */
export async function agentPackages(slug: string, asUser?: string): Promise<{ name: string; version: number }[]> {
  return ((await send(`/fd/agent-bundle/packages/${encodeURIComponent(slug)}`, {}, asUser)).json() as
    Promise<{ packages: { name: string; version: number }[] }>).then((r) => r.packages)
}

/** Your model tiers (no keys): what an import can run on. */
export async function myTiers(asUser?: string): Promise<ModelTier[]> {
  return ((await send('/fd/agent-bundle/tiers', {}, asUser)).json() as Promise<{ tiers: ModelTier[] }>).then((r) => r.tiers)
}

/** Save a made bundle to the person's computer. */
export async function downloadExport(made: ExportResult, asUser?: string): Promise<void> {
  const res = await send(`/fd/agent-bundle/exports/${made.id}/download`, {}, asUser)
  saveBlob(await res.blob(), made.file_name)
}

function saveBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob)
  try {
    const a = document.createElement('a')
    a.href = url
    a.download = fileName
    document.body.appendChild(a)
    a.click()
    a.remove()
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 30_000)
  }
}

export async function deleteExport(id: string, asUser?: string): Promise<void> {
  await send(`/fd/agent-bundle/exports/${id}`, { method: 'DELETE' }, asUser)
}

/** Upload a .captain file: what is in it. Nothing is made yet. */
export async function uploadBundle(file: File, asUser?: string): Promise<UploadResult> {
  const form = new FormData()
  form.append('file', file)
  return (await send('/fd/agent-bundle/upload', { method: 'POST', body: form }, asUser)).json()
}

/** Make a new agent of an upload — or, with `target`, put it into one of yours. */
export async function importUpload(uploadId: string, options: ImportOptions, asUser?: string): Promise<ImportResult | IntoResult> {
  return (await send(`/fd/agent-bundle/uploads/${uploadId}/import`, json(options), asUser)).json()
}

/** Undo an import into an agent: it comes back as it was just before. */
export async function undoImport(safetyId: string, asUser?: string): Promise<IntoResult> {
  return (await send(`/fd/agent-bundle/safety/${safetyId}/restore`, { method: 'POST' }, asUser)).json()
}

/** Save the backup taken before an import into an agent. */
export async function downloadSafety(backup: { id: string; file_name: string }, asUser?: string): Promise<void> {
  const res = await send(`/fd/agent-bundle/safety/${backup.id}/download`, {}, asUser)
  saveBlob(await res.blob(), backup.file_name)
}

export async function discardUpload(uploadId: string, asUser?: string): Promise<void> {
  await send(`/fd/agent-bundle/uploads/${uploadId}`, { method: 'DELETE' }, asUser).catch(() => undefined)
}

/** "2.4 MB" */
export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

const COUNT_LABELS: Record<string, [string, string]> = {
  insights: ['insight', 'insights'],
  intentions: ['intention', 'intentions'],
  intuitions: ['intuition', 'intuitions'],
  reflections: ['reflection', 'reflections'],
  playbooks: ['playbook', 'playbooks'],
  contacts: ['contact', 'contacts'],
  sessions: ['session', 'sessions'],
  topics: ['topic', 'topics'],
  chat_messages: ['chat message', 'chat messages'],
  knowledge: ['document file', 'document files'],
  skills: ['skill file', 'skill files'],
  workspace: ['workspace file', 'workspace files'],
  data_tables: ['data table', 'data tables'],
  schedules: ['scheduled task', 'scheduled tasks'],
}

/** The counts a bundle summary leads with, as "12 insights". */
export function countChips(counts: Record<string, number>): string[] {
  return Object.keys(COUNT_LABELS)
    .filter((k) => (counts[k] ?? 0) > 0)
    .map((k) => `${counts[k]} ${counts[k] === 1 ? COUNT_LABELS[k][0] : COUNT_LABELS[k][1]}`)
}
