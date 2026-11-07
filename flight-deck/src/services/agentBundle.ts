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
}

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
}

export interface ImportResult {
  ok: boolean
  slug: string
  name: string
  web_auth: string
  fleet_instructions: string
  chat_messages: number
  schedules: number
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

/** Make a bundle of one of your agents (it stays on the deck for a day, or until downloaded and deleted). */
export async function exportAgent(slug: string, preset = 'full', asUser?: string): Promise<ExportResult> {
  return (await send(`/fd/agent-bundle/export/${encodeURIComponent(slug)}`, json({ preset }), asUser)).json()
}

/** Save a made bundle to the person's computer. */
export async function downloadExport(made: ExportResult, asUser?: string): Promise<void> {
  const res = await send(`/fd/agent-bundle/exports/${made.id}/download`, {}, asUser)
  const url = URL.createObjectURL(await res.blob())
  try {
    const a = document.createElement('a')
    a.href = url
    a.download = made.file_name
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

/** Make a new agent of an upload. */
export async function importUpload(uploadId: string, name: string, asUser?: string): Promise<ImportResult> {
  return (await send(`/fd/agent-bundle/uploads/${uploadId}/import`, json({ name }), asUser)).json()
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
