// Frontend client for per-app file storage.
// Pairs with captain_claw/flight_deck/app_files_routes.py.

import { useAuthStore } from '../stores/authStore'

export interface FileMeta {
  file_id: string
  filename: string
  mime: string
  size: number
  uploaded_by?: string
  uploaded_at: string
}

function token(): string | null {
  return useAuthStore.getState().token ?? null
}

function authHeader(): Record<string, string> {
  const t = token()
  return t ? { Authorization: `Bearer ${t}` } : {}
}

export async function uploadFile(agentId: string, file: File): Promise<FileMeta> {
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch(`/fd/apps/${encodeURIComponent(agentId)}/files`, {
    method: 'POST',
    credentials: 'include',
    headers: { ...authHeader() },   // do NOT set Content-Type — let browser pick the multipart boundary
    body: form,
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`${resp.status}: ${text || resp.statusText}`)
  }
  const data = await resp.json()
  return data.file as FileMeta
}

export async function listFiles(agentId: string): Promise<FileMeta[]> {
  const resp = await fetch(`/fd/apps/${encodeURIComponent(agentId)}/files`, {
    credentials: 'include',
    headers: { ...authHeader() },
  })
  if (!resp.ok) return []
  const data = await resp.json()
  return (data.files ?? []) as FileMeta[]
}

export async function deleteFile(agentId: string, fileId: string): Promise<boolean> {
  const resp = await fetch(`/fd/apps/${encodeURIComponent(agentId)}/files/${encodeURIComponent(fileId)}`, {
    method: 'DELETE',
    credentials: 'include',
    headers: { ...authHeader() },
  })
  return resp.ok
}

// Returns a URL the browser can hit directly (e.g. `<img src>`). The token
// is added as a query param because the standard auth header can't ride
// along on a plain element src.
export function fileUrl(agentId: string, fileId: string): string {
  const t = token()
  const base = `/fd/apps/${encodeURIComponent(agentId)}/files/${encodeURIComponent(fileId)}`
  return t ? `${base}?fd_token=${encodeURIComponent(t)}` : base
}
