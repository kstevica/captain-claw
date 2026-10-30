// Persistent in-app notifications (the bell), backed by /fd/notifications.

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

export interface ServerNotification {
  id: string
  type: string
  title: string
  body: string
  ref_type: string
  ref_id: string
  read: number
  created_at: string
}

function authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

async function fdFetch(path: string, init?: RequestInit): Promise<Response> {
  let r = await fetch(path, { ...init, headers: { ...authHeaders(), ...(init?.headers || {}) }, credentials: 'include' })
  if (r.status === 401 && useAuthStore.getState().authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) r = await fetch(path, { ...init, headers: { ...authHeaders(), ...(init?.headers || {}) }, credentials: 'include' })
  }
  return r
}

export async function fetchNotifications(): Promise<{ items: ServerNotification[]; unread: number }> {
  const r = await fdFetch('/fd/notifications?limit=50')
  if (!r.ok) return { items: [], unread: 0 }
  const data = await r.json()
  return { items: Array.isArray(data?.notifications) ? data.notifications : [], unread: data?.unread || 0 }
}

export async function markNotificationRead(id: string): Promise<void> {
  try { await fdFetch(`/fd/notifications/${encodeURIComponent(id)}/read`, { method: 'POST' }) } catch { /* ignore */ }
}

export async function markAllNotificationsRead(): Promise<void> {
  try { await fdFetch('/fd/notifications/read-all', { method: 'POST' }) } catch { /* ignore */ }
}

// ── Autonomous Work proposals in the bell (ref_type 'autonomy_action') ──

export interface WaitingProposal {
  /** A Taskboard next step: its wording is a model's, read before it is approved. */
  taskboard: boolean
  text: string
  to: string
}

/** This user's proposals still waiting for approval, by id. */
export async function fetchWaitingProposals(): Promise<Map<string, WaitingProposal>> {
  const r = await fdFetch('/fd/autonomy/actions?status_filter=awaiting_approval&limit=200')
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  const data = await r.json()
  type Row = { id?: string; source?: string; payload?: { text?: string; to?: string; to_options?: { address?: string; display?: string }[] } }
  const actions: Row[] = Array.isArray(data?.actions) ? data.actions : []
  return new Map(actions.filter((a) => a.id).map((a) => {
    const p = a.payload ?? {}
    const to = p.to_options?.find((o) => o.address === p.to)?.display || p.to || ''
    return [String(a.id), { taskboard: a.source === 'taskboard', text: String(p.text || ''), to }]
  }))
}

/** Approve or reject a waiting proposal; throws with the deck's reason. */
export async function resolveProposal(id: string, approve: boolean): Promise<void> {
  const r = await fdFetch(`/fd/autonomy/actions/${encodeURIComponent(id)}/${approve ? 'approve' : 'reject'}`,
    { method: 'POST' })
  if (!r.ok) {
    let detail = `HTTP ${r.status}`
    try {
      const d = await r.json()
      if (typeof d?.detail === 'string') detail = d.detail
    } catch { /* keep the status */ }
    throw new Error(detail)
  }
}
