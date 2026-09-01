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
