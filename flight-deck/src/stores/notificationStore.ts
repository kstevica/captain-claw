import { create } from 'zustand'
import {
  fetchNotifications, markNotificationRead, markAllNotificationsRead,
} from '../services/notifications'

export type NotificationType = 'info' | 'success' | 'warning' | 'error'

// A server notification id is a uuid; a local (toast-only) one is prefixed.
const isServerId = (id: string) => !id.startsWith('notif-')

function mapServerType(t: string): NotificationType {
  if (t === 'run_error') return 'error'
  if (t === 'run') return 'success'
  if (t === 'share' || t === 'info') return 'info'
  if (t === 'success' || t === 'warning' || t === 'error') return t
  return 'info'
}

export interface Notification {
  id: string
  type: NotificationType
  title: string
  message: string
  agentId?: string
  agentName?: string
  createdAt: string
  read: boolean
}

interface NotificationStore {
  notifications: Notification[]
  unreadCount: number
  add: (type: NotificationType, title: string, message: string, agentId?: string, agentName?: string) => void
  markRead: (id: string) => void
  markAllRead: () => void
  remove: (id: string) => void
  clear: () => void
  // Merge server-backed notifications (shares, finished runs) into the store.
  hydrateFromServer: () => Promise<void>
}

export const useNotificationStore = create<NotificationStore>((set, get) => ({
  notifications: [],
  unreadCount: 0,

  add: (type, title, message, agentId, agentName) => {
    const id = `notif-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const notif: Notification = { id, type, title, message, agentId, agentName, createdAt: new Date().toISOString(), read: false }
    const notifications = [notif, ...get().notifications].slice(0, 200)
    set({ notifications, unreadCount: get().unreadCount + 1 })
  },

  markRead: (id) => {
    const notifications = get().notifications.map((n) => n.id === id ? { ...n, read: true } : n)
    const unreadCount = notifications.filter((n) => !n.read).length
    set({ notifications, unreadCount })
    if (isServerId(id)) void markNotificationRead(id)  // persist
  },

  markAllRead: () => {
    const notifications = get().notifications.map((n) => ({ ...n, read: true }))
    set({ notifications, unreadCount: 0 })
    void markAllNotificationsRead()  // persist
  },

  remove: (id) => {
    const notifications = get().notifications.filter((n) => n.id !== id)
    const unreadCount = notifications.filter((n) => !n.read).length
    set({ notifications, unreadCount })
    if (isServerId(id)) void markNotificationRead(id)  // hide by marking read
  },

  clear: () => {
    set({ notifications: [], unreadCount: 0 })
    void markAllNotificationsRead()
  },

  hydrateFromServer: async () => {
    const { items } = await fetchNotifications()
    const mapped: Notification[] = items.map((s) => ({
      id: s.id,
      type: mapServerType(s.type),
      title: s.title,
      message: s.body || '',
      createdAt: s.created_at,
      read: !!s.read,
    }))
    // Keep local toasts (ephemeral), replace the server-sourced set.
    const localToasts = get().notifications.filter((n) => !isServerId(n.id))
    const merged = [...mapped, ...localToasts]
      .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
      .slice(0, 200)
    set({ notifications: merged, unreadCount: merged.filter((n) => !n.read).length })
  },
}))
