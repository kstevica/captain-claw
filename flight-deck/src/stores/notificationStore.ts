import { create } from 'zustand'

export type NotificationType = 'info' | 'success' | 'warning' | 'error'

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
  },

  markAllRead: () => {
    const notifications = get().notifications.map((n) => ({ ...n, read: true }))
    set({ notifications, unreadCount: 0 })
  },

  remove: (id) => {
    const notifications = get().notifications.filter((n) => n.id !== id)
    const unreadCount = notifications.filter((n) => !n.read).length
    set({ notifications, unreadCount })
  },

  clear: () => set({ notifications: [], unreadCount: 0 }),
}))
