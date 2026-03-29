import { useState } from 'react'
import { Bell, X, CheckCheck, Trash2, AlertTriangle, Info, CheckCircle2, AlertCircle } from 'lucide-react'
import { useNotificationStore, type Notification, type NotificationType } from '../../stores/notificationStore'

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const s = Math.floor(diff / 1000)
  if (s < 60) return 'just now'
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

const typeIcon: Record<NotificationType, typeof Info> = {
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  error: AlertCircle,
}
const typeColor: Record<NotificationType, string> = {
  info: 'text-blue-400',
  success: 'text-emerald-400',
  warning: 'text-amber-400',
  error: 'text-red-400',
}

export function NotificationBell() {
  const { unreadCount } = useNotificationStore()
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className={`relative rounded p-1.5 transition-colors ${
          open ? 'bg-zinc-800 text-zinc-200' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
        }`}
        title="Notifications"
      >
        <Bell className="h-4 w-4" />
        {unreadCount > 0 && (
          <span className="absolute -right-0.5 -top-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-[9px] font-bold text-white">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-full z-50 mt-1 w-[360px] rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl">
            <NotificationDropdown onClose={() => setOpen(false)} />
          </div>
        </>
      )}
    </div>
  )
}

function NotificationDropdown({ onClose: _onClose }: { onClose: () => void }) {
  const { notifications, markRead, markAllRead, remove, clear } = useNotificationStore()
  const [filter, setFilter] = useState<NotificationType | 'all'>('all')

  const filtered = filter === 'all' ? notifications : notifications.filter((n) => n.type === filter)

  return (
    <div className="flex max-h-[480px] flex-col">
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
        <span className="text-xs font-semibold">Notifications</span>
        <div className="flex items-center gap-1">
          <button onClick={markAllRead} className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:text-zinc-300" title="Mark all read">
            <CheckCheck className="h-3 w-3" />
          </button>
          {notifications.length > 0 && (
            <button onClick={clear} className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:text-red-400" title="Clear all">
              <Trash2 className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>

      {/* Type filter */}
      <div className="flex gap-0.5 border-b border-zinc-800 px-3 py-1.5">
        {(['all', 'info', 'success', 'warning', 'error'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setFilter(t)}
            className={`rounded px-1.5 py-0.5 text-[10px] font-medium capitalize ${
              filter === t ? 'bg-violet-600/20 text-violet-400' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-zinc-600">No notifications</div>
        ) : (
          filtered.map((n) => (
            <NotificationItem key={n.id} notification={n} onRead={() => markRead(n.id)} onRemove={() => remove(n.id)} />
          ))
        )}
      </div>
    </div>
  )
}

function NotificationItem({ notification: n, onRead, onRemove }: { notification: Notification; onRead: () => void; onRemove: () => void }) {
  const Icon = typeIcon[n.type]
  const color = typeColor[n.type]

  return (
    <div
      className={`group flex items-start gap-2 border-b border-zinc-800/40 px-3 py-2 hover:bg-zinc-900/30 cursor-pointer ${
        !n.read ? 'bg-zinc-900/20' : ''
      }`}
      onClick={onRead}
    >
      <Icon className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${color}`} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-medium text-zinc-200">{n.title}</span>
          {!n.read && <span className="h-1.5 w-1.5 rounded-full bg-violet-500" />}
        </div>
        <p className="text-[10px] text-zinc-500 mt-0.5">{n.message}</p>
        <div className="flex items-center gap-1.5 mt-0.5 text-[10px] text-zinc-600">
          {n.agentName && <span>{n.agentName}</span>}
          <span>{relativeTime(n.createdAt)}</span>
        </div>
      </div>
      <button
        onClick={(e) => { e.stopPropagation(); onRemove() }}
        className="rounded p-0.5 text-zinc-600 opacity-0 group-hover:opacity-100 hover:text-red-400 transition-opacity"
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  )
}
