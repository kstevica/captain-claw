import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bell, X, CheckCheck, Trash2, AlertTriangle, Info, CheckCircle2, AlertCircle, ListChecks, Loader2, Maximize2 } from 'lucide-react'
import { useNotificationStore, type Notification, type NotificationType } from '../../stores/notificationStore'
import { fetchWaitingProposalIds, resolveProposal } from '../../services/notifications'
import { taskboardApi } from '../../services/taskboardApi'
import { useTaskboardStore } from '../../stores/taskboardStore'
import { untrustedUrlTransform } from '../../utils/untrustedContent'
import { useAuthStore, selectKioskLocked } from '../../stores/authStore'
import { useChatStore } from '../../stores/chatStore'
import { useProcessStore } from '../../stores/processStore'
import { useSharedAgentStore } from '../../stores/sharedAgentStore'
import { useUIStore } from '../../stores/uiStore'
import { AgentSetupDialog } from '../layout/AgentSetupDialog'
import { OnboardingDialog } from '../layout/OnboardingDialog'
import { useIsMobile } from '../../hooks/useMediaQuery'
import {
  SHARED_SECTION_ID, sharedNotificationAction, type SharedNotificationAction,
} from '../../utils/sharedAgent'

// What a bell row opens of an agent: its Way of working, or its onboarding's exams or lessons.
type SetupKind = 'setup' | 'exam' | 'lessons'

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

// Autonomous Work's text is the agent's markdown, and may quote an email:
// rendered, but nothing in it is fetched (no images) and links stay text.
const BELL_MD: Components = {
  a: ({ children }) => <span className="underline underline-offset-2">{children}</span>,
  img: () => null,
}
// Its single line breaks are kept (a proposal's title, then its why).
const keepBreaks = (text: string) => text.replace(/([^\n])\n(?!\n)/g, '$1  \n')
const COMPACT_MD = '[&_p]:my-0.5 [&_strong]:font-semibold [&_strong]:text-zinc-300 [&_ul]:list-disc [&_ul]:pl-4 '
  + '[&_ol]:list-decimal [&_ol]:pl-4 [&_h1]:font-semibold [&_h2]:font-semibold [&_h3]:font-semibold'

// `align` picks which edge of the bell the dropdown hangs from: 'right' (the
// default — the bell sits at the right end of the top bar) or 'left' (the
// bell sits in a left-hand column, so the panel must open towards the page).
export function NotificationBell({ align = 'right' }: { align?: 'left' | 'right' } = {}) {
  const { unreadCount } = useNotificationStore()
  const [open, setOpen] = useState(false)
  // The same list, big: a dialog that reads long nudges and proposals whole.
  const [big, setBig] = useState(false)
  // An agent whose way-of-working read is done (or stopped): its setup dialog, opened from here.
  const [setupFor, setSetupFor] = useState<{ slug: string; name: string; readDocs?: string[] } | null>(null)
  // An agent whose onboarding exam ended, or a member suggested a lesson for: its onboarding, on that tab.
  const [examFor, setExamFor] = useState<{ slug: string; name: string; tab: 'exams' | 'lessons' } | null>(null)
  const { isMobile: phone } = useIsMobile()
  const openSetup = (slug: string, kind: SetupKind = 'setup') => {
    setOpen(false)
    setBig(false)
    const proc = useProcessStore.getState().processes.find((p) => p.slug === slug)
    if (kind === 'setup') setSetupFor({ slug, name: proc?.name || slug })
    else setExamFor({ slug, name: proc?.name || slug, tab: kind === 'lessons' ? 'lessons' : 'exams' })
  }

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
          <div className={`fixed left-2 right-2 top-14 z-50 sm:absolute sm:top-full sm:mt-1 sm:w-[360px] rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl ${align === 'left' ? 'sm:right-auto sm:left-0' : 'sm:left-auto sm:right-0'}`}>
            <NotificationDropdown onClose={() => setOpen(false)} onEnlarge={() => { setOpen(false); setBig(true) }} onOpenSetup={openSetup} />
          </div>
        </>
      )}
      {big && <NotificationDialog onClose={() => setBig(false)} onOpenSetup={openSetup} />}
      {setupFor && <AgentSetupDialog agent={setupFor} onClose={() => setSetupFor(null)} />}
      {examFor && (
        <OnboardingDialog agent={examFor} initialTab={examFor.tab} phone={phone} onClose={() => setExamFor(null)}
          onSetup={(readDocs) => { setSetupFor({ ...examFor, readDocs }); setExamFor(null) }} />
      )}
    </div>
  )
}

// The bell's list in a dialog: wide, tall, every text whole.
function NotificationDialog({ onClose, onOpenSetup }: {
  onClose: () => void; onOpenSetup: (slug: string, kind?: SetupKind) => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 sm:items-center sm:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label="Notifications"
        className="flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 shadow-2xl sm:h-[85vh] sm:max-w-3xl sm:rounded-xl sm:border sm:border-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        <NotificationDropdown large onClose={onClose} onOpenSetup={onOpenSetup} />
      </div>
    </div>,
    document.body,
  )
}

// Taskboard: a link to a task in a notice's text, and the notice's own last
// line that is only that link ("[Open in Taskboard](?taskboard=…)") — shown
// as the "Open task" button instead where the board can open.
const TASK_LINK = /[?&]taskboard=([A-Za-z0-9_.:-]{1,200})/
const TASK_LINK_LINE = /\n?\[[^\]\n]*\]\([^)\s]*[?&]taskboard=[^)\s]*\)\s*$/

// The Agent Desktop may still be mounting after setView('desktop'): look for
// the "Shared with me" section for a moment before giving up.
function revealSharedSection(tries = 20): void {
  const el = document.getElementById(SHARED_SECTION_ID)
  if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); return }
  if (tries > 0) setTimeout(() => revealSharedSection(tries - 1), 50)
}

function NotificationDropdown({ onClose, onEnlarge, onOpenSetup, large = false }: {
  onClose: () => void
  onEnlarge?: () => void
  /** A notice that an agent's way-of-working read ended opens its setup. */
  onOpenSetup?: (slug: string, kind?: SetupKind) => void
  /** In the big dialog: larger text, every text whole, a close button. */
  large?: boolean
}) {
  const { notifications, markRead, markAllRead, remove, clear } = useNotificationStore()
  const [filter, setFilter] = useState<NotificationType | 'all'>('all')
  // A notice about an agent someone shared with you leads to it. The list is
  // only polled where agents are shown, so refresh it when the bell opens.
  const sharedAgents = useSharedAgentStore((s) => s.agents)
  const kioskLocked = useAuthStore(selectKioskLocked)
  const layoutMode = useUIStore((s) => s.layoutMode)
  const layout = kioskLocked || layoutMode === 'simple' ? 'simple' : 'full'
  // Where the Taskboard can open (the kiosk, the simple layout on a desktop):
  // a proposal or a notice about a task leads to it there ("Open task").
  const { isMobile, isTablet } = useIsMobile()
  const boardHere = kioskLocked || (layoutMode === 'simple' && !isMobile && !isTablet)
  const aboutAgents = notifications.some((n) => n.refType === 'agent')
  useEffect(() => {
    if (aboutAgents) void useSharedAgentStore.getState().fetch()
  }, [aboutAgents])
  // Autonomous Work proposals: which still wait for an answer (those get
  // Approve / Reject here — the kiosk has no other place to answer them).
  // Asked again when a new one comes in, so it gets its buttons.
  const proposalIds = notifications.filter((n) => n.refType === 'autonomy_action').map((n) => n.refId).join(',')
  const [waiting, setWaiting] = useState<Set<string> | null>(null)
  // Taskboard: the task each waiting proposal is on (a board proposal's own
  // task, else the one the board keeps for the proposal).
  const [proposalTasks, setProposalTasks] = useState<Map<string, string> | null>(null)
  const loadWaiting = () => {
    fetchWaitingProposalIds().then(setWaiting).catch(() => setWaiting(null))
  }
  useEffect(() => {
    if (proposalIds) loadWaiting()
  }, [proposalIds])
  useEffect(() => {
    if (!proposalIds || !boardHere) return
    let gone = false
    taskboardApi.waitingProposals()
      .then((m) => { if (!gone) setProposalTasks(m) })
      .catch(() => { if (!gone) setProposalTasks(null) })
    return () => { gone = true }
  }, [proposalIds, boardHere])
  // The task a row is about — the one its text links to (`?taskboard=<task
  // id>`, the deck's own "Open in Taskboard"), else a waiting proposal's —
  // opened on the board.
  const taskOf = (n: Notification): string => {
    if (!boardHere) return ''
    const linked = TASK_LINK.exec(n.message || '')?.[1]
    return linked || (n.refType === 'autonomy_action' && n.refId ? proposalTasks?.get(n.refId) : '') || ''
  }
  const openTask = (ref: string) => {
    onClose()
    useTaskboardStore.getState().showTask(ref)
  }

  const activate = (action: SharedNotificationAction) => {
    if (action.kind === 'none') return
    onClose()
    if (action.kind === 'chat') {
      useChatStore.getState().openSharedChat(action.agentRef, action.name, action.ownerName)
    } else if (layout === 'full') {
      useUIStore.getState().setView('desktop')
      revealSharedSection()
    } else {
      useUIStore.getState().setSimpleLeftOpen(true)
    }
  }

  const filtered = filter === 'all' ? notifications : notifications.filter((n) => n.type === filter)
  // What still waits for Approve / Reject goes on top; the rest keep their order.
  const waits = (n: Notification) => n.refType === 'autonomy_action' && !!n.refId && !!waiting?.has(n.refId)
  const shown = [...filtered.filter(waits), ...filtered.filter((n) => !waits(n))]

  return (
    <div className={`flex flex-col ${large ? 'h-full' : 'max-h-[480px]'}`}>
      <div className={`flex items-center justify-between border-b border-zinc-800 ${large ? 'px-5 py-3' : 'px-3 py-2'}`}>
        <span className={`${large ? 'text-base' : 'text-xs'} font-semibold`}>Notifications</span>
        <div className="flex items-center gap-1">
          {onEnlarge && (
            <button onClick={onEnlarge} className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:text-zinc-300" title="Show bigger" aria-label="Show bigger">
              <Maximize2 className="h-3 w-3" />
            </button>
          )}
          <button onClick={markAllRead} className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:text-zinc-300" title="Mark all read">
            <CheckCheck className="h-3 w-3" />
          </button>
          {notifications.length > 0 && (
            <button onClick={clear} className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:text-red-400" title="Clear all">
              <Trash2 className="h-3 w-3" />
            </button>
          )}
          {large && (
            <button onClick={onClose} className="ml-1 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Close" aria-label="Close">
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Type filter */}
      <div className={`flex gap-0.5 border-b border-zinc-800 ${large ? 'px-5 py-2' : 'px-3 py-1.5'}`}>
        {(['all', 'info', 'success', 'warning', 'error'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setFilter(t)}
            className={`rounded px-1.5 py-0.5 ${large ? 'text-xs' : 'text-[10px]'} font-medium capitalize ${
              filter === t ? 'bg-violet-600/20 text-violet-400' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto">
        {shown.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-zinc-600">No notifications</div>
        ) : (
          shown.map((n) => {
            const setup = (n.refType === 'agent_setup' || n.refType === 'agent_exam' || n.refType === 'agent_lessons')
              && n.refId && onOpenSetup ? n.refId : ''
            const task = taskOf(n)
            const exam = n.refType === 'agent_exam'
            const lessons = n.refType === 'agent_lessons'
            return (
            <NotificationItem
              key={n.id}
              notification={n}
              large={large}
              action={setup ? { kind: 'none' } : sharedNotificationAction(n.refType, n.refId, sharedAgents, layout)}
              hint={setup ? (exam ? 'Click to see the exam.' : lessons ? 'Click to take it, or decline.'
                : n.type === 'error' ? 'Click to set it up again.' : 'Click to review it.') : ''}
              proposal={n.refType === 'autonomy_action' && n.refId && waiting
                ? (waiting.has(n.refId) ? 'waiting' : 'answered') : undefined}
              onAnswered={() => { markRead(n.id); loadWaiting() }}
              onOpenTask={task ? () => { markRead(n.id); openTask(task) } : undefined}
              onRead={() => markRead(n.id)}
              onActivate={setup ? () => onOpenSetup?.(setup, exam ? 'exam' : lessons ? 'lessons' : 'setup') : activate}
              onRemove={() => remove(n.id)}
            />
            )
          })
        )}
      </div>
    </div>
  )
}

// A proposal's answer, from the bell: Approve sends it to your agent, Reject
// drops it (and teaches Autonomous Work not to suggest its like).
function ProposalButtons({ id, large, onAnswered }: { id: string; large: boolean; onAnswered: () => void }) {
  const [busy, setBusy] = useState<'approve' | 'reject' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const answer = async (approve: boolean) => {
    setBusy(approve ? 'approve' : 'reject'); setError(null)
    try {
      await resolveProposal(id, approve)
      onAnswered()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  const btn = `flex items-center gap-1 rounded font-medium disabled:opacity-50 ${large ? 'px-3 py-1 text-xs' : 'px-2 py-0.5 text-[10px]'}`
  return (
    <div className="mt-1" onClick={(e) => e.stopPropagation()}>
      <div className="flex items-center gap-1.5">
        <button disabled={busy !== null} onClick={() => answer(true)}
          className={`${btn} bg-emerald-600 text-white hover:bg-emerald-500`}>
          {busy === 'approve' && <Loader2 className="h-3 w-3 animate-spin" />} Approve
        </button>
        <button disabled={busy !== null} onClick={() => answer(false)}
          className={`${btn} bg-zinc-800 text-zinc-300 hover:bg-zinc-700`}>
          {busy === 'reject' && <Loader2 className="h-3 w-3 animate-spin" />} Reject
        </button>
      </div>
      {error && <p className="mt-0.5 text-[10px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

function NotificationItem({ notification: n, large, action, hint, proposal, onAnswered, onOpenTask, onRead, onActivate, onRemove }: {
  notification: Notification
  large: boolean
  action: SharedNotificationAction
  /** What a click does, when it isn't the shared-agent action's. */
  hint?: string
  proposal?: 'waiting' | 'answered'
  onAnswered: () => void
  /** Taskboard: the task this is about, on the board. */
  onOpenTask?: () => void
  onRead: () => void
  onActivate: (action: SharedNotificationAction) => void
  onRemove: () => void
}) {
  const Icon = typeIcon[n.type]
  const color = typeColor[n.type]
  // Autonomous Work's text (a nudge, a proposal's why) is markdown; a long
  // one opens up on a click (in the big dialog it's whole).
  const autonomy = n.refType === 'autonomy_action' || n.refType === 'autonomy_nudge'
  const [expanded, setExpanded] = useState(false)
  // Its "Open in Taskboard" link line is the button below, where there is one
  // — and where the board doesn't open here (the full layout, a phone's), no
  // dead link either.
  const message = n.message.replace(TASK_LINK_LINE, '')

  return (
    <div
      className={`group flex items-start gap-2 border-b border-zinc-800/40 hover:bg-zinc-900/30 cursor-pointer ${large ? 'px-5 py-3' : 'px-3 py-2'} ${
        !n.read ? 'bg-zinc-900/20' : ''
      }`}
      onClick={() => { onRead(); onActivate(action); if (autonomy) setExpanded((v) => !v) }}
      title={action.kind === 'chat' ? `Chat with ${action.name}` : undefined}
    >
      <Icon className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${color}`} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className={`${large ? 'text-sm' : 'text-xs'} font-medium text-zinc-200`}>{n.title}</span>
          {!n.read && <span className="h-1.5 w-1.5 rounded-full bg-violet-500" />}
        </div>
        {autonomy && large ? (
          <div className="fd-file-markdown mt-1 [&_ol]:list-decimal [&_ul]:list-disc">
            <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={BELL_MD}>{keepBreaks(message)}</Markdown>
          </div>
        ) : autonomy ? (
          <div className={`mt-0.5 text-[10px] text-zinc-500 ${COMPACT_MD} ${expanded ? '' : 'line-clamp-6'}`}>
            <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={BELL_MD}>{keepBreaks(message)}</Markdown>
          </div>
        ) : (
          <p className={`${large ? 'text-sm' : 'text-[10px]'} text-zinc-500 mt-0.5`}>{message}</p>
        )}
        {proposal === 'waiting' && n.refId && <ProposalButtons id={n.refId} large={large} onAnswered={onAnswered} />}
        {proposal === 'answered' && <p className={`mt-0.5 ${large ? 'text-xs' : 'text-[10px]'} text-zinc-600`}>No longer waiting.</p>}
        {onOpenTask && (
          <button
            onClick={(e) => { e.stopPropagation(); onOpenTask() }}
            className={`mt-1 flex items-center gap-1 font-medium text-teal-600 hover:underline dark:text-teal-400 ${large ? 'text-xs' : 'text-[10px]'}`}
            title="This task on the Taskboard — its whole story, and the next step to edit before you approve"
          >
            <ListChecks className="h-3 w-3" />Open task
          </button>
        )}
        {(hint || action.kind !== 'none') && (
          <p className="mt-0.5 text-[10px] text-sky-700 dark:text-sky-400">{hint || (action.kind !== 'none' ? action.hint : '')}</p>
        )}
        <div className={`flex items-center gap-1.5 mt-0.5 ${large ? 'text-xs' : 'text-[10px]'} text-zinc-600`}>
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
