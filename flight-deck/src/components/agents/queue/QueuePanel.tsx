import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { ChevronLeft, ListTodo, MoreHorizontal, PanelLeftClose, Pause, Play, Trash2, UploadCloud, Wand2, Eraser, EyeOff } from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import { isQueueCommand, useChatStore, LANE_MAIN, LANE_AUTO, type QueuedMessage } from '../../../stores/chatStore'
import { useQuotaStore } from '../../../stores/quotaStore'
import { useFileAttachments, type AgentConn } from '../../../hooks/useFileAttachments'
import { touchKeyboard } from '../../../hooks/useMediaQuery'
import { sharedCloseInfo } from '../../../utils/sharedAgent'
import { Pill } from '../../console/parts'
import { QueuePlannerModal } from '../QueuePlannerModal'
import {
  applyWaitingOrder, deriveQueueModel, isQuiet, joinAttachmentRefs, moveTaskOrder, parseAttachmentRefs, queueSettleMs,
  quietKeyOf, totalsText, type HeroAction, type QueueModel,
} from './queueModel'
import { FOCUS } from './QueueButton'
import { QueueHero } from './QueueHero'
import { DoneSection, UpNextSection } from './QueueList'
import { QueueComposer } from './QueueComposer'
import { QueueMenu, type MenuEntry } from './QueueMenu'
import { UndoToast } from './QueueToast'
import { quietSinceOf, usePassed, useUndoToast, type ToastAction } from './queueHooks'

// The chat's message queue: tasks the agent works through one at a time. One
// hero card on top says what is happening and what the user can do next; the
// list below holds only what waits (Up next) and what finished (Done); the
// composer adds more. The same panel serves the simple layout's column, its
// phone view, and the full layout's fullscreen sidebar.

export type QueueVariant = 'column' | 'phone' | 'sidebar'

/** An inline edit in progress. Kept here, not in the row, so it follows the
 *  item from Up next into the hero if it starts while being edited. */
export interface EditState {
  id: string
  /** 'edit' changes a waiting task; 'retry' saves and runs the item again. */
  mode: 'edit' | 'retry'
  saveLabel: string
  text: string
  refs: string[]
}

export interface QueueActs {
  enqueue: (content: string) => void
  markDone: (id: string) => void
  skipRunning: (id: string) => void
  skipWaiting: (id: string) => void
  removeWaiting: (id: string, message?: string) => void
  removeRunning: (id: string) => void
  removeDone: (id: string) => void
  retry: (id: string, content?: string) => void
  startEdit: (item: QueuedMessage, mode: 'edit' | 'retry', saveLabel: string) => void
  saveEdit: (e: EditState) => void
  move: (id: string, beforeId: string | null) => void
  runNext: () => void
  stop: () => void
  setPaused: (paused: boolean) => void
  setAuto: (on: boolean) => void
  clearFinished: () => void
  clearWaiting: () => void
  /** Hand a waiting task to the running turn now, as an extra instruction
   *  (it leaves the queue); with no turn on, it runs next on its own. */
  insertNow: (id: string) => void
}

/** What every part of the panel reads: the model, the UI state of items,
 *  and the actions bound to this lane. */
export interface QueueCtx {
  sessionKey: string
  agent: string
  shared: boolean
  variant: QueueVariant
  phone: boolean
  compact: boolean
  /** The panel's height, for caps that scale with it. */
  panelHeight: number
  conn: AgentConn | null
  model: QueueModel
  auto: boolean
  paused: boolean
  busy: boolean
  /** Stop was pressed and the agent hasn't ended that turn yet. */
  stopping: boolean
  connected: boolean
  editing: EditState | null
  setEditing: (e: EditState | null) => void
  menuFor: string | null
  menuAnchor: HTMLElement | null
  openMenu: (id: string, anchor: HTMLElement) => void
  closeMenu: () => void
  expanded: Set<string>
  toggleExpanded: (id: string) => void
  /** Items just saved from the editor ("Edited" for 2s). */
  edited: Set<string>
  act: QueueActs
  toast: (message: string, action?: ToastAction) => void
  registerRow: (id: string, el: HTMLElement | null) => void
  focusRow: (id: string | null) => void
  onGoToChat: (o: { messageId?: string; focusComposer?: boolean }) => void
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)

/** "now" to the tick, ticking every `ms` — only while `on`, so an idle
 *  queue costs nothing. A clock read as an external store: fresh the moment
 *  it turns on, never a stale time from the last run. */
function useNow(ms: number, on: boolean): number {
  const subscribe = useCallback((tick: () => void) => {
    if (!on) return () => {}
    const t = setInterval(tick, ms)
    return () => clearInterval(t)
  }, [ms, on])
  const read = useCallback(() => (on ? Math.floor(Date.now() / ms) * ms : 0), [ms, on])
  return useSyncExternalStore(subscribe, read, read)
}

// Shared chats take no files: a drag is accepted and dropped on the floor.
function swallowDrop(e: React.DragEvent) {
  e.preventDefault()
  e.stopPropagation()
}

const isFileDrag = (e: React.DragEvent) => {
  const types = Array.from(e.dataTransfer.types || [])
  return types.includes('Files') && !types.includes('application/x-fd-queue-item')
}

export function QueuePanel({ sessionKey, agentId, shared, conn, variant, onGoToChat, onHide, onBack, onCollapse }: {
  sessionKey: string
  agentId: string
  shared: boolean
  conn: AgentConn | null
  variant: QueueVariant
  onGoToChat: (o: { messageId?: string; focusComposer?: boolean }) => void
  /** column: tuck the queue away. */
  onHide?: () => void
  /** phone: back to the chat. */
  onBack?: () => void
  /** sidebar: collapse to the rail. */
  onCollapse?: () => void
}) {
  const s = useChatStore(useShallow((st) => {
    const x = st.sessions.get(sessionKey)
    return x ? {
      queue: x.queue, auto: x.queueAutoMode, dispatchedId: x.queueDispatchedId,
      paused: !!x.queuePaused, parked: !!x.queueParked, wanted: !!x.queueStartWhenFree,
      forceThrough: !!x.queueForceThrough,
      busy: x.busy, stopping: !!x.stopping, connected: x.connected, unreachable: !!x.unreachable,
      closed: x.closed ?? null, agent: x.containerName, lane: x.lane, sharedInfo: x.shared ?? null,
    } : null
  }))
  const store = useChatStore(useShallow((st) => ({
    enqueue: st.enqueueQueueMessage,
    removeItem: st.removeQueueItem,
    markItemDone: st.markQueueItemDone,
    clearQueue: st.clearQueue,
    clearFinished: st.clearQueueFinished,
    runQueueNext: st.runQueueNext,
    setQueuePaused: st.setQueuePaused,
    setQueueAutoMode: st.setQueueAutoMode,
    retryQueueItem: st.retryQueueItem,
    editQueueItem: st.editQueueItem,
    moveQueueItem: st.moveQueueItem,
    restoreQueueItems: st.restoreQueueItems,
    extendQueueAnswerWait: st.extendQueueAnswerWait,
    cancelTask: st.cancelTask,
    insertNow: st.insertQueueItemNow,
    openSharedChat: st.openSharedChat,
  })))
  const quotaSpent = useQuotaStore((q) => !!q.status?.blocked && !q.status.exempt)
  const resetsAt = useQuotaStore((q) => q.status?.resets_at ?? null)
  const attachments = useFileAttachments(conn)
  const { toast, show: showToast, dismiss: dismissToast, hold: holdToast, release: releaseToast } = useUndoToast()

  const D = s?.dispatchedId ? s.queue.find((q) => q.id === s.dispatchedId) ?? null : null
  // The clock ticks only for what shows time: the working figure and the
  // answer countdown — not for a task held, given up on, or gone quiet.
  const now = useNow(1000, !!D && !D.halted && !D.stuck && (!!s?.busy || !!D.awaitingAnswer))

  // When the running item went quiet: its reply, else the moment the queue
  // saw the agent idle (the same record the toggle's badge reads). Drives
  // "Wrapping up" → "Check it" / "No reply".
  const dId = D?.id
  const quiet = !!D && !!s && isQuiet(D, s.busy, s.connected)
  const idleSince = quietSinceOf(sessionKey, quiet && D ? quietKeyOf(D) : null, D?.repliedAt)
  // …and once it has been quiet for its settle time: one timer, not a clock.
  const settleMs = queueSettleMs(D, !!s?.auto)
  const settled = usePassed(idleSince === null ? 0 : idleSince + settleMs) && quiet

  const closedInfo = useMemo(() => {
    if (!s?.closed || !s.sharedInfo) return null
    const info = sharedCloseInfo(s.closed.code, s.sharedInfo.ownerName, s.closed.reason)
    return { message: info.message, canRetry: info.retry !== 'none' }
  }, [s?.closed, s?.sharedInfo])

  const hasConn = !!conn
  const model = useMemo(() => deriveQueueModel({
    queue: s?.queue ?? [], auto: !!s?.auto, dispatchedId: s?.dispatchedId ?? null,
    paused: !!s?.paused, parked: !!s?.parked, wanted: !!s?.wanted, forceThrough: !!s?.forceThrough,
    busy: !!s?.busy, stopping: !!s?.stopping,
    connected: !!s?.connected, unreachable: !!s?.unreachable, closed: closedInfo,
    agent: s?.agent || 'The agent', shared, quotaSpent, resetsAt, hasConn, now,
    idleSince, settled,
  }), [s, closedInfo, shared, quotaSpent, resetsAt, hasConn, now, idleSince, settled])

  // ── Layout: the panel's own height decides compact mode and the composer cap.
  const rootRef = useRef<HTMLElement>(null)
  const [height, setHeight] = useState(640)
  const hasSession = !!s
  useEffect(() => {
    const el = rootRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(([entry]) => setHeight(entry.contentRect.height))
    ro.observe(el)
    return () => ro.disconnect()
  }, [hasSession])
  const compact = height < 560
  // The hero sticks to the top of the list only while it leaves the list
  // most of the room; a tall one (a working task in a short pane) scrolls
  // away with the list instead of hiding Up next under it.
  const scrollerRef = useRef<HTMLDivElement>(null)
  const heroWrapRef = useRef<HTMLDivElement>(null)
  const [heroH, setHeroH] = useState(0)
  const [scrollerH, setScrollerH] = useState(0)
  useEffect(() => {
    const sc = scrollerRef.current
    const hw = heroWrapRef.current
    if (!sc || !hw || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => {
      setScrollerH(sc.clientHeight)
      setHeroH(hw.offsetHeight)
    })
    ro.observe(sc)
    ro.observe(hw)
    return () => ro.disconnect()
  }, [hasSession])
  const sticky = heroH > 0 && scrollerH > 0 && heroH <= scrollerH * 0.45

  // ── Item-level UI state; a lane switch resets it (the draft stays).
  const [editing, setEditing] = useState<EditState | null>(null)
  const [menu, setMenu] = useState<{ id: string; anchor: HTMLElement } | null>(null)
  const [heroConfirm, setHeroConfirm] = useState(false)
  const [clearConfirm, setClearConfirm] = useState(false)
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())
  const [edited, setEdited] = useState<Set<string>>(() => new Set())
  const [doneOpen, setDoneOpen] = useState<boolean | null>(null)
  // Done opens by itself (few finished, or all done) and then stays open: it
  // never folds up under the user as the 4th task finishes. Per lane.
  const [doneLatch, setDoneLatch] = useState<string | null>(null)
  const doneQualifies = model.hero.id === 'all-done' || model.d <= 3
  useEffect(() => { if (doneQualifies) setDoneLatch(sessionKey) }, [doneQualifies, sessionKey])
  const [doneAll, setDoneAll] = useState(false)
  const [showPlanner, setShowPlanner] = useState(false)
  useEffect(() => {
    setEditing(null)
    setMenu(null)
    setHeroConfirm(false)
    setClearConfirm(false)
    setExpanded(new Set())
    setDoneOpen(null)
    dismissToast()
  }, [sessionKey, dismissToast])
  // The running item changed: its confirm goes, and an editor on an item
  // that is neither running nor waiting any more has nothing left to edit.
  useEffect(() => { setHeroConfirm(false) }, [dId])
  useEffect(() => {
    if (!editing) return
    const it = s?.queue.find((q) => q.id === editing.id)
    if (!it || it.status === 'done') setEditing(null)
  }, [editing, s?.queue])

  // ── Focus helpers.
  const composerRef = useRef<HTMLTextAreaElement>(null)
  const heroPrimaryRef = useRef<HTMLButtonElement>(null)
  const rowRefs = useRef(new Map<string, HTMLElement>())
  const registerRow = useCallback((id: string, el: HTMLElement | null) => {
    if (el) rowRefs.current.set(id, el)
    else rowRefs.current.delete(id)
  }, [])
  // Right inside the tap: iOS raises its keyboard only for a focus made
  // while handling the user's own touch (the composer is always mounted).
  const focusComposer = useCallback(() => { composerRef.current?.focus() }, [])
  // After a file drop the chips render first.
  const focusComposerSoon = useCallback(() => {
    window.setTimeout(() => composerRef.current?.focus(), 0)
  }, [])
  const focusRow = useCallback((id: string | null) => {
    window.setTimeout(() => {
      const el = id ? rowRefs.current.get(id) : null
      if (el) {
        // Into view below the pinned hero (the scroller's scroll padding).
        el.focus({ preventScroll: true })
        el.scrollIntoView({ block: 'nearest' })
      } else if (!touchKeyboard()) composerRef.current?.focus()
    }, 30)
  }, [])
  // After a hero action: its new primary (unless it is disabled), else the
  // composer (not on touch — that would pop the keyboard up).
  const refocusAfterHero = useCallback(() => {
    if (touchKeyboard()) return
    window.setTimeout(() => {
      const primary = heroPrimaryRef.current
      if (primary && !primary.disabled) primary.focus()
      else composerRef.current?.focus()
    }, 60)
  }, [])

  // Announce a change of the hero's state (not on mount, not the empty card).
  const [announce, setAnnounce] = useState('')
  const prevHero = useRef<string | null>(null)
  const heroId = model.hero.id
  const heroLine = `${model.hero.pill ?? ''}: ${model.hero.title}`
  useEffect(() => {
    if (prevHero.current !== null && prevHero.current !== heroId && heroId !== 'empty') setAnnounce(heroLine)
    prevHero.current = heroId
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [heroId])

  // An open menu whose items would change under the pointer closes instead
  // (focus back on its trigger): the running task's when the hero changes
  // state or the agent starts/stops replying (Stop, Run it again…), a
  // waiting task's when a task starts or the queue goes idle (Run now ↔ Do
  // this next).
  const menuNow = useRef(menu)
  menuNow.current = menu
  const closeMenuKeepFocus = useCallback((which: (id: string) => boolean) => {
    const m = menuNow.current
    if (!m || !which(m.id)) return
    setMenu(null)
    requestAnimationFrame(() => {
      const a = document.activeElement
      if ((!a || a === document.body) && m.anchor.isConnected) m.anchor.focus()
    })
  }, [])
  const busyNow = !!s?.busy
  const runningNow = !!model.D
  useEffect(() => { closeMenuKeepFocus((id) => id === 'hero') }, [heroId, busyNow, closeMenuKeepFocus])
  useEffect(() => { closeMenuKeepFocus((id) => id !== 'header' && id !== 'hero') }, [runningNow, closeMenuKeepFocus])

  // ── Actions bound to this lane (undo copies taken before each change).
  const snapshot = useCallback((pick: (q: QueuedMessage) => boolean) => {
    const queue = useChatStore.getState().sessions.get(sessionKey)?.queue ?? []
    return queue.map((item, index) => ({ item, index })).filter(({ item }) => pick(item))
  }, [sessionKey])
  const undo = useCallback((entries: { item: QueuedMessage; index: number }[]): ToastAction => ({
    label: 'Undo', run: () => store.restoreQueueItems(sessionKey, entries),
  }), [store, sessionKey])

  // A task moves among the other waiting tasks (`beforeId` is a task, null
  // for last); the steps between them (fresh starts) stay put, so each task
  // keeps the one in front of it (moveTaskOrder). With no steps waiting, that
  // is the store's own single move.
  const moveTask = (id: string, beforeId: string | null) => {
    const waitingNow = () => (useChatStore.getState().sessions.get(sessionKey)?.queue ?? [])
      .filter((q) => q.status === 'pending')
    const waiting = waitingNow()
    if (!waiting.some((q) => isQueueCommand(q.content))) { store.moveQueueItem(sessionKey, id, beforeId); return }
    const target = moveTaskOrder(waiting, id, beforeId)
    if (target) applyWaitingOrder(target, waitingNow, (a, b) => store.moveQueueItem(sessionKey, a, b))
  }

  const pendingIds = model.pending.map((q) => q.id)
  const doneIds = model.done.map((q) => q.id)
  const neighbour = (ids: string[], id: string) => {
    const at = ids.indexOf(id)
    return at < 0 ? null : ids[at + 1] ?? ids[at - 1] ?? null
  }

  const act: QueueActs = {
    enqueue: (content) => store.enqueue(sessionKey, content),
    markDone: (id) => store.markItemDone(sessionKey, id),
    skipRunning: (id) => store.markItemDone(sessionKey, id, { how: 'skipped' }),
    skipWaiting: (id) => {
      const entries = snapshot((q) => q.id === id)
      const next = neighbour(pendingIds, id)
      store.markItemDone(sessionKey, id, { how: 'skipped' })
      showToast('Task skipped', undo(entries))
      focusRow(next)
    },
    removeWaiting: (id, message = 'Task removed') => {
      const entries = snapshot((q) => q.id === id)
      const next = neighbour(pendingIds, id)
      store.removeItem(sessionKey, id)
      showToast(message, undo(entries))
      focusRow(next)
    },
    removeRunning: (id) => {
      store.removeItem(sessionKey, id)
      setHeroConfirm(false)
      refocusAfterHero()
    },
    removeDone: (id) => {
      const entries = snapshot((q) => q.id === id)
      const next = neighbour(doneIds, id)
      store.removeItem(sessionKey, id)
      showToast('Task removed', undo(entries))
      focusRow(next)
    },
    retry: (id, content) => store.retryQueueItem(sessionKey, id, content),
    startEdit: (item, mode, saveLabel) => {
      const p = parseAttachmentRefs(item.content)
      setEditing({ id: item.id, mode, saveLabel, text: p.text, refs: p.refs })
    },
    saveEdit: (e) => {
      const content = joinAttachmentRefs(e.text.trim(), e.refs)
      if (!content) return
      setEditing(null)
      if (e.mode === 'retry') {
        store.retryQueueItem(sessionKey, e.id, content)
        refocusAfterHero()
        return
      }
      if (!store.editQueueItem(sessionKey, e.id, content)) {
        showToast("It started before you saved, so your edit wasn't applied.", {
          label: 'Add as new task', run: () => store.enqueue(sessionKey, content),
        })
        return
      }
      setEdited((prev) => new Set(prev).add(e.id))
      window.setTimeout(() => setEdited((prev) => { const n = new Set(prev); n.delete(e.id); return n }), 2000)
      focusRow(e.id)
    },
    move: (id, beforeId) => moveTask(id, beforeId),
    runNext: () => store.runQueueNext(sessionKey),
    insertNow: (id) => {
      const next = neighbour(pendingIds, id)
      if (store.insertNow(sessionKey, id)) {
        // No undo: the agent has the note already.
        showToast('Handed to the running task')
        focusRow(next)
        return
      }
      // The turn ended as the button was pressed: it runs on its own, next.
      const first = model.next
      if (first && first.id !== id) moveTask(id, first.id)
      store.runQueueNext(sessionKey)
    },
    stop: () => store.cancelTask(sessionKey),
    setPaused: (paused) => store.setQueuePaused(sessionKey, paused),
    setAuto: (on) => store.setQueueAutoMode(sessionKey, on),
    clearFinished: () => {
      const entries = snapshot((q) => q.status === 'done')
      if (!entries.length) return
      store.clearFinished(sessionKey)
      showToast(`${plural(model.d || entries.length, 'finished task')} cleared`, undo(entries))
    },
    clearWaiting: () => {
      const entries = snapshot((q) => q.status === 'pending')
      if (!entries.length) return
      setClearConfirm(false)
      store.clearQueue(sessionKey)
      showToast(model.k
        ? `${plural(model.k, 'waiting task')} removed`
        : `${plural(entries.length, 'waiting step')} removed`, undo(entries))
      focusRow(null)
    },
  }

  const onHeroAction = (a: HeroAction) => {
    const d = model.D
    switch (a.id) {
      case 'focus-composer': focusComposer(); return
      case 'open-planner': setShowPlanner(true); return
      case 'answer': onGoToChat({ focusComposer: true }); return
      case 'show-in-chat': if (d?.messageId) onGoToChat({ messageId: d.messageId }); return
      case 'show-reply': onGoToChat({}); return
      case 'edit-retry': if (d) act.startEdit(d, 'retry', a.saveLabel ?? 'Save & try again'); return
      case 'reconnect':
        if (s?.sharedInfo) store.openSharedChat(s.sharedInfo.agentRef, s.agent, s.sharedInfo.ownerName, s.lane)
        break
      case 'retry': if (d) act.retry(d.id); break
      case 'skip': if (d) act.skipRunning(d.id); break
      case 'mark-done': if (d) act.markDone(d.id); break
      case 'move-on': if (d) store.markItemDone(sessionKey, d.id, { how: 'no-answer' }); break
      case 'keep-waiting': store.extendQueueAnswerWait(sessionKey); break
      case 'pause-after': act.setPaused(true); break
      case 'keep-going':
      case 'resume': act.setPaused(false); break
      case 'auto-on': act.setAuto(true); break
      case 'run-next': act.runNext(); break
      case 'clear-finished': act.clearFinished(); break
    }
    refocusAfterHero()
  }

  // Cmd/Ctrl+Z takes back what the toast offers to undo — wherever the focus
  // is in the panel, or nowhere (a Clear that took the focused button away).
  // Not while typing, where it undoes the typing; an empty composer has none
  // to undo, and a removal often leaves the focus there.
  const undoRun = toast?.action?.label === 'Undo' ? toast.action.run : null
  useEffect(() => {
    if (!undoRun) return
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || !(e.metaKey || e.ctrlKey) || e.shiftKey || e.altKey || e.key.toLowerCase() !== 'z') return
      const a = document.activeElement as HTMLElement | null
      const nowhere = !a || a === document.body
      if (!nowhere && !rootRef.current?.contains(a)) return
      const composer = composerRef.current
      if (a && a === composer ? !!composer.value : a?.closest('textarea, input, select, [contenteditable="true"]')) return
      e.preventDefault()
      dismissToast()
      undoRun()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undoRun, dismissToast])

  // ── Files dropped anywhere on the panel attach to a new task.
  const [dropping, setDropping] = useState(false)
  const dragDepth = useRef(0)
  // A drag that ends elsewhere (dropped outside, cancelled) clears the overlay.
  useEffect(() => {
    if (!dropping) return
    const reset = () => { dragDepth.current = 0; setDropping(false) }
    window.addEventListener('drop', reset, true)
    window.addEventListener('dragend', reset, true)
    return () => {
      window.removeEventListener('drop', reset, true)
      window.removeEventListener('dragend', reset, true)
    }
  }, [dropping])
  const dropHandlers = shared ? { onDragOver: swallowDrop, onDrop: swallowDrop } : {
    onDragEnter: (e: React.DragEvent) => {
      if (!isFileDrag(e)) return
      e.preventDefault()
      dragDepth.current += 1
      setDropping(true)
    },
    onDragOver: (e: React.DragEvent) => {
      if (!isFileDrag(e)) return
      e.preventDefault()
      e.stopPropagation()
    },
    onDragLeave: (e: React.DragEvent) => {
      if (!isFileDrag(e)) return
      dragDepth.current = Math.max(0, dragDepth.current - 1)
      if (dragDepth.current === 0) setDropping(false)
    },
    onDrop: (e: React.DragEvent) => {
      if (!isFileDrag(e)) return
      e.preventDefault()
      e.stopPropagation()
      dragDepth.current = 0
      setDropping(false)
      if (e.dataTransfer.files.length > 0) {
        // No agent to upload to yet: say so, rather than a chip that waits forever.
        if (!conn) { showToast(`Can't attach files until ${s?.agent || 'the agent'} is connected.`); return }
        attachments.addFiles(e.dataTransfer.files)
        focusComposerSoon()
      }
    },
  }

  if (!s) return null

  const agent = s.agent || 'The agent'
  const phone = variant === 'phone'
  const ctx: QueueCtx = {
    sessionKey, agent, shared, variant, phone, compact, panelHeight: height, conn, model,
    auto: s.auto, paused: s.paused, busy: s.busy, stopping: s.stopping, connected: s.connected,
    editing, setEditing,
    menuFor: menu?.id ?? null, menuAnchor: menu?.anchor ?? null,
    openMenu: (id, anchor) => setMenu((cur) => (cur?.id === id ? null : { id, anchor })),
    closeMenu: () => setMenu(null),
    expanded,
    toggleExpanded: (id) => setExpanded((prev) => {
      const n = new Set(prev)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    }),
    edited, act, toast: showToast, registerRow, focusRow, onGoToChat,
  }

  const { hero, d, N, kAll, left } = model
  const laneLabel = s.lane === LANE_MAIN ? '' : s.lane === LANE_AUTO ? 'Auto lane' : `Lane ${s.lane}`
  // In a narrow column, as the lane strip names it.
  const laneShort = s.lane === LANE_AUTO ? 'Auto' : s.lane
  const laneTitle = s.lane === LANE_AUTO ? 'Automation lane' : laneLabel
  // Paused: the hero says "Paused"; the header only counts, in a quiet tone.
  // All done needs no pill: the hero right under it says so.
  const countPill = s.paused && left
    ? <Pill tone="zinc" icon={Pause}>{left} left</Pill>
    : left > 0
      ? <Pill tone="violet">{left} left</Pill>
      : null
  const planTitle = conn ? 'Plan tasks — describe the job once, get a list of tasks' : 'Waiting for the agent connection…'
  const doneIsOpen = doneOpen ?? (doneQualifies || doneLatch === sessionKey)

  // The header's ⋯ menu.
  const headerMenu: MenuEntry[] = [
    { label: 'Plan tasks…', icon: Wand2, hidden: shared, disabled: !conn, title: conn ? undefined : planTitle, onSelect: () => setShowPlanner(true) },
    s.paused
      // (Start until something has run, as the card's button says.)
      ? { label: model.unstarted ? 'Start the queue' : 'Resume the queue', icon: Play, onSelect: () => act.setPaused(false) }
      : {
        label: model.D ? 'Pause after this' : 'Pause the queue', icon: Pause,
        hidden: s.queue.length === 0, onSelect: () => act.setPaused(true),
      },
    'divider',
    { label: `Clear finished (${d})`, icon: Eraser, hidden: d === 0, onSelect: act.clearFinished },
    {
      label: `Remove waiting tasks (${model.k || kAll})…`, icon: Trash2, hidden: kAll === 0, danger: true,
      onSelect: () => {
        if (kAll < 3) { act.clearWaiting(); return }
        setClearConfirm(true)
        // The confirm replaces Up next's header: bring it out from under the hero.
        requestAnimationFrame(() => {
          scrollerRef.current?.querySelector('[data-fd-upnext-header]')?.scrollIntoView({ block: 'nearest' })
        })
      },
    },
    'divider',
    { label: 'Hide queue', icon: EyeOff, hidden: variant !== 'column' || !onHide, onSelect: () => onHide?.() },
  ]

  const headerBtn = phone
    ? 'h-10 w-10 rounded-xl'
    : 'h-7 w-7 rounded-lg pointer-coarse:h-9 pointer-coarse:w-9'

  return (
    <>
    <section
      ref={rootRef}
      aria-label="Queue"
      className="@container relative flex h-full min-h-0 flex-col"
      {...dropHandlers}
    >
      {/* Header */}
      {phone ? (
        <div className="flex h-12 shrink-0 items-center gap-1.5 border-b border-zinc-800 px-2">
          <button
            type="button"
            onClick={onBack}
            aria-label="Back to chat"
            className={`inline-flex h-10 items-center gap-0.5 rounded-xl px-2 text-sm font-medium text-zinc-300 hover:bg-zinc-800 ${FOCUS}`}
          >
            <ChevronLeft className="h-5 w-5" />
            Chat
          </button>
          <span className="ml-1 shrink-0 text-base font-semibold text-zinc-100">Queue</span>
          {/* The pills give way (the lane's name truncates), never the buttons. */}
          <span className="flex min-w-0 flex-1 items-center gap-1.5 overflow-hidden">
            {laneLabel && <LanePill label={laneLabel} short={laneShort} title={laneTitle} />}
            {countPill}
          </span>
          {!shared && (
            <button
              type="button"
              onClick={() => setShowPlanner(true)}
              disabled={!conn}
              title={planTitle}
              aria-label="Plan tasks"
              // The smallest phones keep the lane's name instead (⋯ has Plan too).
              className={`inline-flex shrink-0 items-center justify-center text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 disabled:opacity-40 ${laneLabel ? '@max-[22rem]:hidden' : ''} ${FOCUS} ${headerBtn}`}
            >
              <Wand2 className="h-4 w-4" />
            </button>
          )}
          <HeaderMenuButton ctx={ctx} items={headerMenu} className={`shrink-0 ${headerBtn}`} />
        </div>
      ) : (
        <div className={`flex min-w-0 shrink-0 items-center gap-1 overflow-hidden border-b border-zinc-800 px-3 @3xs:gap-2 ${variant === 'sidebar' ? 'h-10' : 'h-11'}`}>
          {/* A very narrow column keeps the title and the buttons; the hero
              below says the rest. Beside a lane's pill, the lane's name comes
              first: the icon (and Plan, which the ⋯ menu also has) only once
              there is room for both. */}
          <span className={`hidden h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-violet-500/10 text-violet-600 dark:text-violet-400 ${laneLabel ? '@sm:flex' : '@3xs:flex'}`}>
            <ListTodo className="h-3.5 w-3.5" />
          </span>
          {/* The title and its pills take the room the buttons leave: they
              give way (the lane's name truncates), never Hide or the menu. */}
          <span className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
            <span className="shrink-0 text-sm font-semibold text-zinc-100">Queue</span>
            {laneLabel && <span className="hidden min-w-0 @3xs:flex"><LanePill label={laneLabel} short={laneShort} title={laneTitle} /></span>}
            {countPill && <span className="hidden shrink-0 @3xs:inline-flex">{countPill}</span>}
          </span>
          {!shared && (
            <button
              type="button"
              onClick={() => setShowPlanner(true)}
              disabled={!conn}
              title={planTitle}
              aria-label="Plan tasks"
              className={`h-7 shrink-0 items-center gap-1.5 rounded-lg px-2 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 disabled:opacity-40 pointer-coarse:h-9 ${FOCUS} ${laneLabel ? 'hidden @xs:inline-flex' : 'inline-flex'}`}
            >
              <Wand2 className="h-3.5 w-3.5" />
              {/* Its word gives way to a count pill in a narrow header. */}
              <span className={laneLabel ? 'hidden @md:inline' : countPill ? 'hidden @sm:inline' : 'hidden @3xs:inline'}>Plan</span>
            </button>
          )}
          <HeaderMenuButton ctx={ctx} items={headerMenu} className={`shrink-0 text-zinc-500 hover:text-zinc-200 ${headerBtn}`} />
          {variant === 'column' && onHide && (
            <button
              type="button"
              onClick={onHide}
              title={`Hide queue (${IS_MAC ? '⌘J' : 'Ctrl+J'})`}
              aria-label="Hide queue"
              className={`inline-flex shrink-0 items-center justify-center text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 ${FOCUS} ${headerBtn}`}
            >
              <PanelLeftClose className="h-4 w-4" />
            </button>
          )}
          {variant === 'sidebar' && onCollapse && (
            <button
              type="button"
              onClick={onCollapse}
              title="Collapse queue"
              aria-label="Collapse queue"
              className={`inline-flex shrink-0 items-center justify-center text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 ${FOCUS} ${headerBtn}`}
            >
              <PanelLeftClose className="h-4 w-4" />
            </button>
          )}
        </div>
      )}

      {/* The only scroller. While the hero sticks at its top, the scroll
          padding keeps a focused card or the Clear confirm out from under it. */}
      <div
        ref={scrollerRef}
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 pb-4"
        style={sticky ? { scrollPaddingTop: heroH + 8 } : undefined}
      >
        <QueueHero
          ctx={ctx}
          onAction={onHeroAction}
          primaryRef={heroPrimaryRef}
          wrapRef={heroWrapRef}
          sticky={sticky}
          confirmRemove={heroConfirm}
          setConfirmRemove={setHeroConfirm}
        />

        <ModeBar
          ctx={ctx}
          parked={s.parked}
          summary={N > 0 && hero.id !== 'empty' && hero.id !== 'all-done'}
          showCaption={!compact && (hero.id === 'empty' || (s.parked && !s.auto))}
        />

        {hero.id === 'empty' && shared && (
          <p className="mt-2 px-1 text-[11px] leading-relaxed text-zinc-500">Only you see this list. It's saved in this browser.</p>
        )}

        {hero.id !== 'empty' && (kAll > 0 || d > 0) && (
          <div className="mt-4 space-y-4">
            {kAll > 0 && (
              <UpNextSection ctx={ctx} confirmClear={clearConfirm} setConfirmClear={setClearConfirm} />
            )}
            {d > 0 && (
              <DoneSection
                ctx={ctx}
                open={doneIsOpen}
                onToggle={() => setDoneOpen(!doneIsOpen)}
                showAll={doneAll}
                setShowAll={setDoneAll}
              />
            )}
          </div>
        )}
      </div>

      <QueueComposer
        ctx={ctx}
        taRef={composerRef}
        attachments={attachments}
        panelHeight={height}
        closedInfo={closedInfo}
        quotaSpent={quotaSpent}
        toast={<UndoToast toast={toast} onDismiss={dismissToast} onHold={holdToast} onRelease={releaseToast} />}
      />

      {dropping && !shared && (
        <div className="pointer-events-none absolute inset-2 z-30 flex flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-violet-500/50 bg-violet-500/10 text-[13px] font-medium text-violet-700 backdrop-blur-[1px] dark:text-violet-300">
          <UploadCloud className="h-6 w-6" />
          {conn ? 'Drop to attach to a new task' : `Files can be attached once ${agent} is connected`}
        </div>
      )}

      <span role="status" aria-live="polite" className="sr-only">{announce}</span>
    </section>
    {/* Outside the section, so the planner's own file drops never reach the
        panel's drop overlay. */}
    {showPlanner && conn && !shared && (
      <QueuePlannerModal
        agentId={agentId}
        agentName={agent}
        host={conn.host}
        port={conn.port}
        auth={conn.auth}
        onClose={() => setShowPlanner(false)}
      />
    )}
    </>
  )
}

/** The lane's name as a pill that gives way first in a narrow header: its
 *  short name ("B", as the lane strip says) under 384px, and it truncates
 *  where the console's Pill never shrinks. (Pill's zinc tone.) */
function LanePill({ label, short, title }: { label: string; short: string; title: string }) {
  return (
    <span title={title} className="inline-flex min-w-0 items-center rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] font-medium text-zinc-400">
      <span className="truncate">
        <span className="@sm:hidden">{short}</span>
        <span className="hidden @sm:inline">{label}</span>
      </span>
    </span>
  )
}

function HeaderMenuButton({ ctx, items, className }: { ctx: QueueCtx; items: MenuEntry[]; className: string }) {
  const open = ctx.menuFor === 'header'
  return (
    <>
      <button
        type="button"
        onClick={(e) => ctx.openMenu('header', e.currentTarget)}
        aria-label="Queue options"
        title="Queue options"
        aria-haspopup="menu"
        aria-expanded={open}
        className={`inline-flex items-center justify-center text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 ${FOCUS} ${open ? 'bg-zinc-800 text-zinc-200' : ''} ${className}`}
      >
        <MoreHorizontal className="h-4 w-4" />
      </button>
      {open && ctx.menuAnchor && (
        <QueueMenu
          anchor={ctx.menuAnchor}
          label="Queue options"
          items={items}
          footer="Saved in this browser only."
          onClose={ctx.closeMenu}
        />
      )}
    </>
  )
}

const MODE_CAPTION = {
  auto: 'Each task starts when the one before it finishes. If the agent asks you something, it waits up to 3 min.',
  step: 'You check each reply and mark it done before the next task starts.',
}

/** How the queue moves on (Auto | Step by step), and where the run stands.
 *  The caption explaining the modes shows on the empty card and when the
 *  power-off park needs explaining; otherwise it is the control's tooltip. */
function ModeBar({ ctx, parked, summary, showCaption }: { ctx: QueueCtx; parked: boolean; summary: boolean; showCaption: boolean }) {
  const { model, auto } = ctx
  const caption = parked && !auto
    ? `Switched off when ${ctx.agent} was powered off.`
    : auto ? MODE_CAPTION.auto : MODE_CAPTION.step
  const options: { value: 'auto' | 'step'; label: string; short: string }[] = [
    { value: 'auto', label: 'Auto', short: 'Auto' },
    { value: 'step', label: 'Step by step', short: 'Step' },
  ]
  const current = auto ? 'auto' : 'step'
  const pick = (v: 'auto' | 'step') => { if (v !== current) ctx.act.setAuto(v === 'auto') }
  const totals = totalsText(model.totals)
  return (
    <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1.5 px-1">
      {summary && (
        <span className="text-[11px] tabular-nums text-zinc-500" title={model.totals.tools ? `${model.totals.tools} tool calls` : undefined}>
          {model.d} of {model.N} done{totals ? ` · ${totals}` : ''}
        </span>
      )}
      <div
        role="radiogroup"
        aria-label="How the queue moves on"
        title={showCaption ? undefined : caption}
        // Beside the summary it sits on the right; alone, it lines up with the card.
        className={`${summary ? 'ml-auto ' : ''}inline-flex rounded-lg border border-zinc-800 bg-zinc-950/60 p-0.5`}
        onKeyDown={(e) => {
          // A radio group: the arrows move the choice.
          if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'ArrowDown') {
            e.preventDefault()
            const group = e.currentTarget
            const nextV = current === 'auto' ? 'step' : 'auto'
            pick(nextV)
            window.setTimeout(() => group.querySelector<HTMLButtonElement>(`[data-mode="${nextV}"]`)?.focus(), 0)
          }
        }}
      >
        {options.map((o) => {
          const on = o.value === current
          return (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={on}
              tabIndex={on ? 0 : -1}
              data-mode={o.value}
              onClick={() => pick(o.value)}
              className={`rounded-md px-2 py-1 text-[11px] font-medium transition-colors pointer-coarse:px-2.5 pointer-coarse:py-1.5 pointer-coarse:text-xs ${FOCUS} ${
                on ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-500 hover:text-zinc-200'
              }`}
            >
              <span className="hidden @3xs:inline">{o.label}</span>
              <span className="@3xs:hidden">{o.short}</span>
            </button>
          )
        })}
      </div>
      {showCaption && (
        <p className={`w-full text-[11px] leading-relaxed ${parked && !auto ? 'text-amber-700 dark:text-amber-300' : 'text-zinc-500'}`}>
          {caption}
        </p>
      )}
    </div>
  )
}
