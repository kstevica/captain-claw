import { useState, useRef, useEffect, useCallback, useMemo, memo } from 'react'
import {
  AlarmClock,
  X,
  Send,
  Loader2,
  StopCircle,
  MessageSquare,
  Wrench,
  AlertCircle,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Minus,
  Forward,
  Paperclip,
  Clipboard,
  Copy,
  Check,
  Pin,
  ClipboardList,
  Activity,
  ListChecks,
  Maximize2,
  Minimize2,
  AlertTriangle,
  CircleDot,
  Workflow,
  ListTodo,
  Users,
  RefreshCw,
  Layers,
  History,
  Undo2,
  Sparkles,
  Eraser,
  Pencil,
  Plus,
  ArrowUp,
  Moon,
  Bot,
  Wand2,
  ImageIcon,
  Globe,
  ExternalLink,
  GraduationCap,
} from 'lucide-react'
import { createPortal } from 'react-dom'
import { ChatGreeting, SkillInputDialog, SkillsDialog, SkillsMenu } from './AgentSkills'
import { EffortMenu } from './EffortMenu'
import { ChatFilesMenu } from './ChatFilesMenu'
import { SaveLessonDialog, SuggestLessonDialog } from '../layout/LessonDraft'
import { isManagedAgent } from '../../utils/managedAgents'
import { AttachmentChip } from './AttachmentChip'
import { appendAttachmentRefs, useFileAttachments } from '../../hooks/useFileAttachments'
import { useAgentSkills } from '../../hooks/useAgentSkills'
import { asksForInput, type AgentSkill } from '../../services/agentSkills'
import { fetchTopics, type TopicInfo } from '../../stores/chatStore'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { useShallow } from 'zustand/react/shallow'
import { useChatStore, PLAN_LEVELS, LANE_MAIN, LANE_AUTO, LANES, laneKey, type PlanLevel, type NextStepOption } from '../../stores/chatStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { useContainerStore } from '../../stores/containerStore'
import { useProcessStore } from '../../stores/processStore'
import { usePinnedStore } from '../../stores/pinnedStore'
import { useClipboardStore } from '../../stores/clipboardStore'
import { useTraceStore, selectSpanCount } from '../../stores/traceStore'
import { useUIStore } from '../../stores/uiStore'
import { useAuthStore, selectKioskLocked } from '../../stores/authStore'
import { useScheduledStore } from '../../stores/scheduledStore'
import { useSharedAgentStore } from '../../stores/sharedAgentStore'
import { sharedCaps, sharedCloseInfo } from '../../utils/sharedAgent'
import { SharedAgentNotice } from './SharedAgentNotice'
import { SharedAgentGoogleToggle } from './SharedAgentGoogleToggle'
import { ContextPacksModal } from './ContextPacksModal'
import { PACKS_BUTTON } from '../../utils/contextPacks'
import { SendContextModal } from './SendContextModal'
import { FlowSelectorModal } from './FlowSelectorModal'
import { PlanCard } from './PlanCard'
import { PaymentCard } from '../payments/PaymentCard'
import { PurchaseCard } from '../payments/PurchaseCard'
import { paymentIntentOf, purchaseIntentOf } from '../../services/payments'
import TraceTimeline from '../observability/TraceTimeline'
import { AgentFilesPanel } from './AgentFilesPanel'
import { AgentDatastorePanel } from './AgentDatastorePanel'
import { QueuePanel, type QueueVariant } from './queue/QueuePanel'
import { useQueueProgress, useQueueSignal } from './queue/queueHooks'
import type { SignalTone } from './queue/queueModel'
import { FOCUS } from './queue/QueueButton'
import { usePersistedSize } from '../../hooks/usePersistedSize'
import { touchKeyboard, useIsMobile } from '../../hooks/useMediaQuery'
import { fmtTokens } from '../../utils/format'
import { agentFileViewUrl } from '../../utils/agentFileUrl'
import { useQuotaStore } from '../../stores/quotaStore'
import { PlanUsageBar } from '../common/PlanUsage'
import type { ChatMessage, TokenUsage } from '../../services/agentChat'
import type { ReasoningProfile } from '../../utils/reasoningEffort'

// Plugin arrays hoisted to module scope so they keep a stable identity across
// renders (react-markdown otherwise sees "new" plugins every render).
const REMARK_PLUGINS = [remarkGfm, remarkMath]
const REHYPE_PLUGINS = [rehypeKatex]

// The chat box's "queue it instead of sending" chord, as this keyboard names it.
const QUEUE_CHORD = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)
  ? '⌘Enter'
  : 'Ctrl+Enter'

// Memoized markdown renderer. Parsing markdown + KaTeX is the single most
// expensive thing the chat does, and the transcript re-renders on every
// streaming event (status/monitor/turn_usage fire many times per turn). Keying
// the memo on the already-transformed `content` string means a finished message
// never re-parses while a later turn streams — the core fix for heavy-chat lag.
// IMPORTANT: callers must pass the FINAL transformed string (after
// processImagePaths/stripSuggestions), not raw message.content, or those
// transforms won't invalidate the memo.
const MarkdownMessage = memo(function MarkdownMessage({ content }: { content: string }) {
  return <Markdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={REHYPE_PLUGINS}>{content}</Markdown>
})

// Shared chats take no files: a drag is accepted and dropped on the floor.
function swallowDrop(e: React.DragEvent) {
  e.preventDefault()
  e.stopPropagation()
}

// `variant`:
//  - 'default' — the docked/fullscreen chat in the full layout (unchanged).
//  - 'simple'  — the centre column of the simple layout. The layout itself is
//                fixed there, so the fullscreen and minimise buttons go away,
//                and the left column carries only the queue: the agent's files
//                and datastore live in the layout's own right-hand sidebar.
export function ChatPanel({ variant = 'default' }: { variant?: 'default' | 'simple' } = {}) {
  const simple = variant === 'simple'
  // Kiosk lock: in the locked simple-chat layout the header keeps only the
  // queue toggle — flows, send-to-another-agent, plan routing and traces are
  // all hidden so shared-account users can't reach the wider system.
  const kiosk = useAuthStore(selectKioskLocked)  // admins are exempt from the kiosk lock
  // Simple layout only: the queue column can be tucked away so the
  // conversation gets the whole centre. Fullscreen (full layout) is unchanged.
  // On a phone the queue is a view of its own (it would squeeze the chat to
  // nothing beside it), starting closed — not the desktop column's setting.
  const { isMobile } = useIsMobile()
  const desktopQueueOpen = useUIStore((s) => s.simpleQueueOpen)
  const setDesktopQueueOpen = useUIStore((s) => s.setSimpleQueueOpen)
  const [phoneQueueOpen, setPhoneQueueOpen] = useState(false)
  const simpleQueueOpen = isMobile ? phoneQueueOpen : desktopQueueOpen
  const setSimpleQueueOpen = isMobile ? setPhoneQueueOpen : setDesktopQueueOpen
  // Narrow subscriptions instead of `useChatStore()` (whole store). A bare
  // whole-store read re-renders ChatPanel on EVERY event of EVERY agent/lane,
  // including background agents the user isn't looking at. Actions are stable
  // refs (grouped via useShallow, so they never trigger a render); the booleans
  // and the ACTIVE session are read individually so a background agent's token
  // churn no longer re-renders this pane.
  const activeChatId = useChatStore((s) => s.activeChatId)
  const chatOpen = useChatStore((s) => s.chatOpen)
  const chatFullscreen = useChatStore((s) => s.chatFullscreen)
  const {
    closeChat,
    switchChat,
    disconnectChat,
    sendMessage,
    cancelTask,
    setPlanningEnabled,
    setPlanLevel,
    toggleChatFullscreen,
    openSharedChat,
  } = useChatStore(
    useShallow((s) => ({
      closeChat: s.closeChat,
      switchChat: s.switchChat,
      disconnectChat: s.disconnectChat,
      sendMessage: s.sendMessage,
      cancelTask: s.cancelTask,
      setPlanningEnabled: s.setPlanningEnabled,
      setPlanLevel: s.setPlanLevel,
      toggleChatFullscreen: s.toggleChatFullscreen,
      openSharedChat: s.openSharedChat,
    })),
  )
  // Shared agents: the deck's host-trust warning, for the member notice.
  const sharedHostWarning = useSharedAgentStore((s) => s.hostWarning)
  // Context packs: a member opens "Shared context" from the shared chat's bar.
  const packsEnabled = useSharedAgentStore((s) => s.contextPacks)
  // The deck serves members the agent's saved files and datastore: the
  // member notice says they are shared with everyone who uses it.
  const memberWorkspace = useSharedAgentStore((s) => s.memberWorkspace)
  // The owner's agent can look into members' use of it (PR D): the member
  // notice says so.
  const sharedUsage = useSharedAgentStore((s) => s.sharedUsage)
  const localAgents = useLocalAgentStore((s) => s.agents)
  const containers = useContainerStore((s) => s.containers)
  const processes = useProcessStore((s) => s.processes)
  const [showSendContext, setShowSendContext] = useState(false)
  const [showFlows, setShowFlows] = useState(false)
  const [showTracePanel, setShowTracePanel] = useState(false)
  const [planLevelMenuOpen, setPlanLevelMenuOpen] = useState(false)
  const planLevelMenuRef = useRef<HTMLDivElement>(null)

  // Click-away handler for the plan-level dropdown.
  useEffect(() => {
    if (!planLevelMenuOpen) return
    function onDocClick(e: MouseEvent) {
      if (planLevelMenuRef.current && !planLevelMenuRef.current.contains(e.target as Node)) {
        setPlanLevelMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [planLevelMenuOpen])
  const activeId = activeChatId || ''
  const traceSpanCount = useTraceStore((s) => selectSpanCount(s, activeId))
  // The queue toggle's badge: tasks left, and whether the queue needs you.
  const queueSignal = useQueueSignal(activeId)

  // The queue's "Show in chat" / "Answer in chat": scroll (and focus) inside
  // THIS chat panel only — a phone first leaves the queue view, so the chat
  // can mount before it is scrolled.
  const rootRef = useRef<HTMLDivElement>(null)
  const goToChat = useCallback((o: { messageId?: string; focusComposer?: boolean }) => {
    // On a phone (either layout) the queue is a view of its own.
    const phoneView = isMobile
    if (phoneView) setPhoneQueueOpen(false)
    window.setTimeout(() => {
      const root = rootRef.current
      const el = o.messageId ? root?.querySelector<HTMLElement>(`#${CSS.escape(`fd-msg-${o.messageId}`)}`) : null
      if (o.messageId && el) {
        scrollToMessage(o.messageId)
        if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
          const bubble = (el.firstElementChild as HTMLElement | null) ?? el
          bubble.animate?.(
            [{ boxShadow: '0 0 0 2px rgb(139 92 246 / .6)' }, { boxShadow: '0 0 0 2px rgb(139 92 246 / 0)' }],
            { duration: 1200 },
          )
        }
      } else {
        root?.querySelector('[data-fd-chat-end]')?.scrollIntoView({ behavior: 'smooth' })
      }
      // iOS may not raise its keyboard for a focus it didn't see a tap for.
      if (o.focusComposer) root?.querySelector<HTMLTextAreaElement>('[data-fd-chat-input]')?.focus()
    }, phoneView ? 120 : 0)
  }, [isMobile])
  // Stable, so the memoized queue doesn't re-render with every streamed token.
  // Hiding the column hands the focus to the toggle that brings it back.
  const hideQueue = useCallback(() => {
    setDesktopQueueOpen(false)
    requestAnimationFrame(() => {
      const a = document.activeElement
      if (!a || a === document.body || !a.isConnected) {
        rootRef.current?.querySelector<HTMLElement>('[data-fd-queue-toggle]')?.focus()
      }
    })
  }, [setDesktopQueueOpen])
  const backToChat = useCallback(() => setPhoneQueueOpen(false), [])
  // Bumped when the chat box queues a message: the fullscreen queue, folded
  // to its bar or rail, unfolds to show it.
  const [queueReveal, setQueueReveal] = useState(0)

  // The active session, keyed — its object identity changes only when THIS
  // session mutates, so other agents' events don't re-render the transcript.
  const session = useChatStore((s) => (s.activeChatId ? s.sessions.get(s.activeChatId) ?? null : null))
  // A shared chat's row in the "shared with me" list — only that row, so the
  // list's 10s poll doesn't re-render this pane: what member chats on it can
  // use, and the member's own Google switch.
  const sharedRef = session?.shared?.agentRef
  const sharedRow = useSharedAgentStore((s) => (sharedRef ? s.agents.find((a) => a.agent_ref === sharedRef) : undefined))
  // The member's "Shared context" dialog for this chat's agent; closed when
  // the chat switches to another agent, and when this chat ends or the agent
  // is no longer shared with me (left, revoked, sharing turned off) — so it
  // never lingers on stale data or pops back open after a re-share.
  const [showPacks, setShowPacks] = useState(false)
  const sharedLive = !!sharedRow && !session?.closed
  useEffect(() => { setShowPacks(false) }, [sharedRef, sharedLive])

  // One tab per AGENT. A lane is a context inside an agent, not another agent,
  // so lane sessions don't get their own top-level tab. Subscribe to a PRIMITIVE
  // signature (key + connection + name) so the tab strip re-renders only when
  // membership/labels actually change — not on every streaming tick — then read
  // the current session objects non-reactively when the signature fires.
  const tabsSignature = useChatStore((s) => {
    let sig = ''
    for (const x of s.sessions.values()) {
      if (x.lane === LANE_MAIN) sig += `${x.key} ${x.connected ? 1 : 0} ${x.containerName}\n`
    }
    return sig
  })
  const chatTabs = useMemo(
    () => Array.from(useChatStore.getState().sessions.values()).filter((s) => s.lane === LANE_MAIN),
    [tabsSignature],
  )

  if ((!chatOpen && !simple) || !session) return null

  // An agent another deck user shared with you: chat only. No uploads, files,
  // datastore, plan mode, traces, flows or sending its context elsewhere.
  const shared = session.shared
  const sharedClose = shared && session.closed
    ? sharedCloseInfo(session.closed.code, shared.ownerName, session.closed.reason)
    : null
  // Process agents: member chats use the member's own files and deep memory,
  // and their Google if they turned it on. Docker / older deck: chat-only.
  const caps = sharedCaps(sharedRow)

  // Build target list for context transfer (all reachable agents except the
  // current one). Compare against the AGENT id — `activeChatId` is a lane key,
  // so on lane B it would otherwise fail to exclude this very agent.
  const activeAgentId = session.containerId
  const targets = [
    ...containers
      .filter((c) => c.status === 'running' && c.web_port && c.id !== activeAgentId)
      .map((c) => ({ id: c.id, name: c.agent_name || c.name, host: 'localhost', port: c.web_port!, auth: c.web_auth })),
    ...processes
      .filter((p) => p.status === 'running' && p.web_port && `proc-${p.slug}` !== activeAgentId)
      .map((p) => ({ id: `proc-${p.slug}`, name: p.name, host: 'localhost', port: p.web_port, auth: p.web_auth })),
    ...localAgents
      .filter((a) => a.status === 'online' && a.id !== activeAgentId)
      .map((a) => ({ id: a.id, name: a.name, host: a.host, port: a.port, auth: a.authToken })),
  ]

  // On a phone the simple layout's own header names the agent: the tab bar
  // goes. On a phone in either layout the queue button sits on the lane row
  // and the queue opens as a view of its own.
  const phoneSimple = simple && isMobile
  const queueTitle = simpleQueueOpen
    ? 'Hide queue'
    : `Show queue${queueSignal.left ? ` · ${queueSignal.left} left` : ''}${queueSignal.tone === 'needs' ? ' · needs you' : ''}`
  const queueToggle = (
    <button
      onClick={() => setSimpleQueueOpen(!simpleQueueOpen)}
      title={queueTitle}
      aria-label={queueTitle}
      aria-pressed={simpleQueueOpen}
      data-fd-queue-toggle
      className={`relative mr-1 shrink-0 rounded transition-colors ${FOCUS} ${isMobile ? 'p-2' : 'p-1'} ${
        simpleQueueOpen
          ? 'bg-violet-600/20 text-violet-400'
          : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
      }`}
    >
      <ListTodo className="h-4 w-4" />
      {!simpleQueueOpen && <QueueBadge signal={queueSignal} />}
    </button>
  )

  // Cmd/Ctrl+Enter in the chat box: the message goes to this lane's queue,
  // on Auto, and the queue comes into view — its column (a phone's queue
  // view), or in the full layout the fullscreen chat that carries it.
  const queueFromChat = (content: string) => {
    const st = useChatStore.getState()
    st.setQueueAutoMode(session.key, true)
    st.enqueueQueueMessage(session.key, content)
    if (simple || isMobile) {
      setSimpleQueueOpen(true)
      return
    }
    unfoldFullscreenQueue()
    if (chatFullscreen) setQueueReveal((n) => n + 1)
    else toggleChatFullscreen()
  }

  return (
    <div ref={rootRef} className={`flex h-full flex-col bg-zinc-950/80 ${simple ? '' : 'border-l border-zinc-800'}`}>
      {/* Tab bar */}
      {!phoneSimple && (
      <div className="flex items-center border-b border-zinc-800 bg-zinc-900/60">
        <div className="flex flex-1 items-center gap-0.5 overflow-x-auto px-1 py-1">
          {chatTabs.map((s) => (
            <button
              key={s.key}
              onClick={() => switchChat(s.key)}
              className={`group flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
                s.key === activeChatId
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-300'
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${s.connected ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
              <span className="max-w-[100px] truncate">{s.containerName}</span>
              <button
                onClick={(e) => { e.stopPropagation(); disconnectChat(s.key) }}
                className="ml-0.5 rounded p-0.5 text-zinc-600 opacity-0 transition-opacity hover:bg-zinc-700 hover:text-zinc-300 group-hover:opacity-100"
              >
                <X className="h-3 w-3" />
              </button>
            </button>
          ))}
        </div>
        {shared && (
          <span
            className="mr-1 flex shrink-0 items-center gap-1 rounded border border-sky-500/25 bg-sky-500/15 px-1.5 py-0.5 text-[10px] font-medium text-sky-700 dark:text-sky-300"
            title={`${session.containerName} belongs to ${shared.ownerName || 'another user'}. Your chats are private from other members, not from ${shared.ownerName || 'its owner'} or this deck's admins.`
              + (caps.files || caps.deep_memory ? ' During your chats it uses your own files and deep memory, and your Google only if you turned it on.' : '')}
          >
            <Users className="h-3 w-3" />
            Shared · {shared.ownerName || 'another user'}
          </span>
        )}
        {simple && queueToggle}
        {!kiosk && !shared && (
          <button
            onClick={() => setShowFlows(true)}
            title="Flows — enable/disable and start a flow"
            className="mr-1 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
          >
            <Workflow className="h-4 w-4" />
          </button>
        )}
        {!kiosk && !shared && (
          <button
            onClick={() => setShowSendContext(true)}
            title="Send context to another agent"
            className="mr-1 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
          >
            <Forward className="h-4 w-4" />
          </button>
        )}
        {/* Combined plan toggle + level selector. The button shows the current
            state ("OFF" when planning auto-routing is disabled, otherwise the
            level abbreviation) and opens a 5-row dropdown: Off + the four
            enrichment levels. Picking a level enables planning AND sets the
            level in one click; picking Off disables auto-routing while
            preserving the previously-chosen level. */}
        {!kiosk && !shared && (
        <div className="relative mr-1" ref={planLevelMenuRef}>
          <button
            onClick={() => setPlanLevelMenuOpen((o) => !o)}
            title={
              session.planningEnabled
                ? `Planning ON (${session.planLevel}). Click to change level or turn off.`
                : 'Planning OFF. Click to choose an enrichment level and turn on.'
            }
            className={`relative flex items-center gap-1 rounded px-1.5 py-1 text-[10px] font-bold uppercase tracking-wider transition-colors ${
              session.planningEnabled
                ? 'bg-violet-500 text-white shadow-sm shadow-violet-500/40 ring-1 ring-violet-300/60 hover:bg-violet-400'
                : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
            }`}
          >
            <ListChecks className="h-4 w-4" />
            <span className="leading-none">
              {session.planningEnabled
                ? (session.planLevel || 'plain').slice(0, 3)
                : 'OFF'}
            </span>
            <ChevronDown className="h-3 w-3 opacity-70" />
            {session.planState && session.planState.status === 'running' && (
              <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-pulse rounded-full bg-emerald-400 ring-2 ring-zinc-900" />
            )}
          </button>
          {planLevelMenuOpen && (
            <div className="absolute right-0 top-full z-50 mt-1 w-60 overflow-hidden rounded-md border border-zinc-700 bg-zinc-900 shadow-lg">
              <button
                onClick={() => {
                  // Always send /planning off, even if FD thinks it's already
                  // off — same staleness concern as the level picker below.
                  setPlanningEnabled(session.key, false)
                  setPlanLevelMenuOpen(false)
                }}
                className={`flex w-full flex-col items-start gap-0.5 border-b border-zinc-800 px-3 py-2 text-left text-xs transition-colors ${
                  !session.planningEnabled
                    ? 'bg-zinc-800/80 text-zinc-200'
                    : 'text-zinc-300 hover:bg-zinc-800'
                }`}
              >
                <span className="font-semibold uppercase tracking-wide">Off</span>
                <span className="text-[10px] text-zinc-400">Disable plan auto-routing</span>
              </button>
              {PLAN_LEVELS.map((lvl) => {
                const desc: Record<PlanLevel, string> = {
                  plain: 'No enrichment (default)',
                  enriched: '+ latest reflection',
                  insightful: '+ top user insights',
                  complete: '+ persona-aware planner',
                }
                const active = session.planningEnabled && session.planLevel === lvl
                return (
                  <button
                    key={lvl}
                    onClick={() => {
                      // Always send both commands unconditionally. Skipping
                      // them when the FD mirror "looks right" lets a stale
                      // mirror (e.g., after a ws drop or a missed command_result)
                      // hide the fact that the agent's plan_mode_auto is still
                      // off — the user picks a level, the UI shows it active,
                      // but auto-routing never engages and they have to type
                      // /plan manually. Both commands are idempotent so the
                      // double-send when state already matches is harmless.
                      // Set level first so the /planning on confirmation
                      // already reflects the new level.
                      setPlanLevel(session.key, lvl)
                      setPlanningEnabled(session.key, true)
                      setPlanLevelMenuOpen(false)
                    }}
                    className={`flex w-full flex-col items-start gap-0.5 border-b border-zinc-800 px-3 py-2 text-left text-xs last:border-b-0 transition-colors ${
                      active
                        ? 'bg-violet-500/20 text-violet-200'
                        : 'text-zinc-300 hover:bg-zinc-800'
                    }`}
                  >
                    <span className="font-semibold uppercase tracking-wide">{lvl}</span>
                    <span className="text-[10px] text-zinc-400">{desc[lvl]}</span>
                  </button>
                )
              })}
            </div>
          )}
        </div>
        )}
        {!kiosk && !shared && (
          <button
            onClick={() => setShowTracePanel(!showTracePanel)}
            title={showTracePanel ? 'Hide traces' : 'Show orchestrator traces'}
            className={`relative mr-1 rounded p-1 transition-colors ${
              showTracePanel
                ? 'bg-violet-600/20 text-violet-400'
                : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
            }`}
          >
            <Activity className="h-4 w-4" />
            {traceSpanCount > 0 && (
              <span className="absolute -right-0.5 -top-0.5 flex h-3.5 min-w-[14px] items-center justify-center rounded-full bg-violet-600 px-0.5 text-[8px] font-bold text-white">
                {traceSpanCount > 99 ? '99+' : traceSpanCount}
              </span>
            )}
          </button>
        )}
        {!simple && (
          <button
            onClick={toggleChatFullscreen}
            title={chatFullscreen ? 'Exit fullscreen' : 'Fullscreen chat (hide other panels)'}
            className={`mr-1 rounded p-1 transition-colors ${
              chatFullscreen
                ? 'bg-violet-600/20 text-violet-400'
                : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
            }`}
          >
            {chatFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
        )}
        {!simple && (
          <button onClick={closeChat} className="mr-2 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <Minus className="h-4 w-4" />
          </button>
        )}
      </div>
      )}

      {/* Trace panel (replaces chat when active) */}
      {/* Lanes: three parallel contexts on this one agent. */}
      <LaneStrip
        agentId={session.containerId}
        activeLane={session.lane}
        agentName={session.containerName}
        trailing={isMobile ? queueToggle : undefined}
      />

      {/* Today's plan allowance; when it's spent the composer stays closed */}
      <PlanUsageBar simple={simple} />

      {shared && (
        <SharedAgentNotice
          key={shared.agentRef}
          agentRef={shared.agentRef}
          agentName={session.containerName}
          ownerName={shared.ownerName}
          hostWarning={sharedHostWarning}
          caps={caps}
          workspace={memberWorkspace}
          ownerReads={sharedUsage}
        />
      )}

      {shared && sharedRow && !session.closed && (caps.google || packsEnabled) && (
        <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-1.5">
          {/* The member's Google switch renders only when caps.google (it hides itself otherwise). */}
          <div className="min-w-0 flex-1">
            <SharedAgentGoogleToggle agent={sharedRow} compact />
          </div>
          {packsEnabled && (
            <button
              onClick={() => setShowPacks(true)}
              title={`What you share with everyone who uses ${session.containerName}`}
              className="ml-auto flex shrink-0 items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <Layers className="h-3 w-3" />
              {PACKS_BUTTON}
            </button>
          )}
        </div>
      )}

      {shared && packsEnabled && showPacks && sharedLive && (
        <ContextPacksModal
          key={shared.agentRef}
          agentRef={shared.agentRef}
          agentName={session.containerName}
          onClose={() => setShowPacks(false)}
        />
      )}

      {shared && sharedClose && (
        <div className="flex items-center gap-2 border-b border-amber-500/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
          <span className="min-w-0 flex-1">{sharedClose.message}</span>
          {sharedClose.retry !== 'none' && (
            <button
              onClick={() => openSharedChat(shared.agentRef, session.containerName, shared.ownerName, session.lane)}
              className="flex shrink-0 items-center gap-1 rounded-md border border-amber-500/40 px-2 py-0.5 text-[11px] font-medium hover:bg-amber-500/20"
            >
              <RefreshCw className="h-3 w-3" />
              Retry
            </button>
          )}
        </div>
      )}

      {showTracePanel && !shared ? (
        <div className="flex flex-1 flex-col overflow-hidden">
          <div className="flex items-center gap-2 border-b border-zinc-800/50 bg-zinc-900/30 px-3 py-2">
            <Activity className="h-3.5 w-3.5 text-violet-400" />
            <span className="text-xs font-medium text-zinc-300">Orchestrator Traces</span>
            {traceSpanCount > 0 && (
              <span className="rounded-full bg-violet-600/20 px-1.5 py-0.5 text-[10px] font-medium text-violet-400">
                {traceSpanCount} spans
              </span>
            )}
          </div>
          <div className="flex-1 overflow-y-auto">
            <TraceTimeline containerId={activeId} />
          </div>
        </div>
      ) : (
        /* Chat content (optionally with queue panel on the left in fullscreen) */
        <div className="flex flex-1 overflow-hidden">
          {/* Simple layout: the column (desktop) or its own view (phone). Full
              layout: the fullscreen sidebar — and on a phone, the same phone
              view from the lane row's queue button. */}
          {((simple || isMobile) ? simpleQueueOpen : chatFullscreen) && (
            <QueueSidebar
              sessionKey={session.key}
              agentId={session.containerId}
              queueOnly={simple || isMobile || !!shared}
              shared={!!shared}
              variant={isMobile ? 'phone' : simple ? 'column' : 'sidebar'}
              onGoToChat={goToChat}
              onHide={hideQueue}
              onBack={backToChat}
              reveal={queueReveal}
            />
          )}
          {!(isMobile && simpleQueueOpen) && (
            <ChatContent
              session={session}
              sessionKey={session.key}
              shared={!!shared}
              sharedRef={shared?.agentRef}
              simple={simple}
              minWidth={simple && !isMobile ? 300 : undefined}
              onSend={(content) => sendMessage(session.key, content)}
              onQueue={queueFromChat}
              onCancel={() => cancelTask(session.key)}
            />
          )}
        </div>
      )}

      {/* Send context modal */}
      {showSendContext && !shared && (
        <SendContextModal
          sourceId={session.containerId}
          sourceName={session.containerName}
          messages={session.messages}
          targets={targets}
          onClose={() => setShowSendContext(false)}
        />
      )}

      {/* Flow selector modal */}
      {showFlows && !shared && (
        <FlowSelectorModal
          containerId={session.containerId}
          onClose={() => setShowFlows(false)}
        />
      )}
    </div>
  )
}

// Resolve the agent connection info for the current session
function resolveAgentConnection(
  containerId: string,
  containers: ReturnType<typeof useContainerStore.getState>['containers'],
  localAgents: ReturnType<typeof useLocalAgentStore.getState>['agents'],
  processes: ReturnType<typeof useProcessStore.getState>['processes'],
) {
  const container = containers.find((c) => c.id === containerId)
  if (container && container.web_port) {
    return { host: 'localhost', port: container.web_port, auth: container.web_auth }
  }
  const local = localAgents.find((a) => a.id === containerId)
  if (local) {
    return { host: local.host, port: local.port, auth: local.authToken }
  }
  // Process agents use chatId = `proc-${slug}`
  const proc = processes.find((p) => `proc-${p.slug}` === containerId)
  if (proc && proc.web_port) {
    return { host: 'localhost', port: proc.web_port, auth: proc.web_auth }
  }
  return null
}

// The same object for as long as host, port and auth stay the same — a fresh
// one per render would bust every memo (and upload callback) that depends on it.
function useAgentConnection(containerId: string) {
  const containers = useContainerStore((s) => s.containers)
  const localAgents = useLocalAgentStore((s) => s.agents)
  const processes = useProcessStore((s) => s.processes)
  const conn = resolveAgentConnection(containerId, containers, localAgents, processes)
  const sig = conn ? JSON.stringify([conn.host, conn.port, conn.auth]) : ''
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => conn, [sig])
}

function ChatContent({
  session,
  sessionKey,
  shared = false,
  sharedRef,
  simple = false,
  minWidth,
  onSend,
  onQueue,
  onCancel,
}: {
  session: { containerId: string; containerName: string; messages: ChatMessage[]; connected: boolean; busy: boolean; statusText: string; nextStepOptions?: NextStepOption[]; liveTurnUsage?: TokenUsage | null; closed?: { code: number; reason: string } | null; unreachable?: string | null; asleep?: boolean; waking?: boolean; hasOlder?: boolean; hasNewer?: boolean; holdScrollUntil?: number; reasoning?: ReasoningProfile | null }
  /** This lane's store key — for anything that reads or writes session state.
   *  The agent id lives on `session.containerId`. */
  sessionKey: string
  /** An agent shared with you: chat only — no attachments, no plan card. */
  shared?: boolean
  /** That shared agent's ref: the Files menu lists its files by it. */
  sharedRef?: string
  /** The simple (kiosk) layout: larger text, a greeting with the agent's
   *  skill cards on a fresh chat, and the roomier composer. */
  simple?: boolean
  /** Conversation floor (simple layout): the queue column shrinks before this. */
  minWidth?: number
  onSend: (content: string) => void
  /** Cmd/Ctrl+Enter: the message goes to the lane's queue instead. */
  onQueue: (content: string) => void
  onCancel: () => void
}) {
  const { isMobile } = useIsMobile()
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const conn = useAgentConnection(session.containerId)
  const { attachments, addFiles, removeAttachment, clearAttachments, handlePasteEvent, pasteFromClipboard } = useFileAttachments(conn)
  const loadOlder = useChatStore((s) => s.loadOlder)
  const loadNewer = useChatStore((s) => s.loadNewer)
  const wakeAgent = useChatStore((s) => s.wakeAgent)
  // Asleep for being idle (or waking): what is written waits for the agent.
  const sleeping = !session.connected && (!!session.asleep || !!session.waking)

  // Auto-scroll to bottom when new messages arrive. Deliberately keyed only on
  // message COUNT — not statusText, which changes many times per turn and would
  // fire a fresh smooth-scroll animation on every phase blip ("Working…",
  // "Using X"), a real source of jank during heavy turns.
  // Not while older rows were just prepended or the view jumped to a topic:
  // the count changed, but the reader is looking at the past on purpose.
  useEffect(() => {
    if (Date.now() < (session.holdScrollUntil ?? 0)) return
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.messages.length])

  // Focus input when panel opens
  useEffect(() => {
    inputRef.current?.focus()
  }, [session.containerId])

  // Handle drag-and-drop on the chat area
  const [dragOver, setDragOver] = useState(false)

  // A dialog portalled out of the chat (a skill card's input) still bubbles
  // its React events here: a drop there is the dialog's, not the composer's.
  const fromPortal = (e: React.DragEvent) => !e.currentTarget.contains(e.target as Node)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    if (fromPortal(e)) return
    e.preventDefault()
    e.stopPropagation()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    if (fromPortal(e)) return
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    if (fromPortal(e)) return
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
    if (e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files)
    }
  }, [addFiles])
  // Shared chats take text only: files never reach somebody else's agent. A
  // drop is still swallowed, so the browser doesn't open the file in the tab.
  const dropHandlers = shared
    ? { onDragOver: swallowDrop, onDrop: swallowDrop }
    : { onDragOver: handleDragOver, onDragLeave: handleDragLeave, onDrop: handleDrop }
  const closed = !!session.closed
  // Today's plan allowance is spent: Flight Deck would refuse the message.
  const quotaSpent = useQuotaStore((s) => !!s.status?.blocked && !s.status.exempt)

  // What's in the box (with its uploaded files) goes to `deliver`: sent now,
  // or added to the queue.
  const submit = (deliver: (content: string) => void) => {
    const text = input.trim()
    const uploadedFiles = attachments.filter((a) => a.status === 'uploaded' && a.uploadedPath)
    const hasContent = text || uploadedFiles.length > 0
    if (!hasContent || !(session.connected || sleeping) || quotaSpent) return

    const content = appendAttachmentRefs(text, uploadedFiles)
    deliver(content)
    setInput('')
    // The box grew with what was typed: back to one line.
    if (inputRef.current) inputRef.current.style.height = 'auto'
    clearAttachments()
  }
  const handleSend = () => submit(onSend)

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== 'Enter') return
    // Cmd/Ctrl+Enter queues it, on any keyboard. Plain Enter sends — except
    // on an on-screen keyboard, where it starts a new line (as in any
    // messaging app).
    if (e.metaKey || e.ctrlKey) {
      e.preventDefault()
      submit(onQueue)
    } else if (!e.shiftKey && !touchKeyboard()) {
      e.preventDefault()
      handleSend()
    }
  }

  const handlePasteText = useCallback((text: string) => {
    setInput((prev) => prev + text)
    inputRef.current?.focus()
  }, [])

  // Keep the full transcript — every tool call is shown. The intermediary
  // activity (tools + narration) is grouped into a collapsible panel that
  // streams while the turn runs and collapses to a tools-history summary once
  // it finishes, so showing the whole process doesn't clutter the conversation.
  //
  // Memoized on session.messages: updateSession spreads the session without
  // replacing the messages array, so the many non-message ticks (status,
  // turn_usage) reuse the SAME visibleMessages/groups arrays. That stable
  // identity is what lets the memoized MessageBubble/ActivityGroup children
  // skip re-rendering when only status/usage changed.
  const visibleMessages = useMemo(
    () => session.messages.filter((m) => m.role !== 'tool' || m.tool_name),
    [session.messages],
  )
  const groups = useMemo(() => groupActivity(visibleMessages), [visibleMessages])

  const pendingUploads = attachments.filter((a) => a.status === 'uploading').length
  const hasErrors = attachments.some((a) => a.status === 'error')

  // Simple layout: the agent's skill cards — under a greeting on a fresh chat,
  // and behind the composer's "What can I do?". A card sends its prompt, or
  // first asks for what it needs (text, a file, a picture) in a dialog.
  const skills = useAgentSkills(session.containerId, simple && !shared)
  const [skillsOpen, setSkillsOpen] = useState(false)
  // The File library holds the files of your own process agents (not a
  // run's workers): the Files menu shows a file there.
  const librarySlug = useProcessStore((s) => {
    if (!simple || shared) return undefined
    const p = s.processes.find((x) => `proc-${x.slug}` === session.containerId)
    return p && !isManagedAgent(p.slug, p.description) ? p.slug : undefined
  })
  const [skillsDialog, setSkillsDialog] = useState(false)
  const [skillAsking, setSkillAsking] = useState<AgentSkill | null>(null)
  const canPick = session.connected && !closed && !quotaSpent && !session.busy
  const pickSkill = (skill: AgentSkill) => {
    if (!canPick) return
    if (asksForInput(skill)) setSkillAsking(skill)
    else onSend(skill.prompt)
  }
  // Fresh: nothing said yet, or a new topic (or a cleared context) just began.
  const last = visibleMessages[visibleMessages.length - 1]
  const fresh = simple && session.connected && !session.busy && !session.hasNewer && (
    !last || (last.role === 'system' && (last.topic?.action === 'new' || last.topic?.action === 'clear')))
  const sendDisabled = (!input.trim() && attachments.filter((a) => a.status === 'uploaded').length === 0)
    || (!session.connected && !sleeping) || closed || pendingUploads > 0 || quotaSpent
  const placeholder = closed ? 'Disconnected'
    : sleeping ? (session.waking ? 'Waking up…' : 'Asleep — write to wake it…')
    : !session.connected ? 'Connecting...' : quotaSpent ? 'Daily limit reached'
    : simple ? (skills.available ? 'Ask anything, or try a skill…' : 'Ask anything…')
    : shared || isMobile ? 'Message…' : 'Message, paste image, or drop files...'

  return (
    <div
      className={`flex flex-1 flex-col overflow-hidden ${dragOver ? 'ring-2 ring-inset ring-violet-500/50' : ''}`}
      style={minWidth ? { minWidth } : undefined}
      {...dropHandlers}
    >
      {/* Drag overlay */}
      {dragOver && !shared && (
        <div className="pointer-events-none absolute inset-0 z-50 flex items-center justify-center bg-zinc-950/60">
          <div className="rounded-xl border-2 border-dashed border-violet-500/50 bg-zinc-900/90 px-8 py-6 text-center">
            <Paperclip className="mx-auto h-8 w-8 text-violet-400" />
            <p className="mt-2 text-sm font-medium text-violet-300">Drop files to attach</p>
          </div>
        </div>
      )}

      {/* Plan monitor — pinned above the scroll area so it stays visible
          while messages flow underneath during /plan-execute. */}
      {!shared && <PlanCard containerId={sessionKey} />}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-3">
        <div className={simple ? 'mx-auto w-full max-w-4xl px-4 wide:px-6' : 'px-4'}>
        {/* The loaded window is the newest page of the lane's log. */}
        {session.hasOlder && (
          <div className="mb-3 flex justify-center">
            <button
              onClick={() => loadOlder(sessionKey)}
              className="rounded-md px-2.5 py-1 text-xs text-zinc-500 transition-colors hover:bg-zinc-800/60 hover:text-zinc-300"
            >
              Earlier messages
            </button>
          </div>
        )}
        {visibleMessages.length === 0 && session.connected && !simple && (
          <div className="mt-12 text-center">
            <MessageSquare className="mx-auto h-8 w-8 text-zinc-700" />
            <p className="mt-2 text-sm text-zinc-500">Send a message to start chatting with this agent.</p>
            {!shared && <p className="mt-1 text-xs text-zinc-600">
              {isMobile ? 'Attach files and pictures with the paperclip.' : 'You can drag & drop files or paste images from clipboard.'}
            </p>}
          </div>
        )}

        {!session.connected && !closed && (sleeping ? (
          <div className="mt-12 text-center">
            {session.waking
              ? <Loader2 className="mx-auto h-8 w-8 animate-spin text-zinc-600" />
              : <Moon className="mx-auto h-8 w-8 text-zinc-600" />}
            <p className="mt-2 text-sm text-zinc-400">
              {session.waking ? `Waking ${session.containerName} up…` : `${session.containerName} is asleep`}
            </p>
            {!session.waking && (
              <>
                <p className="mx-auto mt-1 max-w-sm text-xs text-zinc-500">
                  It was stopped after a while without work, to save the server's memory. Write to it and it wakes up — it takes a few seconds.
                </p>
                <button
                  onClick={() => wakeAgent(sessionKey)}
                  className="mt-3 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition-colors hover:bg-zinc-800 hover:text-zinc-100"
                >
                  Wake up now
                </button>
              </>
            )}
          </div>
        ) : (
          <div className="mt-12 text-center" title={session.unreachable || undefined}>
            <AlertCircle className="mx-auto h-8 w-8 text-zinc-700" />
            <p className="mt-2 text-sm text-zinc-500">
              {session.unreachable ? 'Agent unreachable — reconnecting…' : 'Connecting to agent...'}
            </p>
          </div>
        ))}

        {groups.map((grp, gi) => {
          // The live group is the last one (a payment card after it doesn't count).
          const live = session.busy && groups.slice(gi + 1).every((g) => g.kind === 'payment' || g.kind === 'purchase')
          return grp.kind === 'payment' ? (
            <PaymentCard key={`pay-${grp.paymentId}`} id={grp.paymentId!} compact />
          ) : grp.kind === 'purchase' ? (
            <PurchaseCard key={`pur-${grp.paymentId}`} id={grp.paymentId!} compact />
          ) : grp.kind === 'activity' ? (
            <ActivityGroup
              key={`act-${grp.items[0].id}`}
              items={grp.items}
              live={live}
              liveTurnUsage={live ? session.liveTurnUsage ?? null : null}
            />
          ) : (
            <MessageBubble key={grp.items[0].id} message={grp.items[0]} sourceName={session.containerName} agentId={session.containerId} large={simple} />
          )
        })}

        {fresh && (
          <ChatGreeting
            skills={skills}
            disabled={!canPick}
            onPick={pickSkill}
            onMore={() => setSkillsDialog(true)}
          />
        )}

        {/* The window stops before newer rows (a jump, or paging back). */}
        {session.hasNewer && (
          <div className="mb-3 mt-1 flex justify-center">
            <button
              onClick={() => loadNewer(sessionKey)}
              className="rounded-md px-2.5 py-1 text-xs text-zinc-500 transition-colors hover:bg-zinc-800/60 hover:text-zinc-300"
            >
              Later messages
            </button>
          </div>
        )}

        {/* Busy indicator */}
        {session.busy && (
          <div className="mb-3 flex items-center gap-2 text-xs text-zinc-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>{session.statusText || 'Working...'}</span>
          </div>
        )}

        <div ref={messagesEndRef} data-fd-chat-end />
        </div>
      </div>

      {/* Suggested next steps */}
      {session.nextStepOptions && session.nextStepOptions.length > 0 && (
        <NextStepsBar
          options={session.nextStepOptions}
          disabled={!session.connected || session.busy}
          onPick={(action) => onSend(action)}
        />
      )}

      {/* Attachments strip */}
      {attachments.length > 0 && !shared && (
        <div className="border-t border-zinc-800/50 px-3 py-2">
          <div className="flex flex-wrap gap-2">
            {attachments.map((att) => (
              <AttachmentChip key={att.id} attachment={att} onRemove={removeAttachment} />
            ))}
          </div>
          {hasErrors && (
            <p className="mt-1 text-[10px] text-red-400">Some files failed to upload. Remove and retry.</p>
          )}
        </div>
      )}

      {/* Input */}
      {!shared && <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            addFiles(e.target.files)
            e.target.value = ''
          }
        }}
      />}
      {skillsDialog && (
        <SkillsDialog
          skills={skills}
          agentName={session.containerName}
          disabled={!canPick}
          onPick={pickSkill}
          onClose={() => setSkillsDialog(false)}
        />
      )}
      {skillAsking && (
        <SkillInputDialog
          key={skillAsking.title}
          skill={skillAsking}
          conn={conn}
          agentName={session.containerName}
          disabled={!canPick}
          onSend={(content) => { setSkillAsking(null); onSend(content) }}
          onClose={() => setSkillAsking(null)}
        />
      )}
      {simple ? (
        /* The simple layout's composer: the box, and under the text its row
           of attach · paste · "What can I do?" … send. */
        <div className="px-3 pb-3 pt-2 wide:px-6 wide:pb-4">
          <div className="relative mx-auto w-full max-w-4xl rounded-2xl border border-zinc-700 bg-zinc-900 shadow-sm transition-colors focus-within:border-violet-500/50">
            <textarea
              ref={inputRef}
              data-fd-chat-input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={shared ? undefined : handlePasteEvent}
              placeholder={placeholder}
              disabled={(!session.connected && !sleeping) || closed || quotaSpent}
              rows={1}
              className="block max-h-48 min-h-[52px] w-full resize-none rounded-2xl bg-transparent px-4 pb-1 pt-3.5 text-base text-zinc-100 placeholder-zinc-500 focus:outline-none disabled:opacity-40 wide:text-[17px]"
              style={{ height: 'auto', overflow: 'hidden' }}
              onInput={(e) => {
                const el = e.currentTarget
                el.style.height = 'auto'
                el.style.height = Math.min(el.scrollHeight, 192) + 'px'
              }}
            />
            <div className="flex items-center gap-1 px-2 pb-2">
              {!shared && <>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={!session.connected}
                  className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
                  title="Attach files"
                >
                  <Plus className="h-5 w-5" />
                </button>
                <button
                  onClick={() => pasteFromClipboard(handlePasteText)}
                  disabled={!session.connected}
                  className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
                  title="Paste from clipboard"
                >
                  <Clipboard className="h-4 w-4" />
                </button>
              </>}
              <SkillsMenu
                skills={skills}
                open={skillsOpen}
                setOpen={setSkillsOpen}
                disabled={!canPick}
                onPick={pickSkill}
                onMore={() => setSkillsDialog(true)}
                compact={isMobile}
              />
              {!shared && session.reasoning && (
                <EffortMenu
                  agentId={session.containerId}
                  profile={session.reasoning}
                  disabled={!session.connected || closed}
                  compact={isMobile}
                />
              )}
              <ChatFilesMenu
                messages={session.messages}
                conn={conn}
                sharedRef={sharedRef}
                librarySlug={librarySlug}
                busy={session.busy}
                compact={isMobile}
              />
              {/* The keys hint takes what room the row has left, and says
                  as much of itself as fits. */}
              <div className="@container min-w-0 flex-1">
                {!isMobile && (
                  <p className="hidden truncate pr-2 text-right text-xs text-zinc-500 @min-[6rem]:block">
                    Enter to send<span className="hidden @min-[14rem]:inline"> · {QUEUE_CHORD} to queue</span>
                  </p>
                )}
              </div>
              {session.busy ? (
                <button
                  onClick={onCancel}
                  className="flex h-9 items-center gap-1.5 rounded-xl bg-red-600/20 px-3 text-sm font-medium text-red-400 hover:bg-red-600/30"
                >
                  <StopCircle className="h-4 w-4" />
                  Stop
                </button>
              ) : (
                <button
                  onClick={handleSend}
                  disabled={sendDisabled}
                  className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-600 text-white hover:bg-violet-500 disabled:bg-zinc-700 disabled:text-zinc-400"
                  title={pendingUploads > 0 ? 'Waiting for uploads...' : 'Send'}
                  aria-label="Send"
                >
                  {pendingUploads > 0 ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-5 w-5" />}
                </button>
              )}
            </div>
          </div>
        </div>
      ) : (
      <div className="border-t border-zinc-800 p-3">
        <div className="flex items-end gap-2">
          {/* Attach button */}
          {!shared && <div className="flex gap-0.5">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!session.connected}
              className="flex h-[38px] items-center rounded-lg px-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
              title="Attach file"
            >
              <Paperclip className="h-4 w-4" />
            </button>
            <button
              onClick={() => pasteFromClipboard(handlePasteText)}
              disabled={!session.connected}
              className="flex h-[38px] items-center rounded-lg px-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
              title="Paste from clipboard"
            >
              <Clipboard className="h-4 w-4" />
            </button>
          </div>}
          <textarea
            ref={inputRef}
            data-fd-chat-input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={shared ? undefined : handlePasteEvent}
            placeholder={placeholder}
            disabled={(!session.connected && !sleeping) || closed || quotaSpent}
            rows={1}
            className="max-h-32 min-h-[38px] min-w-0 flex-1 resize-none rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-base text-zinc-200 wide:text-sm placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none disabled:opacity-40"
            style={{ height: 'auto', overflow: 'hidden' }}
            onInput={(e) => {
              const el = e.currentTarget
              el.style.height = 'auto'
              el.style.height = Math.min(el.scrollHeight, 128) + 'px'
            }}
          />
          {session.busy ? (
            <button
              onClick={onCancel}
              className="flex h-[38px] items-center gap-1.5 rounded-lg bg-red-600/20 px-3 text-xs font-medium text-red-400 hover:bg-red-600/30"
            >
              <StopCircle className="h-4 w-4" />
              Stop
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={sendDisabled}
              className="flex h-[38px] items-center gap-1.5 rounded-lg bg-violet-600 px-3 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
              title={pendingUploads > 0 ? 'Waiting for uploads...' : `Send (Enter) · ${QUEUE_CHORD} adds it to the queue`}
            >
              {pendingUploads > 0 ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </button>
          )}
        </div>
      </div>
      )}
    </div>
  )
}

function NextStepsBar({
  options,
  disabled,
  onPick,
}: {
  options: NextStepOption[]
  disabled: boolean
  onPick: (action: string) => void
}) {
  return (
    <div className="border-t border-zinc-200 px-4 py-2 dark:border-zinc-800/50">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-500">Suggested next steps</div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((opt, i) => (
          <button
            key={`${i}-${opt.label}`}
            type="button"
            disabled={disabled}
            title={opt.description || opt.action}
            onClick={() => onPick(opt.action)}
            className="rounded-full border border-violet-300 bg-violet-50 px-3 py-1 text-xs font-medium text-violet-700 hover:border-violet-400 hover:bg-violet-100 hover:text-violet-800 dark:border-violet-500/30 dark:bg-violet-500/10 dark:font-normal dark:text-violet-200 dark:hover:bg-violet-500/20 dark:hover:text-violet-100 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Lanes ────────────────────────────────────────────────────────────
//
// Three parallel contexts on ONE agent (docs/queue-lanes-plan.md). Lane A is
// the agent's existing context, so switching to A shows the conversation that
// was always there; B and C are additional rooms that run at the same time.
function LaneStrip({ agentId, activeLane, agentName, trailing }: {
  agentId: string
  activeLane: string
  agentName: string
  /** Phone: the chat's queue button, at the end of the row. */
  trailing?: React.ReactNode
}) {
  const sessions = useChatStore((s) => s.sessions)
  const setActiveLane = useChatStore((s) => s.setActiveLane)
  const newSession = useChatStore((s) => s.newSession)
  const activeSession = sessions.get(laneKey(agentId, activeLane))
  // A shared agent's member has lanes A-C of their own, no automation lane.
  const shared = agentId.startsWith('shared:') || !!activeSession?.shared
  const lanes = shared ? LANES.filter((l) => l !== LANE_AUTO) : LANES

  return (
    <div className="flex items-center gap-1 border-b border-zinc-800 bg-zinc-900/40 px-2 py-1">
      {/* The lanes scroll (without a bar); New topic and Topics stay in view
          on the right. */}
      <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
      {lanes.map((lane) => {
        const s = sessions.get(laneKey(agentId, lane))
        const isActive = lane === activeLane
        const pending = s ? s.queue.filter((q) => q.status === 'pending').length : 0
        const awaiting = s ? s.queue.some((q) => q.awaitingAnswer) : false
        return (
          <button
            key={lane}
            onClick={() => setActiveLane(agentId, lane)}
            title={
              lane === LANE_AUTO ? 'Automated turns (scheduler, autonomy, plans, peer tasks) — results for you also show in lane A'
                : !s ? `Lane ${lane} — not started`
                : awaiting ? `Lane ${lane} — waiting on an answer`
                : s.busy ? `Lane ${lane} — running`
                : `Lane ${lane} — idle`
            }
            className={`flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors wide:px-2 wide:py-1 wide:text-[11px] ${
              isActive
                ? 'bg-violet-600/20 text-violet-700 dark:text-violet-200'
                : 'text-zinc-500 hover:bg-zinc-800/60 hover:text-zinc-300'
            }`}
          >
            {/* State: running (pulsing) · waiting · idle · never opened */}
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                !s ? 'bg-zinc-700'
                  : awaiting ? 'bg-amber-400'
                  : s.busy ? 'animate-pulse bg-violet-400'
                  : 'bg-emerald-400'
              }`}
            />
            {/* A phone has the agent's name in its header already. */}
            <span>{lane === LANE_AUTO ? 'Auto' : lane}<span className="hidden wide:inline"> - {agentName}</span></span>
            {pending > 0 && (
              <span className="rounded-full bg-zinc-800 px-1 text-[9px] text-zinc-400">{pending}</span>
            )}
            {/* Something finished here while you were looking elsewhere. */}
            {s?.unread && !isActive && (
              <span className="h-1.5 w-1.5 rounded-full bg-sky-400" />
            )}
          </button>
        )
      })}
      </div>
      {/* A fresh topic on the lane you're looking at (the agent's /new): its
          context starts over, the transcript keeps going with a divider. */}
      {!shared && <button
        onClick={() => activeSession && newSession(activeSession.key)}
        disabled={!activeSession || activeSession.busy}
        title="Start a new topic — the agent begins with a fresh context; earlier messages stay"
        className={`${TOPIC_BTN} disabled:opacity-40 disabled:hover:border-zinc-700 disabled:hover:bg-transparent`}
      >
        <Plus className="h-4 w-4" />
        New<span className="hidden wide:inline"> topic</span>
      </button>}
      {!shared && activeSession && <TopicsButton sessionKey={activeSession.key} />}
      {trailing}
    </div>
  )
}

// New topic and Topics on the lane row: buttons, a size up from the lane tabs.
const TOPIC_BTN = 'flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg border border-zinc-700 px-3 py-1.5 text-sm font-medium text-zinc-300 transition-colors hover:border-zinc-600 hover:bg-zinc-800 hover:text-zinc-100'

/** Opens the lane's Topics list: one row per divider in its transcript,
 *  newest first. A row jumps to its divider; Resume puts the agent's
 *  context back on that topic's session. */
function TopicsButton({ sessionKey }: { sessionKey: string }) {
  const [open, setOpen] = useState(false)
  // Where the button is when it was clicked: the popover hangs under it.
  const [anchor, setAnchor] = useState<DOMRect | null>(null)
  return (
    <>
      <button
        onClick={(e) => {
          setAnchor(e.currentTarget.getBoundingClientRect())
          setOpen((v) => !v)
        }}
        title="Topics — jump back to or resume an earlier topic"
        aria-label="Topics"
        className={`${TOPIC_BTN} ${open ? 'border-violet-500/50 bg-zinc-800 text-zinc-100' : ''}`}
      >
        <History className="h-4 w-4" />
        <span className="hidden wide:inline">Topics</span>
      </button>
      {open && <TopicsMenu sessionKey={sessionKey} anchor={anchor} onClose={() => setOpen(false)} />}
    </>
  )
}

function TopicsMenu({ sessionKey, anchor, onClose }: {
  sessionKey: string
  anchor: DOMRect | null
  onClose: () => void
}) {
  const { isMobile } = useIsMobile()
  const [topics, setTopics] = useState<TopicInfo[] | null>(null)
  const agentSessionId = useChatStore((s) => s.sessions.get(sessionKey)?.agentSessionId ?? '')
  const busy = useChatStore((s) => s.sessions.get(sessionKey)?.busy ?? false)
  const jumpToTopic = useChatStore((s) => s.jumpToTopic)
  const resumeTopic = useChatStore((s) => s.resumeTopic)
  const renameTopic = useChatStore((s) => s.renameTopic)
  // The row being renamed (by its log row id, else its message id) and the draft.
  const [editing, setEditing] = useState<{ id: string; value: string } | null>(null)

  useEffect(() => {
    let alive = true
    fetchTopics(sessionKey).then((list) => { if (alive) setTopics(list) })
    return () => { alive = false }
  }, [sessionKey])

  const rowKey = (t: TopicInfo) => (t.rowId !== undefined ? `r${t.rowId}` : (t.messageId ?? ''))
  const commitRename = (t: TopicInfo) => {
    if (!editing) return
    const value = editing.value.replace(/\s+/g, ' ').trim().slice(0, 120)
    setEditing(null)
    if (value === (t.title || '')) return
    renameTopic(sessionKey, { rowId: t.rowId, messageId: t.messageId }, value)
    setTopics((prev) => prev ? prev.map((x) => (x === t ? { ...x, title: value } : x)) : prev)
  }

  // Newest first; the current topic is the newest row on the agent's session.
  const rows = useMemo(() => (topics ? [...topics].reverse() : null), [topics])
  const currentIdx = rows ? rows.findIndex((t) => !!t.agentSessionId && t.agentSessionId === agentSessionId) : -1

  const jump = async (t: TopicInfo) => {
    let id: string | null = t.messageId ?? null
    if (!id && t.rowId !== undefined) id = await jumpToTopic(sessionKey, t.rowId)
    onClose()
    if (id) scrollToMessage(id)
  }
  const resume = (t: TopicInfo) => {
    resumeTopic(sessionKey, t.agentSessionId)
    onClose()
  }

  const rect = anchor
  const style: React.CSSProperties = isMobile
    ? { left: 0, right: 0, bottom: 0, maxHeight: '70dvh' }
    : {
      top: (rect?.bottom ?? 48) + 6,
      right: Math.max(8, window.innerWidth - (rect?.right ?? window.innerWidth)),
      width: 340, maxHeight: '60vh',
    }

  return createPortal(
    <div className="fixed inset-0 z-[70]" onClick={onClose}>
      <div
        className={`fixed flex flex-col overflow-hidden border border-zinc-700 bg-zinc-900 shadow-xl ${
          isMobile ? 'rounded-t-2xl pb-[env(safe-area-inset-bottom)]' : 'rounded-lg'
        }`}
        style={style}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-zinc-400">Topics</span>
          <button onClick={onClose} aria-label="Close" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto py-1">
          {rows === null && <p className="px-3 py-3 text-xs text-zinc-500">Loading…</p>}
          {rows !== null && rows.length === 0 && (
            <p className="px-3 py-3 text-xs text-zinc-500">
              No topics yet. "+ New topic" starts one; what was said before stays above its divider.
            </p>
          )}
          {rows?.map((t, i) => {
            const current = i === currentIdx
            const canResume = !!t.agentSessionId && t.agentSessionId !== agentSessionId
            const isEditing = editing?.id === rowKey(t)
            return (
              <div key={t.rowId ?? t.messageId ?? i} className="flex items-center gap-1 px-2 py-0.5">
                {isEditing ? (
                  <input
                    autoFocus
                    value={editing.value}
                    maxLength={120}
                    placeholder={topicTitle(t)}
                    aria-label="Topic title"
                    onChange={(e) => setEditing({ id: rowKey(t), value: e.target.value })}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') { e.preventDefault(); commitRename(t) }
                      if (e.key === 'Escape') { e.preventDefault(); setEditing(null) }
                    }}
                    onBlur={() => commitRename(t)}
                    className="min-w-0 flex-1 rounded-md border border-violet-500/50 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 outline-none"
                  />
                ) : (
                  <button
                    onClick={() => jump(t)}
                    title="Jump to this topic"
                    className="min-w-0 flex-1 rounded-md px-2 py-1.5 text-left transition-colors hover:bg-zinc-800/60"
                  >
                    <div className="truncate text-sm text-zinc-200">{topicTitle(t)}</div>
                    <div className="truncate text-[11px] text-zinc-500">
                      {fmtTopicWhen(t.startedAt)} · {t.messageCount} message{t.messageCount === 1 ? '' : 's'}
                      {current && <span className="ml-1.5 rounded bg-violet-600/20 px-1 text-[10px] text-violet-700 dark:text-violet-300">current</span>}
                    </div>
                  </button>
                )}
                {!isEditing && (
                  <button
                    onClick={() => setEditing({ id: rowKey(t), value: t.title || '' })}
                    title="Rename this topic"
                    aria-label="Rename topic"
                    className="shrink-0 rounded-md p-1.5 text-zinc-600 transition-colors hover:bg-zinc-800/60 hover:text-zinc-200"
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </button>
                )}
                {canResume && !isEditing && (
                  <button
                    onClick={() => resume(t)}
                    disabled={busy}
                    title={busy ? 'Wait for the agent to finish, then resume' : 'Resume: the agent picks this topic\'s context back up'}
                    aria-label="Resume topic"
                    className="shrink-0 rounded-md p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800/60 hover:text-zinc-200 disabled:opacity-40 disabled:hover:bg-transparent"
                  >
                    <Undo2 className="h-4 w-4" />
                  </button>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>,
    document.body,
  )
}

function topicTitle(t: TopicInfo): string {
  if (t.title) return t.title
  const first = t.firstUserMessage.replace(/\s+/g, ' ').trim()
  if (first) return first.length > 72 ? `${first.slice(0, 72)}…` : first
  if (t.action === 'clear') return 'Context cleared'
  if (t.action === 'resume') return 'Back on an earlier topic'
  return 'New topic'
}

/** Where a turn came from, as the person would say it. */
function viaLabel(via: string): string {
  switch (via) {
    case 'whatsapp': return 'WhatsApp'
    case 'telegram': return 'Telegram'
    case 'device': return 'stick'
    case 'glasses': return 'glasses'
    default: return via
  }
}

function fmtTopicWhen(ts: string): string {
  if (!ts) return ''
  try {
    const d = new Date(ts)
    const sameDay = d.toDateString() === new Date().toDateString()
    return sameDay
      ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : d.toLocaleDateString([], { day: 'numeric', month: 'short' })
  } catch {
    return ''
  }
}

function scrollToMessage(id: string) {
  // After a jump the loaded window may have just been replaced: let it paint.
  window.setTimeout(() => {
    document.getElementById(`fd-msg-${id}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' })
  }, 60)
}

// The queue beside the chat, in one of three shapes:
//  - 'column'  (simple layout): a column left of the chat, resizable from its
//              right edge; the layout shows files and datastore elsewhere.
//  - 'phone'   (simple layout on a phone): a view of its own, in place of the chat.
//  - 'sidebar' (full layout, fullscreen chat): the agent's files and datastore
//              above the queue (unless queueOnly), with draggable dividers
//              that never squeeze the queue under its minimum. The queue
//              collapses to a one-line bar under them (to a slim rail when
//              it's alone). Every size persists to localStorage.
// It lives here, not in queue/, to reach useAgentConnection. Memoized: the
// chat around it re-renders with every streamed token, and its props (ids,
// flags, stable callbacks) don't change with them.
const QueueSidebar = memo(function QueueSidebar({ sessionKey, agentId, queueOnly = false, shared = false, variant, onGoToChat, onHide, onBack, reveal }: {
  sessionKey: string
  agentId: string
  queueOnly?: boolean
  shared?: boolean
  variant: QueueVariant
  onGoToChat: (o: { messageId?: string; focusComposer?: boolean }) => void
  onHide: () => void
  onBack: () => void
  /** Changes when the chat box queued a message: a folded queue unfolds. */
  reveal: number
}) {
  const conn = useAgentConnection(agentId)
  const width = usePersistedSize('fd:queue-sidebar-width', 288, 220, 640, 'x')
  const fsWidth = usePersistedSize('fd:queue-fullscreen-width', fullscreenQueueWidthDefault(), 240, 640, 'x')
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem(QUEUE_COLLAPSED_KEY) === '1' } catch { return false }
  })
  const setRail = (v: boolean) => {
    setCollapsed(v)
    try { localStorage.setItem(QUEUE_COLLAPSED_KEY, v ? '1' : '0') } catch { /* private mode */ }
  }
  // (The stored setting already says unfolded: unfoldFullscreenQueue.)
  const [revealSeen, setRevealSeen] = useState(reveal)
  if (revealSeen !== reveal) {
    setRevealSeen(reveal)
    setCollapsed(false)
  }
  // The files and datastore panes give way before the queue drops under
  // ~320px: they share what the column has left over (each keeps 96px).
  const asideRef = useRef<HTMLElement>(null)
  const [asideH, setAsideH] = useState(0)
  const rail = variant === 'sidebar' && collapsed && queueOnly
  const sidebarOpen = variant === 'sidebar' && !rail
  useEffect(() => {
    const el = asideRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(([entry]) => setAsideH(entry.contentRect.height))
    ro.observe(el)
    return () => ro.disconnect()
  }, [sidebarOpen])
  // A collapsed queue leaves the panes all but its one-line bar.
  const avail = Math.max(192, asideH - (collapsed ? 48 : 328))
  const filesH = usePersistedSize('fd:queue-sidebar-files-height', 260, 96, 900, 'y', 'forward', asideH ? avail - 96 : undefined)
  const dsH = usePersistedSize('fd:queue-sidebar-datastore-height', 200, 96, 900, 'y', 'forward', asideH ? avail - filesH.size : undefined)
  const signal = useQueueSignal(sessionKey)
  const progress = useQueueProgress(sessionKey)

  const panel = (
    <QueuePanel
      sessionKey={sessionKey}
      agentId={agentId}
      shared={shared}
      conn={conn}
      variant={variant}
      onGoToChat={onGoToChat}
      onHide={variant === 'column' ? onHide : undefined}
      onBack={variant === 'phone' ? onBack : undefined}
      onCollapse={variant === 'sidebar' ? () => setRail(true) : undefined}
    />
  )

  if (variant === 'phone') {
    return <aside className="relative flex w-full flex-1 flex-col bg-zinc-950/40">{panel}</aside>
  }

  if (variant === 'column') {
    return (
      <aside
        className="relative flex min-w-[180px] shrink flex-col border-r border-zinc-800 bg-zinc-950/40"
        style={{ width: width.size }}
      >
        {panel}
        <ColResizeHandle size={width} label="Resize the queue" />
      </aside>
    )
  }

  const pct = progress.total ? (progress.done / progress.total) * 100 : 0
  // Folded away, the queue still says when it needs you (as the toggle does).
  const expandLabel = `Expand queue · ${signal.left} left${signal.tone === 'needs' ? ' · needs you' : ''}`

  // Fullscreen, collapsed, queue alone: a rail that keeps the badge and the progress.
  if (rail) {
    const label = expandLabel
    return (
      <aside
        onClick={() => setRail(false)}
        className="flex w-11 shrink-0 cursor-pointer flex-col items-center gap-3 border-r border-zinc-800 bg-zinc-950/40 py-3"
      >
        <button
          type="button"
          aria-label={label}
          title={label}
          className={`relative rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100 ${FOCUS}`}
        >
          <ListTodo className="h-4 w-4" />
          <QueueBadge signal={signal} />
        </button>
        <div className="relative h-24 w-1 overflow-hidden rounded-full bg-zinc-800" aria-hidden>
          <div className="absolute bottom-0 left-0 w-full rounded-full bg-emerald-500" style={{ height: `${pct}%` }} />
        </div>
        {progress.working && <Loader2 className="h-3.5 w-3.5 text-violet-600 motion-safe:animate-spin dark:text-violet-400" />}
      </aside>
    )
  }

  return (
    <aside
      ref={asideRef}
      className="relative flex shrink-0 flex-col border-r border-zinc-800 bg-zinc-950/40"
      style={{ width: fsWidth.size }}
    >
      {!queueOnly && (
        <>
          {/* Top: files */}
          <div className="min-h-0 shrink-0 overflow-hidden" style={{ height: filesH.size }}>
            <AgentFilesPanel containerId={agentId} />
          </div>
          <RowResizeHandle size={filesH} label="Resize the files pane" />
          {/* Middle: datastore tables (the rest, while the queue is folded) */}
          <div
            className={`min-h-0 overflow-hidden ${collapsed ? 'flex-1' : 'shrink-0'}`}
            style={collapsed ? undefined : { height: dsH.size }}
          >
            <AgentDatastorePanel containerId={agentId} />
          </div>
          {!collapsed && <RowResizeHandle size={dsH} label="Resize the datastore pane" />}
        </>
      )}
      {/* Bottom: queue, or its one-line bar */}
      {collapsed ? (
        <button
          type="button"
          onClick={() => setRail(false)}
          aria-label={expandLabel}
          title={expandLabel}
          className={`flex h-12 shrink-0 items-center gap-2.5 border-t border-zinc-800 px-3 text-left text-zinc-300 transition-colors hover:bg-zinc-800/60 hover:text-zinc-100 ${FOCUS}`}
        >
          <span className="relative">
            <ListTodo className="h-4 w-4" />
            <QueueBadge signal={signal} />
          </span>
          <span className="text-sm font-semibold">Queue</span>
          {progress.total > 0 && (
            <span className="relative h-1 min-w-0 flex-1 overflow-hidden rounded-full bg-zinc-800" aria-hidden>
              <span className="absolute inset-y-0 left-0 rounded-full bg-emerald-500" style={{ width: `${pct}%` }} />
            </span>
          )}
          {progress.working && <Loader2 className="h-3.5 w-3.5 shrink-0 text-violet-600 motion-safe:animate-spin dark:text-violet-400" />}
          <ChevronUp className="ml-auto h-4 w-4 shrink-0 text-zinc-500" />
        </button>
      ) : (
        <div className="min-h-0 flex-1">{panel}</div>
      )}
      <ColResizeHandle size={fsWidth} label="Resize the queue" />
    </aside>
  )
})

const QUEUE_COLLAPSED_KEY = 'fd:fullscreen-queue-collapsed'

/** The fullscreen queue shows unfolded from now on, as if its bar was clicked. */
function unfoldFullscreenQueue() {
  try { localStorage.setItem(QUEUE_COLLAPSED_KEY, '0') } catch { /* private mode */ }
}

// The fullscreen queue's own width starts from the simple column's, once.
let _fsQueueWidth: number | null = null
function fullscreenQueueWidthDefault(): number {
  if (_fsQueueWidth === null) {
    let v = NaN
    try { v = Number(localStorage.getItem('fd:queue-sidebar-width')) } catch { /* private mode */ }
    _fsQueueWidth = v >= 240 && v <= 640 ? v : 320
  }
  return _fsQueueWidth
}

type PersistedSize = ReturnType<typeof usePersistedSize>

/** The right-edge handle: drag (mouse, touch, pen) or arrow keys (Shift: ×4). */
function ColResizeHandle({ size, label }: { size: PersistedSize; label: string }) {
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={Math.round(size.size)}
      aria-valuemin={size.min}
      aria-valuemax={size.max}
      tabIndex={0}
      title="Drag to resize"
      onPointerDown={size.onResizeStart}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 64 : 16
        if (e.key === 'ArrowLeft') { e.preventDefault(); size.nudge(-step) }
        else if (e.key === 'ArrowRight') { e.preventDefault(); size.nudge(step) }
      }}
      className="group absolute right-0 top-0 z-20 -mr-1 h-full w-2 cursor-col-resize touch-none focus-visible:outline-none"
    >
      <div className="mx-auto h-full w-px bg-transparent transition-colors group-hover:bg-violet-500/50 group-focus-visible:bg-violet-500" />
    </div>
  )
}

/** A divider between the sidebar's panes: drag, or arrow keys up/down. */
function RowResizeHandle({ size, label }: { size: PersistedSize; label: string }) {
  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      aria-label={label}
      aria-valuenow={Math.round(size.size)}
      aria-valuemin={size.min}
      aria-valuemax={size.max}
      tabIndex={0}
      title="Drag to resize"
      onPointerDown={size.onResizeStart}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 64 : 16
        if (e.key === 'ArrowUp') { e.preventDefault(); size.nudge(-step) }
        else if (e.key === 'ArrowDown') { e.preventDefault(); size.nudge(step) }
      }}
      className="group relative z-10 -my-[3px] h-2 shrink-0 cursor-row-resize touch-none focus-visible:outline-none"
    >
      <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-zinc-800 transition-colors group-hover:bg-violet-500/50 group-focus-visible:bg-violet-500" />
    </div>
  )
}

/** The queue toggle's (and the rail's) count: amber when the queue needs
 *  you, violet while a task runs, grey while tasks wait. */
function QueueBadge({ signal }: { signal: { left: number; tone: SignalTone } }) {
  if (signal.tone === 'none') return null
  const cls = signal.tone === 'needs' ? 'bg-amber-500 text-amber-950'
    : signal.tone === 'running' ? 'bg-violet-600 text-white'
    : 'bg-zinc-700 text-zinc-200'
  const text = signal.left > 99 ? '99+' : signal.left > 0 ? String(signal.left) : signal.tone === 'needs' ? '!' : ''
  return (
    <span
      aria-hidden
      className={`pointer-events-none absolute -right-1 -top-1 flex items-center justify-center rounded-full font-semibold tabular-nums ring-2 ring-zinc-950 ${
        text ? 'h-4 min-w-4 px-1 text-[10px]' : 'h-2 w-2'
      } ${cls}`}
    >
      {text}
    </span>
  )
}

/** Convert agent-local image paths in message content to URLs proxied through Flight Deck. */
function processImagePaths(content: string, agentHost?: string, agentPort?: number, agentAuth?: string): string {
  if (!agentPort) return content

  const proxyUrl = (filePath: string): string => agentFileViewUrl(filePath, agentHost, agentPort, agentAuth)

  // Rewrite agent-local paths inside markdown image syntax: ![alt](path)
  // Mirrors captain-claw/computer.js: handles file:///, saved/, output/, and absolute /paths.
  // Leaves http(s)://, data:, blob:, and already-proxied URLs alone.
  let out = content.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (match, alt, src) => {
    if (/^(https?:|data:|blob:)/i.test(src)) return match
    if (src.startsWith('/fd/')) return match
    let filePath = src
    if (/^file:\/\/\//i.test(src)) {
      filePath = '/' + src.replace(/^file:\/\/\//i, '')
    } else if (/^(saved\/|output\/|\/)/.test(src)) {
      filePath = src
    } else {
      return match
    }
    return `![${alt}](${proxyUrl(filePath)})`
  })

  // Also handle bare image paths on their own line (existing behavior).
  out = out.replace(
    /^([`*]*)(\/?(?:\/[\w.@: -]+)+\.(?:png|jpg|jpeg|gif|webp|bmp|svg))([`*]*)$/gm,
    (_m, _pre, filePath) => `![](${proxyUrl(filePath)})`,
  )

  return out
}

// memo: finished messages keep a stable object identity in the store (addMessage
// appends, updateSession spreads without touching the messages array, approval
// resolution replaces the one message with a new object), so a default shallow
// compare on {message, sourceName, agentId} skips every unchanged bubble when a
// later turn streams — no markdown re-parse. This is the heavy-chat fix.
// `large` — the simple layout's chat: the conversation in bigger type.
// A message of the owner's that reads like a correction, or something to remember: offered as a lesson.
const LESSON_HINT = new RegExp(
  "^\\s*(no\\b|nope\\b|wrong\\b|not quite|that'?s not|that is not|actually\\b|incorrect|ne\\b|nije\\b|krivo|pogrešno|zapravo)"
  + "|\\b(from now on|next time|in (the )?future|remember (that|to)|keep in mind|always|never|don'?t ever|ubuduće|od sada|uvijek|nikad|zapamti|imaj na umu)\\b",
  'i')
const looksLikeLesson = (text: string) => text.length >= 12 && text.length <= 1500 && !text.trimStart().startsWith('/')
  && LESSON_HINT.test(text)

/** The agent's answer just before *messageId* in its chat (what the owner corrects). */
function replyBefore(messageId: string): string {
  for (const s of useChatStore.getState().sessions.values()) {
    const at = s.messages.findIndex((m) => m.id === messageId)
    if (at < 0) continue
    for (let i = at - 1; i >= 0; i--) if (s.messages[i].role === 'assistant') return s.messages[i].content || ''
    return ''
  }
  return ''
}

const MessageBubble = memo(function MessageBubble({ message, sourceName, agentId, large = false }: { message: ChatMessage; sourceName?: string; agentId?: string; large?: boolean }) {
  const bodyText = large ? 'text-base leading-relaxed' : 'text-sm'
  const [toolExpanded, setToolExpanded] = useState(false)
  const [lessonFrom, setLessonFrom] = useState<{ said: string; reply: string } | null>(null)
  const [copied, setCopied] = useState(false)
  const [showForward, setShowForward] = useState(false)

  const containers = useContainerStore((s) => s.containers)
  const localAgents = useLocalAgentStore((s) => s.agents)
  const processes = useProcessStore((s) => s.processes)
  const { pin, isPinned } = usePinnedStore()
  const addClipEntry = useClipboardStore((s) => s.addEntry)
  const pinned = agentId ? isPinned(agentId, message.content) : false
  const openChat = useChatStore((s) => s.openChat)
  const sendMessageToAgent = useChatStore((s) => s.sendMessage)

  // Resolve agent connection for image proxying
  const agentConn = useMemo(() => {
    if (!agentId) return null
    const c = containers.find((x) => x.id === agentId)
    if (c?.web_port) return { host: 'localhost', port: c.web_port, auth: c.web_auth || '' }
    const l = localAgents.find((x) => x.id === agentId)
    if (l) return { host: l.host, port: l.port, auth: l.authToken || '' }
    const p = processes.find((x) => `proc-${x.slug}` === agentId)
    if (p?.web_port) return { host: 'localhost', port: p.web_port, auth: p.web_auth || '' }
    return null
  }, [agentId, containers, localAgents, processes])

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(message.content || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }, [message.content])

  const forwardTargets = useMemo(() => {
    const targets: { id: string; name: string; host: string; port: number; auth: string }[] = []
    for (const c of containers) {
      if (c.status === 'running' && c.web_port && c.id !== agentId) {
        targets.push({ id: c.id, name: c.agent_name || c.name, host: 'localhost', port: c.web_port, auth: c.web_auth || '' })
      }
    }
    for (const p of processes) {
      if (p.status === 'running' && p.web_port && `proc-${p.slug}` !== agentId) {
        targets.push({ id: `proc-${p.slug}`, name: p.name, host: 'localhost', port: p.web_port, auth: p.web_auth || '' })
      }
    }
    for (const a of localAgents) {
      if (a.status === 'online' && a.id !== agentId) {
        targets.push({ id: a.id, name: a.name, host: a.host, port: a.port, auth: a.authToken || '' })
      }
    }
    return targets
  }, [containers, processes, localAgents, agentId])

  const handleForward = useCallback(async (target: typeof forwardTargets[0]) => {
    const src = sourceName || 'another agent'
    const role = message.role === 'user' ? 'User' : 'Assistant'
    // Look up forwarding task from target agent's card config
    const container = containers.find((c) => c.id === target.id)
    const process = target.id.startsWith('proc-') ? processes.find((p) => `proc-${p.slug}` === target.id) : undefined
    const localAgent = localAgents.find((a) => a.id === target.id)
    const fwdTask = (container
      ? useContainerStore.getState().getForwardingTask(target.id)
      : process
        ? useProcessStore.getState().getForwardingTask(target.id)
        : localAgent?.forwardingTask
    ) || 'Review the context above and provide your analysis.'
    const composed = `--- Context from "${src}" ---\n\n**${role}:**\n${message.content}\n\n--- End of context ---\n\n${fwdTask}`
    openChat(target.id, target.name, target.host, target.port, target.auth)
    await new Promise((r) => setTimeout(r, 1000))
    sendMessageToAgent(target.id, composed)
    setShowForward(false)
  }, [message, sourceName, containers, processes, localAgents, openChat, sendMessageToAgent])

  const handlePin = useCallback(() => {
    if (!agentId || pinned) return
    pin({ agentId, agentName: sourceName || 'Agent', content: message.content,
      role: message.role === 'rephrase' || message.role === 'image' || message.role === 'html_file'
        ? 'assistant' : message.role, model: message.model })
  }, [agentId, pinned, pin, sourceName, message])

  const handleClip = useCallback(() => {
    addClipEntry(message.content || '', sourceName || 'Agent')
  }, [addClipEntry, message.content, sourceName])

  // Action buttons (copy + pin + clipboard + forward)
  const actionButtons = (align: 'left' | 'right') => (
    <div className={`flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity ${
      align === 'right' ? 'justify-end' : ''
    }`}>
      <button
        onClick={handleCopy}
        className="rounded p-0.5 text-zinc-600 hover:bg-zinc-700/50 hover:text-zinc-400"
        title="Copy"
      >
        {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
      </button>
      {agentId && !pinned && (
        <button
          onClick={handlePin}
          className="rounded p-0.5 text-zinc-600 hover:bg-zinc-700/50 hover:text-amber-400"
          title="Pin message"
        >
          <Pin className="h-3 w-3" />
        </button>
      )}
      {agentId && pinned && (
        <Pin className="h-3 w-3 text-amber-400 mx-0.5" />
      )}
      <button
        onClick={handleClip}
        className="rounded p-0.5 text-zinc-600 hover:bg-zinc-700/50 hover:text-cyan-400"
        title="Add to shared clipboard"
      >
        <ClipboardList className="h-3 w-3" />
      </button>
      {forwardTargets.length > 0 && (
        <div className="relative">
          <button
            onClick={() => setShowForward(!showForward)}
            className="rounded p-0.5 text-zinc-600 hover:bg-zinc-700/50 hover:text-zinc-400"
            title="Send to another agent"
          >
            <Forward className="h-3 w-3" />
          </button>
          {showForward && (
            <div className={`absolute ${align === 'right' ? 'right-0' : 'left-0'} top-full mt-1 z-30 min-w-[160px] rounded-lg border border-zinc-700 bg-zinc-900 py-1 shadow-xl`}>
              <div className="px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Send to
              </div>
              {forwardTargets.map((t) => (
                <button
                  key={t.id}
                  onClick={() => handleForward(t)}
                  className="flex w-full items-center gap-2 px-2 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800"
                >
                  <Send className="h-3 w-3 text-zinc-500" />
                  {t.name}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )

  // The owner's own agent (a process it runs): what they say to it can be taught as a lesson. Someone's
  // shared agent: a member suggests it to its owner instead.
  const ownAgent = message.role === 'user' && agentId
    ? processes.find((p) => `proc-${p.slug}` === agentId) : undefined
  const sharedRef = message.role === 'user' && agentId?.startsWith('shared:') ? agentId.slice('shared:'.length) : ''
  const teachable = !!ownAgent || !!sharedRef
  const saveLesson = () => setLessonFrom({ said: message.content || '', reply: replyBefore(message.id) })

  if (message.role === 'user') {
    return (
      <div id={`fd-msg-${message.id}`} className="group mb-3 flex flex-col items-end gap-0.5">
        <div className={`max-w-[85%] rounded-br-sm bg-violet-600/20 ${large ? 'rounded-2xl px-4 py-3' : 'rounded-xl px-3.5 py-2.5'}`}>
          <div className={`fd-markdown ${bodyText} text-zinc-200`}>
            <MarkdownMessage content={processImagePaths(message.content, agentConn?.host, agentConn?.port, agentConn?.auth)} />
          </div>
          <span className="mt-1 block text-right text-[10px] text-zinc-500">
            {message.via && <span className="mr-1.5 text-zinc-400">via {message.sourceLabel || viaLabel(message.via)}</span>}
            {formatTime(message.timestamp)}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {teachable && looksLikeLesson(message.content || '') && (
            <button onClick={saveLesson}
              className="inline-flex items-center gap-1 rounded-full border border-violet-500/30 px-2 py-0.5 text-[11px] text-violet-700 hover:bg-violet-500/10 dark:text-violet-300">
              <GraduationCap className="h-3 w-3" /> {ownAgent ? 'Save as a lesson?' : 'Suggest as a lesson?'}
            </button>
          )}
          {teachable && !looksLikeLesson(message.content || '') && (
            <button onClick={saveLesson} title={ownAgent ? 'Save as a lesson' : 'Suggest as a lesson to its owner'}
              className="rounded p-0.5 text-zinc-600 opacity-0 transition-opacity hover:bg-zinc-700/50 hover:text-violet-400 group-hover:opacity-100">
              <GraduationCap className="h-3 w-3" />
            </button>
          )}
          {actionButtons('right')}
        </div>
        {lessonFrom && ownAgent && (
          <SaveLessonDialog agent={{ slug: ownAgent.slug, name: ownAgent.name }} said={lessonFrom.said} reply={lessonFrom.reply}
            onClose={() => setLessonFrom(null)} />
        )}
        {lessonFrom && sharedRef && (
          <SuggestLessonDialog agentRef={sharedRef} agentName={sourceName || 'This agent'}
            ownerName={useSharedAgentStore.getState().agents.find((a) => a.agent_ref === sharedRef)?.owner_name ?? ''}
            said={lessonFrom.said} reply={lessonFrom.reply} onClose={() => setLessonFrom(null)} />
        )}
      </div>
    )
  }

  if (message.role === 'tool') {
    return (
      <div className="mb-2">
        <button
          onClick={() => setToolExpanded(!toolExpanded)}
          className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-400"
        >
          <Wrench className="h-3 w-3" />
          {message.peer_name && (
            <span className="font-medium text-sky-600 dark:text-sky-400">{message.peer_name}</span>
          )}
          <span className="font-medium">{message.tool_name}</span>
          {toolExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        </button>
        {toolExpanded && message.content && (
          <pre className="ml-6 mt-1 max-h-40 overflow-auto rounded-md bg-zinc-900/80 p-2 text-xs text-zinc-400 font-mono">
            {message.content.slice(0, 2000)}
            {message.content.length > 2000 && '\n...truncated'}
          </pre>
        )}
      </div>
    )
  }

  if (message.role === 'system') {
    // A topic divider: the agent's context moved (new topic, back on an
    // earlier one) or was reset; the transcript continues underneath.
    if (message.topic) {
      const action = message.topic.action
      const Icon = action === 'resume' ? Undo2 : action === 'clear' ? Eraser : Sparkles
      return (
        <div id={`fd-msg-${message.id}`} data-topic={action} className="my-4 flex items-center gap-3 text-[11px] text-zinc-500">
          <div className="h-px flex-1 bg-zinc-800" />
          <span className="flex min-w-0 items-center gap-1.5 whitespace-nowrap font-medium">
            <Icon className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
            <span className="truncate">
              {message.content || 'New topic'}
              {/* A session named like its title (a scheduled task's) says it once. */}
              {message.topic.title && !(message.content || '').endsWith(message.topic.title) && (
                <span className="text-zinc-400"> · {message.topic.title}</span>
              )}
            </span>
            <span className="shrink-0 font-normal text-zinc-600">{formatTime(message.timestamp)}</span>
          </span>
          <div className="h-px flex-1 bg-zinc-800" />
        </div>
      )
    }
    // Live between-step narration blurb — subtle, in the flow.
    if (message.narration) {
      return (
        <div className="mb-1.5 flex items-start gap-1.5 pl-1 pr-6 text-xs italic text-zinc-500">
          <CircleDot className="mt-0.5 h-3 w-3 shrink-0 text-zinc-600" />
          <div className="fd-markdown min-w-0 flex-1">
            <MarkdownMessage content={message.content} />
          </div>
        </div>
      )
    }
    // Approval request with approve/deny buttons
    if (message.approval_request_id && !message.approval_resolved) {
      const respondToApproval = useChatStore.getState().respondToApproval
      return (
        <div className="mb-3 flex justify-center">
          <div className="max-w-[85%] rounded-lg border border-amber-600/30 dark:border-amber-500/30 bg-amber-50 dark:bg-amber-500/10 px-4 py-3 shadow-sm">
            <p className="whitespace-pre-wrap text-xs text-amber-900 dark:text-amber-200 mb-2.5">{message.content}</p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => agentId && respondToApproval(agentId, message.approval_request_id!, true)}
                className="rounded-md bg-emerald-600 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-700 dark:bg-emerald-600/80 dark:hover:bg-emerald-600"
              >
                Approve
              </button>
              <button
                onClick={() => agentId && respondToApproval(agentId, message.approval_request_id!, false)}
                className="rounded-md bg-zinc-200 px-3 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-600"
              >
                Deny
              </button>
            </div>
          </div>
        </div>
      )
    }
    // Resolved approval
    if (message.approval_request_id && message.approval_resolved) {
      return (
        <div className="mb-3 flex justify-center">
          <div className="max-w-[85%] rounded-lg border border-zinc-200 bg-white px-3 py-2 dark:border-zinc-700/50 dark:bg-zinc-800/40">
            <div className="fd-markdown text-xs">
              <MarkdownMessage content={message.content} />
            </div>
            <span className="text-[10px] text-zinc-400 dark:text-zinc-600 italic">Responded</span>
          </div>
        </div>
      )
    }
    return (
      <div className="mb-3 flex justify-center">
        <div className="max-w-[95%] rounded-lg border border-zinc-200 bg-white px-3 py-2 dark:border-zinc-700/50 dark:bg-zinc-800/40">
          <div className="fd-markdown text-xs">
            <MarkdownMessage content={message.content} />
          </div>
        </div>
      </div>
    )
  }

  if (message.role === 'image' || message.role === 'html_file') {
    return <ChatFileCard message={message} conn={agentConn} large={large} />
  }

  // Assistant — strip trailing suggestions/rating prompts
  const cleanContent = stripSuggestions(message.content)

  // An automated turn's result (Autonomous Work, a plan, a scheduled task) and
  // the agent's restatement of your request are no reply to you: a small
  // preview, the whole of it in a dialog on request.
  if (message.automation || message.task || message.role === 'rephrase') {
    return (
      <CompactMessage
        message={message}
        content={processImagePaths(cleanContent, agentConn?.host, agentConn?.port, agentConn?.auth)}
        actions={actionButtons('left')}
      />
    )
  }

  return (
    <div id={`fd-msg-${message.id}`} className="group mb-3 flex flex-col items-start gap-0.5">
      <div className={`max-w-[85%] rounded-bl-sm bg-zinc-800/60 ${large ? 'rounded-2xl px-4 py-3' : 'rounded-xl px-3.5 py-2.5'} ${message.replay ? 'opacity-60' : ''}`}>
        <div className={`fd-markdown ${bodyText} text-zinc-300`}>
          <MarkdownMessage content={processImagePaths(cleanContent, agentConn?.host, agentConn?.port, agentConn?.auth)} />
        </div>
        <div className="mt-1 flex items-center gap-2 text-[10px] text-zinc-600">
          <span>{formatTime(message.timestamp)}</span>
          {message.model && <span className="font-mono">{message.model}</span>}
          {message.replay && <span className="text-zinc-700">(history)</span>}
        </div>
      </div>
      {actionButtons('left')}
    </div>
  )
})

/** A picture the turn made (image_gen, visualize, a chart the reply hands
 *  over) shown in the chat, or a page it wrote (`html_file`) — each opens full
 *  size in a new tab. A picture that won't load (the agent is off, the file
 *  is gone) falls back to its name, never an empty bubble. */
const ChatFileCard = memo(function ChatFileCard({ message, conn, large }: {
  message: ChatMessage
  conn: { host: string; port: number; auth: string } | null
  large: boolean
}) {
  const [failed, setFailed] = useState(false)
  const path = message.content.trim()
  const name = path.split(/[\\/]/).pop() || path
  const url = conn?.port ? agentFileViewUrl(path, conn.host, conn.port, conn.auth) : ''
  const picture = message.role === 'image'
  if (picture && url && !failed) {
    return (
      <div id={`fd-msg-${message.id}`} className="mb-3 flex flex-col items-start gap-0.5">
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          title={`Open ${name}`}
          className="block max-w-[85%] overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
        >
          <img
            src={url}
            alt={name}
            loading="lazy"
            onError={() => setFailed(true)}
            className={`block h-auto max-w-full object-contain ${large ? 'max-h-[32rem]' : 'max-h-80'}`}
          />
        </a>
        <span className="flex items-center gap-1 pl-1 text-[10px] text-zinc-600">
          <ImageIcon className="h-3 w-3" />
          <span className="truncate">{name}</span>
          <span>· {formatTime(message.timestamp)}</span>
        </span>
      </div>
    )
  }
  const Icon = picture ? ImageIcon : Globe
  return (
    <div id={`fd-msg-${message.id}`} className="mb-3 flex flex-col items-start">
      <div className="flex max-w-[85%] items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2">
        <Icon className="h-4 w-4 shrink-0 text-zinc-500" />
        <div className="min-w-0">
          <div className="truncate text-sm text-zinc-300" title={path}>{name}</div>
          <div className="text-[10px] text-zinc-600">
            {picture ? (failed ? "Picture can't be shown here" : 'Picture') : 'Page'} · {formatTime(message.timestamp)}
          </div>
        </div>
        {url && (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-1 flex shrink-0 items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
          >
            <ExternalLink className="h-3 w-3" />
            Open
          </a>
        )}
      </div>
    </div>
  )
})

/** A message that is no reply to you — an automated turn's result, or the
 *  agent's restatement of your request — kept small and marked as what it
 *  is: its first lines as a preview, the whole of it in a dialog on request. */
const CompactMessage = memo(function CompactMessage({ message, content, actions }: {
  message: ChatMessage
  /** The message as the dialog renders it (images proxied). */
  content: string
  actions: React.ReactNode
}) {
  const [open, setOpen] = useState(false)
  const preview = useMemo(() => previewText(message.content), [message.content])
  const task = message.task
  const rephrase = message.role === 'rephrase'
  const Icon = rephrase ? Wand2 : Bot
  return (
    <div id={`fd-msg-${message.id}`} data-compact={rephrase ? 'rephrase' : 'automated'} className="group mb-3 flex flex-col items-start gap-0.5">
      <div className={`w-full max-w-[85%] rounded-lg border border-dashed border-zinc-700 bg-zinc-900/40 px-3 py-2 ${message.replay ? 'opacity-60' : ''}`}>
        <div className="flex min-w-0 items-center gap-1.5 text-[10px] text-zinc-500">
          <Icon className="h-3 w-3 shrink-0" />
          <span className="shrink-0 font-semibold uppercase tracking-wider">{rephrase ? 'Rephrased task' : 'Automatic'}</span>
          {!rephrase && <span className="shrink-0 text-zinc-600">·</span>}
          {rephrase ? null : task ? (
            <button
              onClick={() => useScheduledStore.getState().openTasks(task.id)}
              className="flex min-w-0 items-center gap-1 font-medium text-amber-700 hover:underline dark:text-amber-400"
              title="Open this scheduled task and its history"
            >
              <AlarmClock className="h-3 w-3 shrink-0" />
              <span className="truncate">{task.name}</span>
            </button>
          ) : (
            <span className="truncate" title={message.automation}>{message.automation}</span>
          )}
          <span className="shrink-0 text-zinc-600">{formatTime(message.timestamp)}</span>
          <button
            onClick={() => setOpen(true)}
            className="ml-auto flex shrink-0 items-center gap-1 rounded-md border border-zinc-700 px-1.5 py-0.5 text-[11px] font-medium text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
            title={rephrase ? 'Show the whole rephrased task' : 'Show the whole result'}
          >
            <Maximize2 className="h-3 w-3" />
            Open
          </button>
        </div>
        {preview && (
          <button onClick={() => setOpen(true)} className="mt-1 block w-full text-left" title={rephrase ? 'Show the whole rephrased task' : 'Show the whole result'}>
            <span className="line-clamp-2 text-xs leading-snug text-zinc-500 hover:text-zinc-400">{preview}</span>
          </button>
        )}
      </div>
      {actions}
      {open && <CompactMessageDialog message={message} content={content} onClose={() => setOpen(false)} />}
    </div>
  )
})

function CompactMessageDialog({ message, content, onClose }: {
  message: ChatMessage
  content: string
  onClose: () => void
}) {
  const { isMobile } = useIsMobile()
  const [copied, setCopied] = useState(false)
  const rephrase = message.role === 'rephrase'
  const Icon = rephrase ? Wand2 : Bot
  const title = rephrase ? 'Rephrased task'
    : message.task ? `Scheduled · ${message.task.name}` : message.automation || 'Automated turn'
  const kind = rephrase ? 'How the agent restated your request' : 'Automatic result'

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const copy = () => {
    navigator.clipboard.writeText(message.content || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return createPortal(
    <div
      className={`fixed inset-0 z-[70] flex bg-black/50 ${isMobile ? 'items-end' : 'items-center justify-center p-6'}`}
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={rephrase ? title : `${kind}: ${title}`}
        className={`flex w-full flex-col overflow-hidden border border-zinc-700 bg-zinc-900 shadow-2xl ${
          isMobile ? 'max-h-[85dvh] rounded-t-2xl pb-[env(safe-area-inset-bottom)]' : 'max-h-[85vh] max-w-3xl rounded-xl'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2.5 border-b border-zinc-800 px-4 py-2.5">
          <Icon className="h-4 w-4 shrink-0 text-zinc-500" />
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-semibold text-zinc-100">{title}</div>
            <div className="flex items-center gap-2 text-[11px] text-zinc-500">
              <span>{kind} · {formatTime(message.timestamp)}</span>
              {message.model && <span className="truncate font-mono">{message.model}</span>}
            </div>
          </div>
          <button onClick={copy} className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Copy" aria-label="Copy">
            {copied ? <Check className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
          </button>
          <button onClick={onClose} className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Close" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <div className="fd-markdown text-sm text-zinc-300">
            <MarkdownMessage content={content} />
          </div>
        </div>
      </div>
    </div>,
    document.body,
  )
}

// ── Intermediary "activity" stream (tool calls + between-step narration) ──
//
// During a turn the agent emits tool calls and short narration blurbs. These
// are grouped into one tinted section that stacks until the turn finishes, so
// they read as a connected activity log rather than loose grey lines. Each
// type gets its own colour: tools (sky), narration (violet).

function isActivityMessage(m: ChatMessage): boolean {
  return m.role === 'tool' || (m.role === 'system' && !!m.narration)
}

type ChatGroup = { kind: 'activity' | 'single' | 'payment' | 'purchase'; items: ChatMessage[]; paymentId?: string }

function groupActivity(msgs: ChatMessage[]): ChatGroup[] {
  const groups: ChatGroup[] = []
  const seenPayments = new Set<string>()
  for (const m of msgs) {
    if (isActivityMessage(m)) {
      const last = groups[groups.length - 1]
      if (last && last.kind === 'activity') last.items.push(m)
      else groups.push({ kind: 'activity', items: [m] })
      // A payment the agent prepared gets its card right after the tool call
      // (once per payment: the card shows the ledger's live state).
      const pid = m.role === 'tool' ? paymentIntentOf(m.tool_name, m.tool_output || m.content) : null
      if (pid && !seenPayments.has(pid)) {
        seenPayments.add(pid)
        groups.push({ kind: 'payment', items: [m], paymentId: pid })
      }
      const purId = m.role === 'tool' ? purchaseIntentOf(m.tool_name, m.tool_output || m.content) : null
      if (purId && !seenPayments.has(purId)) {
        seenPayments.add(purId)
        groups.push({ kind: 'purchase', items: [m], paymentId: purId })
      }
    } else {
      groups.push({ kind: 'single', items: [m] })
    }
  }
  return groups
}

// memo: `items` is a stable array as long as the message list is unchanged
// (see the useMemo'd groups in ChatContent), so non-message ticks skip every
// group except the live/last one, whose `live`/`liveTurnUsage` props change.
const ActivityGroup = memo(function ActivityGroup({ items, live, liveTurnUsage }: { items: ChatMessage[]; live: boolean; liveTurnUsage?: TokenUsage | null }) {
  // Expanded & streaming while the turn runs; auto-collapses to a tools-history
  // summary once it finishes. The user can still toggle it open afterwards.
  const [collapsed, setCollapsed] = useState(!live)
  useEffect(() => { setCollapsed(!live) }, [live])
  const toolCount = items.filter((m) => m.role === 'tool').length
  // Token usage: live cumulative counter while running, else the frozen value
  // stamped onto one of this group's tool messages when the turn ended.
  const usage: TokenUsage | null = live
    ? (liveTurnUsage ?? null)
    : (items.find((m) => m.usage)?.usage ?? null)
  const hasUsage = !!usage && ((usage.prompt_tokens || 0) > 0 || (usage.completion_tokens || 0) > 0)
  return (
    <div className="mb-3 rounded-lg border border-zinc-800 bg-zinc-900/40 px-2.5 py-2">
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="flex w-full items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-400"
      >
        <Wrench className="h-3 w-3" />
        Activity{toolCount > 0 && <span className="text-zinc-600">· {toolCount} tool{toolCount !== 1 ? 's' : ''}</span>}
        {hasUsage && usage && (
          <span className="text-zinc-600 normal-case" title="Tokens this turn — input / output / cached">
            · {fmtTokens(usage.prompt_tokens)}↑ {fmtTokens(usage.completion_tokens)}↓
            {(usage.cache_read_input_tokens || 0) > 0 ? ` · ${fmtTokens(usage.cache_read_input_tokens)} cached` : ''}
          </span>
        )}
        <span className="ml-auto">{collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}</span>
      </button>
      {!collapsed && (
        <div className="mt-1.5 space-y-0.5">
          {items.map((m) =>
            m.role === 'tool'
              ? <ActivityToolRow key={m.id} message={m} />
              : <ActivityNarration key={m.id} message={m} />
          )}
        </div>
      )}
    </div>
  )
})

const ActivityToolRow = memo(function ActivityToolRow({ message }: { message: ChatMessage }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-xs text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-400"
      >
        <Wrench className="h-3 w-3" />
        {message.peer_name && <span className="font-medium text-sky-600 dark:text-sky-400">{message.peer_name}</span>}
        <span className="font-medium">{message.tool_name}</span>
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
      </button>
      {expanded && message.content && (
        <pre className="ml-6 mt-1 max-h-40 overflow-auto rounded-md bg-zinc-900/80 p-2 text-xs text-zinc-400 font-mono">
          {message.content.slice(0, 2000)}
          {message.content.length > 2000 && '\n...truncated'}
        </pre>
      )}
    </div>
  )
})

const ActivityNarration = memo(function ActivityNarration({ message }: { message: ChatMessage }) {
  return (
    <div className="flex items-start gap-1.5 px-1.5 py-0.5 text-xs italic text-zinc-500">
      <CircleDot className="mt-0.5 h-3 w-3 shrink-0 text-zinc-600" />
      <div className="fd-markdown min-w-0 flex-1">
        <MarkdownMessage content={message.content} />
      </div>
    </div>
  )
})

/** Strip trailing suggestion prompts and rating requests from CC responses. */
function stripSuggestions(text: string): string {
  // Remove "---\n💡 If this worked well..." rating block
  let cleaned = text.replace(/\n---\n[💡🔔].*(?:rate good|rate bad).*$/is, '')
  // Remove "SUGGESTED NEXT STEPS" block
  cleaned = cleaned.replace(/\nSUGGESTED NEXT STEPS[\s\S]*$/i, '')
  return cleaned.trimEnd()
}

/** A markdown result's opening words as one plain line, for a preview. */
function previewText(md: string): string {
  return stripSuggestions(md)
    .slice(0, 2000)
    .replace(/```[\s\S]*?(```|$)/g, ' ')                  // code blocks
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')                // images
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')              // links → their text
    .replace(/^\s*\|?(\s*:?-{3,}:?\s*\|?)+\s*$/gm, ' ')    // table rules
    .replace(/^\s{0,3}(#{1,6}\s+|>\s?|[-*+]\s+|\d+\.\s+)/gm, '')
    .replace(/\*+|__|~~|`/g, '')
    .replace(/\|/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 400)
}

function formatTime(ts: string): string {
  if (!ts) return ''
  try {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}
