import { useEffect, useLayoutEffect, useState, useRef, useCallback } from 'react'
import { Sidebar } from './components/layout/Sidebar'
import { TopBar } from './components/layout/TopBar'
import { MobileNav } from './components/layout/MobileNav'
import { SimpleLayout } from './components/layout/SimpleLayout'
import { ConsoleLayout } from './components/console/ConsoleLayout'
import { ChatPanel } from './components/agents/ChatPanel'
import { DirectorPanel } from './components/agents/DirectorPanel'
import { PinnedMessages } from './components/common/PinnedMessages'
import { PinnedFiles } from './components/common/PinnedFiles'
import { SharedClipboard } from './components/common/SharedClipboard'
import { ShortcutsOverlay, useKeyboardShortcuts } from './components/common/KeyboardShortcuts'
import { DesktopPage } from './pages/DesktopPage'
import { QuickChatPage } from './pages/QuickChatPage'
import { HostingPage } from './pages/HostingPage'
import { OperationsPage } from './pages/OperationsPage'
import { WorkflowPage } from './pages/WorkflowPage'
import { SpawnerPage } from './pages/SpawnerPage'
import { ForgePage } from './pages/ForgePage'
import { LibraryPage } from './pages/LibraryPage'
import { LoginPage } from './pages/LoginPage'
import { ChoosePasswordPage } from './pages/ChoosePasswordPage'
import { AcceptInvitePage, INVITE_PATH } from './pages/AcceptInvitePage'
import { AdminPage } from './pages/AdminPage'
import { UsersPage } from './pages/UsersPage'
import { CouncilPage } from './pages/CouncilPage'
import { BasnaPage } from './pages/BasnaPage'
import { BatPage } from './pages/BatPage'
import { CodePage } from './pages/CodePage'
import { DubinaPage } from './pages/DubinaPage'
import { TodayPage } from './pages/TodayPage'
import { SkillsPage } from './pages/SkillsPage'
import { GamesPage } from './pages/GamesPage'
import ConnectionsPage from './pages/ConnectionsPage'
import { ProfilePage } from './pages/ProfilePage'
import { GPUCloudPage } from './pages/GPUCloudPage'
import { PromptBuilderPage } from './pages/PromptBuilderPage'
import { ProjectsPage } from './pages/ProjectsPage'
import { AppPage } from './pages/AppPage'
import { AppCodePage } from './pages/AppCodePage'
import { SchedulerPage } from './pages/SchedulerPage'
import { FlowsPage } from './pages/FlowsPage'
import { ObservatoryPage } from './pages/ObservatoryPage'
import { AutonomousWorkPage } from './pages/AutonomousWorkPage'
import { VFSPage } from './pages/VFSPage'
import { DeepMemoryPage } from './pages/DeepMemoryPage'
import { AgentFoldersPage } from './pages/AgentFoldersPage'
import { SystemPage } from './pages/SystemPage'
import { BrowserLLMPage } from './pages/BrowserLLMPage'
import { BeingsPage } from './pages/BeingsPage'
import { PaymentsPage, paymentIdFromPath, purchaseIdFromPath } from './pages/PaymentsPage'
import { PublicBeingPage } from './pages/PublicBeingPage'
import { useUIStore, startInConsoleUnlessChosen, beginLayoutHydrate } from './stores/uiStore'
import { useConsoleStore, syncConsoleFromUrl, dropConsoleRequest, selectConsoleAdmin, openConsole, startDesktop, consoleHome, pageFromPath } from './stores/consoleStore'
import { isAppShell, reportAndroidSession, reportAndroidSignedOut } from './services/desktopShell'
import { useAgentStore } from './stores/agentStore'
import { useAuthStore, selectKioskLocked, awaitAuthStatus, refreshAccessToken, reloadAfterSessionEnd, DEFAULT_INSTANCE_NAME, DEFAULT_LOGO } from './stores/authStore'
import { useChatStore } from './stores/chatStore'
import { useContainerStore } from './stores/containerStore'
import { useLocalAgentStore } from './stores/localAgentStore'
import { useNotificationStore } from './stores/notificationStore'
import { hasTaskboardLink, takeTaskboardLink, useTaskboardStore } from './stores/taskboardStore'
import { usePipelineStore } from './stores/pipelineStore'
import { hydrateAllStores } from './services/settingsSync'
import { reportPublicOrigin } from './services/publicLinks'
import { useConnectionStore } from './stores/connectionStore'
import { useIsMobile } from './hooks/useMediaQuery'
import { botportWS } from './services/ws'
import type { InstanceInfo } from './types'

import { queueSave, registerHydrator } from './services/settingsSync'

// ── Persisted width helpers ──

const CHAT_WIDTH_KEY = 'fd:chat-panel-width'
const DIRECTOR_WIDTH_KEY = 'fd:director-panel-width'
const TOOL_PANEL_WIDTH_KEY = 'fd:tool-panel-width'
const DIRECTOR_OPEN_KEY = 'fd:director-open'

const DEFAULT_CHAT_WIDTH = 480
const MIN_CHAT_WIDTH = 320
const MAX_CHAT_WIDTH = 900

const DEFAULT_DIRECTOR_WIDTH = 300
const MIN_DIRECTOR_WIDTH = 220
const MAX_DIRECTOR_WIDTH = 500

const DEFAULT_TOOL_PANEL_WIDTH = 340
const MIN_TOOL_PANEL_WIDTH = 280
const MAX_TOOL_PANEL_WIDTH = 500

function loadNum(key: string, fallback: number): number {
  try { return parseInt(localStorage.getItem(key) || '', 10) || fallback } catch { return fallback }
}
function loadBool(key: string, fallback: boolean): boolean {
  const v = localStorage.getItem(key)
  if (v === null) return fallback
  return v === 'true'
}
function persistVal(key: string, val: string) {
  if (useAuthStore.getState().authEnabled) queueSave(key, val)
  else localStorage.setItem(key, val)
}

// ── Reusable resize hook ──

function useResizable(
  key: string,
  defaultW: number,
  minW: number,
  maxW: number,
  direction: 'left' | 'right',
) {
  const [width, setWidth] = useState(() => loadNum(key, defaultW))
  const resizing = useRef(false)
  const startX = useRef(0)
  const startW = useRef(0)

  const onResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    resizing.current = true
    startX.current = e.clientX
    startW.current = width
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const onMove = (ev: MouseEvent) => {
      if (!resizing.current) return
      const dx = ev.clientX - startX.current
      const delta = direction === 'right' ? dx : -dx
      const newW = Math.min(maxW, Math.max(minW, startW.current + delta))
      setWidth(newW)
    }
    const onUp = () => {
      resizing.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
      setWidth((w) => { persistVal(key, String(w)); return w })
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [width, key, minW, maxW, direction])

  return { width, onResizeStart }
}

// ── Tool panel type ──

type ToolPanel = 'pinned' | 'pinned-files' | 'clipboard' | null

// Register hydrator for panel widths
registerHydrator((settings) => {
  // Panel widths will be picked up on next load since they use loadNum()
  // We persist them to localStorage so the existing load functions work
  for (const key of [CHAT_WIDTH_KEY, DIRECTOR_WIDTH_KEY, TOOL_PANEL_WIDTH_KEY, DIRECTOR_OPEN_KEY]) {
    if (settings[key] !== undefined) {
      localStorage.setItem(key, settings[key])
    }
  }
})

function AppContent() {
  const view = useUIStore((s) => s.view)
  // A payment's phone page links here (/payments/<id>) when it needs review.
  useEffect(() => {
    if (window.location.pathname === '/payments' || paymentIdFromPath() || purchaseIdFromPath()) useUIStore.getState().setView('payments')
  }, [])
  const layoutMode = useUIStore((s) => s.layoutMode)
  const kioskLocked = useAuthStore(selectKioskLocked)
  // The Admin console: an admin (the user of a deck without sign-in included)
  // who just signed in, has it as their layout — the default until they pick
  // another — or followed a <deck>/console address. In the desktop app an
  // admin is always in the console or, by its Admin | User switch, the users'
  // kiosk (kioskView — part of kioskLocked). Anyone else gets the usual layout.
  const consoleAdmin = useAuthStore(selectConsoleAdmin)
  const kioskView = useAuthStore((s) => s.kioskView)
  const consoleRequested = useConsoleStore((s) => s.requested)
  const adminSignedIn = useAuthStore((s) => s.adminStartPending)
  // A followed Taskboard link (see below) wins over an admin's default console.
  const [boardVisit] = useState(() => !isAppShell() && hasTaskboardLink())
  const inConsole = consoleAdmin && (isAppShell() || consoleRequested
    || ((layoutMode === 'console' || adminSignedIn) && !PAYMENTS_LINK && !boardVisit))
  useEffect(() => {
    window.addEventListener('popstate', syncConsoleFromUrl)
    return () => window.removeEventListener('popstate', syncConsoleFromUrl)
  }, [])
  useEffect(() => { if (!consoleAdmin) dropConsoleRequest() }, [consoleAdmin])
  const chatOpen = useChatStore((s) => s.chatOpen)
  const chatFullscreen = useChatStore((s) => s.chatFullscreen)
  const { fetchInstances, fetchStats, fetchConcerns, setWsConnected, upsertInstance, removeInstance, updateInstanceActivity } = useAgentStore()
  const addNotification = useNotificationStore((s) => s.add)
  const hydrateNotifications = useNotificationStore((s) => s.hydrateFromServer)

  // Director panel
  const [directorOpen, setDirectorOpen] = useState(() => loadBool(DIRECTOR_OPEN_KEY, false))
  const toggleDirector = useCallback(() => {
    setDirectorOpen((v) => {
      persistVal(DIRECTOR_OPEN_KEY, String(!v))
      return !v
    })
  }, [])

  // Tool panels (pinned, clipboard, pipelines) — only one open at a time, appears between main content and chat
  const [toolPanel, setToolPanel] = useState<ToolPanel>(null)
  const toggleToolPanel = useCallback((panel: ToolPanel) => {
    setToolPanel((prev) => prev === panel ? null : panel)
  }, [])

  // Keyboard shortcuts
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  useKeyboardShortcuts(directorOpen, toggleDirector, shortcutsOpen, setShortcutsOpen)

  // Resizable panels
  const chat = useResizable(CHAT_WIDTH_KEY, DEFAULT_CHAT_WIDTH, MIN_CHAT_WIDTH, MAX_CHAT_WIDTH, 'left')
  const director = useResizable(DIRECTOR_WIDTH_KEY, DEFAULT_DIRECTOR_WIDTH, MIN_DIRECTOR_WIDTH, MAX_DIRECTOR_WIDTH, 'right')
  const tool = useResizable(TOOL_PANEL_WIDTH_KEY, DEFAULT_TOOL_PANEL_WIDTH, MIN_TOOL_PANEL_WIDTH, MAX_TOOL_PANEL_WIDTH, 'left')

  // Pipeline execution: watch for agent responses and auto-forward
  useEffect(() => {
    const unsub = useChatStore.subscribe((state, prevState) => {
      const pipelines = usePipelineStore.getState().pipelines.filter((p) => p.enabled)
      if (pipelines.length === 0) return

      for (const [containerId, session] of state.sessions) {
        const prevSession = prevState.sessions.get(containerId)
        if (!prevSession || session.messages.length <= prevSession.messages.length) continue

        const lastMsg = session.messages[session.messages.length - 1]
        if (lastMsg.role !== 'assistant' || lastMsg.replay) continue

        // Check if this agent is in any pipeline
        for (const pipeline of pipelines) {
          const stepIdx = pipeline.steps.findIndex((s) => s.agentId === containerId)
          if (stepIdx === -1 || stepIdx >= pipeline.steps.length - 1) continue

          const nextStep = pipeline.steps[stepIdx + 1]
          const contextPrefix = `--- Pipeline: "${pipeline.name}" — Output from "${session.containerName}" ---\n\nThe following is output from another agent ("${session.containerName}") as part of the "${pipeline.name}" pipeline. Review it and take appropriate action based on your current context, playbooks, instructions, and selected persona.\n\n`
          const prompt = nextStep.prompt
            ? `${contextPrefix}${nextStep.prompt}\n\n${lastMsg.content}`
            : `${contextPrefix}${lastMsg.content}`

          // Auto-forward to next agent
          const { openChat, sendMessage } = useChatStore.getState()
          const allContainers = useContainerStore.getState().containers
          const allLocalAgents = useLocalAgentStore.getState().agents

          const cTarget = allContainers.find((c) => c.id === nextStep.agentId)
          const lTarget = allLocalAgents.find((a) => a.id === nextStep.agentId)
          const target = cTarget || lTarget
          if (target) {
            const host = ('host' in target ? target.host : 'localhost') || 'localhost'
            const port = ('web_port' in target ? target.web_port : target.port) || 0
            const auth = ('web_auth' in target ? target.web_auth : target.authToken) || ''
            const name = ('agent_name' in target ? target.agent_name : target.name) || ''
            openChat(nextStep.agentId, name, host, port, auth)
            setTimeout(() => sendMessage(nextStep.agentId, prompt), 500)

            addNotification('info', 'Pipeline Forwarded',
              `${session.containerName} → ${name} (${pipeline.name})`,
              nextStep.agentId, name)
          }
        }
      }
    })
    return unsub
  }, [addNotification])

  // Initial data load (only poll BotPort when configured)
  useEffect(() => {
    const { botportUrl } = useConnectionStore.getState()
    if (!botportUrl) return  // No BotPort — skip polling

    fetchInstances()
    fetchStats()
    fetchConcerns(true)

    const interval = setInterval(() => {
      fetchInstances()
      fetchStats()
      fetchConcerns(true)
    }, 15000)

    return () => clearInterval(interval)
  }, [fetchInstances, fetchStats, fetchConcerns])

  // Persistent notifications (shares, finished runs) — poll the bell backend.
  useEffect(() => {
    const { authEnabled, isAuthenticated } = useAuthStore.getState()
    if (authEnabled && !isAuthenticated) return  // wait until logged in
    hydrateNotifications()
    const interval = setInterval(() => { hydrateNotifications() }, 30000)
    return () => clearInterval(interval)
  }, [hydrateNotifications])

  // WebSocket connection for real-time updates (only when BotPort configured)
  useEffect(() => {
    const { botportUrl } = useConnectionStore.getState()
    if (!botportUrl) return  // No BotPort — skip WS

    botportWS.connect()

    const unsubs = [
      botportWS.on('_connected', () => setWsConnected(true)),
      botportWS.on('_disconnected', () => setWsConnected(false)),

      botportWS.on('instance_connected', (msg) => {
        upsertInstance(msg.instance as InstanceInfo)
        addNotification('success', 'Agent Connected', `${(msg.instance as InstanceInfo).name || 'Agent'} is now online`)
      }),
      botportWS.on('instance_disconnected', (msg) => {
        removeInstance(msg.instance_id as string)
        addNotification('warning', 'Agent Disconnected', `Agent ${(msg.instance_id as string).slice(0, 8)} went offline`)
      }),

      botportWS.on('activity', (msg) => {
        updateInstanceActivity(
          msg.instance_id as string,
          msg.step_type as string,
          msg.data as Record<string, unknown>
        )
      }),

      botportWS.on('concern_created', () => fetchConcerns(true)),
      botportWS.on('concern_result', () => {
        fetchConcerns(true)
        fetchStats()
      }),
    ]

    return () => {
      unsubs.forEach((u) => u())
      botportWS.disconnect()
    }
  }, [setWsConnected, upsertInstance, removeInstance, updateInstanceActivity, fetchConcerns, fetchStats, addNotification])

  const { isMobile, isTablet } = useIsMobile()
  // Taskboard v2: a ?taskboard=<task id> link (a nudge, a bell item) opens the
  // board on that task in the simple layout, where the board lives and which
  // takes the link (SimpleLayout). For this visit only, never as their choice
  // (one click or a reload goes back), and on any screen: most links are
  // tapped on a phone, and an admin's own default is the console. Only the
  // apps' console (they have their own switch) and a console address, which
  // says where it wants to be, open no board: the link is taken out of the
  // address there, and they're told.
  const visitLayout = useUIStore((s) => s.visitLayout)
  const deciding = adminSignedIn && consoleAdmin && isAppShell()
  // Before the first paint and any child's effect (SimpleLayout takes the link).
  useLayoutEffect(() => {
    if (kioskLocked || deciding || !hasTaskboardLink()) return
    if (isAppShell() || consoleRequested) {
      if (takeTaskboardLink()) {
        addNotification('info', 'Taskboard', 'That link opens a task on the Taskboard, which isn\'t part of the Admin console.')
      }
      return
    }
    if (layoutMode !== 'simple') visitLayout('simple')
  }, [kioskLocked, deciding, consoleRequested, layoutMode, visitLayout, addNotification])
  const mobilePanel = useUIStore((s) => s.mobilePanel)
  const setMobilePanel = useUIStore((s) => s.setMobilePanel)
  const sidebarDrawerOpen = useUIStore((s) => s.sidebarDrawerOpen)
  const setSidebarDrawerOpen = useUIStore((s) => s.setSidebarDrawerOpen)

  // On mobile, Director/Chat/Tool open via mobilePanel overlay instead of side-by-side
  const mobilePanelToggleDirector = useCallback(() => {
    if (isMobile || isTablet) {
      useUIStore.getState().toggleMobilePanel('director')
    } else {
      toggleDirector()
    }
  }, [isMobile, isTablet, toggleDirector])

  const mainContent = (
    <>
      {view === 'desktop' && <DesktopPage />}
      {view === 'quick-chat' && <QuickChatPage />}
      {view === 'prompt-builder' && <PromptBuilderPage />}
      {view === 'today' && <TodayPage />}
      {view === 'operations' && <OperationsPage />}
      {view === 'workflow' && <WorkflowPage />}
      {view === 'flows' && <FlowsPage />}
      {view === 'observatory' && <ObservatoryPage />}
      {view === 'autonomous-work' && <AutonomousWorkPage />}
      {view === 'skills' && <SkillsPage />}
      {view === 'games' && <GamesPage />}
      {view === 'spawner' && <SpawnerPage />}
      {view === 'forge' && <ForgePage />}
      {view === 'library' && <LibraryPage />}
      {view === 'admin' && <AdminPage />}
      {view === 'users' && <UsersPage />}
      {view === 'council' && <CouncilPage />}
      {view === 'basna' && <BasnaPage />}
      {view === 'bat' && <BatPage />}
      {view === 'payments' && <PaymentsPage />}
      {view === 'code' && <CodePage />}
      {view === 'dubina' && <DubinaPage />}
      {view === 'vfs' && <VFSPage />}
      {view === 'deep-memory' && <DeepMemoryPage />}
      {view === 'hosting' && <HostingPage />}
      {view === 'agent-folders' && <AgentFoldersPage />}
      {view === 'system' && <SystemPage />}
      {view === 'browser-llm' && <BrowserLLMPage />}
      {view === 'beings' && <BeingsPage />}
      {view === 'connections' && <ConnectionsPage />}
      {view === 'profile' && <ProfilePage />}
      {view === 'gpu-cloud' && <GPUCloudPage />}
      {view === 'scheduler' && <SchedulerPage />}
      {view === 'projects' && <ProjectsPage />}
      {view === 'app' && <AppPage />}
      {view === 'code-app' && <AppCodePage />}
    </>
  )

  // ── Mobile / Tablet overlay panels ──
  const mobilePanelOverlay = (isMobile || isTablet) && mobilePanel !== 'none' && (
    <div className="fixed inset-0 z-50 flex flex-col bg-zinc-950">
      <div className="flex h-12 items-center justify-between border-b border-zinc-800 px-4">
        <span className="text-sm font-semibold capitalize">{mobilePanel}</span>
        <button
          onClick={() => setMobilePanel('none')}
          className="rounded p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
        </button>
      </div>
      <div className="flex-1 overflow-hidden">
        {mobilePanel === 'director' && <DirectorPanel />}
        {mobilePanel === 'chat' && <ChatPanel />}
        {mobilePanel === 'tool' && (
          <>
            {toolPanel === 'pinned' && <PinnedMessages onClose={() => { setToolPanel(null); setMobilePanel('none') }} />}
            {toolPanel === 'pinned-files' && <PinnedFiles onClose={() => { setToolPanel(null); setMobilePanel('none') }} />}
            {toolPanel === 'clipboard' && <SharedClipboard onClose={() => { setToolPanel(null); setMobilePanel('none') }} />}
          </>
        )}
      </div>
    </div>
  )

  // ── Tablet sidebar drawer ──
  const tabletSidebarDrawer = isTablet && sidebarDrawerOpen && (
    <>
      <div className="fixed inset-0 z-40 bg-black/50" onClick={() => setSidebarDrawerOpen(false)} />
      <div className="fixed left-0 top-0 bottom-0 z-50 w-[260px]">
        <Sidebar />
      </div>
    </>
  )

  // ── Kiosk lock (Admin → Settings → Kiosk mode; `--simple-chat` default) ──
  // Force the locked, chat-first Simple layout on every form factor and ignore
  // the stored layoutMode entirely. `locked` strips the escape hatches (full
  // view, the Spawner, agent config, start/stop) so shared-account users can
  // only chat, create an agent from a Library archetype, manage their Google /
  // MCP connections, edit their profile, or sign out — nothing else is
  // reachable. Placed before the mobile/tablet/full branches so it always wins.
  // Admins are exempt (selectKioskLocked) — they keep full access even here —
  // except the desktop app's admin who switched to it: the switch
  // (`adminSwitch`) is their one way back to the Admin console.
  if (kioskLocked) {
    return (
      <>
        <SimpleLayout locked adminSwitch={kioskView && consoleAdmin} />
        {shortcutsOpen && <ShortcutsOverlay onClose={() => setShortcutsOpen(false)} />}
      </>
    )
  }

  // The desktop app decides between the kiosk and the console after a sign-in
  // (startDesktop): nothing to show until it has.
  if (adminSignedIn && consoleAdmin && isAppShell()) return <div className="h-screen bg-zinc-950" />

  // ── Admin console ── (every form factor; admins are never kiosk-locked)
  if (inConsole) return <ConsoleLayout />

  // A followed Taskboard link, on a phone or a tablet too (see boardVisit).
  if (boardVisit && layoutMode === 'simple') {
    return (
      <>
        <SimpleLayout />
        {shortcutsOpen && <ShortcutsOverlay onClose={() => setShortcutsOpen(false)} />}
      </>
    )
  }

  // ── Mobile layout ──
  if (isMobile) {
    return (
      <div className="flex h-screen flex-col overflow-hidden">
        <TopBar
          directorOpen={false}
          onToggleDirector={mobilePanelToggleDirector}
          onTogglePinned={() => { toggleToolPanel('pinned'); setMobilePanel('tool') }}
          onTogglePinnedFiles={() => { toggleToolPanel('pinned-files'); setMobilePanel('tool') }}
          onToggleClipboard={() => { toggleToolPanel('clipboard'); setMobilePanel('tool') }}
          onToggleShortcuts={() => setShortcutsOpen(!shortcutsOpen)}
          pinnedOpen={toolPanel === 'pinned'}
          pinnedFilesOpen={toolPanel === 'pinned-files'}
          clipboardOpen={toolPanel === 'clipboard'}
          isMobile
        />
        <main className="flex-1 overflow-hidden pb-14">
          {mainContent}
        </main>
        <MobileNav />
        {mobilePanelOverlay}
        {shortcutsOpen && <ShortcutsOverlay onClose={() => setShortcutsOpen(false)} />}
      </div>
    )
  }

  // ── Tablet layout ──
  if (isTablet) {
    return (
      <div className="flex h-screen flex-col overflow-hidden">
        <TopBar
          directorOpen={false}
          onToggleDirector={mobilePanelToggleDirector}
          onTogglePinned={() => { toggleToolPanel('pinned'); setMobilePanel('tool') }}
          onTogglePinnedFiles={() => { toggleToolPanel('pinned-files'); setMobilePanel('tool') }}
          onToggleClipboard={() => { toggleToolPanel('clipboard'); setMobilePanel('tool') }}
          onToggleShortcuts={() => setShortcutsOpen(!shortcutsOpen)}
          pinnedOpen={toolPanel === 'pinned'}
          pinnedFilesOpen={toolPanel === 'pinned-files'}
          clipboardOpen={toolPanel === 'clipboard'}
          isTablet
          onToggleSidebarDrawer={() => setSidebarDrawerOpen(!sidebarDrawerOpen)}
        />
        <main className="flex-1 overflow-hidden">
          {mainContent}
        </main>
        {tabletSidebarDrawer}
        {mobilePanelOverlay}
        {shortcutsOpen && <ShortcutsOverlay onClose={() => setShortcutsOpen(false)} />}
      </div>
    )
  }

  // ── Desktop: simple layout ──
  // Agents on the left, the chat in the middle, the active agent's files and
  // datastore on the right. Swaps in for everything below (nav, top bar,
  // director, pages, tool panels) and back out with one click; the full
  // layout is left exactly as it was.
  if (layoutMode === 'simple') {
    return (
      <>
        <SimpleLayout />
        {shortcutsOpen && <ShortcutsOverlay onClose={() => setShortcutsOpen(false)} />}
      </>
    )
  }

  // ── Desktop layout (unchanged) ──
  // Fullscreen chat: hides sidebar, director, main content, and tool panels
  // so the chat (with its left-hand queue panel) gets the full viewport
  // width. TopBar stays so the user can still toggle UI controls.
  if (chatFullscreen && chatOpen) {
    return (
      <div className="flex h-screen flex-col overflow-hidden">
        <TopBar
          directorOpen={directorOpen}
          onToggleDirector={toggleDirector}
          onTogglePinned={() => toggleToolPanel('pinned')}
          onTogglePinnedFiles={() => toggleToolPanel('pinned-files')}
          onToggleClipboard={() => toggleToolPanel('clipboard')}
          onToggleShortcuts={() => setShortcutsOpen(!shortcutsOpen)}
          pinnedOpen={toolPanel === 'pinned'}
          pinnedFilesOpen={toolPanel === 'pinned-files'}
          clipboardOpen={toolPanel === 'clipboard'}
        />
        <main className="flex-1 overflow-hidden">
          <div className="h-full w-full">
            <ChatPanel />
          </div>
        </main>
        {shortcutsOpen && <ShortcutsOverlay onClose={() => setShortcutsOpen(false)} />}
      </div>
    )
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <TopBar
          directorOpen={directorOpen}
          onToggleDirector={toggleDirector}
          onTogglePinned={() => toggleToolPanel('pinned')}
          onTogglePinnedFiles={() => toggleToolPanel('pinned-files')}
          onToggleClipboard={() => toggleToolPanel('clipboard')}
          onToggleShortcuts={() => setShortcutsOpen(!shortcutsOpen)}
          pinnedOpen={toolPanel === 'pinned'}
          pinnedFilesOpen={toolPanel === 'pinned-files'}
          clipboardOpen={toolPanel === 'clipboard'}
        />
        <main className="flex-1 overflow-hidden">
          <div className="flex h-full">
            {/* Director panel (left) */}
            {directorOpen && (
              <div className="relative flex-shrink-0" style={{ width: director.width }}>
                <DirectorPanel />
                <div
                  onMouseDown={director.onResizeStart}
                  className="absolute right-0 top-0 h-full w-1.5 cursor-col-resize z-20 hover:bg-violet-500/30 active:bg-violet-500/40 transition-colors"
                />
              </div>
            )}

            {/* Main content */}
            <div className="flex-1 overflow-hidden" style={{ minWidth: 360 }}>
              {mainContent}
            </div>

            {/* Tool panel (pinned / clipboard / pipelines) */}
            {toolPanel && (
              <div className="relative flex-shrink-0 border-l border-zinc-800" style={{ width: tool.width }}>
                <div
                  onMouseDown={tool.onResizeStart}
                  className="absolute left-0 top-0 h-full w-1.5 cursor-col-resize z-20 hover:bg-violet-500/30 active:bg-violet-500/40 transition-colors"
                />
                {toolPanel === 'pinned' && <PinnedMessages onClose={() => setToolPanel(null)} />}
                {toolPanel === 'pinned-files' && <PinnedFiles onClose={() => setToolPanel(null)} />}
                {toolPanel === 'clipboard' && <SharedClipboard onClose={() => setToolPanel(null)} />}
              </div>
            )}

            {/* Chat panel (right) */}
            {chatOpen && (
              <div className="relative flex-shrink-0" style={{ width: chat.width }}>
                <div
                  onMouseDown={chat.onResizeStart}
                  className="absolute left-0 top-0 h-full w-1.5 cursor-col-resize z-20 hover:bg-violet-500/30 active:bg-violet-500/40 transition-colors"
                />
                <ChatPanel />
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Keyboard shortcuts overlay */}
      {shortcutsOpen && <ShortcutsOverlay onClose={() => setShortcutsOpen(false)} />}
    </div>
  )
}

// A link to a payment (its phone page's "Open in Flight Deck") opens that page:
// it wins over the console an admin would otherwise start in. Read at load.
const PAYMENTS_LINK = window.location.pathname === '/payments' || !!paymentIdFromPath() || !!purchaseIdFromPath()

// An admin who signed in from a form: the desktop app's launch choice, or the
// Admin console — on Models while they have no tier set.
async function landAdminAfterSignIn() {
  const auth = useAuthStore.getState()
  if (!auth.adminStartPending) return
  try {
    if (!selectConsoleAdmin(auth) || (PAYMENTS_LINK && !isAppShell())) return
    if (isAppShell()) {
      await startDesktop({ signedIn: true })
      return
    }
    // They signed in to follow a Taskboard link: the board, not the console
    // (settings load slower than the board opens, so the link may be taken).
    if (hasTaskboardLink() || useTaskboardStore.getState().open) return
    auth.setKioskView(false)
    // A console address (a bookmark, the page a session expired on) says where.
    openConsole(pageFromPath(window.location.pathname) ?? consoleHome())
  } finally {
    // Until here the pending sign-in kept the console (or, in the desktop
    // app, a blank screen) up, so nothing flashes before the landing.
    useAuthStore.getState().setAdminStartPending(false)
  }
}

function App() {
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  const mustChangePassword = useAuthStore((s) => s.user?.must_change_password === true)
  const instanceName = useAuthStore((s) => s.branding.name)
  const instanceLogo = useAuthStore((s) => s.branding.logo)
  const [authChecked, setAuthChecked] = useState(false)
  const [fdUnreachable, setFdUnreachable] = useState(false)
  // An invitation link (<deck>/invite#<token>): choose a password, get an account.
  const [onInvitePage, setOnInvitePage] = useState(() => window.location.pathname === INVITE_PATH)

  // The public square (plan §9) is the one surface that lives OUTSIDE auth:
  // /village and /b/<slug> render for anyone, logged in or not, before the
  // login gate below and without hydrating any of the app's stores.
  const _path = window.location.pathname
  if (_path === '/village' || _path.startsWith('/b/') || _path.startsWith('/v/')) {
    return <PublicBeingPage />
  }

  // On mount: check if auth is enabled, try to refresh token. An unreachable
  // FD is retried, never taken as "auth off" — that would render the full,
  // unlocked UI (kiosk lock included) on a deck that enforces auth.
  useEffect(() => {
    let cancelled = false
    async function init() {
      const enabled = await awaitAuthStatus(() => setFdUnreachable(true), () => cancelled)
      if (enabled === null || cancelled) return
      setFdUnreachable(false)
      if (enabled) {
        // Try refreshing an existing session
        const ok = await refreshAccessToken()
        if (ok) {
          // Hydrate stores from server-side settings
          beginLayoutHydrate()
          await hydrateAllStores()
          // The desktop app: the users' kiosk or the Admin console (startDesktop).
          if (isAppShell() && selectConsoleAdmin(useAuthStore.getState())) await startDesktop()
        }
      } else {
        reportPublicOrigin()   // auth-off deck: no login to wait for
        startInConsoleUnlessChosen()   // its user is its admin
        if (isAppShell()) await startDesktop()
      }
      setAuthChecked(true)
    }
    init()
    return () => { cancelled = true }
  }, [])

  // The browser tab carries the deck's name (Admin → Settings → Branding).
  useEffect(() => { document.title = instanceName || DEFAULT_INSTANCE_NAME }, [instanceName])
  // …and its logo as the tab icon (the Captain icon when Branding has none).
  useEffect(() => {
    let link = document.querySelector<HTMLLinkElement>('link[rel="icon"]')
    if (!link) {
      link = document.createElement('link')
      link.rel = 'icon'
      document.head.appendChild(link)
    }
    link.removeAttribute('type')   // PNG, JPEG, SVG… — the browser goes by the response
    link.href = instanceLogo || DEFAULT_LOGO
  }, [instanceLogo])

  // After login, hydrate stores. On the way back out (sign-out, or a session
  // that expired: a 401 whose refresh failed → clearAuth()), reload instead of
  // just showing LoginPage over the old session's sockets and stores — see
  // reloadAfterSessionEnd (it's loop-guarded; startup never goes true → false).
  const prevAuth = useRef(false)
  useEffect(() => {
    if (isAuthenticated && !prevAuth.current) {
      // Once the signed-in settings are in, an admin who just signed in lands
      // (they decide whether a tier set exists).
      beginLayoutHydrate()
      hydrateAllStores().finally(landAdminAfterSignIn)
      // Tell Flight Deck the address this deck is reached at, so a public link
      // an agent makes points here rather than at the agent's internal FD URL.
      reportPublicOrigin()
      reportAndroidSession(useAuthStore.getState().user)
    } else if (!isAuthenticated && prevAuth.current) {
      reportAndroidSignedOut()
      reloadAfterSessionEnd()
    }
    prevAuth.current = isAuthenticated
  }, [isAuthenticated])

  // Still checking auth status
  if (!authChecked) {
    return (
      <div className="min-h-screen bg-[#0a0a1a] flex items-center justify-center">
        <div className="text-zinc-500 text-sm">
          {fdUnreachable ? "Can't reach Flight Deck — retrying…" : 'Loading...'}
        </div>
      </div>
    )
  }

  if (authEnabled && onInvitePage) {
    return <AcceptInvitePage onLeave={() => setOnInvitePage(false)} />
  }

  // Auth enabled but not logged in → show login page
  if (authEnabled && !isAuthenticated) {
    return <LoginPage />
  }

  // An admin set this account's password: the user picks their own first.
  if (authEnabled && mustChangePassword) {
    return <ChoosePasswordPage />
  }

  // Auth disabled or authenticated → show main app
  return <AppContent />
}

export default App
