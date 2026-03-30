import { useEffect, useState, useRef, useCallback } from 'react'
import { Sidebar } from './components/layout/Sidebar'
import { TopBar } from './components/layout/TopBar'
import { ChatPanel } from './components/agents/ChatPanel'
import { DirectorPanel } from './components/agents/DirectorPanel'
import { PinnedMessages } from './components/common/PinnedMessages'
import { PinnedFiles } from './components/common/PinnedFiles'
import { SharedClipboard } from './components/common/SharedClipboard'
import { ShortcutsOverlay, useKeyboardShortcuts } from './components/common/KeyboardShortcuts'
import { DesktopPage } from './pages/DesktopPage'
import { OperationsPage } from './pages/OperationsPage'
import { WorkflowPage } from './pages/WorkflowPage'
import { SpawnerPage } from './pages/SpawnerPage'
import { useUIStore } from './stores/uiStore'
import { useAgentStore } from './stores/agentStore'
import { useChatStore } from './stores/chatStore'
import { useContainerStore } from './stores/containerStore'
import { useLocalAgentStore } from './stores/localAgentStore'
import { useNotificationStore } from './stores/notificationStore'
import { usePipelineStore } from './stores/pipelineStore'
import { botportWS } from './services/ws'
import type { InstanceInfo } from './types'

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
      setWidth((w) => { localStorage.setItem(key, String(w)); return w })
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [width, key, minW, maxW, direction])

  return { width, onResizeStart }
}

// ── Tool panel type ──

type ToolPanel = 'pinned' | 'pinned-files' | 'clipboard' | null

function App() {
  const view = useUIStore((s) => s.view)
  const chatOpen = useChatStore((s) => s.chatOpen)
  const { fetchInstances, fetchStats, fetchConcerns, setWsConnected, upsertInstance, removeInstance, updateInstanceActivity } = useAgentStore()
  const addNotification = useNotificationStore((s) => s.add)

  // Director panel
  const [directorOpen, setDirectorOpen] = useState(() => loadBool(DIRECTOR_OPEN_KEY, false))
  const toggleDirector = useCallback(() => {
    setDirectorOpen((v) => {
      localStorage.setItem(DIRECTOR_OPEN_KEY, String(!v))
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

  // Initial data load
  useEffect(() => {
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

  // WebSocket connection for real-time updates
  useEffect(() => {
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
            <div className="flex-1 overflow-hidden">
              {view === 'desktop' && <DesktopPage />}
              {view === 'operations' && <OperationsPage />}
              {view === 'workflow' && <WorkflowPage />}
              {view === 'spawner' && <SpawnerPage />}
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

export default App
