import { useEffect, useState, useRef, useCallback } from 'react'
import { Sidebar } from './components/layout/Sidebar'
import { TopBar } from './components/layout/TopBar'
import { ChatPanel } from './components/agents/ChatPanel'
import { DesktopPage } from './pages/DesktopPage'
import { WorkflowPage } from './pages/WorkflowPage'
import { SpawnerPage } from './pages/SpawnerPage'
import { useUIStore } from './stores/uiStore'
import { useAgentStore } from './stores/agentStore'
import { useChatStore } from './stores/chatStore'
import { botportWS } from './services/ws'
import type { InstanceInfo } from './types'

const CHAT_WIDTH_KEY = 'fd:chat-panel-width'
const DEFAULT_CHAT_WIDTH = 480
const MIN_CHAT_WIDTH = 320
const MAX_CHAT_WIDTH = 900

function loadChatWidth(): number {
  try { return parseInt(localStorage.getItem(CHAT_WIDTH_KEY) || '', 10) || DEFAULT_CHAT_WIDTH } catch { return DEFAULT_CHAT_WIDTH }
}

function App() {
  const view = useUIStore((s) => s.view)
  const chatOpen = useChatStore((s) => s.chatOpen)
  const { fetchInstances, fetchStats, fetchConcerns, setWsConnected, upsertInstance, removeInstance, updateInstanceActivity } = useAgentStore()

  // Resizable chat panel
  const [chatWidth, setChatWidth] = useState(loadChatWidth)
  const resizing = useRef(false)
  const resizeStartX = useRef(0)
  const resizeStartW = useRef(0)

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    resizing.current = true
    resizeStartX.current = e.clientX
    resizeStartW.current = chatWidth
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const onMove = (ev: MouseEvent) => {
      if (!resizing.current) return
      const dx = resizeStartX.current - ev.clientX // drag left = wider
      const newW = Math.min(MAX_CHAT_WIDTH, Math.max(MIN_CHAT_WIDTH, resizeStartW.current + dx))
      setChatWidth(newW)
    }
    const onUp = () => {
      resizing.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
      // Save after release
      setChatWidth((w) => { localStorage.setItem(CHAT_WIDTH_KEY, String(w)); return w })
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [chatWidth])

  // Initial data load
  useEffect(() => {
    fetchInstances()
    fetchStats()
    fetchConcerns(true)

    // Poll every 15s as fallback
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

      // Instance events
      botportWS.on('instance_connected', (msg) => {
        upsertInstance(msg.instance as InstanceInfo)
      }),
      botportWS.on('instance_disconnected', (msg) => {
        removeInstance(msg.instance_id as string)
      }),

      // Activity updates
      botportWS.on('activity', (msg) => {
        updateInstanceActivity(
          msg.instance_id as string,
          msg.step_type as string,
          msg.data as Record<string, unknown>
        )
      }),

      // Concern events trigger a refetch
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
  }, [setWsConnected, upsertInstance, removeInstance, updateInstanceActivity, fetchConcerns, fetchStats])

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <TopBar />
        <main className="flex-1 overflow-hidden">
          <div className="flex h-full">
            <div className="flex-1 overflow-hidden">
              {view === 'desktop' && <DesktopPage />}
              {view === 'workflow' && <WorkflowPage />}
              {view === 'spawner' && <SpawnerPage />}
            </div>
            {chatOpen && (
              <div className="relative flex-shrink-0" style={{ width: chatWidth }}>
                {/* Resize handle */}
                <div
                  onMouseDown={handleResizeStart}
                  className="absolute left-0 top-0 h-full w-1.5 cursor-col-resize z-20 hover:bg-violet-500/30 active:bg-violet-500/40 transition-colors"
                />
                <ChatPanel />
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
