import { useEffect, useState, useRef, useCallback } from 'react'
import { Sidebar } from './components/layout/Sidebar'
import { TopBar } from './components/layout/TopBar'
import { ChatPanel } from './components/agents/ChatPanel'
import { DirectorPanel } from './components/agents/DirectorPanel'
import { DesktopPage } from './pages/DesktopPage'
import { WorkflowPage } from './pages/WorkflowPage'
import { SpawnerPage } from './pages/SpawnerPage'
import { useUIStore } from './stores/uiStore'
import { useAgentStore } from './stores/agentStore'
import { useChatStore } from './stores/chatStore'
import { botportWS } from './services/ws'
import type { InstanceInfo } from './types'

// ── Persisted width helpers ──

const CHAT_WIDTH_KEY = 'fd:chat-panel-width'
const DIRECTOR_WIDTH_KEY = 'fd:director-panel-width'
const DIRECTOR_OPEN_KEY = 'fd:director-open'

const DEFAULT_CHAT_WIDTH = 480
const MIN_CHAT_WIDTH = 320
const MAX_CHAT_WIDTH = 900

const DEFAULT_DIRECTOR_WIDTH = 300
const MIN_DIRECTOR_WIDTH = 220
const MAX_DIRECTOR_WIDTH = 500

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
  direction: 'left' | 'right', // 'left' = drag left edge to resize (chat), 'right' = drag right edge (director)
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

function App() {
  const view = useUIStore((s) => s.view)
  const chatOpen = useChatStore((s) => s.chatOpen)
  const { fetchInstances, fetchStats, fetchConcerns, setWsConnected, upsertInstance, removeInstance, updateInstanceActivity } = useAgentStore()

  // Director panel
  const [directorOpen, setDirectorOpen] = useState(() => loadBool(DIRECTOR_OPEN_KEY, false))
  const toggleDirector = useCallback(() => {
    setDirectorOpen((v) => {
      localStorage.setItem(DIRECTOR_OPEN_KEY, String(!v))
      return !v
    })
  }, [])

  // Resizable panels
  const chat = useResizable(CHAT_WIDTH_KEY, DEFAULT_CHAT_WIDTH, MIN_CHAT_WIDTH, MAX_CHAT_WIDTH, 'left')
  const director = useResizable(DIRECTOR_WIDTH_KEY, DEFAULT_DIRECTOR_WIDTH, MIN_DIRECTOR_WIDTH, MAX_DIRECTOR_WIDTH, 'right')

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
        <TopBar directorOpen={directorOpen} onToggleDirector={toggleDirector} />
        <main className="flex-1 overflow-hidden">
          <div className="flex h-full">
            {/* Director panel (left) */}
            {directorOpen && (
              <div className="relative flex-shrink-0" style={{ width: director.width }}>
                <DirectorPanel />
                {/* Resize handle on right edge */}
                <div
                  onMouseDown={director.onResizeStart}
                  className="absolute right-0 top-0 h-full w-1.5 cursor-col-resize z-20 hover:bg-violet-500/30 active:bg-violet-500/40 transition-colors"
                />
              </div>
            )}

            {/* Main content */}
            <div className="flex-1 overflow-hidden">
              {view === 'desktop' && <DesktopPage />}
              {view === 'workflow' && <WorkflowPage />}
              {view === 'spawner' && <SpawnerPage />}
            </div>

            {/* Chat panel (right) */}
            {chatOpen && (
              <div className="relative flex-shrink-0" style={{ width: chat.width }}>
                {/* Resize handle on left edge */}
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
    </div>
  )
}

export default App
