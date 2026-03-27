import { useEffect } from 'react'
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

function App() {
  const view = useUIStore((s) => s.view)
  const chatOpen = useChatStore((s) => s.chatOpen)
  const { fetchInstances, fetchStats, fetchConcerns, setWsConnected, upsertInstance, removeInstance, updateInstanceActivity } = useAgentStore()

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
            {chatOpen && <ChatPanel />}
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
