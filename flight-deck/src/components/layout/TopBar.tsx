import { RefreshCw } from 'lucide-react'
import { useAgentStore } from '../../stores/agentStore'

export function TopBar() {
  const { stats, fetchInstances, fetchStats, fetchConcerns } = useAgentStore()

  const refresh = () => {
    fetchInstances()
    fetchStats()
    fetchConcerns(true)
  }

  return (
    <header className="flex h-12 items-center justify-between border-b border-zinc-800 bg-zinc-900/30 px-4">
      <div className="flex items-center gap-6">
        {stats && (
          <>
            <Stat label="Connected" value={stats.connected_instances} color="text-emerald-400" />
            <Stat label="Active" value={stats.active_concerns} color="text-blue-400" />
            <Stat label="Completed" value={stats.completed_concerns} color="text-zinc-400" />
            <Stat label="Failed" value={stats.failed_concerns} color="text-red-400" />
          </>
        )}
      </div>
      <button
        onClick={refresh}
        className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 transition-colors"
        title="Refresh"
      >
        <RefreshCw className="h-4 w-4" />
      </button>
    </header>
  )
}

function Stat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-1.5 text-xs">
      <span className="text-zinc-500">{label}</span>
      <span className={`font-mono font-semibold ${color}`}>{value}</span>
    </div>
  )
}
