import { RefreshCw, PanelLeft, Pin, ClipboardList, Sun, Moon, Keyboard } from 'lucide-react'
import { useAgentStore } from '../../stores/agentStore'
import { useThemeStore } from '../../stores/themeStore'
import { NotificationBell } from '../common/NotificationCenter'

interface TopBarProps {
  directorOpen?: boolean
  onToggleDirector?: () => void
  onTogglePinned?: () => void
  onToggleClipboard?: () => void
  onToggleShortcuts?: () => void
  pinnedOpen?: boolean
  clipboardOpen?: boolean
}

export function TopBar({
  directorOpen, onToggleDirector,
  onTogglePinned, onToggleClipboard, onToggleShortcuts,
  pinnedOpen, clipboardOpen,
}: TopBarProps) {
  const { stats, fetchInstances, fetchStats, fetchConcerns } = useAgentStore()
  const { theme, toggle: toggleTheme } = useThemeStore()

  const refresh = () => {
    fetchInstances()
    fetchStats()
    fetchConcerns(true)
  }

  return (
    <header className="flex h-12 items-center justify-between border-b border-zinc-800 bg-zinc-900/30 px-4">
      <div className="flex items-center gap-3">
        {onToggleDirector && (
          <button
            onClick={onToggleDirector}
            className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium transition-colors ${
              directorOpen
                ? 'bg-violet-600/20 text-violet-400'
                : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
            }`}
            title="Director (Cmd+D)"
          >
            <PanelLeft className="h-3.5 w-3.5" />
            Director
          </button>
        )}

        <div className="h-4 w-px bg-zinc-800" />

        {stats && (
          <div className="flex items-center gap-5">
            <Stat label="Connected" value={stats.connected_instances} color="text-emerald-400" />
            <Stat label="Active" value={stats.active_concerns} color="text-blue-400" />
            <Stat label="Completed" value={stats.completed_concerns} color="text-zinc-400" />
            <Stat label="Failed" value={stats.failed_concerns} color="text-red-400" />
          </div>
        )}
      </div>

      <div className="flex items-center gap-1">
        {/* Tool toggles */}
        {onTogglePinned && (
          <button
            onClick={onTogglePinned}
            className={`rounded p-1.5 transition-colors ${pinnedOpen ? 'bg-amber-600/20 text-amber-400' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'}`}
            title="Pinned Messages"
          >
            <Pin className="h-3.5 w-3.5" />
          </button>
        )}
        {onToggleClipboard && (
          <button
            onClick={onToggleClipboard}
            className={`rounded p-1.5 transition-colors ${clipboardOpen ? 'bg-cyan-600/20 text-cyan-400' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'}`}
            title="Shared Clipboard"
          >
            <ClipboardList className="h-3.5 w-3.5" />
          </button>
        )}
        <div className="h-4 w-px bg-zinc-800 mx-0.5" />

        <NotificationBell />

        <button
          onClick={toggleTheme}
          className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 transition-colors"
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          {theme === 'dark' ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
        </button>

        {onToggleShortcuts && (
          <button
            onClick={onToggleShortcuts}
            className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 transition-colors"
            title="Keyboard Shortcuts (Cmd+K)"
          >
            <Keyboard className="h-3.5 w-3.5" />
          </button>
        )}

        <button
          onClick={refresh}
          className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="h-4 w-4" />
        </button>
      </div>
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
