import { Monitor, BarChart3, GitBranch, Plus, Shield, Users, Wand2 } from 'lucide-react'
import { useUIStore } from '../../stores/uiStore'
import { useAuthStore } from '../../stores/authStore'
import type { ViewMode } from '../../types'

const navItems: { id: ViewMode; icon: typeof Monitor; label: string; adminOnly?: boolean }[] = [
  { id: 'desktop', icon: Monitor, label: 'Desktop' },
  { id: 'council', icon: Users, label: 'Council' },
  { id: 'spawner', icon: Plus, label: 'Spawn' },
  { id: 'forge', icon: Wand2, label: 'Forge' },
  { id: 'workflow', icon: GitBranch, label: 'Workflows' },
  { id: 'operations', icon: BarChart3, label: 'Stats' },
  { id: 'admin', icon: Shield, label: 'Admin', adminOnly: true },
]

export function MobileNav() {
  const { view, setView } = useUIStore()
  const { authEnabled, user: authUser } = useAuthStore()

  const items = navItems.filter((item) => {
    if (item.adminOnly) return authEnabled && authUser?.role === 'admin'
    return true
  })

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-40 flex items-center justify-around border-t border-zinc-800 bg-zinc-900/95 backdrop-blur-sm pb-[env(safe-area-inset-bottom)]">
      {items.map(({ id, icon: Icon, label }) => (
        <button
          key={id}
          onClick={() => setView(id)}
          className={`flex flex-1 flex-col items-center gap-0.5 py-2.5 text-[10px] font-medium transition-colors ${
            view === id
              ? 'text-violet-400'
              : 'text-zinc-500 active:text-zinc-300'
          }`}
        >
          <Icon className="h-5 w-5" />
          {label}
        </button>
      ))}
    </nav>
  )
}
