import { useRef } from 'react'
import type { LucideIcon } from 'lucide-react'

// The Profile page's tabs: Personal · Deck-wide · Password. A segmented bar
// under the header; arrow keys move between them, the panel below is the
// chosen one's.

export type ProfileTab = 'personal' | 'deck' | 'password'

export interface ProfileTabSpec {
  id: ProfileTab
  label: string
  icon: LucideIcon
  /** A word more, as the tab's tooltip. */
  hint: string
}

export function ProfileTabs({ tabs, tab, onChange }: {
  tabs: ProfileTabSpec[]
  tab: ProfileTab
  onChange: (tab: ProfileTab) => void
}) {
  const barRef = useRef<HTMLDivElement>(null)

  const onKey = (e: React.KeyboardEvent) => {
    const step = { ArrowRight: 1, ArrowLeft: -1, Home: -Infinity, End: Infinity }[e.key]
    if (step === undefined) return
    e.preventDefault()
    const at = tabs.findIndex((t) => t.id === tab)
    const next = step === -Infinity ? 0 : step === Infinity ? tabs.length - 1
      : Math.max(0, Math.min(tabs.length - 1, at + step))
    onChange(tabs[next].id)
    barRef.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus()
  }

  return (
    <div
      ref={barRef}
      role="tablist"
      aria-label="Profile sections"
      onKeyDown={onKey}
      className="mb-5 grid gap-1 rounded-xl border border-zinc-800 bg-zinc-900/60 p-1"
      style={{ gridTemplateColumns: `repeat(${tabs.length}, minmax(0, 1fr))` }}
    >
      {tabs.map(({ id, label, icon: Icon, hint }) => {
        const on = id === tab
        return (
          <button
            key={id}
            type="button"
            role="tab"
            id={`profile-tab-${id}`}
            aria-selected={on}
            aria-controls={`profile-panel-${id}`}
            tabIndex={on ? 0 : -1}
            title={hint}
            onClick={() => onChange(id)}
            className={`flex min-w-0 items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${
              on ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-300'
            }`}
          >
            <Icon className={`h-4 w-4 shrink-0 ${on ? 'text-violet-500 dark:text-violet-400' : ''}`} />
            <span className="truncate">{label}</span>
          </button>
        )
      })}
    </div>
  )
}
