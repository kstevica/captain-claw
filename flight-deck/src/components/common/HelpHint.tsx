import { useState } from 'react'
import { HelpCircle, X } from 'lucide-react'
import { useOnboardingStore } from '../../stores/onboardingStore'

interface HelpHintProps {
  id: string
  children: React.ReactNode
  /** Show as a full-width banner instead of inline icon */
  banner?: boolean
}

export function HelpHint({ id, children, banner }: HelpHintProps) {
  const { dismissedHints, dismissHint } = useOnboardingStore()
  const [expanded, setExpanded] = useState(banner ?? false)

  if (dismissedHints.includes(id)) return null

  if (banner) {
    return (
      <div className="fd-help-banner mb-3 rounded-lg border border-violet-500/20 bg-violet-500/5 px-4 py-3">
        <div className="flex items-start gap-3">
          <HelpCircle className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />
          <div className="flex-1 text-xs leading-relaxed text-zinc-300">{children}</div>
          <button
            onClick={() => dismissHint(id)}
            className="shrink-0 rounded p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
            title="Dismiss"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    )
  }

  return (
    <span className="relative inline-flex items-center">
      <button
        onClick={() => setExpanded(!expanded)}
        className="rounded p-0.5 text-zinc-600 hover:text-violet-400 transition-colors"
        title="Help"
      >
        <HelpCircle className="h-3.5 w-3.5" />
      </button>
      {expanded && (
        <div className="absolute left-6 top-0 z-50 w-64 rounded-lg border border-zinc-700/50 bg-zinc-900 p-3 shadow-xl shadow-black/30">
          <div className="flex items-start gap-2">
            <div className="flex-1 text-[11px] leading-relaxed text-zinc-300">{children}</div>
            <button
              onClick={() => dismissHint(id)}
              className="shrink-0 rounded p-0.5 text-zinc-500 hover:text-zinc-300"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        </div>
      )}
    </span>
  )
}
