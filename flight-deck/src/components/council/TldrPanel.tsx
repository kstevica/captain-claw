import { useState } from 'react'
import { RefreshCw, Loader2, MessageSquareQuote, ChevronDown } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { CouncilArtifact } from '../../stores/councilStore'

interface TldrPanelProps {
  tldrs: CouncilArtifact[]
  generating: boolean
  onGenerate: () => void
}

export function TldrPanel({ tldrs, generating, onGenerate }: TldrPanelProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center gap-2 min-w-0"
        >
          <MessageSquareQuote className="h-4 w-4 text-violet-400 shrink-0" />
          <h3 className="text-sm font-medium text-violet-300">TL;DR</h3>
          <ChevronDown className={`h-3.5 w-3.5 text-violet-400 transition-transform ${collapsed ? '-rotate-90' : ''}`} />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onGenerate() }}
          disabled={generating}
          className="ml-auto flex items-center gap-1 rounded-lg border border-violet-500/30 px-2 py-1 text-[10px] font-medium text-violet-400 hover:bg-violet-500/10 disabled:opacity-40"
        >
          {generating
            ? <><Loader2 className="h-3 w-3 animate-spin" /> Generating...</>
            : <><RefreshCw className="h-3 w-3" /> {tldrs.length > 0 ? 'Regenerate' : 'Generate'}</>
          }
        </button>
      </div>

      {!collapsed && (
        <>
          {tldrs.length === 0 && !generating && (
            <p className="text-xs text-zinc-500">
              No TL;DRs yet. Click Generate to have each agent summarize the discussion.
            </p>
          )}

          {generating && tldrs.length === 0 && (
            <div className="flex items-center gap-2 py-2 text-xs text-zinc-400">
              <Loader2 className="h-3 w-3 animate-spin" />
              Collecting TL;DRs from agents...
            </div>
          )}

          <div className="space-y-2">
            {tldrs.map(t => (
              <div
                key={`${t.agentId}-${t.id}`}
                className="rounded-lg bg-zinc-800/40 p-2.5"
              >
                <span className="text-xs font-medium text-zinc-300">{t.agentName}</span>
                <div className="fd-markdown mt-1 text-xs text-zinc-400 leading-relaxed">
                  <Markdown remarkPlugins={[remarkGfm]}>{t.content}</Markdown>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
