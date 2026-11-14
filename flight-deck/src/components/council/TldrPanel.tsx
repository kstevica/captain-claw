import { useState } from 'react'
import { RefreshCw, Loader2, MessageSquareQuote } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { CouncilArtifact } from '../../stores/councilStore'
import { Button } from '../console/parts'
import { AgentAvatar, PanelHead } from './councilLook'
import { SMALL_BTN } from './councilStyles'

interface TldrPanelProps {
  tldrs: CouncilArtifact[]
  generating: boolean
  onGenerate: () => void
}

export function TldrPanel({ tldrs, generating, onGenerate }: TldrPanelProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <section className="rounded-2xl border border-zinc-800 bg-zinc-900/40">
      <PanelHead
        icon={MessageSquareQuote}
        tone="violet"
        title="TL;DR"
        collapsed={collapsed}
        onToggle={() => setCollapsed(!collapsed)}
        action={
          <Button
            onClick={(e) => { e.stopPropagation(); onGenerate() }}
            disabled={generating}
            icon={generating ? Loader2 : RefreshCw}
            className={`${SMALL_BTN} ${generating ? '[&>svg]:animate-spin' : ''}`}
          >
            {generating ? 'Generating...' : tldrs.length > 0 ? 'Regenerate' : 'Generate'}
          </Button>
        }
      />

      {!collapsed && (
        <div className="space-y-2 border-t border-zinc-800/80 px-4 pb-4 pt-3">
          {tldrs.length === 0 && !generating && (
            <p className="text-xs text-zinc-500">
              No TL;DRs yet. Click Generate to have each agent summarize the discussion.
            </p>
          )}

          {generating && tldrs.length === 0 && (
            <div className="flex items-center gap-2 py-1 text-xs text-zinc-400">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Collecting TL;DRs from agents...
            </div>
          )}

          {tldrs.length > 0 && (
            <div className="grid gap-2 @xl:grid-cols-2">
              {tldrs.map(t => (
                <div
                  key={`${t.agentId}-${t.id}`}
                  className="flex gap-2.5 rounded-xl border border-zinc-800 bg-zinc-950/40 p-3"
                >
                  <AgentAvatar name={t.agentName} id={t.agentId} size="xs" />
                  <div className="min-w-0 flex-1">
                    <span className="text-xs font-semibold text-zinc-200">{t.agentName}</span>
                    <div className="fd-markdown mt-1 text-xs leading-relaxed text-zinc-400 [&_ol]:list-decimal [&_ul]:list-disc">
                      <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{t.content}</Markdown>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
