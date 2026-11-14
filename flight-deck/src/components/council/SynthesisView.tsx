import { useState } from 'react'
import { FileText } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { CouncilMessage, CouncilVote } from '../../stores/councilStore'
import { VotingPanel } from './VotingPanel'
import { PanelHead } from './councilLook'

interface SynthesisViewProps {
  message: CouncilMessage
  votes?: CouncilVote[]
}

export function SynthesisView({ message, votes }: SynthesisViewProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <section className="rounded-2xl border border-cyan-500/25 bg-cyan-500/[0.04]">
      <PanelHead
        icon={FileText}
        tone="cyan"
        title="Council Synthesis"
        sub={`by ${message.agentName}`}
        collapsed={collapsed}
        onToggle={() => setCollapsed(!collapsed)}
      />
      {!collapsed && (
        <div className="border-t border-cyan-500/15 px-4 pb-4 pt-3">
          <div className="fd-markdown prose prose-sm dark:prose-invert max-w-none text-sm leading-relaxed text-zinc-300 [&_ol]:list-decimal [&_ul]:list-disc">
            <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</Markdown>
          </div>
          {votes && votes.length > 0 && (
            <div className="mt-4 border-t border-cyan-500/15 pt-4">
              <VotingPanel votes={votes} />
            </div>
          )}
        </div>
      )}
    </section>
  )
}
