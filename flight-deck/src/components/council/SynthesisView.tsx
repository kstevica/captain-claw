import { useState } from 'react'
import { FileText, ChevronDown, ChevronRight } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { CouncilMessage, CouncilVote } from '../../stores/councilStore'
import { VotingPanel } from './VotingPanel'

interface SynthesisViewProps {
  message: CouncilMessage
  votes?: CouncilVote[]
}

export function SynthesisView({ message, votes }: SynthesisViewProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/5">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center gap-2 p-4 text-left hover:bg-cyan-500/5 transition-colors rounded-xl"
      >
        <FileText className="h-4 w-4 text-cyan-400 shrink-0" />
        <h3 className="text-sm font-medium text-cyan-300">Council Synthesis</h3>
        <span className="text-[10px] text-zinc-500">by {message.agentName}</span>
        {collapsed
          ? <ChevronRight className="h-3.5 w-3.5 text-zinc-500 ml-auto shrink-0" />
          : <ChevronDown className="h-3.5 w-3.5 text-zinc-500 ml-auto shrink-0" />
        }
      </button>
      {!collapsed && (
        <div className="px-4 pb-4">
          <div className="fd-markdown prose prose-sm prose-invert max-w-none text-zinc-300 leading-relaxed">
            <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</Markdown>
          </div>
          {votes && votes.length > 0 && (
            <div className="mt-4 border-t border-cyan-500/20 pt-3">
              <VotingPanel votes={votes} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
