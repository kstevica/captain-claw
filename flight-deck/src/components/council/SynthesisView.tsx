import { useState } from 'react'
import { FileText, ChevronDown, ChevronRight } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { CouncilMessage } from '../../stores/councilStore'

interface SynthesisViewProps {
  message: CouncilMessage
}

export function SynthesisView({ message }: SynthesisViewProps) {
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
            <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
          </div>
        </div>
      )}
    </div>
  )
}
