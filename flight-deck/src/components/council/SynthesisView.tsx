import { FileText } from 'lucide-react'
import type { CouncilMessage } from '../../stores/councilStore'

interface SynthesisViewProps {
  message: CouncilMessage
}

export function SynthesisView({ message }: SynthesisViewProps) {
  return (
    <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <FileText className="h-4 w-4 text-cyan-400" />
        <h3 className="text-sm font-medium text-cyan-300">Council Synthesis</h3>
        <span className="text-[10px] text-zinc-500">by {message.agentName}</span>
      </div>
      <div className="prose prose-sm prose-invert max-w-none text-zinc-300 whitespace-pre-wrap leading-relaxed">
        {message.content}
      </div>
    </div>
  )
}
