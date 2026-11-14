import { ThumbsUp, ThumbsDown, Minus } from 'lucide-react'
import type { CouncilVote } from '../../stores/councilStore'
import { SectionLabel } from './councilLook'

interface VotingPanelProps {
  votes: CouncilVote[]
}

export function VotingPanel({ votes }: VotingPanelProps) {
  const agree = votes.filter(v => v.vote === 'agree')
  const disagree = votes.filter(v => v.vote === 'disagree')
  const abstain = votes.filter(v => v.vote === 'abstain')
  const total = votes.length

  return (
    <div className="space-y-3">
      <SectionLabel
        aside={
          <span className="flex items-center gap-3 text-xs font-medium tabular-nums">
            <span className="flex items-center gap-1 text-emerald-700 dark:text-emerald-400">
              <ThumbsUp className="h-3.5 w-3.5" /> {agree.length}
            </span>
            <span className="flex items-center gap-1 text-red-700 dark:text-red-400">
              <ThumbsDown className="h-3.5 w-3.5" /> {disagree.length}
            </span>
            <span className="flex items-center gap-1 text-zinc-500">
              <Minus className="h-3.5 w-3.5" /> {abstain.length}
            </span>
          </span>
        }
      >
        Votes
      </SectionLabel>

      {/* Summary bar */}
      {total > 0 && (
        <div className="flex h-2 w-full gap-0.5 overflow-hidden rounded-full bg-zinc-800">
          {agree.length > 0 && (
            <div
              className="rounded-full bg-emerald-500"
              style={{ width: `${(agree.length / total) * 100}%` }}
            />
          )}
          {disagree.length > 0 && (
            <div
              className="rounded-full bg-red-500"
              style={{ width: `${(disagree.length / total) * 100}%` }}
            />
          )}
          {abstain.length > 0 && (
            <div
              className="rounded-full bg-zinc-500"
              style={{ width: `${(abstain.length / total) * 100}%` }}
            />
          )}
        </div>
      )}

      {/* Individual votes */}
      <div className="space-y-1.5">
        {votes.map(v => (
          <div
            key={v.id || `${v.agentId}-${v.round}`}
            className="flex items-start gap-2.5 rounded-xl bg-zinc-950/40 px-3 py-2 text-xs"
          >
            {v.vote === 'agree' && <ThumbsUp className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />}
            {v.vote === 'disagree' && <ThumbsDown className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-600 dark:text-red-400" />}
            {v.vote === 'abstain' && <Minus className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />}
            <div className="min-w-0 leading-relaxed">
              <span className="font-semibold text-zinc-200">{v.agentName}</span>
              {v.reason && <span className="ml-1 text-zinc-500">— {v.reason}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
