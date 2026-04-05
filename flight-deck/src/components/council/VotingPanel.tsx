import { ThumbsUp, ThumbsDown, Minus } from 'lucide-react'
import type { CouncilVote } from '../../stores/councilStore'

interface VotingPanelProps {
  votes: CouncilVote[]
}

export function VotingPanel({ votes }: VotingPanelProps) {
  const agree = votes.filter(v => v.vote === 'agree')
  const disagree = votes.filter(v => v.vote === 'disagree')
  const abstain = votes.filter(v => v.vote === 'abstain')
  const total = votes.length

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Votes</h3>

      {/* Summary bar */}
      {total > 0 && (
        <div className="flex h-2 w-full overflow-hidden rounded-full">
          {agree.length > 0 && (
            <div
              className="bg-emerald-500"
              style={{ width: `${(agree.length / total) * 100}%` }}
            />
          )}
          {disagree.length > 0 && (
            <div
              className="bg-red-500"
              style={{ width: `${(disagree.length / total) * 100}%` }}
            />
          )}
          {abstain.length > 0 && (
            <div
              className="bg-zinc-500"
              style={{ width: `${(abstain.length / total) * 100}%` }}
            />
          )}
        </div>
      )}

      {/* Counts */}
      <div className="flex items-center gap-3 text-xs">
        <span className="flex items-center gap-1 text-emerald-400">
          <ThumbsUp className="h-3 w-3" /> {agree.length}
        </span>
        <span className="flex items-center gap-1 text-red-400">
          <ThumbsDown className="h-3 w-3" /> {disagree.length}
        </span>
        <span className="flex items-center gap-1 text-zinc-400">
          <Minus className="h-3 w-3" /> {abstain.length}
        </span>
      </div>

      {/* Individual votes */}
      <div className="space-y-1">
        {votes.map(v => (
          <div
            key={v.id || `${v.agentId}-${v.round}`}
            className="flex items-start gap-2 rounded bg-zinc-800/30 px-2 py-1 text-[11px]"
          >
            {v.vote === 'agree' && <ThumbsUp className="h-3 w-3 shrink-0 mt-0.5 text-emerald-400" />}
            {v.vote === 'disagree' && <ThumbsDown className="h-3 w-3 shrink-0 mt-0.5 text-red-400" />}
            {v.vote === 'abstain' && <Minus className="h-3 w-3 shrink-0 mt-0.5 text-zinc-500" />}
            <div>
              <span className="font-medium text-zinc-300">{v.agentName}</span>
              {v.reason && <span className="text-zinc-500 ml-1">— {v.reason}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
