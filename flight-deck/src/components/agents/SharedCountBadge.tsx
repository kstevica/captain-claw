import { Users } from 'lucide-react'
import { useSharedAgentStore } from '../../stores/sharedAgentStore'
import { sharedCountLabel, sharedCountTitle } from '../../utils/sharedAgent'

// The owner's "shared · N" next to an agent of theirs that has members.
// Reads only its own agent's counts (two numbers), so the shared-agent poll
// never re-renders the card for another agent. Nothing when Flight Deck
// doesn't report counts, or nobody has this agent.

export function SharedCountBadge({ agentRef }: { agentRef?: string }) {
  const members = useSharedAgentStore((s) => (agentRef ? Number(s.mine[agentRef]?.members) || 0 : 0))
  const google = useSharedAgentStore((s) => (agentRef ? Number(s.mine[agentRef]?.google) || 0 : 0))
  if (members <= 0) return null
  return (
    <span
      className="flex shrink-0 items-center gap-0.5 rounded border border-sky-500/25 bg-sky-500/15 px-1 py-0.5 text-[9px] font-medium text-sky-700 dark:text-sky-300"
      title={sharedCountTitle(members, google)}
    >
      <Users className="h-2.5 w-2.5" />{sharedCountLabel(members)}
    </span>
  )
}
