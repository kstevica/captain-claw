import type { LessonsChanged } from '../../services/agentLessons'
import { onboardingAs } from '../../services/onboardingAs'
import { useChatStore } from '../../stores/chatStore'
import { useProcessStore } from '../../stores/processStore'
import { stripOwnerBlock } from './agentSetup'

/** What a lesson change did to an agent's instructions, mirrored into the chat's copy and sent to it
 *  (as Apply does): Flight Deck's stored instructions, or — without accounts — the block put in this
 *  browser's copy. Not for an admin acting for its user (as Way of working's Apply). */
export function syncInstructions(slug: string, res: LessonsChanged) {
  // An admin teaching someone's agent: Flight Deck stored them for its user, whose chat sends them.
  if (onboardingAs.get()) return
  const procs = useProcessStore.getState()
  const had = procs.getFleetInstructions(slug)
  const next = res.stored ? res.instructions : [stripOwnerBlock(had), res.block].filter(Boolean).join('\n\n')
  if (next !== had) {
    void procs.setFleetInstructions(slug, next).then(() => useChatStore.getState().resendFleetIdentity(`proc-${slug}`))
  }
}
