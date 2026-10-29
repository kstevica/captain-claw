import type { LessonsChanged } from '../../services/agentLessons'
import { useChatStore } from '../../stores/chatStore'
import { useProcessStore } from '../../stores/processStore'
import { stripOwnerBlock } from './agentSetup'

/** What a lesson change did to an agent's instructions, mirrored into the chat's copy and sent to it
 *  (as Apply does): Flight Deck's stored instructions, or — without accounts — the block put in this
 *  browser's copy. */
export function syncInstructions(slug: string, res: LessonsChanged) {
  const procs = useProcessStore.getState()
  const had = procs.getFleetInstructions(slug)
  const next = res.stored ? res.instructions : [stripOwnerBlock(had), res.block].filter(Boolean).join('\n\n')
  if (next !== had) {
    void procs.setFleetInstructions(slug, next).then(() => useChatStore.getState().resendFleetIdentity(`proc-${slug}`))
  }
}
