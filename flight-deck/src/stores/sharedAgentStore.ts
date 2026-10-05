import { create } from 'zustand'
import { getSharedAgents, type SharedAgent } from '../services/sharedAgents'
import { leaveShare } from '../services/shares'
import { sharedContainerId } from '../utils/sharedAgent'
import { clearSharedSlices, useChatStore } from './chatStore'

// Agents other deck users shared with you. Deliberately its OWN store: the
// process / container stores assume every row is yours (groups, bulk actions,
// the power switch, owner-only menus), so shared agents never go in there.

interface SharedAgentState {
  /** The deck shares agents (FD_AGENT_SHARING on, auth on). Off hides every
   *  sharing affordance — including the owner's "Share…". */
  enabled: boolean
  hostWarning: string
  agents: SharedAgent[]
  loaded: boolean
  fetch: () => Promise<void>
  /** Give up a shared agent: closes its chats here, drops their queue and
   *  plan slices (a later re-share starts clean), then refreshes the list. */
  leave: (agentRef: string, ownerId: string) => Promise<void>
}

export const useSharedAgentStore = create<SharedAgentState>((set, get) => ({
  enabled: false,
  hostWarning: '',
  agents: [],
  loaded: false,

  fetch: async () => {
    try {
      const r = await getSharedAgents()
      const enabled = r?.enabled === true
      const agents = enabled && Array.isArray(r.agents) ? r.agents : []
      const cur = get()
      // Polled every 10s by always-mounted layouts: keep the list's identity
      // when nothing changed, so they don't re-render for nothing.
      const same = JSON.stringify(cur.agents) === JSON.stringify(agents)
      set({
        enabled,
        hostWarning: enabled ? String(r.host_warning || '') : '',
        agents: same ? cur.agents : agents,
        loaded: true,
      })
    } catch {
      // A blip (or a deck that predates sharing) keeps the last list.
    }
  },

  leave: async (agentRef, ownerId) => {
    await leaveShare('agent', agentRef, ownerId)
    const chat = useChatStore.getState()
    const containerId = sharedContainerId(agentRef)
    for (const [key, session] of [...chat.sessions]) {
      if (session.containerId === containerId) chat.disconnectChat(key)
    }
    clearSharedSlices(agentRef)
    await get().fetch()
  },
}))
