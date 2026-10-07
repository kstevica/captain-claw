import { create } from 'zustand'
import { getSharedAgents, setSharedAgentGoogle, type SharedAgent } from '../services/sharedAgents'
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
  /** My own agents that have members (agent_ref → counts), for the owner's
   *  "shared · N" badge. Empty when Flight Deck doesn't say. */
  mine: Record<string, { members: number; google: number }>
  /** Flight Deck offers context packs ("Shared context"). False on an older
   *  deck or with sharing off — every pack affordance hides. */
  contextPacks: boolean
  /** Flight Deck serves members a process agent's saved files and datastore
   *  (the commons panels). False on an older deck or with sharing off — every
   *  member panel and notice paragraph hides (chat-only, as before). */
  memberWorkspace: boolean
  /** Flight Deck tells members and owners that an owner's agent can look into
   *  members' use (PR D). False on an older deck or with sharing off — the
   *  notice paragraphs hide. */
  sharedUsage: boolean
  loaded: boolean
  fetch: () => Promise<void>
  /** A member turns their Google on/off for one agent shared with them.
   *  Optimistic; on failure the old value comes back and the error is
   *  rethrown for the caller to show. */
  setGoogle: (agentRef: string, enabled: boolean) => Promise<void>
  /** Give up a shared agent: closes its chats here, drops their queue and
   *  plan slices (a later re-share starts clean), then refreshes the list. */
  leave: (agentRef: string, ownerId: string) => Promise<void>
}

export const useSharedAgentStore = create<SharedAgentState>((set, get) => ({
  enabled: false,
  hostWarning: '',
  agents: [],
  mine: {},
  contextPacks: false,
  memberWorkspace: false,
  sharedUsage: false,
  loaded: false,

  fetch: async () => {
    try {
      const r = await getSharedAgents()
      const enabled = r?.enabled === true
      const agents = enabled && Array.isArray(r.agents) ? r.agents : []
      const mine = enabled && r.mine && typeof r.mine === 'object' ? r.mine : {}
      const cur = get()
      // Polled every 10s by always-mounted layouts: keep the list's identity
      // when nothing changed, so they don't re-render for nothing.
      const same = JSON.stringify(cur.agents) === JSON.stringify(agents)
      const sameMine = JSON.stringify(cur.mine) === JSON.stringify(mine)
      set({
        enabled,
        hostWarning: enabled ? String(r.host_warning || '') : '',
        agents: same ? cur.agents : agents,
        mine: sameMine ? cur.mine : mine,
        contextPacks: enabled && r.context_packs === true,
        memberWorkspace: enabled && r.member_workspace === true,
        sharedUsage: enabled && r.shared_usage === true,
        loaded: true,
      })
    } catch {
      // A blip (or a deck that predates sharing) keeps the last list.
    }
  },

  setGoogle: async (agentRef, enabled) => {
    const row = get().agents.find((a) => a.agent_ref === agentRef)
    const before = row ? row.google_enabled : undefined
    const put = (value: boolean | undefined) => set((s) => ({
      agents: s.agents.map((a) => (a.agent_ref === agentRef ? { ...a, google_enabled: value as boolean } : a)),
    }))
    if (row) put(enabled)
    try {
      await setSharedAgentGoogle(agentRef, enabled)
    } catch (e) {
      if (row) put(before)
      throw e
    }
    await get().fetch()
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
