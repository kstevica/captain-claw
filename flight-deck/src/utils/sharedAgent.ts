// Pure helpers for chatting with an agent another deck user shared with you.
//
// A shared agent is reached ONLY through Flight Deck's member route: the
// browser names the agent by its `agent_ref` and never learns (or sends) the
// agent's host, port or access token. Kept free of imports so the Node tests
// can lift these declarations straight out of the source.

/** Chat sessions with a shared agent are keyed `shared:<agent_ref>` (lane A). */
export const SHARED_PREFIX = 'shared:'

export function sharedContainerId(agentRef: string): string {
  return `${SHARED_PREFIX}${agentRef}`
}

// ── What this browser keeps per member ──
//
// A shared agent is used alike by every member of it, and several members can
// use one browser (a kiosk machine). So whatever the browser keeps about a
// shared chat — the queue and plan slices, the notice dismissal — is kept per
// deck user, not per agent: a session that ends without a sign-out teardown
// (expired refresh cookie, an admin reset) must not hand the next person the
// last one's queue, nor a notice they never read.

/** The deck user these browser-local keys belong to (`'local'` with auth off). */
function sliceUser(userId: string | null | undefined): string {
  return String(userId || '').trim() || 'local'
}

/** The id a chat's localStorage slices are kept under: a shared chat's lane
 *  key gains `@<user id>` (as FD stores its history); any other is unchanged. */
export function sharedSliceId(chatKey: string, userId: string | null | undefined): string {
  return chatKey.startsWith(SHARED_PREFIX) ? `${chatKey}@${sliceUser(userId)}` : chatKey
}

export const SHARED_ACK_PREFIX = 'fd.sharedAgentAck.'

/** localStorage key recording that this user dismissed this agent's notice. */
export function sharedAckKey(userId: string | null | undefined, agentRef: string): string {
  return `${SHARED_ACK_PREFIX}${sliceUser(userId)}.${agentRef}`
}

// ── Finding a shared agent from its bell notification ──

/** The "Shared with me" section of the Agent Desktop (scrolled to from the bell). */
export const SHARED_SECTION_ID = 'fd-shared-with-me'

/** Where a member finds an agent shared with them, in the layout they use. */
export function sharedAgentWhere(layout: 'full' | 'simple'): string {
  return layout === 'simple'
    ? 'Find it in your agents list, marked “shared”.'
    : 'Find it under “Shared with me” on the Agent Desktop.'
}

export type SharedNotificationAction =
  | { kind: 'none' }
  | { kind: 'chat' | 'reveal'; agentRef: string; name: string; ownerName: string; hint: string }

/** What clicking a bell notification about an agent does: chat with it when
 *  it is still shared with you and running, show where it lives when it is
 *  stopped, nothing when it isn't shared with you (any more). */
export function sharedNotificationAction(
  refType: string | undefined,
  refId: string | undefined,
  agents: readonly {
    agent_ref: string; name: string; slug: string; status: string
    owner_name: string; owner_email: string
  }[],
  layout: 'full' | 'simple',
): SharedNotificationAction {
  if (refType !== 'agent' || !refId) return { kind: 'none' }
  const a = agents.find((x) => x.agent_ref === refId)
  if (!a) return { kind: 'none' }
  const name = a.name || a.slug
  const ownerName = a.owner_name || a.owner_email || 'another user'
  const where = sharedAgentWhere(layout)
  return a.status === 'running'
    ? { kind: 'chat', agentRef: a.agent_ref, name, ownerName, hint: `${where} Click to chat with it.` }
    : { kind: 'reveal', agentRef: a.agent_ref, name, ownerName,
        hint: `${where} It's stopped right now — ask ${ownerName} to start it.` }
}

/** The member socket: agent ref, lane and the caller's FD token — never a
 *  `token`, host or port. */
export function sharedWsUrl(
  agentRef: string,
  lane: string,
  fdToken: string,
  loc: { protocol: string; host: string },
): string {
  const proto = loc.protocol === 'https:' ? 'wss:' : 'ws:'
  const l = (lane || 'A').trim().toUpperCase() || 'A'
  return `${proto}//${loc.host}/fd/agent-ws-shared`
    + `?ref=${encodeURIComponent(agentRef)}`
    + `&lane=${encodeURIComponent(l)}`
    + `&fd_token=${encodeURIComponent(fdToken || '')}`
}

/** What the chat offers after Flight Deck closed a member socket. */
export type SharedRetry = 'refresh' | 'button' | 'none'

/** The reason Flight Deck gives when an owner revokes a member (4403). */
export const GENERIC_REVOKE_REASON = 'Access removed'

export function sharedCloseInfo(
  code: number,
  ownerName: string,
  reason?: string,
): { message: string; retry: SharedRetry } {
  const owner = (ownerName || '').trim()
  const why = (reason || '').trim()
  switch (code) {
    case 4001:
      // Only emitted once the socket has given up (the token refresh failed,
      // or Flight Deck refused the refreshed token too) — nothing reconnects.
      return { message: 'Your session expired — sign in again, or Retry', retry: 'refresh' }
    case 4400:
      return { message: why || "This shared agent can't be opened", retry: 'none' }
    case 4403:
      // Flight Deck says "Access removed" for a revoke; anything else is more
      // specific ("You left this shared agent", "…respawn it to share it") and
      // must not be blamed on the owner.
      return {
        message: why && why !== GENERIC_REVOKE_REASON
          ? why
          : `${owner || 'The owner'} removed your access to this agent`,
        retry: 'none',
      }
    case 4404:
      return { message: 'This agent no longer exists', retry: 'none' }
    case 4409:
      return { message: `This agent is stopped — ask ${owner || 'the owner'} to start it`, retry: 'button' }
    case 4426:
      return { message: `This agent needs a restart before it can be shared — ask ${owner || 'the owner'}`, retry: 'button' }
    case 4429:
      return { message: "Too many open chats with this agent, or it's at capacity — try again shortly", retry: 'button' }
    case 4502:
      return { message: "Couldn't reach the agent", retry: 'button' }
    case 4503:
      return { message: 'Agent sharing is turned off on this Flight Deck', retry: 'none' }
    default:
      return { message: why || 'Disconnected', retry: 'none' }
  }
}

/** Shown to the owner in the Share dialog of an agent. */
export const OWNER_SHARE_NOTE: string =
  'Members chat with this agent in their own private conversations — private from each other, '
  + "not from you or the deck's admins. During member chats the agent can only search the web, read "
  + 'public pages and use its shared insights, playbooks and topics: no shell, files, Google, deep '
  + 'memory, MCP servers, scheduled jobs or your fleet. What it learns from anyone\'s chats, including '
  + 'yours, becomes shared knowledge for everyone using it. Member chats use your LLM keys. '
  + 'If this agent was running before sharing was turned on, restart it once so members can '
  + 'connect.\n\n'
  + 'Anyone on this deck who runs their own shell-capable process agent can act as any agent on '
  + 'this host, including this one.'

/** Shown to a member the first time they open a shared agent. `**…**` is bold. */
export function memberNoticeText(agentName: string, ownerName: string, hostWarning: string): string {
  const agent = (agentName || '').trim() || 'This agent'
  const owner = (ownerName || '').trim() || 'another user'
  return `**${agent} belongs to ${owner}.** Your conversations here are private from other members, `
    + `but not from ${owner} or this deck's admins. The agent can see the profile you set in Flight Deck. `
    + 'What it learns from your chats becomes shared knowledge for everyone using it, including '
    + `${owner}'s own chats with it. In shared chats it can only search the web, read public pages and use its `
    + 'shared insights, playbooks and topics.'
    + (hostWarning ? `\n\n${hostWarning}` : '')
}
