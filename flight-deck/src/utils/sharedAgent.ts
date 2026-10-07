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

/** Bumped with A2 (members use their own files, deep memory and Google), again
 *  with PR C (the agent's saved files and datastore are a commons), and with
 *  PR D (the owner's agent can look into members' chats): everybody sees the
 *  new notice once, even if they dismissed the old one. */
export const SHARED_ACK_PREFIX = 'fd.sharedAgentAck.v4.'

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

// ── What a member chat on this agent can use ──
//
// Flight Deck says, per shared agent, what a member's own chats may use:
// all on for a process agent (their own Google if they opt in, their deep
// memory, their own VFS folders, the agent's saved files and datastore), all
// off for a Docker agent — that one stays chat-only. A row from an older
// Flight Deck has no capabilities: chat-only.

export interface SharedCaps { google: boolean; deep_memory: boolean; files: boolean; datastore: boolean }

export const CHAT_ONLY_CAPS: SharedCaps = { google: false, deep_memory: false, files: false, datastore: false }

/** A row's capabilities: each one only when Flight Deck says exactly `true`. */
export function sharedCaps(row?: { capabilities?: Partial<SharedCaps> | null }): SharedCaps {
  const c: Partial<SharedCaps> = (row && row.capabilities) || {}
  return {
    google: c.google === true, deep_memory: c.deep_memory === true, files: c.files === true,
    datastore: c.datastore === true,
  }
}

/** Shown to the owner in the Share dialog of an agent. */
export const OWNER_SHARE_NOTE: string =
  'Members chat with this agent in their own private conversations — private from each other, '
  + "not from you or the deck's admins. On a process agent, during a member's chat the agent works "
  + 'with THEIR deep memory, THEIR own files and, only if they turn it on, THEIR Google account — '
  + 'never yours. On a Docker agent, member chats can only search the web, read public pages and '
  + "use the shared insights, playbooks and topics. In member chats it can't use the shell, your "
  + "files or accounts, MCP servers, scheduled jobs or your fleet. What it learns from anyone's "
  + "chats — yours, and what it reads from members' own mail, files and deep memory — becomes "
  + 'shared knowledge for everyone using it. Member chats use your LLM keys. If this agent was '
  + 'running before sharing (or this update) was turned on, restart it once so members can '
  + 'connect.\n\n'
  + 'Anyone on this deck who runs their own shell-capable process agent can act as any agent on '
  + "this host, including this one, and can read every user's Flight Deck files."

/** The owner's side of the saved/ + datastore commons (PR C). Flight Deck
 *  sends the owner this same text once as a bell notification, so it is not
 *  to be reworded here alone. */
export const OWNER_WORKSPACE_NOTE: string =
  'Members can also open everything in this agent’s saved/ folder — including what is already '
  + 'there: files you uploaded in your own chats, screenshots and browser captures, script outputs, '
  + 'the scripts and tools it saved for you (check them for passwords or keys), and what its '
  + 'channels and automations saved, such as WhatsApp or email attachments — and see every table '
  + 'and row in its datastore. They can add their own files, tables and rows, shown with their '
  + 'name, and change or delete only what they added; you can change or delete all of it. Its '
  + 'other files (workspace, output/, workflows/) stay yours. Your agent reads what members add, '
  + 'so treat it as untrusted input, especially in automations.'

/** The owner's Share-dialog paragraph (PR D, r2) on a process agent. Says only
 *  what is enforced: replies of turns that used a member's Google or private
 *  deep memory are hidden (not later ones that repeat them), and a turn that
 *  read members' data is refused insights/playbooks/datastore writes and the
 *  write/edit tools, and learns nothing shared. */
export const OWNER_USAGE_NOTE: string =
  'Your agent can also tell you who it is shared with and look into how they use it whenever you '
  + 'ask — in your Flight Deck chats, your channels and your automations: each member’s activity '
  + 'and token use, what they created or shared on it, and their private conversations with it. It '
  + 'leaves out replies from turns where it used a member’s Google account or private deep memory, '
  + 'though a later reply can repeat what it found there. Members are told. Anyone who can talk to '
  + 'this agent through your channels, the API or your other agents can ask it the same. A turn that '
  + 'reads members’ data adds nothing to the shared insights, playbooks or topics, and can’t save to '
  + 'the datastore or write files with its write and edit tools; after reading their conversations '
  + 'the agent also holds back actions that send or change things until your next message. Whatever '
  + 'you then ask it to save to files, the datastore, insights or playbooks is visible to every '
  + 'member. Treat members’ conversations as untrusted input.'

/** The same paragraph on a Docker agent: its members have no Google or deep
 *  memory and no saved/ or datastore commons — only insights, playbooks and
 *  topics are shared. */
export const OWNER_USAGE_NOTE_DOCKER: string =
  'Your agent can also tell you who it is shared with and look into how they use it whenever you '
  + 'ask — in your Flight Deck chats, your channels and your automations: each member’s activity '
  + 'and token use, what they shared with it, and their private conversations with it. Members are '
  + 'told. Anyone who can talk to this agent through your channels, the API or your other agents can '
  + 'ask it the same. A turn that reads members’ data adds nothing to the shared insights, playbooks '
  + 'or topics; after reading their conversations the agent also holds back actions that send or '
  + 'change things until your next message. Whatever you then ask it to save to insights or '
  + 'playbooks is visible to every member. Treat members’ conversations as untrusted input.'

/** The owner's Share dialog note. On a deck with context packs ("Shared
 *  context") it also says, before the closing host-trust paragraph, that
 *  members can publish their own context to the agent and that it is used on
 *  every turn — the owner's channels and automations too. A Docker agent takes
 *  members' profiles only. On a deck that serves members the agent's saved
 *  files and datastore (`memberWorkspace`), a process agent's note says so
 *  too, and that only the owner's files outside saved/ stay out of reach. On
 *  a deck whose owners' agents can look into members' use (`sharedUsage`,
 *  `shared_agent_usage`), either runtime's note says that last, before the
 *  host-trust paragraph: packs, workspace, usage (a Docker agent's own
 *  variant, `OWNER_USAGE_NOTE_DOCKER`). */
export function ownerShareNote(
  contextPacks: boolean,
  runtime: 'process' | 'docker' = 'process',
  memberWorkspace = false,
  sharedUsage = false,
): string {
  const workspace = memberWorkspace === true && runtime === 'process'
  const usage = sharedUsage === true
  if (!contextPacks && !workspace && !usage) return OWNER_SHARE_NOTE
  const base = workspace
    ? OWNER_SHARE_NOTE.replace('the shell, your files or accounts', 'the shell, your accounts or your files outside its saved/ folder')
    : OWNER_SHARE_NOTE
  const paras: string[] = []
  if (contextPacks) {
    const what = runtime === 'docker' ? 'their own profile' : 'their own profile, folders and deep memory'
    paras.push(`Members can also share ${what} with this agent. That is used on every turn, `
      + "including your channels and automations. You're notified and can remove any of it under "
      + 'Shared context.')
  }
  if (workspace) paras.push(OWNER_WORKSPACE_NOTE)
  if (usage) paras.push(runtime === 'docker' ? OWNER_USAGE_NOTE_DOCKER : OWNER_USAGE_NOTE)
  const cut = base.lastIndexOf('\n\n')
  return `${base.slice(0, cut)}\n\n${paras.join('\n\n')}${base.slice(cut)}`
}

/** The member notice's commons paragraph (PR C): the agent's saved/ folder and
 *  its datastore are shared with everyone who uses it, and who can change what. */
export function memberWorkspaceNotice(ownerName: string): string {
  const owner = (ownerName || '').trim() || 'the owner'
  return `Files saved in your chats here (this agent’s saved/ folder) and its data tables are shared with everyone who uses this agent: ${owner} and every other member can open what you or the agent save there — including anything it saves from your mail, Drive or deep memory — and you can open theirs. Only whoever created a file, table or row, and ${owner}, can change or delete it, and everyone sees who created it.`
}

/** The member notice's paragraph about the owner's agent looking into members' use (PR D, r2). */
export function memberOwnerReadsNotice(ownerName: string): string {
  const owner = (ownerName || '').trim() || 'the owner'
  const Owner = owner.charAt(0).toUpperCase() + owner.slice(1)
  return `${Owner}'s agent can also look into your use of it when ${owner} or this deck's admins ask: that `
    + `you use it and how much, what you created or shared on it, and your conversations here, which it `
    + `can quote. Anyone ${owner} lets talk to the agent can ask it the same — people they connect `
    + `through WhatsApp, Telegram, Slack, Discord or the API, ${owner}'s other agents and its `
    + 'automations — and whoever receives those answers may see what it quotes. What it reads this way '
    + 'isn\'t automatically turned into shared knowledge for other members. It never opens your Google '
    + 'account or your private deep memory for this, and it leaves out its replies from the turns in '
    + 'which it used them for you, though a later reply that repeats what it found there can still be '
    + 'shown to it.'
}

/** Shown to a member the first time they open a shared agent. `**…**` is bold.
 *  `caps` is what their chats on it can use (`sharedCaps(row)`); chat-only —
 *  a Docker agent, an older Flight Deck, no row yet — keeps the A1 text.
 *  `workspace` (this deck serves members the agent's saved files and data)
 *  adds the commons paragraph where the agent's files are on. `ownerReads`
 *  (this deck's owners' agents can look into members' use — `shared_usage`)
 *  adds that paragraph. */
export function memberNoticeText(
  agentName: string,
  ownerName: string,
  hostWarning: string,
  caps: SharedCaps = CHAT_ONLY_CAPS,
  workspace = false,
  ownerReads = false,
): string {
  const agent = (agentName || '').trim() || 'This agent'
  const owner = (ownerName || '').trim() || 'another user'
  const ownData = !!caps && (caps.files || caps.deep_memory || caps.google)
  const head = `**${agent} belongs to ${owner}.** Your conversations here are private from other members, `
    + `but not from ${owner} or this deck's admins. The agent can see the profile you set in Flight Deck. `
    + 'What it learns from your chats becomes shared knowledge for everyone using it, including '
  const body = ownData
    ? `${owner}'s own chats with it — and that includes anything it reads from your mail, calendar, `
      + 'Drive, files or deep memory during a chat. During your chats it can read, change and delete '
      + `your own files (your VFS folders) and use your deep memory — never ${owner}'s — and it uses `
      + 'your Google account, including your Drive folders in Flight Deck, only if you turn that on '
      + 'below (Drive files you indexed into deep memory can still turn up in its deep-memory '
      + 'searches with Google off). While it is '
      + `answering you (up to 20 minutes), ${owner}'s agent process acts with those; your VFS folders `
      + `are files on this computer, so ${owner}, who controls the agent, can read them at any time. `
      + "It can't run commands, use MCP servers or other agents, or schedule anything."
    : `${owner}'s own chats with it. In shared chats it can only search the web, read public pages and use its `
      + 'shared insights, playbooks and topics.'
  const commons = ownData && workspace === true && caps.files === true
    ? ' ' + memberWorkspaceNotice(owner)
    : ''
  const reads = ownerReads === true ? ' ' + memberOwnerReadsNotice(owner) : ''
  return head + body + commons + reads + (hostWarning ? `\n\n${hostWarning}` : '')
}

// ── A member's Google, per shared agent (off until they turn it on) ──

/** An agent name in the middle of a sentence. */
function agentInSentence(agentName: string): string {
  return (agentName || '').trim() || 'this agent'
}

/** The switch's label. */
export function googleOptInLabel(agentName: string): string {
  return `Let ${agentInSentence(agentName)} use my Google during my chats`
}

/** Shown in the confirm dialog when a member turns their Google ON. */
export function googleOptInWarning(agentName: string, ownerName: string): string {
  const agent = agentInSentence(agentName)
  const owner = (ownerName || '').trim() || 'another user'
  return `Let ${agent} use your Google account (Gmail, Calendar, Drive) during your chats with it?\n\n`
    + `It acts as you, through ${owner}'s agent: while one of your messages is being answered (up to 20 `
    + 'minutes), that agent can read your mail, calendar and Drive — including the Drive folders you '
    + 'added to Flight Deck — and act in them as you: write drafts and, if your Gmail send settings '
    + 'allow it, send email as you; create, change or delete calendar events; and upload or overwrite '
    + 'Drive files. A Google access token it gets covers everything you allowed when you connected '
    + `Google and stays valid for about an hour, so ${owner}, who controls the agent, could `
    + 'keep using your account for up to about an hour and a half after your message.\n\n'
    + 'Anything the agent reads from your mail, calendar or Drive can also become shared knowledge '
    + `that other members and ${owner} see.\n\n`
    + `Turn this on only if you trust ${owner}. You can turn it off at any time.`
}

/** Under the switch while the member has no Google account connected. */
export function googleOptInHint(row: { google_enabled?: boolean; google_connected?: boolean }): string {
  return row && row.google_connected === true
    ? ''
    : 'Connect your Google account in Connections first — until then this has no effect.'
}

/** The owner's Share dialog: on a member who turned their Google on. */
export function ownerGoogleBadgeTitle(memberName: string): string {
  const member = (memberName || '').trim() || 'This member'
  return `${member} lets this agent use their Google account during their chats`
}

/** The owner's badge on an agent of theirs that has members. */
export function sharedCountLabel(n: number): string {
  return `shared · ${n}`
}

/** That badge's tooltip. */
export function sharedCountTitle(members: number, google: number): string {
  return `Shared with ${members} member${members === 1 ? '' : 's'}`
    + (google > 0 ? `, ${google} with Google on` : '')
}
