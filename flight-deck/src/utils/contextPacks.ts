// Pure helpers and texts for context packs ("Shared context"): a user shares
// their own profile, a folder (read-only) or their deep memory with everyone
// who uses an agent — its owner everywhere the agent answers or works for them
// (Flight Deck chats, its channels and automations) and every member in their
// own chats. Kept free of imports so the Node tests can lift these
// declarations straight out of the source.

type PackKind = 'profile' | 'vfs' | 'deep_memory'

export const PACKS_TITLE = 'Share with this agent’s people'
export const PACKS_BUTTON = 'Shared context'
export const PACKS_MENU_LABEL = 'Shared context…'
export const PROFILE_PACK_LABEL = 'My profile (about me, company)'
export const PROFILE_PACK_HINT = 'Shares the About me and Company from your Profile — never your standing preferences.'
export const VFS_PACK_LABEL = 'Folders (read-only)'
export const VFS_PACK_HINT = 'The agent can read the folder’s files (not hidden ones) but can’t change, move or delete anything. Folders linked from elsewhere and Google Drive folders can’t be shared. The agent reaches a folder as vfs:@<name>; the name starts with your own prefix and stays yours on this agent. Stop sharing a folder before you delete it.'
export const DEEP_PACK_LABEL = 'My deep memory'
export const DEEP_PACK_HINT = 'The agent’s deep-memory searches also cover your deep memory — all of it, including documents you indexed from your files or Google Drive, or only the entries that carry the tags you pick. Files and Drive documents Flight Deck indexed for you carry no tags, so a tag choice leaves them out; files your agents indexed carry your agents’ tags, so picking one of those tags includes them. Nobody else can add to it or delete from it.'
export const TAGS_LABEL = 'Only entries tagged (optional)'
export const NO_TAGS_NOTE = 'Your deep memory has no tags, so it can only be shared whole.'
export const DOCKER_PACKS_NOTE = 'On a Docker agent only your profile can be shared.'
export const AGENT_OUTDATED_NOTE = 'This agent runs an older version that ignores shared context. Ask its owner to restart it (a Docker agent needs its image rebuilt or pulled first). Until then, vfs:@ names typed in its chats open the person’s own folder of that name, not the shared one.'
/** AGENT_OUTDATED_NOTE as the agent's owner reads it: they are the one to restart it. */
export const AGENT_OUTDATED_OWNER_NOTE = 'This agent runs an older version that ignores shared context. Restart this agent (Docker: rebuild or pull its image first) so it uses shared context. Until then, vfs:@ names typed in its chats open the person’s own folder of that name, not the shared one.'
export const AGENT_NOT_RUNNING_NOTE = 'This agent isn’t running, so its version can’t be checked. Shared context takes effect once it runs a current version.'
/** The agent runs, but Flight Deck's version check got no answer (busy, timed out). */
export const AGENT_UNCHECKED_NOTE = 'Couldn’t check this agent’s version. If it runs an older one, it ignores shared context until it’s restarted, and vfs:@ names typed in its chats open the person’s own folder of that name, not the shared one.'
export const REVOKE_NOTE = 'Stopping takes effect on the agent’s next step for folders and deep memory, and from its next message for profiles (a reply already running can keep a profile for up to about 10 minutes). It doesn’t make the agent unlearn what it already read.'
export const ACTIVE_PACKS_HEADING = 'Shared on this agent'
export const NO_PACKS_TEXT = 'Nothing is shared on this agent yet.'
export const MY_PACKS_HEADING = 'Shared with agents’ people'
export const MY_PACKS_EMPTY = 'You don’t share anything with an agent’s people yet. Open an agent (yours or one shared with you) and choose “Shared context”.'
export const INACTIVE_HINT = 'Inactive: you no longer have access to that agent, it changed owner, or the folder was deleted or replaced.'
/** Profile card with agent sharing turned off: FD keeps the packs (and lets
 *  you stop them) but none is in effect until sharing is back. */
export const PACKS_PAUSED_NOTE = 'Paused (sharing is off on this deck) — they come back when it’s turned on.'
/** The Profile page's "Shared with agents’ people" card (the About me note links to it). */
export const MY_PACKS_ID = 'fd-my-context-packs'
export const ALIAS_ERROR = 'Use lowercase letters, digits or dashes — the whole name at most 40 characters'
/** Packs reach the owner's channels and automations too — every publishing text says so. */
export const CHANNELS_NOTE = 'People who reach this agent through its channels (WhatsApp and glasses, Telegram, Slack, Discord, the API), its automations and the owner’s other agents — and anyone it emails or messages — may see what you share.'

/** Who gets what I share on this agent — shown at the top of the dialog. The
 *  owner also learns that members' packs run on the owner's channels and
 *  automations (`runtime` 'docker': members can share only their profile). */
export function packsDisclosure(agentName: string, ownerName: string, role: 'owner' | 'member', runtime = 'process'): string {
  const agent = (agentName || '').trim() || 'this agent'
  const owner = (ownerName || '').trim() || 'its owner'
  const tail = 'Your name and whether you’re a member or the owner are shown next to it. Anything the agent reads from it becomes shared knowledge: it can come up later in any of the agent’s chats and channels, and it stays after you stop sharing — stopping doesn’t make the agent unlearn.'
  if (role === 'owner') {
    const theirs = runtime === 'docker' ? 'their own profile' : 'their own profile, folders and deep memory'
    return `What you share here is available to everyone who uses ${agent}: you, everywhere the agent answers or works for you (your Flight Deck chats, its channels and its automations), and every member, in their own chats. ${CHANNELS_NOTE} Who can reach it there is up to you. ` + tail
      + ` Members can also share ${theirs} with this agent. That is used on every turn, including your channels and automations. You’re notified and can remove any of it under “${ACTIVE_PACKS_HEADING}” below.`
  }
  return `What you share here is available to everyone who uses ${agent}: ${owner}, everywhere the agent answers or works for them (their Flight Deck chats, its channels and its automations), and every member, in their own chats. ${CHANNELS_NOTE} Who can reach it there is up to ${owner}. ` + tail
}

/** The folder name Flight Deck derives when the publisher picks none:
 *  `<prefix>-<folder slug>` (FD's derive_alias; it adds -2 … -9 when taken). */
export function derivedAlias(prefix: string, project: string): string {
  // NFKD, then drop every non-ASCII code unit (FD: encode('ascii', 'ignore')).
  const slug = (project || '').normalize('NFKD').replace(/[\u0080-￿]/g, '')
    .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
  return `${prefix}-${slug || 'folder'}`.slice(0, 40).replace(/-+$/, '')
}

/** The confirm shown before publishing anything. A folder names itself and
 *  its vfs:@ name (`alias` '' = the one FD derives from `prefix`). */
export function packConfirmText(
  kind: PackKind | string,
  agentName: string,
  tags: string[] = [],
  folder: { project?: string; alias?: string; prefix?: string } = {},
): string {
  const agent = (agentName || '').trim() || 'this agent'
  const list = Array.isArray(tags) ? tags : []
  let first: string
  if (kind === 'profile') {
    first = `Share your About me and Company with everyone who uses ${agent}?`
  } else if (kind === 'vfs') {
    const project = (folder.project || '').trim()
    const alias = (folder.alias || '').trim()
    const prefix = (folder.prefix || '').trim()
    const as = alias ? ` (as vfs:@${alias})`
      : prefix && project ? ` (as vfs:@${derivedAlias(prefix, project)}, or the next free name)` : ''
    first = project
      ? `Share your folder “${project}”${as}, read-only, with everyone who uses ${agent}?`
      : `Share this folder, read-only, with everyone who uses ${agent}?`
  } else if (kind === 'deep_memory') {
    first = list.length
      ? `Let everyone who uses ${agent} search your deep memory (only entries tagged ${list.join(', ')})?`
      : `Let everyone who uses ${agent} search ALL of your deep memory, including documents indexed from your files and Google Drive?`
  } else {
    first = `Share this with everyone who uses ${agent}?`
  }
  return first + '\n\n' + CHANNELS_NOTE + ' Your name is shown next to it, and what the agent reads from it can become shared knowledge.'
}

/** One line naming what a pack shares. */
export function packSummary(p: { kind: string; project?: string; alias?: string; tags?: string[] }): string {
  if (p.kind === 'profile') return 'Profile (about me, company)'
  if (p.kind === 'vfs') return `Folder “${p.project || ''}” as vfs:@${p.alias || ''}`
  if (p.kind === 'deep_memory') {
    return p.tags?.length ? `Deep memory (tags: ${p.tags.join(', ')})` : 'Deep memory (all of it)'
  }
  return 'Shared context'
}

// ── Who published a pack ──
//
// A display name is whatever its person typed, so it never carries the role:
// FD's `role` is a separate badge on every row and FD's `owner_tag` tells two
// publishers with one name apart. Names are cleaned as FD cleans them for the
// agent, and only the caller's own rows say You (anyone else's name is quoted —
// a member named "You" reads “You”).

/** A publisher's name as FD cleans it (safe_name): NFKC; format, control and
 *  unassigned characters dropped; anything but letters, marks, digits, spaces
 *  and . ' - (brackets included) becomes a space; one line. */
export function cleanPublisherName(name: string): string {
  return String(name || '').normalize('NFKC')
    .replace(/[\p{Cf}\p{Cs}\p{Co}\p{Cn}]/gu, '')
    .replace(/\s/g, ' ')
    .replace(/\p{Cc}/gu, '')
    .replace(/[^\p{L}\p{M}\p{Nd}.' -]/gu, ' ')
    .replace(/-{2,}/g, ' ')
    .split(' ').filter(Boolean).join(' ')
}

/** 'You' on my own rows; anyone else's cleaned name in quotes ('Someone' when blank). */
export function publisherName(p: { mine?: boolean; owner_name?: string }): string {
  if (p.mine) return 'You'
  const name = cleanPublisherName(p.owner_name || '')
  return name ? `“${name}”` : 'Someone'
}

/** The role badge, from FD's role only. */
export function packRoleBadge(role: string | undefined): 'owner' | 'member' {
  return role === 'owner' ? 'owner' : 'member'
}

/** FD's collision tag ('#3f9a') when it sent one; '' otherwise. */
export function packOwnerTag(p: { owner_tag?: string }): string {
  const tag = String(p.owner_tag || '').trim().replace(/^#/, '')
  return /^[0-9a-f]{1,16}$/i.test(tag) ? `#${tag}` : ''
}

/** Who published a pack, in plain text (the remove confirm):
 *  'You', or `“Ana” (member)` / `“Olga” (owner)` / `“Ana” (member, #3f9a)`. */
export function packOwnerLabel(p: { mine?: boolean; owner_name?: string; role?: string; owner_tag?: string }): string {
  if (p.mine) return 'You'
  const tag = packOwnerTag(p)
  return `${publisherName(p)} (${packRoleBadge(p.role)}${tag ? `, ${tag}` : ''})`
}

/** The owner's confirm before removing someone else's pack. */
export function removePackConfirm(p: {
  mine?: boolean; owner_name?: string; role?: string; owner_tag?: string
  kind: string; project?: string; alias?: string; tags?: string[]
}): string {
  return `Remove ${packSummary(p)}, shared by ${packOwnerLabel(p)}, from this agent?`
}

/** The agents my profile is shared on (active profile packs), each once. */
export function profilePackAgents(packs: readonly { kind: string; active?: boolean; agent_name?: string }[]): string[] {
  const names: string[] = []
  for (const p of packs || []) {
    if (p.kind !== 'profile' || p.active === false) continue
    const n = (p.agent_name || '').trim() || 'an agent'
    if (!names.includes(n)) names.push(n)
  }
  return names
}

/** Above About me / My company while my profile is shared on agents; '' when it isn't. */
export function profileSharedNote(agentNames: readonly string[]): string {
  const names = (agentNames || []).filter((n) => n.trim())
  if (names.length === 0) return ''
  const list = names.length === 1 ? names[0]
    : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`
  return `About me and My company are also shared with everyone who uses ${list} — including their channels and automations. Saving updates what they see.`
}

/** Pick or unpick a deep-memory tag; at most `max` are picked. */
export function toggleTag(selected: string[], tag: string, max = 10): string[] {
  if (selected.includes(tag)) return selected.filter((t) => t !== tag)
  if (selected.length < max) return [...selected, tag]
  return selected
}

/** The full folder name: the publisher's fixed prefix and their chosen
 *  suffix. '' = let Flight Deck derive one from the folder's name. */
export function fullAlias(prefix: string, suffix: string): string {
  const s = (suffix || '').trim()
  if (s === '') return ''
  return `${prefix}-${s}`
}

/** '' when the suffix is empty or makes a valid name; else ALIAS_ERROR. */
export function aliasSuffixError(prefix: string, suffix: string): string {
  if ((suffix || '').trim() === '') return ''
  return /^[a-z0-9][a-z0-9-]{0,39}$/.test(fullAlias(prefix, suffix)) ? '' : ALIAS_ERROR
}

/** What to say about the running agent's support for shared context:
 *  the owner restarts it themselves, a member asks the owner; a running agent
 *  whose version check failed is not called "not running". */
export function capabilityNote(
  supports: boolean | null | undefined,
  role: 'owner' | 'member' | string = 'member',
  agentRunning?: boolean | null,
  ownerName = '',
): string {
  if (supports === true) return ''
  if (supports === false) {
    if (role === 'owner') return AGENT_OUTDATED_OWNER_NOTE
    const owner = (ownerName || '').trim()
    // A function replacement: a name is never read as a `$&` pattern.
    return owner ? AGENT_OUTDATED_NOTE.replace('Ask its owner', () => `Ask ${owner}`) : AGENT_OUTDATED_NOTE
  }
  return agentRunning === true ? AGENT_UNCHECKED_NOTE : AGENT_NOT_RUNNING_NOTE
}

/** The capability note is a warning (amber), not just information. */
export function capabilityWarns(supports: boolean | null | undefined, agentRunning?: boolean | null): boolean {
  return supports === false || (supports !== true && agentRunning === true)
}
