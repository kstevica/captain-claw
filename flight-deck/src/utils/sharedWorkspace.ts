// Pure helpers and texts for a shared agent's saved files and datastore — the
// commons every member of a process agent can open (PR C).
//
// Members see every file in the agent's saved/ folder and every table and row
// in its datastore, the owner's and every member's, each with who created it;
// they change only what they created (the agent enforces it — the UI only
// shows it). A member's panel never sees a host path: a file is named by its
// id relative to saved/. Kept free of imports so the Node tests can lift these
// declarations straight out of the source.

/** Who created a file, table or row. Owner proxies send `owner`/`member`
 *  (`user_id` only to the owner's own panels); Flight Deck's member routes send
 *  `me`/`owner`/`member` and never a user id. */
export interface Creator { kind: 'owner' | 'member' | 'me'; name: string; user_id?: string }

export const MEMBER_FILES_NOTE = 'Everyone who uses this agent can open these files. You can delete only the ones you added.'
export const MEMBER_DATA_NOTE = 'Everyone who uses this agent can see these tables. To add or change data, ask the agent in chat — you can change only what you added.'
export const CHAT_ONLY_TEXT = 'A shared agent is chat only — its files and datastore stay with its owner.'
export const UPLOAD_LABEL = 'Upload'
export const UPLOAD_TITLE = 'Add a file to your own folder on this agent — everyone who uses it can open it'
export const NO_SHARED_FILES = 'No files yet.'
export const NO_SHARED_TABLES = 'No tables yet.'
export const TRUNCATED_NOTE = 'Showing the newest 2,000 files.'
export const CREATED_BY_COLUMN = 'Created by'
export const FILES_BUTTON = 'Files'
export const DATA_BUTTON = 'Data'

/** A member's badge text: "You", "<owner> (owner)" or the member's name. */
export function creatorLabel(c?: Creator | null, ownerName = ''): string {
  if (!c) return ''
  if (c.kind === 'me') return 'You'
  if (c.kind === 'owner') return `${(c.name || ownerName || '').trim() || 'Owner'} (owner)`
  return (c.name || '').trim() || 'A member'
}

/** The owner's badge on something a member added; owner/legacy items get none. */
export function ownerBadgeLabel(c?: Creator | null): string {
  return c && c.kind === 'member' ? `✎ ${(c.name || '').trim() || 'a member'}` : ''
}

export function ownerBadgeTitle(c?: Creator | null): string {
  return c && c.kind === 'member'
    ? `Added by ${(c.name || '').trim() || 'a member'} (a member of this shared agent)`
    : ''
}

export function deleteConfirmText(filename: string): string {
  return `Delete “${filename}”? Everyone who uses this agent loses it.`
}

/** Why this file can't be uploaded ('' when it can) — checked before any
 *  request, against what Flight Deck said it accepts. */
export function uploadError(
  file: { name: string; size: number },
  upload: { max_bytes: number; extensions: string[] },
): string {
  const name = String(file.name || '')
  const dot = name.lastIndexOf('.')
  const ext = dot >= 0 ? '.' + name.slice(dot + 1).toLowerCase() : ''
  if (!ext || !upload.extensions.includes(ext)) return 'That kind of file can’t be uploaded here'
  if (file.size > upload.max_bytes) {
    return `That file is too large (${Math.floor(upload.max_bytes / 1048576)} MB at most)`
  }
  return ''
}

/** Which member panels show: only on a deck that serves them (`member_workspace`)
 *  and only what Flight Deck says this agent offers (a Docker agent: none). */
export function workspaceVisible(
  memberWorkspace: boolean,
  caps: { files: boolean; datastore: boolean },
): { files: boolean; datastore: boolean } {
  return {
    files: memberWorkspace === true && !!caps && caps.files === true,
    datastore: memberWorkspace === true && !!caps && caps.datastore === true,
  }
}

export function isMemberCreated(c?: Creator | null): boolean {
  return c?.kind === 'member'
}

/** A member's file as the shared FileViewer takes it. `physical` is the id
 *  (relative to saved/) — never a host path; the viewer is given the member
 *  routes' URLs, so nothing builds an owner URL from it. */
export function toViewerFile(f: {
  id: string; path: string; filename: string; extension: string; size: number; modified: number
  mime_type: string; is_text: boolean; created_by: Creator
}) {
  return {
    logical: f.path,
    physical: f.id,
    filename: f.filename,
    extension: f.extension,
    exists: true,
    size: f.size,
    modified: f.modified,
    mime_type: f.mime_type,
    is_text: f.is_text,
    source: 'shared',
    created_by: f.created_by,
  }
}
