import type { Creator } from '../../utils/sharedWorkspace'
import {
  CREATED_BY_COLUMN, creatorLabel, isMemberCreated, ownerBadgeLabel, ownerBadgeTitle,
} from '../../utils/sharedWorkspace'

// Who created a file, table or row of a shared agent.
//
// mode="member" — a member's panels: every item says whose it is ("You",
// "Olga (owner)", another member's name). mode="owner" — the owner's own
// panels look as before except on what a member added ("✎ Ana", like the VFS
// browser's author mark); the owner's own and legacy items show nothing.

const MEMBER_TINTS: Record<Creator['kind'], string> = {
  me: 'border-violet-500/25 bg-violet-500/15 text-violet-700 dark:text-violet-300',
  owner: 'border-sky-500/25 bg-sky-500/15 text-sky-700 dark:text-sky-300',
  member: 'border-zinc-700 bg-zinc-800 text-zinc-300',
}

export function CreatorBadge({ creator, ownerName = '', mode }: {
  creator?: Creator | null
  ownerName?: string
  mode: 'member' | 'owner'
}) {
  if (mode === 'owner') {
    if (!isMemberCreated(creator)) return null
    return (
      <span
        className="max-w-[8rem] shrink-0 truncate text-[10px] text-amber-700 dark:text-amber-300"
        title={ownerBadgeTitle(creator)}
      >
        {ownerBadgeLabel(creator)}
      </span>
    )
  }
  const label = creatorLabel(creator, ownerName)
  if (!label || !creator) return null
  const tint = MEMBER_TINTS[creator.kind] || MEMBER_TINTS.member
  return (
    <span
      className={`max-w-[9rem] shrink-0 truncate rounded border px-1 py-0.5 text-[9px] font-medium ${tint}`}
      title={`${CREATED_BY_COLUMN} ${label}`}
    >
      {label}
    </span>
  )
}
