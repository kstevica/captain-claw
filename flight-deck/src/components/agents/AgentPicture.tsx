import { useEffect, useState } from 'react'
import { avatarImageNow, loadAvatarImage } from '../../services/agentAvatars'
import { useAgentAvatarStore } from '../../stores/agentAvatarStore'

// What fills an agent's square wherever it is shown: the picture its person
// gave it (an emoji or their own image), else its initials. The square itself
// — size, colour, corners, the dots on them — stays the caller's; it must be
// `relative`, since an image covers it.

export function agentInitials(name: string): string {
  const words = name.trim().split(/[\s_-]+/).filter(Boolean)
  if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase()
  return name.trim().slice(0, 2).toUpperCase() || '?'
}

/** The icons a person picks from, with what a screen reader calls each. */
export const AGENT_EMOJIS: { emoji: string; label: string }[] = [
  { emoji: '🤖', label: 'Robot' },
  { emoji: '🧠', label: 'Brain' },
  { emoji: '💼', label: 'Briefcase' },
  { emoji: '📊', label: 'Chart' },
  { emoji: '📚', label: 'Books' },
  { emoji: '✍️', label: 'Writing hand' },
  { emoji: '🔍', label: 'Magnifier' },
  { emoji: '💡', label: 'Light bulb' },
  { emoji: '🎯', label: 'Target' },
  { emoji: '📬', label: 'Mailbox' },
  { emoji: '📅', label: 'Calendar' },
  { emoji: '💬', label: 'Speech bubble' },
  { emoji: '🛠️', label: 'Tools' },
  { emoji: '🧪', label: 'Test tube' },
  { emoji: '🎨', label: 'Palette' },
  { emoji: '💰', label: 'Money bag' },
  { emoji: '🌍', label: 'Globe' },
  { emoji: '🚀', label: 'Rocket' },
  { emoji: '⚡', label: 'Lightning' },
  { emoji: '⚓', label: 'Anchor' },
  { emoji: '🧭', label: 'Compass' },
  { emoji: '🦉', label: 'Owl' },
  { emoji: '🦊', label: 'Fox' },
  { emoji: '🐙', label: 'Octopus' },
  { emoji: '🐝', label: 'Bee' },
  { emoji: '🦁', label: 'Lion' },
  { emoji: '🐼', label: 'Panda' },
  { emoji: '🐬', label: 'Dolphin' },
  { emoji: '🦄', label: 'Unicorn' },
  { emoji: '🐢', label: 'Turtle' },
]

export type PictureSize = 'sm' | 'md' | 'lg' | 'xl'

// The emoji's size in each square: 32px, 36–40px, 48px, 80px.
const EMOJI_SIZE: Record<PictureSize, string> = {
  sm: 'text-[18px]',
  md: 'text-xl',
  lg: 'text-2xl',
  xl: 'text-[44px]',
}

/** An uploaded picture's URL, once it has been read (null until then, or when it can't be). */
export function useAvatarImage(id: string | undefined): string | null {
  const [url, setUrl] = useState<string | null>(() => (id ? avatarImageNow(id) : null))
  useEffect(() => {
    if (!id) { setUrl(null); return }
    const now = avatarImageNow(id)
    setUrl(now)
    if (now) return
    let live = true
    void loadAvatarImage(id).then((u) => { if (live) setUrl(u) })
    return () => { live = false }
  }, [id])
  return url
}

export function AgentPicture({ agentId, name, size = 'sm', dim = false }: {
  agentId: string
  name: string
  size?: PictureSize
  /** Switched off: faded, as its initials would be. */
  dim?: boolean
}) {
  const avatar = useAgentAvatarStore((s) => s.avatars[agentId])
  const imageId = avatar && 'image' in avatar ? avatar.image : undefined
  const url = useAvatarImage(imageId)
  const fade = dim ? 'opacity-60 saturate-50' : ''

  if (imageId && url) {
    return <img src={url} alt="" draggable={false} className={`absolute inset-0 h-full w-full rounded-[inherit] object-cover ${fade}`} />
  }
  if (avatar && 'emoji' in avatar) {
    return <span aria-hidden className={`select-none leading-none ${EMOJI_SIZE[size]} ${fade}`}>{avatar.emoji}</span>
  }
  // No picture, or one still on its way: the initials.
  return <>{agentInitials(name)}</>
}
