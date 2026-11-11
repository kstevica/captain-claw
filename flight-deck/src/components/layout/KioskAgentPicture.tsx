import { useEffect, useRef, useState } from 'react'
import { Check, ImagePlus, Loader2, Smile, Type } from 'lucide-react'
import { AgentDialog } from './KioskAgentDialogs'
import { AGENT_EMOJIS, AgentPicture, agentInitials } from '../agents/AgentPicture'
import { rememberAvatarImage, squareDataUrl, uploadAvatarImage } from '../../services/agentAvatars'
import { refreshAgentAvatars, useAgentAvatarStore } from '../../stores/agentAvatarStore'

// ── An agent's picture ──
//
// What its square in the list shows instead of its initials: one of the icons,
// or a picture of the person's own (cut to a square and scaled down here).
// A pick applies at once; "Initials" puts the letters back.

export function KioskAgentPictureDialog({ agentId, agentName, onClose }: {
  agentId: string
  agentName: string
  onClose: () => void
}) {
  const avatar = useAgentAvatarStore((s) => s.avatars[agentId])
  const setAvatar = useAgentAvatarStore((s) => s.setAvatar)
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const live = useRef(true)
  useEffect(() => {
    void refreshAgentAvatars()
    return () => { live.current = false }
  }, [])

  const emoji = avatar && 'emoji' in avatar ? avatar.emoji : ''
  const hasImage = !!avatar && 'image' in avatar
  const lettersOn = !avatar

  const pickEmoji = (e: string) => {
    setError('')
    setAvatar(agentId, { emoji: e })
  }

  const upload = async (file: File | undefined) => {
    if (!file || uploading) return
    setUploading(true); setError('')
    try {
      const dataUrl = await squareDataUrl(file)
      const id = await uploadAvatarImage(dataUrl)
      rememberAvatarImage(id, dataUrl)
      setAvatar(agentId, { image: id })
    } catch (e) {
      if (live.current) setError(`Couldn't use that picture: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      if (live.current) setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const choice = 'flex h-11 items-center justify-center gap-2 rounded-xl border px-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/50 disabled:opacity-60'

  return (
    <AgentDialog title="Picture" icon={Smile} agentName={agentName} onClose={onClose}>
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto overscroll-contain p-5">
        <div className="flex items-center gap-4">
          <span className="relative flex h-20 w-20 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-2xl font-semibold text-white shadow-lg shadow-violet-500/25">
            {uploading ? <Loader2 className="h-7 w-7 animate-spin" /> : <AgentPicture agentId={agentId} name={agentName} size="xl" />}
          </span>
          <div className="min-w-0 flex-1">
            <div className="truncate text-base font-semibold text-zinc-100">{agentName}</div>
            <p className="mt-0.5 text-xs leading-relaxed text-zinc-500">
              Shown in your agent list instead of its initials. Only you see it, on every device you sign in on.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            className={`${choice} ${hasImage ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-300' : 'border-zinc-700 text-zinc-200 hover:bg-zinc-800'}`}
          >
            <ImagePlus className="h-4 w-4" />
            {hasImage ? 'Another picture' : 'Upload a picture'}
          </button>
          <button
            onClick={() => { setError(''); setAvatar(agentId, null) }}
            disabled={uploading}
            aria-pressed={lettersOn}
            className={`${choice} ${lettersOn ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-300' : 'border-zinc-700 text-zinc-200 hover:bg-zinc-800'}`}
          >
            {lettersOn ? <Check className="h-4 w-4" /> : <Type className="h-4 w-4" />}
            Initials ({agentInitials(agentName)})
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif"
            className="hidden"
            onChange={(e) => void upload(e.target.files?.[0])}
          />
        </div>

        {error && <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">{error}</p>}

        <div>
          <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Or an icon</h3>
          <div className="grid grid-cols-6 gap-1.5" role="radiogroup" aria-label="Icons">
            {AGENT_EMOJIS.map(({ emoji: e, label }) => {
              const on = e === emoji
              return (
                <button
                  key={e}
                  role="radio"
                  aria-checked={on}
                  aria-label={label}
                  title={label}
                  onClick={() => pickEmoji(e)}
                  disabled={uploading}
                  className={`flex aspect-square items-center justify-center rounded-xl text-[26px] leading-none transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/50 ${
                    on ? 'bg-violet-500/15 ring-2 ring-violet-500/60' : 'bg-zinc-900 hover:bg-zinc-800'}`}
                >
                  <span aria-hidden className="select-none">{e}</span>
                </button>
              )
            })}
          </div>
        </div>
      </div>
      <div className="flex shrink-0 justify-end border-t border-zinc-800 px-5 py-3">
        <button onClick={onClose} className="h-9 rounded-lg bg-violet-600 px-4 text-sm font-medium text-white transition-colors hover:bg-violet-500">
          Done
        </button>
      </div>
    </AgentDialog>
  )
}
