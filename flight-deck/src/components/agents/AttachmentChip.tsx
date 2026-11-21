import { useState } from 'react'
import { AlertCircle, Loader2, X } from 'lucide-react'
import { formatSize } from '../../services/fileTransfer'
import type { Attachment } from '../../hooks/useFileAttachments'
import { kindOf } from './fileKind'
import { ImageLightbox } from './ImageLightbox'

/** A file waiting in a composer (chat, queue, a skill's form) before it is
 *  sent: a small card — what kind of file, its name, its size — or, for a
 *  picture, its thumbnail, which opens it in the viewer. It says while it
 *  uploads and when it failed; the round × takes it off (on hover with a
 *  mouse, always on a touch screen). */
export function AttachmentChip({ attachment, onRemove }: { attachment: Attachment; onRemove: (id: string) => void }) {
  const { name, size, type, preview, status } = attachment
  const [viewing, setViewing] = useState(false)
  const uploading = status === 'uploading' || status === 'pending'
  const failed = status === 'error'
  const kind = kindOf(name, type)
  const title = failed ? `${name} — upload failed${attachment.error ? `: ${attachment.error}` : ''}` : name

  const remove = (
    <button
      type="button"
      onClick={() => onRemove(attachment.id)}
      title={`Remove ${name}`}
      aria-label={`Remove ${name}`}
      className="absolute -right-1.5 -top-1.5 z-10 flex h-5 w-5 items-center justify-center rounded-full border border-zinc-700 bg-zinc-950 text-zinc-400 shadow-sm transition hover:bg-zinc-800 hover:text-zinc-100 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/60 [@media(hover:hover)]:opacity-0 [@media(hover:hover)]:group-hover:opacity-100"
    >
      <X className="h-3 w-3" strokeWidth={2.5} />
    </button>
  )

  // A picture shows itself: a square thumbnail, the name on hover.
  if (preview && !failed) {
    return (
      <div className="fd-rise group relative shrink-0" title={title} data-fd-attachment={status}>
        <button
          type="button"
          onClick={() => setViewing(true)}
          aria-label={`View ${name}`}
          className="block h-14 w-14 overflow-hidden rounded-xl border border-zinc-700/70 bg-zinc-800 shadow-sm transition hover:border-zinc-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/60"
        >
          <img src={preview} alt={name} className={`h-full w-full object-cover transition ${uploading ? 'scale-105 opacity-50 blur-[1px]' : ''}`} />
          {uploading && (
            <div className="absolute inset-0 flex items-center justify-center">
              <Loader2 className="h-4 w-4 animate-spin text-white drop-shadow" />
            </div>
          )}
        </button>
        {remove}
        {viewing && <ImageLightbox items={[{ src: preview, name }]} index={0} onClose={() => setViewing(false)} />}
      </div>
    )
  }

  const Icon = failed ? AlertCircle : kind.Icon
  return (
    <div
      className={`fd-rise group relative flex h-14 w-56 max-w-full shrink-0 items-center gap-2.5 rounded-xl border p-2 pr-3 shadow-sm transition-colors ${
        failed
          ? 'border-red-500/40 bg-red-500/5'
          : 'border-zinc-700/70 bg-[var(--fd-surface-raised)] dark:bg-zinc-800/60'
      }`}
      title={title}
      data-fd-attachment={status}
    >
      <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${
        failed ? 'bg-red-500/10 text-red-600 dark:text-red-400' : kind.tile
      }`}>
        {uploading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Icon className="h-5 w-5" />}
      </div>
      <div className="min-w-0 flex-1 leading-tight">
        <div className="truncate text-[13px] font-medium text-zinc-200">{name}</div>
        <div className={`mt-0.5 truncate text-[11px] ${failed ? 'text-red-600 dark:text-red-400' : 'text-zinc-500'}`}>
          {failed ? 'Upload failed' : uploading ? 'Uploading…' : `${kind.label} · ${formatSize(size)}`}
        </div>
      </div>
      {remove}
    </div>
  )
}
