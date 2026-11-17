import {
  AlertCircle, Archive, File as FileGeneric, FileCode, FileSpreadsheet, FileText,
  Film, Globe, Image as ImageIcon, Loader2, Music, Presentation, X,
} from 'lucide-react'
import { formatSize } from '../../services/fileTransfer'
import type { Attachment } from '../../hooks/useFileAttachments'

type Kind = { label: string; Icon: typeof FileText; tile: string }

// What the file is, by its extension (the browser's MIME type is often
// empty): a word for under the name, an icon, and the tile's colours. Day
// colours first, night ones under `dark:` (these hues are not remapped).
function kindOf(name: string, mime: string): Kind {
  const ext = name.includes('.') ? name.slice(name.lastIndexOf('.') + 1).toLowerCase() : ''
  const is = (...xs: string[]) => xs.includes(ext)
  if (mime.startsWith('image/') || is('png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp', 'heic'))
    return { label: 'Picture', Icon: ImageIcon, tile: 'bg-sky-500/10 text-sky-600 dark:bg-sky-400/10 dark:text-sky-300' }
  if (is('pdf'))
    return { label: 'PDF', Icon: FileText, tile: 'bg-red-500/10 text-red-600 dark:bg-red-400/10 dark:text-red-300' }
  if (is('xls', 'xlsx', 'ods', 'csv', 'tsv', 'numbers'))
    return { label: is('csv', 'tsv') ? 'Table' : 'Spreadsheet', Icon: FileSpreadsheet, tile: 'bg-emerald-500/10 text-emerald-700 dark:bg-emerald-400/10 dark:text-emerald-300' }
  if (is('doc', 'docx', 'odt', 'rtf', 'pages'))
    return { label: 'Document', Icon: FileText, tile: 'bg-blue-500/10 text-blue-600 dark:bg-blue-400/10 dark:text-blue-300' }
  if (is('ppt', 'pptx', 'odp', 'key'))
    return { label: 'Presentation', Icon: Presentation, tile: 'bg-orange-500/10 text-orange-600 dark:bg-orange-400/10 dark:text-orange-300' }
  if (mime.startsWith('video/') || is('mp4', 'mov', 'webm', 'mkv', 'avi', 'm4v'))
    return { label: 'Video', Icon: Film, tile: 'bg-pink-500/10 text-pink-600 dark:bg-pink-400/10 dark:text-pink-300' }
  if (mime.startsWith('audio/') || is('mp3', 'wav', 'm4a', 'ogg', 'oga', 'opus', 'flac', 'aac'))
    return { label: 'Audio', Icon: Music, tile: 'bg-amber-500/10 text-amber-700 dark:bg-amber-400/10 dark:text-amber-300' }
  if (is('zip', 'tar', 'gz', 'tgz', 'bz2', '7z', 'rar'))
    return { label: 'Archive', Icon: Archive, tile: 'bg-stone-500/10 text-stone-600 dark:bg-stone-400/10 dark:text-stone-300' }
  if (is('html', 'htm'))
    return { label: 'Page', Icon: Globe, tile: 'bg-orange-500/10 text-orange-600 dark:bg-orange-400/10 dark:text-orange-300' }
  if (is('js', 'ts', 'tsx', 'jsx', 'py', 'rb', 'go', 'rs', 'java', 'c', 'cpp', 'h', 'css', 'scss', 'sh', 'sql', 'json', 'yaml', 'yml', 'toml', 'xml'))
    return { label: ext.toUpperCase(), Icon: FileCode, tile: 'bg-teal-500/10 text-teal-700 dark:bg-teal-400/10 dark:text-teal-300' }
  if (mime.startsWith('text/') || is('txt', 'md', 'markdown', 'log', 'rst'))
    return { label: 'Text', Icon: FileText, tile: 'bg-violet-500/10 text-violet-600 dark:bg-violet-400/10 dark:text-violet-300' }
  return { label: ext ? ext.toUpperCase() : 'File', Icon: FileGeneric, tile: 'bg-zinc-800 text-zinc-400' }
}

/** A file waiting in a composer (chat, queue, a skill's form) before it is
 *  sent: a small card — what kind of file, its name, its size — or, for a
 *  picture, its thumbnail. It says while it uploads and when it failed; the
 *  round × takes it off (on hover with a mouse, always on a touch screen). */
export function AttachmentChip({ attachment, onRemove }: { attachment: Attachment; onRemove: (id: string) => void }) {
  const { name, size, type, preview, status } = attachment
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
        <div className="h-14 w-14 overflow-hidden rounded-xl border border-zinc-700/70 bg-zinc-800 shadow-sm">
          <img src={preview} alt={name} className={`h-full w-full object-cover transition ${uploading ? 'scale-105 opacity-50 blur-[1px]' : ''}`} />
          {uploading && (
            <div className="absolute inset-0 flex items-center justify-center">
              <Loader2 className="h-4 w-4 animate-spin text-white drop-shadow" />
            </div>
          )}
        </div>
        {remove}
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
