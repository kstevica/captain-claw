import {
  Archive, File as FileGeneric, FileCode, FileSpreadsheet, FileText,
  Film, Globe, Image as ImageIcon, Music, Presentation,
} from 'lucide-react'

export type Kind = { label: string; Icon: typeof FileText; tile: string }

// What the file is, by its extension (the browser's MIME type is often
// empty): a word for under the name, an icon, and the tile's colours. Day
// colours first, night ones under `dark:` (these hues are not remapped).
// The composer's cards (AttachmentChip) and the sent files' pills (FileChips).
export function kindOf(name: string, mime: string): Kind {
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
