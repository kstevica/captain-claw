const BASE = '/fd'

export interface AgentFile {
  logical: string
  physical: string
  filename: string
  extension: string
  exists: boolean
  size: number
  modified: number
  mime_type: string
  is_text: boolean
  source: string
}

export interface AgentEndpoint {
  id: string
  name: string
  host: string
  port: number
  auth: string
}

export async function listAgentFiles(host: string, port: number, auth: string): Promise<AgentFile[]> {
  const params = new URLSearchParams({ host, port: String(port) })
  if (auth) params.set('token', auth)
  const resp = await fetch(`${BASE}/agent-files/${encodeURIComponent(host)}/${port}?${auth ? `token=${encodeURIComponent(auth)}` : ''}`)
  if (!resp.ok) throw new Error(`Failed to list files: ${resp.status}`)
  return resp.json()
}

export async function transferFile(
  src: AgentEndpoint,
  dst: AgentEndpoint,
  srcPath: string,
): Promise<{ ok: boolean; filename: string; dest_path: string; size: number }> {
  const resp = await fetch(`${BASE}/transfer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      src_host: src.host,
      src_port: src.port,
      src_auth: src.auth,
      src_path: srcPath,
      dst_host: dst.host,
      dst_port: dst.port,
      dst_auth: dst.auth,
    }),
  })
  if (!resp.ok) {
    const detail = await resp.text()
    throw new Error(`Transfer failed: ${detail}`)
  }
  return resp.json()
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function getDownloadUrl(host: string, port: number, path: string, auth: string): string {
  const params = new URLSearchParams({ path })
  if (auth) params.set('token', auth)
  return `${BASE}/agent-file-download/${encodeURIComponent(host)}/${port}?${params}`
}

export function getViewUrl(host: string, port: number, path: string, auth: string): string {
  const params = new URLSearchParams({ path })
  if (auth) params.set('token', auth)
  return `${BASE}/agent-file-view/${encodeURIComponent(host)}/${port}?${params}`
}

/** Upload a file to an agent */
export async function uploadFileToAgent(
  host: string,
  port: number,
  auth: string,
  file: File,
): Promise<{ path: string; size: number; filename: string }> {
  const params = new URLSearchParams()
  if (auth) params.set('token', auth)
  const qs = params.toString() ? `?${params}` : ''
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch(`${BASE}/agent-file-upload/${encodeURIComponent(host)}/${port}${qs}`, {
    method: 'POST',
    body: form,
  })
  if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`)
  return resp.json()
}

/** Extract folder category from logical path (e.g. "downloads/file.txt" -> "downloads") */
export function getFileCategory(file: AgentFile): string {
  const logical = file.logical || file.physical
  const parts = logical.split('/')
  // logical paths like "downloads/session-id/file.txt" or "showcase/file.txt"
  if (parts.length >= 2) return parts[0]
  return 'other'
}

/** Get file type group for filtering */
export function getFileTypeGroup(file: AgentFile): string {
  const ext = file.extension.toLowerCase()
  if (['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.ico', '.bmp'].includes(ext)) return 'image'
  if (['.mp4', '.webm', '.avi', '.mov', '.mkv'].includes(ext)) return 'video'
  if (['.mp3', '.wav', '.ogg', '.flac', '.aac'].includes(ext)) return 'audio'
  if (['.pdf'].includes(ext)) return 'pdf'
  if (['.html', '.htm'].includes(ext)) return 'html'
  if (['.md', '.markdown'].includes(ext)) return 'markdown'
  if (['.json', '.yaml', '.yml', '.toml', '.xml', '.csv', '.tsv'].includes(ext)) return 'data'
  if (['.js', '.ts', '.jsx', '.tsx', '.py', '.rb', '.go', '.rs', '.java', '.c', '.cpp', '.h', '.css', '.scss'].includes(ext)) return 'code'
  if (['.txt', '.log', '.ini', '.cfg', '.conf', '.env'].includes(ext)) return 'text'
  if (['.zip', '.tar', '.gz', '.bz2', '.7z', '.rar'].includes(ext)) return 'archive'
  if (['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods'].includes(ext)) return 'document'
  return 'other'
}

/** Check if file can be viewed inline in browser */
export function isViewable(file: AgentFile): boolean {
  const group = getFileTypeGroup(file)
  return ['html', 'markdown', 'text', 'code', 'data', 'image', 'pdf'].includes(group)
}
