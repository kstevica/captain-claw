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
