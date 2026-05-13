// User-facing MCP client. Hits a user-authed endpoint on flight-deck
// that proxies to the named MCP server. The agent-call endpoint is for
// agents (loopback / shared-secret) — the app-runtime is the user's UI,
// so it needs its own path with session auth.

import { useAuthStore } from '../stores/authStore'

function headers(): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

export interface MCPCallResult {
  ok: boolean
  result?: unknown
  error?: string
}

// Reserved server name that means "use the framework's built-in tool
// catalogue (entity CRUD + file listing) for this app". Generated apps
// that don't bind to an external MCP server fall through to here so
// they can store and retrieve data without any extra provisioning.
export const FRAMEWORK_SERVER = '__framework__'

export async function callTool(
  agentId: string,
  server: string,
  tool: string,
  args: Record<string, unknown> = {},
): Promise<MCPCallResult> {
  const useBuiltin = !server || server === FRAMEWORK_SERVER
  const url = useBuiltin
    ? `/fd/apps/${encodeURIComponent(agentId)}/builtin/call`
    : `/fd/mcp/${encodeURIComponent(server)}/user_call`
  try {
    const resp = await fetch(url, {
      method: 'POST',
      credentials: 'include',
      headers: headers(),
      body: JSON.stringify({ tool, arguments: args }),
    })
    if (!resp.ok) {
      const text = await resp.text().catch(() => '')
      return { ok: false, error: `${resp.status}: ${text || resp.statusText}` }
    }
    const data = await resp.json()
    return { ok: true, result: data?.result }
  } catch (exc) {
    return { ok: false, error: exc instanceof Error ? exc.message : String(exc) }
  }
}

// MCP tool results are typically `{ content: [{ type: "text", text: "..." }] }`.
// Try to extract a list of rows from a tool result, parsing JSON in text blocks
// when present. Renderer doesn't care about MCP framing.
export function extractRows(result: unknown): Record<string, unknown>[] {
  if (!result) return []
  if (Array.isArray(result)) return result as Record<string, unknown>[]

  const r = result as { content?: { type: string; text?: string }[]; structuredContent?: unknown }

  if (r.structuredContent) {
    if (Array.isArray(r.structuredContent)) return r.structuredContent as Record<string, unknown>[]
    const sc = r.structuredContent as { items?: unknown; rows?: unknown; results?: unknown }
    for (const k of ['items', 'rows', 'results'] as const) {
      if (Array.isArray(sc[k])) return sc[k] as Record<string, unknown>[]
    }
  }

  if (Array.isArray(r.content)) {
    for (const block of r.content) {
      if (block.type === 'text' && typeof block.text === 'string') {
        try {
          const parsed = JSON.parse(block.text)
          if (Array.isArray(parsed)) return parsed
          if (parsed && typeof parsed === 'object') {
            for (const k of ['items', 'rows', 'results', 'data'] as const) {
              if (Array.isArray(parsed[k])) return parsed[k]
            }
            return [parsed]
          }
        } catch {
          // not json — leave for extractText
        }
      }
    }
  }
  return []
}

export function extractText(result: unknown): string {
  if (!result) return ''
  if (typeof result === 'string') return result
  const r = result as { content?: { type: string; text?: string }[] }
  if (Array.isArray(r.content)) {
    return r.content
      .filter((b) => b.type === 'text' && typeof b.text === 'string')
      .map((b) => b.text)
      .join('\n\n')
  }
  try { return JSON.stringify(result, null, 2) } catch { return String(result) }
}
