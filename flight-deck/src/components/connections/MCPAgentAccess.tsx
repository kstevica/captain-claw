import { useCallback, useEffect, useState } from 'react'
import {
  Plug, Loader2, Check, AlertCircle, Trash2, Plus, Copy, ChevronDown, KeyRound,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'

interface PAT {
  id: string
  name: string
  created_at: string
  last_used_at: string | null
}

interface Grant {
  client_id: string
  client_name: string
  connected_at: string
  last_used_at: string | null
}

function authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

function fmt(ts: string | null): string {
  if (!ts) return 'never'
  try { return new Date(ts).toLocaleString() } catch { return ts }
}

export default function MCPAgentAccess() {
  const [collapsed, setCollapsed] = useState(true)
  const [tokens, setTokens] = useState<PAT[]>([])
  const [grants, setGrants] = useState<Grant[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [freshToken, setFreshToken] = useState<string | null>(null)
  const [copied, setCopied] = useState<'token' | 'cmd' | 'url' | null>(null)

  const endpoint = `${window.location.origin}/fd/mcp-server`

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [tr, gr] = await Promise.all([
        fetch('/fd/mcp-tokens', { headers: authHeaders(), credentials: 'include' }),
        fetch('/fd/oauth-grants', { headers: authHeaders(), credentials: 'include' }),
      ])
      if (!tr.ok) throw new Error(`HTTP ${tr.status}`)
      const td = await tr.json()
      setTokens(Array.isArray(td?.tokens) ? td.tokens : [])
      if (gr.ok) {
        const gd = await gr.json()
        setGrants(Array.isArray(gd?.grants) ? gd.grants : [])
      }
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load tokens')
    }
    setLoading(false)
  }, [])

  const disconnect = async (clientId: string) => {
    if (!confirm('Disconnect this app? It will need to sign in again to reconnect.')) return
    try {
      await fetch(`/fd/oauth-grants/${clientId}`, { method: 'DELETE', headers: authHeaders(), credentials: 'include' })
      await load()
    } catch { /* ignore */ }
  }

  useEffect(() => { if (!collapsed) load() }, [collapsed, load])

  const create = async () => {
    setCreating(true)
    try {
      const r = await fetch('/fd/mcp-tokens', {
        method: 'POST', headers: authHeaders(), credentials: 'include',
        body: JSON.stringify({ name: newName.trim() }),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const data = await r.json()
      setFreshToken(data.token)
      setNewName('')
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create token')
    }
    setCreating(false)
  }

  const revoke = async (id: string) => {
    if (!confirm('Revoke this token? Any MCP client using it will stop working.')) return
    try {
      await fetch(`/fd/mcp-tokens/${id}`, { method: 'DELETE', headers: authHeaders(), credentials: 'include' })
      await load()
    } catch { /* ignore */ }
  }

  const copy = (text: string, which: 'token' | 'cmd' | 'url') => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(which)
      setTimeout(() => setCopied(null), 1500)
    }).catch(() => {})
  }

  const cmd = `claude mcp add --transport http captain-fleet ${endpoint} --header "Authorization: Bearer ${freshToken || '<your-token>'}"`

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
      <button
        type="button"
        onClick={() => setCollapsed((v) => !v)}
        className={'w-full flex items-center gap-3 px-5 py-4 text-left hover:bg-zinc-900/80 transition-colors ' + (collapsed ? '' : 'border-b border-zinc-800')}
        aria-expanded={!collapsed}
      >
        <div className="h-10 w-10 rounded-md bg-gradient-to-br from-violet-500/20 to-sky-500/20 border border-zinc-800 flex items-center justify-center">
          <Plug className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">Agent access for Claude (MCP)</h3>
            {tokens.length > 0 && (
              <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border bg-emerald-500/15 text-emerald-500 border-emerald-500/30">
                {tokens.length} token{tokens.length === 1 ? '' : 's'}
              </span>
            )}
          </div>
          <p className="text-xs text-zinc-500 mt-0.5">Let Claude connect to Flight Deck, list your agents, send them tasks, and read results.</p>
        </div>
        <ChevronDown className={`h-4 w-4 text-zinc-500 transition-transform ${collapsed ? '' : 'rotate-180'}`} />
      </button>

      {!collapsed && (
        <div className="px-5 py-4 space-y-4">
          {error && (
            <div className="flex items-center gap-2 text-xs text-red-400"><AlertCircle className="h-3.5 w-3.5" />{error}</div>
          )}

          {/* Claude Desktop / claude.ai (custom connector via OAuth) */}
          <div className="rounded-md border border-zinc-800 bg-zinc-950/60 p-3 space-y-2">
            <p className="text-xs font-medium text-zinc-300">Claude Desktop / claude.ai</p>
            <p className="text-[11px] text-zinc-500">Add a <span className="text-zinc-400">custom connector</span> and paste this URL — you'll sign in with your Flight Deck account (no token needed):</p>
            <div className="flex items-center gap-2">
              <code className="block flex-1 text-xs text-zinc-300 font-mono bg-zinc-900 rounded p-2 break-all">{endpoint}</code>
              <button onClick={() => copy(endpoint, 'url')} className="shrink-0 rounded p-1.5 text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800" title="Copy URL">
                {copied === 'url' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            </div>
            {grants.length > 0 && (
              <div className="pt-1 space-y-1">
                <p className="text-[11px] text-zinc-500">Connected apps:</p>
                {grants.map((g) => (
                  <div key={g.client_id} className="flex items-center gap-2 rounded border border-zinc-800 bg-zinc-900/60 px-2.5 py-1.5">
                    <Plug className="h-3 w-3 text-emerald-500/70 shrink-0" />
                    <span className="text-xs text-zinc-300 truncate">{g.client_name || 'Custom connector'}</span>
                    <span className="text-[10px] text-zinc-600 ml-auto shrink-0">used {fmt(g.last_used_at)}</span>
                    <button onClick={() => disconnect(g.client_id)} className="shrink-0 rounded p-0.5 text-zinc-500 hover:text-red-400" title="Disconnect">
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Claude Code CLI (static token) */}
          <div className="rounded-md border border-zinc-800 bg-zinc-950/60 p-3 space-y-2">
            <p className="text-xs font-medium text-zinc-300">Claude Code (CLI)</p>
            <p className="text-xs text-zinc-400 pt-1">Create a token below, then run:</p>
            <div className="flex items-start gap-2">
              <code className="block flex-1 text-[11px] text-zinc-300 font-mono bg-zinc-900 rounded p-2 break-all">{cmd}</code>
              <button onClick={() => copy(cmd, 'cmd')} className="shrink-0 rounded p-1.5 text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800" title="Copy command">
                {copied === 'cmd' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            </div>
            <p className="text-[11px] text-zinc-600">Tools exposed: <span className="text-zinc-400">list_agents</span>, <span className="text-zinc-400">send_task</span>, <span className="text-zinc-400">get_result</span>, <span className="text-zinc-400">cancel_task</span>.</p>
          </div>

          {/* Freshly-minted token (shown once) */}
          {freshToken && (
            <div className="rounded-md border border-amber-500/40 bg-amber-500/[0.06] p-3 space-y-2">
              <p className="text-xs text-amber-500/90 flex items-center gap-1"><KeyRound className="h-3.5 w-3.5" /> Copy this token now — it won't be shown again.</p>
              <div className="flex items-center gap-2">
                <code className="block flex-1 text-xs text-zinc-200 font-mono bg-zinc-950 rounded p-2 break-all">{freshToken}</code>
                <button onClick={() => copy(freshToken, 'token')} className="shrink-0 rounded p-1.5 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800" title="Copy token">
                  {copied === 'token' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                </button>
              </div>
              <button onClick={() => setFreshToken(null)} className="text-[11px] text-zinc-500 hover:text-zinc-300">Done</button>
            </div>
          )}

          {/* Create */}
          <div className="flex items-center gap-2">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Token name (e.g. my-laptop)"
              className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
            />
            <button
              onClick={create}
              disabled={creating}
              className="flex items-center gap-1 rounded-md border border-violet-500/40 bg-violet-500/10 px-3 py-2 text-xs font-medium text-violet-200 hover:bg-violet-500/20 disabled:opacity-50"
            >
              {creating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />} Generate token
            </button>
          </div>

          {/* Existing tokens */}
          <div className="space-y-1">
            {loading && <div className="flex items-center gap-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…</div>}
            {!loading && tokens.length === 0 && <p className="text-xs text-zinc-600">No tokens yet.</p>}
            {tokens.map((t) => (
              <div key={t.id} className="flex items-center gap-2 rounded-md border border-zinc-800 bg-zinc-950/40 px-3 py-2">
                <KeyRound className="h-3.5 w-3.5 text-zinc-600 shrink-0" />
                <span className="text-sm text-zinc-300 truncate">{t.name || '(unnamed)'}</span>
                <span className="text-[11px] text-zinc-600 ml-auto shrink-0">used {fmt(t.last_used_at)}</span>
                <button onClick={() => revoke(t.id)} className="shrink-0 rounded p-1 text-zinc-500 hover:text-red-400 hover:bg-zinc-800" title="Revoke">
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
