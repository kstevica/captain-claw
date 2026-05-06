import { useEffect, useState } from 'react'
import {
  Plug,
  Loader2,
  Check,
  AlertCircle,
  RefreshCw,
  Trash2,
  ChevronDown,
  Plus,
  X,
  Link as LinkIcon,
} from 'lucide-react'
import {
  useMCPStore,
  type MCPServer,
  type MCPServerInput,
  type MCPProbeResult,
} from '../../stores/mcpStore'

const SECRET_PLACEHOLDER = '••••••••'

function emptyForm(): MCPServerInput {
  return {
    name: '',
    url: '',
    client_id: '',
    client_secret: '',
    token_endpoint: '',
    headers: {},
    enabled: true,
  }
}

interface ServerRowProps {
  server: MCPServer
}

function ServerRow({ server }: ServerRowProps) {
  const { testing, lastTestResult, testServer, removeServer } = useMCPStore()
  const [confirming, setConfirming] = useState(false)
  const isTesting = !!testing[server.name]
  const result = lastTestResult[server.name]

  const statusPill = server.initialized ? (
    <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border bg-emerald-500/15 text-emerald-500 border-emerald-500/30">
      <Check className="h-3 w-3" /> Connected
    </span>
  ) : server.last_error ? (
    <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border bg-red-500/15 text-red-500 border-red-500/30">
      <AlertCircle className="h-3 w-3" /> Error
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border bg-zinc-500/15 text-zinc-400 border-zinc-500/30">
      Not initialized
    </span>
  )

  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-zinc-100">{server.name}</span>
            {statusPill}
            {server.client_id && (
              <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border bg-blue-500/10 text-blue-400 border-blue-500/30">
                OAuth2
              </span>
            )}
            {!server.enabled && (
              <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border bg-amber-500/10 text-amber-400 border-amber-500/30">
                Disabled
              </span>
            )}
          </div>
          <div className="text-xs text-zinc-500 mt-1 break-all">
            <LinkIcon className="inline h-3 w-3 mr-1 -mt-0.5" />
            {server.url}
          </div>
          {typeof server.tools_count === 'number' && server.tools_count > 0 && (
            <div className="text-xs text-zinc-400 mt-1">
              {server.tools_count} tool{server.tools_count === 1 ? '' : 's'} discovered
            </div>
          )}
          {server.last_error && (
            <div className="text-xs text-red-400 mt-1 break-all">
              {server.last_error}
            </div>
          )}
          {result && (
            <div
              className={
                'text-xs mt-1 break-all ' +
                (result.ok ? 'text-emerald-400' : 'text-red-400')
              }
            >
              {result.ok
                ? `Probe ok — ${result.tools_count ?? 0} tool${
                    (result.tools_count ?? 0) === 1 ? '' : 's'
                  }`
                : `Probe failed: ${result.error}`}
            </div>
          )}
        </div>
        <div className="flex items-center gap-1.5 flex-wrap shrink-0">
          <button
            type="button"
            onClick={() => testServer(server.name)}
            disabled={isTesting}
            className="inline-flex items-center gap-1.5 rounded-md border border-zinc-700 hover:bg-zinc-800 px-2.5 py-1.5 text-xs text-zinc-200 disabled:opacity-50"
            title="Run an end-to-end probe (initialize + list tools)"
          >
            {isTesting ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            Test
          </button>
          {confirming ? (
            <>
              <button
                type="button"
                onClick={async () => {
                  await removeServer(server.name)
                  setConfirming(false)
                }}
                className="inline-flex items-center gap-1.5 rounded-md bg-red-600 hover:bg-red-500 px-2.5 py-1.5 text-xs text-white"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Confirm
              </button>
              <button
                type="button"
                onClick={() => setConfirming(false)}
                className="inline-flex items-center gap-1.5 rounded-md border border-zinc-700 hover:bg-zinc-800 px-2.5 py-1.5 text-xs text-zinc-300"
              >
                <X className="h-3.5 w-3.5" />
                Cancel
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={() => setConfirming(true)}
              className="inline-flex items-center gap-1.5 rounded-md border border-red-500/30 hover:bg-red-500/10 px-2.5 py-1.5 text-xs text-red-400"
              title="Remove this server"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Remove
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

interface AddServerFormProps {
  onClose: () => void
}

function AddServerForm({ onClose }: AddServerFormProps) {
  const { saving, saveServer, probeTransient } = useMCPStore()
  const [form, setForm] = useState<MCPServerInput>(emptyForm())
  const [probing, setProbing] = useState(false)
  const [probe, setProbe] = useState<MCPProbeResult | null>(null)

  const valid = !!form.name.trim() && !!form.url.trim()

  function update<K extends keyof MCPServerInput>(key: K, value: MCPServerInput[K]) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  async function onProbe() {
    if (!valid) return
    setProbing(true)
    setProbe(null)
    try {
      const result = await probeTransient({
        ...form,
        client_secret: form.client_secret || '',
      })
      setProbe(result)
    } finally {
      setProbing(false)
    }
  }

  async function onSave() {
    if (!valid) return
    const payload: MCPServerInput = {
      ...form,
      // If user pasted the placeholder back in, send empty so the
      // backend keeps the previously stored secret (no-op for a new
      // server because there's nothing to preserve yet).
      client_secret:
        form.client_secret === SECRET_PLACEHOLDER ? '' : form.client_secret,
    }
    const saved = await saveServer(payload)
    if (saved) {
      onClose()
    }
  }

  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-950/70 px-4 py-3 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-zinc-100">Add MCP server</span>
        <button
          type="button"
          onClick={onClose}
          className="text-zinc-500 hover:text-zinc-300"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <label className="text-xs text-zinc-400 space-y-1">
          <span>Name *</span>
          <input
            value={form.name}
            onChange={(e) => update('name', e.target.value)}
            placeholder="fricmcp"
            className="w-full rounded-md bg-zinc-900 border border-zinc-800 px-2.5 py-1.5 text-sm text-zinc-100 focus:outline-none focus:border-violet-500"
          />
        </label>
        <label className="text-xs text-zinc-400 space-y-1">
          <span>Server URL *</span>
          <input
            value={form.url}
            onChange={(e) => update('url', e.target.value)}
            placeholder="https://example.com/mcp"
            className="w-full rounded-md bg-zinc-900 border border-zinc-800 px-2.5 py-1.5 text-sm text-zinc-100 focus:outline-none focus:border-violet-500"
          />
        </label>
        <label className="text-xs text-zinc-400 space-y-1">
          <span>Client ID</span>
          <input
            value={form.client_id || ''}
            onChange={(e) => update('client_id', e.target.value)}
            placeholder="optional, for OAuth2 client_credentials"
            className="w-full rounded-md bg-zinc-900 border border-zinc-800 px-2.5 py-1.5 text-sm text-zinc-100 focus:outline-none focus:border-violet-500"
          />
        </label>
        <label className="text-xs text-zinc-400 space-y-1">
          <span>Client Secret</span>
          <input
            type="password"
            value={form.client_secret || ''}
            onChange={(e) => update('client_secret', e.target.value)}
            placeholder="optional"
            className="w-full rounded-md bg-zinc-900 border border-zinc-800 px-2.5 py-1.5 text-sm text-zinc-100 focus:outline-none focus:border-violet-500"
          />
        </label>
        <label className="text-xs text-zinc-400 space-y-1 sm:col-span-2">
          <span>Token endpoint</span>
          <input
            value={form.token_endpoint || ''}
            onChange={(e) => update('token_endpoint', e.target.value)}
            placeholder="absolute URL or path, e.g. /api/mcp/oauth/token"
            className="w-full rounded-md bg-zinc-900 border border-zinc-800 px-2.5 py-1.5 text-sm text-zinc-100 focus:outline-none focus:border-violet-500"
          />
        </label>
      </div>

      <label className="flex items-center gap-2 text-xs text-zinc-400">
        <input
          type="checkbox"
          checked={form.enabled !== false}
          onChange={(e) => update('enabled', e.target.checked)}
        />
        <span>Enabled (agents will load tools from this server)</span>
      </label>

      {probe && (
        <div
          className={
            'rounded-md border px-3 py-2 text-xs break-all ' +
            (probe.ok
              ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
              : 'border-red-500/30 bg-red-500/10 text-red-300')
          }
        >
          {probe.ok
            ? `Probe ok — discovered ${probe.tools_count ?? 0} tool${
                (probe.tools_count ?? 0) === 1 ? '' : 's'
              }${
                probe.tool_names && probe.tool_names.length
                  ? `: ${probe.tool_names.slice(0, 8).join(', ')}${
                      probe.tool_names.length > 8 ? '…' : ''
                    }`
                  : ''
              }`
            : `Probe failed: ${probe.error}`}
        </div>
      )}

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onProbe}
          disabled={!valid || probing}
          className="inline-flex items-center gap-1.5 rounded-md border border-zinc-700 hover:bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 disabled:opacity-50"
        >
          {probing ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <RefreshCw className="h-3.5 w-3.5" />
          )}
          Test connection
        </button>
        <button
          type="button"
          onClick={onSave}
          disabled={!valid || saving}
          className="inline-flex items-center gap-1.5 rounded-md bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-sm text-white disabled:opacity-50"
        >
          {saving ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Plus className="h-3.5 w-3.5" />
          )}
          Save server
        </button>
      </div>
    </div>
  )
}

export default function MCPConnection() {
  const { servers, loading, error, refresh } = useMCPStore()
  const [collapsed, setCollapsed] = useState(false)
  const [adding, setAdding] = useState(false)

  useEffect(() => {
    refresh()
  }, [refresh])

  const totalTools = servers.reduce(
    (sum, s) => sum + (s.tools_count || 0),
    0,
  )
  const connectedCount = servers.filter((s) => s.initialized).length

  const pill = 'inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border'

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => setCollapsed((v) => !v)}
        className={
          'w-full flex items-center gap-3 px-5 py-4 text-left hover:bg-zinc-900/80 transition-colors ' +
          (collapsed ? '' : 'border-b border-zinc-800')
        }
        aria-expanded={!collapsed}
      >
        <div className="h-10 w-10 rounded-md bg-gradient-to-br from-fuchsia-500/20 to-violet-500/20 border border-zinc-800 flex items-center justify-center">
          <Plug className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">MCP servers</h3>
            {servers.length > 0 ? (
              <span className={`${pill} bg-emerald-500/15 text-emerald-500 border-emerald-500/30`}>
                <Check className="h-3 w-3" />
                {connectedCount}/{servers.length} connected
              </span>
            ) : (
              <span className={`${pill} bg-zinc-500/15 text-zinc-400 border-zinc-500/30`}>
                None configured
              </span>
            )}
            {totalTools > 0 && (
              <span className={`${pill} bg-blue-500/15 text-blue-500 border-blue-500/30`}>
                {totalTools} tool{totalTools === 1 ? '' : 's'}
              </span>
            )}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            Model Context Protocol — Flight Deck proxies tool calls to remote MCP servers for every agent in the fleet.
          </div>
        </div>
        {loading && <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />}
        <ChevronDown
          className={
            'h-4 w-4 text-zinc-500 transition-transform ' +
            (collapsed ? '-rotate-90' : '')
          }
        />
      </button>

      {/* Body */}
      {!collapsed && (
        <div className="px-5 py-4 space-y-3">
          {error && (
            <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-500">
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              <span className="break-all">{error}</span>
            </div>
          )}

          {servers.length === 0 && !adding && (
            <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-3 text-xs text-zinc-400">
              No MCP servers configured yet. Add one to expose its tools to every agent in the fleet.
            </div>
          )}

          <div className="space-y-2">
            {servers.map((srv) => (
              <ServerRow key={srv.name} server={srv} />
            ))}
          </div>

          {adding ? (
            <AddServerForm onClose={() => setAdding(false)} />
          ) : (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setAdding(true)}
                className="inline-flex items-center gap-1.5 rounded-md bg-violet-600 hover:bg-violet-500 px-3.5 py-2 text-sm text-white"
              >
                <Plus className="h-4 w-4" />
                Add MCP server
              </button>
              <button
                type="button"
                onClick={refresh}
                disabled={loading}
                className="inline-flex items-center gap-1.5 rounded-md border border-zinc-700 hover:bg-zinc-800 px-3 py-2 text-sm text-zinc-200 disabled:opacity-50"
              >
                {loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4" />
                )}
                Refresh
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
