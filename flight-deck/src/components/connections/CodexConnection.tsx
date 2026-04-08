import { useEffect, useState } from 'react'
import {
  Sparkles,
  Loader2,
  Check,
  AlertCircle,
  RefreshCw,
  KeyRound,
  ChevronDown,
} from 'lucide-react'
import { useCodexAuthStore } from '../../stores/codexAuthStore'

function formatExpiry(seconds?: number): string {
  if (!seconds || seconds <= 0) return 'expired'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

export default function CodexConnection() {
  const { status, loading, error, lastMessage, refresh, reimport } = useCodexAuthStore()
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    refresh()
  }, [refresh])

  const configured = !!status?.configured
  const connected = !!status?.connected
  const stale = !!status?.stale

  const pill = "inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border"
  const pillTiny = "inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border"

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
        <div className="h-10 w-10 rounded-md bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border border-zinc-800 flex items-center justify-center">
          <Sparkles className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">ChatGPT (Codex)</h3>
            {connected && !stale ? (
              <span className={`${pill} bg-emerald-500/15 text-emerald-500 border-emerald-500/30`}>
                <Check className="h-3 w-3" /> Connected
              </span>
            ) : connected && stale ? (
              <span className={`${pill} bg-amber-500/15 text-amber-600 border-amber-500/30`}>
                Token stale
              </span>
            ) : configured ? (
              <span className={`${pill} bg-amber-500/15 text-amber-600 border-amber-500/30`}>
                Not connected
              </span>
            ) : (
              <span className={`${pill} bg-red-500/15 text-red-500 border-red-500/30`}>
                <AlertCircle className="h-3 w-3" /> Not configured
              </span>
            )}
            <span
              className={`${pillTiny} bg-blue-500/15 text-blue-500 border-blue-500/30`}
              title="Reuses tokens from the Codex CLI on the Flight Deck host."
            >
              <KeyRound className="h-2.5 w-2.5" />
              codex CLI
            </span>
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            Sign in with ChatGPT — uses your ChatGPT plan via the Codex Responses API
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
      <div className="px-5 py-4 space-y-4">
        {error && (
          <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-500">
            <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            <span className="break-all">{error}</span>
          </div>
        )}

        {lastMessage && (
          <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2 text-xs text-zinc-400">
            {lastMessage}
          </div>
        )}

        {/* Not-connected info banner */}
        {!connected && (
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2.5 text-xs text-zinc-300">
            <div className="font-medium text-amber-600 mb-0.5">
              {configured ? 'Tokens unreadable' : 'Run codex login on the Flight Deck host'}
            </div>
            <div className="text-zinc-400">
              {status?.detail || (
                <>
                  Captain Claw reuses the OAuth tokens cached by the Codex CLI in{' '}
                  <code className="text-zinc-200">~/.codex/auth.json</code>. Install the
                  Codex CLI and run <code className="text-zinc-200">codex login</code> on
                  the same machine that runs Flight Deck, then click{' '}
                  <span className="text-zinc-300 font-medium">Reimport</span>.
                </>
              )}
            </div>
            {status?.auth_path && (
              <div className="mt-1.5 text-[11px] text-zinc-500 break-all">
                Looking at: <code className="text-zinc-400">{status.auth_path}</code>
              </div>
            )}
          </div>
        )}

        {/* Connected summary */}
        {connected && (
          <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2.5 space-y-1.5">
            {status?.email && (
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-zinc-500">Email</span>
                <span className="text-zinc-200 truncate">{status.email}</span>
              </div>
            )}
            {status?.plan && (
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-zinc-500">Plan</span>
                <span className="text-zinc-200 capitalize">{status.plan}</span>
              </div>
            )}
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="text-zinc-500">Token expires in</span>
              <span className={stale ? 'text-amber-500' : 'text-zinc-200'}>
                {formatExpiry(status?.seconds_until_expiry)}
              </span>
            </div>
            {status?.access_token_preview && (
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="text-zinc-500">Token</span>
                <code className="text-zinc-400 font-mono">{status.access_token_preview}</code>
              </div>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={reimport}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-md bg-violet-600 hover:bg-violet-500 px-3.5 py-2 text-sm text-white shadow-sm disabled:opacity-50"
            title="Re-read ~/.codex/auth.json from the Flight Deck host"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
            Reimport from Codex CLI
          </button>
          <span className="text-xs text-zinc-500">
            The Codex CLI refreshes tokens automatically while it runs.
          </span>
        </div>
      </div>
      )}
    </div>
  )
}
