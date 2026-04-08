import { useEffect, useState } from 'react'
import {
  Mail,
  Loader2,
  Check,
  Unplug,
  ExternalLink,
  Save,
  Eye,
  EyeOff,
  AlertCircle,
  KeyRound,
  Trash2,
} from 'lucide-react'
import { useGoogleAuthStore } from '../../stores/googleAuthStore'

export default function GoogleConnection() {
  const {
    status,
    config,
    loading,
    error,
    lastPopupMessage,
    refresh,
    saveConfig,
    clearCredentials,
    connect,
    disconnect,
    startMessageListener,
  } = useGoogleAuthStore()

  const [clientId, setClientId] = useState('')
  const [clientSecret, setClientSecret] = useState('')
  const [projectId, setProjectId] = useState('')
  const [location, setLocation] = useState('')
  const [saving, setSaving] = useState(false)
  const [revealSecret, setRevealSecret] = useState(false)

  useEffect(() => {
    refresh()
    const unsub = startMessageListener()
    return unsub
  }, [refresh, startMessageListener])

  useEffect(() => {
    if (config) {
      setClientId(config.client_id || '')
      setProjectId(config.project_id || '')
      setLocation(config.location || 'us-central1')
    }
  }, [config])

  const configured = !!status?.configured
  const connected = !!status?.connected
  const supportsVertex = !!status?.supports_vertex
  const user = status?.user

  const handleSave = async () => {
    setSaving(true)
    const patch: Record<string, string> = {}
    if (clientId !== (config?.client_id || '')) patch.client_id = clientId
    if (clientSecret) patch.client_secret = clientSecret
    if (projectId !== (config?.project_id || '')) patch.project_id = projectId
    if (location !== (config?.location || '')) patch.location = location
    if (Object.keys(patch).length > 0) {
      await saveConfig(patch)
      setClientSecret('')
    }
    setSaving(false)
  }

  const handleClear = async () => {
    if (!confirm(
      'Remove saved Google OAuth credentials?\n\n' +
      'This clears your stored client_id, client_secret, and disconnects ' +
      'the current session. You will need to re-enter credentials before ' +
      'you can connect again.',
    )) return
    setSaving(true)
    await clearCredentials()
    setClientId('')
    setClientSecret('')
    setProjectId('')
    setSaving(false)
  }

  // Accent-color pill helpers. These use /15 opacity over bordered backgrounds
  // so they blend with whatever theme is active — the zinc palette is
  // already inverted via CSS variables in index.css, but the amber/violet/
  // emerald/blue/red shades are not, so we avoid *-950/*-900 accents.
  const pill = "inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border"
  const pillTiny = "inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border"

  const canSave =
    (clientId.trim() !== (config?.client_id || '').trim()) ||
    clientSecret.trim() !== '' ||
    (projectId.trim() !== (config?.project_id || '').trim()) ||
    (location.trim() !== (config?.location || '').trim())

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-5 py-4 border-b border-zinc-800">
        <div className="h-10 w-10 rounded-md bg-gradient-to-br from-blue-500/20 to-red-500/20 border border-zinc-800 flex items-center justify-center">
          <Mail className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">Google</h3>
            {connected ? (
              <span className={`${pill} bg-emerald-500/15 text-emerald-500 border-emerald-500/30`}>
                <Check className="h-3 w-3" /> Connected
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
              title="Uses your own Google Cloud OAuth client."
            >
              <KeyRound className="h-2.5 w-2.5" />
              your credentials
            </span>
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            Gmail, Drive, Calendar{supportsVertex ? ', and Vertex AI / Gemini' : ''}
          </div>
        </div>
        {loading && <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />}
      </div>

      {/* Body */}
      <div className="px-5 py-4 space-y-4">
        {error && (
          <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-500">
            <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            <span className="break-all">{error}</span>
          </div>
        )}

        {lastPopupMessage && (
          <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2 text-xs text-zinc-400">
            {lastPopupMessage}
          </div>
        )}

        {/* Not-configured informational banner */}
        {!configured && (
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2.5 text-xs text-zinc-300">
            <div className="font-medium text-amber-600 mb-0.5">Bring your own OAuth client</div>
            <div className="text-zinc-400">
              Captain Claw does not ship with Google credentials. Create an OAuth 2.0
              Client ID in{' '}
              <a
                href="https://console.cloud.google.com/apis/credentials"
                target="_blank"
                rel="noopener noreferrer"
                className="text-violet-500 hover:underline"
              >
                Google Cloud Console
              </a>{' '}
              (type <span className="text-zinc-200 font-medium">Web application</span>), add the
              redirect URI shown below, then paste your Client ID and Client Secret here.
            </div>
          </div>
        )}

        {/* Connected-user summary */}
        {connected && user && (
          <div className="flex items-center gap-3 rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2.5">
            {user.picture && (
              <img src={user.picture} alt="" className="h-8 w-8 rounded-full" />
            )}
            <div className="flex-1 min-w-0">
              <div className="text-sm text-zinc-200 truncate">{user.name || user.email}</div>
              {user.email && user.name && (
                <div className="text-xs text-zinc-500 truncate">{user.email}</div>
              )}
            </div>
          </div>
        )}

        {/* Granted scopes */}
        {connected && status?.granted_scopes && status.granted_scopes.length > 0 && (
          <div>
            <div className="text-xs text-zinc-500 mb-1.5">Granted access</div>
            <div className="flex flex-wrap gap-1.5">
              {status.granted_scopes.map((s) => (
                <span
                  key={s.scope}
                  className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-300 border border-zinc-700"
                  title={s.scope}
                >
                  {s.label}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Primary actions */}
        <div className="flex items-center gap-2 flex-wrap">
          {!connected && (
            <button
              onClick={connect}
              disabled={!configured}
              className="inline-flex items-center gap-2 rounded-md bg-violet-600 hover:bg-violet-500 px-3.5 py-2 text-sm text-white shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
              title={configured ? 'Start the Google OAuth flow' : 'Save credentials first'}
            >
              <ExternalLink className="h-4 w-4" />
              Connect Google
            </button>
          )}
          {connected && (
            <button
              onClick={disconnect}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-md border border-zinc-700 hover:bg-zinc-800 px-3 py-1.5 text-sm text-zinc-300 disabled:opacity-50"
            >
              <Unplug className="h-3.5 w-3.5" />
              Disconnect
            </button>
          )}
        </div>

        {/* Credentials form — always visible */}
        <div className="space-y-3 rounded-md border border-zinc-800 bg-zinc-950/50 p-3">
          <div className="text-xs text-zinc-500">
            Redirect URI (add this to your OAuth client in Google Cloud Console):
            <code className="block mt-1 text-[11px] text-zinc-200 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 break-all">
              {config?.redirect_uri || 'http://localhost:25080/fd/google/callback'}
            </code>
          </div>

          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1">
              Client ID
              {config?.client_id_set && (
                <span className="ml-2 text-[10px] text-emerald-500">(saved)</span>
              )}
            </label>
            <input
              type="text"
              value={clientId}
              onChange={(e) => setClientId(e.target.value)}
              placeholder="xxxxxx.apps.googleusercontent.com"
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 font-mono focus:border-violet-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1">
              Client Secret
              {config?.client_secret_set && !clientSecret && (
                <span className="ml-2 text-[10px] text-emerald-500">(saved)</span>
              )}
            </label>
            <div className="relative">
              <input
                type={revealSecret ? 'text' : 'password'}
                value={clientSecret}
                onChange={(e) => setClientSecret(e.target.value)}
                placeholder={config?.client_secret_set ? '•••••••• (leave blank to keep)' : 'GOCSPX-...'}
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 pr-9 text-sm text-zinc-200 font-mono focus:border-violet-500 focus:outline-none"
              />
              <button
                type="button"
                onClick={() => setRevealSecret((v) => !v)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
                tabIndex={-1}
              >
                {revealSecret ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1">
                Project ID <span className="text-zinc-600">(for Vertex AI)</span>
              </label>
              <input
                type="text"
                value={projectId}
                onChange={(e) => setProjectId(e.target.value)}
                placeholder="my-gcp-project"
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 font-mono focus:border-violet-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1">Location</label>
              <input
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="us-central1"
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 font-mono focus:border-violet-500 focus:outline-none"
              />
            </div>
          </div>

          <div className="flex items-center gap-2 pt-1 flex-wrap">
            <button
              onClick={handleSave}
              disabled={saving || !canSave}
              className="inline-flex items-center gap-2 rounded-md bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-sm text-white disabled:opacity-50 shadow-sm"
            >
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
              Save credentials
            </button>
            {configured && (
              <button
                onClick={handleClear}
                disabled={saving}
                className="inline-flex items-center gap-2 rounded-md border border-red-500/30 hover:bg-red-500/10 px-3 py-1.5 text-xs text-red-500 disabled:opacity-50"
              >
                <Trash2 className="h-3 w-3" />
                Remove credentials
              </button>
            )}
            {configured && !connected && (
              <span className="text-xs text-zinc-500">
                Then click <span className="text-zinc-300 font-medium">Connect Google</span> above.
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
