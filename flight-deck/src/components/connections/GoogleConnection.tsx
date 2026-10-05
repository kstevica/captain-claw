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
  ChevronDown,
  Send,
  RefreshCw,
} from 'lucide-react'
import {
  gmailSendOptInLocked,
  useGoogleAuthStore,
  type GmailSendPatch,
  type GmailSendPolicy,
} from '../../stores/googleAuthStore'
import { useAuthStore } from '../../stores/authStore'

const RECENT_SENDS = 10

// The recipients box: one address or @domain per line, commas work too. The
// backend trims, lowercases and dedupes the same way, so re-typing a saved
// list in another case doesn't count as a change.
function parseRecipients(text: string): string[] {
  const seen = new Set<string>()
  for (const raw of text.split(/[\n,]+/)) {
    const entry = raw.trim().toLowerCase()
    if (entry) seen.add(entry)
  }
  return [...seen]
}

function fmtSentAt(ts: string): string {
  if (!ts) return ''
  const d = new Date(ts)
  return isNaN(d.getTime()) ? ts : d.toLocaleString()
}

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

  // One OAuth *client* per deck (admin-managed: POST /fd/google/config is
  // require_admin), one Google *account* per FD user (Connect binds the
  // signed-in user). With auth off the single local user is the admin.
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const fdUser = useAuthStore((s) => s.user)
  const canManageClient = !authEnabled || fdUser?.role === 'admin'
  const fdUserLabel = authEnabled && fdUser ? (fdUser.display_name || fdUser.email) : ''
  // A Google connection belongs to a Flight Deck user, so it needs FD sign-in:
  // a deck with auth off (the desktop app runs one) has no Google via FD.
  const unavailable = authEnabled === false

  const [clientId, setClientId] = useState('')
  const [clientSecret, setClientSecret] = useState('')
  const [projectId, setProjectId] = useState('')
  const [location, setLocation] = useState('')
  const [scopes, setScopes] = useState<string[]>([])
  const [saving, setSaving] = useState(false)
  const [revealSecret, setRevealSecret] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    if (unavailable) return
    refresh()
    const unsub = startMessageListener()
    return unsub
  }, [unavailable, refresh, startMessageListener])

  useEffect(() => {
    if (config) {
      setClientId(config.client_id || '')
      setProjectId(config.project_id || '')
      setLocation(config.location || 'us-central1')
      setScopes(config.scopes || config.default_scopes || [])
    }
  }, [config])

  const configured = !!status?.configured
  const connected = !!status?.connected
  const supportsVertex = !!status?.supports_vertex
  const user = status?.user

  const handleSave = async () => {
    const patch: Record<string, string | string[]> = {}
    if (clientId !== (config?.client_id || '')) patch.client_id = clientId
    if (clientSecret) patch.client_secret = clientSecret
    if (projectId !== (config?.project_id || '')) patch.project_id = projectId
    if (location !== (config?.location || '')) patch.location = location
    const prevScopes = new Set(config?.scopes || [])
    const nextScopes = new Set(scopes)
    const scopesChanged =
      prevScopes.size !== nextScopes.size ||
      [...nextScopes].some((s) => !prevScopes.has(s))
    if (scopesChanged) patch.scopes = scopes
    if (Object.keys(patch).length === 0) return
    // The backend wipes EVERY user's stored Google tokens when the client or
    // the scope set changes (the refresh tokens are bound to both).
    if (configured && (patch.client_id !== undefined || scopesChanged) && !confirm(
      'Change the Google OAuth client / scopes?\n\n' +
      'This disconnects the Google account of EVERY user on this deck, not ' +
      'just yours. Their agents lose Google access until each user connects ' +
      'again (and re-consents).',
    )) return
    setSaving(true)
    await saveConfig(patch)
    setClientSecret('')
    setSaving(false)
  }

  const toggleScope = (scope: string) => {
    setScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope],
    )
  }

  const handleClear = async () => {
    if (!confirm(
      'Remove this deck\'s Google OAuth credentials?\n\n' +
      'This clears the stored client_id and client_secret and disconnects ' +
      'the Google account of EVERY user on this deck, not just yours. Their ' +
      'agents lose Google access until an admin re-enters credentials and ' +
      'each user connects again.',
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

  const prevScopeSet = new Set(config?.scopes || [])
  const scopesDirty =
    prevScopeSet.size !== scopes.length ||
    scopes.some((s) => !prevScopeSet.has(s))

  const canSave =
    (clientId.trim() !== (config?.client_id || '').trim()) ||
    clientSecret.trim() !== '' ||
    (projectId.trim() !== (config?.project_id || '').trim()) ||
    (location.trim() !== (config?.location || '').trim()) ||
    scopesDirty

  if (unavailable) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
        <div className="flex items-center gap-3 px-5 py-4 border-b border-zinc-800">
          <div className="h-10 w-10 rounded-md bg-gradient-to-br from-blue-500/20 to-red-500/20 border border-zinc-800 flex items-center justify-center">
            <Mail className="h-5 w-5 text-zinc-200" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-sm font-semibold text-zinc-100">Google</h3>
              <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>
                Not available
              </span>
            </div>
            <div className="text-xs text-zinc-500 mt-0.5">Gmail, Drive, Calendar</div>
          </div>
        </div>
        <div className="px-5 py-4 text-xs text-zinc-400 leading-relaxed">
          Google isn't available on this deck: Flight Deck sign-in is turned off
          (the desktop app always runs it that way), and each Google connection
          belongs to a signed-in Flight Deck user. To connect Google, run Flight
          Deck with sign-in on (<code className="text-zinc-300">FD_AUTH_ENABLED=true</code>,
          the default) and connect from your account.
        </div>
      </div>
    )
  }

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
              title="This deck uses its own Google Cloud OAuth client — one for all users, managed by an admin. Each Flight Deck user connects their own Google account through it."
            >
              <KeyRound className="h-2.5 w-2.5" />
              own OAuth client
            </span>
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            Gmail, Drive, Calendar{supportsVertex ? ', and Vertex AI / Gemini' : ''}
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

        {lastPopupMessage && (
          <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2 text-xs text-zinc-400">
            {lastPopupMessage}
          </div>
        )}

        {/* Not-configured informational banner */}
        {!configured && !canManageClient && (
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2.5 text-xs text-zinc-300">
            <div className="font-medium text-amber-600 mb-0.5">Google isn't set up on this deck yet</div>
            <div className="text-zinc-400">
              Ask an admin to configure Google OAuth. Once they have, you can connect
              your own Google account here.
            </div>
          </div>
        )}
        {!configured && canManageClient && (
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
              The client is shared by every user on this deck; each user then connects
              their own Google account.
            </div>
          </div>
        )}

        {/* Connected-user summary */}
        {connected && user && (
          <div
            className="flex items-center gap-3 rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2.5"
            title={fdUserLabel ? `Google account linked to ${fdUserLabel}` : undefined}
          >
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
              title={
                configured
                  ? 'Link a Google account to your own Flight Deck user'
                  : canManageClient ? 'Save credentials first' : 'An admin has to configure Google first'
              }
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
          {configured && fdUserLabel && (
            <span className="text-xs text-zinc-500">
              {connected ? 'Linked to' : 'Links a Google account to'} your Flight Deck user,{' '}
              <span className="text-zinc-300">{fdUserLabel}</span>. Your agents use it — every
              other user connects their own.
            </span>
          )}
        </div>

        {/* Email sending — the signed-in user's own policy, so it shows for
            non-admins and in the kiosk Connections dialog too. */}
        {connected && <GmailSendSection />}

        {/* Credentials form — deck-wide OAuth client, admins only (the save
            endpoint is require_admin; non-admins would only get 403s). */}
        {canManageClient && (
        <div className="space-y-3 rounded-md border border-zinc-800 bg-zinc-950/50 p-3">
          <div className="text-xs text-zinc-500">
            Redirect URI (add this to your OAuth client in Google Cloud Console):
            <code className="block mt-1 text-[11px] text-zinc-200 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 break-all">
              {config?.redirect_uri || `${window.location.origin}/fd/google/callback`}
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

          {/* Scope picker */}
          {config?.scope_catalog && config.scope_catalog.length > 0 && (
            <div>
              <div className="text-xs font-medium text-zinc-400 mb-1.5">
                Requested scopes
              </div>
              <div className="rounded-md border border-zinc-800 bg-zinc-950/60 divide-y divide-zinc-800">
                {config.scope_catalog.map((entry) => {
                  const checked = scopes.includes(entry.scope)
                  const sensitivity = entry.sensitivity
                  const isIdentity = entry.group === 'identity'
                  const disabled =
                    entry.scope === 'openid' || entry.scope === 'email'
                  return (
                    <label
                      key={entry.scope}
                      className={
                        'flex items-start gap-2 px-3 py-2 cursor-pointer hover:bg-zinc-900/50 ' +
                        (disabled ? 'opacity-70 cursor-not-allowed' : '')
                      }
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={disabled}
                        onChange={() => !disabled && toggleScope(entry.scope)}
                        className="mt-0.5 accent-violet-600"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span className="text-sm text-zinc-200">{entry.label}</span>
                          {sensitivity === 'none' && !isIdentity && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-500 border border-emerald-500/30">
                              non-sensitive
                            </span>
                          )}
                          {sensitivity === 'sensitive' && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-600 border border-amber-500/30">
                              sensitive
                            </span>
                          )}
                          {sensitivity === 'restricted' && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-red-500/15 text-red-500 border border-red-500/30">
                              restricted
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-zinc-500 mt-0.5">
                          {entry.description}
                        </div>
                      </div>
                    </label>
                  )
                })}
              </div>
              <div className="mt-1.5 text-[11px] text-zinc-500 leading-snug">
                Google blocks unverified apps from requesting{' '}
                <span className="text-amber-600">sensitive</span> or{' '}
                <span className="text-red-500">restricted</span> scopes unless
                you add your Google account as a{' '}
                <span className="text-zinc-300">Test user</span> on your OAuth
                consent screen (Google Cloud Console → OAuth consent screen →
                Test users) — every user who will connect needs to be one.
                Changing scopes disconnects every user's Google account; each
                has to reconnect and re-consent.
              </div>
            </div>
          )}

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
        )}
      </div>
      )}
    </div>
  )
}

// Whether this user's agents may send Gmail on their own (POST
// /fd/google/gmail/send), and to whom. Off until the user opts in here — the
// agents then only draft. The deck's FD_GMAIL_SEND=off overrides it.
function GmailSendSection() {
  const policy = useGoogleAuthStore((s) => s.gmailSend)
  const sends = useGoogleAuthStore((s) => s.gmailSends)
  const sendError = useGoogleAuthStore((s) => s.gmailSendError)
  const fetchGmailSend = useGoogleAuthStore((s) => s.fetchGmailSend)
  const saveGmailSend = useGoogleAuthStore((s) => s.saveGmailSend)
  const fetchGmailSends = useGoogleAuthStore((s) => s.fetchGmailSends)

  const [enabled, setEnabled] = useState(false)
  const [recipientsText, setRecipientsText] = useState('')
  const [limitText, setLimitText] = useState('')
  const [saving, setSaving] = useState(false)
  const [reloading, setReloading] = useState(true)

  const syncDraft = (p: GmailSendPolicy) => {
    setEnabled(p.enabled)
    setRecipientsText(p.allowed_recipients.join('\n'))
    setLimitText(String(p.daily_limit))
  }

  const reload = async () => {
    setReloading(true)
    await Promise.all([fetchGmailSend(), fetchGmailSends(RECENT_SENDS)])
    setReloading(false)
  }

  useEffect(() => {
    reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Re-seed the form only when the saved settings themselves change — a
  // Refresh that just brings a new sent-count keeps what's being typed. Done
  // while rendering, not in an effect, so the form never paints a frame empty.
  const savedKey = policy
    ? JSON.stringify([policy.enabled, policy.allowed_recipients, policy.daily_limit])
    : ''
  const [seededKey, setSeededKey] = useState('')
  if (policy && savedKey !== seededKey) {
    setSeededKey(savedKey)
    syncDraft(policy)
  }

  const recipients = parseRecipients(recipientsText)
  const limit = Number(limitText)
  const limitValid = limitText.trim() !== '' && Number.isInteger(limit) && limit >= 1 && limit <= 500
  const enabledDirty = !!policy && enabled !== policy.enabled
  const recipientsDirty = !!policy && recipients.join('\n') !== policy.allowed_recipients.join('\n')
  const limitDirty = !!policy && (!limitValid || limit !== policy.daily_limit)
  const dirty = enabledDirty || recipientsDirty || limitDirty
  const canSave = !!policy && !saving && dirty && limitValid

  const handleSave = async () => {
    if (!policy || !canSave) return
    const patch: GmailSendPatch = {}
    if (enabledDirty) patch.enabled = enabled
    if (recipientsDirty) patch.allowed_recipients = recipients
    if (limitDirty) patch.daily_limit = limit
    setSaving(true)
    const ok = await saveGmailSend(patch)
    setSaving(false)
    if (!ok) return
    // The backend may write an entry differently (a bare domain as '@b.c'),
    // which leaves the saved key as it was — show what it stored regardless.
    const saved = useGoogleAuthStore.getState().gmailSend
    if (saved) syncDraft(saved)
    fetchGmailSends(RECENT_SENDS)
  }

  const deckDisabled = !!policy?.deck_disabled
  const optInLocked = gmailSendOptInLocked(policy)
  const pill = "inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border"
  const field = "w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500 focus:outline-none"

  return (
    <div className="space-y-3 rounded-md border border-zinc-800 bg-zinc-950/50 p-3">
      <div className="flex items-center gap-2 flex-wrap">
        <Send className="h-3.5 w-3.5 text-zinc-400" />
        <span className="text-xs font-medium text-zinc-300">Email sending</span>
        {policy && (
          deckDisabled ? (
            <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Off on this deck</span>
          ) : policy.enabled ? (
            <span className={`${pill} bg-emerald-500/15 text-emerald-500 border-emerald-500/30`}>
              <Check className="h-2.5 w-2.5" /> Agents can send
            </span>
          ) : (
            <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Drafts only</span>
          )
        )}
        <div className="flex-1" />
        <button
          type="button"
          onClick={reload}
          disabled={reloading}
          className="inline-flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300 disabled:opacity-50"
          title="Reload the sending settings and the sent list"
        >
          <RefreshCw className={'h-3 w-3 ' + (reloading ? 'animate-spin' : '')} />
          Refresh
        </button>
      </div>

      {sendError && (
        <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-500">
          <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
          <span className="break-words">{sendError}</span>
        </div>
      )}

      {!policy ? (
        !sendError && (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…
          </div>
        )
      ) : (
        <>
          <label
            className={
              'flex items-start gap-2 ' +
              (optInLocked ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer')
            }
          >
            <input
              type="checkbox"
              checked={enabled}
              disabled={saving || optInLocked}
              onChange={(e) => setEnabled(e.target.checked)}
              className="mt-0.5 accent-violet-600"
            />
            <div className="flex-1 min-w-0">
              <div className="text-sm text-zinc-200">Let my agents send email from this account</div>
              <div className="text-xs text-zinc-500 mt-0.5 leading-snug">
                When on, your agents can send email (including replies) without you
                pressing Send in Gmail. They're instructed to send only when you ask;
                every send is listed below and in your notifications. Off = drafts only.
              </div>
            </div>
          </label>

          {deckDisabled && (
            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-600">
              Sending is disabled on this Flight Deck by the administrator
              (<code>FD_GMAIL_SEND=off</code>).
              {policy.enabled && ' You can still turn yours off here.'}
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="sm:col-span-2">
              <label className="block text-xs font-medium text-zinc-400 mb-1">
                Only to these recipients <span className="text-zinc-600">(optional)</span>
              </label>
              <textarea
                value={recipientsText}
                onChange={(e) => setRecipientsText(e.target.value)}
                rows={3}
                disabled={saving}
                placeholder={'alice@example.com\n@yourcompany.com'}
                className={`${field} font-mono`}
              />
              <div className="mt-1 text-[11px] text-zinc-500 leading-snug">
                One address or <span className="text-zinc-300">@domain</span> per line
                (commas work too). Empty = anyone.
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1">Daily limit</label>
              <input
                type="number"
                min={1}
                max={500}
                step={1}
                value={limitText}
                onChange={(e) => setLimitText(e.target.value)}
                disabled={saving}
                className={field}
              />
              {!limitValid && (
                <div className="mt-1 text-[11px] text-red-500">A whole number from 1 to 500.</div>
              )}
              <div className="mt-1 text-[11px] text-zinc-500">
                Sent in the last 24h:{' '}
                <span className={policy.sent_last_24h >= policy.daily_limit ? 'text-amber-600' : 'text-zinc-300'}>
                  {policy.sent_last_24h} / {policy.daily_limit}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={handleSave}
              disabled={!canSave}
              className="inline-flex items-center gap-2 rounded-md bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-sm text-white disabled:opacity-50 shadow-sm"
            >
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
              Save
            </button>
            {dirty && !saving && (
              <span className="text-xs text-zinc-500">Unsaved changes</span>
            )}
          </div>

          {/* Recently sent */}
          <div>
            <div className="text-xs text-zinc-500 mb-1.5">Recently sent by your agents</div>
            {sends.length === 0 ? (
              <div className="text-xs text-zinc-500">
                {reloading ? 'Loading…' : 'Nothing sent by your agents yet.'}
              </div>
            ) : (
              <div className="rounded-md border border-zinc-800 bg-zinc-950/60 divide-y divide-zinc-800">
                {sends.slice(0, RECENT_SENDS).map((s) => (
                  <div
                    key={String(s.id)}
                    className="px-3 py-2 text-xs"
                    title={[
                      s.cc ? `Cc: ${s.cc}` : '',
                      s.bcc ? `Bcc: ${s.bcc}` : '',
                      s.draft_id ? `From draft ${s.draft_id}` : '',
                    ].filter(Boolean).join('\n') || undefined}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-zinc-500 shrink-0 tabular-nums">{fmtSentAt(s.created_at)}</span>
                      <span className="text-zinc-400 truncate">{s.agent}</span>
                      {s.status === 'unknown' && (
                        <span
                          className="ml-auto shrink-0 text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-600 border border-amber-500/30"
                          title="Gmail never confirmed this send — it may or may not have gone out."
                        >
                          unconfirmed — check Gmail Sent
                        </span>
                      )}
                    </div>
                    <div className="flex items-baseline gap-1.5 min-w-0 mt-0.5">
                      <span className="text-zinc-500 shrink-0">To</span>
                      <span className="text-zinc-300 truncate">{s.to}</span>
                    </div>
                    <div className="text-zinc-200 truncate mt-0.5">{s.subject || '(no subject)'}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
