import { useEffect, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  X, Globe, Loader2, AlertCircle, AlertTriangle, Copy, Check, ExternalLink, RefreshCw,
  Trash2, Lock, MessageSquare, Files, ShieldCheck, Eye, ChevronDown, ChevronRight,
} from 'lucide-react'
import type { AgentEndpoint, AgentFile } from '../../services/fileTransfer'
import { formatSize, getFileTypeGroup } from '../../services/fileTransfer'
import type { PublicLink, PublicLinkConfig } from '../../services/publicLinks'
import { getFileLink, saveFileLink, listMyLinks, revokeLink, linkUrl } from '../../services/publicLinks'

// ── Public link for one agent file ───────────────────────────────────
//
// Opens on the file's Share button. With no link yet it explains what
// publishing means before anything happens; "Create public link" copies the
// file out of the agent and shows the address. With a link it shows the
// address, its views and expiry, and lets the owner refresh the copy or stop
// sharing. The footer lists every link the user has, so one whose file has
// since disappeared from the agent can still be turned off.

const EXPIRY_LABELS: Record<number, string> = {
  0: 'Never', 1: 'After 1 day', 7: 'After 7 days', 30: 'After 30 days',
}

// What a visitor gets for each file type (mirrors public_file_shares).
const SHOWN_AS: Record<string, string> = {
  markdown: 'a formatted page — the Markdown is rendered, any HTML inside it shows as plain text',
  html: 'the page itself, running in a locked sandbox',
  image: 'the image',
  video: 'a video player',
  audio: 'an audio player',
  pdf: 'the PDF',
  text: 'the text',
  code: 'the code as text',
  data: 'the data as text',
}

// Names that usually mean credentials or keys.
const SENSITIVE_NAME = /(^|[._-])(env|secrets?|credentials?|passwords?|tokens?)([._-]|$)|\.(pem|key|p12|pfx|kdbx)$|^id_(rsa|ed25519|ecdsa)/i

// A deck reached at one of these can't be opened from the internet.
function localOnlyHost(base: string): string | null {
  let host = ''
  try { host = new URL(base).hostname } catch { return null }
  const local = host === 'localhost' || host.endsWith('.localhost') || host === '[::1]' || host === '::1'
    || /^127\./.test(host) || /^10\./.test(host) || /^192\.168\./.test(host)
    || /^172\.(1[6-9]|2\d|3[01])\./.test(host) || host.endsWith('.local')
  return local ? host : null
}

function fmtDate(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' })
    + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function PublicLinkDialog({ agent, file, onClose }: {
  agent: AgentEndpoint
  file: AgentFile
  onClose: () => void
}) {
  const [config, setConfig] = useState<PublicLinkConfig | null>(null)
  const [link, setLink] = useState<PublicLink | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState<'save' | 'revoke' | null>(null)
  const [error, setError] = useState('')
  // -1: keep the link's current expiry (only offered once a link exists).
  // A new link starts at the deck's default: every link expires (cohort001).
  const [expiry, setExpiry] = useState(7)
  const [justCreated, setJustCreated] = useState(false)
  const [confirmStop, setConfirmStop] = useState(false)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    let live = true
    getFileLink(agent, file.physical)
      .then(({ share, ...cfg }) => {
        if (!live) return
        setConfig(cfg); setLink(share)
        setExpiry(share ? -1 : cfg.default_days ?? 7)
      })
      .catch((e) => { if (live) setError(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (live) setLoading(false) })
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent.id, agent.host, agent.port, file.physical])

  // Esc closes this dialog only: marked, so a dialog it opened from (the
  // kiosk's Files & data) stays open.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose() } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const group = getFileTypeGroup(file)
  const url = link ? linkUrl(link, config) : ''
  const localHost = localOnlyHost(config?.public_base || window.location.origin)
  const tooBig = !!config && file.size > config.max_mb * 1024 * 1024

  const save = async () => {
    setBusy('save'); setError('')
    try {
      const { share, ...cfg } = await saveFileLink(agent, file.physical, expiry < 0 ? null : expiry)
      setConfig(cfg)
      setJustCreated(!link)
      setLink(share)
      setExpiry(-1)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const stop = async () => {
    if (!link) return
    setBusy('revoke'); setError('')
    try {
      await revokeLink(link.token)
      setLink(null); setConfirmStop(false); setJustCreated(false); setExpiry(config?.default_days ?? 7)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const copy = () => {
    navigator.clipboard?.writeText(url)
      .then(() => { setCopied(true); setTimeout(() => setCopied(false), 2000) })
      .catch(() => {})
  }

  return createPortal(
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center bg-black/60 p-3"
      onClick={(e) => { e.stopPropagation(); onClose() }}
    >
      <div
        role="dialog"
        aria-label={`Public link — ${file.filename}`}
        className="flex max-h-[92vh] w-full max-w-lg flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start gap-3 border-b border-zinc-800 px-5 py-3.5">
          <Globe className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold text-zinc-100">Public link</h2>
            <div className="mt-0.5 truncate text-xs text-zinc-500" title={file.filename}>
              {file.filename} · {formatSize(file.size)}
            </div>
          </div>
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
          {loading && (
            <div className="flex justify-center py-8"><Loader2 className="h-5 w-5 animate-spin text-zinc-500" /></div>
          )}

          {!loading && config && !config.enabled && !link && (
            <Banner tone="warn" icon={Lock}>Public links are turned off on this deck.</Banner>
          )}

          {/* ── The link exists ───────────────────────────── */}
          {!loading && link && (
            <>
              <Banner tone="ok" icon={Check}>
                {justCreated ? 'Link created. ' : ''}Anyone with this link can view this file — no sign-in needed.
              </Banner>

              <div>
                <label className="mb-1.5 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Sharing link</label>
                <div className="flex items-center gap-2">
                  <input
                    readOnly
                    value={url}
                    onFocus={(e) => e.currentTarget.select()}
                    className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-2 font-mono text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none"
                  />
                  <button
                    onClick={copy}
                    className="flex shrink-0 items-center gap-1 rounded-md bg-violet-600 px-3 py-2 text-xs font-semibold text-white hover:bg-violet-500"
                  >
                    {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="shrink-0 rounded-md bg-zinc-800 p-2 text-zinc-300 hover:bg-zinc-700"
                    title="Open the public page"
                  >
                    <ExternalLink className="h-3.5 w-3.5" />
                  </a>
                </div>
              </div>

              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
                <dt className="text-zinc-500">Copy taken</dt>
                <dd className="text-zinc-300">{fmtDate(link.updated_at)}</dd>
                <dt className="text-zinc-500">Expires</dt>
                <dd className="text-zinc-300">{link.expires_at ? fmtDate(link.expires_at) : 'Never'}</dd>
                <dt className="text-zinc-500">Views</dt>
                <dd className="text-zinc-300">
                  {link.view_count}{link.last_viewed_at ? ` · last ${fmtDate(link.last_viewed_at)}` : ''}
                </dd>
              </dl>

              <p className="text-xs leading-relaxed text-zinc-500">
                This is a copy of the file from when it was taken — the agent's later changes don't appear.
                Update the copy to publish the current version; the link stays the same.
              </p>
              {localHost && <LocalHostNote host={localHost} />}
            </>
          )}

          {/* ── No link yet: explain first ───────────────── */}
          {!loading && !link && config?.enabled && (
            <>
              <p className="text-xs leading-relaxed text-zinc-400">
                Visitors will see {SHOWN_AS[group] || 'a download button — this type can’t be previewed in a browser'}.
              </p>
              <div>
                <div className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Before you share</div>
                <ul className="space-y-2.5">
                  <Point icon={Globe}>
                    <b className="font-medium text-zinc-200">Anyone with the link can open it</b>, without signing in.
                    Links get forwarded — you can’t control who ends up seeing it.
                  </Point>
                  <Point icon={Files}>
                    It publishes a <b className="font-medium text-zinc-200">copy of the file as it is now</b>. Later changes
                    by the agent won’t show until you update the copy, and deleting the file in the agent doesn’t
                    remove the link — stop sharing here.
                  </Point>
                  <Point icon={Eye}>
                    <b className="font-medium text-zinc-200">Check it for private details first</b> — names, emails, phone
                    numbers, API keys, internal links. Agent-written files can hold more than you expect.
                  </Point>
                  <Point icon={MessageSquare}>
                    Chat apps (WhatsApp, Slack, iMessage…) open links to build a preview, so the file name can show
                    up in the chat and count as a view.
                  </Point>
                  {group === 'html' && (
                    <Point icon={ShieldCheck}>
                      Scripts in the page run in a locked sandbox, cut off from Flight Deck and your account — but they
                      can still load things from the internet.
                    </Point>
                  )}
                  {(group === 'html' || group === 'markdown') && (
                    <Point icon={AlertTriangle}>
                      Only this one file is published: images or other files it points to by a relative path won’t load.
                    </Point>
                  )}
                  <Point icon={Lock}>
                    Nothing else is shared — not who you are, which agent made it, its folder, or anything else in
                    Flight Deck. Search engines are asked not to index it.
                  </Point>
                </ul>
              </div>

              {SENSITIVE_NAME.test(file.filename) && (
                <Banner tone="danger" icon={AlertTriangle}>
                  “{file.filename}” looks like a file that holds keys or passwords. Don’t publish it unless you’re sure.
                </Banner>
              )}
              {localHost && <LocalHostNote host={localHost} />}
              {tooBig && (
                <Banner tone="danger" icon={AlertCircle}>
                  This file is larger than the {config.max_mb} MB limit for public links.
                </Banner>
              )}
            </>
          )}

          {error && <Banner tone="danger" icon={AlertCircle}>{error}</Banner>}

          {!loading && <MyLinks current={link?.token} config={config} onStopped={(t) => { if (t === link?.token) { setLink(null); setExpiry(config?.default_days ?? 7) } }} />}
        </div>

        {/* Footer */}
        {!loading && (link || config?.enabled) && (
          <div className="flex flex-wrap items-center gap-2 border-t border-zinc-800 px-5 py-3">
            {!confirmStop && <label className="flex items-center gap-1.5 text-xs text-zinc-500">
              Expires
              <select
                value={expiry}
                onChange={(e) => setExpiry(Number(e.target.value))}
                className="rounded-md border border-zinc-700 bg-zinc-900 px-1.5 py-1 text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none"
              >
                {link && <option value={-1}>Keep current</option>}
                {(config?.expiry_days || [1, 7, 30]).map((d) => (
                  <option key={d} value={d}>{EXPIRY_LABELS[d] || `After ${d} days`}</option>
                ))}
              </select>
            </label>}
            <div className="flex-1" />
            {link && !confirmStop && (
              <button
                onClick={() => setConfirmStop(true)}
                className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-500/10 dark:text-red-400"
              >
                <Trash2 className="h-3.5 w-3.5" /> Stop sharing
              </button>
            )}
            {link && confirmStop && (
              <>
                <span className="text-xs text-zinc-400">The link stops working for everyone.</span>
                <button onClick={() => setConfirmStop(false)} className="rounded-md px-2.5 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">
                  Keep
                </button>
                <button
                  onClick={stop}
                  disabled={busy !== null}
                  className="flex items-center gap-1.5 rounded-md bg-red-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-red-500 disabled:opacity-50"
                >
                  {busy === 'revoke' && <Loader2 className="h-3.5 w-3.5 animate-spin" />} Stop sharing
                </button>
              </>
            )}
            {!link && (
              <button onClick={onClose} className="rounded-md px-3 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800">
                Cancel
              </button>
            )}
            {!confirmStop && (link ? config?.enabled : true) && (
              <button
                onClick={save}
                disabled={busy !== null || tooBig}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold disabled:opacity-50 ${
                  link ? 'bg-zinc-800 text-zinc-200 hover:bg-zinc-700' : 'bg-violet-600 text-white hover:bg-violet-500'
                }`}
              >
                {busy === 'save'
                  ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  : link ? <RefreshCw className="h-3.5 w-3.5" /> : <Globe className="h-3.5 w-3.5" />}
                {link ? 'Update copy' : 'Create public link'}
              </button>
            )}
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}

// Every link the user has made — so one whose file is gone can still be stopped.
function MyLinks({ current, config, onStopped }: {
  current?: string
  config: PublicLinkConfig | null
  onStopped: (token: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [links, setLinks] = useState<PublicLink[] | null>(null)
  const [stopping, setStopping] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!open) return
    let live = true
    listMyLinks()
      .then((r) => { if (live) setLinks(r.shares) })
      .catch((e) => { if (live) setError(e instanceof Error ? e.message : String(e)) })
    return () => { live = false }
  }, [open, current])

  const stop = async (token: string) => {
    setStopping(token)
    try {
      await revokeLink(token)
      setLinks((l) => (l || []).filter((x) => x.token !== token))
      onStopped(token)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setStopping(null)
    }
  }

  return (
    <div className="border-t border-zinc-800/60 pt-3">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        All your public links{links ? ` (${links.length})` : ''}
      </button>
      {open && (
        <div className="mt-2">
          {!links && !error && <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />}
          {error && <p className="text-xs text-red-600 dark:text-red-400">{error}</p>}
          {links && links.length === 0 && <p className="text-xs text-zinc-500">You haven’t shared any files.</p>}
          {links && links.length > 0 && (
            <ul className="divide-y divide-zinc-800/60">
              {links.map((l) => (
                <li key={l.token} className="flex items-center gap-2 py-1.5">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs text-zinc-200" title={l.filename}>
                      {l.filename}{l.token === current && <span className="ml-1.5 text-violet-400">· this file</span>}
                    </div>
                    <div className="truncate text-[11px] text-zinc-500">
                      {l.agent_name || 'Agent'} · {l.view_count} view{l.view_count === 1 ? '' : 's'}
                      {l.expires_at ? ` · expires ${fmtDate(l.expires_at)}` : ''}
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      navigator.clipboard?.writeText(linkUrl(l, config))
                        .then(() => { setCopied(l.token); setTimeout(() => setCopied(null), 2000) })
                        .catch(() => {})
                    }}
                    className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
                    title="Copy link"
                  >
                    {copied === l.token ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  </button>
                  <button
                    onClick={() => stop(l.token)}
                    disabled={stopping !== null}
                    className="rounded p-1 text-zinc-500 hover:bg-red-500/10 hover:text-red-600 disabled:opacity-50 dark:hover:text-red-400"
                    title="Stop sharing"
                  >
                    {stopping === l.token ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

function LocalHostNote({ host }: { host: string }) {
  return (
    <Banner tone="warn" icon={AlertTriangle}>
      This deck is reached at <span className="font-mono">{host}</span>, so the link only works for people who can
      reach that address — not from the internet.
    </Banner>
  )
}

function Point({ icon: Icon, children }: { icon: typeof Globe; children: ReactNode }) {
  return (
    <li className="flex items-start gap-2.5 text-xs leading-relaxed text-zinc-400">
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />
      <span>{children}</span>
    </li>
  )
}

function Banner({ tone, icon: Icon, children }: {
  tone: 'ok' | 'warn' | 'danger'
  icon: typeof Globe
  children: ReactNode
}) {
  const cls = {
    ok: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
    warn: 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300',
    danger: 'border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300',
  }[tone]
  return (
    <div className={`flex items-start gap-2 rounded-md border px-3 py-2 text-xs leading-relaxed ${cls}`}>
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <span>{children}</span>
    </div>
  )
}
