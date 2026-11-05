import { useEffect, useState } from 'react'
import {
  AtSign,
  Loader2,
  Check,
  Plus,
  Trash2,
  Save,
  Eye,
  EyeOff,
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Send,
  Zap,
  Info,
  RefreshCw,
} from 'lucide-react'
import {
  useImapMailStore,
  type ImapAccount,
  type ImapAccountPatch,
  type ImapSecurity,
  type ImapTestResult,
} from '../../stores/imapMailStore'
import { useAuthStore } from '../../stores/authStore'
import { AgenticSignatureField, normalizeSignature } from './AgenticSignatureField'
import { AgentAllowlistPicker } from './AgentAllowlistPicker'

const field =
  'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500 focus:outline-none disabled:opacity-60'
const label = 'block text-xs font-medium text-zinc-400 mb-1'
const pill = 'inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border'
const btnPrimary =
  'inline-flex items-center gap-2 rounded-md bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-sm text-white disabled:opacity-50 shadow-sm'
const btnSecondary =
  'inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 disabled:opacity-50'

const DEFAULT_PORT: Record<'imap' | 'smtp', Record<ImapSecurity, number>> = {
  imap: { ssl: 993, starttls: 143, none: 143 },
  smtp: { ssl: 465, starttls: 587, none: 25 },
}

function parseRecipients(text: string): string[] {
  const seen = new Set<string>()
  for (const raw of text.split(/[\n,]+/)) {
    const entry = raw.trim().toLowerCase()
    if (entry) seen.add(entry)
  }
  return [...seen]
}

function fmtTime(ts: string | undefined): string {
  if (!ts) return ''
  const d = new Date(ts)
  return isNaN(d.getTime()) ? ts : d.toLocaleString()
}

// ── the connection form (add, or edit an inbox's settings) ──────────

interface FormState {
  label: string
  email: string
  username: string
  password: string
  display_name: string
  imap_host: string
  imap_port: string
  imap_security: ImapSecurity
  receive_only: boolean
  smtp_host: string
  smtp_port: string
  smtp_security: ImapSecurity
  separate_smtp: boolean
  smtp_username: string
  smtp_password: string
  save_sent: boolean
  provider: string
}

function blankForm(): FormState {
  return {
    label: '', email: '', username: '', password: '', display_name: '',
    imap_host: '', imap_port: '993', imap_security: 'ssl', receive_only: false,
    smtp_host: '', smtp_port: '465', smtp_security: 'ssl', separate_smtp: false,
    smtp_username: '', smtp_password: '', save_sent: true, provider: '',
  }
}

function formFrom(a: ImapAccount): FormState {
  return {
    label: a.label || '', email: a.email, username: a.username, password: '',
    display_name: a.display_name || '', imap_host: a.imap_host, imap_port: String(a.imap_port),
    imap_security: a.imap_security, receive_only: !a.smtp_host, smtp_host: a.smtp_host,
    smtp_port: String(a.smtp_port || 465), smtp_security: a.smtp_security || 'ssl',
    separate_smtp: !!a.smtp_username, smtp_username: a.smtp_username || '', smtp_password: '',
    save_sent: a.save_sent, provider: a.provider || '',
  }
}

function patchFrom(f: FormState): ImapAccountPatch {
  const p: ImapAccountPatch = {
    label: f.label.trim(), email: f.email.trim(), username: f.username.trim() || f.email.trim(),
    display_name: f.display_name.trim(), imap_host: f.imap_host.trim(),
    imap_port: Number(f.imap_port) || DEFAULT_PORT.imap[f.imap_security],
    imap_security: f.imap_security,
    smtp_host: f.receive_only ? '' : f.smtp_host.trim(),
    smtp_port: Number(f.smtp_port) || DEFAULT_PORT.smtp[f.smtp_security],
    smtp_security: f.smtp_security,
    smtp_username: !f.receive_only && f.separate_smtp ? f.smtp_username.trim() : '',
    save_sent: f.save_sent,
    provider: f.provider,
  }
  if (f.password) p.password = f.password
  if (!f.receive_only && f.separate_smtp && f.smtp_password) p.smtp_password = f.smtp_password
  return p
}

function SecuritySelect({ value, onChange, disabled }: {
  value: ImapSecurity; onChange: (v: ImapSecurity) => void; disabled?: boolean
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value as ImapSecurity)}
      className={field}
    >
      <option value="ssl">SSL/TLS</option>
      <option value="starttls">STARTTLS</option>
      <option value="none">None (this machine only)</option>
    </select>
  )
}

function TestResult({ result }: { result: ImapTestResult }) {
  const row = (name: string, ok: boolean, text: string) => (
    <div className="flex items-start gap-2">
      {ok ? <Check className="h-3.5 w-3.5 text-emerald-500 shrink-0 mt-0.5" />
        : <AlertCircle className="h-3.5 w-3.5 text-red-500 shrink-0 mt-0.5" />}
      <span className={ok ? 'text-zinc-300' : 'text-red-500'}>
        <span className="font-medium">{name}:</span> {text}
      </span>
    </div>
  )
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-xs space-y-1">
      {row('IMAP', result.imap.ok, result.imap.ok
        ? `signed in — ${result.imap.folders ?? 0} folders${result.imap.roles?.length ? ` (${result.imap.roles.join(', ')})` : ''}`
        : result.imap.error || 'failed')}
      {row('SMTP', result.smtp.ok, result.smtp.skipped ? 'none (receive-only)'
        : result.smtp.ok ? 'signed in' : result.smtp.error || 'failed')}
    </div>
  )
}

function ConnectionForm({ initial, editing, onDone, onCancel }: {
  initial: FormState
  editing?: ImapAccount
  onDone: (account: ImapAccount) => void
  onCancel?: () => void
}) {
  const { suggest, test, add, update } = useImapMailStore()
  const [f, setF] = useState<FormState>(initial)
  const [showPw, setShowPw] = useState(false)
  const [hint, setHint] = useState('')
  const [microsoft, setMicrosoft] = useState(false)
  const [busy, setBusy] = useState<'' | 'test' | 'save' | 'suggest'>('')
  const [error, setError] = useState('')
  const [result, setResult] = useState<ImapTestResult | null>(null)
  const [suggestedFor, setSuggestedFor] = useState(editing ? editing.email.toLowerCase() : '')
  // The username the last lookup filled in: replaced by the next lookup; one
  // the person typed themselves is kept.
  const [autoUsername, setAutoUsername] = useState(editing ? editing.username : '')

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) => setF((s) => ({ ...s, [k]: v }))

  const applySuggestion = async () => {
    const email = f.email.trim().toLowerCase()
    if (!email.includes('@') || email === suggestedFor) return
    setSuggestedFor(email)
    setBusy('suggest')
    const s = await suggest(email)
    setBusy('')
    if (!s) return
    setHint(s.hint || (s.source === 'guess'
      ? "Couldn't find this provider's settings — these are the usual ones; check them with your mail provider."
      : ''))
    setMicrosoft(!!s.microsoft)
    const nextUser = s.username || email
    setAutoUsername(nextUser)
    setF((cur) => ({
      ...cur,
      username: cur.username && cur.username !== autoUsername && cur.username.toLowerCase() !== email
        ? cur.username : nextUser,
      imap_host: s.imap_host, imap_port: String(s.imap_port), imap_security: s.imap_security,
      smtp_host: s.smtp_host, smtp_port: String(s.smtp_port), smtp_security: s.smtp_security,
      receive_only: !s.smtp_host ? true : cur.receive_only,
      save_sent: s.save_sent, provider: s.provider || '',
    }))
  }

  const needsPassword = !editing && !f.password
  const valid = f.email.includes('@') && f.imap_host.trim() && !needsPassword
    && (f.receive_only || f.smtp_host.trim())
    && (!f.separate_smtp || f.receive_only || (f.smtp_username.trim() && (editing?.separate_smtp_login || f.smtp_password)))

  const runTest = async () => {
    setError('')
    setResult(null)
    setBusy('test')
    const body = patchFrom(f)
    if (editing) body.account_id = editing.id
    const r = await test(body)
    setBusy('')
    if (r.error) setError(r.error)
    else if (r.result) setResult(r.result)
  }

  const save = async () => {
    setError('')
    setBusy('save')
    const body = patchFrom(f)
    const r = editing ? await update(editing.id, body) : await add(body)
    setBusy('')
    if (r.error) {
      setError(r.error)
      return
    }
    if (r.account) onDone(r.account)
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <label className={label}>Email address</label>
          <input
            type="email" value={f.email} autoComplete="off" placeholder="you@company.com"
            onChange={(e) => set('email', e.target.value)} onBlur={applySuggestion}
            className={field} disabled={!!busy && busy !== 'suggest'}
          />
        </div>
        <div>
          <label className={label}>Name <span className="text-zinc-600">(optional — e.g. Work, Support)</span></label>
          <input value={f.label} onChange={(e) => set('label', e.target.value)} className={field}
                 placeholder="Work" maxLength={60} />
        </div>
        <div>
          <label className={label}>Username</label>
          <input value={f.username} onChange={(e) => set('username', e.target.value)} className={field}
                 placeholder={f.email || 'usually the email address'} autoComplete="off" />
        </div>
        <div>
          <label className={label}>
            Password {editing && <span className="text-zinc-600">(leave empty to keep the saved one)</span>}
          </label>
          <div className="relative">
            <input
              type={showPw ? 'text' : 'password'} value={f.password} autoComplete="new-password"
              onChange={(e) => set('password', e.target.value)} className={field + ' pr-8'}
              placeholder={editing ? '••••••••' : 'password or app password'}
            />
            <button type="button" onClick={() => setShowPw((v) => !v)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
                    aria-label={showPw ? 'Hide password' : 'Show password'}>
              {showPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
            </button>
          </div>
        </div>
      </div>

      {busy === 'suggest' && (
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <Loader2 className="h-3 w-3 animate-spin" /> Looking up this provider's settings…
        </div>
      )}
      {(hint || f.provider) && busy !== 'suggest' && (
        <div className={'flex items-start gap-2 rounded-md border px-3 py-2 text-xs ' + (microsoft
          ? 'border-amber-500/30 bg-amber-500/10 text-amber-600'
          : 'border-zinc-800 bg-zinc-950/60 text-zinc-400')}>
          <Info className="h-3.5 w-3.5 shrink-0 mt-0.5" />
          <span>{f.provider && <span className="text-zinc-200">{f.provider}. </span>}{hint}</span>
        </div>
      )}

      <div className="rounded-md border border-zinc-800 p-3 space-y-3">
        <div className="text-xs font-medium text-zinc-300">Incoming mail (IMAP)</div>
        <div>
          <label className={label}>Server</label>
          <input value={f.imap_host} onChange={(e) => set('imap_host', e.target.value)} className={field}
                 placeholder="imap.example.com" />
        </div>
        <div className="grid grid-cols-[minmax(0,1fr)_6rem] gap-3">
          <div>
            <label className={label}>Security</label>
            <SecuritySelect value={f.imap_security} onChange={(v) => {
              setF((s) => ({ ...s, imap_security: v, imap_port: String(DEFAULT_PORT.imap[v]) }))
            }} />
          </div>
          <div>
            <label className={label}>Port</label>
            <input value={f.imap_port} inputMode="numeric" onChange={(e) => set('imap_port', e.target.value)}
                   className={field} />
          </div>
        </div>
      </div>

      <div className="rounded-md border border-zinc-800 p-3 space-y-3">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="text-xs font-medium text-zinc-300">Outgoing mail (SMTP)</div>
          <label className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer">
            <input type="checkbox" checked={f.receive_only} className="accent-violet-600"
                   onChange={(e) => set('receive_only', e.target.checked)} />
            Receive only (no sending)
          </label>
        </div>
        {!f.receive_only && (
          <>
            <div>
              <label className={label}>Server</label>
              <input value={f.smtp_host} onChange={(e) => set('smtp_host', e.target.value)} className={field}
                     placeholder="smtp.example.com" />
            </div>
            <div className="grid grid-cols-[minmax(0,1fr)_6rem] gap-3">
              <div>
                <label className={label}>Security</label>
                <SecuritySelect value={f.smtp_security} onChange={(v) => {
                  setF((s) => ({ ...s, smtp_security: v, smtp_port: String(DEFAULT_PORT.smtp[v]) }))
                }} />
              </div>
              <div>
                <label className={label}>Port</label>
                <input value={f.smtp_port} inputMode="numeric" onChange={(e) => set('smtp_port', e.target.value)}
                       className={field} />
              </div>
            </div>
            <label className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer">
              <input type="checkbox" checked={f.separate_smtp} className="accent-violet-600"
                     onChange={(e) => set('separate_smtp', e.target.checked)} />
              SMTP uses a different login (a relay service, or other SMTP credentials)
            </label>
            {f.separate_smtp && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className={label}>SMTP username</label>
                  <input value={f.smtp_username} onChange={(e) => set('smtp_username', e.target.value)}
                         className={field} autoComplete="off" />
                </div>
                <div>
                  <label className={label}>
                    SMTP password {editing?.separate_smtp_login && <span className="text-zinc-600">(empty = keep)</span>}
                  </label>
                  <input type={showPw ? 'text' : 'password'} value={f.smtp_password} autoComplete="new-password"
                         onChange={(e) => set('smtp_password', e.target.value)} className={field} />
                </div>
              </div>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className={label}>Sender name <span className="text-zinc-600">(optional)</span></label>
                <input value={f.display_name} onChange={(e) => set('display_name', e.target.value)}
                       className={field} placeholder="Ana Horvat" maxLength={120} />
              </div>
              <label className="flex items-start gap-1.5 text-xs text-zinc-400 cursor-pointer sm:mt-6">
                <input type="checkbox" checked={f.save_sent} className="accent-violet-600 mt-0.5"
                       onChange={(e) => set('save_sent', e.target.checked)} />
                <span>Put a copy of each sent email in Sent
                  <span className="block text-zinc-600">Turn off for providers that do it themselves (Gmail, Yahoo, Outlook.com) or you'll get two.</span>
                </span>
              </label>
            </div>
          </>
        )}
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-500">
          <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
          <span className="break-words">{error}</span>
        </div>
      )}
      {result && <TestResult result={result} />}

      <div className="flex items-center gap-2 flex-wrap">
        <button onClick={save} disabled={!valid || !!busy} className={btnPrimary}>
          {busy === 'save' ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
            : editing ? <Save className="h-3.5 w-3.5" /> : <Plus className="h-3.5 w-3.5" />}
          {editing ? 'Save' : 'Connect inbox'}
        </button>
        <button onClick={runTest} disabled={!valid || !!busy} className={btnSecondary}>
          {busy === 'test' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Zap className="h-3.5 w-3.5" />}
          Test
        </button>
        {onCancel && (
          <button onClick={onCancel} disabled={!!busy} className="text-sm text-zinc-500 hover:text-zinc-300 px-2">
            Cancel
          </button>
        )}
        <span className="text-[11px] text-zinc-500">
          {busy === 'save' ? 'Signing in to check the settings…'
            : 'Flight Deck signs in before saving. The password is stored encrypted and never shown to agents.'}
        </span>
      </div>
    </div>
  )
}

// ── one connected inbox ─────────────────────────────────────────────

function SendingSection({ account }: { account: ImapAccount }) {
  const { update, deckSendDisabled, sends, fetchSends } = useImapMailStore()
  const [enabled, setEnabled] = useState(account.sending.enabled)
  const [recipientsText, setRecipientsText] = useState(account.sending.allowed_recipients.join('\n'))
  const [limitText, setLimitText] = useState(String(account.sending.daily_limit))
  const [signatureText, setSignatureText] = useState(account.signature || '')
  const [agents, setAgents] = useState<string[]>(account.agents || [])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchSends(account.id)
  }, [account.id, fetchSends])

  const recipients = parseRecipients(recipientsText)
  const limit = Number(limitText)
  const limitValid = limitText.trim() !== '' && Number.isInteger(limit) && limit >= 1 && limit <= 500
  const signature = normalizeSignature(signatureText)
  const agentsDirty = agents.slice().sort().join(',') !== (account.agents || []).slice().sort().join(',')
  const dirty = enabled !== account.sending.enabled
    || recipients.join('\n') !== account.sending.allowed_recipients.join('\n')
    || (limitValid && limit !== account.sending.daily_limit) || signature !== (account.signature || '')
    || agentsDirty

  const handleSave = async () => {
    setSaving(true)
    setError('')
    const r = await update(account.id, {
      sending: { enabled, allowed_recipients: recipients, daily_limit: limit },
      signature, agents,
    })
    setSaving(false)
    if (r.error) setError(r.error)
    else if (r.account) {
      setRecipientsText(r.account.sending.allowed_recipients.join('\n'))
      fetchSends(account.id)
    }
  }

  const list = sends[account.id] || []

  return (
    <div className="space-y-3">
      <div className="rounded-md border border-zinc-800 bg-zinc-950/50 p-3 space-y-3">
        <div className="flex items-center gap-2 flex-wrap">
          <Send className="h-3.5 w-3.5 text-zinc-400" />
          <span className="text-xs font-medium text-zinc-300">Email sending</span>
          {!account.can_send ? (
            <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Receive only</span>
          ) : deckSendDisabled ? (
            <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Off on this deck</span>
          ) : account.sending.enabled ? (
            <span className={`${pill} bg-emerald-500/15 text-emerald-500 border-emerald-500/30`}>
              <Check className="h-2.5 w-2.5" /> Agents can send
            </span>
          ) : (
            <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Drafts only</span>
          )}
        </div>
        {account.can_send && (
          <label className="flex items-start gap-2 cursor-pointer">
            <input type="checkbox" checked={enabled} disabled={saving} className="mt-0.5 accent-violet-600"
                   onChange={(e) => setEnabled(e.target.checked)} />
            <div className="flex-1 min-w-0">
              <div className="text-sm text-zinc-200">Let my agents send email from this inbox</div>
              <div className="text-xs text-zinc-500 mt-0.5 leading-snug">
                When on, your agents can send email (including replies) through this inbox's SMTP
                server. They're instructed to send only when you ask; every send is listed below and
                in your notifications. Off = drafts only (saved to this inbox's Drafts folder).
              </div>
            </div>
          </label>
        )}
        {deckSendDisabled && account.can_send && (
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-600">
            Sending is disabled on this Flight Deck by the administrator (<code>FD_IMAP_SEND=off</code>).
          </div>
        )}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div className="sm:col-span-2 space-y-3">
            {account.can_send && (
              <div>
                <label className={label}>Only to these recipients <span className="text-zinc-600">(optional)</span></label>
                <textarea value={recipientsText} onChange={(e) => setRecipientsText(e.target.value)} rows={3}
                          disabled={saving} placeholder={'alice@example.com\n@yourcompany.com'}
                          className={`${field} font-mono`} />
                <div className="mt-1 text-[11px] text-zinc-500">One address or @domain per line. Empty = anyone.</div>
              </div>
            )}
            <AgenticSignatureField value={signatureText} onChange={setSignatureText} disabled={saving}
                                   writes={account.can_send ? 'email and draft' : 'draft'} />
          </div>
          {account.can_send && (
            <div>
              <label className={label}>Daily limit</label>
              <input type="number" min={1} max={500} step={1} value={limitText} disabled={saving}
                     onChange={(e) => setLimitText(e.target.value)} className={field} />
              {!limitValid && <div className="mt-1 text-[11px] text-red-500">A whole number from 1 to 500.</div>}
              <div className="mt-1 text-[11px] text-zinc-500">
                Sent in the last 24h:{' '}
                <span className={account.sent_last_24h >= account.sending.daily_limit ? 'text-amber-600' : 'text-zinc-300'}>
                  {account.sent_last_24h} / {account.sending.daily_limit}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="rounded-md border border-zinc-800 bg-zinc-950/50 p-3 space-y-2">
        <div className="text-xs font-medium text-zinc-300">Which agents can use this inbox</div>
        <AgentAllowlistPicker
          value={agents}
          onChange={setAgents}
          hint="Pick the agents that may read, draft and send from this inbox. Leave empty to let all your agents use it."
        />
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-500">
          <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" /><span className="break-words">{error}</span>
        </div>
      )}
      <div className="flex items-center gap-2">
        <button onClick={handleSave} disabled={!dirty || saving || !limitValid} className={btnPrimary}>
          {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
          Save
        </button>
        {dirty && !saving && <span className="text-xs text-zinc-500">Unsaved changes</span>}
      </div>

      {account.can_send && (
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-xs text-zinc-500">Recently sent by your agents</span>
            <button type="button" onClick={() => fetchSends(account.id)}
                    className="text-zinc-500 hover:text-zinc-300" title="Reload">
              <RefreshCw className="h-3 w-3" />
            </button>
          </div>
          {list.length === 0 ? (
            <div className="text-xs text-zinc-500">Nothing sent from this inbox by your agents yet.</div>
          ) : (
            <div className="rounded-md border border-zinc-800 bg-zinc-950/60 divide-y divide-zinc-800">
              {list.map((s) => (
                <div key={s.id} className="px-3 py-2 text-xs"
                     title={[s.cc && `Cc: ${s.cc}`, s.bcc && `Bcc: ${s.bcc}`].filter(Boolean).join('\n') || undefined}>
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-zinc-500 shrink-0 tabular-nums">{fmtTime(s.created_at)}</span>
                    <span className="text-zinc-400 truncate">{s.agent}</span>
                    {s.status === 'unknown' && (
                      <span className="ml-auto shrink-0 text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-600 border border-amber-500/30"
                            title="The mail server never confirmed this send — it may or may not have gone out.">
                        unconfirmed — check Sent
                      </span>
                    )}
                  </div>
                  <div className="flex items-baseline gap-1.5 min-w-0 mt-0.5">
                    <span className="text-zinc-500 shrink-0">To</span>
                    <span className="text-zinc-300 truncate">{s.to}</span>
                  </div>
                  <div className="text-zinc-200 truncate mt-0.5">{s.subject || '(no subject)'}</div>
                  {!!s.attachments?.length && (
                    <div className="text-zinc-500 truncate mt-0.5">Attached: {s.attachments.join(', ')}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function AccountRow({ account }: { account: ImapAccount }) {
  const remove = useImapMailStore((s) => s.remove)
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<'use' | 'settings'>('use')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const handleRemove = async () => {
    if (!confirm(`Disconnect ${account.email}?\n\nFlight Deck forgets its settings and password; your agents lose access to it. Its send history stays.`)) return
    setBusy(true)
    const err = await remove(account.id)
    setBusy(false)
    if (err) setError(err)
  }

  const sendPill = !account.can_send
    ? <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Receive only</span>
    : account.sending.enabled
      ? <span className={`${pill} bg-emerald-500/15 text-emerald-500 border-emerald-500/30`}>Sending on</span>
      : <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Drafts only</span>

  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-950/40">
      <button type="button" onClick={() => setOpen((v) => !v)}
              className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-zinc-900/60"
              aria-expanded={open}>
        {open ? <ChevronDown className="h-4 w-4 text-zinc-500" /> : <ChevronRight className="h-4 w-4 text-zinc-500" />}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm text-zinc-100 truncate">{account.label || account.email}</span>
            {account.label && <span className="text-xs text-zinc-500 truncate">{account.email}</span>}
            {sendPill}
            {account.agents.length > 0 && (
              <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>
                {account.agents.length} agent{account.agents.length === 1 ? '' : 's'}
              </span>
            )}
          </div>
          <div className="text-[11px] text-zinc-500 truncate mt-0.5">
            {account.imap_host}{account.smtp_host ? ` · ${account.smtp_host}` : ''} · agents use{' '}
            <code className="text-zinc-400">account={account.id}</code>
          </div>
        </div>
      </button>
      {open && (
        <div className="border-t border-zinc-800 p-3 space-y-3">
          <div className="flex items-center gap-1 text-xs">
            {(['use', 'settings'] as const).map((t) => (
              <button key={t} type="button" onClick={() => setTab(t)}
                      className={'px-2.5 py-1 rounded-md ' + (tab === t ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300')}>
                {t === 'use' ? 'Sending & access' : 'Connection'}
              </button>
            ))}
            <div className="flex-1" />
            <button type="button" onClick={handleRemove} disabled={busy}
                    className="inline-flex items-center gap-1 text-xs text-red-500 hover:text-red-400 disabled:opacity-50">
              {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />} Disconnect
            </button>
          </div>
          {error && <div className="text-xs text-red-500">{error}</div>}
          {tab === 'use' ? <SendingSection key={account.updated_at} account={account} />
            : <ConnectionForm key={account.updated_at} initial={formFrom(account)} editing={account}
                              onDone={() => setTab('use')} />}
        </div>
      )}
    </div>
  )
}

// ── the card ────────────────────────────────────────────────────────

export default function ImapConnection() {
  const { accounts, loading, error, refresh, maxAccounts, cryptoAvailable } = useImapMailStore()
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const desktopDeck = useAuthStore((s) => s.desktopDeck)
  const unavailable = authEnabled === false && !desktopDeck
  const [adding, setAdding] = useState(false)

  useEffect(() => {
    if (!unavailable) refresh()
  }, [unavailable, refresh])

  const count = accounts?.length ?? 0
  const icon = (
    <div className="h-10 w-10 rounded-md bg-gradient-to-br from-violet-500/20 to-sky-500/20 border border-zinc-800 flex items-center justify-center">
      <AtSign className="h-5 w-5 text-zinc-200" />
    </div>
  )

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
      <div className="flex items-center gap-3 px-5 py-4 border-b border-zinc-800">
        {icon}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">Email (IMAP)</h3>
            {unavailable ? (
              <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Not available</span>
            ) : count > 0 ? (
              <span className={`${pill} bg-emerald-500/15 text-emerald-500 border-emerald-500/30`}>
                <Check className="h-2.5 w-2.5" /> {count} inbox{count === 1 ? '' : 'es'}
              </span>
            ) : null}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            Any mailbox with IMAP and SMTP — company mail, Fastmail, iCloud, Yahoo, Zoho, GMX … — as many as you need.
          </div>
        </div>
        {!unavailable && (
          <button type="button" onClick={() => refresh()} disabled={loading}
                  className="text-zinc-500 hover:text-zinc-300 disabled:opacity-50" title="Reload">
            <RefreshCw className={'h-3.5 w-3.5 ' + (loading ? 'animate-spin' : '')} />
          </button>
        )}
      </div>

      <div className="px-5 py-4 space-y-3">
        {unavailable ? (
          <div className="text-xs text-zinc-400 leading-relaxed">
            Email (IMAP) isn't available on this deck: Flight Deck sign-in is turned off, and each inbox
            belongs to a signed-in Flight Deck user. Run Flight Deck with sign-in on
            (<code className="text-zinc-300">FD_AUTH_ENABLED=true</code>) — or use the desktop app.
          </div>
        ) : (
          <>
            {error && (
              <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-500">
                <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" /><span className="break-words">{error}</span>
              </div>
            )}
            {!cryptoAvailable && (
              <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-600">
                This Flight Deck can't store passwords encrypted (the <code>cryptography</code> package is
                missing), so inboxes can't be connected until the administrator installs it.
              </div>
            )}
            {accounts === null && loading && (
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…
              </div>
            )}
            {accounts && accounts.length > 0 && (
              <div className="space-y-2">
                {accounts.map((a) => <AccountRow key={a.id} account={a} />)}
              </div>
            )}
            {accounts && accounts.length === 0 && !adding && (
              <div className="text-xs text-zinc-500 leading-relaxed">
                No inbox connected yet. Connect one with its email address and password (many providers
                want an app password). Your agents can then read, search, draft and — if you turn it on
                per inbox — send email from it.
              </div>
            )}
            {adding ? (
              <div className="rounded-md border border-violet-500/30 bg-zinc-950/40 p-3">
                <div className="text-xs font-medium text-zinc-200 mb-3">Connect an inbox</div>
                <ConnectionForm initial={blankForm()} onDone={() => setAdding(false)} onCancel={() => setAdding(false)} />
              </div>
            ) : (
              accounts && cryptoAvailable && count < maxAccounts && (
                <button type="button" onClick={() => setAdding(true)} className={btnSecondary}>
                  <Plus className="h-3.5 w-3.5" /> {count ? 'Connect another inbox' : 'Connect an inbox'}
                </button>
              )
            )}
          </>
        )}
      </div>
    </div>
  )
}
