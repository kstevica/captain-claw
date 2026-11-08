import { useCallback, useEffect, useMemo, useState } from 'react'
import { Check, Copy, Link2, Loader2, Mail, MailPlus, RefreshCw, Send, Trash2, UserCheck, X } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { matchesQuery } from './deckConfig'
import { FilterSelect, FilterSummary, SearchBox, Segmented } from './ListFilters'

// Users → Invitations: a one-time link to <deck>/invite#<token>,
// emailed from the admin's own connected Google (Gmail) or Microsoft (Outlook)
// account — or no email, and the admin passes the link on. The invitee
// chooses their own password and the account is created then
// (captain_claw/flight_deck/invite_routes.py).

type Via = 'google' | 'microsoft' | 'link'
export type Provider = Exclude<Via, 'link'>

export interface MailOption {
  available: boolean
  from: string
  reason: string
}

export type MailOptions = Record<Provider, MailOption>

export interface Invite {
  id: string
  email: string
  display_name: string
  role: string
  kiosk: boolean | null
  note: string
  status: 'pending' | 'accepted'
  invited_by: string
  invited_by_name: string
  created_at: string
  expires_at: string
  expired: boolean
  sent_via: '' | Provider
  sent_from: string
  sent_at: string
  accepted_at: string  // blank until accepted
}

// Pending invitations split by whether their link still works.
type StatusFilter = 'pending' | 'expired' | 'accepted' | 'all'
type RoleFilter = 'all' | 'admin' | 'user'
type SentFilter = 'any' | Provider | 'link'
type SortKey = 'newest' | 'oldest'

function inviteState(inv: Invite): Exclude<StatusFilter, 'all'> {
  if (inv.status === 'accepted') return 'accepted'
  return inv.expired ? 'expired' : 'pending'
}

interface EmailOutcome {
  sent: boolean
  via: Via
  from: string
  error: string
}

interface InviteResult {
  invite: Invite
  url: string
  email: EmailOutcome
}

export const PROVIDER_LABEL: Record<Provider, string> = { google: 'Gmail', microsoft: 'Outlook' }
export const PROVIDERS: Provider[] = ['google', 'microsoft']

function _headers(): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: _headers(), credentials: 'include', ...init })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`)
  }
  return res.json()
}

function shortDate(iso: string): string {
  if (!iso) return ''
  try { return new Date(iso).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) } catch { return iso }
}

function when(iso: string): string {
  if (!iso) return ''
  try { return new Date(iso).toLocaleString() } catch { return iso }
}

// A link the invitee can't open from elsewhere: the deck was reached on this machine.
export function localOnly(url: string): boolean {
  try {
    const host = new URL(url).hostname
    return host === 'localhost' || host.endsWith('.localhost') || host === '127.0.0.1' || host === '[::1]'
  } catch { return false }
}

function firstAvailable(mail: MailOptions | null): Via {
  return PROVIDERS.find((p) => mail?.[p].available) ?? 'link'
}

export function InvitesAdmin({ refreshTick, kioskDefault, kioskNote, onOpenCount }: {
  refreshTick: number
  kioskDefault: boolean
  kioskNote: string | null
  // Invitations still waiting to be used (expired ones left out).
  onOpenCount: (n: number) => void
}) {
  const [invites, setInvites] = useState<Invite[] | null>(null)
  const [mail, setMail] = useState<MailOptions | null>(null)
  const [ttlDays, setTtlDays] = useState(7)
  const [open, setOpen] = useState(false)
  const [result, setResult] = useState<InviteResult | null>(null)
  const [busyId, setBusyId] = useState('')
  const [error, setError] = useState('')
  const [status, setStatus] = useState<StatusFilter>('pending')
  const [query, setQuery] = useState('')
  const [role, setRole] = useState<RoleFilter>('all')
  const [sent, setSent] = useState<SentFilter>('any')
  const [sort, setSort] = useState<SortKey>('newest')

  const load = useCallback(async () => {
    try {
      const data = await call<{ invites: Invite[]; mail: MailOptions; ttl_days: number }>('/fd/admin/invites?status=all')
      setInvites(data.invites)
      // Waiting to be used, like Overview's count: an expired link isn't.
      onOpenCount(data.invites.filter((i) => i.status === 'pending' && !i.expired).length)
      setMail(data.mail)
      setTtlDays(data.ttl_days)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [onOpenCount])

  useEffect(() => { load() }, [load, refreshTick])

  const resend = async (inv: Invite, via: Via) => {
    setBusyId(inv.id); setError('')
    try {
      const r = await call<InviteResult>(`/fd/admin/invites/${inv.id}/resend`, {
        method: 'POST', body: JSON.stringify({ send_via: via }),
      })
      setResult(r)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyId('')
      load()
    }
  }

  const remove = async (inv: Invite) => {
    const ask = inv.status === 'accepted'
      ? `Remove the record of ${inv.email}'s accepted invitation? Their account stays.`
      : `Remove the invitation for ${inv.email}? Its link stops working.`
    if (!confirm(ask)) return
    setBusyId(inv.id); setError('')
    try {
      await call(`/fd/admin/invites/${inv.id}`, { method: 'DELETE' })
      if (result?.invite.id === inv.id) setResult(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyId('')
      load()
    }
  }

  const counts = useMemo(() => {
    const c = { pending: 0, expired: 0, accepted: 0, all: 0 }
    for (const inv of invites ?? []) { c[inviteState(inv)] += 1; c.all += 1 }
    return c
  }, [invites])

  const shown = useMemo(() => {
    const list = (invites ?? []).filter((inv) => {
      if (status !== 'all' && inviteState(inv) !== status) return false
      if (!matchesQuery(query, inv.email, inv.display_name, inv.invited_by_name, inv.note)) return false
      if (role !== 'all' && inv.role !== role) return false
      if (sent === 'link' ? !!inv.sent_via : sent !== 'any' && inv.sent_via !== sent) return false
      return true
    })
    return list.sort((a, b) => (sort === 'oldest'
      ? a.created_at.localeCompare(b.created_at)
      : b.created_at.localeCompare(a.created_at)))
  }, [invites, status, query, role, sent, sort])

  // The status switch is the view; these narrow it.
  const filtered = query.trim() !== '' || role !== 'all' || sent !== 'any'
  const clearFilters = () => { setQuery(''); setRole('all'); setSent('any') }
  const inView = status === 'all' ? counts.all : counts[status]

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <SearchBox value={query} onChange={setQuery} placeholder="Search email, name, inviter or note" />
        <button
          onClick={() => { setOpen(true); setResult(null) }}
          disabled={open}
          className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50"
        >
          <MailPlus className="h-3.5 w-3.5" />
          Invite by email
        </button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Segmented
          label="Show"
          value={status}
          onChange={setStatus}
          options={[
            { value: 'pending', label: 'Pending', count: counts.pending },
            { value: 'expired', label: 'Expired', count: counts.expired, warn: true },
            { value: 'accepted', label: 'Accepted', count: counts.accepted },
            { value: 'all', label: 'All', count: counts.all },
          ]}
        />
        <FilterSelect label="Role" value={role} onChange={(v) => setRole(v as RoleFilter)} active={role !== 'all'}>
          <option value="all">All roles</option>
          <option value="admin">Admins</option>
          <option value="user">Users</option>
        </FilterSelect>
        <FilterSelect label="Sent" value={sent} onChange={(v) => setSent(v as SentFilter)} active={sent !== 'any'}>
          <option value="any">Any delivery</option>
          {PROVIDERS.map((p) => <option key={p} value={p}>Emailed from {PROVIDER_LABEL[p]}</option>)}
          <option value="link">Link only, not emailed</option>
        </FilterSelect>
        <span className="flex-1" />
        <FilterSelect label="Sort" value={sort} onChange={(v) => setSort(v as SortKey)} active={false}>
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
        </FilterSelect>
      </div>

      {open && (
        <InviteForm
          mail={mail}
          kioskDefault={kioskDefault}
          kioskNote={kioskNote}
          onCancel={() => setOpen(false)}
          onDone={(r) => { setResult(r); setOpen(false); setStatus('pending'); load() }}
        />
      )}

      {result && <ResultBanner result={result} ttlDays={ttlDays} onClose={() => setResult(null)} />}

      {error && (
        <div className="rounded-md border border-red-900/50 bg-red-950/20 px-4 py-2 text-sm text-red-400">{error}</div>
      )}

      {invites === null && !error && (
        <div className="flex justify-center py-6">
          <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
        </div>
      )}

      {invites && filtered && (
        <FilterSummary shown={shown.length} total={inView} noun="invitations" onClear={clearFilters} />
      )}

      {shown.length > 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/30">
          {shown.map((inv) => (
            <InviteRow
              key={inv.id}
              inv={inv}
              mail={mail}
              busy={busyId === inv.id}
              disabled={!!busyId}
              onResend={(via) => resend(inv, via)}
              onRemove={() => remove(inv)}
            />
          ))}
        </div>
      )}

      {invites && shown.length === 0 && (
        <p className="text-center text-sm text-zinc-500 py-8">
          {filtered
            ? 'No invitations match these filters.'
            : status === 'pending'
              ? counts.expired > 0
                ? `No invitations waiting — ${counts.expired} expired (see Expired).`
                : 'No invitations waiting to be accepted.'
              : status === 'expired' ? 'No expired invitations.'
                : status === 'accepted' ? 'No accepted invitations yet.'
                  : 'No invitations yet.'}
        </p>
      )}
    </div>
  )
}

function InviteRow({ inv, mail, busy, disabled, onResend, onRemove }: {
  inv: Invite
  mail: MailOptions | null
  busy: boolean
  disabled: boolean
  onResend: (via: Via) => void
  onRemove: () => void
}) {
  const accepted = inv.status === 'accepted'
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-zinc-800/60 px-4 py-2.5 last:border-b-0">
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="flex flex-wrap items-center gap-2">
          {accepted
            ? <UserCheck className="h-3.5 w-3.5 shrink-0 text-emerald-500" />
            : <Mail className="h-3.5 w-3.5 shrink-0 text-zinc-500" />}
          <span className="truncate text-sm text-zinc-100">{inv.display_name || inv.email}</span>
          {inv.display_name && <span className="truncate text-xs text-zinc-400">{inv.email}</span>}
          {inv.role === 'admin' && (
            <span className="rounded bg-amber-950/60 px-1.5 py-0.5 text-[10px] font-medium text-amber-400">admin</span>
          )}
          {inv.kiosk === true && (
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-300">kiosk</span>
          )}
          {accepted ? (
            <span className="rounded bg-emerald-950/60 px-1.5 py-0.5 text-[10px] font-medium text-emerald-400"
              title={inv.accepted_at ? when(inv.accepted_at) : undefined}>
              Accepted{inv.accepted_at ? ` ${shortDate(inv.accepted_at)}` : ''}
            </span>
          ) : inv.expired ? (
            <span className="rounded bg-red-950/60 px-1.5 py-0.5 text-[10px] font-medium text-red-400">Expired</span>
          ) : (
            <span className="text-[11px] text-zinc-500">until {shortDate(inv.expires_at)}</span>
          )}
        </div>
        <div className="text-[11px] text-zinc-500" title={when(inv.created_at)}>
          Invited{inv.invited_by_name ? ` by ${inv.invited_by_name}` : ''} {shortDate(inv.created_at)}
          {inv.sent_via
            ? <> · emailed from {inv.sent_from || PROVIDER_LABEL[inv.sent_via]} {shortDate(inv.sent_at)}</>
            : <> · link only, not emailed</>}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {busy && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
        {!accepted && PROVIDERS.filter((p) => mail?.[p].available).map((p) => (
          <button
            key={p}
            onClick={() => onResend(p)}
            disabled={disabled}
            title={`A new link, emailed from ${mail?.[p].from || PROVIDER_LABEL[p]} — the old link stops working`}
            className="flex items-center gap-1 rounded-md bg-zinc-800 px-2.5 py-1 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
          >
            <Send className="h-3 w-3" /> Resend via {PROVIDER_LABEL[p]}
          </button>
        ))}
        {!accepted && (
          <button
            onClick={() => onResend('link')}
            disabled={disabled}
            title="A new link to copy and pass on yourself — the old link stops working"
            className="flex items-center gap-1 rounded-md bg-zinc-800 px-2.5 py-1 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
          >
            <Link2 className="h-3 w-3" /> New link
          </button>
        )}
        <button
          onClick={onRemove}
          disabled={disabled}
          title={accepted ? 'Remove from the list (their account stays)' : 'Remove invitation'}
          className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-red-400 disabled:opacity-50"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}

function InviteForm({ mail, kioskDefault, kioskNote, onCancel, onDone }: {
  mail: MailOptions | null
  kioskDefault: boolean
  kioskNote: string | null
  onCancel: () => void
  onDone: (r: InviteResult) => void
}) {
  const [email, setEmail] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [role, setRole] = useState('user')
  const [kiosk, setKiosk] = useState(kioskDefault)
  const [note, setNote] = useState('')
  const [via, setVia] = useState<Via>(() => firstAvailable(mail))
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState('')

  // The options arrive after the form may already be open.
  useEffect(() => { setVia((v) => (v === 'link' || !mail?.[v].available ? firstAvailable(mail) : v)) }, [mail])

  const valid = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email.trim())

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!valid || saving) return
    setSaving(true); setErr('')
    try {
      const r = await call<InviteResult>('/fd/admin/invites', {
        method: 'POST',
        body: JSON.stringify({
          email: email.trim(),
          display_name: displayName.trim(),
          role,
          note: via === 'link' ? '' : note.trim(),
          send_via: via,
          ...(role === 'admin' ? {} : { kiosk }),
        }),
      })
      onDone(r)
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Failed to invite')
      setSaving(false)
    }
  }

  const field = 'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none'

  return (
    <form onSubmit={submit} className="rounded-lg border border-violet-800/40 bg-zinc-900/50 p-4 space-y-3">
      <div>
        <div className="flex items-center gap-2">
          <MailPlus className="h-4 w-4 text-violet-400" />
          <h3 className="text-sm font-semibold text-zinc-200">Invite a member</h3>
        </div>
        <p className="mt-0.5 text-xs text-zinc-500">
          They get a link that works once; opening it, they choose their own password and their account is created.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 wide:grid-cols-2">
        <div className="wide:col-span-2">
          <label className="block text-xs font-medium text-zinc-500 mb-1">Email</label>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus
            placeholder="teammate@example.com" autoComplete="off" className={field} />
        </div>
        <div>
          <label className="block text-xs font-medium text-zinc-500 mb-1">
            Name <span className="text-zinc-600">(optional)</span>
          </label>
          <input type="text" value={displayName} onChange={(e) => setDisplayName(e.target.value)} maxLength={100}
            placeholder="They can change it" className={field} />
        </div>
        <div>
          <label className="block text-xs font-medium text-zinc-500 mb-1">Role</label>
          <select value={role} onChange={(e) => setRole(e.target.value)} className={field}>
            <option value="user">user</option>
            <option value="admin">admin</option>
          </select>
        </div>
        {role !== 'admin' && (
          <div className="wide:col-span-2">
            <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
              <input type="checkbox" checked={kiosk} onChange={(e) => setKiosk(e.target.checked)}
                className="h-3.5 w-3.5 rounded border-zinc-600 bg-zinc-950 accent-violet-600" />
              Kiosk mode <span className="text-xs text-zinc-500">— simple chat only</span>
            </label>
            {kioskNote && <p className="mt-1 ml-5 text-[11px] text-zinc-500">{kioskNote}</p>}
          </div>
        )}
        <fieldset className="wide:col-span-2">
          <legend className="block text-xs font-medium text-zinc-500 mb-1.5">Send</legend>
          <div className="space-y-1.5">
            {PROVIDERS.map((p) => {
              const opt = mail?.[p]
              const disabled = !opt?.available
              return (
                <label key={p} className={`flex items-start gap-2.5 rounded-md border px-3 py-2 ${
                  via === p ? 'border-violet-600/60 bg-violet-950/20' : 'border-zinc-800'
                } ${disabled ? 'opacity-60' : 'cursor-pointer hover:border-zinc-700'}`}>
                  <input type="radio" name="invite-via" checked={via === p} disabled={disabled}
                    onChange={() => setVia(p)} className="mt-0.5 accent-violet-600" />
                  <span className="min-w-0">
                    <span className="block text-sm text-zinc-200">
                      Email from {PROVIDER_LABEL[p]}
                      {opt?.from && <span className="text-zinc-400"> · {opt.from}</span>}
                    </span>
                    {disabled && opt?.reason && <span className="block text-[11px] text-zinc-500">{opt.reason}</span>}
                  </span>
                </label>
              )
            })}
            <label className={`flex items-start gap-2.5 rounded-md border px-3 py-2 cursor-pointer hover:border-zinc-700 ${
              via === 'link' ? 'border-violet-600/60 bg-violet-950/20' : 'border-zinc-800'
            }`}>
              <input type="radio" name="invite-via" checked={via === 'link'} onChange={() => setVia('link')}
                className="mt-0.5 accent-violet-600" />
              <span className="block text-sm text-zinc-200">
                Don't email <span className="text-zinc-400">— I'll copy the link and send it myself</span>
              </span>
            </label>
          </div>
        </fieldset>
        {/* Only an email carries it: a link you pass on yourself has no room for a note. */}
        {via !== 'link' && (
          <div className="wide:col-span-2">
            <label className="block text-xs font-medium text-zinc-500 mb-1">
              Personal note <span className="text-zinc-600">(optional, goes in the email)</span>
            </label>
            <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2} maxLength={1000}
              placeholder="Welcome aboard — here's your access to our agents." className={`${field} resize-y`} />
          </div>
        )}
      </div>
      {err && <p className="text-xs text-red-400">{err}</p>}
      <div className="flex items-center gap-2 pt-1">
        <button type="submit" disabled={saving || !valid}
          className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50">
          {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : via === 'link' ? <Link2 className="h-3 w-3" /> : <Send className="h-3 w-3" />}
          {saving ? 'Inviting…' : via === 'link' ? 'Create invitation link' : 'Send invitation'}
        </button>
        <button type="button" onClick={onCancel}
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">
          Cancel
        </button>
      </div>
    </form>
  )
}

function ResultBanner({ result, ttlDays, onClose }: { result: InviteResult; ttlDays: number; onClose: () => void }) {
  const [copied, setCopied] = useState(false)
  const { email: outcome, invite, url } = result
  const failed = outcome.via !== 'link' && !outcome.sent
  const copy = () => {
    navigator.clipboard?.writeText(url)
      .then(() => { setCopied(true); setTimeout(() => setCopied(false), 2000) })
      .catch(() => {})
  }
  const tone = failed
    ? 'border-amber-800/50 bg-amber-950/20'
    : 'border-emerald-800/50 bg-emerald-950/20'

  return (
    <div className={`rounded-lg border px-4 py-3 space-y-2 ${tone}`} role="status">
      <div className="flex items-start gap-2">
        {failed
          ? <RefreshCw className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
          : <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />}
        <div className="min-w-0 flex-1 text-sm">
          {outcome.sent ? (
            <p className="text-zinc-200">
              Invitation emailed to <span className="text-white">{invite.email}</span>
              {outcome.from && <> from {outcome.from}</>}.
            </p>
          ) : failed ? (
            <p className="text-zinc-200">
              The invitation for <span className="text-white">{invite.email}</span> is ready, but the email didn't go
              out: <span className="text-amber-300">{outcome.error}</span> Send them this link yourself:
            </p>
          ) : (
            <p className="text-zinc-200">
              Invitation for <span className="text-white">{invite.email}</span> is ready. Send them this link — it works
              once and expires in {ttlDays} days.
            </p>
          )}
        </div>
        <button onClick={onClose} title="Dismiss" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="flex items-center gap-2">
        <input readOnly value={url} onFocus={(e) => e.currentTarget.select()}
          className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 font-mono text-xs text-zinc-300 focus:outline-none" />
        <button onClick={copy}
          className="flex shrink-0 items-center gap-1 rounded-md bg-zinc-800 px-2.5 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700">
          {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
          {copied ? 'Copied' : 'Copy link'}
        </button>
      </div>
      {localOnly(url) && (
        <p className="text-[11px] text-amber-400/90">
          This link points at this machine (localhost), so others can't open it. Open Flight Deck at its public
          address, or set FD_PUBLIC_URL, and make a new link.
        </p>
      )}
    </div>
  )
}
