import { useCallback, useEffect, useMemo, useState } from 'react'
import { Check, Copy, X, Trash2, Loader2, ClipboardList, RefreshCw } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { localOnly, PROVIDER_LABEL, PROVIDERS, type MailOptions, type Provider } from './InvitesAdmin'
import { matchesQuery } from './deckConfig'
import { FilterSelect, FilterSummary, SearchBox, Segmented } from './ListFilters'

// Users → Waitlist: people who applied from the sign-up page while
// self-registration is closed, each naming who recommended them. Approving
// creates the account with the password they chose and emails them, from the
// admin's own Gmail or Outlook, that they can sign in
// (captain_claw/flight_deck/admin_routes.py approve_waitlist).

export interface WaitlistEntry {
  id: string
  email: string
  display_name: string
  referrer_name: string
  referrer_email: string
  cohort: string
  status: 'pending' | 'approved' | 'rejected'
  created_at: string
  decided_at: string
}

type Status = WaitlistEntry['status']
type StatusFilter = Status | 'all'
type SortKey = 'newest' | 'oldest'
type Via = Provider | 'none'

interface ApproveResult {
  user: { email: string; display_name: string }
  email: { sent: boolean; via: Via; from: string; error: string; url: string }
}

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

export async function fetchWaitlist(status: StatusFilter = 'pending'): Promise<{
  entries: WaitlistEntry[]
  pending: number
  counts: Record<Status, number>
  mail: MailOptions
  sign_in_url: string
}> {
  return call(`/fd/admin/waitlist?status=${status}`)
}

function when(iso: string): string {
  if (!iso) return ''
  try { return new Date(iso).toLocaleString() } catch { return iso }
}

function firstAvailable(mail: MailOptions | null): Via {
  return PROVIDERS.find((p) => mail?.[p].available) ?? 'none'
}

export function WaitlistAdmin({ refreshTick, onPendingCount, onApproved, note, kioskDefault = false, kioskNote = null }: {
  refreshTick: number
  onPendingCount: (n: number) => void
  onApproved: () => void
  note: string | null
  // The deck's "New users start in kiosk mode": what the box below starts at.
  kioskDefault?: boolean
  kioskNote?: string | null
}) {
  const [filter, setFilter] = useState<StatusFilter>('pending')
  const [entries, setEntries] = useState<WaitlistEntry[] | null>(null)
  const [counts, setCounts] = useState<Record<Status, number> | null>(null)
  const [query, setQuery] = useState('')
  const [cohort, setCohort] = useState('all')
  const [sort, setSort] = useState<SortKey>('newest')
  const [busyId, setBusyId] = useState('')
  const [error, setError] = useState('')
  const [mail, setMail] = useState<MailOptions | null>(null)
  const [signInUrl, setSignInUrl] = useState('')
  const [via, setVia] = useState<Via>('none')
  const [kiosk, setKiosk] = useState(kioskDefault)
  useEffect(() => { setKiosk(kioskDefault) }, [kioskDefault])
  const [result, setResult] = useState<ApproveResult | null>(null)

  const load = useCallback(async () => {
    try {
      const data = await fetchWaitlist(filter)
      setEntries(data.entries)
      setCounts(data.counts)
      onPendingCount(data.pending)
      setMail(data.mail)
      setSignInUrl(data.sign_in_url)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [filter, onPendingCount])

  useEffect(() => { load() }, [load, refreshTick])
  // The options arrive with the list; keep the admin's pick while it still works.
  useEffect(() => { setVia((v) => (v === 'none' || !mail?.[v].available ? firstAvailable(mail) : v)) }, [mail])

  const act = async (entry: WaitlistEntry, fn: () => Promise<unknown>, after?: () => void) => {
    setBusyId(entry.id)
    setError('')
    try {
      await fn()
      after?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyId('')
      load()
    }
  }

  const approve = (e: WaitlistEntry) => {
    setResult(null)
    act(e, async () => {
      const r = await call<ApproveResult>(`/fd/admin/waitlist/${e.id}/approve`, {
        method: 'POST', body: JSON.stringify({ send_via: via, kiosk }),
      })
      setResult(r)
    }, onApproved)
  }
  const canEmail = PROVIDERS.some((p) => mail?.[p].available)
  const reject = (e: WaitlistEntry) => {
    if (!confirm(`Reject ${e.display_name || e.email}'s application? They can apply again later.`)) return
    act(e, () => call(`/fd/admin/waitlist/${e.id}/reject`, { method: 'POST' }))
  }
  const remove = (e: WaitlistEntry) => {
    if (!confirm(`Delete ${e.display_name || e.email}'s application from the list?`)) return
    act(e, () => call(`/fd/admin/waitlist/${e.id}`, { method: 'DELETE' }))
  }

  // Waitlist names applicants joined under, for the cohort filter.
  const cohorts = useMemo(
    () => [...new Set((entries ?? []).map((e) => e.cohort).filter(Boolean))].sort(),
    [entries],
  )
  // The menu shows only with two or more; a pick it no longer offers is dropped.
  useEffect(() => {
    if (entries && cohort !== 'all' && (cohorts.length < 2 || !cohorts.includes(cohort))) setCohort('all')
  }, [entries, cohorts, cohort])

  const shown = useMemo(() => {
    const list = (entries ?? []).filter((e) => (
      matchesQuery(query, e.display_name, e.email, e.referrer_name, e.referrer_email, e.cohort)
      && (cohort === 'all' || e.cohort === cohort)
    ))
    return list.sort((a, b) => (sort === 'oldest'
      ? a.created_at.localeCompare(b.created_at)
      : b.created_at.localeCompare(a.created_at)))
  }, [entries, query, cohort, sort])

  const filtered = query.trim() !== '' || cohort !== 'all'
  const clearFilters = () => { setQuery(''); setCohort('all') }

  return (
    <div className="space-y-2">
      <SearchBox value={query} onChange={setQuery} placeholder="Search applicant, email or who recommended them" />
      <div className="flex flex-wrap items-center gap-2">
        <Segmented
          label="Show"
          value={filter}
          onChange={setFilter}
          options={[
            { value: 'pending', label: 'Waiting', count: counts?.pending },
            { value: 'approved', label: 'Approved', count: counts?.approved },
            { value: 'rejected', label: 'Rejected', count: counts?.rejected },
            { value: 'all', label: 'All', count: counts ? counts.pending + counts.approved + counts.rejected : undefined },
          ]}
        />
        {cohorts.length > 1 && (
          <FilterSelect label="Waitlist" value={cohort} onChange={setCohort} active={cohort !== 'all'}>
            <option value="all">Every waitlist</option>
            {cohorts.map((c) => <option key={c} value={c}>{c}</option>)}
          </FilterSelect>
        )}
        <span className="flex-1" />
        <FilterSelect label="Sort" value={sort} onChange={(v) => setSort(v as SortKey)} active={false}>
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
        </FilterSelect>
      </div>

      <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-3">
        <p className="flex gap-2 text-xs text-zinc-500">
          <ClipboardList className="h-4 w-4 shrink-0 text-violet-400" />
          <span>
            People who applied while registration is closed, with who recommended them. Approving creates
            their account with the password they chose, so they can sign in right away, and emails them
            that they're in.
          </span>
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor="waitlist-via" className="text-xs text-zinc-500">When I approve someone, email them</label>
          <select
            id="waitlist-via"
            value={via}
            onChange={(ev) => setVia(ev.target.value as Via)}
            className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
          >
            {PROVIDERS.map((p) => {
              const opt = mail?.[p]
              return (
                <option key={p} value={p} disabled={!opt?.available} title={opt?.reason || undefined}>
                  from {PROVIDER_LABEL[p]}{opt?.from ? ` · ${opt.from}` : ''}{opt?.available ? '' : ' (not connected)'}
                </option>
              )
            })}
            <option value="none">— don't email, I'll tell them</option>
          </select>
        </div>
        <label className="flex items-center gap-2 text-xs text-zinc-400">
          <input type="checkbox" checked={kiosk} onChange={(ev) => setKiosk(ev.target.checked)} className="accent-violet-500" />
          Approve them into kiosk mode <span className="text-zinc-500">— simple chat only</span>
        </label>
        {kioskNote && <p className="text-[11px] text-zinc-500">{kioskNote}</p>}
        {mail && !canEmail && (
          <p className="text-[11px] text-zinc-500">
            To email approvals, connect your Google or Microsoft account with permission to send mail
            (Connections).
          </p>
        )}
        {via !== 'none' && localOnly(signInUrl) && (
          <p className="text-[11px] text-amber-400/90">
            The email's sign-in link would be {signInUrl}, which only works on this machine. Open Flight Deck
            at its public address, or set FD_PUBLIC_URL, before approving.
          </p>
        )}
        {note && <p className="text-xs text-amber-500/90">{note}</p>}
      </div>

      {result && <ApprovedBanner result={result} onClose={() => setResult(null)} />}

      {error && (
        <div className="rounded-md border border-red-900/50 bg-red-950/20 px-4 py-2 text-sm text-red-400">{error}</div>
      )}

      {entries === null && !error && (
        <div className="flex justify-center py-6">
          <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
        </div>
      )}

      {entries && filtered && (
        <FilterSummary shown={shown.length} total={entries.length} noun="applications" onClear={clearFilters} />
      )}

      {shown.map((e) => (
        <div
          key={e.id}
          className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-3"
        >
          <div className="min-w-0 flex-1 space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-zinc-100 truncate">{e.display_name || e.email}</span>
              {e.display_name && <span className="text-xs text-zinc-400 truncate">{e.email}</span>}
              {e.cohort && (
                <span className="rounded border border-violet-800/50 bg-violet-950/40 px-1.5 py-0.5 text-[10px] font-medium text-violet-300">
                  {e.cohort}
                </span>
              )}
              {e.status !== 'pending' && (
                <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                  e.status === 'approved' ? 'bg-emerald-950/60 text-emerald-400' : 'bg-zinc-800 text-zinc-400'
                }`}>
                  {e.status === 'approved' ? 'Approved' : 'Rejected'}
                </span>
              )}
            </div>
            <div className="text-xs text-zinc-400">
              Recommended by <span className="text-zinc-200">{e.referrer_name}</span>
              {e.referrer_email && <span className="text-zinc-500"> · {e.referrer_email}</span>}
            </div>
            <div className="text-[11px] text-zinc-600">
              Applied {when(e.created_at)}
              {e.decided_at && <> · {e.status} {when(e.decided_at)}</>}
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            {busyId === e.id && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
            {e.status === 'pending' ? (
              <>
                <button
                  onClick={() => approve(e)}
                  disabled={!!busyId}
                  title={via === 'none'
                    ? 'Create their account — no email'
                    : `Create their account and email them from ${mail?.[via].from || PROVIDER_LABEL[via]}`}
                  className="flex items-center gap-1 rounded-md bg-emerald-600 px-2.5 py-1 text-xs text-white hover:bg-emerald-500 disabled:opacity-50"
                >
                  <Check className="h-3 w-3" /> {via === 'none' ? 'Approve' : 'Approve & email'}
                </button>
                <button
                  onClick={() => reject(e)}
                  disabled={!!busyId}
                  className="flex items-center gap-1 rounded-md bg-zinc-800 px-2.5 py-1 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-50"
                >
                  <X className="h-3 w-3" /> Reject
                </button>
              </>
            ) : (
              <button
                onClick={() => remove(e)}
                disabled={!!busyId}
                title="Delete application"
                className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-red-400 disabled:opacity-50"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>
      ))}

      {entries && shown.length === 0 && (
        <p className="text-center text-sm text-zinc-500 py-8">
          {filtered ? 'No applications match these filters.'
            : filter === 'pending' ? 'Nobody is waiting for approval.'
              : filter === 'approved' ? 'Nobody has been approved yet.'
                : filter === 'rejected' ? 'No rejected applications.'
                  : 'No applications yet.'}
        </p>
      )}
    </div>
  )
}

function ApprovedBanner({ result, onClose }: { result: ApproveResult; onClose: () => void }) {
  const [copied, setCopied] = useState(false)
  const { user, email: outcome } = result
  const who = user.display_name || user.email
  const failed = outcome.via !== 'none' && !outcome.sent
  const copy = () => {
    navigator.clipboard?.writeText(outcome.url)
      .then(() => { setCopied(true); setTimeout(() => setCopied(false), 2000) })
      .catch(() => {})
  }

  return (
    <div
      className={`rounded-lg border px-4 py-3 space-y-2 ${
        failed ? 'border-amber-800/50 bg-amber-950/20' : 'border-emerald-800/50 bg-emerald-950/20'
      }`}
      role="status"
    >
      <div className="flex items-start gap-2">
        {failed
          ? <RefreshCw className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
          : <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />}
        <p className="min-w-0 flex-1 text-sm text-zinc-200">
          {outcome.sent ? (
            <>
              Approved <span className="text-white">{who}</span> and emailed {user.email}
              {outcome.from && <> from {outcome.from}</>} that they can sign in.
            </>
          ) : failed ? (
            <>
              Approved <span className="text-white">{who}</span>, but the email didn't go
              out: <span className="text-amber-300">{outcome.error}</span> Let them know they can sign in here:
            </>
          ) : (
            <>
              Approved <span className="text-white">{who}</span> — no email sent. Let them know they can sign in here:
            </>
          )}
        </p>
        <button onClick={onClose} title="Dismiss" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      {!outcome.sent && (
        <div className="flex items-center gap-2">
          <input readOnly value={outcome.url} onFocus={(e) => e.currentTarget.select()}
            className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 font-mono text-xs text-zinc-300 focus:outline-none" />
          <button onClick={copy}
            className="flex shrink-0 items-center gap-1 rounded-md bg-zinc-800 px-2.5 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700">
            {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
            {copied ? 'Copied' : 'Copy address'}
          </button>
        </div>
      )}
    </div>
  )
}
