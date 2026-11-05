import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity, Mail, MailCheck, MessageCircle, Check, Trash2, RefreshCw, Send, ChevronDown, Reply,
  UserX, Moon, Loader2, NotebookPen, Plug,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { useConsoleStore } from '../../stores/consoleStore'
import { PROVIDER_LABEL, PROVIDERS, type Provider } from '../admin/InvitesAdmin'
import { SearchBox, FilterSelect } from '../admin/ListFilters'
import { matchesQuery } from '../admin/deckConfig'
import {
  Page, PageHeader, Card, StatTile, Segmented, Button, Notice, Pill, EmptyState, Sheet, Dialog, Field,
  inputCls, type Tone,
} from './parts'
import { timeAgo, fmtNumber } from './format'
import {
  fetchActivity, fetchMailbox, fetchDraft, sendEmail, noteContact, deleteContact,
  type ActivityReport, type ActivityUser, type ActivityDay, type Contact, type Mailbox,
  type MessageDraft, type Template, type ActivityStatus,
} from './activityApi'

// Activity: who on the deck is using it and whether you've written to them.
// "Typed" is the messages people sent their agents, from any channel; their
// agents' own work (scheduled tasks too) shows apart, so it never passes for
// them. "You wrote" is the deck's record — its emails, and what you marked —
// and your own Gmail, read when the page opens (user_activity.py).

type WindowDays = '14' | '30' | '90'
type StatusFilter = 'all' | ActivityStatus
type Sort = 'active' | 'typed' | 'wrote' | 'name' | 'joined'

const SORT_LABEL: Record<Sort, string> = {
  active: 'Last typed',
  typed: 'Most typed',
  wrote: 'Longest since you wrote',
  name: 'Name',
  joined: 'Newest account',
}

const STATUS: Record<ActivityStatus, { label: string; tone: Tone }> = {
  active: { label: 'Active', tone: 'emerald' },
  quiet: { label: 'Quiet', tone: 'amber' },
  never: { label: 'Never started', tone: 'zinc' },
}

const CHANNELS = ['WhatsApp', 'Phone call', 'In person', 'Email', 'Telegram', 'Other'] as const

const DAY_MS = 86_400_000

function dayMinus(day: string, n: number): string {
  return new Date(Date.parse(`${day}T00:00:00Z`) - n * DAY_MS).toISOString().slice(0, 10)
}

function shortDay(day: string): string {
  return new Date(`${day}T00:00:00Z`).toLocaleDateString(undefined, { day: 'numeric', month: 'short', timeZone: 'UTC' })
}

/** "today", "yesterday", "5 days ago", or the date — for a UTC day. */
function dayAgo(day: string, today: string): string {
  if (!day) return ''
  const n = Math.round((Date.parse(`${today}T00:00:00Z`) - Date.parse(`${day}T00:00:00Z`)) / DAY_MS)
  if (n <= 0) return 'today'
  if (n === 1) return 'yesterday'
  if (n < 45) return `${n} days ago`
  return new Date(`${day}T00:00:00Z`).toLocaleDateString(undefined, { timeZone: 'UTC' })
}

/** The same for a moment: "3 h ago" within a day. */
function whenAgo(iso: string): string {
  if (!iso) return ''
  const ms = Date.now() - Date.parse(iso)
  if (ms < DAY_MS) return timeAgo(iso)
  const n = Math.floor(ms / DAY_MS)
  if (n === 1) return 'yesterday'
  if (n < 45) return `${n} days ago`
  return new Date(iso).toLocaleDateString()
}

function nameOf(u: ActivityUser): string {
  return u.display_name || u.email.split('@')[0] || u.email
}

function initials(s: string): string {
  const parts = s.trim().split(/[\s@._-]+/).filter(Boolean)
  return ((parts[0]?.[0] || '') + (parts[1]?.[0] || '')).toUpperCase() || '?'
}

function viaLabel(channel: string): string {
  return channel === 'google' ? 'Gmail' : channel === 'microsoft' ? 'Outlook' : channel
}

/** "Feedback request · Gmail", "Gmail" for a message, the channel for a note. */
function contactHow(c: Contact): string {
  if (c.kind === 'feedback') return `Feedback request · ${viaLabel(c.channel)}`
  return viaLabel(c.channel)
}

// The last time you wrote: the deck's record or your Gmail, whichever is newer.
interface Wrote { at: string; how: string; text: string }

function lastWrote(u: ActivityUser, mb: Mailbox | null): Wrote | null {
  const c = u.contacts[0]
  const deck: Wrote | null = c
    ? { at: c.created_at, how: contactHow(c), text: c.kind === 'noted' ? c.note : c.subject }
    : null
  const sent = mb?.available ? mb.users[u.id]?.sent : null
  const mail: Wrote | null = sent ? { at: sent.at, how: 'Gmail', text: sent.subject } : null
  if (!deck || !mail) return deck || mail
  // The deck's own emails are in Gmail's Sent too: the deck says what they were.
  const gap = Date.parse(mail.at) - Date.parse(deck.at)
  return gap > 120_000 ? mail : deck
}

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

export function ActivityConsole() {
  const go = useConsoleStore((s) => s.go)
  const [windowDays, setWindowDays] = useState<WindowDays>('30')
  const [report, setReport] = useState<ActivityReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [mailbox, setMailbox] = useState<Mailbox | null>(null)
  const [mailboxLoading, setMailboxLoading] = useState(true)
  const [mailboxError, setMailboxError] = useState('')
  const [status, setStatus] = useState<StatusFilter>('all')
  const [notWritten, setNotWritten] = useState(false)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<Sort>('active')
  const [open, setOpen] = useState<string | null>(null)
  const [writing, setWriting] = useState<ActivityUser | null>(null)
  const [noting, setNoting] = useState<ActivityUser | null>(null)
  const [flash, setFlash] = useState('')

  // Each tick reads again; the first mailbox read may take the server's copy.
  const [tick, setTick] = useState(0)
  const [mailboxTick, setMailboxTick] = useState(0)

  useEffect(() => {
    let live = true
    fetchActivity(Number(windowDays))
      .then((r) => { if (live) { setReport(r); setError('') } })
      .catch((e) => { if (live) setError(errText(e)) })
      .finally(() => { if (live) setLoading(false) })
    return () => { live = false }
  }, [windowDays, tick])

  useEffect(() => {
    let live = true
    fetchMailbox(mailboxTick > 0)
      .then((m) => { if (live) { setMailbox(m); setMailboxError('') } })
      .catch((e) => { if (live) setMailboxError(errText(e)) })
      .finally(() => { if (live) setMailboxLoading(false) })
    return () => { live = false }
  }, [mailboxTick])

  const reload = useCallback(() => { setLoading(true); setTick((t) => t + 1) }, [])
  const recheckMailbox = useCallback(() => { setMailboxLoading(true); setMailboxTick((t) => t + 1) }, [])
  const changeWindow = (w: WindowDays) => { if (w !== windowDays) { setLoading(true); setWindowDays(w) } }

  useEffect(() => {
    if (!flash) return
    const t = setTimeout(() => setFlash(''), 6000)
    return () => clearTimeout(t)
  }, [flash])

  const users = useMemo(() => report?.users ?? [], [report])
  const wrote = useMemo(() => new Map(users.map((u) => [u.id, lastWrote(u, mailbox)])), [users, mailbox])

  const counts = useMemo(() => ({
    active: users.filter((u) => u.status === 'active').length,
    quiet: users.filter((u) => u.status === 'quiet').length,
    never: users.filter((u) => u.status === 'never').length,
    notWritten: users.filter((u) => !wrote.get(u.id)).length,
  }), [users, wrote])

  const shown = useMemo(() => {
    const list = users.filter((u) =>
      (status === 'all' || u.status === status)
      && (!notWritten || !wrote.get(u.id))
      && matchesQuery(query, u.display_name, u.email))
    const wroteAt = (u: ActivityUser) => wrote.get(u.id)?.at ?? ''
    const lastSeen = (u: ActivityUser) => u.typed.last_day || ''
    return list.sort((a, b) => {
      if (sort === 'typed') return b.typed.messages - a.typed.messages || lastSeen(b).localeCompare(lastSeen(a))
      if (sort === 'wrote') return wroteAt(a).localeCompare(wroteAt(b)) || lastSeen(b).localeCompare(lastSeen(a))
      if (sort === 'name') return nameOf(a).localeCompare(nameOf(b), undefined, { sensitivity: 'base' })
      if (sort === 'joined') return b.created_at.localeCompare(a.created_at)
      // Last typed; those who never typed after, their agents' work next.
      return lastSeen(b).localeCompare(lastSeen(a))
        || (b.agents.last_day || '').localeCompare(a.agents.last_day || '')
        || nameOf(a).localeCompare(nameOf(b))
    })
  }, [users, status, notWritten, query, sort, wrote])

  const filtered = status !== 'all' || notWritten || query.trim() !== ''
  const clearFilters = () => { setStatus('all'); setNotWritten(false); setQuery('') }

  const afterChange = (message: string) => {
    setFlash(message)
    reload()
  }

  return (
    <Page>
      <PageHeader
        icon={Activity}
        title="Activity"
        description={<>
          Who is using the deck, and whether you've written to them. <em>Typed</em> counts the messages someone sent their
          agents on any channel; their agents' own work, scheduled tasks included, shows apart. Days are UTC.
        </>}
        actions={<>
          <Segmented
            value={windowDays}
            onChange={changeWindow}
            label="Window"
            options={[{ value: '14', label: '14 days' }, { value: '30', label: '30 days' }, { value: '90', label: '90 days' }]}
          />
          <Button icon={RefreshCw} busy={loading} onClick={() => { reload(); recheckMailbox() }}>Refresh</Button>
        </>}
      />

      {flash && <Notice tone="emerald">{flash}</Notice>}
      {error && <Notice tone="red">{error}</Notice>}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile icon={Activity} tone="emerald" label="Active this week" value={counts.active} loading={loading && !report}
          sub={`typed in ${report?.active_days ?? 7} days`} onClick={() => setStatus(status === 'active' ? 'all' : 'active')} />
        <StatTile icon={Moon} tone="amber" label="Quiet" value={counts.quiet} loading={loading && !report}
          sub="none this week" onClick={() => setStatus(status === 'quiet' ? 'all' : 'quiet')} />
        <StatTile icon={UserX} label="Never started" value={counts.never} loading={loading && !report}
          sub="no message typed yet" onClick={() => setStatus(status === 'never' ? 'all' : 'never')} />
        <StatTile icon={Mail} tone="violet" label="Not written to" value={counts.notWritten} loading={(loading && !report) || mailboxLoading}
          sub={mailbox?.available ? 'not here, not in Gmail' : 'nothing from you yet'}
          onClick={() => setNotWritten(!notWritten)} />
      </div>

      <MailboxBar
        mailbox={mailbox}
        loading={mailboxLoading}
        error={mailboxError}
        onRefresh={recheckMailbox}
        onConnect={() => go('connections', { connection: 'google' })}
      />

      <Card bodyClassName="">
        <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800/80 px-4 py-3">
          <div className="flex min-w-0 basis-full wide:basis-0 wide:flex-1">
            <SearchBox value={query} onChange={setQuery} placeholder="Search people" />
          </div>
          <FilterSelect value={status} onChange={(v) => setStatus(v as StatusFilter)} label="Status" active={status !== 'all'}>
            <option value="all">Everyone</option>
            <option value="active">Active ({counts.active})</option>
            <option value="quiet">Quiet ({counts.quiet})</option>
            <option value="never">Never started ({counts.never})</option>
          </FilterSelect>
          <label className={`flex cursor-pointer items-center gap-1.5 rounded-md border px-2 py-1 text-xs ${
            notWritten ? 'border-violet-600/60 text-zinc-100' : 'border-zinc-700 text-zinc-400'
          }`}>
            <input type="checkbox" checked={notWritten} onChange={(e) => setNotWritten(e.target.checked)} className="accent-violet-600" />
            Not written to
          </label>
          <FilterSelect value={sort} onChange={(v) => setSort(v as Sort)} label="Sort" active={false}>
            {(Object.keys(SORT_LABEL) as Sort[]).map((s) => <option key={s} value={s}>{SORT_LABEL[s]}</option>)}
          </FilterSelect>
        </div>

        {filtered && report && (
          <div className="flex items-center gap-2 border-b border-zinc-800/80 px-4 py-2 text-xs text-zinc-500">
            <span>{shown.length} of {users.length} people</span>
            <button onClick={clearFilters} className="text-violet-600 underline underline-offset-2 hover:no-underline dark:text-violet-400">
              Clear filters
            </button>
          </div>
        )}

        {loading && !report ? (
          <div className="flex justify-center py-12"><Loader2 className="h-5 w-5 animate-spin text-zinc-600" /></div>
        ) : users.length === 0 ? (
          <div className="p-5"><EmptyState icon={Activity} title="No one else on the deck yet">Invite people on the Users page.</EmptyState></div>
        ) : shown.length === 0 ? (
          <p className="px-5 py-10 text-center text-sm text-zinc-500">No one matches these filters.</p>
        ) : (
          <ul className="divide-y divide-zinc-800/70">
            {report && shown.map((u) => (
              <PersonRow
                key={u.id}
                user={u}
                report={report}
                wrote={wrote.get(u.id) ?? null}
                mailbox={mailbox}
                open={open === u.id}
                onToggle={() => setOpen(open === u.id ? null : u.id)}
                onWrite={() => setWriting(u)}
                onNote={() => setNoting(u)}
                onDeleted={() => afterChange('Note removed.')}
              />
            ))}
          </ul>
        )}
      </Card>

      {writing && (
        <MessageSheet
          user={writing}
          onClose={() => setWriting(null)}
          onSent={(text) => { setWriting(null); afterChange(text) }}
        />
      )}
      {noting && (
        <NoteDialog
          user={noting}
          onClose={() => setNoting(null)}
          onSaved={() => { const n = nameOf(noting); setNoting(null); afterChange(`Marked: you wrote to ${n}.`) }}
        />
      )}
    </Page>
  )
}

// ── Your Gmail ──

function MailboxBar({ mailbox, loading, error, onRefresh, onConnect }: {
  mailbox: Mailbox | null
  loading: boolean
  error: string
  onRefresh: () => void
  onConnect: () => void
}) {
  let text: React.ReactNode
  if (loading) text = 'Reading your Gmail…'
  else if (error) text = `Your Gmail couldn't be read: ${error}`
  else if (mailbox?.available) {
    text = <>
      Your Gmail{mailbox.address ? ` (${mailbox.address})` : ''} is counted too, checked {timeAgo(mailbox.checked_at)}
      {mailbox.partial ? '. Some people couldn’t be read this time.' : '.'}
    </>
  } else text = mailbox?.reason || 'Your Gmail isn’t read.'
  const off = !loading && !error && mailbox && !mailbox.available
  return (
    <div className={`flex flex-wrap items-center gap-3 rounded-2xl border px-4 py-3 text-sm ${
      off ? 'border-amber-500/30 bg-amber-500/5' : 'border-zinc-800 bg-zinc-900/40'
    }`}>
      {loading ? <Loader2 className="h-4 w-4 shrink-0 animate-spin text-zinc-500" /> : <MailCheck className={`h-4 w-4 shrink-0 ${off ? 'text-amber-600 dark:text-amber-400' : 'text-violet-600 dark:text-violet-400'}`} />}
      <div className="min-w-[12rem] flex-1">
        <div className="text-zinc-300">{text}</div>
        <div className="mt-0.5 text-[11px] text-zinc-500">
          Only the date and subject of the newest email each way, between you and each person — read when this page opens, never stored.
        </div>
      </div>
      {off ? (
        <Button icon={Plug} onClick={onConnect}>Connections</Button>
      ) : (
        <Button variant="ghost" icon={RefreshCw} busy={loading} onClick={onRefresh}>Check again</Button>
      )}
    </div>
  )
}

// ── One person ──

function PersonRow({ user: u, report, wrote, mailbox, open, onToggle, onWrite, onNote, onDeleted }: {
  user: ActivityUser
  report: ActivityReport
  wrote: Wrote | null
  mailbox: Mailbox | null
  open: boolean
  onToggle: () => void
  onWrite: () => void
  onNote: () => void
  onDeleted: () => void
}) {
  const st = STATUS[u.status]
  const received = mailbox?.available ? mailbox.users[u.id]?.received ?? null : null
  const replied = !!(received && wrote && Date.parse(received.at) > Date.parse(wrote.at))
  const typedSince = !!(wrote && u.typed.last_day && u.typed.last_day > wrote.at.slice(0, 10))
  const name = nameOf(u)

  return (
    <li className="px-4 py-4 wide:px-5">
      <div className="flex flex-col gap-4 lg:grid lg:grid-cols-[13rem_minmax(0,1fr)_15rem] lg:items-start lg:gap-5">
        {/* Who */}
        <div className="flex min-w-0 items-start gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-violet-500/15 text-xs font-semibold text-violet-700 dark:text-violet-300">
            {initials(name)}
          </span>
          <div className="min-w-0">
            <div className="truncate text-sm font-medium text-zinc-100" title={name}>{name}</div>
            <div className="truncate text-xs text-zinc-500" title={u.email}>{u.email}</div>
            <div className="mt-1.5 flex flex-wrap items-center gap-1">
              <Pill tone={st.tone}>{st.label}</Pill>
              {u.role === 'admin' && <Pill tone="violet">Admin</Pill>}
            </div>
          </div>
        </div>

        {/* What they did */}
        <div className="min-w-0">
          <Strip days={u.days} today={report.today} window={report.days} />
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-zinc-500">
            {u.typed.last_day ? (
              <span className="text-zinc-300">Typed {dayAgo(u.typed.last_day, report.today)}</span>
            ) : (
              <span>Never typed</span>
            )}
            {u.typed.messages > 0
              ? <span>{u.typed.messages} message{u.typed.messages === 1 ? '' : 's'} on {u.typed.days} day{u.typed.days === 1 ? '' : 's'}</span>
              : u.typed.last_day && <span>nothing in {report.days} days</span>}
            {u.agents.days > 0 && <span>agents worked {u.agents.days} day{u.agents.days === 1 ? '' : 's'}</span>}
            <span>{u.last_login_at ? `seen ${whenAgo(u.last_login_at)}` : 'never signed in'}</span>
            <button
              onClick={onToggle}
              aria-expanded={open}
              aria-label={`Details for ${name}`}
              className="inline-flex items-center gap-0.5 text-violet-600 hover:underline dark:text-violet-400"
            >
              Details <ChevronDown className={`h-3 w-3 transition-transform ${open ? 'rotate-180' : ''}`} />
            </button>
          </div>
        </div>

        {/* You wrote */}
        <div className="min-w-0 space-y-2">
          {wrote ? (
            <div className="min-w-0 text-xs">
              <div className="flex flex-wrap items-center gap-1.5 text-zinc-300">
                <Check className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                You wrote {whenAgo(wrote.at)}
              </div>
              <div className="mt-0.5 truncate text-zinc-500" title={wrote.text || wrote.how}>
                {wrote.how}{wrote.text ? ` · ${wrote.text}` : ''}
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-xs font-medium text-amber-700 dark:text-amber-300">
              <Mail className="h-3.5 w-3.5" /> Not written to yet
            </div>
          )}
          {(replied || typedSince) && (
            <div className="flex flex-wrap gap-1">
              {replied && received && <Pill tone="emerald" icon={Reply} title={received.subject}>Replied {whenAgo(received.at)}</Pill>}
              {typedSince && <Pill tone="sky" title="They typed a message on a later day">Typed since</Pill>}
            </div>
          )}
          <div className="flex flex-wrap items-center gap-1.5">
            <Button variant="primary" icon={Send} onClick={onWrite} className="!px-2.5 !py-1 !text-xs">Write</Button>
            <Button icon={NotebookPen} onClick={onNote} className="!px-2.5 !py-1 !text-xs" title="You got in touch another way: WhatsApp, a call…">Mark as written</Button>
          </div>
        </div>
      </div>

      {open && <History user={u} report={report} mailbox={mailbox} onDeleted={onDeleted} />}
    </li>
  )
}

/** One bar a day: how much they typed (violet), or their agents' work only (sky). */
function Strip({ days, today, window }: { days: ActivityDay[]; today: string; window: number }) {
  const byDay = new Map(days.map((d) => [d.day, d]))
  const most = Math.max(1, ...days.map((d) => d.messages))
  const cells = Array.from({ length: window }, (_, i) => dayMinus(today, window - 1 - i))
  return (
    <div>
      <div className={`flex h-7 items-end ${window > 45 ? 'gap-0' : 'gap-px'}`} role="img"
        aria-label={`${days.filter((d) => d.messages).length} days typed in the last ${window} days`}>
        {cells.map((day) => {
          const d = byDay.get(day)
          const typed = d?.messages ?? 0
          const agents = d?.tokens ?? 0
          const title = `${shortDay(day)}: ${typed ? `${typed} message${typed === 1 ? '' : 's'}` : 'nothing typed'}${agents ? ` · agents ${fmtNumber(agents)} tokens` : ''}`
          const cls = typed ? 'bg-violet-500' : agents ? 'bg-sky-500/50' : 'bg-zinc-800'
          const height = typed ? `${35 + 65 * (typed / most)}%` : agents ? '22%' : '8%'
          return <span key={day} title={title} className={`min-w-0 flex-1 rounded-[2px] ${cls}`} style={{ height }} />
        })}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-zinc-600">
        <span>{shortDay(cells[0])}</span>
        <span>today</span>
      </div>
    </div>
  )
}

function History({ user: u, report, mailbox, onDeleted }: {
  user: ActivityUser
  report: ActivityReport
  mailbox: Mailbox | null
  onDeleted: () => void
}) {
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const mail = mailbox?.available ? mailbox.users[u.id] : undefined

  const remove = async (c: Contact) => {
    setBusy(c.id); setError('')
    try {
      await deleteContact(c.id)
      onDeleted()
    } catch (e) {
      setError(errText(e))
      setBusy('')
    }
  }

  return (
    <div className="mt-4 grid gap-4 rounded-xl border border-zinc-800 bg-zinc-950/50 p-4 text-xs lg:grid-cols-2">
      <div className="space-y-1.5">
        <div className="font-medium text-zinc-300">What they did</div>
        <Line label="Joined">{u.created_at ? new Date(u.created_at).toLocaleDateString() : '—'}</Line>
        <Line label="First typed">{u.typed.first_day ? shortDay(u.typed.first_day) : 'never'}</Line>
        <Line label="All time">{u.typed.messages_all} message{u.typed.messages_all === 1 ? '' : 's'} on {u.typed.days_all} day{u.typed.days_all === 1 ? '' : 's'}</Line>
        <Line label="This week">{u.typed.messages_week} message{u.typed.messages_week === 1 ? '' : 's'}</Line>
        <Line label="Agents">{u.agents.last_day ? `last worked ${dayAgo(u.agents.last_day, report.today)}` : 'no work on record'}{u.agents.tokens ? ` · ${fmtNumber(u.agents.tokens)} tokens in ${report.days} days` : ''}</Line>
        <Line label="Last seen">{u.last_login_at ? new Date(u.last_login_at).toLocaleString() : 'never signed in'}</Line>
      </div>
      <div className="space-y-1.5">
        <div className="font-medium text-zinc-300">You and them</div>
        {mail && (
          <>
            <Line label="Gmail, you">{mail.sent ? `${new Date(mail.sent.at).toLocaleDateString()} · ${mail.sent.subject || '(no subject)'}` : 'no email to them'}</Line>
            <Line label="Gmail, them">{mail.received ? `${new Date(mail.received.at).toLocaleDateString()} · ${mail.received.subject || '(no subject)'}` : 'no email from them'}</Line>
          </>
        )}
        {u.contacts.length === 0 && <p className="text-zinc-500">Nothing on record from this deck.</p>}
        <ul className="space-y-1.5">
          {u.contacts.map((c) => (
            <li key={c.id} className="flex items-start gap-2">
              {c.kind === 'noted'
                ? <MessageCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />
                : <Mail className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />}
              <div className="min-w-0 flex-1">
                <div className="text-zinc-300">
                  {c.kind === 'message' ? `Email · ${viaLabel(c.channel)}` : contactHow(c)}
                  <span className="text-zinc-500"> · {new Date(c.created_at).toLocaleDateString()}{c.admin_name ? ` · ${c.admin_name}` : ''}</span>
                </div>
                {(c.subject || c.note) && <div className="truncate text-zinc-500" title={c.subject || c.note}>{c.subject || c.note}</div>}
              </div>
              {c.kind === 'noted' && (
                <button
                  onClick={() => remove(c)}
                  disabled={busy === c.id}
                  title="Remove this note"
                  aria-label="Remove this note"
                  className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-red-600 disabled:opacity-50 dark:hover:text-red-400"
                >
                  {busy === c.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                </button>
              )}
            </li>
          ))}
        </ul>
        {u.contacts_total > u.contacts.length && <p className="text-zinc-500">…and {u.contacts_total - u.contacts.length} earlier.</p>}
        {error && <p className="text-red-600 dark:text-red-400">{error}</p>}
      </div>
    </div>
  )
}

function Line({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="w-24 shrink-0 text-zinc-500">{label}</span>
      <span className="min-w-0 flex-1 break-words text-zinc-300">{children}</span>
    </div>
  )
}

// ── Writing ──

function MessageSheet({ user, onClose, onSent }: {
  user: ActivityUser
  onClose: () => void
  onSent: (text: string) => void
}) {
  const [template, setTemplate] = useState<Template>('message')
  const [drafts, setDrafts] = useState<Partial<Record<Template, MessageDraft>>>({})
  const [edits, setEdits] = useState<Partial<Record<Template, { subject: string; body: string }>>>({})
  const [via, setVia] = useState<Provider | ''>('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const draft = drafts[template]
  const edit = edits[template]

  useEffect(() => {
    if (drafts[template]) return
    let live = true
    fetchDraft(user.id, template)
      .then((d) => {
        if (!live) return
        setDrafts((m) => ({ ...m, [template]: d }))
        setEdits((m) => ({ ...m, [template]: { subject: d.subject, body: d.body } }))
        setVia((v) => v || (PROVIDERS.find((p) => d.mail[p].available) ?? ''))
      })
      .catch((e) => { if (live) setError(errText(e)) })
    return () => { live = false }
  }, [template, user.id, drafts])

  const change = (field: 'subject' | 'body', value: string) =>
    setEdits((m) => ({ ...m, [template]: { subject: m[template]?.subject ?? '', body: m[template]?.body ?? '', [field]: value } }))

  const canSend = !!draft && !!via && !sending && !!edit?.subject.trim() && !!edit?.body.trim()
  const available = draft ? PROVIDERS.filter((p) => draft.mail[p].available) : []

  const send = async () => {
    if (!canSend || !edit || !via) return
    setSending(true); setError('')
    try {
      const r = await sendEmail(user.id, template, edit.subject, edit.body, via)
      if (r.ok) { onSent(`Sent to ${user.email}${r.email.from ? ` from ${r.email.from}` : ''}.`); return }
      setError(r.email.error || 'The email did not go out.')
    } catch (e) {
      setError(errText(e))
    }
    setSending(false)
  }

  return (
    <Sheet
      title={`Write to ${nameOf(user)}`}
      subtitle={`${user.email} · from your own mailbox, so their reply comes straight to you`}
      icon={Send}
      wide
      onClose={onClose}
      footer={<>
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button variant="primary" icon={Send} busy={sending} disabled={!canSend} onClick={send}>Send email</Button>
      </>}
    >
      <div className="space-y-4">
        <Segmented
          value={template}
          onChange={setTemplate}
          label="Kind of email"
          options={[{ value: 'message', label: 'Message' }, { value: 'feedback', label: 'Ask for feedback' }]}
        />
        {template === 'feedback' && draft?.asked_at && (
          <p className="text-xs text-zinc-500">You last asked them for feedback on {new Date(draft.asked_at).toLocaleDateString()}.</p>
        )}
        {!draft && !error && <Loader2 className="h-5 w-5 animate-spin text-zinc-600" />}
        {draft && edit && (
          <>
            {available.length > 0 ? (
              <Field label="Send from">
                <select value={via} onChange={(e) => setVia(e.target.value as Provider)} className={inputCls}>
                  {available.map((p) => (
                    <option key={p} value={p}>{PROVIDER_LABEL[p]}{draft.mail[p].from ? ` · ${draft.mail[p].from}` : ''}</option>
                  ))}
                </select>
              </Field>
            ) : (
              <Notice tone="amber">
                {PROVIDERS.map((p) => <p key={p}>{draft.mail[p].reason}</p>)}
              </Notice>
            )}
            <Field label="Subject">
              <input type="text" value={edit.subject} onChange={(e) => change('subject', e.target.value)} maxLength={200}
                placeholder="What it's about" className={inputCls} />
            </Field>
            <Field label="Message">
              <textarea value={edit.body} onChange={(e) => change('body', e.target.value)} rows={14} maxLength={5000}
                className={`${inputCls} font-[inherit] leading-relaxed`} />
            </Field>
          </>
        )}
        {error && <Notice tone="red">{error}</Notice>}
      </div>
    </Sheet>
  )
}

function NoteDialog({ user, onClose, onSaved }: {
  user: ActivityUser
  onClose: () => void
  onSaved: () => void
}) {
  const [channel, setChannel] = useState<string>('WhatsApp')
  const [other, setOther] = useState('')
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const me = useAuthStore((s) => s.user?.display_name || s.user?.email || 'you')
  const how = channel === 'Other' ? other.trim() : channel

  const save = async () => {
    if (!how) return
    setSaving(true); setError('')
    try {
      await noteContact(user.id, how, note)
      onSaved()
    } catch (e) {
      setError(errText(e))
      setSaving(false)
    }
  }

  return (
    <Dialog
      title={`Mark as written to ${nameOf(user)}`}
      icon={NotebookPen}
      onClose={onClose}
      footer={<>
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button variant="primary" icon={Check} busy={saving} disabled={!how} onClick={save}>Mark as written</Button>
      </>}
    >
      <div className="space-y-4">
        <p className="text-sm text-zinc-400">
          For when you got in touch another way. It's kept on the deck as from {me}, today, and every admin sees it.
        </p>
        <Field label="How">
          <div className="flex flex-wrap gap-1.5">
            {CHANNELS.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => setChannel(c)}
                className={`rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors ${
                  c === channel
                    ? 'border-violet-500/50 bg-violet-500/15 text-violet-700 dark:text-violet-200'
                    : 'border-zinc-800 bg-zinc-950 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200'
                }`}
              >
                {c}
              </button>
            ))}
          </div>
          {channel === 'Other' && (
            <input type="text" value={other} onChange={(e) => setOther(e.target.value)} maxLength={40} autoFocus
              placeholder="Signal, LinkedIn…" className={`${inputCls} mt-2`} />
          )}
        </Field>
        <Field label="About (optional)">
          <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={3} maxLength={500}
            placeholder="Asked how the tender radar is going" className={inputCls} />
        </Field>
        {error && <Notice tone="red">{error}</Notice>}
      </div>
    </Dialog>
  )
}
