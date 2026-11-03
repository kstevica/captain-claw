import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Bot, CheckCheck, CircleHelp, ClipboardCheck, Lightbulb, Loader2, Mail, MessageCircle, MessagesSquare, Network,
  Forward, Radio, Receipt, Send, Server, Share2, SlidersHorizontal, Sunrise, Waypoints, X, type LucideIcon,
} from 'lucide-react'
import {
  addLineSummary, askLine, fetchLine, handLineItem, LINE_PAGE, markAllLineRead, markLineRead, type LineItem,
} from '../../services/line'
import { fetchWaitingProposals, resolveProposal, type WaitingProposal } from '../../services/notifications'
import { useLineStore } from '../../stores/lineStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { untrustedUrlTransform } from '../../utils/untrustedContent'
import NotificationsConnection from '../connections/NotificationsConnection'

const KIND_ICON: Record<string, LucideIcon> = {
  proposal: Lightbulb,
  nudge: MessageCircle,
  result: ClipboardCheck,
  run: Network,
  ask: CircleHelp,
  payment: Receipt,
  mail: Mail,
  agent: Bot,
  fleet: Radio,
  share: Share2,
  deck: Server,
  talk: MessagesSquare,
}

// Nothing on the Line is fetched or followed from its text: an item can quote
// an email or a web page. Links stay text, pictures are dropped.
const LINE_MD: Components = {
  a: ({ children }) => <span className="underline underline-offset-2">{children}</span>,
  img: () => null,
}
const keepBreaks = (text: string) => text.replace(/([^\n])\n(?!\n)/g, '$1  \n')
const MD = '[&_p]:my-0.5 [&_strong]:font-semibold [&_strong]:text-zinc-300 [&_ul]:list-disc [&_ul]:pl-4 '
  + '[&_ol]:list-decimal [&_ol]:pl-4 [&_h1]:font-semibold [&_h2]:font-semibold [&_h3]:font-semibold'
// The deck's own "[Open in Taskboard](?taskboard=…)" line: a dead link here.
const TASK_LINK_LINE = /\n?\[[^\]\n]*\]\([^)\s]*[?&]taskboard=[^)\s]*\)\s*$/

function when(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const s = Math.floor((Date.now() - d.getTime()) / 1000)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  if (s < 7 * 86400) return `${Math.floor(s / 86400)}d ago`
  return d.toLocaleDateString()
}

type Filter = { kind: string } | { needsYou: true } | null
const filterKey = (f: Filter) => (f === null ? 'all' : 'needsYou' in f ? 'needs' : `kind:${f.kind}`)
const POLL_MS = 20000
// While an answer is coming, read the Line this often, for at most this long.
const ASK_POLL_MS = 3000
const ASK_WAIT_MS = 6 * 60_000

/** The Line: what the deck and the agents said on their own, newest first.
 *  The bell holds the part of it that needs you. */
export function LinePanel({ onClose, large = false }: { onClose?: () => void; large?: boolean }) {
  const [items, setItems] = useState<LineItem[]>([])
  const [kinds, setKinds] = useState<Record<string, string>>({})
  const [seenKinds, setSeenKinds] = useState<string[]>([])
  const [filter, setFilter] = useState<Filter>(null)
  const [loading, setLoading] = useState(true)
  const [older, setOlder] = useState(false)
  const [more, setMore] = useState(false)
  const [error, setError] = useState('')
  const [open, setOpen] = useState<Set<string>>(new Set())
  // Where the Line reaches you besides Flight Deck: the Notifications card, here.
  const [channels, setChannels] = useState(false)
  // The agent that answers on the Line, and a question waiting for its answer.
  const [agent, setAgent] = useState('')
  // The user's agents an item can be handed to.
  const [agents, setAgents] = useState<{ slug: string; name: string }[]>([])
  const [asked, setAsked] = useState<{ id: string; agent: string; at: number } | null>(null)
  // Proposals on the Line still waiting, by id: Approve / Reject here too.
  const [proposals, setProposals] = useState<Map<string, WaitingProposal> | null>(null)
  const setCounts = useLineStore((s) => s.set)
  const counts = useLineStore()
  const key = filterKey(filter)
  const keyRef = useRef(key)
  keyRef.current = key

  const query = useCallback((before?: string) => fetchLine({
    before,
    kind: filter && 'kind' in filter ? filter.kind : undefined,
    needsYou: !!filter && 'needsYou' in filter,
  }), [filter])

  // The newest page: on open, on a filter, and every POLL_MS (new items join
  // on top; what was paged in below stays).
  const loadNewest = useCallback(async () => {
    const asked = keyRef.current
    try {
      const page = await query()
      if (keyRef.current !== asked) return
      setItems((prev) => {
        const ids = new Set(page.items.map((i) => i.id))
        const oldest = page.items[page.items.length - 1]?.created_at || ''
        return [...page.items, ...prev.filter((i) => !ids.has(i.id) && i.created_at < oldest)]
      })
      setKinds(page.kinds)
      setAgent(page.agent)
      setAgents(page.agents)
      if (asked === 'all') {
        setSeenKinds((prev) => Array.from(new Set([...prev, ...page.items.map((i) => i.kind)])))
      }
      setMore((m) => m || page.items.length >= LINE_PAGE)
      setCounts({ unread: page.unread, waiting: page.waiting })
      setError('')
    } catch (e) {
      if (keyRef.current === asked) setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (keyRef.current === asked) setLoading(false)
    }
  }, [query, setCounts])

  useEffect(() => {
    setItems([]); setMore(false); setLoading(true)
    void loadNewest()
    const t = setInterval(() => { void loadNewest() }, POLL_MS)
    return () => clearInterval(t)
  }, [loadNewest])

  // Something read or answered in the bell (the same items): read again now.
  const bellState = useNotificationStore((s) => s.notifications
    .map((n) => `${n.id}${n.read ? 'r' : ''}${n.resolved ? 'x' : ''}`).join(','))
  const firstBell = useRef(true)
  const loadRef = useRef(loadNewest)
  loadRef.current = loadNewest
  useEffect(() => {
    if (firstBell.current) { firstBell.current = false; return }
    void loadRef.current()
  }, [bellState])

  // A question asked: read the Line every few seconds until its answer comes.
  useEffect(() => {
    if (!asked) return
    const t = setInterval(() => {
      if (Date.now() - asked.at > ASK_WAIT_MS) { setAsked(null); return }
      void loadRef.current()
    }, ASK_POLL_MS)
    return () => clearInterval(t)
  }, [asked])
  useEffect(() => {
    if (!asked) return
    const q = items.find((i) => i.id === asked.id)
    if (q && items.some((i) => i.kind === 'talk' && i.source === asked.agent && i.id !== q.id
        && i.created_at > q.created_at && !i.title.startsWith('You '))) setAsked(null)
  }, [items, asked])

  const waitingIds = items.filter((i) => i.ref_type === 'autonomy_action' && i.needs_you && !i.resolved)
    .map((i) => i.ref_id).join(',')
  const loadProposals = useCallback(() => {
    fetchWaitingProposals().then(setProposals).catch(() => setProposals(null))
  }, [])
  useEffect(() => {
    if (waitingIds) loadProposals()
  }, [waitingIds, loadProposals])
  const answered = () => {
    loadProposals()
    void loadRef.current()
    void useNotificationStore.getState().hydrateFromServer()
  }

  const loadOlder = async () => {
    const last = items[items.length - 1]
    if (!last) return
    setOlder(true)
    try {
      const page = await query(last.created_at)
      setItems((prev) => {
        const ids = new Set(prev.map((i) => i.id))
        return [...prev, ...page.items.filter((i) => !ids.has(i.id))]
      })
      setMore(page.items.length >= LINE_PAGE)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setOlder(false)
    }
  }

  const read = (item: LineItem) => {
    if (item.read) return
    setItems((prev) => prev.map((i) => (i.id === item.id ? {
      ...i, read: true, resolved: i.resolved || (i.needs_you && i.ref_type !== 'autonomy_action'),
    } : i)))
    setCounts({ unread: Math.max(0, counts.unread - 1), waiting: counts.waiting })
    void markLineRead(item.id).then(() => {
      void useLineStore.getState().refresh()
      void useNotificationStore.getState().hydrateFromServer()
    })
  }
  const readAll = async () => {
    setItems((prev) => prev.map((i) => ({
      ...i, read: true, resolved: i.resolved || (i.needs_you && i.ref_type !== 'autonomy_action'),
    })))
    await markAllLineRead()
    void useLineStore.getState().refresh()
    void useNotificationStore.getState().hydrateFromServer()
  }
  const toggle = (item: LineItem) => {
    read(item)
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(item.id)) next.delete(item.id)
      else next.add(item.id)
      return next
    })
  }

  const chips = useMemo(() => Object.keys(kinds).filter((k) => seenKinds.includes(k)), [kinds, seenKinds])
  const chip = (label: string, f: Filter, n?: number) => {
    const on = filterKey(f) === key
    return (
      <button key={filterKey(f)} onClick={() => setFilter(f)}
        className={`shrink-0 rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
          on ? 'bg-violet-600/20 text-violet-600 dark:text-violet-300' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'}`}>
        {label}{n ? <span className="ml-1 text-amber-600 dark:text-amber-400">{n}</span> : null}
      </button>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className={`flex items-center gap-2 border-b border-zinc-800 ${large ? 'px-5 py-3' : 'px-4 py-2.5'}`}>
        <Waypoints className="h-4 w-4 shrink-0 text-violet-500" />
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold text-zinc-200">Line</h2>
          <p className="truncate text-[11px] text-zinc-500">
            Everything the deck and your agents said on their own. What needs you is in the bell too.
          </p>
        </div>
        <button onClick={() => setChannels((v) => !v)} aria-expanded={channels}
          className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] hover:bg-zinc-800 hover:text-zinc-300 ${
            channels ? 'text-violet-600 dark:text-violet-300' : 'text-zinc-500'}`}
          title="Where your Line reaches you: WhatsApp, Telegram, devices, one agent's chat">
          <SlidersHorizontal className="h-3.5 w-3.5" /><span className="hidden sm:inline">Channels</span>
        </button>
        <button onClick={() => void readAll()} disabled={!counts.unread}
          className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
          title="Mark everything read">
          <CheckCheck className="h-3.5 w-3.5" /><span className="hidden sm:inline">Mark all read</span>
        </button>
        {onClose && (
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Close" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <div className={`flex gap-1 overflow-x-auto border-b border-zinc-800 ${large ? 'px-5 py-2' : 'px-4 py-1.5'}`}>
        {chip('All', null)}
        {chip('Needs you', { needsYou: true }, counts.waiting)}
        {chips.map((k) => chip(kinds[k] || k, { kind: k }))}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {channels && (
          <div className={`space-y-2 ${large ? 'px-5 pt-3' : 'px-4 pt-3'}`}>
            <NotificationsConnection kiosk open />
            <SummaryRow agent={agent} />
          </div>
        )}
        {loading && items.length === 0 ? (
          <div className="flex items-center justify-center gap-2 py-10 text-xs text-zinc-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…
          </div>
        ) : items.length === 0 ? (
          <div className="px-6 py-10 text-center text-xs text-zinc-500">
            {error ? `Couldn't load the Line: ${error}`
              : filter ? 'Nothing here.'
                : 'Nothing on your Line yet. Proposals, finished runs, payments to look at and fleet changes land here.'}
          </div>
        ) : (
          <ul className={large ? 'px-3 py-2' : 'px-2 py-1.5'}>
            {items.map((item) => (
              <LineRow key={item.id} item={item} expanded={open.has(item.id)} onToggle={() => toggle(item)}
                proposal={item.ref_type === 'autonomy_action' ? proposals?.get(item.ref_id) : undefined}
                onAnswered={() => { read(item); answered() }}
                agents={agents} lineAgent={agent}
                onHanded={(id, name) => { setAsked({ id, agent: name, at: Date.now() }); void loadRef.current() }} />
            ))}
          </ul>
        )}
        {more && items.length > 0 && (
          <div className="flex justify-center pb-4">
            <button onClick={() => void loadOlder()} disabled={older}
              className="flex items-center gap-1.5 rounded-md border border-zinc-800 px-3 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800 disabled:opacity-50">
              {older && <Loader2 className="h-3 w-3 animate-spin" />} Older
            </button>
          </div>
        )}
      </div>
      <AskBox agent={agent} large={large} asking={asked !== null}
        onAsked={(id, name) => { setAsked({ id, agent: name, at: Date.now() }); void loadRef.current() }}
        onPickAgent={() => setChannels(true)} />
    </div>
  )
}

// Ask the agent picked for the Line — about the Line or anything. The question
// goes on the Line now, its answer when it comes.
function AskBox({ agent, large, asking, onAsked, onPickAgent }: {
  agent: string; large: boolean; asking: boolean
  onAsked: (id: string, agent: string) => void; onPickAgent: () => void
}) {
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const pad = large ? 'px-5 py-3' : 'px-4 py-2.5'
  if (!agent) {
    return (
      <div className={`border-t border-zinc-800 text-[11px] text-zinc-500 ${pad}`}>
        Pick an agent for your Line to ask it what happened —{' '}
        <button onClick={onPickAgent} className="text-violet-600 underline hover:no-underline dark:text-violet-400">Channels</button>.
      </div>
    )
  }
  const send = async () => {
    const q = text.trim()
    if (!q || busy) return
    setBusy(true); setError('')
    try {
      const r = await askLine(q)
      setText('')
      onAsked(r.id, r.agent)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className={`border-t border-zinc-800 ${pad}`}>
      <form className="flex items-center gap-2" onSubmit={(e) => { e.preventDefault(); void send() }}>
        <input value={text} onChange={(e) => setText(e.target.value)} maxLength={2000}
          placeholder={`Ask ${agent} about your Line…`} aria-label={`Ask ${agent}`}
          className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none" />
        <button type="submit" disabled={!text.trim() || busy}
          className="flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40">
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />} Ask
        </button>
      </form>
      {asking && (
        <p className="mt-1 flex items-center gap-1 text-[11px] text-zinc-500">
          <Loader2 className="h-3 w-3 animate-spin" /> {agent} is answering — it lands on the Line.
        </p>
      )}
      {error && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

// A daily summary of the Line by the agent picked for it: a scheduled task.
function SummaryRow({ agent }: { agent: string }) {
  const [time, setTime] = useState('08:00')
  const [telegram, setTelegram] = useState(false)
  const [email, setEmail] = useState(false)
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  if (!agent) return null
  const add = async () => {
    setBusy(true); setError(''); setNote('')
    try {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || ''
      const r = await addLineSummary({ time, tz, telegram, email })
      setNote(`${r.updated ? 'Updated' : 'Added'}: ${r.agent} sums up your Line every day at ${time} — in its chat and on the Line${
        telegram ? ', and on Telegram' : ''}${email ? ', and by email' : ''}. Change it in Scheduled tasks.`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-4 py-3">
      <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-200">
        <Sunrise className="h-3.5 w-3.5 text-zinc-400" />
        Daily summary by {agent} at
        <input type="time" value={time} onChange={(e) => setTime(e.target.value)} aria-label="Summary time"
          className="rounded-md border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-[11px] text-zinc-200" />
        <label className="flex items-center gap-1 text-[11px] text-zinc-400">
          <input type="checkbox" checked={telegram} onChange={(e) => setTelegram(e.target.checked)} className="h-3 w-3 accent-violet-600" /> Telegram
        </label>
        <label className="flex items-center gap-1 text-[11px] text-zinc-400">
          <input type="checkbox" checked={email} onChange={(e) => setEmail(e.target.checked)} className="h-3 w-3 accent-violet-600" /> email
        </label>
        <button onClick={() => void add()} disabled={busy || !time}
          className="ml-auto flex items-center gap-1 rounded-md bg-zinc-800 px-2.5 py-1 text-[11px] text-zinc-200 hover:bg-zinc-700 disabled:opacity-40">
          {busy && <Loader2 className="h-3 w-3 animate-spin" />} Add
        </button>
      </div>
      <p className="mt-1 text-[11px] text-zinc-500">What waits for you first, then what failed, then results — in {agent}'s chat and on the Line.</p>
      {note && <p className="mt-1 text-[11px] text-emerald-700 dark:text-emerald-300">{note}</p>}
      {error && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

// A proposal's answer, on the Line: Approve sends it to your agent, Reject drops it.
function ProposalButtons({ id, onAnswered }: { id: string; onAnswered: () => void }) {
  const [busy, setBusy] = useState<'approve' | 'reject' | null>(null)
  const [error, setError] = useState('')
  const answer = async (approve: boolean) => {
    setBusy(approve ? 'approve' : 'reject'); setError('')
    try {
      await resolveProposal(id, approve)
      onAnswered()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  const btn = 'flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium disabled:opacity-50'
  return (
    <div className="mt-1" onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
      <div className="flex items-center gap-1.5">
        <button disabled={busy !== null} onClick={() => void answer(true)} className={`${btn} bg-emerald-600 text-white hover:bg-emerald-500`}>
          {busy === 'approve' && <Loader2 className="h-3 w-3 animate-spin" />} Approve
        </button>
        <button disabled={busy !== null} onClick={() => void answer(false)} className={`${btn} bg-zinc-800 text-zinc-300 hover:bg-zinc-700`}>
          {busy === 'reject' && <Loader2 className="h-3 w-3 animate-spin" />} Reject
        </button>
      </div>
      {error && <p className="mt-0.5 text-[10px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

// Hand an item to one of the user's agents, with a few words of theirs; the
// answer lands on the Line. The item goes along quoted, as information.
function HandForm({ item, agents, lineAgent, onHanded, onCancel }: {
  item: LineItem; agents: { slug: string; name: string }[]; lineAgent: string
  onHanded: (id: string, agent: string) => void; onCancel: () => void
}) {
  const [slug, setSlug] = useState(() => (agents.find((a) => a.name === lineAgent) || agents[0])?.slug || '')
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const send = async () => {
    if (!slug || busy) return
    setBusy(true); setError('')
    try {
      const r = await handLineItem(item.id, slug, note.trim())
      onHanded(r.id, r.agent)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  const field = 'rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none'
  return (
    <div className="mt-1.5 space-y-1.5 rounded-lg border border-zinc-800 bg-zinc-950/60 p-2"
      onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-zinc-400">
        Hand to
        <select value={slug} onChange={(e) => setSlug(e.target.value)} className={field} aria-label="The agent to hand it to">
          {agents.map((a) => <option key={a.slug} value={a.slug}>{a.name}</option>)}
        </select>
      </div>
      <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2} maxLength={2000}
        placeholder="What should it do? (optional — e.g. draft a reply, find out why it failed)"
        aria-label="What the agent should do" className={`w-full resize-y ${field}`} />
      <div className="flex items-center gap-1.5">
        <button onClick={() => void send()} disabled={!slug || busy}
          className="flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-violet-500 disabled:opacity-40">
          {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Forward className="h-3 w-3" />} Hand it over
        </button>
        <button onClick={onCancel} className="rounded-md px-2 py-1 text-[11px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">Cancel</button>
        <span className="text-[10px] text-zinc-600">The answer lands on the Line.</span>
      </div>
      {error && <p className="text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

function LineRow({ item, expanded, onToggle, proposal, onAnswered, agents, lineAgent, onHanded }: {
  item: LineItem; expanded: boolean; onToggle: () => void
  /** A proposal still waiting (Approve / Reject here, unless its words are read on its task first). */
  proposal?: WaitingProposal
  onAnswered: () => void
  agents: { slug: string; name: string }[]
  lineAgent: string
  onHanded: (id: string, agent: string) => void
}) {
  const [handing, setHanding] = useState(false)
  const Icon = KIND_ICON[item.kind] || Server
  const waiting = item.needs_you && !item.resolved
  const body = item.body.replace(TASK_LINK_LINE, '').trim()
  const failed = /_(error|failed)$/.test(item.type) || item.type === 'error'
  return (
    <li>
      <div role="button" tabIndex={0} onClick={onToggle}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle() } }}
        aria-expanded={expanded}
        className={`group flex w-full cursor-pointer items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-zinc-900/60 focus:outline-none focus-visible:ring-1 focus-visible:ring-violet-500/60 ${
          item.read ? '' : 'bg-zinc-900/30'}`}>
        <span className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md ${
          failed ? 'bg-red-500/10 text-red-500' : waiting ? 'bg-amber-500/10 text-amber-600 dark:text-amber-400'
            : 'bg-zinc-800/60 text-zinc-400'}`}>
          <Icon className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className={`min-w-0 truncate text-xs ${item.read ? 'text-zinc-300' : 'font-semibold text-zinc-100'}`}>{item.title}</span>
            {!item.read && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-violet-500" aria-label="Unread" />}
            {waiting && (
              <span className="shrink-0 rounded-full bg-amber-500/15 px-1.5 text-[10px] font-medium text-amber-700 dark:text-amber-300">
                {item.ref_type !== 'autonomy_action' ? 'Needs you'
                  : proposal?.taskboard ? 'Waiting — read it on its task first' : 'Waiting for you'}
              </span>
            )}
          </div>
          {body && (
            <div className={`mt-0.5 text-[11px] text-zinc-500 ${MD} ${expanded ? '' : 'line-clamp-2'}`}>
              <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={LINE_MD}>{keepBreaks(body)}</Markdown>
            </div>
          )}
          {waiting && proposal && !proposal.taskboard && <ProposalButtons id={item.ref_id} onAnswered={onAnswered} />}
          {expanded && agents.length > 0 && !handing && (
            <button onClick={(e) => { e.stopPropagation(); setHanding(true) }}
              onKeyDown={(e) => e.stopPropagation()}
              className="mt-1 flex items-center gap-1 rounded px-1 py-0.5 text-[11px] font-medium text-violet-600 hover:bg-violet-500/10 dark:text-violet-400">
              <Forward className="h-3 w-3" /> Hand to agent…
            </button>
          )}
          {expanded && handing && (
            <HandForm item={item} agents={agents} lineAgent={lineAgent} onCancel={() => setHanding(false)}
              onHanded={(id, name) => { setHanding(false); onHanded(id, name) }} />
          )}
          <div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-zinc-600">
            {item.source && <span className="truncate">{item.source}</span>}
            {item.source && <span>·</span>}
            <span>{when(item.created_at)}</span>
          </div>
        </div>
      </div>
    </li>
  )
}
