import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  ArrowUp, Bot, CheckCheck, ChevronDown, ChevronLeft, CircleHelp, ClipboardCheck, Inbox, Lightbulb, Loader2, Mail,
  MessageCircle, MessagesSquare, Network, Radio, Receipt, Server, Share2, SlidersHorizontal, Sunrise, Waypoints, X,
  type LucideIcon,
} from 'lucide-react'
import {
  addLineSummary, askLine, fetchLine, handLineItem, LINE_PAGE, markAllLineRead, markLineRead, type LineItem,
} from '../../services/line'
import { fetchWaitingProposals, resolveProposal, type WaitingProposal } from '../../services/notifications'
import { useLineStore } from '../../stores/lineStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { untrustedUrlTransform } from '../../utils/untrustedContent'
import NotificationsConnection from '../connections/NotificationsConnection'

type Agent = { slug: string; name: string }

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
// One item's kind, as a word.
const KIND_WORD: Record<string, string> = {
  proposal: 'Proposal', nudge: 'Nudge', result: 'Result', run: 'Run', ask: 'Question', payment: 'Payment',
  mail: 'Mail', agent: 'Agent', fleet: 'Fleet', share: 'Sharing', deck: 'Deck', talk: 'Asked',
}
// An agent's colour, picked by its name (full class strings, for Tailwind).
const HUES = [
  'bg-violet-500/15 text-violet-700 dark:text-violet-300',
  'bg-sky-500/15 text-sky-700 dark:text-sky-300',
  'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  'bg-rose-500/15 text-rose-700 dark:text-rose-300',
  'bg-teal-500/15 text-teal-700 dark:text-teal-300',
  'bg-indigo-500/15 text-indigo-700 dark:text-indigo-300',
]
const hueOf = (name: string) => HUES[[...name].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7) % HUES.length]
function initials(name: string): string {
  const words = name.trim().split(/[\s_-]+/).filter(Boolean)
  if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase()
  return name.trim().slice(0, 2).toUpperCase() || '?'
}

// Nothing on the Line is fetched or followed from its text: an item can quote
// an email or a web page. Links stay text, pictures are dropped.
const LINE_MD: Components = {
  a: ({ children }) => <span className="underline decoration-zinc-600 underline-offset-2">{children}</span>,
  img: () => null,
}
const keepBreaks = (text: string) => text.replace(/([^\n])\n(?!\n)/g, '$1  \n')
// The deck's own "[Open in Taskboard](?taskboard=…)" line: a dead link here.
const TASK_LINK_LINE = /\n?\[[^\]\n]*\]\([^)\s]*[?&]taskboard=[^)\s]*\)\s*$/
const PROSE = 'text-[13px] leading-relaxed text-zinc-300 [&_p]:my-2 [&_p:first-child]:mt-0 '
  + '[&_strong]:font-semibold [&_strong]:text-zinc-100 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 '
  + '[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-1 [&_li]:pl-0.5 '
  + '[&_h1]:mb-1 [&_h1]:mt-4 [&_h1]:text-sm [&_h1]:font-semibold [&_h1]:text-zinc-100 '
  + '[&_h2]:mb-1 [&_h2]:mt-4 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-zinc-100 '
  + '[&_h3]:mb-1 [&_h3]:mt-3 [&_h3]:font-semibold [&_h3]:text-zinc-100 '
  + '[&_code]:rounded [&_code]:bg-zinc-800 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[12px] '
  + '[&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-zinc-700 [&_blockquote]:pl-3 [&_blockquote]:text-zinc-400 '
  + '[&_table]:my-2 [&_table]:w-full [&_table]:text-xs [&_td]:border-t [&_td]:border-zinc-800 [&_td]:py-1 [&_td]:pr-3 '
  + '[&_th]:py-1 [&_th]:pr-3 [&_th]:text-left [&_th]:font-medium [&_th]:text-zinc-400 [&_hr]:my-3 [&_hr]:border-zinc-800'

/** An item's text as plain lines: no markdown marks, no Taskboard link line. */
function plainLines(md: string): string[] {
  return md.replace(TASK_LINK_LINE, '').split('\n').map((l) => l
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^\s{0,3}(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s?)/, '')
    .replace(/\*\*|__|`/g, '')
    .trim()).filter(Boolean)
}
/** Who an item is from (an agent), or "" when the deck said it. A nudge came
 *  through the bell, which keeps no sender: its title names it. */
const whoOf = (item: LineItem) => (item.source && !item.source.startsWith('via ') ? item.source
  : /^Nudge from (.+)$/.exec(item.title)?.[1]?.trim() || '')
/** A title that only says who or what kind ("Nudge from Mira", "Waiting for
 *  your approval"): the text says what it's about. */
const GENERIC_TITLE = /^(Nudge from |Waiting for your approval)/
/** The line a row leads with, and what follows it. */
function headlineOf(item: LineItem): { headline: string; preview: string } {
  const lines = plainLines(item.body)
  if (GENERIC_TITLE.test(item.title) && lines.length) {
    return { headline: lines[0], preview: lines.slice(1).join(' ') }
  }
  return { headline: item.title, preview: lines.join(' ') }
}
const isFailed = (item: LineItem) => /_(error|failed)$/.test(item.type) || item.type === 'error'

const sameDay = (a: Date, b: Date) => a.toDateString() === b.toDateString()
function dayLabel(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const now = new Date()
  if (sameDay(d, now)) return 'Today'
  const y = new Date(now); y.setDate(now.getDate() - 1)
  if (sameDay(d, y)) return 'Yesterday'
  if (now.getTime() - d.getTime() < 6 * 86_400_000) return d.toLocaleDateString(undefined, { weekday: 'long' })
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'long', ...(d.getFullYear() !== now.getFullYear() ? { year: 'numeric' } : {}) })
}
function shortTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const now = new Date()
  if (sameDay(d, now)) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  if (now.getTime() - d.getTime() < 6 * 86_400_000) return d.toLocaleDateString(undefined, { weekday: 'short' })
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}
function ago(iso: string): string {
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (!Number.isFinite(s)) return ''
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return `${Math.floor(s / 86400)} d ago`
}
function fullTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return `${d.toLocaleString(undefined, { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })} · ${ago(iso)}`
}

type Filter = { kind: string } | { needsYou: true } | null
const filterKey = (f: Filter) => (f === null ? 'all' : 'needsYou' in f ? 'needs' : `kind:${f.kind}`)
const POLL_MS = 20000
// While an answer is coming, read the Line this often, for at most this long.
const ASK_POLL_MS = 3000
const ASK_WAIT_MS = 6 * 60_000
// From this width the Line shows its list and the open item side by side.
const TWO_PANES = 720
const SUGGESTIONS = ['Look into it', 'Draft a reply', 'What should I do?']

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T | null>(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    setWidth(el.getBoundingClientRect().width)
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, width]
}

/** The Line: what the deck and the agents said on their own, newest first —
 *  a list, the open item beside it (on a phone, in its place), and one box to
 *  ask an agent about the Line or hand it the open item. The bell holds the
 *  part of it that needs you. */
export function LinePanel({ onClose }: { onClose?: () => void; large?: boolean }) {
  const [rootRef, width] = useWidth<HTMLDivElement>()
  const wide = width >= TWO_PANES
  const [items, setItems] = useState<LineItem[]>([])
  const [kinds, setKinds] = useState<Record<string, string>>({})
  const [seenKinds, setSeenKinds] = useState<string[]>([])
  const [filter, setFilter] = useState<Filter>(null)
  const [loading, setLoading] = useState(true)
  const [older, setOlder] = useState(false)
  const [more, setMore] = useState(false)
  const [error, setError] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  // Where the Line reaches you besides Flight Deck, in the open item's place.
  const [channels, setChannels] = useState(false)
  // The agent that answers on the Line, and the user's agents.
  const [lineAgent, setLineAgent] = useState({ slug: '', name: '' })
  const [agents, setAgents] = useState<Agent[]>([])
  // A question or a hand-over waiting for its answer.
  const [asked, setAsked] = useState<{ id: string; agent: string; at: number; from: string | null } | null>(null)
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
    const asking = keyRef.current
    try {
      const page = await query()
      if (keyRef.current !== asking) return
      setItems((prev) => {
        const ids = new Set(page.items.map((i) => i.id))
        const oldest = page.items[page.items.length - 1]?.created_at || ''
        return [...page.items, ...prev.filter((i) => !ids.has(i.id) && i.created_at < oldest)]
      })
      setKinds(page.kinds)
      setLineAgent({ slug: page.agent_slug, name: page.agent })
      setAgents(page.agents)
      if (asking === 'all') {
        setSeenKinds((prev) => Array.from(new Set([...prev, ...page.items.map((i) => i.kind)])))
      }
      setMore((m) => m || page.items.length >= LINE_PAGE)
      setCounts({ unread: page.unread, waiting: page.waiting })
      setError('')
    } catch (e) {
      if (keyRef.current === asking) setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (keyRef.current === asking) setLoading(false)
    }
  }, [query, setCounts])
  const loadRef = useRef(loadNewest)
  loadRef.current = loadNewest

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
  useEffect(() => {
    if (firstBell.current) { firstBell.current = false; return }
    void loadRef.current()
  }, [bellState])

  // A question asked: read the Line every few seconds until its answer comes,
  // then open the answer (unless the user went on to something else).
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
    const answer = q && items.find((i) => i.kind === 'talk' && i.source === asked.agent && i.id !== q.id
      && i.created_at > q.created_at && !i.title.startsWith('You '))
    if (!answer) return
    setAsked(null)
    setSelectedId((cur) => (cur === null || cur === asked.from || cur === q.id ? answer.id : cur))
  }, [items, asked])

  const waitingIds = items.filter((i) => i.ref_type === 'autonomy_action' && i.needs_you && !i.resolved)
    .map((i) => i.ref_id).join(',')
  const loadProposals = useCallback(() => {
    fetchWaitingProposals().then(setProposals).catch(() => setProposals(null))
  }, [])
  useEffect(() => {
    if (waitingIds) loadProposals()
  }, [waitingIds, loadProposals])

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

  const syncBell = () => {
    void useLineStore.getState().refresh()
    void useNotificationStore.getState().hydrateFromServer()
  }
  const read = (item: LineItem) => {
    if (item.read) return
    setItems((prev) => prev.map((i) => (i.id === item.id ? {
      ...i, read: true, resolved: i.resolved || (i.needs_you && i.ref_type !== 'autonomy_action'),
    } : i)))
    setCounts({ unread: Math.max(0, counts.unread - 1), waiting: counts.waiting })
    void markLineRead(item.id).then(syncBell)
  }
  const readAll = async () => {
    setItems((prev) => prev.map((i) => ({
      ...i, read: true, resolved: i.resolved || (i.needs_you && i.ref_type !== 'autonomy_action'),
    })))
    await markAllLineRead()
    syncBell()
  }
  const open = (item: LineItem) => {
    setChannels(false)
    setSelectedId(item.id)
  }
  const back = () => { setSelectedId(null); setChannels(false) }
  const answered = (item: LineItem) => {
    read(item)
    loadProposals()
    void loadRef.current()
    syncBell()
  }

  const selected = useMemo(() => items.find((i) => i.id === selectedId) || null, [items, selectedId])
  // Open is read — opened by a click, or an answer that opened on its own.
  const readRef = useRef(read)
  readRef.current = read
  useEffect(() => {
    if (selected && !selected.read) readRef.current(selected)
  }, [selected])
  const chips = useMemo(() => Object.keys(kinds).filter((k) => seenKinds.includes(k)), [kinds, seenKinds])
  const detailShown = channels || selected !== null
  const iconBtn = (on = false) => `flex h-8 items-center gap-1.5 rounded-lg px-2 text-xs transition-colors ${
    on ? 'bg-violet-500/15 text-violet-700 dark:text-violet-300' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`

  const list = (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex shrink-0 gap-1 overflow-x-auto px-3 py-2 [mask-image:linear-gradient(to_right,black_88%,transparent)] [scrollbar-width:none]">
        <FilterChip label="All" on={key === 'all'} onClick={() => setFilter(null)} />
        <FilterChip label="Needs you" on={key === 'needs'} onClick={() => setFilter({ needsYou: true })} count={counts.waiting} />
        {chips.map((k) => (
          <FilterChip key={k} label={kinds[k] || k} on={key === `kind:${k}`} onClick={() => setFilter({ kind: k })} />
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {loading && items.length === 0 ? (
          <div className="flex items-center justify-center gap-2 py-16 text-xs text-zinc-500">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading your Line…
          </div>
        ) : items.length === 0 ? (
          <EmptyList error={error} filtered={!!filter} />
        ) : (
          <ul>
            {items.map((item, i) => {
              const day = dayLabel(item.created_at)
              const showDay = i === 0 || dayLabel(items[i - 1].created_at) !== day
              return (
                <li key={item.id}>
                  {showDay && (
                    <div className="px-3 pb-1 pt-3 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">{day}</div>
                  )}
                  <ListRow item={item} selected={item.id === selectedId} onOpen={() => open(item)}
                    proposal={item.ref_type === 'autonomy_action' ? proposals?.get(item.ref_id) : undefined} />
                </li>
              )
            })}
          </ul>
        )}
        {more && items.length > 0 && (
          <div className="flex justify-center pt-3">
            <button onClick={() => void loadOlder()} disabled={older}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-800 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
              {older && <Loader2 className="h-3 w-3 animate-spin" />} Show older
            </button>
          </div>
        )}
      </div>
    </div>
  )

  const detail = channels ? (
    <ChannelsView agent={lineAgent.name} onBack={wide ? undefined : back} onClose={() => setChannels(false)} />
  ) : selected ? (
    <Detail item={selected} onBack={wide ? undefined : back}
      proposal={selected.ref_type === 'autonomy_action' ? proposals?.get(selected.ref_id) : undefined}
      onAnswered={() => answered(selected)} />
  ) : (
    <EmptyDetail unread={counts.unread} waiting={counts.waiting} />
  )

  return (
    <div ref={rootRef} className="flex h-full min-h-0 flex-col bg-zinc-950">
      <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/15 text-violet-600 dark:text-violet-300">
          <Waypoints className="h-[18px] w-[18px]" />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold text-zinc-100">Line</h2>
          <p className="truncate text-[11px] text-zinc-500">What the deck and your agents did and said on their own</p>
        </div>
        <button onClick={() => setChannels((v) => !v)} aria-pressed={channels} className={iconBtn(channels)}
          title="Where your Line reaches you: WhatsApp, Telegram, devices, one agent's chat — and a daily summary">
          <SlidersHorizontal className="h-4 w-4" /><span className={wide ? '' : 'sr-only'}>Channels</span>
        </button>
        <button onClick={() => void readAll()} disabled={!counts.unread} className={`${iconBtn()} disabled:opacity-40`}
          title="Mark everything read">
          <CheckCheck className="h-4 w-4" /><span className={wide ? '' : 'sr-only'}>Mark all read</span>
        </button>
        {onClose && (
          <button onClick={onClose} className={iconBtn()} title="Close" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <div className="flex min-h-0 flex-1">
        {wide ? (
          <>
            <div className="flex w-[min(400px,42%)] shrink-0 flex-col border-r border-zinc-800">{list}</div>
            <div className="flex min-w-0 flex-1 flex-col">{detail}</div>
          </>
        ) : detailShown ? (
          <div className="flex min-w-0 flex-1 flex-col">{detail}</div>
        ) : list}
      </div>

      {!channels && (
        <Composer agents={agents} lineAgent={lineAgent} target={selected} pending={asked?.agent || ''} compact={!wide}
          onDetach={() => setSelectedId(null)}
          onSent={(id, agent) => { setAsked({ id, agent, at: Date.now(), from: selected?.id || null }); void loadRef.current() }}
          onSetUp={() => setChannels(true)} />
      )}
    </div>
  )
}

function FilterChip({ label, on, onClick, count = 0 }: { label: string; on: boolean; onClick: () => void; count?: number }) {
  return (
    <button onClick={onClick}
      className={`flex h-7 shrink-0 items-center gap-1.5 rounded-lg px-2.5 text-xs font-medium transition-colors ${
        on ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:bg-zinc-800/60 hover:text-zinc-300'}`}>
      {label}
      {count > 0 && (
        <span className="rounded-full bg-amber-500/20 px-1.5 text-[10px] font-semibold leading-4 text-amber-700 dark:text-amber-300">{count}</span>
      )}
    </button>
  )
}

/** Who an item is from: the agent's initials in its colour, else the kind's icon. */
function Avatar({ item, size = 'md' }: { item: LineItem; size?: 'md' | 'lg' }) {
  const who = whoOf(item)
  const box = size === 'lg' ? 'h-10 w-10 rounded-xl text-xs' : 'h-9 w-9 rounded-xl text-[11px]'
  if (who) {
    return <span className={`flex shrink-0 items-center justify-center font-semibold ${box} ${hueOf(who)}`}>{initials(who)}</span>
  }
  const Icon = KIND_ICON[item.kind] || Server
  return (
    <span className={`flex shrink-0 items-center justify-center bg-zinc-800 text-zinc-400 ${box}`}>
      <Icon className={size === 'lg' ? 'h-5 w-5' : 'h-4 w-4'} />
    </span>
  )
}

function StatusPill({ item, proposal }: { item: LineItem; proposal?: WaitingProposal }) {
  const waiting = item.needs_you && !item.resolved
  const pill = 'shrink-0 rounded-md px-1.5 py-px text-[10px] font-semibold'
  if (isFailed(item)) return <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-400`}>Failed</span>
  if (waiting && item.ref_type === 'autonomy_action') {
    return <span className={`${pill} bg-amber-500/15 text-amber-700 dark:text-amber-300`}>{proposal ? 'Approve?' : 'Waiting'}</span>
  }
  if (waiting) return <span className={`${pill} bg-amber-500/15 text-amber-700 dark:text-amber-300`}>Needs you</span>
  return null
}

function ListRow({ item, selected, onOpen, proposal }: {
  item: LineItem; selected: boolean; onOpen: () => void; proposal?: WaitingProposal
}) {
  const who = whoOf(item)
  const { headline, preview } = headlineOf(item)
  return (
    <button onClick={onOpen} aria-current={selected || undefined}
      className={`group relative flex w-full items-start gap-3 rounded-xl px-3 py-2.5 text-left transition-colors ${
        selected ? 'bg-violet-500/10 ring-1 ring-inset ring-violet-500/25' : 'hover:bg-zinc-900'}`}>
      {!item.read && <span className="absolute left-1 top-[22px] h-1.5 w-1.5 rounded-full bg-violet-500" aria-label="Unread" />}
      <Avatar item={item} />
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-1.5">
          <span className={`min-w-0 truncate text-xs ${item.read ? 'text-zinc-400' : 'font-semibold text-zinc-200'}`}>
            {who || KIND_WORD[item.kind] || 'Deck'}
          </span>
          {who && <span className="shrink-0 text-[11px] text-zinc-600">· {KIND_WORD[item.kind] || item.kind}</span>}
          <span className="ml-auto shrink-0 pl-2 text-[11px] tabular-nums text-zinc-500">{shortTime(item.created_at)}</span>
        </span>
        <span className="mt-0.5 flex items-center gap-1.5">
          <span className={`min-w-0 truncate text-[13px] ${item.read ? 'text-zinc-300' : 'font-medium text-zinc-100'}`}>{headline}</span>
          <StatusPill item={item} proposal={proposal} />
        </span>
        {preview && <span className="mt-0.5 block truncate text-xs text-zinc-500">{preview}</span>}
      </span>
    </button>
  )
}

function Detail({ item, proposal, onBack, onAnswered }: {
  item: LineItem; proposal?: WaitingProposal; onBack?: () => void; onAnswered: () => void
}) {
  const who = whoOf(item)
  const full = item.body.replace(TASK_LINK_LINE, '').trim()
  // A generic title gives way to the text's own first line, as the heading.
  const generic = GENERIC_TITLE.test(item.title) && !!full
  const heading = generic ? headlineOf(item).headline : item.title
  const body = generic ? full.replace(/^[^\n]*\n*/, '').trim() : full
  const waiting = item.needs_you && !item.resolved
  const scroller = useRef<HTMLDivElement>(null)
  useEffect(() => { scroller.current?.scrollTo({ top: 0 }) }, [item.id])
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {onBack && (
        <button onClick={onBack} className="flex shrink-0 items-center gap-1 border-b border-zinc-800 px-3 py-2 text-xs font-medium text-zinc-400 hover:text-zinc-200">
          <ChevronLeft className="h-4 w-4" /> Line
        </button>
      )}
      <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-2xl px-5 py-5 sm:px-6">
          <div className="flex items-center gap-3">
            <Avatar item={item} size="lg" />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-zinc-100">{who || KIND_WORD[item.kind] || 'Flight Deck'}</span>
                <StatusPill item={item} proposal={proposal} />
              </div>
              <div className="truncate text-[11px] text-zinc-500">
                {who ? `${KIND_WORD[item.kind] || item.kind} · ` : ''}{fullTime(item.created_at)}
              </div>
            </div>
          </div>

          <h3 className="mt-4 text-base font-semibold leading-snug text-zinc-100">{heading}</h3>

          {waiting && item.ref_type === 'autonomy_action' && (
            <div className="mt-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3.5 py-3">
              {proposal && !proposal.taskboard ? (
                <>
                  <div className="text-xs font-medium text-amber-800 dark:text-amber-200">Waiting for your approval</div>
                  <ProposalButtons id={item.ref_id} onAnswered={onAnswered} />
                </>
              ) : (
                <div className="text-xs text-amber-800 dark:text-amber-200">
                  {proposal?.taskboard ? 'A next step on the Taskboard — read and approve it on its task.'
                    : 'Waiting for your answer — in the bell.'}
                </div>
              )}
            </div>
          )}

          {body ? (
            <article className={`mt-4 ${PROSE}`}>
              <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={LINE_MD}>{keepBreaks(body)}</Markdown>
            </article>
          ) : !generic && (
            <p className="mt-4 text-xs text-zinc-500">Nothing more to it.</p>
          )}
          {item.trust === 'untrusted' && (
            <p className="mt-4 text-[11px] text-zinc-500">This quotes someone else's words — links aren't followed and pictures aren't shown.</p>
          )}
        </div>
      </div>
    </div>
  )
}

function EmptyDetail({ unread, waiting }: { unread: number; waiting: number }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-8 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-900 text-zinc-500 ring-1 ring-zinc-800">
        <Inbox className="h-6 w-6" />
      </span>
      <p className="mt-3 text-sm font-medium text-zinc-200">
        {waiting ? `${waiting} waiting for you` : unread ? `${unread} new` : 'All caught up'}
      </p>
      <p className="mt-1 max-w-xs text-xs leading-relaxed text-zinc-500">
        Open an item to read it whole, answer it, or hand it to one of your agents. Or ask an agent about your Line below.
      </p>
    </div>
  )
}

function EmptyList({ error, filtered }: { error: string; filtered: boolean }) {
  return (
    <div className="px-6 py-16 text-center">
      <p className="text-sm font-medium text-zinc-300">{error ? "Couldn't load the Line" : filtered ? 'Nothing here' : 'Nothing on your Line yet'}</p>
      <p className="mx-auto mt-1 max-w-xs text-xs leading-relaxed text-zinc-500">
        {error || (filtered ? 'Try another filter.'
          : 'Results, finished runs, proposals, payments to look at and fleet changes land here as they happen.')}
      </p>
    </div>
  )
}

function ChannelsView({ agent, onBack, onClose }: { agent: string; onBack?: () => void; onClose: () => void }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-4 py-2">
        {onBack ? (
          <button onClick={onBack} className="-ml-1 flex items-center gap-1 text-xs font-medium text-zinc-400 hover:text-zinc-200">
            <ChevronLeft className="h-4 w-4" /> Line
          </button>
        ) : (
          <>
            <span className="text-xs font-medium text-zinc-300">Where your Line reaches you</span>
            <button onClick={onClose} className="ml-auto rounded-md p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" aria-label="Close channels">
              <X className="h-3.5 w-3.5" />
            </button>
          </>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-2xl space-y-3 px-4 py-4">
          <NotificationsConnection kiosk open />
          <SummaryRow agent={agent} />
        </div>
      </div>
    </div>
  )
}

/** Pick one of the user's agents: a pill with its initials (and name, where
 *  there's room), a native select under it. */
function AgentPill({ agents, value, onChange, compact = false }: {
  agents: Agent[]; value: string; onChange: (slug: string) => void; compact?: boolean
}) {
  const name = agents.find((a) => a.slug === value)?.name || ''
  return (
    <label title={name} className="relative flex h-8 shrink-0 cursor-pointer items-center gap-1.5 rounded-lg bg-zinc-800/80 pl-1 pr-1.5 text-xs font-medium text-zinc-200 transition-colors hover:bg-zinc-800">
      <span className={`flex h-6 w-6 items-center justify-center rounded-md text-[10px] font-semibold ${hueOf(name || '?')}`}>{initials(name || '?')}</span>
      {!compact && <span className="max-w-[8rem] truncate">{name}</span>}
      <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
      <select value={value} onChange={(e) => onChange(e.target.value)} aria-label="Which agent"
        className="absolute inset-0 cursor-pointer opacity-0">
        {agents.map((a) => <option key={a.slug} value={a.slug}>{a.name}</option>)}
      </select>
    </label>
  )
}

/** One box at the bottom: ask an agent about the Line or, with an item open,
 *  hand it that item with a few words. The answer lands on the Line. */
function Composer({ agents, lineAgent, target, pending, compact, onDetach, onSent, onSetUp }: {
  agents: Agent[]; lineAgent: { slug: string; name: string }; target: LineItem | null; pending: string
  /** Narrow: the agent pill shows only initials. */
  compact: boolean
  onDetach: () => void; onSent: (id: string, agent: string) => void; onSetUp: () => void
}) {
  const [slug, setSlug] = useState('')
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const box = useRef<HTMLTextAreaElement>(null)
  // The Line's own agent by default, else the first; kept while it still exists.
  useEffect(() => {
    if (!agents.some((a) => a.slug === slug)) setSlug(agents.some((a) => a.slug === lineAgent.slug) ? lineAgent.slug : agents[0]?.slug || '')
  }, [agents, lineAgent.slug, slug])
  useLayoutEffect(() => {
    const el = box.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 132)}px`
  }, [text])
  const name = agents.find((a) => a.slug === slug)?.name || ''
  const shell = 'shrink-0 border-t border-zinc-800 bg-zinc-950 px-3 pb-[max(0.625rem,env(safe-area-inset-bottom))] pt-2.5 sm:px-4'
  if (!agents.length) {
    return (
      <div className={`${shell} text-xs text-zinc-500`}>
        Create an agent to ask it about your Line.{' '}
        <button onClick={onSetUp} className="text-violet-600 hover:underline dark:text-violet-400">Channels</button>
      </div>
    )
  }
  const send = async () => {
    const words = text.trim()
    if (busy || !slug || (!words && !target)) return
    setBusy(true); setError('')
    try {
      const r = target ? await handLineItem(target.id, slug, words) : await askLine(words, slug)
      setText('')
      onSent(r.id, r.agent)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  const about = target ? headlineOf(target).headline : ''
  return (
    <div className={shell}>
      {pending && (
        <div className="mb-2 flex items-center gap-1.5 text-[11px] text-zinc-500">
          <Loader2 className="h-3 w-3 animate-spin" /> {pending} is on it — the answer lands on the Line.
        </div>
      )}
      {target && (
        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          <span className="flex min-w-0 max-w-full items-center gap-1 rounded-lg bg-violet-500/10 py-0.5 pl-2 pr-1 text-[11px] text-violet-700 dark:text-violet-300">
            <span className="shrink-0 font-medium">Hand over:</span>
            <span className="min-w-0 truncate">{about}</span>
            <button onClick={onDetach} className="shrink-0 rounded p-0.5 hover:bg-violet-500/20" aria-label="Not about this item">
              <X className="h-3 w-3" />
            </button>
          </span>
          {SUGGESTIONS.map((s) => (
            <button key={s} onClick={() => { setText(s); box.current?.focus() }}
              className="rounded-lg border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-400 transition-colors hover:border-zinc-700 hover:text-zinc-200">
              {s}
            </button>
          ))}
        </div>
      )}
      <div className="flex items-end gap-1.5 rounded-xl border border-zinc-800 bg-zinc-900/70 p-1.5 transition-colors focus-within:border-violet-500/50">
        <AgentPill agents={agents} value={slug} onChange={setSlug} compact={compact} />
        <textarea ref={box} value={text} rows={1} maxLength={2000}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void send() } }}
          placeholder={target ? `What should ${name} do with it?` : `Ask ${name} about your Line…`}
          aria-label={target ? `Hand it to ${name}` : `Ask ${name}`}
          className="max-h-[132px] min-h-[32px] min-w-0 flex-1 resize-none bg-transparent px-1.5 py-1.5 text-base text-zinc-100 placeholder-zinc-500 focus:outline-none sm:text-[13px]" />
        <button onClick={() => void send()} disabled={busy || (!text.trim() && !target)}
          aria-label={target ? 'Hand it over' : 'Ask'} title={target ? 'Hand it over' : 'Ask'}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-600 text-white transition-colors hover:bg-violet-500 disabled:opacity-30">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
        </button>
      </div>
      {error && <p className="mt-1.5 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
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
  if (!agent) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-5 py-4 text-xs text-zinc-500">
        <span className="font-medium text-zinc-300">Daily summary</span> — pick an agent for your Line above (One agent's chat) and it can sum the Line up every morning.
      </div>
    )
  }
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
  const check = 'flex items-center gap-1.5 text-xs text-zinc-400'
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-5 py-4">
      <div className="flex items-start gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-zinc-800 bg-gradient-to-br from-amber-500/20 to-orange-500/20">
          <Sunrise className="h-5 w-5 text-zinc-200" />
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-zinc-100">Daily summary</h3>
          <p className="mt-0.5 text-xs text-zinc-500">{agent} sums up your Line — what waits for you first, then what failed, then results.</p>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <label className={check}>
              At
              <input type="time" value={time} onChange={(e) => setTime(e.target.value)} aria-label="Summary time"
                className="rounded-md border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none" />
            </label>
            <label className={check}><input type="checkbox" checked={telegram} onChange={(e) => setTelegram(e.target.checked)} className="h-3.5 w-3.5 accent-violet-600" /> Also Telegram</label>
            <label className={check}><input type="checkbox" checked={email} onChange={(e) => setEmail(e.target.checked)} className="h-3.5 w-3.5 accent-violet-600" /> Also email</label>
            <button onClick={() => void add()} disabled={busy || !time}
              className="ml-auto flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40">
              {busy && <Loader2 className="h-3 w-3 animate-spin" />} Set it up
            </button>
          </div>
          {note && <p className="mt-2 text-xs text-emerald-700 dark:text-emerald-300">{note}</p>}
          {error && <p className="mt-2 text-xs text-red-600 dark:text-red-400">{error}</p>}
        </div>
      </div>
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
  const btn = 'flex h-8 items-center gap-1.5 rounded-lg px-3.5 text-xs font-medium transition-colors disabled:opacity-50'
  return (
    <div className="mt-2.5">
      <div className="flex items-center gap-2">
        <button disabled={busy !== null} onClick={() => void answer(true)} className={`${btn} bg-emerald-600 text-white hover:bg-emerald-500`}>
          {busy === 'approve' && <Loader2 className="h-3.5 w-3.5 animate-spin" />} Approve
        </button>
        <button disabled={busy !== null} onClick={() => void answer(false)} className={`${btn} bg-zinc-800 text-zinc-200 hover:bg-zinc-700`}>
          {busy === 'reject' && <Loader2 className="h-3.5 w-3.5 animate-spin" />} Reject
        </button>
      </div>
      {error && <p className="mt-1.5 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}
