import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  ArrowUp, Bot, CheckCheck, ChevronDown, ChevronLeft, CircleHelp, ClipboardCheck, Inbox, Lightbulb, Loader2, Mail,
  MessageCircle, MessagesSquare, Network, Radio, Receipt, Server, Share2, SlidersHorizontal, Sunrise, Waypoints, X,
  type LucideIcon,
} from 'lucide-react'
import {
  addLineSummary, answeredByReading, askLine, fetchLine, fetchThread, handLineItem, LINE_PAGE, markAllLineRead,
  markThreadRead, type LineItem,
} from '../../services/line'
import { OPEN_COURSE_EVENT } from '../../services/course'
import { fetchWaitingProposals, resolveProposal, type WaitingProposal } from '../../services/notifications'
import { useLineStore } from '../../stores/lineStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { untrustedUrlTransform } from '../../utils/untrustedContent'
import NotificationsConnection from '../connections/NotificationsConnection'
import { FullscreenButton } from '../library/libraryParts'

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
/** The user's own words on the Line: a question asked, an item handed over. */
const isMine = (item: LineItem) => item.kind === 'talk' && item.title.startsWith('You ')
/** Who said an item, in a thread's last line. */
const actorOf = (item: LineItem) => (isMine(item) ? 'You' : whoOf(item) || KIND_WORD[item.kind] || 'Deck')
/** An item in one line: what was said, for what was talked; else its headline. */
const snippetOf = (item: LineItem) => (item.kind === 'talk'
  ? plainLines(item.body)[0] || item.title : headlineOf(item).headline)
/** A later headline without what it shares with the thread's topic
 *  ("Taskboard: Q4 update — step 2 done" under "Taskboard: Q4 update — …"). */
function past(topic: string, text: string): string {
  let n = 0
  while (n < topic.length && topic[n] === text[n]) n++
  const cut = text.lastIndexOf(' — ', n - 1)
  return cut > 0 && cut + 3 < text.length ? text.slice(cut + 3) : text
}

/** What belongs together on the Line: an item, its hand-overs and their
 *  answers, or everything about one Taskboard task — newest first. */
type Thread = { id: string; items: LineItem[]; root: LineItem; latest: LineItem }
function groupThreads(items: LineItem[]): Thread[] {
  const byId = new Map<string, LineItem[]>()
  for (const item of items) {
    const id = item.thread_id || item.id
    const list = byId.get(id)
    if (list) list.push(item)
    else byId.set(id, [item])
  }
  return [...byId].map(([id, list]) => ({ id, items: list, root: list[list.length - 1], latest: list[0] }))
}
/** What a thread is about: its first item's headline — a question's own words,
 *  or (when the first item is out of reach) the item a hand-over names. */
function topicOf(root: LineItem): string {
  if (isMine(root)) {
    return /^You handed “(.+)” to /.exec(root.title)?.[1] || plainLines(root.body)[0] || root.title
  }
  return headlineOf(root).headline
}
const threadWaits = (t: Thread) => t.items.some((i) => i.needs_you && !i.resolved)

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
function clock(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  return sameDay(d, new Date()) ? time : `${shortTime(iso)}, ${time}`
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

/** The Line: what the deck and the agents said on their own, newest first and
 *  in threads — a list, the open thread beside it (on a phone, in its place) as
 *  a conversation, and one box to ask an agent about the Line or hand it the
 *  open thread. The bell holds the part of it that needs you. */
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
  const [threadId, setThreadId] = useState<string | null>(null)
  // Where the Line reaches you besides Flight Deck, in the open thread's place.
  const [channels, setChannels] = useState(false)
  // The agent that answers on the Line, and the user's agents.
  const [lineAgent, setLineAgent] = useState({ slug: '', name: '' })
  const [agents, setAgents] = useState<Agent[]>([])
  // A question or a hand-over waiting for its answer, and its thread.
  const [asked, setAsked] = useState<{ id: string; agent: string; at: number; thread: string } | null>(null)
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

  // A question asked: read the Line every few seconds until its answer comes
  // (it joins the question's thread, which is open — or opens, unless the user
  // went on to something else).
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
    const answer = q && items.find((i) => i.kind === 'talk' && !isMine(i) && i.source === asked.agent
      && i.created_at > q.created_at && (i.thread_id || i.id) === asked.thread)
    if (!answer) return
    setAsked(null)
    setThreadId((cur) => cur ?? asked.thread)
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
  const readAll = async () => {
    setItems((prev) => prev.map((i) => ({
      ...i, read: true, resolved: i.resolved || answeredByReading(i.needs_you, i.ref_type),
    })))
    await markAllLineRead()
    syncBell()
  }
  const back = () => { setThreadId(null); setChannels(false) }
  const answered = () => {
    loadProposals()
    void loadRef.current()
    syncBell()
  }

  const threads = useMemo(() => groupThreads(items), [items])
  const thread = useMemo(() => threads.find((t) => t.id === threadId) || null, [threads, threadId])
  // An open thread is read — opened by a click, or one that opened by itself
  // or got a new item while open.
  const unreadKey = thread ? thread.items.filter((i) => !i.read).map((i) => i.id).join(',') : ''
  useEffect(() => {
    if (!threadId || !unreadKey) return
    const n = unreadKey.split(',').length
    setItems((prev) => prev.map((i) => ((i.thread_id || i.id) === threadId && !i.read ? {
      ...i, read: true, resolved: i.resolved || answeredByReading(i.needs_you, i.ref_type),
    } : i)))
    setCounts({ unread: Math.max(0, useLineStore.getState().unread - n), waiting: useLineStore.getState().waiting })
    void markThreadRead(threadId).then(syncBell)
  }, [threadId, unreadKey, setCounts])

  const chips = useMemo(() => Object.keys(kinds).filter((k) => seenKinds.includes(k)), [kinds, seenKinds])
  const detailShown = channels || threadId !== null
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
            {threads.map((t, i) => {
              const day = dayLabel(t.latest.created_at)
              const showDay = i === 0 || dayLabel(threads[i - 1].latest.created_at) !== day
              return (
                <li key={t.id}>
                  {showDay && (
                    <div className="px-3 pb-1 pt-3 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">{day}</div>
                  )}
                  <ThreadRow thread={t} selected={t.id === threadId} proposals={proposals}
                    onOpen={() => { setChannels(false); setThreadId(t.id) }} />
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
  ) : threadId ? (
    <ThreadView key={threadId} threadId={threadId} known={thread?.items || []} proposals={proposals}
      pending={asked && asked.thread === threadId ? asked.agent : ''}
      onBack={wide ? undefined : back} onAnswered={answered} />
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
        <FullscreenButton className={iconBtn()} />
        {onClose && (
          <button onClick={onClose} className={iconBtn()} title="Close" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <div className="flex min-h-0 flex-1">
        {wide ? (
          <>
            <div className="flex w-[min(400px,40%)] shrink-0 flex-col border-r border-zinc-800">{list}</div>
            <div className="flex min-w-0 flex-1 flex-col">{detail}</div>
          </>
        ) : detailShown ? (
          <div className="flex min-w-0 flex-1 flex-col">{detail}</div>
        ) : list}
      </div>

      {!channels && (
        <Composer agents={agents} lineAgent={lineAgent} compact={!wide}
          talkedTo={thread?.items.find((i) => i.kind === 'talk')?.source || ''}
          target={thread?.latest || null} about={thread ? topicOf(thread.root) : ''}
          more={!!thread && (thread.items.length > 1 || isMine(thread.root))}
          pending={asked && asked.thread !== threadId ? asked.agent : ''}
          onDetach={() => setThreadId(null)}
          onSent={(id, agent, ask) => {
            const t = ask ? id : threadId || id
            setAsked({ id, agent, at: Date.now(), thread: t })
            if (ask) { setChannels(false); setThreadId(id) }
            void loadRef.current()
          }}
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
const AVATAR_BOX = { sm: 'h-7 w-7 rounded-lg text-[10px]', md: 'h-9 w-9 rounded-xl text-[11px]', lg: 'h-10 w-10 rounded-xl text-xs' }
function Avatar({ item, size = 'md' }: { item: LineItem; size?: 'sm' | 'md' | 'lg' }) {
  const who = whoOf(item)
  const box = AVATAR_BOX[size]
  if (who) {
    return <span className={`flex shrink-0 items-center justify-center font-semibold ${box} ${hueOf(who)}`}>{initials(who)}</span>
  }
  const Icon = KIND_ICON[item.kind] || Server
  return (
    <span className={`flex shrink-0 items-center justify-center bg-zinc-800 text-zinc-400 ${box}`}>
      <Icon className={size === 'lg' ? 'h-5 w-5' : size === 'sm' ? 'h-3.5 w-3.5' : 'h-4 w-4'} />
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

/** A thread's state, in one pill: its last item failed, a proposal in it
 *  waits, or something in it needs the user. */
function ThreadPill({ thread, proposals }: { thread: Thread; proposals: Map<string, WaitingProposal> | null }) {
  const pill = 'shrink-0 rounded-md px-1.5 py-px text-[10px] font-semibold'
  if (isFailed(thread.latest)) return <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-400`}>Failed</span>
  const proposal = thread.items.find((i) => i.ref_type === 'autonomy_action' && i.needs_you && !i.resolved)
  if (proposal) {
    return <span className={`${pill} bg-amber-500/15 text-amber-700 dark:text-amber-300`}>{proposals?.get(proposal.ref_id) ? 'Approve?' : 'Waiting'}</span>
  }
  if (threadWaits(thread)) return <span className={`${pill} bg-amber-500/15 text-amber-700 dark:text-amber-300`}>Needs you</span>
  return null
}

function ThreadRow({ thread, selected, onOpen, proposals }: {
  thread: Thread; selected: boolean; onOpen: () => void; proposals: Map<string, WaitingProposal> | null
}) {
  const { root, latest } = thread
  const who = whoOf(root)
  const n = thread.items.length
  const unread = thread.items.some((i) => !i.read)
  // One item: what follows its headline; a thread: who said the last thing, and what.
  const last = n > 1 ? `${actorOf(latest)}: ${past(topicOf(root), snippetOf(latest))}` : headlineOf(root).preview
  return (
    <button onClick={onOpen} aria-current={selected || undefined}
      className={`group relative flex w-full items-start gap-3 rounded-xl px-3 py-2.5 text-left transition-colors ${
        selected ? 'bg-violet-500/10 ring-1 ring-inset ring-violet-500/25' : 'hover:bg-zinc-900'}`}>
      {unread && <span className="absolute left-1 top-[22px] h-1.5 w-1.5 rounded-full bg-violet-500" aria-label="Unread" />}
      <Avatar item={root} />
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-1.5">
          <span className={`min-w-0 truncate text-xs ${unread ? 'font-semibold text-zinc-200' : 'text-zinc-400'}`}>
            {who || KIND_WORD[root.kind] || 'Deck'}
          </span>
          {who && <span className="shrink-0 text-[11px] text-zinc-600">· {KIND_WORD[root.kind] || root.kind}</span>}
          <span className="ml-auto shrink-0 pl-2 text-[11px] tabular-nums text-zinc-500">{shortTime(latest.created_at)}</span>
        </span>
        <span className="mt-0.5 flex items-center gap-1.5">
          <span className={`min-w-0 truncate text-[13px] ${unread ? 'font-medium text-zinc-100' : 'text-zinc-300'}`}>{topicOf(root)}</span>
          {n > 1 && (
            <span className="flex shrink-0 items-center gap-0.5 rounded-md bg-zinc-800 px-1.5 py-px text-[10px] font-semibold text-zinc-400"
              title={`${n} in this thread`}>
              <MessagesSquare className="h-3 w-3" />{n}
            </span>
          )}
          <ThreadPill thread={thread} proposals={proposals} />
        </span>
        {last && <span className="mt-0.5 block truncate text-xs text-zinc-500">{last}</span>}
      </span>
    </button>
  )
}

/** A thread as a conversation: the item it started with, then what followed —
 *  the user's hand-overs on the right, the agents' answers on the left, other
 *  items about the same task as cards — and who's still on it. */
function ThreadView({ threadId, known, proposals, pending, onBack, onAnswered }: {
  threadId: string; known: LineItem[]; proposals: Map<string, WaitingProposal> | null; pending: string
  onBack?: () => void; onAnswered: () => void
}) {
  const [items, setItems] = useState<LineItem[] | null>(null)
  // Read again whenever the list sees this thread change (a new item, an answer, a read).
  const changes = known.map((i) => `${i.id}${i.read ? 'r' : ''}${i.resolved ? 'x' : ''}`).join(',')
  useEffect(() => {
    let gone = false
    fetchThread(threadId).then((rows) => { if (!gone) setItems(rows) }).catch(() => {})
    return () => { gone = true }
  }, [threadId, changes])
  const shown = items ?? [...known].reverse()
  // Something new at the end (an answer, a hand-over, who's on it): bring it into view.
  const scroller = useRef<HTMLDivElement>(null)
  const seen = useRef(0)
  const length = shown.length + (pending ? 1 : 0)
  useEffect(() => {
    if (seen.current && length > seen.current) {
      scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' })
    }
    seen.current = length
  }, [length])

  const root = shown[0]
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {onBack && (
        <button onClick={onBack} className="flex shrink-0 items-center gap-1 border-b border-zinc-800 px-3 py-2 text-xs font-medium text-zinc-400 hover:text-zinc-200">
          <ChevronLeft className="h-4 w-4" /> Line
        </button>
      )}
      <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
        {!root ? (
          <div className="flex items-center justify-center gap-2 py-16 text-xs text-zinc-500">
            <Loader2 className="h-4 w-4 animate-spin" /> Opening…
          </div>
        ) : (
          <div className="mx-auto max-w-2xl space-y-5 px-5 py-5 sm:px-6">
            <ThreadStart item={root} proposal={root.ref_type === 'autonomy_action' ? proposals?.get(root.ref_id) : undefined}
              onAnswered={onAnswered} />
            {shown.slice(1).map((item) => (
              <ThreadEntry key={item.id} item={item} onAnswered={onAnswered}
                proposal={item.ref_type === 'autonomy_action' ? proposals?.get(item.ref_id) : undefined} />
            ))}
            {pending && (
              <div className="flex items-center gap-3">
                <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-[10px] font-semibold ${hueOf(pending)}`}>
                  {initials(pending)}
                </span>
                <span className="flex items-center gap-2 rounded-2xl rounded-tl-md bg-zinc-900 px-3.5 py-2 text-xs text-zinc-400 ring-1 ring-zinc-800">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> {pending} is on it…
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/** The item a thread starts with: a header, its heading, its text — or a
 *  question the user asked. */
function ThreadStart({ item, proposal, onAnswered }: { item: LineItem; proposal?: WaitingProposal; onAnswered: () => void }) {
  if (isMine(item)) {
    return (
      <div>
        <div className="text-[11px] text-zinc-500">You asked {item.source} · {fullTime(item.created_at)}</div>
        <h3 className="mt-1.5 text-base font-semibold leading-snug text-zinc-100">{topicOf(item)}</h3>
      </div>
    )
  }
  const who = whoOf(item)
  const full = item.body.replace(TASK_LINK_LINE, '').trim()
  // A generic title gives way to the text's own first line, as the heading.
  const generic = GENERIC_TITLE.test(item.title) && !!full
  const heading = generic ? headlineOf(item).headline : item.title
  const body = generic ? full.replace(/^[^\n]*\n*/, '').trim() : full
  return (
    <div>
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
      <ProposalCard item={item} proposal={proposal} onAnswered={onAnswered} />
      <CourseReviewCard item={item} />
      {body && (
        <article className={`mt-3 ${PROSE}`}>
          <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={LINE_MD}>{keepBreaks(body)}</Markdown>
        </article>
      )}
      {item.trust === 'untrusted' && (
        <p className="mt-3 text-[11px] text-zinc-500">This quotes someone else's words — links aren't followed and pictures aren't shown.</p>
      )}
    </div>
  )
}

/** One later item in a thread. */
function ThreadEntry({ item, proposal, onAnswered }: { item: LineItem; proposal?: WaitingProposal; onAnswered: () => void }) {
  const body = item.body.replace(TASK_LINK_LINE, '').trim()
  if (isMine(item)) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%]">
          <div className="mb-1 text-right text-[11px] text-zinc-500">You → {item.source} · {clock(item.created_at)}</div>
          <div className="whitespace-pre-wrap rounded-2xl rounded-tr-md bg-violet-500/15 px-3.5 py-2 text-[13px] leading-relaxed text-zinc-100">
            {body || 'Look into it.'}
          </div>
        </div>
      </div>
    )
  }
  if (item.kind === 'talk') {
    const failed = /couldn't answer/.test(item.title)
    return (
      <div className="flex gap-3">
        <Avatar item={item} size="sm" />
        <div className="min-w-0 flex-1">
          <div className="text-[11px] text-zinc-500"><span className="font-medium text-zinc-300">{whoOf(item)}</span> · {clock(item.created_at)}</div>
          {failed ? (
            <p className="mt-1 text-xs text-zinc-500">Couldn't answer — {body}</p>
          ) : (
            <article className={`mt-1 ${PROSE}`}>
              <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={LINE_MD}>{keepBreaks(body)}</Markdown>
            </article>
          )}
        </div>
      </div>
    )
  }
  const who = whoOf(item)
  const generic = GENERIC_TITLE.test(item.title) && !!body
  const heading = generic ? headlineOf(item).headline : item.title
  const rest = generic ? body.replace(/^[^\n]*\n*/, '').trim() : body
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
      <div className="flex items-center gap-2.5">
        <Avatar item={item} size="sm" />
        <div className="min-w-0 flex-1">
          <div className="truncate text-xs font-medium text-zinc-200">
            {who || KIND_WORD[item.kind] || 'Flight Deck'}
            {who && <span className="font-normal text-zinc-500"> · {KIND_WORD[item.kind] || item.kind}</span>}
          </div>
          <div className="text-[11px] text-zinc-500">{clock(item.created_at)}</div>
        </div>
        <StatusPill item={item} proposal={proposal} />
      </div>
      <h4 className="mt-2.5 text-sm font-semibold leading-snug text-zinc-100">{heading}</h4>
      <ProposalCard item={item} proposal={proposal} onAnswered={onAnswered} />
      <CourseReviewCard item={item} />
      {rest && (
        <article className={`mt-1.5 ${PROSE}`}>
          <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={LINE_MD}>{keepBreaks(rest)}</Markdown>
        </article>
      )}
    </div>
  )
}

/** Approve / Reject on a proposal still waiting (a Taskboard next step is
 *  read and approved on its task). */
function ProposalCard({ item, proposal, onAnswered }: { item: LineItem; proposal?: WaitingProposal; onAnswered: () => void }) {
  if (!(item.needs_you && !item.resolved && item.ref_type === 'autonomy_action')) return null
  return (
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
  )
}

/** A goal due for review: answered on Course (Still on course, edit, pause,
 *  done or drop), not here. */
function CourseReviewCard({ item }: { item: LineItem }) {
  if (!(item.needs_you && !item.resolved && item.ref_type === 'course_review')) return null
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3.5 py-3">
      <span className="flex-1 text-xs text-amber-800 dark:text-amber-200">Is this goal still on course?</span>
      <button
        onClick={() => window.dispatchEvent(new CustomEvent(OPEN_COURSE_EVENT))}
        className="rounded-md bg-amber-600 px-3 py-1 text-xs text-white hover:bg-amber-500"
      >
        Open Course
      </button>
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
function Composer({ agents, lineAgent, talkedTo, target, about, more, pending, compact, onDetach, onSent, onSetUp }: {
  agents: Agent[]; lineAgent: { slug: string; name: string }
  /** The agent the open thread last talked with: picked when the thread opens. */
  talkedTo: string
  /** The open thread's newest item (handed over with the thread behind it), and what the thread is about. */
  target: LineItem | null; about: string
  /** The thread has more than its first item. */
  more: boolean
  pending: string
  /** Narrow: the agent pill shows only initials. */
  compact: boolean
  onDetach: () => void; onSent: (id: string, agent: string, ask: boolean) => void; onSetUp: () => void
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
  useEffect(() => {
    const agent = agents.find((a) => a.name === talkedTo)
    if (agent) setSlug(agent.slug)
  }, [talkedTo]) // eslint-disable-line react-hooks/exhaustive-deps -- only when the thread's agent changes
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
      onSent(r.id, r.agent, !target)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
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
            <span className="shrink-0 font-medium">{more ? 'Thread:' : 'Hand over:'}</span>
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
          placeholder={target ? `What should ${name} do ${more ? 'next' : 'with it'}?` : `Ask ${name ? `${name} ` : ''}about your Line…`}
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
