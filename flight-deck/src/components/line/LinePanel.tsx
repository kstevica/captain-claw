import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Bot, CheckCheck, CircleHelp, ClipboardCheck, Lightbulb, Loader2, Mail, MessageCircle, Network, Radio,
  Receipt, Server, Share2, Waypoints, X, type LucideIcon,
} from 'lucide-react'
import { fetchLine, LINE_PAGE, markAllLineRead, markLineRead, type LineItem } from '../../services/line'
import { useLineStore } from '../../stores/lineStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { untrustedUrlTransform } from '../../utils/untrustedContent'

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
              <LineRow key={item.id} item={item} expanded={open.has(item.id)} onToggle={() => toggle(item)} />
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
    </div>
  )
}

function LineRow({ item, expanded, onToggle }: { item: LineItem; expanded: boolean; onToggle: () => void }) {
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
                {item.ref_type === 'autonomy_action' ? 'Waiting — answer it in the bell' : 'Needs you'}
              </span>
            )}
          </div>
          {body && (
            <div className={`mt-0.5 text-[11px] text-zinc-500 ${MD} ${expanded ? '' : 'line-clamp-2'}`}>
              <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={LINE_MD}>{keepBreaks(body)}</Markdown>
            </div>
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
