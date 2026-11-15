import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  Activity, ChevronDown, ChevronRight, Download, EyeOff, HelpCircle, Loader2, RotateCw, Search, X,
} from 'lucide-react'
import { useNotificationStore } from '../../stores/notificationStore'
import { fmtRelative } from '../../services/schedulerApi'
import {
  OUTCOMES, RISKS, toolAuditApi,
  type AuditScope, type CallQuery, type Facets, type Outcome, type Risk, type Summary, type ToolCall, type WhyAnswer,
} from '../../services/toolAuditApi'
import { FullscreenButton, Highlight, libraryIconBtn as iconBtn } from '../library/libraryParts'
import { Pill, type Tone } from '../console/parts'

// The tool-call register: every tool call the agents made — which agent, when,
// which tool, with what, what started it and how it ended. The kiosk opens it
// for the user's own agents (Settings → Activity, an agent's "⋯" → Activity);
// the Admin console's Audit page for everyone's.

const PAGE = 100

type Range = '1h' | '24h' | '7d' | '30d' | 'all'
const RANGES: { id: Range; label: string; seconds: number }[] = [
  { id: '1h', label: 'Last hour', seconds: 3600 },
  { id: '24h', label: 'Last 24 hours', seconds: 86400 },
  { id: '7d', label: 'Last 7 days', seconds: 7 * 86400 },
  { id: '30d', label: 'Last 30 days', seconds: 30 * 86400 },
  { id: 'all', label: 'Everything kept', seconds: 0 },
]

const RISK_TONE: Record<Risk, Tone> = {
  exec: 'amber', send: 'violet', money: 'red', write: 'cyan', read: 'zinc', other: 'zinc',
}
const RISK_LABEL = Object.fromEntries(RISKS.map((r) => [r.id, r.label])) as Record<Risk, string>
const OUTCOME_TONE: Record<Outcome, Tone> = {
  ok: 'emerald', error: 'red', blocked: 'amber', timeout: 'amber', aborted: 'zinc', interrupted: 'zinc', running: 'sky',
}
const OUTCOME_LABEL = Object.fromEntries(OUTCOMES.map((o) => [o.id, o.label])) as Record<Outcome, string>

const selectCls = 'h-9 shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 text-sm text-zinc-300 focus:border-violet-500/60 focus:outline-none'

function since(range: Range): number | undefined {
  const r = RANGES.find((x) => x.id === range)
  return r && r.seconds ? Math.floor(Date.now() / 1000) - r.seconds : undefined
}

function fmtTime(sec: number): string {
  const d = new Date(sec * 1000)
  const today = new Date()
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  return d.toDateString() === today.toDateString() ? time : `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`
}

function fmtDuration(ms: number | null): string {
  if (ms == null) return ''
  if (ms < 1000) return `${ms} ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`
  return `${Math.floor(ms / 60_000)} min ${Math.round((ms % 60_000) / 1000)} s`
}

function shortValue(v: unknown): string {
  if (v == null) return ''
  if (typeof v === 'string') return v
  if (typeof v === 'object' && v && '_head' in v && '_len' in v) {
    const big = v as { _head: string; _len: number }
    return `${big._head}… (${big._len.toLocaleString()} chars)`
  }
  try { return JSON.stringify(v) } catch { return String(v) }
}

/** The parameters in a line: the action first, then the rest. */
function preview(params: Record<string, unknown> | null): string {
  if (!params) return ''
  const entries = Object.entries(params).sort(([a], [b]) => (a === 'action' ? -1 : b === 'action' ? 1 : 0))
  return entries.map(([k, v]) => `${k}: ${shortValue(v).replace(/\s+/g, ' ').slice(0, 120)}`).join(' · ')
}

export function ToolCallRegister({ scope, phone = false, initialAgent = '', initialRef = '', users, embedded = false, onClose }: {
  scope: AuditScope
  phone?: boolean
  /** Open filtered to one agent: its slug, or (a container) the id the app keys it by. */
  initialAgent?: string
  initialRef?: string
  /** Admin: the people to filter by. */
  users?: { id: string; name: string }[]
  /** Inside a page (the Admin console): no header of its own. */
  embedded?: boolean
  onClose?: () => void
}) {
  const notify = useNotificationStore((s) => s.add)
  const [range, setRange] = useState<Range>('7d')
  const [agent, setAgent] = useState(initialAgent)
  const [ref, setRef] = useState(initialRef)
  const [tool, setTool] = useState('')
  const [risk, setRisk] = useState<Risk | ''>('')
  const [outcome, setOutcome] = useState<Outcome | ''>('')
  const [userId, setUserId] = useState('')
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [calls, setCalls] = useState<ToolCall[]>([])
  const [next, setNext] = useState<number | null>(null)
  const [facets, setFacets] = useState<Facets | null>(null)
  const [summary, setSummary] = useState<Summary | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')
  const [open, setOpen] = useState<number | null>(null)
  const [exportOpen, setExportOpen] = useState(false)
  const reqId = useRef(0)

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 300)
    return () => clearTimeout(t)
  }, [query])

  const filters: CallQuery = useMemo(() => ({
    agent, ref, tool, risk, outcome, q: debounced, since: since(range),
    ...(scope === 'admin' && userId ? { user_id: userId } : {}),
  }), [agent, ref, tool, risk, outcome, debounced, range, scope, userId])

  const load = useCallback(async (quiet = false) => {
    const id = ++reqId.current
    if (!quiet) setLoading(true)
    try {
      // The chips count every kind and outcome, whichever one is picked.
      const [page, sum] = await Promise.all([
        toolAuditApi.calls(scope, { ...filters, limit: PAGE }),
        toolAuditApi.summary(scope, { ...filters, risk: '', outcome: '' }),
      ])
      if (id !== reqId.current) return
      setCalls(page.calls)
      setNext(page.next_before)
      setSummary(sum)
      setError('')
    } catch (e) {
      if (id === reqId.current && !quiet) setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (id === reqId.current) setLoading(false)
    }
  }, [filters, scope])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    toolAuditApi.facets(scope, { since: since(range), ...(scope === 'admin' && userId ? { user_id: userId } : {}) })
      .then(setFacets).catch(() => setFacets(null))
  }, [scope, range, userId])

  // While only the first page shows, keep it fresh.
  const loadRef = useRef(load)
  loadRef.current = load
  const paged = calls.length > PAGE
  useEffect(() => {
    if (paged) return
    const t = setInterval(() => loadRef.current(true), 30_000)
    return () => clearInterval(t)
  }, [paged])

  const loadMore = async () => {
    if (!next) return
    setLoadingMore(true)
    try {
      const page = await toolAuditApi.calls(scope, { ...filters, before_id: next, limit: PAGE })
      setCalls((cur) => [...cur, ...page.calls.filter((c) => !cur.some((x) => x.id === c.id))])
      setNext(page.next_before)
    } catch (e) {
      notify('error', 'Could not load more calls', e instanceof Error ? e.message : String(e))
    } finally {
      setLoadingMore(false)
    }
  }

  const download = async (format: 'csv' | 'jsonl') => {
    setExportOpen(false)
    try {
      await toolAuditApi.download(scope, filters, format)
    } catch (e) {
      notify('error', 'Could not export the calls', e instanceof Error ? e.message : String(e))
    }
  }

  useEffect(() => {
    if (embedded || !onClose) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (exportOpen) { setExportOpen(false); return }
      onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [embedded, onClose, exportOpen])

  const terms = useMemo(() => debounced.toLowerCase().split(/\s+/).filter(Boolean), [debounced])
  const filtered = !!(agent || ref || tool || risk || outcome || debounced || userId)
  const clear = () => { setAgent(''); setRef(''); setTool(''); setRisk(''); setOutcome(''); setQuery(''); setUserId('') }
  const agentName = (slug: string) => facets?.agents.find((a) => a.slug === slug)?.name || slug

  const exportButton = (
    <div className="relative">
      <button onClick={() => setExportOpen((v) => !v)} className={iconBtn} title="Export these calls" aria-label="Export these calls" aria-expanded={exportOpen}>
        <Download className="h-4 w-4" />
      </button>
      {exportOpen && (
        <div className="absolute right-0 top-full z-20 mt-1 w-44 overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 py-1 shadow-xl">
          <button onClick={() => download('csv')} className="block w-full px-3 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-800">CSV (spreadsheet)</button>
          <button onClick={() => download('jsonl')} className="block w-full px-3 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-800">JSON lines</button>
        </div>
      )}
    </div>
  )

  const header = embedded ? null : (
    <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/15 text-violet-600 dark:text-violet-400">
        <Activity className="h-[18px] w-[18px]" />
      </span>
      <div className="min-w-0 flex-1">
        <h2 className="truncate text-base font-semibold text-zinc-100">Activity</h2>
        {!phone && <p className="truncate text-xs text-zinc-500">Every tool your agents used — when, with what, and how it went.</p>}
      </div>
      <button onClick={() => load()} className={iconBtn} title="Refresh" aria-label="Refresh">
        <RotateCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
      </button>
      {exportButton}
      <FullscreenButton />
      {onClose && (
        <button onClick={onClose} className={iconBtn} title="Close" aria-label="Close">
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  )

  const toolbar = (
    <div className="flex flex-col gap-2 border-b border-zinc-800 px-3 py-2.5 wide:px-4">
      <div className="flex items-center gap-2">
        <label className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={phone ? 'Search calls…' : 'Search tools, parameters, errors…'}
            aria-label="Search calls"
            className="h-9 w-full rounded-lg border border-zinc-800 bg-zinc-900 pl-9 pr-3 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
          />
        </label>
        <select value={range} onChange={(e) => setRange(e.target.value as Range)} aria-label="Time" className={selectCls}>
          {RANGES.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}
        </select>
        {embedded && (
          <>
            <button onClick={() => load()} className={iconBtn} title="Refresh" aria-label="Refresh">
              <RotateCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            {exportButton}
          </>
        )}
      </div>
      <div className="-mx-1 flex items-center gap-2 overflow-x-auto px-1 pb-0.5">
        {users && (
          <select value={userId} onChange={(e) => setUserId(e.target.value)} aria-label="Person" className={selectCls}>
            <option value="">Everyone</option>
            {users.map((u) => <option key={u.id} value={u.id}>{u.name}</option>)}
          </select>
        )}
        <select
          value={agent}
          onChange={(e) => { setAgent(e.target.value); setRef('') }}
          aria-label="Agent"
          className={selectCls}
        >
          <option value="">{ref && !agent ? 'This agent' : 'All agents'}</option>
          {agent && !facets?.agents.some((a) => a.slug === agent) && <option value={agent}>{agent}</option>}
          {facets?.agents.map((a) => <option key={a.slug} value={a.slug}>{a.name}</option>)}
        </select>
        <select value={tool} onChange={(e) => setTool(e.target.value)} aria-label="Tool" className={selectCls}>
          <option value="">All tools</option>
          {tool && !facets?.tools.some((t) => t.tool === tool) && <option value={tool}>{tool}</option>}
          {facets?.tools.map((t) => <option key={t.tool} value={t.tool}>{t.tool} ({t.count})</option>)}
        </select>
        <select value={risk} onChange={(e) => setRisk(e.target.value as Risk | '')} aria-label="Kind" className={selectCls}>
          <option value="">Any kind</option>
          {RISKS.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}
        </select>
        <select value={outcome} onChange={(e) => setOutcome(e.target.value as Outcome | '')} aria-label="Outcome" className={selectCls}>
          <option value="">Any outcome</option>
          {OUTCOMES.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
        </select>
        {filtered && (
          <button onClick={clear} className="h-9 shrink-0 rounded-lg px-2.5 text-sm text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            Clear
          </button>
        )}
      </div>
      {summary && summary.total > 0 && (
        <SummaryChips summary={summary} risk={risk} outcome={outcome} onRisk={setRisk} onOutcome={setOutcome} />
      )}
    </div>
  )

  let body
  if (loading && !calls.length) {
    body = <div className="flex justify-center py-16 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
  } else if (error) {
    body = <div className="px-5 py-12 text-center text-sm text-red-600 dark:text-red-400">{error}</div>
  } else if (!calls.length) {
    body = (
      <div className="flex flex-col items-center gap-2 px-6 py-16 text-center">
        <Activity className="h-8 w-8 text-zinc-700" />
        <div className="text-sm font-medium text-zinc-300">{filtered ? 'No calls match' : 'No tool calls yet'}</div>
        <p className="max-w-sm text-xs leading-relaxed text-zinc-500">
          {filtered
            ? 'Try another time range or clear the filters.'
            : 'Agents list every tool they use here, as they use it. An agent that was already running starts after its next restart.'}
        </p>
      </div>
    )
  } else {
    body = (
      <ul className="divide-y divide-zinc-800/70">
        {calls.map((c) => (
          <CallRow
            key={c.id}
            call={c}
            phone={phone}
            admin={scope === 'admin'}
            terms={terms}
            open={open === c.id}
            onToggle={() => setOpen((cur) => (cur === c.id ? null : c.id))}
            onAgent={() => { setAgent(c.agent.slug); setRef('') }}
          />
        ))}
        {next && (
          <li className="flex justify-center py-4">
            <button
              onClick={loadMore}
              disabled={loadingMore}
              className="flex items-center gap-2 rounded-lg border border-zinc-800 px-4 py-2 text-sm text-zinc-300 hover:bg-zinc-800 disabled:opacity-60"
            >
              {loadingMore && <Loader2 className="h-4 w-4 animate-spin" />}Older calls
            </button>
          </li>
        )}
      </ul>
    )
  }

  return (
    <div className={`flex min-h-0 flex-col ${embedded ? '' : 'h-full'}`}>
      {header}
      {toolbar}
      {agent && !phone && (
        <div className="border-b border-zinc-800 px-4 py-1.5 text-xs text-zinc-500 wide:px-5">
          Showing <span className="font-medium text-zinc-300">{agentName(agent)}</span>
        </div>
      )}
      <div className={embedded ? '' : 'min-h-0 flex-1 overflow-y-auto'}>{body}</div>
    </div>
  )
}

function SummaryChips({ summary, risk, outcome, onRisk, onOutcome }: {
  summary: Summary
  risk: Risk | ''
  outcome: Outcome | ''
  onRisk: (r: Risk | '') => void
  onOutcome: (o: Outcome | '') => void
}) {
  const chip = (on: boolean, tone: Tone, label: string, n: number, click: () => void) => (
    <button key={label} onClick={click} aria-pressed={on} className={`rounded-full ${on ? 'ring-2 ring-violet-500/50' : ''}`}>
      <Pill tone={tone}>{label} · {n.toLocaleString()}</Pill>
    </button>
  )
  const outs = (['blocked', 'error', 'timeout', 'interrupted'] as Outcome[]).filter((o) => summary.outcomes[o])
  const risks = (['exec', 'send', 'money'] as Risk[]).filter((r) => summary.risks[r])
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs text-zinc-500">
      <span className="mr-1 tabular-nums">{summary.total.toLocaleString()} call{summary.total === 1 ? '' : 's'}</span>
      {outs.map((o) => chip(outcome === o, OUTCOME_TONE[o], OUTCOME_LABEL[o], summary.outcomes[o] || 0,
        () => onOutcome(outcome === o ? '' : o)))}
      {risks.map((r) => chip(risk === r, RISK_TONE[r], RISK_LABEL[r], summary.risks[r] || 0,
        () => onRisk(risk === r ? '' : r)))}
    </div>
  )
}

function CallRow({ call: c, phone, admin, terms, open, onToggle, onAgent }: {
  call: ToolCall
  phone: boolean
  admin: boolean
  terms: string[]
  open: boolean
  onToggle: () => void
  onAgent: () => void
}) {
  const line = c.hidden ? '' : preview(c.params)
  const who = c.member ? c.speaker.label || 'Member' : ''
  const started = [c.source_label || c.source, who && (c.speaker.you ? 'you' : who)].filter(Boolean).join(' · ')
  return (
    <li>
      <button onClick={onToggle} aria-expanded={open} className="flex w-full items-start gap-3 px-4 py-2.5 text-left hover:bg-zinc-900/60 wide:px-5">
        <span className="mt-0.5 text-zinc-600">{open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}</span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="font-mono text-[13px] font-medium text-zinc-100">{c.tool}</span>
            <Pill tone={RISK_TONE[c.risk] || 'zinc'}>{RISK_LABEL[c.risk] || c.risk}</Pill>
            {c.outcome !== 'ok' && <Pill tone={OUTCOME_TONE[c.outcome] || 'zinc'}>{OUTCOME_LABEL[c.outcome] || c.outcome}</Pill>}
            {c.hidden && <Pill icon={EyeOff}>{who || 'Member'}</Pill>}
            <span className="ml-auto shrink-0 text-[11px] tabular-nums text-zinc-500" title={new Date(c.at * 1000).toString()}>
              {phone ? fmtRelative(c.at) : fmtTime(c.at)}
            </span>
          </span>
          {line && (
            <span className="mt-0.5 block truncate font-mono text-xs text-zinc-400">
              <Highlight text={line} terms={terms} />
            </span>
          )}
          <span className="mt-0.5 block truncate text-[11px] text-zinc-500">
            {c.agent.name}
            {admin && !c.owner.you && c.owner.name ? ` · ${c.owner.name}` : ''}
            {started ? ` · ${started}` : ''}
            {c.duration_ms != null ? ` · ${fmtDuration(c.duration_ms)}` : ''}
          </span>
        </span>
      </button>
      {open && <CallDetail call={c} admin={admin} onAgent={onAgent} />}
    </li>
  )
}

function CallDetail({ call: c, admin, onAgent }: { call: ToolCall; admin: boolean; onAgent: () => void }) {
  const rows: [string, ReactNode][] = [
    ['Started', new Date(c.at * 1000).toLocaleString()],
    ['Took', c.outcome === 'running' ? 'Still running' : fmtDuration(c.duration_ms) || '—'],
    ['Outcome', OUTCOME_LABEL[c.outcome] || c.outcome],
    ['Agent', (
      <button onClick={onAgent} className="text-violet-600 hover:underline dark:text-violet-400" title="Only this agent's calls">
        {c.agent.name}
      </button>
    )],
  ]
  if (!c.owner.you && c.owner.name) rows.push(['Agent owner', c.owner.name])
  if (c.source_label || c.source) rows.push(['Started from', c.source_label || c.source])
  if (c.member) rows.push(['Speaking', c.speaker.you ? 'You' : c.speaker.label || 'Member'])
  if (c.session_id) rows.push(['Session', <code key="s" className="text-[11px]">{c.session_id}</code>])
  // An allowed call keeps who allowed it (approval cards) as its reason.
  if (c.reason) rows.push([c.outcome === 'blocked' ? 'Why refused' : c.outcome === 'ok' ? 'Note' : 'Error', c.reason])
  rows.push(['Record', (
    <span key="r" className="font-mono text-[11px] text-zinc-500" title={admin ? c.hash : undefined}>
      #{c.seq} · run {c.boot.slice(0, 8)}{admin ? ` · ${c.hash.slice(0, 12)}…` : ''}
    </span>
  )])
  return (
    <div className="space-y-3 bg-zinc-900/40 px-4 pb-4 pl-11 pt-1 wide:px-5 wide:pl-12">
      <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-zinc-500">{k}</dt>
            <dd className="min-w-0 break-words text-zinc-300">{v}</dd>
          </div>
        ))}
      </dl>
      <WhyPanel call={c} admin={admin} />
      {c.hidden ? (
        <p className="flex items-center gap-2 rounded-lg border border-zinc-800 px-3 py-2 text-xs text-zinc-500">
          <EyeOff className="h-3.5 w-3.5 shrink-0" />
          A member made this call on your shared agent. Its parameters can hold their own data, so only they and the deck's admins see them.
        </p>
      ) : (
        <div>
          <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Parameters</div>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-[12px] leading-relaxed text-zinc-300">
            {JSON.stringify(c.params ?? {}, null, 2)}
          </pre>
          <p className="mt-1 text-[11px] text-zinc-600">Passwords, keys and tokens are masked; long values keep their start, length and fingerprint.</p>
        </div>
      )}
    </div>
  )
}

// "Why did this happen?": the deck's deciders re-run on this call's recorded
// facts (why.py) — no model call. A layer that would decide differently today
// says so, so a setting's effect shows before the next call.
function WhyPanel({ call: c, admin }: { call: ToolCall; admin: boolean }) {
  const [why, setWhy] = useState<WhyAnswer | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const ask = async () => {
    setBusy(true)
    setError('')
    try {
      setWhy(await toolAuditApi.why(admin ? 'admin' : 'mine', c.id))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setBusy(false)
    }
  }
  if (!why) {
    return (
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" onClick={ask} disabled={busy}
          className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:border-zinc-700 hover:text-zinc-100 disabled:opacity-50">
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <HelpCircle className="h-3.5 w-3.5" />}
          Why did this happen?
        </button>
        {error && <span className="text-xs text-red-600 dark:text-red-400">{error}</span>}
      </div>
    )
  }
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3">
      <p className="text-sm font-medium text-zinc-100">{why.summary}</p>
      <ol className="mt-2 space-y-2">
        {why.layers.map((l) => (
          <li key={l.layer} className="text-xs leading-relaxed">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium text-zinc-300">{l.title}</span>
              {l.changed && <Pill tone="amber">Different today</Pill>}
            </div>
            <p className="text-zinc-400">{l.then}</p>
            {l.today && <p className={l.changed ? 'text-amber-700 dark:text-amber-300' : 'text-zinc-500'}>{l.today}</p>}
          </li>
        ))}
      </ol>
      <p className="mt-2 text-[11px] text-zinc-600">Worked out from what the register recorded and today's settings; no AI involved.</p>
    </div>
  )
}
