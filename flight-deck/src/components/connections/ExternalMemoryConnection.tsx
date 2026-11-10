import { useEffect, useId, useMemo, useState, type FormEvent, type ReactNode } from 'react'
import {
  Brain,
  Loader2,
  Check,
  AlertCircle,
  ChevronDown,
  Save,
  Trash2,
  Server,
  ExternalLink,
  RotateCcw,
  Search,
  Plus,
  Pencil,
  KeyRound,
  X,
} from 'lucide-react'
import {
  useExternalMemoryStore,
  type MemoryView,
  type MirrorKind,
  type OutboxItem,
  type ProviderInfo,
  type ReadingPriority,
} from '../../stores/externalMemoryStore'
import CredentialInputs from './CredentialInputs'

// External memories: several per Flight Deck user (per deck with sign-in
// off), all shared by every agent of theirs — their own, plus the one an admin
// connected for everyone (which they can switch off). Each has its own
// reading priority and its own "copied in on its own" choices. Flight Deck
// keeps the credentials and talks to the providers; agents reach them through
// Flight Deck with the `external_memory` tool
// (captain_claw/flight_deck/external_memory/).

type Busy = 'save' | 'test' | 'remove' | 'retry' | 'mirror' | 'priority' | 'enabled' | 'rename' | null
interface Notice { tone: 'ok' | 'warn'; text: string }

const pill = 'inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border'
const input = 'w-full min-w-0 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500 focus:outline-none disabled:opacity-50'
const ghostBtn = 'inline-flex items-center gap-1.5 rounded-md bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-40'
const primaryBtn = 'inline-flex items-center gap-2 rounded-md bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-sm text-white disabled:opacity-50 disabled:cursor-not-allowed shadow-sm'
const dangerBtn = 'inline-flex items-center gap-1.5 rounded-md border border-red-500/30 hover:bg-red-500/10 px-3 py-1.5 text-xs text-red-600 dark:text-red-500 disabled:opacity-50'

const MIRROR_LABELS: Record<MirrorKind, { label: string; hint: string }> = {
  insight: { label: 'Insights', hint: 'Facts, contacts, decisions, preferences and deadlines your agents pick up.' },
  intention: { label: 'Intentions', hint: 'Reminders and follow-ups, once they are active.' },
  lesson: { label: 'Lessons', hint: 'Rules and facts you teach your agents, or approve from what they noticed.' },
  reflection: { label: 'Reflections', hint: "Your agents' own notes on how to work better for you." },
  intuition: { label: 'Intuitions', hint: "Your agents' unchecked hunches and connections." },
}

const PRIORITY_LABELS: Record<ReadingPriority, { label: string; hint: string }> = {
  first: {
    label: 'First — my main memory',
    hint: "Looked up before every reply (up to 8 things), kept even in long conversations, and trusted over your agents' own notes when they disagree.",
  },
  alongside: {
    label: 'Alongside',
    hint: "Looked up before every reply (up to 5 things), next to your agents' own memory.",
  },
  on_demand: {
    label: 'Only when needed',
    hint: 'Nothing is looked up on its own — your agents search it when they decide to.',
  },
}
const PRIORITY_SHORT: Record<ReadingPriority, string> = { first: 'main memory', alongside: 'alongside', on_demand: 'when needed' }

function duration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`
}

function when(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === '') return ''
  const d = typeof value === 'number' ? new Date(value * 1000) : new Date(value)
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function NoticeBox({ notice }: { notice: Notice | null }) {
  if (!notice) return null
  return (
    <div className={'rounded-md border px-3 py-2 text-xs break-words ' + (notice.tone === 'ok'
      ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
      : 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300')}
    >
      {notice.text}
    </div>
  )
}

function memoryPill(m: MemoryView): ReactNode {
  if (!m.enabled) return <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Off</span>
  if (m.verified === false && m.verify_error === 'refused') {
    return <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-500 border-red-500/30`}><AlertCircle className="h-3 w-3" /> Key refused</span>
  }
  if (m.verified === false) {
    return <span className={`${pill} bg-amber-500/15 text-amber-600 dark:text-amber-500 border-amber-500/30`}><AlertCircle className="h-3 w-3" /> Not verified</span>
  }
  return <span className={`${pill} bg-emerald-500/15 text-emerald-600 dark:text-emerald-500 border-emerald-500/30`}><Check className="h-3 w-3" /> {PRIORITY_SHORT[m.priority] ?? 'on'}</span>
}

// The credential form: a new memory (provider picker + optional name), or new
// credentials for one of the person's own (provider fixed, saved hints shown).
function CredentialForm({ providers, memory, onDone, onCancel }: {
  providers: ProviderInfo[]
  memory?: MemoryView
  onDone: (notice: Notice) => void
  onCancel: () => void
}) {
  const { connect } = useExternalMemoryStore()
  const [providerName, setProviderName] = useState(memory?.provider ?? providers[0]?.name ?? '')
  const [name, setName] = useState('')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const baseId = useId()
  const chosen = providers.find((p) => p.name === providerName) ?? providers[0]
  if (!chosen) return null
  const missing = chosen.credential_fields.some((f) => f.required && !(drafts[f.name] || '').trim())

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (busy || missing) return
    const creds: Record<string, string> = {}
    for (const f of chosen.credential_fields) {
      const v = (drafts[f.name] || '').trim()
      if (v) creds[f.name] = v
    }
    setBusy(true)
    const res = await connect(chosen.name, creds, memory ? { memory: memory.id } : { name: name.trim() })
    setBusy(false)
    if (!res.ok) return
    onDone(res.warning ? { tone: 'warn', text: res.warning }
      : { tone: 'ok', text: `${memory ? 'New credentials saved' : 'Connected'} and checked with ${chosen.label}.` })
  }

  return (
    <form onSubmit={onSubmit} className="space-y-3 rounded-md border border-zinc-800 bg-zinc-950/40 p-3">
      {!memory && (
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label htmlFor={`${baseId}-provider`} className="block text-xs font-medium text-zinc-400 mb-1">Service</label>
            <select
              id={`${baseId}-provider`}
              value={chosen.name}
              onChange={(e) => { setProviderName(e.target.value); setDrafts({}) }}
              className={input}
            >
              {providers.map((p) => <option key={p.name} value={p.name}>{p.label}</option>)}
            </select>
          </div>
          <div>
            <label htmlFor={`${baseId}-name`} className="block text-xs font-medium text-zinc-400 mb-1">
              Name <span className="text-zinc-600">(optional)</span>
            </label>
            <input id={`${baseId}-name`} value={name} onChange={(e) => setName(e.target.value)} maxLength={40}
              placeholder={chosen.label} className={input} />
          </div>
        </div>
      )}
      <div className="text-xs text-zinc-400">
        <span className="font-medium text-zinc-300">{chosen.label}</span>
        {chosen.description ? ` — ${chosen.description}` : ''}
      </div>
      <CredentialInputs
        key={`${chosen.name}-${memory?.id ?? 'new'}`}
        idPrefix={baseId}
        fields={chosen.credential_fields}
        drafts={drafts}
        onChange={(n, v) => setDrafts((d) => ({ ...d, [n]: v }))}
        hints={memory?.credential_hints}
        disabled={busy}
        inputClass={input}
      />
      <div className="flex items-center gap-2 flex-wrap">
        <button type="submit" disabled={busy || missing} className={primaryBtn}
          title={`Flight Deck checks it with ${chosen.label} before saving it`}>
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
          {busy ? 'Checking…' : memory ? 'Replace' : 'Connect'}
        </button>
        <button type="button" onClick={onCancel} disabled={busy} className={ghostBtn}><X className="h-3 w-3" /> Cancel</button>
        {chosen.credentials_url && (
          <a href={chosen.credentials_url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 text-xs text-violet-500 hover:underline">
            Open {chosen.label} <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>
    </form>
  )
}

function MemoryCard({ mem, providers, mirrorKinds, priorities, defaultOpen }: {
  mem: MemoryView
  providers: ProviderInfo[]
  mirrorKinds: MirrorKind[]
  priorities: ReadingPriority[]
  defaultOpen: boolean
}) {
  const { test, disconnect, updateMemory, outbox, retryOutbox } = useExternalMemoryStore()
  const [open, setOpen] = useState(defaultOpen)
  const [busy, setBusy] = useState<Busy>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [replacing, setReplacing] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [nameDraft, setNameDraft] = useState(mem.name)
  const [showRules, setShowRules] = useState(false)
  const [showQueue, setShowQueue] = useState(false)
  const [queue, setQueue] = useState<OutboxItem[] | null>(null)
  const baseId = useId()
  const shared = mem.source === 'deck'
  const counts = mem.outbox
  const providerInfo = providers.filter((p) => p.name === mem.provider)

  const loadQueue = async () => setQueue(await outbox(mem.id))
  const step = async (b: Busy, fn: () => Promise<unknown>) => {
    if (busy) return
    setBusy(b); setNotice(null)
    try { await fn() } finally { setBusy(null) }
  }

  const onTest = () => step('test', async () => {
    const res = await test(mem.id)
    if (res.ok) setNotice({ tone: 'ok', text: `${mem.label} answered in ${duration(res.latency_ms ?? 0)}.` })
  })
  const onRemove = () => {
    if (!window.confirm(
      `Remove ${mem.name}?\n\nYour agents stop saving to it and searching it. What is already in it stays there; anything still waiting to be copied in is dropped.`,
    )) return
    step('remove', () => disconnect(mem.id))
  }
  const onRename = (e: FormEvent) => {
    e.preventDefault()
    const name = nameDraft.trim()
    if (!name || name === mem.name) { setRenaming(false); return }
    step('rename', async () => { if (await updateMemory(mem.id, { name })) setRenaming(false) })
  }
  const onRetry = () => step('retry', async () => {
    const n = await retryOutbox(mem.id)
    if (n !== null) {
      setNotice({ tone: 'ok', text: n ? `${n} item(s) will be sent again now.` : 'Nothing was waiting.' })
      if (showQueue) loadQueue()
    }
  })

  return (
    <div className={'rounded-md border ' + (mem.enabled ? 'border-zinc-800' : 'border-zinc-800/60 opacity-80')}>
      <div className="flex items-center gap-2 px-3 py-2.5">
        <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
          className="flex-1 min-w-0 flex items-center gap-2 text-left">
          <ChevronDown className={'h-3.5 w-3.5 shrink-0 text-zinc-500 transition-transform ' + (open ? '' : '-rotate-90')} />
          <span className="min-w-0">
            <span className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-medium text-zinc-100 truncate">{mem.name}</span>
              {memoryPill(mem)}
            </span>
            <span className="block text-[11px] text-zinc-500 truncate">
              {mem.label}{shared ? ' · shared by your admin' : ''}{mem.reaches.length ? ` · ${mem.reaches.join(', ')}` : ''}
            </span>
          </span>
        </button>
        <label className="flex items-center gap-1.5 text-[11px] text-zinc-400 cursor-pointer shrink-0" title="Your agents use it while it is on">
          <input
            type="checkbox"
            checked={mem.enabled}
            disabled={busy !== null}
            onChange={(e) => step('enabled', () => updateMemory(mem.id, { enabled: e.target.checked }))}
            className="accent-violet-600"
          />
          On
        </label>
      </div>

      {open && (
        <div className="border-t border-zinc-800 px-3 py-3 space-y-4">
          <NoticeBox notice={notice} />

          {shared && (
            <div className="text-[11px] leading-relaxed text-zinc-400">
              Your admin connected this {mem.label} for everyone; your agents use a {mem.scope_label || 'space'} of
              your own on it{mem.scope_id ? <> (<code className="text-zinc-300">{mem.scope_id}</code>)</> : null} —
              nobody else's agents see it. Switch it off if you don't want it.
            </div>
          )}

          <div className="text-[11px] leading-snug text-zinc-500">
            {mem.verified === true
              ? <>Checked{mem.verified_at ? ` ${when(mem.verified_at)}` : ''}.</>
              : mem.verify_error === 'refused'
                ? <span className="text-red-600 dark:text-red-400">{mem.label} refused the key — {shared ? 'tell your admin.' : 'replace it below.'} Nothing is copied in until then; what is waiting stays queued.</span>
                : mem.verified === false
                  ? <>Not verified — {mem.label} couldn't be reached. Test checks it again.</>
                  : <>Not checked yet.</>}
          </div>

          <section className="space-y-2">
            <h4 className="text-xs font-semibold text-zinc-300">Reading priority</h4>
            <div role="radiogroup" aria-label={`Reading priority of ${mem.name}`} className="space-y-1.5">
              {priorities.map((p) => {
                const on = mem.priority === p
                return (
                  <label key={p} className={'flex items-start gap-2 rounded-md border px-2.5 py-2 text-xs cursor-pointer ' + (on
                    ? 'border-violet-500/50 bg-violet-500/10' : 'border-zinc-800 hover:border-zinc-700')}>
                    <input type="radio" name={`${baseId}-priority`} checked={on} disabled={busy !== null}
                      onChange={() => step('priority', () => updateMemory(mem.id, { priority: p }))}
                      className="mt-0.5 accent-violet-600" />
                    <span>
                      <span className="text-zinc-200">{PRIORITY_LABELS[p]?.label ?? p}</span>
                      <span className="block text-[11px] leading-snug text-zinc-500">{PRIORITY_LABELS[p]?.hint ?? ''}</span>
                    </span>
                    {busy === 'priority' && on && <Loader2 className="ml-auto h-3 w-3 animate-spin text-zinc-500" />}
                  </label>
                )
              })}
            </div>
          </section>

          <section className="space-y-2">
            <h4 className="text-xs font-semibold text-zinc-300">Copied in on its own</h4>
            <div className="space-y-1.5">
              {mirrorKinds.map((kind) => (
                <label key={kind} className="flex items-start gap-2 text-xs cursor-pointer">
                  <input type="checkbox" checked={!!mem.mirror?.[kind]} disabled={busy !== null}
                    onChange={(e) => step('mirror', () => updateMemory(mem.id, { mirror: { [kind]: e.target.checked } }))}
                    className="mt-0.5 accent-violet-600" />
                  <span>
                    <span className="text-zinc-200">{MIRROR_LABELS[kind]?.label ?? kind}</span>
                    <span className="text-zinc-500"> — {MIRROR_LABELS[kind]?.hint ?? ''}</span>
                  </span>
                </label>
              ))}
            </div>
            {counts && (
              <div className="flex items-center gap-2 flex-wrap text-[11px] text-zinc-500">
                <span><span className="text-zinc-300">{counts.sent}</span> copied</span>
                <span>· <span className="text-zinc-300">{counts.pending}</span> waiting</span>
                {counts.failed > 0 && <span>· <span className="text-red-600 dark:text-red-400">{counts.failed}</span> failed</span>}
                {counts.last_sent_at && <span>· last {when(counts.last_sent_at)}</span>}
                {(counts.failed > 0 || counts.pending > 0) && (
                  <button type="button" onClick={onRetry} disabled={busy !== null} className="inline-flex items-center gap-1 text-violet-500 hover:underline disabled:opacity-50">
                    {busy === 'retry' ? <Loader2 className="h-3 w-3 animate-spin" /> : <RotateCcw className="h-3 w-3" />} Send now
                  </button>
                )}
                <button type="button" onClick={() => { const next = !showQueue; setShowQueue(next); if (next) loadQueue() }}
                  className="text-violet-500 hover:underline">
                  {showQueue ? 'Hide' : 'Show'} recent
                </button>
              </div>
            )}
            {counts?.last_error && (
              <div className="text-[11px] leading-snug text-amber-700 dark:text-amber-400 break-words">Last problem: {counts.last_error}</div>
            )}
            {showQueue && (
              <div className="rounded-md border border-zinc-800 divide-y divide-zinc-800 max-h-64 overflow-y-auto">
                {queue === null && <div className="px-3 py-2 text-[11px] text-zinc-500"><Loader2 className="inline h-3 w-3 animate-spin" /> Loading…</div>}
                {queue !== null && queue.length === 0 && <div className="px-3 py-2 text-[11px] text-zinc-500">Nothing copied in yet.</div>}
                {(queue ?? []).map((it) => (
                  <div key={it.id} className="px-3 py-1.5 text-[11px] flex items-start gap-2">
                    <span className={`shrink-0 ${it.state === 'sent' ? 'text-emerald-600 dark:text-emerald-500' : it.state === 'failed' ? 'text-red-600 dark:text-red-400' : 'text-amber-600 dark:text-amber-400'}`}>
                      {it.state === 'sent' ? '✓' : it.state === 'failed' ? '✕' : '…'}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="text-zinc-400">{MIRROR_LABELS[it.kind]?.label ?? it.kind}</span>
                      {it.agent && <span className="text-zinc-600"> · {it.agent}</span>}
                      <span className="block text-zinc-300 truncate">{it.title}</span>
                      {it.state !== 'sent' && it.last_error && <span className="block text-zinc-500 break-words">{it.last_error}</span>}
                    </span>
                    <span className="shrink-0 text-zinc-600">{when(it.sent_at ?? it.created_at)}</span>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="space-y-2">
            <h4 className="text-xs font-semibold text-zinc-300">What your agents can do here</h4>
            <div className="flex flex-wrap gap-1.5">
              {mem.actions.map((a) => (
                <span key={a.name} title={a.description} className={`${pill} ${a.destructive
                  ? 'border-red-500/30 text-red-600 dark:text-red-400'
                  : a.writes ? 'border-violet-500/30 text-violet-600 dark:text-violet-300' : 'border-zinc-700 text-zinc-300'}`}>
                  {a.name}
                </span>
              ))}
            </div>
            <button type="button" onClick={() => setShowRules((v) => !v)} className="text-[11px] text-violet-500 hover:underline">
              {showRules ? 'Hide' : 'Show'} the rules they follow ({mem.rules.length})
            </button>
            {showRules && (
              <ul className="list-disc pl-5 space-y-1 text-[11px] leading-snug text-zinc-400">
                {mem.rules.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            )}
          </section>

          {renaming && (
            <form onSubmit={onRename} className="flex gap-2">
              <input value={nameDraft} onChange={(e) => setNameDraft(e.target.value)} maxLength={40} autoFocus
                aria-label="Name" className={input} />
              <button type="submit" disabled={busy !== null || !nameDraft.trim()} className={ghostBtn}>
                {busy === 'rename' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />} Save
              </button>
              <button type="button" onClick={() => { setRenaming(false); setNameDraft(mem.name) }} className={ghostBtn}>Cancel</button>
            </form>
          )}
          {replacing && providerInfo.length > 0 && (
            <CredentialForm providers={providerInfo} memory={mem}
              onDone={(n) => { setReplacing(false); setNotice(n) }} onCancel={() => setReplacing(false)} />
          )}

          <div className="flex items-center gap-2 flex-wrap">
            <button type="button" onClick={onTest} disabled={busy !== null} className={ghostBtn}>
              {busy === 'test' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Server className="h-3 w-3" />} Test
            </button>
            {!shared && (
              <>
                <button type="button" onClick={() => { setRenaming(true); setReplacing(false) }} disabled={busy !== null || renaming} className={ghostBtn}>
                  <Pencil className="h-3 w-3" /> Rename
                </button>
                <button type="button" onClick={() => { setReplacing(true); setRenaming(false) }} disabled={busy !== null || replacing} className={ghostBtn}>
                  <KeyRound className="h-3 w-3" /> Replace key
                </button>
                <button type="button" onClick={onRemove} disabled={busy !== null} className={dangerBtn}>
                  {busy === 'remove' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                  Remove
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default function ExternalMemoryConnection() {
  const { status, loading, error, refresh, run } = useExternalMemoryStore()
  const [collapsed, setCollapsed] = useState(false)
  const [adding, setAdding] = useState(false)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [target, setTarget] = useState('')
  const [query, setQuery] = useState('')
  const [fact, setFact] = useState('')
  const [trying, setTrying] = useState<'search' | 'remember' | null>(null)
  const [answer, setAnswer] = useState<{ tone: 'ok' | 'err'; text: string } | null>(null)
  const baseId = useId()

  useEffect(() => { refresh() }, [refresh])

  const memories = status?.memories ?? []
  const enabled = memories.filter((m) => m.enabled)
  const own = memories.filter((m) => m.source === 'own')
  const canAdd = !!status && own.length < (status.max_memories ?? 5)
  const deckScope = status?.scope === 'deck'
  const priorities = status?.priorities ?? (['first', 'alongside', 'on_demand'] as ReadingPriority[])
  const offers = (action: string) => enabled.filter((m) => m.actions.some((a) => a.name === action))
  const searchable = useMemo(() => offers('search'), [enabled]) // eslint-disable-line react-hooks/exhaustive-deps
  const savable = useMemo(() => offers('remember'), [enabled]) // eslint-disable-line react-hooks/exhaustive-deps
  const targetName = enabled.find((m) => m.id === target)?.name

  const onSearch = async (e: FormEvent) => {
    e.preventDefault()
    if (!query.trim() || trying) return
    setTrying('search'); setAnswer(null)
    const res = await run('search', { query: query.trim(), limit: 8 }, target || undefined)
    setTrying(null)
    setAnswer('text' in res ? { tone: 'ok', text: res.text } : { tone: 'err', text: res.error })
  }
  const onRemember = async (e: FormEvent) => {
    e.preventDefault()
    if (!fact.trim() || trying) return
    setTrying('remember'); setAnswer(null)
    const res = await run('remember', { content: fact.trim() }, target || undefined)
    setTrying(null)
    if ('text' in res) setFact('')
    setAnswer('text' in res ? { tone: 'ok', text: res.text } : { tone: 'err', text: res.error })
  }

  let statusPill: ReactNode
  if (!status && error && !loading) {
    statusPill = <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-500 border-red-500/30`}><AlertCircle className="h-3 w-3" /> Unavailable</span>
  } else if (!status) {
    statusPill = <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}><Loader2 className="h-3 w-3 animate-spin" /> Checking</span>
  } else if (enabled.some((m) => m.verify_error === 'refused')) {
    statusPill = <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-500 border-red-500/30`}><AlertCircle className="h-3 w-3" /> Key refused</span>
  } else if (enabled.length === 1) {
    statusPill = <span className={`${pill} bg-emerald-500/15 text-emerald-600 dark:text-emerald-500 border-emerald-500/30`}><Check className="h-3 w-3" /> {enabled[0].name}</span>
  } else if (enabled.length > 1) {
    statusPill = <span className={`${pill} bg-emerald-500/15 text-emerald-600 dark:text-emerald-500 border-emerald-500/30`}><Check className="h-3 w-3" /> {enabled.length} memories</span>
  } else {
    statusPill = <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Not connected</span>
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
      <button
        type="button"
        onClick={() => setCollapsed((v) => !v)}
        className={'w-full flex items-center gap-3 px-5 py-4 text-left hover:bg-zinc-900/80 transition-colors ' + (collapsed ? '' : 'border-b border-zinc-800')}
        aria-expanded={!collapsed}
      >
        <div className="h-10 w-10 shrink-0 rounded-md bg-gradient-to-br from-emerald-500/20 to-violet-500/20 border border-zinc-800 flex items-center justify-center">
          <Brain className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">External memory</h3>
            {statusPill}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">Memories all of your agents share</div>
        </div>
        {loading && status && <Loader2 className="h-4 w-4 shrink-0 animate-spin text-zinc-500" />}
        <ChevronDown className={'h-4 w-4 shrink-0 text-zinc-500 transition-transform ' + (collapsed ? '-rotate-90' : '')} />
      </button>

      <div hidden={collapsed} className="px-5 py-4 space-y-4">
        <p className="text-xs leading-relaxed text-zinc-400">
          Connect one or more memory services and every one of {deckScope ? "this deck's" : 'your'} agents shares
          them: what one agent learns about you, the others can find. Each memory has its own reading priority and
          its own choice of what is copied in. Agents use them with the <code className="text-zinc-300">external_memory</code>{' '}
          tool — a search looks in all of them, a save goes to all of them unless they name one. Flight Deck keeps
          the keys — agents never see them.
          {status?.capture_every ? <> While one is on, your agents pick facts out of your chats every {status.capture_every} messages,
            and right after you tell them something to remember.</> : null}
        </p>

        {error && (
          <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
            <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            <span className="min-w-0 flex-1 break-words">{error}</span>
            {!status && !loading && (
              <button type="button" onClick={() => refresh()} className="shrink-0 underline hover:no-underline">Try again</button>
            )}
          </div>
        )}
        <NoticeBox notice={notice} />

        {status && memories.length > 0 && (
          <div className="space-y-2">
            {memories.map((m) => (
              <MemoryCard key={m.id} mem={m} providers={status.providers} mirrorKinds={status.mirror_kinds}
                priorities={priorities} defaultOpen={memories.length === 1} />
            ))}
          </div>
        )}

        {status && (adding ? (
          <CredentialForm providers={status.providers}
            onDone={(n) => { setAdding(false); setNotice(n) }} onCancel={() => setAdding(false)} />
        ) : canAdd ? (
          <button type="button" onClick={() => { setAdding(true); setNotice(null) }} className={memories.length ? ghostBtn : primaryBtn}>
            <Plus className="h-3.5 w-3.5" /> {memories.length ? 'Add a memory' : 'Connect a memory'}
          </button>
        ) : (
          <div className="text-[11px] text-zinc-500">You have {own.length} memories of your own — the most there can be. Remove one to add another.</div>
        ))}

        {(searchable.length > 0 || savable.length > 0) && (
          <section className="space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <h4 className="text-xs font-semibold text-zinc-300">Try it</h4>
              {enabled.length > 1 && (
                <select aria-label="Which memory" value={target} onChange={(e) => setTarget(e.target.value)}
                  className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-300 focus:border-violet-500 focus:outline-none">
                  <option value="">All memories</option>
                  {enabled.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
                </select>
              )}
            </div>
            {searchable.length > 0 && (
              <form onSubmit={onSearch} className="flex gap-2">
                <input id={`${baseId}-q`} value={query} onChange={(e) => setQuery(e.target.value)}
                  placeholder={targetName ? `Search ${targetName}…` : enabled.length > 1 ? 'Search all your memories…' : 'Search your memory…'}
                  className={input} />
                <button type="submit" disabled={!query.trim() || trying !== null} className={ghostBtn}>
                  {trying === 'search' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Search className="h-3 w-3" />} Search
                </button>
              </form>
            )}
            {savable.length > 0 && (
              <form onSubmit={onRemember} className="flex gap-2">
                <input value={fact} onChange={(e) => setFact(e.target.value)}
                  placeholder={targetName ? `Save a fact to ${targetName}…` : enabled.length > 1 ? 'Save a fact to all of them…' : 'Save a fact…'}
                  className={input} />
                <button type="submit" disabled={!fact.trim() || trying !== null} className={ghostBtn}>
                  {trying === 'remember' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />} Save
                </button>
              </form>
            )}
            {answer && (
              <pre className={'whitespace-pre-wrap break-words rounded-md border px-3 py-2 text-[11px] leading-snug font-sans ' + (answer.tone === 'ok'
                ? 'border-zinc-800 bg-zinc-950 text-zinc-300'
                : 'border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300')}>
                {answer.text}
              </pre>
            )}
          </section>
        )}
      </div>
    </div>
  )
}
