import { useEffect, useId, useMemo, useState, type FormEvent, type ReactNode } from 'react'
import {
  Brain,
  Loader2,
  Check,
  AlertCircle,
  ChevronDown,
  Eye,
  EyeOff,
  Save,
  Trash2,
  Server,
  ExternalLink,
  RotateCcw,
  Search,
  Plus,
} from 'lucide-react'
import {
  useExternalMemoryStore,
  type MirrorKind,
  type OutboxItem,
} from '../../stores/externalMemoryStore'

// External memory: one per Flight Deck user (per deck with sign-in off),
// shared by all of their agents. Flight Deck keeps the credentials and talks
// to the provider; agents reach it through Flight Deck with the
// `external_memory` tool, and what they store as insights, intentions and the
// like is copied in on its own (captain_claw/flight_deck/external_memory/).

type Busy = 'save' | 'test' | 'remove' | 'retry' | 'mirror' | 'search' | 'remember' | null
interface Notice { tone: 'ok' | 'warn'; text: string }

const pill = 'inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border'
const input = 'w-full min-w-0 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500 focus:outline-none disabled:opacity-50'
const ghostBtn = 'inline-flex items-center gap-1.5 rounded-md bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-40'

const MIRROR_LABELS: Record<MirrorKind, { label: string; hint: string }> = {
  insight: { label: 'Insights', hint: 'Facts, contacts, decisions, preferences and deadlines your agents pick up.' },
  intention: { label: 'Intentions', hint: 'Reminders and follow-ups, once they are active.' },
  lesson: { label: 'Lessons', hint: 'Rules and facts you teach your agents, or approve from what they noticed.' },
  reflection: { label: 'Reflections', hint: "Your agents' own notes on how to work better for you." },
  intuition: { label: 'Intuitions', hint: "Your agents' unchecked hunches and connections." },
}

function duration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`
}

function when(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === '') return ''
  const d = typeof value === 'number' ? new Date(value * 1000) : new Date(value)
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export default function ExternalMemoryConnection() {
  const { status, loading, error, refresh, connect, disconnect, test, setMirror, run, outbox, retryOutbox } =
    useExternalMemoryStore()

  const [collapsed, setCollapsed] = useState(false)
  const [providerName, setProviderName] = useState('')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [reveal, setReveal] = useState(false)
  const [busy, setBusy] = useState<Busy>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [showRules, setShowRules] = useState(false)
  const [showQueue, setShowQueue] = useState(false)
  const [queue, setQueue] = useState<OutboxItem[] | null>(null)
  const [query, setQuery] = useState('')
  const [fact, setFact] = useState('')
  const [answer, setAnswer] = useState<{ tone: 'ok' | 'err'; text: string } | null>(null)
  const baseId = useId()

  useEffect(() => { refresh() }, [refresh])

  const connected = !!status?.connected
  // On the deck's connection (an admin's, a bank of your own on it) rather than your own.
  const onDeck = connected && status?.source === 'deck'
  const own = connected && !onDeck
  const providers = status?.providers ?? []
  const chosen = useMemo(() => {
    const name = own ? status?.provider : providerName || (onDeck ? status?.provider : '') || providers[0]?.name
    return providers.find((p) => p.name === name) ?? providers[0]
  }, [own, onDeck, status?.provider, providerName, providers])
  const actions = status?.actions ?? []
  const canRemember = actions.some((a) => a.name === 'remember')
  const canSearch = actions.some((a) => a.name === 'search')
  const counts = status?.outbox
  const deckScope = status?.scope === 'deck'

  const loadQueue = async () => setQueue(await outbox())

  const onSave = async (e: FormEvent) => {
    e.preventDefault()
    if (!chosen || busy) return
    const creds: Record<string, string> = {}
    for (const f of chosen.credential_fields) {
      const v = (drafts[f.name] || '').trim()
      if (v) creds[f.name] = v
    }
    if (chosen.credential_fields.some((f) => f.required && !creds[f.name])) return
    setBusy('save'); setNotice(null)
    const res = await connect(chosen.name, creds)
    setBusy(null)
    if (!res.ok) return
    setDrafts({})
    setReveal(false)
    setNotice(res.warning ? { tone: 'warn', text: res.warning } : { tone: 'ok', text: `Connected and checked with ${chosen.label}.` })
  }

  const onTest = async () => {
    if (busy) return
    setBusy('test'); setNotice(null)
    const res = await test()
    setBusy(null)
    if (res.ok) setNotice({ tone: 'ok', text: `${status?.label ?? 'The memory'} answered in ${duration(res.latency_ms ?? 0)}.` })
  }

  const onRemove = async () => {
    if (busy) return
    const deckOffer = status?.deck_connection
    const ok = window.confirm(
      `Disconnect ${status?.label ?? 'the external memory'}?\n\n` +
      (deckOffer
        ? `Your agents switch to the ${deckOffer.label} your admin connected for everyone, in a bank of your own.`
        : (deckScope ? "This deck's agents" : 'Your agents') + ' stop saving to it and searching it.') +
      ' What is already in it stays there; anything still waiting to be copied in is dropped.',
    )
    if (!ok) return
    setBusy('remove'); setNotice(null)
    const done = await disconnect()
    setBusy(null)
    setQueue(null)
    if (done) setNotice({ tone: 'ok', text: 'Disconnected.' })
  }

  const onMirror = async (kind: MirrorKind, value: boolean) => {
    if (busy) return
    setBusy('mirror')
    await setMirror({ [kind]: value })
    setBusy(null)
  }

  const onRetry = async () => {
    if (busy) return
    setBusy('retry'); setNotice(null)
    const n = await retryOutbox()
    setBusy(null)
    if (n !== null) {
      setNotice({ tone: 'ok', text: n ? `${n} item(s) will be sent again now.` : 'Nothing was waiting.' })
      if (showQueue) loadQueue()
    }
  }

  const onSearch = async (e: FormEvent) => {
    e.preventDefault()
    if (!query.trim() || busy) return
    setBusy('search'); setAnswer(null)
    const res = await run('search', { query: query.trim(), limit: 8 })
    setBusy(null)
    setAnswer('text' in res ? { tone: 'ok', text: res.text } : { tone: 'err', text: res.error })
  }

  const onRemember = async (e: FormEvent) => {
    e.preventDefault()
    if (!fact.trim() || busy) return
    setBusy('remember'); setAnswer(null)
    const res = await run('remember', { content: fact.trim() })
    setBusy(null)
    if ('text' in res) setFact('')
    setAnswer('text' in res ? { tone: 'ok', text: res.text } : { tone: 'err', text: res.error })
  }

  let statusPill: ReactNode
  if (!status && error && !loading) {
    statusPill = <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-500 border-red-500/30`}><AlertCircle className="h-3 w-3" /> Unavailable</span>
  } else if (!status) {
    statusPill = <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}><Loader2 className="h-3 w-3 animate-spin" /> Checking</span>
  } else if (connected && status.verified === false && status.verify_error === 'refused') {
    statusPill = <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-500 border-red-500/30`}><AlertCircle className="h-3 w-3" /> Key refused</span>
  } else if (connected && status.verified === false) {
    statusPill = <span className={`${pill} bg-amber-500/15 text-amber-600 dark:text-amber-500 border-amber-500/30`}><AlertCircle className="h-3 w-3" /> Saved, not verified</span>
  } else if (connected) {
    statusPill = <span className={`${pill} bg-emerald-500/15 text-emerald-600 dark:text-emerald-500 border-emerald-500/30`}><Check className="h-3 w-3" /> {status.label}{onDeck ? ' · deck' : ''}</span>
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
          <div className="text-xs text-zinc-500 mt-0.5">One memory all of your agents share</div>
        </div>
        {loading && status && <Loader2 className="h-4 w-4 shrink-0 animate-spin text-zinc-500" />}
        <ChevronDown className={'h-4 w-4 shrink-0 text-zinc-500 transition-transform ' + (collapsed ? '-rotate-90' : '')} />
      </button>

      <div hidden={collapsed} className="px-5 py-4 space-y-5">
        <p className="text-xs leading-relaxed text-zinc-400">
          Connect a memory service and every one of {deckScope ? "this deck's" : 'your'} agents shares it: what one
          agent learns about you, the others can find. Agents save and search it with the{' '}
          <code className="text-zinc-300">external_memory</code> tool, following the service's own rules, and what
          they keep as insights, intentions or lessons is copied in on its own. Flight Deck keeps the key — agents
          never see it.
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

        {notice && (
          <div className={'rounded-md border px-3 py-2 text-xs break-words ' + (notice.tone === 'ok'
            ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
            : 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300')}
          >
            {notice.text}
          </div>
        )}

        {onDeck && status && (
          <div className="rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-xs leading-relaxed text-zinc-400">
            Your agents use the <span className="text-zinc-200">{status.label}</span> your admin connected for
            everyone, in a {status.scope_label || 'space'} of your own
            {status.scope_id ? <> (<code className="text-zinc-300">{status.scope_id}</code>)</> : null} — nobody
            else's agents see it. To use a memory of your own instead, connect it below.
          </div>
        )}

        {status && chosen && (
          <form onSubmit={onSave} className="space-y-3">
            {!own && providers.length > 1 && (
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
            )}
            <div className="text-xs text-zinc-400">
              <span className="font-medium text-zinc-300">{chosen.label}</span>
              {chosen.description ? ` — ${chosen.description}` : ''}
            </div>

            {chosen.credential_fields.map((f) => {
              const id = `${baseId}-${f.name}`
              const hint = own ? status.credential_hints?.[f.name] : undefined
              return (
                <div key={f.name} className="space-y-1">
                  <label htmlFor={id} className="block text-xs font-medium text-zinc-400">
                    {f.label}
                    {!f.required && <span className="ml-1 text-zinc-600">(optional)</span>}
                    {own && hint && f.secret && <span className="ml-2 text-[10px] text-emerald-600 dark:text-emerald-500">(saved)</span>}
                  </label>
                  <div className="relative">
                    <input
                      id={id}
                      type={f.secret && !reveal ? 'password' : 'text'}
                      value={drafts[f.name] || ''}
                      onChange={(e) => setDrafts((d) => ({ ...d, [f.name]: e.target.value }))}
                      placeholder={own && hint ? (f.secret ? `•••• saved (${hint}) — paste a new one to replace` : hint) : f.placeholder}
                      autoComplete="off"
                      spellCheck={false}
                      disabled={busy === 'save'}
                      className={`${input} font-mono ${f.secret ? 'pr-9' : ''}`}
                    />
                    {f.secret && (
                      <button
                        type="button"
                        onClick={() => setReveal((v) => !v)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
                        tabIndex={-1}
                        aria-label={reveal ? 'Hide' : 'Show'}
                      >
                        {reveal ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      </button>
                    )}
                  </div>
                  {f.help && <div className="text-[11px] leading-snug text-zinc-500">{f.help}</div>}
                </div>
              )
            })}

            <div className="flex items-center gap-2 flex-wrap">
              <button
                type="submit"
                disabled={busy !== null || chosen.credential_fields.some((f) => f.required && !(drafts[f.name] || '').trim())}
                className="inline-flex items-center gap-2 rounded-md bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-sm text-white disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                title={`Flight Deck checks it with ${chosen.label} before saving it`}
              >
                {busy === 'save' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                {busy === 'save' ? 'Checking…' : own ? 'Replace' : onDeck ? 'Connect my own' : 'Connect'}
              </button>
              {connected && (
                <>
                  <button type="button" onClick={onTest} disabled={busy !== null} className={ghostBtn}>
                    {busy === 'test' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Server className="h-3 w-3" />}
                    Test
                  </button>
                  {own && (
                    <button
                      type="button"
                      onClick={onRemove}
                      disabled={busy !== null}
                      className="inline-flex items-center gap-1.5 rounded-md border border-red-500/30 hover:bg-red-500/10 px-3 py-1.5 text-xs text-red-600 dark:text-red-500 disabled:opacity-50"
                    >
                      {busy === 'remove' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                      Disconnect
                    </button>
                  )}
                </>
              )}
              {chosen.credentials_url && (
                <a href={chosen.credentials_url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 text-xs text-violet-500 hover:underline">
                  Open {chosen.label} <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </div>

            {connected && (
              <div className="text-[11px] leading-snug text-zinc-500">
                {status.verified === true
                  ? <>Checked{status.verified_at ? ` ${when(status.verified_at)}` : ''}.</>
                  : status.verify_error === 'refused'
                    ? <span className="text-red-600 dark:text-red-400">{status.label} refused the key — {onDeck ? 'tell your admin.' : 'paste a new one above.'} Nothing is copied in until then; what is waiting stays queued.</span>
                    : status.verified === false
                      ? <>Not verified — {status.label} couldn't be reached. Test checks it again.</>
                      : <>Not checked yet.</>}
                {status.reaches && status.reaches.length > 0 && <> Reaches: <span className="text-zinc-300">{status.reaches.join(', ')}</span>.</>}
              </div>
            )}
          </form>
        )}

        {connected && status && (
          <>
            <section className="space-y-2">
              <h4 className="text-xs font-semibold text-zinc-300">What your agents can do</h4>
              <div className="flex flex-wrap gap-1.5">
                {actions.map((a) => (
                  <span
                    key={a.name}
                    title={a.description}
                    className={`${pill} ${a.destructive
                      ? 'border-red-500/30 text-red-600 dark:text-red-400'
                      : a.writes ? 'border-violet-500/30 text-violet-600 dark:text-violet-300' : 'border-zinc-700 text-zinc-300'}`}
                  >
                    {a.name}
                  </span>
                ))}
              </div>
              <button type="button" onClick={() => setShowRules((v) => !v)} className="text-[11px] text-violet-500 hover:underline">
                {showRules ? 'Hide' : 'Show'} the rules they follow ({status.rules?.length ?? 0})
              </button>
              {showRules && (
                <ul className="list-disc pl-5 space-y-1 text-[11px] leading-snug text-zinc-400">
                  {(status.rules ?? []).map((r, i) => <li key={i}>{r}</li>)}
                </ul>
              )}
            </section>

            <section className="space-y-2">
              <h4 className="text-xs font-semibold text-zinc-300">Copied in on its own</h4>
              <div className="space-y-1.5">
                {(status.mirror_kinds ?? []).map((kind) => (
                  <label key={kind} className="flex items-start gap-2 text-xs cursor-pointer">
                    <input
                      type="checkbox"
                      checked={!!status.mirror?.[kind]}
                      onChange={(e) => onMirror(kind, e.target.checked)}
                      disabled={busy === 'mirror'}
                      className="mt-0.5 accent-violet-600"
                    />
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
                  <button
                    type="button"
                    onClick={() => { const next = !showQueue; setShowQueue(next); if (next) loadQueue() }}
                    className="text-violet-500 hover:underline"
                  >
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

            {(canSearch || canRemember) && (
              <section className="space-y-2">
                <h4 className="text-xs font-semibold text-zinc-300">Try it</h4>
                {canSearch && (
                  <form onSubmit={onSearch} className="flex gap-2">
                    <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search your memory…" className={input} />
                    <button type="submit" disabled={!query.trim() || busy !== null} className={ghostBtn}>
                      {busy === 'search' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Search className="h-3 w-3" />} Search
                    </button>
                  </form>
                )}
                {canRemember && (
                  <form onSubmit={onRemember} className="flex gap-2">
                    <input value={fact} onChange={(e) => setFact(e.target.value)} placeholder="Save a fact…" className={input} />
                    <button type="submit" disabled={!fact.trim() || busy !== null} className={ghostBtn}>
                      {busy === 'remember' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />} Save
                    </button>
                  </form>
                )}
                {answer && (
                  <pre className={'whitespace-pre-wrap break-words rounded-md border px-3 py-2 text-[11px] leading-snug font-sans ' + (answer.tone === 'ok'
                    ? 'border-zinc-800 bg-zinc-950 text-zinc-300'
                    : 'border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300')}
                  >
                    {answer.text}
                  </pre>
                )}
              </section>
            )}
          </>
        )}
      </div>
    </div>
  )
}
