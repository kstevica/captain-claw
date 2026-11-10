import { useEffect, useId, useMemo, useState, type FormEvent } from 'react'
import { Users, Loader2, Check, AlertCircle, Save, Trash2, Server, ExternalLink } from 'lucide-react'
import { useExternalMemoryStore } from '../../stores/externalMemoryStore'
import CredentialInputs from './CredentialInputs'

// The deck's external memory (admins): one server that keeps people apart
// itself — Hindsight gives each person a bank, Qdrant and ArcadeDB a space —
// connected once for everyone. Users without a memory of their own use it,
// each in their own; their
// own connection wins. A memory whose key is one person's (Membase) can't be
// shared this way, so only those that keep people apart are offered.

type Busy = 'save' | 'test' | 'remove' | null
interface Notice { tone: 'ok' | 'warn'; text: string }

const pill = 'inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border'
const input = 'w-full min-w-0 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500 focus:outline-none disabled:opacity-50'

function duration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`
}

export default function ExternalMemoryDeckConnection() {
  const { deck, deckError, refreshDeck, connectDeck, disconnectDeck, testDeck } = useExternalMemoryStore()
  const [providerName, setProviderName] = useState('')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState<Busy>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const baseId = useId()

  useEffect(() => { refreshDeck() }, [refreshDeck])

  const providers = deck?.providers ?? []
  const chosen = useMemo(() => {
    const name = deck?.connected ? deck.provider : providerName || providers[0]?.name
    return providers.find((p) => p.name === name) ?? providers[0]
  }, [deck?.connected, deck?.provider, providerName, providers])
  // Each person gets their own bank / space: the deck's connection never names one.
  const fields = (chosen?.credential_fields ?? []).filter((f) => !chosen?.scope_field || f.name !== chosen.scope_field)
  const scope = (deck?.connected ? deck.scope_label : chosen?.scope_label) || 'space'


  if (deck && !deck.available) return null

  const onSave = async (e: FormEvent) => {
    e.preventDefault()
    if (!chosen || busy) return
    const creds: Record<string, string> = {}
    for (const f of fields) {
      const v = (drafts[f.name] || '').trim()
      if (v) creds[f.name] = v
    }
    if (fields.some((f) => f.required && !creds[f.name])) return
    setBusy('save'); setNotice(null)
    const res = await connectDeck(chosen.name, creds)
    setBusy(null)
    if (!res.ok) return
    setDrafts({})
    setNotice(res.warning ? { tone: 'warn', text: res.warning } : { tone: 'ok', text: `Connected for everyone and checked with ${chosen.label}.` })
  }

  const onTest = async () => {
    if (busy) return
    setBusy('test'); setNotice(null)
    const res = await testDeck()
    setBusy(null)
    if (res.ok) setNotice({ tone: 'ok', text: `${deck?.label ?? 'The memory'} answered in ${duration(res.latency_ms ?? 0)}.` })
  }

  const onRemove = async () => {
    if (busy) return
    if (!window.confirm(
      `Stop sharing ${deck?.label ?? 'this memory'} with everyone?\n\n` +
      `People without a memory of their own lose theirs in Flight Deck (what is in their ${scope}s stays on the server). People who connected their own aren't affected.`,
    )) return
    setBusy('remove'); setNotice(null)
    const done = await disconnectDeck()
    setBusy(null)
    if (done) setNotice({ tone: 'ok', text: 'No longer shared.' })
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
      <div className="flex items-center gap-3 px-5 py-4 border-b border-zinc-800">
        <div className="h-10 w-10 shrink-0 rounded-md bg-gradient-to-br from-emerald-500/20 to-sky-500/20 border border-zinc-800 flex items-center justify-center">
          <Users className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">Shared memory server</h3>
            {!deck ? (
              <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}><Loader2 className="h-3 w-3 animate-spin" /> Checking</span>
            ) : deck.connected && deck.verify_error === 'refused' ? (
              <span className={`${pill} bg-red-500/15 text-red-600 dark:text-red-500 border-red-500/30`}><AlertCircle className="h-3 w-3" /> Key refused</span>
            ) : deck.connected ? (
              <span className={`${pill} bg-emerald-500/15 text-emerald-600 dark:text-emerald-500 border-emerald-500/30`}><Check className="h-3 w-3" /> {deck.label}</span>
            ) : (
              <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Not shared</span>
            )}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">One memory server, a {scope} per person</div>
        </div>
      </div>

      <div className="px-5 py-4 space-y-4">
        <p className="text-xs leading-relaxed text-zinc-400">
          Connect one memory server for the whole deck. Everyone who hasn't connected a memory of their own uses
          it, each in a {scope} of their own
          {deck?.scope_example ? <> (<code className="text-zinc-300">{deck.scope_example}</code>)</> : null} — agents
          never pick a {scope}, so nobody's agents can read anyone else's. A person's own connection wins over this one.
          Only memories that keep people apart themselves can be shared like this.
        </p>

        {deckError && (
          <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
            <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            <span className="min-w-0 flex-1 break-words">{deckError}</span>
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

        {deck && chosen && (
          <form onSubmit={onSave} className="space-y-3">
            {!deck.connected && providers.length > 1 && (
              <select
                aria-label="Service"
                value={chosen.name}
                onChange={(e) => { setProviderName(e.target.value); setDrafts({}) }}
                className={input}
              >
                {providers.map((p) => <option key={p.name} value={p.name}>{p.label}</option>)}
              </select>
            )}
            <div className="text-xs text-zinc-400">
              <span className="font-medium text-zinc-300">{chosen.label}</span>
              {chosen.description ? ` — ${chosen.description}` : ''}
            </div>
            <CredentialInputs
              key={`${chosen.name}-${deck.connected ? 'saved' : 'new'}`}
              idPrefix={baseId}
              fields={fields}
              drafts={drafts}
              onChange={(name, value) => setDrafts((d) => ({ ...d, [name]: value }))}
              hints={deck.connected ? deck.credential_hints : undefined}
              disabled={busy === 'save'}
              inputClass={input}
            />
            <div className="flex items-center gap-2 flex-wrap">
              <button
                type="submit"
                disabled={busy !== null || fields.some((f) => f.required && !(drafts[f.name] || '').trim())}
                className="inline-flex items-center gap-2 rounded-md bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-sm text-white disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
              >
                {busy === 'save' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                {busy === 'save' ? 'Checking…' : deck.connected ? 'Replace' : 'Share with everyone'}
              </button>
              {deck.connected && (
                <>
                  <button type="button" onClick={onTest} disabled={busy !== null}
                    className="inline-flex items-center gap-1.5 rounded-md bg-zinc-800 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-700 disabled:opacity-40">
                    {busy === 'test' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Server className="h-3 w-3" />}
                    Test
                  </button>
                  <button type="button" onClick={onRemove} disabled={busy !== null}
                    className="inline-flex items-center gap-1.5 rounded-md border border-red-500/30 hover:bg-red-500/10 px-3 py-1.5 text-xs text-red-600 dark:text-red-500 disabled:opacity-50">
                    {busy === 'remove' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                    Stop sharing
                  </button>
                </>
              )}
              {chosen.docs_url && (
                <a href={chosen.docs_url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 text-xs text-violet-500 hover:underline">
                  {chosen.label} docs <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </div>
          </form>
        )}
        {deck && providers.length === 0 && (
          <div className="text-xs text-zinc-500">No memory this deck knows can be shared yet.</div>
        )}
      </div>
    </div>
  )
}
