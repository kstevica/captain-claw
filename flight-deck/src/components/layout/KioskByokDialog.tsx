import { useEffect, useId, useMemo, useState, type FormEvent } from 'react'
import { AlertCircle, Check, Eye, EyeOff, KeyRound, Loader2, RotateCcw, Trash2 } from 'lucide-react'
import {
  getByok, removeByokKey, saveByokKey, saveByokModels, setByokEnabled,
  type ByokChange, type ByokProvider, type ByokStatus, type ByokTier, type ByokTierPick,
} from '../../services/byok'
import { useQuotaStore } from '../../stores/quotaStore'
import { AgentDialog } from './KioskAgentDialogs'

// ── Kiosk: "Your own AI keys" ────────────────────────────────────────
//
// A user brings their own provider keys and turns them on: their agents then
// run on those keys and the deck's daily limits stop applying to them. Keys go
// in and never come back (a hint only). Each kind of work (tier) runs on the
// model the deck uses when the user has a key for that provider, else on a
// default of one of theirs — they can pick another. Turning the switch on or
// off moves their agents and restarts the running ones (byok.py).

const when = (iso: string | null) => {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

function movedText(c: ByokChange): string {
  const m = c.moved
  if (!m) return ''
  if (m.error) return `Your agents couldn't all be moved: ${m.error}`
  if (!m.switched) return ''
  return `${plural(m.switched, 'agent')} moved${m.restarted ? ` — ${m.restarted} restarting` : ''}.`
}

export function KioskByokDialog({ onClose }: { onClose: () => void }) {
  const [status, setStatus] = useState<ByokStatus | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const [note, setNote] = useState<{ tone: 'ok' | 'warn'; text: string } | null>(null)
  const refreshQuota = useQuotaStore((s) => s.refresh)

  useEffect(() => {
    getByok().then(setStatus).catch((e) => setLoadErr(e instanceof Error ? e.message : String(e)))
  }, [])

  const apply = (c: ByokChange) => {
    setStatus(c)
    const moved = movedText(c)
    if (c.warning) setNote({ tone: 'warn', text: [c.warning, moved].filter(Boolean).join(' ') })
    else if (moved) setNote({ tone: c.moved?.error ? 'warn' : 'ok', text: moved })
    void refreshQuota()
  }

  return (
    <AgentDialog title="Your own AI keys" icon={KeyRound} agentName="Run your agents on your own API keys — no daily limits" size="md" onClose={onClose}>
      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        {loadErr && <p className="py-10 text-center text-sm text-red-600 dark:text-red-400">Couldn't load your keys: {loadErr}</p>}
        {!loadErr && !status && (
          <p className="flex items-center justify-center gap-2 py-10 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</p>
        )}
        {status && (
          <div className="space-y-5">
            <SwitchCard status={status} onChange={apply} />
            {note && (
              <p className={`flex items-start gap-2 rounded-lg px-3 py-2 text-xs ${note.tone === 'ok'
                ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                : 'bg-amber-500/10 text-amber-700 dark:text-amber-300'}`}>
                {note.tone === 'ok' ? <Check className="mt-0.5 h-3.5 w-3.5 shrink-0" /> : <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />}
                {note.text}
              </p>
            )}
            <section>
              <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Your keys</h3>
              {!status.encryption && (
                <p className="mb-2 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                  This deck can't store keys safely yet (no encryption available). Ask your admin.
                </p>
              )}
              <ul className="space-y-2">
                {status.providers.map((p) => (
                  <KeyRow key={p.id} p={p} locked={status.enabled && status.providers.filter((x) => x.connected).length === 1 && p.connected}
                    disabled={!status.encryption} onChange={apply} onError={(text) => setNote({ tone: 'warn', text })} />
                ))}
              </ul>
            </section>
            <ModelsSection status={status} onChange={apply} onError={(text) => setNote({ tone: 'warn', text })} />
            <p className="text-[11px] leading-relaxed text-zinc-500">
              Your keys are stored encrypted on this deck and are never shown again — only their last characters.
              You pay your provider directly for what your agents use on them. Web search and other tool keys still come from the deck.
            </p>
          </div>
        )}
      </div>
    </AgentDialog>
  )
}

// ── The switch ───────────────────────────────────────────────────────

function SwitchCard({ status, onChange }: { status: ByokStatus; onChange: (c: ByokChange) => void }) {
  const [asking, setAsking] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const on = status.enabled
  const keyed = status.providers.some((p) => p.connected)
  const running = status.agents.running
  const canTurnOn = status.allowed && keyed && status.missing.length === 0

  const flip = async (enabled: boolean) => {
    setBusy(true); setErr('')
    try {
      onChange(await setByokEnabled(enabled))
      setAsking(false)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
    setBusy(false)
  }

  const why = !status.allowed
    ? 'Your admin has turned own keys off on this deck.'
    : !keyed
      ? 'Add a key below to turn this on.'
      : status.missing.length
        ? `Pick a model for: ${status.missing.join(', ')}.`
        : ''

  return (
    <div className={`rounded-xl border p-4 ${on ? 'border-emerald-500/40 bg-emerald-500/5' : 'border-zinc-800 bg-zinc-900/60'}`}>
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-zinc-100">{on ? 'On — your agents run on your keys' : 'Off — your agents run on the deck\'s models'}</div>
          <p className="mt-0.5 text-xs leading-relaxed text-zinc-500">
            {on
              ? `No daily limits while this is on. ${plural(status.agents.on_own_keys, 'agent')} on your keys.`
              : 'Within your plan\'s daily limits. Turn this on to use your own keys with no limits.'}
          </p>
        </div>
        <button
          role="switch"
          aria-checked={on}
          aria-label="Use my own keys"
          disabled={busy || (!on && !canTurnOn)}
          onClick={() => {
            if (!on && running > 0) { setAsking(true); return }
            void flip(!on)
          }}
          className={`relative mt-0.5 h-6 w-11 shrink-0 rounded-full transition-colors disabled:opacity-50 ${on ? 'bg-emerald-500' : 'bg-zinc-700'}`}
        >
          <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${on ? 'left-[22px]' : 'left-0.5'}`} />
        </button>
      </div>
      {!on && why && <p className="mt-2 text-xs text-zinc-400">{why}</p>}
      {asking && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-zinc-300">
          <span>This restarts your {plural(running, 'running agent')} to move {running === 1 ? 'it' : 'them'} onto your keys.</span>
          <button onClick={() => flip(true)} disabled={busy} className="rounded-md bg-violet-600 px-2.5 py-1 font-medium text-white hover:bg-violet-500 disabled:opacity-50">
            {busy ? 'Turning on…' : 'Turn on'}
          </button>
          <button onClick={() => setAsking(false)} disabled={busy} className="rounded-md border border-zinc-700 px-2.5 py-1 hover:bg-zinc-800">Cancel</button>
        </div>
      )}
      {on && running > 0 && !busy && (
        <p className="mt-2 text-[11px] text-zinc-500">Turning it off restarts your running agents on the deck's models.</p>
      )}
      {busy && !asking && <p className="mt-2 flex items-center gap-1.5 text-xs text-zinc-400"><Loader2 className="h-3 w-3 animate-spin" /> Moving your agents…</p>}
      {err && <p className="mt-2 text-xs text-red-600 dark:text-red-400">{err}</p>}
    </div>
  )
}

// ── One provider's key ───────────────────────────────────────────────

function KeyRow({ p, locked, disabled, onChange, onError }: {
  p: ByokProvider
  /** The last key while the switch is on: can't go. */
  locked: boolean
  disabled: boolean
  onChange: (c: ByokChange) => void
  onError: (text: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState('')
  const [show, setShow] = useState(false)
  const [busy, setBusy] = useState<'save' | 'remove' | null>(null)
  const [err, setErr] = useState('')
  const inputId = useId()

  const save = async (e: FormEvent) => {
    e.preventDefault()
    if (!value.trim() || busy) return
    setBusy('save'); setErr('')
    try {
      onChange(await saveByokKey(p.id, value.trim()))
      setValue(''); setEditing(false); setShow(false)
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex))
    }
    setBusy(null)
  }

  const remove = async () => {
    setBusy('remove')
    try {
      onChange(await removeByokKey(p.id))
    } catch (ex) {
      onError(ex instanceof Error ? ex.message : String(ex))
    }
    setBusy(null)
  }

  return (
    <li className="rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium text-zinc-200">{p.label}</span>
          <span className="block text-[11px] text-zinc-500">
            {p.connected
              ? `Key ${p.hint || ''}${p.verified === false ? ' · not checked yet' : p.checked_at ? ` · checked ${when(p.checked_at)}` : ''}${p.models.length ? ` · ${plural(p.models.length, 'model')}` : ''}`
              : 'No key'}
          </span>
        </span>
        {p.connected && !editing && (
          <button onClick={remove} disabled={!!busy || locked} title={locked ? 'Your last key — turn your own keys off first' : 'Forget this key'}
            className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-red-500 disabled:opacity-40">
            {busy === 'remove' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
          </button>
        )}
        {!editing && (
          <button onClick={() => { setEditing(true); setErr('') }} disabled={disabled}
            className="shrink-0 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:border-violet-500/50 hover:text-zinc-100 disabled:opacity-50">
            {p.connected ? 'Replace' : 'Add key'}
          </button>
        )}
      </div>
      {editing && (
        <form onSubmit={save} className="mt-2 space-y-1.5">
          <label htmlFor={inputId} className="sr-only">{p.label} API key</label>
          <div className="flex gap-2">
            <div className="relative min-w-0 flex-1">
              <input
                id={inputId}
                autoFocus
                type={show ? 'text' : 'password'}
                autoComplete="off"
                spellCheck={false}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                disabled={!!busy}
                placeholder={`Paste your ${p.label} API key`}
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 py-2 pl-3 pr-9 text-base text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none wide:text-sm"
              />
              <button type="button" onClick={() => setShow(!show)} aria-label={show ? 'Hide key' : 'Show key'}
                className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-zinc-500 hover:text-zinc-300">
                {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            <button type="submit" disabled={!value.trim() || !!busy}
              className="flex shrink-0 items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-2 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50">
              {busy === 'save' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
              {busy === 'save' ? 'Checking…' : 'Save'}
            </button>
            <button type="button" onClick={() => { setEditing(false); setValue(''); setErr('') }} disabled={!!busy}
              className="shrink-0 rounded-lg border border-zinc-700 px-2.5 py-2 text-xs text-zinc-400 hover:bg-zinc-800">
              Cancel
            </button>
          </div>
          <p className="text-[11px] text-zinc-500">{p.label} is asked whether the key works before it is saved.</p>
          {err && <p className="text-xs text-red-600 dark:text-red-400">{err}</p>}
        </form>
      )}
    </li>
  )
}

// ── Which model each kind of work uses ───────────────────────────────

function ModelsSection({ status, onChange, onError }: {
  status: ByokStatus; onChange: (c: ByokChange) => void; onError: (text: string) => void
}) {
  const keyed = useMemo(() => status.providers.filter((p) => p.connected), [status.providers])
  const [draft, setDraft] = useState<Record<string, ByokTierPick>>({})
  const [busy, setBusy] = useState(false)
  const listId = useId()
  if (!keyed.length) return null

  const dirty = Object.keys(draft).length > 0
  const pick = (t: ByokTier): ByokTierPick => draft[t.id] || t.mine || { provider: keyed[0].id, model: '' }

  const save = async (picks: Record<string, ByokTierPick>) => {
    setBusy(true)
    try {
      onChange(await saveByokModels(picks))
      setDraft({})
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e))
    }
    setBusy(false)
  }

  const shown = status.tiers.filter((t) => !t.local)
  const local = status.tiers.filter((t) => t.local)
  return (
    <section>
      <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Which model each kind of work uses</h3>
      <p className="mb-2 text-xs text-zinc-500">
        Where you have a key for the provider the deck uses, the model stays the same. Otherwise one of yours is picked — change it here.
      </p>
      <ul className="space-y-2">
        {shown.map((t) => {
          const cur = pick(t)
          const models = keyed.find((p) => p.id === cur.provider)?.models || []
          return (
            <li key={t.id} className="rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2.5">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm font-medium text-zinc-200">{t.label}</span>
                {t.team.model && <span className="truncate text-[11px] text-zinc-600" title="What the deck uses">Deck: {t.team.model}</span>}
              </div>
              {t.use && <p className="text-[11px] text-zinc-500">{t.use}</p>}
              <div className="mt-2 flex flex-col gap-2 wide:flex-row">
                <select
                  value={cur.provider}
                  disabled={busy}
                  aria-label={`${t.label}: provider`}
                  onChange={(e) => {
                    const provider = e.target.value
                    setDraft({ ...draft, [t.id]: { provider, model: '' } })
                  }}
                  className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-200 focus:border-violet-500/60 focus:outline-none wide:w-40"
                >
                  {keyed.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
                </select>
                <input
                  value={cur.model}
                  disabled={busy}
                  list={`${listId}-${cur.provider}`}
                  aria-label={`${t.label}: model`}
                  placeholder="Model"
                  onChange={(e) => setDraft({ ...draft, [t.id]: { provider: cur.provider, model: e.target.value } })}
                  className="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-base text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none wide:text-sm"
                />
                {t.chosen && !draft[t.id] && (
                  <button onClick={() => save({ [t.id]: { provider: '', model: '' } })} disabled={busy} title="Back to the default"
                    className="flex shrink-0 items-center gap-1 self-start rounded-lg border border-zinc-700 px-2 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">
                    <RotateCcw className="h-3 w-3" /> Default
                  </button>
                )}
              </div>
              {models.length > 0 && (
                <datalist id={`${listId}-${cur.provider}`}>
                  {models.map((m) => <option key={m} value={m} />)}
                </datalist>
              )}
              {!t.mine && !draft[t.id] && <p className="mt-1 text-[11px] text-amber-600 dark:text-amber-400">No model yet — pick one.</p>}
            </li>
          )
        })}
      </ul>
      {local.length > 0 && (
        <p className="mt-2 text-[11px] text-zinc-500">
          {local.map((t) => t.label).join(', ')} {local.length === 1 ? 'runs' : 'run'} on the deck's own local model{local.length === 1 ? '' : 's'} — {local.length === 1 ? 'it stays' : 'they stay'} as {local.length === 1 ? 'it is' : 'they are'}.
        </p>
      )}
      {dirty && (
        <div className="mt-3 flex items-center gap-2">
          <button onClick={() => save(draft)} disabled={busy || Object.values(draft).some((d) => !d.model.trim())}
            className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50">
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
            Save models
          </button>
          <button onClick={() => setDraft({})} disabled={busy} className="rounded-lg border border-zinc-700 px-2.5 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">Cancel</button>
          {status.enabled && <span className="text-[11px] text-zinc-500">Agents whose model changes restart.</span>}
        </div>
      )}
    </section>
  )
}
