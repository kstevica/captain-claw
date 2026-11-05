import { useMemo, useState } from 'react'
import { Check, ExternalLink, Eye, EyeOff, AlertTriangle } from 'lucide-react'
import { TIER_ORDER, type TierConfig } from '../../services/tierConfig'
import {
  checkPresetKey, presetForKey, presetModelGroups, presetSetName, presetTiers, type TierPreset,
} from '../../services/tierPresets'
import { Field, Notice, inputCls } from './parts'

// Quick start: pick the AI provider, paste its key, and get a tier set named
// after the provider with its current models on every tier (the catalog is
// captain_claw/instructions/tier_presets.json). The key is checked with the
// provider first, so a mistyped key never becomes a set that can't run.

function keyHost(url: string): string {
  try { return new URL(url).host.replace(/^www\./, '') } catch { return url }
}

function fmtChecked(day: string): string {
  const d = new Date(`${day}T00:00:00`)
  return Number.isNaN(d.getTime()) ? day : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

// What stands between the person and the set after a check.
type Hold =
  | { kind: 'refused'; preset: string }
  | { kind: 'unreachable'; preset: string }
  | { kind: 'missing'; preset: string; models: string[] }

export function QuickStartForm({ presets, checked, tierLabel, existingNames, onCreate }: {
  presets: TierPreset[]
  checked?: string
  tierLabel: (tier: string) => string
  existingNames: string[]
  onCreate: (name: string, perTier: Record<string, Partial<TierConfig>>) => void
}) {
  const [picked, setPicked] = useState<string>('')
  const [apiKey, setApiKey] = useState('')
  const [shown, setShown] = useState(false)
  const [busy, setBusy] = useState(false)
  const [hold, setHold] = useState<Hold | null>(null)

  const preset = presets.find((p) => p.id === picked) || null
  const key = apiKey.trim()
  // The provider the key's own prefix names, when it names one.
  const keyFor = presetForKey(presets, key)
  const mismatch = preset && keyFor && keyFor.id !== preset.id ? keyFor : null
  const groups = useMemo(
    () => (preset ? presetModelGroups(preset, tierLabel, TIER_ORDER) : []),
    [preset, tierLabel],
  )
  const setName = preset ? presetSetName(preset, existingNames) : ''
  // A hold only counts for the provider it was about.
  const held = hold && preset && hold.preset === preset.id ? hold : null

  const pick = (id: string) => { setPicked(id); setHold(null) }
  const onKey = (v: string) => {
    setApiKey(v); setHold(null)
    // Nothing picked yet: a key that names its provider picks it.
    if (!picked) {
      const p = presetForKey(presets, v)
      if (p) setPicked(p.id)
    }
  }

  const create = () => {
    if (!preset || !key) return
    onCreate(setName, presetTiers(preset, key))
  }

  const submit = () => {
    if (!preset || !key || busy) return
    // Already told: create anyway.
    if (held && held.kind !== 'refused') create()
    else runCheck()
  }

  const runCheck = async () => {
    if (!preset || !key) return
    setBusy(true); setHold(null)
    try {
      const res = await checkPresetKey(preset.id, key)
      if (res.ok) {
        if (res.missing.length) setHold({ kind: 'missing', preset: preset.id, models: res.missing })
        else create()
      } else if (res.refused) {
        setHold({ kind: 'refused', preset: preset.id })
      } else {
        setHold({ kind: 'unreachable', preset: preset.id })
      }
    } finally {
      setBusy(false)
    }
  }

  const ready = !!preset && !!key && !(held?.kind === 'refused')
  const submitLabel = busy
    ? 'Checking your key…'
    : held && held.kind !== 'refused'
      ? 'Create anyway'
      : preset ? `Create “${setName}”` : 'Create tier set'

  return (
    <form className="space-y-5" onSubmit={(e) => { e.preventDefault(); submit() }}>
      <Field label="Your AI provider">
        <div role="radiogroup" aria-label="Your AI provider" className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {presets.map((p) => {
            const on = p.id === picked
            return (
              <button
                key={p.id}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => pick(p.id)}
                className={`flex min-h-[4.25rem] flex-col items-start justify-center rounded-xl border px-3 py-2 text-left transition-colors ${
                  on
                    ? 'border-violet-500/60 bg-violet-500/10 ring-1 ring-violet-500/30'
                    : 'border-zinc-800 bg-zinc-950/40 hover:border-zinc-600'
                }`}
              >
                <span className="flex w-full items-center gap-1.5 text-sm font-semibold text-zinc-100">
                  {p.label}
                  {on && <Check className="ml-auto h-4 w-4 shrink-0 text-violet-600 dark:text-violet-400" />}
                </span>
                <span className="mt-0.5 text-[11px] leading-snug text-zinc-500">{p.blurb}</span>
              </button>
            )
          })}
        </div>
      </Field>

      <Field
        label={preset ? `${preset.label} API key` : 'API key'}
        hint={preset ? (
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <a
              href={preset.key_url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 font-medium text-violet-600 hover:underline dark:text-violet-400"
            >
              Get a key at {keyHost(preset.key_url)} <ExternalLink className="h-3 w-3" />
            </a>
            <span>Saved with this tier set; Captain uses it only to call {preset.label}.</span>
          </span>
        ) : 'Pick your provider, or paste a key and Captain picks it for you.'}
      >
        <div className="relative">
          <input
            type={shown ? 'text' : 'password'}
            value={apiKey}
            onChange={(e) => onKey(e.target.value)}
            placeholder={preset?.key_placeholder || 'Paste your API key'}
            autoComplete="off"
            spellCheck={false}
            aria-label={preset ? `${preset.label} API key` : 'API key'}
            className={`${inputCls} pr-10 font-mono`}
          />
          <button
            type="button"
            onClick={() => setShown((v) => !v)}
            aria-label={shown ? 'Hide the key' : 'Show the key'}
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-zinc-500 hover:text-zinc-200"
          >
            {shown ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        </div>
        {mismatch && (
          <p className="mt-2 flex flex-wrap items-center gap-2 text-xs text-amber-700 dark:text-amber-300">
            <AlertTriangle className="h-3.5 w-3.5" />
            This looks like a {mismatch.label} key.
            <button type="button" onClick={() => pick(mismatch.id)} className="font-medium underline">Use {mismatch.label}</button>
          </p>
        )}
      </Field>

      {preset && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-950/40 px-4 py-3">
          <div className="text-xs font-medium text-zinc-400">What each kind of work runs on</div>
          <ul className="mt-2 space-y-1.5">
            {groups.map((g) => (
              <li key={g.model} className="flex flex-col gap-0.5 text-xs sm:flex-row sm:items-baseline sm:gap-3">
                <span className="shrink-0 font-mono text-zinc-200 sm:w-56 sm:truncate" title={g.model}>{g.model}</span>
                <span className="text-zinc-500">{g.tiers.join(', ')}</span>
              </li>
            ))}
          </ul>
          <p className="mt-2.5 text-[11px] text-zinc-500">
            Change any tier afterwards.{checked && ` Current models as of ${fmtChecked(checked)}.`}
          </p>
        </div>
      )}

      {held?.kind === 'refused' && preset && (
        <Notice tone="red">
          {preset.label} didn't accept this key. Check that you copied all of it
          {mismatch ? <>, or use it with {mismatch.label}</> : null}.
        </Notice>
      )}
      {held?.kind === 'unreachable' && preset && (
        <Notice tone="amber">
          Couldn't reach {preset.label} to check the key. Create the set anyway, or{' '}
          <button type="button" onClick={runCheck} className="font-medium underline">check again</button>.
        </Notice>
      )}
      {held?.kind === 'missing' && preset && (
        <Notice tone="amber">
          The key works, but {preset.label} doesn't offer{' '}
          {held.models.map((m, i) => <span key={m}>{i > 0 && ', '}<code className="font-mono">{m}</code></span>)}
          {' '}to it yet. Tiers on {held.models.length === 1 ? 'that model' : 'those models'} won't run until it does
          — pick another model for them after.
        </Notice>
      )}

      <div className="flex justify-end pt-1">
        <button
          type="submit"
          disabled={!ready || busy}
          className="inline-flex items-center gap-1.5 rounded-xl bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50"
        >
          <Check className="h-4 w-4" /> {submitLabel}
        </button>
      </div>
    </form>
  )
}
