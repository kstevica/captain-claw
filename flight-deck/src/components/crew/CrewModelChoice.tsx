import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Loader2 } from 'lucide-react'
import { Segmented } from '../console/parts'
import { crewApi, type CrewHarness, type CrewModelChoices, type CrewModelOption, type CrewModelSource, type CrewOwnModel } from '../../services/crewApi'
import { ErrorLine, errorOf, fieldClass, quietBtn } from './crewParts'

// ── What a worker runs on, when it isn't its CLI's own account ──
//
// Whichever it is, the key stays on the deck: the worker calls the model through it,
// with a key good for one of its turns (`captain_claw/flight_deck/crew_model.py`).

export type ModelKind = 'own' | 'agent' | 'tier' | 'typed'

export interface ModelChoice {
  kind: ModelKind
  agent: string
  tier: string
  url: string
  model: string
  context: string
  key: string
}

export const NOTHING_CHOSEN: ModelChoice = { kind: 'own', agent: '', tier: '', url: '', model: '', context: '', key: '' }

const fits = (option: CrewModelOption, harness: string) => option.works_with.includes(harness)

/** The one chosen, or the first that this CLI can use. */
export function picked(options: CrewModelOption[] | undefined, id: string, harness: string): CrewModelOption | undefined {
  const usable = (options ?? []).filter((o) => fits(o, harness))
  return usable.find((o) => o.id === id) ?? usable[0]
}

/** What goes with a new task. Nothing, for its CLI's own; undefined too when what's needed isn't there yet. */
export function sourceOf(choice: ModelChoice, choices: CrewModelChoices | null, harness: string): CrewModelSource | undefined {
  if (choice.kind === 'agent') {
    const agent = picked(choices?.agents, choice.agent, harness)
    return agent && { kind: 'agent', agent: agent.id }
  }
  if (choice.kind === 'tier') {
    const tier = picked(choices?.tiers, choice.tier, harness)
    return tier && { kind: 'tier', tier: tier.id }
  }
  if (choice.kind === 'typed' && choice.url.trim() && choice.model.trim()) {
    return { kind: 'typed', url: choice.url.trim(), model: choice.model.trim(), context: choice.context.trim(), key: choice.key.trim() }
  }
  return undefined
}

function Options({ noun, options, id, harness, onPick }: {
  noun: string; options: CrewModelOption[]; id: string; harness: CrewHarness; onPick: (id: string) => void
}) {
  const chosen = picked(options, id, harness.id)
  const cant = options.filter((o) => !fits(o, harness.id))
  const why = cant.find((o) => o.why_not)?.why_not
  return (
    <div className="space-y-1.5">
      <select value={chosen?.id ?? ''} onChange={(e) => onPick(e.target.value)} aria-label={noun} className={fieldClass}>
        {!chosen && <option value="">{harness.label} can't use any of these</option>}
        {options.map((o) => (
          <option key={o.id} value={o.id} disabled={!fits(o, harness.id)}>
            {o.label}{o.model ? ` · ${o.model}` : ''}{fits(o, harness.id) ? '' : ` (not for ${harness.label})`}
          </option>
        ))}
      </select>
      {!chosen && why && <p className="text-xs text-zinc-500">{why}</p>}
      {!chosen && !why && <p className="text-xs text-zinc-500">OpenCode, Crush and Pi can talk to any of them. Claude Code talks only to servers with Anthropic's API, and Codex only to those with OpenAI's Responses API.</p>}
      {chosen && (
        <p className="text-xs text-zinc-500">
          {harness.label} calls it through Captain, and never sees its key. Its tokens count toward your daily limit, unless they're on your own key.
        </p>
      )}
    </div>
  )
}

/** Its address, name, context size and key, for this task. */
function Typed({ choice, own, harness, onChange }: {
  choice: ModelChoice; own?: CrewOwnModel | null; harness: CrewHarness; onChange: (next: Partial<ModelChoice>) => void
}) {
  const [found, setFound] = useState<string[]>([])
  const [checking, setChecking] = useState(false)
  const [said, setSaid] = useState('')
  const [error, setError] = useState('')
  const check = async () => {
    setChecking(true); setError(''); setSaid('')
    try {
      const got = await crewApi.checkOwnModel(choice.url.trim(), choice.key.trim())
      if (got.error) setError(got.error)
      else {
        setFound(got.models)
        setSaid(got.models.length ? 'It answered. Pick a model.' : 'It answered, but listed no models.')
        if (!choice.model.trim() && got.models[0]) onChange({ model: got.models[0] })
      }
    } catch (e) {
      setError(errorOf(e))
    }
    setChecking(false)
  }
  const savedKey = !!own?.has_key && own.url === choice.url.trim()
  return (
    <div className="space-y-2">
      <div className="grid gap-2 sm:grid-cols-2">
        <input value={choice.url} onChange={(e) => onChange({ url: e.target.value })} placeholder="API address, like http://192.168.1.20:11434/v1" aria-label="API address" autoComplete="off" spellCheck={false} className={`${fieldClass} sm:col-span-2`} />
        <input list="crew-typed-models" value={choice.model} onChange={(e) => onChange({ model: e.target.value })} placeholder="Model name, as the server calls it" aria-label="Model name" autoComplete="off" spellCheck={false} className={fieldClass} />
        <datalist id="crew-typed-models">{found.map((m) => <option key={m} value={m} />)}</datalist>
        <input value={choice.context} onChange={(e) => onChange({ context: e.target.value.replace(/\D/g, '') })} inputMode="numeric" placeholder="Context size in tokens (optional)" aria-label="Context size" autoComplete="off" className={fieldClass} />
        <input type="password" value={choice.key} onChange={(e) => onChange({ key: e.target.value })} placeholder={savedKey ? 'API key: the one saved in Logins' : 'API key (blank if it needs none)'} aria-label="API key" autoComplete="new-password" className={`${fieldClass} sm:col-span-2`} />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" disabled={checking || !choice.url.trim()} onClick={check} className={quietBtn}>
          {checking && <Loader2 className="h-4 w-4 animate-spin" />} Check it
        </button>
        {said && <span className="text-xs text-zinc-500">{said}</span>}
      </div>
      <ErrorLine text={error} />
      <p className="text-xs leading-relaxed text-zinc-500">
        It's kept for this task only, and {harness.label} never sees the key. The address has to be reachable from this deck.
        {harness.id === 'claude' && ' Claude Code needs a server with Anthropic\'s API: Ollama and OpenRouter have one.'}
        {harness.id === 'codex' && ' Codex needs a server with OpenAI\'s Responses API: Ollama and OpenRouter have one.'}
      </p>
    </div>
  )
}

export function ModelChooser({ choice, choices, harness, own, loadError, onChange, onLogins, children }: {
  choice: ModelChoice
  choices: CrewModelChoices | null
  harness?: CrewHarness
  own?: CrewOwnModel | null
  loadError: string
  onChange: (next: Partial<ModelChoice>) => void
  onLogins: () => void
  children: ReactNode  // choosing among its CLI's own
}) {
  const noLogin = !!harness && !harness.login.ok
  const movedForThem = useRef(false)
  // Nothing to run on as it stands: start it on what there is, not on a dead end. Back on a CLI with
  // a login, it goes back too: nobody chose to spend the tier for that one.
  useEffect(() => {
    if (noLogin && choice.kind === 'own' && choices && harness) {
      movedForThem.current = true
      onChange({ kind: picked(choices.tiers, '', harness.id) ? 'tier' : picked(choices.agents, '', harness.id) ? 'agent' : 'typed' })
    } else if (!noLogin && movedForThem.current) {
      movedForThem.current = false
      onChange({ kind: 'own' })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [noLogin, choices, harness?.id])
  const pick = (kind: ModelKind) => {
    movedForThem.current = false
    // What's saved as their own model is what they'd type here.
    const fill = kind === 'typed' && !choice.url && own ? { url: own.url, model: own.models[0] ?? '', context: own.context ? String(own.context) : '' } : {}
    onChange({ kind, ...fill })
  }
  return (
    <div className="space-y-2">
      <Segmented
        label="What it runs on" value={choice.kind} onChange={pick}
        options={[
          { value: 'own' as const, label: 'Its own' },
          { value: 'agent' as const, label: "An agent's model" },
          { value: 'tier' as const, label: 'A deck tier' },
          { value: 'typed' as const, label: 'Enter one' },
        ]}
      />
      {choice.kind === 'own' && (noLogin
        ? <p className="text-xs text-zinc-500">{harness?.label} has no login yet. <button type="button" onClick={onLogins} className="underline">Add one</button>, or choose another model above.</p>
        : children)}
      {choice.kind !== 'own' && <ErrorLine text={loadError} />}
      {choice.kind !== 'own' && !choices && !loadError && <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />}
      {harness && choices && choice.kind === 'agent' && (choices.agents.length
        ? <Options noun="Agent" options={choices.agents} id={choice.agent} harness={harness} onPick={(agent) => onChange({ agent })} />
        : <p className="text-xs text-zinc-500">You have no agents on this deck.</p>)}
      {harness && choices && choice.kind === 'tier' && (choices.tiers.length
        ? <Options noun="Tier" options={choices.tiers} id={choice.tier} harness={harness} onPick={(tier) => onChange({ tier })} />
        : <p className="text-xs text-zinc-500">This deck has no tiers set up. An admin can publish them in Library.</p>)}
      {harness && choice.kind === 'typed' && <Typed choice={choice} own={own} harness={harness} onChange={onChange} />}
    </div>
  )
}
