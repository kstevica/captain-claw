// The chat box's model picker: the other models of the agent's own provider
// it can switch to — same key, same endpoint (captain_claw/llm/model_switch.py).
// The agent says which: the big providers' current models from the tier
// presets catalog, the models an admin listed for a custom endpoint (Console
// → Models), and its own allowed models. Tier sets are not involved.

export interface ModelChoice {
  model: string
  /** As a person reads it ("Claude Opus 5.5"). */
  name: string
  /** What it is for ("Smartest", "Everyday", "Fastest and cheapest"), or "". */
  label: string
  /** The admin's or the agent's note on it, or "". */
  note: string
  /** USD per 1M tokens, when known. */
  price: { input: number; output: number } | null
  /** The model the agent is configured to run on. */
  default: boolean
}

export interface ModelSwitchInfo {
  /** What it runs on now. */
  current: string
  /** What it is configured to run on. */
  default: string
  /** The current model is for the rest of the agent's turn only. */
  turnOnly: boolean
  /** The owner lets it switch on its own (Agent settings). */
  selfSwitch: boolean
  choices: ModelChoice[]
}

/** The `model_switch` block of the agent's session info, or null (no picker). */
export function parseModelSwitch(raw: unknown): ModelSwitchInfo | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const choices: ModelChoice[] = (Array.isArray(r.choices) ? r.choices : [])
    .filter((c): c is Record<string, unknown> => !!c && typeof c === 'object' && typeof (c as Record<string, unknown>).model === 'string')
    .map((c) => {
      const p = c.price as Record<string, unknown> | null | undefined
      const price = p && typeof p === 'object' && (Number(p.input) > 0 || Number(p.output) > 0)
        ? { input: Number(p.input) || 0, output: Number(p.output) || 0 }
        : null
      return {
        model: String(c.model),
        name: String(c.name || c.model),
        label: String(c.label ?? ''),
        note: String(c.note ?? ''),
        price,
        default: !!c.default,
      }
    })
  if (choices.length < 2) return null
  return {
    current: String(r.current ?? ''),
    default: String(r.default ?? ''),
    turnOnly: !!r.turn_only,
    selfSwitch: !!r.self_switch,
    choices,
  }
}

/** Is `running` (maybe with a routing prefix) the choice `model`? */
export function isModel(running: string, model: string): boolean {
  const a = running.trim().toLowerCase()
  const b = model.trim().toLowerCase()
  if (!a || !b) return false
  return a === b || a.endsWith(`/${b}`) || b.endsWith(`/${a}`) || a.split('/').pop() === b.split('/').pop()
}

function usd(v: number): string {
  if (v >= 1) return `$${Number.isInteger(v) ? v : v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')}`
  return `$${v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '') || '0'}`
}

/** "$3 / $15 per 1M tokens" (in / out), or "". */
export function priceText(price: ModelChoice['price']): string {
  if (!price) return ''
  return `${usd(price.input)} in · ${usd(price.output)} out per 1M tokens`
}
