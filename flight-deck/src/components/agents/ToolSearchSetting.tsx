import { useEffect, useState } from 'react'
import { AlertTriangle, Check, Loader2 } from 'lucide-react'
import { getAgentToolSearch, setAgentToolSearch, type AgentToolSearch } from '../../services/docker'
import { SwitchTrack } from '../common/SwitchTrack'

// Tool search (captain_claw/tool_search.py): past a budget of tool descriptions
// per message, the agent sends the everyday tools and the ones it used, and
// looks up the rest when it needs them. Full mode only — eco mode keeps its own
// smaller set. Saved at once; read on every message.
const BUDGETS = [8000, 16000, 24000, 32000]

const thousands = (n: number) => `${Math.round(n / 1000)}k`

export function ToolSearchSetting({ kind, identifier }: { kind: 'docker' | 'process'; identifier: string }) {
  // undefined: loading; null: the deck couldn't say.
  const [state, setState] = useState<AgentToolSearch | null | undefined>(undefined)
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let live = true
    getAgentToolSearch(kind, identifier)
      .then((r) => { if (live) setState(r) })
      .catch(() => { if (live) setState(null) })
    return () => { live = false }
  }, [kind, identifier])

  if (state === undefined) {
    return <p className="flex items-center gap-1.5 text-xs text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" /> Loading…</p>
  }
  if (state === null) return <p className="text-xs text-zinc-500">Not available for this agent.</p>

  const save = async (enabled: boolean, budget: number) => {
    if (busy) return
    const was = state
    setState({ ...state, enabled, budget_tokens: budget }); setBusy(true); setError(''); setSaved(false)
    try {
      await setAgentToolSearch(kind, identifier, enabled, budget)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      setState(was)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  // A budget set by hand in its config shows beside the usual ones.
  const budgets = BUDGETS.includes(state.budget_tokens)
    ? BUDGETS
    : [...BUDGETS, state.budget_tokens].sort((a, b) => a - b)

  return (
    <div className="space-y-2">
      <button
        type="button" role="switch" aria-checked={state.enabled} disabled={busy}
        onClick={() => save(!state.enabled, state.budget_tokens)}
        className="flex w-full items-start gap-3 rounded-lg border border-zinc-800 px-3 py-2.5 text-left transition-colors hover:border-zinc-700 hover:bg-zinc-900 disabled:cursor-wait"
      >
        <span className="mt-0.5"><SwitchTrack on={state.enabled} /></span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5 text-xs font-medium text-zinc-200">
            Look up rarely used tools when needed
            {busy && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
            {saved && <Check className="h-3 w-3 text-emerald-500" />}
          </span>
          <span className="block text-[11px] leading-relaxed text-zinc-500">
            Every message carries a description of each tool the agent can use — about 40,000 tokens for a
            typical agent. On, only the everyday tools and the ones it has used go with every message, up to the
            limit below; it looks up the rest when it needs one, which can take one extra step. Applies from its
            next message.
          </span>
        </span>
      </button>

      {state.enabled && (
        <div>
          <span className="mb-1 block text-[11px] font-medium text-zinc-400">Tool descriptions per message, at most</span>
          <div role="radiogroup" aria-label="Tool descriptions per message" className="flex flex-wrap gap-1.5">
            {budgets.map((b) => {
              const on = b === state.budget_tokens
              return (
                <button
                  key={b}
                  type="button" role="radio" aria-checked={on} disabled={busy}
                  onClick={() => { if (!on) save(true, b) }}
                  className={`min-w-[3.5rem] rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors disabled:cursor-wait ${
                    on ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-300' :'border-zinc-800 text-zinc-300 hover:border-zinc-700 hover:bg-zinc-900'
                  }`}
                  title={`${b.toLocaleString()} tokens`}
                >
                  {thousands(b)}
                </button>
              )
            })}
          </div>
          <span className="mt-1 block text-[11px] leading-snug text-zinc-500">
            Lower saves more; higher means fewer look-ups. 16k suits most agents.
          </span>
        </div>
      )}

      {state.eco && (
        <p className="flex items-start gap-1.5 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[11px] leading-relaxed text-amber-700 dark:text-amber-300">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
          Eco mode is on, and it already sends a smaller set of tools. This setting applies when Eco mode is off.
        </p>
      )}
      {error && <p className="text-[11px] text-red-500">{error}</p>}
    </div>
  )
}
