import { useEffect, useState } from 'react'
import { Check, Loader2 } from 'lucide-react'
import { getAgentModelSwitch, setAgentModelSwitch } from '../../services/docker'
import { SwitchTrack } from '../common/SwitchTrack'

// Whether the agent may pick another model of its own provider for a task by
// itself (captain_claw/llm/model_switch.py) — off, it switches only when asked
// or picked in the chat box. Saved at once; read on every message.
export function ModelSwitchSetting({ kind, identifier }: { kind: 'docker' | 'process'; identifier: string }) {
  const [on, setOn] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let live = true
    getAgentModelSwitch(kind, identifier)
      .then((r) => { if (live) setOn(r.enabled) })
      .catch(() => { if (live) setOn(null) })
    return () => { live = false }
  }, [kind, identifier])

  if (on === null) return <p className="text-xs text-zinc-500">Not available for this agent.</p>

  const toggle = async () => {
    if (busy) return
    const next = !on
    setOn(next); setBusy(true); setError(''); setSaved(false)
    try {
      await setAgentModelSwitch(kind, identifier, next)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      setOn(!next)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <button
        type="button" role="switch" aria-checked={on} onClick={toggle} disabled={busy}
        className="flex w-full items-start gap-3 rounded-lg border border-zinc-800 px-3 py-2.5 text-left transition-colors hover:border-zinc-700 hover:bg-zinc-900 disabled:cursor-wait"
      >
        <span className="mt-0.5"><SwitchTrack on={on} /></span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5 text-xs font-medium text-zinc-200">
            Pick a smarter or cheaper model on its own
            {busy && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
            {saved && <Check className="h-3 w-3 text-emerald-500" />}
          </span>
          <span className="block text-[11px] leading-relaxed text-zinc-500">
            For one task at a time it may move to another model of the same provider and key — a smarter one for hard
            work, a cheaper one for simple work — and tells you when it does. Off, it switches only when you ask or pick
            a model in the chat box.
          </span>
        </span>
      </button>
      {error && <p className="mt-1.5 text-[11px] text-red-500">{error}</p>}
    </div>
  )
}
