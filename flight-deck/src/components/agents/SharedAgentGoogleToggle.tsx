import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import type { SharedAgent } from '../../services/sharedAgents'
import { useSharedAgentStore } from '../../stores/sharedAgentStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { googleOptInHint, googleOptInLabel, googleOptInWarning, sharedCaps } from '../../utils/sharedAgent'

// A member's own switch: may this shared agent use MY Google account during
// MY chats with it? Off until they turn it on (with a warning about what that
// hands the owner's agent). Only for agents whose member chats can use it at
// all — process agents on a deck that says so; a Docker agent (or an older
// Flight Deck) gets nothing here. Member-only APIs, so it works in the kiosk
// too (a kiosk user connects Google in the kiosk Connections dialog).

export function SharedAgentGoogleToggle({ agent, compact = false }: { agent: SharedAgent; compact?: boolean }) {
  const [busy, setBusy] = useState(false)
  if (!sharedCaps(agent).google) return null

  const name = agent.name || agent.slug
  const owner = agent.owner_name || agent.owner_email || 'another user'
  const on = agent.google_enabled === true
  const label = googleOptInLabel(name)
  const hint = googleOptInHint(agent)

  const toggle = async () => {
    if (busy) return
    const next = !on
    if (next && !window.confirm(googleOptInWarning(name, owner))) return
    setBusy(true)
    try {
      await useSharedAgentStore.getState().setGoogle(agent.agent_ref, next)
    } catch (e) {
      useNotificationStore.getState().add('error', 'Could not change Google access',
        e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const control = (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      onClick={toggle}
      disabled={busy}
      className={`relative h-4 w-7 shrink-0 rounded-full transition-colors disabled:cursor-wait disabled:opacity-60 ${
        on ? 'bg-sky-500' : 'bg-zinc-700'}`}
    >
      <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${
        on ? 'translate-x-3.5' : 'translate-x-0.5'}`} />
    </button>
  )

  if (compact) {
    return (
      <div className="flex min-w-0 items-center gap-2 text-[11px] text-zinc-400" title={hint || undefined}>
        {control}
        <span className="min-w-0 truncate">{label}</span>
        {busy && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-zinc-500" />}
        {/* Visible, not only a tooltip: kiosks are touch screens with no hover. */}
        {hint && !busy && (
          <span className="shrink-0 rounded-full border border-amber-500/30 bg-amber-500/10 px-1.5 text-[10px] text-amber-600 dark:text-amber-400">
            Google not connected
          </span>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex min-w-0 items-center gap-2 text-xs text-zinc-300">
        {control}
        <span className="min-w-0 flex-1">{label}</span>
        {busy && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-zinc-500" />}
      </div>
      {hint && <p className="text-[11px] text-zinc-500">{hint}</p>}
    </div>
  )
}
