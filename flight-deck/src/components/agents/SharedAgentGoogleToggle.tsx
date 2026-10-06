import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import type { SharedAgent } from '../../services/sharedAgents'
import { useSharedAgentStore } from '../../stores/sharedAgentStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { googleOptInHint, googleOptInLabel, googleOptInWarning, sharedCaps } from '../../utils/sharedAgent'
import { SwitchTrack } from '../common/SwitchTrack'

// A member's own switch: may this shared agent use MY Google account during
// MY chats with it? Off until they turn it on (with a warning about what that
// hands the owner's agent). Only for agents whose member chats can use it at
// all — process agents on a deck that says so; a Docker agent (or an older
// Flight Deck) gets nothing here. Member-only APIs, so it works in the kiosk
// too (a kiosk user connects Google in the kiosk Connections dialog).

// Google's "G": the switch is about the member's own Google account.
function GoogleMark() {
  return (
    <svg viewBox="0 0 48 48" aria-hidden="true" className="h-3.5 w-3.5 shrink-0">
      <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z" />
      <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z" />
      <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z" />
    </svg>
  )
}

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

  if (compact) {
    // The chat bar: one pill, the whole of it the switch. The agent is the
    // chat's own, so its name stays in the accessible name and the tooltip.
    // The words stay the same either way (aria-checked, the track and the
    // colour carry the state), and they sit inside the accessible name, so
    // voice control can say what it sees.
    // Sized by its container, not the viewport: the services drop out of a
    // narrow bar and the "not connected" chip wraps under the pill.
    return (
      <div className="@container flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1" title={hint || undefined}>
        <button
          type="button"
          role="switch"
          aria-checked={on}
          aria-label={label}
          title={hint || label}
          onClick={toggle}
          disabled={busy}
          className={`flex min-w-0 items-center gap-2 rounded-full border py-1 pl-1.5 pr-3 text-[11px] font-medium transition-colors disabled:cursor-wait disabled:opacity-70 ${
            on
              ? 'border-sky-500/40 bg-sky-500/10 text-sky-700 hover:bg-sky-500/15 dark:text-sky-300'
              : 'border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:bg-zinc-800/60 hover:text-zinc-200'}`}
        >
          <SwitchTrack on={on} />
          <GoogleMark />
          <span className="min-w-0 truncate">Use my Google</span>
          <span className="hidden min-w-0 truncate font-normal text-zinc-500 @md:block">Gmail · Calendar · Drive</span>
        </button>
        {busy && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-zinc-500" />}
        {/* Visible, not only a tooltip: kiosks are touch screens with no hover. */}
        {hint && !busy && (
          <span className="shrink-0 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-600 dark:text-amber-400">
            Google not connected
          </span>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1">
      <button
        type="button"
        role="switch"
        aria-checked={on}
        aria-label={label}
        onClick={toggle}
        disabled={busy}
        className="-mx-1.5 flex min-w-0 items-center gap-2 rounded-md px-1.5 py-1 text-left text-xs text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:cursor-wait disabled:opacity-70"
      >
        <GoogleMark />
        <span className="min-w-0 flex-1">{label}</span>
        {busy && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-zinc-500" />}
        <SwitchTrack on={on} />
      </button>
      {hint && <p className="text-[11px] text-zinc-500">{hint}</p>}
    </div>
  )
}
