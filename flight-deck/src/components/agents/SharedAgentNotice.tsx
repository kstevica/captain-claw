import { useState, type ReactNode } from 'react'
import { Info, X } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { memberNoticeText, sharedAckKey } from '../../utils/sharedAgent'

// What a member should know before chatting with an agent somebody shared
// with them: whose it is, who can read along, what the agent keeps. Shown
// until dismissed, once per agent per deck user in this browser — several
// members can share one machine (a kiosk), and each must see it themselves.
// Storage is a convenience only — blocked storage just means the notice shows
// again.

function ackKey(agentRef: string): string {
  return sharedAckKey(useAuthStore.getState().user?.id, agentRef)
}

function readAck(agentRef: string): boolean {
  try { return window.localStorage.getItem(ackKey(agentRef)) === '1' } catch { return false }
}

function writeAck(agentRef: string): void {
  try { window.localStorage.setItem(ackKey(agentRef), '1') } catch { /* storage blocked */ }
}

/** `**bold**` spans → <strong>; everything else as text (line breaks kept by CSS). */
function withBold(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean).map((part, i) =>
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={i} className="font-semibold">{part.slice(2, -2)}</strong>
      : <span key={i}>{part}</span>,
  )
}

/** Render with `key={agentRef}` so each agent gets its own dismissal. */
export function SharedAgentNotice({ agentRef, agentName, ownerName, hostWarning }: {
  agentRef: string
  agentName: string
  ownerName: string
  hostWarning: string
}) {
  const [acked, setAcked] = useState(() => readAck(agentRef))
  if (acked) return null
  return (
    <div className="flex items-start gap-2 border-b border-sky-500/25 bg-sky-500/10 px-3 py-2 text-[11px] leading-relaxed text-sky-900 dark:text-sky-100">
      <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-sky-600 dark:text-sky-400" />
      <div className="min-w-0 flex-1 whitespace-pre-line">
        {withBold(memberNoticeText(agentName, ownerName, hostWarning))}
      </div>
      <button
        onClick={() => { writeAck(agentRef); setAcked(true) }}
        title="Got it"
        aria-label="Dismiss"
        className="shrink-0 rounded p-0.5 text-sky-700 hover:bg-sky-500/20 dark:text-sky-300"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}
