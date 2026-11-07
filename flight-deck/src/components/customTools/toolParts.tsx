import { Globe, KeyRound, Timer } from 'lucide-react'
import type { Declares, ToolSummary } from '../../services/customToolsApi'
import { Pill } from '../console/parts'

// Shown on the Custom tools page and in a tool's sheet.

export function statusPill(t: ToolSummary) {
  if (t.publish?.state === 'running') return <Pill tone="amber">Publishing…</Pill>
  if (t.live) return <Pill tone="emerald">Live · {t.live.version.slice(0, 7)}</Pill>
  if (t.publish?.state === 'failed') return <Pill tone="red">Publish failed</Pill>
  return <Pill>Not published</Pill>
}

export function DeclaresLine({ d }: { d: Declares | null | undefined }) {
  if (!d) return null
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      <Pill tone={d.network ? 'amber' : 'zinc'} icon={Globe}>{d.network ? 'Uses the internet' : 'No internet'}</Pill>
      {d.secrets.length > 0 && <Pill icon={KeyRound}>Needs {d.secrets.join(', ')}</Pill>}
      <Pill icon={Timer}>Up to {d.timeout_s} s</Pill>
    </div>
  )
}
