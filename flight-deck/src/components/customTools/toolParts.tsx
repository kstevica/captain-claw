import { Bot, Globe, KeyRound, MessageCircleQuestion, Timer, Workflow } from 'lucide-react'
import type { Declares, ToolSummary, ToolUse } from '../../services/customToolsApi'
import { Pill } from '../console/parts'

// Shown on the Custom tools page and in a tool's sheet.

export function statusPill(t: ToolSummary) {
  if (t.publish?.state === 'running') return <Pill tone="amber">Publishing…</Pill>
  if (t.live) return <Pill tone="emerald">Live · {t.live.version.slice(0, 7)}</Pill>
  if (t.publish?.state === 'failed') return <Pill tone="red">Publish failed</Pill>
  return <Pill>Not published</Pill>
}

/** A tool.json's uses, in the words Flight Deck's declares use. */
export function usesWords(uses: ToolUse[] | undefined): { uses: string[]; runs: string[] } {
  const list = uses || []
  return {
    uses: list.filter((u) => u.tool).map((u) => {
      const extra = u.actions?.length ? u.actions : u.hosts?.length ? u.hosts : []
      return extra.length ? `${u.tool} (${extra.join(', ')})` : String(u.tool)
    }),
    runs: list.filter((u) => u.custom_tool).map((u) => String(u.custom_tool)),
  }
}

export function DeclaresLine({ d }: { d: Declares | null | undefined }) {
  if (!d) return null
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      <Pill tone={d.network ? 'amber' : 'zinc'} icon={Globe}>{d.network ? 'Uses the internet' : 'No internet'}</Pill>
      {d.secrets.length > 0 && <Pill icon={KeyRound}>Needs {d.secrets.join(', ')}</Pill>}
      {(d.uses || []).length > 0 && <Pill tone="violet" icon={Bot}>Calls your agent: {(d.uses || []).join(', ')}</Pill>}
      {(d.runs || []).length > 0 && <Pill icon={Workflow}>Runs {(d.runs || []).join(', ')}</Pill>}
      {d.asks && <Pill tone="violet" icon={MessageCircleQuestion}>Asks you questions</Pill>}
      <Pill icon={Timer}>Up to {d.timeout_s} s</Pill>
    </div>
  )
}
