import { useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react'
import { Check, CornerDownRight, Loader2, Undo2, UserRound } from 'lucide-react'
import { AgentPicture } from '../agents/AgentPicture'
import { GroupSheet } from './KioskAgentGroups'
import { adoptRefusal, inFamily, parentIn, type FamilyAgent, type FamilyMove } from '../../utils/agentFamily'

// ── The agent family in the agent column (docs/agent-family-plan.md, P3) ──
//
// An agent another agent made sits under it; the person can move one of their
// agents under another (drag it onto that agent's row, or "⋯" → Put under).
// The deck holds the family (agent_family.py) and refuses what it won't do;
// these mirror its limits so a drag can say why before the drop.

// ── "⋯" → Put under ──

export function AgentParentPicker({ agent, agents, phone, onPick, onClose }: {
  agent: FamilyAgent
  agents: FamilyAgent[]
  phone: boolean
  onPick: (parentId: string) => Promise<void> | void
  onClose: () => void
}) {
  const byId = useMemo(() => new Map(agents.map((a) => [a.id, a])), [agents])
  const current = parentIn(byId, agent)?.id ?? ''
  const options = useMemo(() => agents
    .filter((a) => inFamily(a) && a.id !== agent.id)
    .map((a) => ({ a, why: adoptRefusal(agents, agent.id, a.id) }))
    .filter((o) => o.why !== null)
    .sort((x, y) => x.a.name.localeCompare(y.a.name)), [agents, agent.id])
  const [busy, setBusy] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (phone) return
    const here = listRef.current?.querySelector<HTMLElement>('[data-pick][aria-checked="true"]')
    ;(here ?? listRef.current?.querySelector<HTMLElement>('[data-pick]'))?.focus()
  }, [phone])

  const onArrows = (e: ReactKeyboardEvent) => {
    const way = { ArrowDown: 1, ArrowUp: -1 }[e.key]
    if (!way) return
    const rows = [...(listRef.current?.querySelectorAll<HTMLElement>('[data-pick]') ?? [])]
    const at = rows.indexOf(document.activeElement as HTMLElement)
    if (at < 0) return
    e.preventDefault()
    rows[Math.max(0, Math.min(rows.length - 1, at + way))]?.focus()
  }

  const pick = async (id: string) => {
    if (id === current) { onClose(); return }
    setBusy(id)
    try { await onPick(id) } finally { setBusy(null) }
  }
  const row = 'flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors hover:bg-zinc-800/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 disabled:opacity-50 disabled:hover:bg-transparent'
  const parentName = current ? byId.get(current)?.name : ''

  return (
    <GroupSheet
      label={`Put ${agent.name} under another agent`}
      picture={<span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-violet-500/15 text-violet-600 dark:text-violet-300"><CornerDownRight className="h-5 w-5" /></span>}
      title={agent.name}
      subtitle={parentName ? `Under ${parentName}` : 'One of your own agents'}
      phone={phone}
      onClose={onClose}
    >
      <div ref={listRef} onKeyDown={onArrows} className="min-h-0 flex-1 space-y-0.5 overflow-y-auto overscroll-contain px-3 pb-3 wide:px-4" role="radiogroup" aria-label="Put under">
        <button data-pick role="radio" aria-checked={!current} onClick={() => pick('')} disabled={busy !== null} className={`${row} ${!current ? 'bg-zinc-800/60' : ''}`}>
          <UserRound className="h-4 w-4 shrink-0 text-zinc-500" />
          <span className="min-w-0 flex-1">
            <span className="block truncate text-sm font-medium text-zinc-200">None — one of your own</span>
            <span className="block truncate text-xs text-zinc-500">At the top of the list, in its group</span>
          </span>
          {busy === '' ? <Loader2 className="h-4 w-4 animate-spin text-zinc-400" /> : !current && <Check className="h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />}
        </button>
        {options.map(({ a, why }) => {
          const here = a.id === current
          return (
            <button key={a.id} data-pick role="radio" aria-checked={here} onClick={() => pick(a.id)}
              disabled={busy !== null || (!!why && !here)} title={why || undefined}
              className={`${row} ${here ? 'bg-zinc-800/60' : ''}`}>
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-[var(--fd-selected)] text-[11px] font-semibold text-[var(--fd-ink)]">
                <AgentPicture agentId={a.id} name={a.name} />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-zinc-200">{a.name}</span>
                {why && !here && <span className="block truncate text-xs text-zinc-500">{why}</span>}
              </span>
              {busy === a.id ? <Loader2 className="h-4 w-4 animate-spin text-zinc-400" /> : here && <Check className="h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />}
            </button>
          )
        })}
        {options.length === 0 && (
          <p className="px-3 py-4 text-sm text-zinc-500">No other agent of yours can have agents under it yet.</p>
        )}
      </div>
    </GroupSheet>
  )
}

// ── What just moved, with an Undo ──

export function FamilyUndoBar({ move, onDone }: { move: FamilyMove | null; onDone: () => void }) {
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (!move) return
    const t = setTimeout(onDone, 8000)
    return () => clearTimeout(t)
  }, [move, onDone])
  if (!move) return null
  return (
    <div role="status" className="fd-fade mx-1 mt-1.5 flex items-center gap-2 rounded-[var(--fd-radius-control)] border border-violet-500/30 bg-violet-500/10 px-2.5 py-1.5 text-[11px] text-zinc-200">
      <CornerDownRight className="h-3.5 w-3.5 shrink-0 text-violet-600 dark:text-violet-300" />
      <span className="min-w-0 flex-1 truncate" title={move.text}>{move.text}</span>
      <button
        onClick={async () => { setBusy(true); try { await move.undo() } finally { setBusy(false); onDone() } }}
        disabled={busy}
        className="flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 font-medium text-violet-700 hover:bg-violet-500/15 disabled:opacity-60 dark:text-violet-300"
      >
        {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Undo2 className="h-3 w-3" />}Undo
      </button>
    </div>
  )
}

