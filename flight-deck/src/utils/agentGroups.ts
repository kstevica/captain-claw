import type { AgentGroup } from '../stores/groupStore'

// ── Agent groups: colours and sections ──
//
// A group's colour, as whole class names (Tailwind keeps only what it can read),
// and how a list of agents falls into the sections the kiosk's column shows. A
// "role" is a label of the full layout's, not a place in the list.

export interface GroupTone {
  bg: string
  text: string
  border: string
  dot: string
  /** The line beside a group's agents. */
  rail: string
  /** A section something is being dragged over. */
  over: string
}

export const GROUP_TONES: Record<string, GroupTone> = {
  violet: { bg: 'bg-violet-500/15', text: 'text-violet-400', border: 'border-violet-500/25', dot: 'bg-violet-400', rail: 'border-violet-400/30', over: 'bg-violet-500/10 ring-violet-400/50' },
  blue: { bg: 'bg-blue-500/15', text: 'text-blue-400', border: 'border-blue-500/25', dot: 'bg-blue-400', rail: 'border-blue-400/30', over: 'bg-blue-500/10 ring-blue-400/50' },
  emerald: { bg: 'bg-emerald-500/15', text: 'text-emerald-400', border: 'border-emerald-500/25', dot: 'bg-emerald-400', rail: 'border-emerald-400/30', over: 'bg-emerald-500/10 ring-emerald-400/50' },
  amber: { bg: 'bg-amber-500/15', text: 'text-amber-400', border: 'border-amber-500/25', dot: 'bg-amber-400', rail: 'border-amber-400/30', over: 'bg-amber-500/10 ring-amber-400/50' },
  pink: { bg: 'bg-pink-500/15', text: 'text-pink-400', border: 'border-pink-500/25', dot: 'bg-pink-400', rail: 'border-pink-400/30', over: 'bg-pink-500/10 ring-pink-400/50' },
  cyan: { bg: 'bg-cyan-500/15', text: 'text-cyan-400', border: 'border-cyan-500/25', dot: 'bg-cyan-400', rail: 'border-cyan-400/30', over: 'bg-cyan-500/10 ring-cyan-400/50' },
  red: { bg: 'bg-red-500/15', text: 'text-red-400', border: 'border-red-500/25', dot: 'bg-red-400', rail: 'border-red-400/30', over: 'bg-red-500/10 ring-red-400/50' },
  indigo: { bg: 'bg-indigo-500/15', text: 'text-indigo-400', border: 'border-indigo-500/25', dot: 'bg-indigo-400', rail: 'border-indigo-400/30', over: 'bg-indigo-500/10 ring-indigo-400/50' },
}

export const GROUP_COLORS = Object.keys(GROUP_TONES)

export function groupTone(color: string): GroupTone {
  return GROUP_TONES[color] || GROUP_TONES.violet
}

/** The colour fewest groups have: a new group stands out from the others. */
export function nextGroupColor(groups: AgentGroup[]): string {
  const uses = (color: string) => groups.filter((g) => g.color === color).length
  return GROUP_COLORS.reduce((best, c) => (uses(c) < uses(best) ? c : best))
}

export function isFolder(g: AgentGroup): boolean {
  return (g.type || 'group') === 'group'
}

/** The groups as the list shows them: the pinned one first, the rest in their order. */
export function orderedFolders(groups: AgentGroup[]): AgentGroup[] {
  const folders = groups.filter(isFolder)
  const top = folders.find((g) => g.pinned)
  return top ? [top, ...folders.filter((g) => g !== top)] : folders
}

/** The section of the agents in no group (collapses like the others). */
export const OTHER_SECTION = 'other'

export interface AgentSection<A> {
  id: string
  /** Null: the agents in no group. */
  group: AgentGroup | null
  name: string
  agents: A[]
}

/** Those pinned first, as they were pinned (a power switch doesn't move them), then the rest as they come. */
function pinnedFirst<A extends { id: string }>(agents: A[], pinned: string[] = []): A[] {
  if (pinned.length === 0) return agents
  const on = pinned.flatMap((id) => agents.filter((a) => a.id === id))
  return [...on, ...agents.filter((a) => !pinned.includes(a.id))]
}

/** Each group in the list's order (empty ones too), then the agents in none.
 *  Within a section the agents keep the order they come in, those pinned first.
 *  One placed in two groups in the full layout shows in both. */
export function agentSections<A extends { id: string }>(agents: A[], groups: AgentGroup[]): AgentSection<A>[] {
  const folders = orderedFolders(groups)
  const placed = new Set(folders.flatMap((g) => g.agentIds))
  return [
    ...folders.map((g) => {
      const members = new Set(g.agentIds)
      return { id: g.id, group: g, name: g.name, agents: pinnedFirst(agents.filter((a) => members.has(a.id)), g.pinnedAgentIds) }
    }),
    { id: OTHER_SECTION, group: null, name: 'Other agents', agents: agents.filter((a) => !placed.has(a.id)) },
  ]
}
