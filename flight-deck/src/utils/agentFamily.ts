// The agent family on the client (docs/agent-family-plan.md, P3): the
// deck's rules (agent_family.py) mirrored, so a drag can say why before the
// drop; the deck still decides. Used by the agent column (AgentFamily.tsx).

export const MAX_DEPTH = 3
export const MAX_CHILDREN = 12

/** What the family rules need of an agent in the list. */
export interface FamilyAgent {
  id: string
  name: string
  kind: string
  /** The id of the agent it sits under, when that one is in the list too. */
  parent?: string
  role?: string
  managed?: boolean
}

/** A process agent of the person's own that stays: the only kind that can sit under another or have agents under it. */
export const inFamily = (a: FamilyAgent | undefined): a is FamilyAgent =>
  !!a && a.kind === 'process' && a.role !== 'worker' && !a.managed

export function parentIn(byId: Map<string, FamilyAgent>, a: FamilyAgent): FamilyAgent | undefined {
  const p = a.parent ? byId.get(a.parent) : undefined
  return p && p.id !== a.id ? p : undefined
}

/** Nearest first; stops at a loop, should the data hold one. */
export function ancestorsOf(byId: Map<string, FamilyAgent>, a: FamilyAgent): FamilyAgent[] {
  const out: FamilyAgent[] = []
  const seen = new Set([a.id])
  let cur = parentIn(byId, a)
  while (cur && !seen.has(cur.id) && out.length < 64) {
    out.push(cur)
    seen.add(cur.id)
    cur = parentIn(byId, cur)
  }
  return out
}

function kidsOf(all: FamilyAgent[], byId: Map<string, FamilyAgent>, id: string): FamilyAgent[] {
  return all.filter((x) => x.id !== id && inFamily(x) && parentIn(byId, x)?.id === id)
}

function heightOf(all: FamilyAgent[], byId: Map<string, FamilyAgent>, id: string): number {
  let level = [id]
  let h = 0
  const seen = new Set([id])
  for (;;) {
    const next = level.flatMap((s) => kidsOf(all, byId, s).map((k) => k.id)).filter((k) => !seen.has(k))
    if (!next.length) return h
    next.forEach((k) => seen.add(k))
    h += 1
    level = next
  }
}

/** Why ``child`` can't go under ``parent`` (the deck's words), "" when it can,
 *  or null when the pair isn't a family move at all (not two agents of yours). */
export function adoptRefusal(all: FamilyAgent[], childId: string, parentId: string): string | null {
  const byId = new Map(all.map((a) => [a.id, a]))
  const child = byId.get(childId)
  const parent = byId.get(parentId)
  if (!inFamily(child) || !inFamily(parent) || childId === parentId) return null
  if (ancestorsOf(byId, parent).some((x) => x.id === childId)) return `${parent.name} sits under ${child.name}: that would make a loop.`
  if (ancestorsOf(byId, parent).length + 1 + heightOf(all, byId, childId) > MAX_DEPTH) return `That would go past ${MAX_DEPTH} levels.`
  if (kidsOf(all, byId, parentId).filter((k) => k.id !== childId).length >= MAX_CHILDREN) return `${parent.name} already has ${MAX_CHILDREN} agents under it.`
  return ''
}

/** A process agent's slug from its list id (`proc-<slug>`). */
export const slugOf = (id: string) => (id.startsWith('proc-') ? id.slice(5) : id)

export interface FamilyMove {
  /** Says what happened: "Deal digest is now under Glasses". */
  text: string
  /** Puts it back. */
  undo: () => Promise<void> | void
}

// ── A run's workers, under whoever started the run ──

export interface WorkerLineItem {
  /** "Working · 2 workers", or a First Mate factory's "Inbox triage · Working · 2 workers". */
  label: string
  /** The workers' names, for the tooltip. */
  names: string[]
  onOpen?: () => void
}

/** Group a parent's running workers into lines: one per First Mate factory, one for the rest. */
export function workerLines(workers: { name: string; factory?: string; madeFor?: string }[], onFactory?: () => void): WorkerLineItem[] {
  const byKey = new Map<string, { title: string; names: string[] }>()
  for (const w of workers) {
    const key = w.factory || ''
    const title = w.factory ? (w.madeFor || '').split(' · ')[0] || 'A First Mate factory' : ''
    const cur = byKey.get(key) ?? { title, names: [] }
    cur.names.push(w.name)
    byKey.set(key, cur)
  }
  return [...byKey.entries()].map(([key, { title, names }]) => {
    const n = `${names.length} worker${names.length === 1 ? '' : 's'}`
    return {
      label: key ? `${title} · Working · ${n}` : `Working · ${n}`,
      names,
      onOpen: key ? onFactory : undefined,
    }
  })
}
