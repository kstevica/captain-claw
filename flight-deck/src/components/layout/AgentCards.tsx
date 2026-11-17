import { useMemo, useState, type DragEvent, type ReactNode } from 'react'
import { ArrowRight, Compass, CornerDownRight, Moon, Pin, Store, Workflow } from 'lucide-react'
import { AgentPicture } from '../agents/AgentPicture'
import { useAgentAvatarStore } from '../../stores/agentAvatarStore'
import { useGroupStore } from '../../stores/groupStore'
import { agentSections, groupTone, isFolder } from '../../utils/agentGroups'
import type { WorkerLineItem } from '../../utils/agentFamily'
import { AGENT_DRAG_TYPE } from './KioskAgentGroups'
import type { HomeAgent } from './AgentHome'

// ── Centre: Agents — the agents that are up, as cards ──
//
// The nav's Agents (under Today): a card for each agent that is on or asleep,
// grouped as the list on the left is: its picture, name, what it does and how
// it is right now (ready, working, a new reply, asleep, starting). One click
// opens its chat — an asleep one wakes. Agents that are off stay in the list
// on the left; with none up, the caller's empty card shows instead.
//
// A family (agent_family) is one card across the row: the agent, and beside it
// a tray of the agents under it (theirs too, indented) and its runs' workers
// as a line ("Inbox triage · Working · 2 workers"). Pointing at a family dims
// the rest. Hold a card over another for a moment and it goes under that one;
// a child dropped on the page around the cards is one of the person's own again.

// How long a card is held over another before it would go under it (as the column's rows).
const ADOPT_HOLD_MS = 450

type Tone = 'emerald' | 'amber' | 'sky' | 'indigo'

const PILL: Record<Tone, { pill: string; dot: string }> = {
  emerald: { pill: 'bg-emerald-500/10 text-emerald-700 ring-emerald-500/20 dark:text-emerald-300', dot: 'bg-emerald-500 dark:bg-emerald-400' },
  amber: { pill: 'bg-amber-500/10 text-amber-700 ring-amber-500/25 dark:text-amber-300', dot: 'bg-amber-500 dark:bg-amber-400' },
  sky: { pill: 'bg-sky-500/10 text-sky-700 ring-sky-500/25 dark:text-sky-300', dot: 'bg-sky-500 dark:bg-sky-400' },
  indigo: { pill: 'bg-indigo-500/10 text-indigo-700 ring-indigo-500/20 dark:text-indigo-300', dot: 'bg-indigo-500 dark:bg-indigo-400' },
}

// The square behind initials and icons: one of these, the same for an agent every time.
const HUES = [
  'from-violet-500 to-fuchsia-500',
  'from-sky-500 to-indigo-500',
  'from-emerald-500 to-teal-500',
  'from-amber-500 to-orange-500',
  'from-rose-500 to-pink-500',
  'from-cyan-500 to-blue-500',
]

function hueOf(id: string): string {
  let h = 0
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0
  return HUES[h % HUES.length]
}

const isUp = (a: HomeAgent) => a.state === 'running' || a.state === 'starting' || !!a.asleep

/** The top of ``id``'s family among ``all`` (itself when it sits under nobody; a loop stops the walk). */
function rootOf(all: HomeAgent[], id: string): string {
  const byId = new Map(all.map((a) => [a.id, a]))
  const seen = new Set<string>()
  let cur = byId.get(id)
  while (cur?.parent && byId.has(cur.parent) && !seen.has(cur.parent)) {
    seen.add(cur.id)
    cur = byId.get(cur.parent)
  }
  return cur?.id ?? id
}

function statusOf(a: HomeAgent, info: { busy: boolean; unread: boolean } | undefined, starting: boolean): { label: string; tone: Tone; pulse?: boolean; title: string } {
  if (starting || a.state === 'starting') return { label: 'Starting', tone: 'amber', pulse: true, title: 'Starting up — it opens in a moment' }
  if (info?.busy) return { label: 'Working', tone: 'amber', pulse: true, title: 'Working on something for you' }
  if (info?.unread) return { label: 'New reply', tone: 'sky', title: 'Finished something while you were elsewhere' }
  if (a.asleep) return { label: 'Asleep', tone: 'indigo', title: 'Asleep — it wakes when you open it' }
  if (!a.reachable) return { label: 'No web port', tone: 'amber', title: 'Running, but it has no web port to chat on' }
  return { label: 'Ready', tone: 'emerald', title: 'Running — ready to chat' }
}

export function AgentCards<A extends HomeAgent>({
  agents: every, sessionInfo, booting, switching, onOpen, onGuide, empty, onStore,
  workerLines, adoptRefusal, onAdopt, onDetach,
}: {
  agents: A[]
  sessionInfo: Map<string, { busy: boolean; unread: boolean }>
  booting: Set<string>
  switching: Set<string>
  onOpen: (a: A) => void
  locked?: boolean
  /** Kiosk only: open the Getting started guide. */
  onGuide?: () => void
  /** Shown instead when no agent is up. */
  empty: ReactNode
  /** Open the Agent store over the layout (New agent's window). */
  onStore?: () => void
  /** The family (AgentFamily): an agent's runs' workers, as lines. */
  workerLines?: (a: A) => WorkerLineItem[]
  adoptRefusal?: (childId: string, parentId: string) => string | null
  onAdopt?: (childId: string, parentId: string) => void
  onDetach?: (childId: string) => void
}) {
  const groups = useGroupStore((s) => s.groups)
  const avatars = useAgentAvatarStore((s) => s.avatars)

  // The person's agents: a run's workers show in their starter's line, not as cards.
  const agents = useMemo(() => every.filter((a) => !a.managed), [every])
  // The family: who sits under whom among them (a loop in the data makes each one a top).
  const { tops, kidsOf, childIds } = useMemo(() => {
    const byId = new Map(agents.map((a) => [a.id, a]))
    const up = (a: A) => (a.parent && a.parent !== a.id ? byId.get(a.parent) : undefined)
    const isChild = (a: A) => {
      const seen = new Set([a.id])
      for (let p = up(a); p; p = up(p)) {
        if (seen.has(p.id)) return false
        seen.add(p.id)
      }
      return !!up(a)
    }
    const kids = new Map<string, A[]>()
    const child = new Set<string>()
    for (const a of agents) {
      if (!isChild(a)) continue
      child.add(a.id)
      kids.set(a.parent!, [...(kids.get(a.parent!) ?? []), a])
    }
    return { tops: agents.filter((a) => !child.has(a.id)), kidsOf: kids, childIds: child }
  }, [agents])
  const below = (id: string): { a: A; depth: number }[] => {
    const out: { a: A; depth: number }[] = []
    const seen = new Set([id])
    const walk = (pid: string, depth: number) => {
      for (const k of kidsOf.get(pid) ?? []) {
        if (seen.has(k.id)) continue
        seen.add(k.id)
        out.push({ a: k, depth })
        walk(k.id, depth + 1)
      }
    }
    walk(id, 1)
    return out
  }

  // As the list on the left has them: the pinned group first, the others in
  // their order, those pinned in a group at its top, then the agents in none.
  // A family shows while any of it is up.
  const upIds = useMemo(() => new Set(agents.filter(isUp).map((a) => a.id)), [agents])
  const shownTops = useMemo(() => tops.filter((t) => upIds.has(t.id)
    || [...childIds].some((c) => upIds.has(c) && rootOf(agents, c) === t.id)), [tops, upIds, childIds, agents])
  const grouped = groups.some(isFolder)
  const sections = useMemo(() => agentSections(shownTops, groups).filter((sec) => sec.agents.length > 0), [shownTops, groups])

  // Dragging a card: which, which card it's held over (and since when), and the family pointed at.
  const [dragging, setDragging] = useState<string | null>(null)
  const [hold, setHold] = useState<{ id: string; since: number } | null>(null)
  const [adoptOver, setAdoptOver] = useState<string | null>(null)
  const [focus, setFocus] = useState<string | null>(null)
  const canDrag = !!onAdopt && !!adoptRefusal

  if (upIds.size === 0) return <>{empty}</>

  const shown = agents.filter(isUp)
  const asleep = shown.filter((a) => a.asleep).length
  const working = shown.filter((a) => sessionInfo.get(a.id)?.busy).length
  const awake = shown.length - asleep
  const off = agents.length - shown.length
  const counts = [
    awake ? `${awake} awake` : '',
    working ? `${working} working` : '',
    asleep ? `${asleep} asleep` : '',
  ].filter(Boolean).join(' · ')

  const endDrag = () => { setDragging(null); setHold(null); setAdoptOver(null) }
  const draggedName = dragging ? agents.find((x) => x.id === dragging)?.name ?? '' : ''
  const refusal = (child: string, parent: string) => (adoptRefusal ? adoptRefusal(child, parent) : null)
  // A card that can be dragged, and held over to take the dragged one under it.
  const dragProps = (a: A) => (canDrag ? {
    draggable: true,
    onDragStart: (e: DragEvent<HTMLElement>) => {
      e.stopPropagation()
      e.dataTransfer.setData(AGENT_DRAG_TYPE, a.id)
      e.dataTransfer.effectAllowed = 'move'
      setTimeout(() => setDragging(a.id), 0)
    },
    onDragEnd: endDrag,
    onDragOver: (e: DragEvent<HTMLElement>) => {
      if (!dragging || dragging === a.id) return
      const why = refusal(dragging, a.id)
      if (why === null) return
      e.stopPropagation()
      const now = Date.now()
      if (hold?.id !== a.id) {
        setHold({ id: a.id, since: now })
        return
      }
      if (adoptOver !== a.id && now - hold.since < ADOPT_HOLD_MS) return
      e.preventDefault()
      e.dataTransfer.dropEffect = why ? 'none' : 'move'
      setAdoptOver(a.id)
    },
    onDragLeave: (e: DragEvent<HTMLElement>) => {
      if (e.currentTarget.contains(e.relatedTarget as Node | null)) return
      if (hold?.id === a.id) setHold(null)
      setAdoptOver((o) => (o === a.id ? null : o))
    },
    onDrop: (e: DragEvent<HTMLElement>) => {
      if (!dragging || adoptOver !== a.id) return
      e.preventDefault()
      e.stopPropagation()
      if (refusal(dragging, a.id) === '') onAdopt?.(dragging, a.id)
      endDrag()
    },
  } : {})
  // While a card is held over this one: what letting go would do.
  const adoptNote = (a: A) => {
    if (adoptOver !== a.id || !dragging) return null
    const why = refusal(dragging, a.id) ?? ''
    return (
      <span className={`pointer-events-none absolute inset-x-2 bottom-2 z-[1] rounded-lg px-2 py-1 text-center text-[11px] font-medium shadow ${why
        ? 'bg-red-500/15 text-red-700 ring-1 ring-red-500/40 dark:text-red-300'
        : 'bg-violet-500/20 text-violet-800 ring-1 ring-violet-500/50 dark:text-violet-200'}`}>
        {why || `Put ${draggedName} under ${a.name}`}
      </span>
    )
  }

  // An agent under another, in its family's tray.
  const trayCard = (a: A, depth: number, parentName: string) => {
    const info = sessionInfo.get(a.id)
    const starting = booting.has(a.id) && a.state !== 'stopped'
    const openable = !switching.has(a.id) && ((a.reachable && !starting) || sessionInfo.has(a.id))
    const st = isUp(a) ? statusOf(a, info, starting) : { label: 'Off', tone: 'indigo' as Tone, title: 'Off — switch it on from the list on the left' }
    return (
      <li key={a.id} className="relative" style={depth > 1 ? { marginLeft: (depth - 1) * 12 } : undefined} {...dragProps(a)}>
        <button
          onClick={() => onOpen(a)}
          disabled={!openable}
          title={openable ? `Chat with ${a.name}` : `${a.name} — ${st.title}`}
          className={`flex h-full w-full items-start gap-2.5 rounded-xl border p-2.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/50 ${
            adoptOver === a.id ? 'border-violet-500/60' : info?.busy ? 'border-amber-500/30 bg-amber-500/[0.04]' : 'border-zinc-800 bg-zinc-900/40'
          } ${openable ? 'hover:border-violet-500/40 hover:bg-zinc-900' : 'cursor-default opacity-60'}`}
        >
          <span className={`flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden rounded-xl text-[11px] font-semibold text-white ${avatars[a.id] ? '' : `bg-gradient-to-br ${hueOf(a.id)}`} ${isUp(a) ? '' : 'saturate-0'}`}>
            <AgentPicture agentId={a.id} name={a.name} />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-[13px] font-medium text-zinc-200">{a.name}</span>
            <span className="block truncate text-[11px] text-zinc-500">
              {depth > 1 ? `Under ${parentName}` : a.madeFor ? `For: ${a.madeFor}` : a.description || st.label}
            </span>
            <span className={`mt-1 inline-flex items-center gap-1 rounded-full px-1.5 text-[10px] font-medium ring-1 ring-inset ${PILL[st.tone].pill}`}>
              <span className={`h-1 w-1 rounded-full ${PILL[st.tone].dot} ${'pulse' in st && st.pulse ? 'animate-pulse' : ''}`} />{st.label}
            </span>
          </span>
        </button>
        {adoptNote(a)}
      </li>
    )
  }

  // What sits beside a family's agent: the agents under it, and its runs' workers.
  const tray = (a: A) => {
    const kids = below(a.id)
    const lines = workerLines?.(a) ?? []
    if (!kids.length && !lines.length) return null
    const nameOf = (id: string | undefined) => agents.find((x) => x.id === id)?.name ?? ''
    return (
      <div className="min-w-0 flex-1">
        {kids.length > 0 && (
          <>
            <div className="mb-2 flex items-center gap-1.5 text-[11px] font-medium text-zinc-500">
              <CornerDownRight className="h-3.5 w-3.5" />
              {kids.length === 1 ? 'One agent under it' : `${kids.length} agents under it`}
            </div>
            <ul className="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-2">
              {kids.map(({ a: k, depth }) => trayCard(k, depth, nameOf(k.parent)))}
            </ul>
          </>
        )}
        {lines.length > 0 && (
          <ul className={`${kids.length ? 'mt-2.5' : ''} space-y-1`}>
            {lines.map((line, i) => {
              const body = (
                <>
                  <Workflow className="h-3.5 w-3.5 shrink-0 animate-pulse motion-reduce:animate-none" />
                  <span className="min-w-0 truncate">{line.label}</span>
                </>
              )
              const cls = 'flex w-full items-center gap-2 rounded-lg border border-amber-500/25 bg-amber-500/[0.06] px-2.5 py-1.5 text-left text-xs font-medium text-amber-700 dark:text-amber-300'
              return (
                <li key={`${a.id}-line-${i}`} data-workers-of={a.id} title={line.names.join(', ')}>
                  {line.onOpen ? <button onClick={line.onOpen} className={`${cls} hover:bg-amber-500/10`}>{body}</button> : <span className={cls}>{body}</span>}
                </li>
              )
            })}
          </ul>
        )}
      </div>
    )
  }

  // One agent's card. `pinned`: at the top of its group.
  const card = (a: A, key: string, i: number, pinned: boolean) => {
    const info = sessionInfo.get(a.id)
    const starting = booting.has(a.id) && a.state !== 'stopped'
    const switchingNow = switching.has(a.id)
    const openable = !switchingNow && ((a.reachable && !starting) || sessionInfo.has(a.id))
    const st = statusOf(a, info, starting || switchingNow)
    const tone = PILL[st.tone]
    const avatar = avatars[a.id]
    const fade = a.asleep ? 'opacity-75 saturate-[0.6]' : ''
    const about = a.description || (a.kind === 'shared' ? `Shared by ${a.ownerName || 'another user'}` : '')
    const family = tray(a)
    const dimmed = focus !== null && focus !== a.id
    const button = (
        <button
          onClick={() => onOpen(a)}
          disabled={!openable}
          title={openable ? (a.asleep ? `Chat with ${a.name} — it wakes when you open it` : `Chat with ${a.name}`) : `${a.name} — ${st.title}`}
          className={`group/card relative flex h-full w-full flex-col rounded-2xl border p-4 text-left transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/50 motion-reduce:transition-none ${
            info?.busy ? 'border-amber-500/30 bg-amber-500/[0.04]' : 'border-zinc-800 bg-zinc-900/60'
          } ${openable
            ? 'hover:-translate-y-0.5 hover:border-violet-500/40 hover:bg-zinc-900 hover:shadow-xl hover:shadow-violet-500/[0.07] motion-reduce:hover:translate-y-0'
            : 'cursor-default opacity-70'}`}
        >
          <div className="flex items-start justify-between gap-3">
            {/* A picked icon or picture stands on its own; only initials get the coloured tile. */}
            {avatar && 'emoji' in avatar ? (
              <span aria-hidden className={`flex h-14 w-14 shrink-0 select-none items-center justify-center text-5xl leading-none drop-shadow-sm transition-transform duration-200 group-hover/card:scale-105 motion-reduce:transition-none ${fade}`}>
                {avatar.emoji}
              </span>
            ) : avatar ? (
              <span className={`relative flex h-14 w-14 shrink-0 items-center justify-center overflow-hidden rounded-2xl text-base font-semibold text-zinc-500 ${fade}`}>
                <AgentPicture agentId={a.id} name={a.name} size="lg" />
              </span>
            ) : (
              <span className={`relative flex h-14 w-14 shrink-0 items-center justify-center overflow-hidden rounded-2xl bg-gradient-to-br ${hueOf(a.id)} text-base font-semibold text-white shadow-md shadow-black/10 ${fade}`}>
                <AgentPicture agentId={a.id} name={a.name} size="lg" />
              </span>
            )}
            <span title={st.title} className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-1 text-[11px] font-medium ring-1 ring-inset ${tone.pill}`}>
              {a.asleep && !info?.busy && !info?.unread && !starting
                ? <Moon className="h-3 w-3" />
                : <span className={`h-1.5 w-1.5 rounded-full ${tone.dot} ${st.pulse ? 'animate-pulse' : ''}`} />}
              {st.label}
            </span>
          </div>

          <div className="mt-3.5 min-w-0">
            <div className="truncate text-[15px] font-semibold text-zinc-100">{a.name}</div>
            <p className={`mt-1 line-clamp-2 min-h-[2rem] text-xs leading-4 ${about ? 'text-zinc-500' : 'italic text-zinc-600'}`}>
              {about || 'No description yet'}
            </p>
          </div>

          <div className="mt-3 flex items-center justify-between gap-2 border-t border-zinc-800/80 pt-3 text-[11px]">
            {pinned ? (
              <span className="flex items-center gap-1 font-medium text-violet-600 dark:text-violet-400">
                <Pin className="h-3 w-3 fill-current" />Pinned
              </span>
            ) : <span className="text-zinc-600">{a.kind === 'shared' ? 'Shared with you' : a.kind === 'local' ? 'On this computer' : ''}</span>}
            {openable && (
              <span className="ml-auto flex shrink-0 items-center gap-1 font-medium text-zinc-500 transition-colors group-hover/card:text-violet-600 group-focus-visible/card:text-violet-600 dark:group-hover/card:text-violet-400 dark:group-focus-visible/card:text-violet-400">
                {a.asleep ? 'Wake and chat' : 'Chat'}
                <ArrowRight className="h-3 w-3 transition-transform group-hover/card:translate-x-0.5" />
              </span>
            )}
          </div>
        </button>
    )
    const common = {
      style: { animationDelay: `${Math.min(i, 8) * 35}ms` },
      onMouseEnter: family ? () => setFocus(a.id) : undefined,
      onMouseLeave: family ? () => setFocus((f) => (f === a.id ? null : f)) : undefined,
    }
    if (!family) {
      return (
        <li key={key} className={`fd-rise relative transition-opacity duration-200 ${dimmed ? 'opacity-50' : ''}`} {...common} {...dragProps(a)}>
          {button}
          {adoptNote(a)}
        </li>
      )
    }
    // A family: across the row, its agent beside the tray of the ones under it.
    return (
      <li key={key} data-family={a.id} className={`fd-rise col-span-full transition-opacity duration-200 ${dimmed ? 'opacity-50' : ''}`} {...common}>
        <div className="flex flex-col gap-3 rounded-[1.25rem] border border-zinc-800/80 bg-zinc-900/25 p-2 sm:flex-row">
          <div className="relative sm:w-60 sm:shrink-0" {...dragProps(a)}>
            {button}
            {adoptNote(a)}
          </div>
          <div className="min-w-0 flex-1 px-1 pb-1 sm:py-2 sm:pr-2">{family}</div>
        </div>
      </li>
    )
  }

  // Cards arrive one after another, across the sections.
  let n = 0

  // A child dropped on the page around the cards: one of the person's own again.
  const detaching = !!dragging && childIds.has(dragging) && !!onDetach
  return (
    <div
      className="h-full overflow-y-auto bg-zinc-950/80"
      {...(detaching ? {
        onDragOver: (e: DragEvent<HTMLElement>) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move' },
        onDrop: (e: DragEvent<HTMLElement>) => { e.preventDefault(); onDetach?.(dragging!); endDrag() },
      } : {})}
    >
      <div className="mx-auto w-full max-w-5xl px-6 pb-10 pt-10 wide:px-10 wide:pt-14">
        <header className="fd-rise mb-7 flex flex-wrap items-end justify-between gap-3">
          <div className="min-w-0">
            <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">Agents</h1>
            <p className="mt-1 text-sm text-zinc-500">
              {counts}. Pick one to start a conversation.
            </p>
          </div>
          {/* The store adds ready-made agents: the same window New agent opens. */}
          {onStore && (
            <button onClick={onStore} className="flex shrink-0 items-center gap-1.5 rounded-full border border-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-400 transition-colors hover:border-violet-500/40 hover:text-violet-600 dark:hover:text-violet-400">
              <Store className="h-3.5 w-3.5" />
              Agent store
            </button>
          )}
        </header>

        <div className="space-y-8" aria-label="Agents that are up">
          {sections.map((sec) => {
            const gTone = sec.group ? groupTone(sec.group.color) : null
            const pins = sec.group?.pinnedAgentIds ?? []
            return (
              <section key={sec.id} aria-label={sec.name}>
                {/* Only with groups: one section alone needs no heading. */}
                {grouped && (
                  <h2 className="fd-rise mb-3 flex items-center gap-2 text-sm font-semibold text-zinc-300">
                    <span className={`h-2 w-2 shrink-0 rounded-full ${gTone ? gTone.dot : 'bg-zinc-500'}`} />
                    <span className="truncate">{sec.name}</span>
                    <span className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[11px] font-medium leading-none text-zinc-500">{sec.agents.length + sec.agents.reduce((n, t) => n + below(t.id).length, 0)}</span>
                    {sec.group?.pinned && <Pin className="h-3.5 w-3.5 shrink-0 fill-current text-violet-600 dark:text-violet-400" aria-label="Pinned to the top" />}
                  </h2>
                )}
                <ul className="grid grid-cols-[repeat(auto-fill,minmax(15rem,1fr))] gap-3">
                  {sec.agents.map((a) => card(a, `${sec.id}:${a.id}`, n++, pins.includes(a.id)))}
                </ul>
              </section>
            )
          })}
        </div>

        <footer className="mt-8 flex flex-col items-center gap-2 text-center text-xs text-zinc-500">
          <p>
            {off > 0 && <>{off === 1 ? 'One more agent is off: switch it on' : `${off} more agents are off: switch them on`} from the list on the left. </>}
            What they save for you lands in Library, Data and Folders, bottom left.
          </p>
          {onGuide && (
            <button onClick={onGuide} className="flex items-center gap-1.5 font-medium transition-colors hover:text-violet-600 dark:hover:text-violet-400">
              <Compass className="h-3.5 w-3.5" />
              New here? Getting started
            </button>
          )}
        </footer>
      </div>
    </div>
  )
}
