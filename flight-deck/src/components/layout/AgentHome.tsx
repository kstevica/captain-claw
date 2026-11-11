import { useMemo, type ReactNode } from 'react'
import { ArrowRight, Compass, Moon } from 'lucide-react'
import { AgentPicture } from '../agents/AgentPicture'
import { useAgentAvatarStore } from '../../stores/agentAvatarStore'
import { useAuthStore } from '../../stores/authStore'
import { useGroupStore } from '../../stores/groupStore'
import { groupTone, isFolder } from '../../utils/agentGroups'

// ── Centre: no chat open — the agents that are up ──
//
// A card for each agent that is on or asleep: its picture, name, what it does
// and how it is right now (ready, working, a new reply, asleep, starting). One
// click opens its chat — an asleep one wakes. Agents that are off stay in the
// list on the left; with none up, the caller's empty card shows instead.

/** What a card needs of an agent row (SimpleLayout's agents fit it). */
export interface HomeAgent {
  id: string
  kind: 'docker' | 'process' | 'local' | 'shared'
  name: string
  description: string
  state: 'running' | 'starting' | 'stopped' | 'unknown'
  reachable: boolean
  asleep?: boolean
  ownerName?: string
}

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

function greeting(): string {
  const h = new Date().getHours()
  if (h < 5) return 'Working late'
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}

function statusOf(a: HomeAgent, info: { busy: boolean; unread: boolean } | undefined, starting: boolean): { label: string; tone: Tone; pulse?: boolean; title: string } {
  if (starting || a.state === 'starting') return { label: 'Starting', tone: 'amber', pulse: true, title: 'Starting up — it opens in a moment' }
  if (info?.busy) return { label: 'Working', tone: 'amber', pulse: true, title: 'Working on something for you' }
  if (info?.unread) return { label: 'New reply', tone: 'sky', title: 'Finished something while you were elsewhere' }
  if (a.asleep) return { label: 'Asleep', tone: 'indigo', title: 'Asleep — it wakes when you open it' }
  if (!a.reachable) return { label: 'No web port', tone: 'amber', title: 'Running, but it has no web port to chat on' }
  return { label: 'Ready', tone: 'emerald', title: 'Running — ready to chat' }
}

export function AgentHome<A extends HomeAgent>({ agents, sessionInfo, booting, switching, onOpen, locked = false, onGuide, empty }: {
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
}) {
  const user = useAuthStore((s) => s.user)
  const groups = useGroupStore((s) => s.groups)
  const avatars = useAgentAvatarStore((s) => s.avatars)

  // Awake ones first, then asleep; each by name.
  const shown = useMemo(
    () => agents.filter(isUp).sort((x, y) => Number(!!x.asleep) - Number(!!y.asleep) || x.name.localeCompare(y.name)),
    [agents],
  )
  const groupOf = useMemo(() => {
    const m = new Map<string, { name: string; color: string }>()
    for (const g of groups) if (isFolder(g)) for (const id of g.agentIds) if (!m.has(id)) m.set(id, { name: g.name, color: g.color })
    return m
  }, [groups])

  if (shown.length === 0) return <>{empty}</>

  const firstName = (user?.display_name || '').trim().split(/\s+/)[0]
  const asleep = shown.filter((a) => a.asleep).length
  const working = shown.filter((a) => sessionInfo.get(a.id)?.busy).length
  const awake = shown.length - asleep
  const off = agents.length - shown.length
  const counts = [
    awake ? `${awake} awake` : '',
    working ? `${working} working` : '',
    asleep ? `${asleep} asleep` : '',
  ].filter(Boolean).join(' · ')

  return (
    <div className="h-full overflow-y-auto bg-zinc-950/80">
      <div className="mx-auto w-full max-w-5xl px-6 pb-10 pt-10 wide:px-10 wide:pt-14">
        <header className="fd-rise mb-7">
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">
            {greeting()}{firstName ? `, ${firstName}` : ''}
          </h1>
          <p className="mt-1 text-sm text-zinc-500">
            {counts}. Pick one to start a conversation.
          </p>
        </header>

        <ul className="grid grid-cols-[repeat(auto-fill,minmax(15rem,1fr))] gap-3" aria-label="Agents that are up">
          {shown.map((a, i) => {
            const info = sessionInfo.get(a.id)
            const starting = booting.has(a.id) && a.state !== 'stopped'
            const switchingNow = switching.has(a.id)
            const openable = !switchingNow && ((a.reachable && !starting) || sessionInfo.has(a.id))
            const st = statusOf(a, info, starting || switchingNow)
            const tone = PILL[st.tone]
            const group = groupOf.get(a.id)
            const gTone = group ? groupTone(group.color) : null
            const avatar = avatars[a.id]
            const fade = a.asleep ? 'opacity-75 saturate-[0.6]' : ''
            const about = a.description || (a.kind === 'shared' ? `Shared by ${a.ownerName || 'another user'}` : '')
            return (
              <li key={a.id} className="fd-rise" style={{ animationDelay: `${Math.min(i, 8) * 35}ms` }}>
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
                    {group && gTone ? (
                      <span className="flex min-w-0 items-center gap-1.5 text-zinc-500">
                        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${gTone.dot}`} />
                        <span className="truncate">{group.name}</span>
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
              </li>
            )
          })}
        </ul>

        <footer className="mt-8 flex flex-col items-center gap-2 text-center text-xs text-zinc-500">
          <p>
            {off > 0 && <>{off === 1 ? 'One more agent is off: switch it on' : `${off} more agents are off: switch them on`} from the list on the left. </>}
            {locked
              ? 'What they save for you lands in Library, Data and Folders, bottom left.'
              : 'An agent’s files and data show up on the right once its chat is open.'}
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
