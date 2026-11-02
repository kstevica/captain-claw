import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ArrowDown, ArrowUp, Check, ChevronRight, Folder, FolderMinus, FolderPlus, MoreHorizontal, Pin, Plus, Search, Trash2, X } from 'lucide-react'
import { useGroupStore, type AgentGroup } from '../../stores/groupStore'
import { GROUP_COLORS, OTHER_SECTION, agentSections, groupTone, isFolder, nextGroupColor, orderedFolders, type AgentSection } from '../../utils/agentGroups'

// ── Agent groups in the agent column ──
//
// Folders for agents: a section with a name and a colour that folds away, the
// agents in no group below them under "Other agents". Off a phone an agent is
// dragged onto a group (onto "New group" to start one) and a group dragged up
// or down; everywhere an agent's "⋯" → Group, or a group's "⋯" with its tick
// list, does the same. With no groups the list is the plain list it always was.
// What a folded group holds still shows on its header: an agent working or with
// a new reply, and the one whose chat is open stays in view. One group can be
// pinned to the top of the list, and any number of agents to the top of their
// group; nothing goes above the pinned group, and "Other agents" has no pins.

export interface GroupableAgent {
  id: string
  name: string
  description: string
  state: string
  asleep?: boolean
}

type SessionInfo = Map<string, { busy: boolean; unread: boolean }>

/** Goes on an agent's row (its <li>): what makes it draggable. */
export interface RowDrag {
  draggable?: boolean
  onDragStart?: (e: DragEvent<HTMLElement>) => void
  onDragEnd?: () => void
  style?: { opacity: number }
}

/** A row in a group: its pin, to the top of that group. */
export interface RowPin {
  pinned: boolean
  groupName: string
  onToggle: () => void
}

/** What an agent's row gets from its section. */
export interface RowParts {
  drag: RowDrag
  /** None in "Other agents". */
  pin?: RowPin
}

type Dragging = { kind: 'agent'; id: string; from: string } | { kind: 'group'; id: string }

const agentInitials = (name: string) => {
  const words = name.trim().split(/[\s_-]+/).filter(Boolean)
  if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase()
  return name.trim().slice(0, 2).toUpperCase() || '?'
}

const headCount = (n: number) => `${n} agent${n === 1 ? '' : 's'}`

function folded(section: AgentSection<GroupableAgent>, sessionInfo: SessionInfo) {
  const infos = section.agents.map((a) => sessionInfo.get(a.id))
  return { busy: infos.some((x) => x?.busy), unread: infos.some((x) => x?.unread) }
}

export function AgentSections<A extends GroupableAgent>({ agents, query, activeAgentId, sessionInfo, renderRow, onEdit, onNew, phone = false }: {
  agents: A[]
  /** The search box: an agent's name or description, or its group's name. */
  query: string
  activeAgentId: string | null
  sessionInfo: SessionInfo
  renderRow: (a: A, parts: RowParts) => ReactNode
  onEdit: (groupId: string) => void
  /** Start a group, with these agents in it. */
  onNew: (agentIds?: string[]) => void
  phone?: boolean
}) {
  const groups = useGroupStore((s) => s.groups)
  const collapsed = useGroupStore((s) => s.collapsed)
  const toggleCollapsed = useGroupStore((s) => s.toggleCollapsed)
  const placeAgent = useGroupStore((s) => s.placeAgent)
  const moveGroup = useGroupStore((s) => s.moveGroup)
  const pinGroup = useGroupStore((s) => s.pinGroup)
  const toggleAgentPin = useGroupStore((s) => s.toggleAgentPin)

  const [dragging, setDragging] = useState<Dragging | null>(null)
  const [over, setOver] = useState<{ id: string; above: boolean } | null>(null)
  const [overNew, setOverNew] = useState(false)
  // The section that just took an agent or the pin, to say so.
  const [landed, setLanded] = useState<string | null>(null)
  const landedTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(landedTimer.current), [])
  const showLanded = (id: string) => {
    setLanded(id)
    clearTimeout(landedTimer.current)
    landedTimer.current = setTimeout(() => setLanded(null), 900)
  }

  const endDrag = useCallback(() => {
    setDragging(null)
    setOver(null)
    setOverNew(false)
  }, [])
  // The row a drop moves is gone before its dragend would fire.
  useEffect(() => {
    if (!dragging) return
    window.addEventListener('dragend', endDrag)
    return () => window.removeEventListener('dragend', endDrag)
  }, [dragging, endDrag])

  const q = query.trim().toLowerCase()
  const searching = q.length > 0
  const allSections = useMemo(() => agentSections(agents, groups), [agents, groups])
  const sections = useMemo(() => {
    if (!searching) return allSections
    const hit = (a: A) => a.name.toLowerCase().includes(q) || a.description.toLowerCase().includes(q)
    return allSections
      .map((s) => (s.group && s.name.toLowerCase().includes(q) ? s : { ...s, agents: s.agents.filter(hit) }))
      .filter((s) => s.agents.length > 0)
  }, [allSections, q, searching])

  const grouped = groups.some(isFolder)
  const pinnedName = groups.find((g) => g.pinned && isFolder(g))?.name
  const agentDrag = dragging?.kind === 'agent'
  const canDrag = !phone

  // Set after the event: Chrome ends a drag whose start changes the page under it.
  const startDrag = (e: DragEvent<HTMLElement>, what: Dragging, type: string) => {
    e.dataTransfer.setData(type, what.id)
    e.dataTransfer.effectAllowed = 'move'
    setTimeout(() => setDragging(what), 0)
  }
  const rowDrag = (a: A, from: string): RowDrag => (canDrag ? {
    draggable: true,
    onDragStart: (e) => startDrag(e, { kind: 'agent', id: a.id, from }, 'application/x-fd-agent'),
    onDragEnd: endDrag,
    ...(dragging?.kind === 'agent' && dragging.id === a.id ? { style: { opacity: 0.4 } } : {}),
  } : {})
  const rowParts = (a: A, s: AgentSection<A>): RowParts => {
    const { group } = s
    return {
      drag: rowDrag(a, s.id),
      pin: group ? {
        pinned: !!group.pinnedAgentIds?.includes(a.id),
        groupName: group.name,
        onToggle: () => toggleAgentPin(group.id, a.id),
      } : undefined,
    }
  }

  const togglePinned = (g: AgentGroup) => {
    pinGroup(g.pinned ? null : g.id)
    if (!g.pinned) showLanded(g.id)
  }

  // Nothing goes above the pinned group: a drop on it lands below it.
  const dropsAbove = (e: DragEvent<HTMLElement>, s: AgentSection<A>) => {
    if (s.group?.pinned) return false
    const r = e.currentTarget.getBoundingClientRect()
    return e.clientY < r.top + r.height / 2
  }

  const dropOn = (s: AgentSection<A>, above: boolean) => {
    if (dragging?.kind === 'agent') {
      if (dragging.from !== s.id) {
        placeAgent(dragging.id, s.group?.id ?? null)
        showLanded(s.id)
      }
    } else if (dragging?.kind === 'group' && dragging.id !== s.id) {
      if (!s.group) moveGroup(dragging.id, null)
      else if (above) moveGroup(dragging.id, s.id)
      else {
        const others = orderedFolders(groups).filter((g) => g.id !== dragging.id)
        moveGroup(dragging.id, others[others.findIndex((g) => g.id === s.id) + 1]?.id ?? null)
      }
    }
    endDrag()
  }

  const sectionDrop = (s: AgentSection<A>) => (canDrag ? {
    onDragOver: (e: DragEvent<HTMLElement>) => {
      if (!dragging || (dragging.kind === 'group' && dragging.id === s.id)) return
      e.preventDefault()
      e.dataTransfer.dropEffect = 'move'
      const above = dropsAbove(e, s)
      setOver((o) => (o?.id === s.id && o.above === above ? o : { id: s.id, above }))
    },
    onDragLeave: (e: DragEvent<HTMLElement>) => {
      if (e.currentTarget.contains(e.relatedTarget as Node | null)) return
      setOver((o) => (o?.id === s.id ? null : o))
    },
    onDrop: (e: DragEvent<HTMLElement>) => {
      e.preventDefault()
      dropOn(s, dropsAbove(e, s))
    },
  } : {})

  const listClass = phone ? 'flex flex-col gap-2' : 'flex flex-col gap-0.5'

  const newGroup = !searching && (grouped || agents.length > 1) && (
    <button
      onClick={() => onNew()}
      {...(canDrag && agentDrag ? {
        onDragOver: (e: DragEvent<HTMLElement>) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; setOverNew(true) },
        onDragLeave: () => setOverNew(false),
        onDrop: (e: DragEvent<HTMLElement>) => {
          e.preventDefault()
          const id = dragging?.id
          endDrag()
          if (id) onNew([id])
        },
      } : {})}
      className={`flex w-full items-center gap-2 rounded-lg border border-dashed text-left transition-colors ${phone ? 'mt-3 h-11 px-3 text-sm' : 'mt-1.5 px-2 py-1.5 text-xs'} ${
        agentDrag
          ? overNew ? 'border-violet-400 bg-violet-500/10 text-zinc-100' : 'border-zinc-600 text-zinc-300'
          : 'border-transparent text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-300'
      }`}
    >
      <FolderPlus className={phone ? 'h-4 w-4 shrink-0' : 'h-3.5 w-3.5 shrink-0'} />
      {agentDrag ? 'Drop here for a new group' : 'New group'}
    </button>
  )

  if (searching && sections.length === 0) {
    return <p className={phone ? 'py-8 text-center text-sm text-zinc-500' : 'px-2.5 py-6 text-center text-xs text-zinc-600'}>No agents match your search.</p>
  }

  // No groups yet: the plain list.
  if (!grouped) {
    const only = sections[0]?.agents ?? []
    return (
      <>
        <ul className={listClass}>{only.map((a) => renderRow(a, { drag: rowDrag(a, OTHER_SECTION) }))}</ul>
        {newGroup}
      </>
    )
  }

  return (
    <>
      <div className={`flex flex-col ${phone ? 'gap-3' : 'gap-1'}`}>
        {sections.map((s) => {
          if (!s.group && s.agents.length === 0 && !agentDrag) return null
          const isCollapsed = !searching && collapsed.includes(s.id)
          const tone = s.group ? groupTone(s.group.color) : null
          const isOver = over?.id === s.id
          // Only somewhere else: an agent already here isn't going anywhere.
          const taking = isOver && dragging?.kind === 'agent' && dragging.from !== s.id
          const signals = isCollapsed ? folded(s, sessionInfo) : null
          const stillShown = isCollapsed ? s.agents.filter((a) => a.id === activeAgentId) : []
          const running = s.agents.filter((a) => a.state === 'running' || a.asleep).length
          const rail = `${phone ? 'ml-3 pl-2' : 'ml-[11px]'} border-l ${tone ? tone.rail : 'border-transparent'}`
          return (
            <section
              key={s.id}
              data-section={s.name}
              aria-label={s.name}
              {...sectionDrop(s)}
              className={`relative rounded-lg transition-colors ${taking ? `ring-1 ring-inset ${tone ? tone.over : 'bg-zinc-500/10 ring-zinc-500/40'}` : ''}`}
            >
              {isOver && dragging?.kind === 'group' && (
                <span className={`pointer-events-none absolute inset-x-1 h-0.5 rounded-full bg-violet-500 ${over?.above && s.group ? '-top-[3px]' : '-bottom-[3px]'}`} />
              )}
              <div
                className={`group/head flex items-center gap-0.5 rounded-md ${landed === s.id ? 'fd-group-landed' : ''}`}
                {...(canDrag && s.group && !s.group.pinned && !searching ? {
                  draggable: true,
                  onDragStart: (e: DragEvent<HTMLElement>) => startDrag(e, { kind: 'group', id: s.id }, 'application/x-fd-group'),
                  onDragEnd: endDrag,
                } : {})}
                style={dragging?.kind === 'group' && dragging.id === s.id ? { opacity: 0.4 } : undefined}
              >
                <button
                  onClick={() => toggleCollapsed(s.id)}
                  aria-expanded={!isCollapsed}
                  title={`${s.name} — ${headCount(s.agents.length)}, ${running} on`}
                  className={`flex min-w-0 flex-1 items-center gap-1.5 rounded-md text-left transition-colors hover:bg-zinc-800/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${phone ? 'h-11 px-2' : 'h-7 px-1.5'}`}
                >
                  <ChevronRight className={`shrink-0 text-zinc-500 transition-transform duration-200 motion-reduce:transition-none ${isCollapsed ? '' : 'rotate-90'} ${phone ? 'h-4 w-4' : 'h-3 w-3'}`} />
                  {tone && <span className={`shrink-0 rounded-[3px] ${tone.dot} ${phone ? 'h-2.5 w-2.5' : 'h-2 w-2'}`} />}
                  <span className={`min-w-0 truncate font-semibold ${phone ? 'text-sm' : 'text-xs'} ${s.group ? 'text-zinc-300' : 'text-zinc-500'}`}>{s.name}</span>
                  <span className={`shrink-0 tabular-nums text-zinc-500 ${phone ? 'text-xs' : 'text-[10px]'}`}>{s.agents.length}</span>
                  {/* Off a phone its pin button says so. */}
                  {phone && s.group?.pinned && <Pin className="h-3 w-3 shrink-0 fill-current text-violet-600 dark:text-violet-400" aria-label="Pinned to the top" />}
                  {signals?.busy && <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-amber-500 dark:bg-amber-400" title="An agent here is working" />}
                  {signals?.unread && !signals.busy && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400" title="A new reply here" />}
                </button>
                {s.group && !phone && (
                  <button
                    onClick={() => togglePinned(s.group!)}
                    aria-pressed={!!s.group.pinned}
                    aria-label={`${s.group.pinned ? 'Unpin' : 'Pin'} group ${s.name}`}
                    title={s.group.pinned ? `Unpin ${s.name}` : pinnedName ? `Pin to the top, in place of ${pinnedName}` : 'Pin to the top'}
                    className={`shrink-0 rounded p-1.5 transition-colors hover:bg-zinc-800 ${s.group.pinned
                      ? 'text-violet-600 dark:text-violet-400'
                      : 'text-zinc-500 opacity-0 hover:text-zinc-200 focus:opacity-100 group-hover/head:opacity-100 pointer-coarse:opacity-100'}`}
                  >
                    <Pin className={`h-3.5 w-3.5 ${s.group.pinned ? 'fill-current' : ''}`} />
                  </button>
                )}
                {s.group && (
                  <button
                    onClick={() => onEdit(s.id)}
                    aria-label={`Edit group ${s.name}`}
                    title={`Edit ${s.name}`}
                    className={phone
                      ? 'flex h-11 w-11 shrink-0 items-center justify-center rounded-lg text-zinc-500 active:bg-zinc-800'
                      : 'shrink-0 rounded p-1.5 text-zinc-500 opacity-0 transition-colors hover:bg-zinc-800 hover:text-zinc-200 focus:opacity-100 group-hover/head:opacity-100 pointer-coarse:opacity-100'}
                  >
                    <MoreHorizontal className={phone ? 'h-5 w-5' : 'h-3.5 w-3.5'} />
                  </button>
                )}
              </div>

              {/* Rows of 0fr → 1fr: folds to nothing without knowing its height. */}
              <div className={`grid transition-[grid-template-rows] duration-200 ease-out motion-reduce:transition-none ${isCollapsed ? 'grid-rows-[0fr]' : 'grid-rows-[1fr]'}`}>
                <div className="min-h-0 overflow-hidden" inert={isCollapsed}>
                  <div className={`${rail} ${phone ? 'pb-1 pt-1' : 'pb-0.5 pt-0.5'}`}>
                    <ul className={listClass}>{s.agents.map((a) => renderRow(a, rowParts(a, s)))}</ul>
                    {s.group && s.agents.length === 0 && (
                      <button
                        onClick={() => onEdit(s.id)}
                        className={`flex w-full items-center gap-1.5 rounded-md text-left text-zinc-500 transition-colors hover:bg-zinc-800/50 hover:text-zinc-300 ${phone ? 'h-11 px-2 text-sm' : 'px-2 py-1.5 text-[11px]'}`}
                      >
                        <Plus className="h-3 w-3 shrink-0" />
                        {phone ? 'Add agents' : 'Drag agents here, or add them'}
                      </button>
                    )}
                    {!s.group && s.agents.length === 0 && (
                      <p className={`text-zinc-500 ${phone ? 'px-2 py-2 text-sm' : 'px-2 py-1.5 text-[11px]'}`}>Drop here to take it out of its group</p>
                    )}
                  </div>
                </div>
              </div>
              {stillShown.length > 0 && (
                <div className={rail}><ul className={listClass}>{stillShown.map((a) => renderRow(a, rowParts(a, s)))}</ul></div>
              )}
            </section>
          )
        })}
      </div>
      {newGroup}
    </>
  )
}

// ── The collapsed rail: a mark before each group's agents ──

export function RailGroupMark({ section, sessionInfo }: { section: AgentSection<GroupableAgent>; sessionInfo: SessionInfo }) {
  const isCollapsed = useGroupStore((s) => s.collapsed.includes(section.id))
  const toggleCollapsed = useGroupStore((s) => s.toggleCollapsed)
  const tone = section.group ? groupTone(section.group.color) : null
  const signals = isCollapsed ? folded(section, sessionInfo) : null
  return (
    <button
      onClick={() => toggleCollapsed(section.id)}
      aria-expanded={!isCollapsed}
      aria-label={section.name}
      title={`${section.name} — ${headCount(section.agents.length)}. Click to ${isCollapsed ? 'show' : 'hide'} them`}
      className="group/mark relative flex w-9 shrink-0 items-center justify-center rounded py-1 transition-colors hover:bg-zinc-800"
    >
      {isCollapsed ? (
        <span className={`flex h-5 min-w-6 items-center justify-center rounded-md px-1 text-[10px] font-semibold tabular-nums ${tone ? `${tone.bg} ${tone.text}` : 'bg-zinc-800 text-zinc-400'}`}>
          {section.agents.length}
        </span>
      ) : (
        <span className={`h-1 w-6 rounded-full ${tone ? tone.dot : 'bg-zinc-700'}`} />
      )}
      {signals?.busy && <span className="absolute right-0 top-0 h-2 w-2 animate-pulse rounded-full bg-amber-500 ring-2 ring-zinc-900 dark:bg-amber-400" />}
      {signals?.unread && !signals.busy && <span className="absolute right-0 top-0 h-2 w-2 rounded-full bg-sky-400 ring-2 ring-zinc-900" />}
    </button>
  )
}

// ── A sheet of ours ──
//
// Looks like the kiosk's menus (MenuSheet): a card in the middle of a larger
// screen, a sheet from the bottom on a phone. Esc closes it and the focus goes
// back where it was.

function GroupSheet({ label, picture, title, subtitle, phone, onClose, children, footer }: {
  label: string
  /** Inside the coloured square at the top. */
  picture: ReactNode
  title: ReactNode
  subtitle?: string
  phone: boolean
  onClose: () => void
  children: ReactNode
  footer?: ReactNode
}) {
  const [opener] = useState(() => document.activeElement as HTMLElement | null)
  const close = useCallback(() => {
    onClose()
    opener?.focus?.()
  }, [onClose, opener])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.stopPropagation()
      close()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [close])

  return createPortal(
    <div
      className={`fd-fade fixed inset-0 z-50 flex justify-center bg-black/50 backdrop-blur-[2px] ${phone ? 'items-end' : 'items-end wide:items-center wide:p-4'}`}
      onClick={close}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={label}
        onClick={(e) => e.stopPropagation()}
        className={`${phone ? 'fd-sheet rounded-t-3xl' : 'fd-sheet fd-card-wide rounded-t-3xl wide:max-w-lg wide:rounded-3xl wide:pb-0'} relative flex max-h-[88dvh] w-full flex-col overflow-hidden border border-zinc-800 bg-zinc-900 pb-[env(safe-area-inset-bottom)] shadow-2xl shadow-black/40 wide:max-h-[92dvh]`}
      >
        <div className={`mx-auto mt-2.5 h-1 w-10 shrink-0 rounded-full bg-zinc-700 ${phone ? '' : 'wide:hidden'}`} aria-hidden />
        <div className="flex shrink-0 items-center gap-3.5 px-5 pb-4 pt-4 wide:pt-5">
          {picture}
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-base font-semibold text-zinc-100">{title}</h2>
            {subtitle && <p className="truncate text-xs text-zinc-500">{subtitle}</p>}
          </div>
          <button onClick={close} aria-label="Close" title="Close" className="-mr-1.5 rounded-full p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-5 w-5" />
          </button>
        </div>
        {children}
        {footer}
      </div>
    </div>,
    document.body,
  )
}

function GroupPicture({ color }: { color: string }) {
  return (
    <span className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl text-white shadow-lg shadow-black/20 transition-colors ${groupTone(color).dot}`}>
      <Folder className="h-6 w-6" />
    </span>
  )
}

// ── New group / edit a group ──
//
// One sheet for all of it: the name, the colour, and a tick for each agent. An
// agent in another group moves here when it's ticked, and says so.

export function AgentGroupDialog<A extends GroupableAgent>({ groupId, agentIds, agents, phone, onClose }: {
  /** The group to edit. None: a new one. */
  groupId?: string
  /** A new group's first agents. */
  agentIds?: string[]
  agents: A[]
  phone: boolean
  onClose: () => void
}) {
  const groups = useGroupStore((s) => s.groups)
  const createGroup = useGroupStore((s) => s.createGroup)
  const updateGroup = useGroupStore((s) => s.updateGroup)
  const deleteGroup = useGroupStore((s) => s.deleteGroup)
  const moveGroup = useGroupStore((s) => s.moveGroup)
  const pinGroup = useGroupStore((s) => s.pinGroup)
  const setGroupMembers = useGroupStore((s) => s.setGroupMembers)

  const folders = useMemo(() => orderedFolders(groups), [groups])
  const group = groupId ? folders.find((g) => g.id === groupId) : undefined
  const [name, setName] = useState(group?.name ?? '')
  const [color, setColor] = useState(() => group?.color ?? nextGroupColor(groups))
  const [picked, setPicked] = useState(() => new Set(group?.agentIds ?? agentIds ?? []))
  // Those in it when the sheet opened come first, and stay put as they're ticked.
  const [first] = useState(() => new Set(group?.agentIds ?? agentIds ?? []))
  const [filter, setFilter] = useState('')
  const [deleting, setDeleting] = useState(false)
  // A new group's pin, until it's created; a group's own goes on at once, like its place.
  const [newPinned, setNewPinned] = useState(false)

  // It went away on another device while this was open.
  useEffect(() => { if (groupId && !group) onClose() }, [groupId, group, onClose])

  const clean = name.trim()
  const clash = folders.some((g) => g.id !== groupId && g.name.trim().toLowerCase() === clean.toLowerCase())
  const canSave = clean.length > 0 && !clash

  const listed = useMemo(() => {
    const q = filter.trim().toLowerCase()
    const sorted = [...agents.filter((a) => first.has(a.id)), ...agents.filter((a) => !first.has(a.id))]
    return q ? sorted.filter((a) => a.name.toLowerCase().includes(q) || a.description.toLowerCase().includes(q)) : sorted
  }, [agents, first, filter])
  const count = agents.filter((a) => picked.has(a.id)).length
  const tone = groupTone(color)

  const elsewhere = (id: string) => folders.find((g) => g.id !== groupId && g.agentIds.includes(id))?.name

  const toggle = (id: string) => setPicked((prev) => {
    const next = new Set(prev)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    return next
  })

  const save = () => {
    if (!canSave) return
    const here = new Set(agents.map((a) => a.id))
    const ticked = agents.filter((a) => picked.has(a.id)).map((a) => a.id)
    if (group) {
      updateGroup(group.id, { name: clean, color })
      // An agent not listed right now (an owner took a shared one back for a while) keeps its place.
      setGroupMembers(group.id, [...group.agentIds.filter((id) => !here.has(id)), ...ticked])
    } else {
      const id = createGroup(clean, 'group', color)
      setGroupMembers(id, ticked)
      if (newPinned) pinGroup(id)
    }
    onClose()
  }

  const isPinned = group ? !!group.pinned : newPinned
  const pinnedElsewhere = folders.find((g) => g.pinned && g.id !== groupId)
  const togglePinned = () => {
    if (group) pinGroup(group.pinned ? null : group.id)
    else setNewPinned((v) => !v)
  }

  // Its place among the groups below the pinned one.
  const at = group ? folders.findIndex((g) => g.id === group.id) : -1
  const highest = pinnedElsewhere ? 1 : 0
  const moveUp = () => { if (group && at > highest) moveGroup(group.id, folders[at - 1].id) }
  const moveDown = () => { if (group && at >= 0 && at < folders.length - 1) moveGroup(group.id, folders[at + 2]?.id ?? null) }

  const headLabel = 'mb-2 px-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500'
  const stepBtn = 'rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-zinc-500'

  return (
    <GroupSheet
      label={group ? `Edit group ${group.name}` : 'New group'}
      picture={<GroupPicture color={color} />}
      title={clean || (group ? group.name : 'New group')}
      subtitle={headCount(count)}
      phone={phone}
      onClose={onClose}
      footer={
        <div className="flex shrink-0 items-center gap-2 border-t border-zinc-800 px-4 py-3 wide:px-5">
          {group && (deleting ? (
            <>
              <span className="min-w-0 flex-1 text-xs text-zinc-400">Delete it? Its agents stay, in Other agents.</span>
              <button onClick={() => setDeleting(false)} className="rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200">Keep</button>
              <button onClick={() => { deleteGroup(group.id); onClose() }} className="rounded-lg bg-red-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-red-500">Delete</button>
            </>
          ) : (
            <button
              onClick={() => (group.agentIds.length ? setDeleting(true) : (deleteGroup(group.id), onClose()))}
              className="flex items-center gap-1.5 rounded-lg px-2.5 py-2 text-sm font-medium text-red-600 transition-colors hover:bg-red-500/10 dark:text-red-400"
            >
              <Trash2 className="h-4 w-4" />
              Delete
            </button>
          ))}
          {!deleting && (
            <>
              <span className="flex-1" />
              <button onClick={onClose} className="rounded-lg px-3 py-2 text-sm font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200">Cancel</button>
              <button onClick={save} disabled={!canSave} className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-40 disabled:hover:bg-violet-600">
                {group ? 'Save' : 'Create group'}
              </button>
            </>
          )}
        </div>
      }
    >
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto overscroll-contain px-4 pb-4 wide:px-5">
        <div>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') save() }}
            maxLength={40}
            autoFocus={!phone && !group}
            placeholder="Name it: Research, Clients, Home…"
            aria-label="Group name"
            aria-invalid={clash}
            className="h-11 w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3.5 text-base font-medium text-zinc-100 placeholder:font-normal placeholder:text-zinc-600 focus:border-violet-500/60 focus:outline-none wide:text-sm"
          />
          {clash && <p className="mt-1.5 px-1 text-xs text-amber-600 dark:text-amber-400">You already have a group with this name.</p>}
        </div>

        <div className="flex items-center gap-1" role="radiogroup" aria-label="Colour">
          {GROUP_COLORS.map((c) => (
            <button
              key={c}
              role="radio"
              aria-checked={c === color}
              aria-label={c}
              title={c}
              onClick={() => setColor(c)}
              className={`flex h-9 w-9 items-center justify-center rounded-full transition-transform hover:scale-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/60 motion-reduce:transition-none ${c === color ? 'ring-2 ring-zinc-500' : ''}`}
            >
              <span className={`flex h-6 w-6 items-center justify-center rounded-full ${groupTone(c).dot}`}>
                {c === color && <Check className="h-3.5 w-3.5 text-white" strokeWidth={3} />}
              </span>
            </button>
          ))}
          {group && !isPinned && folders.length > 1 && (
            <span className="ml-auto flex items-center">
              <button onClick={moveUp} disabled={at <= highest} className={stepBtn} aria-label="Higher in the list" title="Higher in the list"><ArrowUp className="h-4 w-4" /></button>
              <button onClick={moveDown} disabled={at >= folders.length - 1} className={stepBtn} aria-label="Lower in the list" title="Lower in the list"><ArrowDown className="h-4 w-4" /></button>
            </span>
          )}
        </div>

        <button
          role="switch"
          aria-checked={isPinned}
          onClick={togglePinned}
          className="flex w-full items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-950/40 px-3 py-2.5 text-left transition-colors hover:border-zinc-700 hover:bg-zinc-800/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40"
        >
          <Pin className={`h-4 w-4 shrink-0 ${isPinned ? 'fill-current text-violet-600 dark:text-violet-400' : 'text-zinc-500'}`} />
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-medium text-zinc-200">Pin to the top</span>
            <span className="block truncate text-[11px] text-zinc-500">
              {isPinned ? 'First in your list, above every other group'
                : pinnedElsewhere ? `Takes the place of ${pinnedElsewhere.name}: one group can be pinned`
                  : 'Keep it first in your list'}
            </span>
          </span>
          <span className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${isPinned ? 'bg-violet-500' : 'bg-zinc-700'}`} aria-hidden>
            <span className={`absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform motion-reduce:transition-none ${isPinned ? 'translate-x-4' : ''}`} />
          </span>
        </button>

        <div>
          <h3 className={headLabel}>Agents in it</h3>
          {agents.length > 6 && (
            <div className="relative mb-2">
              <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
              <input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Search agents…"
                aria-label="Search agents"
                className="h-9 w-full rounded-lg border border-zinc-800 bg-zinc-950/60 pl-8 pr-3 text-base text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none wide:text-xs"
              />
            </div>
          )}
          {agents.length === 0 && <p className="px-1 py-3 text-sm text-zinc-500">No agents yet. They can join this group once you have some.</p>}
          {agents.length > 0 && listed.length === 0 && <p className="px-1 py-3 text-sm text-zinc-500">No agents match your search.</p>}
          <div className="space-y-0.5">
            {listed.map((a) => {
              const on = picked.has(a.id)
              const other = elsewhere(a.id)
              const live = a.state === 'running' || a.asleep
              return (
                <button
                  key={a.id}
                  role="checkbox"
                  aria-checked={on}
                  onClick={() => toggle(a.id)}
                  className={`flex w-full items-center gap-3 rounded-xl px-2.5 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${on ? 'bg-zinc-800/60' : 'hover:bg-zinc-800/40'}`}
                >
                  <span className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-zinc-800 text-[11px] font-semibold text-zinc-300">
                    {agentInitials(a.name)}
                    <span className={`absolute -bottom-0.5 -right-0.5 h-2 w-2 rounded-full ring-2 ring-zinc-900 ${live ? 'bg-emerald-600 dark:bg-emerald-400' : 'bg-zinc-600'}`} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium text-zinc-200">{a.name}</span>
                    <span className={`block truncate text-[11px] ${on && other ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-500'}`}>
                      {other ? (on ? `Moves here from ${other}` : `In ${other}`) : a.description || (live ? 'Running' : 'Off')}
                    </span>
                  </span>
                  <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-md border transition-colors ${on ? `${tone.dot} border-transparent` : 'border-zinc-600'}`}>
                    {on && <Check className="h-3.5 w-3.5 text-white" strokeWidth={3} />}
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </GroupSheet>
  )
}

// ── An agent's "⋯" → Group ──
//
// One tap and it's there: its groups as a list, or a new one named right here.

export function AgentGroupPicker({ agentId, agentName, phone, onClose }: {
  agentId: string
  agentName: string
  phone: boolean
  onClose: () => void
}) {
  const groups = useGroupStore((s) => s.groups)
  const placeAgent = useGroupStore((s) => s.placeAgent)
  const createGroup = useGroupStore((s) => s.createGroup)
  const expand = useGroupStore((s) => s.expand)
  const folders = useMemo(() => orderedFolders(groups), [groups])
  const inGroups = folders.filter((g) => g.agentIds.includes(agentId))
  const [name, setName] = useState('')
  const listRef = useRef<HTMLDivElement>(null)

  const pick = (groupId: string | null) => {
    placeAgent(agentId, groupId)
    // It shows up where it went, not inside a folded section.
    if (groupId) expand(groupId)
    onClose()
  }
  const createAndPick = () => {
    const clean = name.trim()
    if (!clean) return
    const same = folders.find((g) => g.name.trim().toLowerCase() === clean.toLowerCase())
    pick(same ? same.id : createGroup(clean))
  }

  // Off a phone the focus starts on where it is now, or on the name for a first group.
  useEffect(() => {
    if (phone) return
    const rows = listRef.current?.querySelectorAll<HTMLElement>('[data-pick]')
    const here = listRef.current?.querySelector<HTMLElement>('[data-pick][aria-checked="true"]')
    ;(here ?? rows?.[0])?.focus()
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

  const row = 'flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors hover:bg-zinc-800/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40'

  return (
    <GroupSheet
      label={`Group for ${agentName}`}
      picture={<GroupPicture color={inGroups[0]?.color ?? 'violet'} />}
      title={agentName}
      subtitle={inGroups.length ? `In ${inGroups.map((g) => g.name).join(', ')}` : 'Not in a group'}
      phone={phone}
      onClose={onClose}
    >
      <div ref={listRef} onKeyDown={onArrows} className="min-h-0 flex-1 space-y-0.5 overflow-y-auto overscroll-contain px-3 pb-2 wide:px-4" role="radiogroup" aria-label="Group">
        {folders.map((g) => {
          const here = g.agentIds.includes(agentId)
          return (
            <button key={g.id} data-pick role="radio" aria-checked={here} onClick={() => pick(g.id)} className={`${row} ${here ? 'bg-zinc-800/60' : ''}`}>
              <span className={`h-3 w-3 shrink-0 rounded ${groupTone(g.color).dot}`} />
              <span className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-200">{g.name}</span>
              {here && <Check className="h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />}
            </button>
          )
        })}
        {folders.length > 0 && (
          <button data-pick role="radio" aria-checked={inGroups.length === 0} onClick={() => pick(null)} className={`${row} ${inGroups.length === 0 ? 'bg-zinc-800/60' : ''}`}>
            <FolderMinus className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
            <span className="min-w-0 flex-1 truncate text-sm text-zinc-400">No group</span>
            {inGroups.length === 0 && <Check className="h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />}
          </button>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2 border-t border-zinc-800 px-4 py-3 wide:px-5">
        <FolderPlus className="h-4 w-4 shrink-0 text-zinc-500" />
        <input
          autoFocus={!phone && folders.length === 0}
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') createAndPick() }}
          maxLength={40}
          placeholder={folders.length ? 'Or a new group…' : 'Name its group: Research, Clients…'}
          aria-label="New group name"
          className="h-9 min-w-0 flex-1 bg-transparent text-base text-zinc-100 placeholder-zinc-500 focus:outline-none wide:text-sm"
        />
        <button onClick={createAndPick} disabled={!name.trim()} className="rounded-lg bg-violet-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-40 disabled:hover:bg-violet-600">
          Create
        </button>
      </div>
    </GroupSheet>
  )
}
