import { useRef, useState } from 'react'
import {
  ArrowDown, ArrowUp, ArrowUpToLine, Check, ChevronRight, Clock, CopyPlus, CornerDownRight, GripVertical,
  Merge, MessageSquare, MoreHorizontal, PenLine, Play, RotateCcw, SkipForward, Trash2, X,
} from 'lucide-react'
import { isQueueCommand, useChatStore, type QueuedMessage } from '../../../stores/chatStore'
import { fmtTokens } from '../../../utils/format'
import { Pill } from '../../console/parts'
import { fmtClock, fmtElapsed, firstLine, isFreshStart, parseAttachmentRefs, taskTokens, usageTitle, workEndedAt } from './queueModel'
import { FOCUS, QBtn } from './QueueButton'
import { FileChips, TaskBlock, TaskEditor } from './QueueHero'
import { QueueMenu, type MenuEntry } from './QueueMenu'
import type { QueueCtx } from './QueuePanel'

// The list under the hero: "Up next" (what waits, in run order) and "Done"
// (what finished, newest first, folded away). The running task is never
// here — it lives in the hero.

const ITEM_TYPE = 'application/x-fd-queue-item'
const isItemDrag = (e: React.DragEvent) => Array.from(e.dataTransfer.types || []).includes(ITEM_TYPE)

const HEADER_ACTION = 'ml-auto rounded-md px-1.5 py-0.5 text-[11px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 pointer-coarse:px-2.5 pointer-coarse:py-1.5'

export function UpNextSection({ ctx, confirmClear, setConfirmClear }: {
  ctx: QueueCtx
  confirmClear: boolean
  setConfirmClear: (v: boolean) => void
}) {
  const { model } = ctx
  const items = model.pending
  const tasks = items.filter((q) => !isQueueCommand(q.content))
  const clearBtn = useRef<HTMLButtonElement>(null)
  // Drag to reorder (fine pointers): what's being dragged and where it would land.
  const [dragId, setDragId] = useState<string | null>(null)
  const [over, setOver] = useState<{ id: string; pos: 'above' | 'below' } | null>(null)
  const n = model.k || model.kAll
  const keep = () => { setConfirmClear(false); window.setTimeout(() => clearBtn.current?.focus(), 0) }

  return (
    <section aria-label="Up next">
      <div
        data-fd-upnext-header
        className="flex min-h-7 items-center gap-2 px-1 pb-1.5"
        onKeyDown={(e) => { if (confirmClear && e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); keep() } }}
      >
        {confirmClear ? (
          <>
            <span className="text-xs font-medium text-zinc-200">Remove all {n} waiting tasks?</span>
            <div className="ml-auto flex shrink-0 gap-1">
              <QBtn variant="danger" onClick={ctx.act.clearWaiting} className="h-7 rounded-lg px-2 text-[11px]">Remove all</QBtn>
              <QBtn variant="ghost" autoFocus onClick={keep} className="h-7 rounded-lg px-2 text-[11px]">Keep</QBtn>
            </div>
          </>
        ) : (
          <>
            <h4 className="text-xs font-medium text-zinc-400">Up next</h4>
            <span className="text-[11px] tabular-nums text-zinc-500">{model.k}</span>
            <button
              ref={clearBtn}
              type="button"
              title="Remove all waiting tasks"
              onClick={() => { if (model.kAll >= 3) setConfirmClear(true); else ctx.act.clearWaiting() }}
              className={HEADER_ACTION}
            >
              Clear
            </button>
          </>
        )}
      </div>
      <ol className="space-y-1.5">
        {items.map((item) => {
          if (isQueueCommand(item.content)) return <CommandRow key={item.id} ctx={ctx} item={item} />
          const at = tasks.indexOf(item)
          const nextTask = tasks[at + 1]
          return (
            <UpNextRow
              key={item.id}
              ctx={ctx}
              item={item}
              n={model.numberOf.get(item.id) ?? at + 1}
              isNext={item.id === model.next?.id}
              firstTask={tasks[0]}
              prevTask={tasks[at - 1]}
              nextTask={nextTask}
              twoAfter={tasks[at + 2]}
              dragging={dragId === item.id}
              dropLine={dragId && over?.id === item.id && dragId !== item.id ? over.pos : null}
              onGripStart={(e) => {
                e.dataTransfer.setData(ITEM_TYPE, item.id)
                e.dataTransfer.effectAllowed = 'move'
                const li = (e.currentTarget as HTMLElement).closest('li')
                if (li) e.dataTransfer.setDragImage(li, 16, 16)
                setDragId(item.id)
              }}
              onGripEnd={() => { setDragId(null); setOver(null) }}
              dropProps={{
                onDragOver: (e) => {
                  if (!dragId || !isItemDrag(e)) return
                  e.preventDefault()
                  e.dataTransfer.dropEffect = 'move'
                  const r = e.currentTarget.getBoundingClientRect()
                  const pos = e.clientY < r.top + r.height / 2 ? 'above' : 'below'
                  setOver((o) => (o?.id === item.id && o.pos === pos ? o : { id: item.id, pos }))
                },
                onDragLeave: (e) => {
                  if (e.currentTarget.contains(e.relatedTarget as Node | null)) return
                  setOver((o) => (o?.id === item.id ? null : o))
                },
                onDrop: (e) => {
                  if (!isItemDrag(e)) return
                  e.preventDefault()
                  e.stopPropagation()
                  const id = e.dataTransfer.getData(ITEM_TYPE) || dragId
                  const r = e.currentTarget.getBoundingClientRect()
                  const above = e.clientY < r.top + r.height / 2
                  // Above: before this card. Below: before the task after it.
                  // A task that started mid-drag doesn't move (the store refuses).
                  if (id) ctx.act.move(id, above ? item.id : (nextTask?.id ?? null))
                  setDragId(null)
                  setOver(null)
                },
              }}
            />
          )
        })}
      </ol>
    </section>
  )
}

function UpNextRow({ ctx, item, n, isNext, firstTask, prevTask, nextTask, twoAfter, dragging, dropLine, onGripStart, onGripEnd, dropProps }: {
  ctx: QueueCtx
  item: QueuedMessage
  n: number
  isNext: boolean
  firstTask?: QueuedMessage
  prevTask?: QueuedMessage
  nextTask?: QueuedMessage
  twoAfter?: QueuedMessage
  dragging: boolean
  dropLine: 'above' | 'below' | null
  onGripStart: (e: React.DragEvent) => void
  onGripEnd: () => void
  dropProps: Pick<React.HTMLAttributes<HTMLLIElement>, 'onDragOver' | 'onDragLeave' | 'onDrop'>
}) {
  const menuBtn = useRef<HTMLButtonElement>(null)
  const menuOpen = ctx.menuFor === item.id
  const editing = ctx.editing?.id === item.id ? ctx.editing : null
  const running = !!ctx.model.D
  // A turn is on (the queue's, or a message typed in the chat) that this
  // task can join now, as an extra instruction, instead of waiting its turn.
  const joinable = ctx.busy && !ctx.stopping && ctx.connected
  const startEdit = () => ctx.act.startEdit(item, 'edit', 'Save')
  const moveUp = () => { if (prevTask) { ctx.act.move(item.id, prevTask.id); ctx.focusRow(item.id) } }
  const moveDown = () => { if (nextTask) { ctx.act.move(item.id, twoAfter?.id ?? null); ctx.focusRow(item.id) } }

  const menu: MenuEntry[] = [
    {
      label: 'Insert now', icon: Merge, hidden: !joinable,
      onSelect: () => ctx.act.insertNow(item.id),
    },
    {
      label: 'Run now', icon: Play, hidden: running,
      onSelect: () => {
        if (firstTask && firstTask.id !== item.id) ctx.act.move(item.id, firstTask.id)
        ctx.act.runNext()
      },
    },
    {
      // Before the first waiting TASK, so a fresh start ahead of it still runs first.
      label: 'Do this next', icon: ArrowUpToLine, hidden: !running || !firstTask || firstTask.id === item.id,
      onSelect: () => { if (firstTask) { ctx.act.move(item.id, firstTask.id); ctx.focusRow(item.id) } },
    },
    { label: 'Move up', icon: ArrowUp, hidden: !prevTask, onSelect: moveUp },
    { label: 'Move down', icon: ArrowDown, hidden: !nextTask, onSelect: moveDown },
    { label: 'Edit', icon: PenLine, onSelect: startEdit },
    { label: 'Skip', icon: SkipForward, onSelect: () => ctx.act.skipWaiting(item.id) },
    'divider',
    { label: 'Remove', icon: Trash2, danger: true, onSelect: () => ctx.act.removeWaiting(item.id) },
  ]

  // Shown over the card's corner on hover (fine pointers), out of the flow,
  // so the text keeps the card's width; E and Delete on the card and the ⋯
  // menu do the same from the keyboard, so they're no tab stops.
  const quick = `inline-flex h-6 w-6 items-center justify-center rounded-md text-zinc-500 hover:bg-zinc-800 ${FOCUS}`
  const p = parseAttachmentRefs(item.content)
  const summary = firstLine(p.text, 80) || firstLine(p.files.map((f) => f.name).join(', '), 80)

  return (
    <li
      ref={(el) => ctx.registerRow(item.id, el)}
      tabIndex={0}
      aria-label={`Task ${n}${isNext ? ', runs next' : ''}: ${summary}`}
      {...dropProps}
      onKeyDown={(e) => {
        // Only when the card itself has focus — not its buttons or editor.
        if (e.target !== e.currentTarget || editing) return
        const plain = !e.metaKey && !e.ctrlKey && !e.altKey
        if ((e.key === 'Enter' || e.key === ' ') && plain) { e.preventDefault(); ctx.toggleExpanded(item.id) }
        else if ((e.key === 'e' || e.key === 'E') && plain) { e.preventDefault(); startEdit() }
        else if ((e.key === 'i' || e.key === 'I') && plain && joinable) { e.preventDefault(); ctx.act.insertNow(item.id) }
        else if ((e.key === 'Delete' || e.key === 'Backspace') && plain) { e.preventDefault(); ctx.act.removeWaiting(item.id) }
        else if (e.altKey && e.key === 'ArrowUp') { e.preventDefault(); moveUp() }
        else if (e.altKey && e.key === 'ArrowDown') { e.preventDefault(); moveDown() }
        else if ((e.shiftKey && e.key === 'F10') || e.key === 'ContextMenu') {
          e.preventDefault()
          if (menuBtn.current) ctx.openMenu(item.id, menuBtn.current)
        }
      }}
      className={`group relative flex gap-2.5 rounded-xl border bg-zinc-900/40 px-3 py-2.5 transition-colors hover:border-zinc-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${
        isNext ? 'border-violet-500/25' : 'border-zinc-800'
      } ${dragging ? 'opacity-50' : ''}`}
    >
      {dropLine && (
        <span className={`pointer-events-none absolute left-2 right-2 h-0.5 rounded-full bg-violet-500 ${dropLine === 'above' ? '-top-[4px]' : '-bottom-[4px]'}`} />
      )}
      <span
        className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-medium tabular-nums ${
          isNext ? 'bg-violet-500/15 text-violet-700 dark:text-violet-300' : 'bg-zinc-800 text-zinc-400'
        }`}
      >
        <span className={editing ? '' : 'pointer-fine:group-hover:hidden'}>{n}</span>
        {!editing && (
          <span
            draggable
            onDragStart={onGripStart}
            onDragEnd={onGripEnd}
            title="Drag to reorder"
            aria-hidden
            className="hidden cursor-grab active:cursor-grabbing pointer-fine:group-hover:flex"
          >
            <GripVertical className="h-3.5 w-3.5" />
          </span>
        )}
      </span>

      <div className="min-w-0 flex-1">
        {editing ? (
          <TaskEditor
            value={editing}
            onChange={ctx.setEditing}
            onSave={() => ctx.act.saveEdit(editing)}
            onCancel={() => { ctx.setEditing(null); ctx.focusRow(item.id) }}
            conn={ctx.conn}
            phone={ctx.phone}
            className=""
          />
        ) : (
          <TaskBlock
            ctx={ctx}
            item={item}
            lines={3}
            clickToggle
            className=""
            textClass={`text-zinc-200 ${ctx.phone ? 'text-sm leading-6' : 'text-[13px] leading-snug'}`}
          />
        )}
        {(item.stuck || ctx.edited.has(item.id)) && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {item.stuck && <Pill tone="red">Gave up last time</Pill>}
            {ctx.edited.has(item.id) && <Pill tone="zinc">Edited</Pill>}
          </div>
        )}
        {joinable && !editing && (
          <div className="mt-2">
            <button
              type="button"
              onClick={() => ctx.act.insertNow(item.id)}
              title="Hand this to the running task now, as an extra instruction for its remaining steps (I)"
              className={`inline-flex items-center gap-1 rounded-md border border-zinc-700 px-2 text-[11px] font-medium text-zinc-300 hover:border-violet-500/40 hover:bg-violet-500/10 hover:text-violet-700 dark:hover:text-violet-300 ${FOCUS} ${
                ctx.phone ? 'h-9 px-3 text-xs' : 'h-6 pointer-coarse:h-8'
              }`}
            >
              <Merge className="h-3.5 w-3.5" />
              Insert now
            </button>
          </div>
        )}
        {isNext && ctx.model.caption && (
          <p className="mt-1.5 flex items-center gap-1 text-[11px] text-zinc-500">
            <Clock className="h-3 w-3 shrink-0" />
            {ctx.model.caption}
          </p>
        )}
      </div>

      {!editing && !menuOpen && (
        <div className="absolute right-10 top-2 z-[1] hidden gap-0.5 rounded-lg border border-zinc-800 bg-zinc-900 p-0.5 shadow-sm pointer-fine:group-hover:flex pointer-fine:group-focus-within:flex">
          <button type="button" tabIndex={-1} onClick={startEdit} aria-label={`Edit task ${n}`} title="Edit (E)" className={`${quick} hover:text-zinc-200`}>
            <PenLine className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            tabIndex={-1}
            onClick={() => ctx.act.removeWaiting(item.id)}
            aria-label={`Remove task ${n}`}
            title="Remove (Delete)"
            className={`${quick} hover:text-red-600 dark:hover:text-red-400`}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
      {!editing && (
        <div className="-mr-1 -mt-0.5 flex shrink-0 items-start">
          <button
            ref={menuBtn}
            type="button"
            onClick={(e) => ctx.openMenu(item.id, e.currentTarget)}
            aria-label={`Task ${n} options`}
            title="Task options"
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            className={`inline-flex items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 pointer-coarse:text-zinc-400 ${FOCUS} ${
              ctx.phone ? 'h-10 w-10' : 'h-7 w-7 pointer-coarse:h-9 pointer-coarse:w-9'
            } ${menuOpen ? 'bg-zinc-800 text-zinc-200' : ''}`}
          >
            <MoreHorizontal className="h-4 w-4" />
          </button>
        </div>
      )}
      {menuOpen && ctx.menuAnchor && (
        <QueueMenu anchor={ctx.menuAnchor} label={`Task ${n} options`} items={menu} onClose={ctx.closeMenu} />
      )}
    </li>
  )
}

/** A slash command between tasks (the planner's /new): a quiet divider, not
 *  a card — not numbered, not counted, not draggable. Tasks move past it, it
 *  stays put (QueuePanel's move), so each task keeps the step in front of it. */
function CommandRow({ ctx, item }: { ctx: QueueCtx; item: QueuedMessage }) {
  const fresh = isFreshStart(item.content)
  return (
    <li
      className="group flex items-center gap-2 px-1 py-0.5 text-[11px] text-zinc-500"
      title={fresh ? 'The agent starts the next task with a clean context (/new)' : undefined}
    >
      {fresh ? (
        <>
          <RotateCcw className="h-3 w-3 shrink-0" />
          <span className="shrink-0">Fresh start</span>
          <span className="h-px flex-1 bg-zinc-800" />
        </>
      ) : (
        <>
          <span className="shrink-0">Command</span>
          <span className="min-w-0 flex-1 truncate font-mono text-zinc-400">{item.content.trim()}</span>
        </>
      )}
      <button
        ref={(el) => ctx.registerRow(item.id, el)}
        type="button"
        onClick={() => ctx.act.removeWaiting(item.id, 'Step removed')}
        aria-label="Remove step"
        title="Remove step"
        className={`shrink-0 rounded p-0.5 text-zinc-500 opacity-0 hover:bg-zinc-800 hover:text-zinc-200 focus-visible:opacity-100 group-hover:opacity-100 pointer-coarse:p-2.5 pointer-coarse:opacity-100 ${FOCUS}`}
      >
        <X className="h-3 w-3" />
      </button>
    </li>
  )
}

export function DoneSection({ ctx, open, onToggle, showAll, setShowAll }: {
  ctx: QueueCtx
  open: boolean
  onToggle: () => void
  showAll: boolean
  setShowAll: (v: boolean) => void
}) {
  const { model } = ctx
  const rows = showAll ? model.done : model.done.slice(0, 5)
  return (
    <section aria-label="Done">
      <div className="flex min-h-7 items-center gap-2 px-1 pb-1.5">
        <button
          type="button"
          aria-expanded={open}
          onClick={onToggle}
          className="group flex min-w-0 items-center gap-1.5 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 pointer-coarse:min-h-11 pointer-coarse:py-2.5"
        >
          <ChevronRight className={`h-3.5 w-3.5 shrink-0 text-zinc-500 motion-safe:transition-transform ${open ? 'rotate-90' : ''}`} />
          <span className="text-xs font-medium text-zinc-400 group-hover:text-zinc-200">Done</span>
          {/* The totals are in the summary above; the count is enough here. */}
          <span className="text-[11px] tabular-nums text-zinc-500">{model.d}</span>
        </button>
        <button
          type="button"
          title="Remove finished tasks — waiting and running ones stay"
          // The section goes with them: the focus moves on, not to the page.
          onClick={() => { ctx.act.clearFinished(); ctx.focusRow(null) }}
          className={HEADER_ACTION}
        >
          Clear
        </button>
      </div>
      {open && (
        <>
          <ul className="space-y-0.5">
            {rows.map((q) => <DoneRow key={q.id} ctx={ctx} item={q} />)}
          </ul>
          {model.done.length > 5 && (
            <button
              type="button"
              onClick={() => setShowAll(!showAll)}
              className="mt-1 rounded px-2 text-[11px] font-medium text-violet-700 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 pointer-coarse:py-2 dark:text-violet-300"
            >
              {showAll ? 'Show fewer' : `Show all ${model.done.length}`}
            </button>
          )}
        </>
      )}
    </section>
  )
}

function DoneRow({ ctx, item }: { ctx: QueueCtx; item: QueuedMessage }) {
  const open = ctx.expanded.has(item.id)
  const p = parseAttachmentRefs(item.content)
  const skipped = item.endedHow === 'skipped'
  const inserted = item.endedHow === 'inserted'
  const noAnswer = item.endedHow === 'no-answer'
  const Icon = skipped ? SkipForward : noAnswer ? CornerDownRight : inserted ? Merge : Check
  const iconCls = skipped ? 'text-zinc-500' : noAnswer ? 'text-amber-600 dark:text-amber-400'
    : inserted ? 'text-violet-700 dark:text-violet-300' : 'text-emerald-600 dark:text-emerald-400'
  const ran = !!item.dispatchedAt
  const end = workEndedAt(item)
  const elapsed = ran && end ? fmtElapsed(item.dispatchedAt, end) : ''
  const tokens = taskTokens(item)
  const label = (skipped ? 'Skipped · ' : noAnswer ? 'Moved on, no answer · ' : inserted ? 'Inserted into the running task · ' : '')
    + (firstLine(p.text, 200) || p.files.map((f) => f.name).join(', '))
  const u = item.usage
  const cached = (u?.cache_read_input_tokens || 0) + (u?.cache_creation_input_tokens || 0)
  const stats = [
    ran ? `Started ${fmtClock(item.dispatchedAt)}` : '',
    item.completedAt ? `${skipped && !ran ? 'Skipped' : inserted ? 'Inserted' : 'Finished'} ${fmtClock(item.completedAt)}` : '',
    elapsed ? `took ${elapsed}` : '',
    item.toolCount ? `${item.toolCount} tool${item.toolCount === 1 ? '' : 's'}` : '',
    u?.prompt_tokens ? `${fmtTokens(u.prompt_tokens)} in` : '',
    cached ? `${fmtTokens(cached)} cached` : '',
    u?.completion_tokens ? `${fmtTokens(u.completion_tokens)} out` : '',
  ].filter(Boolean).join(' · ')

  const showInChat = () => {
    const msgs = useChatStore.getState().sessions.get(ctx.sessionKey)?.messages ?? []
    if (!item.messageId || !msgs.some((m) => m.id === item.messageId)) {
      ctx.toast("That message isn't loaded in the chat any more.")
      return
    }
    ctx.onGoToChat({ messageId: item.messageId })
  }

  const action = `inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${ctx.phone ? 'h-9' : ''}`

  return (
    <li>
      <button
        ref={(el) => ctx.registerRow(item.id, el)}
        type="button"
        aria-expanded={open}
        onClick={() => ctx.toggleExpanded(item.id)}
        onKeyDown={(e) => { if (e.key === 'Delete') { e.preventDefault(); ctx.act.removeDone(item.id) } }}
        className="group flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left hover:bg-zinc-900/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 pointer-coarse:min-h-11 pointer-coarse:py-2.5"
      >
        <Icon className={`h-3.5 w-3.5 shrink-0 ${iconCls}`} />
        <span className="min-w-0 flex-1 truncate text-xs text-zinc-500">{label}</span>
        {elapsed && (
          <span className="shrink-0 text-[11px] tabular-nums text-zinc-500" title={usageTitle(item)}>
            {elapsed}
            {tokens > 0 && <span className="hidden @md:inline"> · {fmtTokens(tokens)}</span>}
          </span>
        )}
      </button>
      {open && (
        <div className="mb-1 mt-1 rounded-xl bg-zinc-900/50 px-3 py-2.5">
          {p.text && <p className="whitespace-pre-wrap break-words text-xs leading-relaxed text-zinc-400">{p.text}</p>}
          {p.files.length > 0 && <FileChips files={p.files} conn={ctx.conn} className="mt-1.5" />}
          {stats && <p className="mt-1.5 text-[11px] tabular-nums text-zinc-500" title={usageTitle(item)}>{stats}</p>}
          <div className="mt-2 flex flex-wrap gap-1">
            <button type="button" onClick={() => ctx.act.enqueue(item.content)} title="Adds a copy to the end of the queue" className={action}>
              <CopyPlus className="h-3.5 w-3.5" />
              Add again
            </button>
            {item.messageId && (
              <button type="button" onClick={showInChat} className={action}>
                <MessageSquare className="h-3.5 w-3.5" />
                Show in chat
              </button>
            )}
            <button type="button" onClick={() => ctx.act.removeDone(item.id)} className={`${action} hover:text-red-600 dark:hover:text-red-400`}>
              <Trash2 className="h-3.5 w-3.5" />
              Remove
            </button>
          </div>
        </div>
      )}
    </li>
  )
}
