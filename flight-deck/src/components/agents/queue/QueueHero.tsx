import { forwardRef, useEffect, useLayoutEffect, useRef, useState, type Ref } from 'react'
import {
  AlertTriangle, Check, CheckCircle2, Clock, Gauge, HelpCircle, ListTodo, Loader2,
  MessageSquare, MessageSquareText, MoreHorizontal, Pause, PenLine, Play, Plus, Power, RefreshCw, RotateCcw,
  SkipForward, Square, Trash2, Unplug, Wand2, WifiOff, X, XCircle, type LucideIcon, type LucideProps,
} from 'lucide-react'
import type { AgentConn } from '../../../hooks/useFileAttachments'
import { kindOf } from '../fileKind'
import { touchKeyboard } from '../../../hooks/useMediaQuery'
import { agentFileViewUrl } from '../../../utils/agentFileUrl'
import { fmtTokens } from '../../../utils/format'
import type { QueuedMessage } from '../../../stores/chatStore'
import { Pill } from '../../console/parts'
import {
  firstLine, fmtClock, parseAttachmentRefs, taskTokens, usageTitle,
  type ActionIcon, type AttachedFile, type HeroAction, type HeroIcon, type HeroTone,
} from './queueModel'
import { FOCUS, QBtn } from './QueueButton'
import { QueueMenu, type MenuEntry } from './QueueMenu'
import type { EditState, QueueCtx } from './QueuePanel'

// The hero card: the one place that says what the queue is doing, what
// happens next by itself, and what the user can do — with the running task
// (its text, stats and ⋯ menu) inside it, so it never shows twice.

const Spinner = forwardRef<SVGSVGElement, LucideProps>(function Spinner({ className = '', ...p }, ref) {
  return <Loader2 ref={ref} {...p} className={`${className} motion-safe:animate-spin`} />
})

const HERO_ICON: Record<HeroIcon, LucideIcon> = {
  list: ListTodo, alert: AlertTriangle, gauge: Gauge, x: XCircle, square: Square, 'wifi-off': WifiOff,
  loader: Loader2, help: HelpCircle, reply: MessageSquareText, unplug: Unplug, pause: Pause, power: Power,
  clock: Clock, play: Play, check: CheckCircle2,
}

const ACTION_ICON: Record<ActionIcon, LucideIcon> = {
  plus: Plus, wand: Wand2, refresh: RefreshCw, rotate: RotateCcw, skip: SkipForward, pen: PenLine,
  message: MessageSquare, check: Check, play: Play,
}

const TONE_CARD: Record<HeroTone, string> = {
  violet: 'border-violet-500/30 bg-violet-500/[0.06]',
  emerald: 'border-emerald-500/30 bg-emerald-500/[0.06]',
  amber: 'border-amber-500/40 bg-amber-500/[0.08]',
  red: 'border-red-500/40 bg-red-500/[0.07]',
  zinc: 'border-zinc-800 bg-zinc-900/50',
}

export function QueueHero({ ctx, onAction, primaryRef, wrapRef, sticky, confirmRemove, setConfirmRemove }: {
  ctx: QueueCtx
  onAction: (a: HeroAction) => void
  primaryRef: Ref<HTMLButtonElement>
  /** The wrapper, measured by the panel to decide whether it may stick. */
  wrapRef: Ref<HTMLDivElement>
  /** Pinned to the top of the list — only while it leaves the list room. */
  sticky: boolean
  confirmRemove: boolean
  setConfirmRemove: (v: boolean) => void
}) {
  const { model, compact, phone } = ctx
  const hero = model.hero
  const D = hero.running ? model.D : null
  const editing = D && ctx.editing?.id === D.id ? ctx.editing : null
  const menuOpen = ctx.menuFor === 'hero'
  const menuBtn = useRef<HTMLButtonElement>(null)
  const Icon = hero.spin ? Spinner : HERO_ICON[hero.icon]
  const backToCard = () => window.setTimeout(() => menuBtn.current?.focus(), 0)

  // A stop still landing: the agent is replying, but whatever you pick now
  // waits for the stop and then goes by itself — only a second Stop is moot.
  const replying = ctx.busy && !ctx.stopping
  const menuItems: MenuEntry[] = D ? [
    { label: 'Show in chat', icon: MessageSquare, hidden: !D.messageId, onSelect: () => ctx.onGoToChat({ messageId: D.messageId }) },
    // Disabled rather than hidden: the rows below never move under the pointer.
    {
      label: ctx.stopping ? 'Stopping…' : 'Stop', icon: Square, disabled: !replying,
      title: ctx.stopping ? `${ctx.agent} stops at its next step` : ctx.busy ? "Stop the agent's current reply" : "The agent isn't replying",
      onSelect: ctx.act.stop,
    },
    { label: 'Mark done', icon: Check, onSelect: () => ctx.act.markDone(D.id) },
    {
      label: 'Run it again', icon: RotateCcw, disabled: replying,
      title: replying ? 'Wait for the agent to finish, or stop it first' : ctx.stopping ? 'It runs again once it has stopped' : undefined,
      onSelect: () => ctx.act.retry(D.id),
    },
    {
      label: 'Edit and run again', icon: PenLine, hidden: replying || model.Dcmd,
      onSelect: () => ctx.act.startEdit(D, 'retry', 'Save & run again'),
    },
    // While it is still replying, a skip stops that reply too — else the
    // agent works on for minutes at a task the queue has moved past. The
    // skip goes first, so the next task waits for the stop to land and then
    // starts by itself. (Already stopping: the skip alone.)
    replying
      ? { label: 'Stop & skip', icon: SkipForward, onSelect: () => { ctx.act.skipRunning(D.id); ctx.act.stop() } }
      : { label: 'Skip', icon: SkipForward, onSelect: () => ctx.act.skipRunning(D.id) },
    'divider',
    { label: 'Remove from queue', icon: Trash2, danger: true, onSelect: () => setConfirmRemove(true) },
  ] : []

  // A long label wraps inside its button instead of pushing it out of the card.
  const btnFit = 'h-auto min-w-0 max-w-full text-left leading-tight'
  const btnPrimary = `${btnFit} ${phone ? 'min-h-11 w-full rounded-xl px-3 py-2 text-sm' : 'min-h-8 rounded-xl px-3 py-1.5 text-[13px]'}`
  const btnSecondary = `${btnFit} ${phone ? 'min-h-10 rounded-xl px-2.5 py-2 text-[13px]' : 'min-h-8 rounded-xl px-2.5 py-1.5 text-[13px]'}`
  // A short panel: the task as one line, no stats — the list keeps its room.
  const oneLine = D && compact ? (() => {
    const p = parseAttachmentRefs(D.content)
    return { text: firstLine(p.text, 80) || firstLine(p.files.map((f) => f.name).join(', '), 80), full: p.text }
  })() : null

  return (
    <div ref={wrapRef} className={`-mx-3 px-3 pb-2 pt-3 ${sticky ? 'sticky top-0 z-10 bg-zinc-950/95 backdrop-blur' : ''}`}>
      <div className={`relative rounded-2xl border shadow-sm ${compact ? 'p-3' : 'p-3.5'} ${TONE_CARD[hero.tone]}`}>
        {/* The running task's ⋯ sits outside the crossfade, so a change of
            state never remounts it (nor drops the focus on it). */}
        {D && (
          <button
            ref={menuBtn}
            type="button"
            onClick={(e) => ctx.openMenu('hero', e.currentTarget)}
            aria-label="Task options"
            title="Task options"
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            className={`absolute right-2.5 top-2.5 z-[1] inline-flex h-7 w-7 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 pointer-coarse:h-9 pointer-coarse:w-9 ${FOCUS} ${menuOpen ? 'bg-zinc-800 text-zinc-200' : ''}`}
          >
            <MoreHorizontal className="h-4 w-4" />
          </button>
        )}
        {/* Keyed by the run (or by the row when nothing runs): a new state
            crossfades in, while working → wrapping up → your turn keeps the
            same buttons, and the focus on them. */}
        <div
          key={hero.running && model.D ? `run:${model.D.id}` : hero.id}
          className="transition-opacity duration-150 starting:opacity-0 motion-reduce:transition-none"
        >
          <div className={`flex flex-wrap items-center gap-1.5 ${D ? 'pr-8 pointer-coarse:pr-10' : ''}`}>
            {hero.pill === null ? (
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
                <ListTodo className="h-[18px] w-[18px]" />
              </span>
            ) : (
              <Pill tone={hero.tone} icon={Icon}>{hero.pill}</Pill>
            )}
            {hero.tryPill && <Pill tone="amber">{hero.tryPill}</Pill>}
            {hero.figure && (
              <span
                className={`ml-auto text-[11px] font-medium tabular-nums ${hero.figure.kind === 'countdown'
                  ? 'text-amber-700 dark:text-amber-300' : 'text-violet-700 dark:text-violet-300'}`}
                title={hero.figure.kind === 'countdown' ? 'Until the queue moves on' : 'Running for'}
              >
                {hero.figure.text}
              </span>
            )}
          </div>

          <h3 className="mt-2 text-sm font-semibold leading-snug text-zinc-100">{hero.title}</h3>

          {D && (editing ? (
            <TaskEditor
              value={editing}
              onChange={ctx.setEditing}
              onSave={() => ctx.act.saveEdit(editing)}
              onCancel={() => { ctx.setEditing(null); backToCard() }}
              conn={ctx.conn}
              phone={phone}
            />
          ) : oneLine ? (
            oneLine.text && <p className="mt-1 truncate text-[13px] leading-snug text-zinc-300" title={oneLine.full}>{oneLine.text}</p>
          ) : (
            <TaskBlock ctx={ctx} item={D} lines={3} capped />
          ))}

          {hero.quote && (
            <p className="mt-1.5 line-clamp-2 break-words text-xs italic leading-relaxed text-zinc-300" title={hero.quote}>{hero.quote}</p>
          )}
          {/* Never clamped, even in a short panel: it is the one place that
              says what happens next (the reset time, the countdown). */}
          <p className="mt-1.5 text-xs leading-relaxed text-zinc-400">{hero.body}</p>

          {!editing && (confirmRemove && D ? (
            <div
              className="mt-3 rounded-xl border border-zinc-800 bg-zinc-950/60 p-2.5"
              onKeyDown={(e) => {
                if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); setConfirmRemove(false); backToCard() }
              }}
            >
              <p className="text-xs leading-relaxed text-zinc-300">
                Remove it from the queue? {ctx.agent} may still finish its reply in the chat.
              </p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                <QBtn variant="danger" onClick={() => ctx.act.removeRunning(D.id)} className={btnSecondary} icon={Trash2}>Remove</QBtn>
                <QBtn variant="ghost" autoFocus onClick={() => { setConfirmRemove(false); backToCard() }} className={btnSecondary}>Keep</QBtn>
              </div>
            </div>
          ) : (hero.primary || hero.secondary.length > 0) && (
            <div className={`mt-3 flex gap-1.5 ${phone ? 'flex-wrap items-center' : 'flex-col items-start @3xs:flex-row @3xs:flex-wrap @3xs:items-center'}`}>
              {hero.primary && (
                <QBtn
                  ref={primaryRef}
                  variant="primary"
                  icon={hero.primary.icon ? ACTION_ICON[hero.primary.icon] : undefined}
                  iconClassName={phone ? 'h-4 w-4' : 'h-3.5 w-3.5'}
                  disabled={hero.primary.disabled}
                  title={hero.primary.title}
                  onClick={() => onAction(hero.primary!)}
                  className={btnPrimary}
                >
                  {hero.primary.label}
                </QBtn>
              )}
              {hero.secondary.map((a) => (
                <QBtn
                  key={a.id + a.label}
                  variant="ghost"
                  icon={a.icon ? ACTION_ICON[a.icon] : undefined}
                  disabled={a.disabled}
                  title={a.title}
                  onClick={() => onAction(a)}
                  className={btnSecondary}
                >
                  {a.label}
                </QBtn>
              ))}
            </div>
          ))}

          {D?.dispatchedAt && !compact && (
            <div className="mt-2.5 flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-zinc-800/70 pt-2 text-[11px] tabular-nums text-zinc-500" title={usageTitle(D)}>
              <span title={new Date(D.dispatchedAt).toLocaleString()}>Started {fmtClock(D.dispatchedAt)}</span>
              {!!D.toolCount && <span>· {D.toolCount} tool{D.toolCount === 1 ? '' : 's'}</span>}
              {taskTokens(D) > 0 && <span>· {fmtTokens(taskTokens(D))} tokens</span>}
              {/* The Gave up card already says how often it tried. */}
              {D.stuck && hero.id !== 'gave-up' && <Pill tone="red">Gave up after {D.attempts ?? 0} tries</Pill>}
            </div>
          )}

          {hero.progress === 'bar' && model.N > 0 && (
            <div
              role="progressbar"
              aria-label="Tasks done"
              aria-valuemin={0}
              aria-valuemax={model.N}
              aria-valuenow={model.d}
              className="mt-3 flex h-1 overflow-hidden rounded-full bg-zinc-800"
            >
              <div className="h-full bg-emerald-500 transition-[width] duration-500 motion-reduce:transition-none" style={{ width: `${(model.d / model.N) * 100}%` }} />
              {model.D && !model.Dcmd && (
                <div className={`h-full bg-violet-500 ${ctx.busy ? 'motion-safe:animate-pulse' : ''}`} style={{ width: `${100 / model.N}%` }} />
              )}
            </div>
          )}
          {hero.progress === 'countdown' && (
            <div className="mt-3 h-1 overflow-hidden rounded-full bg-zinc-800" aria-hidden>
              <div
                className="h-full rounded-full bg-amber-500 transition-[width] duration-1000 ease-linear motion-reduce:transition-none"
                style={{ width: `${(hero.remaining ?? 1) * 100}%` }}
              />
            </div>
          )}
        </div>
      </div>
      {menuOpen && ctx.menuAnchor && D && (
        <QueueMenu anchor={ctx.menuAnchor} label="Task options" items={menuItems} onClose={ctx.closeMenu} />
      )}
    </div>
  )
}

/** A task's text (refs stripped, clamped with Show more) and its file chips. */
export function TaskBlock({ ctx, item, lines, capped = false, clickToggle = false, className = 'mt-1.5', textClass = 'text-[13px] leading-snug text-zinc-300' }: {
  ctx: QueueCtx
  item: QueuedMessage
  lines: 1 | 2 | 3
  className?: string
  /** The hero: an opened text scrolls inside the card instead of growing it. */
  capped?: boolean
  clickToggle?: boolean
  textClass?: string
}) {
  const p = parseAttachmentRefs(item.content)
  const open = ctx.expanded.has(item.id)
  // The hero's opened text scrolls inside a quarter of the panel.
  const cap = capped && open ? { maxHeight: Math.max(96, Math.round(ctx.panelHeight * 0.25)) } : undefined
  return (
    <div className={`min-w-0 ${className} ${cap ? 'overflow-y-auto overscroll-contain' : ''}`} style={cap}>
      {p.text && (
        <ClampText
          text={p.text}
          lines={lines}
          open={open}
          onToggle={() => ctx.toggleExpanded(item.id)}
          clickToggle={clickToggle}
          className={textClass}
        />
      )}
      {p.files.length > 0 && <FileChips files={p.files} conn={ctx.conn} className="mt-1.5" />}
    </div>
  )
}

/** Text clamped to a few lines; "Show more" only when it actually overflows. */
export function ClampText({ text, lines, open, onToggle, clickToggle = false, className = '' }: {
  text: string
  lines: 1 | 2 | 3
  open: boolean
  onToggle: () => void
  clickToggle?: boolean
  className?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [over, setOver] = useState(false)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el || open) return
    const measure = () => setOver(el.scrollHeight > el.clientHeight + 1)
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [text, lines, open])
  const clamp = open ? '' : lines === 1 ? 'line-clamp-1' : lines === 2 ? 'line-clamp-2' : 'line-clamp-3'
  const canToggle = over || open
  return (
    <>
      <div
        ref={ref}
        onClick={clickToggle && canToggle ? onToggle : undefined}
        className={`whitespace-pre-wrap break-words ${clamp} ${clickToggle && canToggle ? 'cursor-pointer' : ''} ${className}`}
      >
        {text}
      </div>
      {canToggle && (
        <button
          type="button"
          onClick={onToggle}
          className="mt-0.5 rounded text-[11px] font-medium text-violet-700 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 pointer-coarse:-ml-2 pointer-coarse:px-2 pointer-coarse:py-2 dark:text-violet-300"
        >
          {open ? 'Show less' : 'Show more'}
        </button>
      )}
    </>
  )
}

const PILL = 'inline-flex h-7 max-w-[220px] items-center gap-1.5 rounded-full border border-zinc-700/60 bg-zinc-950/50 pl-0.5 text-xs text-zinc-300'

/** Files as pills — a task's, or a message's you sent: a picture with its
 *  thumbnail once the agent is known, a file with its kind's icon, and the
 *  name a person gave it (never the server's path). With `viewUrl` a picture
 *  opens full size in a new tab. */
export function FileChips({ files, conn, viewUrl, onRemove, className = '' }: {
  files: AttachedFile[]
  conn: AgentConn | null
  /** Where a picture is seen in the browser (a shared agent's files go
   *  another way than `conn`); a picture with one is a link. */
  viewUrl?: (f: AttachedFile) => string | null
  onRemove?: (ref: string) => void
  className?: string
}) {
  return (
    <div className={`flex flex-wrap gap-1.5 ${className}`}>
      {files.map((f) => {
        const src = f.kind !== 'image' ? null
          : viewUrl ? viewUrl(f)
          : conn ? agentFileViewUrl(f.path, conn.host, conn.port, conn.auth) : null
        const body = (
          <>
            {f.kind === 'image' ? <Thumb key={src ?? ''} src={src} /> : <KindDot name={f.name} />}
            <span className="min-w-0 truncate">{f.name}</span>
          </>
        )
        if (viewUrl && src && !onRemove) {
          return (
            <a
              key={f.ref}
              href={src}
              target="_blank"
              rel="noopener noreferrer"
              title={`Open ${f.name}`}
              className={`${PILL} pr-2.5 transition-colors hover:border-zinc-600 hover:bg-zinc-900 hover:text-zinc-100 ${FOCUS}`}
            >
              {body}
            </a>
          )
        }
        return (
          <span key={f.ref} title={f.name} className={`${PILL} ${onRemove ? 'pr-1' : 'pr-2.5'}`}>
            {body}
            {onRemove && (
              <button
                type="button"
                onClick={() => onRemove(f.ref)}
                aria-label={`Remove ${f.name}`}
                className={`shrink-0 rounded-full p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 ${FOCUS}`}
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </span>
        )
      })}
    </div>
  )
}

/** A file's kind in a pill: its icon on its colour (as the composer's cards). */
function KindDot({ name }: { name: string }) {
  const { Icon, tile } = kindOf(name, '')
  return (
    <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full ${tile}`}>
      <Icon className="h-3.5 w-3.5" />
    </span>
  )
}

function Thumb({ src }: { src: string | null }) {
  const [broken, setBroken] = useState(false)
  if (!src || broken) return <KindDot name="picture.png" />
  return (
    <img
      src={src}
      alt=""
      loading="lazy"
      onError={() => setBroken(true)}
      className="h-6 w-6 shrink-0 rounded-full bg-zinc-800 object-cover"
    />
  )
}

/** The inline editor: the text without its file refs, the files as chips
 *  that can be dropped, Save and Cancel. */
export function TaskEditor({ value, onChange, onSave, onCancel, conn, phone = false, className = 'mt-1.5' }: {
  value: EditState
  onChange: (v: EditState) => void
  onSave: () => void
  onCancel: () => void
  conn: AgentConn | null
  phone?: boolean
  className?: string
}) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.focus()
    el.setSelectionRange(el.value.length, el.value.length)
  }, [])
  // Grows with the text, up to about 12 lines.
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const max = 12 * 18 + 18
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight + 2, max)}px`
    el.style.overflowY = el.scrollHeight + 2 > max ? 'auto' : 'hidden'
  }, [value.text])
  const files = parseAttachmentRefs(value.refs.join('\n')).files
  const canSave = !!value.text.trim() || files.length > 0
  const btn = phone ? 'h-10 rounded-xl px-3 text-sm' : 'h-7 rounded-lg px-2.5 text-xs'
  return (
    <div className={className} onClick={(e) => e.stopPropagation()}>
      <textarea
        ref={ref}
        value={value.text}
        rows={2}
        aria-label="Task text"
        onChange={(e) => onChange({ ...value, text: e.target.value })}
        onKeyDown={(e) => {
          if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); onCancel(); return }
          if (e.key !== 'Enter') return
          if (e.metaKey || e.ctrlKey || (!e.shiftKey && !touchKeyboard())) {
            e.preventDefault()
            if (canSave) onSave()
          }
        }}
        className="w-full resize-none rounded-xl border border-violet-500/50 bg-zinc-950 px-2.5 py-2 text-base leading-snug text-zinc-100 ring-2 ring-violet-500/15 focus:outline-none wide:text-[13px]"
      />
      {files.length > 0 && (
        <FileChips
          files={files}
          conn={conn}
          onRemove={(r) => onChange({ ...value, refs: value.refs.filter((x) => x !== r) })}
          className="mt-1.5"
        />
      )}
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <QBtn variant="primary" disabled={!canSave} onClick={onSave} className={btn}>{value.saveLabel}</QBtn>
        <QBtn variant="ghost" onClick={onCancel} className={btn}>Cancel</QBtn>
      </div>
    </div>
  )
}
