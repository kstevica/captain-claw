import type { ReactNode } from 'react'
import { ArrowLeft, ChevronDown, X, type LucideIcon } from 'lucide-react'
import { Pill, type Tone } from '../console/parts'
import { FullscreenButton } from '../library/libraryParts'
import type { CouncilStatus } from '../../stores/councilStore'
import { TONE_PILL, TONE_SQUARE, agentTone, typeLook, type CouncilTone } from './councilStyles'

// Council's own look, on top of the console parts: tinted icon squares (as in
// the kiosk's menu tiles), a colour per session type, agent avatars, status
// pills, and the page header. Dark-first zinc classes; coloured text carries
// its light/dark pair.

const SQUARE_SIZE = {
  sm: 'h-8 w-8 rounded-lg [&>svg]:h-4 [&>svg]:w-4',
  md: 'h-10 w-10 rounded-xl [&>svg]:h-5 [&>svg]:w-5',
  lg: 'h-12 w-12 rounded-2xl [&>svg]:h-6 [&>svg]:w-6',
  xl: 'h-14 w-14 rounded-2xl [&>svg]:h-7 [&>svg]:w-7',
}

export function IconSquare({ icon: Icon, tone = 'violet', size = 'md', className = '' }: {
  icon: LucideIcon
  tone?: CouncilTone
  size?: keyof typeof SQUARE_SIZE
  className?: string
}) {
  return (
    <span className={`flex shrink-0 items-center justify-center ring-1 ${SQUARE_SIZE[size]} ${TONE_SQUARE[tone]} ${className}`}>
      <Icon />
    </span>
  )
}

export function TypePill({ type }: { type: string }) {
  const look = typeLook(type)
  const Icon = look.icon
  return (
    <span className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${TONE_PILL[look.tone]}`}>
      <Icon className="h-3 w-3" />
      {type}
    </span>
  )
}

// ── Status ──

const STATUS_TONE: Record<CouncilStatus, Tone> = {
  setup: 'zinc',
  active: 'emerald',
  synthesizing: 'violet',
  concluded: 'zinc',
}

export function StatusPill({ status }: { status: string }) {
  const tone = STATUS_TONE[status as CouncilStatus] || 'zinc'
  const live = status === 'active' || status === 'synthesizing'
  return (
    <Pill tone={tone}>
      {live && (
        <span className="relative mr-0.5 flex h-1.5 w-1.5">
          <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${status === 'active' ? 'bg-emerald-500' : 'bg-violet-500'}`} />
          <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${status === 'active' ? 'bg-emerald-500' : 'bg-violet-500'}`} />
        </span>
      )}
      <span className="capitalize">{status}</span>
    </Pill>
  )
}

/** How far the rounds are: a thin bar. */
export function RoundBar({ current, max, className = '' }: { current: number; max: number; className?: string }) {
  const pct = max > 0 ? Math.min(100, Math.round((current / max) * 100)) : 0
  return (
    <span className={`block h-1 overflow-hidden rounded-full bg-zinc-800 ${className}`} aria-hidden>
      <span className="block h-full rounded-full bg-violet-500/80" style={{ width: `${pct}%` }} />
    </span>
  )
}

function initials(name: string): string {
  const words = (name || '?').replace(/[^\p{L}\p{N}\s-]/gu, ' ').split(/[\s-]+/).filter(Boolean)
  if (words.length === 0) return '?'
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase()
  return (words[0][0] + words[1][0]).toUpperCase()
}

/** An agent's initials on a colour of its own (the same colour everywhere). */
export function AgentAvatar({ name, id, size = 'md', tone, children }: {
  name: string
  id?: string
  size?: 'xs' | 'sm' | 'md'
  /** A colour by role (the person, the moderator) instead of by agent. */
  tone?: CouncilTone
  /** On the avatar's corner: a dot for its connection. */
  children?: ReactNode
}) {
  const cls = size === 'xs' ? 'h-6 w-6 rounded-md text-[10px]' : size === 'sm' ? 'h-8 w-8 rounded-lg text-[11px]' : 'h-9 w-9 rounded-xl text-xs'
  return (
    <span className={`relative flex shrink-0 items-center justify-center font-semibold ring-1 ${cls} ${TONE_SQUARE[tone || agentTone(id || name)]}`}>
      {initials(name)}
      {children}
    </span>
  )
}

// ── Header ──

const ICON_BTN = 'rounded-xl p-2.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200'

/** A Council page's header: a tinted icon, the title and a line on it, and
 *  what can be done on the right. ``compact`` over a running session. */
export function CouncilHeader({ icon, tone = 'amber', title, meta, description, onBack, action, onClose, compact = false }: {
  icon: LucideIcon
  tone?: CouncilTone
  title: ReactNode
  /** Pills beside the title. */
  meta?: ReactNode
  description?: ReactNode
  onBack?: () => void
  /** The page's main action (a primary button): right on a wide page, under the title on a narrow one. */
  action?: ReactNode
  onClose?: () => void
  compact?: boolean
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className={`flex items-start ${compact ? 'gap-3' : 'gap-3 @md:gap-4'}`}>
        {onBack && (
          <button
            onClick={onBack}
            aria-label="Back to sessions"
            title="Back to sessions"
            className={`-ml-1.5 shrink-0 ${ICON_BTN} ${compact ? '-mt-px' : 'mt-0.5'}`}
          >
            <ArrowLeft className="h-[18px] w-[18px]" />
          </button>
        )}
        <IconSquare icon={icon} tone={tone} size={compact ? 'md' : 'lg'} className={compact ? 'hidden @md:flex' : ''} />
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-1">
            <h1 className={`min-w-0 max-w-full truncate font-semibold tracking-tight text-zinc-100 ${compact ? 'text-lg leading-7' : 'text-2xl'}`}>
              {title}
            </h1>
            {meta && <div className="flex flex-wrap items-center gap-1.5">{meta}</div>}
          </div>
          {description && (
            <p className={`max-w-3xl text-sm leading-relaxed text-zinc-500 ${compact ? 'line-clamp-1' : 'mt-1'}`}>{description}</p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {action && <div className="hidden @2xl:flex">{action}</div>}
          <FullscreenButton className={ICON_BTN} />
          {onClose && (
            <button onClick={onClose} aria-label="Close" title="Close" className={ICON_BTN}>
              <X className="h-[18px] w-[18px]" />
            </button>
          )}
        </div>
      </div>
      {action && <div className="flex @2xl:hidden">{action}</div>}
    </div>
  )
}

/** A small heading inside a panel. */
export function SectionLabel({ icon: Icon, children, aside }: { icon?: LucideIcon; children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="flex items-center gap-1.5">
      {Icon && <Icon className="h-3.5 w-3.5 text-zinc-500" />}
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">{children}</h3>
      {aside && <div className="ml-auto flex items-center gap-1.5">{aside}</div>}
    </div>
  )
}

// ── Panels over the discussion (synthesis, TL;DRs, action points) ──

/** A panel's head: its tinted icon, title and a word under it — a click
 *  folds it — and its action on the right. */
export function PanelHead({ icon, tone, title, sub, collapsed, onToggle, action }: {
  icon: LucideIcon
  tone: CouncilTone
  title: ReactNode
  sub?: ReactNode
  collapsed: boolean
  onToggle: () => void
  action?: ReactNode
}) {
  return (
    <div className="flex items-center gap-2 px-3 py-2.5">
      <button
        onClick={onToggle}
        aria-expanded={!collapsed}
        className="flex min-w-0 flex-1 items-center gap-3 rounded-xl px-1 py-0.5 text-left"
      >
        <IconSquare icon={icon} tone={tone} size="sm" />
        <span className="min-w-0">
          <span className="block text-sm font-semibold text-zinc-100">{title}</span>
          {sub && <span className="block truncate text-xs text-zinc-500">{sub}</span>}
        </span>
        <ChevronDown className={`ml-0.5 h-4 w-4 shrink-0 text-zinc-500 transition-transform ${collapsed ? '-rotate-90' : ''}`} />
      </button>
      {action && <div className="flex shrink-0 items-center gap-2">{action}</div>}
    </div>
  )
}
