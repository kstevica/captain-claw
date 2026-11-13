import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import type { LucideIcon } from 'lucide-react'
import { Check, ChevronDown, ChevronUp, Maximize2, Minimize2, Search, X } from 'lucide-react'
import { TOKEN_RE } from '../../services/anonymizeApi'
import { kindLabel, kindTone } from '../anonymize/anonFormat'
import { baseToken, tokenKind, type Range } from './roomText'

// Pieces the data room's views share: a dialog that can fill the window, a
// search field that steps through its matches, a picker that looks like the
// rest of the deck, and text with tokens as chips and matches marked.

export function TokenChip({ token, active = false, onClick }: { token: string; active?: boolean; onClick?: () => void }) {
  const kind = tokenKind(token)
  const cls = `inline-flex items-center rounded-md px-1.5 py-px font-mono text-[0.8em] leading-5 ${kindTone(kind)} ${
    active ? 'ring-2 ring-amber-500 ring-offset-1 ring-offset-zinc-950' : ''}`
  if (!onClick) return <span className={cls} title={kindLabel(kind)}>{token}</span>
  return (
    <button type="button" onClick={onClick} className={`${cls} cursor-pointer transition hover:brightness-110`}
      title={`${kindLabel(kind)} — show every place it appears`}>
      {token}
    </button>
  )
}

// ── a dialog that can fill the window ──────────────────────────────

/** A data-room dialog: big by default, with a switch to fill the window.
 *  Its body lays itself out (``scroll`` gives it its own scrolling). */
export function RoomDialog({ title, subtitle, icon: Icon, onClose, actions, footer, children, size = 'large',
  scroll = true }: {
  title: ReactNode
  subtitle?: ReactNode
  icon?: LucideIcon
  onClose: () => void
  actions?: ReactNode
  footer?: ReactNode
  children: ReactNode
  size?: 'large' | 'medium'
  scroll?: boolean
}) {
  const [full, setFull] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      const el = e.target as HTMLElement | null
      // A picker closes itself first; typed words in a box aren't thrown away.
      if (el?.closest?.('[data-esc-own]')) return
      if (el instanceof HTMLTextAreaElement && el.value.trim()) return
      e.stopPropagation()
      onClose()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  const box = full
    ? 'h-[100dvh] w-full rounded-none'
    : size === 'large'
      ? 'h-[100dvh] w-full wide:h-[92dvh] wide:w-[min(1440px,96vw)] wide:rounded-2xl wide:border wide:border-zinc-800'
      : 'max-h-[100dvh] w-full wide:max-h-[88dvh] wide:w-[min(860px,94vw)] wide:rounded-2xl wide:border wide:border-zinc-800'
  return createPortal(
    <div className={`fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 backdrop-blur-[2px] wide:items-center ${full ? '' : 'wide:p-4'}`}
      onClick={onClose}>
      <div role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}
        className={`flex flex-col overflow-hidden bg-zinc-950 shadow-2xl ${box}`}>
        <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
          {Icon && (
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
              <Icon className="h-[18px] w-[18px]" />
            </span>
          )}
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-[15px] font-semibold text-zinc-100">{title}</h2>
            {subtitle && <div className="mt-0.5 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-zinc-500">{subtitle}</div>}
          </div>
          {actions && <div className="hidden shrink-0 items-center gap-2 sm:flex">{actions}</div>}
          <div className="flex shrink-0 items-center">
            <button type="button" onClick={() => setFull(!full)} className="hidden rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 wide:inline-flex"
              title={full ? 'Exit fullscreen' : 'Fullscreen'} aria-label={full ? 'Exit fullscreen' : 'Fullscreen'} aria-pressed={full}>
              {full ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </button>
            <button type="button" onClick={onClose} className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
              title="Close" aria-label="Close">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-4 py-2 sm:hidden">{actions}</div>}
        <div className={`min-h-0 flex-1 ${scroll ? 'overflow-y-auto' : 'flex flex-col'}`}>{children}</div>
        {footer && (
          <div className="flex flex-wrap items-center justify-end gap-2 border-t border-zinc-800 px-4 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:px-5">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}

// ── search with matches ────────────────────────────────────────────

/** A search box that says how many it found and steps through them. */
export function SearchField({ value, onChange, placeholder, total, at, onStep, autoFocus = false, className = '' }: {
  value: string
  onChange: (v: string) => void
  placeholder: string
  total?: number
  at?: number
  onStep?: (dir: 1 | -1) => void
  autoFocus?: boolean
  className?: string
}) {
  const show = value.trim() !== '' && total !== undefined
  return (
    <div className={`flex min-w-0 items-center gap-1 rounded-xl border border-zinc-800 bg-zinc-950 pl-3 pr-1 focus-within:border-violet-500/60 focus-within:ring-2 focus-within:ring-violet-500/15 ${className}`}>
      <Search className="h-4 w-4 shrink-0 text-zinc-500" />
      <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} autoFocus={autoFocus}
        onKeyDown={(e) => { if (e.key === 'Enter' && onStep) { e.preventDefault(); onStep(e.shiftKey ? -1 : 1) } }}
        className="min-w-0 flex-1 bg-transparent py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none" />
      {show && (
        <span className={`shrink-0 px-1 text-xs tabular-nums ${total ? 'text-zinc-400' : 'text-amber-700 dark:text-amber-400'}`}>
          {total ? `${(at ?? 0) + 1} / ${total}` : 'none'}
        </span>
      )}
      {show && onStep && total! > 0 && (
        <>
          <IconBtn label="Previous (Shift+Enter)" onClick={() => onStep(-1)}><ChevronUp className="h-4 w-4" /></IconBtn>
          <IconBtn label="Next (Enter)" onClick={() => onStep(1)}><ChevronDown className="h-4 w-4" /></IconBtn>
        </>
      )}
      {value && <IconBtn label="Clear" onClick={() => onChange('')}><X className="h-3.5 w-3.5" /></IconBtn>}
    </div>
  )
}

export function IconBtn({ label, onClick, children, active = false }: {
  label: string; onClick: () => void; children: ReactNode; active?: boolean
}) {
  return (
    <button type="button" onClick={onClick} title={label} aria-label={label} aria-pressed={active}
      className={`shrink-0 rounded-lg p-1.5 transition-colors ${active ? 'bg-violet-500/15 text-violet-600 dark:text-violet-300' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`}>
      {children}
    </button>
  )
}

// ── a picker ───────────────────────────────────────────────────────

export interface PickOption {
  value: string
  label: ReactNode
  /** What typing in the picker matches. */
  search: string
  group?: string
  meta?: ReactNode
}

/** A searchable list in a popover — the deck's own look, not the browser's. */
export function Picker({ value, options, onChange, placeholder, icon: Icon, empty = 'Nothing to pick.', className = '' }: {
  value: string
  options: PickOption[]
  onChange: (v: string) => void
  placeholder: string
  icon?: LucideIcon
  empty?: string
  className?: string
}) {
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const [hi, setHi] = useState(0)
  const box = useRef<HTMLDivElement>(null)
  const chosen = options.find((o) => o.value === value)
  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return needle ? options.filter((o) => o.search.toLowerCase().includes(needle)) : options
  }, [options, q])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false) }
    window.addEventListener('mousedown', onDown)
    return () => window.removeEventListener('mousedown', onDown)
  }, [open])

  const pick = (v: string) => { onChange(v); setOpen(false); setQ('') }

  return (
    <div ref={box} className={`relative min-w-0 ${className}`} data-esc-own={open ? '' : undefined}>
      <button type="button" onClick={() => { setOpen(!open); setHi(0) }}
        className={`flex w-full min-w-0 items-center gap-2 rounded-xl border bg-zinc-950 px-3 py-2 text-left text-sm transition-colors ${
          open ? 'border-violet-500/60 ring-2 ring-violet-500/15' : 'border-zinc-800 hover:border-zinc-600'}`}>
        {Icon && <Icon className="h-4 w-4 shrink-0 text-zinc-500" />}
        <span className={`min-w-0 flex-1 truncate ${chosen ? 'text-zinc-100' : 'text-zinc-500'}`}>{chosen ? chosen.label : placeholder}</span>
        {chosen && (
          <span role="button" tabIndex={0} aria-label="Clear" onClick={(e) => { e.stopPropagation(); onChange('') }}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.stopPropagation(); onChange('') } }}
            className="rounded p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"><X className="h-3.5 w-3.5" /></span>
        )}
        <ChevronDown className={`h-4 w-4 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="absolute left-0 right-0 top-full z-20 mt-1.5 min-w-[18rem] overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl">
          <div className="border-b border-zinc-800 p-2">
            <input autoFocus value={q} onChange={(e) => { setQ(e.target.value); setHi(0) }} placeholder="Type to filter"
              onKeyDown={(e) => {
                if (e.key === 'ArrowDown') { e.preventDefault(); setHi((h) => Math.min(h + 1, shown.length - 1)) }
                if (e.key === 'ArrowUp') { e.preventDefault(); setHi((h) => Math.max(h - 1, 0)) }
                if (e.key === 'Enter' && shown[hi]) { e.preventDefault(); pick(shown[hi].value) }
                if (e.key === 'Escape') { e.stopPropagation(); setOpen(false) }
              }}
              className="w-full rounded-lg bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none" />
          </div>
          <ul role="listbox" className="max-h-72 overflow-y-auto p-1">
            {shown.length === 0 && <li className="px-3 py-4 text-center text-xs text-zinc-500">{empty}</li>}
            {shown.map((o, i) => {
              const head = o.group && (i === 0 || shown[i - 1].group !== o.group) ? o.group : undefined
              return (
                <li key={o.value}>
                  {head && <div className="px-2.5 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">{head}</div>}
                  <button type="button" role="option" aria-selected={o.value === value} onMouseEnter={() => setHi(i)} onClick={() => pick(o.value)}
                    className={`flex w-full min-w-0 items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-sm ${i === hi ? 'bg-zinc-800/80' : ''}`}>
                    <span className="min-w-0 flex-1 truncate text-zinc-200">{o.label}</span>
                    {o.meta && <span className="shrink-0 text-xs tabular-nums text-zinc-500">{o.meta}</span>}
                    {o.value === value && <Check className="h-3.5 w-3.5 shrink-0 text-violet-500" />}
                  </button>
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </div>
  )
}

// ── text with tokens and marks ─────────────────────────────────────

const MARK = 'rounded-[3px] bg-yellow-200 text-yellow-950 shadow-[0_0_0_1px_rgba(234,179,8,0.45)] dark:bg-yellow-400/25 dark:text-yellow-50'
const MARK_NOW = 'rounded-[3px] bg-amber-300 text-amber-950 shadow-[0_0_0_2px_rgba(245,158,11,0.9)] dark:bg-amber-400/60 dark:text-white'

/** One stretch of text: tokens as chips (``tokens``), the parts in *ranges*
 *  marked, the *current* range marked brighter (its element gets
 *  ``data-current``). Offsets in *ranges* are into the whole text; *offset*
 *  is where this stretch starts in it. */
export function Marked({ text, offset = 0, ranges, current = -1, tokens = false, onToken, activeToken }: {
  text: string
  offset?: number
  ranges: Range[]
  current?: number
  tokens?: boolean
  onToken?: (token: string) => void
  activeToken?: string
}) {
  const end = offset + text.length
  // The ranges that touch this stretch (ranges are sorted).
  let lo = 0, hi = ranges.length
  while (lo < hi) { const mid = (lo + hi) >> 1; if (ranges[mid][1] <= offset) lo = mid + 1; else hi = mid }
  const here: { r: Range; i: number }[] = []
  for (let i = lo; i < ranges.length && ranges[i][0] < end; i++) here.push({ r: ranges[i], i })

  const markPiece = (s: number, e: number, key: string): ReactNode[] => {
    const out: ReactNode[] = []
    let pos = s
    for (const { r, i } of here) {
      const a = Math.max(r[0], s), b = Math.min(r[1], e)
      if (b <= a) continue
      if (a > pos) out.push(text.slice(pos - offset, a - offset))
      out.push(<mark key={`${key}-${i}-${a}`} data-current={i === current ? '' : undefined} className={i === current ? MARK_NOW : MARK}>
        {text.slice(a - offset, b - offset)}
      </mark>)
      pos = b
    }
    if (pos < e) out.push(text.slice(pos - offset, e - offset))
    return out
  }

  if (!tokens) return <>{markPiece(offset, end, 't')}</>
  const out: ReactNode[] = []
  let pos = offset
  for (const m of text.matchAll(TOKEN_RE)) {
    const a = offset + (m.index ?? 0), b = a + m[0].length
    if (a > pos) out.push(...markPiece(pos, a, `p${pos}`))
    const kind = m[1]
    const active = !!activeToken && baseToken(m[0]) === activeToken
    out.push(
      <span key={`k${a}`} onClick={onToken ? () => onToken(baseToken(m[0])) : undefined} title={kindLabel(kind)}
        className={`mx-px inline-flex rounded-md px-1 font-mono text-[0.82em] leading-[1.45] ${kindTone(kind)} ${onToken ? 'cursor-pointer hover:brightness-110' : ''} ${active ? 'ring-1 ring-amber-500/70' : ''}`}>
        {markPiece(a, b, `k${a}`)}
      </span>,
    )
    pos = b
  }
  if (pos < end) out.push(...markPiece(pos, end, `p${pos}`))
  return <>{out}</>
}
