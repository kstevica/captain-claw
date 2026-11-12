import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown, Cpu } from 'lucide-react'
import { useChatStore } from '../../stores/chatStore'
import { isModel, priceText, type ModelChoice, type ModelSwitchInfo } from '../../utils/modelSwitch'

// The composer's model picker: the other models of the agent's own provider
// (same key, same endpoint) it can run this conversation on — the agent's
// configured model is the default. The pick is the agent's, kept with the
// conversation; the agent can switch too (when asked, or on its own when its
// owner allows), and the picker follows.
export function ModelMenu({ containerId, info, disabled, compact }: {
  containerId: string
  info: ModelSwitchInfo
  disabled?: boolean
  /** Phone: a shorter label. */
  compact?: boolean
}) {
  const switchModel = useChatStore((s) => s.switchModel)
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const current = info.choices.find((c) => isModel(info.current, c.model)) ?? null
  const onDefault = current ? current.default : isModel(info.current, info.default)
  const name = current?.name || info.current || 'Model'
  const choose = (c: ModelChoice) => {
    setOpen(false)
    if (current && c.model === current.model && !info.turnOnly) return
    switchModel(containerId, c.default ? 'default' : c.model)
  }

  return (
    // Phone: the menu spans the composer box (its positioned ancestor), as
    // the skills menu does — anchored to the button it would run off-screen.
    <div className={compact ? 'shrink-0' : 'relative shrink-0'} ref={ref}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="listbox"
        title={`Model: ${name}${info.turnOnly ? ' (for this turn)' : ''} — the AI model the agent runs on`}
        aria-label={`Model: ${name}`}
        // Phone: the icon only — the composer row is full; the menu names it.
        className={`flex h-9 items-center rounded-lg border text-sm transition-colors disabled:opacity-40 ${
          compact ? 'w-9 justify-center' : 'max-w-[11rem] gap-1.5 px-2.5'
        } ${
          open
            ? 'border-sky-500/40 bg-sky-500/10 text-zinc-100'
            : !onDefault
              ? 'border-sky-500/30 text-zinc-200 hover:bg-zinc-800'
              : 'border-zinc-700 text-zinc-300 hover:bg-zinc-800'
        }`}
      >
        <Cpu className={`h-4 w-4 shrink-0 ${onDefault ? 'text-zinc-500' : 'text-sky-500'}`} />
        {!compact && <>
          <span className="min-w-0 truncate">{name}</span>
          <ChevronDown className={`h-3.5 w-3.5 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
        </>}
      </button>
      {open && (
        <div
          role="listbox"
          aria-label="Model"
          className={`absolute bottom-full z-40 mb-2 overflow-hidden rounded-2xl border border-zinc-700 bg-zinc-900 shadow-2xl ${
            compact ? 'inset-x-0' : 'left-0 w-80'
          }`}
        >
          <div className="border-b border-zinc-800 px-4 py-2.5">
            <div className="text-sm font-medium text-zinc-200">Model</div>
            <div className="text-xs text-zinc-500">Its provider's models, on the same key — for this conversation</div>
          </div>
          <div className="max-h-[min(55vh,26rem)] overflow-y-auto p-1.5">
            {info.choices.map((c) => {
              const active = !!current && c.model === current.model
              const hint = [c.label, c.note, priceText(c.price)].filter(Boolean).join(' · ')
              return (
                <button
                  key={c.model}
                  type="button"
                  role="option"
                  aria-selected={active}
                  onClick={() => choose(c)}
                  className={`flex w-full items-center gap-3 rounded-xl px-2.5 py-2 text-left transition-colors ${
                    active ? 'bg-sky-500/10' : 'hover:bg-zinc-800'
                  }`}
                >
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2 text-[15px] font-medium text-zinc-100">
                      <span className="truncate">{c.name}</span>
                      {c.default && <Tag>default</Tag>}
                      {active && info.turnOnly && <Tag>this turn</Tag>}
                    </span>
                    <span className="block truncate text-sm text-zinc-400" title={`${c.model}${hint ? ` — ${hint}` : ''}`}>
                      {hint || c.model}
                    </span>
                  </span>
                  {active && <Check className="h-4 w-4 shrink-0 text-sky-500" />}
                </button>
              )
            })}
          </div>
          <div className="border-t border-zinc-800 px-4 py-2 text-xs text-zinc-500">
            {info.selfSwitch
              ? 'It may also pick a smarter or cheaper one for a task on its own, and tells you when it does.'
              : 'It switches only when you pick here or ask it. Agent settings can let it pick on its own.'}
          </div>
        </div>
      )}
    </div>
  )
}

function Tag({ children }: { children: string }) {
  return (
    <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-px text-[10px] font-normal uppercase tracking-wide text-zinc-400">
      {children}
    </span>
  )
}
