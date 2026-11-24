import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown, Gauge } from 'lucide-react'
import { useDriveStore } from '../../stores/driveStore'
import { DRIVE_LABEL, DRIVE_LEVELS, type DrivePick } from '../../services/driveLevels'

// How far Captain pushes to finish this turn — the composer's Drive picker,
// next to reasoning effort (how hard the model thinks). Auto leaves it to the
// agent's own default; the pick is kept per agent and travels with every
// message sent to it.
export function DriveMenu({ agentId, disabled, compact }: {
  agentId: string
  disabled?: boolean
  /** Phone: icon only when nothing is picked. */
  compact?: boolean
}) {
  const pick: DrivePick = useDriveStore((s) => s.byAgent[agentId]) ?? 'auto'
  const setDrive = useDriveStore((s) => s.setDrive)
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

  const choose = (p: DrivePick) => { setDrive(agentId, p); setOpen(false) }
  const label = pick === 'auto' ? 'Auto' : DRIVE_LABEL[pick]

  return (
    <div className={compact ? 'shrink-0' : 'relative shrink-0'} ref={ref}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={`Drive: ${label}`}
        title={`Drive: ${pick === 'auto' ? "Auto (the agent's own default)" : label} — how far it pushes to finish`}
        className={`flex h-9 items-center gap-1.5 rounded-lg border px-2.5 text-sm transition-colors disabled:opacity-40 ${
          open
            ? 'border-amber-500/40 bg-amber-500/10 text-zinc-100'
            : pick !== 'auto'
              ? 'border-amber-500/30 text-zinc-200 hover:bg-zinc-800'
              : 'border-zinc-700 text-zinc-300 hover:bg-zinc-800'
        }`}
      >
        <Gauge className={`h-4 w-4 ${pick === 'auto' ? 'text-zinc-500' : 'text-amber-500'}`} />
        {!(compact && pick === 'auto') && <span>{label}</span>}
        <ChevronDown className={`h-3.5 w-3.5 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div
          role="listbox"
          aria-label="Drive"
          className={`absolute bottom-full z-40 mb-2 overflow-hidden rounded-2xl border border-zinc-700 bg-zinc-900 shadow-2xl ${
            compact ? 'inset-x-0' : 'left-0 w-80'
          }`}
        >
          <div className="border-b border-zinc-800 px-4 py-2.5">
            <div className="text-sm font-medium text-zinc-200">Drive</div>
            <div className="text-xs text-zinc-500">How far it pushes to finish what you asked</div>
          </div>
          <div className="max-h-[min(55vh,26rem)] overflow-y-auto p-1.5">
            <DriveRow active={pick === 'auto'} title="Auto" hint="The agent's own default" onClick={() => choose('auto')} />
            {DRIVE_LEVELS.map((lvl) => (
              <DriveRow key={lvl.id} active={pick === lvl.id} title={lvl.label} hint={lvl.hint} onClick={() => choose(lvl.id)} />
            ))}
          </div>
          <div className="border-t border-zinc-800 px-4 py-2 text-xs text-zinc-500">
            Higher levels take more steps and more tokens.
          </div>
        </div>
      )}
    </div>
  )
}

function DriveRow({ active, title, hint, onClick }: {
  active: boolean
  title: string
  hint: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      role="option"
      aria-selected={active}
      onClick={onClick}
      className={`flex w-full items-center gap-3 rounded-xl px-2.5 py-2 text-left transition-colors ${
        active ? 'bg-amber-500/10' : 'hover:bg-zinc-800'
      }`}
    >
      <span className="min-w-0 flex-1">
        <span className="block text-[15px] font-medium text-zinc-100">{title}</span>
        <span className="block text-sm text-zinc-400">{hint}</span>
      </span>
      {active && <Check className="h-4 w-4 shrink-0 text-amber-500" />}
    </button>
  )
}
