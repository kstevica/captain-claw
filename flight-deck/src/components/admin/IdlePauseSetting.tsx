import { Switch } from './Switch'

// Scheduled tasks pause for people who are away (Flight Deck's idle_pause.py):
// a switch, and how many days without activity first. The Admin page and the
// console's Settings share it. Decks with accounts only.

export const IDLE_PAUSE_TITLE = 'Pause scheduled tasks for people who are away'
export const IDLE_PAUSE_HINT =
  'When someone hasn’t signed in or written to their agents for this long, their scheduled tasks — and the jobs their agents set up for themselves — are paused, and their Line says so. Nothing is deleted; they resume each one in Scheduled tasks.'

const DEFAULT_DAYS = 7
const CHOICES = [3, 5, 7, 14, 30, 60, 90]

function daysLabel(d: number): string {
  return d === 1 ? '1 day' : `${d} days`
}

export function IdlePauseSetting({ days, busy, onSave }: {
  days: number
  busy: boolean
  onSave: (days: number) => void
}) {
  const on = days > 0
  // A count set outside the list (the API takes 1 to 365) stays pickable.
  const choices = on && !CHOICES.includes(days) ? [...CHOICES, days].sort((a, b) => a - b) : CHOICES
  return (
    <div className="flex items-center gap-3">
      {on && (
        <select
          value={days}
          disabled={busy}
          onChange={(e) => onSave(Number(e.target.value))}
          aria-label="Days away before scheduled tasks pause"
          className="rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/60 focus:outline-none disabled:opacity-60"
        >
          {choices.map((d) => <option key={d} value={d}>after {daysLabel(d)}</option>)}
        </select>
      )}
      <Switch
        on={on}
        disabled={busy}
        label={IDLE_PAUSE_TITLE}
        onToggle={() => onSave(on ? 0 : DEFAULT_DAYS)}
      />
    </div>
  )
}
