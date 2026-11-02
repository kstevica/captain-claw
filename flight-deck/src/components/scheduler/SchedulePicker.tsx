import { DAY_SHORT, fmtWhen, type Frequency, type ScheduleForm } from '../../services/schedulerApi'
import type { SchedulePreview } from './schedulePreview'

// When a task runs, in words a person would use: Scheduled tasks and Forge agents.

const FREQS: { id: Frequency; label: string }[] = [
  { id: 'daily', label: 'Every day' },
  { id: 'weekdays', label: 'Weekdays' },
  { id: 'weekly', label: 'Weekly' },
  { id: 'every', label: 'Repeat every…' },
  { id: 'once', label: 'Once' },
]

const field = 'rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-violet-500/60 focus:outline-none'

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function SchedulePicker({ form, onChange, preview, tz }: {
  form: ScheduleForm
  onChange: (update: (was: ScheduleForm) => ScheduleForm) => void
  preview: SchedulePreview | null
  tz: string
}) {
  return (
    <>
      <div className="flex flex-wrap gap-1 rounded-lg border border-zinc-800 bg-zinc-900/60 p-1">
        {FREQS.map((f) => (
          <button
            key={f.id}
            type="button"
            onClick={() => onChange((s) => ({ ...s, freq: f.id }))}
            className={`flex-1 whitespace-nowrap rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
              form.freq === f.id ? 'bg-zinc-700 text-zinc-100 shadow-sm' : 'text-zinc-400 hover:text-zinc-200'}`}
          >{f.label}</button>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {form.freq === 'weekly' && (
          <div className="flex gap-1">
            {DAY_SHORT.map((d, i) => {
              const on = form.days.includes(i)
              return (
                <button
                  key={d}
                  type="button"
                  aria-pressed={on}
                  onClick={() => onChange((s) => ({
                    ...s, days: on ? (s.days.length > 1 ? s.days.filter((x) => x !== i) : s.days) : [...s.days, i],
                  }))}
                  className={`h-9 w-10 rounded-lg text-xs font-medium transition-colors ${
                    on ? 'bg-violet-600 text-white' : 'border border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`}
                >{d.slice(0, 2)}</button>
              )
            })}
          </div>
        )}
        {form.freq === 'every' ? (
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            Every
            <input
              type="number" min={1} max={999}
              value={form.every}
              aria-label="How many"
              onChange={(e) => onChange((s) => ({ ...s, every: Number(e.target.value) || 1 }))}
              className={`${field} w-20`}
            />
            <select value={form.unit} aria-label="Minutes, hours or days" onChange={(e) => onChange((s) => ({ ...s, unit: e.target.value as ScheduleForm['unit'] }))} className={field}>
              <option value="m">minutes</option>
              <option value="h">hours</option>
              <option value="d">days</option>
            </select>
          </div>
        ) : (
          <>
            {form.freq === 'once' && (
              <input type="date" value={form.date} aria-label="Date" onChange={(e) => onChange((s) => ({ ...s, date: e.target.value }))} className={field} />
            )}
            <span className="text-sm text-zinc-400">at</span>
            <input type="time" value={form.time} aria-label="Time" onChange={(e) => onChange((s) => ({ ...s, time: e.target.value }))} className={field} />
          </>
        )}
      </div>
      {form.freq === 'every' && form.unit === 'm' && form.every < 15 && (
        <p className="mt-1.5 text-[11px] text-amber-700 dark:text-amber-400">That's very often — each run costs a full agent turn.</p>
      )}
      <p className={`mt-2 text-xs ${preview?.error ? 'text-red-600 dark:text-red-400' : 'text-zinc-500'}`}>
        {preview?.error
          ? capitalize(preview.error)
          : preview?.next.length
            ? <>Next: <span className="text-zinc-300">{fmtWhen(preview.next[0])}</span>
                {preview.next.slice(1).map((t) => <span key={t}>, then {fmtWhen(t)}</span>)}
                {tz && <span className="text-zinc-600"> · {tz.replace(/_/g, ' ')} time</span>}</>
            : ' '}
      </p>
    </>
  )
}
