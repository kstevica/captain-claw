import { useEffect, useRef, useState } from 'react'
import { Check, Loader2, Minus, Plus, Users } from 'lucide-react'
import { getAgentTeamUp, setAgentTeamUp, type TeamUpSettings } from '../../../services/docker'
import { FOCUS } from '../queue/QueueButton'

// Team up's per-agent setting (captain_claw/team_up.py): whether the agent may
// split a big task among helpers — copies of itself — and how far it may go.
// Saved at once; the agent reads it on every message.

const MODES: { id: TeamUpSettings['mode']; label: string; help: string }[] = [
  { id: 'off', label: 'Off', help: 'It never offers a team and does everything itself.' },
  { id: 'ask', label: 'Ask me', help: 'It offers a team on a card when a task splits; you tap Team up or Do it alone.' },
  { id: 'on', label: 'On', help: 'It starts a team by itself when a task splits — you see it working.' },
]

export function TeamUpSetting({ kind, identifier, title = false }: {
  kind: 'docker' | 'process'; identifier: string; title?: boolean
}) {
  const [s, setS] = useState<TeamUpSettings | null>(null)
  const [missing, setMissing] = useState(false)
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    let live = true
    getAgentTeamUp(kind, identifier)
      .then((r) => { if (live) setS(r) })
      .catch(() => { if (live) setMissing(true) })
    return () => { live = false; if (timer.current) clearTimeout(timer.current) }
  }, [kind, identifier])

  if (missing) return title ? null : <p className="text-xs text-zinc-500">Not available for this agent.</p>
  if (!s) return null

  const save = async (next: TeamUpSettings) => {
    const prev = s
    setS(next); setBusy(true); setError(''); setSaved(false)
    try {
      const r = await setAgentTeamUp(kind, identifier, next)
      setS({ mode: r.mode, max_helpers: r.max_helpers, budget_x: r.budget_x })
      setSaved(true)
      if (timer.current) clearTimeout(timer.current)
      timer.current = setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      setS(prev)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const mode = MODES.find((m) => m.id === s.mode) ?? MODES[1]
  const off = s.mode === 'off'

  return (
    <div>
      {title && (
        <h3 className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-400">
          <Users className="h-3.5 w-3.5" />
          Team up
          {busy && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
          {saved && <span className="flex items-center gap-0.5 normal-case tracking-normal text-emerald-600 dark:text-emerald-400"><Check className="h-3 w-3" />saved</span>}
        </h3>
      )}
      <div className="rounded-lg border border-zinc-800 px-3 py-3">
        <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-200">
          Split big tasks among copies of itself
          {!title && busy && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
          {!title && saved && <Check className="h-3 w-3 text-emerald-500" />}
        </div>
        <p className="mt-0.5 text-[11px] leading-relaxed text-zinc-500">
          Helpers share its memory, tools and model and work at the same time, in the background, while you keep
          chatting. They never send, publish or pay — the agent does that once, after. Web chat only.
        </p>

        <div role="radiogroup" aria-label="Team up" className="mt-3 inline-flex rounded-lg border border-zinc-800 bg-zinc-900/60 p-0.5">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button" role="radio" aria-checked={s.mode === m.id}
              onClick={() => s.mode !== m.id && !busy && save({ ...s, mode: m.id })}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${FOCUS} ${
                s.mode === m.id ? 'bg-violet-600 text-white shadow-sm' : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <p className="mt-1.5 text-[11px] text-zinc-500">{mode.help}</p>

        <div className={`mt-3 grid grid-cols-2 gap-3 transition-opacity ${off ? 'pointer-events-none opacity-40' : ''}`}>
          <Stepper
            label="Helpers at most" value={s.max_helpers} min={2} max={4} disabled={busy || off}
            onChange={(v) => save({ ...s, max_helpers: v })}
          />
          {/* No team budget for now (team_up.BUDGET_CAP): the setting is kept, not shown. */}
        </div>
      </div>
      {error && <p className="mt-1.5 text-[11px] text-red-500">{error}</p>}
    </div>
  )
}

function Stepper({ label, value, min, max, suffix, disabled, onChange }: {
  label: string; value: number; min: number; max: number; suffix?: string; disabled: boolean; onChange: (v: number) => void
}) {
  const btn = `flex h-7 w-7 items-center justify-center rounded-md text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40 ${FOCUS}`
  return (
    <div>
      <div className="text-[11px] text-zinc-500">{label}</div>
      <div className="mt-1 flex items-center gap-1">
        <button type="button" className={btn} disabled={disabled || value <= min} onClick={() => onChange(value - 1)} aria-label={`Less: ${label}`}>
          <Minus className="h-3.5 w-3.5" />
        </button>
        <span className="min-w-[1.75rem] text-center font-mono text-sm tabular-nums text-zinc-100">{value}{suffix ? '×' : ''}</span>
        <button type="button" className={btn} disabled={disabled || value >= max} onClick={() => onChange(value + 1)} aria-label={`More: ${label}`}>
          <Plus className="h-3.5 w-3.5" />
        </button>
        {suffix && <span className="ml-1 truncate text-[11px] text-zinc-500">{suffix.replace(/^×\s*/, '')}</span>}
      </div>
    </div>
  )
}
