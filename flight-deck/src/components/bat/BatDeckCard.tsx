import { useEffect, useState } from 'react'
import { Hammer, Save } from 'lucide-react'
import { Button, Card, Notice, SettingRow, inputCls } from '../console/parts'
import { getBatSettings, saveBatSettings, type BatSettings, type BatSettingsView } from '../../services/batSettings'

// Bat's deck settings on the admin console's Settings page. A run going now takes
// a change up: the steps at the same time between its steps, the others with its
// next worker.

type Field = keyof BatSettings

const FIELDS: { key: Field; title: string; hint: string; unit: string }[] = [
  { key: 'max_parallel', title: 'Steps at the same time',
    hint: 'How many steps of one run work at once. On a single local model server, 1 or 2: more only make each step wait for the others.',
    unit: 'steps' },
  { key: 'step_ceiling', title: 'Step ceiling per worker turn',
    hint: 'A worker that keeps making progress keeps going, up to this many steps in one turn. One going nowhere is stopped much sooner either way.',
    unit: 'steps' },
  { key: 'idle_minutes', title: 'Quiet minutes before a step stops',
    hint: 'A step has no time limit while its worker keeps working. This long without a sound from it (a hung model call) ends it.',
    unit: 'minutes' },
]

export function BatDeckCard() {
  const [s, setS] = useState<BatSettingsView | null>(null)
  const [draft, setDraft] = useState<Record<Field, string>>({ max_parallel: '', step_ceiling: '', idle_minutes: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  const take = (got: BatSettingsView) => {
    setS(got)
    setDraft({ max_parallel: String(got.max_parallel), step_ceiling: String(got.step_ceiling), idle_minutes: String(got.idle_minutes) })
  }

  useEffect(() => {
    getBatSettings().then(take).catch(() => setError("Couldn't load Bat's settings"))
  }, [])

  const changed = !!s && FIELDS.some((f) => draft[f.key] !== String(s[f.key]))
  const save = async () => {
    if (!s) return
    setBusy(true); setError(''); setSaved(false)
    try {
      const patch: Partial<BatSettings> = {}
      for (const f of FIELDS) if (draft[f.key] !== String(s[f.key])) patch[f.key] = Number(draft[f.key])
      take(await saveBatSettings(patch))
      setSaved(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card icon={Hammer} title="Bat"
      hint="How Bat, the stubborn finisher, runs on this deck (First Mate's factories run through it). A run going now takes a change up."
      bodyClassName="divide-y divide-zinc-800/80 px-5 py-4">
      {error && <div className="pb-4"><Notice tone="red">{error}</Notice></div>}
      {!s ? (
        <p className="text-xs text-zinc-500">{error ? '' : 'Loading…'}</p>
      ) : (
        <form onSubmit={(e) => { e.preventDefault(); void save() }}>
          {FIELDS.map((f) => {
            const [lo, hi] = s.ranges[f.key]
            return (
              <SettingRow key={f.key} title={f.title}
                hint={`${f.hint} ${lo}–${hi} ${f.unit}; started with ${s.defaults[f.key]}.`}>
                <input id={`bat-${f.key}`} aria-label={f.title} inputMode="numeric" className={`${inputCls} w-28 text-right`}
                  value={draft[f.key]} onChange={(e) => { setSaved(false); setDraft({ ...draft, [f.key]: e.target.value.replace(/\D/g, '') }) }} />
              </SettingRow>
            )
          })}
          <div className="flex items-center justify-end gap-3 pt-4">
            {saved && !changed && <span className="text-xs text-emerald-600 dark:text-emerald-400">Saved: in force now</span>}
            <Button type="submit" icon={Save} busy={busy} disabled={!changed || FIELDS.some((f) => !draft[f.key])}>Save</Button>
          </div>
        </form>
      )}
    </Card>
  )
}
