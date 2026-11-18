import { useEffect, useState } from 'react'
import { Hammer, Save } from 'lucide-react'
import { Button, Card, Notice, SettingRow, inputCls } from '../console/parts'
import { getBatSettings, saveBatSettings, type BatNumberSetting, type BatSettings, type BatSettingsView } from '../../services/batSettings'

// Bat's deck settings on the admin console's Settings page. A run going now takes
// a change up: the steps at the same time between its steps, the others with its
// next worker.

type Field = BatNumberSetting

const TIER_LABELS: Record<string, string> = {
  reason: 'Reasoning', balanced: 'Balanced', fast: 'Fast', longctx: 'Long context', coding: 'Coding', vision: 'Vision', micro: 'Micro',
}

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
  { key: 'call_minutes', title: 'Minutes one model call may take',
    hint: "Bat's own calls: planning, its checks, the reviewers, First Mate's drafts. A thinking model on one machine can think for several minutes before it answers; cut off, it starts again from nothing.",
    unit: 'minutes' },
]

const draftOf = (s: BatSettings) => Object.fromEntries(FIELDS.map((f) => [f.key, String(s[f.key])])) as Record<Field, string>

/** The tier Bat plans on: reasoning, else fast, else the first set (as the deck picks it). */
function planTier(s: BatSettingsView): string {
  for (const name of ['reason', 'fast']) if (s.your_tiers[name]) return name
  return Object.keys(s.your_tiers)[0] ?? 'reason'
}

export function BatDeckCard() {
  const [s, setS] = useState<BatSettingsView | null>(null)
  const [draft, setDraft] = useState<Record<Field, string>>({ max_parallel: '', step_ceiling: '', idle_minutes: '', call_minutes: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  const take = (got: BatSettingsView) => {
    setS(got)
    setDraft(draftOf(got))
    setTier(got.worker_tier || '')
  }

  useEffect(() => {
    getBatSettings().then(take).catch(() => setError("Couldn't load Bat's settings"))
  }, [])

  const [tier, setTier] = useState('')
  const changed = !!s && (FIELDS.some((f) => draft[f.key] !== String(s[f.key])) || tier !== s.worker_tier)
  const save = async () => {
    if (!s) return
    setBusy(true); setError(''); setSaved(false)
    try {
      const patch: Partial<BatSettings> = {}
      for (const f of FIELDS) if (draft[f.key] !== String(s[f.key])) patch[f.key] = Number(draft[f.key])
      if (tier !== s.worker_tier) patch.worker_tier = tier
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
          <SettingRow title="Workers use"
            hint={`Which tier of each person's own set works a run's steps: the searching, reading and checking, where nearly all the tokens go. Planning, the first step's conventions, the checks between steps and the reviewers stay on the ${TIER_LABELS[planTier(s)] ?? 'planning'} tier, so a stronger model there is spent only where a mistake costs hours. A tier someone's set leaves empty falls back to the planning one.`}>
            <select id="bat-worker_tier" aria-label="Workers use" className={`${inputCls} w-64`}
              value={tier} onChange={(e) => { setSaved(false); setTier(e.target.value) }}>
              <option value="">Same as planning{s.your_tiers[planTier(s)] ? ` · ${s.your_tiers[planTier(s)]}` : ''}</option>
              {s.tiers.map((name) => (
                <option key={name} value={name} disabled={!s.your_tiers[name] && name !== tier}>
                  {TIER_LABELS[name] ?? name}{s.your_tiers[name] ? ` · ${s.your_tiers[name]}` : ' · not set in your tiers'}
                </option>
              ))}
            </select>
          </SettingRow>
          {tier && s.your_tiers[tier] && s.your_tiers[tier] === s.your_tiers[planTier(s)] && (
            <p className="pt-3 text-xs text-zinc-500">
              In your own tiers that's the same model as planning ({s.your_tiers[tier]}): put a stronger model on the {TIER_LABELS[planTier(s)] ?? 'planning'} tier in Library → Model Tiers to split them.
            </p>
          )}
          <div className="flex items-center justify-end gap-3 pt-4">
            {saved && !changed && <span className="text-xs text-emerald-600 dark:text-emerald-400">Saved: in force now</span>}
            <Button type="submit" icon={Save} busy={busy} disabled={!changed || FIELDS.some((f) => !draft[f.key])}>Save</Button>
          </div>
        </form>
      )}
    </Card>
  )
}
