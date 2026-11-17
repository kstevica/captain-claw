import { useEffect, useState } from 'react'
import { Anchor, Save } from 'lucide-react'
import { Switch } from '../admin/Switch'
import { Button, Card, Notice, SettingRow, inputCls } from '../console/parts'
import { getDeckSettings, saveDeckSettings, type DeckSettings } from '../../services/firstMate'
import { useFirstMateStore } from '../../stores/firstMateStore'

// First Mate's deck switches on the admin console's Settings page: the same
// ones as "This deck" on the First Mate page. Each person approves their own
// factories there; admins don't see other people's factories.

export function FirstMateDeckCard() {
  const [s, setS] = useState<DeckSettings | null>(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [limits, setLimits] = useState({ max_factories: '', runs_per_day: '' })
  const reloadNav = useFirstMateStore((x) => x.load)

  useEffect(() => {
    getDeckSettings()
      .then((got) => {
        setS(got)
        setLimits({ max_factories: String(got.max_factories), runs_per_day: String(got.runs_per_day) })
      })
      .catch(() => setError("Couldn't load First Mate's settings"))
  }, [])

  const save = async (key: string, patch: Partial<DeckSettings>) => {
    setBusy(key); setError('')
    try {
      const next = await saveDeckSettings(patch)
      setS(next)
      setLimits({ max_factories: String(next.max_factories), runs_per_day: String(next.runs_per_day) })
      void reloadNav()  // who sees First Mate in the nav and the kiosk follows the switch
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setBusy('')
    }
  }

  const changed = !!s && (limits.max_factories !== String(s.max_factories) || limits.runs_per_day !== String(s.runs_per_day))

  return (
    <Card icon={Anchor} title="First Mate"
      hint="Standing jobs that run for people, each on a charter its owner approved. Each person approves their own factories, on First Mate and on their Line; admins don't see other people's factories."
      bodyClassName="divide-y divide-zinc-800/80 px-5 py-4">
      {error && <div className="pb-4"><Notice tone="red">{error}</Notice></div>}
      {!s ? (
        <p className="text-xs text-zinc-500">{error ? '' : 'Loading…'}</p>
      ) : (
        <>
          <SettingRow title="On for this deck" hint="Off: nobody sees it but admins, and nothing runs.">
            <Switch on={s.enabled} disabled={busy === 'enabled'} label="First Mate on for this deck"
              onToggle={() => save('enabled', { enabled: !s.enabled })} />
          </SettingRow>
          <SettingRow title="Admins only"
            hint="While First Mate is being tried out, only admins use it. Off: everyone signed in gets it (improvement runs stay with admins, like Captain Swarm).">
            <Switch on={s.admins_only} disabled={busy === 'admins_only'} label="First Mate for admins only"
              onToggle={() => save('admins_only', { admins_only: !s.admins_only })} />
          </SettingRow>
          <div className="py-4 last:pb-0">
            <div className="text-sm font-medium text-zinc-200">Limits</div>
            <div className="mt-0.5 text-xs text-zinc-500">How many factories each person may have (1 to 50), and how many runs each factory may start a day (1 to 500).</div>
            <form className="mt-3 flex flex-wrap items-end gap-3"
              onSubmit={(e) => {
                e.preventDefault()
                void save('limits', { max_factories: Number(limits.max_factories), runs_per_day: Number(limits.runs_per_day) })
              }}>
              <label className="w-40 text-xs text-zinc-400">
                <span className="mb-1 block">Factories per person</span>
                <input inputMode="numeric" className={inputCls} value={limits.max_factories}
                  onChange={(e) => setLimits({ ...limits, max_factories: e.target.value.replace(/\D/g, '') })} />
              </label>
              <label className="w-40 text-xs text-zinc-400">
                <span className="mb-1 block">Runs per factory a day</span>
                <input inputMode="numeric" className={inputCls} value={limits.runs_per_day}
                  onChange={(e) => setLimits({ ...limits, runs_per_day: e.target.value.replace(/\D/g, '') })} />
              </label>
              <Button type="submit" icon={Save} busy={busy === 'limits'} disabled={!changed || !limits.max_factories || !limits.runs_per_day}>Save</Button>
            </form>
          </div>
        </>
      )}
    </Card>
  )
}
