import { useEffect, useState } from 'react'
import { ShieldCheck, Save } from 'lucide-react'
import { anonymizeApi, type GateInfo, type GateSettings } from '../../services/anonymizeApi'
import { Card, Button, Field, Notice, Segmented, SettingRow, Switch, inputCls } from '../console/parts'

// The privacy gate: for the sources the person picks, what their agents read
// is anonymized before the agent's model sees it — when that model isn't
// trusted (or always). Mail and messages the agents send get the real values
// back just before they leave.

export function GatePanel() {
  const [info, setInfo] = useState<GateInfo | null>(null)
  const [gate, setGate] = useState<GateSettings | null>(null)
  const [folders, setFolders] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    anonymizeApi.gate()
      .then((i) => { setInfo(i); setGate(i.gate); setFolders(i.gate.folders.join('\n')) })
      .catch((e) => setError(e instanceof Error ? e.message : "Couldn't load the privacy gate"))
  }, [])

  if (!gate || !info) return error ? <Notice tone="red">{error}</Notice> : null

  const set = (patch: Partial<GateSettings>) => { setGate({ ...gate, ...patch }); setSaved(false) }
  const toggleGroup = (id: string) =>
    set({ groups: gate.groups.includes(id) ? gate.groups.filter((g) => g !== id) : [...gate.groups, id] })

  const save = async () => {
    setBusy(true); setError(''); setSaved(false)
    try {
      const r = await anonymizeApi.saveGate({
        ...gate, folders: folders.split('\n').map((f) => f.trim()).filter(Boolean),
      })
      setGate(r.gate)
      setSaved(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't save")
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-5">
      <Notice>
        With the gate on, what your agents read from the sources you pick reaches their model as tokens (⟦PERSON_1⟧) — the
        real values stay in your vault “{info.vault}”. Mail and WhatsApp messages they send get the real values back just
        before they leave. In a chat, “Real names” shows a reply as you'd write it.
      </Notice>
      <Card icon={ShieldCheck} title="Privacy gate" bodyClassName="space-y-5 p-5">
        <div className="divide-y divide-zinc-800/80">
          <SettingRow title="On" hint="Off: your agents read everything as it is.">
            <Switch on={gate.enabled} label="Privacy gate on" onChange={(v) => set({ enabled: v })} />
          </SettingRow>
          <SettingRow title="For which agents" hint="A model on this machine or your network, or the one your admin trusts, can see the real thing.">
            <Segmented label="For which agents" value={gate.when} onChange={(v) => set({ when: v })}
              options={[{ value: 'untrusted', label: 'On an untrusted model' }, { value: 'always', label: 'Every agent' }]} />
          </SettingRow>
          <SettingRow title="Who looks" hint={info.model_available
            ? 'Code is quick; the trusted model also catches names with nothing around them, but every result waits for it.'
            : 'This deck has no trusted model, so code looks.'}>
            <Segmented label="Who looks" value={info.model_available ? gate.lane : 'deterministic'} onChange={(v) => set({ lane: v })}
              options={[
                { value: 'deterministic', label: 'Code' },
                ...(info.model_available ? [{ value: 'both' as const, label: 'Code + trusted model' }] : []),
              ]} />
          </SettingRow>
        </div>

        <Field label="What it covers">
          <div className="grid gap-2 wide:grid-cols-2">
            {info.groups.map((g) => (
              <label key={g.id} className="flex items-center gap-2.5 rounded-xl border border-zinc-800 px-3 py-2 text-sm text-zinc-200">
                <input type="checkbox" checked={gate.groups.includes(g.id)} onChange={() => toggleGroup(g.id)} />
                {g.label}
              </label>
            ))}
          </div>
        </Field>

        {gate.groups.includes('files') && (
          <Field label="Only these folders (optional)"
            hint="One per line, as your agents name them: vfs:clients or /Users/you/Clients. Empty: every file they read. Drive and OneDrive are always covered.">
            <textarea value={folders} onChange={(e) => { setFolders(e.target.value); setSaved(false) }} rows={3} className={`${inputCls} font-mono`} />
          </Field>
        )}

        {error && <Notice tone="red">{error}</Notice>}
        {saved && <Notice tone="emerald">{gate.enabled ? 'Saved. Your agents follow it from their next step (within a minute).' : 'Saved. The gate is off.'}</Notice>}
        <div className="flex justify-end">
          <Button variant="primary" icon={Save} busy={busy} onClick={save}>Save</Button>
        </div>
      </Card>
    </div>
  )
}
