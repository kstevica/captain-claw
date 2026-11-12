import { useEffect, useState } from 'react'
import { ShieldCheck, Save, RotateCcw } from 'lucide-react'
import { anonymizeApi, type GateInfo, type GateSettings } from '../../services/anonymizeApi'
import { Card, Button, Dialog, Field, Notice, Segmented, SettingRow, Switch, inputCls } from '../console/parts'
import { RESET_TEXT } from './anonFormat'

// The privacy gate: what their agents' models read is anonymized first — when
// that model isn't trusted (or always). Either the sources the person picks
// (mail, chats, files…), or the whole chat: their messages, every tool result,
// the notes and the system prompt. Mail and messages the agents send get the
// real values back just before they leave; with the whole chat, the person's
// own chat shows them too.

export function GatePanel() {
  const [info, setInfo] = useState<GateInfo | null>(null)
  const chat = (g: GateSettings | null) => g?.coverage === 'chat'
  const [gate, setGate] = useState<GateSettings | null>(null)
  const [folders, setFolders] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const [confirmReset, setConfirmReset] = useState(false)
  const [resetting, setResetting] = useState(false)
  const [resetDone, setResetDone] = useState('')

  const resetChat = async () => {
    setResetting(true); setError('')
    try {
      await anonymizeApi.resetVault('Chat')
      setResetDone('The chat vault is empty. Start a new conversation with your agents.')
    } catch (e) {
      const msg = e instanceof Error ? e.message : ''
      setError(/No such vault/.test(msg) ? 'There is no chat vault yet — nothing to reset.' : msg || "Couldn't reset it")
    } finally {
      setResetting(false); setConfirmReset(false)
    }
  }

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
        With the gate on, names, numbers and addresses reach your agents' model as tokens (⟦PERSON_1⟧) — the real values
        stay in your vault “{info.vault}”. Mail and WhatsApp messages they send get the real values back just before they
        leave. {chat(gate)
          ? 'With the whole chat covered, your own chat shows the real values again.'
          : 'In a chat, “Real names” shows a reply as you\'d write it.'}
      </Notice>
      <Card icon={ShieldCheck} title="Privacy gate" bodyClassName="space-y-5 p-5">
        <div className="divide-y divide-zinc-800/80">
          <SettingRow title="On" hint="Off: your agents read everything as it is.">
            <Switch on={gate.enabled} label="Privacy gate on" onChange={(v) => set({ enabled: v })} />
          </SettingRow>
          <SettingRow title="What it covers" hint={chat(gate)
            ? 'Everything the model reads: your messages, what every tool returns, the notes and the system prompt.'
            : 'Only what the sources you pick below give your agents.'}>
            <Segmented label="What it covers" value={gate.coverage || 'sources'} onChange={(v) => set({ coverage: v })}
              options={[{ value: 'sources', label: 'Sources I pick' }, { value: 'chat', label: 'The whole chat' }]} />
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

        {chat(gate) ? (
          <Notice>
            Your messages are anonymized before the agent reads them, and every tool's result before its model does. Your
            chat in Flight Deck — and your own Telegram chat — shows the real values; files the agent writes and searches it
            runs use them too. Anyone else talking to your agent (another Telegram user, a group, WhatsApp, a public page)
            sees the tokens. Pictures go to the model as they are.
          </Notice>
        ) : (<>
        <Field label="Sources">
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
        </>)}

        {error && <Notice tone="red">{error}</Notice>}
        {saved && <Notice tone="emerald">{gate.enabled ? 'Saved. Your agents follow it from their next step (within a minute).' : 'Saved. The gate is off.'}</Notice>}
        {resetDone && <Notice tone="emerald">{resetDone}</Notice>}
        <div className="flex justify-end gap-2">
          {chat(gate) && <Button icon={RotateCcw} onClick={() => { setConfirmReset(true); setResetDone('') }} className="mr-auto">Reset the chat vault</Button>}
          <Button variant="primary" icon={Save} busy={busy} onClick={save}>Save</Button>
        </div>
      </Card>
      {confirmReset && (
        <Dialog title="Reset the chat vault?" icon={RotateCcw} onClose={() => setConfirmReset(false)} footer={<>
          <Button onClick={() => setConfirmReset(false)}>Keep it</Button>
          <Button variant="danger" icon={RotateCcw} busy={resetting} onClick={resetChat}>Reset</Button>
        </>}>
          <p className="text-sm leading-relaxed text-zinc-300">{RESET_TEXT}</p>
        </Dialog>
      )}
    </div>
  )
}
