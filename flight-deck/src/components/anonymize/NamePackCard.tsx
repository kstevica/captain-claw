import { useCallback, useEffect, useState } from 'react'
import { ScanFace, Download } from 'lucide-react'
import { anonymizeApi, type NamePack } from '../../services/anonymizeApi'
import { Card, Button, Field, Notice, Pill, SettingRow, Switch, inputCls } from '../console/parts'

// Admin → Settings: the name-recognition pack — a local model that finds names
// in plain sentences ("we met Marko") for every anonymization on this deck.
// It runs here; nothing leaves the machine. It needs transformers + torch and a
// download, so it is off until an admin turns it on.

const STATUS: Record<NamePack['status'], { text: string; tone: 'emerald' | 'amber' | 'zinc' | 'red' }> = {
  off: { text: 'Off', tone: 'zinc' },
  not_installed: { text: 'Not installed', tone: 'amber' },
  not_downloaded: { text: 'Needs a download', tone: 'amber' },
  not_loaded: { text: 'Downloaded, not running', tone: 'amber' },
  downloading: { text: 'Downloading…', tone: 'amber' },
  loading: { text: 'Starting…', tone: 'amber' },
  ready: { text: 'On', tone: 'emerald' },
  error: { text: 'Failed', tone: 'red' },
}

export function NamePackCard() {
  const [pack, setPack] = useState<NamePack | null>(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    try {
      setPack(await anonymizeApi.nerGet())
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't load the name-recognition pack")
    }
  }, [])

  useEffect(() => {
    anonymizeApi.nerGet().then(setPack).catch((e) => setError(e instanceof Error ? e.message : "Couldn't load it"))
  }, [])

  // While it downloads or starts, look again every few seconds.
  useEffect(() => {
    if (!pack || (pack.status !== 'downloading' && pack.status !== 'loading')) return
    const t = setTimeout(load, 3000)
    return () => clearTimeout(t)
  }, [pack, load])

  if (!pack) return error ? <Notice tone="red">{error}</Notice> : null

  const save = async (enabled: boolean, model: string) => {
    setBusy('save'); setError('')
    try {
      setPack(await anonymizeApi.nerPut(enabled, model))
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't save")
    } finally {
      setBusy('')
    }
  }

  const download = async () => {
    setBusy('download'); setError('')
    try {
      setPack(await anonymizeApi.nerDownload())
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't start the download")
    } finally {
      setBusy('')
    }
  }

  const s = STATUS[pack.status]
  return (
    <Card
      icon={ScanFace}
      title="Name recognition"
      hint="A local model that finds names in plain sentences (“we met Marko”) for every anonymization here, alongside the code. It runs on this machine; nothing leaves it."
      action={<Pill tone={s.tone}>{s.text}</Pill>}
      bodyClassName="space-y-4 p-5"
    >
      <div className="divide-y divide-zinc-800/80">
        <SettingRow title="On" hint="Off: code alone looks (and the trusted model, if there is one).">
          <Switch on={pack.enabled} busy={busy === 'save'} label="Name recognition on" onChange={(v) => save(v, pack.model)} />
        </SettingRow>
      </div>
      <Field label="Model" hint="Downloaded once from Hugging Face; only these can be chosen, and only their safetensors weights are loaded.">
        <select value={pack.model} onChange={(e) => save(pack.enabled, e.target.value)} className={inputCls} disabled={busy !== ''}>
          {pack.models.map((m) => <option key={m.id} value={m.id}>{m.id} — {m.label}</option>)}
        </select>
      </Field>
      {pack.status === 'not_installed' && (
        <Notice tone="amber">
          This deck can't run it yet: it needs <code>transformers</code> and <code>torch</code>. Install them where Flight Deck runs
          (<code>pip install transformers torch</code>), then restart Flight Deck. The desktop app and a Raspberry Pi don't have them.
        </Notice>
      )}
      {pack.status === 'error' && <Notice tone="red">{pack.error}</Notice>}
      {error && <Notice tone="red">{error}</Notice>}
      {pack.enabled && (pack.status === 'not_downloaded' || pack.status === 'not_loaded' || pack.status === 'error') && pack.runtime && (
        <div className="flex justify-end">
          <Button variant="primary" icon={Download} busy={busy === 'download'} onClick={download}>
            {pack.downloaded ? 'Start it' : 'Download and start'}
          </Button>
        </div>
      )}
    </Card>
  )
}
