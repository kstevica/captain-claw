/** Kiosk Settings → Agent backups: when your agents are backed up, the backups
 *  kept, and which decks' bundles this one trusts (agent_bundle.backups / signing). */

import { useEffect, useState } from 'react'
import { Archive, Check, Copy, Download, Fingerprint, Loader2, Play, RotateCcw, ShieldCheck, Trash2, TriangleAlert, X } from 'lucide-react'
import { AgentDialog } from './KioskAgentDialogs'
import {
  deleteBackup, downloadBackup, formatBytes, listBackups, restoreBackup, runBackups, saveBackupSettings, saveTrust,
  signingInfo, type BackupRecord, type BackupSettings, type SigningInfo, type Trusted,
} from '../../services/agentBundle'

const field = 'h-10 rounded-xl border border-zinc-800 bg-zinc-900 px-3 text-sm text-zinc-100 outline-none focus:border-violet-500'
const btn = 'inline-flex h-10 items-center justify-center gap-2 rounded-xl px-4 text-sm font-medium transition-colors disabled:opacity-50'
const primary = `${btn} bg-violet-600 text-white hover:bg-violet-500`
const quiet = `${btn} border border-zinc-800 text-zinc-300 hover:bg-zinc-900`
const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

const Title = ({ children }: { children: string }) => (
  <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">{children}</p>
)

function when(ts: number): string {
  return ts ? new Date(ts * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : 'never'
}

export function AgentBackupsDialog({ agents, onClose, onRestored }: {
  /** Your own process agents. */
  agents: { slug: string; name: string }[]
  onClose: () => void
  /** An agent was put back as a backup held it (it restarts if it was running). */
  onRestored?: (slug: string, restarted: boolean) => void
}) {
  const [settings, setSettings] = useState<BackupSettings | null>(null)
  const [saved, setSaved] = useState<BackupSettings | null>(null)
  const [list, setList] = useState<BackupRecord[]>([])
  const [signing, setSigning] = useState<SigningInfo | null>(null)
  const [busy, setBusy] = useState('')
  const [err, setErr] = useState('')
  const [note, setNote] = useState('')
  const [confirm, setConfirm] = useState<string | null>(null)

  const reload = async () => {
    const r = await listBackups()
    setSettings(r.settings)
    setSaved(r.settings)
    setList(r.backups)
  }
  useEffect(() => {
    reload().catch((e) => setErr(String(e instanceof Error ? e.message : e)))
    signingInfo().then(setSigning).catch(() => setSigning(null))
  }, [])

  const act = async (what: string, fn: () => Promise<void>) => {
    setErr('')
    setNote('')
    setBusy(what)
    try { await fn() } catch (e) { setErr(e instanceof Error ? e.message : String(e)) } finally { setBusy('') }
  }
  const set = (patch: Partial<BackupSettings>) => setSettings((s) => (s ? { ...s, ...patch } : s))
  const dirty = !!settings && !!saved && JSON.stringify({ ...settings, last_run: 0, last_error: '' }) !== JSON.stringify({ ...saved, last_run: 0, last_error: '' })
  const allAgents = !settings?.agents.length

  return (
    <AgentDialog title="Agent backups" icon={Archive} agentName="Your agents, on a schedule" size="md" onClose={onClose}>
      <div className="min-h-0 flex-1 space-y-6 overflow-y-auto p-5">
        {!settings && !err && <p className="flex items-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" />Loading…</p>}
        {settings && (
          <section className="space-y-3">
            <Title>Schedule</Title>
            <label className="flex cursor-pointer items-center gap-3">
              <input type="checkbox" className="h-4 w-4 accent-violet-600" checked={settings.enabled} onChange={(e) => set({ enabled: e.target.checked })} />
              <span className="text-sm text-zinc-200">Back up my agents automatically</span>
            </label>
            <div className={`grid gap-3 sm:grid-cols-3 ${settings.enabled ? '' : 'pointer-events-none opacity-50'}`}>
              <label className="block">
                <span className="text-xs text-zinc-500">How often</span>
                <select value={settings.frequency} onChange={(e) => set({ frequency: e.target.value as 'daily' | 'weekly' })} className={`mt-1 w-full ${field}`}>
                  <option value="daily">Every day</option>
                  <option value="weekly">Every week</option>
                </select>
              </label>
              {settings.frequency === 'weekly' && (
                <label className="block">
                  <span className="text-xs text-zinc-500">On</span>
                  <select value={settings.weekday} onChange={(e) => set({ weekday: Number(e.target.value) })} className={`mt-1 w-full ${field}`}>
                    {DAYS.map((d, i) => <option key={d} value={i}>{d}</option>)}
                  </select>
                </label>
              )}
              <label className="block">
                <span className="text-xs text-zinc-500">From</span>
                <select value={settings.hour} onChange={(e) => set({ hour: Number(e.target.value) })} className={`mt-1 w-full ${field}`}>
                  {Array.from({ length: 24 }, (_, h) => <option key={h} value={h}>{String(h).padStart(2, '0')}:00</option>)}
                </select>
              </label>
              <label className="block">
                <span className="text-xs text-zinc-500">Keep per agent</span>
                <input type="number" min={1} max={60} value={settings.keep} onChange={(e) => set({ keep: Math.max(1, Math.min(60, Number(e.target.value) || 1)) })}
                  className={`mt-1 w-full ${field}`} />
              </label>
            </div>
            <div className={settings.enabled ? '' : 'pointer-events-none opacity-50'}>
              <span className="text-xs text-zinc-500">Which agents</span>
              <div className="mt-1 flex flex-wrap gap-1.5">
                <button onClick={() => set({ agents: [] })}
                  className={`rounded-full border px-3 py-1 text-xs ${allAgents ? 'border-violet-500 bg-violet-500/10 text-zinc-100' : 'border-zinc-800 text-zinc-400'}`}>All of them</button>
                {agents.map((a) => {
                  const on = settings.agents.includes(a.slug)
                  return (
                    <button key={a.slug} onClick={() => set({ agents: on ? settings.agents.filter((x) => x !== a.slug) : [...settings.agents, a.slug] })}
                      className={`inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs ${on ? 'border-violet-500 bg-violet-500/10 text-zinc-100' : 'border-zinc-800 text-zinc-400'}`}>
                      {on && <Check className="h-3 w-3" />}{a.name}
                    </button>
                  )
                })}
              </div>
            </div>
            <label className={`block ${settings.enabled ? '' : 'pointer-events-none opacity-50'}`}>
              <span className="text-xs text-zinc-500">Also copy to a folder in your files (optional)</span>
              <input value={settings.folder} onChange={(e) => set({ folder: e.target.value })} maxLength={300}
                placeholder="e.g. backups/agents — a linked folder can be a synced disk" className={`mt-1 w-full ${field}`} />
            </label>
            <p className="text-xs text-zinc-500">
              Last scheduled run: {when(settings.last_run)}. Backups are full, without keys; the agents keep running while they're taken.
            </p>
            {settings.last_error && (
              <p className="flex gap-2 text-xs text-amber-700 dark:text-amber-300"><TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />{settings.last_error}</p>
            )}
            <div className="flex flex-wrap gap-2">
              <button className={primary} disabled={!dirty || !!busy}
                onClick={() => act('save', async () => { const s = await saveBackupSettings(settings); setSettings(s); setSaved(s); setNote('Saved.') })}>
                {busy === 'save' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}Save
              </button>
              <button className={quiet} disabled={!!busy}
                onClick={() => act('run', async () => {
                  const r = await runBackups(settings.agents.length ? settings.agents : undefined)
                  await reload()
                  setNote(`${r.made.length} backup${r.made.length === 1 ? '' : 's'} made.${r.problems.length ? ` ${r.problems.join(' ')}` : ''}`)
                })}>
                {busy === 'run' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}Back up now
              </button>
            </div>
          </section>
        )}

        {settings && (
          <section className="space-y-2">
            <Title>Backups kept</Title>
            {list.length === 0 && <p className="text-sm text-zinc-500">None yet.</p>}
            {list.length > 0 && (
              <ul className="divide-y divide-zinc-800 rounded-xl border border-zinc-800">
                {list.map((b) => (
                  <li key={b.id} className="flex flex-wrap items-center gap-2 px-3 py-2">
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm text-zinc-200">{b.name}</span>
                      <span className="block text-xs text-zinc-500">{when(b.made)} · {formatBytes(b.size)}</span>
                    </span>
                    {confirm === b.id ? (
                      <span className="flex items-center gap-2 text-xs">
                        <span className="text-zinc-400">Put it back as it was then?</span>
                        <button className="font-medium text-red-700 hover:underline dark:text-red-300" disabled={!!busy}
                          onClick={() => act(`restore:${b.id}`, async () => {
                            const r = await restoreBackup(b.id)
                            setConfirm(null)
                            setNote(`${r.name} is back as it was on ${when(b.made)}. Its state just before was backed up too (kept a week).`)
                            onRestored?.(r.slug, r.restarted)
                          })}>{busy === `restore:${b.id}` ? 'Restoring…' : 'Yes, restore'}</button>
                        <button className="text-zinc-500 hover:text-zinc-300" onClick={() => setConfirm(null)} aria-label="Cancel"><X className="h-3.5 w-3.5" /></button>
                      </span>
                    ) : (
                      <span className="flex items-center gap-1">
                        <button title="Download" aria-label={`Download ${b.file_name}`} className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                          onClick={() => act(`dl:${b.id}`, () => downloadBackup(b))}><Download className="h-4 w-4" /></button>
                        <button title="Restore it" aria-label={`Restore ${b.name} from this backup`} className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                          onClick={() => setConfirm(b.id)}><RotateCcw className="h-4 w-4" /></button>
                        <button title="Delete" aria-label={`Delete ${b.file_name}`} className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-red-400"
                          onClick={() => act(`del:${b.id}`, async () => { await deleteBackup(b.id); setList((l) => l.filter((x) => x.id !== b.id)) })}><Trash2 className="h-4 w-4" /></button>
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}

        {signing && <SigningSection info={signing} onSaved={setSigning} />}
        {note && <p className="flex gap-2 text-sm text-emerald-700 dark:text-emerald-300"><Check className="mt-0.5 h-4 w-4 shrink-0" />{note}</p>}
        {err && <p role="alert" className="rounded-xl border border-red-500/30 bg-red-500/5 p-3 text-sm text-red-700 dark:text-red-300">{err}</p>}
      </div>
      <div className="flex shrink-0 justify-end border-t border-zinc-800 p-3">
        <button className={quiet} onClick={onClose}>Close</button>
      </div>
    </AgentDialog>
  )
}

/** The console's Settings → Agent files: this deck's fingerprint and the decks it trusts. */
export function AgentFilesSigning() {
  const [info, setInfo] = useState<SigningInfo | null>(null)
  useEffect(() => { signingInfo().then(setInfo).catch(() => setInfo(null)) }, [])
  // The console's card already says "Agent files": no second heading inside it.
  return info ? <SigningSection info={info} onSaved={setInfo} untitled /> : null
}

function SigningSection({ info, onSaved, untitled = false }: {
  info: SigningInfo; onSaved: (i: SigningInfo) => void; untitled?: boolean
}) {
  const [trusted, setTrusted] = useState<Trusted[]>(info.trusted)
  const [only, setOnly] = useState(info.require_trusted)
  const [fp, setFp] = useState('')
  const [label, setLabel] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [copied, setCopied] = useState(false)
  const dirty = only !== info.require_trusted || JSON.stringify(trusted) !== JSON.stringify(info.trusted)

  const save = async () => {
    setBusy(true)
    setErr('')
    try {
      const r = await saveTrust(trusted, only)
      setTrusted(r.trusted)
      onSaved({ ...info, trusted: r.trusted, require_trusted: r.require_trusted })
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="space-y-2">
      {!untitled && <Title>Signed files</Title>}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
        <p className="flex items-center gap-2 text-xs text-zinc-400"><Fingerprint className="h-4 w-4 text-violet-500" />
          {info.deck} signs every file it exports. Its fingerprint — give it to whoever imports your agents:</p>
        <div className="mt-2 flex items-center gap-2">
          <code className="min-w-0 flex-1 truncate rounded-lg bg-zinc-950 px-2 py-1.5 text-xs text-zinc-200">{info.fingerprint}</code>
          <button className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" aria-label="Copy the fingerprint"
            onClick={() => { void navigator.clipboard?.writeText(info.fingerprint); setCopied(true); setTimeout(() => setCopied(false), 1500) }}>
            {copied ? <Check className="h-4 w-4 text-emerald-500" /> : <Copy className="h-4 w-4" />}
          </button>
        </div>
      </div>
      {info.admin && (
        <div className="space-y-2 rounded-xl border border-zinc-800 p-3">
          <p className="flex items-center gap-2 text-xs font-medium text-zinc-300"><ShieldCheck className="h-4 w-4 text-emerald-500" />Decks this one trusts</p>
          {trusted.length === 0 && <p className="text-xs text-zinc-500">None besides this deck itself.</p>}
          <ul className="space-y-1">
            {trusted.map((t) => (
              <li key={t.fingerprint} className="flex items-center gap-2 text-xs">
                <span className="w-32 shrink-0 truncate text-zinc-300">{t.label || 'Unnamed'}</span>
                <code className="min-w-0 flex-1 truncate text-zinc-500">{t.fingerprint}</code>
                <button className="rounded p-1 text-zinc-500 hover:text-red-400" aria-label={`Stop trusting ${t.label || t.fingerprint}`}
                  onClick={() => setTrusted((l) => l.filter((x) => x.fingerprint !== t.fingerprint))}><X className="h-3.5 w-3.5" /></button>
              </li>
            ))}
          </ul>
          <div className="grid gap-2 sm:grid-cols-[1fr_2fr_auto]">
            <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Name, e.g. the consultant" maxLength={80} className={field} />
            <input value={fp} onChange={(e) => setFp(e.target.value)} placeholder="Fingerprint" className={field} />
            <button className={quiet} disabled={fp.replace(/[^0-9a-f]/gi, '').length !== 32}
              onClick={() => { setTrusted((l) => [...l, { fingerprint: fp.trim(), label: label.trim() }]); setFp(''); setLabel('') }}>Add</button>
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-xs text-zinc-300">
            <input type="checkbox" className="h-4 w-4 accent-violet-600" checked={only} onChange={(e) => setOnly(e.target.checked)} />
            Only take files signed by this deck or a trusted one
          </label>
          {err && <p className="text-xs text-red-600 dark:text-red-400">{err}</p>}
          <button className={primary} onClick={save} disabled={!dirty || busy}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}Save
          </button>
        </div>
      )}
    </section>
  )
}
