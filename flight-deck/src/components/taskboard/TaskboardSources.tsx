import { useEffect, useState } from 'react'
import { Check, History, Layers, Loader2, Mail, MessageCircle, Plug, Send, ShieldCheck, X } from 'lucide-react'
import type { TabId as ConnectionsTab } from '../../pages/ConnectionsPage'
import { fmtWhen } from '../../services/schedulerApi'
import {
  PASSIVE_KEYS, taskboardApi, type PassiveKey, type PassiveSource, type TaskboardBackfill, type TaskboardSettings,
  type TaskboardStatus,
} from '../../services/taskboardApi'
import { libraryIconBtn } from '../library/libraryParts'
import { BoardSwitch } from './taskboardParts'

// The Sources sheet: what the Taskboard reads, whether it does yet, and what
// it can't. Your own inboxes (Telegram, WhatsApp, Gmail) are each off until
// you switch one on here, having read what it means: what is read, that it
// is kept 30 days, that switching off deletes the tasks it made, and that
// they stay yours. History is the last 30 days of your agents' chats, filled
// in once.

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))

const PASSIVE: Record<PassiveKey, {
  label: string; short: string; Icon: typeof Mail; tone: string; reads: string
  /** Its key among the status sources (an older deck names Gmail's otherwise). */
  keys: string[]
  connections: ConnectionsTab
}> = {
  telegram_inbox: {
    label: 'Your Telegram inbox', short: 'Telegram inbox', Icon: Send, tone: 'text-sky-600 dark:text-sky-400',
    reads: 'The chats your Telegram inbox keeps (Connections → Telegram → Inbox): what people write to you there and what you write back.',
    keys: ['telegram_inbox'], connections: 'telegram',
  },
  whatsapp_web: {
    label: 'Your WhatsApp chats', short: 'WhatsApp chats', Icon: MessageCircle, tone: 'text-emerald-600 dark:text-emerald-400',
    reads: 'Every chat of the WhatsApp you linked under Connections → WhatsApp Web: what people write to you and what you write back.',
    keys: ['whatsapp_web'], connections: 'whatsapp_web',
  },
  gmail: {
    label: 'Your Gmail inbox', short: 'Gmail inbox', Icon: Mail, tone: 'text-rose-600 dark:text-rose-400',
    reads: 'The email in the Gmail inbox you connected under Connections → Google: who wrote, the subject and the text.',
    keys: ['gmail', 'gmail_inbox'], connections: 'google',
  },
}
const PASSIVE_SOURCE_KEYS = new Set(PASSIVE_KEYS.flatMap((k) => PASSIVE[k].keys))

export function SourcesSheet({ status, phone, onClose, onChanged, onOpenConnections }: {
  status: TaskboardStatus | null
  phone: boolean
  onClose: () => void
  /** Something switched (tasks may be gone, or coming): the board reads again. */
  onChanged: () => void
  onOpenConnections?: (tab: ConnectionsTab) => void
}) {
  const [settings, setSettings] = useState<TaskboardSettings | null>(null)
  const [settingsError, setSettingsError] = useState('')

  useEffect(() => {
    let gone = false
    taskboardApi.settings()
      .then((s) => { if (!gone) { setSettings(s); setSettingsError('') } })
      .catch((e) => { if (!gone) setSettingsError(errText(e)) })
    return () => { gone = true }
  }, [])

  const passive = settings?.passive
  // Your own inboxes have switches of their own (once the settings say how they are).
  const sources = (status?.sources ?? []).filter((s) => s.key !== 'not_covered' && !(passive && PASSIVE_SOURCE_KEYS.has(s.key)))
  const uncovered = status?.sources?.find((s) => s.key === 'not_covered')
  return (
    <div className="absolute inset-0 z-30 flex items-stretch justify-center bg-black/50 wide:items-center wide:p-6" onClick={onClose}>
      <div
        role="dialog"
        aria-label="Sources"
        onClick={(e) => e.stopPropagation()}
        className="flex w-full flex-col overflow-hidden bg-zinc-950 shadow-2xl wide:max-h-full wide:max-w-lg wide:rounded-xl wide:border wide:border-zinc-800"
      >
        <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-500/15 text-teal-600 dark:text-teal-400">
            <Layers className="h-4 w-4" />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-base font-semibold text-zinc-100">Sources</h3>
            <p className="truncate text-xs text-zinc-500">Where the Taskboard finds its tasks</p>
          </div>
          <button onClick={onClose} className={libraryIconBtn} title="Close" aria-label="Close sources"><X className="h-4 w-4" /></button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          <section className="border-b border-zinc-800 px-4 py-3">
            <h4 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">Your own inboxes</h4>
            <p className="mt-1 text-xs leading-relaxed text-zinc-500">
              Off until you switch one on. What they find is yours alone — nobody else on this deck sees it.
            </p>
            {passive ? (
              <ul className="mt-2 flex flex-col divide-y divide-zinc-800/70">
                {PASSIVE_KEYS.map((key) => (
                  <PassiveRow
                    key={key}
                    id={key}
                    source={passive[key] ?? { on: false, consented_at: null, available: false, note: 'Not on this deck yet.' }}
                    phone={phone}
                    onSaved={(s) => { setSettings(s); onChanged() }}
                    onOpenConnections={onOpenConnections}
                  />
                ))}
              </ul>
            ) : settingsError ? (
              <p className="mt-2 text-xs text-red-600 dark:text-red-400">{settingsError}</p>
            ) : (
              <div className="flex justify-center py-4 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>
            )}
          </section>
          {!status ? (
            <div className="flex justify-center py-8 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>
          ) : sources.length === 0 ? (
            <p className="px-4 py-6 text-center text-sm text-zinc-500">No sources yet.</p>
          ) : (
            <ul className="divide-y divide-zinc-800/70">
              {sources.map((s) => {
                const fill = s.key === 'history' ? status.backfill : undefined
                const on = s.tracked && s.on
                const state = fill ? backfillState(fill) : s.tracked ? (s.on ? 'Tracked' : 'Off') : 'Not tracked yet'
                const lit = fill ? fill.state === 'done' || fill.state === 'running' : on
                return (
                  <li key={s.key} className="flex items-start gap-3 px-4 py-3">
                    <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${lit ? 'bg-emerald-500' : s.tracked ? 'bg-zinc-500' : 'bg-zinc-700'}`} />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-zinc-200">{s.label}</div>
                      {s.note && <div className="mt-0.5 text-xs leading-relaxed text-zinc-500">{s.note}</div>}
                      {fill && <BackfillLine fill={fill} onStarted={onChanged} />}
                    </div>
                    <span className={`shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-medium ${
                      lit ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300' : 'bg-zinc-800 text-zinc-400'}`}>
                      {state}
                    </span>
                  </li>
                )
              })}
            </ul>
          )}
          {uncovered?.note && (
            <p className="border-t border-zinc-800 px-4 py-3 text-[11px] leading-relaxed text-zinc-500">
              <span className="font-medium text-zinc-400">{uncovered.label || 'Not tracked'}:</span> {uncovered.note}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function backfillState(fill: TaskboardBackfill): string {
  if (fill.state === 'running') return 'Reading…'
  if (fill.state === 'done') return 'Filled in'
  if (fill.state === 'error') return 'Stopped'
  return 'Not yet'
}

/** The history fill under its row: how far it got, when it finished, or
 *  why it stopped — and a way to start it (once) or try again. */
function BackfillLine({ fill, onStarted }: { fill: TaskboardBackfill; onStarted: () => void }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const start = async () => {
    setBusy(true)
    setError('')
    try {
      await taskboardApi.backfill()
      onStarted()
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(false)
    }
  }
  const { agents_done: done = 0, agents_total: total = 0, tasks = 0 } = fill.progress ?? {}
  const found = `${tasks} task${tasks === 1 ? '' : 's'}`
  const line = fill.state === 'running'
    ? `Reading the last 30 days…${total ? ` ${Math.min(done, total)} of ${total} agent${total === 1 ? '' : 's'}` : ''}${tasks ? ` · ${found} so far` : ''}`
    : fill.state === 'done'
      ? `Filled in${fill.done_at ? ` ${fmtWhen(fill.done_at)}` : ''}${tasks ? ` · ${found}` : ''} — marked “From history”, and never given a next step.`
      : fill.state === 'error'
        ? fill.message || 'It stopped before it finished.'
        : 'Your agents\' chats of the last 30 days are read once, the first time the board opens.'
  const canStart = fill.state === 'error' || (fill.state === 'idle' && !fill.done_at)
  return (
    <div className="mt-1">
      <p className={`flex items-start gap-1 text-[11px] leading-relaxed ${fill.state === 'error' ? 'text-amber-700 dark:text-amber-400' : 'text-zinc-500'}`}>
        {fill.state === 'running'
          ? <Loader2 className="mt-0.5 h-3 w-3 shrink-0 animate-spin text-teal-600 dark:text-teal-400" />
          : <History className="mt-0.5 h-3 w-3 shrink-0" />}
        <span className="min-w-0">{line}</span>
      </p>
      {canStart && (
        <button onClick={start} disabled={busy} className="mt-1 flex items-center gap-1 text-[11px] font-medium text-teal-600 hover:underline disabled:opacity-60 dark:text-teal-400">
          {busy && <Loader2 className="h-3 w-3 animate-spin" />}{fill.state === 'error' ? 'Try again' : 'Fill it in now'}
        </button>
      )}
      {error && <p className="mt-0.5 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}

/** One of your own inboxes: its switch, and before it goes on, what it
 *  means (you agree first); before it goes off, that its tasks go with it. */
function PassiveRow({ id, source, phone, onSaved, onOpenConnections }: {
  id: PassiveKey
  source: PassiveSource
  phone: boolean
  onSaved: (s: TaskboardSettings) => void
  onOpenConnections?: (tab: ConnectionsTab) => void
}) {
  const info = PASSIVE[id]
  const [asking, setAsking] = useState<'on' | 'off' | null>(null)
  const [agreed, setAgreed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [done, setDone] = useState('')

  // Escape closes the question first, not the sheet (it marks the key used).
  useEffect(() => {
    if (!asking) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); setAsking(null) } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [asking])

  const ask = (on: boolean) => {
    setAsking(on ? 'on' : 'off')
    setAgreed(false)
    setError('')
    setDone('')
  }
  const apply = async (on: boolean) => {
    setBusy(true)
    setError('')
    try {
      const { settings, deleted } = await taskboardApi.setPassive(id, on)
      setAsking(null)
      setDone(on
        ? 'Switched on. What it finds shows up here as tasks.'
        : `Switched off. ${deleted === 0 ? 'No task came only from it.' : `${deleted} task${deleted === 1 ? '' : 's'} deleted.`}`)
      onSaved(settings)
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy(false)
    }
  }

  const disabled = !source.available && !source.on
  const btn = `flex items-center justify-center gap-1.5 rounded-lg font-medium disabled:opacity-50 ${phone ? 'h-11 flex-1 px-3 text-sm' : 'px-3 py-1.5 text-xs'}`
  return (
    <li className="py-3">
      <div className="flex items-start gap-3">
        <info.Icon className={`mt-0.5 h-4 w-4 shrink-0 ${info.tone}`} />
        <div className="min-w-0 flex-1">
          <div className="text-sm text-zinc-200">{info.label}</div>
          <p className="mt-0.5 text-xs leading-relaxed text-zinc-500">{info.reads}</p>
          {disabled && (
            <p className="mt-1 text-[11px] leading-relaxed text-amber-700 dark:text-amber-400">
              {source.note || 'It can\'t be switched on here yet.'}
              {onOpenConnections && (
                <button onClick={() => onOpenConnections(info.connections)} className="ml-1 inline-flex items-center gap-0.5 font-medium text-teal-600 hover:underline dark:text-teal-400">
                  <Plug className="h-3 w-3" />Connections
                </button>
              )}
            </p>
          )}
          {source.on && source.consented_at && !asking && (
            <p className="mt-1 text-[11px] text-zinc-500">On since {fmtWhen(source.consented_at)} · text kept 30 days</p>
          )}
        </div>
        <BoardSwitch
          on={source.on}
          onChange={(on) => ask(on)}
          disabled={busy || disabled || asking !== null}
          label={`Read ${info.short} for the Taskboard`}
        />
      </div>

      {asking === 'on' && (
        <div className="ml-7 mt-2.5 rounded-lg border border-teal-500/30 bg-teal-500/5 px-3 py-2.5">
          <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-200">
            <ShieldCheck className="h-3.5 w-3.5 shrink-0 text-teal-600 dark:text-teal-400" />Before you switch it on
          </div>
          <ul className="mt-1.5 flex list-disc flex-col gap-1 pl-4 text-[11px] leading-relaxed text-zinc-400">
            <li><span className="font-medium text-zinc-300">What is read:</span> {info.reads}</li>
            <li>What it reads is kept 30 days, then deleted — the tasks made from it with it.</li>
            <li>Switching it off deletes every task that came only from it, at once.</li>
            <li>It stays yours: only you see these tasks — not the deck's admin, and not the owner of an agent you use.</li>
            <li>Nothing goes out from it on your behalf: a next step waits for your Approve, and email stays a draft.</li>
          </ul>
          <label className="mt-2 flex cursor-pointer items-start gap-2 text-xs text-zinc-300">
            <input
              type="checkbox"
              checked={agreed}
              onChange={(e) => setAgreed(e.target.checked)}
              className={`mt-0.5 shrink-0 accent-teal-600 ${phone ? 'h-5 w-5' : 'h-3.5 w-3.5'}`}
            />
            I agree: Captain may read my {info.short} for the Taskboard.
          </label>
          <div className="mt-2 flex gap-2">
            <button onClick={() => setAsking(null)} disabled={busy} className={`${btn} border border-zinc-800 text-zinc-300 hover:bg-zinc-800/60`}>Cancel</button>
            <button onClick={() => apply(true)} disabled={busy || !agreed} className={`${btn} bg-teal-600 text-white hover:bg-teal-500`}>
              {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}Switch on
            </button>
          </div>
        </div>
      )}

      {asking === 'off' && (
        <div role="alertdialog" aria-label={`Switch off ${info.short}`} className="ml-7 mt-2.5 rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2.5">
          <div className="text-xs font-medium text-red-700 dark:text-red-300">Switch off {info.short}?</div>
          <p className="mt-1 text-[11px] leading-relaxed text-zinc-400">
            Every task that came only from it is deleted now, and that can't be undone. Tasks that also came in
            some other way stay.
          </p>
          <div className="mt-2 flex gap-2">
            <button onClick={() => setAsking(null)} disabled={busy} className={`${btn} border border-zinc-800 text-zinc-300 hover:bg-zinc-800/60`}>Keep it on</button>
            <button onClick={() => apply(false)} disabled={busy} className={`${btn} bg-red-600 text-white hover:bg-red-500`}>
              {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}Switch off and delete
            </button>
          </div>
        </div>
      )}

      {done && !asking && (
        <p className="ml-7 mt-1.5 flex items-center gap-1 text-[11px] text-emerald-700 dark:text-emerald-400">
          <Check className="h-3 w-3 shrink-0" />{done}
        </p>
      )}
      {error && <p className="ml-7 mt-1.5 text-[11px] text-red-600 dark:text-red-400">{error}</p>}
    </li>
  )
}
