import { useEffect, useState } from 'react'
import { Check, Loader2, Plug, Settings2, X } from 'lucide-react'
import {
  EXTRACT_PER_DAY_MAX, JEV_PER_DAY_MAX, taskboardApi, type TaskboardSettings, type TaskboardSettingsPatch,
} from '../../services/taskboardApi'
import { libraryIconBtn } from '../library/libraryParts'
import { BoardSwitch } from './taskboardParts'

// The board's own settings: whether Jev may close a task by itself, how
// many Jev decisions and summaries a day it may spend, what today used, and
// whether Jev is connected at all (when it isn't, Connect Jev opens
// Connections on Jev — anyone can add their own key there). A popover under
// the header's gear on a desktop, a full-screen sheet on a phone (like the
// filters). Nothing changes until Save.

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))

interface Form { autoClose: boolean; jev: string; extract: string }

function formOf(s: TaskboardSettings): Form {
  return { autoClose: !!s.auto_close, jev: String(s.jev_per_day ?? 0), extract: String(s.extract_per_day ?? 0) }
}

/** A whole number within 0..max, or null. */
function limitOf(v: string, max: number): number | null {
  const n = Number(v)
  return v.trim() !== '' && Number.isInteger(n) && n >= 0 && n <= max ? n : null
}

/** Today's counts — none when the server's day is gone (its UTC day, or
 *  the local one near midnight). */
function usedToday(c: TaskboardSettings['counters']): { jev: number; extract: number } | null {
  if (!c) return null
  const now = new Date()
  const pad = (n: number) => String(n).padStart(2, '0')
  const local = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
  const current = !c.day || c.day === now.toISOString().slice(0, 10) || c.day === local
  return current ? { jev: Number(c.jev) || 0, extract: Number(c.extract) || 0 } : { jev: 0, extract: 0 }
}

/** Whether Jev is connected, and with whose key; when it isn't, the way to
 *  Connections → Jev (`onConnect`, when the board can open it). */
export function JevSection({ jev, phone, onConnect }: {
  jev: NonNullable<TaskboardSettings['jev']>; phone: boolean; onConnect?: () => void
}) {
  return (
    <section>
      <h4 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">Jev</h4>
      {jev.connected ? (
        <p className="mt-1.5 flex items-center gap-1.5 text-sm text-zinc-200">
          <span className="h-2 w-2 shrink-0 rounded-full bg-emerald-500" />
          Connected{jev.source === 'own' ? ' with your own key' : jev.source === 'team' ? ' with the team key' : ''}
        </p>
      ) : (
        <>
          <p className="mt-1.5 flex items-center gap-1.5 text-sm text-zinc-300">
            <span className="h-2 w-2 shrink-0 rounded-full bg-zinc-600" />
            {onConnect ? 'Not connected' : 'Not connected — connect Jev in Connections'}
          </p>
          <p className="mt-0.5 text-[11px] leading-relaxed text-zinc-500">
            Until then, simple rules sort your tasks.{onConnect ? ' Your own Jev key works too.' : ''}
          </p>
          {onConnect && (
            <button
              type="button"
              onClick={onConnect}
              className={`mt-2 flex items-center gap-1.5 rounded-lg border border-zinc-800 font-medium text-teal-600 transition-colors hover:bg-teal-500/10 dark:text-teal-400 ${
                phone ? 'h-11 px-3.5 text-sm' : 'px-2.5 py-1.5 text-xs'}`}
            >
              <Plug className={phone ? 'h-4 w-4' : 'h-3.5 w-3.5'} />Connect Jev
            </button>
          )}
        </>
      )}
    </section>
  )
}

export function BoardSettings({ phone, onClose, onConnectJev }: {
  phone: boolean; onClose: () => void
  /** Connections → Jev (the board closes its settings first). */
  onConnectJev?: () => void
}) {
  const [saved, setSaved] = useState<TaskboardSettings | null>(null)
  const [form, setForm] = useState<Form | null>(null)
  const [loadError, setLoadError] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [justSaved, setJustSaved] = useState(false)

  useEffect(() => {
    let gone = false
    taskboardApi.settings()
      .then((s) => { if (!gone) { setSaved(s); setForm(formOf(s)) } })
      .catch((e) => { if (!gone) setLoadError(errText(e)) })
    return () => { gone = true }
  }, [])

  const jev = form ? limitOf(form.jev, JEV_PER_DAY_MAX) : null
  const extract = form ? limitOf(form.extract, EXTRACT_PER_DAY_MAX) : null
  const patch: TaskboardSettingsPatch = {}
  if (form && saved) {
    if (form.autoClose !== !!saved.auto_close) patch.auto_close = form.autoClose
    if (jev !== null && jev !== saved.jev_per_day) patch.jev_per_day = jev
    if (extract !== null && extract !== saved.extract_per_day) patch.extract_per_day = extract
  }
  const invalid = !!form && (jev === null || extract === null)
  const changed = Object.keys(patch).length > 0
  const used = usedToday(saved?.counters)

  const edit = (next: Partial<Form>) => {
    setForm((cur) => cur && { ...cur, ...next })
    setJustSaved(false)
    setError('')
  }
  const save = async () => {
    if (invalid || !changed || saving) return
    setSaving(true)
    setError('')
    try {
      const s = await taskboardApi.saveSettings(patch)
      setSaved(s)
      setForm(formOf(s))
      setJustSaved(true)
    } catch (e) {
      setError(errText(e))
    } finally {
      setSaving(false)
    }
  }

  const input = `w-24 shrink-0 rounded-lg border bg-zinc-900 px-2.5 text-right tabular-nums text-zinc-200 focus:outline-none ${
    phone ? 'h-11 text-base' : 'py-1.5 text-sm'}`
  const limitRow = (label: string, value: string, max: number, ok: boolean, count: number | undefined, onChange: (v: string) => void) => (
    <label className="flex items-center gap-3">
      <span className="min-w-0 flex-1">
        <span className="block text-sm text-zinc-200">{label}</span>
        <span className={`block text-[11px] ${ok ? 'text-zinc-500' : 'text-red-600 dark:text-red-400'}`}>
          {ok ? (count !== undefined ? `${count} used today` : `Up to ${max.toLocaleString()} a day`) : `A whole number from 0 to ${max.toLocaleString()}`}
        </span>
      </span>
      <input
        type="number"
        inputMode="numeric"
        min={0}
        max={max}
        step={1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label={label}
        aria-invalid={!ok}
        className={`${input} ${ok ? 'border-zinc-800 focus:border-teal-500/60' : 'border-red-500/60'}`}
      />
    </label>
  )

  const jevState = saved?.jev
  const body = !form ? (
    loadError
      ? <p className="px-4 py-6 text-center text-sm text-red-600 dark:text-red-400">{loadError}</p>
      : <div className="flex justify-center py-8 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>
  ) : (
    <div className="flex flex-col gap-5 px-4 py-4">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="text-sm text-zinc-200">Close tasks automatically when Jev is sure</div>
          <p className="mt-0.5 text-[11px] leading-relaxed text-zinc-500">
            Undo brings one back. Off, Jev never closes a task by itself — you do.
          </p>
        </div>
        <BoardSwitch
          on={form.autoClose}
          onChange={(on) => edit({ autoClose: on })}
          disabled={saving}
          label="Close tasks automatically when Jev is sure"
        />
      </div>

      <section className="flex flex-col gap-3">
        <h4 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">Daily limits</h4>
        {limitRow('Jev decisions', form.jev, JEV_PER_DAY_MAX, jev !== null, used?.jev, (v) => edit({ jev: v }))}
        {limitRow('Summaries', form.extract, EXTRACT_PER_DAY_MAX, extract !== null, used?.extract, (v) => edit({ extract: v }))}
        <p className="text-[11px] leading-relaxed text-zinc-500">
          0 means none. Once Jev's limit is reached, simple rules sort new tasks until the next day.
        </p>
      </section>

      {jevState && <JevSection jev={jevState} phone={phone} onConnect={onConnectJev} />}
    </div>
  )

  const status = error ? (
    <p className="text-[11px] text-red-600 dark:text-red-400">{error}</p>
  ) : justSaved && !changed ? (
    <p className="flex items-center gap-1 text-[11px] text-emerald-700 dark:text-emerald-400"><Check className="h-3 w-3" />Saved</p>
  ) : null

  if (phone) {
    return (
      <form onSubmit={(e) => { e.preventDefault(); void save() }} className="flex h-full flex-col">
        <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <h3 className="flex-1 text-base font-semibold text-zinc-100">Board settings</h3>
          <button type="button" onClick={onClose} className={libraryIconBtn} aria-label="Close settings"><X className="h-5 w-5" /></button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">{body}</div>
        <div className="border-t border-zinc-800 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
          {status && <div className="mb-2">{status}</div>}
          <div className="flex gap-2">
            <button type="button" onClick={onClose} className="h-11 flex-1 rounded-xl border border-zinc-800 text-sm font-medium text-zinc-300">
              Close
            </button>
            <button
              type="submit"
              disabled={!changed || invalid || saving}
              className="flex h-11 flex-1 items-center justify-center gap-1.5 rounded-xl bg-teal-600 text-sm font-medium text-white disabled:opacity-50"
            >
              {saving && <Loader2 className="h-4 w-4 animate-spin" />}Save
            </button>
          </div>
        </div>
      </form>
    )
  }

  return (
    <form
      onSubmit={(e) => { e.preventDefault(); void save() }}
      role="dialog"
      aria-label="Board settings"
      className="flex max-h-[70vh] flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl shadow-black/30"
    >
      <div className="flex items-center gap-2.5 border-b border-zinc-800 px-4 py-2.5">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-teal-500/15 text-teal-600 dark:text-teal-400">
          <Settings2 className="h-3.5 w-3.5" />
        </span>
        <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-zinc-100">Board settings</h3>
        <button type="button" onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" aria-label="Close settings">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">{body}</div>
      <div className="flex items-center gap-2 border-t border-zinc-800 px-4 py-2.5">
        <div className="min-w-0 flex-1">{status}</div>
        <button
          type="submit"
          disabled={!changed || invalid || saving}
          className="flex shrink-0 items-center gap-1.5 rounded-lg bg-teal-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-teal-500 disabled:opacity-50"
        >
          {saving && <Loader2 className="h-3.5 w-3.5 animate-spin" />}Save
        </button>
      </div>
    </form>
  )
}
