import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, ChevronDown, ChevronRight, Download, Eye, FileText, Loader2, Play, Plus, Trash2 } from 'lucide-react'
import {
  addRecordingData, deleteRecording, dropRecordingData, getRecording, listRecordings, recordingDataUrl,
  type RecordingDetail, type RecordingRun, type RecordingSummary,
} from '../../services/browserClip'
import { AddDataPanel } from '../context/AddData'
import { sizeText } from '../../utils/format'

// Connections → Browser → Recordings: what the person recorded in Captain for
// Chrome. Each one opens to its steps, the extra data every run reads first
// (added here from uploads, pasted text, links, agents' files, Flight Deck
// files or a conversation), and how its runs went. Recording and running
// happen in the extension, in the browser the agent works in.

const STATUS: Record<string, [string, string]> = {
  done: ['Done', 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30'],
  thinking: ['Working…', 'bg-violet-500/15 text-violet-700 dark:text-violet-300 border-violet-500/30'],
  starting: ['Starting…', 'bg-violet-500/15 text-violet-700 dark:text-violet-300 border-violet-500/30'],
  error: ['Went wrong', 'bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30'],
  stopped: ['Stopped', 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30'],
  lost: ['Ended — answer in the agent’s chat', 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30'],
}

function when(ts: string): string {
  if (!ts) return ''
  const d = new Date(ts)
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function took(run: RecordingRun): string {
  const s = (new Date(run.ended_at).getTime() - new Date(run.started_at).getTime()) / 1000
  if (!run.ended_at || Number.isNaN(s) || s < 0) return ''
  return s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`
}

const errorOf = (e: unknown) => (e instanceof Error ? e.message : String(e))

function RunRow({ run, showName }: { run: RecordingRun; showName?: boolean }) {
  const [open, setOpen] = useState(false)
  const [label, tone] = STATUS[run.status] || [run.status, STATUS.stopped[1]]
  const given = [...run.inputs.map((i) => `${i.label}: ${i.value}`), run.note].filter(Boolean).join(' · ')
  return (
    <li className="px-3 py-2">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 text-left" aria-expanded={open}>
        {open ? <ChevronDown className="h-3.5 w-3.5 shrink-0 text-zinc-500" /> : <ChevronRight className="h-3.5 w-3.5 shrink-0 text-zinc-500" />}
        <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] ${tone}`}>{label}</span>
        <span className="min-w-0 flex-1 truncate text-xs text-zinc-300">
          {showName && <span className="font-medium text-zinc-200">{run.name} · </span>}
          {run.agent} · {when(run.started_at)}{took(run) ? ` · ${took(run)}` : ''}
          {given && <span className="text-zinc-500"> · {given}</span>}
        </span>
      </button>
      {open && (
        <div className="mt-2 space-y-2 pl-5 text-xs">
          {run.error && <p className="text-red-600 dark:text-red-400">{run.error}</p>}
          {run.result
            ? <div className="max-h-72 overflow-auto whitespace-pre-wrap rounded-md border border-zinc-800 bg-zinc-950/40 px-3 py-2 text-zinc-300">{run.result}</div>
            : <p className="text-zinc-500">{run.status === 'thinking' ? 'No answer yet.' : 'No answer was kept: look in the agent’s chat.'}</p>}
          {!!run.steps_done?.length && (
            <details>
              <summary className="cursor-pointer text-zinc-500">What it did ({run.steps_done.length})</summary>
              <ul className="mt-1 space-y-0.5 text-zinc-400">{run.steps_done.map((t, i) => <li key={i} className="truncate">· {t}</li>)}</ul>
            </details>
          )}
          <p className="text-zinc-500">
            From {run.browser || 'the browser'} · its folder: <span className="font-mono text-[11px] text-zinc-400">{run.vfs}</span>
            {run.extra.length > 0 && <> · read {run.extra.length} file{run.extra.length === 1 ? '' : 's'} of extra data</>}
          </p>
        </div>
      )}
    </li>
  )
}

function Detail({ id, onChanged, onGone }: { id: string; onChanged: () => void; onGone: () => void }) {
  const [rec, setRec] = useState<RecordingDetail | null>(null)
  const [error, setError] = useState('')
  const [panel, setPanel] = useState(false)
  const [removing, setRemoving] = useState('')
  const [deleting, setDeleting] = useState(false)

  const load = useCallback(async () => {
    try {
      setRec(await getRecording(id))
      setError('')
    } catch (e) {
      setError(errorOf(e))
    }
  }, [id])
  useEffect(() => { void load() }, [load])
  // A run going on: its line follows it.
  const going = !!rec?.history.some((r) => r.status === 'thinking' || r.status === 'starting')
  useEffect(() => {
    if (!going) return
    const t = setInterval(() => void load(), 4000)
    return () => clearInterval(t)
  }, [going, load])

  const drop = async (itemId: string) => {
    setRemoving(itemId)
    try {
      const got = await dropRecordingData(id, itemId)
      setRec((old) => (old ? { ...old, data: got.items } : old))
      onChanged()
    } catch (e) {
      setError(errorOf(e))
    }
    setRemoving('')
  }

  const remove = async () => {
    if (!rec || !confirm(`Delete “${rec.name}”? Its extra data goes with it; what its runs made stays in web-clips.`)) return
    setDeleting(true)
    try {
      await deleteRecording(id)
      onGone()
    } catch (e) {
      setError(errorOf(e))
      setDeleting(false)
    }
  }

  if (!rec) {
    return error
      ? <p className="px-3 py-2 text-xs text-red-600 dark:text-red-400">{error}</p>
      : <p className="flex items-center gap-2 px-3 py-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" />Loading…</p>
  }
  const icon = 'inline-flex h-7 w-7 items-center justify-center rounded-md text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'
  const asked = [
    rec.asks.length ? `asked before it starts: ${rec.asks.map((a) => a.label).join(', ')}` : '',
    rec.asks_during.length ? `asked on the way: ${rec.asks_during.join(', ')}` : '',
    rec.agent_decides.length ? `the agent decides: ${rec.agent_decides.join(', ')}` : '',
  ].filter(Boolean).join(' · ')
  return (
    <div className="space-y-4 border-t border-zinc-800 bg-zinc-950/30 px-3 py-3">
      {error && <p className="flex items-center gap-2 text-xs text-red-600 dark:text-red-400"><AlertCircle className="h-3.5 w-3.5" />{error}</p>}
      {rec.about && <p className="whitespace-pre-wrap text-xs text-zinc-400">{rec.about}</p>}
      {asked && <p className="text-xs text-zinc-500">Values — {asked}.</p>}

      <details>
        <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Steps ({rec.steps})</summary>
        <ol className="mt-2 space-y-1 text-xs text-zinc-300">
          {rec.step_lines.map((line, i) => <li key={i} className="whitespace-pre-wrap">{line}</li>)}
        </ol>
      </details>

      <section className="space-y-2">
        <div className="flex items-center gap-2">
          <h5 className="flex-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Extra data · read before every run</h5>
          <button onClick={() => setPanel(true)} className="inline-flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-100 hover:bg-zinc-700">
            <Plus className="h-3.5 w-3.5" />Add data
          </button>
        </div>
        {rec.data.length === 0 ? (
          <p className="text-xs text-zinc-500">
            Nothing yet. Add what the agent should go by: how things are done for each client, a list, rules, an example, a conversation
            where you explained it — from your computer, your agents’ files or Flight Deck.
          </p>
        ) : (
          <ul className="divide-y divide-zinc-800 rounded-md border border-zinc-800">
            {rec.data.map((d) => (
              <li key={d.id} className="flex items-center gap-2 px-3 py-1.5">
                <FileText className="h-4 w-4 shrink-0 text-zinc-500" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-xs text-zinc-200" title={d.name}>{d.name}</div>
                  <div className="truncate text-[11px] text-zinc-500">{[d.origin, sizeText(d.size), when(d.added_at)].filter(Boolean).join(' · ')}</div>
                </div>
                <a href={recordingDataUrl(id, d.id)} target="_blank" rel="noopener noreferrer" className={icon} title="Open" aria-label={`Open ${d.name}`}><Eye className="h-3.5 w-3.5" /></a>
                <a href={recordingDataUrl(id, d.id, true)} className={icon} title="Download" aria-label={`Download ${d.name}`}><Download className="h-3.5 w-3.5" /></a>
                <button onClick={() => void drop(d.id)} disabled={removing === d.id} className={`${icon} hover:!text-red-600 dark:hover:!text-red-400`} title="Take it off" aria-label={`Take ${d.name} off`}>
                  {removing === d.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="space-y-2">
        <h5 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Runs ({rec.history.length})</h5>
        {rec.history.length === 0
          ? <p className="text-xs text-zinc-500">Not run yet. Run it from Captain for Chrome → Recordings.</p>
          : <ul className="divide-y divide-zinc-800 rounded-md border border-zinc-800">{rec.history.map((r) => <RunRow key={r.id} run={r} />)}</ul>}
      </section>

      <button onClick={() => void remove()} disabled={deleting} className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-red-600 dark:text-red-400 hover:bg-red-500/10 disabled:opacity-50">
        {deleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}Delete this recording
      </button>

      {panel && (
        <AddDataPanel title={`Extra data for “${rec.name}”`}
          subtitle="Every run from now on gets a copy in its folder, and the agent reads it before it starts."
          action={(n) => (n ? `Add ${n} to the recording` : 'Add to the recording')}
          onDone={async (staged) => {
            const got = await addRecordingData(id, staged.map((s) => s.id))
            setRec((old) => (old ? { ...old, data: got.items } : old))
            onChanged()
          }}
          onClose={() => setPanel(false)} />
      )}
    </div>
  )
}

export default function BrowserRecordings({ active = true }: { active?: boolean }) {
  const [recs, setRecs] = useState<RecordingSummary[] | null>(null)
  const [runs, setRuns] = useState<RecordingRun[]>([])
  const [open, setOpen] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    try {
      const got = await listRecordings()
      setRecs(got.recordings)
      setRuns(got.runs)
      setError('')
    } catch (e) {
      setError(errorOf(e))
    }
  }, [])
  useEffect(() => {
    void load()
    if (!active) return
    const t = setInterval(() => void load(), 10000)
    return () => clearInterval(t)
  }, [active, load])

  // Runs of recordings that are gone since: still part of what happened.
  const orphans = runs.filter((r) => recs && !recs.some((x) => x.id === r.recording))

  return (
    <section className="space-y-2">
      <div className="flex items-center gap-2">
        <Play className="h-4 w-4 text-zinc-300" />
        <h4 className="text-sm font-semibold text-zinc-100">Recordings</h4>
        <span className="text-[10px] uppercase tracking-wider text-zinc-500">show it once, an agent does it after that</span>
      </div>
      <p className="text-xs text-zinc-500">
        In Captain for Chrome press <b className="text-zinc-300">Record what I do</b>, do the thing on the site, and say what matters at each step.
        Run it from <b className="text-zinc-300">Recordings</b> there: the agent works in that browser, signed in as you. Here you give it what to go by, and see how each run went.
      </p>
      {error && <div className="flex items-center gap-2 text-xs text-red-500 dark:text-red-400"><AlertCircle className="h-3.5 w-3.5" />{error}</div>}
      {recs === null ? (
        !error && <div className="flex items-center gap-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" />Loading…</div>
      ) : recs.length === 0 ? (
        <p className="text-xs text-zinc-500">Nothing recorded yet.</p>
      ) : (
        <ul className="overflow-hidden rounded-md border border-zinc-800 divide-y divide-zinc-800">
          {recs.map((r) => (
            <li key={r.id}>
              <button onClick={() => setOpen(open === r.id ? '' : r.id)} aria-expanded={open === r.id}
                className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-zinc-800/40">
                {open === r.id ? <ChevronDown className="h-4 w-4 shrink-0 text-zinc-500" /> : <ChevronRight className="h-4 w-4 shrink-0 text-zinc-500" />}
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm text-zinc-200">{r.name}</div>
                  <div className="truncate text-[11px] text-zinc-500">
                    {[r.site, `${r.steps} step${r.steps === 1 ? '' : 's'}`, r.data ? `${r.data} file${r.data === 1 ? '' : 's'} of extra data` : '',
                      r.runs ? `ran ${r.runs}× · last ${when(r.last_run_at)}` : 'not run yet'].filter(Boolean).join(' · ')}
                  </div>
                </div>
              </button>
              {open === r.id && <Detail id={r.id} onChanged={() => void load()} onGone={() => { setOpen(''); void load() }} />}
            </li>
          ))}
        </ul>
      )}
      {orphans.length > 0 && (
        <details>
          <summary className="cursor-pointer text-xs text-zinc-500">Runs of recordings since deleted ({orphans.length})</summary>
          <ul className="mt-1 divide-y divide-zinc-800 rounded-md border border-zinc-800">{orphans.map((r) => <RunRow key={r.id} run={r} showName />)}</ul>
        </details>
      )}
    </section>
  )
}
