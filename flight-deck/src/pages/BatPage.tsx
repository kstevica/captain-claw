// Bat — the stubborn finisher. Start a run, watch it work, and answer what it
// needs from you. Reuses the Basna/Vatra run components (progress feed, live
// agent cards, VFS files) over Bat's own store.
import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { Hammer, Loader2, Square } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useBatStore } from '../stores/batStore'
import type { BatRun } from '../stores/batStore'
import { ProgressFeed, LiveAgentsPanel, ResizableSplit } from '../components/basna/RunWorkspace'
import { RunFilesPanel, RunDatastorePanel } from '../components/basna/RunArtifacts'
import { buildLiveAgents, STATUS_DOT } from '../components/basna/shared'
import { BatAskCard } from '../components/bat/BatAskCard'

type TabKey = 'steps' | 'agents' | 'progress' | 'result' | 'files'

const RUNNING = ['planning', 'running', 'retrying', 'waiting', 'awaiting_plan', 'awaiting_human']

const DOT: Record<string, string> = {
  ...STATUS_DOT,
  planning: 'bg-sky-400',
  retrying: 'bg-amber-400',
  waiting: 'bg-amber-400',
  awaiting_plan: 'bg-fuchsia-400',
  awaiting_human: 'bg-fuchsia-400',
  cancelled: 'bg-zinc-500',
}

function dollars(n?: number): string {
  return `$${(n || 0).toFixed(2)}`
}

function Compose() {
  const startRun = useBatStore((s) => s.startRun)
  const busy = useBatStore((s) => s.busy)
  const [task, setTask] = useState('')
  const [adv, setAdv] = useState(false)
  const [realCap, setRealCap] = useState('')
  const [perItem, setPerItem] = useState('')
  const [mode, setMode] = useState<'plain' | 'archetype'>('plain')

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!task.trim()) return
    const id = await startRun(task.trim(), {
      real_usd_cap: parseFloat(realCap) || 0,
      per_item_usd: parseFloat(perItem) || 0,
      worker_mode: mode,
    })
    if (id) {
      setTask('')
      setRealCap('')
      setPerItem('')
    }
  }

  return (
    <form onSubmit={submit} className="space-y-2">
      <textarea
        value={task}
        onChange={(e) => setTask(e.target.value)}
        rows={2}
        placeholder="Give Bat a goal to finish — it keeps working until an independent judge says it's done."
        className="w-full resize-y rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-amber-500 focus:outline-none"
      />
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="submit"
          disabled={busy || !task.trim()}
          className="inline-flex items-center gap-1.5 rounded-md bg-amber-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-500 disabled:opacity-50"
        >
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Hammer size={14} />}
          Start Bat run
        </button>
        <div className="inline-flex overflow-hidden rounded-md border border-zinc-700 text-xs">
          <button type="button" onClick={() => setMode('plain')}
            className={`px-2.5 py-1.5 ${mode === 'plain' ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800'}`}
            title="A generic full-toolset worker per step">Plain</button>
          <button type="button" onClick={() => setMode('archetype')}
            className={`px-2.5 py-1.5 ${mode === 'archetype' ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800'}`}
            title="A best-fit specialist archetype per step">Archetype</button>
        </div>
        <button type="button" onClick={() => setAdv((v) => !v)} className="text-xs text-zinc-400 hover:text-zinc-200">
          {adv ? 'Hide' : 'Spend cap…'}
        </button>
        {adv && (
          <div className="flex items-center gap-2 text-xs text-zinc-400">
            <label className="flex items-center gap-1">
              run cap $
              <input value={realCap} onChange={(e) => setRealCap(e.target.value)} inputMode="decimal"
                className="w-16 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-1 text-zinc-100" />
            </label>
            <label className="flex items-center gap-1">
              auto-approve ≤ $
              <input value={perItem} onChange={(e) => setPerItem(e.target.value)} inputMode="decimal"
                className="w-16 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-1 text-zinc-100" />
            </label>
          </div>
        )}
      </div>
    </form>
  )
}

function RunsList() {
  const runs = useBatStore((s) => s.runs)
  const activeId = useBatStore((s) => s.activeId)
  const select = useBatStore((s) => s.select)
  if (runs.length === 0) {
    return <p className="p-3 text-sm text-zinc-500">No Bat runs yet.</p>
  }
  return (
    <ul className="space-y-1 p-1">
      {runs.map((r: BatRun) => (
        <li key={r.id}>
          <button
            onClick={() => select(r.id)}
            className={`w-full rounded-md px-2.5 py-2 text-left text-sm ${
              activeId === r.id ? 'bg-zinc-800' : 'hover:bg-zinc-800/60'
            }`}
          >
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 shrink-0 rounded-full ${DOT[r.status] || 'bg-zinc-500'}`} />
              <span className="truncate text-zinc-100">{r.title || r.task.slice(0, 48) || r.id}</span>
            </div>
            <div className="mt-0.5 pl-4 text-[11px] text-zinc-500">
              {r.status}
              {r.cumulative_usd ? ` · ${dollars(r.cumulative_usd)} LLM` : ''}
            </div>
          </button>
        </li>
      ))}
    </ul>
  )
}

function Budget({ run, committed, cap }: { run: BatRun; committed: number; cap: number }) {
  const llmCap = run.llm_usd_cap || 0
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 rounded-md border border-zinc-800 bg-zinc-900/50 px-3 py-2 text-xs text-zinc-300">
      <span>LLM spend: <b className="text-zinc-100">{dollars(run.cumulative_usd)}</b>{llmCap ? ` / ${dollars(llmCap)}` : ''}</span>
      {cap > 0 && (
        <span>
          Real money: <b className={committed >= cap ? 'text-rose-400' : 'text-zinc-100'}>{dollars(committed)}</b> / {dollars(cap)}
        </span>
      )}
      {run.email_allowed && <span className="text-emerald-400">email ✓</span>}
      {run.spend_allowed && <span className="text-emerald-400">spend ✓</span>}
      {run.account_allowed && <span className="text-emerald-400">accounts ✓</span>}
    </div>
  )
}

function Detail() {
  const active = useBatStore((s) => s.active)
  const asks = useBatStore((s) => s.asks)
  const cancel = useBatStore((s) => s.cancel)
  const [tab, setTab] = useState<TabKey>('steps')
  if (!active) {
    return <p className="p-4 text-sm text-zinc-500">Select a run to watch it work.</p>
  }
  const { run, steps, events, spend } = active
  const running = RUNNING.includes(run.status)
  const myAsks = asks.filter((a) => a.run_id === run.id)
  const agents = buildLiveAgents(events)
  const doneSteps = steps.filter((s) => s.status === 'done').length

  const tabs: { key: TabKey; label: string }[] = [
    { key: 'steps', label: `Steps${steps.length ? ` (${doneSteps}/${steps.length})` : ''}` },
    { key: 'agents', label: `Live agents${agents.length ? ` (${agents.length})` : ''}` },
    { key: 'progress', label: `Progress${events.length ? ` (${events.length})` : ''}` },
    { key: 'result', label: 'Result' },
    { key: 'files', label: 'Files' },
  ]

  return (
    <div className="flex h-full flex-col p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-zinc-100">{run.title || run.task.slice(0, 60)}</h2>
          <p className="mt-0.5 flex items-center gap-2 text-xs text-zinc-400">
            <span className={`h-2 w-2 rounded-full ${DOT[run.status] || 'bg-zinc-500'}`} />
            {run.status}
            {run.stopped_reason ? ` — ${run.stopped_reason}` : ''}
          </p>
        </div>
        {running && (
          <button
            onClick={() => cancel(run.id)}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
          >
            <Square size={12} /> Stop
          </button>
        )}
      </div>

      <div className="mt-2">
        <Budget run={run} committed={spend?.committed_usd || 0} cap={spend?.cap || 0} />
      </div>

      {myAsks.length > 0 && <div className="mt-2"><BatAskCard asks={myAsks} /></div>}

      <div className="mt-3 flex gap-1 border-b border-zinc-800">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`-mb-px border-b-2 px-3 py-1.5 text-xs font-medium ${
              tab === t.key
                ? 'border-amber-500 text-zinc-100'
                : 'border-transparent text-zinc-400 hover:text-zinc-200'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="mt-3 min-h-0 flex-1 overflow-y-auto">
        {tab === 'steps' && (
          steps.length === 0
            ? <p className="text-sm text-zinc-500">No steps planned yet.</p>
            : (
              <ul className="divide-y divide-zinc-800/60 rounded-md border border-zinc-800">
                {steps.map((s) => (
                  <li key={s.step_key} className="flex items-center gap-2 px-3 py-2 text-sm">
                    <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${DOT[s.status] || 'bg-zinc-500'}`} />
                    <span className="flex-1 truncate text-zinc-200">{s.title || s.step_key}</span>
                    <span className="shrink-0 text-[11px] text-zinc-500">{s.status}{s.attempt > 1 ? ` ·${s.attempt}` : ''}</span>
                  </li>
                ))}
              </ul>
            )
        )}
        {tab === 'agents' && (
          agents.length === 0
            ? <p className="text-sm text-zinc-500">No agents spawned yet.</p>
            : <LiveAgentsPanel agents={agents} />
        )}
        {tab === 'progress' && <ProgressFeed progress={events} running={running} fill defaultOpen />}
        {tab === 'result' && (
          run.truth
            ? <div className="fd-markdown text-sm text-zinc-200"><Markdown remarkPlugins={[remarkGfm]}>{run.truth}</Markdown></div>
            : <p className="text-sm text-zinc-500">{running ? 'No result yet — still working.' : 'No result produced.'}</p>
        )}
        {tab === 'files' && (
          run.vfs_project
            ? (
              <div className="space-y-3">
                <RunFilesPanel project={run.vfs_project} live={running} variant="tab" />
                <RunDatastorePanel project={run.vfs_project} live={running} variant="tab" hideWhenEmpty />
              </div>
            )
            : <p className="text-sm text-zinc-500">No workspace.</p>
        )}
      </div>
    </div>
  )
}

export function BatPage() {
  const loadRuns = useBatStore((s) => s.loadRuns)
  const loadAsks = useBatStore((s) => s.loadAsks)
  const poll = useBatStore((s) => s.poll)
  const asks = useBatStore((s) => s.asks)
  const error = useBatStore((s) => s.error)

  useEffect(() => {
    void loadRuns()
    void loadAsks()
    const t = setInterval(() => void poll(), 3500)
    return () => clearInterval(t)
  }, [loadRuns, loadAsks, poll])

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      <div className="flex items-center gap-2">
        <Hammer size={18} className="text-amber-400" />
        <h1 className="text-lg font-semibold text-zinc-100">Bat</h1>
        <span className="text-xs text-zinc-500">the stubborn finisher — works a goal to a judged “done”.</span>
      </div>

      <Compose />
      {error && <p className="text-xs text-rose-400">{error}</p>}

      {asks.length > 0 && (
        <div>
          <div className="mb-1 text-xs font-medium text-amber-300">Bat needs you</div>
          <BatAskCard asks={asks} />
        </div>
      )}

      <div className="min-h-0 flex-1">
        <ResizableSplit
          storageKey="fd.bat.split"
          left={<div className="h-full overflow-y-auto">{<RunsList />}</div>}
          right={<Detail />}
        />
      </div>
    </div>
  )
}
