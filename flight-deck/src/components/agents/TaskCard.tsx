import { useCallback, useEffect, useState } from 'react'
import { CheckCircle2, CircleStop, ExternalLink, Hand, Loader2, RotateCw, XCircle, Zap } from 'lucide-react'
import { api } from '../../services/fileLibraryApi'
import { useAuthStore } from '../../stores/authStore'
import { useUIStore } from '../../stores/uiStore'
import { useBatStore } from '../../stores/batStore'

// Bigger work started from the chat (a Bat run) as a live card under the step
// that started it: working · needs you · done · failed · stopped, what it cost
// so far, and Stop / Try again / Open. The agent still reports the result in
// the chat when it ends; the card is how it looks meanwhile.

type State = 'working' | 'needs_you' | 'done' | 'failed' | 'stopped'

const STATE_OF: Record<string, State> = {
  planning: 'working', running: 'working', retrying: 'working', waiting: 'working',
  awaiting_plan: 'needs_you', awaiting_human: 'needs_you',
  done: 'done', error: 'failed', cancelled: 'stopped',
}

const LOOK: Record<State, { label: string; cls: string; Icon: typeof Zap }> = {
  working: { label: 'Working', cls: 'bg-amber-500/10 text-amber-700 dark:text-amber-300', Icon: Loader2 },
  needs_you: { label: 'Needs you', cls: 'bg-violet-500/10 text-violet-700 dark:text-violet-300', Icon: Hand },
  done: { label: 'Done', cls: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300', Icon: CheckCircle2 },
  failed: { label: 'Failed', cls: 'bg-red-500/10 text-red-700 dark:text-red-300', Icon: XCircle },
  stopped: { label: 'Stopped', cls: 'bg-zinc-800 text-zinc-400', Icon: CircleStop },
}

interface Run {
  id: string
  title: string
  task: string
  status: string
  cumulative_usd?: number
  stopped_reason?: string
}

/** The Bat run a `bat` tool call started, from its answer ("… (run <id>)"). */
export function batRunOf(toolName: string | undefined, output: string | undefined): string | null {
  if (toolName !== 'bat' || !output || !output.trimStart().startsWith('Started Bat run')) return null
  const m = /\(run ([A-Za-z0-9_-]{6,})\)/.exec(output)
  return m ? m[1] : null
}

export function TaskCard({ id: firstId }: { id: string }) {
  const [id, setId] = useState(firstId)
  const [run, setRun] = useState<Run | null>(null)
  const [gone, setGone] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const simple = useAuthStore((s) => s.simpleChatOnly)

  const load = useCallback(async () => {
    try {
      const r = await api<{ run: Run }>(`/fd/bat/runs/${encodeURIComponent(id)}`)
      setRun(r.run)
      setGone(false)
      return r.run
    } catch {
      setGone(true)
      return null
    }
  }, [id])

  useEffect(() => {
    let timer = 0
    let alive = true
    const tick = async () => {
      const now = await load()
      const state = now ? STATE_OF[now.status] : undefined
      if (alive && (state === 'working' || state === 'needs_you')) timer = window.setTimeout(tick, 6000)
    }
    void tick()
    return () => { alive = false; window.clearTimeout(timer) }
  }, [load])

  if (gone && !run) return null
  const state: State = run ? STATE_OF[run.status] || 'working' : 'working'
  const look = LOOK[state]

  const stop = async () => {
    setBusy('stop'); setError('')
    try {
      await api(`/fd/bat/runs/${encodeURIComponent(id)}/cancel`, { method: 'POST' })
      await load()
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed') } finally { setBusy('') }
  }
  const again = async () => {
    if (!run) return
    setBusy('again'); setError('')
    try {
      const r = await api<{ run_id?: string }>('/fd/bat/start', {
        method: 'POST', body: JSON.stringify({ task: run.task, title: run.title }),
      })
      if (r.run_id) setId(r.run_id)
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed') } finally { setBusy('') }
  }
  const open = () => {
    useUIStore.getState().setView('bat')
    void useBatStore.getState().select(id)
  }

  return (
    <div className="my-2 rounded-xl border border-zinc-800 bg-zinc-900/50 px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <Zap className="h-4 w-4 shrink-0 text-violet-500" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-100">{run?.title || 'Bat run'}</span>
        <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${look.cls}`}>
          <look.Icon className={`h-3.5 w-3.5 ${state === 'working' ? 'animate-spin' : ''}`} />{look.label}
        </span>
      </div>
      <p className="mt-1 text-xs text-zinc-500">
        {state === 'needs_you' ? 'It waits for you — answer it from the bell (Needs you).'
          : state === 'failed' ? (run?.stopped_reason || 'It stopped with an error.')
          : state === 'stopped' ? (run?.stopped_reason || 'It was stopped.')
          : state === 'done' ? 'Finished — the agent reports the result in the chat.'
          : 'Working on it; the agent reports back here when it ends.'}
        {run?.cumulative_usd ? ` · $${run.cumulative_usd.toFixed(2)} so far` : ''}
      </p>
      {error && <p className="mt-1 text-xs text-red-600 dark:text-red-400">{error}</p>}
      <div className="mt-2 flex flex-wrap gap-2">
        {(state === 'working' || state === 'needs_you') && (
          <button type="button" onClick={stop} disabled={!!busy} className="inline-flex items-center gap-1 rounded-lg border border-zinc-800 px-2.5 py-1 text-xs text-zinc-300 hover:border-zinc-700 disabled:opacity-50">
            {busy === 'stop' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CircleStop className="h-3.5 w-3.5" />} Stop
          </button>
        )}
        {(state === 'failed' || state === 'stopped') && run?.task && (
          <button type="button" onClick={again} disabled={!!busy} className="inline-flex items-center gap-1 rounded-lg border border-zinc-800 px-2.5 py-1 text-xs text-zinc-300 hover:border-zinc-700 disabled:opacity-50">
            {busy === 'again' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCw className="h-3.5 w-3.5" />} Try again
          </button>
        )}
        {!simple && (
          <button type="button" onClick={open} className="inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs text-violet-600 hover:underline dark:text-violet-400">
            <ExternalLink className="h-3.5 w-3.5" /> Open
          </button>
        )}
      </div>
    </div>
  )
}
