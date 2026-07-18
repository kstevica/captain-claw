// Local inference panel (System page) — opt-in: this tab becomes an
// inference worker for YOUR mrav agents. See src/inference/localInference.ts.
import { useState } from 'react'
import { Bug, Loader2, Play, Square, Zap } from 'lucide-react'
import {
  MODEL_LADDER,
  localInference,
  pickDefaultModel,
  webgpuAvailable,
} from '../../inference/localInference'
import { useInferenceStore } from '../../stores/inferenceStore'

const STATUS_META: Record<string, { label: string; cls: string }> = {
  off: { label: 'off', cls: 'bg-zinc-800 text-zinc-500 border-zinc-700' },
  starting: { label: 'loading', cls: 'bg-amber-500/15 text-amber-400 border-amber-500/30' },
  ready: { label: 'serving', cls: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' },
  error: { label: 'error', cls: 'bg-red-500/15 text-red-400 border-red-500/30' },
}

export function LocalInferencePanel() {
  const { status, modelId, progress, error, wsConnected, jobsDone, lastCall } = useInferenceStore()
  const [selectedModel, setSelectedModel] = useState(pickDefaultModel())
  const gpu = webgpuAvailable()
  const running = status === 'starting' || status === 'ready'
  const meta = STATUS_META[status] ?? STATUS_META.off

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Bug className="h-4 w-4 text-violet-400" />
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-zinc-100">Local inference</h2>
              <span className={`inline-flex items-center rounded-full border px-1.5 py-0.5 text-[10px] font-medium ${meta.cls}`}>
                {meta.label}
              </span>
            </div>
            <p className="text-xs text-zinc-500">
              Serve your Mrav agents from this tab — WebLLM on WebGPU, no server GPU needed.
              Agents with provider <span className="font-mono text-zinc-400">browser</span> route here.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={running ? modelId : selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={running}
            className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none disabled:opacity-60"
          >
            {MODEL_LADDER.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
          {running ? (
            <button
              onClick={() => localInference.stop()}
              className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800"
            >
              <Square className="h-3 w-3" /> Stop
            </button>
          ) : (
            <button
              onClick={() => void localInference.start(selectedModel)}
              disabled={!gpu}
              title={gpu ? 'Download the model (cached after the first time) and start serving' : 'WebGPU is not available in this browser'}
              className="flex items-center gap-1.5 rounded-md border border-violet-500/40 bg-violet-500/15 px-3 py-1.5 text-xs font-medium text-violet-300 hover:bg-violet-500/25 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Play className="h-3 w-3" /> Start
            </button>
          )}
        </div>
      </div>

      {(running || status === 'error') && (
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
          {status === 'starting' && <Loader2 className="h-3 w-3 animate-spin text-amber-400" />}
          {status === 'error'
            ? <span className="text-red-400">{error}</span>
            : <span className="text-zinc-400">{progress}</span>}
          {status === 'ready' && (
            <>
              <span className={wsConnected ? 'text-emerald-500' : 'text-amber-500'}>
                {wsConnected ? 'connected' : 'reconnecting…'}
              </span>
              <span className="text-zinc-500">jobs: {jobsDone}</span>
              {lastCall && (
                <span className="flex items-center gap-1 text-zinc-500">
                  <Zap className="h-3 w-3 text-violet-400" /> {lastCall}
                </span>
              )}
            </>
          )}
        </div>
      )}
      {!gpu && status === 'off' && (
        <p className="mt-2 text-[11px] text-amber-500/80">
          WebGPU not detected — use a current Chrome, Edge, Safari 26+ or Firefox.
        </p>
      )}
    </div>
  )
}
