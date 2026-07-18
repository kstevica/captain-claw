// Local inference worker — this tab serves LLM completions to the backend.
//
// Flow (docs/mrav-micro-agent-plan.md, Phase 2): WebLLM runs a small model
// in a WebWorker (WebGPU); this manager registers the tab on Flight Deck's
// /fd/infer-ws and answers completion jobs routed by the broker. The tab
// never executes tools — it only turns (messages, schema) into tokens.
// Grammar-constrained JSON comes from WebLLM's xgrammar (response_format).
//
// Explicit opt-in only: nothing here runs until the user clicks Start in
// the Local inference panel (System page).
import {
  CreateWebWorkerMLCEngine,
  type MLCEngineInterface,
} from '@mlc-ai/web-llm'
import { useAuthStore } from '../stores/authStore'
import { useInferenceStore } from '../stores/inferenceStore'

// Total engine context: 8k input cap + 1k output, matching the broker
// and BrowserProvider sizing.
const CONTEXT_WINDOW = 9216

// Qwen3.5 is the live-eval champion family (5/6 on the mrav eval at 4B);
// all three sizes ship in WebLLM 0.2.84's prebuilt catalog.
export const MODEL_LADDER: { id: string; label: string; minMemGB: number }[] = [
  { id: 'Qwen3.5-4B-q4f16_1-MLC', label: 'Qwen3.5 4B — best quality (~4.2 GB GPU)', minMemGB: 16 },
  { id: 'Qwen3.5-2B-q4f16_1-MLC', label: 'Qwen3.5 2B — balanced (~2.3 GB GPU)', minMemGB: 8 },
  { id: 'Qwen3.5-0.8B-q4f16_1-MLC', label: 'Qwen3.5 0.8B — smallest (~1.1 GB GPU)', minMemGB: 4 },
]

export function webgpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!(navigator as any).gpu
}

export function pickDefaultModel(): string {
  // navigator.deviceMemory is Chromium-only and capped at 8 — treat 8 as
  // "could be anything ≥8 GB" and let desktop users upgrade manually.
  const mem = Number((navigator as any).deviceMemory || 8)
  if (mem >= 16) return MODEL_LADDER[0].id
  if (mem >= 8) return MODEL_LADDER[1].id
  return MODEL_LADDER[2].id
}

type Job = {
  type: 'job'
  job_id: string
  messages: { role: string; content: string }[]
  max_tokens?: number
  temperature?: number
  response_schema?: unknown
}

class LocalInferenceManager {
  private engine: MLCEngineInterface | null = null
  private worker: Worker | null = null
  private ws: WebSocket | null = null
  private desired = false
  private modelId = ''
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null

  private patch(partial: Parameters<ReturnType<typeof useInferenceStore.getState>['patch']>[0]) {
    useInferenceStore.getState().patch(partial)
  }

  async start(modelId?: string): Promise<void> {
    if (this.desired) return
    if (!webgpuAvailable()) {
      this.patch({ status: 'error', error: 'WebGPU is not available in this browser.' })
      return
    }
    this.desired = true
    this.modelId = modelId || pickDefaultModel()
    this.patch({ status: 'starting', modelId: this.modelId, error: '', progress: 'Loading model…' })

    try {
      this.worker = new Worker(new URL('./webllmWorker.ts', import.meta.url), { type: 'module' })
      this.engine = await CreateWebWorkerMLCEngine(
        this.worker,
        this.modelId,
        {
          initProgressCallback: (report) => {
            this.patch({ progress: report.text || `${Math.round((report.progress || 0) * 100)}%` })
          },
        },
        { context_window_size: CONTEXT_WINDOW },
      )
    } catch (err) {
      this.patch({
        status: 'error',
        error: `Model load failed: ${err instanceof Error ? err.message : String(err)}`,
      })
      this.desired = false
      this.teardown()
      return
    }

    this.patch({ progress: 'Model ready — connecting to Flight Deck…' })
    this.connect()
  }

  stop(): void {
    this.desired = false
    this.teardown()
    this.patch({ status: 'off', wsConnected: false, progress: '', error: '' })
  }

  private teardown(): void {
    if (this.reconnectTimer) { clearTimeout(this.reconnectTimer); this.reconnectTimer = null }
    if (this.ws) { try { this.ws.close() } catch { /* already gone */ } this.ws = null }
    if (this.engine) { try { void this.engine.unload() } catch { /* best effort */ } this.engine = null }
    if (this.worker) { try { this.worker.terminate() } catch { /* best effort */ } this.worker = null }
  }

  private connect(): void {
    if (!this.desired) return
    const { token, authEnabled } = useAuthStore.getState()
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const query = authEnabled && token ? `?token=${encodeURIComponent(token)}` : ''
    const ws = new WebSocket(`${proto}://${window.location.host}/fd/infer-ws${query}`)
    this.ws = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'register',
        engine: 'webllm',
        model: this.modelId,
        ctx_max: CONTEXT_WINDOW,
        schema: true,
      }))
    }
    ws.onmessage = (event) => {
      let msg: any
      try { msg = JSON.parse(event.data) } catch { return }
      if (msg.type === 'registered') {
        this.patch({ status: 'ready', wsConnected: true, progress: 'Serving as inference worker.' })
      } else if (msg.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }))
      } else if (msg.type === 'job') {
        void this.serve(msg as Job)
      }
    }
    ws.onclose = () => {
      this.patch({ wsConnected: false })
      if (this.desired) {
        this.patch({ progress: 'Connection lost — reconnecting…' })
        this.reconnectTimer = setTimeout(() => this.connect(), 3000)
      }
    }
    ws.onerror = () => { try { ws.close() } catch { /* triggers onclose */ } }
  }

  private async serve(job: Job): Promise<void> {
    const started = performance.now()
    if (!this.engine || !this.ws) return
    try {
      const request: any = {
        messages: job.messages,
        temperature: job.temperature ?? 0.2,
        max_tokens: job.max_tokens ?? 1024,
      }
      if (job.response_schema) {
        // xgrammar-enforced JSON — the whole point of serving mrav from
        // a browser tab with a 2-4B model.
        request.response_format = { type: 'json_object', schema: JSON.stringify(job.response_schema) }
      }
      const reply = await this.engine.chat.completions.create(request)
      const choice = reply?.choices?.[0]
      const usage = reply?.usage
      this.ws.send(JSON.stringify({
        type: 'result',
        job_id: job.job_id,
        content: choice?.message?.content ?? '',
        finish_reason: choice?.finish_reason ?? '',
        model: this.modelId,
        usage: {
          prompt_tokens: usage?.prompt_tokens ?? 0,
          completion_tokens: usage?.completion_tokens ?? 0,
          total_tokens: usage?.total_tokens ?? 0,
        },
      }))
      const seconds = ((performance.now() - started) / 1000).toFixed(1)
      const state = useInferenceStore.getState()
      this.patch({
        jobsDone: state.jobsDone + 1,
        lastCall: `${usage?.prompt_tokens ?? '?'}→${usage?.completion_tokens ?? '?'} tok · ${seconds}s`,
      })
    } catch (err) {
      try {
        this.ws?.send(JSON.stringify({
          type: 'error',
          job_id: job.job_id,
          message: err instanceof Error ? err.message : String(err),
        }))
      } catch { /* socket already gone; broker fails the job on disconnect */ }
    }
  }
}

export const localInference = new LocalInferenceManager()
