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
  prebuiltAppConfig,
  type MLCEngineInterface,
} from '@mlc-ai/web-llm'
import { useAuthStore } from '../stores/authStore'
import { useInferenceStore } from '../stores/inferenceStore'

// Engine context window (total: input + output). 9216 matches mrav's
// default caps (8k in + 1k out); 40960 matches a raised 32k/8k tier.
// Bigger windows cost GPU memory (KV grows linearly with the window).
export const DEFAULT_ENGINE_WINDOW = 9216
export const ENGINE_WINDOW_OPTIONS: { value: number; label: string }[] = [
  { value: 4096, label: '4k' },
  { value: 8192, label: '8k' },
  { value: 9216, label: '9k — mrav default (8k in + 1k out)' },
  { value: 16384, label: '16k' },
  { value: 32768, label: '32k' },
  { value: 40960, label: '40k — raised mrav tier (32k in + 8k out)' },
  { value: 65536, label: '64k' },
]

// Curated picks (q4f16, VRAM at the catalog's 4k reference — our 9216-token
// engine window adds KV on top). Qwen3.5 is the live-eval champion family;
// 9B is the largest Qwen3.5 that WebLLM ships (Qwen3.6 exists only as
// 27B/35B — not convertible to a browser tab).
export interface BrowserModel {
  id: string
  label: string
  vramGB: number
  note: string
  recommended?: boolean
}

export const MODEL_CATALOG: BrowserModel[] = [
  { id: 'Qwen3.5-4B-q4f16_1-MLC', label: 'Qwen3.5 4B', vramGB: 3.8, note: 'recommended — mrav live-eval champion', recommended: true },
  { id: 'Qwen3.5-9B-q4f16_1-MLC', label: 'Qwen3.5 9B', vramGB: 6.3, note: 'best quality in a tab — 16 GB+ machines', recommended: true },
  { id: 'Qwen3.5-2B-q4f16_1-MLC', label: 'Qwen3.5 2B', vramGB: 2.2, note: 'balanced small', recommended: true },
  { id: 'Qwen3.5-0.8B-q4f16_1-MLC', label: 'Qwen3.5 0.8B', vramGB: 1.6, note: 'smallest useful', recommended: true },
  { id: 'Qwen3-8B-q4f16_1-MLC', label: 'Qwen3 8B', vramGB: 5.6, note: 'previous-gen large' },
  { id: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC', label: 'Hermes 3 · Llama 3.1 8B', vramGB: 4.8, note: 'function-calling tuned' },
  { id: 'Phi-4-mini-instruct-q4f16_1-MLC', label: 'Phi-4 mini (3.8B)', vramGB: 3.4, note: 'Microsoft, MIT license' },
  { id: 'Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC', label: 'Ministral 3 3B', vramGB: 2.8, note: 'Mistral small, tool-use' },
  { id: 'gemma3-1b-it-q4f16_1-MLC', label: 'Gemma 3 1B', vramGB: 0.7, note: 'floor — weakest, fastest' },
]

// Everything else the pinned WebLLM build can run: q4f16 chat models under
// ~8.5 GB that are not already curated (no embeddings, no crippled -1k ctx
// variants). Future package bumps surface here automatically.
export function extraCatalogModels(): BrowserModel[] {
  const curated = new Set(MODEL_CATALOG.map((m) => m.id))
  const out: BrowserModel[] = []
  for (const m of prebuiltAppConfig.model_list) {
    const id = m.model_id
    if (curated.has(id)) continue
    if (!id.endsWith('q4f16_1-MLC')) continue
    if (id.startsWith('snowflake')) continue
    const vram = (m.vram_required_MB || 0) / 1024
    if (!vram || vram > 8.5) continue
    out.push({ id, label: id.replace('-q4f16_1-MLC', ''), vramGB: Math.round(vram * 10) / 10, note: '' })
  }
  return out.sort((a, b) => a.vramGB - b.vramGB)
}

export function webgpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!(navigator as any).gpu
}

export function pickDefaultModel(): string {
  // navigator.deviceMemory is Chromium-only and usually capped at 8 — treat
  // 8 as "could be anything ≥8 GB" and let users pick bigger models by hand.
  const mem = Number((navigator as any).deviceMemory || 8)
  if (mem >= 16) return 'Qwen3.5-4B-q4f16_1-MLC'
  if (mem >= 8) return 'Qwen3.5-2B-q4f16_1-MLC'
  return 'Qwen3.5-0.8B-q4f16_1-MLC'
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
  private ctxWindow = DEFAULT_ENGINE_WINDOW
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null

  private patch(partial: Parameters<ReturnType<typeof useInferenceStore.getState>['patch']>[0]) {
    useInferenceStore.getState().patch(partial)
  }

  async start(modelId?: string, ctxWindow?: number): Promise<void> {
    if (this.desired) return
    if (!webgpuAvailable()) {
      this.patch({ status: 'error', error: 'WebGPU is not available in this browser.' })
      return
    }
    this.desired = true
    this.modelId = modelId || pickDefaultModel()
    this.ctxWindow = ctxWindow || DEFAULT_ENGINE_WINDOW
    this.patch({
      status: 'starting', modelId: this.modelId, ctxWindow: this.ctxWindow,
      error: '', progress: 'Loading model…',
    })

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
        { context_window_size: this.ctxWindow },
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
        ctx_max: this.ctxWindow,
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
