import { useAuthStore, refreshAccessToken } from '../stores/authStore'

const BASE = '/fd'

// ── Auth helper (mirrors src/services/fileTransfer.ts) ──

function _authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = {}
  if (authEnabled && token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  return headers
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = { ..._authHeaders(), ...(init?.headers as Record<string, string> | undefined) }
  let res = await fetch(`${BASE}${path}`, { ...init, headers, credentials: 'include' })
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) {
      const h2 = { ..._authHeaders(), ...(init?.headers as Record<string, string> | undefined) }
      res = await fetch(`${BASE}${path}`, { ...init, headers: h2, credentials: 'include' })
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(body.error || body.detail || `${res.status} ${res.statusText}`)
  }
  // DELETE / enable may return empty body in some cases — guard.
  const text = await res.text()
  return (text ? JSON.parse(text) : {}) as T
}

function jsonInit(method: string, body?: unknown): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  }
}

// ── Data model (mirrors PROCESS_ENGINE_DESIGN.md) ──

export type TriggerOn = 'message' | 'schedule' | 'decision'
export type TriggerChannel = 'any' | 'whatsapp' | 'glasses' | 'web'
export type MatchKind = 'rule' | 'classifier' | 'always'

export interface FlowMatch {
  kind: MatchKind
  rules: string[]
  labels: string[]
}

export interface FlowTrigger {
  on: TriggerOn
  channel: TriggerChannel
  match: FlowMatch
}

export type StepType = 'tool' | 'agent' | 'vision' | 'input' | 'branch' | 'emit'

export interface FlowStep {
  id: string
  type: StepType
  /** agent selector: "origin" | "capability:vision" | "name:Agent" */
  on?: string
  // tool step
  tool?: string
  args?: Record<string, string>
  // agent step
  prompt?: string
  /** optional file/image to send to the agent, e.g. {{trigger.image_path}} */
  attach?: string
  guardrails?: { allow?: string[]; deny?: string[] }
  // branch step
  when?: string          // legacy single-condition (still honored)
  goto?: string          // legacy single-target
  /** switch/case: first case whose condition is true jumps to its goto */
  cases?: { when: string; goto: string }[]
  /** goto when no case matches (the "else") */
  default?: string
  // emit step
  channel?: string
  body?: string
  // input step — pause the run, ask the user, resume with their reply
  /** seconds to wait for the user's reply before failing the run */
  timeout?: number
}

export interface FlowGuardrails {
  max_steps: number
  timeout_s: number
}

export interface FlowOutput {
  channel: string
  format: string
}

export interface FlowLastRun {
  id: string
  status: string
  started_at?: string
  ended_at?: string
  error?: string | null
}

export interface Flow {
  id: string
  name: string
  description: string
  enabled: boolean
  priority: number
  trigger: FlowTrigger
  steps: FlowStep[]
  output: FlowOutput
  guardrails: FlowGuardrails
  updated_at?: string
  last_run?: FlowLastRun | null
}

/** Payload sent to create/update — no server-managed fields. */
export type FlowInput = Pick<
  Flow,
  'name' | 'description' | 'enabled' | 'priority' | 'trigger' | 'steps' | 'guardrails' | 'output'
>

export interface FlowRunSummary {
  id: string
  status: string
  started_at?: string
  ended_at?: string
  error?: string | null
}

export interface FlowRunStep {
  step_id: string
  seq: number
  status: string
  agent?: string
  output_text?: string
  ms?: number
}

export interface FlowRunDetail {
  run: FlowRunSummary & { flow_id?: string; trigger_payload?: unknown }
  steps: FlowRunStep[]
}

/** A single step result from the /test dry-run. */
export interface FlowTestStep {
  step_id: string
  status: string
  agent?: string
  output?: string
  ms?: number
}

// ── Endpoints ──

export async function listFlows(): Promise<Flow[]> {
  const r = await fdFetch<{ flows: Flow[] }>('/flows')
  return r.flows || []
}

export interface FleetAgentLite { name: string; status?: string; description?: string }

/** Running/known agents — used to populate the step "agent selector" dropdown. */
export async function listFleet(): Promise<FleetAgentLite[]> {
  try {
    return (await fdFetch<FleetAgentLite[]>('/fleet')) || []
  } catch {
    return []
  }
}

export async function getFlow(id: string): Promise<Flow> {
  return fdFetch<Flow>(`/flows/${encodeURIComponent(id)}`)
}

export async function createFlow(input: FlowInput): Promise<{ id: string }> {
  return fdFetch<{ id: string }>('/flows', jsonInit('POST', input))
}

export async function updateFlow(id: string, flow: Flow): Promise<{ ok: boolean }> {
  return fdFetch<{ ok: boolean }>(`/flows/${encodeURIComponent(id)}`, jsonInit('PUT', flow))
}

export async function deleteFlow(id: string): Promise<{ ok: boolean }> {
  return fdFetch<{ ok: boolean }>(`/flows/${encodeURIComponent(id)}`, jsonInit('DELETE'))
}

export async function enableFlow(id: string, enabled: boolean): Promise<{ ok: boolean }> {
  return fdFetch<{ ok: boolean }>(`/flows/${encodeURIComponent(id)}/enable`, jsonInit('POST', { enabled }))
}

export async function runFlow(id: string, payload?: Record<string, unknown>): Promise<{ run_id: string }> {
  return fdFetch<{ run_id: string }>(`/flows/${encodeURIComponent(id)}/run`, jsonInit('POST', { payload }))
}

export async function testFlow(
  id: string,
  payload: Record<string, unknown>,
): Promise<{ steps: FlowTestStep[] }> {
  return fdFetch<{ steps: FlowTestStep[] }>(`/flows/${encodeURIComponent(id)}/test`, jsonInit('POST', { payload }))
}

export async function listFlowRuns(id: string): Promise<FlowRunSummary[]> {
  const r = await fdFetch<{ runs: FlowRunSummary[] }>(`/flows/${encodeURIComponent(id)}/runs`)
  return r.runs || []
}

export async function getFlowRun(runId: string): Promise<FlowRunDetail> {
  return fdFetch<FlowRunDetail>(`/flows/runs/${encodeURIComponent(runId)}`)
}

// ── Helpers ──

/** Empty flow scaffold for the New Flow form. */
export function emptyFlow(): FlowInput {
  return {
    name: '',
    description: '',
    enabled: true,
    priority: 50,
    trigger: { on: 'message', channel: 'any', match: { kind: 'rule', rules: [], labels: [] } },
    steps: [],
    guardrails: { max_steps: 12, timeout_s: 600 },
    output: { channel: 'same', format: 'text' },
  }
}

/** Human-readable trigger summary like "WhatsApp · has video". */
export function triggerSummary(t: FlowTrigger): string {
  const channelLabel: Record<TriggerChannel, string> = {
    any: 'Any',
    whatsapp: 'WhatsApp',
    glasses: 'Glasses',
    web: 'Web',
  }
  const ch = channelLabel[t.channel] || t.channel
  let match = ''
  if (t.match.kind === 'always') {
    match = 'always'
  } else if (t.match.kind === 'classifier') {
    match = t.match.labels.length ? `classify: ${t.match.labels.join(', ')}` : 'classify'
  } else {
    match = t.match.rules.length ? t.match.rules.join(', ').replace(/_/g, ' ') : 'no rules'
  }
  return `${ch} · ${match}`
}
