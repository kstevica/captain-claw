// ── BotPort domain types (mirrors botport/models.py + swarm/models.py) ──

export interface PersonaInfo {
  name: string
  description: string
  background: string
  expertise_tags: string[]
}

export interface InstanceInfo {
  id: string
  name: string
  personas: PersonaInfo[]
  tools: string[]
  models: string[]
  active_persona: string
  active_model: string
  max_concurrent: number
  active_concerns: number
  status: 'connected' | 'disconnected'
  connected_at: string
  last_heartbeat: string
  activity: Record<string, ActivityData>
}

export interface ActivityData {
  text?: string
  tool?: string
  phase?: string
  chunk?: string
  tool_name?: string
  arguments?: Record<string, unknown>
  output?: string
  updated_at?: string
}

export interface Concern {
  id: string
  from_instance: string
  from_instance_name: string
  from_session: string
  assigned_instance: string | null
  assigned_instance_name: string
  assigned_session: string | null
  task: string
  context: Record<string, unknown>
  expertise_tags: string[]
  status: 'pending' | 'assigned' | 'in_progress' | 'responded' | 'closed' | 'failed' | 'timeout'
  messages: ConcernExchange[]
  created_at: string
  updated_at: string
  timeout_at: string
  metadata: Record<string, unknown>
}

export interface ConcernExchange {
  direction: 'request' | 'response' | 'follow_up' | 'context_request' | 'context_reply'
  content: string
  timestamp: string
  from_instance: string
  metadata: Record<string, unknown>
}

// ── Swarm types ──

export type SwarmStatus = 'draft' | 'decomposing' | 'ready' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'
export type TaskStatus = 'queued' | 'waiting' | 'pending_approval' | 'running' | 'completed' | 'failed' | 'retrying' | 'paused' | 'skipped'

export interface SwarmProject {
  id: string
  name: string
  description: string
  created_at: string
  updated_at: string
  metadata: Record<string, unknown>
}

export interface Swarm {
  id: string
  project_id: string
  name: string
  original_task: string
  rephrased_task: string
  status: SwarmStatus
  priority: number
  concurrency_limit: number
  error_policy: 'fail_fast' | 'continue_on_error' | 'manual_review'
  agent_mode: 'connected' | 'designed'
  created_at: string
  updated_at: string
  started_at: string
  completed_at: string
  template_id: string
  metadata: Record<string, unknown>
}

export interface SwarmTask {
  id: string
  swarm_id: string
  name: string
  description: string
  status: TaskStatus
  priority: number
  assigned_instance: string
  assigned_persona: string
  concern_id: string
  position_x: number
  position_y: number
  retry_count: number
  max_retries: number
  requires_approval: boolean
  approval_status: '' | 'approved' | 'rejected'
  timeout_seconds: number
  created_at: string
  updated_at: string
  started_at: string
  completed_at: string
  input_data: Record<string, unknown>
  output_data: Record<string, unknown>
  error_message: string
  metadata: Record<string, unknown>
  // Shared workspace data flow declarations
  workspace_outputs?: string[]
  workspace_inputs?: string[]
  // Structured output validation
  output_schema?: Record<string, unknown>
  output_schema_name?: string
}

// ── Shared Workspace types ──

export interface WorkspaceEntry {
  value: unknown
  task_id: string
  session_id: string
  content_type: 'text' | 'json' | 'binary_ref'
}

export interface WorkspaceSnapshot {
  orchestration_id: string
  entry_count: number
  entries: Record<string, WorkspaceEntry>
}

// ── Trace / Observability types ──

export interface TraceSpan {
  span_id: string
  trace_id: string
  parent_span_id: string
  span_type: string       // orchestration | task | llm_call | tool_execution |
                          // validation | workspace_op | decompose | synthesize | execution
  name: string
  started_at: number
  ended_at: number
  duration_ms: number
  status: 'running' | 'completed' | 'failed'
  attributes: Record<string, unknown>
}

export interface TraceSummary {
  trace_id: string
  span_count: number
  by_type: Record<string, number>
  total_duration_ms: number
  total_input_tokens: number
  total_output_tokens: number
  completed: number
  failed: number
  running: number
}

export interface TraceData {
  spans: TraceSpan[]
  summary: TraceSummary
}

export interface SwarmEdge {
  id: number
  swarm_id: string
  from_task_id: string
  to_task_id: string
  edge_type: 'dependency' | 'data_flow'
}

export interface BotPortStats {
  connected_instances: number
  total_concerns: number
  active_concerns: number
  completed_concerns: number
  failed_concerns: number
}

// ── UI types ──

export type ViewMode = 'desktop' | 'workflow' | 'spawner' | 'forge' | 'operations' | 'admin'
