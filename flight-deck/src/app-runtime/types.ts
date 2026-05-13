// Manifest schema for agent-native apps rendered by the App runtime.
//
// An agent declares: typed entities, read-only feeds, write actions, and
// composed surfaces. Renderers (web today, voice/glasses later) bind to
// this manifest — they never speak to the agent directly, only via the
// MCP tools the manifest names.

export type FieldType =
  | 'string'
  | 'text'
  | 'number'
  | 'boolean'
  | 'date'
  | 'datetime'
  | 'enum'
  | 'markdown'
  | 'file'           // upload — value is a file_id from /fd/apps/{id}/files
  | { ref: string } // ref(entityId)

export interface EntityField {
  type: FieldType
  label?: string
  values?: string[]      // for enum
  primary?: boolean      // identifier field
  title?: boolean        // human-readable title field
  required?: boolean
}

export interface EntityDef {
  id: string
  label: string
  plural?: string
  fields: Record<string, EntityField>
  default_view?: 'card' | 'row' | 'summary'
}

export interface FeedDef {
  id: string
  label: string
  mcp_tool: string                  // tool name on the agent's MCP server
  arguments?: Record<string, unknown> // static args (templated values allowed: $entity.id etc.)
  returns: string                   // entity id
  surfaces?: string[]               // surface ids this feed appears on
  refresh_seconds?: number
  proactive?: boolean
  description?: string
}

export interface ActionInputDef {
  type: FieldType
  label?: string
  required?: boolean
  values?: string[]
}

export interface ActionDef {
  id: string
  label: string
  mcp_tool: string
  inputs: Record<string, ActionInputDef>
  nl_aliases?: string[]
  surfaces?: string[]
  prefill?: Record<string, string>  // values like "$entity.id"
  returns?: 'markdown' | 'entity' | 'none'
  confirm?: boolean
  prominent?: boolean
  description?: string
}

export interface SurfaceSection {
  type: 'feed' | 'action' | 'chat'
  id: string
  filter?: Record<string, string>
  prefill?: Record<string, string>
  prominent?: boolean
}

export interface SurfaceDef {
  id: string
  label?: string
  layout: 'dashboard' | 'list' | 'entity' | 'inbox' | 'upload'
  entity?: string                   // for layout=entity
  sources?: string[]                // for layout=inbox: feed ids; for layout=upload: action ids
  sections?: SurfaceSection[]       // for layout=dashboard/entity
  accept?: string                   // for layout=upload — input accept filter (e.g. "image/*")
  multiple?: boolean                // for layout=upload — allow multi-file picker
}

export interface ChatDef {
  enabled: boolean
  context_aware?: boolean
  default_actions?: string[]
}

export interface AgentManifest {
  manifest_version: 1
  agent: {
    id: string
    name: string
    tagline?: string
    mcp_server: string              // MCP server name in flight-deck
  }
  entities: Record<string, EntityDef>
  feeds: Record<string, FeedDef>
  actions: Record<string, ActionDef>
  surfaces: Record<string, SurfaceDef>
  chat?: ChatDef
  home_surface?: string             // surface id to land on
}
