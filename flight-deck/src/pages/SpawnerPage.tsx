import { useState, useEffect } from 'react'
import { Server, Plus, Trash2, Save, FolderOpen, Copy, X, FileDown, ClipboardCopy, ChevronDown, ChevronRight, Rocket, Cpu, Menu } from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import { useIsMobile } from '../hooks/useMediaQuery'
import { useConnectionStore } from '../stores/connectionStore'
import { useContainerStore } from '../stores/containerStore'
import { useProcessStore } from '../stores/processStore'
import { spawnAgent, spawnProcess, type SpawnConfig } from '../services/docker'

// ── Agent config model ──

interface AgentConfig {
  // Identity
  name: string
  description: string
  hostname: string
  image: string

  // LLM
  provider: string
  model: string
  temperature: number
  maxTokens: number
  providerApiKey: string

  // BotPort
  botportEnabled: boolean
  botportUrl: string
  botportInstanceName: string
  botportKey: string
  botportSecret: string
  botportMaxConcurrent: number

  // Tools
  tools: string[]

  // Web
  webEnabled: boolean
  webPort: number
  webAuthToken: string

  // Platforms
  telegramEnabled: boolean
  telegramBotToken: string
  discordEnabled: boolean
  discordBotToken: string
  slackEnabled: boolean
  slackBotToken: string

  // Cognitive mode
  cognitiveMode: string

  // Docker
  networkMode: string
  restartPolicy: string
  extraVolumes: { host: string; container: string }[]
  envVars: { key: string; value: string }[]
}

interface SavedPreset {
  id: string
  label: string
  config: AgentConfig
  savedAt: string
}

const PRESETS_KEY = 'fd:agent-presets'

const ALL_TOOLS = [
  'shell', 'read', 'write', 'glob', 'edit',
  'web_fetch', 'web_search', 'browser',
  'pdf_extract', 'docx_extract', 'xlsx_extract', 'pptx_extract',
  'image_gen', 'image_ocr', 'image_vision',
  'pocket_tts', 'send_mail',
  'google_drive', 'google_calendar', 'google_mail',
  'todo', 'contacts', 'scripts', 'apis', 'playbooks',
  'typesense', 'datastore', 'personality', 'insights',
  'screen_capture', 'desktop_action', 'direct_api',
  'cron', 'twitter', 'summarize_files', 'termux', 'pinchtab', 'gws',
  'botport',
]

const defaultConfig: AgentConfig = {
  name: '',
  description: '',
  hostname: 'captain-claw',
  image: 'kstevica/captain-claw:latest',
  provider: 'ollama',
  model: 'minimax-m2.7:cloud',
  temperature: 0.7,
  maxTokens: 32768,
  providerApiKey: '',
  botportEnabled: false,
  botportUrl: '',
  botportInstanceName: '',
  botportKey: '',
  botportSecret: '',
  botportMaxConcurrent: 5,
  tools: [
    'shell', 'read', 'write', 'glob', 'edit', 'web_fetch', 'web_search', 'browser',
    'pdf_extract', 'docx_extract', 'xlsx_extract', 'pptx_extract',
    'scripts', 'playbooks', 'personality',
  ],
  webEnabled: true,
  webPort: 24080,
  webAuthToken: '',
  telegramEnabled: false,
  telegramBotToken: '',
  discordEnabled: false,
  discordBotToken: '',
  slackEnabled: false,
  slackBotToken: '',
  cognitiveMode: 'neutra',
  networkMode: 'host',
  restartPolicy: 'unless-stopped',
  extraVolumes: [],
  envVars: [],
}

function loadPresets(): SavedPreset[] {
  try { return JSON.parse(localStorage.getItem(PRESETS_KEY) || '[]') } catch { return [] }
}
function persistPresets(presets: SavedPreset[]) {
  localStorage.setItem(PRESETS_KEY, JSON.stringify(presets))
}

// ── Generators ──

function generateConfigYaml(c: AgentConfig): string {
  const tools = c.tools.length > 0
    ? c.tools.map((t) => `  - ${t}`).join('\n')
    : '  - shell\n  - read\n  - write'

  return `model:
  provider: ${c.provider}
  model: ${c.model}
  temperature: ${c.temperature}
  max_tokens: ${c.maxTokens}
  api_key: '${c.providerApiKey}'
  base_url: ''
context:
  max_tokens: 160000
  compaction_threshold: 0.8
  compaction_ratio: 0.4
memory:
  enabled: true
  path: /home/claw/.captain-claw/memory.db
  index_workspace: true
  index_sessions: true
  embeddings:
    provider: auto
    ollama_model: nomic-embed-text
    ollama_base_url: http://127.0.0.1:11434
    fallback_to_local_hash: true
tools:
  enabled:
${tools}
  shell:
    timeout: 120
    default_policy: ask
  browser:
    headless: true
    viewport_width: 1280
    viewport_height: 720
  web_search:
    provider: brave
    max_results: 5
  require_confirmation:
  - shell
  - write
  - edit
session:
  storage: sqlite
  path: /data/sessions/sessions.db
  auto_save: true
workspace:
  path: /data/workspace
web:
  enabled: ${c.webEnabled}
  host: 0.0.0.0
  port: ${c.webPort}
  api_enabled: true
  auth_token: '${c.webAuthToken}'
botport:
  enabled: ${c.botportEnabled}
  url: '${c.botportUrl}'
  instance_name: ${c.botportInstanceName || c.name || 'default'}
  key: '${c.botportKey}'
  secret: '${c.botportSecret}'
  advertise_personas: true
  advertise_tools: true
  advertise_models: true
  max_concurrent: ${c.botportMaxConcurrent}
  reconnect_delay_seconds: 5.0
  heartbeat_interval_seconds: 30.0
telegram:
  enabled: ${c.telegramEnabled}
  bot_token: '${c.telegramBotToken}'
discord:
  enabled: ${c.discordEnabled}
  bot_token: '${c.discordBotToken}'
slack:
  enabled: ${c.slackEnabled}
  bot_token: '${c.slackBotToken}'
logging:
  level: INFO
  format: console
`
}

function generateDockerCompose(c: AgentConfig): string {
  const svcName = c.name || 'captain-claw'
  const slug = svcName.replace(/[^a-z0-9-]/gi, '-').toLowerCase()

  const volumes = [
    `      - ./${slug}/config.yaml:/app/config.yaml`,
    `      - ./${slug}/.env:/app/.env:ro`,
    `      - ./${slug}/data/home-config:/home/claw/.captain-claw`,
    `      - ./${slug}/data/workspace:/data/workspace`,
    `      - ./${slug}/data/sessions:/data/sessions`,
    `      - ./${slug}/data/skills:/data/skills`,
    ...c.extraVolumes
      .filter((v) => v.host && v.container)
      .map((v) => `      - ${v.host}:${v.container}`),
  ].join('\n')

  const envLines = c.envVars
    .filter((e) => e.key)
    .map((e) => `      ${e.key}: '${e.value}'`)
    .join('\n')
  const envSection = envLines ? `\n    environment:\n${envLines}` : ''

  return `services:
  ${slug}:
    image: ${c.image}
    hostname: ${c.hostname || slug}
    network_mode: ${c.networkMode}

    # Security hardening
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined  # needed for Chromium
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETUID
      - SETGID
      - SYS_CHROOT
    tmpfs:
      - /tmp
      - /run

    volumes:
${volumes}${envSection}
    restart: ${c.restartPolicy}
    stop_grace_period: 5s
`
}

function generateEnvFile(c: AgentConfig): string {
  const lines: string[] = []
  if (c.providerApiKey) lines.push(`ANTHROPIC_API_KEY=${c.providerApiKey}`)
  c.envVars.filter((e) => e.key).forEach((e) => lines.push(`${e.key}=${e.value}`))
  return lines.join('\n') + '\n'
}

// ── Component ──

export function SpawnerPage() {
  const globalBotportUrl = useConnectionStore((s) => s.botportUrl)
  const { fetchContainers, dockerAvailable, checkHealth } = useContainerStore()
  const { fetchProcesses } = useProcessStore()
  const { isMobile, isTablet } = useIsMobile()
  const compact = isMobile || isTablet
  const [presetsOpen, setPresetsOpen] = useState(false)
  const [spawnMode, setSpawnMode] = useState<'docker' | 'process'>('process')
  const [config, setConfig] = useState<AgentConfig>(() => ({
    ...defaultConfig,
    botportUrl: globalBotportUrl ? globalBotportUrl.replace(/^http/, 'ws') + '/ws' : '',
  }))
  const [presets, setPresets] = useState<SavedPreset[]>(loadPresets)
  const [activePresetId, setActivePresetId] = useState<string | null>(null)
  const [saveLabel, setSaveLabel] = useState('')
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [showOutput, setShowOutput] = useState<'compose' | 'config' | 'env' | null>(null)
  const [spawning, setSpawning] = useState(false)
  const [spawnResult, setSpawnResult] = useState<{ ok: boolean; message: string } | null>(null)
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    identity: true, llm: true, botport: false, tools: false, web: false, platforms: false, docker: false, env: false,
  })

  useEffect(() => { checkHealth() }, [checkHealth])

  useEffect(() => { persistPresets(presets) }, [presets])

  const update = <K extends keyof AgentConfig>(key: K, value: AgentConfig[K]) =>
    setConfig((prev) => ({ ...prev, [key]: value }))

  const toggleSection = (key: string) =>
    setExpandedSections((prev) => ({ ...prev, [key]: !prev[key] }))

  const toggleTool = (tool: string) => {
    const tools = config.tools.includes(tool)
      ? config.tools.filter((t) => t !== tool)
      : [...config.tools, tool]
    update('tools', tools)
  }

  // ── Preset management ──

  const handleSave = () => {
    const label = saveLabel.trim() || config.name || 'Untitled Agent'
    if (activePresetId && presets.find((p) => p.id === activePresetId)) {
      setPresets(presets.map((p) =>
        p.id === activePresetId ? { ...p, label, config: { ...config }, savedAt: new Date().toISOString() } : p
      ))
    } else {
      const preset: SavedPreset = { id: crypto.randomUUID(), label, config: { ...config }, savedAt: new Date().toISOString() }
      setPresets([...presets, preset])
      setActivePresetId(preset.id)
    }
    setShowSaveDialog(false)
    setSaveLabel('')
  }

  const handleLoadPreset = (preset: SavedPreset) => {
    setConfig({ ...defaultConfig, ...preset.config })
    setActivePresetId(preset.id)
  }

  const handleDeletePreset = (id: string) => {
    setPresets(presets.filter((p) => p.id !== id))
    if (activePresetId === id) setActivePresetId(null)
  }

  const handleDuplicate = (preset: SavedPreset) => {
    const dup: SavedPreset = {
      id: crypto.randomUUID(),
      label: `${preset.label} (copy)`,
      config: { ...preset.config },
      savedAt: new Date().toISOString(),
    }
    setPresets([...presets, dup])
  }

  const handleNewAgent = () => {
    setConfig({ ...defaultConfig, botportUrl: globalBotportUrl ? globalBotportUrl.replace(/^http/, 'ws') + '/ws' : '' })
    setActivePresetId(null)
  }

  const openSaveDialog = () => {
    const existing = activePresetId ? presets.find((p) => p.id === activePresetId) : null
    setSaveLabel(existing?.label || config.name || '')
    setShowSaveDialog(true)
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const downloadFile = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const isDirty = activePresetId
    ? JSON.stringify(config) !== JSON.stringify(presets.find((p) => p.id === activePresetId)?.config)
    : config.name !== ''

  // ── Render ──

  return (
    <div className="flex h-full">
      {/* Saved presets sidebar — hidden on mobile, shown as overlay */}
      {compact ? (
        presetsOpen && (
          <>
            <div className="fixed inset-0 z-40 bg-black/50" onClick={() => setPresetsOpen(false)} />
            <div className="fixed left-0 top-0 bottom-0 z-50 w-64 border-r border-zinc-800 bg-zinc-900 flex flex-col">
              <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2.5">
                <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">Saved Agents</span>
                <div className="flex gap-1">
                  <button onClick={handleNewAgent} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="New blank agent">
                    <Plus className="h-3.5 w-3.5" />
                  </button>
                  <button onClick={() => setPresetsOpen(false)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-2">
                {presets.map((preset) => (
                  <div
                    key={preset.id}
                    className={`group mb-1 rounded-lg px-2.5 py-2 cursor-pointer transition-colors ${
                      activePresetId === preset.id ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
                    }`}
                    onClick={() => { handleLoadPreset(preset); setPresetsOpen(false) }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium truncate">{preset.label}</span>
                      <div className="flex gap-0.5">
                        <button onClick={(e) => { e.stopPropagation(); handleDuplicate(preset) }} className="rounded p-0.5 text-zinc-500 hover:text-zinc-300" title="Duplicate"><Copy className="h-3 w-3" /></button>
                        <button onClick={(e) => { e.stopPropagation(); handleDeletePreset(preset.id) }} className="rounded p-0.5 text-zinc-500 hover:text-red-400" title="Delete"><Trash2 className="h-3 w-3" /></button>
                      </div>
                    </div>
                    <div className="mt-0.5 text-xs text-zinc-600 truncate">{preset.config.provider}:{preset.config.model}</div>
                  </div>
                ))}
                {presets.length === 0 && (
                  <p className="px-2 py-6 text-center text-xs text-zinc-600">No saved agents yet.</p>
                )}
              </div>
            </div>
          </>
        )
      ) : (
        <div className="w-56 shrink-0 border-r border-zinc-800 bg-zinc-900/30 flex flex-col">
          <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2.5">
            <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">Saved Agents</span>
            <button onClick={handleNewAgent} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="New blank agent">
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {presets.map((preset) => (
              <div
                key={preset.id}
                className={`group mb-1 rounded-lg px-2.5 py-2 cursor-pointer transition-colors ${
                  activePresetId === preset.id ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
                }`}
                onClick={() => handleLoadPreset(preset)}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium truncate">{preset.label}</span>
                  <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button onClick={(e) => { e.stopPropagation(); handleDuplicate(preset) }} className="rounded p-0.5 text-zinc-500 hover:text-zinc-300" title="Duplicate"><Copy className="h-3 w-3" /></button>
                    <button onClick={(e) => { e.stopPropagation(); handleDeletePreset(preset.id) }} className="rounded p-0.5 text-zinc-500 hover:text-red-400" title="Delete"><Trash2 className="h-3 w-3" /></button>
                  </div>
                </div>
                <div className="mt-0.5 text-xs text-zinc-600 truncate">{preset.config.provider}:{preset.config.model}</div>
              </div>
            ))}
            {presets.length === 0 && (
              <p className="px-2 py-6 text-center text-xs text-zinc-600">No saved agents yet.</p>
            )}
          </div>
        </div>
      )}

      {/* Config form */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="mx-auto max-w-2xl">
          {/* Header */}
          <div className="mb-4 flex items-start justify-between gap-2 md:mb-6">
            <div className="min-w-0">
              <h1 className="text-lg font-semibold truncate">
                {activePresetId ? presets.find((p) => p.id === activePresetId)?.label || 'Edit Agent' : 'New Agent'}
              </h1>
              <p className="text-xs text-zinc-500 sm:text-sm">
                Configure a Captain Claw instance — {spawnMode === 'docker' ? 'Docker container' : 'local process (pip)'}
              </p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {compact && (
                <button onClick={() => setPresetsOpen(true)} className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800 transition-colors">
                  <Menu className="h-3.5 w-3.5" /> Presets
                </button>
              )}
              <button onClick={openSaveDialog} className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800 transition-colors">
                <Save className="h-3.5 w-3.5" /> {activePresetId ? 'Update' : 'Save'}
              </button>
            </div>
          </div>

          {/* Spawn mode toggle */}
          {useAuthStore.getState().dockerSpawnEnabled ? (
            <div className="mb-5 flex rounded-lg border border-zinc-700 overflow-hidden">
              <button
                onClick={() => setSpawnMode('process')}
                className={`flex flex-1 items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors ${
                  spawnMode === 'process'
                    ? 'bg-violet-600 text-white'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
                }`}
              >
                <Cpu className="h-3.5 w-3.5" /> Local Process (pip)
              </button>
              <button
                onClick={() => setSpawnMode('docker')}
                className={`flex flex-1 items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors ${
                  spawnMode === 'docker'
                    ? 'bg-violet-600 text-white'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
                }`}
              >
                <Server className="h-3.5 w-3.5" /> Docker Container
              </button>
            </div>
          ) : (
            <div className="mb-5 rounded-lg border border-zinc-700 px-4 py-2.5 text-center text-sm text-zinc-500">
              <Cpu className="h-3.5 w-3.5 inline mr-1.5" />Local Process (pip) — Docker spawning is disabled by admin
            </div>
          )}

          {/* Save dialog */}
          {showSaveDialog && (
            <div className="mb-5 rounded-xl border border-zinc-700 bg-zinc-900 p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium">{activePresetId ? 'Update saved agent' : 'Save as new agent preset'}</span>
                <button onClick={() => setShowSaveDialog(false)} className="rounded p-1 text-zinc-500 hover:text-zinc-300"><X className="h-3.5 w-3.5" /></button>
              </div>
              <div className="flex gap-2">
                <input value={saveLabel} onChange={(e) => setSaveLabel(e.target.value)} placeholder="Preset name" className="input flex-1" autoFocus onKeyDown={(e) => e.key === 'Enter' && handleSave()} />
                <button onClick={handleSave} className="rounded-lg bg-violet-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-violet-500">Save</button>
              </div>
              {activePresetId && (
                <button onClick={() => { setActivePresetId(null); handleSave() }} className="mt-2 text-xs text-zinc-500 hover:text-zinc-300">
                  or save as new preset instead
                </button>
              )}
            </div>
          )}

          {isDirty && activePresetId && (
            <div className="mb-4 flex items-center gap-2 rounded-lg bg-amber-500/10 border border-amber-500/20 px-3 py-2 text-xs text-amber-400">
              <FolderOpen className="h-3.5 w-3.5 shrink-0" /> Unsaved changes
            </div>
          )}

          {/* ── Sections ── */}
          <div className="space-y-1">

            {/* Identity */}
            <Section title="Identity & Image" expanded={expandedSections.identity} onToggle={() => toggleSection('identity')}>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Field label="Agent Name">
                  <input value={config.name} onChange={(e) => update('name', e.target.value)} placeholder="e.g. research-agent" className="input" />
                </Field>
                <Field label="Hostname">
                  <input value={config.hostname} onChange={(e) => update('hostname', e.target.value)} placeholder="captain-claw" className="input" />
                </Field>
              </div>
              <Field label="Description">
                <input value={config.description} onChange={(e) => update('description', e.target.value)} placeholder="What this agent does..." className="input" />
              </Field>
              {spawnMode === 'docker' && (
                <Field label="Docker Image">
                  <input value={config.image} onChange={(e) => update('image', e.target.value)} className="input" />
                </Field>
              )}
            </Section>

            {/* LLM */}
            <Section title="LLM Model" expanded={expandedSections.llm} onToggle={() => toggleSection('llm')}>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Field label="Provider">
                  <select value={config.provider} onChange={(e) => update('provider', e.target.value)} className="input">
                    <option value="anthropic">Anthropic</option>
                    <option value="ollama">Ollama</option>
                    <option value="openai">OpenAI</option>
                    <option value="gemini">Gemini</option>
                    <option value="openrouter">OpenRouter</option>
                    <option value="xai">xAI</option>
                  </select>
                </Field>
                <Field label="Model ID">
                  <input value={config.model} onChange={(e) => update('model', e.target.value)} placeholder="model name" className="input" />
                </Field>
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Field label="Temperature">
                  <input type="number" step="0.1" min="0" max="2" value={config.temperature} onChange={(e) => update('temperature', parseFloat(e.target.value) || 0)} className="input" />
                </Field>
                <Field label="Max Tokens">
                  <input type="number" value={config.maxTokens} onChange={(e) => update('maxTokens', parseInt(e.target.value) || 0)} className="input" />
                </Field>
              </div>
              <Field label="API Key" hint="Stored in .env file, not in config.yaml">
                <input type="password" value={config.providerApiKey} onChange={(e) => update('providerApiKey', e.target.value)} placeholder="sk-..." className="input font-mono text-xs" />
              </Field>
              <Field label="Cognitive Mode" hint="How the agent thinks — reasoning strategy">
                <select value={config.cognitiveMode} onChange={(e) => update('cognitiveMode', e.target.value)} className="input">
                  <option value="neutra">Neutra — Default (balanced)</option>
                  <option value="ionian">Ionian — The Resolver (convergent, decisive)</option>
                  <option value="dorian">Dorian — The Pragmatic Empath (tradeoff-aware)</option>
                  <option value="phrygian">Phrygian — The Adversarial Analyst (threat modeling)</option>
                  <option value="lydian">Lydian — The Visionary Explorer (creative, divergent)</option>
                  <option value="mixolydian">Mixolydian — The Iterative Builder (ship & improve)</option>
                  <option value="aeolian">Aeolian — The Depth Researcher (thorough analysis)</option>
                  <option value="locrian">Locrian — The Deconstructionist (challenges premises)</option>
                </select>
              </Field>
            </Section>

            {/* BotPort */}
            <Section title="BotPort Connection" expanded={expandedSections.botport} onToggle={() => toggleSection('botport')}>
              <label className="mb-3 flex items-center gap-2 text-sm">
                <input type="checkbox" checked={config.botportEnabled} onChange={(e) => update('botportEnabled', e.target.checked)} className="accent-violet-500" />
                <span className="text-zinc-300">Enable BotPort connection</span>
              </label>
              {config.botportEnabled && (
                <>
                  <Field label="BotPort WebSocket URL">
                    <input value={config.botportUrl} onChange={(e) => update('botportUrl', e.target.value)} placeholder="ws://host.docker.internal:23180/ws" className="input font-mono text-xs" />
                  </Field>
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <Field label="Instance Name">
                      <input value={config.botportInstanceName} onChange={(e) => update('botportInstanceName', e.target.value)} placeholder={config.name || 'default'} className="input" />
                    </Field>
                    <Field label="Max Concurrent">
                      <input type="number" min="1" max="20" value={config.botportMaxConcurrent} onChange={(e) => update('botportMaxConcurrent', parseInt(e.target.value) || 5)} className="input" />
                    </Field>
                  </div>
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <Field label="Key">
                      <input value={config.botportKey} onChange={(e) => update('botportKey', e.target.value)} className="input font-mono text-xs" />
                    </Field>
                    <Field label="Secret">
                      <input type="password" value={config.botportSecret} onChange={(e) => update('botportSecret', e.target.value)} className="input font-mono text-xs" />
                    </Field>
                  </div>
                </>
              )}
            </Section>

            {/* Tools */}
            <Section title={`Tools (${config.tools.length})`} expanded={expandedSections.tools} onToggle={() => toggleSection('tools')}>
              <div className="flex flex-wrap gap-1.5">
                {ALL_TOOLS.map((tool) => (
                  <button
                    key={tool}
                    onClick={() => toggleTool(tool)}
                    className={`rounded-md px-2 py-1 text-xs transition-colors ${
                      config.tools.includes(tool)
                        ? 'bg-violet-600/20 text-violet-300 border border-violet-500/30'
                        : 'bg-zinc-800/50 text-zinc-500 border border-zinc-800 hover:text-zinc-300'
                    }`}
                  >
                    {tool}
                  </button>
                ))}
              </div>
              <div className="mt-2 flex gap-2">
                <button onClick={() => update('tools', [...ALL_TOOLS])} className="text-xs text-zinc-500 hover:text-zinc-300">Select all</button>
                <button onClick={() => update('tools', [])} className="text-xs text-zinc-500 hover:text-zinc-300">Clear all</button>
              </div>
            </Section>

            {/* Web */}
            <Section title="Web Server" expanded={expandedSections.web} onToggle={() => toggleSection('web')}>
              <label className="mb-3 flex items-center gap-2 text-sm">
                <input type="checkbox" checked={config.webEnabled} onChange={(e) => update('webEnabled', e.target.checked)} className="accent-violet-500" />
                <span className="text-zinc-300">Enable web interface</span>
              </label>
              {config.webEnabled && (
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Field label="Port">
                    <input type="number" value={config.webPort} onChange={(e) => update('webPort', parseInt(e.target.value) || 24080)} className="input" />
                  </Field>
                  <Field label="Auth Token">
                    <input type="password" value={config.webAuthToken} onChange={(e) => update('webAuthToken', e.target.value)} placeholder="optional" className="input font-mono text-xs" />
                  </Field>
                </div>
              )}
            </Section>

            {/* Platforms */}
            <Section title="Platforms" expanded={expandedSections.platforms} onToggle={() => toggleSection('platforms')}>
              <PlatformToggle label="Telegram" enabled={config.telegramEnabled} onToggle={(v) => update('telegramEnabled', v)} tokenValue={config.telegramBotToken} onTokenChange={(v) => update('telegramBotToken', v)} />
              <PlatformToggle label="Discord" enabled={config.discordEnabled} onToggle={(v) => update('discordEnabled', v)} tokenValue={config.discordBotToken} onTokenChange={(v) => update('discordBotToken', v)} />
              <PlatformToggle label="Slack" enabled={config.slackEnabled} onToggle={(v) => update('slackEnabled', v)} tokenValue={config.slackBotToken} onTokenChange={(v) => update('slackBotToken', v)} />
            </Section>

            {/* Docker — only shown in docker mode */}
            {spawnMode === 'docker' && (
              <Section title="Docker Settings" expanded={expandedSections.docker} onToggle={() => toggleSection('docker')}>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Field label="Network Mode">
                    <select value={config.networkMode} onChange={(e) => update('networkMode', e.target.value)} className="input">
                      <option value="host">host</option>
                      <option value="bridge">bridge</option>
                    </select>
                  </Field>
                  <Field label="Restart Policy">
                    <select value={config.restartPolicy} onChange={(e) => update('restartPolicy', e.target.value)} className="input">
                      <option value="unless-stopped">unless-stopped</option>
                      <option value="always">always</option>
                      <option value="on-failure">on-failure</option>
                      <option value="no">no</option>
                    </select>
                  </Field>
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between">
                    <label className="text-sm font-medium text-zinc-300">Extra Volumes</label>
                    <button onClick={() => update('extraVolumes', [...config.extraVolumes, { host: '', container: '' }])} className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
                      <Plus className="h-3 w-3" /> Add
                    </button>
                  </div>
                  {config.extraVolumes.map((vol, idx) => (
                    <div key={idx} className="mb-1.5 flex gap-2">
                      <input value={vol.host} onChange={(e) => update('extraVolumes', config.extraVolumes.map((v, i) => i === idx ? { ...v, host: e.target.value } : v))} placeholder="./host/path" className="input flex-1 font-mono text-xs" />
                      <span className="self-center text-zinc-600">:</span>
                      <input value={vol.container} onChange={(e) => update('extraVolumes', config.extraVolumes.map((v, i) => i === idx ? { ...v, container: e.target.value } : v))} placeholder="/container/path" className="input flex-1 font-mono text-xs" />
                      <button onClick={() => update('extraVolumes', config.extraVolumes.filter((_, i) => i !== idx))} className="rounded p-1.5 text-zinc-600 hover:bg-zinc-800 hover:text-red-400"><Trash2 className="h-3.5 w-3.5" /></button>
                    </div>
                  ))}
                </div>
              </Section>
            )}

            {/* Env vars */}
            <Section title="Environment Variables" expanded={expandedSections.env} onToggle={() => toggleSection('env')}>
              <div className="mb-2 flex justify-end">
                <button onClick={() => update('envVars', [...config.envVars, { key: '', value: '' }])} className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
                  <Plus className="h-3 w-3" /> Add
                </button>
              </div>
              {config.envVars.map((env, idx) => (
                <div key={idx} className="mb-1.5 flex gap-2">
                  <input value={env.key} onChange={(e) => update('envVars', config.envVars.map((v, i) => i === idx ? { ...v, key: e.target.value } : v))} placeholder="KEY" className="input flex-1 font-mono text-xs" />
                  <input value={env.value} onChange={(e) => update('envVars', config.envVars.map((v, i) => i === idx ? { ...v, value: e.target.value } : v))} placeholder="value" className="input flex-[2] text-xs" />
                  <button onClick={() => update('envVars', config.envVars.filter((_, i) => i !== idx))} className="rounded p-1.5 text-zinc-600 hover:bg-zinc-800 hover:text-red-400"><Trash2 className="h-3.5 w-3.5" /></button>
                </div>
              ))}
            </Section>
          </div>

          {/* ── Generate buttons ── */}
          <div className={`mt-6 grid gap-3 ${spawnMode === 'docker' ? 'grid-cols-1 sm:grid-cols-3' : 'grid-cols-1 sm:grid-cols-2'}`}>
            {spawnMode === 'docker' && (
              <OutputButton label="docker-compose.yml" active={showOutput === 'compose'} onClick={() => setShowOutput(showOutput === 'compose' ? null : 'compose')} />
            )}
            <OutputButton label="config.yaml" active={showOutput === 'config'} onClick={() => setShowOutput(showOutput === 'config' ? null : 'config')} />
            <OutputButton label=".env" active={showOutput === 'env'} onClick={() => setShowOutput(showOutput === 'env' ? null : 'env')} />
          </div>

          {/* Output preview */}
          {showOutput && (
            <div className="mt-4 rounded-xl border border-zinc-700 bg-zinc-950 overflow-hidden">
              <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
                <span className="text-xs font-medium text-zinc-400">
                  {showOutput === 'compose' ? 'docker-compose.yml' : showOutput === 'config' ? 'config.yaml' : '.env'}
                </span>
                <div className="flex gap-1.5">
                  <button
                    onClick={() => copyToClipboard(
                      showOutput === 'compose' ? generateDockerCompose(config) :
                      showOutput === 'config' ? generateConfigYaml(config) :
                      generateEnvFile(config)
                    )}
                    className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                  >
                    <ClipboardCopy className="h-3 w-3" /> Copy
                  </button>
                  <button
                    onClick={() => downloadFile(
                      showOutput === 'compose' ? 'docker-compose.yml' : showOutput === 'config' ? 'config.yaml' : '.env',
                      showOutput === 'compose' ? generateDockerCompose(config) :
                      showOutput === 'config' ? generateConfigYaml(config) :
                      generateEnvFile(config)
                    )}
                    className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                  >
                    <FileDown className="h-3 w-3" /> Download
                  </button>
                </div>
              </div>
              <pre className="max-h-80 overflow-auto p-4 text-xs text-zinc-300 font-mono leading-relaxed">
                {showOutput === 'compose' ? generateDockerCompose(config) :
                 showOutput === 'config' ? generateConfigYaml(config) :
                 generateEnvFile(config)}
              </pre>
            </div>
          )}

          {/* Spawn result */}
          {spawnResult && (
            <div className={`mt-4 rounded-lg border px-4 py-3 text-sm ${
              spawnResult.ok
                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                : 'border-red-500/30 bg-red-500/10 text-red-400'
            }`}>
              {spawnResult.message}
            </div>
          )}

          {/* Spawn + Download buttons */}
          <div className="mt-6 flex gap-3">
            <button
              className={`flex flex-1 items-center justify-center gap-2 rounded-xl py-3 text-sm font-semibold text-white disabled:opacity-50 transition-colors ${
                spawnMode === 'process' ? 'bg-emerald-600 hover:bg-emerald-500' : 'bg-violet-600 hover:bg-violet-500'
              }`}
              disabled={spawning || (spawnMode === 'docker' && !dockerAvailable)}
              onClick={async () => {
                setSpawning(true)
                setSpawnResult(null)
                try {
                  const payload: SpawnConfig = {
                    name: config.name,
                    description: config.description,
                    hostname: config.hostname,
                    image: config.image,
                    provider: config.provider,
                    model: config.model,
                    temperature: config.temperature,
                    max_tokens: config.maxTokens,
                    provider_api_key: config.providerApiKey,
                    botport_enabled: config.botportEnabled,
                    botport_url: config.botportUrl,
                    botport_instance_name: config.botportInstanceName,
                    botport_key: config.botportKey,
                    botport_secret: config.botportSecret,
                    botport_max_concurrent: config.botportMaxConcurrent,
                    tools: config.tools,
                    web_enabled: config.webEnabled,
                    web_port: config.webPort,
                    web_auth_token: config.webAuthToken,
                    telegram_enabled: config.telegramEnabled,
                    telegram_bot_token: config.telegramBotToken,
                    discord_enabled: config.discordEnabled,
                    discord_bot_token: config.discordBotToken,
                    slack_enabled: config.slackEnabled,
                    slack_bot_token: config.slackBotToken,
                    cognitive_mode: config.cognitiveMode,
                    network_mode: config.networkMode,
                    restart_policy: config.restartPolicy,
                    extra_volumes: config.extraVolumes,
                    env_vars: config.envVars,
                  }
                  if (spawnMode === 'process') {
                    const result = await spawnProcess(payload)
                    setSpawnResult({ ok: result.ok, message: result.message })
                    fetchProcesses()
                  } else {
                    const result = await spawnAgent(payload)
                    setSpawnResult({ ok: result.ok, message: `${result.message} (${result.container_id})` })
                    fetchContainers()
                  }
                } catch (e) {
                  setSpawnResult({ ok: false, message: `Failed: ${e instanceof Error ? e.message : String(e)}` })
                } finally {
                  setSpawning(false)
                }
              }}
            >
              {spawning ? (
                <>{spawnMode === 'process' ? <Cpu className="h-4 w-4 animate-pulse" /> : <Server className="h-4 w-4 animate-pulse" />} Spawning...</>
              ) : spawnMode === 'process' ? (
                <><Rocket className="h-4 w-4" /> Spawn Process</>
              ) : (
                <><Rocket className="h-4 w-4" /> {dockerAvailable ? 'Spawn Container' : 'Docker not available'}</>
              )}
            </button>
            <button
              className="flex items-center gap-2 rounded-xl border border-zinc-700 px-4 py-3 text-sm font-medium text-zinc-300 hover:bg-zinc-800 transition-colors"
              onClick={() => {
                const slug = (config.name || 'captain-claw').replace(/[^a-z0-9-]/gi, '-').toLowerCase()
                if (spawnMode === 'docker') {
                  downloadFile(`${slug}-docker-compose.yml`, generateDockerCompose(config))
                }
                setTimeout(() => downloadFile(`${slug}-config.yaml`, generateConfigYaml(config)), spawnMode === 'docker' ? 100 : 0)
                setTimeout(() => downloadFile(`${slug}-.env`, generateEnvFile(config)), spawnMode === 'docker' ? 200 : 100)
              }}
            >
              <FileDown className="h-4 w-4" /> Download Files
            </button>
          </div>
          {spawnMode === 'docker' && !dockerAvailable && (
            <p className="mt-2 text-xs text-zinc-600 text-center">
              Docker is not available. Switch to <button onClick={() => setSpawnMode('process')} className="text-emerald-500 hover:text-emerald-400 underline">Process mode</button> to spawn without Docker.
            </p>
          )}
          {spawnMode === 'process' && (
            <p className="mt-2 text-xs text-zinc-600 text-center">
              Requires <code className="text-zinc-500">captain-claw</code> installed via pip. Agent data stored in <code className="text-zinc-500">fd-data/{(config.name || 'agent').replace(/[^a-z0-9-]/gi, '-').toLowerCase()}/</code>
            </p>
          )}

          <style>{`
            .input {
              width: 100%;
              border-radius: 0.5rem;
              border: 1px solid #3f3f46;
              background: #09090b;
              padding: 0.5rem 0.75rem;
              font-size: 0.875rem;
              color: #e4e4e7;
            }
            .input::placeholder { color: #52525b; }
            .input:focus { outline: none; border-color: rgba(139, 92, 246, 0.5); }
            html.light .input {
              background: #ffffff;
              border-color: #d4d4d8;
              color: #18181b;
            }
            html.light .input::placeholder { color: #a1a1aa; }
            html.light .input:focus { border-color: rgba(139, 92, 246, 0.5); }
          `}</style>
        </div>
      </div>
    </div>
  )
}

// ── Sub-components ──

function Section({ title, expanded, onToggle, children }: { title: string; expanded: boolean; onToggle: () => void; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-zinc-800 overflow-hidden">
      <button onClick={onToggle} className="flex w-full items-center gap-2 px-4 py-3 text-sm font-medium text-zinc-200 hover:bg-zinc-900/50 transition-colors">
        {expanded ? <ChevronDown className="h-4 w-4 text-zinc-500" /> : <ChevronRight className="h-4 w-4 text-zinc-500" />}
        {title}
      </button>
      {expanded && <div className="space-y-4 border-t border-zinc-800/50 px-4 py-4">{children}</div>}
    </div>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1.5 block text-sm font-medium text-zinc-300">{label}</label>
      {children}
      {hint && <p className="mt-1 text-xs text-zinc-600">{hint}</p>}
    </div>
  )
}

function PlatformToggle({ label, enabled, onToggle, tokenValue, onTokenChange }: {
  label: string; enabled: boolean; onToggle: (v: boolean) => void; tokenValue: string; onTokenChange: (v: string) => void
}) {
  return (
    <div className="mb-3">
      <label className="mb-2 flex items-center gap-2 text-sm">
        <input type="checkbox" checked={enabled} onChange={(e) => onToggle(e.target.checked)} className="accent-violet-500" />
        <span className="text-zinc-300">{label}</span>
      </label>
      {enabled && (
        <input value={tokenValue} onChange={(e) => onTokenChange(e.target.value)} placeholder="Bot token" className="input font-mono text-xs" />
      )}
    </div>
  )
}

function OutputButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${
        active ? 'border-violet-500/50 bg-violet-500/10 text-violet-300' : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
      }`}
    >
      {label}
    </button>
  )
}
