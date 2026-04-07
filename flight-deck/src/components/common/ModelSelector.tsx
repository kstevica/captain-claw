import { useState, useEffect, useCallback } from 'react'
import { Cpu, Check, Loader2, Pencil, KeyRound } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'

interface ModelSelectorProps {
  kind: 'docker' | 'process'
  identifier: string  // container id or process slug
  onModelChange?: (provider: string, model: string) => void
}

interface ModelInfo {
  provider: string
  model: string
  api_key: string
}

const KNOWN_PROVIDERS = [
  'ollama', 'anthropic', 'openai', 'gemini', 'groq', 'mistral',
  'deepseek', 'openrouter', 'xai', 'together', 'cerebras', 'sambanova', 'fireworks',
  'litert',
]

function parseModelFromYaml(yaml: string): ModelInfo {
  const info: ModelInfo = { provider: '', model: '', api_key: '' }
  // Simple YAML parser for the model section
  const lines = yaml.split('\n')
  let inModel = false
  for (const line of lines) {
    const trimmed = line.trimStart()
    // Check if we're entering/leaving the model section
    if (/^model\s*:/.test(trimmed)) {
      inModel = true
      continue
    }
    if (inModel && trimmed.length > 0 && !trimmed.startsWith(' ') && !trimmed.startsWith('\t') && !trimmed.startsWith('#')) {
      // Left the model section (new top-level key)
      inModel = false
    }
    if (!inModel) continue

    const providerMatch = trimmed.match(/^\s*provider\s*:\s*["']?([^"'\s#]+)/)
    if (providerMatch) info.provider = providerMatch[1]

    const modelMatch = trimmed.match(/^\s*model\s*:\s*["']?([^"'\s#]+)/)
    if (modelMatch) info.model = modelMatch[1]

    const keyMatch = trimmed.match(/^\s*api_key\s*:\s*["']?([^"'#]*)/)
    if (keyMatch) info.api_key = keyMatch[1].trim()
  }
  return info
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...(init?.headers as Record<string, string> || {}) }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`

  let res = await fetch(`/fd${path}`, { ...init, headers, credentials: 'include' })
  if (res.status === 401 && authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) {
      const h2: Record<string, string> = { 'Content-Type': 'application/json', ...(init?.headers as Record<string, string> || {}) }
      const t2 = useAuthStore.getState().token
      if (t2) h2['Authorization'] = `Bearer ${t2}`
      res = await fetch(`/fd${path}`, { ...init, headers: h2, credentials: 'include' })
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

export function ModelSelector({ kind, identifier, onModelChange }: ModelSelectorProps) {
  const [editing, setEditing] = useState(false)
  const [provider, setProvider] = useState('')
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)
  const [originalProvider, setOriginalProvider] = useState('')
  const [originalModel, setOriginalModel] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fdFetch<{ config_yaml: string }>(`/agent-config/${kind}/${identifier}`)
      const info = parseModelFromYaml(data.config_yaml || '')
      setProvider(info.provider)
      setModel(info.model)
      setApiKey(info.api_key)
      setOriginalProvider(info.provider)
      setOriginalModel(info.model)
    } catch {
      // Silently fail — card still works
    } finally {
      setLoading(false)
    }
  }, [kind, identifier])

  useEffect(() => { load() }, [load])

  const handleSave = async () => {
    if (!provider.trim() || !model.trim()) return
    setSaving(true)
    setError('')
    try {
      // Use the dedicated model endpoint which patches all 3 config locations
      const body: Record<string, string> = {
        provider: provider.trim(),
        model: model.trim(),
      }
      // Only send api_key if it was changed (non-empty means user typed something)
      if (showApiKey && apiKey.trim()) {
        body.api_key = apiKey.trim()
      }
      await fdFetch<{ ok: boolean }>(`/agent-model/${kind}/${identifier}`, {
        method: 'PUT',
        body: JSON.stringify(body),
      })
      setOriginalProvider(provider.trim())
      setOriginalModel(model.trim())
      setEditing(false)
      setShowApiKey(false)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
      onModelChange?.(provider.trim(), model.trim())
    } catch (e: any) {
      setError(e.message || 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  const handleCancel = () => {
    setProvider(originalProvider)
    setModel(originalModel)
    setApiKey('')
    setShowApiKey(false)
    setEditing(false)
    setError('')
  }

  if (loading) {
    return (
      <div className="mb-3 flex items-center gap-2 text-xs text-zinc-600">
        <Cpu className="h-3 w-3" />
        <span>Loading model...</span>
      </div>
    )
  }

  if (!editing) {
    return (
      <div className="mb-3 group/model flex items-center gap-2">
        <Cpu className="h-3 w-3 text-zinc-500 shrink-0" />
        <span className="text-xs text-zinc-400">
          {originalProvider ? (
            <>
              <span className="text-zinc-500">{originalProvider}</span>
              <span className="text-zinc-600 mx-1">/</span>
              <span className="text-zinc-300">{originalModel || '—'}</span>
            </>
          ) : (
            <span className="text-zinc-600 italic">No model override set</span>
          )}
        </span>
        {saved && <Check className="h-3 w-3 text-emerald-400" />}
        {saved && <span className="text-[10px] text-emerald-500">saved — restart agent</span>}
        <button
          onClick={() => setEditing(true)}
          className="rounded p-0.5 text-zinc-600 opacity-0 transition-opacity group-hover/model:opacity-100 hover:bg-zinc-800 hover:text-zinc-300"
        >
          <Pencil className="h-3 w-3" />
        </button>
      </div>
    )
  }

  return (
    <div className="mb-3 space-y-2 rounded-lg border border-zinc-700/50 bg-zinc-950/50 p-2.5">
      <div className="flex items-center gap-1.5">
        <Cpu className="h-3 w-3 text-violet-400 shrink-0" />
        <span className="text-xs font-medium text-zinc-300">Model Configuration</span>
      </div>

      <div className="flex items-center gap-2">
        {/* Provider */}
        <div className="flex-1 min-w-0">
          <label className="text-[10px] text-zinc-500 uppercase tracking-wider mb-0.5 block">Provider</label>
          <input
            list="model-providers"
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            placeholder="ollama"
            className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
          <datalist id="model-providers">
            {KNOWN_PROVIDERS.map(p => <option key={p} value={p} />)}
          </datalist>
        </div>

        {/* Model */}
        <div className="flex-[2] min-w-0">
          <label className="text-[10px] text-zinc-500 uppercase tracking-wider mb-0.5 block">Model</label>
          <input
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="claude-sonnet-4-20250514"
            className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
        </div>
      </div>

      {/* API Key toggle + input */}
      <div>
        <button
          onClick={() => setShowApiKey(!showApiKey)}
          className={`flex items-center gap-1 text-[11px] transition-colors ${
            showApiKey ? 'text-violet-400' : 'text-zinc-500 hover:text-zinc-400'
          }`}
        >
          <KeyRound className="h-3 w-3" />
          {showApiKey ? 'Hide API key' : 'Change API key'}
        </button>
        {showApiKey && (
          <input
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            type="password"
            placeholder="sk-..."
            className="mt-1 w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none font-mono"
          />
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="text-[11px] text-red-400">{error}</div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-1.5 pt-0.5">
        <button
          onClick={handleSave}
          disabled={saving || !provider.trim() || !model.trim()}
          className="flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-violet-500 disabled:opacity-40 transition-colors"
        >
          {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
          Save
        </button>
        <button
          onClick={handleCancel}
          className="rounded-md px-2.5 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}
