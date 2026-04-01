import { useState, useEffect } from 'react'
import { X, Save, Loader2, AlertTriangle, FileText, KeyRound } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'

interface AgentConfigEditorProps {
  kind: 'docker' | 'process'
  identifier: string    // container id or process slug
  agentName: string
  onClose: () => void
}

async function fdFetchConfig<T>(path: string, init?: RequestInit): Promise<T> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`

  let res = await fetch(`/fd${path}`, { headers, credentials: 'include', ...init })
  if (res.status === 401 && authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) {
      const h2: Record<string, string> = { 'Content-Type': 'application/json' }
      const t2 = useAuthStore.getState().token
      if (t2) h2['Authorization'] = `Bearer ${t2}`
      res = await fetch(`/fd${path}`, { headers: h2, credentials: 'include', ...init })
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

export function AgentConfigEditor({ kind, identifier, agentName, onClose }: AgentConfigEditorProps) {
  const [configYaml, setConfigYaml] = useState('')
  const [env, setEnv] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [activeTab, setActiveTab] = useState<'config' | 'env'>('config')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    fdFetchConfig<{ config_yaml: string; env: string }>(`/agent-config/${kind}/${identifier}`)
      .then((data) => {
        if (cancelled) return
        setConfigYaml(data.config_yaml || '')
        setEnv(data.env || '')
      })
      .catch((e) => {
        if (cancelled) return
        setError(e.message || 'Failed to load config')
      })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [kind, identifier])

  const handleSave = async () => {
    setSaving(true)
    setError('')
    setSuccess('')
    try {
      const body: Record<string, string> = {}
      body.config_yaml = configYaml
      body.env = env
      const result = await fdFetchConfig<{ ok: boolean; message: string }>(`/agent-config/${kind}/${identifier}`, {
        method: 'PUT',
        body: JSON.stringify(body),
      })
      setSuccess(result.message || 'Saved. Restart agent for changes to take effect.')
      setTimeout(() => setSuccess(''), 5000)
    } catch (e: any) {
      setError(e.message || 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="w-[700px] max-h-[85vh] flex flex-col rounded-xl border border-zinc-700/50 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800">
          <div className="flex items-center gap-2.5">
            <FileText className="h-4 w-4 text-violet-400" />
            <span className="text-sm font-medium text-zinc-200">Config — {agentName}</span>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 transition-colors">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-0 border-b border-zinc-800">
          <button
            onClick={() => setActiveTab('config')}
            className={`px-4 py-2 text-xs font-medium transition-colors ${
              activeTab === 'config'
                ? 'text-violet-400 border-b-2 border-violet-400 bg-zinc-800/30'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <span className="flex items-center gap-1.5">
              <FileText className="h-3 w-3" /> config.yaml
            </span>
          </button>
          <button
            onClick={() => setActiveTab('env')}
            className={`px-4 py-2 text-xs font-medium transition-colors ${
              activeTab === 'env'
                ? 'text-violet-400 border-b-2 border-violet-400 bg-zinc-800/30'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <span className="flex items-center gap-1.5">
              <KeyRound className="h-3 w-3" /> .env
            </span>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
              <span className="ml-2 text-sm text-zinc-500">Loading config...</span>
            </div>
          ) : (
            <div className="h-[420px]">
              {activeTab === 'config' ? (
                <textarea
                  value={configYaml}
                  onChange={(e) => setConfigYaml(e.target.value)}
                  spellCheck={false}
                  className="w-full h-full resize-none bg-zinc-950/50 px-4 py-3 text-[13px] font-mono text-zinc-300 leading-relaxed focus:outline-none placeholder-zinc-700"
                  placeholder="# config.yaml — agent configuration&#10;provider: anthropic&#10;model: claude-sonnet-4-20250514&#10;..."
                />
              ) : (
                <textarea
                  value={env}
                  onChange={(e) => setEnv(e.target.value)}
                  spellCheck={false}
                  className="w-full h-full resize-none bg-zinc-950/50 px-4 py-3 text-[13px] font-mono text-zinc-300 leading-relaxed focus:outline-none placeholder-zinc-700"
                  placeholder="# .env — environment variables&#10;BRAVE_API_KEY=...&#10;ANTHROPIC_API_KEY=...&#10;..."
                />
              )}
            </div>
          )}
        </div>

        {/* Status messages */}
        {error && (
          <div className="mx-4 mb-2 flex items-center gap-2 rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-400">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0" /> {error}
          </div>
        )}
        {success && (
          <div className="mx-4 mb-2 flex items-center gap-2 rounded-lg bg-emerald-500/10 px-3 py-2 text-xs text-emerald-400">
            <Save className="h-3.5 w-3.5 shrink-0" /> {success}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-zinc-800">
          <span className="text-[11px] text-zinc-600">
            <AlertTriangle className="inline h-3 w-3 mr-1" />
            Restart the agent after saving for changes to take effect
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-xs font-medium text-zinc-400 hover:text-zinc-200 rounded-lg hover:bg-zinc-800 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving || loading}
              className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-white bg-violet-600 hover:bg-violet-500 rounded-lg transition-colors disabled:opacity-40"
            >
              {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
