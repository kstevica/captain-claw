import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Sparkles,
  Loader2,
  RefreshCw,
  AlertCircle,
  Download,
  Power,
  ExternalLink,
  Upload,
  FileArchive,
} from 'lucide-react'
import { useContainerStore } from '../stores/containerStore'
import { useProcessStore } from '../stores/processStore'
import { useLocalAgentStore } from '../stores/localAgentStore'
import { useAuthStore } from '../stores/authStore'

// ── Types ──

interface AgentTarget {
  id: string
  name: string
  kind: 'docker' | 'process' | 'local'
  host: string
  port: number
  auth: string
}

interface Skill {
  name: string
  description: string
  source: string
  file_path: string
  base_dir: string
  emoji?: string | null
  homepage?: string | null
  user_invocable: boolean
  model_invocation: boolean
  active: boolean
  enabled: boolean | null
  skill_key: string
  requires: { bins: string[]; any_bins: string[]; env: string[]; config: string[] }
  has_install: boolean
}

interface AgentSkillsState {
  loading: boolean
  error: string | null
  skills: Skill[]
}

// ── Helpers ──

function fdAuthHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = {}
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

// ── Page ──

export function SkillsPage() {
  const containers = useContainerStore((s) => s.containers)
  const processes = useProcessStore((s) => s.processes)
  const localAgents = useLocalAgentStore((s) => s.agents)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [data, setData] = useState<Map<string, AgentSkillsState>>(new Map())
  const [installUrl, setInstallUrl] = useState('')
  const [installing, setInstalling] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [filter, setFilter] = useState('')
  const uploadRef = useRef<HTMLInputElement>(null)

  const targets: AgentTarget[] = useMemo(() => {
    const out: AgentTarget[] = []
    for (const c of containers) {
      if (c.web_port) {
        out.push({
          id: c.id, name: c.agent_name || c.name, kind: 'docker',
          host: 'localhost', port: c.web_port, auth: c.web_auth || '',
        })
      }
    }
    for (const p of processes) {
      if (p.web_port) {
        out.push({
          id: p.slug, name: p.name, kind: 'process',
          host: 'localhost', port: p.web_port, auth: p.web_auth || '',
        })
      }
    }
    for (const a of localAgents) {
      out.push({
        id: a.id, name: a.name, kind: 'local',
        host: a.host, port: a.port, auth: a.authToken || '',
      })
    }
    return out
  }, [containers, processes, localAgents])

  const selected = useMemo(
    () => targets.find((t) => t.id === selectedId) || targets[0] || null,
    [targets, selectedId]
  )

  // Default to first agent
  useEffect(() => {
    if (!selectedId && targets.length > 0) setSelectedId(targets[0].id)
  }, [targets, selectedId])

  const loadSkills = useCallback(async (target: AgentTarget) => {
    setData((prev) => {
      const next = new Map(prev)
      next.set(target.id, { loading: true, error: null, skills: [] })
      return next
    })
    try {
      const params = new URLSearchParams()
      if (target.auth) params.set('token', target.auth)
      const res = await fetch(
        `/fd/agent-skills/${encodeURIComponent(target.host)}/${target.port}?${params}`,
        { headers: fdAuthHeaders(), credentials: 'include' }
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      const skills: Skill[] = Array.isArray(json?.skills) ? json.skills : []
      setData((prev) => {
        const next = new Map(prev)
        next.set(target.id, { loading: false, error: null, skills })
        return next
      })
    } catch (e) {
      setData((prev) => {
        const next = new Map(prev)
        next.set(target.id, {
          loading: false,
          error: e instanceof Error ? e.message : String(e),
          skills: [],
        })
        return next
      })
    }
  }, [])

  useEffect(() => {
    if (selected) void loadSkills(selected)
  }, [selected?.id, loadSkills]) // eslint-disable-line react-hooks/exhaustive-deps

  const toggleSkill = useCallback(async (skill: Skill) => {
    if (!selected) return
    const newEnabled = skill.enabled === false ? true : false
    try {
      const params = new URLSearchParams()
      if (selected.auth) params.set('token', selected.auth)
      const res = await fetch(
        `/fd/agent-skills/${encodeURIComponent(selected.host)}/${selected.port}/toggle?${params}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...fdAuthHeaders() },
          credentials: 'include',
          body: JSON.stringify({ skill_key: skill.skill_key, enabled: newEnabled }),
        }
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      await loadSkills(selected)
    } catch (e) {
      alert(`Toggle failed: ${e instanceof Error ? e.message : String(e)}`)
    }
  }, [selected, loadSkills])

  const installSkill = useCallback(async () => {
    if (!selected) return
    const url = installUrl.trim()
    if (!url) return
    setInstalling(true)
    try {
      const params = new URLSearchParams()
      if (selected.auth) params.set('token', selected.auth)
      const res = await fetch(
        `/fd/agent-skills/${encodeURIComponent(selected.host)}/${selected.port}/install?${params}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...fdAuthHeaders() },
          credentials: 'include',
          body: JSON.stringify({ url }),
        }
      )
      const json = await res.json().catch(() => ({}))
      if (!res.ok || !json?.ok) {
        throw new Error(json?.error || `HTTP ${res.status}`)
      }
      setInstallUrl('')
      await loadSkills(selected)
    } catch (e) {
      alert(`Install failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setInstalling(false)
    }
  }, [selected, installUrl, loadSkills])

  const installSkillUpload = useCallback(async (file: File) => {
    if (!selected) return
    const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
    if (ext !== '.md' && ext !== '.zip') {
      alert('Only .md or .zip files are supported.')
      return
    }
    setUploading(true)
    try {
      const params = new URLSearchParams()
      if (selected.auth) params.set('token', selected.auth)
      const fd = new FormData()
      fd.append('file', file, file.name)
      const res = await fetch(
        `/fd/agent-skills/${encodeURIComponent(selected.host)}/${selected.port}/install-upload?${params}`,
        {
          method: 'POST',
          headers: fdAuthHeaders(),
          credentials: 'include',
          body: fd,
        }
      )
      const json = await res.json().catch(() => ({}))
      if (!res.ok || !json?.ok) {
        throw new Error(json?.error || `HTTP ${res.status}`)
      }
      await loadSkills(selected)
    } catch (e) {
      alert(`Upload failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setUploading(false)
      if (uploadRef.current) uploadRef.current.value = ''
    }
  }, [selected, loadSkills])

  const state = selected ? data.get(selected.id) : null
  const filteredSkills = useMemo(() => {
    const list = state?.skills || []
    const q = filter.trim().toLowerCase()
    if (!q) return list
    return list.filter((s) =>
      s.name.toLowerCase().includes(q) || (s.description || '').toLowerCase().includes(q)
    )
  }, [state, filter])

  if (targets.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center bg-zinc-950 text-zinc-500">
        <Sparkles className="mb-3 h-10 w-10 text-zinc-700" />
        <div className="text-sm">No agents reachable.</div>
        <div className="mt-1 text-xs text-zinc-600">Spawn an agent to manage skills.</div>
      </div>
    )
  }

  return (
    <div className="flex h-full overflow-hidden bg-zinc-950">
      {/* Agent picker */}
      <aside className="flex w-60 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/30">
        <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <Sparkles className="h-4 w-4 text-violet-400" />
          <span className="text-sm font-semibold text-zinc-200">Skills</span>
        </div>
        <div className="px-2 py-2">
          <div className="px-2 pb-1 text-[9px] font-semibold uppercase tracking-wider text-zinc-600">Agents</div>
          {targets.map((t) => (
            <button
              key={t.id}
              onClick={() => setSelectedId(t.id)}
              className={`mb-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors ${
                selected?.id === t.id
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
              }`}
            >
              <span className="min-w-0 flex-1 truncate">{t.name}</span>
              <span className="rounded-full border border-zinc-800 px-1.5 py-0.5 text-[8px] text-zinc-500">{t.kind}</span>
            </button>
          ))}
        </div>
      </aside>

      {/* Main */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="flex items-center justify-between border-b border-zinc-800 px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">
              {selected ? selected.name : 'Skills'}
            </h1>
            <p className="text-xs text-zinc-500">
              {selected ? `${selected.host}:${selected.port} · ${state?.skills.length || 0} skills` : ''}
            </p>
          </div>
          <button
            onClick={() => selected && loadSkills(selected)}
            disabled={state?.loading}
            className="flex items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
          >
            {state?.loading
              ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
              : <RefreshCw className="h-3.5 w-3.5" />}
            Refresh
          </button>
        </div>

        {/* Install bar */}
        <div className="border-b border-zinc-800 px-6 py-3">
          <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-zinc-500">
            <Download className="h-3 w-3" />
            Install from GitHub
          </div>
          <div className="flex gap-2">
            <div className="flex flex-1 items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900 px-2.5 focus-within:border-violet-500/50">
              <ExternalLink className="h-3.5 w-3.5 text-zinc-500" />
              <input
                type="text"
                value={installUrl}
                onChange={(e) => setInstallUrl(e.target.value)}
                placeholder="https://github.com/owner/repo or owner/repo"
                onKeyDown={(e) => e.key === 'Enter' && void installSkill()}
                disabled={installing}
                className="flex-1 bg-transparent py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none disabled:opacity-50"
              />
            </div>
            <button
              onClick={() => void installSkill()}
              disabled={installing || !installUrl.trim()}
              className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
            >
              {installing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
              Install
            </button>
            <button
              onClick={() => uploadRef.current?.click()}
              disabled={uploading}
              title="Upload a .md or .zip skill file"
              className="flex items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
            >
              {uploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
              <FileArchive className="h-3.5 w-3.5" />
              Upload
            </button>
            <input
              ref={uploadRef}
              type="file"
              accept=".md,.zip"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void installSkillUpload(f)
              }}
            />
          </div>
        </div>

        {/* Filter */}
        <div className="border-b border-zinc-800 px-6 py-2">
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter skills…"
            className="w-full rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
        </div>

        {/* Skill list */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {state?.error && (
            <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <div>
                <div className="font-semibold">Failed to load skills</div>
                <div className="text-red-300/80">{state.error}</div>
              </div>
            </div>
          )}

          {state?.loading && (
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading skills…
            </div>
          )}

          {!state?.loading && !state?.error && filteredSkills.length === 0 && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-6 text-center text-xs text-zinc-500">
              {state?.skills.length === 0
                ? 'No skills installed on this agent.'
                : 'No skills match your filter.'}
            </div>
          )}

          <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
            {filteredSkills.map((s) => {
              const isOff = s.enabled === false
              const isActive = s.active && !isOff
              return (
                <div
                  key={`${s.skill_key}-${s.file_path}`}
                  className={`rounded-lg border p-3 transition-colors ${
                    isActive
                      ? 'border-zinc-800 bg-zinc-900/40'
                      : 'border-zinc-800/50 bg-zinc-900/20 opacity-60'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        {s.emoji && <span className="text-sm">{s.emoji}</span>}
                        <span className="truncate text-sm font-semibold text-zinc-200">{s.name}</span>
                        <span
                          className={`rounded-full px-1.5 py-0.5 text-[8px] ${
                            isActive
                              ? 'bg-emerald-500/15 text-emerald-300'
                              : isOff
                                ? 'bg-zinc-800 text-zinc-500'
                                : 'bg-amber-500/15 text-amber-300'
                          }`}
                        >
                          {isOff ? 'disabled' : isActive ? 'active' : 'unmet reqs'}
                        </span>
                      </div>
                      {s.description && (
                        <div className="mt-1 line-clamp-2 text-[11px] text-zinc-400">{s.description}</div>
                      )}
                      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[9px] text-zinc-600">
                        <span>source: {s.source}</span>
                        {s.user_invocable && <span>· user-invocable</span>}
                        {s.model_invocation && <span>· model-callable</span>}
                        {s.requires.bins.length > 0 && (
                          <span>· bins: {s.requires.bins.join(', ')}</span>
                        )}
                        {s.requires.env.length > 0 && (
                          <span>· env: {s.requires.env.join(', ')}</span>
                        )}
                      </div>
                    </div>
                    <div className="flex shrink-0 flex-col items-end gap-1">
                      <button
                        onClick={() => void toggleSkill(s)}
                        title={isOff ? 'Enable skill' : 'Disable skill'}
                        className={`flex items-center gap-1 rounded-md border px-2 py-1 text-[10px] transition-colors ${
                          isOff
                            ? 'border-zinc-800 text-zinc-500 hover:border-emerald-500/30 hover:text-emerald-300'
                            : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:border-zinc-700 hover:bg-zinc-800 hover:text-zinc-400'
                        }`}
                      >
                        <Power className="h-3 w-3" />
                        {isOff ? 'Enable' : 'Disable'}
                      </button>
                      {s.homepage && (
                        <a
                          href={s.homepage}
                          target="_blank"
                          rel="noreferrer"
                          className="flex items-center gap-1 text-[9px] text-zinc-500 hover:text-zinc-300"
                        >
                          <ExternalLink className="h-2.5 w-2.5" />
                          homepage
                        </a>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
