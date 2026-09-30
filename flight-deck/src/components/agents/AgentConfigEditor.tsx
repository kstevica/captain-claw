import { useState, useEffect, useRef } from 'react'
import { X, Save, Loader2, AlertTriangle, FileText, KeyRound, BookOpen, SlidersHorizontal, Server } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'
import { useContainerStore } from '../../stores/containerStore'
import { useProcessStore } from '../../stores/processStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { ACT_AS_HEADER } from '../../services/docker'
import { ModelSelector } from '../common/ModelSelector'
import { CognitiveModeSelector } from '../common/CognitiveModeSelector'

interface AgentConfigEditorProps {
  /** 'local' agents live on a remote host — only their label/description are
   *  stored here; config.yaml / .env / instructions don't apply. */
  kind: 'docker' | 'process' | 'local'
  identifier: string    // container id, process slug, or local agent id
  agentName: string
  onClose: () => void
  /** Admin managing ANOTHER user's agent (docker / process): that user's id.
   *  Everything then goes through the server on their behalf — nothing is read
   *  from or written to the admin's own browser stores — and the controls that
   *  only work on the viewer's own agents (model, cognitive mode) are hidden. */
  asUser?: string
  /** With `asUser`: who the agent belongs to, and its current description. */
  ownerLabel?: string
  initialDescription?: string
}

async function fdFetchConfig<T>(path: string, init?: RequestInit, asUser?: string): Promise<T> {
  const build = (): Record<string, string> => {
    const { token, authEnabled } = useAuthStore.getState()
    const h: Record<string, string> = { 'Content-Type': 'application/json' }
    if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
    if (asUser) h[ACT_AS_HEADER] = asUser
    return h
  }
  const { authEnabled } = useAuthStore.getState()

  let res = await fetch(`/fd${path}`, { headers: build(), credentials: 'include', ...init })
  if (res.status === 401 && authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) {
      res = await fetch(`/fd${path}`, { headers: build(), credentials: 'include', ...init })
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

type Tab = 'general' | 'instructions' | 'config' | 'env'

export function AgentConfigEditor({
  kind, identifier, agentName, onClose, asUser, ownerLabel, initialDescription,
}: AgentConfigEditorProps) {
  const isLocal = kind === 'local'
  // A container's name / description are browser-side labels of whoever views
  // it, so an admin acting for its owner has nothing to edit on General.
  const hasGeneral = !(asUser && kind === 'docker')

  const [activeTab, setActiveTab] = useState<Tab>(hasGeneral ? 'general' : 'instructions')
  const [configYaml, setConfigYaml] = useState('')
  const [env, setEnv] = useState('')
  const [loadingConfig, setLoadingConfig] = useState(!isLocal)
  // A failed load leaves the editors empty; saving then would write that
  // emptiness over the real files. Save stays off until a load succeeds.
  const [loadFailed, setLoadFailed] = useState(false)
  const [reloadKey, setReloadKey] = useState(0)
  // What the server held when this opened — the as-user Save sends only what
  // differs, so it never rewrites parts the admin didn't touch.
  const loaded = useRef({ configYaml: '', env: '', fleet: '', name: '', description: '' })
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  // Stores. For docker/process the overrides live in the container/process
  // store; for local agents, name + description live in the local-agent store.
  const containerStore = useContainerStore()
  const processStore = useProcessStore()
  const localStore = useLocalAgentStore()
  const store = kind === 'docker' ? containerStore : processStore

  const localAgent = isLocal ? localStore.agents.find((a) => a.id === identifier) : undefined

  // Friendly field state, seeded from the stores (instant, no fetch needed).
  const [name, setName] = useState(() =>
    asUser ? agentName
      : isLocal ? (localAgent?.name ?? agentName) : (store.nameOverrides[identifier] || agentName))
  const [description, setDescription] = useState(() =>
    asUser ? (initialDescription ?? '')
      : isLocal ? (localAgent?.description ?? '') : (store.descriptionOverrides[identifier] || ''))
  const [fleet, setFleet] = useState(() => (isLocal || asUser ? '' : store.getFleetInstructions(identifier)))
  const [cog, setCog] = useState(() => (isLocal ? 'neutra' : store.getCognitiveMode(identifier)))
  const [cogSaved, setCogSaved] = useState(false)

  // config.yaml / .env come from disk — only docker/process have them.
  useEffect(() => {
    if (isLocal) return
    let cancelled = false
    setLoadingConfig(true)
    setLoadFailed(false)
    setError('')
    const loads: Promise<unknown>[] = [
      fdFetchConfig<{ config_yaml: string; env: string }>(`/agent-config/${kind}/${identifier}`, undefined, asUser)
        .then((data) => {
          if (cancelled) return
          setConfigYaml(data.config_yaml || '')
          setEnv(data.env || '')
          loaded.current.configYaml = data.config_yaml || ''
          loaded.current.env = data.env || ''
        }),
    ]
    // The owner's instructions live in THEIR settings — ask the server.
    if (asUser) {
      loaded.current.name = agentName
      loaded.current.description = initialDescription ?? ''
      loads.push(
        fdFetchConfig<{ instructions: string }>(`/agent-instructions/${kind}/${identifier}`, undefined, asUser)
          .then((data) => {
            if (cancelled) return
            setFleet(data.instructions || '')
            loaded.current.fleet = data.instructions || ''
          }),
      )
    }
    Promise.all(loads)
      .catch((e) => {
        if (cancelled) return
        setLoadFailed(true)
        setError(`${e.message || 'Failed to load'} — nothing can be saved until it loads.`)
      })
      .finally(() => { if (!cancelled) setLoadingConfig(false) })
    return () => { cancelled = true }
    // agentName / initialDescription only seed the snapshot for this open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, identifier, isLocal, asUser, reloadKey])

  // Cognitive mode applies instantly (store + best-effort live push), matching
  // the agent cards. A stopped agent just stores the choice for next start.
  const applyCognitive = async (mode: string) => {
    setCog(mode)
    if (isLocal) return
    store.setCognitiveMode(identifier, mode)
    try {
      const { token, authEnabled } = useAuthStore.getState()
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
      await fetch(`/fd/agent-mode/${kind}/${identifier}`, {
        method: 'PUT', headers, credentials: 'include', body: JSON.stringify({ mode }),
      })
    } catch { /* agent may be stopped — the stored choice still applies on start */ }
    setCogSaved(true)
    setTimeout(() => setCogSaved(false), 2000)
  }

  const handleSave = async () => {
    setSaving(true)
    setError('')
    setSuccess('')
    try {
      if (isLocal) {
        localStore.updateAgent(identifier, { name: name.trim() || agentName, description })
        setSuccess('Saved.')
      } else if (asUser) {
        // On the owner's behalf: registry identity, their instructions, the
        // files on disk — all server-side, none of it in this browser's stores.
        // Only what changed is sent, so the owner's own edits to the rest (made
        // since this opened) are left alone.
        const was = loaded.current
        const saved: string[] = []
        try {
          const newName = name.trim() || agentName
          if (kind === 'process' && (newName !== was.name || description !== was.description)) {
            await fdFetchConfig(`/processes/${identifier}/identity`, {
              method: 'POST',
              body: JSON.stringify({
                ...(newName !== was.name ? { name: newName } : {}),
                ...(description !== was.description ? { description } : {}),
              }),
            }, asUser)
            was.name = newName
            was.description = description
            saved.push('name and description')
          }
          if (fleet !== was.fleet) {
            await fdFetchConfig(`/agent-instructions/${kind}/${identifier}`, {
              method: 'PUT', body: JSON.stringify({ instructions: fleet }),
            }, asUser)
            was.fleet = fleet
            saved.push('instructions')
          }
          if (configYaml !== was.configYaml || env !== was.env) {
            await fdFetchConfig<{ ok: boolean; message: string }>(`/agent-config/${kind}/${identifier}`, {
              method: 'PUT',
              body: JSON.stringify({
                ...(configYaml !== was.configYaml ? { config_yaml: configYaml } : {}),
                ...(env !== was.env ? { env } : {}),
              }),
            }, asUser)
            was.configYaml = configYaml
            was.env = env
            saved.push('config.yaml / .env (restart the agent to apply)')
          }
        } catch (e) {
          const why = e instanceof Error ? e.message : 'Failed to save'
          throw new Error(saved.length ? `Saved ${saved.join(', ')} — then failed: ${why}` : why)
        }
        setSuccess(saved.length ? `Saved ${saved.join(', ')}.` : 'Nothing changed.')
      } else {
        // Update the FD-side override so the list reflects it instantly…
        store.setNameOverride(identifier, name.trim() || agentName)
        store.setDescription(identifier, description)
        store.setFleetInstructions(identifier, fleet)
        // …and, for process agents, persist name/description canonically into
        // the registry so they survive a browser wipe and reach every client.
        if (kind === 'process') {
          await fdFetchConfig(`/processes/${identifier}/identity`, {
            method: 'POST',
            body: JSON.stringify({ name: name.trim() || agentName, description }),
          })
        }
        // config.yaml + .env are written to disk and need a restart.
        await fdFetchConfig<{ ok: boolean; message: string }>(`/agent-config/${kind}/${identifier}`, {
          method: 'PUT',
          body: JSON.stringify({ config_yaml: configYaml, env }),
        })
        setSuccess('Saved. Name, description and instructions apply now; restart for config.yaml / .env changes.')
      }
      setTimeout(() => setSuccess(''), 6000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  const tabs: { id: Tab; label: string; icon: typeof FileText }[] = isLocal
    ? [{ id: 'general', label: 'General', icon: SlidersHorizontal }]
    : [
        ...(hasGeneral ? [{ id: 'general' as Tab, label: 'General', icon: SlidersHorizontal }] : []),
        { id: 'instructions', label: 'Instructions', icon: BookOpen },
        { id: 'config', label: 'config.yaml', icon: FileText },
        { id: 'env', label: '.env', icon: KeyRound },
      ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="flex h-[80vh] w-[80vw] flex-col rounded-2xl border border-zinc-700/50 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-violet-600/20 text-violet-400">
              <SlidersHorizontal className="h-4 w-4" />
            </div>
            <div>
              <div className="text-sm font-semibold text-zinc-100">Agent options</div>
              <div className="text-xs text-zinc-500">
                {agentName}{asUser && ownerLabel ? ` · managed for ${ownerLabel}` : ''}
              </div>
            </div>
          </div>
          <button onClick={onClose} className="rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex shrink-0 gap-1 border-b border-zinc-800 px-4">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-1.5 px-3 py-2.5 text-xs font-medium transition-colors ${
                activeTab === id
                  ? 'border-b-2 border-violet-400 text-violet-400'
                  : 'border-b-2 border-transparent text-zinc-500 hover:text-zinc-300'
              }`}
            >
              <Icon className="h-3.5 w-3.5" /> {label}
            </button>
          ))}
        </div>

        {/* Content — the modal is a fixed size, so each tab owns the space. */}
        <div className="min-h-0 flex-1 overflow-hidden">
          {activeTab === 'general' && (
            <div className="mx-auto flex h-full max-w-2xl flex-col gap-6 overflow-y-auto p-6">
              <Field label="Display name" hint="Shown in the agent list and chat. Only changes the label — not the agent's hostname.">
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder={agentName}
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
                />
              </Field>

              <Field label="Description" hint="A short note about what this agent is for.">
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={2}
                  placeholder="e.g. Research assistant for market analysis"
                  className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
                />
              </Field>

              {isLocal ? (
                <div className="flex items-start gap-2.5 rounded-lg border border-zinc-800 bg-zinc-950/50 px-4 py-3 text-xs text-zinc-500">
                  <Server className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-600" />
                  <span>
                    This agent runs on a remote host, so its model, instructions and environment
                    live on that machine. Only its label and description are stored here.
                  </span>
                </div>
              ) : asUser ? (
                <div className="flex items-start gap-2.5 rounded-lg border border-zinc-800 bg-zinc-950/50 px-4 py-3 text-xs text-zinc-500">
                  <Server className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-600" />
                  <span>
                    You're managing this agent for its owner. Change its model, keys and mode on the
                    config.yaml and .env tabs, then restart it.
                  </span>
                </div>
              ) : (
                <>
                  <Field label="Model" hint="Provider, model and API key. Applies on the agent's next turn.">
                    <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3">
                      <ModelSelector kind={kind} identifier={identifier} />
                    </div>
                  </Field>

                  <Field label="Cognitive mode" hint="How the agent thinks — its reasoning strategy. Applies immediately.">
                    <CognitiveModeSelector value={cog} saved={cogSaved} onChange={applyCognitive} />
                  </Field>
                </>
              )}
            </div>
          )}

          {activeTab === 'instructions' && (asUser && loadingConfig ? (
            <Loading label="Loading instructions…" />
          ) : (
            <div className="flex h-full flex-col p-6">
              <p className="shrink-0 pb-3 text-xs leading-relaxed text-zinc-500">
                Standing instructions added to this agent's system prompt on every conversation.
                They take effect immediately — no restart. Use them for tone, format, house rules,
                or context the agent should always keep in mind.
              </p>
              <textarea
                value={fleet}
                onChange={(e) => setFleet(e.target.value)}
                spellCheck={false}
                placeholder={'Standing instructions for this agent…\n\nExamples:\n- Always answer in British English\n- Prefer concise, bulleted answers\n- Cite sources for factual claims\n- Assume the reader is a domain expert'}
                className="min-h-0 w-full flex-1 resize-none rounded-lg border border-zinc-700 bg-zinc-950 px-4 py-3 text-sm leading-relaxed text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
              />
            </div>
          ))}

          {activeTab === 'config' && (
            loadingConfig ? (
              <Loading label="Loading config…" />
            ) : (
              <div className="flex h-full flex-col p-4">
                <p className="shrink-0 px-1 pb-2 text-[11px] text-zinc-500">
                  The full agent config. Editing here needs a restart to take effect.
                </p>
                <textarea
                  value={configYaml}
                  onChange={(e) => setConfigYaml(e.target.value)}
                  spellCheck={false}
                  className="min-h-0 w-full flex-1 resize-none rounded-lg border border-zinc-800 bg-zinc-950/50 px-4 py-3 font-mono text-[13px] leading-relaxed text-zinc-300 placeholder-zinc-700 focus:border-violet-500/40 focus:outline-none"
                  placeholder={'# config.yaml — agent configuration\nprovider: anthropic\nmodel: claude-sonnet-4-20250514\n…'}
                />
              </div>
            )
          )}

          {activeTab === 'env' && (
            loadingConfig ? (
              <Loading label="Loading .env…" />
            ) : (
              <div className="flex h-full flex-col p-4">
                <p className="flex shrink-0 items-center gap-1.5 px-1 pb-2 text-[11px] text-zinc-500">
                  <BookOpen className="h-3 w-3" /> Environment variables (API keys, endpoints). Restart to apply.
                </p>
                <textarea
                  value={env}
                  onChange={(e) => setEnv(e.target.value)}
                  spellCheck={false}
                  className="min-h-0 w-full flex-1 resize-none rounded-lg border border-zinc-800 bg-zinc-950/50 px-4 py-3 font-mono text-[13px] leading-relaxed text-zinc-300 placeholder-zinc-700 focus:border-violet-500/40 focus:outline-none"
                  placeholder={'# .env — environment variables\nANTHROPIC_API_KEY=…\nBRAVE_API_KEY=…\n…'}
                />
              </div>
            )
          )}
        </div>

        {/* Status */}
        {error && (
          <div className="mx-6 mb-2 flex shrink-0 items-center gap-2 rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-400">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0" /> <span className="min-w-0 flex-1">{error}</span>
            {loadFailed && (
              <button onClick={() => setReloadKey((k) => k + 1)} className="shrink-0 rounded px-2 py-0.5 font-medium underline hover:no-underline">
                Retry
              </button>
            )}
          </div>
        )}
        {success && (
          <div className="mx-6 mb-2 flex shrink-0 items-center gap-2 rounded-lg bg-emerald-500/10 px-3 py-2 text-xs text-emerald-400">
            <Save className="h-3.5 w-3.5 shrink-0" /> {success}
          </div>
        )}

        {/* Footer */}
        <div className="flex shrink-0 items-center justify-between border-t border-zinc-800 px-6 py-4">
          <span className="text-[11px] text-zinc-600">
            {isLocal
              ? 'Label and description are saved locally.'
              : asUser
                ? 'Saved for the owner. config.yaml / .env need a restart.'
              : kind === 'process'
                ? 'Name, description and instructions apply immediately. config.yaml / .env need a restart.'
                : 'General changes apply immediately. config.yaml / .env need a restart.'}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="rounded-lg px-3.5 py-2 text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
            >
              Close
            </button>
            <button
              onClick={handleSave}
              disabled={saving || (!isLocal && (loadingConfig || loadFailed))}
              className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-40"
            >
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function Loading({ label }: { label: string }) {
  return (
    <div className="flex h-full items-center justify-center text-sm text-zinc-500">
      <Loader2 className="mr-2 h-5 w-5 animate-spin" /> {label}
    </div>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400">{label}</label>
      {hint && <p className="text-[11px] leading-relaxed text-zinc-600">{hint}</p>}
      <div className="mt-0.5">{children}</div>
    </div>
  )
}
