import { useEffect, useState } from 'react'
import type { AgentManifest } from './types'
import { generateManifest, saveManifest, deleteManifest } from './authoring'
import { useMCPStore } from '../stores/mcpStore'
import { useAgentTargets } from './useAgentTargets'

interface Props {
  open: boolean
  onClose: () => void
  onSaved: (agentId: string) => void
  baseAgentId?: string                 // when editing
  baseMcpServer?: string               // pre-fill server when editing
}

const LAST_AGENT_KEY = 'fd:app-runtime:authoring-agent'

export function AuthoringDialog({ open, onClose, onSaved, baseAgentId, baseMcpServer }: Props) {
  const [description, setDescription] = useState('')
  const [mcpServer, setMcpServer] = useState<string>('')
  const [agentId, setAgentId] = useState<string>('')
  const [busy, setBusy] = useState(false)
  const [draft, setDraft] = useState<AgentManifest | null>(null)
  const [errors, setErrors] = useState<string[]>([])
  const [info, setInfo] = useState<string | null>(null)

  const servers = useMCPStore((s) => s.servers)
  const refreshServers = useMCPStore((s) => s.refresh)
  const agentTargets = useAgentTargets()

  useEffect(() => {
    if (open) {
      refreshServers()
      setDescription('')
      setDraft(null)
      setErrors([])
      setInfo(null)
      setMcpServer(baseMcpServer ?? '')
      try {
        const remembered = localStorage.getItem(LAST_AGENT_KEY) ?? ''
        setAgentId(remembered)
      } catch { setAgentId('') }
    }
  }, [open, baseMcpServer, refreshServers])

  // If the remembered agent isn't (or is no longer) available, fall back
  // to the first online one so the user isn't silently routed to nothing.
  useEffect(() => {
    if (!open) return
    if (agentId && agentTargets.some((a) => a.id === agentId)) return
    if (agentTargets.length > 0) setAgentId(agentTargets[0].id)
  }, [open, agentTargets, agentId])

  if (!open) return null

  const onGenerate = async () => {
    setBusy(true)
    setErrors([])
    setInfo(null)
    const target = agentTargets.find((a) => a.id === agentId) ?? null
    try { if (agentId) localStorage.setItem(LAST_AGENT_KEY, agentId) } catch { /* ignore */ }
    const res = await generateManifest({
      description,
      mcp_server: mcpServer || undefined,
      base_agent_id: baseAgentId,
      agent: target
        ? { host: target.host, port: target.port, auth: target.auth, name: target.name }
        : undefined,
    })
    setBusy(false)
    setDraft(res.manifest)
    setErrors(res.errors ?? [])
    if (!res.manifest) setInfo('LLM produced no manifest. Try again or rephrase.')
  }

  const onSave = async () => {
    if (!draft) return
    setBusy(true)
    setErrors([])
    const res = await saveManifest(draft)
    setBusy(false)
    if (!res.ok) {
      setErrors(res.errors ?? ['save failed'])
      return
    }
    onSaved(draft.agent.id)
    onClose()
  }

  const onDelete = async () => {
    if (!baseAgentId) return
    if (!confirm(`Delete app "${baseAgentId}"? This cannot be undone.`)) return
    setBusy(true)
    const res = await deleteManifest(baseAgentId)
    setBusy(false)
    if (!res.ok) {
      setErrors([res.error ?? 'delete failed'])
      return
    }
    onSaved('')   // signal parent to refresh and clear selection
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-auto bg-black/70 p-4" onClick={onClose}>
      <div
        onClick={(e) => e.stopPropagation()}
        className="my-8 w-full max-w-3xl rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
      >
        <header className="mb-4 flex items-center justify-between">
          <h2 className="text-base font-semibold text-zinc-100">
            {baseAgentId ? `Edit app: ${baseAgentId}` : 'New app from description'}
          </h2>
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">✕</button>
        </header>

        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-[11px] uppercase tracking-wide text-zinc-500">
              Generate with agent
            </label>
            <select
              value={agentId}
              onChange={(e) => setAgentId(e.target.value)}
              className="w-full rounded border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
            >
              {agentTargets.length === 0 && (
                <option value="">No agents online — start one first</option>
              )}
              {agentTargets.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name} {a.model ? `· ${a.model}` : ''} · {a.kind}
                </option>
              ))}
            </select>
            <p className="mt-1 text-[10px] text-zinc-600">
              The selected agent's LLM (model + credentials) generates the manifest.
            </p>
          </div>

          <div>
            <label className="mb-1 block text-[11px] uppercase tracking-wide text-zinc-500">
              {baseAgentId ? 'Describe the change' : 'Describe the app'}
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={5}
              className="w-full rounded border border-zinc-800 bg-zinc-900 px-2 py-2 text-sm text-zinc-100"
              placeholder={baseAgentId
                ? 'e.g. add a follow-ups inbox surface and a Mark Done action'
                : 'e.g. I want an app to track support signals from my portfolio companies…'}
            />
          </div>

          <div>
            <label className="mb-1 block text-[11px] uppercase tracking-wide text-zinc-500">
              MCP server <span className="text-zinc-600 normal-case tracking-normal">(optional)</span>
            </label>
            <select
              value={mcpServer}
              onChange={(e) => setMcpServer(e.target.value)}
              className="w-full rounded border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
            >
              <option value="">None — model will guess tool names</option>
              {servers.map((s) => (
                <option key={s.name} value={s.name}>{s.name}</option>
              ))}
            </select>
            <p className="mt-1 text-[10px] text-zinc-600">
              If set, the generator sees the live tool catalogue and binds feeds/actions to real tools.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onGenerate}
              disabled={busy || !description.trim() || !agentId}
              className="rounded bg-violet-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50"
            >
              {busy ? 'Working…' : draft ? 'Regenerate' : 'Generate'}
            </button>
            {baseAgentId && (
              <button
                onClick={onDelete}
                disabled={busy}
                className="ml-auto rounded border border-red-900/50 px-3 py-1.5 text-xs text-red-300 hover:bg-red-950/40"
              >
                Delete app
              </button>
            )}
          </div>

          {info && <div className="rounded bg-zinc-900 px-3 py-2 text-xs text-zinc-400">{info}</div>}

          {errors.length > 0 && (
            <div className="rounded bg-red-950/40 px-3 py-2 text-xs text-red-300">
              <div className="mb-1 font-semibold">Validation errors</div>
              <ul className="list-disc space-y-0.5 pl-4">
                {errors.map((e, i) => <li key={i}>{e}</li>)}
              </ul>
            </div>
          )}

          {draft && (
            <div>
              <div className="mb-1 flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wide text-zinc-500">Manifest preview</span>
                <span className="text-[10px] text-zinc-600">
                  {draft.agent?.name} · {Object.keys(draft.feeds ?? {}).length} feeds · {Object.keys(draft.actions ?? {}).length} actions · {Object.keys(draft.surfaces ?? {}).length} surfaces
                </span>
              </div>
              <pre className="max-h-72 overflow-auto rounded border border-zinc-800 bg-zinc-900 p-3 text-[11px] leading-snug text-zinc-200">
                {JSON.stringify(draft, null, 2)}
              </pre>
            </div>
          )}
        </div>

        <footer className="mt-5 flex justify-end gap-2 border-t border-zinc-800 pt-4">
          <button onClick={onClose} className="rounded px-3 py-1.5 text-sm text-zinc-400 hover:text-zinc-200">Cancel</button>
          <button
            onClick={onSave}
            disabled={!draft || busy || errors.length > 0}
            className="rounded bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
            title={errors.length > 0 ? 'Resolve validation errors before saving' : ''}
          >
            Save
          </button>
        </footer>
      </div>
    </div>
  )
}
