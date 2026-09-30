import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Bot, Box, Cpu, Loader2, Play, Plus, RefreshCw, RotateCw, Settings, Square, Trash2, X } from 'lucide-react'
import {
  listContainers, listProcesses,
  startContainer, stopContainer, restartContainer, removeContainer,
  startProcess, stopProcess, restartProcess, removeProcess,
  type ContainerInfo, type ProcessInfo,
} from '../../services/docker'
import { useAuthStore } from '../../stores/authStore'
import { AgentConfigEditor } from './AgentConfigEditor'
import { ArchetypeSpawnDialog } from '../layout/ArchetypeSpawnDialog'

// ── Admin: another user's agents ─────────────────────────────────────
//
// Everything here runs through the ordinary agent endpoints with the admin
// acting FOR the user (services/docker `asUser` → X-FD-Act-As), so an agent
// created, configured or removed here behaves exactly as if its owner had done
// it: their ownership, their tier set and keys, their plan limits.

interface ManagedAgent {
  kind: 'docker' | 'process'
  /** What the endpoints take: process slug, or container id. */
  id: string
  name: string
  description: string
  running: boolean
  detail: string
}

function fromProcess(p: ProcessInfo): ManagedAgent {
  return {
    kind: 'process', id: p.slug, name: p.name || p.slug, description: p.description || '',
    running: p.status === 'running',
    detail: [p.provider, p.model].filter(Boolean).join(' / '),
  }
}

function fromContainer(c: ContainerInfo): ManagedAgent {
  return {
    kind: 'docker', id: c.id, name: c.agent_name || c.name, description: c.description || '',
    running: c.status === 'running',
    detail: c.image,
  }
}

const btn = 'flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 transition-colors hover:border-zinc-600 hover:bg-zinc-800 disabled:opacity-40'

export function UserAgentsDialog({ userId, userLabel, onClose }: {
  userId: string
  userLabel: string
  onClose: () => void
}) {
  // An admin opening their OWN row isn't acting for anyone: the plain paths
  // keep their own browser stores (instructions, labels) in step.
  const actAs = useAuthStore((s) => (s.user?.id === userId ? undefined : userId))
  const [agents, setAgents] = useState<ManagedAgent[] | null>(null)
  const [loadError, setLoadError] = useState('')
  // Kept until the next action or dismissal — the list poll must not wipe it.
  const [actionError, setActionError] = useState('')
  const [busy, setBusy] = useState<Set<string>>(() => new Set())   // agents with an action running
  const [creating, setCreating] = useState(false)
  const [editing, setEditing] = useState<ManagedAgent | null>(null)

  const load = useCallback(async () => {
    try {
      // Docker may be unavailable on a process-only deck — that's not an error.
      const [procs, containers] = await Promise.all([
        listProcesses(actAs),
        listContainers(actAs).catch(() => [] as ContainerInfo[]),
      ])
      const list = [...procs.map(fromProcess), ...containers.map(fromContainer)]
      list.sort((a, b) => Number(b.running) - Number(a.running) || a.name.localeCompare(b.name))
      setAgents(list)
      setLoadError('')
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e))
    }
  }, [actAs])

  useEffect(() => {
    load()
    const interval = setInterval(load, 8000)
    return () => clearInterval(interval)
  }, [load])

  // Escape closes this dialog only when no child dialog is open on top of it.
  useEffect(() => {
    if (creating || editing) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [creating, editing, onClose])

  const run = async (a: ManagedAgent, action: 'start' | 'stop' | 'restart' | 'remove') => {
    if (action === 'remove' && !confirm(
      `Remove "${a.name}" from ${userLabel}'s agents? It is stopped and taken off their list; its files stay on disk.`)) return
    const key = `${a.kind}:${a.id}`
    setBusy((prev) => new Set(prev).add(key))
    setActionError('')
    try {
      if (a.kind === 'process') {
        if (action === 'start') await startProcess(a.id, actAs)
        else if (action === 'stop') await stopProcess(a.id, actAs)
        else if (action === 'restart') await restartProcess(a.id, actAs)
        else await removeProcess(a.id, actAs)
      } else {
        if (action === 'start') await startContainer(a.id, actAs)
        else if (action === 'stop') await stopContainer(a.id, actAs)
        else if (action === 'restart') await restartContainer(a.id, actAs)
        else await removeContainer(a.id, true, actAs)
      }
      await load()
    } catch (e) {
      setActionError(`${a.name}: could not ${action} — ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy((prev) => { const next = new Set(prev); next.delete(key); return next })
    }
  }

  return createPortal(
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label={`Agents of ${userLabel}`}
        className="flex max-h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-violet-500/10 text-violet-400">
            <Bot className="h-3.5 w-3.5" />
          </span>
          <div className="min-w-0">
            <h3 className="truncate text-sm font-semibold text-zinc-200">Agents of {userLabel}</h3>
            <p className="text-[11px] text-zinc-500">
              Changes apply to this user's agents as if they made them. Config shows each agent's settings, including its keys.
            </p>
          </div>
          <div className="ml-auto flex items-center gap-1">
            <button onClick={() => setCreating(true)} className="flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-violet-500">
              <Plus className="h-3.5 w-3.5" /> New agent
            </button>
            <button onClick={load} className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Refresh">
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
            <button onClick={onClose} className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Close">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* List */}
        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {actionError && (
            <p className="mb-3 flex items-start gap-2 rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-600 dark:text-red-400">
              <span className="min-w-0 flex-1">{actionError}</span>
              <button onClick={() => setActionError('')} title="Dismiss" className="shrink-0"><X className="h-3.5 w-3.5" /></button>
            </p>
          )}
          {loadError && <p className="mb-3 rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-600 dark:text-red-400">Couldn't load the agents: {loadError}</p>}
          {agents === null && !loadError && (
            <p className="flex items-center justify-center gap-2 py-8 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</p>
          )}
          {agents?.length === 0 && (
            <p className="py-8 text-center text-sm text-zinc-500">This user has no agents yet.</p>
          )}
          <ul className="flex flex-col gap-2">
            {agents?.map((a) => {
              const KindIcon = a.kind === 'docker' ? Box : Cpu
              const working = busy.has(`${a.kind}:${a.id}`)
              return (
                <li key={`${a.kind}:${a.id}`} className="flex items-center gap-3 rounded-lg border border-zinc-800 px-3 py-2.5">
                  <span className={`h-2 w-2 shrink-0 rounded-full ${a.running ? 'bg-emerald-600 dark:bg-emerald-400' : 'bg-zinc-600'}`} title={a.running ? 'Running' : 'Stopped'} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate text-sm font-medium text-zinc-100">{a.name}</span>
                      <KindIcon className="h-3 w-3 shrink-0 text-zinc-600" />
                      <span className="shrink-0 text-[11px] text-zinc-500">{a.running ? 'running' : 'stopped'}</span>
                    </div>
                    <div className="truncate text-[11px] text-zinc-500">{a.detail || a.description || a.id}</div>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    {working && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
                    {a.running ? (
                      <>
                        <button onClick={() => run(a, 'stop')} disabled={working} className={btn}><Square className="h-3 w-3" /> Stop</button>
                        <button onClick={() => run(a, 'restart')} disabled={working} className={btn}><RotateCw className="h-3 w-3" /> Restart</button>
                      </>
                    ) : (
                      <button onClick={() => run(a, 'start')} disabled={working} className={btn}><Play className="h-3 w-3" /> Start</button>
                    )}
                    <button onClick={() => setEditing(a)} disabled={working} className={btn}><Settings className="h-3 w-3" /> Config</button>
                    <button onClick={() => run(a, 'remove')} disabled={working} className={`${btn} hover:border-red-500/40 hover:text-red-600 dark:hover:text-red-400`}>
                      <Trash2 className="h-3 w-3" /> Remove
                    </button>
                  </div>
                </li>
              )
            })}
          </ul>
        </div>
      </div>

      {/* Child dialogs sit above (z-50) and must not close this one on click. */}
      <div onClick={(e) => e.stopPropagation()}>
        {creating && (
          <ArchetypeSpawnDialog
            takenSlugs={new Set((agents || []).filter((a) => a.kind === 'process').map((a) => a.id))}
            asUser={actAs}
            ownerLabel={userLabel}
            onClose={() => setCreating(false)}
            onSpawned={() => { setCreating(false); load() }}
          />
        )}
        {editing && (
          <AgentConfigEditor
            kind={editing.kind}
            identifier={editing.id}
            agentName={editing.name}
            initialDescription={editing.description}
            asUser={actAs}
            ownerLabel={userLabel}
            onClose={() => { setEditing(null); load() }}
          />
        )}
      </div>
    </div>,
    document.body,
  )
}
