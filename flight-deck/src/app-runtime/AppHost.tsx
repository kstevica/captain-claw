import { useEffect, useState } from 'react'
import { useAppRuntime } from './store'
import { SurfaceRenderer } from './surfaces/SurfaceRenderer'
import { AuthoringDialog } from './AuthoringDialog'
import { useMCPStore } from '../stores/mcpStore'

export function AppHost() {
  const {
    manifest,
    agentId,
    surfaceId,
    availableApps,
    appsLoading,
    manifestLoading,
    error,
    refreshAppList,
    loadAgent,
    setSurface,
  } = useAppRuntime()

  const [authoring, setAuthoring] = useState<{ mode: 'new' | 'edit' } | null>(null)
  const pendingAuthoring = useAppRuntime((s) => s.pendingAuthoring)
  const clearAuthoringRequest = useAppRuntime((s) => s.clearAuthoringRequest)
  const mcpServers = useMCPStore((s) => s.servers)
  const refreshMCPServers = useMCPStore((s) => s.refresh)
  useEffect(() => { refreshMCPServers() }, [refreshMCPServers])

  // Fetch app list once on mount.
  useEffect(() => {
    refreshAppList()
  }, [refreshAppList])

  // Honour an authoring request raised from elsewhere (e.g. sidebar).
  useEffect(() => {
    if (pendingAuthoring) {
      setAuthoring({ mode: pendingAuthoring })
      clearAuthoringRequest()
    }
  }, [pendingAuthoring, clearAuthoringRequest])

  const onAuthoringSaved = async (newAgentId: string) => {
    await refreshAppList()
    if (newAgentId) await loadAgent(newAgentId)
  }

  // Pick a default agent once apps are loaded.
  useEffect(() => {
    if (availableApps.length === 0) return
    const candidate = agentId && availableApps.some((a) => a.id === agentId)
      ? agentId
      : availableApps[0].id
    if (!manifest || manifest.agent.id !== candidate) {
      loadAgent(candidate)
    }
  }, [availableApps, agentId, manifest, loadAgent])

  if (appsLoading && availableApps.length === 0) {
    return <CenterPanel>Loading apps…</CenterPanel>
  }

  if (availableApps.length === 0) {
    return (
      <>
        <CenterPanel>
          <h2 className="mb-2 text-lg font-semibold text-zinc-200">No apps registered</h2>
          <p className="text-sm text-zinc-500">
            Describe what you want and Captain Claw will write the manifest for you.
          </p>
          <button
            onClick={() => setAuthoring({ mode: 'new' })}
            className="mt-4 rounded bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500"
          >
            + Describe a new app
          </button>
          {error && <p className="mt-3 text-xs text-red-400">{error}</p>}
        </CenterPanel>
        <AuthoringDialog
          open={authoring?.mode === 'new'}
          onClose={() => setAuthoring(null)}
          onSaved={onAuthoringSaved}
        />
      </>
    )
  }

  if (manifestLoading && !manifest) {
    return <CenterPanel>Loading manifest…</CenterPanel>
  }

  if (!manifest) {
    return (
      <CenterPanel>
        <p className="text-sm text-zinc-500">Failed to load manifest.</p>
        {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
      </CenterPanel>
    )
  }

  const surface = surfaceId ? manifest.surfaces[surfaceId] : null
  const surfaces = Object.values(manifest.surfaces)
  const declaredMcp = manifest.agent.mcp_server
  // `__framework__` and empty/missing mean "use the built-in tool catalogue",
  // so don't flag those as unknown servers.
  const isBuiltin = !declaredMcp || declaredMcp === '__framework__'
  const mcpKnown = isBuiltin
    ? true
    : mcpServers.length === 0
      ? true   // server list not loaded yet — don't flag a false positive
      : mcpServers.some((s) => s.name === declaredMcp)

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="flex items-center justify-between border-b border-zinc-800 bg-zinc-950 px-4 py-3">
        <div className="flex items-center gap-3">
          <AppSelector
            available={availableApps}
            currentId={manifest.agent.id}
            onSelect={(id) => loadAgent(id)}
          />
          {manifest.agent.tagline && (
            <span className="text-[11px] text-zinc-500">{manifest.agent.tagline}</span>
          )}
          <div className="ml-2 flex gap-1">
            <button
              onClick={() => setAuthoring({ mode: 'new' })}
              className="rounded border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-zinc-800"
              title="Create a new app from a description"
            >
              + New
            </button>
            <button
              onClick={() => setAuthoring({ mode: 'edit' })}
              className="rounded border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-zinc-800"
              title="Edit this app via natural language"
            >
              Edit
            </button>
          </div>
        </div>
        <nav className="flex gap-1">
          {surfaces.map((s) => (
            <button
              key={s.id}
              onClick={() => setSurface(s.id)}
              className={
                s.id === surfaceId
                  ? 'rounded bg-violet-600/20 px-3 py-1 text-xs font-medium text-violet-300'
                  : 'rounded px-3 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
              }
            >
              {s.label ?? s.id}
            </button>
          ))}
        </nav>
      </header>

      {!mcpKnown && (
        <div className="border-b border-amber-800/40 bg-amber-950/40 px-4 py-2 text-xs text-amber-200">
          <span className="font-medium">This app is bound to MCP server </span>
          <code className="rounded bg-amber-900/40 px-1 py-0.5 font-mono">{declaredMcp || '(none)'}</code>
          <span>, which isn&rsquo;t registered. Tool calls will fail with 404.</span>
          <button
            onClick={() => setAuthoring({ mode: 'edit' })}
            className="ml-3 rounded border border-amber-700/60 px-2 py-0.5 text-[11px] font-medium text-amber-100 hover:bg-amber-900/60"
          >
            Edit app
          </button>
        </div>
      )}

      <main className="flex-1 overflow-auto p-4">
        {surface ? (
          <SurfaceRenderer manifest={manifest} surface={surface} />
        ) : (
          <div className="text-sm text-zinc-500">No surface selected.</div>
        )}
      </main>

      <AuthoringDialog
        open={authoring !== null}
        onClose={() => setAuthoring(null)}
        onSaved={onAuthoringSaved}
        baseAgentId={authoring?.mode === 'edit' ? manifest.agent.id : undefined}
        baseMcpServer={authoring?.mode === 'edit' ? manifest.agent.mcp_server : manifest.agent.mcp_server}
      />
    </div>
  )
}

function CenterPanel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="max-w-md text-center">{children}</div>
    </div>
  )
}

interface AppSelectorProps {
  available: { id: string; name: string; tagline?: string }[]
  currentId: string
  onSelect: (id: string) => void
}

function AppSelector({ available, currentId, onSelect }: AppSelectorProps) {
  if (available.length <= 1) {
    const a = available[0]
    return <h1 className="text-base font-semibold text-zinc-100">{a?.name ?? currentId}</h1>
  }
  return (
    <div className="relative">
      <select
        value={currentId}
        onChange={(e) => onSelect(e.target.value)}
        className="cursor-pointer appearance-none rounded border border-zinc-800 bg-zinc-900 py-1 pl-2 pr-7 text-base font-semibold text-zinc-100 hover:border-zinc-700 focus:border-violet-500 focus:outline-none"
      >
        {available.map((a) => (
          <option key={a.id} value={a.id}>{a.name}</option>
        ))}
      </select>
      <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500">▾</span>
    </div>
  )
}
