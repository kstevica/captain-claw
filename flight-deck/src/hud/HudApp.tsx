// Smart-glasses HUD root.
//
//   /hud/        — the glasses app (600×600 Meta Ray-Ban Display, Rokid Lumen…)
//   /hud/pair    — approve a glasses sign-in from a phone / desktop
//
// Auth reuses the dashboard's store (JWT in memory + httpOnly refresh cookie)
// but none of its side effects: no hydrateAllStores, no polling stores, no
// dashboard bundle.

import { useEffect, useState } from 'react'
import { awaitAuthStatus, refreshAccessToken, useAuthStore } from '../stores/authStore'
import { AgentsScreen } from './agents/AgentsScreen'
import { lastAgentId, rememberAgent, useAgents } from './agentsStore'
import { PairApprovePage } from './auth/PairApprovePage'
import { PairScreen } from './auth/PairScreen'
import { chat } from './chat/chatStore'
import { ChatScreen } from './chat/ChatScreen'
import { RecordScreen } from './data/RecordScreen'
import { RowsScreen } from './data/RowsScreen'
import { TablesScreen } from './data/TablesScreen'
import { FileScreen } from './files/FileScreen'
import { FilesScreen } from './files/FilesScreen'
import { installFocusEngine } from './focus'
import { getRoute, initRouter, navigate, routeKey, useRoute, type Route } from './router'
import { ScreenFrame, StateView } from './ui'
import { registerWebMcpTools } from './webmcp'

export function HudApp() {
  if (window.location.pathname.replace(/\/+$/, '') === '/hud/pair') return <PairApprovePage />
  return <GlassesApp />
}

function GlassesApp() {
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  const [booted, setBooted] = useState(false)
  const [unreachable, setUnreachable] = useState(false)

  useEffect(() => installFocusEngine(), [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      // Never treat "Flight Deck unreachable" as "auth off" (see authStore).
      const enabled = await awaitAuthStatus(() => setUnreachable(true), () => cancelled)
      if (enabled === null || cancelled) return
      setUnreachable(false)
      if (enabled) await refreshAccessToken()
      if (!cancelled) setBooted(true)
    })()
    return () => { cancelled = true }
  }, [])

  if (!booted) {
    return (
      <main className="hud-screen hud-boot-screen">
        <div className="hud-state hud-state--loading">
          <div className="hud-state-text">{unreachable ? "Can't reach Flight Deck — retrying…" : 'Captain Claw…'}</div>
        </div>
      </main>
    )
  }
  if (authEnabled && !isAuthenticated) return <PairScreen />
  return <Shell />
}

/** Keep the 15-minute access token fresh: the glasses idle a lot (display
 *  sleeps after ~25 s) and an own-agent chat socket never refreshes it. */
function useTokenKeepAlive() {
  useEffect(() => {
    if (!useAuthStore.getState().authEnabled) return
    let lastRefresh = Date.now()
    const unsub = useAuthStore.subscribe((s, prev) => { if (s.token && s.token !== prev.token) lastRefresh = Date.now() })
    const id = setInterval(() => { void refreshAccessToken() }, 10 * 60_000)
    const onVis = () => {
      if (document.visibilityState === 'visible' && Date.now() - lastRefresh > 5 * 60_000) void refreshAccessToken()
    }
    document.addEventListener('visibilitychange', onVis)
    return () => { unsub(); clearInterval(id); document.removeEventListener('visibilitychange', onVis) }
  }, [])
}

function Shell() {
  const route = useRoute()
  const agents = useAgents((s) => s.agents)
  const agentsError = useAgents((s) => s.error)
  // Router state must exist before the first render (lazy init runs once).
  const [fromUrl] = useState(() => initRouter({ v: 'agents' }))

  useTokenKeepAlive()

  useEffect(() => {
    void useAgents.getState().refresh().then(() => {
      // A plain launch (no screen in the URL) resumes the last agent — once
      // we know it still exists, so Back from it lands on the agent list.
      if (fromUrl || getRoute().v !== 'agents') return
      const last = lastAgentId()
      const found = last ? useAgents.getState().agents?.find((a) => a.id === last) : null
      if (found?.running) navigate({ v: 'agent', a: found.id, t: 'chat' })
    })
    const unregister = registerWebMcpTools()
    return () => {
      unregister()
      chat.detach()
      useAgents.getState().reset()
    }
  }, [fromUrl])

  // Attach the chat session to the agent the route points at.
  const agentId = 'a' in route ? route.a : null
  const agent = agentId && agents ? agents.find((a) => a.id === agentId) ?? null : null
  useEffect(() => {
    if (!agent) return
    rememberAgent(agent.id)
    if (agent.running) chat.attach(agent)
  }, [agent])

  // Unread counting: the Chat tab is "visible" only on the agent chat route.
  const chatVisible = route.v === 'agent' && route.t === 'chat'
  useEffect(() => { chat.setChatVisible(chatVisible) }, [chatVisible])

  return <div className="hud-app" key={routeKey(route)}>{renderRoute(route, agents, agentsError)}</div>
}

function renderRoute(route: Route, agents: ReturnType<typeof useAgents.getState>['agents'], error: string | null) {
  if (route.v === 'agents') return <AgentsScreen />
  if (!agents) {
    return (
      <ScreenFrame title="Captain Claw">
        {error
          ? <StateView kind="error" message={error} onRetry={() => { void useAgents.getState().refresh() }} />
          : <StateView kind="loading" />}
      </ScreenFrame>
    )
  }
  const agent = agents.find((a) => a.id === route.a)
  if (!agent) {
    return (
      <ScreenFrame title="Agent not found">
        <StateView kind="empty" message="This agent is gone or no longer shared with you."
          action={{ label: 'All agents', onActivate: () => navigate({ v: 'agents' }, { replace: true }) }} />
      </ScreenFrame>
    )
  }
  if (!agent.running) {
    return (
      <ScreenFrame title={agent.name} subtitle="stopped">
        <StateView kind="empty" message={`${agent.name} is not running. Start it from Flight Deck.`}
          action={{ label: 'All agents', onActivate: () => navigate({ v: 'agents' }, { replace: true }) }} />
      </ScreenFrame>
    )
  }
  switch (route.v) {
    case 'agent':
      if (route.t === 'files' && agent.caps.files) return <FilesScreen agent={agent} />
      if (route.t === 'data' && agent.caps.data) return <TablesScreen agent={agent} />
      return <ChatScreen agent={agent} />
    case 'file':
      return <FileScreen agent={agent} fileKey={route.p} name={route.n} />
    case 'rows':
      return <RowsScreen agent={agent} table={route.tb} />
    case 'record':
      return <RecordScreen agent={agent} table={route.tb} index={route.i} />
  }
}
