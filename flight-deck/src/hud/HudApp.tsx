// Smart-glasses HUD root.
//
//   /hud/        — the glasses app (600×600 Meta Ray-Ban Display, Rokid Lumen…)
//   /hud/pair    — approve a glasses sign-in from a phone / desktop
//
// Auth reuses the dashboard's store (JWT in memory + httpOnly refresh cookie)
// but none of its side effects: no hydrateAllStores, no polling stores, no
// dashboard bundle.

import { useEffect, useRef, useState } from 'react'
import type { ComponentProps } from 'react'
import { awaitAuthStatus, refreshAccessToken, useAuthStore } from '../stores/authStore'
import { AgentsScreen } from './agents/AgentsScreen'
import { restoreSession } from './api'
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
import { useAutoFocus } from './hooks'
import { collapseHistory, getRoute, initRouter, navigate, navigateUp, routeKey, useRoute, type Route } from './router'
import { ScreenFrame, StateView } from './ui'
import { registerWebMcpTools } from './webmcp'

export function HudApp() {
  if (window.location.pathname.replace(/\/+$/, '') === '/hud/pair') return <PairApprovePage />
  return <GlassesApp />
}

function GlassesApp() {
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  const signingOut = useAuthStore((s) => s.signingOut)
  const [booted, setBooted] = useState(false)
  const [unreachable, setUnreachable] = useState(false)
  const wasSignedIn = useRef(false)

  useEffect(() => installFocusEngine(), [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      // Never treat "Flight Deck unreachable" as "auth off" (see authStore).
      const enabled = await awaitAuthStatus(() => setUnreachable(true), () => cancelled)
      if (enabled === null || cancelled) return
      setUnreachable(false)
      // A refresh that got no real answer (the link blinked, a stalled
      // proxy) retries instead of sending the wearer to re-pair.
      for (let attempt = 0; enabled; attempt++) {
        const r = await restoreSession()
        if (cancelled) return
        if (r !== 'offline') break
        setUnreachable(true)
        await new Promise((resolve) => setTimeout(resolve, Math.min(1000 * 2 ** attempt, 10_000)))
        if (cancelled) return
      }
      setUnreachable(false)
      if (!cancelled) setBooted(true)
    })()
    return () => { cancelled = true }
  }, [])

  // The session ended mid-use (a refresh failed): the app's screens are
  // gone, so take their history entries with them — otherwise each Back on
  // the sign-in screen pops one with nothing visible happening. A sign-out
  // reloads instead (signingOut), from the agent list.
  useEffect(() => {
    if (!booted || !authEnabled || signingOut) return
    if (isAuthenticated) wasSignedIn.current = true
    else if (wasSignedIn.current) { wasSignedIn.current = false; collapseHistory() }
  }, [booted, authEnabled, isAuthenticated, signingOut])

  if (!booted) return <BootScreen text={unreachable ? "Can't reach Flight Deck — retrying…" : 'Captain Claw…'} />
  // Signing out reloads the page in a moment; the sign-in screen mounted
  // meanwhile would request a pairing code only to drop it.
  if (authEnabled && !isAuthenticated) return signingOut ? <BootScreen text="Signing out…" /> : <PairScreen />
  return <Shell />
}

function BootScreen({ text }: { text: string }) {
  return (
    <main className="hud-screen hud-boot-screen">
      <div className="hud-state hud-state--loading">
        <div className="hud-state-text">{text}</div>
      </div>
    </main>
  )
}

/** Refresh when the access token has less than this left. */
const REFRESH_WITHIN_MS = 3 * 60_000

/** Whether the access token's `exp` is within `ms` (or unreadable). No
 *  verification — it only decides whether to refresh. */
function tokenExpiresWithin(ms: number): boolean {
  const token = useAuthStore.getState().token
  if (!token) return true
  try {
    const part = token.split('.')[1] || ''
    const b64 = part.replace(/-/g, '+').replace(/_/g, '/')
    const json = JSON.parse(atob(b64 + '='.repeat((4 - (b64.length % 4)) % 4))) as { exp?: unknown }
    const exp = Number(json.exp)
    return !Number.isFinite(exp) || exp * 1000 - Date.now() < ms
  } catch {
    return true
  }
}

/** Keep the 15-minute access token from lapsing while the HUD is in use (an
 *  own-agent chat socket never refreshes it): check once a minute and when
 *  the display wakes, and refresh only when it is about to expire. Every
 *  refresh rotates the refresh cookie, and one whose reply is lost — the
 *  link is often still reconnecting at wake — ends the session, so no
 *  rotation that isn't needed. */
function useTokenKeepAlive() {
  useEffect(() => {
    if (!useAuthStore.getState().authEnabled) return
    const check = () => {
      if (document.visibilityState !== 'visible') return
      const s = useAuthStore.getState()
      if (!s.isAuthenticated || s.signingOut) return
      if (tokenExpiresWithin(REFRESH_WITHIN_MS)) void refreshAccessToken()
    }
    const id = setInterval(check, 60_000)
    document.addEventListener('visibilitychange', check)
    return () => { clearInterval(id); document.removeEventListener('visibilitychange', check) }
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

/** Loading / error / agent-gone screen of a route: its Retry or All agents
 *  gets focus, so a pinch works without a swipe first. */
function RouteState({ title, subtitle, ...view }: { title: string; subtitle?: string } & ComponentProps<typeof StateView>) {
  useAutoFocus(view.kind !== 'loading')
  return (
    <ScreenFrame title={title} subtitle={subtitle}>
      <StateView {...view} />
    </ScreenFrame>
  )
}

function renderRoute(route: Route, agents: ReturnType<typeof useAgents.getState>['agents'], error: string | null) {
  if (route.v === 'agents') return <AgentsScreen />
  if (!agents) {
    return error
      ? <RouteState title="Captain Claw" kind="error" message={error} onRetry={() => { void useAgents.getState().refresh() }} />
      : <RouteState title="Captain Claw" kind="loading" />
  }
  const agent = agents.find((a) => a.id === route.a)
  if (!agent) {
    return (
      <RouteState title="Agent not found" kind="empty" message="This agent is gone or no longer shared with you."
        action={{ label: 'All agents', onActivate: () => navigateUp({ v: 'agents' }) }} />
    )
  }
  if (!agent.running) {
    return (
      <RouteState title={agent.name} subtitle="stopped" kind="empty" message={`${agent.name} is not running. Start it from Flight Deck.`}
        action={{ label: 'All agents', onActivate: () => navigateUp({ v: 'agents' }) }} />
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
