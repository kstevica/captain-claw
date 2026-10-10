// The HUD's home: the signed-in user's agents (own processes / containers and
// agents shared with them). Running agents are focus stops that open the
// agent's chat; stopped ones are shown dimmed in one reading block (Flight
// Deck starts agents, not the glasses). Ends with Refresh and Sign out.

import { useEffect, useRef, useState } from 'react'
import { logoutUser, useAuthStore } from '../../stores/authStore'
import type { HudAgent } from '../api'
import { lastAgentId, useAgents } from '../agentsStore'
import { useAutoFocus } from '../hooks'
import { navigate } from '../router'
import { Btn, Pill, Row, ScreenFrame, StateView } from '../ui'
import './agents.css'

/** Re-list on arrival when the list is older than this (agents start / stop). */
const STALE_MS = 60_000
/** Sign out needs a second pinch within this window. */
const CONFIRM_MS = 5_000

function kindLabel(a: HudAgent): string {
  if (a.kind === 'shared') return a.ownerName ? `shared by ${a.ownerName}` : 'shared'
  return a.kind === 'container' ? 'docker' : 'process'
}

function agentMeta(a: HudAgent): string {
  return a.model ? `${kindLabel(a)} · ${a.model}` : kindLabel(a)
}

function refresh() {
  void useAgents.getState().refresh()
}

function AgentRow(props: { agent: HudAgent; last: boolean }) {
  const a = props.agent
  const badge = (
    <span className="hud-agents-badges">
      {props.last ? <Pill>last</Pill> : null}
      {a.running ? <Pill tone="ok">live</Pill> : null}
    </span>
  )
  return (
    <Row
      title={a.name}
      meta={agentMeta(a)}
      badge={badge}
      fk={a.running ? `agent-${a.id}` : undefined}
      dim={!a.running}
      autoFocus={a.running && props.last}
      onActivate={a.running ? () => navigate({ v: 'agent', a: a.id, t: 'chat' }) : null}
    />
  )
}

export function AgentsScreen() {
  const agents = useAgents((s) => s.agents)
  const loading = useAgents((s) => s.loading)
  const error = useAgents((s) => s.error)
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const userName = useAuthStore((s) => s.user?.display_name || s.user?.email || '')
  const signingOut = useAuthStore((s) => s.signingOut)
  const [lastId] = useState(lastAgentId)
  const [confirming, setConfirming] = useState(false)
  const [signOutFailed, setSignOutFailed] = useState(false)
  const confirmTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // HudApp loads the list once; load it here if that has not happened, and
  // re-list a stale one (in the background — the old list stays on screen).
  useEffect(() => {
    const st = useAgents.getState()
    if (st.loading) return
    if (!st.agents || Date.now() - st.loadedAt > STALE_MS) void st.refresh()
  }, [])

  useEffect(() => () => clearTimeout(confirmTimer.current), [])

  useAutoFocus(agents !== null || !!error)

  const signOut = () => {
    if (signingOut) return
    if (!confirming) {
      // A stray pinch must not sign the glasses out (re-pairing needs a phone).
      setConfirming(true)
      setSignOutFailed(false)
      clearTimeout(confirmTimer.current)
      confirmTimer.current = setTimeout(() => setConfirming(false), CONFIRM_MS)
      return
    }
    clearTimeout(confirmTimer.current)
    setConfirming(false)
    // Success reloads the page (→ pairing screen); failure keeps the session.
    void logoutUser().then((ok) => { if (!ok) setSignOutFailed(true) })
  }

  const subtitle = authEnabled === false ? 'Sign-in off' : userName || undefined

  const signOutBtn = authEnabled ? (
    <Btn
      variant="chip"
      className={confirming ? 'hud-agents-confirm' : undefined}
      fk="agents-signout"
      onActivate={signOut}
      disabled={signingOut}
    >
      {signingOut ? 'Signing out…' : confirming ? 'Confirm sign out' : 'Sign out'}
    </Btn>
  ) : null

  const footerNotes = (
    <>
      {signOutFailed ? (
        <p className="hud-note hud-agents-error" role="alert">Sign-out failed — you are still signed in. Try again.</p>
      ) : null}
      {authEnabled === false ? (
        <p className="hud-note hud-agents-warn">
          <Pill tone="warn">Sign-in off</Pill> Sign-in is off on this deck — anyone with the URL can use it.
        </p>
      ) : null}
    </>
  )

  if (!agents) {
    return (
      <ScreenFrame title="Agents" subtitle={subtitle}>
        {error ? (
          <>
            <StateView kind="error" message={error} onRetry={refresh} />
            {signOutBtn ? <div className="hud-actions hud-agents-actions">{signOutBtn}</div> : null}
            {footerNotes}
          </>
        ) : (
          <StateView kind="loading" message="Loading agents…" />
        )}
      </ScreenFrame>
    )
  }

  const running = agents.filter((a) => a.running)
  const stopped = agents.filter((a) => !a.running)

  const actions = (
    <div className="hud-actions hud-agents-actions">
      <Btn variant="chip" fk="agents-refresh" onActivate={refresh}>{loading ? 'Refreshing…' : 'Refresh'}</Btn>
      {signOutBtn}
    </div>
  )

  if (agents.length === 0) {
    return (
      <ScreenFrame title="Agents" subtitle={subtitle}>
        <StateView kind="empty" message="No agents yet — create one in Flight Deck." />
        {actions}
        {footerNotes}
      </ScreenFrame>
    )
  }

  return (
    <ScreenFrame title="Agents" subtitle={subtitle}>
      {error ? <p className="hud-note hud-agents-error">Couldn't refresh — {error}</p> : null}
      {running.map((a) => <AgentRow key={a.id} agent={a} last={a.id === lastId} />)}
      {running.length === 0 ? (
        <p className="hud-note">No agent is running — start one in Flight Deck.</p>
      ) : null}
      {stopped.length > 0 ? (
        <>
          <div className="hud-section-label">Stopped</div>
          {/* Not actionable, but readable: one focus stop for the whole group
              so the D-pad can scroll through it. */}
          <div className="hud-block hud-agents-stopped" tabIndex={0}>
            {stopped.map((a) => <AgentRow key={a.id} agent={a} last={a.id === lastId} />)}
          </div>
        </>
      ) : null}
      {actions}
      {footerNotes}
    </ScreenFrame>
  )
}
