// Shared HUD building blocks. Every screen renders inside <ScreenFrame>, which
// gives the one `.hud-screen` root the focus engine works in, a compact header
// (title, agent tabs, live dot, clock) and a single vertical scroll owner.

import type { ReactNode, Ref } from 'react'
import type { HudAgent } from './api'
import { useHudChat } from './chat/chatStore'
import { clock } from './format'
import { useActivate, useNow } from './hooks'
import { navigate, type Tab } from './router'

// ── Buttons and rows ──

export function Btn(props: {
  children: ReactNode
  onActivate: (() => void) | null
  variant?: 'primary' | 'ghost' | 'danger' | 'chip'
  disabled?: boolean
  fk?: string
  autoFocus?: boolean
  className?: string
  title?: string
}) {
  const act = useActivate(props.onActivate, { disabled: props.disabled })
  return (
    <div
      className={`hud-btn hud-btn--${props.variant || 'ghost'}${props.className ? ' ' + props.className : ''}`}
      data-fk={props.fk}
      data-autofocus={props.autoFocus ? '' : undefined}
      aria-label={props.title}
      {...act}
    >
      {props.children}
    </div>
  )
}

export function Row(props: {
  title: ReactNode
  meta?: ReactNode
  badge?: ReactNode
  onActivate: (() => void) | null
  fk?: string
  dim?: boolean
  autoFocus?: boolean
}) {
  const act = useActivate(props.onActivate)
  return (
    <div
      className={`hud-row${props.dim ? ' hud-row--dim' : ''}`}
      data-fk={props.fk}
      data-autofocus={props.autoFocus ? '' : undefined}
      {...act}
    >
      <div className="hud-row-main">
        <div className="hud-row-title">{props.title}</div>
        {props.meta ? <div className="hud-row-meta">{props.meta}</div> : null}
      </div>
      {props.badge != null ? <div className="hud-row-badge">{props.badge}</div> : null}
    </div>
  )
}

export function Pill(props: { children: ReactNode; tone?: 'ok' | 'warn' | 'err' | 'dim' }) {
  return <span className={`hud-pill hud-pill--${props.tone || 'dim'}`}>{props.children}</span>
}

/** Loading / error / empty placeholder. An error with `onRetry` gets a focusable Retry. */
export function StateView(props: {
  kind: 'loading' | 'error' | 'empty'
  message?: string
  onRetry?: () => void
  action?: { label: string; onActivate: () => void }
}) {
  const text = props.message ?? (props.kind === 'loading' ? 'Loading…' : props.kind === 'empty' ? 'Nothing here yet.' : 'Something went wrong.')
  return (
    <div className={`hud-state hud-state--${props.kind}`} role={props.kind === 'error' ? 'alert' : 'status'}>
      <div className="hud-state-text">{text}</div>
      {props.onRetry ? <Btn variant="primary" onActivate={props.onRetry} autoFocus>Retry</Btn> : null}
      {props.action ? <Btn variant="primary" onActivate={props.action.onActivate} autoFocus>{props.action.label}</Btn> : null}
    </div>
  )
}

// ── Header ──

const TAB_LABEL: Record<Tab, string> = { chat: 'Chat', files: 'Files', data: 'Data' }

function TabBtn(props: { agent: HudAgent; tab: Tab; active: boolean; badge?: number }) {
  const act = useActivate(props.active ? () => {} : () => navigate({ v: 'agent', a: props.agent.id, t: props.tab }, { replace: true }))
  return (
    <div className={`hud-tab${props.active ? ' hud-tab--on' : ''}`} data-fk={`tab-${props.tab}`} aria-selected={props.active} {...act}>
      {TAB_LABEL[props.tab]}
      {props.badge ? <span className="hud-tab-badge">{props.badge}</span> : null}
    </div>
  )
}

function LiveDot() {
  const connected = useHudChat((s) => s.connected)
  const busy = useHudChat((s) => s.busy)
  const agentId = useHudChat((s) => s.agentId)
  if (!agentId) return null
  const tone = !connected ? 'off' : busy ? 'busy' : 'on'
  const label = !connected ? 'offline' : busy ? 'thinking' : 'live'
  return <span className={`hud-dot hud-dot--${tone}`} aria-label={label} title={label} />
}

export function ScreenFrame({ scrollRef, ...props }: {
  title: string
  subtitle?: string
  /** With `tab`, shows the Chat / Files / Data tabs for this agent. */
  agent?: HudAgent | null
  tab?: Tab
  footer?: ReactNode
  children: ReactNode
  scrollRef?: Ref<HTMLDivElement>
  className?: string
}) {
  const now = useNow(15_000)
  const unread = useHudChat((s) => s.unread)
  const tabs: Tab[] = props.agent
    ? (['chat', 'files', 'data'] as Tab[]).filter((t) => t === 'chat' || (t === 'files' ? props.agent!.caps.files : props.agent!.caps.data))
    : []
  return (
    <main className={`hud-screen${props.className ? ' ' + props.className : ''}`}>
      <header className="hud-header">
        <div className="hud-title">
          <div className="hud-title-main">{props.title}</div>
          {props.subtitle ? <div className="hud-title-sub">{props.subtitle}</div> : null}
        </div>
        {props.agent && props.tab ? (
          <nav className="hud-tabs" aria-label="Sections">
            {tabs.map((t) => (
              <TabBtn key={t} agent={props.agent!} tab={t} active={t === props.tab} badge={t === 'chat' && props.tab !== 'chat' ? unread : 0} />
            ))}
          </nav>
        ) : null}
        <div className="hud-header-end">
          <LiveDot />
          <span className="hud-clock">{clock(now)}</span>
        </div>
      </header>
      <div className="hud-scroll" ref={scrollRef}>
        {props.children}
      </div>
      {props.footer ? <footer className="hud-footer">{props.footer}</footer> : null}
    </main>
  )
}
