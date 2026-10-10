// Chat with one agent on the glasses.
//
// Layout (one scroll owner, D-pad walk top → bottom):
//   transcript — each message is ONE focusable reading block (the focus
//                engine pages through tall replies); markdown-file chips sit
//                right after the reply that mentions them;
//   tail       — busy line, approval card, error, next-step chips, then the
//                Stop / New chat / Read aloud chips;
//   footer     — the composer (textarea + Send).
// While the wearer is at the composer the view follows the conversation;
// once they focus an older message it stays put.

import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { FocusEvent } from 'react'
import type { HudAgent } from '../api'
import { focusEl } from '../focus'
import { truncate } from '../format'
import { useAutoFocus } from '../hooks'
import { HudMarkdown } from '../markdown/HudMarkdown'
import { getRoute, navigate } from '../router'
import { Btn, ScreenFrame } from '../ui'
import { chat, resolveFileRef, useHudChat, type Approval, type HudMsg } from './chatStore'
import { Composer } from './Composer'
import { findFileRefs, refName } from './text'
import './chat.css'

const NO_MSGS: HudMsg[] = []
/** File chips are offered for this many of the latest replies. */
const FILE_REPLIES = 3
const MAX_STEPS = 4

function focusComposer(): void {
  const field = document.querySelector<HTMLElement>('.hud-screen .hud-composer-field')
  if (field) focusEl(field)
}

const MsgBlock = memo(function MsgBlock({ m }: { m: HudMsg }) {
  const fk = `msg-${m.id}`
  if (m.role === 'user') {
    return (
      <div className="hud-block hud-msg hud-msg--user" tabIndex={0} data-fk={fk}>
        <div className="hud-msg-label">You</div>
        <div className="hud-msg-text">{m.text}</div>
      </div>
    )
  }
  if (m.role === 'system') {
    return (
      <div className="hud-block hud-msg hud-msg--system" tabIndex={0} data-fk={fk}>
        <div className="hud-msg-text">{m.text}</div>
      </div>
    )
  }
  return (
    <div className="hud-block hud-msg hud-msg--assistant" tabIndex={0} data-fk={fk}>
      <HudMarkdown text={m.text} />
    </div>
  )
})

function FileChips({ msgId, refs, onOpen }: { msgId: string; refs: string[]; onOpen: (ref: string) => void }) {
  return (
    <div className="hud-chat-files">
      {refs.map((ref, i) => (
        <Btn key={ref} variant="chip" className="hud-chat-file" fk={`file-${msgId}-${i}`} title={`Open ${refName(ref)}`}
          onActivate={() => onOpen(ref)}>
          <span aria-hidden="true">📄</span>
          <span className="hud-chat-file-name">{refName(ref)}</span>
        </Btn>
      ))}
    </div>
  )
}

function StatusLine({ status, narration }: { status: string; narration: string }) {
  return (
    <div className="hud-chat-status" role="status">
      <span className="hud-chat-status-dot" aria-hidden="true" />
      <div className="hud-chat-status-body">
        <div className="hud-chat-status-text">{status || 'Thinking…'}</div>
        {narration ? <div className="hud-chat-narration">{narration}</div> : null}
      </div>
    </div>
  )
}

function categoryLabel(category: string): string {
  if (!category) return ''
  if (category === 'peer_consult') return 'Ask another agent'
  const t = category.replace(/_/g, ' ')
  return t[0].toUpperCase() + t.slice(1)
}

function ApprovalCard({ approval }: { approval: Approval }) {
  const answer = (ok: boolean) => {
    chat.respondApproval(ok)
    focusComposer()
  }
  const cat = categoryLabel(approval.category)
  return (
    <section className="hud-chat-approval" aria-label="Approval needed">
      <div className="hud-chat-approval-title">Approval needed{cat ? ` · ${cat}` : ''}</div>
      <div className="hud-block hud-chat-approval-msg" tabIndex={0} data-fk={`approval-${approval.id}`}>
        {approval.message}
      </div>
      <div className="hud-actions">
        <Btn variant="primary" fk="approval-yes" onActivate={() => answer(true)}>Approve</Btn>
        <Btn variant="danger" fk="approval-no" onActivate={() => answer(false)}>Deny</Btn>
      </div>
    </section>
  )
}

export function ChatScreen({ agent }: { agent: HudAgent }) {
  // Until the shell attaches the session to this agent, show nothing of the
  // previous one.
  const mine = useHudChat((s) => s.agentId === agent.id)
  const messages = useHudChat((s) => s.messages)
  const busy = useHudChat((s) => s.busy)
  const status = useHudChat((s) => s.status)
  const narration = useHudChat((s) => s.narration)
  const approval = useHudChat((s) => s.approval)
  const error = useHudChat((s) => s.error)
  const nextSteps = useHudChat((s) => s.nextSteps)
  const tts = useHudChat((s) => s.tts)
  const closed = useHudChat((s) => s.closed)
  const shown = mine ? messages : NO_MSGS

  const scrollRef = useRef<HTMLDivElement>(null)
  const logRef = useRef<HTMLDivElement>(null)
  /** data-fk of the message block the wearer last focused. */
  const enteredRef = useRef<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [confirmNew, setConfirmNew] = useState(false)
  const confirmTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => { chat.markRead() }, [])
  useEffect(() => () => {
    if (noticeTimer.current) clearTimeout(noticeTimer.current)
    if (confirmTimer.current) clearTimeout(confirmTimer.current)
  }, [])
  useAutoFocus(true)

  const flash = useCallback((text: string) => {
    setNotice(text)
    if (noticeTimer.current) clearTimeout(noticeTimer.current)
    noticeTimer.current = setTimeout(() => { noticeTimer.current = null; setNotice(null) }, 3500)
  }, [])

  // Markdown files mentioned in the latest replies.
  const fileRefs = useMemo(() => {
    const map = new Map<string, string[]>()
    if (!agent.caps.files) return map
    let seen = 0
    for (let i = shown.length - 1; i >= 0 && seen < FILE_REPLIES; i--) {
      const m = shown[i]
      if (m.role !== 'assistant') continue
      seen++
      const refs = findFileRefs(m.text, 3)
      if (refs.length) map.set(m.id, refs)
    }
    return map
  }, [shown, agent.caps.files])

  const openRef = useCallback((ref: string) => {
    void (async () => {
      try {
        const file = await resolveFileRef(agent, ref)
        // The wearer may have moved on while the list loaded.
        const r = getRoute()
        if (r.v !== 'agent' || r.a !== agent.id || r.t !== 'chat') return
        if (file) navigate({ v: 'file', a: agent.id, p: file.key, n: file.name })
        else flash('File not found')
      } catch (e) {
        flash(e instanceof Error && e.message ? e.message : 'Could not list files')
      }
    })()
  }, [agent, flash])

  /** The newest message if it is a reply taller than the view. */
  const tallLastReply = useCallback((): HTMLElement | null => {
    const sc = scrollRef.current
    const log = logRef.current
    if (!sc || !log) return null
    const blocks = log.querySelectorAll<HTMLElement>('.hud-msg')
    const last = blocks[blocks.length - 1]
    if (!last || !last.classList.contains('hud-msg--assistant')) return null
    return last.offsetHeight > sc.clientHeight - 24 ? last : null
  }, [])

  // Follow the conversation while the wearer is not reading back: the start
  // of a long new reply, else the bottom (status, chips, newest message).
  const follow = useCallback(() => {
    const sc = scrollRef.current
    if (!sc) return
    const active = document.activeElement
    const inside = active instanceof HTMLElement && active !== sc && sc.contains(active)
    if (inside && active.closest('[data-chat-turn]')) return
    if (!inside) {
      const tall = tallLastReply()
      if (tall) {
        sc.scrollTop += tall.getBoundingClientRect().top - sc.getBoundingClientRect().top - 8
        return
      }
    }
    sc.scrollTop = sc.scrollHeight
    if (inside) {
      // Keep the focused tail control (approval, chips) in view.
      const over = sc.getBoundingClientRect().top - active.getBoundingClientRect().top
      if (over > 0) sc.scrollTop -= over + 8
    }
  }, [tallLastReply])

  useLayoutEffect(() => { follow() }, [follow, shown, busy, status, narration, approval, error, nextSteps, notice, mine])

  // Late layout (fonts, markdown that renders in a second pass) moves things.
  useEffect(() => {
    const log = logRef.current
    if (!log || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => follow())
    ro.observe(log)
    return () => ro.disconnect()
  }, [follow])

  // Up from the composer: into the start of a long new reply first (rather
  // than its end, then reading backwards), once per reply.
  const jumpToReply = useCallback((): boolean => {
    const el = tallLastReply()
    if (!el || enteredRef.current === el.dataset.fk) return false
    focusEl(el)
    return true
  }, [tallLastReply])

  const onLogFocus = (e: FocusEvent<HTMLDivElement>) => {
    const fk = e.target instanceof HTMLElement ? e.target.dataset.fk : undefined
    if (fk && fk.startsWith('msg-')) enteredRef.current = fk
  }

  const sendStep = (action: string) => {
    if (chat.send(action)) focusComposer()
    else flash(useHudChat.getState().connected ? 'Wait for the reply to finish' : 'Not connected')
  }

  const onNewChat = () => {
    if (confirmTimer.current) { clearTimeout(confirmTimer.current); confirmTimer.current = null }
    if (!confirmNew) {
      setConfirmNew(true)
      confirmTimer.current = setTimeout(() => { confirmTimer.current = null; setConfirmNew(false) }, 4000)
      return
    }
    setConfirmNew(false)
    chat.newSession()
    focusComposer()
  }

  const onStop = () => {
    chat.cancel()
    focusComposer()
  }

  const steps = mine && !busy ? nextSteps.slice(0, MAX_STEPS) : []

  return (
    <ScreenFrame
      title={agent.name}
      agent={agent}
      tab="chat"
      scrollRef={scrollRef}
      className="hud-chat"
      footer={<Composer onArrowUp={jumpToReply} />}
    >
      <div className="hud-chat-log" ref={logRef} onFocus={onLogFocus}>
        {shown.length === 0 ? (
          <p className="hud-chat-empty">Ask {agent.name} anything. Pinch the field below to write or speak.</p>
        ) : null}
        {shown.map((m) => {
          const refs = fileRefs.get(m.id)
          return (
            <div className="hud-chat-turn" key={m.id} data-chat-turn="">
              <MsgBlock m={m} />
              {refs ? <FileChips msgId={m.id} refs={refs} onOpen={openRef} /> : null}
            </div>
          )
        })}
      </div>

      {mine && busy ? <StatusLine status={status} narration={narration} /> : null}
      {mine && approval ? <ApprovalCard approval={approval} /> : null}
      {mine && error ? <div className="hud-chat-error" role="alert">{error}</div> : null}
      {notice ? <div className="hud-chat-notice" role="status">{notice}</div> : null}

      {steps.length ? (
        <div className="hud-actions hud-chat-steps">
          {steps.map((o, i) => (
            <Btn key={`${i}-${o.label}`} variant="chip" fk={`step-${i}`} onActivate={() => sendStep(o.action)}>
              {truncate(o.label, 48)}
            </Btn>
          ))}
        </div>
      ) : null}

      <div className="hud-actions hud-chat-tools">
        {mine && closed ? (
          <Btn variant="chip" fk="chat-reconnect" onActivate={() => chat.reconnect()}>Reconnect</Btn>
        ) : null}
        {mine && busy ? (
          <Btn variant="chip" className="hud-chat-stop" fk="chat-stop" onActivate={onStop}>Stop</Btn>
        ) : null}
        <Btn variant="chip" className={confirmNew ? 'hud-chat-confirm' : undefined} fk="chat-new" onActivate={onNewChat}>
          {confirmNew ? 'Start a new chat?' : 'New chat'}
        </Btn>
        <Btn variant="chip" fk="chat-tts" onActivate={() => chat.setTts(!tts)}>
          Read aloud: {tts ? 'On' : 'Off'}
        </Btn>
      </div>
    </ScreenFrame>
  )
}
