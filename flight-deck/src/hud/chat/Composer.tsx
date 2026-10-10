// Chat composer (screen footer): a plain <textarea> so the glasses open their
// system composer (handwriting / dictation / keyboard) on a pinch, and a Send
// button.
//
// Glasses facts this is built around (Meta Ray-Ban Display, fw v127+):
//   - focus + pinch on the field opens the composer; programmatic focus does
//     not. Committed text arrives in ONE update: `input`, then `change` — no
//     per-key keydowns;
//   - the pinch that closes the composer can arrive late as Enter / click on
//     whatever is focused next, and the host may reset focus to the first
//     control. So on `change` we walk focus to Send (pinned) and mute every
//     activation for 800 ms from that move (Glasscast's on-device guard,
//     added after this pinch pressed its Send by itself), making
//     "pinch field → speak → pinch Send" the whole flow;
//   - a message Meta AI drafted by voice (WebMCP) lands here unsent, with
//     focus on Send: the wearer reads it and pinches.
//   - Enter sends only for physical typing (desktop / phone keyboard): on the
//     glasses Enter on the field is the pinch that opens the composer.
// The app stays usable without the composer (next-step chips, WebMCP).

import { useCallback, useEffect, useLayoutEffect, useRef } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import { IS_GLASSES } from '../device'
import { focusEl, pinFocus, quietActivations } from '../focus'
import { Btn } from '../ui'
import { chat, useHudChat } from './chatStore'
import './chat.css'

const MAX_FIELD_PX = 104 // three lines of 20 px text, then the field scrolls
const TYPING_WINDOW_MS = 10_000
/** Late composer pinch guard (device-tested length), from the focus move. */
const COMPOSER_GUARD_MS = 800
const NAV_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Tab'])

export function Composer(props: {
  /** ArrowUp out of the composer; return true when it moved focus itself. */
  onArrowUp?: () => boolean
}) {
  const draft = useHudChat((s) => s.draft)
  const connected = useHudChat((s) => s.connected)
  const busy = useHudChat((s) => s.busy)
  const closed = useHudChat((s) => s.closed)
  const staged = useHudChat((s) => s.draftStaged)
  const fieldRef = useRef<HTMLTextAreaElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  /** Last printable keydown in the field (physical keyboard present). */
  const typedAt = useRef(0)
  /** The wearer moved focus themselves since the field got focus. */
  const navSinceFocus = useRef(false)

  const canSend = connected && !busy && draft.trim().length > 0

  const send = useCallback(() => {
    const text = useHudChat.getState().draft
    if (!chat.send(text)) return
    chat.setDraft('')
    const field = fieldRef.current
    if (field) focusEl(field)
  }, [])

  // Grow with the text up to three lines.
  useLayoutEffect(() => {
    const field = fieldRef.current
    if (!field) return
    field.style.height = 'auto'
    field.style.height = `${Math.min(field.scrollHeight + 2, MAX_FIELD_PX)}px`
  }, [draft])

  // The system composer committed text (`change` after `input`).
  useEffect(() => {
    const field = fieldRef.current
    if (!field) return
    const onFocus = () => { navSinceFocus.current = false }
    const onNavKey = (e: KeyboardEvent) => { if (NAV_KEYS.has(e.key)) navSinceFocus.current = true }
    const onCommit = () => {
      // A desktop blur-change after typing, or the wearer already moved on.
      if (navSinceFocus.current || Date.now() - typedAt.current < TYPING_WINDOW_MS) return
      // The pinch that closed the composer must not press what gets focus.
      quietActivations(COMPOSER_GUARD_MS)
      // After React re-rendered with the new text (Send enabled).
      requestAnimationFrame(() => {
        if (!field.isConnected) return
        const sendEl = wrapRef.current?.querySelector<HTMLElement>('.hud-composer-send')
        const target = sendEl && sendEl.getAttribute('aria-disabled') !== 'true' && field.value.trim() ? sendEl : field
        focusEl(target)
        // The guard counts from the move (the render can take a while on the
        // glasses' CPU), and the pin covers the same window.
        quietActivations(COMPOSER_GUARD_MS)
        pinFocus(target, COMPOSER_GUARD_MS)
      })
    }
    field.addEventListener('focus', onFocus)
    field.addEventListener('change', onCommit)
    document.addEventListener('keydown', onNavKey, true)
    return () => {
      field.removeEventListener('focus', onFocus)
      field.removeEventListener('change', onCommit)
      document.removeEventListener('keydown', onNavKey, true)
    }
  }, [])

  // A voice draft arrived: put focus on Send so one pinch sends what the
  // wearer is reading (after the screen's own initial focus has run).
  useEffect(() => {
    if (!staged) return
    let raf2 = 0
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => {
        const sendEl = wrapRef.current?.querySelector<HTMLElement>('.hud-composer-send')
        const target = sendEl && sendEl.getAttribute('aria-disabled') !== 'true' ? sendEl : fieldRef.current
        if (target) focusEl(target)
      })
    })
    return () => { cancelAnimationFrame(raf1); cancelAnimationFrame(raf2) }
  }, [staged])

  const onFieldKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key.length === 1) typedAt.current = Date.now()
    if (e.key !== 'Enter' || e.shiftKey || e.altKey || e.ctrlKey || e.metaKey || e.nativeEvent.isComposing) return
    if (Date.now() - typedAt.current > TYPING_WINDOW_MS) return // glasses pinch: let it open the composer
    e.preventDefault()
    send()
  }

  // Up out of the composer: let the screen jump to the start of a long new
  // reply instead of the nearest chip. Only where the focus engine would
  // leave the field (desktop typing keeps the caret moving inside it).
  const onWrapKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'ArrowUp' || e.defaultPrevented || !props.onArrowUp) return
    const t = e.target
    if (t instanceof HTMLTextAreaElement && !IS_GLASSES) {
      const atStart = t.selectionStart === 0 && t.selectionEnd === 0
      if (!atStart && t.value.includes('\n')) return
    }
    if (props.onArrowUp()) e.preventDefault()
  }

  const hint = !connected ? (closed ? 'Disconnected' : 'Connecting…')
    : busy ? 'Waiting for reply…'
      : staged ? 'Written by Meta AI — not sent. Pinch Send to send it.'
        : ''

  return (
    <div className="hud-composer-wrap" ref={wrapRef} onKeyDown={onWrapKeyDown}>
      <div className="hud-composer">
        <textarea
          ref={fieldRef}
          className="hud-composer-field"
          rows={1}
          maxLength={4000}
          placeholder="Pinch to write or speak…"
          aria-label="Message"
          data-autofocus=""
          value={draft}
          onChange={(e) => chat.setDraft(e.target.value)}
          onKeyDown={onFieldKeyDown}
        />
        <Btn variant="primary" className="hud-composer-send" disabled={!canSend} onActivate={send} title="Send">
          Send
        </Btn>
      </div>
      {hint ? <div className="hud-composer-hint" role="status">{hint}</div> : null}
    </div>
  )
}
