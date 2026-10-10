// Email + password sign-in on the glasses — the fallback to pairing.
//
// Text entry on Meta Ray-Ban Display (fw v127+): focusing a standard text
// field and pinching it opens the system composer (handwriting, dictation,
// keyboard on v129+); committed text arrives as ONE `input` + `change`, no
// keydowns. `type=password` never opens the composer, so on glasses the
// password is a plain text field masked with CSS (-webkit-text-security).
// After a commit we mute activations briefly (the pinch that closed the
// composer arrives late as Enter / click on whatever is focused next) and
// walk focus to the next empty field, else to Sign in — pinned, because the
// glasses may reset focus to the first control when the composer closes.
// A `change` from a field that already lost focus to a tap / click (paste,
// autofill or a soft keyboard leave no printable keydown) is an ordinary
// blur commit: it fires between that press and its click, so muting there
// would swallow the tap on Sign in.

import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import { loginUser } from '../../stores/authStore'
import { IS_GLASSES } from '../device'
import { focusEl, pinFocus, quietActivations } from '../focus'
import { useAutoFocus } from '../hooks'
import { Btn, ScreenFrame } from '../ui'
import './auth.css'

/** A physical keystroke this recent means "someone is typing" (not the composer). */
const TYPING_WINDOW_MS = 10_000
/** A `change` this soon after a pointer press came from the blur it caused. */
const POINTER_BLUR_MS = 300
const NAV_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Tab'])
/** Symbols the glasses keyboard has no key for. */
const UNTYPABLE = `' " _ \\ < > [ ] { } | ~ ^ \``
const SUBMIT_FK = 'login-submit'

function loginErrorText(e: unknown): string {
  if (e instanceof TypeError) return "Can't reach Flight Deck — check the connection."
  const m = e instanceof Error ? e.message.trim() : ''
  if (!m || m.includes('[object')) return 'Sign-in failed.'
  if (/invalid credentials/i.test(m)) return 'Wrong email or password.'
  return m
}

/** Where focus belongs next: the first empty field, else Sign in. */
function nextStop(email: HTMLInputElement, password: HTMLInputElement): HTMLElement | null {
  if (!email.value.trim()) return email
  if (!password.value) return password
  return document.querySelector<HTMLElement>(`.hud-screen [data-fk="${SUBMIT_FK}"]`)
}

export function EmailLogin(props: {
  /** Back to the pairing code (button hidden when absent). */
  onUsePairing?: () => void
}) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const emailRef = useRef<HTMLInputElement>(null)
  const passwordRef = useRef<HTMLInputElement>(null)
  /** Last printable keydown in a field (a physical keyboard is present). */
  const typedAt = useRef(0)
  /** The wearer moved focus themselves since a field got focus. */
  const navSinceFocus = useRef(false)
  /** Last pointer press anywhere (a tap / click that may blur a field). */
  const pointerAt = useRef(0)

  useAutoFocus(true)

  // The system composer committed text (`change` after `input`).
  useEffect(() => {
    const emailEl = emailRef.current
    const passwordEl = passwordRef.current
    if (!emailEl || !passwordEl) return
    const onFocus = () => { navSinceFocus.current = false }
    const onNavKey = (e: KeyboardEvent) => { if (NAV_KEYS.has(e.key)) navSinceFocus.current = true }
    // Touch: the focus change (and the blur-change) comes with the compat
    // mousedown after the finger lifts, so note every phase of the press.
    const onPointer = () => { pointerAt.current = Date.now() }
    const onCommit = (e: Event) => {
      // A desktop blur-change after typing, or the wearer already moved on.
      if (navSinceFocus.current || Date.now() - typedAt.current < TYPING_WINDOW_MS) return
      // The field lost focus to a tap / click (or, off the glasses, to
      // anything: there is no composer there) — let that press through.
      const field = e.target as HTMLElement
      if (field !== document.activeElement
        && (!IS_GLASSES || Date.now() - pointerAt.current < POINTER_BLUR_MS)) return
      // The pinch that closed the composer must not press what gets focus.
      quietActivations(800)
      // After React re-rendered with the new value.
      requestAnimationFrame(() => {
        const target = nextStop(emailEl, passwordEl)
        if (!target || !target.isConnected) return
        focusEl(target)
        pinFocus(target, 800)
      })
    }
    for (const el of [emailEl, passwordEl]) {
      el.addEventListener('focus', onFocus)
      el.addEventListener('change', onCommit)
    }
    document.addEventListener('keydown', onNavKey, true)
    const pointerEvents = ['pointerdown', 'pointerup', 'mousedown'] as const
    for (const t of pointerEvents) document.addEventListener(t, onPointer, true)
    return () => {
      for (const el of [emailEl, passwordEl]) {
        el.removeEventListener('focus', onFocus)
        el.removeEventListener('change', onCommit)
      }
      document.removeEventListener('keydown', onNavKey, true)
      for (const t of pointerEvents) document.removeEventListener(t, onPointer, true)
    }
  }, [])

  const submit = async () => {
    if (busy) return
    const em = email.replace(/\s+/g, '')
    if (!em || !password) {
      setError('Enter your email and password.')
      const emailEl = emailRef.current
      const passwordEl = passwordRef.current
      if (emailEl && passwordEl) {
        const target = nextStop(emailEl, passwordEl)
        if (target) focusEl(target)
      }
      return
    }
    setBusy(true)
    setError(null)
    try {
      // Success sets the session; HudApp then swaps this screen for the app.
      await loginUser(em, password)
    } catch (e) {
      setBusy(false)
      setError(loginErrorText(e))
    }
  }

  const onFieldKeyDown = (e: ReactKeyboardEvent<HTMLInputElement>) => {
    if (e.key.length === 1) typedAt.current = Date.now()
    if (e.key !== 'Enter' || e.shiftKey || e.altKey || e.ctrlKey || e.metaKey || e.nativeEvent.isComposing) return
    // On glasses Enter on a field is the pinch that opens the composer.
    if (IS_GLASSES && Date.now() - typedAt.current > TYPING_WINDOW_MS) return
    e.preventDefault()
    const passwordEl = passwordRef.current
    if (e.currentTarget === emailRef.current && passwordEl && !passwordEl.value) focusEl(passwordEl)
    else void submit()
  }

  return (
    <ScreenFrame title="Sign in with email" subtitle="Captain Claw" className="hud-login-screen">
      <label className="hud-field">
        <span className="hud-field-label">Email</span>
        <input
          ref={emailRef}
          className="hud-input"
          type="email"
          name="email"
          autoComplete="username"
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
          placeholder={IS_GLASSES ? 'Pinch to write or speak' : 'you@example.com'}
          value={email}
          readOnly={busy}
          onChange={(e) => { setEmail(e.target.value); setError(null) }}
          onKeyDown={onFieldKeyDown}
        />
      </label>
      <label className="hud-field">
        <span className="hud-field-label">Password</span>
        {IS_GLASSES ? (
          // type=password never opens the glasses composer: a masked text field.
          <input
            ref={passwordRef}
            className={password ? 'hud-input hud-secret' : 'hud-input'}
            type="text"
            name="hud-secret"
            autoComplete="off"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            placeholder="Pinch to enter"
            value={password}
            readOnly={busy}
            onChange={(e) => { setPassword(e.target.value); setError(null) }}
            onKeyDown={onFieldKeyDown}
          />
        ) : (
          <input
            ref={passwordRef}
            className="hud-input"
            type="password"
            name="password"
            autoComplete="current-password"
            value={password}
            readOnly={busy}
            onChange={(e) => { setPassword(e.target.value); setError(null) }}
            onKeyDown={onFieldKeyDown}
          />
        )}
      </label>
      {error ? <div className="hud-login-error" role="alert">{error}</div> : null}
      <div className="hud-actions">
        <Btn variant="primary" fk={SUBMIT_FK} onActivate={() => { void submit() }} disabled={busy}>
          {busy ? 'Signing in…' : 'Sign in'}
        </Btn>
        {props.onUsePairing ? (
          <Btn variant="ghost" fk="login-pair" onActivate={props.onUsePairing}>Use a pairing code instead</Btn>
        ) : null}
      </div>
      <p className="hud-note">
        The glasses keyboard can't type <span className="hud-login-symbols">{UNTYPABLE}</span>, and
        dictating says your password out loud. A pairing code is safer.
      </p>
    </ScreenFrame>
  )
}
