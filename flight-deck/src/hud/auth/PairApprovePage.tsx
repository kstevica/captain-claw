// /hud/pair — approve a glasses sign-in from a phone or desktop.
//
// A normal page (not a .hud-screen; the D-pad focus engine is not installed
// here): real <form>, <input> and <button> elements, one centered column,
// 48 px touch targets. Flow:
//   auth status → (auth off: nothing to pair) → refresh the session →
//   sign in if needed → type the code shown in the glasses → see WHICH
//   device is asking (IP, age, and what it says about itself) →
//   Approve / Deny → done.
// Device-code phishing is the main risk of this flow ("open this link /
// enter this code"), hence: the code is never taken from the URL (nothing
// links here with one — the glasses show the bare address — so a prefilled
// link could only come from someone else), the card separates the one fact
// the deck observed (the IP) from the label and browser any caller can make
// up, and the explicit "only approve a code showing in YOUR glasses" warning.

import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { awaitAuthStatus, loginUser, logoutUser, refreshAccessToken, useAuthStore } from '../../stores/authStore'
import { HudApiError, pairApprove, pairLookup, type PairInfo } from '../api'
import { relTime, truncate } from '../format'
import { useNow } from '../hooks'
import './auth.css'

/** User codes: two groups of four consonants (no vowels, no Y). */
const CODE_RE = /^[BCDFGHJKLMNPQRSTVWXZ]{8}$/

function codeLetters(raw: string): string {
  return raw.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 8)
}

/** "bcdf ghjk" / "BCDFGHJK" / "bcdf-ghjk" → "BCDF-GHJK" (as typed: "BCD", "BCDF-G"). */
function formatCode(raw: string): string {
  const l = codeLetters(raw)
  return l.length > 4 ? `${l.slice(0, 4)}-${l.slice(4)}` : l
}

function mmss(totalSeconds: number): string {
  const s = Math.max(0, Math.ceil(totalSeconds))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

function loginErrorText(e: unknown): string {
  if (e instanceof TypeError) return "Can't reach Flight Deck — check your connection."
  const m = e instanceof Error ? e.message.trim() : ''
  if (!m || m.includes('[object')) return 'Sign-in failed.'
  if (/invalid credentials/i.test(m)) return 'Wrong email or password.'
  return m
}

function pairErrorText(e: unknown): string {
  const status = e instanceof HudApiError ? e.status : 0
  if (status === 404) return 'Code not found or expired — check the glasses for a fresh code.'
  if (status === 429) return 'Too many attempts — wait a minute.'
  if (status === 401) return 'Your session ended — sign in again.'
  if (status === 0) return "Can't reach Flight Deck — check your connection."
  return e instanceof Error && e.message ? e.message : 'Something went wrong.'
}

/** "Android 14 · Chrome 146 WebView" — enough to recognise a device (the label names it). */
function shortUA(ua: string): string {
  if (!ua) return 'Unknown'
  const pick = (re: RegExp) => ua.match(re)?.[1] ?? null
  const parts: string[] = []
  const android = pick(/Android (\d+(?:\.\d+)?)/)
  const ios = pick(/(?:iPhone|iPad|CPU) OS (\d+)[._]/)
  if (android) parts.push(`Android ${android}`)
  else if (ios) parts.push(`iOS ${ios}`)
  else if (/Windows NT/.test(ua)) parts.push('Windows')
  else if (/CrOS/.test(ua)) parts.push('ChromeOS')
  else if (/Mac OS X/.test(ua)) parts.push('macOS')
  else if (/Linux/.test(ua)) parts.push('Linux')
  let browser = ''
  const edge = pick(/Edg[A-Za-z]*\/(\d+)/)
  const firefox = pick(/(?:Firefox|FxiOS)\/(\d+)/)
  const chrome = pick(/(?:Chrome|CriOS)\/(\d+)/)
  const safari = pick(/Version\/(\d+)[\d.]*.*Safari\//)
  if (edge) browser = `Edge ${edge}`
  else if (firefox) browser = `Firefox ${firefox}`
  else if (chrome) browser = `Chrome ${chrome}`
  else if (safari) browser = `Safari ${safari}`
  if (browser) parts.push(/; wv\)/.test(ua) ? `${browser} WebView` : browser)
  return parts.length ? parts.join(' · ') : truncate(ua, 80)
}

/** /hud/pair on a phone or desktop: sign in, enter the glasses' code, approve. */
export function PairApprovePage() {
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  const email = useAuthStore((s) => s.user?.email || s.user?.display_name || '')
  const [booted, setBooted] = useState(false)
  const [unreachable, setUnreachable] = useState(false)

  useEffect(() => {
    const prev = document.title
    document.title = 'Pair glasses · Captain Claw'
    return () => { document.title = prev }
  }, [])

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

  let content
  if (!booted) {
    content = (
      <div className="hud-pp-card" role="status">
        <p className="hud-pp-muted">{unreachable ? "Can't reach Flight Deck — retrying…" : 'Loading…'}</p>
      </div>
    )
  } else if (authEnabled === false) {
    content = (
      <div className="hud-pp-card">
        <h2 className="hud-pp-h2">No pairing needed</h2>
        <p className="hud-pp-text">
          This deck has sign-in turned off, so glasses need no pairing — open{' '}
          <strong className="hud-pp-url">{window.location.host}/hud/</strong> on them directly.
        </p>
        <p className="hud-pp-warn">
          With sign-in off, anyone who can reach this deck's address can use it. Turn sign-in on before
          exposing it to the internet.
        </p>
      </div>
    )
  } else if (!isAuthenticated) {
    content = <LoginCard />
  } else {
    content = <ApproveFlow email={email} />
  }

  return (
    <div className="hud-page hud-pair-page">
      <div className="hud-pp-wrap">
        <header className="hud-pp-head">
          <div className="hud-pp-brand">Captain Claw</div>
          <h1 className="hud-pp-title">Pair glasses</h1>
        </header>
        {content}
      </div>
    </div>
  )
}

function LoginCard() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (busy) return
    const em = email.trim()
    if (!em || !password) { setError('Enter your email and password.'); return }
    setBusy(true)
    setError(null)
    try {
      // Success flips isAuthenticated; the page moves on to the code step.
      await loginUser(em, password)
    } catch (err) {
      setBusy(false)
      setError(loginErrorText(err))
    }
  }

  return (
    <form className="hud-pp-card" onSubmit={onSubmit} noValidate>
      <h2 className="hud-pp-h2">Sign in to approve</h2>
      <p className="hud-pp-muted">Use the Flight Deck account the glasses should sign in to.</p>
      <label className="hud-pp-field">
        <span className="hud-pp-label">Email</span>
        <input className="hud-pp-input" type="email" name="email" autoComplete="username" autoCapitalize="off"
          autoCorrect="off" spellCheck={false} inputMode="email" value={email} readOnly={busy}
          onChange={(e) => { setEmail(e.target.value); setError(null) }} />
      </label>
      <label className="hud-pp-field">
        <span className="hud-pp-label">Password</span>
        <input className="hud-pp-input" type="password" name="password" autoComplete="current-password"
          value={password} readOnly={busy} onChange={(e) => { setPassword(e.target.value); setError(null) }} />
      </label>
      {error ? <p className="hud-pp-error" role="alert">{error}</p> : null}
      <button className="hud-pp-btn hud-pp-btn--primary" type="submit" disabled={busy}>
        {busy ? 'Signing in…' : 'Sign in'}
      </button>
    </form>
  )
}

type Step =
  | { k: 'enter' }
  | { k: 'review'; code: string; info: PairInfo; expiresAt: number }
  | { k: 'done'; approved: boolean }

function ApproveFlow(props: { email: string }) {
  // Typed by the wearer from the glasses — never prefilled (see the top).
  const [code, setCode] = useState('')
  const [step, setStep] = useState<Step>({ k: 'enter' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [signOutFailed, setSignOutFailed] = useState(false)
  const signingOut = useAuthStore((s) => s.signingOut)

  const lookup = async (e: FormEvent) => {
    e.preventDefault()
    if (busy) return
    const letters = codeLetters(code)
    if (letters.length !== 8) { setError('Enter the 8-letter code shown in the glasses.'); return }
    if (!CODE_RE.test(letters)) { setError('Check the code — it only uses consonants (no A, E, I, O, U or Y).'); return }
    const formatted = formatCode(letters)
    setBusy(true)
    setError(null)
    try {
      const info = await pairLookup(formatted)
      const ttl = Number(info.expires_in)
      setStep({ k: 'review', code: formatted, info, expiresAt: Date.now() + (Number.isFinite(ttl) ? ttl : 0) * 1_000 })
    } catch (err) {
      setError(pairErrorText(err))
    } finally {
      setBusy(false)
    }
  }

  const decide = async (approve: boolean) => {
    if (busy || step.k !== 'review') return
    setBusy(true)
    setError(null)
    try {
      const res = await pairApprove(step.code, approve)
      setStep({ k: 'done', approved: res.status ? res.status === 'approved' : approve })
    } catch (err) {
      setError(pairErrorText(err))
    } finally {
      setBusy(false)
    }
  }

  const reset = () => {
    setStep({ k: 'enter' })
    setCode('')
    setError(null)
  }

  const signOut = () => {
    setSignOutFailed(false)
    // Success reloads the page (back to the sign-in form).
    void logoutUser().then((ok) => { if (!ok) setSignOutFailed(true) })
  }

  let card
  if (step.k === 'enter') {
    card = (
      <form className="hud-pp-card" onSubmit={lookup} noValidate>
        <h2 className="hud-pp-h2">Enter the code from your glasses</h2>
        <label className="hud-pp-field">
          <span className="hud-pp-label">Code</span>
          <input className="hud-pp-input hud-pp-code-input" type="text" name="code" inputMode="text"
            autoComplete="off" autoCapitalize="characters" autoCorrect="off" spellCheck={false}
            enterKeyHint="go" placeholder="XXXX-XXXX" value={code} readOnly={busy}
            aria-invalid={error ? true : undefined}
            onChange={(e) => { setCode(formatCode(e.target.value)); setError(null) }} />
        </label>
        {error ? <p className="hud-pp-error" role="alert">{error}</p> : null}
        <button className="hud-pp-btn hud-pp-btn--primary" type="submit" disabled={busy}>
          {busy ? 'Checking…' : 'Continue'}
        </button>
      </form>
    )
  } else if (step.k === 'review') {
    card = (
      <ReviewCard code={step.code} info={step.info} expiresAt={step.expiresAt} email={props.email}
        busy={busy} error={error} onDecide={(approve) => { void decide(approve) }} onCancel={reset} />
    )
  } else {
    card = (
      <div className="hud-pp-card" role="status">
        {step.approved ? (
          <>
            <p className="hud-pp-ok">Glasses signed in — they continue automatically.</p>
            <p className="hud-pp-muted">
              They stay signed in as long as they are used at least once every 7 days. Only Sign out on the
              glasses (in their agent list) ends it sooner — this page can't sign them out.
            </p>
          </>
        ) : (
          <>
            <p className="hud-pp-ok hud-pp-ok--denied">Request denied.</p>
            <p className="hud-pp-muted">The glasses will say that sign-in was denied.</p>
          </>
        )}
        <button className="hud-pp-btn" type="button" onClick={reset}>Pair another device</button>
      </div>
    )
  }

  return (
    <>
      <div className="hud-pp-user">
        <span className="hud-pp-user-name">Signed in as <strong>{props.email || 'your account'}</strong></span>
        <button className="hud-pp-link" type="button" onClick={signOut} disabled={signingOut}>
          {signingOut ? 'Signing out…' : 'Sign out'}
        </button>
      </div>
      {signOutFailed ? <p className="hud-pp-error" role="alert">Sign-out failed — you are still signed in. Try again.</p> : null}
      {card}
    </>
  )
}

/** Which device is asking, plus Approve / Deny. Ticks its own countdown. */
function ReviewCard(props: {
  code: string
  info: PairInfo
  expiresAt: number
  email: string
  busy: boolean
  error: string | null
  onDecide: (approve: boolean) => void
  onCancel: () => void
}) {
  const now = useNow(1_000)
  const left = (props.expiresAt - now) / 1_000
  const expired = left <= 0
  const createdMs = Date.parse(props.info.created_at)
  const requested = Number.isFinite(createdMs) ? relTime(createdMs, now) : ''

  return (
    <div className="hud-pp-card">
      <h2 className="hud-pp-h2">Sign in this device?</h2>
      <div className="hud-pp-code-show">{props.code}</div>
      <dl className="hud-pp-facts">
        <dt>IP address</dt>
        <dd>{props.info.ip || 'unknown'}</dd>
        <dt>Requested</dt>
        <dd>{requested ? (requested === 'now' ? 'just now' : requested) : '—'}</dd>
        <dt>Expires</dt>
        <dd className={expired ? 'hud-pp-expired' : undefined}>{expired ? 'expired' : `in ${mmss(left)}`}</dd>
      </dl>
      {/* Sent by whoever started the pairing — anyone can claim to be glasses. */}
      <div className="hud-pp-claimed">
        <div className="hud-pp-claimed-h">Reported by the device (not verified)</div>
        <dl className="hud-pp-facts">
          <dt>Name</dt>
          <dd>{truncate(props.info.label || 'none given', 80)}</dd>
          <dt>Browser</dt>
          <dd title={props.info.user_agent}>{shortUA(props.info.user_agent || '')}</dd>
        </dl>
      </div>
      <p className="hud-pp-warn">
        Only approve a code that is showing in <strong>YOUR</strong> glasses right now. Approving signs that
        device in as <strong>{props.email || 'you'}</strong> until it signs out or goes 7 days unused.
      </p>
      {expired ? (
        <p className="hud-pp-error" role="alert">This code has expired — check the glasses for a fresh code.</p>
      ) : null}
      {props.error ? <p className="hud-pp-error" role="alert">{props.error}</p> : null}
      <div className="hud-pp-actions">
        <button className="hud-pp-btn hud-pp-btn--primary" type="button" disabled={props.busy || expired}
          onClick={() => props.onDecide(true)}>
          Approve
        </button>
        <button className="hud-pp-btn hud-pp-btn--danger" type="button" disabled={props.busy || expired}
          onClick={() => props.onDecide(false)}>
          Deny
        </button>
      </div>
      <button className="hud-pp-link" type="button" onClick={props.onCancel}>Enter a different code</button>
    </div>
  )
}
