// Glasses sign-in (shown by HudApp instead of the app while auth is on and
// there is no session): a device-code pairing, RFC 8628 style.
//
//   1. POST /fd/auth/pair/start → a short user code (shown huge) and a secret
//      device code (never shown);
//   2. the wearer opens <host>/hud/pair on a phone / computer where they are
//      signed in, types the code and approves;
//   3. meanwhile we poll /fd/auth/pair/poll every `interval` s; "approved"
//      carries a normal session (access token + httpOnly refresh cookie), so
//      setAuth() is all it takes — HudApp then switches to the app.
//
// Nothing secret is typed on the glasses. Email + password stays available
// as a fallback (EmailLogin, local state — not a route, so Back still exits),
// on every view — a stalled request never leaves a screen without controls.
// Polling pauses while the page is hidden (display asleep) and resumes the
// moment it is visible again; network trouble backs off quietly.
// The pending code is kept in sessionStorage until it is used up, so a reload
// (universal menu → Restart, a sign-out hand-over) shows the SAME code the
// wearer may be typing on the phone instead of orphaning it, and does not
// spend another start (10 per 10 minutes per network).

import { useCallback, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { useAuthStore } from '../../stores/authStore'
import { HudApiError, pairPoll, pairStart } from '../api'
import { deviceLabel } from '../device'
import { clock } from '../format'
import { focusEl } from '../focus'
import { useAutoFocus, useNow } from '../hooks'
import { Btn, ScreenFrame, StateView } from '../ui'
import { EmailLogin } from './EmailLogin'
import './auth.css'

type View = 'starting' | 'active' | 'denied' | 'expired' | 'limited' | 'error'

interface PairState {
  view: View
  /** The code the wearer reads out ("BCDF-GHJK"). */
  code?: string
  /** Epoch ms when the code stops working. */
  expiresAt?: number
  /** Lifetime of the code in seconds (caps the countdown). */
  ttl?: number
  /** Where to approve ("/hud/pair"). */
  path?: string
  /** Network trouble: still trying, quietly. */
  reconnecting?: boolean
  /** view 'error': what went wrong. */
  error?: string
  /** view 'error': the deck no longer requires sign-in (pair/start → 400). */
  authOff?: boolean
  /** view 'limited': epoch ms of the refusal; message: the deck's own reason. */
  limitedAt?: number
  message?: string
}

const MAX_BACKOFF_MS = 30_000
/** pair/start allows this many codes per window per client IP (auth_routes PAIR_START_LIMIT). */
const START_LIMIT = 10
const START_WINDOW_MIN = 10

// ── The pending pairing, across reloads ──

const STORE_KEY = 'hud.pair.v1'
/** Resume a stored code only if it still has this long to live. */
const MIN_RESUME_MS = 10_000

interface StoredPairing {
  v: 1
  /** Secret poll credential — same tab only (sessionStorage), dropped once used. */
  deviceCode: string
  userCode: string
  /** Epoch ms when the code stops working. */
  expiresAt: number
  ttl: number
  intervalMs: number
  path: string
}

function loadPairing(): StoredPairing | null {
  try {
    const raw = sessionStorage.getItem(STORE_KEY)
    if (!raw) return null
    const p = JSON.parse(raw) as Partial<StoredPairing> | null
    if (
      p && p.v === 1 && typeof p.deviceCode === 'string' && p.deviceCode && typeof p.userCode === 'string' && p.userCode
      && typeof p.path === 'string' && Number.isFinite(p.expiresAt) && Number.isFinite(p.ttl) && Number.isFinite(p.intervalMs)
      && p.expiresAt! - Date.now() >= MIN_RESUME_MS && p.expiresAt! - Date.now() <= p.ttl! * 1_000
    ) return p as StoredPairing
    sessionStorage.removeItem(STORE_KEY)
  } catch { /* storage blocked / bad JSON: start a new code */ }
  return null
}

function savePairing(p: StoredPairing): void {
  try { sessionStorage.setItem(STORE_KEY, JSON.stringify(p)) } catch { /* storage blocked */ }
}

function clearPairing(): void {
  try { sessionStorage.removeItem(STORE_KEY) } catch { /* storage blocked */ }
}

function statusOf(e: unknown): number {
  return e instanceof HudApiError ? e.status : 0
}

/** No answer / gateway trouble: worth retrying without bothering the wearer. */
function isTransient(status: number): boolean {
  return status === 0 || status === 408 || status >= 500
}

function clamp(n: unknown, lo: number, hi: number, fallback: number): number {
  const v = Number(n)
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback
}

function mmss(totalSeconds: number): string {
  const s = Math.max(0, Math.ceil(totalSeconds))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

/**
 * The pairing state machine. `restart()` starts over with a fresh code.
 * One setTimeout chain drives both "get a code" retries and polling; a
 * second timer ends the code when it expires. A stored, still-valid code is
 * resumed instead of starting a new one.
 */
function usePairing(): { state: PairState; restart: () => void } {
  const [state, setState] = useState<PairState>({ view: 'starting' })
  const [gen, setGen] = useState(0)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let expiryTimer: ReturnType<typeof setTimeout> | undefined
    /** The step to run next; null once we reached an end state. */
    let next: (() => Promise<void>) | null = null
    /** The step came due while the page was hidden. */
    let parked = false
    let failures = 0
    let deviceCode = ''
    let expiresAt = 0
    let intervalMs = 3_000
    /** Polling a code restored from storage that no poll has confirmed yet. */
    let resumed = false

    const schedule = (step: () => Promise<void>, ms: number) => {
      clearTimeout(timer)
      next = step
      parked = false
      timer = setTimeout(() => {
        if (cancelled) return
        if (document.visibilityState === 'hidden') { parked = true; return }
        void step()
      }, ms)
    }
    const stop = () => { clearTimeout(timer); clearTimeout(expiryTimer); next = null; parked = false }
    /** Every end state uses the code up: a reload must not bring it back. */
    const end = (view: View, extra?: Partial<PairState>) => {
      stop()
      clearPairing()
      setState((s) => ({ ...s, ...extra, view, reconnecting: false }))
    }
    const backoff = () => Math.min(1_000 * 2 ** failures, MAX_BACKOFF_MS)

    /** Show a code and poll it until it is used up or expires. */
    const activate = (p: StoredPairing) => {
      deviceCode = p.deviceCode
      intervalMs = p.intervalMs
      expiresAt = p.expiresAt
      setState({ view: 'active', code: p.userCode, expiresAt, ttl: p.ttl, path: p.path })
      clearTimeout(expiryTimer)
      expiryTimer = setTimeout(() => { if (!cancelled) end('expired') }, Math.max(0, expiresAt - Date.now()))
    }

    const start = async () => {
      try {
        const res = await pairStart(deviceLabel())
        if (cancelled) return
        if (!res?.device_code || !res.user_code) throw new HudApiError('Unexpected answer from Flight Deck.', 422)
        failures = 0
        const ttl = clamp(res.expires_in, 1, 3_600, 600)
        const p: StoredPairing = {
          v: 1,
          deviceCode: res.device_code,
          userCode: res.user_code,
          expiresAt: Date.now() + ttl * 1_000,
          ttl,
          intervalMs: clamp(res.interval, 1, 30, 3) * 1_000,
          path: res.verification_path || '/hud/pair',
        }
        savePairing(p)
        activate(p)
        schedule(poll, intervalMs)
      } catch (e) {
        if (cancelled) return
        const status = statusOf(e)
        if (status === 429) {
          end('limited', { limitedAt: Date.now(), message: e instanceof Error ? e.message : '' })
          return
        }
        if (isTransient(status)) {
          failures++
          setState({ view: 'starting', reconnecting: true })
          schedule(start, backoff())
          return
        }
        stop()
        setState({
          view: 'error',
          authOff: status === 400,
          error: status === 400
            ? 'Sign-in is turned off on this deck.'
            : e instanceof Error && e.message ? e.message : 'Could not get a sign-in code.',
        })
      }
    }

    /** The deck does not know the code. A restored one: the deck restarted
     *  (or the code was used up elsewhere) — quietly get a new one. */
    const expire = () => {
      if (!resumed) { end('expired'); return }
      resumed = false
      stop()
      clearPairing()
      setState({ view: 'starting' })
      schedule(start, 0)
    }

    const poll = async () => {
      if (Date.now() >= expiresAt) { end('expired'); return }
      try {
        const res = await pairPoll(deviceCode)
        if (cancelled) return
        if (res.status === 'approved') {
          // Even if the code timed out meanwhile: the server already handed
          // us the session (and set the refresh cookie) — take it.
          stop()
          clearPairing()
          if (res.access_token && res.user) {
            // HudApp sees isAuthenticated and swaps this screen for the app.
            useAuthStore.getState().setAuth(res.user, res.access_token)
          } else {
            end('expired')
          }
          return
        }
        if (!next) return // ended while the request was in flight
        if (failures) { failures = 0; setState((s) => ({ ...s, reconnecting: false })) }
        if (res.status === 'denied') { end('denied'); return }
        if (res.status === 'expired') { expire(); return }
        resumed = false
        schedule(poll, intervalMs)
      } catch (e) {
        if (cancelled || !next) return
        const status = statusOf(e)
        if (status === 429) {
          // RFC 8628 "slow_down": poll less often and carry on.
          intervalMs = Math.min(intervalMs + 5_000, MAX_BACKOFF_MS)
          schedule(poll, intervalMs)
          return
        }
        if (isTransient(status)) {
          failures++
          setState((s) => ({ ...s, reconnecting: true }))
          schedule(poll, Math.max(intervalMs, backoff()))
          return
        }
        // Unknown / already used device code (e.g. the deck restarted).
        expire()
      }
    }

    // Display asleep → nothing runs; awake → catch up immediately.
    const onVisibility = () => {
      if (cancelled || document.visibilityState !== 'visible' || !parked || !next) return
      schedule(next, 0)
    }
    document.addEventListener('visibilitychange', onVisibility)
    const saved = loadPairing()
    if (saved) {
      // The code from before the reload: keep showing it, ask about it now.
      resumed = true
      activate(saved)
      schedule(poll, 0)
    } else {
      schedule(start, 0)
    }
    return () => {
      cancelled = true
      stop()
      document.removeEventListener('visibilitychange', onVisibility)
      // Signed in another way (email): this code is no longer needed.
      if (useAuthStore.getState().isAuthenticated) clearPairing()
    }
  }, [gen])

  const restart = useCallback(() => {
    clearPairing()
    setState({ view: 'starting' })
    setGen((g) => g + 1)
  }, [])

  return { state, restart }
}

/**
 * pair/start answered 429. Usually the per-network limit: a sliding window, so
 * a new code can take up to the whole window — say so, with the latest time.
 * Otherwise the deck's own reason (too many pairings in progress overall).
 */
function limitedText(p: PairState): string {
  if (p.message && /in progress/i.test(p.message)) return p.message
  const by = clock((p.limitedAt ?? Date.now()) + START_WINDOW_MIN * 60_000)
  return `Too many new codes from this network (${START_LIMIT} per ${START_WINDOW_MIN} minutes). `
    + `Try again in a few minutes — by ${by} at the latest.`
}

/** "Expires in 9:41" — ticks on its own so the rest of the screen stays still. */
function Countdown(props: { expiresAt: number; ttl: number }) {
  const now = useNow(1_000)
  const left = Math.min(props.ttl, Math.max(0, (props.expiresAt - now) / 1_000))
  return <>Expires in {mmss(left)}</>
}

/** Glasses sign-in (unauthenticated): shows a pairing code, email fallback. */
export function PairScreen() {
  const { state: pairing, restart } = usePairing()
  const [mode, setMode] = useState<'pair' | 'email'>('pair')
  const view = pairing.view === 'starting' && pairing.reconnecting ? 'retrying' : pairing.view

  const toEmail = useCallback(() => setMode('email'), [])
  const toPairing = useCallback(() => setMode('pair'), [])

  // First paint: focus its landing spot (while a code is on its way, that is
  // "Sign in with email instead" — never a screen without a focus stop).
  useAutoFocus(mode === 'pair')

  // Later view changes unmount the focused element: land on the new spot
  // ("New code" after an expiry, the code after a restart).
  useEffect(() => {
    if (mode !== 'pair') return
    let raf2 = 0
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => {
        const el = document.querySelector<HTMLElement>('.hud-screen [data-autofocus]')
        if (el && el !== document.activeElement) focusEl(el)
      })
    })
    return () => { cancelAnimationFrame(raf1); cancelAnimationFrame(raf2) }
  }, [mode, view])

  if (mode === 'email') return <EmailLogin onUsePairing={toPairing} />

  const emailBtn = (
    <div className="hud-actions">
      <Btn variant="ghost" onActivate={toEmail} fk="pair-email">Sign in with email instead</Btn>
    </div>
  )

  let body: ReactNode
  switch (view) {
    case 'starting':
      body = (
        <>
          <StateView kind="loading" message="Getting a sign-in code…" />
          {emailBtn}
        </>
      )
      break
    case 'retrying':
      // No answer (or a timeout): retrying with backoff; "Try now" skips the wait.
      body = (
        <>
          <StateView kind="loading" message="Can't reach Flight Deck — trying again…"
            action={{ label: 'Try now', onActivate: restart }} />
          {emailBtn}
        </>
      )
      break
    case 'active': {
      const url = `${window.location.host}${pairing.path || '/hud/pair'}`
      body = (
        <>
          <div className="hud-block hud-pair-card" tabIndex={0} data-autofocus="">
            <div className="hud-pair-lead">On your phone or computer, open</div>
            <div className="hud-pair-url">{url}</div>
            <div className="hud-pair-lead">and enter this code</div>
            <div className="hud-pair-code">{pairing.code}</div>
            <div className="hud-pair-expiry">
              <Countdown expiresAt={pairing.expiresAt ?? 0} ttl={pairing.ttl ?? 0} />
              {pairing.reconnecting ? <span className="hud-pair-reconnect"> · Reconnecting…</span> : null}
            </div>
          </div>
          <p className="hud-note">
            Nothing secret is typed on the glasses — you approve this code where you are already signed in.
          </p>
          {emailBtn}
        </>
      )
      break
    }
    case 'denied':
    case 'expired':
    case 'limited': {
      const text = view === 'denied'
        ? 'Sign-in was denied.'
        : view === 'expired' ? 'Code expired.' : limitedText(pairing)
      body = (
        <>
          <StateView kind={view === 'expired' ? 'empty' : 'error'} message={text}
            action={{ label: view === 'limited' ? 'Try again' : 'New code', onActivate: restart }} />
          {emailBtn}
        </>
      )
      break
    }
    case 'error':
      body = (
        <>
          <StateView kind="error" message={pairing.error}
            action={pairing.authOff
              ? { label: 'Reload', onActivate: () => window.location.reload() }
              : { label: 'Try again', onActivate: restart }} />
          {emailBtn}
        </>
      )
      break
  }

  return (
    <ScreenFrame title="Sign in" subtitle="Captain Claw" className="hud-pair-screen">
      {body}
    </ScreenFrame>
  )
}
