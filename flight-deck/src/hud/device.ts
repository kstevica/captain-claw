// Which host (if any) is running the HUD.
//
// The /hud route is the source of truth for "glasses mode" — this is only used
// to tune input quirks (e.g. the Display never delivers Escape, and its system
// Back already calls history.back()) and to label the device when pairing.
//
// Meta Ray-Ban Display: Android WebView whose UA carries the device codename
// "Greatwhite" plus the "; wv)" WebView token — the same check Meta's own UI
// toolkit uses (PageTransition.tsx). Rokid Lumen (community host that runs
// Display web apps unchanged) injects `window.lumen` on the app's origin.
//
// The Captain Android app (kept out of this repo until it is published): a
// phone WebView shell that appends ` CaptainClawHUD/<version>` to its user
// agent; its first versions loaded a deck's /hud/. It is a touch phone, never
// glasses.

export type GlassesHost = 'meta-display' | 'rokid-lumen' | null
export type HudHost = 'meta-display' | 'rokid-lumen' | 'android-app' | null

function detect(): HudHost {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent || '' : ''
  if (/\bGreatwhite\b/.test(ua) && /; wv\)/.test(ua)) return 'meta-display'
  const w = typeof window !== 'undefined' ? (window as unknown as { lumen?: unknown }) : {}
  if (w.lumen && typeof w.lumen === 'object') return 'rokid-lumen'
  if (/\bCaptainClawHUD\/\S+/.test(ua)) return 'android-app'
  return null
}

export const HUD_HOST: HudHost = detect()

export const GLASSES_HOST: GlassesHost = HUD_HOST === 'android-app' ? null : HUD_HOST

/** True on a known glasses host (no physical keyboard, no pointer). */
export const IS_GLASSES = GLASSES_HOST !== null

/** The Captain Claw Android app. */
export const IS_APP = HUD_HOST === 'android-app'

function touchFirst(): boolean {
  if (IS_GLASSES) return false
  if (IS_APP) return true
  try {
    // A phone / tablet browser: the main pointer is a finger and nothing hovers.
    return window.matchMedia('(pointer: coarse)').matches && !window.matchMedia('(any-hover: hover)').matches
  } catch {
    return false
  }
}

/** A touch phone or tablet (the app, or a mobile browser) — never glasses. */
export const IS_TOUCH = touchFirst()

/** The word for pressing a control: a pinch on glasses, a tap on a phone. */
export const PRESS = IS_GLASSES ? 'Pinch' : IS_TOUCH ? 'Tap' : 'Click'

/** Short human label for the pairing request ("which device is asking?"). */
export function deviceLabel(): string {
  if (GLASSES_HOST === 'meta-display') return 'Meta Ray-Ban Display'
  if (GLASSES_HOST === 'rokid-lumen') return 'Rokid glasses (Lumen)'
  if (IS_APP) return 'Captain app (Android)'
  const ua = navigator.userAgent || ''
  if (/Android/.test(ua)) return 'Android browser'
  if (/iPhone|iPad/.test(ua)) return 'iOS browser'
  if (/Mac OS X/.test(ua)) return 'Mac browser'
  if (/Windows/.test(ua)) return 'Windows browser'
  return 'Browser'
}

/** The Android app's native side (its MainActivity), injected for the deck's
 *  origin only. Absent everywhere else. */
interface AppBridge { postMessage(message: string): void }

function appBridge(): AppBridge | null {
  if (!IS_APP) return null
  const b = (window as unknown as { CaptainClawHUD?: AppBridge }).CaptainClawHUD
  return b && typeof b.postMessage === 'function' ? b : null
}

/** The app can open its own settings (change deck, reload). */
export function hasAppSettings(): boolean {
  return appBridge() !== null
}

/** Open the app's settings screen (no-op outside the app). */
export function openAppSettings(): void {
  try { appBridge()?.postMessage(JSON.stringify({ t: 'settings' })) } catch { /* bridge gone */ }
}
