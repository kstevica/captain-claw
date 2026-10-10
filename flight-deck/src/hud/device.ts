// Which glasses host (if any) is running the HUD.
//
// The /hud route is the source of truth for "glasses mode" — this is only used
// to tune input quirks (e.g. the Display never delivers Escape, and its system
// Back already calls history.back()) and to label the device when pairing.
//
// Meta Ray-Ban Display: Android WebView whose UA carries the device codename
// "Greatwhite" plus the "; wv)" WebView token — the same check Meta's own UI
// toolkit uses (PageTransition.tsx). Rokid Lumen (community host that runs
// Display web apps unchanged) injects `window.lumen` on the app's origin.

export type GlassesHost = 'meta-display' | 'rokid-lumen' | null

function detect(): GlassesHost {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent || '' : ''
  if (/\bGreatwhite\b/.test(ua) && /; wv\)/.test(ua)) return 'meta-display'
  const w = typeof window !== 'undefined' ? (window as unknown as { lumen?: unknown }) : {}
  if (w.lumen && typeof w.lumen === 'object') return 'rokid-lumen'
  return null
}

export const GLASSES_HOST: GlassesHost = detect()

/** True on a known glasses host (no physical keyboard, no pointer). */
export const IS_GLASSES = GLASSES_HOST !== null

/** Short human label for the pairing request ("which device is asking?"). */
export function deviceLabel(): string {
  if (GLASSES_HOST === 'meta-display') return 'Meta Ray-Ban Display'
  if (GLASSES_HOST === 'rokid-lumen') return 'Rokid glasses (Lumen)'
  const ua = navigator.userAgent || ''
  if (/Android/.test(ua)) return 'Android browser'
  if (/iPhone|iPad/.test(ua)) return 'iOS browser'
  if (/Mac OS X/.test(ua)) return 'Mac browser'
  if (/Windows/.test(ua)) return 'Windows browser'
  return 'Browser'
}
