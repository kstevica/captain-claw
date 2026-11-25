import { useEffect, useState } from 'react'
import { getChromeActivity, type ChromeActivity } from '../../services/browserClip'

// Captain for Chrome's state for the sidebar (ChromeExtension.tsx): the deck
// is asked again every few seconds — quicker while something runs or the
// dialog is open, slower while it isn't connected, never while the tab is
// hidden, and at once when the person comes back to the tab (perhaps from
// setting it up in Chrome).

/** The deck's answer, kept fresh (`watching`: quicker, the dialog is open; `enabled`
 *  false: not asked at all). null until the first answer, or when the deck has no such route. */
export function useChromeActivity(watching: boolean, enabled = true): ChromeActivity | null {
  const [data, setData] = useState<ChromeActivity | null>(null)
  useEffect(() => {
    if (!enabled) return
    let live = true
    let busy = false   // this effect's own: a newer one never waits on an older one's request
    let timer: ReturnType<typeof setTimeout> | undefined
    const tick = async () => {
      clearTimeout(timer)
      if (!live || busy || document.hidden) return
      busy = true
      let next = 30000
      try {
        const d = await getChromeActivity()
        if (!live) return
        setData(d)
        next = watching || d.now.length ? 2500 : d.state === 'connected' ? 6000 : 15000
      } catch {
        // Signed out, or a deck without the route: try again now and then.
      } finally {
        busy = false
      }
      if (live) timer = setTimeout(tick, next)
    }
    tick()
    // Back to the tab (perhaps from setting it up in Chrome): ask at once.
    const again = () => { if (!document.hidden) tick() }
    document.addEventListener('visibilitychange', again)
    window.addEventListener('focus', again)
    return () => {
      live = false
      clearTimeout(timer)
      document.removeEventListener('visibilitychange', again)
      window.removeEventListener('focus', again)
    }
  }, [watching, enabled])
  return data
}
