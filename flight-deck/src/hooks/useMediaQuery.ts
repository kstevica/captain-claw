import { useSyncExternalStore } from 'react'

const MOBILE = '(max-width: 767px)'
const TABLET = '(min-width: 768px) and (max-width: 1023px)'
const DESKTOP = '(min-width: 1024px)'

function subscribe(query: string) {
  return (cb: () => void) => {
    const mql = window.matchMedia(query)
    mql.addEventListener('change', cb)
    return () => mql.removeEventListener('change', cb)
  }
}

function snapshot(query: string) {
  return () => window.matchMedia(query).matches
}

const server = () => false

export function useIsMobile() {
  const isMobile = useSyncExternalStore(subscribe(MOBILE), snapshot(MOBILE), server)
  const isTablet = useSyncExternalStore(subscribe(TABLET), snapshot(TABLET), server)
  const isDesktop = useSyncExternalStore(subscribe(DESKTOP), snapshot(DESKTOP), server)
  return { isMobile, isTablet, isDesktop }
}
