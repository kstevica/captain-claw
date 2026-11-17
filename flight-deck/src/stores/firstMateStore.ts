import { useEffect } from 'react'
import { create } from 'zustand'
import { useAuthStore } from './authStore'
import { getStatus, type Status } from '../services/firstMate'

/** First Mate in the nav and the kiosk: the deck's answer to "may I use it",
 *  read once a session (and again after the page changes the deck's switch),
 *  and whether the kiosk's dialog is open. */
interface FirstMateStore {
  status: Status | null
  open: boolean
  openFirstMate: () => void
  close: () => void
  load: () => Promise<void>
}

let inflight: Promise<void> | null = null

export const useFirstMateStore = create<FirstMateStore>((set) => ({
  status: null,
  open: false,
  openFirstMate: () => set({ open: true }),
  close: () => set({ open: false }),
  load: () => {
    if (!inflight) {
      inflight = getStatus()
        .then((status) => set({ status }))
        .catch(() => set({ status: null }))
        .finally(() => { inflight = null })
    }
    return inflight
  },
}))

/** Shown to whoever may use First Mate, and to admins (a deck with sign-in off
 *  has one user, its admin) so they can turn it on. */
export function useFirstMateVisible(): boolean {
  const signedIn = useAuthStore((s) => s.isAuthenticated || s.authEnabled === false)
  const admin = useAuthStore((s) => s.authEnabled === false || s.user?.role === 'admin')
  const status = useFirstMateStore((s) => s.status)
  const load = useFirstMateStore((s) => s.load)
  useEffect(() => { if (signedIn && !status) void load() }, [signedIn, status, load])
  return signedIn && (admin || !!status?.available)
}
