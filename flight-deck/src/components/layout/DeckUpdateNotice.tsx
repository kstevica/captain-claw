import { useCallback, useEffect, useRef, useState } from 'react'
import { CheckCircle2, Info, Loader2, RefreshCw, X } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'

// What everyone signed in sees of a deck update (deck_update.py), on every
// layout, at the top right: a toast when one is coming, one they can't close
// counting down to the restart, and, when the deck is back, a way to the new
// version's page. It only ever follows the deck: it asks every 15 seconds, and
// every 3 while something is going on.

interface DeckNotice {
  level: 'none' | 'soon' | 'restart'
  id?: number
  seconds_left?: number
  last?: { id: number; phase: string } | null
}

const POLL_QUIET_MS = 15000
const POLL_ACTIVE_MS = 3000

async function fetchNotice(): Promise<DeckNotice> {
  const ask = () => {
    const { token, authEnabled } = useAuthStore.getState()
    return fetch('/fd/system/update/notice', {
      headers: authEnabled && token ? { Authorization: `Bearer ${token}` } : {},
      credentials: 'include',
    })
  }
  let res = await ask()
  // A deck restart can end the access token (no FD_JWT_SECRET).
  if (res.status === 401 && useAuthStore.getState().authEnabled && await refreshAccessToken()) res = await ask()
  if (!res.ok) throw new Error(`${res.status}`)
  return res.json()
}

const clock = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`

const CARD = 'rounded-xl border bg-zinc-900 px-4 py-3 text-sm shadow-xl shadow-black/30'

export function DeckUpdateNotice() {
  const name = useAuthStore((s) => s.branding.name) || 'Captain'
  const [notice, setNotice] = useState<DeckNotice>({ level: 'none' })
  // No answer while it restarts.
  const [away, setAway] = useState(false)
  const [secondsLeft, setSecondsLeft] = useState(0)
  const [closed, setClosed] = useState<number | null>(null)
  const [back, setBack] = useState<string | null>(null)
  const deadline = useRef(0)
  // The update whose restart this page saw coming: it's the one told how it ended.
  const following = useRef<number | null>(null)

  const poll = useCallback(async () => {
    try {
      const next = await fetchNotice()
      setAway(false)
      if (next.level === 'restart') {
        following.current = next.id ?? null
        deadline.current = Date.now() + (next.seconds_left ?? 0) * 1000
        setSecondsLeft(next.seconds_left ?? 0)
        setBack(null)
      } else if (next.level === 'none' && following.current !== null && next.last?.id === following.current) {
        following.current = null
        setBack(next.last.phase)
      }
      setNotice(next)
    } catch {
      if (following.current !== null) setAway(true)
    }
  }, [])

  const busy = away || notice.level !== 'none'
  useEffect(() => {
    let timer = 0
    let stopped = false
    const run = async () => {
      await poll()
      if (!stopped) timer = window.setTimeout(run, busy ? POLL_ACTIVE_MS : POLL_QUIET_MS)
    }
    timer = window.setTimeout(run, busy ? POLL_ACTIVE_MS : 0)
    return () => { stopped = true; window.clearTimeout(timer) }
  }, [busy, poll])

  const counting = notice.level === 'restart'
  useEffect(() => {
    if (!counting) return
    const timer = window.setInterval(() => {
      setSecondsLeft(Math.max(0, Math.ceil((deadline.current - Date.now()) / 1000)))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [counting])

  const restarting = away || counting
  const soon = notice.level === 'soon' && closed !== notice.id && !away
  if (!restarting && !soon && !back) return null

  return (
    <div className="pointer-events-none fixed right-4 top-4 z-[10000] flex w-[min(22rem,calc(100vw-2rem))] flex-col gap-2">
      {restarting && (
        <div role="alert" data-testid="deck-update-countdown" className={`${CARD} border-amber-500/40`}>
          {!away && secondsLeft > 0 ? (
            <>
              <p className="flex items-center gap-2 font-semibold text-zinc-100">
                <RefreshCw className="h-4 w-4 shrink-0 text-amber-500" />
                <span>{name} restarts in <span className="tabular-nums">{clock(secondsLeft)}</span></span>
              </p>
              <p className="mt-1 text-xs leading-relaxed text-zinc-400">It's being updated. Finish what you're writing: agents stop when this reaches zero.</p>
            </>
          ) : (
            <>
              <p className="flex items-center gap-2 font-semibold text-zinc-100">
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-amber-500" />
                <span>{name} is restarting</span>
              </p>
              <p className="mt-1 text-xs leading-relaxed text-zinc-400">This goes away by itself when it's back.</p>
            </>
          )}
        </div>
      )}

      {soon && (
        <div role="status" data-testid="deck-update-soon" className={`${CARD} pointer-events-auto flex items-start gap-2 border-zinc-700`}>
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-sky-500" />
          <p className="min-w-0 flex-1 leading-relaxed text-zinc-200">{name} will be updated soon. You'll see a countdown before it restarts.</p>
          <button type="button" aria-label="Close" onClick={() => setClosed(notice.id ?? null)} className="shrink-0 rounded p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {back && !restarting && (
        <div role="status" data-testid="deck-update-back" className={`${CARD} pointer-events-auto flex items-start gap-2 border-zinc-700`}>
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />
          <div className="min-w-0 flex-1 leading-relaxed text-zinc-200">
            {back === 'done' ? (
              <>
                <p>{name} is back on the new version.</p>
                <button type="button" onClick={() => window.location.reload()} className="mt-1 text-xs font-medium text-violet-600 hover:underline dark:text-violet-400">
                  Reload this page to get it
                </button>
              </>
            ) : (
              <p>{name} is back. The update didn't go through, so nothing changed for you.</p>
            )}
          </div>
          <button type="button" aria-label="Close" onClick={() => setBack(null)} className="shrink-0 rounded p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  )
}
