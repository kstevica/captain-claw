import { useCallback, useEffect, useState } from 'react'
import { Layers, Loader2 } from 'lucide-react'
import { getMyPacks, deletePack, type MyPackRow } from '../../services/contextPacks'
import {
  INACTIVE_HINT,
  MY_PACKS_EMPTY,
  MY_PACKS_HEADING,
  MY_PACKS_ID,
  PACKS_PAUSED_NOTE,
  packSummary,
} from '../../utils/contextPacks'

// Profile page: everything I share with agents' people, on every agent — so I
// can stop sharing in one place, including on an agent I no longer reach
// (shown as inactive).
//
// `sharingOff` — agent sharing is off on this deck: Flight Deck keeps my packs
// (none is in effect) and still lets me stop them, so they're listed as paused
// rather than hidden. `showEmpty` — show the card (with its "nothing yet"
// text) even when I share nothing; off, the card shows only when Flight Deck
// returns packs (an older deck without the route shows nothing). `onPacks`
// hears every list loaded (the page's About me note).

export function MyContextPacks({ sharingOff = false, showEmpty = true, onPacks }: {
  sharingOff?: boolean
  showEmpty?: boolean
  onPacks?: (packs: MyPackRow[]) => void
} = {}) {
  const [packs, setPacks] = useState<MyPackRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busyId, setBusyId] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const r = await getMyPacks()
      const list = Array.isArray(r?.packs) ? r.packs : []
      setPacks(list)
      onPacks?.(list)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }, [onPacks])

  useEffect(() => { void load() }, [load])

  const stop = async (id: string) => {
    if (busyId) return
    setBusyId(id)
    let failure = ''
    try {
      await deletePack(id)
    } catch (e) {
      failure = e instanceof Error ? e.message : 'Failed to stop sharing'
    }
    await load()
    if (failure) setError(failure)
    setBusyId(null)
  }

  // Nothing to show (still loading, none, or no such route on this deck).
  if (!showEmpty && packs.length === 0) return null

  return (
    <div id={MY_PACKS_ID} className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Layers className="h-4 w-4 text-violet-500" />
        <h3 className="text-sm font-semibold text-zinc-200">{MY_PACKS_HEADING}</h3>
      </div>
      {error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
          {error}
        </div>
      )}
      {sharingOff && packs.length > 0 && (
        <p className="text-[11px] text-amber-700 dark:text-amber-400">{PACKS_PAUSED_NOTE}</p>
      )}
      {loading && packs.length === 0 ? (
        <div className="flex justify-center py-4"><Loader2 className="h-5 w-5 animate-spin text-zinc-500" /></div>
      ) : packs.length === 0 ? (
        <p className="text-xs text-zinc-500">{MY_PACKS_EMPTY}</p>
      ) : (
        <div className="space-y-1.5">
          {packs.map((p) => (
            <div key={p.id} className="flex items-start gap-2 rounded-md border border-zinc-800 px-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="truncate text-xs text-zinc-300">
                  {packSummary(p)}
                  <span className="text-zinc-600"> · </span>
                  <span className="text-zinc-400">{p.agent_name}</span>
                  {sharingOff && (
                    <span className="ml-1.5 rounded border border-amber-500/30 px-1 py-px text-[9px] font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-400">
                      paused
                    </span>
                  )}
                </div>
                {/* With sharing off nothing is active — the inactive reasons would be wrong. */}
                {!sharingOff && !p.active && (
                  <p className="mt-0.5 text-[11px] text-amber-700 dark:text-amber-400">{INACTIVE_HINT}</p>
                )}
              </div>
              <button
                type="button"
                onClick={() => { void stop(p.id) }}
                disabled={!!busyId}
                className="flex shrink-0 items-center gap-1 rounded-md border border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
              >
                {busyId === p.id && <Loader2 className="h-3 w-3 animate-spin" />}
                Stop sharing
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
