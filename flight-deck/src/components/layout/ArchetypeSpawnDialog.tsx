import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Library, Loader2, Search, Star, X } from 'lucide-react'
import { listArchetypes } from '../../services/archetypes'
import { spawnArchetypeProcess } from '../../services/docker'
import type { Archetype } from '../../services/tierConfig'
import { useProcessStore } from '../../stores/processStore'

// ── Kiosk "New agent" picker ─────────────────────────────────────────
//
// The locked Simple layout has no Spawner, so this is the one way a kiosk user
// creates an agent: pick a Library archetype, name it, done. Everything else —
// tools, cognitive mode, model — is resolved by FD from the archetype and the
// account's tier set (or the admin's team default), so no model config or API
// key is exposed here.

/** Mirrors the backend `_slug`: a process agent's name is its directory. */
function agentSlug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/^-+|-+$/g, '') || 'cc-agent'
}

function uniqueName(base: string, taken: Set<string>): string {
  if (!taken.has(agentSlug(base))) return base
  for (let n = 2; ; n++) {
    const candidate = `${base} ${n}`
    if (!taken.has(agentSlug(candidate))) return candidate
  }
}

export function ArchetypeSpawnDialog({ takenSlugs, onClose, onSpawned, asUser, ownerLabel }: {
  /** Slugs of the agents this user can see — the default name avoids them. */
  takenSlugs: Set<string>
  onClose: () => void
  onSpawned: (slug: string) => void
  /** Admin creating the agent FOR that user: their gallery, their ownership,
   *  their tier set and plan. */
  asUser?: string
  ownerLabel?: string
}) {
  const [archetypes, setArchetypes] = useState<Archetype[] | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const [query, setQuery] = useState('')
  const [picked, setPicked] = useState<Archetype | null>(null)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const nameRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    listArchetypes(asUser)
      .then((reg) => setArchetypes(reg.archetypes || []))
      .catch((e) => setLoadErr(e instanceof Error ? e.message : String(e)))
  }, [asUser])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !busy) onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onClose])

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase()
    const shown = (archetypes || []).filter((a) => !q
      || a.role.toLowerCase().includes(q) || a.id.toLowerCase().includes(q)
      || a.family.toLowerCase().includes(q) || a.description.toLowerCase().includes(q))
    const byFamily = new Map<string, Archetype[]>()
    for (const a of shown) {
      const fam = a.family || 'Other'
      byFamily.set(fam, [...(byFamily.get(fam) || []), a])
    }
    return [...byFamily.entries()].sort(([x], [y]) => x.localeCompare(y))
  }, [archetypes, query])

  const pick = (a: Archetype) => {
    setPicked(a)
    setName(uniqueName(a.role || a.id, takenSlugs))
    setErr('')
  }

  // Focus the name once the pick has re-rendered it enabled, so Enter creates.
  useEffect(() => { if (picked) nameRef.current?.select() }, [picked])

  const create = async () => {
    if (!picked || !name.trim() || busy) return
    setBusy(true)
    setErr('')
    let finalName = name.trim()
    try {
      let res
      try {
        res = await spawnArchetypeProcess(picked.id, finalName, picked.description, asUser)
      } catch (e) {
        // Taken by an agent this user can't see (another account's): try once
        // more with a short suffix rather than make a kiosk user guess.
        const msg = e instanceof Error ? e.message : String(e)
        if (!/already (exists|running)/i.test(msg)) throw e
        finalName = `${finalName} ${Math.random().toString(36).slice(2, 6)}`
        res = await spawnArchetypeProcess(picked.id, finalName, picked.description, asUser)
      }
      if (!res.ok) throw new Error(res.message)
      const slug = res.slug || agentSlug(finalName)
      // FD records the archetype's instructions for the owner; mirror them
      // into this browser's store only when the agent is the viewer's own.
      if (!asUser) {
        const procs = useProcessStore.getState()
        if (picked.fleet_instructions) procs.setFleetInstructions(slug, picked.fleet_instructions)
        procs.fetchProcesses()
      }
      onSpawned(slug)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => { if (!busy) onClose() }}>
      <div
        role="dialog"
        aria-label="New agent from an archetype"
        className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
            <Library className="h-3.5 w-3.5" />
          </span>
          <div>
            <h3 className="text-sm font-semibold text-zinc-200">New agent{asUser && ownerLabel ? ` for ${ownerLabel}` : ''}</h3>
            <p className="text-[11px] text-zinc-500">Pick an archetype — its role, tools and model come with it.</p>
          </div>
          <button onClick={onClose} disabled={busy} className="ml-auto rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50" title="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Search */}
        <div className="shrink-0 px-4 pt-3">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-600" />
            <input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search archetypes…"
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 py-1.5 pl-8 pr-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
            />
          </div>
        </div>

        {/* Gallery */}
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
          {loadErr && <p className="py-6 text-center text-sm text-red-600 dark:text-red-400">Couldn't load archetypes: {loadErr}</p>}
          {!loadErr && archetypes === null && (
            <p className="flex items-center justify-center gap-2 py-6 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</p>
          )}
          {archetypes && groups.length === 0 && (
            <p className="py-6 text-center text-sm text-zinc-500">
              {archetypes.length === 0 ? 'No archetypes available. Ask an administrator to add some.' : 'No archetypes match your search.'}
            </p>
          )}
          {groups.map(([family, list]) => (
            <div key={family} className="mb-3">
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">{family}</div>
              <ul className="flex flex-col gap-1">
                {list.map((a) => {
                  const active = picked?.id === a.id
                  return (
                    <li key={a.id}>
                      <button
                        onClick={() => pick(a)}
                        disabled={busy}
                        className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                          active
                            ? 'border-violet-500/60 bg-violet-600/10'
                            : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-800/50'
                        }`}
                      >
                        <div className="flex items-center gap-1.5">
                          <span className="text-sm font-medium text-zinc-100">{a.role || a.id}</span>
                          {a.lead && <Star className="h-3 w-3 text-amber-500" aria-label="Lead" />}
                          {a.tier && (
                            <span className="ml-auto rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">{a.tier}</span>
                          )}
                        </div>
                        {a.description && <p className="mt-0.5 line-clamp-2 text-xs text-zinc-500">{a.description}</p>}
                      </button>
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </div>

        {/* Name + create */}
        <div className="shrink-0 border-t border-zinc-800 px-4 py-3">
          {err && <p className="mb-2 text-xs text-red-600 dark:text-red-400">{err}</p>}
          <form
            className="flex items-center gap-2"
            onSubmit={(e) => { e.preventDefault(); create() }}
          >
            <input
              ref={nameRef}
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={!picked || busy}
              placeholder={picked ? 'Agent name' : 'Pick an archetype first'}
              aria-label="Agent name"
              className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={!picked || !name.trim() || busy}
              className="flex shrink-0 items-center gap-1.5 rounded-lg bg-violet-600 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
            >
              {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              {busy ? 'Creating…' : 'Create agent'}
            </button>
          </form>
        </div>
      </div>
    </div>,
    document.body,
  )
}
