import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ArrowLeft, Check, Link2, Plus, Sparkles, X } from 'lucide-react'
import { archetypeLook } from '../layout/archetypeLook'
import type { SetupRequest } from '../layout/ArchetypeSpawnDialog'
import { Button, Switch, inputCls, labelCls } from '../console/parts'
import { ErrorNote } from '../forge/forgeParts'
import { spawnArchetypeProcess } from '../../services/docker'
import type { Archetype } from '../../services/tierConfig'
import { probeGoogle, probeMicrosoft } from '../../stores/connectionsStore'
import { useProcessStore } from '../../stores/processStore'
import { agentSlug, creatorOf, uniqueName, worksInMail } from './storeModel'

// ── "Add" from the Agent store ───────────────────────────────────────
//
// The store's one step between a card and a new agent: name it, and create it
// exactly as the New agent picker (ArchetypeSpawnDialog) does — POST
// /fd/spawn-process with the archetype, a retry with a short suffix when the
// name is taken by an agent this person can't see, then the instructions FD
// stored mirrored into the process store. "Set up its way of working next"
// hands the new agent to the caller, which opens AgentSetupDialog on it.

export function AddAgentDialog({ archetype: a, takenSlugs, onClose, onAdded }: {
  archetype: Archetype
  /** Slugs of the agents this person has — the default name avoids them. */
  takenSlugs: Set<string>
  onClose: () => void
  /** `setup`: they asked to set up its way of working next. */
  onAdded: (slug: string, name: string, setup?: SetupRequest) => void
}) {
  const [name, setName] = useState(() => uniqueName(a.role || a.id, takenSlugs))
  const [setupNext, setSetupNext] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  // The name it was created under, once it is.
  const [added, setAdded] = useState<string | null>(null)
  const nameRef = useRef<HTMLInputElement>(null)
  const look = archetypeLook(a.id, a.family || 'Other')
  const Icon = look.icon

  // Works in mail or a calendar and none of this person's accounts is connected:
  // say so before it's created, as the New agent picker does. null = unknown.
  const [mail, setMail] = useState<{ connected: boolean; microsoft: boolean } | null>(null)
  const needsMail = worksInMail(a)
  useEffect(() => {
    if (!needsMail) return
    let live = true
    Promise.all([probeGoogle(), probeMicrosoft()]).then(([g, m]) => {
      if (!live || g.health === 'gray') return  // sign-in off: no accounts to connect
      const up = (c: { health: string } | null) => !!c && (c.health === 'green' || c.health === 'yellow')
      setMail({ connected: up(g) || up(m), microsoft: m !== null })
    }).catch(() => {})
    return () => { live = false }
  }, [needsMail])

  useEffect(() => { nameRef.current?.select() }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !busy) onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onClose])

  const add = async () => {
    if (!name.trim() || busy) return
    setBusy(true)
    setErr('')
    let finalName = name.trim()
    try {
      let res
      try {
        res = await spawnArchetypeProcess(a.id, finalName, a.description, undefined)
      } catch (e) {
        // Taken by an agent this person can't see (another account's): once more with a short suffix.
        const msg = e instanceof Error ? e.message : String(e)
        if (!/already (exists|running)/i.test(msg)) throw e
        finalName = `${finalName} ${Math.random().toString(36).slice(2, 6)}`
        res = await spawnArchetypeProcess(a.id, finalName, a.description, undefined)
      }
      if (!res.ok) throw new Error(res.message)
      const slug = res.slug || agentSlug(finalName)
      const procs = useProcessStore.getState()
      const stored = res.fleet_instructions || a.fleet_instructions
      if (stored) procs.setFleetInstructions(slug, stored)
      procs.fetchProcesses()
      onAdded(slug, finalName, setupNext ? { slug, name: finalName, role: a.role || a.id } : undefined)
      if (setupNext) return
      setAdded(finalName)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return createPortal(
    <div className="fd-fade fixed inset-0 z-50 flex items-end justify-center bg-black/60 backdrop-blur-[2px] wide:items-center wide:p-4" onClick={() => { if (!busy) onClose() }}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`Add ${a.role || a.id}`}
        className="fd-sheet w-full overflow-hidden rounded-t-3xl border border-zinc-800 bg-zinc-950 shadow-2xl wide:max-w-md wide:rounded-3xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={`relative bg-gradient-to-b ${look.wash} to-transparent px-6 pb-5 pt-6`}>
          <button onClick={onClose} disabled={busy} aria-label="Close" title="Close"
            className="absolute right-3 top-3 rounded-full p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
            <X className="h-4 w-4" />
          </button>
          <span className={`flex h-16 w-16 items-center justify-center rounded-2xl ring-1 ${look.bg} ${look.text} ${look.ring}`}>
            <Icon className="h-8 w-8" />
          </span>
          <h2 className="mt-4 text-xl font-bold tracking-tight text-zinc-100">{a.role || a.id}</h2>
          <p className="text-xs text-zinc-500">{creatorOf(a)} · {a.family || 'Other'}</p>
          {a.description && <p className="mt-3 text-sm leading-relaxed text-zinc-400">{a.description}</p>}
        </div>

        {added ? (
          <div className="space-y-4 px-6 pb-6">
            <p className="flex items-start gap-2.5 rounded-2xl border border-emerald-500/30 bg-emerald-500/[0.07] px-4 py-3 text-sm text-zinc-200">
              <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
              <span><span className="font-semibold">{added}</span> is joining your team. It starts in a few seconds — you'll find it with your agents in Flight Deck.</span>
            </p>
            <div className="flex flex-col-reverse gap-2 wide:flex-row wide:justify-end">
              <Button icon={ArrowLeft} onClick={onClose}>Keep browsing</Button>
              <a href="/" className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-violet-600 px-3.5 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500">
                Go to my agents
              </a>
            </div>
          </div>
        ) : (
          <form className="space-y-3 px-6 pb-6" onSubmit={(e) => { e.preventDefault(); add() }}>
            {err && <ErrorNote>{err}</ErrorNote>}
            {needsMail && mail?.connected === false && (
              <p role="note" className="flex items-start gap-2 rounded-xl border border-sky-500/30 bg-sky-500/5 px-3 py-2.5 text-xs leading-relaxed text-sky-800 dark:text-sky-200">
                <Link2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span className="min-w-0">
                  It works in your mail and calendar, and {mail.microsoft ? 'neither Google nor Microsoft is' : 'Google isn\'t'} connected
                  yet. Connect {mail.microsoft ? 'one' : 'it'} under Connections in Flight Deck, or it can't read your inbox.
                </span>
              </p>
            )}
            <div>
              <label className={labelCls} htmlFor="store-agent-name">Its name on your team</label>
              <input id="store-agent-name" ref={nameRef} value={name} onChange={(e) => setName(e.target.value)} disabled={busy}
                className={`${inputCls} !text-base wide:!text-sm`} />
            </div>
            <label className={`flex cursor-pointer items-center justify-between gap-3 rounded-xl border border-zinc-800 bg-zinc-900/40 p-3 transition-colors hover:border-zinc-700 ${busy ? 'pointer-events-none opacity-60' : ''}`}>
              <span className="min-w-0">
                <span className="block text-sm font-medium text-zinc-200">Set up its way of working next</span>
                <span className="block text-xs text-zinc-500">Your instructions and documents become its rules and knowledge.</span>
              </span>
              <Switch on={setupNext} busy={busy} label="Set up its way of working next" onChange={setSetupNext} />
            </label>
            <Button type="submit" variant="primary" icon={setupNext ? Sparkles : Plus} busy={busy} disabled={!name.trim()} className="w-full !rounded-full !py-2.5">
              {busy ? 'Adding…' : setupNext ? 'Add and set up' : 'Add to my team'}
            </Button>
          </form>
        )}
      </div>
    </div>,
    document.body,
  )
}
