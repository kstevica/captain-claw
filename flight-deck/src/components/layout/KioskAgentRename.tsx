import { useEffect, useRef, useState } from 'react'
import { Loader2, PencilLine } from 'lucide-react'
import { AgentDialog } from './KioskAgentDialogs'
import { SwitchTrack } from '../common/SwitchTrack'
import { renameProcess } from '../../services/docker'
import { useProcessStore } from '../../stores/processStore'
import { useContainerStore } from '../../stores/containerStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'

// ── Rename: what an agent is called ──────────────────────────────────
//
// One of your own agents (Flight Deck runs it): the name goes in the deck's
// registry, so every browser of yours shows it, and — unless the person says
// no — the agent goes by it too: its personality ("You are …") takes the new
// name, at once when it runs, else from its next start (server.py,
// /fd/processes/{slug}/identity with `introduce`). A Docker or local agent,
// or one shared with you, gets the name in your own list only.

export const AGENT_NAME_MAX = 60

export interface RenameTarget {
  kind: 'docker' | 'process' | 'local' | 'shared'
  name: string
  /** The process slug, the container id, the local agent's id, or a shared agent's slug. */
  key: string
  /** A shared agent's owner. */
  ownerName?: string
}

export function KioskAgentRename({ agent: given, onClose }: { agent: RenameTarget; onClose: () => void }) {
  // As it was when the dialog opened: the list shows the new name at once,
  // and the title shouldn't change under the person while it saves.
  const [agent] = useState(given)
  const [name, setName] = useState(agent.name)
  const [introduce, setIntroduce] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  // Renamed, but the agent didn't take the name the usual way (it runs and
  // couldn't be told, or its file couldn't be written): said before closing.
  const [note, setNote] = useState('')
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // Ready to type over: the whole name selected.
    input.current?.focus()
    input.current?.select()
  }, [])

  const clean = name.replace(/\s+/g, ' ').trim()
  const changed = clean !== agent.name
  const own = agent.kind === 'process'

  const save = async () => {
    if (note) { onClose(); return }
    if (!clean || saving) return
    if (!changed) { onClose(); return }
    setSaving(true)
    setError('')
    try {
      if (agent.kind === 'process') {
        const procs = useProcessStore.getState()
        const before = procs.nameOverrides[agent.key]
        procs.setNameOverride(agent.key, clean)   // the list shows it at once
        let res
        try {
          res = await renameProcess(agent.key, clean, introduce)
        } catch (e) {
          procs.setNameOverride(agent.key, before ?? agent.name)
          throw e
        }
        // The new name in the list says it went fine; only a surprise is told.
        if (res.introduced === 'restart' || res.introduced === 'failed') {
          setNote(res.message)
          setSaving(false)
          return
        }
      } else if (agent.kind === 'docker') {
        useContainerStore.getState().setNameOverride(agent.key, clean)
      } else if (agent.kind === 'local') {
        useLocalAgentStore.getState().updateAgent(agent.key, { name: clean })
      } else {
        useProcessStore.getState().setNameOverride(agent.key, clean)
      }
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setSaving(false)
    }
  }

  const where = agent.kind === 'shared'
    ? `Only you see this name${agent.ownerName ? ` — ${agent.ownerName} keeps theirs` : ''}.`
    : agent.kind === 'process' ? '' : 'The name in your list; the agent itself isn’t told.'

  return (
    <AgentDialog title="Rename" icon={PencilLine} agentName={agent.name} onClose={onClose}>
      <form
        className="min-h-0 flex-1 space-y-4 overflow-y-auto overscroll-contain p-5"
        onSubmit={(e) => { e.preventDefault(); save() }}
      >
        <label className="block">
          <span className="text-xs font-medium text-zinc-300">Name</span>
          <input
            ref={input}
            value={name}
            maxLength={AGENT_NAME_MAX}
            onChange={(e) => { setName(e.target.value); setError('') }}
            placeholder={agent.name}
            aria-label="New name"
            className="mt-1 w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-violet-500/60 focus:outline-none"
          />
          {where && <span className="mt-1 block text-[11px] leading-relaxed text-zinc-500">{where}</span>}
          {own && <span className="mt-1 block text-[11px] leading-relaxed text-zinc-500">Its folder stays <code className="rounded bg-zinc-800 px-1">{agent.key}</code>.</span>}
        </label>
        {own && (
          <button
            type="button" role="switch" aria-checked={introduce} onClick={() => setIntroduce((v) => !v)}
            className="flex w-full items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-3 text-left hover:border-zinc-700"
          >
            <span className="mt-0.5"><SwitchTrack on={introduce} /></span>
            <span className="min-w-0">
              <span className="block text-sm font-medium text-zinc-100">It goes by this name too</span>
              <span className="mt-1 block text-[11px] leading-relaxed text-zinc-500">
                {introduce
                  ? `It calls itself ${clean || 'this'} when it talks to you and to others.`
                  : 'Only the name in your list changes; it keeps calling itself what it did.'}
              </span>
            </span>
          </button>
        )}
        {error && <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">{error}</p>}
        {note && <p role="status" className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-800 dark:text-amber-200">{note}</p>}
        <div className="flex justify-end gap-2 pt-1">
          {note ? (
            <button type="button" onClick={onClose} className="h-9 rounded-lg bg-violet-600 px-4 text-sm font-medium text-white transition-colors hover:bg-violet-500">Done</button>
          ) : (
            <>
              <button type="button" onClick={onClose} className="h-9 rounded-lg px-4 text-sm text-zinc-300 hover:bg-zinc-800">Cancel</button>
              <button
                type="submit" disabled={!clean || saving}
                className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-violet-600 px-4 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {saving && <Loader2 className="h-4 w-4 animate-spin" />}Rename
              </button>
            </>
          )}
        </div>
      </form>
    </AgentDialog>
  )
}
