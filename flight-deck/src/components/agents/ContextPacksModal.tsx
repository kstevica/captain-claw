import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { X, Layers, Loader2, Trash2 } from 'lucide-react'
import {
  getAgentPacks,
  createPack,
  deletePack,
  type AgentPacks,
  type PackRow,
} from '../../services/contextPacks'
import { useSharedAgentStore } from '../../stores/sharedAgentStore'
import {
  PACKS_TITLE,
  PROFILE_PACK_LABEL,
  PROFILE_PACK_HINT,
  VFS_PACK_LABEL,
  VFS_PACK_HINT,
  DEEP_PACK_LABEL,
  DEEP_PACK_HINT,
  TAGS_LABEL,
  NO_TAGS_NOTE,
  DOCKER_PACKS_NOTE,
  REVOKE_NOTE,
  ACTIVE_PACKS_HEADING,
  NO_PACKS_TEXT,
  INACTIVE_HINT,
  packsDisclosure,
  packConfirmText,
  packSummary,
  publisherName,
  packRoleBadge,
  packOwnerTag,
  removePackConfirm,
  toggleTag,
  fullAlias,
  aliasSuffixError,
  capabilityNote,
  capabilityWarns,
} from '../../utils/contextPacks'

// "Share with this agent's people": the caller publishes their OWN profile,
// folders (read-only) and deep memory to everyone who uses an agent they own
// or are a member of — and sees (and, as the owner, can remove) what everyone
// else shares there. Flight Deck checks every request; this dialog only says
// what sharing means, asks before publishing, and shows FD's answer.

const sectionLabel = 'text-[10px] font-semibold uppercase tracking-wider text-zinc-500'

// Who published a row: the name (You, or the quoted name), then FD's role as
// its own badge on every row, then FD's collision tag when it sent one. The
// name never carries the role, so a member can't name themselves into one.
function Publisher({ p }: { p: PackRow }) {
  const tag = packOwnerTag(p)
  const role = packRoleBadge(p.role)
  return (
    <>
      <span className={p.mine ? 'font-medium text-sky-700 dark:text-sky-300' : 'text-zinc-100'}>{publisherName(p)}</span>
      {' '}
      <span
        title={role === 'owner' ? 'The agent’s owner' : 'A member of this agent'}
        className={`rounded border px-1 py-px text-[9px] font-semibold uppercase tracking-wider ${
          role === 'owner' ? 'border-violet-500/40 text-violet-700 dark:text-violet-300' : 'border-zinc-700 text-zinc-500'}`}
      >
        {role}
      </span>
      {tag && <span className="ml-1 font-mono text-[10px] text-zinc-500" title="Tells publishers with the same name apart">{tag}</span>}
    </>
  )
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function Switch({ on, label, busy, disabled, onClick }: {
  on: boolean; label: string; busy: boolean; disabled: boolean; onClick: () => void
}) {
  return (
    <div className="flex min-w-0 items-center gap-2 text-xs text-zinc-300">
      <button
        type="button"
        role="switch"
        aria-checked={on}
        aria-label={label}
        onClick={onClick}
        disabled={disabled}
        className={`relative h-4 w-7 shrink-0 rounded-full transition-colors disabled:cursor-wait disabled:opacity-60 ${
          on ? 'bg-sky-500' : 'bg-zinc-700'}`}
      >
        <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${
          on ? 'translate-x-3.5' : 'translate-x-0.5'}`} />
      </button>
      <span className="min-w-0 flex-1 font-medium">{label}</span>
      {busy && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-zinc-500" />}
    </div>
  )
}

export function ContextPacksModal({ agentRef, agentName, onClose }: {
  agentRef: string
  agentName: string
  onClose: () => void
}) {
  const hostWarning = useSharedAgentStore((s) => s.hostWarning)
  const [data, setData] = useState<AgentPacks | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  // The request running now ('profile' | 'vfs' | 'deep' | 'rm:<id>'); every
  // control that changes something is disabled until it answers.
  const [busy, setBusy] = useState<string | null>(null)
  const [project, setProject] = useState('')
  const [suffix, setSuffix] = useState('')
  const [tags, setTags] = useState<string[]>([])

  // Esc closes.
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', h)
    return () => document.removeEventListener('keydown', h)
  }, [onClose])

  // A failed load keeps the last data and shows FD's reason.
  const load = useCallback(async () => {
    try {
      const d = await getAgentPacks(agentRef)
      setData(d)
      setError('')
    } catch (e) {
      setError(errorText(e) || 'Failed to load')
    } finally {
      setLoading(false)
    }
  }, [agentRef])

  useEffect(() => {
    setLoading(true)
    void load()
  }, [load])

  // Every create/delete reloads, success or not (FD's answer may have
  // changed what's there); a failure's reason stays on screen.
  const run = async (key: string, fn: () => Promise<unknown>) => {
    if (busy) return
    setBusy(key)
    let failure = ''
    try {
      await fn()
    } catch (e) {
      failure = errorText(e) || 'Request failed'
    }
    await load()
    if (failure) setError(failure)
    setBusy(null)
  }

  const name = (data?.agent_name || agentName || '').trim()
  const kinds = data?.kinds || []
  const mine = data?.mine || []
  const profileRow = mine.find((p) => p.kind === 'profile')
  const deepRow = mine.find((p) => p.kind === 'deep_memory')
  const vfsRows = mine.filter((p) => p.kind === 'vfs')
  const prefix = data?.alias_prefix || ''
  const maxVfs = data?.limits?.max_vfs_per_owner ?? 0
  const maxTags = data?.limits?.max_tags ?? 10
  const sharedProjects = new Set(vfsRows.map((p) => p.project))
  const available = (data?.eligible_projects || []).filter((p) => !sharedProjects.has(p))
  const selProject = available.includes(project) ? project : (available[0] || '')
  const suffixError = aliasSuffixError(prefix, suffix)
  const facets = data?.deep_memory_tags || []
  const facetTags = new Set(facets.map((f) => f.tag))
  const capNote = data
    ? capabilityNote(data.agent_supports_packs, data.role, data.agent_running, data.owner_name)
    : ''

  const toggleProfile = () => {
    if (profileRow) return run('profile', () => deletePack(profileRow.id))
    if (!window.confirm(packConfirmText('profile', name))) return
    return run('profile', () => createPack({ agent_ref: agentRef, kind: 'profile' }))
  }

  const shareFolder = () => {
    if (!selProject || suffixError) return
    const folder = { project: selProject, alias: fullAlias(prefix, suffix), prefix }
    if (!window.confirm(packConfirmText('vfs', name, [], folder))) return
    return run('vfs', async () => {
      await createPack({ agent_ref: agentRef, kind: 'vfs', project: selProject, alias: fullAlias(prefix, suffix) })
      setSuffix('')
    })
  }

  const toggleDeep = () => {
    if (deepRow) return run('deep', () => deletePack(deepRow.id))
    const picked = tags.filter((t) => facetTags.has(t))
    if (!window.confirm(packConfirmText('deep_memory', name, picked))) return
    return run('deep', async () => {
      await createPack({ agent_ref: agentRef, kind: 'deep_memory', tags: picked })
      setTags([])
    })
  }

  const remove = (p: PackRow) => {
    if (!p.mine && !window.confirm(removePackConfirm(p))) return
    return run(`rm:${p.id}`, () => deletePack(p.id))
  }

  const stopButton = (p: PackRow) => (
    <button
      type="button"
      onClick={() => run(`rm:${p.id}`, () => deletePack(p.id))}
      disabled={!!busy}
      className="flex shrink-0 items-center gap-1 rounded-md border border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
    >
      {busy === `rm:${p.id}` && <Loader2 className="h-3 w-3 animate-spin" />}
      Stop sharing
    </button>
  )

  return createPortal(
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-full max-w-lg flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <Layers className="h-4 w-4 text-violet-400" />
          <div className="min-w-0 flex-1">
            <div className="text-sm font-semibold text-zinc-100">{PACKS_TITLE}</div>
            <div className="truncate text-xs text-zinc-500">{name || agentName}</div>
          </div>
          <button onClick={onClose} title="Close" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>

        {error && (
          <div className="border-b border-red-900/50 bg-red-950/20 px-4 py-2 text-xs text-red-400">{error}</div>
        )}

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {!data && loading && (
            <div className="flex justify-center py-6"><Loader2 className="h-5 w-5 animate-spin text-zinc-600" /></div>
          )}

          {data && (<>
            {/* 1. Who gets it (incl. the agent's channels and automations) */}
            <p className="whitespace-pre-line rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-[11px] leading-relaxed text-zinc-400">
              {packsDisclosure(data.agent_name || agentName, data.owner_name, data.role, data.runtime)}
            </p>

            {/* 2. Does the running agent understand shared context? */}
            {capNote && (capabilityWarns(data.agent_supports_packs, data.agent_running) ? (
              <div className="rounded-md border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-[11px] leading-relaxed text-amber-800 dark:text-amber-200">
                {capNote}
              </div>
            ) : (
              <p className="text-[11px] leading-relaxed text-zinc-500">{capNote}</p>
            ))}

            {/* 3. What you share */}
            <div className="space-y-3">
              <div className={sectionLabel}>What you share</div>
              {data.runtime === 'docker' && (
                <p className="text-[11px] text-zinc-500">{DOCKER_PACKS_NOTE}</p>
              )}

              {kinds.includes('profile') && (
                <div className="space-y-1 rounded-md border border-zinc-800 px-3 py-2">
                  <Switch
                    on={!!profileRow}
                    label={PROFILE_PACK_LABEL}
                    busy={busy === 'profile'}
                    disabled={!!busy}
                    onClick={() => { void toggleProfile() }}
                  />
                  <p className="text-[11px] text-zinc-500">{PROFILE_PACK_HINT}</p>
                </div>
              )}

              {kinds.includes('vfs') && (
                <div className="space-y-2 rounded-md border border-zinc-800 px-3 py-2">
                  <div className="text-xs font-medium text-zinc-300">{VFS_PACK_LABEL}</div>
                  <p className="text-[11px] text-zinc-500">{VFS_PACK_HINT}</p>
                  {vfsRows.map((p) => (
                    <div key={p.id} className="space-y-0.5">
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate text-xs text-zinc-300">{packSummary(p)}</span>
                        {stopButton(p)}
                      </div>
                      {p.active === false && (
                        <p className="text-[11px] text-amber-700 dark:text-amber-400">{INACTIVE_HINT}</p>
                      )}
                    </div>
                  ))}
                  {vfsRows.length < maxVfs && available.length > 0 && (
                    <div className="space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <select
                          value={selProject}
                          onChange={(e) => setProject(e.target.value)}
                          disabled={!!busy}
                          className="min-w-0 max-w-[10rem] rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
                        >
                          {available.map((p) => <option key={p} value={p}>{p}</option>)}
                        </select>
                        <div className="flex min-w-0 flex-1 items-center rounded border border-zinc-700 bg-zinc-950 text-xs focus-within:border-violet-500/50">
                          <span className="shrink-0 pl-2 font-mono text-zinc-500">vfs:@{prefix}-</span>
                          <input
                            value={suffix}
                            onChange={(e) => setSuffix(e.target.value)}
                            placeholder="name (optional)"
                            disabled={!!busy}
                            className="min-w-0 flex-1 bg-transparent py-1 pr-2 font-mono text-zinc-200 placeholder-zinc-600 focus:outline-none"
                          />
                        </div>
                        <button
                          type="button"
                          onClick={() => { void shareFolder() }}
                          disabled={!!busy || !selProject || !!suffixError}
                          className="flex shrink-0 items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
                        >
                          {busy === 'vfs' && <Loader2 className="h-3 w-3 animate-spin" />}
                          Share
                        </button>
                      </div>
                      {suffixError && <p className="text-[11px] text-red-600 dark:text-red-400">{suffixError}</p>}
                    </div>
                  )}
                  {available.length === 0 && vfsRows.length === 0 && (
                    <p className="text-[11px] text-zinc-600">No folder of yours can be shared.</p>
                  )}
                </div>
              )}

              {kinds.includes('deep_memory') && (
                <div className="space-y-2 rounded-md border border-zinc-800 px-3 py-2">
                  <Switch
                    on={!!deepRow}
                    label={DEEP_PACK_LABEL}
                    busy={busy === 'deep'}
                    disabled={!!busy}
                    onClick={() => { void toggleDeep() }}
                  />
                  <p className="text-[11px] text-zinc-500">{DEEP_PACK_HINT}</p>
                  {deepRow ? (
                    <div className="flex items-center gap-2">
                      <span className="min-w-0 flex-1 truncate text-xs text-zinc-300">{packSummary(deepRow)}</span>
                      {stopButton(deepRow)}
                    </div>
                  ) : facets.length > 0 ? (
                    <div className="space-y-1">
                      <div className="text-[11px] text-zinc-400">{TAGS_LABEL}</div>
                      <div className="flex flex-wrap gap-1">
                        {facets.map((f) => {
                          const on = tags.includes(f.tag)
                          const full = !on && tags.length >= maxTags
                          return (
                            <label
                              key={f.tag}
                              className={`flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] ${
                                on ? 'border-sky-500/40 bg-sky-500/15 text-sky-700 dark:text-sky-300'
                                  : 'border-zinc-700 text-zinc-400'} ${full || busy ? 'opacity-50' : 'cursor-pointer'}`}
                            >
                              <input
                                type="checkbox"
                                className="h-3 w-3"
                                checked={on}
                                disabled={full || !!busy}
                                onChange={() => setTags((cur) => toggleTag(cur, f.tag, maxTags))}
                              />
                              {`${f.tag} (${f.count})`}
                            </label>
                          )
                        })}
                      </div>
                    </div>
                  ) : (
                    <p className="text-[11px] text-zinc-500">{NO_TAGS_NOTE}</p>
                  )}
                </div>
              )}
            </div>

            {/* 4. Everyone's active packs */}
            <div>
              <div className={`mb-1.5 ${sectionLabel}`}>{ACTIVE_PACKS_HEADING}</div>
              {data.packs.length === 0 ? (
                <p className="text-xs text-zinc-600">{NO_PACKS_TEXT}</p>
              ) : (
                <div className="space-y-1">
                  {data.packs.map((p) => (
                    <div key={p.id} className="flex items-center gap-2 rounded-md border border-zinc-800 px-2.5 py-1.5">
                      <span className="min-w-0 flex-1 truncate text-xs text-zinc-300">
                        <Publisher p={p} />
                        <span className="text-zinc-600"> · </span>
                        {packSummary(p)}
                      </span>
                      {p.can_remove && (
                        <button
                          type="button"
                          onClick={() => { void remove(p) }}
                          disabled={!!busy}
                          title={p.mine ? 'Stop sharing' : 'Remove from this agent'}
                          className="rounded p-1 text-zinc-500 hover:bg-red-950/40 hover:text-red-400 disabled:opacity-50"
                        >
                          {busy === `rm:${p.id}` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>)}
        </div>

        {/* 5. Footer */}
        <div className="space-y-1 border-t border-zinc-800 px-4 py-2.5">
          <p className="text-[11px] leading-relaxed text-zinc-500">{REVOKE_NOTE}</p>
          {hostWarning && <p className="text-[11px] leading-relaxed text-zinc-500">{hostWarning}</p>}
        </div>
      </div>
    </div>,
    document.body,
  )
}
