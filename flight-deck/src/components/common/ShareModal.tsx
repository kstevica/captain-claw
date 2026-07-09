import { useEffect, useState, useCallback } from 'react'
import { X, Users, Trash2, Loader2, Check, Search, Eye, Pencil } from 'lucide-react'
import {
  listShareUsers,
  listResourceShares,
  createShare,
  deleteShare,
  type ResourceType,
  type Permission,
  type ShareUser,
  type ResourceShare,
} from '../../services/shares'

interface Props {
  resourceType: ResourceType
  resourceId: string
  resourceName: string
  /** Archetypes are use-only — hide the View/Edit toggle and force 'view'. */
  allowEdit?: boolean
  onClose: () => void
}

export function ShareModal({ resourceType, resourceId, resourceName, allowEdit = true, onClose }: Props) {
  const [users, setUsers] = useState<ShareUser[]>([])
  const [shares, setShares] = useState<ResourceShare[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('')
  const [permission, setPermission] = useState<Permission>('view')
  const [busyId, setBusyId] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [u, s] = await Promise.all([listShareUsers(), listResourceShares(resourceType, resourceId)])
      setUsers(u)
      setShares(s)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }, [resourceType, resourceId])

  useEffect(() => { load() }, [load])

  const sharedIds = new Set(shares.map((s) => s.grantee_id))

  const grant = async (granteeId: string, perm: Permission) => {
    setBusyId(granteeId)
    try {
      await createShare(resourceType, resourceId, granteeId, allowEdit ? perm : 'view')
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to share')
    } finally {
      setBusyId(null)
    }
  }

  const revoke = async (granteeId: string) => {
    setBusyId(granteeId)
    try {
      await deleteShare(resourceType, resourceId, granteeId)
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to revoke')
    } finally {
      setBusyId(null)
    }
  }

  const q = filter.trim().toLowerCase()
  const candidates = users.filter(
    (u) =>
      !sharedIds.has(u.id) &&
      (!q || u.email.toLowerCase().includes(q) || (u.display_name || '').toLowerCase().includes(q)),
  )

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-full max-w-md flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <Users className="h-4 w-4 text-violet-400" />
          <div className="min-w-0 flex-1">
            <div className="text-sm font-semibold text-zinc-100">Share</div>
            <div className="truncate text-xs text-zinc-500">{resourceName}</div>
          </div>
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>

        {error && (
          <div className="border-b border-red-900/50 bg-red-950/20 px-4 py-2 text-xs text-red-400">{error}</div>
        )}

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Current shares */}
          <div>
            <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
              Shared with ({shares.length})
            </div>
            {shares.length === 0 ? (
              <p className="text-xs text-zinc-600">Not shared with anyone yet.</p>
            ) : (
              <div className="space-y-1">
                {shares.map((s) => (
                  <div key={s.grantee_id} className="flex items-center gap-2 rounded-md border border-zinc-800 px-2.5 py-1.5">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm text-zinc-200">{s.grantee_name || s.grantee_email}</div>
                      <div className="truncate text-[11px] text-zinc-500">{s.grantee_email}</div>
                    </div>
                    {allowEdit && (
                      <select
                        value={s.permission}
                        onChange={(e) => grant(s.grantee_id, e.target.value as Permission)}
                        disabled={busyId === s.grantee_id}
                        className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
                      >
                        <option value="view">View</option>
                        <option value="edit">Edit</option>
                      </select>
                    )}
                    <button
                      onClick={() => revoke(s.grantee_id)}
                      disabled={busyId === s.grantee_id}
                      className="rounded p-1 text-zinc-500 hover:bg-red-950/40 hover:text-red-400"
                      title="Revoke"
                    >
                      {busyId === s.grantee_id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Add people */}
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Add people</span>
              {allowEdit && (
                <div className="flex overflow-hidden rounded-md border border-zinc-700 text-[11px]">
                  <button
                    onClick={() => setPermission('view')}
                    className={`flex items-center gap-1 px-2 py-0.5 ${permission === 'view' ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:bg-zinc-800'}`}
                  >
                    <Eye className="h-3 w-3" /> View
                  </button>
                  <button
                    onClick={() => setPermission('edit')}
                    className={`flex items-center gap-1 px-2 py-0.5 ${permission === 'edit' ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:bg-zinc-800'}`}
                  >
                    <Pencil className="h-3 w-3" /> Edit
                  </button>
                </div>
              )}
            </div>
            <div className="relative mb-2">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-600" />
              <input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Search people by name or email…"
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 py-1.5 pl-8 pr-2.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              />
            </div>
            {loading ? (
              <div className="flex justify-center py-6"><Loader2 className="h-5 w-5 animate-spin text-zinc-600" /></div>
            ) : candidates.length === 0 ? (
              <p className="py-3 text-center text-xs text-zinc-600">
                {users.length === 0 ? 'No other users to share with.' : 'No matching people.'}
              </p>
            ) : (
              <div className="max-h-56 space-y-1 overflow-y-auto">
                {candidates.map((u) => (
                  <button
                    key={u.id}
                    onClick={() => grant(u.id, permission)}
                    disabled={busyId === u.id}
                    className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left hover:bg-zinc-800/60 disabled:opacity-50"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm text-zinc-200">{u.display_name || u.email}</div>
                      <div className="truncate text-[11px] text-zinc-500">{u.email}</div>
                    </div>
                    {busyId === u.id
                      ? <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />
                      : <Check className="h-3.5 w-3.5 text-zinc-600" />}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
