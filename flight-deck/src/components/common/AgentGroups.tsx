import { useState, useRef, useEffect } from 'react'
import { Tag, Plus, X, Trash2, Pencil, Check, Users, Briefcase } from 'lucide-react'
import { useGroupStore } from '../../stores/groupStore'
import type { GroupType } from '../../stores/groupStore'

// ── Color mapping for group badges ──

const colorMap: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  violet: { bg: 'bg-violet-500/15', text: 'text-violet-400', border: 'border-violet-500/25', dot: 'bg-violet-400' },
  blue: { bg: 'bg-blue-500/15', text: 'text-blue-400', border: 'border-blue-500/25', dot: 'bg-blue-400' },
  emerald: { bg: 'bg-emerald-500/15', text: 'text-emerald-400', border: 'border-emerald-500/25', dot: 'bg-emerald-400' },
  amber: { bg: 'bg-amber-500/15', text: 'text-amber-400', border: 'border-amber-500/25', dot: 'bg-amber-400' },
  pink: { bg: 'bg-pink-500/15', text: 'text-pink-400', border: 'border-pink-500/25', dot: 'bg-pink-400' },
  cyan: { bg: 'bg-cyan-500/15', text: 'text-cyan-400', border: 'border-cyan-500/25', dot: 'bg-cyan-400' },
  red: { bg: 'bg-red-500/15', text: 'text-red-400', border: 'border-red-500/25', dot: 'bg-red-400' },
  indigo: { bg: 'bg-indigo-500/15', text: 'text-indigo-400', border: 'border-indigo-500/25', dot: 'bg-indigo-400' },
}

function getColors(color: string) {
  return colorMap[color] || colorMap.violet
}

// ── Group badges shown on agent cards ──

export function AgentGroupBadges({ agentId }: { agentId: string }) {
  const groups = useGroupStore((s) => s.groups)
  const { addToGroup, removeFromGroup, createGroup } = useGroupStore()
  const [showPicker, setShowPicker] = useState(false)
  const [newGroupName, setNewGroupName] = useState('')
  const [newGroupType, setNewGroupType] = useState<GroupType>('group')
  const pickerRef = useRef<HTMLDivElement>(null)

  const agentGroups = groups.filter((g) => g.agentIds.includes(agentId))
  const availableGroups = groups.filter((g) => !g.agentIds.includes(agentId))

  // Close picker on outside click
  useEffect(() => {
    if (!showPicker) return
    const handler = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setShowPicker(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showPicker])

  const handleCreateAndAdd = () => {
    if (!newGroupName.trim()) return
    const id = createGroup(newGroupName.trim(), newGroupType)
    addToGroup(id, agentId)
    setNewGroupName('')
  }

  return (
    <div className="flex flex-wrap items-center gap-1 relative">
      {agentGroups.map((g) => {
        const c = getColors(g.color)
        return (
          <span
            key={g.id}
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${c.bg} ${c.text} ${c.border} group/badge`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${c.dot}`} />
            {g.name}
            <button
              onClick={(e) => { e.stopPropagation(); removeFromGroup(g.id, agentId) }}
              className="opacity-0 group-hover/badge:opacity-100 transition-opacity -mr-0.5 rounded-full hover:bg-white/10 p-0.5"
            >
              <X className="h-2 w-2" />
            </button>
          </span>
        )
      })}

      {/* Add to group button */}
      <button
        onClick={(e) => { e.stopPropagation(); setShowPicker(!showPicker) }}
        className="inline-flex items-center gap-0.5 rounded-full border border-dashed border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-600 hover:border-zinc-500 hover:text-zinc-400 transition-colors"
      >
        <Tag className="h-2.5 w-2.5" />
        {agentGroups.length === 0 && 'Group'}
      </button>

      {/* Group picker dropdown */}
      {showPicker && (
        <div
          ref={pickerRef}
          className="absolute top-full left-0 z-50 mt-1 w-52 rounded-xl border border-zinc-700/50 bg-zinc-900 shadow-xl shadow-black/30"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-2">
            {/* Groups section */}
            {(() => {
              const availGroups = availableGroups.filter(g => (g.type || 'group') === 'group')
              const availRoles = availableGroups.filter(g => g.type === 'role')
              return (
                <>
                  <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500 px-1">Groups</p>
                  {availGroups.length > 0 ? (
                    <div className="space-y-0.5 max-h-24 overflow-y-auto">
                      {availGroups.map((g) => {
                        const c = getColors(g.color)
                        return (
                          <button
                            key={g.id}
                            onClick={() => { addToGroup(g.id, agentId); setShowPicker(false) }}
                            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 transition-colors"
                          >
                            <span className={`h-2 w-2 rounded-full ${c.dot}`} />
                            {g.name}
                          </button>
                        )
                      })}
                    </div>
                  ) : (
                    <p className="px-1 py-1 text-[10px] text-zinc-600">
                      {groups.some(g => (g.type || 'group') === 'group') ? 'In all groups' : 'No groups yet'}
                    </p>
                  )}

                  {/* Roles section */}
                  <p className="mt-2 mb-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500 px-1">Roles</p>
                  {availRoles.length > 0 ? (
                    <div className="space-y-0.5 max-h-24 overflow-y-auto">
                      {availRoles.map((g) => {
                        const c = getColors(g.color)
                        return (
                          <button
                            key={g.id}
                            onClick={() => { addToGroup(g.id, agentId); setShowPicker(false) }}
                            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 transition-colors"
                          >
                            <span className={`h-2 w-2 rounded-full ${c.dot}`} />
                            {g.name}
                          </button>
                        )
                      })}
                    </div>
                  ) : (
                    <p className="px-1 py-1 text-[10px] text-zinc-600">
                      {groups.some(g => g.type === 'role') ? 'Has all roles' : 'No roles yet'}
                    </p>
                  )}
                </>
              )
            })()}

            {/* Create new group/role inline */}
            <div className="mt-1.5 border-t border-zinc-800 pt-1.5">
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setNewGroupType(newGroupType === 'group' ? 'role' : 'group')}
                  className="shrink-0 rounded-md border border-zinc-700/50 bg-zinc-950 p-1 text-zinc-400 hover:text-zinc-200 transition-colors"
                  title={newGroupType === 'group' ? 'Creating group (click for role)' : 'Creating role (click for group)'}
                >
                  {newGroupType === 'group' ? <Users className="h-3 w-3" /> : <Briefcase className="h-3 w-3" />}
                </button>
                <input
                  value={newGroupName}
                  onChange={(e) => setNewGroupName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleCreateAndAdd() }}
                  placeholder={newGroupType === 'group' ? 'New group...' : 'New role...'}
                  className="flex-1 rounded-md border border-zinc-700/50 bg-zinc-950 px-2 py-1 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
                  autoFocus
                />
                <button
                  onClick={handleCreateAndAdd}
                  disabled={!newGroupName.trim()}
                  className="rounded-md bg-violet-600 p-1 text-white hover:bg-violet-500 disabled:opacity-40"
                >
                  <Plus className="h-3 w-3" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Group Manager panel for the Director ──

export function GroupManager() {
  const { groups: allItems, createGroup, deleteGroup, renameGroup } = useGroupStore()
  const [newName, setNewName] = useState('')
  const [newType, setNewType] = useState<GroupType>('group')
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameInput, setRenameInput] = useState('')

  const groups = allItems.filter(g => (g.type || 'group') === 'group')
  const roles = allItems.filter(g => g.type === 'role')

  const handleCreate = () => {
    if (!newName.trim()) return
    createGroup(newName.trim(), newType)
    setNewName('')
  }

  const handleRename = (id: string) => {
    if (renameInput.trim()) renameGroup(id, renameInput.trim())
    setRenamingId(null)
  }

  const renderList = (items: typeof allItems, emptyText: string) => (
    items.length === 0 ? (
      <p className="py-2 text-center text-[11px] text-zinc-600">{emptyText}</p>
    ) : (
      <div className="space-y-1">
        {items.map((g) => {
          const c = getColors(g.color)
          return (
            <div key={g.id} className="flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-zinc-800/50 group/row">
              <span className={`h-2.5 w-2.5 rounded-full ${c.dot} shrink-0`} />
              {renamingId === g.id ? (
                <div className="flex flex-1 items-center gap-1">
                  <input
                    value={renameInput}
                    onChange={(e) => setRenameInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleRename(g.id); if (e.key === 'Escape') setRenamingId(null) }}
                    onBlur={() => handleRename(g.id)}
                    className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-xs text-zinc-200 focus:outline-none"
                    autoFocus
                  />
                  <button onClick={() => handleRename(g.id)} className="rounded p-0.5 text-emerald-400 hover:bg-zinc-800">
                    <Check className="h-3 w-3" />
                  </button>
                </div>
              ) : (
                <>
                  <span className="flex-1 text-xs text-zinc-300">{g.name}</span>
                  <span className="text-[10px] text-zinc-600">{g.agentIds.length}</span>
                  <div className="flex items-center gap-0.5 opacity-0 group-hover/row:opacity-100 transition-opacity">
                    <button
                      onClick={() => { setRenamingId(g.id); setRenameInput(g.name) }}
                      className="rounded p-0.5 text-zinc-600 hover:text-zinc-300 hover:bg-zinc-800"
                    >
                      <Pencil className="h-3 w-3" />
                    </button>
                    <button
                      onClick={() => { if (confirm(`Delete '${g.name}'?`)) deleteGroup(g.id) }}
                      className="rounded p-0.5 text-zinc-600 hover:text-red-400 hover:bg-red-500/10"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </>
              )}
            </div>
          )
        })}
      </div>
    )
  )

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Users className="h-4 w-4 text-violet-400" />
        <span className="text-sm font-semibold text-zinc-200">Groups</span>
        <span className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">{groups.length}</span>
      </div>
      {renderList(groups, 'No groups yet.')}

      <div className="flex items-center gap-2 pt-1">
        <Briefcase className="h-4 w-4 text-amber-400" />
        <span className="text-sm font-semibold text-zinc-200">Roles</span>
        <span className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">{roles.length}</span>
      </div>
      {renderList(roles, 'No roles yet.')}

      {/* Create new */}
      <div className="flex gap-1.5 pt-1 border-t border-zinc-800">
        <button
          onClick={() => setNewType(newType === 'group' ? 'role' : 'group')}
          className="shrink-0 rounded-lg border border-zinc-700/50 bg-zinc-950 px-2 py-1.5 text-zinc-400 hover:text-zinc-200 transition-colors"
          title={newType === 'group' ? 'Creating group (click for role)' : 'Creating role (click for group)'}
        >
          {newType === 'group' ? <Users className="h-3.5 w-3.5" /> : <Briefcase className="h-3.5 w-3.5" />}
        </button>
        <input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
          placeholder={newType === 'group' ? 'New group name...' : 'New role name...'}
          className="flex-1 rounded-lg border border-zinc-700/50 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />
        <button
          onClick={handleCreate}
          disabled={!newName.trim()}
          className="rounded-lg bg-violet-600 px-2.5 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-40"
        >
          <Plus className="h-3 w-3" />
        </button>
      </div>
    </div>
  )
}

// ── Group filter for Desktop page ──

export function GroupFilter({ selected, onChange }: { selected: string | null; onChange: (id: string | null) => void }) {
  const allGroups = useGroupStore((s) => s.groups)
  // Only show groups (not roles) in the filter bar
  const groups = allGroups.filter(g => (g.type || 'group') === 'group')

  if (groups.length === 0) return null

  return (
    <div className="flex items-center gap-1">
      <button
        onClick={() => onChange(null)}
        className={`rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors ${
          selected === null
            ? 'bg-zinc-700 text-zinc-200'
            : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
        }`}
      >
        All
      </button>
      {groups.map((g) => {
        const c = getColors(g.color)
        const isActive = selected === g.id
        return (
          <button
            key={g.id}
            onClick={() => onChange(isActive ? null : g.id)}
            className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors ${
              isActive
                ? `${c.bg} ${c.text} border ${c.border}`
                : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${c.dot}`} />
            {g.name}
            <span className="text-[9px] opacity-60">{g.agentIds.length}</span>
          </button>
        )
      })}
    </div>
  )
}
