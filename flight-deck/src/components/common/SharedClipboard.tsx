import { useState, useRef } from 'react'
import {
  ClipboardList,
  X,
  Plus,
  Pin,
  Trash2,
  Copy,
  Check,
  Send,
  Edit3,
} from 'lucide-react'
import { useClipboardStore, type ClipboardEntry } from '../../stores/clipboardStore'
import { useChatStore } from '../../stores/chatStore'
import { useContainerStore } from '../../stores/containerStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'

export function SharedClipboard({ onClose }: { onClose: () => void }) {
  const { entries, addEntry, removeEntry, togglePin, updateEntry, clear } = useClipboardStore()
  const [newContent, setNewContent] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editContent, setEditContent] = useState('')
  const [sendingTo, setSendingTo] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const { sessions, openChat, sendMessage } = useChatStore()
  const containers = useContainerStore((s) => s.containers)
  const localAgents = useLocalAgentStore((s) => s.agents)

  const allAgents = [
    ...containers.filter((c) => c.status === 'running' && c.web_port).map((c) => ({
      id: c.id, name: c.agent_name || c.name, host: 'localhost', port: c.web_port!, auth: c.web_auth || '',
    })),
    ...localAgents.filter((a) => a.status === 'online').map((a) => ({
      id: a.id, name: a.name, host: a.host, port: a.port, auth: a.authToken || '',
    })),
  ]

  const handleAdd = () => {
    if (!newContent.trim()) return
    addEntry(newContent.trim(), 'user')
    setNewContent('')
    textareaRef.current?.focus()
  }

  const handleSendToAgent = (entryContent: string, agentId: string) => {
    const agent = allAgents.find((a) => a.id === agentId)
    if (!agent) return
    const session = sessions.get(agentId)
    if (!session) {
      openChat(agentId, agent.name, agent.host, agent.port, agent.auth)
      setTimeout(() => sendMessage(agentId, entryContent), 500)
    } else {
      sendMessage(agentId, entryContent)
    }
    setSendingTo(null)
  }

  const startEdit = (entry: ClipboardEntry) => {
    setEditingId(entry.id)
    setEditContent(entry.content)
  }

  const saveEdit = (id: string) => {
    if (editContent.trim()) updateEntry(id, editContent.trim())
    setEditingId(null)
  }

  const pinnedEntries = entries.filter((e) => e.pinned)
  const unpinnedEntries = entries.filter((e) => !e.pinned)

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2.5">
        <div className="flex items-center gap-2">
          <ClipboardList className="h-4 w-4 text-cyan-400" />
          <span className="text-sm font-semibold">Shared Clipboard</span>
          <span className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">{entries.length}</span>
        </div>
        <div className="flex items-center gap-1">
          {entries.length > 0 && (
            <button onClick={clear} className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:text-red-400" title="Clear unpinned">
              Clear
            </button>
          )}
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Add new entry */}
      <div className="border-b border-zinc-800 p-2.5">
        <textarea
          ref={textareaRef}
          value={newContent}
          onChange={(e) => setNewContent(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleAdd() }}
          placeholder="Add a note to the shared clipboard..."
          className="w-full resize-none rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          rows={2}
        />
        <div className="mt-1.5 flex items-center justify-between">
          <span className="text-[10px] text-zinc-600">Cmd+Enter to add</span>
          <button
            onClick={handleAdd}
            disabled={!newContent.trim()}
            className="flex items-center gap-1 rounded-md bg-violet-600 px-2 py-1 text-[10px] font-medium text-white hover:bg-violet-500 disabled:opacity-40"
          >
            <Plus className="h-3 w-3" />
            Add
          </button>
        </div>
      </div>

      {/* Entries */}
      <div className="flex-1 overflow-y-auto">
        {entries.length === 0 ? (
          <div className="px-3 py-8 text-center text-xs text-zinc-600">
            Clipboard is empty. Add notes to share context between agents.
          </div>
        ) : (
          <>
            {pinnedEntries.length > 0 && (
              <div>
                <div className="px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-600">Pinned</div>
                {pinnedEntries.map((e) => (
                  <ClipEntry
                    key={e.id} entry={e}
                    editing={editingId === e.id} editContent={editContent}
                    onEdit={() => startEdit(e)} onEditChange={setEditContent}
                    onSaveEdit={() => saveEdit(e.id)} onCancelEdit={() => setEditingId(null)}
                    onTogglePin={() => togglePin(e.id)} onRemove={() => removeEntry(e.id)}
                    sendingTo={sendingTo === e.id} onSendTo={() => setSendingTo(sendingTo === e.id ? null : e.id)}
                    agents={allAgents} onSendToAgent={(agentId) => handleSendToAgent(e.content, agentId)}
                  />
                ))}
              </div>
            )}
            {unpinnedEntries.length > 0 && (
              <div>
                {pinnedEntries.length > 0 && (
                  <div className="px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-600">Recent</div>
                )}
                {unpinnedEntries.map((e) => (
                  <ClipEntry
                    key={e.id} entry={e}
                    editing={editingId === e.id} editContent={editContent}
                    onEdit={() => startEdit(e)} onEditChange={setEditContent}
                    onSaveEdit={() => saveEdit(e.id)} onCancelEdit={() => setEditingId(null)}
                    onTogglePin={() => togglePin(e.id)} onRemove={() => removeEntry(e.id)}
                    sendingTo={sendingTo === e.id} onSendTo={() => setSendingTo(sendingTo === e.id ? null : e.id)}
                    agents={allAgents} onSendToAgent={(agentId) => handleSendToAgent(e.content, agentId)}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function ClipEntry({
  entry, editing, editContent, onEdit, onEditChange, onSaveEdit, onCancelEdit,
  onTogglePin, onRemove, sendingTo, onSendTo, agents, onSendToAgent,
}: {
  entry: ClipboardEntry
  editing: boolean
  editContent: string
  onEdit: () => void
  onEditChange: (v: string) => void
  onSaveEdit: () => void
  onCancelEdit: () => void
  onTogglePin: () => void
  onRemove: () => void
  sendingTo: boolean
  onSendTo: () => void
  agents: { id: string; name: string }[]
  onSendToAgent: (agentId: string) => void
}) {
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(entry.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="group border-b border-zinc-800/40 px-3 py-2 hover:bg-zinc-900/30">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 text-[10px] text-zinc-600">
          <span className="font-medium text-zinc-500">{entry.source}</span>
          <span>·</span>
          <span>{new Date(entry.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <button onClick={copy} className="rounded p-0.5 text-zinc-600 hover:text-zinc-300" title="Copy">
            {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
          </button>
          <button onClick={onSendTo} className="rounded p-0.5 text-zinc-600 hover:text-zinc-300" title="Send to agent">
            <Send className="h-3 w-3" />
          </button>
          <button onClick={onEdit} className="rounded p-0.5 text-zinc-600 hover:text-zinc-300" title="Edit">
            <Edit3 className="h-3 w-3" />
          </button>
          <button onClick={onTogglePin} className={`rounded p-0.5 ${entry.pinned ? 'text-amber-400' : 'text-zinc-600 hover:text-amber-400'}`} title={entry.pinned ? 'Unpin' : 'Pin'}>
            <Pin className="h-3 w-3" />
          </button>
          <button onClick={onRemove} className="rounded p-0.5 text-zinc-600 hover:text-red-400" title="Delete">
            <Trash2 className="h-3 w-3" />
          </button>
        </div>
      </div>

      {editing ? (
        <div className="mt-1">
          <textarea
            value={editContent}
            onChange={(e) => onEditChange(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) onSaveEdit(); if (e.key === 'Escape') onCancelEdit() }}
            className="w-full resize-none rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
            rows={3}
            autoFocus
          />
          <div className="mt-1 flex gap-1">
            <button onClick={onSaveEdit} className="rounded bg-violet-600 px-1.5 py-0.5 text-[10px] text-white hover:bg-violet-500">Save</button>
            <button onClick={onCancelEdit} className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:text-zinc-300">Cancel</button>
          </div>
        </div>
      ) : (
        <p className="mt-1 text-xs text-zinc-400 whitespace-pre-wrap break-words">{entry.content.length > 300 ? entry.content.slice(0, 300) + '...' : entry.content}</p>
      )}

      {/* Send to agent dropdown */}
      {sendingTo && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {agents.map((a) => (
            <button
              key={a.id}
              onClick={() => onSendToAgent(a.id)}
              className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-300 hover:bg-violet-600/20 hover:text-violet-400"
            >
              {a.name}
            </button>
          ))}
          {agents.length === 0 && <span className="text-[10px] text-zinc-600">No agents online</span>}
        </div>
      )}
    </div>
  )
}
