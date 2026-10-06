import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Users, MessageSquare, LogOut, Loader2, Layers, FolderOpen, Database, X } from 'lucide-react'
import type { SharedAgent } from '../../services/sharedAgents'
import { useChatStore } from '../../stores/chatStore'
import { useSharedAgentStore } from '../../stores/sharedAgentStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { SharedAgentGoogleToggle } from './SharedAgentGoogleToggle'
import { ContextPacksModal } from './ContextPacksModal'
import { SharedFilesPanel } from './SharedFilesPanel'
import { DatastoreBrowser } from './DatastoreBrowser'
import { PACKS_BUTTON } from '../../utils/contextPacks'
import { sharedCaps } from '../../utils/sharedAgent'
import { DATA_BUTTON, FILES_BUTTON, workspaceVisible } from '../../utils/sharedWorkspace'

// An agent another deck user shared with you, on the Agent Desktop. No power,
// config or the owner's private files and memory — those stay with its owner.
// On a process agent the member's own Google switch lives here (and in the
// chat), and — on a deck that serves them — "Files" and "Data" open the
// agent's saved/ folder and datastore, which everyone who uses it shares.
// "Shared context" opens what the member shares with the agent's people.

/** The member's Files panel as a dialog (Esc or the X closes it; a file
 *  viewer open inside handles its own Esc first). */
function SharedFilesDialog({ agentRef, agentName, ownerName, onClose }: {
  agentRef: string
  agentName: string
  ownerName: string
  onClose: () => void
}) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // Wait for every listener: the viewer marks the Esc it used to close.
      setTimeout(() => { if (!e.defaultPrevented) onClose() }, 0)
    }
    document.addEventListener('keydown', h)
    return () => document.removeEventListener('keydown', h)
  }, [onClose])
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="relative flex flex-col overflow-hidden rounded-xl border border-zinc-700/50 bg-zinc-900 shadow-2xl"
        style={{ width: 'min(520px, 92vw)', height: '75vh' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex h-9 shrink-0 items-center justify-between border-b border-zinc-800 px-3">
          <span className="min-w-0 truncate text-xs font-medium text-zinc-400">{agentName}</span>
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Close (Esc)">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1">
          <SharedFilesPanel agentRef={agentRef} agentName={agentName} ownerName={ownerName} />
        </div>
      </div>
    </div>,
    document.body,
  )
}

export function SharedAgentCard({ agent }: { agent: SharedAgent }) {
  const openSharedChat = useChatStore((s) => s.openSharedChat)
  const leave = useSharedAgentStore((s) => s.leave)
  const contextPacks = useSharedAgentStore((s) => s.contextPacks)
  const memberWorkspace = useSharedAgentStore((s) => s.memberWorkspace)
  const [leaving, setLeaving] = useState(false)
  const [showPacks, setShowPacks] = useState(false)
  const [showFiles, setShowFiles] = useState(false)
  const [showData, setShowData] = useState(false)
  const running = agent.status === 'running'
  const owner = agent.owner_name || agent.owner_email || 'another user'
  const name = agent.name || agent.slug
  const vis = workspaceVisible(memberWorkspace, sharedCaps(agent))

  const handleLeave = async () => {
    if (!confirm(`Leave ${name}? You'll lose access until ${owner} shares it again.`)) return
    setLeaving(true)
    try {
      await leave(agent.agent_ref, agent.owner_id)
    } catch (e) {
      useNotificationStore.getState().add('error', 'Could not leave agent',
        `${name}: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setLeaving(false)
    }
  }

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${running ? 'bg-emerald-600 dark:bg-emerald-400' : 'bg-zinc-600'}`}
          title={running ? 'Running' : 'Stopped'}
        />
        <span className="min-w-0 flex-1 truncate text-sm font-semibold text-zinc-100">{name}</span>
        <span
          className="flex shrink-0 items-center gap-0.5 rounded border border-sky-500/25 bg-sky-500/15 px-1 py-0.5 text-[9px] font-medium text-sky-700 dark:text-sky-300"
          title={`Shared by ${owner}`}
        >
          <Users className="h-2.5 w-2.5" />shared
        </span>
      </div>
      <div className="text-[11px] text-zinc-500">
        Shared by {owner}{running ? '' : ' · Stopped'}
      </div>
      {agent.description && (
        <p className="line-clamp-2 text-xs text-zinc-400">{agent.description}</p>
      )}
      <SharedAgentGoogleToggle agent={agent} />
      <div className="mt-1 flex items-center gap-2">
        <button
          onClick={() => openSharedChat(agent.agent_ref, name, agent.owner_name || owner)}
          disabled={!running}
          title={running ? `Chat with ${name}` : `${name} is stopped — ask ${owner} to start it`}
          className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
        >
          <MessageSquare className="h-3.5 w-3.5" />
          Chat
        </button>
        {vis.files && (
          <button
            onClick={() => setShowFiles(true)}
            disabled={!running}
            title={running ? `Files saved on ${name} — everyone who uses it can open them` : `${name} is stopped — ask ${owner} to start it`}
            className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40 disabled:hover:bg-transparent"
          >
            <FolderOpen className="h-3.5 w-3.5" />
            {FILES_BUTTON}
          </button>
        )}
        {vis.datastore && (
          <button
            onClick={() => setShowData(true)}
            disabled={!running}
            title={running ? `${name}'s data tables — everyone who uses it can see them` : `${name} is stopped — ask ${owner} to start it`}
            className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40 disabled:hover:bg-transparent"
          >
            <Database className="h-3.5 w-3.5" />
            {DATA_BUTTON}
          </button>
        )}
        {contextPacks && (
          <button
            onClick={() => setShowPacks(true)}
            title={`What you share with everyone who uses ${name}`}
            className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
          >
            <Layers className="h-3.5 w-3.5" />
            {PACKS_BUTTON}
          </button>
        )}
        <button
          onClick={handleLeave}
          disabled={leaving}
          title={`Leave ${name}`}
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-red-600 disabled:opacity-40 dark:hover:text-red-400"
        >
          {leaving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <LogOut className="h-3.5 w-3.5" />}
          Leave
        </button>
      </div>
      {contextPacks && showPacks && (
        <ContextPacksModal agentRef={agent.agent_ref} agentName={name} onClose={() => setShowPacks(false)} />
      )}
      {vis.files && showFiles && (
        <SharedFilesDialog agentRef={agent.agent_ref} agentName={name} ownerName={owner} onClose={() => setShowFiles(false)} />
      )}
      {vis.datastore && showData && createPortal(
        <DatastoreBrowser sharedRef={agent.agent_ref} agentName={name} ownerName={owner} onClose={() => setShowData(false)} />,
        document.body,
      )}
    </div>
  )
}
