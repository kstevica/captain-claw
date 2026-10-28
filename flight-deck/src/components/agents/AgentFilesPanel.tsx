import { useState, useEffect, useMemo, useRef } from 'react'
import {
  FileText, Loader2, AlertCircle, FolderOpen, RefreshCw, Download, Eye,
  Search, Image, FileCode, FileSpreadsheet, Film, Music, Archive,
  Pin, MonitorPlay, Pencil, Globe,
} from 'lucide-react'
import type { AgentFile, AgentEndpoint } from '../../services/fileTransfer'
import {
  listAgentFiles, formatSize, getDownloadUrl, getViewUrl, getDeckUrl,
  getFileTypeGroup, isViewable, isEditable, processEndpoint,
} from '../../services/fileTransfer'
import { FileViewer } from './FileViewer'
import { FileActionSheet } from './FileActionSheet'
import { PublicLinkDialog } from './PublicLinkDialog'
import { usePublicLinksEnabled } from '../../services/publicLinks'
import { useIsMobile } from '../../hooks/useMediaQuery'
import { CreatorBadge } from './CreatorBadge'
import { isMemberCreated } from '../../utils/sharedWorkspace'
import { usePinnedFilesStore } from '../../stores/pinnedFilesStore'
import { useContainerStore } from '../../stores/containerStore'
import { useProcessStore } from '../../stores/processStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { useChatStore } from '../../stores/chatStore'

const TYPE_ICONS: Record<string, typeof FileText> = {
  image: Image, video: Film, audio: Music, code: FileCode,
  data: FileSpreadsheet, archive: Archive,
}
const TYPE_COLORS: Record<string, string> = {
  image: 'text-blue-400', video: 'text-pink-400', audio: 'text-amber-400',
  code: 'text-emerald-400', data: 'text-cyan-400', archive: 'text-orange-400',
  html: 'text-orange-300', markdown: 'text-violet-400', pdf: 'text-red-400',
}

// Resolve the full agent endpoint (id + name + connection) for a chat container.
// Mirrors DesktopPage's allAgentEndpoints construction so pinning/transfer keys
// line up with the rest of the app.
export function useAgentEndpoint(containerId: string): AgentEndpoint | null {
  const containers = useContainerStore((s) => s.containers)
  const processes = useProcessStore((s) => s.processes)
  const localAgents = useLocalAgentStore((s) => s.agents)

  const container = containers.find((c) => c.id === containerId)
  if (container && container.web_port) {
    return { id: container.id, name: container.agent_name || container.name, host: 'localhost', port: container.web_port, auth: container.web_auth }
  }
  const local = localAgents.find((a) => a.id === containerId)
  if (local) {
    return { id: local.id, name: local.name, host: local.host, port: local.port, auth: local.authToken }
  }
  const proc = processes.find((p) => `proc-${p.slug}` === containerId)
  return (proc && processEndpoint(proc)) || null
}

/** Goes up by one each time the agent's chat (any lane) gets through again
 *  after being cut off — Flight Deck restarted it, or it moved port. A panel
 *  left showing an error from while it was away reloads on it. */
export function useAgentBack(containerId: string): number {
  const live = useChatStore((s) => {
    for (const x of s.sessions.values()) if (x.containerId === containerId && x.connected) return true
    return false
  })
  const [back, setBack] = useState(0)
  const wasLive = useRef(live)
  useEffect(() => {
    if (live && !wasLive.current) setBack((n) => n + 1)
    wasLive.current = live
  }, [live])
  return back
}

export function FileIcon({ file }: { file: AgentFile }) {
  const group = getFileTypeGroup(file)
  const Icon = TYPE_ICONS[group] || FileText
  return <Icon className={`h-3.5 w-3.5 shrink-0 ${TYPE_COLORS[group] || 'text-zinc-500'}`} />
}

function IconBtn({ onClick, title, Icon, active, hoverClass }: {
  onClick: () => void
  title: string
  Icon: typeof FileText
  active?: boolean
  hoverClass?: string
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`rounded p-0.5 transition-colors ${
        active ? 'text-amber-400' : `text-zinc-500 hover:bg-zinc-800 ${hoverClass || 'hover:text-zinc-300'}`
      }`}
    >
      <Icon className="h-3.5 w-3.5" />
    </button>
  )
}

// Compact, sidebar-sized files view for a single agent. Reuses the same
// listAgentFiles service, FileViewer modal and pinned-files store as the
// full-page FileBrowser, trimmed to the per-file actions (pin, view, edit,
// present, download) that fit a narrow column.
export function AgentFilesPanel({ containerId }: { containerId: string }) {
  const agent = useAgentEndpoint(containerId)
  const [files, setFiles] = useState<AgentFile[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [viewingFile, setViewingFile] = useState<AgentFile | null>(null)
  const [viewerStartEdit, setViewerStartEdit] = useState(false)
  const { pin: pinFile, isPinned: isFilePinned } = usePinnedFilesStore()
  // A phone has no hover: a tap on a file opens its actions in a sheet.
  const { isMobile: phone } = useIsMobile()
  const [actionFile, setActionFile] = useState<AgentFile | null>(null)
  const [sharingFile, setSharingFile] = useState<AgentFile | null>(null)
  const publicLinks = usePublicLinksEnabled()

  const agentId = agent?.id
  const loadFiles = async () => {
    if (!agent) return
    setLoading(true)
    setError('')
    try {
      const result = await listAgentFiles(agent.host, agent.port, agent.auth)
      setFiles(result.filter((f) => f.exists))
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { loadFiles() }, [agentId, agent?.host, agent?.port])
  // Failed while the agent was away (restarting with Flight Deck): its chat
  // getting through again means it's back.
  const back = useAgentBack(containerId)
  useEffect(() => { if (back && error) loadFiles() }, [back])

  const processedFiles = useMemo(() => {
    let r = [...files]
    if (search) {
      const q = search.toLowerCase()
      r = r.filter((f) =>
        f.filename.toLowerCase().includes(q) ||
        f.logical.toLowerCase().includes(q) ||
        f.extension.toLowerCase().includes(q))
    }
    r.sort((a, b) => b.modified - a.modified)
    return r
  }, [files, search])

  if (!agent) {
    return (
      <div className="flex h-full items-center justify-center px-3 text-center text-[11px] text-zinc-500">
        Agent not connected — files unavailable.
      </div>
    )
  }

  // A member's HTML never runs as a deck (Flight Deck's /deck/view refuses it too).
  const isDeckable = (f: AgentFile) => getFileTypeGroup(f) === 'html' && !isMemberCreated(f.created_by)

  const handleView = (f: AgentFile) => {
    if (getFileTypeGroup(f) === 'pdf') {
      window.open(getViewUrl(agent.host, agent.port, f.physical, agent.auth), '_blank')
    } else {
      setViewerStartEdit(false)
      setViewingFile(f)
    }
  }
  const handleEdit = (f: AgentFile) => { setViewerStartEdit(true); setViewingFile(f) }
  const handleDownload = (f: AgentFile) => {
    const a = document.createElement('a')
    a.href = getDownloadUrl(agent.host, agent.port, f.physical, agent.auth)
    a.download = f.filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }
  const handleDeck = (f: AgentFile) => {
    window.open(getDeckUrl(agent.host, agent.port, f.physical, agent.auth), '_blank')
  }
  const handlePin = (f: AgentFile) => {
    if (isFilePinned(agent.id, f.physical)) return
    pinFile({
      agentId: agent.id,
      agentName: agent.name,
      host: agent.host,
      port: agent.port,
      auth: agent.auth,
      filename: f.filename,
      extension: f.extension,
      physical: f.physical,
      logical: f.logical,
      size: f.size,
      mime_type: f.mime_type,
      source: f.source,
      created_by: f.created_by,
    })
  }

  // Prev/next navigation for the viewer, over the currently-visible viewables.
  const viewableFiles = processedFiles.filter((f) => isViewable(f))
  const viewIdx = viewingFile ? viewableFiles.findIndex((f) => f.physical === viewingFile.physical) : -1

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center gap-2">
          <FolderOpen className="h-3.5 w-3.5 text-violet-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-zinc-300">Files</span>
          {files.length > 0 && (
            <span className="rounded-full bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-medium text-violet-300">{files.length}</span>
          )}
        </div>
        <button
          onClick={loadFiles}
          className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
          title="Refresh"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Search */}
      <div className="border-b border-zinc-800/50 px-2 py-1.5">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-600" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search files…"
            className="w-full rounded-md border border-zinc-700 bg-zinc-900 py-1 pl-7 pr-2 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
          />
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto px-1 py-1">
        {loading && (
          <div className="flex justify-center py-6"><Loader2 className="h-4 w-4 animate-spin text-zinc-500" /></div>
        )}
        {error && (
          <div className="flex items-start gap-1.5 px-2 py-3 text-[11px] text-red-400">
            <AlertCircle className="h-3.5 w-3.5 shrink-0" />{error}
          </div>
        )}
        {!loading && !error && processedFiles.length === 0 && (
          <p className="px-2 py-6 text-center text-[11px] text-zinc-500">
            {files.length === 0 ? 'No files yet.' : 'No files match your search.'}
          </p>
        )}
        {!loading && !error && processedFiles.length > 0 && (
          <ul className="flex flex-col gap-0.5">
            {processedFiles.map((f) => phone ? (
              <li key={f.physical}>
                <button
                  onClick={() => setActionFile(f)}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-3 text-left active:bg-zinc-800/60"
                >
                  <FileIcon file={f} />
                  <span className="min-w-0 flex-1 truncate text-sm text-zinc-200">{f.filename}</span>
                  <CreatorBadge mode="owner" creator={f.created_by} />
                  <span className="shrink-0 text-xs text-zinc-500">{formatSize(f.size)}</span>
                </button>
              </li>
            ) : (
              <li key={f.physical} className="group rounded-md px-1.5 py-1 hover:bg-zinc-900/60">
                <div className="flex items-center gap-1.5">
                  <FileIcon file={f} />
                  <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-300" title={f.logical || f.physical}>
                    {f.filename}
                  </span>
                  <CreatorBadge mode="owner" creator={f.created_by} />
                  <span className="shrink-0 text-[10px] text-zinc-600">{formatSize(f.size)}</span>
                </div>
                <div className="mt-0.5 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
                  <IconBtn onClick={() => handlePin(f)} title={isFilePinned(agent.id, f.physical) ? 'Pinned' : 'Pin file'} Icon={Pin} active={isFilePinned(agent.id, f.physical)} hoverClass="hover:text-amber-400" />
                  {isViewable(f) && <IconBtn onClick={() => handleView(f)} title="View" Icon={Eye} />}
                  {isEditable(f) && <IconBtn onClick={() => handleEdit(f)} title="Edit" Icon={Pencil} hoverClass="hover:text-amber-400" />}
                  {isDeckable(f) && <IconBtn onClick={() => handleDeck(f)} title="Present as deck" Icon={MonitorPlay} hoverClass="hover:text-emerald-400" />}
                  <IconBtn onClick={() => handleDownload(f)} title="Download" Icon={Download} />
                  {publicLinks && <IconBtn onClick={() => setSharingFile(f)} title="Share — public link" Icon={Globe} hoverClass="hover:text-violet-400" />}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Phone: what a tap on a file offers */}
      {actionFile && (
        <FileActionSheet
          filename={actionFile.filename}
          size={actionFile.size}
          mime={actionFile.mime_type}
          icon={<FileIcon file={actionFile} />}
          detail={<CreatorBadge mode="owner" creator={actionFile.created_by} />}
          downloadUrl={getDownloadUrl(agent.host, agent.port, actionFile.physical, agent.auth)}
          onView={isViewable(actionFile) ? () => { setActionFile(null); handleView(actionFile) } : undefined}
          onDownload={() => handleDownload(actionFile)}
          onPublicLink={publicLinks ? () => { setActionFile(null); setSharingFile(actionFile) } : undefined}
          onClose={() => setActionFile(null)}
        />
      )}

      {sharingFile && (
        <PublicLinkDialog agent={agent} file={sharingFile} onClose={() => setSharingFile(null)} />
      )}

      {/* Viewer modal */}
      {viewingFile && (
        <FileViewer
          file={viewingFile}
          host={agent.host}
          port={agent.port}
          auth={agent.auth}
          startInEdit={viewerStartEdit}
          onClose={() => { setViewingFile(null); setViewerStartEdit(false) }}
          hasPrev={viewIdx > 0}
          hasNext={viewIdx >= 0 && viewIdx < viewableFiles.length - 1}
          onPrev={() => { if (viewIdx > 0) setViewingFile(viewableFiles[viewIdx - 1]) }}
          onNext={() => { if (viewIdx >= 0 && viewIdx < viewableFiles.length - 1) setViewingFile(viewableFiles[viewIdx + 1]) }}
        />
      )}
    </div>
  )
}
