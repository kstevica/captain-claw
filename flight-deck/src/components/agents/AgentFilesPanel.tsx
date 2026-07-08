import { useState, useEffect, useMemo } from 'react'
import {
  FileText, Loader2, AlertCircle, FolderOpen, RefreshCw, Download, Eye,
  Search, Image, FileCode, FileSpreadsheet, Film, Music, Archive,
  Pin, MonitorPlay, Pencil,
} from 'lucide-react'
import type { AgentFile, AgentEndpoint } from '../../services/fileTransfer'
import {
  listAgentFiles, formatSize, getDownloadUrl, getViewUrl,
  getFileTypeGroup, isViewable,
} from '../../services/fileTransfer'
import { FileViewer } from './FileViewer'
import { usePinnedFilesStore } from '../../stores/pinnedFilesStore'
import { useContainerStore } from '../../stores/containerStore'
import { useProcessStore } from '../../stores/processStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'

// File groups whose text content can be edited in place (mirrors FileBrowser).
const EDITABLE_GROUPS = new Set(['markdown', 'code', 'data', 'text', 'html'])

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
function useAgentEndpoint(containerId: string): AgentEndpoint | null {
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
  if (proc && proc.web_port) {
    return { id: `proc-${proc.slug}`, name: proc.name, host: 'localhost', port: proc.web_port, auth: proc.web_auth }
  }
  return null
}

function FileIcon({ file }: { file: AgentFile }) {
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
  useEffect(() => { loadFiles() }, [agentId])

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

  const isEditable = (f: AgentFile) => EDITABLE_GROUPS.has(getFileTypeGroup(f))
  const isDeckable = (f: AgentFile) => getFileTypeGroup(f) === 'html'

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
    const q = new URLSearchParams({
      c: 'deck', path: f.physical, host: agent.host, port: String(agent.port), auth: agent.auth || '',
    })
    window.open(`${window.location.origin}/deck/view?${q.toString()}`, '_blank')
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
            {processedFiles.map((f) => (
              <li key={f.physical} className="group rounded-md px-1.5 py-1 hover:bg-zinc-900/60">
                <div className="flex items-center gap-1.5">
                  <FileIcon file={f} />
                  <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-300" title={f.logical || f.physical}>
                    {f.filename}
                  </span>
                  <span className="shrink-0 text-[10px] text-zinc-600">{formatSize(f.size)}</span>
                </div>
                <div className="mt-0.5 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
                  <IconBtn onClick={() => handlePin(f)} title={isFilePinned(agent.id, f.physical) ? 'Pinned' : 'Pin file'} Icon={Pin} active={isFilePinned(agent.id, f.physical)} hoverClass="hover:text-amber-400" />
                  {isViewable(f) && <IconBtn onClick={() => handleView(f)} title="View" Icon={Eye} />}
                  {isEditable(f) && <IconBtn onClick={() => handleEdit(f)} title="Edit" Icon={Pencil} hoverClass="hover:text-amber-400" />}
                  {isDeckable(f) && <IconBtn onClick={() => handleDeck(f)} title="Present as deck" Icon={MonitorPlay} hoverClass="hover:text-emerald-400" />}
                  <IconBtn onClick={() => handleDownload(f)} title="Download" Icon={Download} />
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

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
