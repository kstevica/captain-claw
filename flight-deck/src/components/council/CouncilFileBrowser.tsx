import { useState, useEffect, useMemo, useCallback } from 'react'
import {
  FolderOpen, RefreshCw, Eye, Download, Loader2, AlertCircle,
  ChevronDown, ChevronRight, Search, X, FileText, Image, FileCode,
  FileSpreadsheet, Film, Music, Archive, Pin,
} from 'lucide-react'
import type { AgentFile } from '../../services/fileTransfer'
import {
  listAgentFiles, formatSize, getDownloadUrl, getViewUrl,
  getFileTypeGroup, isViewable,
} from '../../services/fileTransfer'
import { FileViewer } from '../agents/FileViewer'
import { usePinnedFilesStore } from '../../stores/pinnedFilesStore'
import type { CouncilAgent } from '../../stores/councilStore'

interface CouncilFileBrowserProps {
  agents: CouncilAgent[]
  sessionCreatedAt: string   // ISO timestamp — only show files modified after this
  sessionConcludedAt: string // ISO timestamp — only show files modified before this (empty = no upper bound)
  onClose: () => void
}

interface AgentFileEntry extends AgentFile {
  agentId: string
  agentName: string
  host: string
  port: number
  auth: string
}

const TYPE_ICONS: Record<string, typeof FileText> = {
  image: Image,
  video: Film,
  audio: Music,
  code: FileCode,
  data: FileSpreadsheet,
  archive: Archive,
}

function formatDate(ts: number): string {
  if (!ts) return '--'
  const d = new Date(ts * 1000)
  const now = new Date()
  const isToday = d.toDateString() === now.toDateString()
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  if (isToday) return `Today ${time}`
  const yesterday = new Date(now)
  yesterday.setDate(yesterday.getDate() - 1)
  if (d.toDateString() === yesterday.toDateString()) return `Yesterday ${time}`
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ` ${time}`
}

export function CouncilFileBrowser({ agents, sessionCreatedAt, sessionConcludedAt, onClose }: CouncilFileBrowserProps) {
  const [files, setFiles] = useState<AgentFileEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [collapsedAgents, setCollapsedAgents] = useState<Set<string>>(new Set())
  const [viewingFile, setViewingFile] = useState<AgentFileEntry | null>(null)

  const { pin: pinFile, isPinned: isFilePinned } = usePinnedFilesStore()

  // Only show files created/modified within the council session time window
  const sessionStartEpoch = sessionCreatedAt ? Math.floor(new Date(sessionCreatedAt).getTime() / 1000) : 0
  const sessionEndEpoch = sessionConcludedAt ? Math.floor(new Date(sessionConcludedAt).getTime() / 1000) : 0

  const loadAllFiles = useCallback(async () => {
    setLoading(true)
    setError('')
    const allFiles: AgentFileEntry[] = []
    let anyError = false

    await Promise.all(
      agents.filter(a => a.connected).map(async (agent) => {
        try {
          const agentFiles = await listAgentFiles(agent.host, agent.port, agent.auth)
          for (const f of agentFiles) {
            if (!f.exists || f.modified < sessionStartEpoch) continue
            if (sessionEndEpoch && f.modified > sessionEndEpoch) continue
            allFiles.push({
              ...f,
              agentId: agent.id,
              agentName: agent.name,
              host: agent.host,
              port: agent.port,
              auth: agent.auth,
            })
          }
        } catch {
          anyError = true
        }
      })
    )

    // Sort by modified date descending
    allFiles.sort((a, b) => b.modified - a.modified)
    setFiles(allFiles)
    if (anyError && allFiles.length === 0) setError('Failed to load files from agents')
    setLoading(false)
  }, [agents, sessionStartEpoch, sessionEndEpoch])

  useEffect(() => { loadAllFiles() }, [loadAllFiles])

  // Filter
  const filtered = useMemo(() => {
    if (!searchQuery) return files
    const q = searchQuery.toLowerCase()
    return files.filter(f =>
      f.filename.toLowerCase().includes(q) ||
      f.logical.toLowerCase().includes(q) ||
      f.agentName.toLowerCase().includes(q)
    )
  }, [files, searchQuery])

  // Group by agent
  const grouped = useMemo(() => {
    const map = new Map<string, AgentFileEntry[]>()
    for (const f of filtered) {
      const key = f.agentId
      if (!map.has(key)) map.set(key, [])
      map.get(key)!.push(f)
    }
    return map
  }, [filtered])

  const toggleAgent = (id: string) => {
    setCollapsedAgents(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleView = (f: AgentFileEntry) => {
    const group = getFileTypeGroup(f)
    if (group === 'pdf') {
      window.open(getViewUrl(f.host, f.port, f.physical, f.auth), '_blank')
    } else {
      setViewingFile(f)
    }
  }

  const handleDownload = (f: AgentFileEntry) => {
    const a = document.createElement('a')
    a.href = getDownloadUrl(f.host, f.port, f.physical, f.auth)
    a.download = f.filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  // Navigation for viewer: all viewable files in current filtered list
  const viewableFiles = filtered.filter(f => isViewable(f))
  const viewingIndex = viewingFile
    ? viewableFiles.findIndex(f => f.physical === viewingFile.physical && f.agentId === viewingFile.agentId)
    : -1

  const FileIcon = ({ file }: { file: AgentFile }) => {
    const group = getFileTypeGroup(file)
    const Icon = TYPE_ICONS[group] || FileText
    const colors: Record<string, string> = {
      image: 'text-blue-400', video: 'text-pink-400', audio: 'text-amber-400',
      code: 'text-emerald-400', data: 'text-cyan-400', archive: 'text-orange-400',
      html: 'text-orange-300', markdown: 'text-violet-400', pdf: 'text-red-400',
    }
    return <Icon className={`h-3.5 w-3.5 shrink-0 ${colors[group] || 'text-zinc-500'}`} />
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="flex h-[85vh] w-[900px] flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div className="flex items-center gap-2">
            <FolderOpen className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-semibold">Council Files</h2>
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-xs text-zinc-500 font-mono">{files.length}</span>
          </div>
          <div className="flex items-center gap-1">
            <button onClick={loadAllFiles} className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Refresh">
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button onClick={onClose} className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="border-b border-zinc-800/50 px-5 py-2.5">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-600" />
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search files or agents..."
              className="w-full rounded-md border border-zinc-800 bg-zinc-900 py-1.5 pl-8 pr-3 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-600 hover:text-zinc-400"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
        </div>

        {/* File list */}
        <div className="flex-1 overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 px-5 py-4 text-xs text-red-400">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}

          {!loading && !error && files.length === 0 && (
            <div className="py-12 text-center text-sm text-zinc-500">No files generated yet</div>
          )}

          {!loading && !error && filtered.length === 0 && files.length > 0 && (
            <div className="py-12 text-center text-sm text-zinc-500">
              No files match your search
              <button onClick={() => setSearchQuery('')} className="ml-2 text-violet-400 hover:underline">
                Clear
              </button>
            </div>
          )}

          {!loading && filtered.length > 0 && (
            <div>
              {Array.from(grouped.entries()).map(([agentId, agentFiles]) => {
                const agentName = agentFiles[0].agentName
                const collapsed = collapsedAgents.has(agentId)

                return (
                  <div key={agentId}>
                    {/* Agent group header */}
                    <button
                      onClick={() => toggleAgent(agentId)}
                      className="flex w-full items-center gap-2 bg-zinc-900/80 px-5 py-2 text-xs font-medium text-zinc-400 hover:bg-zinc-900 sticky top-0 z-10 border-b border-zinc-800/30"
                    >
                      {collapsed
                        ? <ChevronRight className="h-3 w-3" />
                        : <ChevronDown className="h-3 w-3" />}
                      <span className="text-zinc-300">{agentName}</span>
                      <span className="text-zinc-600">({agentFiles.length} file{agentFiles.length !== 1 ? 's' : ''})</span>
                    </button>

                    {/* Files */}
                    {!collapsed && agentFiles.map((f) => (
                      <div
                        key={`${f.agentId}-${f.physical}`}
                        className="flex items-center gap-2 border-b border-zinc-800/20 px-5 py-2 hover:bg-zinc-900/50 transition-colors"
                      >
                        <FileIcon file={f} />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm text-zinc-200">{f.filename}</div>
                          <div className="truncate text-[11px] text-zinc-600 font-mono">{f.logical || f.physical}</div>
                        </div>
                        <span className="w-28 text-right text-[11px] text-zinc-500 shrink-0">{formatDate(f.modified)}</span>
                        <span className="w-16 text-right text-[11px] text-zinc-600 font-mono shrink-0">{f.extension || '--'}</span>
                        <span className="w-16 text-right text-[11px] text-zinc-600 shrink-0">{formatSize(f.size)}</span>
                        {/* Actions */}
                        <div className="w-20 flex items-center justify-end gap-0.5 shrink-0">
                          <button
                            onClick={() => {
                              if (!isFilePinned(f.agentId, f.physical)) {
                                pinFile({
                                  agentId: f.agentId,
                                  agentName: f.agentName,
                                  host: f.host,
                                  port: f.port,
                                  auth: f.auth,
                                  filename: f.filename,
                                  extension: f.extension,
                                  physical: f.physical,
                                  logical: f.logical,
                                  size: f.size,
                                  mime_type: f.mime_type,
                                })
                              }
                            }}
                            className={`rounded p-1 transition-colors ${
                              isFilePinned(f.agentId, f.physical)
                                ? 'text-amber-400'
                                : 'text-zinc-600 hover:bg-zinc-800 hover:text-amber-400'
                            }`}
                            title={isFilePinned(f.agentId, f.physical) ? 'Pinned' : 'Pin file'}
                          >
                            <Pin className="h-3 w-3" />
                          </button>
                          {isViewable(f) && (
                            <button
                              onClick={() => handleView(f)}
                              className="rounded p-1 text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300"
                              title="View"
                            >
                              <Eye className="h-3 w-3" />
                            </button>
                          )}
                          <button
                            onClick={() => handleDownload(f)}
                            className="rounded p-1 text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300"
                            title="Download"
                          >
                            <Download className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-zinc-800 px-5 py-2.5">
          <span className="text-xs text-zinc-500">
            {filtered.length} file{filtered.length !== 1 ? 's' : ''} from {grouped.size} agent{grouped.size !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* File Viewer modal */}
      {viewingFile && (
        <FileViewer
          file={viewingFile}
          host={viewingFile.host}
          port={viewingFile.port}
          auth={viewingFile.auth}
          onClose={() => setViewingFile(null)}
          hasPrev={viewingIndex > 0}
          hasNext={viewingIndex < viewableFiles.length - 1}
          onPrev={() => {
            if (viewingIndex > 0) setViewingFile(viewableFiles[viewingIndex - 1])
          }}
          onNext={() => {
            if (viewingIndex < viewableFiles.length - 1) setViewingFile(viewableFiles[viewingIndex + 1])
          }}
        />
      )}
    </div>
  )
}
