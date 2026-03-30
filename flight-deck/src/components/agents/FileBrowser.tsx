import { useState, useEffect, useMemo } from 'react'
import {
  X, FileText, Send, Loader2, Check, AlertCircle, FolderOpen, RefreshCw,
  Download, Eye, Search, ChevronDown, ChevronRight, Image, FileCode,
  FileSpreadsheet, Film, Music, Archive, Filter, Pin,
} from 'lucide-react'
import type { AgentFile, AgentEndpoint } from '../../services/fileTransfer'
import {
  listAgentFiles, transferFile, formatSize, getDownloadUrl, getViewUrl,
  getFileCategory, getFileTypeGroup, isViewable,
} from '../../services/fileTransfer'
import { FileViewer } from './FileViewer'
import { usePinnedFilesStore } from '../../stores/pinnedFilesStore'

interface FileBrowserProps {
  agent: AgentEndpoint
  allAgents: AgentEndpoint[]
  onClose: () => void
}

type SortField = 'name' | 'date' | 'type' | 'size'
type SortDir = 'asc' | 'desc'

const TYPE_ICONS: Record<string, typeof FileText> = {
  image: Image,
  video: Film,
  audio: Music,
  code: FileCode,
  data: FileSpreadsheet,
  archive: Archive,
}

const TYPE_LABELS: Record<string, string> = {
  image: 'Images',
  video: 'Videos',
  audio: 'Audio',
  pdf: 'PDF',
  html: 'HTML',
  markdown: 'Markdown',
  data: 'Data',
  code: 'Code',
  text: 'Text',
  archive: 'Archives',
  document: 'Documents',
  other: 'Other',
}

function formatDate(ts: number): string {
  if (!ts) return '--'
  const d = new Date(ts * 1000)
  const now = new Date()
  const isToday = d.toDateString() === now.toDateString()
  const yesterday = new Date(now)
  yesterday.setDate(yesterday.getDate() - 1)
  const isYesterday = d.toDateString() === yesterday.toDateString()

  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  if (isToday) return `Today ${time}`
  if (isYesterday) return `Yesterday ${time}`
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ` ${time}`
}

export function FileBrowser({ agent, allAgents, onClose }: FileBrowserProps) {
  const [files, setFiles] = useState<AgentFile[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [sendingTo, setSendingTo] = useState<string | null>(null)
  const [transferStatus, setTransferStatus] = useState<Record<string, 'sending' | 'done' | 'error'>>({})
  const [resultMessage, setResultMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null)

  // Filters & sorting
  const [searchQuery, setSearchQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState<string>('')
  const [sortField, setSortField] = useState<SortField>('date')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [groupByFolder, setGroupByFolder] = useState(true)
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())
  const [showFilters, setShowFilters] = useState(false)
  const [viewingFile, setViewingFile] = useState<AgentFile | null>(null)

  const { pin: pinFile, isPinned: isFilePinned } = usePinnedFilesStore()
  const otherAgents = allAgents.filter((a) => a.id !== agent.id)

  const loadFiles = async () => {
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

  useEffect(() => { loadFiles() }, [agent.id])

  // Available type groups from current files
  const availableTypes = useMemo(() => {
    const types = new Set<string>()
    files.forEach((f) => types.add(getFileTypeGroup(f)))
    return Array.from(types).sort()
  }, [files])

  // Filtered and sorted files
  const processedFiles = useMemo(() => {
    let result = [...files]

    // Search
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      result = result.filter((f) =>
        f.filename.toLowerCase().includes(q) ||
        f.logical.toLowerCase().includes(q) ||
        f.extension.toLowerCase().includes(q)
      )
    }

    // Type filter
    if (typeFilter) {
      result = result.filter((f) => getFileTypeGroup(f) === typeFilter)
    }

    // Sort
    result.sort((a, b) => {
      let cmp = 0
      switch (sortField) {
        case 'name': cmp = a.filename.localeCompare(b.filename); break
        case 'date': cmp = a.modified - b.modified; break
        case 'type': cmp = a.extension.localeCompare(b.extension); break
        case 'size': cmp = a.size - b.size; break
      }
      return sortDir === 'asc' ? cmp : -cmp
    })

    return result
  }, [files, searchQuery, typeFilter, sortField, sortDir])

  // Grouped files
  const groupedFiles = useMemo(() => {
    if (!groupByFolder) return new Map([['all', processedFiles]])
    const groups = new Map<string, AgentFile[]>()
    for (const f of processedFiles) {
      const cat = getFileCategory(f)
      if (!groups.has(cat)) groups.set(cat, [])
      groups.get(cat)!.push(f)
    }
    return groups
  }, [processedFiles, groupByFolder])

  const toggleSelect = (path: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  const selectAll = () => {
    if (selected.size === processedFiles.length) setSelected(new Set())
    else setSelected(new Set(processedFiles.map((f) => f.physical)))
  }

  const toggleGroup = (group: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(group)) next.delete(group)
      else next.add(group)
      return next
    })
  }

  const handleSort = (field: SortField) => {
    if (sortField === field) setSortDir((d) => d === 'asc' ? 'desc' : 'asc')
    else { setSortField(field); setSortDir(field === 'name' ? 'asc' : 'desc') }
  }

  const handleSend = async (dst: AgentEndpoint) => {
    if (selected.size === 0) return
    setSendingTo(dst.id)
    setResultMessage(null)
    const paths = Array.from(selected)
    let ok = 0, fail = 0

    for (const path of paths) {
      setTransferStatus((prev) => ({ ...prev, [path]: 'sending' }))
      try {
        await transferFile(agent, dst, path)
        setTransferStatus((prev) => ({ ...prev, [path]: 'done' }))
        ok++
      } catch {
        setTransferStatus((prev) => ({ ...prev, [path]: 'error' }))
        fail++
      }
    }
    setSendingTo(null)
    if (fail === 0) setResultMessage({ text: `Sent ${ok} file${ok > 1 ? 's' : ''} to ${dst.name}`, type: 'success' })
    else setResultMessage({ text: `${ok} sent, ${fail} failed`, type: 'error' })
  }

  const handleDownloadSelected = () => {
    for (const path of selected) {
      const url = getDownloadUrl(agent.host, agent.port, path, agent.auth)
      const a = document.createElement('a')
      a.href = url
      a.download = path.split('/').pop() || 'file'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    }
  }

  const handleView = (f: AgentFile) => {
    const group = getFileTypeGroup(f)
    if (group === 'pdf') {
      // PDF opens in a new tab
      window.open(getViewUrl(agent.host, agent.port, f.physical, agent.auth), '_blank')
    } else {
      setViewingFile(f)
    }
  }

  // Navigation for the viewer: prev/next viewable file in processedFiles
  const viewableFiles = processedFiles.filter((f) => isViewable(f))
  const viewingViewableIndex = viewingFile ? viewableFiles.findIndex((f) => f.physical === viewingFile.physical) : -1

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
            <h2 className="text-sm font-semibold">Files — {agent.name}</h2>
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-xs text-zinc-500 font-mono">{files.length}</span>
          </div>
          <div className="flex items-center gap-1">
            <button onClick={loadFiles} className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Refresh">
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button onClick={onClose} className="rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Search & filters bar */}
        <div className="border-b border-zinc-800/50 px-5 py-2.5 space-y-2">
          <div className="flex items-center gap-2">
            {/* Search */}
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-600" />
              <input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search files..."
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

            {/* Filter toggle */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`flex items-center gap-1 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
                showFilters || typeFilter ? 'bg-violet-600/20 text-violet-300' : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
              }`}
            >
              <Filter className="h-3 w-3" />
              Filter
              {typeFilter && <span className="rounded bg-violet-500/30 px-1 text-[10px]">1</span>}
            </button>

            {/* Group toggle */}
            <button
              onClick={() => setGroupByFolder(!groupByFolder)}
              className={`flex items-center gap-1 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
                groupByFolder ? 'bg-violet-600/20 text-violet-300' : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
              }`}
            >
              <FolderOpen className="h-3 w-3" />
              Group
            </button>
          </div>

          {/* Filter row */}
          {showFilters && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-zinc-500">Type:</span>
              <button
                onClick={() => setTypeFilter('')}
                className={`rounded-md px-2 py-0.5 text-xs font-medium transition-colors ${
                  !typeFilter ? 'bg-violet-600/30 text-violet-300' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
                }`}
              >
                All
              </button>
              {availableTypes.map((t) => (
                <button
                  key={t}
                  onClick={() => setTypeFilter(typeFilter === t ? '' : t)}
                  className={`rounded-md px-2 py-0.5 text-xs font-medium transition-colors ${
                    typeFilter === t ? 'bg-violet-600/30 text-violet-300' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
                  }`}
                >
                  {TYPE_LABELS[t] || t}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Column headers */}
        {!loading && files.length > 0 && (
          <div className="flex items-center gap-2 border-b border-zinc-800/50 px-5 py-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-600">
            <input
              type="checkbox"
              checked={selected.size === processedFiles.length && processedFiles.length > 0}
              onChange={selectAll}
              className="rounded border-zinc-600 mr-1"
            />
            <SortHeader label="Name" field="name" current={sortField} dir={sortDir} onClick={handleSort} className="flex-1" />
            <SortHeader label="Date" field="date" current={sortField} dir={sortDir} onClick={handleSort} className="w-28 text-right" />
            <SortHeader label="Type" field="type" current={sortField} dir={sortDir} onClick={handleSort} className="w-16 text-right" />
            <SortHeader label="Size" field="size" current={sortField} dir={sortDir} onClick={handleSort} className="w-16 text-right" />
            <div className="w-20" /> {/* actions column */}
          </div>
        )}

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
            <div className="py-12 text-center text-sm text-zinc-500">No files found</div>
          )}

          {!loading && !error && processedFiles.length === 0 && files.length > 0 && (
            <div className="py-12 text-center text-sm text-zinc-500">
              No files match your filters
              <button onClick={() => { setSearchQuery(''); setTypeFilter('') }} className="ml-2 text-violet-400 hover:underline">
                Clear filters
              </button>
            </div>
          )}

          {!loading && processedFiles.length > 0 && (
            <div>
              {Array.from(groupedFiles.entries()).map(([group, groupFiles]) => (
                <div key={group}>
                  {/* Group header */}
                  {groupByFolder && group !== 'all' && (
                    <button
                      onClick={() => toggleGroup(group)}
                      className="flex w-full items-center gap-2 bg-zinc-900/80 px-5 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-900 sticky top-0 z-10"
                    >
                      {collapsedGroups.has(group)
                        ? <ChevronRight className="h-3 w-3" />
                        : <ChevronDown className="h-3 w-3" />}
                      <FolderOpen className="h-3 w-3 text-zinc-500" />
                      <span className="capitalize">{group}</span>
                      <span className="text-zinc-600">({groupFiles.length})</span>
                    </button>
                  )}

                  {/* Files in group */}
                  {(!collapsedGroups.has(group) || group === 'all') && groupFiles.map((f) => (
                    <div
                      key={f.physical}
                      className={`flex items-center gap-2 border-b border-zinc-800/20 px-5 py-2 hover:bg-zinc-900/50 transition-colors ${
                        selected.has(f.physical) ? 'bg-violet-500/5' : ''
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={selected.has(f.physical)}
                        onChange={() => toggleSelect(f.physical)}
                        className="rounded border-zinc-600 mr-1"
                      />
                      <FileIcon file={f} />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm text-zinc-200">{f.filename}</div>
                        <div className="truncate text-[11px] text-zinc-600 font-mono">{f.logical || f.physical}</div>
                      </div>
                      {/* Transfer status */}
                      {transferStatus[f.physical] === 'sending' && <Loader2 className="h-3 w-3 animate-spin text-violet-400" />}
                      {transferStatus[f.physical] === 'done' && <Check className="h-3 w-3 text-emerald-400" />}
                      {transferStatus[f.physical] === 'error' && <AlertCircle className="h-3 w-3 text-red-400" />}
                      {/* Date */}
                      <span className="w-28 text-right text-[11px] text-zinc-500 shrink-0">{formatDate(f.modified)}</span>
                      {/* Extension */}
                      <span className="w-16 text-right text-[11px] text-zinc-600 font-mono shrink-0">
                        {f.extension || '--'}
                      </span>
                      {/* Size */}
                      <span className="w-16 text-right text-[11px] text-zinc-600 shrink-0">{formatSize(f.size)}</span>
                      {/* Actions */}
                      <div className="w-20 flex items-center justify-end gap-0.5 shrink-0">
                        <button
                          onClick={() => {
                            if (!isFilePinned(agent.id, f.physical)) {
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
                          }}
                          className={`rounded p-1 transition-colors ${
                            isFilePinned(agent.id, f.physical)
                              ? 'text-amber-400'
                              : 'text-zinc-600 hover:bg-zinc-800 hover:text-amber-400'
                          }`}
                          title={isFilePinned(agent.id, f.physical) ? 'Pinned' : 'Pin file'}
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
                          onClick={() => {
                            const a = document.createElement('a')
                            a.href = getDownloadUrl(agent.host, agent.port, f.physical, agent.auth)
                            a.download = f.filename
                            document.body.appendChild(a)
                            a.click()
                            document.body.removeChild(a)
                          }}
                          className="rounded p-1 text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300"
                          title="Download"
                        >
                          <Download className="h-3 w-3" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Result message */}
        {resultMessage && (
          <div className={`border-t px-5 py-2 flex items-center gap-2 ${
            resultMessage.type === 'success'
              ? 'border-emerald-500/20 bg-emerald-500/10'
              : 'border-red-500/20 bg-red-500/10'
          }`}>
            {resultMessage.type === 'success'
              ? <Check className="h-3.5 w-3.5 text-emerald-400" />
              : <AlertCircle className="h-3.5 w-3.5 text-red-400" />}
            <span className={`text-xs font-medium ${
              resultMessage.type === 'success' ? 'text-emerald-300' : 'text-red-300'
            }`}>{resultMessage.text}</span>
          </div>
        )}

        {/* Bottom action bar */}
        <div className="border-t border-zinc-800 px-5 py-2.5">
          <div className="flex items-center gap-3">
            {/* Selection info */}
            <span className="text-xs text-zinc-500">
              {selected.size > 0
                ? `${selected.size} selected`
                : `${processedFiles.length} file${processedFiles.length !== 1 ? 's' : ''}`
              }
            </span>

            <div className="flex-1" />

            {/* Download selected */}
            {selected.size > 0 && (
              <button
                onClick={handleDownloadSelected}
                className="flex items-center gap-1.5 rounded-md bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-700"
              >
                <Download className="h-3 w-3" />
                Download {selected.size > 1 ? `(${selected.size})` : ''}
              </button>
            )}

            {/* Send to other agents */}
            {selected.size > 0 && otherAgents.length > 0 && (
              <>
                <span className="text-xs text-zinc-600">|</span>
                <Send className="h-3 w-3 text-zinc-500" />
                <div className="flex flex-wrap gap-1">
                  {otherAgents.map((dst) => (
                    <button
                      key={dst.id}
                      onClick={() => handleSend(dst)}
                      disabled={sendingTo !== null}
                      className="flex items-center gap-1 rounded-md bg-violet-600/20 px-2.5 py-1 text-xs font-medium text-violet-300 hover:bg-violet-600/30 disabled:opacity-40"
                    >
                      {sendingTo === dst.id && <Loader2 className="h-3 w-3 animate-spin" />}
                      {dst.name}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* File Viewer modal */}
      {viewingFile && (
        <FileViewer
          file={viewingFile}
          host={agent.host}
          port={agent.port}
          auth={agent.auth}
          onClose={() => setViewingFile(null)}
          hasPrev={viewingViewableIndex > 0}
          hasNext={viewingViewableIndex < viewableFiles.length - 1}
          onPrev={() => {
            if (viewingViewableIndex > 0) setViewingFile(viewableFiles[viewingViewableIndex - 1])
          }}
          onNext={() => {
            if (viewingViewableIndex < viewableFiles.length - 1) setViewingFile(viewableFiles[viewingViewableIndex + 1])
          }}
        />
      )}
    </div>
  )
}

function SortHeader({ label, field, current, dir, onClick, className }: {
  label: string
  field: SortField
  current: SortField
  dir: SortDir
  onClick: (f: SortField) => void
  className?: string
}) {
  const active = current === field
  return (
    <button
      onClick={() => onClick(field)}
      className={`flex items-center gap-0.5 hover:text-zinc-400 transition-colors ${active ? 'text-zinc-300' : ''} ${className || ''}`}
    >
      {label}
      {active && <span className="text-violet-400">{dir === 'asc' ? '\u2191' : '\u2193'}</span>}
    </button>
  )
}
