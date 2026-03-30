import { useState } from 'react'
import {
  Pin, X, Tag, FileText, Download, Eye, Copy, Check,
  Trash2, Image, FileCode, FileSpreadsheet, Film, Music, Archive,
} from 'lucide-react'
import { usePinnedFilesStore, type PinnedFile } from '../../stores/pinnedFilesStore'
import { getDownloadUrl, getViewUrl, formatSize, getFileTypeGroup, isViewable } from '../../services/fileTransfer'
import { FileViewer } from '../agents/FileViewer'
import type { AgentFile } from '../../services/fileTransfer'

const TYPE_ICONS: Record<string, typeof FileText> = {
  image: Image, video: Film, audio: Music,
  code: FileCode, data: FileSpreadsheet, archive: Archive,
}

const TYPE_COLORS: Record<string, string> = {
  image: 'text-blue-400', video: 'text-pink-400', audio: 'text-amber-400',
  code: 'text-emerald-400', data: 'text-cyan-400', archive: 'text-orange-400',
  html: 'text-orange-300', markdown: 'text-violet-400', pdf: 'text-red-400',
}

export function PinnedFiles({ onClose }: { onClose: () => void }) {
  const { pins, unpin, updatePin } = usePinnedFilesStore()
  const [filter, setFilter] = useState('')
  const [editingTag, setEditingTag] = useState<string | null>(null)
  const [tagInput, setTagInput] = useState('')
  const [viewingFile, setViewingFile] = useState<{ file: PinnedFile } | null>(null)

  const allTags = [...new Set(pins.flatMap((p) => p.tags))]

  const filtered = filter
    ? pins.filter((p) =>
        p.tags.includes(filter) ||
        p.agentName.toLowerCase().includes(filter.toLowerCase()) ||
        p.filename.toLowerCase().includes(filter.toLowerCase())
      )
    : pins

  const addTag = (pinId: string) => {
    const pin = pins.find((p) => p.id === pinId)
    if (!pin || !tagInput.trim()) return
    const tags = [...new Set([...pin.tags, tagInput.trim().toLowerCase()])]
    updatePin(pinId, { tags })
    setTagInput('')
    setEditingTag(null)
  }

  const removeTag = (pinId: string, tag: string) => {
    const pin = pins.find((p) => p.id === pinId)
    if (!pin) return
    updatePin(pinId, { tags: pin.tags.filter((t) => t !== tag) })
  }

  const handleView = (pin: PinnedFile) => {
    const fakeFile: AgentFile = {
      logical: pin.logical,
      physical: pin.physical,
      filename: pin.filename,
      extension: pin.extension,
      exists: true,
      size: pin.size,
      modified: 0,
      mime_type: pin.mime_type,
      is_text: false,
      source: '',
    }
    const group = getFileTypeGroup(fakeFile)
    if (group === 'pdf') {
      window.open(getViewUrl(pin.host, pin.port, pin.physical, pin.auth), '_blank')
    } else {
      setViewingFile({ file: pin })
    }
  }

  const handleDownload = (pin: PinnedFile) => {
    const url = getDownloadUrl(pin.host, pin.port, pin.physical, pin.auth)
    const a = document.createElement('a')
    a.href = url
    a.download = pin.filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  // Build viewable list for prev/next navigation
  const viewableFiltered = filtered.filter((p) => {
    const fakeFile: AgentFile = { logical: p.logical, physical: p.physical, filename: p.filename, extension: p.extension, exists: true, size: p.size, modified: 0, mime_type: p.mime_type, is_text: false, source: '' }
    return isViewable(fakeFile)
  })
  const viewingIdx = viewingFile ? viewableFiltered.findIndex((p) => p.id === viewingFile.file.id) : -1

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2.5">
        <div className="flex items-center gap-2">
          <Pin className="h-4 w-4 text-blue-400" />
          <span className="text-sm font-semibold">Pinned Files</span>
          <span className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">{pins.length}</span>
        </div>
        <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Tags filter */}
      {allTags.length > 0 && (
        <div className="flex flex-wrap gap-1 border-b border-zinc-800 px-3 py-2">
          <button
            onClick={() => setFilter('')}
            className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${!filter ? 'bg-violet-600/20 text-violet-400' : 'text-zinc-500 hover:text-zinc-300'}`}
          >
            All
          </button>
          {allTags.map((tag) => (
            <button
              key={tag}
              onClick={() => setFilter(filter === tag ? '' : tag)}
              className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${filter === tag ? 'bg-violet-600/20 text-violet-400' : 'text-zinc-500 hover:text-zinc-300'}`}
            >
              #{tag}
            </button>
          ))}
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="px-3 py-8 text-center text-xs text-zinc-600">
            {pins.length === 0 ? 'No pinned files yet. Pin files from the file browser to save them here.' : 'No matches.'}
          </div>
        ) : (
          <div className="divide-y divide-zinc-800/60">
            {filtered.map((pin) => (
              <PinnedFileCard
                key={pin.id}
                pin={pin}
                onUnpin={() => unpin(pin.id)}
                onView={() => handleView(pin)}
                onDownload={() => handleDownload(pin)}
                editingTag={editingTag === pin.id}
                onStartEditTag={() => { setEditingTag(pin.id); setTagInput('') }}
                tagInput={tagInput}
                onTagInputChange={setTagInput}
                onAddTag={() => addTag(pin.id)}
                onRemoveTag={(tag) => removeTag(pin.id, tag)}
                onCancelTag={() => setEditingTag(null)}
              />
            ))}
          </div>
        )}
      </div>

      {/* File Viewer */}
      {viewingFile && (
        <FileViewer
          file={{
            logical: viewingFile.file.logical,
            physical: viewingFile.file.physical,
            filename: viewingFile.file.filename,
            extension: viewingFile.file.extension,
            exists: true,
            size: viewingFile.file.size,
            modified: 0,
            mime_type: viewingFile.file.mime_type,
            is_text: false,
            source: '',
          }}
          host={viewingFile.file.host}
          port={viewingFile.file.port}
          auth={viewingFile.file.auth}
          onClose={() => setViewingFile(null)}
          hasPrev={viewingIdx > 0}
          hasNext={viewingIdx < viewableFiltered.length - 1}
          onPrev={() => {
            if (viewingIdx > 0) setViewingFile({ file: viewableFiltered[viewingIdx - 1] })
          }}
          onNext={() => {
            if (viewingIdx < viewableFiltered.length - 1) setViewingFile({ file: viewableFiltered[viewingIdx + 1] })
          }}
        />
      )}
    </div>
  )
}

function PinnedFileCard({
  pin, onUnpin, onView, onDownload, editingTag, onStartEditTag, tagInput, onTagInputChange, onAddTag, onRemoveTag, onCancelTag,
}: {
  pin: PinnedFile
  onUnpin: () => void
  onView: () => void
  onDownload: () => void
  editingTag: boolean
  onStartEditTag: () => void
  tagInput: string
  onTagInputChange: (v: string) => void
  onAddTag: () => void
  onRemoveTag: (tag: string) => void
  onCancelTag: () => void
}) {
  const [copied, setCopied] = useState(false)

  const fakeFile: AgentFile = { logical: pin.logical, physical: pin.physical, filename: pin.filename, extension: pin.extension, exists: true, size: pin.size, modified: 0, mime_type: pin.mime_type, is_text: false, source: '' }
  const group = getFileTypeGroup(fakeFile)
  const canView = isViewable(fakeFile)

  const Icon = TYPE_ICONS[group] || FileText
  const color = TYPE_COLORS[group] || 'text-zinc-500'

  const copyUrl = () => {
    const url = getViewUrl(pin.host, pin.port, pin.physical, pin.auth)
    navigator.clipboard.writeText(`${window.location.origin}${url}`)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="group px-3 py-2.5 hover:bg-zinc-900/30">
      <div className="flex items-start gap-2">
        <Icon className={`h-4 w-4 mt-0.5 shrink-0 ${color}`} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm text-zinc-200 truncate font-medium">{pin.filename}</span>
            <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
              {canView && (
                <button onClick={onView} className="rounded p-0.5 text-zinc-600 hover:text-zinc-300" title="View">
                  <Eye className="h-3 w-3" />
                </button>
              )}
              <button onClick={onDownload} className="rounded p-0.5 text-zinc-600 hover:text-zinc-300" title="Download">
                <Download className="h-3 w-3" />
              </button>
              <button onClick={copyUrl} className="rounded p-0.5 text-zinc-600 hover:text-zinc-300" title="Copy URL">
                {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
              </button>
              <button onClick={onStartEditTag} className="rounded p-0.5 text-zinc-600 hover:text-zinc-300" title="Tag">
                <Tag className="h-3 w-3" />
              </button>
              <button onClick={onUnpin} className="rounded p-0.5 text-zinc-600 hover:text-red-400" title="Unpin">
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-zinc-500 mt-0.5">
            <span className="font-medium text-zinc-400">{pin.agentName}</span>
            <span>·</span>
            <span className="font-mono">{pin.extension}</span>
            <span>·</span>
            <span>{formatSize(pin.size)}</span>
            <span>·</span>
            <span>{new Date(pin.pinnedAt).toLocaleDateString()}</span>
          </div>
        </div>
      </div>

      {/* Tags */}
      <div className="mt-1.5 flex flex-wrap items-center gap-1 ml-6">
        {pin.tags.map((tag) => (
          <span key={tag} className="group/tag flex items-center gap-0.5 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
            #{tag}
            <button onClick={() => onRemoveTag(tag)} className="opacity-0 group-hover/tag:opacity-100 text-zinc-600 hover:text-red-400">
              <X className="h-2.5 w-2.5" />
            </button>
          </span>
        ))}
        {editingTag && (
          <div className="flex items-center gap-1">
            <input
              value={tagInput}
              onChange={(e) => onTagInputChange(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') onAddTag(); if (e.key === 'Escape') onCancelTag() }}
              placeholder="tag..."
              className="w-16 rounded border border-zinc-700 bg-zinc-950 px-1 py-0.5 text-[10px] text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              autoFocus
            />
          </div>
        )}
      </div>
    </div>
  )
}
