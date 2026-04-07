import { useState } from 'react'
import { Pin, X, Tag, MessageSquare, Copy, Check, ChevronDown, ChevronRight, Trash2 } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { usePinnedStore, type PinnedMessage } from '../../stores/pinnedStore'

export function PinnedMessages({ onClose }: { onClose: () => void }) {
  const { pins, unpin, updatePin } = usePinnedStore()
  const [filter, setFilter] = useState('')
  const [editingTag, setEditingTag] = useState<string | null>(null)
  const [tagInput, setTagInput] = useState('')

  const allTags = [...new Set(pins.flatMap((p) => p.tags))]

  const filtered = filter
    ? pins.filter((p) => p.tags.includes(filter) || p.agentName.toLowerCase().includes(filter.toLowerCase()) || p.content.toLowerCase().includes(filter.toLowerCase()))
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

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2.5">
        <div className="flex items-center gap-2">
          <Pin className="h-4 w-4 text-amber-400" />
          <span className="text-sm font-semibold">Pinned Messages</span>
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
            {pins.length === 0 ? 'No pinned messages yet. Pin messages from chat to save them here.' : 'No matches.'}
          </div>
        ) : (
          <div className="divide-y divide-zinc-800/60">
            {filtered.map((pin) => (
              <PinnedCard
                key={pin.id}
                pin={pin}
                onUnpin={() => unpin(pin.id)}
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
    </div>
  )
}

function PinnedCard({
  pin, onUnpin, editingTag, onStartEditTag, tagInput, onTagInputChange, onAddTag, onRemoveTag, onCancelTag,
}: {
  pin: PinnedMessage
  onUnpin: () => void
  editingTag: boolean
  onStartEditTag: () => void
  tagInput: string
  onTagInputChange: (v: string) => void
  onAddTag: () => void
  onRemoveTag: (tag: string) => void
  onCancelTag: () => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(pin.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const preview = pin.content.length > 200 ? pin.content.slice(0, 200) + '...' : pin.content

  return (
    <div className="group px-3 py-2.5 hover:bg-zinc-900/30">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 text-[10px] text-zinc-500">
          <MessageSquare className="h-3 w-3" />
          <span className="font-medium text-zinc-400">{pin.agentName}</span>
          <span>·</span>
          <span>{new Date(pin.pinnedAt).toLocaleDateString()}</span>
          {pin.model && <><span>·</span><span className="font-mono">{pin.model}</span></>}
        </div>
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <button onClick={copy} className="rounded p-0.5 text-zinc-600 hover:text-zinc-300" title="Copy">
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

      {/* Content */}
      <div className="mt-1">
        {expanded ? (
          <div className="fd-markdown prose-xs text-xs text-zinc-300">
            <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{pin.content}</Markdown>
          </div>
        ) : (
          <p className="text-xs text-zinc-400 cursor-pointer" onClick={() => setExpanded(true)}>
            {preview}
          </p>
        )}
        {pin.content.length > 200 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-0.5 flex items-center gap-0.5 text-[10px] text-violet-400 hover:text-violet-300"
          >
            {expanded ? <ChevronDown className="h-2.5 w-2.5" /> : <ChevronRight className="h-2.5 w-2.5" />}
            {expanded ? 'Collapse' : 'Expand'}
          </button>
        )}
      </div>

      {/* Tags */}
      <div className="mt-1.5 flex flex-wrap items-center gap-1">
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
