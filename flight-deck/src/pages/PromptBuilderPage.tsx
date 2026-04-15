import { useState, useEffect, useCallback, useRef } from 'react'
import {
  FileText,
  Plus,
  Save,
  Trash2,
  Send,
  Wand2,
  Paperclip,
  X,
  Loader2,
  Tag,
  Clock,
  ChevronDown,
  Copy,
  Check,
  Upload,
  GitBranch,
} from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import { useContainerStore } from '../stores/containerStore'
import { useProcessStore } from '../stores/processStore'
import { useChatStore } from '../stores/chatStore'
import { uploadFileToAgent } from '../services/fileTransfer'
import { useTraceStore } from '../stores/traceStore'

// ── Types ──

interface SavedPrompt {
  id: string
  title: string
  content: string
  files: string // JSON array
  tags: string // JSON array
  created_at: string
  updated_at: string
}

interface AgentTarget {
  id: string
  name: string
  host: string
  port: number
  auth: string
}

// ── API helpers ──

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...(init?.headers as Record<string, string> || {}) }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`

  let res = await fetch(`/fd${path}`, { ...init, headers, credentials: 'include' })
  if (res.status === 401 && authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) {
      const t2 = useAuthStore.getState().token
      if (t2) headers['Authorization'] = `Bearer ${t2}`
      res = await fetch(`/fd${path}`, { ...init, headers, credentials: 'include' })
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

// ── Component ──

export function PromptBuilderPage() {
  // Saved prompts
  const [prompts, setPrompts] = useState<SavedPrompt[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Editor state
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [files, setFiles] = useState<string[]>([])
  const [tags, setTags] = useState<string[]>([])
  const [tagInput, setTagInput] = useState('')
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [rephrasing, setRephrasing] = useState(false)
  const [copied, setCopied] = useState(false)
  const [orchestrate, setOrchestrate] = useState(false)
  const [rephraseOpen, setRephraseOpen] = useState(false)
  const [sendOpen, setSendOpen] = useState(false)
  const [sendStatus, setSendStatus] = useState('')

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Agent targets for "Send to agent"
  const containers = useContainerStore((s) => s.containers)
  const processes = useProcessStore((s) => s.processes)
  const { openChat, sendMessage } = useChatStore()
  const clearTraces = useTraceStore((s) => s.clearTraces)

  const agents: AgentTarget[] = [
    ...containers
      .filter((c) => c.status === 'running' && c.web_port)
      .map((c) => ({
        id: c.id,
        name: c.name || c.id.slice(0, 12),
        host: 'localhost',
        port: c.web_port!,
        auth: c.web_auth || '',
      })),
    ...processes
      .filter((p) => p.status === 'running' && p.web_port)
      .map((p) => ({
        id: p.slug,
        name: p.name || p.slug.slice(0, 12),
        host: 'localhost',
        port: p.web_port!,
        auth: p.web_auth || '',
      })),
  ]

  // ── Load prompts ──

  const fetchPrompts = useCallback(async () => {
    try {
      const data = await fdFetch<SavedPrompt[]>('/prompts')
      setPrompts(data)
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { fetchPrompts() }, [fetchPrompts])

  // ── Select a prompt ──

  const selectPrompt = useCallback((p: SavedPrompt) => {
    setSelectedId(p.id)
    setTitle(p.title)
    setContent(p.content)
    try { setFiles(JSON.parse(p.files)) } catch { setFiles([]) }
    try { setTags(JSON.parse(p.tags)) } catch { setTags([]) }
    setDirty(false)
  }, [])

  const newPrompt = useCallback(() => {
    setSelectedId(null)
    setTitle('')
    setContent('')
    setFiles([])
    setTags([])
    setDirty(false)
    textareaRef.current?.focus()
  }, [])

  // ── Save ──

  const savePrompt = useCallback(async () => {
    setSaving(true)
    try {
      if (selectedId) {
        const updated = await fdFetch<SavedPrompt>(`/prompts/${selectedId}`, {
          method: 'PUT',
          body: JSON.stringify({ title, content, files, tags }),
        })
        setPrompts((prev) => prev.map((p) => p.id === selectedId ? updated : p))
      } else {
        const created = await fdFetch<SavedPrompt>('/prompts', {
          method: 'POST',
          body: JSON.stringify({ title: title || 'Untitled', content, files, tags }),
        })
        setPrompts((prev) => [created, ...prev])
        setSelectedId(created.id)
      }
      setDirty(false)
    } catch { /* ignore */ }
    finally { setSaving(false) }
  }, [selectedId, title, content, files, tags])

  // ── Delete ──

  const deletePrompt = useCallback(async () => {
    if (!selectedId) return
    try {
      await fdFetch(`/prompts/${selectedId}`, { method: 'DELETE' })
      setPrompts((prev) => prev.filter((p) => p.id !== selectedId))
      newPrompt()
    } catch { /* ignore */ }
  }, [selectedId, newPrompt])

  // ── Rephrase ──

  const rephrasePrompt = useCallback(async (agent: AgentTarget) => {
    if (!content.trim()) return
    setRephraseOpen(false)
    setRephrasing(true)
    try {
      const data = await fdFetch<{ content: string }>(
        `/agent-rephrase/${agent.host}/${agent.port}`,
        { method: 'POST', body: JSON.stringify({ content }) },
      )
      if (data.content) {
        setContent(data.content)
        setDirty(true)
      }
    } catch { /* ignore */ }
    finally { setRephrasing(false) }
  }, [content])

  // ── Copy ──

  const copyToClipboard = useCallback(() => {
    let text = content
    if (files.length > 0) {
      text += '\n\n' + files.map((f) => `[Attached file: ${f}]`).join('\n')
    }
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [content, files])

  // ── Send to agent ──

  const sendToAgent = useCallback((agent: AgentTarget) => {
    let msg = content
    if (files.length > 0) {
      const refs = files.map((f) => `[Attached file: ${f}]`).join('\n')
      msg = msg ? `${msg}\n\n${refs}` : refs
    }
    if (orchestrate) {
      msg = `/orchestrate ${msg}`
      clearTraces(agent.id)
    }
    openChat(agent.id, agent.name, agent.host, agent.port, agent.auth)
    // Small delay to let WS connect
    setTimeout(() => {
      sendMessage(agent.id, msg)
    }, 1000)
    setSendOpen(false)
    setSendStatus(`${orchestrate ? 'Orchestrated' : 'Sent'} to ${agent.name}`)
    setTimeout(() => setSendStatus(''), 3000)
  }, [content, files, orchestrate, openChat, sendMessage, clearTraces])

  // ── File upload ──

  const handleFileUpload = useCallback(async (fileList: FileList) => {
    if (agents.length === 0) return
    const agent = agents[0]
    for (const file of Array.from(fileList)) {
      try {
        const result = await uploadFileToAgent(agent.host, agent.port, agent.auth, file)
        setFiles((prev) => [...prev, result.path])
        setDirty(true)
      } catch { /* ignore */ }
    }
  }, [agents])

  // ── Tags ──

  const addTag = useCallback(() => {
    const t = tagInput.trim()
    if (t && !tags.includes(t)) {
      setTags((prev) => [...prev, t])
      setDirty(true)
    }
    setTagInput('')
  }, [tagInput, tags])

  const removeTag = useCallback((tag: string) => {
    setTags((prev) => prev.filter((t) => t !== tag))
    setDirty(true)
  }, [])

  // ── Mark dirty on edits ──

  const onTitleChange = (v: string) => { setTitle(v); setDirty(true) }
  const onContentChange = (v: string) => { setContent(v); setDirty(true) }

  // ── Keyboard shortcuts ──

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 's') {
      e.preventDefault()
      savePrompt()
    }
  }, [savePrompt])

  const selectedPrompt = prompts.find((p) => p.id === selectedId)

  return (
    <div className="flex h-full" onKeyDown={handleKeyDown}>
      {/* ── Left sidebar: saved prompts ── */}
      <div className="flex w-72 shrink-0 flex-col border-r border-zinc-800 bg-zinc-950/50">
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-amber-400" />
            <span className="text-sm font-medium text-zinc-200">Prompts</span>
            <span className="rounded-full bg-zinc-800 px-1.5 text-[10px] text-zinc-500">{prompts.length}</span>
          </div>
          <button
            onClick={newPrompt}
            className="rounded-md bg-amber-600 p-1 text-white hover:bg-amber-500 transition-colors"
            title="New prompt"
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-4 w-4 animate-spin text-zinc-600" />
            </div>
          ) : prompts.length === 0 ? (
            <div className="px-4 py-12 text-center">
              <FileText className="mx-auto h-8 w-8 text-zinc-700 mb-2" />
              <p className="text-xs text-zinc-600">No saved prompts yet</p>
              <p className="text-[10px] text-zinc-700 mt-1">Click + to create one</p>
            </div>
          ) : (
            <div className="p-2 space-y-0.5">
              {prompts.map((p) => {
                const promptTags = (() => { try { return JSON.parse(p.tags) } catch { return [] } })()
                return (
                  <button
                    key={p.id}
                    onClick={() => selectPrompt(p)}
                    className={`group flex w-full flex-col rounded-lg px-3 py-2.5 text-left transition-colors ${
                      selectedId === p.id
                        ? 'bg-amber-500/10 border border-amber-500/20'
                        : 'hover:bg-zinc-800/50 border border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className={`text-sm font-medium truncate ${selectedId === p.id ? 'text-zinc-100' : 'text-zinc-300'}`}>
                        {p.title || 'Untitled'}
                      </span>
                    </div>
                    <div className="text-[10px] text-zinc-600 mt-1 truncate">
                      {p.content.slice(0, 80) || 'Empty prompt'}
                    </div>
                    {promptTags.length > 0 && (
                      <div className="flex gap-1 mt-1.5 flex-wrap">
                        {promptTags.slice(0, 3).map((tag: string) => (
                          <span key={tag} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[9px] text-zinc-500">{tag}</span>
                        ))}
                      </div>
                    )}
                    <div className="flex items-center gap-1 mt-1.5 text-[9px] text-zinc-600">
                      <Clock className="h-2.5 w-2.5" />
                      {new Date(p.updated_at).toLocaleDateString()}
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Main editor ── */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Toolbar */}
        <div className="flex items-center gap-2 border-b border-zinc-800 px-5 py-2.5 bg-zinc-900/50">
          <div className="flex items-center gap-2 flex-1">
            <button
              onClick={savePrompt}
              disabled={saving || !dirty}
              className="flex items-center gap-1.5 rounded-lg bg-amber-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-amber-500 disabled:opacity-40 disabled:hover:bg-amber-600 transition-colors"
            >
              {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
              Save
            </button>

            <div className="relative">
              <button
                onClick={() => setRephraseOpen(!rephraseOpen)}
                disabled={rephrasing || !content.trim() || agents.length === 0}
                className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40 transition-colors"
                title={agents.length === 0 ? 'Connect an agent to use rephrase' : 'Rephrase with AI'}
              >
                {rephrasing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Wand2 className="h-3 w-3" />}
                Rephrase
                <ChevronDown className="h-3 w-3" />
              </button>
              {rephraseOpen && agents.length > 0 && (
                <>
                  <div className="fixed inset-0 z-20" onClick={() => setRephraseOpen(false)} />
                  <div className="absolute left-0 top-full mt-1 z-30 rounded-xl border border-zinc-700 bg-zinc-800 shadow-2xl py-1 min-w-[200px] max-h-[300px] overflow-y-auto">
                    {agents.map((agent) => (
                      <button
                        key={agent.id}
                        onClick={() => rephrasePrompt(agent)}
                        className="flex items-center gap-2 w-full text-left px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-colors"
                      >
                        <div className="h-1.5 w-1.5 rounded-full bg-violet-400 shrink-0" />
                        <span className="truncate">{agent.name}</span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>

            <button
              onClick={copyToClipboard}
              disabled={!content.trim()}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-40 transition-colors"
            >
              {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
              {copied ? 'Copied' : 'Copy'}
            </button>

            {selectedId && (
              <button
                onClick={deletePrompt}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/10 hover:border-red-500/30 transition-colors"
              >
                <Trash2 className="h-3 w-3" />
                Delete
              </button>
            )}
          </div>

          {/* Orchestrate toggle */}
          <button
            onClick={() => setOrchestrate(!orchestrate)}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              orchestrate
                ? 'bg-violet-600 text-white'
                : 'border border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
            }`}
            title={orchestrate ? 'Will decompose into parallel sub-tasks' : 'Will send as a single message'}
          >
            <GitBranch className="h-3 w-3" />
            Orchestrate
          </button>

          {/* Send to agent */}
          <div className="relative">
            <button
              onClick={() => setSendOpen(!sendOpen)}
              disabled={!content.trim() || agents.length === 0}
              className="flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-40 transition-colors"
            >
              <Send className="h-3 w-3" />
              Send to Agent
              <ChevronDown className="h-3 w-3" />
            </button>
            {sendOpen && agents.length > 0 && (
              <>
                <div className="fixed inset-0 z-20" onClick={() => setSendOpen(false)} />
                <div className="absolute right-0 top-full mt-1 z-30 rounded-xl border border-zinc-700 bg-zinc-800 shadow-2xl py-1 min-w-[200px] max-h-[300px] overflow-y-auto">
                  {agents.map((agent) => (
                    <button
                      key={agent.id}
                      onClick={() => sendToAgent(agent)}
                      className="flex items-center gap-2 w-full text-left px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-colors"
                    >
                      <div className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                      <span className="truncate">{agent.name}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          {sendStatus && (
            <span className="text-[11px] text-emerald-400 animate-pulse">{sendStatus}</span>
          )}
        </div>

        {/* Editor body */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Title */}
          <div className="mx-auto w-full max-w-4xl shrink-0 px-8 pt-6">
            <input
              value={title}
              onChange={(e) => onTitleChange(e.target.value)}
              placeholder="Prompt title..."
              className="block w-full bg-transparent text-2xl font-semibold text-zinc-100 placeholder-zinc-600 outline-none border-none"
            />
          </div>

          {/* Content — fills remaining space */}
          <div className="flex-1 overflow-y-auto min-h-0">
            <div className="mx-auto max-w-4xl px-8 py-4 h-full">
              <textarea
                ref={textareaRef}
                value={content}
                onChange={(e) => onContentChange(e.target.value)}
                placeholder="Write your prompt here...&#10;&#10;Be specific about what you want the agent to do. Include context, constraints, and expected output format."
                className="block w-full h-full resize-none bg-transparent text-sm text-zinc-300 placeholder-zinc-700 outline-none border-none leading-relaxed font-mono"
              />
            </div>
          </div>

          {/* Bottom panels — attachments, tags, meta */}
          <div className="shrink-0 border-t border-zinc-800 bg-zinc-950/30">
            <div className="mx-auto max-w-4xl px-8 py-3">
              {/* Attachments */}
              <div className="flex items-center gap-2">
                <Paperclip className="h-3.5 w-3.5 text-zinc-500" />
                <span className="text-xs font-medium text-zinc-400">Attachments</span>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={agents.length === 0}
                  className="flex items-center gap-1 rounded-md border border-dashed border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-500 hover:border-zinc-500 hover:text-zinc-300 disabled:opacity-40 transition-colors"
                  title={agents.length === 0 ? 'Connect an agent to upload files' : 'Upload file'}
                >
                  <Upload className="h-3 w-3" />
                  Upload
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  className="hidden"
                  onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
                />
                {files.length > 0 && (
                  <div className="flex items-center gap-1 ml-2 flex-1 overflow-x-auto">
                    {files.map((f, i) => (
                      <span key={i} className="flex items-center gap-1 rounded bg-zinc-800/50 px-2 py-1 text-[10px] text-zinc-400 font-mono shrink-0 group">
                        {f.split('/').pop()}
                        <button
                          onClick={() => { setFiles((prev) => prev.filter((_, j) => j !== i)); setDirty(true) }}
                          className="text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                        >
                          <X className="h-2.5 w-2.5" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}
                {files.length === 0 && <span className="text-[10px] text-zinc-700 ml-1">None</span>}
              </div>

              {/* Tags */}
              <div className="flex items-center gap-2 mt-2">
                <Tag className="h-3.5 w-3.5 text-zinc-500" />
                <span className="text-xs font-medium text-zinc-400">Tags</span>
                <div className="flex flex-wrap items-center gap-1 ml-1">
                  {tags.map((tag) => (
                    <span
                      key={tag}
                      className="group flex items-center gap-1 rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400"
                    >
                      {tag}
                      <button
                        onClick={() => removeTag(tag)}
                        className="text-zinc-600 hover:text-red-400 transition-colors"
                      >
                        <X className="h-2.5 w-2.5" />
                      </button>
                    </span>
                  ))}
                  <input
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addTag() } }}
                    placeholder="Add tag..."
                    className="rounded-full bg-transparent px-2 py-0.5 text-[10px] text-zinc-400 placeholder-zinc-700 outline-none w-20"
                  />
                </div>
              </div>

              {/* Meta info */}
              {selectedPrompt && (
                <div className="mt-2 pt-2 border-t border-zinc-800/50 text-[10px] text-zinc-600 flex items-center gap-4">
                  <span>Created: {new Date(selectedPrompt.created_at).toLocaleString()}</span>
                  <span>Updated: {new Date(selectedPrompt.updated_at).toLocaleString()}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
