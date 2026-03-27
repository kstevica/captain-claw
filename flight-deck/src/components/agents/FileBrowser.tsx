import { useState, useEffect } from 'react'
import {
  X, FileText, Send, Loader2, Check, AlertCircle, FolderOpen, RefreshCw,
} from 'lucide-react'
import type { AgentFile, AgentEndpoint } from '../../services/fileTransfer'
import { listAgentFiles, transferFile, formatSize } from '../../services/fileTransfer'

interface FileBrowserProps {
  agent: AgentEndpoint
  allAgents: AgentEndpoint[]
  onClose: () => void
}

export function FileBrowser({ agent, allAgents, onClose }: FileBrowserProps) {
  const [files, setFiles] = useState<AgentFile[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [sendingTo, setSendingTo] = useState<string | null>(null)
  const [transferStatus, setTransferStatus] = useState<Record<string, 'sending' | 'done' | 'error'>>({})
  const [resultMessage, setResultMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null)

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

  const toggleSelect = (path: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  const selectAll = () => {
    if (selected.size === files.length) setSelected(new Set())
    else setSelected(new Set(files.map((f) => f.physical)))
  }

  const handleSend = async (dst: AgentEndpoint) => {
    if (selected.size === 0) return
    setSendingTo(dst.id)
    setResultMessage(null)
    const paths = Array.from(selected)
    let ok = 0
    let fail = 0

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
    if (fail === 0) {
      setResultMessage({ text: `Sent ${ok} file${ok > 1 ? 's' : ''} to ${dst.name}`, type: 'success' })
    } else {
      setResultMessage({ text: `${ok} sent, ${fail} failed`, type: 'error' })
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="flex max-h-[80vh] w-[600px] flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div className="flex items-center gap-2">
            <FolderOpen className="h-4 w-4 text-zinc-400" />
            <h2 className="text-sm font-semibold">Files — {agent.name}</h2>
            <span className="text-xs text-zinc-500">{agent.host}:{agent.port}</span>
          </div>
          <div className="flex items-center gap-1">
            <button onClick={loadFiles} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
            <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
              <X className="h-4 w-4" />
            </button>
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
            <div className="py-12 text-center text-sm text-zinc-500">No files found</div>
          )}

          {!loading && files.length > 0 && (
            <div>
              <div className="flex items-center gap-2 border-b border-zinc-800/50 px-5 py-2">
                <input
                  type="checkbox"
                  checked={selected.size === files.length && files.length > 0}
                  onChange={selectAll}
                  className="rounded border-zinc-600"
                />
                <span className="text-xs text-zinc-500">
                  {selected.size > 0 ? `${selected.size} selected` : `${files.length} files`}
                </span>
              </div>
              {files.map((f) => (
                <label
                  key={f.physical}
                  className="flex cursor-pointer items-center gap-3 border-b border-zinc-800/30 px-5 py-2 hover:bg-zinc-900/50"
                >
                  <input
                    type="checkbox"
                    checked={selected.has(f.physical)}
                    onChange={() => toggleSelect(f.physical)}
                    className="rounded border-zinc-600"
                  />
                  <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm text-zinc-200">{f.filename}</div>
                    <div className="truncate text-xs text-zinc-600 font-mono">{f.logical || f.physical}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {transferStatus[f.physical] === 'sending' && <Loader2 className="h-3 w-3 animate-spin text-violet-400" />}
                    {transferStatus[f.physical] === 'done' && <Check className="h-3 w-3 text-emerald-400" />}
                    {transferStatus[f.physical] === 'error' && <AlertCircle className="h-3 w-3 text-red-400" />}
                    <span className="text-xs text-zinc-600">{formatSize(f.size)}</span>
                  </div>
                </label>
              ))}
            </div>
          )}
        </div>

        {/* Result message */}
        {resultMessage && (
          <div className={`border-t px-5 py-2.5 flex items-center gap-2 ${
            resultMessage.type === 'success'
              ? 'border-emerald-500/20 bg-emerald-500/10'
              : 'border-red-500/20 bg-red-500/10'
          }`}>
            {resultMessage.type === 'success'
              ? <Check className="h-4 w-4 text-emerald-400" />
              : <AlertCircle className="h-4 w-4 text-red-400" />}
            <span className={`text-xs font-medium ${
              resultMessage.type === 'success' ? 'text-emerald-300' : 'text-red-300'
            }`}>{resultMessage.text}</span>
          </div>
        )}

        {/* Send bar */}
        {selected.size > 0 && otherAgents.length > 0 && (
          <div className="border-t border-zinc-800 px-5 py-3">
            <div className="flex items-center gap-2">
              <Send className="h-3.5 w-3.5 text-zinc-400" />
              <span className="text-xs text-zinc-400">Send to:</span>
              <div className="flex flex-wrap gap-1.5">
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
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
