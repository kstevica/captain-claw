import { useState, useMemo } from 'react'
import { X, Send, ChevronDown, ChevronUp, MessageSquare } from 'lucide-react'
import type { ChatMessage } from '../../services/agentChat'
import { useChatStore } from '../../stores/chatStore'
import { useContainerStore } from '../../stores/containerStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'

interface AgentTarget {
  id: string
  name: string
  host: string
  port: number
  auth: string
}

interface SendContextModalProps {
  sourceId: string
  sourceName: string
  messages: ChatMessage[]
  targets: AgentTarget[]
  onClose: () => void
}

export function SendContextModal({ sourceId: _sourceId, sourceName, messages, targets, onClose }: SendContextModalProps) {
  // Only user + assistant messages are useful as context
  const contextMessages = useMemo(
    () => messages.filter((m) => m.role === 'user' || m.role === 'assistant'),
    [messages],
  )

  const [messageCount, setMessageCount] = useState(Math.min(10, contextMessages.length))
  const [task, setTask] = useState('')
  const [taskTouched, setTaskTouched] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [sending, setSending] = useState(false)
  const [sent, setSent] = useState<string | null>(null)

  const openChat = useChatStore((s) => s.openChat)
  const sendMessage = useChatStore((s) => s.sendMessage)
  const getContainerFwdTask = useContainerStore((s) => s.getForwardingTask)
  const localAgents = useLocalAgentStore((s) => s.agents)

  // Look up forwarding task for a given target
  const getTargetFwdTask = (targetId: string): string => {
    const containerTask = getContainerFwdTask(targetId)
    if (containerTask) return containerTask
    const localAgent = localAgents.find((a) => a.id === targetId)
    return localAgent?.forwardingTask || ''
  }

  const selectedMessages = contextMessages.slice(-messageCount)

  const formatContext = (): string => {
    const lines: string[] = []
    lines.push(`--- Context transferred from another AI agent "${sourceName}" (last ${selectedMessages.length} messages from their conversation) ---`)
    lines.push('')
    for (const m of selectedMessages) {
      const role = m.role === 'user' ? 'User' : 'Assistant'
      const content = m.content
      lines.push(`**${role}:**`)
      lines.push(content)
      lines.push('')
    }
    lines.push('--- End of context ---')
    lines.push('')
    lines.push(task)
    return lines.join('\n')
  }

  const handleSend = async (target: AgentTarget) => {
    if (!task.trim()) return
    setSending(true)

    // Ensure chat session exists for the target
    openChat(target.id, target.name, target.host, target.port, target.auth)

    // Wait briefly for WebSocket connection to establish
    await new Promise((r) => setTimeout(r, 1500))

    const composed = formatContext()
    sendMessage(target.id, composed)

    setSending(false)
    setSent(target.name)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="flex max-h-[85vh] w-[560px] flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div className="flex items-center gap-2">
            <Send className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-semibold">Send Context to Agent</h2>
          </div>
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Source info */}
          <div className="text-xs text-zinc-500">
            From: <span className="font-medium text-zinc-300">{sourceName}</span>
            <span className="ml-2 text-zinc-600">({contextMessages.length} messages available)</span>
          </div>

          {/* Message count slider */}
          <div>
            <label className="mb-2 flex items-center justify-between text-xs font-medium text-zinc-400">
              <span>Include last {messageCount} messages</span>
              <span className="text-zinc-600">{messageCount} / {contextMessages.length}</span>
            </label>
            <input
              type="range"
              min={0}
              max={contextMessages.length}
              value={messageCount}
              onChange={(e) => setMessageCount(Number(e.target.value))}
              className="w-full accent-violet-500"
            />
            <div className="mt-1 flex justify-between text-[10px] text-zinc-600">
              <span>0 (no context)</span>
              <span>All</span>
            </div>
          </div>

          {/* Preview toggle */}
          {messageCount > 0 && (
            <div>
              <button
                onClick={() => setShowPreview(!showPreview)}
                className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300"
              >
                {showPreview ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                <MessageSquare className="h-3 w-3" />
                Preview context ({selectedMessages.length} messages)
              </button>
              {showPreview && (
                <div className="mt-2 max-h-48 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 space-y-2">
                  {selectedMessages.map((m) => (
                    <div key={m.id} className="text-xs">
                      <span className={`font-medium ${m.role === 'user' ? 'text-violet-400' : 'text-zinc-400'}`}>
                        {m.role === 'user' ? 'User' : 'Assistant'}:
                      </span>
                      <span className="ml-1.5 text-zinc-500">
                        {m.content.length > 200 ? m.content.slice(0, 200) + '...' : m.content}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Task input */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-zinc-400">Task / Prompt</label>
            <textarea
              value={task}
              onChange={(e) => { setTask(e.target.value); setTaskTouched(true) }}
              placeholder="What should the receiving agent do with this context? (hover a target to see its suggested task)"
              rows={4}
              className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              autoFocus
            />
          </div>

          {/* Success message */}
          {sent && (
            <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-3 py-2 text-xs text-emerald-300">
              Context + task sent to <span className="font-medium">{sent}</span>. Switch to their chat tab to see the response.
            </div>
          )}
        </div>

        {/* Destination buttons */}
        <div className="border-t border-zinc-800 px-5 py-3">
          {targets.length === 0 ? (
            <p className="text-xs text-zinc-500">No other agents available. Connect to another agent first.</p>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-400">Send to:</span>
              <div className="flex flex-wrap gap-1.5">
                {targets.map((t) => {
                  const fwdTask = getTargetFwdTask(t.id)
                  return (
                    <button
                      key={t.id}
                      onClick={() => handleSend(t)}
                      onMouseEnter={() => {
                        if (!taskTouched && fwdTask) setTask(fwdTask)
                      }}
                      onMouseLeave={() => {
                        if (!taskTouched && fwdTask) setTask('')
                      }}
                      disabled={sending || !task.trim()}
                      className="flex items-center gap-1 rounded-md bg-violet-600/20 px-2.5 py-1.5 text-xs font-medium text-violet-300 hover:bg-violet-600/30 disabled:opacity-40"
                      title={fwdTask ? `Suggested task: ${fwdTask}` : undefined}
                    >
                      <Send className="h-3 w-3" />
                      {t.name}
                    </button>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
