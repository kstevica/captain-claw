import { useEffect, useRef, useState } from 'react'
import { Loader2, Send, Square, Paperclip, Wrench, X } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AgentChatWS } from '../../services/agentChat'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'

// A live chat scoped to one topic. Reuses AgentChatWS for streaming (narration,
// tools, busy) but renders ONLY the turns the user initiates here (ignoring the
// agent's session replay so topics never mix), and persists each completed turn
// to the selected topic.

type LiveMsg =
  | { kind: 'user' | 'agent'; text: string }
  | { kind: 'narration'; text: string }
  | { kind: 'tool'; tool: string; detail: string }

interface Attachment { name: string; status: 'uploading' | 'uploaded' | 'error'; path?: string }

async function authedFetch(path: string, init?: RequestInit): Promise<Response> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { ...(init?.headers as Record<string, string> | undefined) }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
  let res = await fetch(`/fd${path}`, { ...init, headers, credentials: 'include' })
  if (res.status === 401 && authEnabled && await refreshAccessToken()) {
    const t2 = useAuthStore.getState().token
    if (t2) headers['Authorization'] = `Bearer ${t2}`
    res = await fetch(`/fd${path}`, { ...init, headers, credentials: 'include' })
  }
  return res
}

export function TopicChat({ host, port, auth, topicId, onPersisted }: {
  host: string; port: number; auth?: string; topicId: string; onPersisted?: () => void
}) {
  const wsRef = useRef<AgentChatWS | null>(null)
  const [live, setLive] = useState<LiveMsg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const scrollRef = useRef<HTMLDivElement>(null)
  // Turn state held in refs so WS handlers (registered once) see current values.
  const awaiting = useRef(false)
  const userText = useRef('')
  const agentText = useRef('')
  const topicRef = useRef(topicId)
  topicRef.current = topicId
  const tokenQs = auth ? `?token=${encodeURIComponent(auth)}` : ''

  // One WS for the panel's lifetime (the agent is the same across topics).
  useEffect(() => {
    const ws = new AgentChatWS(`topic-${host}-${port}`, host, port, auth || '')
    wsRef.current = ws
    const offs = [
      ws.on('status', (d) => {
        const s = String(d.status || d.text || '')
        if (s) setStatus(String(d.text || s))
        if (['idle', 'ready', 'done', 'error'].includes(s.toLowerCase())) finishTurn()
      }),
      ws.on('narration', (d) => {
        if (!awaiting.current) return
        const t = String(d.text || '').trim()
        if (t) setLive((m) => [...m, { kind: 'narration', text: t }])
      }),
      ws.on('monitor', (d) => {
        if (!awaiting.current || d.replay) return
        setLive((m) => [...m, { kind: 'tool', tool: String(d.tool_name || d.tool || 'tool'),
          detail: _summarize(d.arguments) }])
      }),
      ws.on('chat_message', (d) => {
        if (!awaiting.current || d.replay || d.role !== 'assistant') return
        const c = String(d.content || '').trim()
        if (!c) return
        agentText.current = c
        setLive((m) => {
          const copy = [...m]
          // update the trailing agent bubble, or add one
          if (copy.length && (copy[copy.length - 1] as LiveMsg).kind === 'agent') {
            copy[copy.length - 1] = { kind: 'agent', text: c }
          } else copy.push({ kind: 'agent', text: c })
          return copy
        })
        finishTurn()
      }),
    ]
    ws.connect()
    return () => { offs.forEach((off) => off()); ws.disconnect() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [host, port, auth])

  // Switching topics resets the live view (prior topic's turns were persisted).
  useEffect(() => {
    setLive([]); setInput(''); setAttachments([]); setBusy(false); setStatus('')
    awaiting.current = false; userText.current = ''; agentText.current = ''
  }, [topicId])

  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight }) }, [live])

  const finishTurn = () => {
    if (!awaiting.current) return
    awaiting.current = false
    setBusy(false); setStatus('')
    const u = userText.current, a = agentText.current
    if (u || a) {
      authedFetch(`/agent-topic-append/${host}/${port}/${encodeURIComponent(topicRef.current)}${tokenQs}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [
          ...(u ? [{ role: 'user', content: u }] : []),
          ...(a ? [{ role: 'agent', content: a }] : []),
        ] }),
      }).then(() => onPersisted?.()).catch(() => { /* non-fatal */ })
    }
    userText.current = ''; agentText.current = ''
  }

  const uploadOne = async (file: File) => {
    const idx = attachments.length
    setAttachments((a) => [...a, { name: file.name, status: 'uploading' }])
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await authedFetch(`/agent-file-upload/${host}/${port}${tokenQs}`, { method: 'POST', body: fd })
      const j = await res.json()
      setAttachments((a) => a.map((x, i) => i === idx ? { ...x, status: 'uploaded', path: j.path } : x))
    } catch {
      setAttachments((a) => a.map((x, i) => i === idx ? { ...x, status: 'error' } : x))
    }
  }
  const onPaste = (e: React.ClipboardEvent) => {
    const files = Array.from(e.clipboardData.items).map((it) => it.getAsFile()).filter(Boolean) as File[]
    files.forEach(uploadOne)
  }

  const uploading = attachments.some((a) => a.status === 'uploading')
  const canSend = !busy && !uploading && (input.trim() || attachments.some((a) => a.status === 'uploaded'))

  const send = () => {
    if (!canSend || !wsRef.current) return
    let content = input.trim()
    for (const a of attachments) if (a.status === 'uploaded' && a.path) content += `\n[Attached file: ${a.name} → ${a.path}]`
    userText.current = content; agentText.current = ''
    setLive((m) => [...m, { kind: 'user', text: content }])
    awaiting.current = true; setBusy(true); setStatus('Thinking…')
    wsRef.current.send(content)
    setInput(''); setAttachments([])
  }

  return (
    <div className="mt-2 flex flex-col border-t border-zinc-800 pt-2">
      {live.length > 0 && (
        <div ref={scrollRef} className="mb-2 max-h-[40vh] overflow-auto rounded-lg bg-zinc-950/40 p-2">
          {live.map((m, i) => (
            <div key={i} className="mb-1.5 last:mb-0">
              {m.kind === 'narration' ? (
                <div className="text-[10px] italic text-zinc-500">{m.text}</div>
              ) : m.kind === 'tool' ? (
                <div className="flex items-center gap-1 text-[10px] text-zinc-500"><Wrench className="h-3 w-3" /> {m.tool}{m.detail ? ` · ${m.detail}` : ''}</div>
              ) : (
                <div className={m.kind === 'user' ? 'flex justify-end' : ''}>
                  <div className={`fd-markdown text-xs ${m.kind === 'user' ? 'rounded-lg bg-violet-600/20 px-2 py-1 text-zinc-200' : 'text-zinc-300'}`}>
                    <Markdown remarkPlugins={[remarkGfm]}>{m.text}</Markdown>
                  </div>
                </div>
              )}
            </div>
          ))}
          {busy && <div className="flex items-center gap-1 text-[10px] text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" /> {status || 'Working…'}</div>}
        </div>
      )}

      {attachments.length > 0 && (
        <div className="mb-1 flex flex-wrap gap-1">
          {attachments.map((a, i) => (
            <span key={i} className="flex items-center gap-1 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-300">
              {a.status === 'uploading' ? <Loader2 className="h-3 w-3 animate-spin" /> : a.status === 'error' ? '⚠' : '📎'} {a.name}
              <button onClick={() => setAttachments((x) => x.filter((_, j) => j !== i))} className="text-zinc-500 hover:text-zinc-300"><X className="h-3 w-3" /></button>
            </span>
          ))}
        </div>
      )}

      <div className="flex items-end gap-1.5">
        <label className="cursor-pointer p-1.5 text-zinc-500 hover:text-zinc-300" title="Attach file">
          <Paperclip className="h-4 w-4" />
          <input type="file" multiple className="hidden" onChange={(e) => { Array.from(e.target.files || []).forEach(uploadOne); e.target.value = '' }} />
        </label>
        <textarea
          value={input} onChange={(e) => setInput(e.target.value)} onPaste={onPaste}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
          rows={1} placeholder={busy ? 'Agent is working…' : 'Message this topic…'}
          className="max-h-32 min-h-[2.25rem] flex-1 resize-none rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-sky-600"
        />
        {busy ? (
          <button onClick={() => { wsRef.current?.cancel(); finishTurn() }} title="Stop"
            className="rounded-lg bg-rose-600 p-2 text-white hover:bg-rose-500"><Square className="h-4 w-4" /></button>
        ) : (
          <button onClick={send} disabled={!canSend} title="Send"
            className="rounded-lg bg-sky-600 p-2 text-white hover:bg-sky-500 disabled:opacity-40"><Send className="h-4 w-4" /></button>
        )}
      </div>
    </div>
  )
}

function _summarize(args: unknown): string {
  if (!args || typeof args !== 'object') return ''
  try {
    const entries = Object.entries(args as Record<string, unknown>).slice(0, 2)
    return entries.map(([k, v]) => `${k}=${String(v).slice(0, 40)}`).join(', ')
  } catch { return '' }
}
