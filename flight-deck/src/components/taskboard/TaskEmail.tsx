import { useEffect, useState } from 'react'
import { Loader2, Paperclip } from 'lucide-react'
import { fmtWhen } from '../../services/schedulerApi'
import { taskboardApi, type TaskEmailMessage } from '../../services/taskboardApi'
import { FoldHighlight } from './taskboardParts'
import { readableEmail } from './taskboardText'

const sectionTitle = 'text-[11px] font-medium uppercase tracking-wider text-zinc-500'
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))

function Addressed({ message }: { message: TaskEmailMessage }) {
  const to = [...message.to, ...message.cc]
  return (
    <span className="min-w-0 text-xs text-zinc-500">
      <span className="font-medium text-zinc-300">{message.from_name || message.from_address}</span>
      {message.from_name && message.from_address ? ` <${message.from_address}>` : ''}
      {' · '}{fmtWhen(message.date)}
      {to.length > 0 && <span className="block truncate">To {to.join(', ')}</span>}
    </span>
  )
}

function Body({ message, terms }: { message: TaskEmailMessage; terms: string[] }) {
  return (
    <>
      <p className="mt-2 whitespace-pre-wrap break-words text-[15px] leading-7 text-zinc-200">
        <FoldHighlight text={readableEmail(message.text) || '(No text in this email)'} terms={terms} />
      </p>
      {message.attachments.length > 0 && (
        <p className="mt-2 flex items-center gap-1.5 text-xs text-zinc-500">
          <Paperclip className="h-3.5 w-3.5 shrink-0" /><span className="min-w-0 truncate">{message.attachments.join(', ')}</span>
        </p>
      )}
    </>
  )
}

/** The whole email behind a task, read from Gmail now and kept nowhere: the bigger
 *  view's, where there is room to read it. As plain text, whatever it came as:
 *  nothing in it is followed or loaded. The latest is open, older ones a click away. */
export function TaskEmail({ taskId, terms }: { taskId: string; terms: string[] }) {
  const [messages, setMessages] = useState<TaskEmailMessage[] | null>(null)
  const [more, setMore] = useState(0)
  const [reason, setReason] = useState('')

  useEffect(() => {
    let gone = false
    taskboardApi.email(taskId)
      .then((res) => { if (!gone) { setMessages(res.messages); setMore(res.more); setReason(res.reason) } })
      .catch((e) => { if (!gone) { setMessages([]); setReason(errText(e)) } })
    return () => { gone = true }
  }, [taskId])

  if (messages !== null && !messages.length && !reason) return null
  const latest = messages?.[messages.length - 1]
  return (
    <section className="mt-4">
      <h4 className={sectionTitle}>{messages && messages.length > 1 ? `The email · ${messages.length + more} in this thread` : 'The email'}</h4>
      {messages === null ? (
        <Loader2 className="mt-2 h-4 w-4 animate-spin text-zinc-500" />
      ) : !latest ? (
        <p className="mt-1 text-xs leading-relaxed text-zinc-500">{reason}</p>
      ) : (
        <div className="mt-2 flex flex-col gap-2">
          {more > 0 && <p className="text-xs text-zinc-500">{more} older {more === 1 ? 'email' : 'emails'} not shown.</p>}
          {messages.slice(0, -1).map((m) => (
            <details key={m.id} className="rounded-lg border border-zinc-800 px-3 py-2">
              <summary className="cursor-pointer"><Addressed message={m} /></summary>
              <Body message={m} terms={terms} />
            </details>
          ))}
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-3">
            <Addressed message={latest} />
            <Body message={latest} terms={terms} />
          </div>
        </div>
      )}
    </section>
  )
}
