// Bat needs you — renders the run's open human-in-the-loop asks and lets the
// owner answer them: approve/cancel a plan or a spend, type a value, or enter a
// private code (masked, kept server-side out of logs). One card per open ask.
import { useState } from 'react'
import { AlertTriangle, Check, X } from 'lucide-react'
import { useBatStore } from '../../stores/batStore'
import type { BatAsk } from '../../stores/batStore'

function AskRow({ ask }: { ask: BatAsk }) {
  const answer = useBatStore((s) => s.answer)
  const [value, setValue] = useState('')
  const [sending, setSending] = useState(false)

  const send = async (text: string) => {
    setSending(true)
    try {
      await answer(ask.id, text)
    } finally {
      setSending(false)
    }
  }

  const isDecision = ask.kind === 'plan_approval' || ask.kind === 'spend_approval'
  const label =
    ask.kind === 'plan_approval' ? 'Plan approval'
      : ask.kind === 'spend_approval' ? 'Spend approval'
        : ask.secret ? 'Private value' : 'Input needed'

  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-3">
      <div className="mb-1 flex items-center gap-2 text-xs font-medium text-amber-300">
        <AlertTriangle size={14} />
        {label}
      </div>
      <p className="mb-3 whitespace-pre-wrap text-sm text-zinc-200">{ask.question}</p>

      {isDecision ? (
        <div className="flex gap-2">
          <button
            disabled={sending}
            onClick={() => send('approve')}
            className="inline-flex items-center gap-1 rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
          >
            <Check size={14} /> Approve
          </button>
          <button
            disabled={sending}
            onClick={() => send('cancel')}
            className="inline-flex items-center gap-1 rounded-md bg-zinc-700 px-3 py-1.5 text-sm font-medium text-zinc-100 hover:bg-zinc-600 disabled:opacity-50"
          >
            <X size={14} /> Cancel
          </button>
        </div>
      ) : ask.options && ask.options.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {ask.options.map((opt) => (
            <button
              key={opt}
              disabled={sending}
              onClick={() => send(opt)}
              className="rounded-md bg-zinc-700 px-3 py-1.5 text-sm text-zinc-100 hover:bg-zinc-600 disabled:opacity-50"
            >
              {opt}
            </button>
          ))}
        </div>
      ) : (
        <form
          onSubmit={(e) => {
            e.preventDefault()
            if (value.trim()) void send(value.trim())
          }}
          className="flex gap-2"
        >
          <input
            type={ask.secret ? 'password' : 'text'}
            autoComplete="off"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={ask.secret ? 'Enter the private value…' : 'Your answer…'}
            className="flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-amber-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={sending || !value.trim()}
            className="rounded-md bg-amber-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-500 disabled:opacity-50"
          >
            Send
          </button>
        </form>
      )}
      {ask.secret && (
        <p className="mt-2 text-[11px] text-zinc-500">Kept private — never stored in logs or sent to a chat channel.</p>
      )}
    </div>
  )
}

export function BatAskCard({ asks }: { asks: BatAsk[] }) {
  if (!asks || asks.length === 0) return null
  return (
    <div className="space-y-2">
      {asks.map((a) => (
        <AskRow key={a.id} ask={a} />
      ))}
    </div>
  )
}
