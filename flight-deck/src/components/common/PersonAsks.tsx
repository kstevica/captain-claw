import { useCallback, useEffect, useRef, useState } from 'react'
import { MessageCircleQuestion, Send, X, ShieldAlert } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { asksApi, type AskQuestion, type OpenAsk } from '../../services/asksApi'
import { Dialog, Button, Notice, inputCls } from '../console/parts'

// Read at chat size: the question and its answers match the chat around it,
// not the console's small form text.
const answerCls = inputCls.replace('text-sm', 'text-base leading-relaxed')

// A question for this person: from a custom tool running for them (captain.ask)
// or from one of their agents (its ask_user tool). Mounted once beside the app
// (every layout): while signed in, the tab keeps one request waiting on
// /fd/asks, which answers as soon as a question opens or closes. The first open
// question shows in a dialog; answered here, it closes in their other tabs too.
// The only ways out are an answer or "Don't answer": the asker waits on it.

type Answer = string | string[] | boolean | null

function startOf(q: AskQuestion): Answer {
  if (q.default !== undefined && q.default !== null) return q.default as Answer
  return q.kind === 'multiple' ? [] : q.kind === 'text' ? '' : null
}

function missing(q: AskQuestion, v: Answer): string {
  if (q.kind === 'text') return !q.optional && !String(v ?? '').trim() ? 'needs an answer' : ''
  if (q.kind === 'multiple') {
    const n = Array.isArray(v) ? v.length : 0
    const low = q.min ?? (q.optional ? 0 : 1)
    if (n < low) return `pick at least ${low}`
    if (q.max !== undefined && n > q.max) return `pick at most ${q.max}`
    return ''
  }
  return !q.optional && (v === null || v === undefined) ? 'needs an answer' : ''
}

function QuestionField({ q, value, onChange }: { q: AskQuestion; value: Answer; onChange: (v: Answer) => void }) {
  const label = <>{q.question}{q.optional && <span className="text-zinc-600"> (optional)</span>}</>
  return (
    <fieldset className="space-y-2">
      <legend className="mb-1 block text-base font-medium leading-snug text-zinc-100">{label}</legend>
      {q.help && <p className="text-sm leading-relaxed text-zinc-500">{q.help}</p>}
      {q.kind === 'text' ? (
        (q.lines ?? 1) > 1 ? (
          <textarea className={answerCls} rows={Math.max(3, q.lines ?? 3)} maxLength={4000} value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} />
        ) : (
          <input className={answerCls} maxLength={4000} autoComplete="off" value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} />
        )
      ) : q.kind === 'confirm' ? (
        <div className="flex gap-2">
          {[true, false].map((yes) => (
            <button key={String(yes)} type="button" onClick={() => onChange(yes)}
              className={`rounded-xl border px-5 py-2 text-base font-medium transition-colors ${value === yes
                ? 'border-violet-500/60 bg-violet-500/10 text-violet-700 dark:text-violet-300' : 'border-zinc-800 text-zinc-300 hover:border-zinc-700'}`}>
              {yes ? 'Yes' : 'No'}
            </button>
          ))}
        </div>
      ) : (
        <div className="space-y-1.5">
          {(q.choices || []).map((c) => {
            const multi = q.kind === 'multiple'
            const on = multi ? Array.isArray(value) && value.includes(c) : value === c
            return (
              <label key={c} className={`flex cursor-pointer items-start gap-3 rounded-xl border px-3.5 py-2.5 text-base transition-colors ${on
                ? 'border-violet-500/60 bg-violet-500/5 text-zinc-100' : 'border-zinc-800 text-zinc-300 hover:border-zinc-700'}`}>
                <input type={multi ? 'checkbox' : 'radio'} name={q.id} className="mt-1.5 shrink-0" checked={on}
                  onChange={() => {
                    if (!multi) return onChange(c)
                    const now = Array.isArray(value) ? value : []
                    onChange(on ? now.filter((x) => x !== c) : [...now, c])
                  }} />
                <span className="min-w-0 break-words">{c}</span>
              </label>
            )
          })}
          {q.kind === 'multiple' && (
            <p className="text-sm text-zinc-500">
              {q.min === q.max ? `Pick ${q.min}.` : `Pick ${q.min ?? 0}${q.max !== undefined ? `–${q.max}` : ' or more'}.`}
            </p>
          )}
        </div>
      )}
    </fieldset>
  )
}

function AskDialog({ ask, onDone }: { ask: OpenAsk; onDone: () => void }) {
  const [values, setValues] = useState<Record<string, Answer>>(() =>
    Object.fromEntries(ask.questions.map((q) => [q.id, startOf(q)])))
  const [busy, setBusy] = useState<'' | 'send' | 'decline'>('')
  const [error, setError] = useState('')
  const [left, setLeft] = useState(ask.seconds_left)
  useEffect(() => {
    const timer = window.setInterval(() => setLeft((s) => Math.max(0, s - 1)), 1000)
    return () => window.clearInterval(timer)
  }, [])
  const problems = ask.questions.map((q) => missing(q, values[q.id])).filter(Boolean)
  const reply = async (declined: boolean) => {
    setBusy(declined ? 'decline' : 'send')
    setError('')
    try {
      await asksApi.answer(ask.id, declined ? { declined: true } : { answers: values })
      onDone()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
      setBusy('')
    }
  }
  const mins = Math.floor(left / 60)
  const timeLeft = left <= 0 ? 'time is up' : mins >= 1 ? `${mins} min left` : `${left} s left`
  return (
    <Dialog title={`${ask.title} asks`} icon={MessageCircleQuestion} onClose={onDone} keepOpen closable={false} wide footer={<>
      <Button variant="ghost" icon={X} busy={busy === 'decline'} disabled={!!busy} onClick={() => reply(true)}>Don't answer</Button>
      <Button variant="primary" icon={Send} busy={busy === 'send'} disabled={!!busy || problems.length > 0} onClick={() => reply(false)}>Send</Button>
    </>}>
      <div className="space-y-4">
        <p className="text-sm leading-relaxed text-zinc-500">
          {ask.source === 'agent'
            ? <>{ask.agent_name || 'Your agent'} needs your answer to go on · {timeLeft}.</>
            : <>{ask.label || 'A custom tool'}{ask.owner_name ? ` (a tool by ${ask.owner_name})` : ''} is running
              {ask.agent_name ? ` for ${ask.agent_name}` : ''} and waits for your answer · {timeLeft}.</>}
        </p>
        {error && <Notice tone="red">{error}</Notice>}
        {ask.questions.map((q) => (
          <QuestionField key={q.id} q={q} value={values[q.id]} onChange={(v) => setValues((now) => ({ ...now, [q.id]: v }))} />
        ))}
        <p className="flex items-start gap-1.5 text-xs leading-relaxed text-zinc-500">
          <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          {ask.source === 'agent' ? 'Never type a password or key here.'
            : "Never type a password or key here: a tool's secrets go in its Secrets, on the Custom tools page."}
        </p>
      </div>
    </Dialog>
  )
}

export function PersonAsks() {
  const signedIn = useAuthStore((s) => s.authEnabled === false || !!s.user)
  const [asks, setAsks] = useState<OpenAsk[]>([])
  const known = useRef<string[]>([])

  useEffect(() => {
    if (!signedIn) return
    let alive = true
    const loop = async () => {
      while (alive) {
        try {
          const r = await asksApi.list(known.current, 25)
          if (!alive) return
          known.current = r.asks.map((a) => a.id)
          setAsks(r.asks)
        } catch {
          await new Promise((res) => window.setTimeout(res, 5000))
        }
      }
    }
    void loop()
    return () => { alive = false }
  }, [signedIn])

  const done = useCallback((id: string) => setAsks((now) => now.filter((a) => a.id !== id)), [])
  const current = signedIn ? asks[0] : undefined
  if (!current) return null
  return (
    <AskDialog key={current.id} ask={current} onDone={() => done(current.id)} />
  )
}
