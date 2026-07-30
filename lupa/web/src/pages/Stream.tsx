import { useCallback, useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ArrowLeft, FileText, Loader2, Send } from 'lucide-react'
import { api, post } from '../api'
import { useVocab, type Round, type Stream } from '../stores'

interface Detail {
  id: string
  status: string
  title?: string
  intent?: string
  truth?: string
  route?: { group0_plan?: unknown }
  analysis?: Record<string, unknown> | null
}

interface ProgressEvent { i: number; stage: string; message: string; usd?: number }

interface Entry { name: string; dir?: boolean; size?: number }

const LIVE = new Set(['routing', 'routed', 'planning', 'awaiting_plan', 'running'])

export default function StreamView({ streamId, onBack }: { streamId: string; onBack: () => void }) {
  const v = useVocab()
  const [stream, setStream] = useState<(Stream & { rounds: Round[] }) | null>(null)
  const [detail, setDetail] = useState<Detail | null>(null)
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [error, setError] = useState('')

  const activeSid = stream?.rounds?.length
    ? stream.rounds[stream.rounds.length - 1].session_id : null

  const loadStream = useCallback(async () => {
    setStream(await api(`/api/streams/${streamId}`))
  }, [streamId])

  useEffect(() => { void loadStream() }, [loadStream])

  // Poll the active commission while it is live.
  useEffect(() => {
    if (!activeSid) return
    let stop = false
    const tick = async () => {
      try {
        const d = await api<Detail>(`/api/commissions/${activeSid}`)
        if (stop) return
        setDetail(d)
        if (LIVE.has(d.status)) {
          const p = await api<{ events: ProgressEvent[] }>(`/api/commissions/${activeSid}/progress`)
          if (!stop) setEvents(p.events ?? [])
        }
      } catch (e) {
        if (!stop) setError(e instanceof Error ? e.message : 'failed to load')
      }
    }
    void tick()
    const iv = setInterval(() => {
      void tick()
    }, 3000)
    return () => { stop = true; clearInterval(iv) }
  }, [activeSid])

  const commission = async (brief: string) => {
    setError('')
    try {
      await post(`/api/streams/${streamId}/commissions`, { brief })
      await loadStream()
    } catch (e) { setError(e instanceof Error ? e.message : 'commission failed') }
  }

  const status = detail?.status ?? ''
  const showComposer = !activeSid || status === 'done'

  return (
    <div className="max-w-4xl mx-auto px-5 py-6 space-y-5">
      <div className="flex items-center gap-3">
        <button onClick={onBack} className="p-1.5 rounded hover:bg-[var(--lp-border)] text-[var(--lp-text-dim)]">
          <ArrowLeft size={17} />
        </button>
        <h1 className="text-xl font-bold">{stream?.title ?? '…'}</h1>
        {stream?.rounds && stream.rounds.length > 0 && (
          <div className="flex gap-1.5 ml-2">
            {stream.rounds.map((r) => (
              <span key={r.session_id}
                    title={r.kind}
                    className={`text-[11px] px-2 py-0.5 rounded-full border ${
                      r.session_id === activeSid
                        ? 'border-[var(--lp-accent)] text-[var(--lp-accent)]'
                        : 'border-[var(--lp-border)] text-[var(--lp-text-dim)]'}`}>
                {v('round', 'Round')} {r.round_no}
              </span>
            ))}
          </div>
        )}
        <div className="flex-1" />
        {status && (
          <span className="text-xs px-2.5 py-1 rounded-full bg-[var(--lp-surface)] border border-[var(--lp-border)] text-[var(--lp-text-dim)]">
            {status}
          </span>
        )}
      </div>

      {error && <div className="text-sm text-red-400">{error}</div>}

      {activeSid && (status === 'planning' || status === 'routing' || status === 'routed') && (
        <Waiting label="Your team is drafting the research plan…" events={events} />
      )}

      {activeSid && status === 'awaiting_plan' && detail?.route?.group0_plan !== undefined && (
        <PlanGate
          sid={activeSid}
          plan={detail.route.group0_plan}
          onDecided={() => void loadStream()}
        />
      )}

      {activeSid && status === 'running' && (
        <Waiting label="The desk is working your brief…" events={events} />
      )}

      {activeSid && (status === 'error' || status === 'cancelled' || status === 'rejected') && (
        <div className="rounded-xl border border-red-900/60 bg-[var(--lp-surface)] px-5 py-4 text-sm">
          This {v('round', 'Round').toLowerCase()} ended with status “{status}”.
          {detail?.truth ? ' Partial output is shown below.' : ''}
        </div>
      )}

      {activeSid && status === 'done' && stream && (
        <Report streamId={streamId} truth={detail?.truth ?? ''} />
      )}

      {showComposer && (
        <Composer
          placeholder={activeSid ? v('continue_placeholder') : v('composer_placeholder')}
          cta={activeSid ? `Next ${v('round', 'Round').toLowerCase()}` : v('commission', 'Commission')}
          onSubmit={commission}
        />
      )}
    </div>
  )
}

// ── pieces ───────────────────────────────────────────────────────────

function Composer({ placeholder, cta, onSubmit }:
  { placeholder: string; cta: string; onSubmit: (brief: string) => Promise<void> }) {
  const [brief, setBrief] = useState('')
  const [busy, setBusy] = useState(false)
  return (
    <form
      onSubmit={async (e) => {
        e.preventDefault()
        if (!brief.trim()) return
        setBusy(true)
        try { await onSubmit(brief.trim()); setBrief('') } finally { setBusy(false) }
      }}
      className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] p-4 space-y-3"
    >
      <textarea
        value={brief}
        onChange={(e) => setBrief(e.target.value)}
        placeholder={placeholder}
        rows={4}
        className="w-full rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2 text-sm outline-none focus:border-[var(--lp-accent)] resize-y"
      />
      <div className="flex justify-end">
        <button
          type="submit" disabled={busy || !brief.trim()}
          className="rounded-lg px-4 py-2 text-sm font-semibold text-black disabled:opacity-40 flex items-center gap-1.5"
          style={{ background: 'var(--lp-accent)' }}
        >
          {busy ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />} {cta}
        </button>
      </div>
    </form>
  )
}

function Waiting({ label, events }: { label: string; events: ProgressEvent[] }) {
  const feedRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight })
  }, [events.length])
  return (
    <div className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] p-4 space-y-3">
      <div className="flex items-center gap-2 text-sm">
        <Loader2 size={15} className="animate-spin" style={{ color: 'var(--lp-accent)' }} />
        {label}
      </div>
      {events.length > 0 && (
        <div ref={feedRef} className="max-h-64 overflow-y-auto space-y-1 text-xs font-mono">
          {events.map((e) => (
            <div key={e.i} className={e.stage === 'phase'
              ? 'font-bold mt-2' : 'text-[var(--lp-text-dim)]'}>
              {e.stage === 'phase' ? `— ${e.message} —` : `${e.stage}: ${e.message}`}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function PlanGate({ sid, plan, onDecided }:
  { sid: string; plan: unknown; onDecided: () => void }) {
  const v = useVocab()
  const [text, setText] = useState(() => JSON.stringify(plan ?? {}, null, 2))
  const [busy, setBusy] = useState<'approve' | 'cancel' | null>(null)
  const [error, setError] = useState('')

  const decide = async (action: 'approve' | 'cancel') => {
    setBusy(action); setError('')
    try {
      if (action === 'approve') {
        let edited: unknown = null
        try { edited = JSON.parse(text) } catch { setError('The plan is not valid JSON.'); setBusy(null); return }
        const original = JSON.stringify(plan ?? {}, null, 2)
        await post(`/api/commissions/${sid}/approve`,
                   { plan: text === original ? null : edited })
      } else {
        await post(`/api/commissions/${sid}/cancel`, {})
      }
      onDecided()
    } catch (e) { setError(e instanceof Error ? e.message : 'failed') }
    finally { setBusy(null) }
  }

  return (
    <div className="rounded-xl border border-[var(--lp-accent)] bg-[var(--lp-surface)] p-5 space-y-3">
      <div className="font-semibold">{v('plan_gate_title')}</div>
      <div className="text-sm text-[var(--lp-text-dim)]">{v('plan_gate_hint')}</div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={14}
        spellCheck={false}
        className="w-full rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2 text-xs font-mono outline-none focus:border-[var(--lp-accent)] resize-y"
      />
      {error && <div className="text-sm text-red-400">{error}</div>}
      <div className="flex gap-2 justify-end">
        <button
          onClick={() => void decide('cancel')} disabled={busy !== null}
          className="rounded-lg px-4 py-2 text-sm border border-[var(--lp-border)] text-[var(--lp-text-dim)] hover:text-[var(--lp-text)] disabled:opacity-40"
        >
          {busy === 'cancel' ? '…' : 'Cancel'}
        </button>
        <button
          onClick={() => void decide('approve')} disabled={busy !== null}
          className="rounded-lg px-4 py-2 text-sm font-semibold text-black disabled:opacity-40"
          style={{ background: 'var(--lp-accent)' }}
        >
          {busy === 'approve' ? '…' : 'Approve & run'}
        </button>
      </div>
    </div>
  )
}

function Report({ streamId, truth }: { streamId: string; truth: string }) {
  const [entries, setEntries] = useState<Entry[]>([])
  const [selected, setSelected] = useState('')
  const [content, setContent] = useState('')

  useEffect(() => {
    void (async () => {
      const data = await api<{ entries: Entry[] }>(`/api/streams/${streamId}/files`)
      const files = (data.entries ?? []).filter((e) => !e.dir && e.name.endsWith('.md'))
      setEntries(files)
      const main = files.find((f) => /report|output|synthesis/i.test(f.name)) ?? files[0]
      if (main) setSelected(main.name)
    })()
  }, [streamId])

  useEffect(() => {
    if (!selected) { setContent(''); return }
    void (async () => {
      const data = await api<{ text?: string }>(
        `/api/streams/${streamId}/file?path=${encodeURIComponent(selected)}`)
      setContent(data.text ?? '')
    })()
  }, [streamId, selected])

  const body = content || truth
  return (
    <div className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)]">
      {entries.length > 1 && (
        <div className="flex gap-1.5 flex-wrap px-5 pt-4">
          {entries.map((f) => (
            <button key={f.name}
                    onClick={() => setSelected(f.name)}
                    className={`text-[11px] px-2 py-1 rounded border flex items-center gap-1 ${
                      f.name === selected
                        ? 'border-[var(--lp-accent)] text-[var(--lp-accent)]'
                        : 'border-[var(--lp-border)] text-[var(--lp-text-dim)] hover:text-[var(--lp-text)]'}`}>
              <FileText size={11} /> {f.name}
            </button>
          ))}
        </div>
      )}
      <div className="lp-prose px-6 py-5 text-[15px]">
        {body
          ? <Markdown remarkPlugins={[remarkGfm]}>{body}</Markdown>
          : <div className="text-sm text-[var(--lp-text-dim)]">No report file yet.</div>}
      </div>
    </div>
  )
}
