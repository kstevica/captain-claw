import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ArrowLeft, ArrowLeftRight, FileText, Loader2, Send, SlidersHorizontal } from 'lucide-react'
import { api, post } from '../api'
import { useVocab, type Round, type Stream } from '../stores'
import ReceiptsPanel from '../components/Receipts'
import { collapseSame, lineDiff } from '../lib/diff'

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

interface StreamSettings { quality_profile?: string; same_cast?: boolean; max_agents?: number }

type ContinueKind = 'continue' | 'revise' | 'fill_gaps'

const KIND_LABELS: Record<ContinueKind, string> = {
  continue: 'Deepen', revise: 'Revise', fill_gaps: 'Fill gaps',
}

const LIVE = new Set(['routing', 'routed', 'planning', 'awaiting_plan', 'running'])

export default function StreamView({ streamId, onBack }: { streamId: string; onBack: () => void }) {
  const v = useVocab()
  const [stream, setStream] = useState<(Stream & { rounds: Round[]; settings?: StreamSettings }) | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<Detail | null>(null)
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [error, setError] = useState('')
  const [tipStatus, setTipStatus] = useState('')
  const [draft, setDraft] = useState('')
  const [draftKind, setDraftKind] = useState<ContinueKind>('continue')
  const [showSettings, setShowSettings] = useState(false)
  const [showDiff, setShowDiff] = useState(false)
  const composerRef = useRef<HTMLDivElement>(null)

  const rounds = stream?.rounds ?? []
  const tipSid = rounds.length ? rounds[rounds.length - 1].session_id : null
  const activeSid = selected ?? tipSid

  const loadStream = useCallback(async (selectTip = false) => {
    const s = await api<Stream & { rounds: Round[]; settings?: StreamSettings }>(`/api/streams/${streamId}`)
    setStream(s)
    if (selectTip && s.rounds.length) {
      setSelected(s.rounds[s.rounds.length - 1].session_id)
    }
  }, [streamId])

  useEffect(() => { void loadStream() }, [loadStream])

  // Poll the selected commission; stop once it reaches a terminal state.
  useEffect(() => {
    if (!activeSid) return
    let stop = false
    let iv: ReturnType<typeof setInterval> | null = null
    setDetail(null); setEvents([])
    const tick = async () => {
      try {
        const d = await api<Detail>(`/api/commissions/${activeSid}`)
        if (stop) return
        setDetail(d)
        if (activeSid === (stream?.rounds?.at(-1)?.session_id ?? activeSid)) {
          setTipStatus(d.status)
        }
        if (LIVE.has(d.status)) {
          const p = await api<{ events: ProgressEvent[] }>(`/api/commissions/${activeSid}/progress`)
          if (!stop) setEvents(p.events ?? [])
        } else if (iv) { clearInterval(iv); iv = null }
      } catch (e) {
        if (!stop) setError(e instanceof Error ? e.message : 'failed to load')
        if (iv) { clearInterval(iv); iv = null }
      }
    }
    void tick()
    iv = setInterval(() => { void tick() }, 3000)
    return () => { stop = true; if (iv) clearInterval(iv) }
  }, [activeSid, stream])

  const commission = async (brief: string) => {
    setError('')
    try {
      await post(`/api/streams/${streamId}/commissions`,
                 { brief, kind: tipSid ? draftKind : 'auto' })
      setDraft('')
      setShowDiff(false)
      await loadStream(true)
    } catch (e) { setError(e instanceof Error ? e.message : 'commission failed') }
  }

  const followUp = (brief: string) => {
    setDraft(brief)
    setDraftKind('fill_gaps')
    composerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  const status = detail?.status ?? ''
  const tipDone = !tipSid || tipStatus === 'done'
  const showComposer = !tipSid || (tipStatus ? tipDone : false)
  const selectedRound = rounds.find((r) => r.session_id === activeSid)
  const prevRound = selectedRound && selectedRound.round_no > 1
    ? rounds.find((r) => r.round_no === selectedRound.round_no - 1) : undefined

  return (
    <div className="max-w-4xl mx-auto px-5 py-6 space-y-5">
      <div className="flex items-center gap-3">
        <button onClick={onBack} className="p-1.5 rounded hover:bg-[var(--lp-border)] text-[var(--lp-text-dim)]">
          <ArrowLeft size={17} />
        </button>
        <h1 className="text-xl font-bold">{stream?.title ?? '…'}</h1>
        {rounds.length > 0 && (
          <div className="flex gap-1.5 ml-2 flex-wrap">
            {rounds.map((r) => (
              <button key={r.session_id}
                      title={r.kind}
                      onClick={() => setSelected(r.session_id)}
                      className={`text-[11px] px-2 py-0.5 rounded-full border transition-colors ${
                        r.session_id === activeSid
                          ? 'border-[var(--lp-accent)] text-[var(--lp-accent)]'
                          : 'border-[var(--lp-border)] text-[var(--lp-text-dim)] hover:text-[var(--lp-text)]'}`}>
                {v('round', 'Round')} {r.round_no}
              </button>
            ))}
          </div>
        )}
        <div className="flex-1" />
        {status && (
          <span className="text-xs px-2.5 py-1 rounded-full bg-[var(--lp-surface)] border border-[var(--lp-border)] text-[var(--lp-text-dim)]">
            {status}
          </span>
        )}
        <button
          onClick={() => setShowSettings(!showSettings)}
          title="Stream settings"
          className={`p-1.5 rounded hover:bg-[var(--lp-border)] ${
            showSettings ? 'text-[var(--lp-accent)]' : 'text-[var(--lp-text-dim)]'}`}
        >
          <SlidersHorizontal size={16} />
        </button>
      </div>

      {showSettings && stream && (
        <SettingsPanel
          streamId={streamId}
          settings={stream.settings ?? {}}
          onSaved={() => void loadStream()}
        />
      )}

      {error && <div className="text-sm text-red-400">{error}</div>}

      {activeSid && (status === 'planning' || status === 'routing' || status === 'routed') && (
        <Waiting label="Your team is drafting the research plan…" events={events} />
      )}

      {activeSid && status === 'awaiting_plan' && detail?.route?.group0_plan !== undefined && (
        <PlanGate
          sid={activeSid}
          plan={detail.route.group0_plan}
          onDecided={() => void loadStream(true)}
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
        <>
          {prevRound && (
            <div className="flex justify-end">
              <button
                onClick={() => setShowDiff(!showDiff)}
                className={`text-xs px-2.5 py-1 rounded-full border flex items-center gap-1.5 transition-colors ${
                  showDiff
                    ? 'border-[var(--lp-accent)] text-[var(--lp-accent)]'
                    : 'border-[var(--lp-border)] text-[var(--lp-text-dim)] hover:text-[var(--lp-text)]'}`}
              >
                <ArrowLeftRight size={12} />
                {showDiff ? 'Show the report' : `What changed vs ${v('round', 'Round').toLowerCase()} ${prevRound.round_no}`}
              </button>
            </div>
          )}
          {showDiff && prevRound
            ? <RoundDiff prevSid={prevRound.session_id} currTruth={detail?.truth ?? ''} />
            : <Report streamId={streamId} truth={detail?.truth ?? ''} />}
          <ReceiptsPanel sid={activeSid} onFollowUp={tipDone ? followUp : undefined} />
        </>
      )}

      {showComposer && (
        <div ref={composerRef}>
          <Composer
            placeholder={tipSid ? v('continue_placeholder') : v('composer_placeholder')}
            cta={tipSid ? `Next ${v('round', 'Round').toLowerCase()}` : v('commission', 'Commission')}
            value={draft}
            onChange={setDraft}
            kind={tipSid ? draftKind : null}
            onKind={setDraftKind}
            onSubmit={commission}
          />
        </div>
      )}
    </div>
  )
}

// ── pieces ───────────────────────────────────────────────────────────

function Composer({ placeholder, cta, value, onChange, kind, onKind, onSubmit }: {
  placeholder: string; cta: string
  value: string; onChange: (v: string) => void
  kind: ContinueKind | null; onKind: (k: ContinueKind) => void
  onSubmit: (brief: string) => Promise<void>
}) {
  const [busy, setBusy] = useState(false)
  return (
    <form
      onSubmit={async (e) => {
        e.preventDefault()
        if (!value.trim()) return
        setBusy(true)
        try { await onSubmit(value.trim()) } finally { setBusy(false) }
      }}
      className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] p-4 space-y-3"
    >
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={4}
        className="w-full rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2 text-sm outline-none focus:border-[var(--lp-accent)] resize-y"
      />
      <div className="flex items-center justify-end gap-2">
        {kind !== null && (
          <div className="flex gap-1 mr-auto">
            {(Object.keys(KIND_LABELS) as ContinueKind[]).map((k) => (
              <button key={k} type="button"
                      onClick={() => onKind(k)}
                      className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                        kind === k
                          ? 'border-[var(--lp-accent)] text-[var(--lp-accent)]'
                          : 'border-[var(--lp-border)] text-[var(--lp-text-dim)] hover:text-[var(--lp-text)]'}`}>
                {KIND_LABELS[k]}
              </button>
            ))}
          </div>
        )}
        <button
          type="submit" disabled={busy || !value.trim()}
          className="rounded-lg px-4 py-2 text-sm font-semibold text-black disabled:opacity-40 flex items-center gap-1.5"
          style={{ background: 'var(--lp-accent)' }}
        >
          {busy ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />} {cta}
        </button>
      </div>
    </form>
  )
}

function SettingsPanel({ streamId, settings, onSaved }:
  { streamId: string; settings: StreamSettings; onSaved: () => void }) {
  const [busy, setBusy] = useState(false)
  const save = async (patch: Record<string, unknown>) => {
    setBusy(true)
    try {
      await api(`/api/streams/${streamId}/settings`,
                { method: 'PATCH', body: JSON.stringify(patch) })
      onSaved()
    } finally { setBusy(false) }
  }
  const quality = settings.quality_profile ?? ''
  const sameCast = settings.same_cast ?? true
  const maxAgents = settings.max_agents ?? 6
  return (
    <div className={`rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm ${busy ? 'opacity-60' : ''}`}>
      <label className="flex items-center gap-2">
        <span className="text-[var(--lp-text-dim)]">Quality</span>
        <select
          value={quality}
          onChange={(e) => void save({ quality_profile: e.target.value || null })}
          className="rounded bg-[var(--lp-bg)] border border-[var(--lp-border)] px-2 py-1 text-sm outline-none"
        >
          <option value="">Pack default</option>
          <option value="thorough">Thorough</option>
          <option value="balanced">Balanced</option>
          <option value="off">Off</option>
        </select>
      </label>
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox" checked={sameCast}
          onChange={(e) => void save({ same_cast: e.target.checked })}
        />
        <span className="text-[var(--lp-text-dim)]">Keep the same team across rounds</span>
      </label>
      <label className="flex items-center gap-2">
        <span className="text-[var(--lp-text-dim)]">Max agents</span>
        <input
          type="number" min={1} max={10} value={maxAgents}
          onChange={(e) => {
            const n = Number(e.target.value)
            if (n >= 1 && n <= 10) void save({ max_agents: n })
          }}
          className="w-16 rounded bg-[var(--lp-bg)] border border-[var(--lp-border)] px-2 py-1 text-sm outline-none"
        />
      </label>
    </div>
  )
}

function RoundDiff({ prevSid, currTruth }: { prevSid: string; currTruth: string }) {
  const [prevTruth, setPrevTruth] = useState<string | null>(null)

  useEffect(() => {
    setPrevTruth(null)
    void api<Detail>(`/api/commissions/${prevSid}`)
      .then((d) => setPrevTruth(d.truth ?? ''))
      .catch(() => setPrevTruth(''))
  }, [prevSid])

  const lines = useMemo(
    () => prevTruth === null ? [] : collapseSame(lineDiff(prevTruth, currTruth)),
    [prevTruth, currTruth])

  if (prevTruth === null) {
    return <div className="text-sm text-[var(--lp-text-dim)] px-1">Comparing rounds…</div>
  }
  const changes = lines.filter((l) => l.type === 'add' || l.type === 'del').length
  return (
    <div className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] px-5 py-4">
      <div className="text-xs text-[var(--lp-text-dim)] mb-2">
        {changes === 0 ? 'The conclusions are identical.' : `${changes} changed line(s) vs the previous round.`}
      </div>
      <div className="font-mono text-xs leading-5 overflow-x-auto">
        {lines.map((l, i) => l.type === 'skip'
          ? <div key={i} className="text-[var(--lp-text-dim)] select-none">··· {l.count} unchanged lines ···</div>
          : (
            <div key={i} className={
              l.type === 'add' ? 'bg-emerald-950/50 text-emerald-300'
              : l.type === 'del' ? 'bg-red-950/40 text-red-400/90 line-through decoration-red-800'
              : 'text-[var(--lp-text-dim)]'}>
              <span className="select-none inline-block w-4">
                {l.type === 'add' ? '+' : l.type === 'del' ? '−' : ' '}
              </span>
              {l.text || ' '}
            </div>
          ))}
      </div>
    </div>
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
