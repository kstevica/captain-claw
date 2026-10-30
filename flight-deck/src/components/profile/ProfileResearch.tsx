import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, Check, FileText, Globe, Loader2, Search, Sparkles, Square } from 'lucide-react'
import {
  charCount,
  RESEARCH_STAGES,
  type ProfileCaps,
  type ProfileFields,
  type ProfileResearch as Research,
  type ResearchSource,
} from '../../services/profile'
import { newDrafts, sourcesSummary, timeLabel } from '../../utils/profileSources'

// Profile → Personal: "Fill it in for me". The person agrees, Flight Deck
// reads their documents and links, looks them and their company up on the
// web, and drafts About me / My company / My standing preferences on the
// team's model. The drafts wait here; "Use" puts one into its text box, where
// the person edits and saves it like any other text. Nothing reaches the
// agents before that.

const FIELD_LABELS: Record<keyof ProfileFields, string> = {
  about_me: 'About me',
  company: 'My company',
  instructions: 'My standing preferences',
}
const FIELDS = Object.keys(FIELD_LABELS) as (keyof ProfileFields)[]

const SOURCE_ICON: Record<ResearchSource['kind'], typeof Globe> = {
  document: FileText, link: Globe, page: Globe, search: Search,
}

export function ProfileResearch({ research, canStart, busy, error, caps, current, onStart, onStop, onUse }: {
  research: Research
  /** There is something to read from (a link, a company, a document). */
  canStart: boolean
  /** A start or stop is on its way. */
  busy: boolean
  /** Why the last start failed. */
  error: string | null
  caps: ProfileCaps
  /** What is in the text boxes now — a draft that matches needn't be used. */
  current: ProfileFields
  onStart: () => void
  onStop: () => void
  onUse: (fields: Partial<ProfileFields>) => void
}) {
  const running = research.state === 'running'
  const done = research.state === 'done' && !!research.drafts
  const failed = research.state === 'error' || research.state === 'cancelled'
  // Which drafts were put into the text boxes — of THIS run (a new run starts over).
  const [usedOf, setUsedOf] = useState<{ run: number | null; keys: Set<keyof ProfileFields> }>({ run: null, keys: new Set() })
  const used = usedOf.run === research.started ? usedOf.keys : new Set<keyof ProfileFields>()
  const logEnd = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (running) logEnd.current?.scrollIntoView({ block: 'nearest' })
  }, [running, research.log.length])

  const pending = newDrafts(research.drafts, current)
  const use = (fields: (keyof ProfileFields)[]) => {
    const picked: Partial<ProfileFields> = {}
    for (const k of fields) if (pending[k] !== undefined) picked[k] = pending[k]
    if (!Object.keys(picked).length) return
    onUse(picked)
    setUsedOf({ run: research.started, keys: new Set([...used, ...fields]) })
  }

  const where = research.model_source === 'team' ? ' (the team model)' : research.model_source === 'own' ? ' (your model)' : ''

  return (
    <div className="rounded-lg border border-violet-500/25 bg-violet-500/[0.04] p-5 space-y-4">
      <div className="flex items-start gap-2">
        <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-zinc-200">Fill it in for me</h3>
          <p className="mt-0.5 text-xs text-zinc-500">
            Flight Deck reads your documents and links, looks you and your company up on the web, and drafts
            the three texts below on the team's AI model. You see every word before it's saved — nothing reaches
            your agents until you save.
          </p>
        </div>
      </div>

      {/* Idle, or the last run's outcome in one line, and the button. */}
      {!running && (
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onStart}
            disabled={busy || !canStart}
            title={canStart ? undefined : 'Add a link, your company or a document first.'}
            className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
          >
            {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}
            {busy ? 'Starting…' : done || failed ? 'Research again' : 'Research and draft'}
          </button>
          <span className="text-[11px] text-zinc-600">
            {canStart
              ? 'A minute or two. Unsaved details are saved first; the run counts toward your daily AI allowance.'
              : 'Add a link, your company or a document above first.'}
          </span>
        </div>
      )}

      {/* Under way: the stage, the log, Stop. */}
      {running && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs text-zinc-300">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-500" />
            <span>{RESEARCH_STAGES[research.stage] ?? 'Working…'}</span>
            {research.model && <span className="text-zinc-600">· {research.model}{where}</span>}
            <button
              type="button"
              onClick={onStop}
              disabled={busy}
              className="ml-auto flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-400 hover:bg-zinc-800 disabled:opacity-50"
            >
              <Square className="h-3 w-3" /> Stop
            </button>
          </div>
          <LogList log={research.log} endRef={logEnd} />
        </div>
      )}

      {error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
          {error}
        </div>
      )}

      {/* Stopped or failed: why. */}
      {failed && !running && (
        <div className={`flex items-start gap-2 rounded-md border px-3 py-2 text-xs ${
          research.state === 'cancelled'
            ? 'border-zinc-700 bg-zinc-900/60 text-zinc-400'
            : 'border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-200'
        }`}>
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <div className="min-w-0">
            <div>{research.error || 'The run stopped.'}</div>
            {research.log.length > 0 && <LogDetails log={research.log} />}
          </div>
        </div>
      )}

      {/* Done: what it read, what it noted, the three drafts. */}
      {done && research.drafts && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-zinc-500">
            <span className="flex items-center gap-1 text-emerald-700 dark:text-emerald-300">
              <Check className="h-3.5 w-3.5" /> Drafted {timeLabel(research.finished)}
            </span>
            <span>{sourcesSummary(research.sources)}</span>
            {research.model && <span>on {research.model}{where}</span>}
            {research.tokens > 0 && <span>{research.tokens.toLocaleString()} tokens</span>}
          </div>
          {research.sources.length > 0 && (
            <ul className="flex flex-wrap gap-1.5">
              {research.sources.map((s, i) => {
                const Icon = SOURCE_ICON[s.kind]
                return (
                  <li
                    key={`${s.kind}-${s.name}-${i}`}
                    title={s.url ? `${s.url}${s.why ? ` — ${s.why}` : ''}` : s.why}
                    className={`flex max-w-full items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] ${
                      s.ok ? 'border-zinc-700 text-zinc-300' : 'border-amber-500/30 text-amber-700 line-through decoration-amber-500/60 dark:text-amber-300'
                    }`}
                  >
                    <Icon className="h-3 w-3 shrink-0" />
                    <span className="truncate">{s.kind === 'search' ? `“${s.name.replace(/^"|"$/g, '')}”` : s.name}</span>
                  </li>
                )
              })}
            </ul>
          )}
          {research.notes.length > 0 && (
            <ul className="space-y-0.5 rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-[11px] text-zinc-400">
              {research.notes.map((n) => <li key={n}>· {n}</li>)}
            </ul>
          )}
          <LogDetails log={research.log} />

          <div className="grid gap-3 lg:grid-cols-3">
            {FIELDS.map((k) => {
              const text = research.drafts?.[k] ?? ''
              const n = charCount(text)
              const inBox = !pending[k]
              return (
                <div key={k} className="flex min-w-0 flex-col rounded-md border border-zinc-800 bg-zinc-950/60">
                  <div className="flex items-baseline justify-between gap-2 border-b border-zinc-800 px-3 py-2">
                    <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">{FIELD_LABELS[k]}</span>
                    <span className={`text-[11px] tabular-nums ${n > caps[k] ? 'text-red-500' : 'text-zinc-600'}`}>
                      {n.toLocaleString()} / {caps[k].toLocaleString()}
                    </span>
                  </div>
                  {text ? (
                    <pre className="max-h-56 flex-1 overflow-y-auto whitespace-pre-wrap break-words px-3 py-2 font-sans text-xs leading-relaxed text-zinc-300">
                      {text}
                    </pre>
                  ) : (
                    <p className="flex-1 px-3 py-3 text-center text-[11px] text-zinc-600">Nothing found for this one.</p>
                  )}
                  <div className="border-t border-zinc-800 px-3 py-2">
                    <button
                      type="button"
                      onClick={() => use([k])}
                      disabled={!text || inBox}
                      className="flex items-center gap-1 rounded-md border border-zinc-700 px-2.5 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
                    >
                      {inBox && text ? <><Check className="h-3 w-3" /> {used.has(k) ? 'In the text box' : 'Same as the text box'}</> : 'Use this'}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => use(FIELDS)}
              disabled={!Object.keys(pending).length}
              className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
            >
              Use all three
            </button>
            <span className="text-[11px] text-zinc-600">
              Puts them into the text boxes below — read them over, change what you like, then Save.
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

function LogList({ log, endRef }: { log: Research['log']; endRef?: React.RefObject<HTMLDivElement | null> }) {
  return (
    <div className="max-h-40 space-y-0.5 overflow-y-auto rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2">
      {log.map((e, i) => (
        <div key={`${e.ts}-${i}`} className={`text-[11px] leading-snug ${e.kind === 'warn' ? 'text-amber-700 dark:text-amber-300' : 'text-zinc-400'}`}>
          {e.message}
        </div>
      ))}
      {endRef && <div ref={endRef} />}
    </div>
  )
}

function LogDetails({ log }: { log: Research['log'] }) {
  if (!log.length) return null
  return (
    <details className="text-[11px] text-zinc-500">
      <summary className="cursor-pointer select-none hover:text-zinc-300">What it did ({log.length} step{log.length === 1 ? '' : 's'})</summary>
      <div className="mt-1.5">
        <LogList log={log} />
      </div>
    </details>
  )
}
