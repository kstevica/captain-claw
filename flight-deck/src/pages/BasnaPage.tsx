import { useEffect, useRef, useState } from 'react'
import {
  Network, Play, Sparkles, Plus, Trash2, ThumbsUp, ThumbsDown,
  Loader2, Check, X, Wrench, Maximize2, Download, Paperclip, FileText, Image as ImageIcon,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useBasnaStore, type BasnaSession, type BasnaRun, type ProgressEvent } from '../stores/basnaStore'
import { useTierConfig, TIER_ORDER } from '../services/tierConfig'

// Demo / debug tasks — each exercises a different path through the pipeline.
const BASNA_EXAMPLES: { label: string; text: string }[] = [
  {
    label: 'EU expansion (complex)',
    text: `We're a 20-person B2B SaaS selling AI-powered contract-review software to US mid-market law firms (~$2M ARR, growing ~8%/month). Decide whether we should expand into the EU in the next two quarters.

Weigh, with evidence: (1) GDPR and the EU AI Act's implications for a legal-AI tool — obligations, timelines, and risk classification; (2) the competitive landscape in DACH and France — who's already there and how we'd differentiate; (3) data-residency and localization costs (hosting, language, support, legal); (4) the realistic revenue opportunity versus the distraction risk to our US growth.

End with ONE clear recommendation (go / no-go / phased) and the three specific conditions that would flip the decision.`,
  },
  {
    label: 'Data store choice',
    text: "We're adding an append-only event log: ~50M records/month, written continuously, queried by time-range and user_id, with daily aggregations. Pick ONE data store — Postgres, ClickHouse, or DynamoDB — and give the single most important reason.",
  },
  {
    label: 'Quick fact',
    text: 'What does the SQL keyword EXPLAIN do? One sentence.',
  },
  {
    label: 'Brainstorm options',
    text: 'Brainstorm ways to cut cold-start latency for a serverless API. List distinct approaches — breadth over depth.',
  },
  {
    label: 'Migration risk',
    text: 'Name the single biggest risk in migrating a monolith to microservices.',
  },
]

const DIFFICULTY_COLOR: Record<string, string> = {
  trivial: 'text-emerald-700 dark:text-emerald-300',
  moderate: 'text-amber-700 dark:text-amber-300',
  hard: 'text-rose-700 dark:text-rose-300',
}

function Badge({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={`rounded-full border border-zinc-700/60 bg-zinc-800/60 px-2 py-0.5 text-[11px] font-medium ${className}`}>
      {children}
    </span>
  )
}

function WeightBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
      <div className="h-full rounded-full bg-sky-500" style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  )
}

function SessionCard({ s, active, onOpen, onDelete }: {
  s: BasnaSession; active: boolean; onOpen: () => void; onDelete: () => void
}) {
  return (
    <button
      onClick={onOpen}
      className={`group w-full rounded-lg border p-2.5 text-left transition-colors ${
        active ? 'border-sky-600/60 bg-sky-950/30' : 'border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800/50'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="line-clamp-2 text-xs font-medium text-zinc-200">{s.intent || '(untitled)'}</p>
        <span
          onClick={(e) => { e.stopPropagation(); onDelete() }}
          className="shrink-0 rounded p-0.5 text-zinc-600 opacity-0 transition-opacity hover:text-rose-400 group-hover:opacity-100"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </span>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        {s.domain && <Badge className="text-zinc-300">{s.domain}</Badge>}
        {s.difficulty && <Badge className={DIFFICULTY_COLOR[s.difficulty] || 'text-zinc-300'}>{s.difficulty}</Badge>}
        <Badge className="text-zinc-400">{s.status}</Badge>
      </div>
    </button>
  )
}

function formatProgress(events: ProgressEvent[]): string {
  return events.map((e) => {
    const t = e.ts ? new Date(e.ts * 1000).toLocaleTimeString([], { hour12: false }) : ''
    return `${t}  ${e.stage.toUpperCase().padEnd(10)} ${e.message}`
  }).join('\n')
}

function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'basna'
}

function downloadMarkdown(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

// Fullscreen modal that renders markdown content, with an export-to-.md button.
function MarkdownModal({ title, content, onClose }: { title: string; content: string; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="flex max-h-[90vh] w-full max-w-4xl flex-col rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-800 px-4 py-3">
          <span className="truncate text-sm font-medium text-zinc-200">{title}</span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => downloadMarkdown(`${slugify(title)}.md`, content)}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-200 hover:bg-zinc-800"
            >
              <Download className="h-3.5 w-3.5" /> Export .md
            </button>
            <button onClick={onClose} className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        <div className="fd-markdown overflow-auto p-5 text-sm text-zinc-200 leading-relaxed">
          <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
        </div>
      </div>
    </div>
  )
}

// Compact fullscreen + export buttons reused by the truth card and agent cards.
function OutputActions({ title, content, onView }: { title: string; content: string; onView: (t: string, c: string) => void }) {
  return (
    <div className="flex items-center gap-1">
      <button onClick={() => onView(title, content)} title="Fullscreen" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
        <Maximize2 className="h-3.5 w-3.5" />
      </button>
      <button onClick={() => downloadMarkdown(`${slugify(title)}.md`, content)} title="Export .md" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
        <Download className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

function AgentRow({ run, onFeedback, onView }: { run: BasnaRun; onFeedback: (success: boolean) => void; onView: (t: string, c: string) => void }) {
  const scored = run.success !== null
  let actions: { tool: string; detail?: string }[] = []
  try { actions = JSON.parse(run.actions || '[]') } catch { actions = [] }
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-zinc-200">{run.role || run.archetype_id}</span>
          <Badge className="text-sky-700 dark:text-sky-300">{run.tier}</Badge>
          {scored && (
            run.success === 1
              ? <Badge className="text-emerald-700 dark:text-emerald-300">success</Badge>
              : <Badge className="text-rose-700 dark:text-rose-300">fail</Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          {run.output && <OutputActions title={run.role || run.archetype_id} content={run.output} onView={onView} />}
          <span className="text-[11px] text-zinc-500">{(run.latency_ms / 1000).toFixed(1)}s</span>
        </div>
      </div>
      <div className="mt-2 flex items-center gap-2">
        <span className="w-16 shrink-0 text-[11px] text-zinc-500">weight {run.weight_at_run.toFixed(2)}</span>
        <WeightBar value={run.weight_at_run} />
      </div>
      {run.output && (
        <div className="fd-markdown mt-2 max-h-48 overflow-auto text-xs text-zinc-400 leading-relaxed">
          <Markdown remarkPlugins={[remarkGfm]}>{run.output}</Markdown>
        </div>
      )}
      {actions.length > 0 && (
        <div className="mt-2 rounded-md border border-zinc-800 bg-zinc-950/40 p-2">
          <div className="mb-1 flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">Activity ({actions.length})</span>
            <button
              onClick={() => downloadMarkdown(
                `${slugify(run.role || run.archetype_id)}-activity.md`,
                actions.map((a) => `- ${a.tool}${a.detail ? ': ' + a.detail : ''}`).join('\n'),
              )}
              title="Export activity"
              className="rounded p-0.5 text-zinc-600 hover:text-zinc-300"
            >
              <Download className="h-3 w-3" />
            </button>
          </div>
          <div className="space-y-0.5">
            {actions.map((a, i) => (
              <div key={i} className="flex items-baseline gap-2 text-[11px]">
                <Wrench className="h-3 w-3 shrink-0 text-zinc-600" />
                <span className="font-mono text-zinc-400">{a.tool}</span>
                {a.detail && <span className="truncate text-zinc-600">{a.detail}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="mt-2 flex items-center gap-2">
        <span className="text-[11px] text-zinc-500">Was this contribution good?</span>
        <button
          onClick={() => onFeedback(true)}
          className={`rounded p-1 transition-colors ${run.success === 1 ? 'text-emerald-400' : 'text-zinc-500 hover:text-emerald-400'}`}
          title="Mark as good"
        >
          <ThumbsUp className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => onFeedback(false)}
          className={`rounded p-1 transition-colors ${run.success === 0 ? 'text-rose-400' : 'text-zinc-500 hover:text-rose-400'}`}
          title="Mark as poor"
        >
          <ThumbsDown className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}

export function BasnaPage() {
  const {
    sessions, activeSession, routePlan, runs, lastExecute, progress, attachments,
    routing, executing, error,
    routerTier, maxAgents, setRouterTier, setMaxAgents, addFiles, removeFile, downloadFile,
    loadSessions, selectSession, newSession, route, execute, sendFeedback, deleteSession,
  } = useBasnaStore()
  const { tiers, registry, envVars } = useTierConfig()

  const [intent, setIntent] = useState('')
  const [modal, setModal] = useState<{ title: string; content: string } | null>(null)
  const viewFull = (title: string, content: string) => setModal({ title, content })
  const progressRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const handlePaste = (e: React.ClipboardEvent) => {
    const imgs: File[] = []
    for (const it of Array.from(e.clipboardData?.items || [])) {
      if (it.type.startsWith('image/')) {
        const f = it.getAsFile()
        if (f) {
          const ext = it.type.split('/')[1] || 'png'
          imgs.push(f.name && f.name !== 'image.png' ? f : new File([f], `pasted-${Date.now()}.${ext}`, { type: it.type }))
        }
      }
    }
    if (imgs.length) addFiles(imgs)
  }

  // Keep the progress log pinned to the newest line as events stream in.
  useEffect(() => {
    const el = progressRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [progress.length])

  useEffect(() => { loadSessions() }, [loadSessions])
  useEffect(() => { setIntent(activeSession?.intent || '') }, [activeSession?.id, activeSession?.intent])

  const canRoute = intent.trim().length > 0 && !routing
  const canRun = !!routePlan && !!activeSession && !executing
  const truth = lastExecute?.truth ?? activeSession?.truth ?? ''
  const confidence = lastExecute?.confidence ?? activeSession?.confidence ?? 0

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex shrink-0 items-center gap-2 border-b border-zinc-700/50 bg-zinc-900/50 px-4 py-3 md:px-6">
        <Network className="h-5 w-5 text-sky-600 dark:text-sky-400" />
        <div>
          <h1 className="text-sm font-semibold text-zinc-100">Basna</h1>
          <p className="text-[11px] text-zinc-500">Route → spawn the minimal team → merge by reliability</p>
        </div>
        <button
          onClick={() => { newSession(); setIntent('') }}
          className="ml-auto flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
        >
          <Plus className="h-3.5 w-3.5" /> New
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Session list */}
        <div className="w-64 shrink-0 space-y-2 overflow-auto border-r border-zinc-800 p-3">
          {sessions.length === 0 && <p className="px-1 text-xs text-zinc-600">No runs yet.</p>}
          {sessions.map((s) => (
            <SessionCard
              key={s.id}
              s={s}
              active={activeSession?.id === s.id}
              onOpen={() => selectSession(s.id)}
              onDelete={() => deleteSession(s.id)}
            />
          ))}
        </div>

        {/* Detail */}
        <div className="flex-1 overflow-auto p-4 md:p-6">
          <div className="mx-auto max-w-3xl space-y-5">
            {/* Intent + controls */}
            <div
              className={`rounded-lg border bg-zinc-900/50 p-4 ${dragOver ? 'border-sky-500' : 'border-zinc-800'}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files) }}
            >
              <label className="mb-1.5 block text-xs font-medium text-zinc-400">Task / intent</label>
              <textarea
                value={intent}
                onChange={(e) => setIntent(e.target.value)}
                onPaste={handlePaste}
                rows={9}
                placeholder="Describe the task, or attach/drop/paste files. The router picks the smallest team that can answer it well."
                className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 p-2.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
              />

              {/* Attachments */}
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
                >
                  <Paperclip className="h-3.5 w-3.5" /> Attach
                </button>
                <input
                  ref={fileInputRef} type="file" multiple className="hidden"
                  onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }}
                />
                {attachments.filter((a) => a.kind !== 'generated').map((a) => (
                  <span key={a.name} className="flex items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-800/60 px-2 py-0.5 text-[11px] text-zinc-300">
                    {a.mime.startsWith('image/') ? <ImageIcon className="h-3 w-3 text-zinc-500" /> : <FileText className="h-3 w-3 text-zinc-500" />}
                    {a.name}
                    <span className="text-zinc-600">{Math.max(1, Math.round(a.size / 1024))}kb</span>
                    <button onClick={() => removeFile(a.name)} className="text-zinc-500 hover:text-rose-400"><X className="h-3 w-3" /></button>
                  </span>
                ))}
              </div>
              {!intent.trim() && (
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <span className="text-[11px] text-zinc-600">Try:</span>
                  {BASNA_EXAMPLES.map((ex) => (
                    <button
                      key={ex.label}
                      onClick={() => setIntent(ex.text)}
                      title={ex.text}
                      className="rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-1 text-[11px] text-sky-300 hover:bg-sky-500/20 transition-colors"
                    >
                      {ex.label}
                    </button>
                  ))}
                </div>
              )}
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2 text-xs text-zinc-400">
                  Router tier
                  <select
                    value={routerTier}
                    onChange={(e) => setRouterTier(e.target.value)}
                    title="Which Library tier picks the archetypes"
                    className="rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-sky-600 focus:outline-none"
                  >
                    {TIER_ORDER.filter((t) => tiers[t]).map((t) => (
                      <option key={t} value={t}>{registry?.tiers[t]?.label || t}</option>
                    ))}
                    {Object.keys(tiers).length === 0 && <option value="reason">reason</option>}
                  </select>
                </label>
                <label className="flex items-center gap-2 text-xs text-zinc-400">
                  Max agents
                  <input
                    type="number" min={1} max={10} value={maxAgents}
                    onChange={(e) => setMaxAgents(Number(e.target.value))}
                    className="w-16 rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-sky-600 focus:outline-none"
                  />
                </label>
                <div className="ml-auto flex items-center gap-2">
                  <button
                    onClick={() => route(intent, tiers)}
                    disabled={!canRoute}
                    className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800 disabled:opacity-40"
                  >
                    {routing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                    Route
                  </button>
                  <button
                    onClick={() => execute(tiers, envVars)}
                    disabled={!canRun}
                    className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
                  >
                    {executing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                    Run ensemble
                  </button>
                </div>
              </div>
              {error && (
                <div className="mt-3 flex items-start gap-2 rounded-lg border border-rose-900/50 bg-rose-950/30 p-2.5 text-xs text-rose-300">
                  <X className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {error}
                </div>
              )}
            </div>

            {/* Route plan */}
            {routePlan && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Route plan</span>
                  <Badge className="text-zinc-300">{routePlan.domain}</Badge>
                  <Badge className={DIFFICULTY_COLOR[routePlan.difficulty] || 'text-zinc-300'}>{routePlan.difficulty}</Badge>
                  <Badge className="text-sky-700 dark:text-sky-300">{routePlan.merge_kind}</Badge>
                  {routePlan.source && <Badge className="text-zinc-500">{routePlan.source}</Badge>}
                  <span className="ml-auto text-[11px] text-zinc-600">{routePlan.selected.length} agent(s)</span>
                </div>
                {routePlan.rationale && <p className="mb-3 text-xs text-zinc-400">{routePlan.rationale}</p>}
                <div className="space-y-2">
                  {routePlan.selected.map((sel) => {
                    const tc = tiers[sel.tier]
                    return (
                    <div key={sel.archetype_id} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-medium text-zinc-200">{sel.role || sel.archetype_id}</span>
                        <Badge className="text-sky-700 dark:text-sky-300">{sel.tier}</Badge>
                      </div>
                      <p className="mt-0.5 font-mono text-[11px] text-zinc-600">
                        {tc?.model ? `${tc.provider}/${tc.model}` : `${sel.tier} tier (model from server)`}
                      </p>
                      {sel.why && <p className="mt-1 text-xs text-zinc-500">{sel.why}</p>}
                      <div className="mt-2 flex items-center gap-2">
                        <span className="w-20 shrink-0 text-[11px] text-zinc-500">prior {sel.prior_weight.toFixed(2)}</span>
                        <WeightBar value={sel.prior_weight} />
                      </div>
                    </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Live progress log */}
            {(executing || progress.length > 0) && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="mb-2 flex items-center gap-2">
                  {executing
                    ? <Loader2 className="h-4 w-4 animate-spin text-sky-400" />
                    : <Check className="h-4 w-4 text-emerald-500" />}
                  <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Progress</span>
                  {progress.length > 30 && (
                    <span className="text-[10px] text-zinc-600">showing last 30 of {progress.length}</span>
                  )}
                  <button
                    onClick={() => downloadMarkdown('basna-progress.md', formatProgress(progress))}
                    title="Export progress log"
                    className="ml-auto rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                  >
                    <Download className="h-3.5 w-3.5" />
                  </button>
                </div>
                <div ref={progressRef} className="max-h-72 space-y-1 overflow-auto">
                  {progress.slice(-30).map((ev) => (
                    <div key={ev.i} className="flex items-baseline gap-2 text-xs">
                      <span className="shrink-0 font-mono text-[10px] tabular-nums text-zinc-600">
                        {ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString([], { hour12: false }) : ''}
                      </span>
                      <span className="w-16 shrink-0 font-mono text-[10px] uppercase text-zinc-600">{ev.stage}</span>
                      <span className={
                        ev.stage === 'narration' ? 'text-zinc-100 font-medium'
                          : ev.ok === false ? 'text-rose-400'
                          : 'text-zinc-400'
                      }>{ev.message}</span>
                    </div>
                  ))}
                  {executing && progress.length === 0 && (
                    <div className="text-xs text-zinc-500">Starting…</div>
                  )}
                </div>
              </div>
            )}

            {/* Truth */}
            {truth && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <Check className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                  <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Compiled truth</span>
                  {lastExecute?.method && <Badge className="text-sky-700 dark:text-sky-300">{lastExecute.method}</Badge>}
                  <span className="ml-auto flex items-center gap-2 text-[11px] text-zinc-500">
                    confidence {(confidence * 100).toFixed(0)}%
                    <span className="h-1.5 w-20 overflow-hidden rounded-full bg-zinc-800">
                      <span className="block h-full rounded-full bg-emerald-500" style={{ width: `${Math.round(confidence * 100)}%` }} />
                    </span>
                  </span>
                  <OutputActions title="Compiled truth" content={truth} onView={viewFull} />
                </div>
                <div className="fd-markdown text-sm text-zinc-200 leading-relaxed">
                  <Markdown remarkPlugins={[remarkGfm]}>{truth}</Markdown>
                </div>
              </div>
            )}

            {/* Agent runs */}
            {runs.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Agents ({runs.length})</span>
                {runs.map((run) => (
                  <AgentRow key={run.id} run={run} onFeedback={(s) => sendFeedback(run.id, s)} onView={viewFull} />
                ))}
              </div>
            )}

            {/* Generated files */}
            {attachments.some((a) => a.kind === 'generated') && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">Generated files</div>
                <div className="space-y-1">
                  {attachments.filter((a) => a.kind === 'generated').map((a) => (
                    <div key={a.name} className="flex items-center gap-2 text-xs">
                      <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                      <span className="truncate text-zinc-300">{a.name}</span>
                      <span className="shrink-0 text-zinc-600">{Math.max(1, Math.round(a.size / 1024))}kb</span>
                      <button
                        onClick={() => downloadFile(a.name)}
                        title="Download"
                        className="ml-auto shrink-0 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                      >
                        <Download className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {modal && <MarkdownModal title={modal.title} content={modal.content} onClose={() => setModal(null)} />}
    </div>
  )
}
