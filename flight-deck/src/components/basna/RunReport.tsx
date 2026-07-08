import { useMemo, useState } from 'react'
import {
  AlertTriangle, Check, CornerDownRight, Download, Eye, FileText, Loader2,
  RefreshCw, ScanSearch, ThumbsDown, ThumbsUp, Wrench, ClipboardList, ListTree, Activity,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { AttachedFile, BasnaAnalysis, BasnaRun, BasnaSession, ProgressEvent, RunCost, VatraSubtask } from '../../stores/basnaStore'
import { CostCard } from '../CostCard'
import { VatraBlackboard } from '../VatraDelegation'
import { ProgressFeed } from './RunWorkspace'
import {
  analysisToMarkdown, Badge, downloadMarkdown, isViewable, OutputActions, slugify, WeightBar,
} from './shared'

// ── Run report: the finished run as one tabbed destination ──────────────────
// Report | Analysis | Gaps | Board | Files | Agents | Log — with the follow-up
// actions (continue / investigate blind spots / fill gaps) first-class in the header.

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

type TabId = 'report' | 'analysis' | 'gaps' | 'board' | 'files' | 'agents' | 'log'

export interface ContinueOpts { instruction: string; kind: string; sameCast: boolean }

export function RunReport({
  session, vatraMode, truth, confidence, method, analysis, runs, generatedFiles, runCost, subject,
  progress, subtasks, recompiling, onRecompile, onView, onViewFile, onDownloadFile, onFeedback,
  onDeepen, onFillGaps, deepening, onContinue,
}: {
  session: BasnaSession
  vatraMode: boolean
  truth: string
  confidence: number
  method?: string
  analysis: BasnaAnalysis | null
  runs: BasnaRun[]
  generatedFiles: AttachedFile[]
  runCost?: RunCost | null
  subject: string
  progress: ProgressEvent[]
  subtasks?: VatraSubtask[]
  recompiling: boolean
  onRecompile: () => void
  onView: (title: string, content: string) => void
  onViewFile: (name: string) => void
  onDownloadFile: (name: string) => void
  onFeedback: (runId: number, success: boolean) => void
  onDeepen: () => Promise<void>
  onFillGaps: () => Promise<void>
  deepening: boolean
  onContinue: (opts: ContinueOpts) => Promise<void>
}) {
  const hasAnalysis = !!analysis && !!(analysis.agreement?.length || analysis.differences?.length || analysis.unique?.length || analysis.blind_spots?.length)
  const hasGaps = vatraMode && !!analysis && !!(analysis.coverage_summary || analysis.gaps?.length)
  const done = session.status === 'done'
  const failed = session.status === 'error'

  const tabs = useMemo(() => {
    const t: { id: TabId; label: string; count?: number }[] = [{ id: 'report', label: vatraMode ? 'Report' : 'Truth' }]
    if (hasAnalysis) t.push({ id: 'analysis', label: 'Analysis' })
    if (hasGaps) t.push({ id: 'gaps', label: 'Gaps', count: analysis?.gaps?.length || 0 })
    if (vatraMode) t.push({ id: 'board', label: 'Board' })
    if (generatedFiles.length > 0) t.push({ id: 'files', label: 'Files', count: generatedFiles.length })
    if (runs.length > 0) t.push({ id: 'agents', label: 'Agents', count: runs.length })
    if (progress.length > 0) t.push({ id: 'log', label: 'Log' })
    return t
  }, [vatraMode, hasAnalysis, hasGaps, generatedFiles.length, runs.length, progress.length, analysis?.gaps?.length])

  const [tab, setTab] = useState<TabId>('report')
  const activeTab: TabId = tabs.some((t) => t.id === tab) ? tab : 'report'

  // Continue panel state — the next-round launcher lives in the report header.
  const [continuePanel, setContinuePanel] = useState(false)
  const [continueText, setContinueText] = useState('')
  const [continueKind, setContinueKind] = useState('continue')
  const [continueSameCast, setContinueSameCast] = useState(true)
  const [continuing, setContinuing] = useState(false)

  const deepenButton = done && truth && !!analysis?.blind_spots?.length && (
    <button
      onClick={() => { void onDeepen() }}
      disabled={deepening}
      title="Spawn a follow-up run focused on these blind spots, seeded with this run's result"
      className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
    >
      {deepening ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ScanSearch className="h-3.5 w-3.5" />}
      Investigate blind spots
    </button>
  )

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50">
      {/* Header: tabs left, outcome + actions right. */}
      <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-3 py-2">
        <div className="flex flex-wrap items-center gap-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                activeTab === t.id ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              {t.id === 'report' && <Check className={`h-3 w-3 ${failed ? 'text-rose-400' : 'text-emerald-500'}`} />}
              {t.id === 'analysis' && <ScanSearch className="h-3 w-3 text-violet-500" />}
              {t.id === 'gaps' && <AlertTriangle className="h-3 w-3 text-amber-500" />}
              {t.id === 'board' && <ClipboardList className="h-3 w-3 text-violet-500" />}
              {t.id === 'files' && <FileText className="h-3 w-3 text-zinc-500" />}
              {t.id === 'agents' && <ListTree className="h-3 w-3 text-zinc-500" />}
              {t.id === 'log' && <Activity className="h-3 w-3 text-zinc-500" />}
              {t.label}
              {typeof t.count === 'number' && t.count > 0 && (
                <span className="rounded bg-zinc-800/80 px-1 text-[10px] tabular-nums text-zinc-400">{t.count}</span>
              )}
            </button>
          ))}
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {truth && (
            <span className="flex items-center gap-2 text-[11px] text-zinc-500" title={vatraMode ? 'How completely the report covers the task' : 'Merge confidence'}>
              {vatraMode ? 'assembled' : 'confidence'} {(confidence * 100).toFixed(0)}%
              <span className="h-1.5 w-16 overflow-hidden rounded-full bg-zinc-800">
                <span className="block h-full rounded-full bg-emerald-500" style={{ width: `${Math.round(confidence * 100)}%` }} />
              </span>
            </span>
          )}
          {!vatraMode && method && <Badge className="text-sky-700 dark:text-sky-300">{method}</Badge>}
          {!vatraMode && truth && (
            <button
              onClick={onRecompile}
              disabled={recompiling}
              title="Recompile the truth + analysis from the agent outputs"
              className="flex items-center gap-1 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${recompiling ? 'animate-spin' : ''}`} />
            </button>
          )}
          {truth && <OutputActions title={`${subject} — ${vatraMode ? 'Final report' : 'Compiled truth'}`} content={truth} onView={onView} />}
          {done && truth && (
            <button
              onClick={() => { setContinuePanel((v) => !v); setContinueKind('continue'); setContinueSameCast(true) }}
              title="Carry this run forward into another round — same shared folder, building on this conclusion"
              className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1 text-xs font-medium text-white hover:bg-violet-500"
            >
              <CornerDownRight className="h-3.5 w-3.5" /> Continue…
            </button>
          )}
        </div>
      </div>

      {/* Continue launcher — next round, same shared folder & conclusion. */}
      {continuePanel && (
        <div className="border-b border-violet-300/60 bg-violet-50/40 px-4 py-3 dark:border-violet-500/30 dark:bg-violet-950/10">
          <div className="mb-2 flex items-center gap-2 text-[11px] text-zinc-500">
            <CornerDownRight className="h-3 w-3 text-violet-600 dark:text-violet-400" />
            next round · same shared folder &amp; conclusion
          </div>
          <textarea
            value={continueText}
            onChange={(e) => setContinueText(e.target.value)}
            rows={3}
            placeholder="What should the next round do? e.g. extend the research, write the next chapter, act on the conclusion…"
            className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 p-2.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-violet-500 focus:outline-none"
          />
          <div className="mt-2 flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-xs text-zinc-400">
              Mode
              <select
                value={continueKind}
                onChange={(e) => setContinueKind(e.target.value)}
                className="rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500 focus:outline-none"
              >
                <option value="continue">Continue (extend)</option>
                <option value="revise">Revise (improve)</option>
                {vatraMode
                  ? <option value="fill_gaps">Fill gaps</option>
                  : <option value="deepen">Deepen (blind spots)</option>}
              </select>
            </label>
            <label className="flex items-center gap-2 text-xs text-zinc-400" title="Reuse the prior run's team; uncheck to let the router pick a fresh team">
              <input
                type="checkbox"
                checked={continueSameCast}
                onChange={(e) => setContinueSameCast(e.target.checked)}
                className="h-3.5 w-3.5"
              />
              Same team
            </label>
            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={() => { setContinuePanel(false); setContinueText('') }}
                disabled={continuing}
                className="rounded-lg px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800 disabled:opacity-40"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  setContinuing(true)
                  try {
                    await onContinue({ instruction: continueText.trim(), kind: continueKind, sameCast: continueSameCast })
                    setContinuePanel(false); setContinueText('')
                  } catch { /* surfaced by store error path */ }
                  finally { setContinuing(false) }
                }}
                disabled={continuing || ((continueKind === 'continue' || continueKind === 'revise') && !continueText.trim())}
                title="Spawn the next round, seeded with this run's conclusion and shared files"
                className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
              >
                {continuing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CornerDownRight className="h-3.5 w-3.5" />}
                Start round
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="p-4">
        {/* Report tab */}
        {activeTab === 'report' && (
          <div className="space-y-3">
            {failed && (
              <div className="flex items-start gap-2 rounded-lg border border-rose-900/50 bg-rose-950/30 p-3 text-xs text-rose-300">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>
                  The run ended with an error.{runs.length > 0 ? ' The agent outputs below are saved — you can recompile from them.' : ' Adjust the task or setup and run it again.'}
                </span>
              </div>
            )}
            {/* Recovery: agents produced outputs but the merge didn't finish */}
            {!truth && runs.length > 0 && (
              <div className="flex items-center gap-3 rounded-lg border border-yellow-400 bg-yellow-100 p-4 dark:border-amber-500/40 dark:bg-amber-400/10">
                <AlertTriangle className="h-4 w-4 shrink-0 text-orange-600 dark:text-orange-400" />
                <span className="text-xs font-medium text-orange-700 dark:text-orange-300">
                  No compiled truth — the merge may have stalled or failed. The {runs.length} agent output(s) are saved; recompile from them.
                </span>
                <button
                  onClick={onRecompile}
                  disabled={recompiling}
                  className="ml-auto flex shrink-0 items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
                >
                  {recompiling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                  Compile truth
                </button>
              </div>
            )}
            {truth && (
              <div className="fd-markdown text-sm text-zinc-200 leading-relaxed">
                <Markdown remarkPlugins={[remarkGfm]}>{truth}</Markdown>
              </div>
            )}
            {runCost && <CostCard cost={runCost} />}
          </div>
        )}

        {/* Analysis tab */}
        {activeTab === 'analysis' && analysis && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Cross-agent analysis</span>
              <div className="ml-auto flex items-center gap-2">
                {deepenButton}
                <OutputActions title={`${subject} — Analysis`} content={analysisToMarkdown(analysis)} onView={onView} />
              </div>
            </div>
            {!!analysis.agreement?.length && (
              <div>
                <div className="mb-1 text-xs font-semibold text-emerald-700 dark:text-emerald-300">Agreement</div>
                <ul className="ml-4 list-disc space-y-1 text-sm text-zinc-200">
                  {analysis.agreement.map((a, i) => <li key={i}>{a}</li>)}
                </ul>
              </div>
            )}
            {!!analysis.differences?.length && (
              <div>
                <div className="mb-1 text-xs font-semibold text-amber-700 dark:text-amber-300">Key differences</div>
                <div className="space-y-2">
                  {analysis.differences.map((d, i) => (
                    <div key={i} className="text-sm">
                      <div className="text-zinc-200">{d.point}</div>
                      {!!d.positions?.length && (
                        <div className="ml-3 mt-0.5 space-y-0.5">
                          {d.positions.map((p, j) => (
                            <div key={j} className="text-zinc-400">
                              <span className="font-mono text-zinc-400">{p.by}</span> — {p.stance}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {!!analysis.unique?.length && (
              <div>
                <div className="mb-1 text-xs font-semibold text-sky-700 dark:text-sky-300">Unique insights</div>
                <ul className="ml-4 list-disc space-y-1 text-sm text-zinc-200">
                  {analysis.unique.map((u, i) => (
                    <li key={i}><span className="font-mono text-zinc-400">{u.by}</span> — {u.insight}</li>
                  ))}
                </ul>
              </div>
            )}
            {!!analysis.blind_spots?.length && (
              <div className="rounded-md border border-rose-300 bg-rose-50 p-3 dark:border-rose-900/40 dark:bg-rose-950/20">
                <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-rose-700 dark:text-rose-300">
                  <AlertTriangle className="h-3.5 w-3.5" /> Blind spots — covered by none
                </div>
                <ul className="ml-4 list-disc space-y-1 text-sm text-rose-800 dark:text-rose-200/90">
                  {analysis.blind_spots.map((b, i) => <li key={i}>{b}</li>)}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Coverage gaps tab (Vatra) */}
        {activeTab === 'gaps' && analysis && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Coverage gaps</span>
              {!analysis.gaps?.length && <Badge className="text-emerald-700 dark:text-emerald-300">complete</Badge>}
              {done && !!analysis.gaps?.length && (
                <button
                  onClick={() => { void onFillGaps() }}
                  disabled={deepening}
                  title="Spawn a follow-up Vatra that fills these gaps, seeded with this report"
                  className="ml-auto flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
                >
                  {deepening ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ScanSearch className="h-3.5 w-3.5" />}
                  Fill the gaps
                </button>
              )}
            </div>
            {analysis.coverage_summary && (
              <p className="text-xs text-zinc-400">{analysis.coverage_summary}</p>
            )}
            {!!analysis.gaps?.length && (
              <ul className="space-y-1.5">
                {analysis.gaps.map((g, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-zinc-300">
                    <span className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                      g.severity === 'major'
                        ? 'bg-rose-500/15 text-rose-700 dark:text-rose-300'
                        : 'bg-amber-500/15 text-amber-700 dark:text-amber-300'}`}>
                      {g.severity}
                    </span>
                    <span><span className="text-zinc-200">{g.item}</span>{g.note ? <span className="text-zinc-500"> — {g.note}</span> : null}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Shared board tab (Vatra) */}
        {activeTab === 'board' && (
          <VatraBlackboard sessionId={session.id} subtasks={subtasks} active={false} />
        )}

        {/* Generated files tab */}
        {activeTab === 'files' && (
          <div className="space-y-1">
            {generatedFiles.map((a) => (
              <div key={a.name} className="flex items-center gap-2 text-xs">
                <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                <span className="truncate text-zinc-300">{a.name}</span>
                {a.agent && (
                  <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400" title={`Generated by ${a.agent}`}>
                    {a.agent}
                  </span>
                )}
                <span className="shrink-0 text-zinc-600">{Math.max(1, Math.round(a.size / 1024))}kb</span>
                <div className="ml-auto flex shrink-0 items-center gap-0.5">
                  {isViewable(a.name) && (
                    <button
                      onClick={() => onViewFile(a.name)}
                      title="View"
                      className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                    >
                      <Eye className="h-3.5 w-3.5" />
                    </button>
                  )}
                  <button
                    onClick={() => onDownloadFile(a.name)}
                    title="Download"
                    className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                  >
                    <Download className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Agent runs tab */}
        {activeTab === 'agents' && (
          <div className="space-y-2">
            {runs.map((run) => (
              <AgentRow key={run.id} run={run} onFeedback={(s) => onFeedback(run.id, s)} onView={onView} />
            ))}
          </div>
        )}

        {/* Progress log tab */}
        {activeTab === 'log' && (
          <ProgressFeed progress={progress} running={false} defaultOpen />
        )}
      </div>
    </div>
  )
}
