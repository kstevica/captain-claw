import { useEffect, useRef, useState } from 'react'
import {
  X, Sparkles, Loader2, Gauge, ScanSearch, ListChecks,
  Brain, ArrowRight, ArrowLeft, Wand2, Check, Paperclip, FileText, Image as ImageIcon, FolderSearch, ChevronDown,
} from 'lucide-react'
import { useBasnaStore, apiRecommend, type Recommendation, type BasnaSession } from '../../stores/basnaStore'
import { applyPreset } from '../../services/quality'
import { KnowledgePicker } from './KnowledgePicker'
import { ReferenceFolderPicker } from './ReferenceFolderPicker'
import { Button, Segmented, inputCls, labelCls } from '../console/parts'
import { IconSquare, SectionLabel } from '../basna/look'
import { MODE_LOOK, MODE_TILE_ON, checkCls, chipToggleCls, ctaCls, iconBtnCls, smallInputCls } from '../basna/lookCls'

type Mode = 'basna' | 'vatra'
type Effort = 'standard' | 'deep' | 'plan'
type QLevel = 'basic' | 'balanced' | 'thorough'

interface Creds { provider?: string; model?: string; api_key?: string; base_url?: string }

const STEPS = ['Task', 'Setup', 'Knowledge'] as const

export function BasnaWizardModal({
  sessions,
  creds,
  mode: lockMode,
  onClose,
  onPrepare,
}: {
  sessions: BasnaSession[]
  creds: Creds
  /** The page is locked to this mode: no Basna/Vatra choice here. */
  mode?: Mode
  onClose: () => void
  onPrepare: (args: { intent: string; title: string; mode: Mode }) => void
}) {
  const s = useBasnaStore()
  const [step, setStep] = useState(0)
  const [intent, setIntent] = useState('')
  const [title, setTitle] = useState('')
  const [mode, setMode] = useState<Mode>(lockMode ?? 'vatra')
  const [effort, setEffort] = useState<Effort>('standard')
  const [qlevel, setQlevel] = useState<QLevel>('balanced')
  const [maxAgents, setMaxAgents] = useState(4)
  const [grouped, setGrouped] = useState(true)
  const [sharedDatastore, setSharedDatastore] = useState(true)
  const [useKnowledge, setUseKnowledge] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [rec, setRec] = useState<Recommendation | null>(null)
  const [err, setErr] = useState('')
  const [showRefFolders, setShowRefFolders] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  // Load VFS folders once so the reference picker is ready on the Knowledge step.
  useEffect(() => { s.loadProjects() }, [s.loadProjects])

  // Attachments live in the store so onPrepare's route/plan upload them (same as
  // the manual form). Paste of images is supported like the main chat inputs.
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
    if (imgs.length) s.addFiles(imgs)
  }
  const files = s.attachments.filter((a) => a.kind !== 'generated')

  const applyRec = (r: Recommendation) => {
    setRec(r)
    if (!lockMode) setMode(r.mode)
    setEffort(r.effort)
    setQlevel(r.quality)
    setMaxAgents(r.max_agents)
    setGrouped(r.grouped)
    setSharedDatastore(r.shared_datastore)
  }

  const analyze = async () => {
    if (!intent.trim()) return
    setAnalyzing(true)
    setErr('')
    try {
      applyRec(await apiRecommend(intent.trim(), creds as Record<string, unknown>))
      setStep(1)
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'analysis failed')
    } finally {
      setAnalyzing(false)
    }
  }

  const finish = () => {
    // Apply the wizard's choices to the store, then hand off to the page to
    // prepare the run (which reads intent/mode + these store options).
    s.setQuality(applyPreset(qlevel === 'basic' ? 'off' : qlevel, s.quality))
    s.setMaxAgents(maxAgents)
    if (effort === 'deep') s.setDeep(true)
    else if (effort === 'plan') s.setPlanMode(true)
    else { s.setDeep(false); s.setPlanMode(false) }
    s.setExecutionGroups(mode === 'vatra' ? grouped : false)
    s.setSharedDatastore(sharedDatastore)
    if (!useKnowledge) s.setKnowledgeSessionIds([])
    onPrepare({ intent: intent.trim(), title: title.trim(), mode })
  }

  const canNext = step === 0 ? intent.trim().length > 0 : true

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="New run wizard"
        className="flex max-h-[92dvh] w-full flex-col rounded-t-2xl border border-zinc-800 bg-zinc-950 shadow-2xl wide:max-w-[640px] wide:rounded-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 border-b border-zinc-800 px-5 py-4">
          <IconSquare icon={Wand2} tone="violet" size="sm" />
          <h2 className="min-w-0 flex-1 truncate text-base font-semibold text-zinc-100">New run wizard</h2>
          <button onClick={onClose} aria-label="Close" title="Close" className={iconBtnCls}>
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Step rail */}
        <div className="flex items-center gap-1.5 border-b border-zinc-800/70 px-5 py-3">
          {STEPS.map((label, i) => (
            <div key={label} className="flex items-center gap-1.5">
              <button
                onClick={() => i < step && setStep(i)}
                className={`flex items-center gap-1.5 rounded-full px-2 py-1 text-xs font-medium transition-colors ${
                  i === step ? 'bg-zinc-800 text-zinc-100' : i < step ? 'text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200' : 'cursor-default text-zinc-600'
                }`}
              >
                <span className={`flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-semibold ${
                  i < step ? 'bg-emerald-600 text-white' : i === step ? 'bg-violet-600 text-white' : 'border border-zinc-700 text-zinc-500'
                }`}>
                  {i < step ? <Check className="h-2.5 w-2.5" /> : i + 1}
                </span>
                {label}
              </button>
              {i < STEPS.length - 1 && <span className="text-zinc-700" aria-hidden>›</span>}
            </div>
          ))}
        </div>

        {/* Content */}
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
          {step === 0 && (
            <div className="space-y-4">
              <div>
                <label className={labelCls}>Title <span className="font-normal text-zinc-600">— optional</span></label>
                <input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g. Q3 competitor scan"
                  className={inputCls}
                />
              </div>
              <div>
                <label className={labelCls}>What do you want to accomplish?</label>
                <textarea
                  value={intent}
                  onChange={(e) => setIntent(e.target.value)}
                  onPaste={handlePaste}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => { e.preventDefault(); if (e.dataTransfer.files?.length) s.addFiles(e.dataTransfer.files) }}
                  rows={6}
                  autoFocus
                  placeholder={lockMode
                    ? 'Describe the task, or attach/drop/paste files. The wizard can recommend options.'
                    : 'Describe the task, or attach/drop/paste files. The wizard can recommend Basna vs Vatra and options.'}
                  className={`${inputCls} resize-y leading-relaxed`}
                />
                <div className="mt-2.5 flex flex-wrap items-center gap-2">
                  <Button icon={Paperclip} onClick={() => fileRef.current?.click()} className="px-3 py-1.5 text-xs">
                    Attach
                  </Button>
                  <input
                    ref={fileRef} type="file" multiple className="hidden"
                    onChange={(e) => { if (e.target.files) s.addFiles(e.target.files); e.target.value = '' }}
                  />
                  {files.map((a) => (
                    <span key={a.name} className="flex max-w-full items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-900/60 py-1 pl-2.5 pr-1.5 text-xs text-zinc-300">
                      {a.mime.startsWith('image/') ? <ImageIcon className="h-3.5 w-3.5 shrink-0 text-zinc-500" /> : <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />}
                      <span className="truncate">{a.name}</span>
                      <button onClick={() => s.removeFile(a.name)} aria-label={`Remove ${a.name}`} className="shrink-0 rounded-full p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-red-600 dark:hover:text-red-400"><X className="h-3 w-3" /></button>
                    </span>
                  ))}
                </div>
              </div>
              {err && <p className="text-xs text-red-600 dark:text-red-400">{err}</p>}
              <button
                onClick={analyze}
                disabled={!intent.trim() || analyzing}
                className={`${ctaCls('violet')} w-full`}
              >
                {analyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                {analyzing ? 'Analyzing the task…' : 'Analyze & recommend a setup'}
              </button>
              <p className="text-center text-[11px] text-zinc-500">One quick LLM call · or skip to set it up yourself →</p>
            </div>
          )}

          {step === 1 && (
            <div className="space-y-5">
              {rec && (
                <div className="flex items-start gap-2.5 rounded-xl border border-violet-500/30 bg-violet-500/[0.07] px-3.5 py-3">
                  <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-violet-600 dark:text-violet-400" />
                  <p className="text-xs leading-relaxed text-violet-700 dark:text-violet-200">{rec.rationale || 'Recommended setup below — tweak anything.'}</p>
                </div>
              )}
              {/* Mode — only where the page isn't locked to one */}
              {!lockMode && (
                <div>
                  <SectionLabel className="mb-2">Mode</SectionLabel>
                  <div className="grid gap-2.5 wide:grid-cols-2">
                    {([
                      { id: 'basna', sub: 'Independent ensemble — answers merged by reliability' },
                      { id: 'vatra', sub: 'Collaborative team — a Lead splits work, a reporter assembles' },
                    ] as const).map((m) => {
                      const ml = MODE_LOOK[m.id]
                      const sel = mode === m.id
                      return (
                        <button
                          key={m.id}
                          onClick={() => setMode(m.id)}
                          aria-pressed={sel}
                          className={`flex min-w-0 items-start gap-3 rounded-2xl border p-3 text-left transition-colors ${
                            sel ? MODE_TILE_ON[ml.tone] : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-800/60'}`}
                        >
                          <IconSquare icon={ml.icon} tone={sel ? ml.tone : 'zinc'} size="sm" />
                          <span className="min-w-0 flex-1">
                            <span className="flex items-center gap-1.5">
                              <span className="text-sm font-semibold text-zinc-100">{ml.name}</span>
                              {rec?.mode === m.id && <span className="ml-auto rounded-full bg-violet-500/15 px-1.5 text-[10px] font-medium leading-4 text-violet-700 dark:text-violet-300">suggested</span>}
                            </span>
                            <span className="mt-0.5 block text-xs leading-snug text-zinc-500">{m.sub}</span>
                          </span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}
              {/* Effort */}
              <Row label="Effort">
                <Segmented
                  label="Effort"
                  value={effort}
                  onChange={setEffort}
                  options={[
                    { value: 'standard', label: <><Gauge className="h-3.5 w-3.5" />Standard</> },
                    { value: 'deep', label: <><ScanSearch className="h-3.5 w-3.5" />Deep</> },
                    { value: 'plan', label: <><ListChecks className="h-3.5 w-3.5" />Plan</> },
                  ]}
                />
              </Row>
              {/* Quality */}
              <Row label="Quality">
                <Segmented
                  label="Quality"
                  value={qlevel}
                  onChange={setQlevel}
                  options={[
                    { value: 'basic', label: 'Basic' },
                    { value: 'balanced', label: 'Balanced' },
                    { value: 'thorough', label: 'Thorough' },
                  ]}
                />
              </Row>
              {/* Team size + toggles */}
              <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 text-sm">
                <label className="flex items-center gap-2 text-zinc-400">
                  Max agents
                  <input
                    type="number" min={1} max={10} value={maxAgents}
                    onChange={(e) => setMaxAgents(Math.max(1, Math.min(10, Number(e.target.value) || 1)))}
                    className={`${smallInputCls} w-16`}
                  />
                </label>
                {mode === 'vatra' && (
                  <Toggle checked={grouped} onChange={setGrouped} label="Grouped phases" />
                )}
                <Toggle checked={sharedDatastore} onChange={setSharedDatastore} label="Shared datastore" />
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-4">
              <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3">
                <input
                  type="checkbox"
                  checked={useKnowledge}
                  onChange={(e) => setUseKnowledge(e.target.checked)}
                  className={`${checkCls} mt-0.5`}
                />
                <span>
                  <span className="flex items-center gap-1.5 text-sm font-medium text-zinc-200">
                    <Brain className="h-4 w-4 text-violet-600 dark:text-violet-400" /> Seed with prior runs' knowledge
                  </span>
                  <span className="mt-0.5 block text-xs leading-relaxed text-zinc-500">
                    Fold earlier runs' reports + gaps/blind spots into this run's instructions. Optional — it adds tokens (cost) to every agent.
                  </span>
                </span>
              </label>
              {useKnowledge && (
                <KnowledgePicker
                  sessions={sessions}
                  selectedIds={s.knowledgeSessionIds}
                  onToggle={s.toggleKnowledgeSession}
                  includeBoard={s.knowledgeIncludeBoard}
                  onIncludeBoard={s.setKnowledgeIncludeBoard}
                />
              )}
              {/* Reference folders (read-only) */}
              <div>
                <button
                  onClick={() => setShowRefFolders((v) => !v)}
                  aria-expanded={showRefFolders}
                  className={chipToggleCls(s.referenceFolders.length > 0, 'emerald')}
                >
                  <FolderSearch className="h-4 w-4" />
                  {s.referenceFolders.length > 0 ? `Reference folders: ${s.referenceFolders.length}` : 'Reference folders (read-only)'}
                  <ChevronDown className={`h-3.5 w-3.5 opacity-60 transition-transform ${showRefFolders ? 'rotate-180' : ''}`} />
                </button>
                {showRefFolders && (
                  <div className="mt-3 rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
                    <ReferenceFolderPicker projects={s.projects} selected={s.referenceFolders} onToggle={s.toggleReferenceFolder} />
                  </div>
                )}
              </div>
              {/* Summary */}
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 text-xs leading-relaxed text-zinc-400">
                <SectionLabel className="mb-1">Summary</SectionLabel>
                <span className="font-medium text-zinc-200">{mode === 'vatra' ? 'Vatra' : 'Basna'}</span> · {effort} · {qlevel} quality · up to {maxAgents} agents
                {mode === 'vatra' && grouped ? ' · grouped' : ''}{sharedDatastore ? ' · shared datastore' : ''}
                {useKnowledge && s.knowledgeSessionIds.length ? ` · ${s.knowledgeSessionIds.length} prior run(s)` : ''}
                {s.referenceFolders.length ? ` · ${s.referenceFolders.length} reference folder(s)` : ''}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between gap-2 border-t border-zinc-800 px-5 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
          <Button onClick={() => (step === 0 ? onClose() : setStep((x) => x - 1))}>
            {step === 0 ? 'Cancel' : <><ArrowLeft className="h-4 w-4" /> Back</>}
          </Button>
          {step < STEPS.length - 1 ? (
            <Button onClick={() => setStep((x) => x + 1)} disabled={!canNext}>
              {step === 0 ? 'Set up manually' : 'Next'} <ArrowRight className="h-4 w-4" />
            </Button>
          ) : (
            <button
              onClick={finish}
              disabled={!intent.trim()}
              className={ctaCls(mode === 'vatra' ? 'violet' : 'sky')}
            >
              <Sparkles className="h-4 w-4" /> {mode === 'vatra' ? 'Create & plan team' : 'Create & route'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <SectionLabel className="w-16 shrink-0">{label}</SectionLabel>
      {children}
    </div>
  )
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-zinc-300">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className={checkCls}
      />
      {label}
    </label>
  )
}
