import { useEffect, useRef, useState } from 'react'
import {
  X, Sparkles, Loader2, Network, Users, Gauge, ScanSearch, ListChecks,
  Brain, ArrowRight, ArrowLeft, Wand2, Check, Paperclip, FileText, Image as ImageIcon, FolderSearch,
} from 'lucide-react'
import { useBasnaStore, apiRecommend, type Recommendation, type BasnaSession } from '../../stores/basnaStore'
import { applyPreset } from '../../services/quality'
import { KnowledgePicker } from './KnowledgePicker'
import { ReferenceFolderPicker } from './ReferenceFolderPicker'

type Mode = 'basna' | 'vatra'
type Effort = 'standard' | 'deep' | 'plan'
type QLevel = 'basic' | 'balanced' | 'thorough'

interface Creds { provider?: string; model?: string; api_key?: string; base_url?: string }

const STEPS = ['Task', 'Setup', 'Knowledge'] as const

export function BasnaWizardModal({
  sessions,
  creds,
  onClose,
  onPrepare,
}: {
  sessions: BasnaSession[]
  creds: Creds
  onClose: () => void
  onPrepare: (args: { intent: string; title: string; mode: Mode }) => void
}) {
  const s = useBasnaStore()
  const [step, setStep] = useState(0)
  const [intent, setIntent] = useState('')
  const [title, setTitle] = useState('')
  const [mode, setMode] = useState<Mode>('vatra')
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
    setMode(r.mode)
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="flex max-h-[88vh] w-[640px] flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div className="flex items-center gap-2">
            <Wand2 className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-semibold text-zinc-100">New run wizard</h2>
          </div>
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Step rail */}
        <div className="flex items-center gap-2 border-b border-zinc-800/60 px-5 py-2.5">
          {STEPS.map((label, i) => (
            <div key={label} className="flex items-center gap-2">
              <button
                onClick={() => i < step && setStep(i)}
                className={`flex items-center gap-1.5 text-[11px] font-medium ${
                  i === step ? 'text-zinc-100' : i < step ? 'text-zinc-400 hover:text-zinc-200' : 'text-zinc-600'
                }`}
              >
                <span className={`flex h-4 w-4 items-center justify-center rounded-full text-[9px] ${
                  i < step ? 'bg-emerald-600 text-white' : i === step ? 'bg-violet-600 text-white' : 'border border-zinc-700 text-zinc-600'
                }`}>
                  {i < step ? <Check className="h-2.5 w-2.5" /> : i + 1}
                </span>
                {label}
              </button>
              {i < STEPS.length - 1 && <span className="text-zinc-700">›</span>}
            </div>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {step === 0 && (
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-zinc-400">Title <span className="font-normal text-zinc-600">— optional</span></label>
                <input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g. Q3 competitor scan"
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-zinc-400">What do you want to accomplish?</label>
                <textarea
                  value={intent}
                  onChange={(e) => setIntent(e.target.value)}
                  onPaste={handlePaste}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => { e.preventDefault(); if (e.dataTransfer.files?.length) s.addFiles(e.dataTransfer.files) }}
                  rows={6}
                  autoFocus
                  placeholder="Describe the task, or attach/drop/paste files. The wizard can recommend Basna vs Vatra and options."
                  className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 p-2.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
                />
                <div className="mt-1.5 flex flex-wrap items-center gap-2">
                  <button
                    onClick={() => fileRef.current?.click()}
                    className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
                  >
                    <Paperclip className="h-3.5 w-3.5" /> Attach
                  </button>
                  <input
                    ref={fileRef} type="file" multiple className="hidden"
                    onChange={(e) => { if (e.target.files) s.addFiles(e.target.files); e.target.value = '' }}
                  />
                  {files.map((a) => (
                    <span key={a.name} className="flex items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-800/60 px-2 py-0.5 text-[11px] text-zinc-300">
                      {a.mime.startsWith('image/') ? <ImageIcon className="h-3 w-3 text-zinc-500" /> : <FileText className="h-3 w-3 text-zinc-500" />}
                      {a.name}
                      <button onClick={() => s.removeFile(a.name)} className="text-zinc-500 hover:text-rose-400"><X className="h-3 w-3" /></button>
                    </span>
                  ))}
                </div>
              </div>
              {err && <p className="text-[11px] text-red-400">{err}</p>}
              <button
                onClick={analyze}
                disabled={!intent.trim() || analyzing}
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-violet-600 px-3 py-2 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
              >
                {analyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                {analyzing ? 'Analyzing the task…' : 'Analyze & recommend a setup'}
              </button>
              <p className="text-center text-[10px] text-zinc-600">One quick LLM call · or skip to set it up yourself →</p>
            </div>
          )}

          {step === 1 && (
            <div className="space-y-4">
              {rec && (
                <div className="flex items-start gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2">
                  <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0 text-violet-400" />
                  <p className="text-[11px] text-violet-700 dark:text-violet-200">{rec.rationale || 'Recommended setup below — tweak anything.'}</p>
                </div>
              )}
              {/* Mode */}
              <div>
                <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Mode</div>
                <div className="grid grid-cols-2 gap-2">
                  {([
                    { id: 'basna', Icon: Network, name: 'Basna', sub: 'Independent ensemble — answers merged by reliability', on: 'border-sky-400 bg-sky-50 dark:border-sky-500/70 dark:bg-sky-950/30', dot: 'text-sky-600 dark:text-sky-400' },
                    { id: 'vatra', Icon: Users, name: 'Vatra', sub: 'Collaborative team — a Lead splits work, a reporter assembles', on: 'border-violet-400 bg-violet-50 dark:border-violet-500/70 dark:bg-violet-950/30', dot: 'text-violet-600 dark:text-violet-400' },
                  ] as const).map((m) => {
                    const sel = mode === m.id
                    return (
                      <button
                        key={m.id}
                        onClick={() => setMode(m.id)}
                        className={`rounded-lg border p-2.5 text-left transition-colors ${sel ? m.on : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700'}`}
                      >
                        <div className="flex items-center gap-1.5">
                          <m.Icon className={`h-4 w-4 ${sel ? m.dot : 'text-zinc-500'}`} />
                          <span className="text-xs font-semibold text-zinc-200">{m.name}</span>
                          {rec?.mode === m.id && <span className="ml-auto rounded bg-violet-500/20 px-1.5 py-0.5 text-[9px] font-medium text-violet-700 dark:text-violet-300">suggested</span>}
                        </div>
                        <p className="mt-1 text-[10px] leading-snug text-zinc-500">{m.sub}</p>
                      </button>
                    )
                  })}
                </div>
              </div>
              {/* Effort */}
              <Row label="Effort">
                <Segmented
                  options={[
                    { id: 'standard', label: 'Standard', Icon: Gauge },
                    { id: 'deep', label: 'Deep', Icon: ScanSearch },
                    { id: 'plan', label: 'Plan', Icon: ListChecks },
                  ]}
                  value={effort}
                  onChange={(v) => setEffort(v as Effort)}
                />
              </Row>
              {/* Quality */}
              <Row label="Quality">
                <Segmented
                  options={[
                    { id: 'basic', label: 'Basic' },
                    { id: 'balanced', label: 'Balanced' },
                    { id: 'thorough', label: 'Thorough' },
                  ]}
                  value={qlevel}
                  onChange={(v) => setQlevel(v as QLevel)}
                />
              </Row>
              {/* Team size + toggles */}
              <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
                <label className="flex items-center gap-2 text-zinc-400">
                  Max agents
                  <input
                    type="number" min={1} max={10} value={maxAgents}
                    onChange={(e) => setMaxAgents(Math.max(1, Math.min(10, Number(e.target.value) || 1)))}
                    className="w-14 rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-violet-500/60 focus:outline-none"
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
            <div className="space-y-3">
              <label className="flex cursor-pointer items-start gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2.5">
                <input
                  type="checkbox"
                  checked={useKnowledge}
                  onChange={(e) => setUseKnowledge(e.target.checked)}
                  className="mt-0.5 h-4 w-4 rounded border-zinc-700 bg-zinc-950/60 accent-violet-600"
                />
                <span>
                  <span className="flex items-center gap-1.5 text-xs font-medium text-zinc-200">
                    <Brain className="h-3.5 w-3.5 text-violet-400" /> Seed with prior runs' knowledge
                  </span>
                  <span className="mt-0.5 block text-[10px] text-zinc-500">
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
                  className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                    s.referenceFolders.length > 0
                      ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                      : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'
                  }`}
                >
                  <FolderSearch className="h-3.5 w-3.5" />
                  {s.referenceFolders.length > 0 ? `Reference folders: ${s.referenceFolders.length}` : 'Reference folders (read-only)'}
                </button>
                {showRefFolders && (
                  <div className="mt-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                    <ReferenceFolderPicker projects={s.projects} selected={s.referenceFolders} onToggle={s.toggleReferenceFolder} />
                  </div>
                )}
              </div>
              {/* Summary */}
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2.5 text-[11px] text-zinc-400">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Summary</div>
                <span className="text-zinc-300">{mode === 'vatra' ? 'Vatra' : 'Basna'}</span> · {effort} · {qlevel} quality · up to {maxAgents} agents
                {mode === 'vatra' && grouped ? ' · grouped' : ''}{sharedDatastore ? ' · shared datastore' : ''}
                {useKnowledge && s.knowledgeSessionIds.length ? ` · ${s.knowledgeSessionIds.length} prior run(s)` : ''}
                {s.referenceFolders.length ? ` · ${s.referenceFolders.length} reference folder(s)` : ''}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-zinc-800 px-5 py-3">
          <button
            onClick={() => (step === 0 ? onClose() : setStep((x) => x - 1))}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800"
          >
            {step === 0 ? 'Cancel' : <><ArrowLeft className="h-3.5 w-3.5" /> Back</>}
          </button>
          {step < STEPS.length - 1 ? (
            <button
              onClick={() => setStep((x) => x + 1)}
              disabled={!canNext}
              className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-100 hover:bg-zinc-700 disabled:opacity-40"
            >
              {step === 0 ? 'Set up manually' : 'Next'} <ArrowRight className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button
              onClick={finish}
              disabled={!intent.trim()}
              className={`flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-medium text-white disabled:opacity-40 ${
                mode === 'vatra' ? 'bg-violet-600 hover:bg-violet-500' : 'bg-sky-600 hover:bg-sky-500'
              }`}
            >
              <Sparkles className="h-3.5 w-3.5" /> {mode === 'vatra' ? 'Create & plan team' : 'Create & route'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-14 shrink-0 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">{label}</span>
      {children}
    </div>
  )
}

function Segmented({ options, value, onChange }: {
  options: { id: string; label: string; Icon?: typeof Gauge }[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="inline-flex rounded-lg border border-zinc-700 bg-zinc-900/50 p-0.5">
      {options.map((o) => (
        <button
          key={o.id}
          onClick={() => onChange(o.id)}
          className={`flex items-center gap-1 rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
            value === o.id ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          {o.Icon && <o.Icon className="h-3 w-3" />}
          {o.label}
        </button>
      ))}
    </div>
  )
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex cursor-pointer items-center gap-1.5 text-zinc-400">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-3.5 w-3.5 rounded border-zinc-700 bg-zinc-950/60 accent-violet-600"
      />
      {label}
    </label>
  )
}
