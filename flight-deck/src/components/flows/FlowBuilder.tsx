import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowLeft,
  Plus,
  Trash2,
  ChevronUp,
  ChevronDown,
  Save,
  FlaskConical,
  Loader2,
  Wrench,
  Bot,
  GitBranch,
  Send,
  X,
  CheckCircle2,
  XCircle,
} from 'lucide-react'
import { useFlowsStore } from '../../stores/flowsStore'
import {
  emptyFlow,
  listFleet,
  type FlowInput,
  type FlowStep,
  type StepType,
  type TriggerChannel,
  type TriggerOn,
  type MatchKind,
  type FlowTestStep,
} from '../../services/flowsApi'

// ── Rule presets for the match builder ──
const RULE_PRESETS = ['has_image', 'has_video', 'has_audio', 'has_document', 'has_text']

const STEP_TYPE_META: Record<StepType, { icon: typeof Wrench; label: string; color: string }> = {
  tool: { icon: Wrench, label: 'Tool', color: 'text-emerald-400' },
  agent: { icon: Bot, label: 'Agent', color: 'text-violet-400' },
  branch: { icon: GitBranch, label: 'Branch', color: 'text-amber-400' },
  emit: { icon: Send, label: 'Emit', color: 'text-sky-400' },
}

function newStep(type: StepType, idx: number): FlowStep {
  const base: FlowStep = { id: `step_${idx}`, type }
  if (type === 'tool') { base.on = 'origin'; base.tool = ''; base.args = {} }
  if (type === 'agent') { base.on = 'origin'; base.prompt = ''; base.guardrails = { deny: [] } }
  if (type === 'branch') { base.when = ''; base.goto = '' }
  if (type === 'emit') { base.channel = 'same'; base.body = '' }
  return base
}

const inputCls =
  'w-full rounded-lg border border-zinc-700/50 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none focus:ring-1 focus:ring-violet-500/20'
const labelCls = 'mb-1 block text-[10px] font-medium uppercase tracking-wider text-zinc-500'

export function FlowBuilder() {
  const { flows, editingId, saveFlow, openList, testFlow, testSteps, testing, clearTest, error } =
    useFlowsStore()

  const initial: FlowInput = useMemo(() => {
    if (editingId) {
      const f = flows.find((x) => x.id === editingId)
      if (f) {
        const { name, description, enabled, priority, trigger, steps, guardrails, output } = f
        return { name, description, enabled, priority, trigger, steps, guardrails, output }
      }
    }
    return emptyFlow()
  }, [editingId, flows])

  const [draft, setDraft] = useState<FlowInput>(initial)
  const [saving, setSaving] = useState(false)
  const [showTest, setShowTest] = useState(false)
  const [testPayload, setTestPayload] = useState('{\n  "video_path": "/path/to/sample.mp4"\n}')

  useEffect(() => { setDraft(initial); clearTest() }, [initial, clearTest])

  const [fleet, setFleet] = useState<string[]>([])
  useEffect(() => {
    listFleet().then((a) => setFleet(a.map((x) => x.name).filter(Boolean))).catch(() => {})
  }, [])

  const patch = (p: Partial<FlowInput>) => setDraft((d) => ({ ...d, ...p }))

  // ── Trigger helpers ──
  const setMatch = (m: Partial<FlowInput['trigger']['match']>) =>
    patch({ trigger: { ...draft.trigger, match: { ...draft.trigger.match, ...m } } })

  const toggleRule = (rule: string) => {
    const has = draft.trigger.match.rules.includes(rule)
    setMatch({ rules: has ? draft.trigger.match.rules.filter((r) => r !== rule) : [...draft.trigger.match.rules, rule] })
  }

  // ── Step helpers ──
  const setStep = (i: number, p: Partial<FlowStep>) =>
    patch({ steps: draft.steps.map((s, j) => (j === i ? { ...s, ...p } : s)) })

  const addStep = (type: StepType) =>
    patch({ steps: [...draft.steps, newStep(type, draft.steps.length + 1)] })

  const removeStep = (i: number) => patch({ steps: draft.steps.filter((_, j) => j !== i) })

  const moveStep = (i: number, dir: -1 | 1) => {
    const j = i + dir
    if (j < 0 || j >= draft.steps.length) return
    const next = [...draft.steps]
    ;[next[i], next[j]] = [next[j], next[i]]
    patch({ steps: next })
  }

  const handleSave = async () => {
    if (!draft.name.trim()) return
    setSaving(true)
    try {
      await saveFlow(draft, editingId)
    } catch {
      // error surfaced by store
    } finally {
      setSaving(false)
    }
  }

  const handleTest = () => {
    if (!editingId) return
    let payload: Record<string, unknown> = {}
    try {
      payload = JSON.parse(testPayload || '{}')
    } catch {
      payload = {}
    }
    testFlow(editingId, payload)
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-6 py-6">
        {/* Header */}
        <div className="mb-5 flex items-center justify-between">
          <button
            onClick={openList}
            className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" /> Back to flows
          </button>
          <div className="flex items-center gap-2">
            {editingId && (
              <button
                onClick={() => setShowTest((v) => !v)}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-700/50 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 transition-colors"
              >
                <FlaskConical className="h-3.5 w-3.5" /> Test with sample
              </button>
            )}
            <button
              onClick={openList}
              className="rounded-lg px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={!draft.name.trim() || saving}
              className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40 transition-colors"
            >
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
              Save
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-red-500/20 bg-red-500/[0.06] px-3 py-2 text-xs text-red-300">
            {error}
          </div>
        )}

        <h1 className="mb-5 text-lg font-semibold text-zinc-100">
          {editingId ? 'Edit flow' : 'New flow'}
        </h1>

        {/* ── Basics ── */}
        <Section title="Basics">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="sm:col-span-2">
              <label className={labelCls}>Name</label>
              <input
                value={draft.name}
                onChange={(e) => patch({ name: e.target.value })}
                placeholder="Describe attached video"
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>Priority</label>
              <input
                type="number"
                value={draft.priority}
                onChange={(e) => patch({ priority: parseInt(e.target.value, 10) || 0 })}
                className={inputCls}
              />
            </div>
          </div>
          <div className="mt-3">
            <label className={labelCls}>Description</label>
            <input
              value={draft.description}
              onChange={(e) => patch({ description: e.target.value })}
              placeholder="What this flow does"
              className={inputCls}
            />
          </div>
          <label className="mt-3 flex items-center gap-2 text-xs text-zinc-300">
            <input
              type="checkbox"
              checked={draft.enabled}
              onChange={(e) => patch({ enabled: e.target.checked })}
              className="h-3.5 w-3.5 accent-violet-500"
            />
            Enabled
          </label>
        </Section>

        {/* ── Trigger ── */}
        <Section title="Trigger">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div>
              <label className={labelCls}>On</label>
              <select
                value={draft.trigger.on}
                onChange={(e) => patch({ trigger: { ...draft.trigger, on: e.target.value as TriggerOn } })}
                className={inputCls}
              >
                <option value="message">message</option>
                <option value="schedule">schedule</option>
                <option value="decision">decision</option>
              </select>
            </div>
            <div>
              <label className={labelCls}>Channel</label>
              <select
                value={draft.trigger.channel}
                onChange={(e) => patch({ trigger: { ...draft.trigger, channel: e.target.value as TriggerChannel } })}
                className={inputCls}
              >
                <option value="any">any</option>
                <option value="whatsapp">whatsapp</option>
                <option value="glasses">glasses</option>
                <option value="web">web</option>
              </select>
            </div>
          </div>

          <div className="mt-3">
            <label className={labelCls}>Match kind</label>
            <div className="flex gap-1.5">
              {(['rule', 'classifier', 'always'] as MatchKind[]).map((k) => (
                <button
                  key={k}
                  onClick={() => setMatch({ kind: k })}
                  className={`rounded-lg px-3 py-1 text-xs transition-colors ${
                    draft.trigger.match.kind === k
                      ? 'bg-violet-600/20 text-violet-300 ring-1 ring-violet-500/30'
                      : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>

          {draft.trigger.match.kind === 'rule' && (
            <div className="mt-3">
              <label className={labelCls}>Rules</label>
              <div className="flex flex-wrap gap-1.5">
                {RULE_PRESETS.map((r) => {
                  const on = draft.trigger.match.rules.includes(r)
                  return (
                    <button
                      key={r}
                      onClick={() => toggleRule(r)}
                      className={`rounded-full px-2.5 py-1 text-[11px] transition-colors ${
                        on
                          ? 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30'
                          : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      {r.replace(/_/g, ' ')}
                    </button>
                  )
                })}
              </div>
              <CsvField
                className="mt-2"
                label="Custom rules (comma-separated, e.g. from:+123, mime:image/png)"
                value={draft.trigger.match.rules.filter((r) => !RULE_PRESETS.includes(r))}
                presets={draft.trigger.match.rules.filter((r) => RULE_PRESETS.includes(r))}
                onChange={(custom, presets) => setMatch({ rules: [...presets, ...custom] })}
              />
            </div>
          )}

          {draft.trigger.match.kind === 'classifier' && (
            <div className="mt-3">
              <label className={labelCls}>Labels (comma-separated)</label>
              <input
                value={draft.trigger.match.labels.join(', ')}
                onChange={(e) =>
                  setMatch({ labels: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })
                }
                placeholder="urgent, invoice, support"
                className={inputCls}
              />
            </div>
          )}
        </Section>

        {/* ── Steps ── */}
        <Section title={`Steps (${draft.steps.length})`}>
          {draft.steps.length === 0 && (
            <p className="mb-3 rounded-lg border border-dashed border-zinc-800 px-4 py-6 text-center text-xs text-zinc-600">
              No steps yet. Add a tool, agent, branch, or emit step below.
            </p>
          )}

          <div className="space-y-3">
            {draft.steps.map((step, i) => (
              <StepCard
                key={i}
                step={step}
                index={i}
                total={draft.steps.length}
                priorIds={draft.steps.slice(0, i).map((s) => s.id)}
                allIds={draft.steps.map((s) => s.id)}
                fleet={fleet}
                onChange={(p) => setStep(i, p)}
                onRemove={() => removeStep(i)}
                onMove={(dir) => moveStep(i, dir)}
              />
            ))}
          </div>

          {/* Add step buttons */}
          <div className="mt-3 flex flex-wrap gap-2">
            {(Object.keys(STEP_TYPE_META) as StepType[]).map((t) => {
              const { icon: Icon, label, color } = STEP_TYPE_META[t]
              return (
                <button
                  key={t}
                  onClick={() => addStep(t)}
                  className="flex items-center gap-1.5 rounded-lg border border-dashed border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:border-violet-500/30 hover:bg-violet-500/5 hover:text-zinc-200 transition-colors"
                >
                  <Plus className="h-3 w-3" />
                  <Icon className={`h-3.5 w-3.5 ${color}`} /> {label}
                </button>
              )
            })}
          </div>
        </Section>

        {/* ── Guardrails & Output ── */}
        <Section title="Guardrails & Output">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div>
              <label className={labelCls}>Max steps</label>
              <input
                type="number"
                value={draft.guardrails.max_steps}
                onChange={(e) =>
                  patch({ guardrails: { ...draft.guardrails, max_steps: parseInt(e.target.value, 10) || 0 } })
                }
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>Timeout (s)</label>
              <input
                type="number"
                value={draft.guardrails.timeout_s}
                onChange={(e) =>
                  patch({ guardrails: { ...draft.guardrails, timeout_s: parseInt(e.target.value, 10) || 0 } })
                }
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>Output channel</label>
              <select
                value={draft.output.channel}
                onChange={(e) => patch({ output: { ...draft.output, channel: e.target.value } })}
                className={inputCls}
              >
                <option value="same">same (reply on origin)</option>
                <option value="whatsapp">whatsapp</option>
                <option value="glasses">glasses</option>
                <option value="web">web</option>
                <option value="none">none</option>
              </select>
            </div>
            <div>
              <label className={labelCls}>Output format</label>
              <select
                value={draft.output.format}
                onChange={(e) => patch({ output: { ...draft.output, format: e.target.value } })}
                className={inputCls}
              >
                <option value="text">text</option>
                <option value="whisper">whisper</option>
                <option value="markdown">markdown</option>
                <option value="json">json</option>
              </select>
            </div>
          </div>
        </Section>

        {/* ── Test drawer ── */}
        {showTest && editingId && (
          <Section title="Test with sample">
            <label className={labelCls}>Sample payload (JSON)</label>
            <textarea
              value={testPayload}
              onChange={(e) => setTestPayload(e.target.value)}
              rows={5}
              className={`${inputCls} font-mono resize-y`}
            />
            <button
              onClick={handleTest}
              disabled={testing}
              className="mt-2 flex items-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40 transition-colors"
            >
              {testing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FlaskConical className="h-3.5 w-3.5" />}
              Run test
            </button>

            {testSteps && (
              <div className="mt-4 space-y-2">
                <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">Trace</div>
                {testSteps.length === 0 ? (
                  <p className="text-xs text-zinc-600">No steps returned.</p>
                ) : (
                  testSteps.map((s, i) => <TestStepRow key={i} step={s} />)
                )}
              </div>
            )}
          </Section>
        )}
      </div>
    </div>
  )
}

// ── Sub-components ──

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-5 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
      <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">{title}</h2>
      {children}
    </section>
  )
}

function CsvField({
  label,
  value,
  presets,
  onChange,
  className,
}: {
  label: string
  value: string[]
  presets: string[]
  onChange: (custom: string[], presets: string[]) => void
  className?: string
}) {
  return (
    <div className={className}>
      <label className={labelCls}>{label}</label>
      <input
        value={value.join(', ')}
        onChange={(e) => onChange(e.target.value.split(',').map((s) => s.trim()).filter(Boolean), presets)}
        placeholder="from:+38591..., mime:image/png"
        className={inputCls}
      />
    </div>
  )
}

const _TRIGGER_VARS = [
  'trigger.text', 'trigger.channel', 'trigger.waid', 'trigger.mime',
  'trigger.image_path', 'trigger.video_path', 'trigger.audio_path', 'trigger.origin_name',
]
const _SYSTEM_VARS = ['system.now', 'system.date', 'system.time', 'system.agent', 'system.channel']

/** Clickable chips that insert {{...}} variables into the focused field. */
function VarChips({ priorIds, onInsert }: { priorIds: string[]; onInsert: (token: string) => void }) {
  const groups: { label: string; vars: string[] }[] = [
    { label: 'Trigger', vars: _TRIGGER_VARS },
    ...(priorIds.length ? [{ label: 'Prior steps', vars: priorIds.map((id) => `steps.${id}.output`) }] : []),
    { label: 'System', vars: _SYSTEM_VARS },
  ]
  return (
    <div className="mt-1.5 space-y-1 rounded-lg border border-zinc-800/60 bg-zinc-900/30 p-1.5">
      <div className="text-[9px] uppercase tracking-wider text-zinc-600">Insert variable (click into a field first)</div>
      {groups.map((g) => (
        <div key={g.label} className="flex flex-wrap items-center gap-1">
          <span className="mr-1 w-16 shrink-0 text-[9px] uppercase tracking-wider text-zinc-600">{g.label}</span>
          {g.vars.map((v) => (
            <button
              type="button"
              key={v}
              onMouseDown={(e) => { e.preventDefault(); onInsert(v) }}
              className="rounded bg-zinc-800/70 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300 hover:bg-violet-500/20 hover:text-violet-200"
              title={`Insert {{${v}}}`}
            >
              {`{{${v}}}`}
            </button>
          ))}
        </div>
      ))}
    </div>
  )
}

function StepCard({
  step,
  index,
  total,
  priorIds,
  allIds,
  fleet,
  onChange,
  onRemove,
  onMove,
}: {
  step: FlowStep
  index: number
  total: number
  priorIds: string[]
  allIds: string[]
  fleet: string[]
  onChange: (p: Partial<FlowStep>) => void
  onRemove: () => void
  onMove: (dir: -1 | 1) => void
}) {
  const meta = STEP_TYPE_META[step.type]
  const Icon = meta.icon

  // Track the last-focused templatable field so a variable chip inserts at the
  // cursor of whatever the user was editing in this step.
  const lastField = useRef<{ el: HTMLTextAreaElement | HTMLInputElement; set: (v: string) => void } | null>(null)
  const registerFocus = (el: HTMLTextAreaElement | HTMLInputElement, set: (v: string) => void) => {
    lastField.current = { el, set }
  }
  const insertVar = (token: string) => {
    const f = lastField.current
    const ins = `{{${token}}}`
    if (!f) return
    const el = f.el
    const cur = el.value || ''
    const s = el.selectionStart ?? cur.length
    const e = el.selectionEnd ?? cur.length
    f.set(cur.slice(0, s) + ins + cur.slice(e))
    requestAnimationFrame(() => {
      try { el.focus(); const p = s + ins.length; el.setSelectionRange(p, p) } catch { /* ignore */ }
    })
  }

  // {{...}} hint built from prior step outputs
  const hint = priorIds.length
    ? `Use {{trigger.*}} or ${priorIds.map((id) => `{{steps.${id}.output}}`).join(', ')}`
    : 'Use {{trigger.*}} (e.g. {{trigger.video_path}})'

  return (
    <div className="rounded-xl border border-zinc-700/50 bg-zinc-950/50 p-3">
      <div className="mb-2.5 flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-zinc-800 text-[11px] font-bold text-zinc-400">
          {index + 1}
        </span>
        <Icon className={`h-3.5 w-3.5 ${meta.color}`} />
        <span className="text-xs font-medium text-zinc-200">{meta.label}</span>
        <div className="ml-auto flex items-center gap-0.5">
          <button
            onClick={() => onMove(-1)}
            disabled={index === 0}
            className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-30"
          >
            <ChevronUp className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => onMove(1)}
            disabled={index === total - 1}
            className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-30"
          >
            <ChevronDown className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={onRemove}
            className="rounded p-1 text-zinc-500 hover:bg-red-500/10 hover:text-red-400"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Step id (common) */}
      <div className="mb-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div>
          <label className={labelCls}>Step id</label>
          <input
            value={step.id}
            onChange={(e) => onChange({ id: e.target.value })}
            className={`${inputCls} font-mono`}
          />
        </div>
        {(step.type === 'tool' || step.type === 'agent') && (
          <div>
            <label className={labelCls}>Run on (agent)</label>
            <select
              value={step.on || 'origin'}
              onChange={(e) => onChange({ on: e.target.value })}
              className={inputCls}
            >
              <option value="origin">origin (the triggering agent)</option>
              <option value="any">any running agent</option>
              <option value="capability:vision">capability: vision</option>
              {step.type === 'tool' && <option value="fd">Flight Deck (internal tool)</option>}
              {fleet.length > 0 && (
                <optgroup label="Specific agent">
                  {fleet.map((name) => (
                    <option key={name} value={`name:${name}`}>{name}</option>
                  ))}
                </optgroup>
              )}
              {/* Preserve a custom value set outside the presets (e.g. from JSON). */}
              {step.on &&
                !['origin', 'any', 'capability:vision', 'fd'].includes(step.on) &&
                !fleet.some((n) => `name:${n}` === step.on) && (
                  <option value={step.on}>{step.on}</option>
                )}
            </select>
          </div>
        )}
      </div>

      {/* Typed sub-form */}
      {step.type === 'tool' && (
        <>
          <div className="mb-2">
            <label className={labelCls}>Tool</label>
            <input
              value={step.tool || ''}
              onChange={(e) => onChange({ tool: e.target.value })}
              placeholder="video_vision"
              className={inputCls}
            />
          </div>
          <ArgsEditor args={step.args || {}} onChange={(args) => onChange({ args })} hint={hint} registerFocus={registerFocus} />
          <VarChips priorIds={priorIds} onInsert={insertVar} />
        </>
      )}

      {step.type === 'agent' && (
        <>
          <div className="mb-2">
            <label className={labelCls}>Prompt</label>
            <textarea
              value={step.prompt || ''}
              onChange={(e) => onChange({ prompt: e.target.value })}
              onFocus={(e) => registerFocus(e.currentTarget, (v) => onChange({ prompt: v }))}
              rows={3}
              placeholder="Describe the video using ONLY this analysis:\n{{steps.analyze.output}}"
              className={`${inputCls} resize-y`}
            />
            <p className="mt-1 text-[10px] text-zinc-600">{hint}</p>
            <VarChips priorIds={priorIds} onInsert={insertVar} />
          </div>
          <div className="mb-2">
            <label className={labelCls}>Attach to agent (file/image path — optional)</label>
            <input
              value={step.attach || ''}
              onChange={(e) => onChange({ attach: e.target.value })}
              onFocus={(e) => registerFocus(e.currentTarget, (v) => onChange({ attach: v }))}
              placeholder="{{trigger.image_path}}"
              className={`${inputCls} font-mono`}
            />
            <p className="mt-1 text-[10px] text-zinc-600">
              Sends a file (e.g. the photo) to the agent so it can see it. Use {`{{trigger.image_path}}`} for an attached image.
            </p>
          </div>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div>
              <label className={labelCls}>Allowed tools (comma-sep)</label>
              <input
                value={(step.guardrails?.allow || []).join(', ')}
                onChange={(e) =>
                  onChange({
                    guardrails: {
                      ...step.guardrails,
                      allow: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
                    },
                  })
                }
                placeholder="(empty = none)"
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>Denied tools (comma-sep)</label>
              <input
                value={(step.guardrails?.deny || []).join(', ')}
                onChange={(e) =>
                  onChange({
                    guardrails: {
                      ...step.guardrails,
                      deny: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
                    },
                  })
                }
                placeholder="scripts, shell"
                className={inputCls}
              />
            </div>
          </div>
        </>
      )}

      {step.type === 'branch' && (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <div>
            <label className={labelCls}>When (condition)</label>
            <input
              value={step.when || ''}
              onChange={(e) => onChange({ when: e.target.value })}
              onFocus={(e) => registerFocus(e.currentTarget, (v) => onChange({ when: v }))}
              placeholder="{{steps.who.label}} == none"
              className={`${inputCls} font-mono`}
            />
            <VarChips priorIds={priorIds} onInsert={insertVar} />
          </div>
          <div>
            <label className={labelCls}>Goto step</label>
            <select
              value={step.goto || ''}
              onChange={(e) => onChange({ goto: e.target.value })}
              className={inputCls}
            >
              <option value="">— select step —</option>
              {allIds.filter((id) => id !== step.id).map((id) => (
                <option key={id} value={id}>{id}</option>
              ))}
            </select>
          </div>
        </div>
      )}

      {step.type === 'emit' && (
        <>
          <div className="mb-2">
            <label className={labelCls}>Channel</label>
            <select
              value={step.channel || 'same'}
              onChange={(e) => onChange({ channel: e.target.value })}
              className={inputCls}
            >
              <option value="same">same</option>
              <option value="whatsapp">whatsapp</option>
              <option value="glasses">glasses</option>
              <option value="web">web</option>
            </select>
          </div>
          <div>
            <label className={labelCls}>Body</label>
            <textarea
              value={step.body || ''}
              onChange={(e) => onChange({ body: e.target.value })}
              onFocus={(e) => registerFocus(e.currentTarget, (v) => onChange({ body: v }))}
              rows={2}
              placeholder="{{steps.whisper.output}}"
              className={`${inputCls} resize-y`}
            />
            <p className="mt-1 text-[10px] text-zinc-600">{hint}</p>
            <VarChips priorIds={priorIds} onInsert={insertVar} />
          </div>
        </>
      )}
    </div>
  )
}

function ArgsEditor({
  args,
  onChange,
  hint,
  registerFocus,
}: {
  args: Record<string, string>
  onChange: (args: Record<string, string>) => void
  hint: string
  registerFocus: (el: HTMLTextAreaElement | HTMLInputElement, set: (v: string) => void) => void
}) {
  const entries = Object.entries(args)

  const setKey = (oldKey: string, newKey: string) => {
    const next: Record<string, string> = {}
    for (const [k, v] of Object.entries(args)) next[k === oldKey ? newKey : k] = v
    onChange(next)
  }
  const setVal = (key: string, val: string) => onChange({ ...args, [key]: val })
  const remove = (key: string) => {
    const next = { ...args }
    delete next[key]
    onChange(next)
  }
  const add = () => {
    let i = 1
    let key = 'arg'
    while (key in args) { key = `arg${i++}` }
    onChange({ ...args, [key]: '' })
  }

  return (
    <div>
      <label className={labelCls}>Args</label>
      <div className="space-y-1.5">
        {entries.map(([k, v]) => (
          <div key={k} className="flex items-center gap-1.5">
            <input
              value={k}
              onChange={(e) => setKey(k, e.target.value)}
              placeholder="key"
              className={`${inputCls} max-w-[35%] font-mono`}
            />
            <input
              value={v}
              onChange={(e) => setVal(k, e.target.value)}
              onFocus={(e) => registerFocus(e.currentTarget, (nv) => setVal(k, nv))}
              placeholder="{{trigger.video_path}}"
              className={`${inputCls} font-mono`}
            />
            <button onClick={() => remove(k)} className="rounded p-1 text-zinc-500 hover:bg-red-500/10 hover:text-red-400">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
      </div>
      <button
        onClick={add}
        className="mt-1.5 flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300"
      >
        <Plus className="h-3 w-3" /> Add arg
      </button>
      <p className="mt-1 text-[10px] text-zinc-600">{hint}</p>
    </div>
  )
}

function TestStepRow({ step }: { step: FlowTestStep }) {
  const ok = (step.status || '').toLowerCase() === 'ok' || (step.status || '').toLowerCase() === 'done'
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 px-3 py-2">
      <div className="flex items-center gap-2">
        {ok ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" /> : <XCircle className="h-3.5 w-3.5 text-red-400" />}
        <span className="font-mono text-xs text-zinc-200">{step.step_id}</span>
        <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
          {step.status}
        </span>
        {step.agent && <span className="text-[11px] text-zinc-500">· {step.agent}</span>}
        {typeof step.ms === 'number' && <span className="ml-auto text-[10px] text-zinc-500">{step.ms} ms</span>}
      </div>
      {step.output && (
        <pre className="mt-1.5 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-zinc-900/70 px-2 py-1.5 text-[11px] text-zinc-300">
          {step.output}
        </pre>
      )}
    </div>
  )
}
