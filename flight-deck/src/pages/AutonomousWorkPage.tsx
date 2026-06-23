import { useEffect, useMemo, useState } from 'react'
import { Cpu, Sliders, ListChecks, Loader2, AlertCircle, Check, ShieldCheck, Gauge, Play } from 'lucide-react'
import { useAutonomyStore, type AutonomyConfig, type AutonomyAction, type CustomAction, type CustomSource } from '../stores/autonomyStore'

// Theming (see index.css): zinc is auto-remapped in light mode, so zinc classes
// are written dark-first with NO dark: pairs. Non-zinc accents use explicit
// light/dark: pairs.

const AUTONOMY_LEVELS = ['off', 'propose', 'act_low_risk', 'act'] as const
const LEVEL_LABEL: Record<string, string> = {
  off: 'Off', propose: 'Propose only', act_low_risk: 'Act (low risk)', act: 'Act (all)',
}
const RISK_LEVELS = ['low', 'normal', 'high']
const JUDGE_MODES = [
  { v: 'auto', label: 'Auto (LLM judges)' },
  { v: 'human', label: 'Human only' },
  { v: 'both', label: 'Both (LLM + human)' },
]

const inputCls =
  'rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none'

function Section({ title, desc, children }: { title: string; desc?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
        {desc && <p className="mt-0.5 text-[11px] text-zinc-500">{desc}</p>}
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">{children}</div>
    </div>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-medium text-zinc-400">{label}</span>
      {children}
      {hint && <span className="text-[10px] text-zinc-600">{hint}</span>}
    </label>
  )
}

function Toggle({ label, hint, checked, disabled, onChange }: {
  label: string; hint?: string; checked: boolean; disabled?: boolean; onChange: (v: boolean) => void
}) {
  return (
    <label className={`flex items-start gap-2 ${disabled ? 'opacity-50' : ''}`}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 rounded border border-zinc-700 bg-zinc-950 accent-sky-600"
      />
      <span className="flex flex-col">
        <span className="text-xs font-medium text-zinc-300">{label}</span>
        {hint && <span className="text-[10px] text-zinc-600">{hint}</span>}
      </span>
    </label>
  )
}

const _inp = 'rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 w-full'

function JsonField({ value, onChange }: { value: Record<string, unknown>; onChange: (v: Record<string, unknown>) => void }) {
  const [text, setText] = useState(JSON.stringify(value || {}))
  const [bad, setBad] = useState(false)
  return (
    <div className="flex flex-col gap-0.5">
      <textarea
        value={text} rows={2} spellCheck={false}
        onChange={(e) => {
          setText(e.target.value)
          try { const p = JSON.parse(e.target.value || '{}'); if (p && typeof p === 'object' && !Array.isArray(p)) { onChange(p as Record<string, unknown>); setBad(false) } else setBad(true) }
          catch { setBad(true) }
        }}
        className={`${_inp} font-mono`}
      />
      {bad && <span className="text-[10px] text-rose-400">invalid JSON — not saved</span>}
    </div>
  )
}

// How the loop uses a promoted tool — short worked walk-throughs.
const TS_EXAMPLES: { title: string; body: string }[] = [
  { title: 'Granola → meeting follow-ups', body: 'Add the Granola meetings source. When a new meeting is transcribed, the loop reads it BY ID (grounded) and either nudges you with the action items or drafts a follow-up email.' },
  { title: 'Gmail → drafted replies', body: 'Add the Gmail search source. Important unread threads surface as events; the arbiter reads the thread by id and prepares a draft reply (never sent) for you to review — and learns from what you dismiss.' },
  { title: 'Promote a tool by hand', body: 'Click any tool chip below (or “+ action”), give it a label + required args, keep it propose-only, and Save. It joins the catalog and the arbiter can propose it — auto-firing only once it earns trust.' },
]

// One-click templates. Tool names are best-guess for a typical Granola/Google
// stack — confirm each against your AGENT TOOLS list and tweak before saving.
const LIB_SOURCES: (Partial<CustomSource> & { _desc: string })[] = [
  { _desc: 'New Granola meetings → events', name: 'granola', label: 'Granola meetings', tool: 'query_granola_meetings', id_field: 'id', fetch_tool: 'get_meeting_transcript', summary_template: 'Meeting: {title}' },
  { _desc: 'Important unread Gmail threads', name: 'gmail_search', label: 'Gmail search', tool: 'search_threads', id_field: 'thread_id', fetch_tool: 'get_thread', summary_template: 'Email: {subject}', args: { query: 'is:important is:unread in:inbox' }, requires_google: true },
  { _desc: 'Recently changed Drive files', name: 'drive_recent', label: 'Drive recent files', tool: 'list_recent_files', id_field: 'id', fetch_tool: 'read_file_content', summary_template: 'File: {name}', requires_google: true },
  { _desc: 'A standing web search', name: 'web_watch', label: 'Web watch', tool: 'web_search', id_field: 'url', summary_template: '{title}', args: { query: 'your query here' } },
]
const LIB_ACTIONS: (Partial<CustomAction> & { _desc: string })[] = [
  { _desc: 'Save to project memory', id: 'custom.project_memory', label: 'Save to project memory', tool: 'project_memory', required: ['content'], risk: 'low', reversibility: 'reversible', grant: 'memory' },
  { _desc: 'Save a key/value to the datastore', id: 'custom.datastore', label: 'Save to datastore', tool: 'datastore', required: ['key', 'value'], risk: 'low', reversibility: 'reversible', grant: 'data' },
  { _desc: 'Create a Drive doc', id: 'custom.drive_create', label: 'Create Drive doc', tool: 'create_file', required: ['name', 'content'], risk: 'normal', reversibility: 'reversible', grant: 'drive' },
  { _desc: 'Label a Gmail thread', id: 'custom.gmail_label', label: 'Label Gmail thread', tool: 'label_thread', required: ['thread_id', 'label'], risk: 'low', reversibility: 'reversible', reverse_tool: 'unlabel_thread', grant: 'mail' },
  { _desc: 'Suggest a meeting time', id: 'custom.cal_suggest', label: 'Suggest meeting time', tool: 'suggest_time', required: ['duration'], risk: 'low', reversibility: 'read_only', grant: 'calendar' },
  { _desc: 'Fetch a web page', id: 'custom.web_fetch', label: 'Fetch a web page', tool: 'web_fetch', required: ['url'], risk: 'low', reversibility: 'read_only', grant: 'web' },
  { _desc: 'Run a quick web research brief', id: 'custom.web_research', label: 'Web research brief', tool: 'web_search', required: ['query'], risk: 'low', reversibility: 'read_only', grant: 'web' },
  { _desc: 'Summarize workspace files', id: 'custom.summarize', label: 'Summarize files', tool: 'summarize_files', required: ['paths'], risk: 'low', reversibility: 'read_only', grant: 'files' },
]

// Mirror of backend AUTONOMY_HARD_EXCLUDE — tools the loop may never drive.
const _HARD_EXCLUDE = ['shell', 'bash', 'exec', 'subprocess', 'browser', 'playwright', 'selenium', 'tweet', 'post_to', 'social', 'pay', 'payment', 'stripe', 'checkout', 'transfer', 'wire', 'basna']
const _isExcludedTool = (t: string) => _HARD_EXCLUDE.some((x) => t.toLowerCase().includes(x))

function _blankAction(): CustomAction {
  return { id: 'custom.', label: '', tool: '', base_args: {}, required: [], optional: [], risk: 'normal', reversibility: 'irreversible', reverse_tool: '', grant: 'custom', human_only: true, enabled: true }
}
function _blankSource(): CustomSource {
  return { name: '', label: '', tool: '', args: {}, interval_seconds: 600, items_path: '', id_field: 'id', summary_template: '', fetch_tool: '', requires_google: false, enabled: false }
}

function ToolsAndSourcesPanel({ config, set }: {
  config: AutonomyConfig
  set: <K extends keyof AutonomyConfig>(k: K, v: AutonomyConfig[K]) => void
}) {
  const { agentTools, fetchAgentTools } = useAutonomyStore()
  useEffect(() => { if (!agentTools) fetchAgentTools() }, [agentTools, fetchAgentTools])

  const actions = config.custom_actions || []
  const sources = config.custom_sources || []
  const upA = (i: number, patch: Partial<CustomAction>) => set('custom_actions', actions.map((a, idx) => idx === i ? { ...a, ...patch } : a))
  const upS = (i: number, patch: Partial<CustomSource>) => set('custom_sources', sources.map((s, idx) => idx === i ? { ...s, ...patch } : s))
  const csv = (s: string) => s.split(',').map((x) => x.trim()).filter(Boolean)
  const addAction = (a: CustomAction) => set('custom_actions', [...actions, a])
  const addSource = (s: CustomSource) => set('custom_sources', [...sources, s])
  // A read/list-shaped tool is most useful as a sense; everything else as a hand.
  const looksLikeSense = (t: string) => /(list|search|recent|watch|detect|get_|read|fetch|digest)/i.test(t)
  const promoteTool = (t: string) => {
    if (looksLikeSense(t)) addSource({ ..._blankSource(), name: t, label: t, tool: t })
    else addAction({ ..._blankAction(), id: `custom.${t}`, label: t, tool: t })
  }

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-zinc-100">Tools &amp; Sources</h3>
        <p className="mt-0.5 text-[11px] text-zinc-500">
          Promote your agent's own tools into autonomous <b>actions</b> (hands) or polled <b>sources</b> (senses).
          New actions default to propose-only; shell/browser/social/payment tools are always excluded.
        </p>
      </div>

      {/* How it works — worked examples */}
      <div className="mb-4 rounded-lg border border-sky-900/40 bg-sky-950/20 p-3">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-sky-400">How it works</span>
        <ul className="mt-1.5 flex flex-col gap-1.5">
          {TS_EXAMPLES.map((ex) => (
            <li key={ex.title} className="text-[11px] text-zinc-400">
              <span className="font-medium text-zinc-200">{ex.title}.</span> {ex.body}
            </li>
          ))}
        </ul>
      </div>

      {/* Preset library */}
      <div className="mb-4">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Library — one-click templates</span>
        <p className="mb-1.5 mt-0.5 text-[10px] text-zinc-600">Tool names are best-guess for a Granola / Google stack — confirm each against your Agent tools and tweak before saving.</p>
        <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
          {LIB_SOURCES.map((p) => (
            <button key={p.name} onClick={() => addSource({ ..._blankSource(), ...p } as CustomSource)}
              className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-950/40 px-2 py-1.5 text-left hover:border-zinc-600">
              <span className="flex flex-col"><span className="text-[11px] font-medium text-zinc-200">{p.label}</span><span className="text-[10px] text-zinc-600">sense · {p._desc}</span></span>
              <span className="text-[10px] text-sky-500">+ add</span>
            </button>
          ))}
          {LIB_ACTIONS.map((p) => (
            <button key={p.id} onClick={() => addAction({ ..._blankAction(), ...p } as CustomAction)}
              className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-950/40 px-2 py-1.5 text-left hover:border-zinc-600">
              <span className="flex flex-col"><span className="text-[11px] font-medium text-zinc-200">{p.label}</span><span className="text-[10px] text-zinc-600">hand · {p._desc}</span></span>
              <span className="text-[10px] text-sky-500">+ add</span>
            </button>
          ))}
        </div>
      </div>

      {/* Agent tool menu */}
      <div className="mb-4">
        <div className="mb-1 flex items-center gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Agent tools</span>
          <button onClick={() => fetchAgentTools()} className="text-[10px] text-sky-500 hover:text-sky-400">refresh</button>
          {agentTools?.error && <span className="text-[10px] text-rose-400">{agentTools.error}</span>}
        </div>
        <p className="mb-1 text-[10px] text-zinc-600">Click a tool to promote it (read/list tools → a source, others → an action).</p>
        <div className="flex flex-wrap gap-1">
          {(agentTools?.tools || []).map((t) => {
            const excluded = _isExcludedTool(t)
            return (
              <button key={t} onClick={() => !excluded && promoteTool(t)} disabled={excluded}
                title={excluded ? 'excluded — the loop can never drive this' : 'promote this tool'}
                className={excluded
                  ? 'cursor-not-allowed rounded-full bg-zinc-900 px-2 py-0.5 text-[10px] text-zinc-600 line-through'
                  : 'rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-300 hover:bg-sky-900/60 hover:text-sky-200'}>{t}</button>
            )
          })}
          {agentTools && (agentTools.tools || []).length === 0 && <span className="text-[10px] text-zinc-600">No tools (agent running?)</span>}
        </div>
      </div>

      {/* Custom actions */}
      <div className="mb-4 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Custom actions ({actions.length})</span>
          <button
            onClick={() => addAction(_blankAction())}
            className="rounded-md border border-zinc-700 px-2 py-0.5 text-[10px] text-zinc-300 hover:bg-zinc-800">+ action</button>
        </div>
        {actions.map((a, i) => (
          <div key={i} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
            <div className="grid grid-cols-2 gap-2">
              <Field label="id"><input className={_inp} value={a.id} onChange={(e) => upA(i, { id: e.target.value })} /></Field>
              <Field label="tool"><input className={_inp} value={a.tool} onChange={(e) => upA(i, { tool: e.target.value })} placeholder="agent tool name" /></Field>
              <Field label="label"><input className={_inp} value={a.label} onChange={(e) => upA(i, { label: e.target.value })} /></Field>
              <Field label="grant"><input className={_inp} value={a.grant} onChange={(e) => upA(i, { grant: e.target.value })} /></Field>
              <Field label="risk"><select className={_inp} value={a.risk} onChange={(e) => upA(i, { risk: e.target.value })}><option>low</option><option>normal</option><option>high</option></select></Field>
              <Field label="reversibility"><select className={_inp} value={a.reversibility} onChange={(e) => upA(i, { reversibility: e.target.value })}><option>read_only</option><option>reversible</option><option>irreversible</option></select></Field>
              <Field label="required args (csv)"><input className={_inp} value={a.required.join(', ')} onChange={(e) => upA(i, { required: csv(e.target.value) })} /></Field>
              <Field label="reverse tool (opt)"><input className={_inp} value={a.reverse_tool} onChange={(e) => upA(i, { reverse_tool: e.target.value })} /></Field>
              <Field label="base args (json)"><JsonField value={a.base_args} onChange={(v) => upA(i, { base_args: v })} /></Field>
            </div>
            <div className="mt-2 flex items-center gap-4">
              <Toggle label="human-only" hint="never auto-fire" checked={a.human_only} onChange={(v) => upA(i, { human_only: v })} />
              <Toggle label="enabled" checked={a.enabled} onChange={(v) => upA(i, { enabled: v })} />
              <button onClick={() => set('custom_actions', actions.filter((_, idx) => idx !== i))} className="ml-auto text-[10px] text-rose-400 hover:text-rose-300">remove</button>
            </div>
          </div>
        ))}
      </div>

      {/* Custom sources */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Custom sources ({sources.length})</span>
          <button
            onClick={() => addSource(_blankSource())}
            className="rounded-md border border-zinc-700 px-2 py-0.5 text-[10px] text-zinc-300 hover:bg-zinc-800">+ source</button>
        </div>
        {sources.map((s, i) => (
          <div key={i} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
            <div className="grid grid-cols-2 gap-2">
              <Field label="name (source slug)"><input className={_inp} value={s.name} onChange={(e) => upS(i, { name: e.target.value })} /></Field>
              <Field label="poll tool"><input className={_inp} value={s.tool} onChange={(e) => upS(i, { tool: e.target.value })} placeholder="list/search tool" /></Field>
              <Field label="label"><input className={_inp} value={s.label} onChange={(e) => upS(i, { label: e.target.value })} /></Field>
              <Field label="interval (s)"><input type="number" className={_inp} value={s.interval_seconds} onChange={(e) => upS(i, { interval_seconds: Number(e.target.value) || 600 })} /></Field>
              <Field label="id field" hint="dedup + grounding handle"><input className={_inp} value={s.id_field} onChange={(e) => upS(i, { id_field: e.target.value })} /></Field>
              <Field label="fetch tool" hint="opens one item by id"><input className={_inp} value={s.fetch_tool} onChange={(e) => upS(i, { fetch_tool: e.target.value })} /></Field>
              <Field label="items path (opt)" hint="dotted path to list"><input className={_inp} value={s.items_path} onChange={(e) => upS(i, { items_path: e.target.value })} /></Field>
              <Field label="summary template"><input className={_inp} value={s.summary_template} onChange={(e) => upS(i, { summary_template: e.target.value })} placeholder="New: {title}" /></Field>
              <Field label="args (json)"><JsonField value={s.args} onChange={(v) => upS(i, { args: v })} /></Field>
            </div>
            <div className="mt-2 flex items-center gap-4">
              <Toggle label="needs Google" checked={s.requires_google} onChange={(v) => upS(i, { requires_google: v })} />
              <Toggle label="enabled" checked={s.enabled} onChange={(v) => upS(i, { enabled: v })} />
              <button onClick={() => set('custom_sources', sources.filter((_, idx) => idx !== i))} className="ml-auto text-[10px] text-rose-400 hover:text-rose-300">remove</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function weightColor(w: number): string {
  if (w >= 0.66) return 'text-emerald-600 dark:text-emerald-400'
  if (w >= 0.4) return 'text-amber-600 dark:text-amber-400'
  return 'text-rose-600 dark:text-rose-400'
}

function relTime(iso: string | null): string {
  if (!iso) return ''
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return ''
  const s = Math.floor((Date.now() - t) / 1000)
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}

// Like relTime but signed — "due in 3d" / "due 2d ago" / "due now".
function dueLabel(iso: string | null): string {
  if (!iso) return ''
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return ''
  const s = Math.floor((t - Date.now()) / 1000)
  const a = Math.abs(s)
  const span = a < 3600 ? `${Math.max(1, Math.floor(a / 60))}m`
    : a < 86400 ? `${Math.floor(a / 3600)}h` : `${Math.floor(a / 86400)}d`
  if (a < 60) return 'due now'
  return s < 0 ? `due ${span} ago` : `due in ${span}`
}

const STATUS_CHIP: Record<string, string> = {
  candidate: 'bg-zinc-800 text-zinc-300',
  queued: 'bg-sky-50 text-sky-700 dark:bg-sky-950/40 dark:text-sky-400',
  awaiting_approval: 'bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400',
  dispatched: 'bg-violet-50 text-violet-700 dark:bg-violet-950/40 dark:text-violet-400',
  done: 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400',
  rejected: 'bg-rose-50 text-rose-700 dark:bg-rose-950/40 dark:text-rose-400',
  expired: 'bg-zinc-800 text-zinc-500',
}

function ActionCard({ action, onApprove, onReject, onUndo }: {
  action: AutonomyAction; onApprove: (id: string) => void; onReject: (id: string) => void
  onUndo: (id: string) => void
}) {
  const pending = action.status === 'awaiting_approval'
  const canUndo = action.status === 'done' && action.outcome === 'success'
    && !!(action.payload as { reverse?: unknown } | undefined)?.reverse
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
      <div className="mb-1 flex flex-wrap items-center gap-2 text-[11px]">
        <span className={`rounded-full px-2 py-0.5 ${STATUS_CHIP[action.status] || 'bg-zinc-800 text-zinc-300'}`}>
          {action.status.replace('_', ' ')}
        </span>
        <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-zinc-300">{action.kind}</span>
        <span className="text-zinc-500">· {action.source}</span>
        <span className="text-zinc-500">· risk {action.risk}</span>
        <span className="ml-auto text-zinc-600">{relTime(action.created_at)}</span>
      </div>
      <div className="text-sm font-medium text-zinc-100">{action.title || '(untitled)'}</div>
      {action.rationale && <p className="mt-0.5 text-xs text-zinc-400">{action.rationale}</p>}
      <div className="mt-2 flex items-center gap-2 text-[11px] text-zinc-500">
        <Gauge className="h-3 w-3" /> score {action.score.toFixed(2)}
        {action.outcome && <span>· outcome {action.outcome}</span>}
      </div>
      {pending && (
        <div className="mt-3 flex gap-2">
          <button
            onClick={() => onApprove(action.id)}
            className="rounded-lg bg-emerald-600 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-500"
          >
            Approve
          </button>
          <button
            onClick={() => onReject(action.id)}
            className="rounded-lg border border-zinc-700 px-3 py-1 text-xs font-medium text-zinc-200 hover:bg-zinc-800"
          >
            Reject
          </button>
        </div>
      )}
      {canUndo && (
        <div className="mt-3">
          <button
            onClick={() => onUndo(action.id)}
            title="Reverse this action (e.g. delete the created event/job)"
            className="rounded-lg border border-zinc-700 px-3 py-1 text-xs font-medium text-zinc-200 hover:bg-zinc-800"
          >
            Undo
          </button>
        </div>
      )}
    </div>
  )
}

export function AutonomousWorkPage() {
  const {
    config, defaults, actions, reliability, log, catalog, events, plans, followUps, loading, saving, error,
    loadAll, loadActions, setField, save, approve, reject, undo, nudge,
    createPlan, advancePlan, abandonPlan, doneFollowUp, dismissFollowUp,
  } = useAutonomyStore()
  const [planGoal, setPlanGoal] = useState('')
  const [creatingPlan, setCreatingPlan] = useState(false)
  const [tab, setTab] = useState<'control' | 'tools' | 'activity'>('control')
  const [savedAt, setSavedAt] = useState(false)
  const [nudging, setNudging] = useState(false)
  const [nudgeMsg, setNudgeMsg] = useState<string | null>(null)

  useEffect(() => { loadAll() }, [loadAll])

  // While watching the Activity feed, poll — dispatched actions finish and get
  // judged in the background, so status/outcome/reliability move without input.
  useEffect(() => {
    if (tab !== 'activity') return
    const id = setInterval(() => { loadActions() }, 8000)
    return () => clearInterval(id)
  }, [tab, loadActions])

  const ceiling = config?.max_autonomy_level || 'propose'
  const allowedLevels = useMemo(() => {
    const cap = AUTONOMY_LEVELS.indexOf(ceiling as typeof AUTONOMY_LEVELS[number])
    return AUTONOMY_LEVELS.slice(0, (cap < 0 ? 1 : cap) + 1)
  }, [ceiling])

  const set = <K extends keyof AutonomyConfig>(k: K, v: AutonomyConfig[K]) => {
    setField(k, v)
    setSavedAt(false)
  }

  const onSave = async () => {
    try { await save(); setSavedAt(true); setTimeout(() => setSavedAt(false), 2500) } catch { /* surfaced via error */ }
  }

  const pending = actions.filter((a) => a.status === 'awaiting_approval')
  const recent = actions.filter((a) => a.status !== 'awaiting_approval')

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex shrink-0 items-center gap-2 border-b border-zinc-700/50 bg-zinc-900/50 px-4 py-3 md:px-6">
        <Cpu className="h-5 w-5 text-sky-600 dark:text-sky-400" />
        <div className="min-w-0">
          <h1 className="text-sm font-semibold text-zinc-100">Autonomous Work</h1>
          <p className="truncate text-[11px] text-zinc-500">
            The closed loop: notice → decide → act → judge → adjust. Everything here is per-user.
          </p>
        </div>
        <div className="ml-auto flex items-center gap-1 rounded-lg border border-zinc-800 bg-zinc-950 p-0.5">
          <button
            onClick={() => setTab('control')}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1 text-xs font-medium ${tab === 'control' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}
          >
            <Sliders className="h-3.5 w-3.5" /> Control
          </button>
          <button
            onClick={() => setTab('tools')}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1 text-xs font-medium ${tab === 'tools' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}
          >
            <Cpu className="h-3.5 w-3.5" /> Tools &amp; Sources
          </button>
          <button
            onClick={() => setTab('activity')}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1 text-xs font-medium ${tab === 'activity' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}
          >
            <ListChecks className="h-3.5 w-3.5" /> Activity
            {pending.length > 0 && (
              <span className="rounded-full bg-amber-500 px-1.5 text-[10px] font-bold text-white">{pending.length}</span>
            )}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 md:p-6">
        {error && (
          <div className="mb-4 flex items-start gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-700 dark:text-rose-300">
            <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {loading && !config && (
          <div className="flex items-center gap-2 text-sm text-zinc-500">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading…
          </div>
        )}

        {/* ── Control tab ── */}
        {tab === 'control' && config && (
          <div className="mx-auto flex max-w-3xl flex-col gap-4">
            {/* Ceiling banner */}
            <div className="flex items-start gap-2 rounded-xl border border-sky-200 bg-sky-50 p-3 text-xs text-sky-800 dark:border-sky-900/40 dark:bg-sky-950/20 dark:text-sky-300">
              <ShieldCheck className="mt-0.5 h-4 w-4 flex-shrink-0" />
              <span>
                Shipped autonomy ceiling: <strong>{LEVEL_LABEL[ceiling] || ceiling}</strong>. In this build the loop
                can never exceed it — at <em>propose</em> it only suggests work and waits for your approval.
              </span>
            </div>

            <Section title="Master" desc="The global kill switch and how far the loop is allowed to go.">
              <Toggle
                label="Enable autonomous work"
                hint="Hard off-switch for the whole loop."
                checked={config.enabled}
                onChange={(v) => set('enabled', v)}
              />
              <Field label="Autonomy level" hint="Clamped to the shipped ceiling.">
                <select className={inputCls} value={config.autonomy_level} onChange={(e) => set('autonomy_level', e.target.value)}>
                  {allowedLevels.map((l) => <option key={l} value={l}>{LEVEL_LABEL[l]}</option>)}
                </select>
              </Field>
            </Section>

            <Section title="Arbiter" desc="The decider that runs inside the heartbeat: ranks candidates, picks the single best next action.">
              <Toggle label="Run arbiter on each pulse" checked={config.arbiter_on_pulse} onChange={(v) => set('arbiter_on_pulse', v)} />
              <Field label={`Min score to act — ${config.arbiter_min_score.toFixed(2)}`} hint="Ignore candidates below this priority.">
                <input type="range" min={0} max={1} step={0.05} value={config.arbiter_min_score}
                  onChange={(e) => set('arbiter_min_score', Number(e.target.value))} className="accent-sky-600" />
              </Field>
              <Field label="Max actions / day">
                <input type="number" min={0} max={100} className={inputCls} value={config.max_actions_per_day}
                  onChange={(e) => set('max_actions_per_day', Number(e.target.value))} />
              </Field>
              <Field label="Max concurrent actions">
                <input type="number" min={1} max={20} className={inputCls} value={config.max_concurrent_actions}
                  onChange={(e) => set('max_concurrent_actions', Number(e.target.value))} />
              </Field>
              <Field label="Candidate lookback (hours)">
                <input type="number" min={1} max={168} className={inputCls} value={config.candidate_lookback_hours}
                  onChange={(e) => set('candidate_lookback_hours', Number(e.target.value))} />
              </Field>
              <Field label="Quiet hours">
                <div className="flex items-center gap-2">
                  <input type="number" min={0} max={23} className={`${inputCls} w-20`} value={config.quiet_hours_start}
                    onChange={(e) => set('quiet_hours_start', Number(e.target.value))} />
                  <span className="text-xs text-zinc-500">to</span>
                  <input type="number" min={0} max={23} className={`${inputCls} w-20`} value={config.quiet_hours_end}
                    onChange={(e) => set('quiet_hours_end', Number(e.target.value))} />
                </div>
              </Field>
            </Section>

            <Section title="Dispatch" desc="How chosen actions become real work. Low-risk only fires automatically once the level allows it.">
              <Toggle label="Allow auto-dispatch (low-risk)" hint="Fires low-risk actions without approval (needs level ≥ act_low_risk)."
                checked={config.allow_auto_dispatch} onChange={(v) => set('allow_auto_dispatch', v)} />
              <Toggle label="High-risk requires approval" checked={config.high_risk_requires_approval}
                onChange={(v) => set('high_risk_requires_approval', v)} />
              <Field label="Low-risk kinds" hint="Comma-separated action kinds treated as low-risk.">
                <input className={inputCls} value={config.low_risk_kinds.join(', ')}
                  onChange={(e) => set('low_risk_kinds', e.target.value.split(',').map((s) => s.trim()).filter(Boolean))} />
              </Field>
            </Section>

            <Section title="Judge & learn" desc="How outcomes are scored and fed back so the arbiter stops repeating what fails.">
              <Toggle label="Learning enabled" checked={config.learning_enabled} onChange={(v) => set('learning_enabled', v)} />
              <Field label="Judge mode">
                <select className={inputCls} value={config.judge_mode} onChange={(e) => set('judge_mode', e.target.value)}>
                  {JUDGE_MODES.map((m) => <option key={m.v} value={m.v}>{m.label}</option>)}
                </select>
              </Field>
              <Field label={`Reliability seed — ${config.reliability_seed.toFixed(2)}`} hint="Prior weight for an unproven action kind.">
                <input type="range" min={0} max={1} step={0.05} value={config.reliability_seed}
                  onChange={(e) => set('reliability_seed', Number(e.target.value))} className="accent-sky-600" />
              </Field>
              <Field label={`Suppress below weight — ${config.suppress_below_weight.toFixed(2)}`} hint="Stop proposing kinds whose weight drops under this.">
                <input type="range" min={0} max={1} step={0.05} value={config.suppress_below_weight}
                  onChange={(e) => set('suppress_below_weight', Number(e.target.value))} className="accent-sky-600" />
              </Field>
            </Section>

            <Section title="Reflections → intentions" desc="Turn self-reflection bullets into candidate goals the arbiter can pick up.">
              <Toggle label="Feed reflections into intentions" checked={config.reflection_to_intention}
                onChange={(v) => set('reflection_to_intention', v)} />
              <Field label="Max intentions per reflection">
                <input type="number" min={0} max={10} className={inputCls} value={config.max_intentions_per_reflection}
                  onChange={(e) => set('max_intentions_per_reflection', Number(e.target.value))} />
              </Field>
              <Field label="Max risk to auto-create">
                <select className={inputCls} value={config.reflection_intention_max_risk}
                  onChange={(e) => set('reflection_intention_max_risk', e.target.value)}>
                  {RISK_LEVELS.map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
              </Field>
            </Section>

            {/* Allowed actions (grants → auto-fire) */}
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
              <div className="mb-3">
                <h3 className="text-sm font-semibold text-zinc-100">Allowed actions (auto-fire)</h3>
                <p className="mt-0.5 text-[11px] text-zinc-500">
                  Checked reversible, low-risk actions fire <em>without</em> approval once the level is ≥ Act (low risk).
                  Everything else is always proposed for your approval.
                </p>
              </div>
              <div className="flex flex-col gap-1.5">
                {catalog.map((a) => {
                  const eligible = !a.human_only && a.risk === 'low'
                    && (a.reversibility === 'reversible' || a.reversibility === 'read_only')
                  const granted = (config.granted_actions || []).includes(a.id)
                  // Trust rung from learned reliability (kind=tool_action, domain=action id).
                  const rel = reliability.find((r) => r.kind === 'tool_action' && r.domain === a.id)
                  let badge = { text: 'approval only', cls: 'bg-zinc-800 text-zinc-400' }
                  if (eligible) {
                    if (granted) badge = { text: 'auto · granted', cls: 'bg-sky-50 text-sky-700 dark:bg-sky-950/40 dark:text-sky-400' }
                    else if (rel && rel.weight >= (config.trust_threshold ?? 0.85) && rel.runs >= (config.trust_min_runs ?? 3))
                      badge = { text: `auto · trusted ${rel.weight.toFixed(2)}`, cls: 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400' }
                    else if (rel && rel.weight < (config.suppress_below_weight ?? 0.25))
                      badge = { text: `suppressed ${rel.weight.toFixed(2)}`, cls: 'bg-rose-50 text-rose-700 dark:bg-rose-950/40 dark:text-rose-400' }
                    else if (rel) badge = { text: `learning ${rel.weight.toFixed(2)} · ${rel.runs}✓✗`, cls: 'bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400' }
                    else badge = { text: 'propose', cls: 'bg-zinc-800 text-zinc-400' }
                  }
                  return (
                    <label key={a.id} className={`flex items-center gap-2 text-xs ${eligible ? '' : 'opacity-60'}`}>
                      <input
                        type="checkbox"
                        disabled={!eligible}
                        checked={eligible && granted}
                        onChange={(e) => {
                          const cur = new Set(config.granted_actions || [])
                          if (e.target.checked) cur.add(a.id); else cur.delete(a.id)
                          set('granted_actions', Array.from(cur))
                        }}
                        className="rounded border border-zinc-700 bg-zinc-950 accent-sky-600 disabled:opacity-40"
                      />
                      <span className="font-medium text-zinc-200">{a.label}</span>
                      <span className="text-zinc-600">· {a.risk} · {a.reversibility}</span>
                      <span className={`ml-auto rounded-full px-2 py-0.5 text-[10px] ${badge.cls}`}>{badge.text}</span>
                    </label>
                  )
                })}
                {catalog.length === 0 && <span className="text-xs text-zinc-600">No catalog actions.</span>}
              </div>
            </div>

            <Section title="Event sources" desc="What the loop watches in your world. New events become arbiter candidates. (Calendar/Gmail polling takes effect once a Google account is connected.)">
              <Toggle label="Google Calendar" hint="Poll upcoming/changed events"
                checked={config.event_calendar_enabled} onChange={(v) => set('event_calendar_enabled', v)} />
              <Toggle label="Gmail" hint="Poll important new mail"
                checked={config.event_gmail_enabled} onChange={(v) => set('event_gmail_enabled', v)} />
              <p className="col-span-full text-[11px] text-zinc-500">
                More hands &amp; senses live in the <button onClick={() => setTab('tools')} className="text-sky-500 hover:text-sky-400">Tools &amp; Sources</button> tab.
              </p>
            </Section>

            {/* Save bar */}
            <div className="flex items-center gap-3">
              <button onClick={onSave} disabled={saving}
                className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40">
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                Save settings
              </button>
              {savedAt && <span className="text-xs text-emerald-600 dark:text-emerald-400">Saved</span>}
              {defaults && (
                <button
                  onClick={() => { Object.entries(defaults).forEach(([k, v]) => { if (k !== 'max_autonomy_level' && k !== 'db_path') setField(k as keyof AutonomyConfig, v as never) }); setSavedAt(false) }}
                  className="ml-auto text-xs text-zinc-500 hover:text-zinc-300"
                >
                  Reset to defaults
                </button>
              )}
            </div>
          </div>
        )}

        {/* ── Tools & Sources tab ── */}
        {tab === 'tools' && config && (
          <div className="mx-auto flex max-w-3xl flex-col gap-4">
            <ToolsAndSourcesPanel config={config} set={set} />
            <div className="flex items-center gap-3">
              <button onClick={onSave} disabled={saving}
                className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40">
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                Save settings
              </button>
              {savedAt && <span className="text-xs text-emerald-600 dark:text-emerald-400">Saved</span>}
            </div>
          </div>
        )}

        {/* ── Activity tab ── */}
        {tab === 'activity' && (
          <div className="mx-auto flex max-w-3xl flex-col gap-6">
            <div className="flex items-center gap-3">
              <button
                onClick={async () => {
                  setNudging(true); setNudgeMsg(null)
                  try {
                    const r = await nudge()
                    setNudgeMsg(r.proposed > 0 ? `Proposed ${r.proposed} action` : `Nothing proposed (${r.reason || 'quiet'})`)
                  } catch (e) {
                    setNudgeMsg(e instanceof Error ? e.message : String(e))
                  } finally { setNudging(false) }
                }}
                disabled={nudging || !config?.enabled}
                title={config?.enabled ? 'Force one arbiter pass now' : 'Enable autonomous work first'}
                className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
              >
                {nudging ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                Run arbiter now
              </button>
              {nudgeMsg && <span className="text-xs text-zinc-400">{nudgeMsg}</span>}
            </div>

            {/* Plans (#4) */}
            <div className="flex flex-col gap-2">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Plans</h2>
              <div className="flex gap-2">
                <input
                  value={planGoal}
                  onChange={(e) => setPlanGoal(e.target.value)}
                  placeholder="Give it a goal, e.g. 'prep for the FRC AGM'…"
                  className={`${inputCls} flex-1`}
                />
                <button
                  onClick={async () => {
                    if (!planGoal.trim()) return
                    setCreatingPlan(true)
                    try { await createPlan(planGoal.trim()); setPlanGoal('') }
                    catch { /* surfaced via error */ }
                    finally { setCreatingPlan(false) }
                  }}
                  disabled={creatingPlan || !planGoal.trim()}
                  className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
                >
                  {creatingPlan ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                  Plan it
                </button>
              </div>
              {plans.map((p) => (
                <div key={p.id} className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
                  <div className="mb-1 flex items-center gap-2 text-[11px]">
                    <span className={`rounded-full px-2 py-0.5 ${STATUS_CHIP[p.status] || 'bg-zinc-800 text-zinc-300'}`}>{p.status}</span>
                    <span className="font-medium text-zinc-100">{p.goal}</span>
                    <span className="ml-auto text-zinc-600">{relTime(p.created_at)}</span>
                  </div>
                  <ol className="ml-4 list-decimal space-y-0.5 text-xs">
                    {p.steps.map((s) => (
                      <li key={s.idx} className={
                        s.status === 'done' ? 'text-emerald-600 dark:text-emerald-400'
                          : s.status === 'failed' ? 'text-rose-600 dark:text-rose-400'
                          : s.status === 'skipped' ? 'text-zinc-600 line-through' : 'text-zinc-300'}>
                        {s.title} <span className="text-zinc-600">· {s.kind}{s.action_id ? `:${s.action_id}` : ''} · {s.status}</span>
                      </li>
                    ))}
                  </ol>
                  {(p.status === 'active' || p.status === 'paused') && (
                    <div className="mt-2 flex gap-2">
                      <button onClick={() => advancePlan(p.id)}
                        className="rounded-lg bg-emerald-600 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-500">
                        Run next step
                      </button>
                      <button onClick={() => abandonPlan(p.id)}
                        className="rounded-lg border border-zinc-700 px-3 py-1 text-xs font-medium text-zinc-200 hover:bg-zinc-800">
                        Abandon
                      </button>
                    </div>
                  )}
                  {p.note && <p className="mt-1 text-[11px] text-zinc-500">{p.note}</p>}
                </div>
              ))}
            </div>

            {(() => {
              const openFu = followUps.filter((f) => f.status === 'open')
              const staleFu = followUps.filter((f) => f.status === 'stale')
              const shown = [...openFu, ...staleFu]
              if (shown.length === 0) return null
              return (
                <div className="flex flex-col gap-2">
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-sky-600 dark:text-sky-400">
                    Waiting on you ({openFu.length}{staleFu.length ? ` · ${staleFu.length} stale` : ''})
                  </h2>
                  <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-2">
                    {shown.map((f) => {
                      const overdue = new Date(f.follow_up_at).getTime() <= Date.now()
                      return (
                        <div key={f.id} className="flex items-start gap-2 border-b border-zinc-800/40 py-2 last:border-0">
                          <span className="mt-0.5 shrink-0 rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-300">{f.source || 'note'}</span>
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-xs text-zinc-100">{f.summary}</div>
                            {f.detail && <div className="truncate text-[11px] text-zinc-500">{f.detail}</div>}
                            <div className="mt-0.5 flex items-center gap-2 text-[10px] text-zinc-600">
                              <span className={f.status === 'stale' ? 'text-zinc-500'
                                : overdue ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-500'}>
                                {f.status === 'stale' ? 'stale — not nudging' : dueLabel(f.follow_up_at)}
                              </span>
                              {f.nudged_count > 0 && <span>· nudged {f.nudged_count}×</span>}
                              <span>· {relTime(f.created_at)}</span>
                            </div>
                          </div>
                          <div className="flex shrink-0 gap-1">
                            <button onClick={() => doneFollowUp(f.id)}
                              className="rounded-md bg-emerald-600 px-2 py-1 text-[10px] font-medium text-white hover:bg-emerald-500">Done</button>
                            <button onClick={() => dismissFollowUp(f.id)}
                              className="rounded-md border border-zinc-700 px-2 py-1 text-[10px] font-medium text-zinc-300 hover:bg-zinc-800">Dismiss</button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })()}

            {pending.length > 0 && (
              <div className="flex flex-col gap-2">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                  Awaiting your approval ({pending.length})
                </h2>
                {pending.map((a) => <ActionCard key={a.id} action={a} onApprove={approve} onReject={reject} onUndo={undo} />)}
              </div>
            )}

            {events.length > 0 && (
              <div className="flex flex-col gap-2">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Recent events</h2>
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-2 text-[11px]">
                  {events.slice(0, 20).map((e) => (
                    <div key={e.id} className="flex items-center gap-2 border-b border-zinc-800/40 py-1 last:border-0">
                      <span className="shrink-0 rounded-full bg-zinc-800 px-2 py-0.5 text-zinc-300">{e.source}</span>
                      <span className="truncate text-zinc-200">{e.summary}</span>
                      <span className="ml-auto shrink-0 text-zinc-600">{e.status}</span>
                      <span className="shrink-0 text-zinc-600">{relTime(e.ingested_at)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex flex-col gap-2">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Action ledger</h2>
              {recent.length === 0 ? (
                <div className="rounded-xl border border-dashed border-zinc-800 p-6 text-center text-sm text-zinc-500">
                  No autonomous actions yet. The arbiter that fills this feed ships in a later phase —
                  for now this is the audit trail it will write to.
                </div>
              ) : (
                recent.map((a) => <ActionCard key={a.id} action={a} onApprove={approve} onReject={reject} onUndo={undo} />)
              )}
            </div>

            <div className="flex flex-col gap-2">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Live log</h2>
              {log.length === 0 ? (
                <div className="rounded-xl border border-dashed border-zinc-800 p-4 text-center text-xs text-zinc-500">
                  No log entries yet. Hit "Run arbiter now" — every pass, skip reason, dispatch, and error lands here.
                </div>
              ) : (
                <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-2 font-mono text-[11px] leading-relaxed">
                  {log.map((e) => (
                    <div key={e.id} className="flex gap-2 border-b border-zinc-800/40 py-1 last:border-0">
                      <span className="shrink-0 text-zinc-600">{relTime(e.ts)}</span>
                      <span className={`shrink-0 w-10 uppercase ${
                        e.level === 'error' ? 'text-rose-600 dark:text-rose-400'
                          : e.level === 'warn' ? 'text-amber-600 dark:text-amber-400'
                          : 'text-emerald-600 dark:text-emerald-400'}`}>{e.level}</span>
                      <span className="text-zinc-300">{e.event}</span>
                      {e.detail && <span className="text-zinc-600">— {e.detail}</span>}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {reliability.length > 0 && (
              <div className="flex flex-col gap-2">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Learned reliability</h2>
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
                  {reliability.map((r) => (
                    <div key={`${r.kind}:${r.domain}`} className="flex items-center gap-3 border-b border-zinc-800/60 py-1.5 text-xs last:border-0">
                      <span className="font-medium text-zinc-200">{r.kind}</span>
                      <span className="text-zinc-500">· {r.domain}</span>
                      <span className="ml-auto text-zinc-500">{r.successes}✓ / {r.fails}✗</span>
                      <span className={`w-12 text-right font-semibold ${weightColor(r.weight)}`}>{r.weight.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
