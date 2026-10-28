import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  AlertTriangle,
  BookOpen,
  Brain,
  BrainCog,
  Check,
  Cpu,
  Database,
  Eye,
  Info,
  LibraryBig,
  Loader2,
  MoreHorizontal,
  RotateCw,
  SlidersHorizontal,
  Tags,
  Target,
  Trash2,
  X,
} from 'lucide-react'
import { IntentionsPanel } from '../agents/IntentionsPanel'
import { KioskInsightsPanel } from './KioskAgentInsights'
import { TopicsPanel } from '../agents/TopicsPanel'
import { getAgentConfig, getAgentMode, setAgentMode, setAgentModel, setAgentVision } from '../../services/docker'
import { LLM_PROVIDERS } from '../../services/tierConfig'
import { VISION_OPTIONS, parseModelFromYaml } from '../../services/agentModel'
import { MODES } from '../../services/cognitiveModes'
import { KioskReasoningSettings } from './KioskReasoningSettings'
import { useChatStore } from '../../stores/chatStore'
import { useContainerStore } from '../../stores/containerStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { useProcessStore } from '../../stores/processStore'

// ── Kiosk: one agent's menu and dialogs ──────────────────────────────
//
// The "⋯" on a kiosk agent row opens what a kiosk user may do with one agent
// of theirs besides chatting and the power switch: look at what it means to
// do and what they talked about (Info: its intentions and topics), open its
// files and data, give it standing instructions and see what it remembers
// (Instructions & insights), change its model, how it thinks and whether it
// sees images itself (Settings), or remove it for good, folder and all (Remove).

/** What these dialogs need of an agent row. */
export interface KioskAgent {
  id: string
  kind: 'docker' | 'process' | 'local' | 'shared'
  name: string
  state: string
  reachable: boolean
  host: string
  port: number
  auth: string
  startKey: string
}

export interface AgentMenuItem {
  icon: typeof Info
  label: string
  /** A word more under the label, or why it's off. */
  hint?: string
  onSelect: () => void
  danger?: boolean
  disabled?: boolean
}

// ── The "⋯" menu ──

const MENU_WIDTH = 240
/** Kept clear of the screen's edges, and the gap to the trigger. */
const MENU_MARGIN = 8
const MENU_GAP = 4

type MenuPos = { left: number; top: number; maxHeight: number }

/** Under the trigger, right edges aligned — above it when the menu doesn't fit
 *  below. When it fits on neither side (a long menu on a short screen), it
 *  sits as low as it can without leaving the screen and scrolls if it must:
 *  never an item above the top, out of reach. */
function placeMenu(t: DOMRect, height: number): MenuPos {
  const vh = window.innerHeight
  const left = Math.max(MENU_MARGIN, Math.min(t.right - MENU_WIDTH, window.innerWidth - MENU_WIDTH - MENU_MARGIN))
  const below = vh - MENU_MARGIN - (t.bottom + MENU_GAP)
  const above = t.top - MENU_GAP - MENU_MARGIN
  if (height <= below) return { left, top: t.bottom + MENU_GAP, maxHeight: below }
  if (height <= above) return { left, top: t.top - MENU_GAP - height, maxHeight: above }
  const top = Math.max(MENU_MARGIN, vh - MENU_MARGIN - height)
  return { left, top, maxHeight: vh - MENU_MARGIN - top }
}

export function AgentMenu({ agentName, items, phone = false, shown = false }: {
  agentName: string
  items: AgentMenuItem[]
  /** Phone row: a 44px touch target, always shown. */
  phone?: boolean
  /** Shown all the time — otherwise only while its row is hovered (or on a touch screen). */
  shown?: boolean
}) {
  const [open, setOpen] = useState(false)
  // Where it sits (placeMenu); none yet while it is drawn unseen to be measured.
  const [pos, setPos] = useState<MenuPos | null>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  const toggle = () => {
    if (open) { setOpen(false); return }
    setPos(null)
    setOpen(true)
  }
  // Placed by its real height, before it shows: a hint that wraps makes its
  // row taller than any guess.
  useLayoutEffect(() => {
    if (!open) return
    const t = triggerRef.current?.getBoundingClientRect()
    const m = menuRef.current
    if (t && m) setPos(placeMenu(t, m.offsetHeight))
  }, [open])

  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => {
      const n = e.target as Node
      if (menuRef.current?.contains(n) || triggerRef.current?.contains(n)) return
      setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.stopPropagation()
      setOpen(false)
    }
    const close = () => setOpen(false)
    // The page scrolling moves the trigger away; the menu's own scroll doesn't.
    const onScroll = (e: Event) => {
      if (!menuRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey, true)
    window.addEventListener('resize', close)
    window.addEventListener('scroll', onScroll, true)
    return () => {
      document.removeEventListener('pointerdown', onDown)
      window.removeEventListener('keydown', onKey, true)
      window.removeEventListener('resize', close)
      window.removeEventListener('scroll', onScroll, true)
    }
  }, [open])

  return (
    <>
      <button
        ref={triggerRef}
        onClick={toggle}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`More for ${agentName}`}
        title={`More — ${agentName}`}
        className={phone
          ? 'flex h-11 w-11 shrink-0 items-center justify-center rounded-lg text-zinc-500 transition-colors active:bg-zinc-800'
          : `shrink-0 rounded p-1.5 transition-colors hover:bg-zinc-800 hover:text-zinc-200 focus:opacity-100 group-hover:opacity-100 pointer-coarse:opacity-100 ${
            open ? 'bg-zinc-800 text-zinc-200 opacity-100' : `text-zinc-500 ${shown ? '' : 'opacity-0'}`}`}
      >
        <MoreHorizontal className={phone ? 'h-5 w-5' : 'h-3.5 w-3.5'} />
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          role="menu"
          aria-label={agentName}
          className="fixed z-50 overflow-y-auto overscroll-contain rounded-xl border border-zinc-700/60 bg-zinc-900 p-1 shadow-2xl shadow-black/40"
          style={pos ? { ...pos, width: MENU_WIDTH } : { left: 0, top: 0, width: MENU_WIDTH, visibility: 'hidden' }}
        >
          {items.map(({ icon: Icon, label, hint, onSelect, danger, disabled }) => (
            <button
              key={label}
              role="menuitem"
              disabled={disabled}
              onClick={() => { setOpen(false); onSelect() }}
              className={`flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                danger
                  ? 'text-red-600 hover:bg-red-500/10 dark:text-red-400'
                  : 'text-zinc-200 hover:bg-zinc-800'
              } ${phone ? 'py-3' : ''}`}
            >
              <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${danger ? '' : 'text-zinc-400'}`} />
              <span className="min-w-0">
                <span className="block text-sm font-medium">{label}</span>
                {hint && <span className="block text-[11px] leading-snug text-zinc-500">{hint}</span>}
              </span>
            </button>
          ))}
        </div>,
        document.body,
      )}
    </>
  )
}

// ── Dialog shell ──
//
// `lg` (Info) and `md` (Instructions): a panel on a larger screen, the whole
// screen on a phone. `sm`: a narrow panel — a bottom sheet on a phone — like
// kiosk Settings.

const DIALOG_SIZE = {
  sm: 'max-h-[100dvh] max-w-md rounded-t-2xl border border-zinc-800 wide:max-h-[88vh]',
  // A sheet on a phone like `sm`; on a big screen wide enough for two columns.
  // (Plain sm:/lg: widths: a `wide:` one would outrank lg: in the stylesheet.)
  wide: 'max-h-[100dvh] rounded-t-2xl border border-zinc-800 wide:max-h-[88vh] sm:max-w-2xl lg:max-w-5xl',
  md: 'h-[100dvh] wide:h-[80vh] wide:max-w-2xl wide:border wide:border-zinc-800',
  lg: 'h-[100dvh] wide:h-[88vh] wide:max-w-5xl wide:border wide:border-zinc-800',
}

export function AgentDialog({ title, icon: Icon, agentName, size = 'sm', onClose, children }: {
  title: string
  icon: typeof Info
  agentName: string
  size?: keyof typeof DIALOG_SIZE
  onClose: () => void
  children: ReactNode
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // Wait for every listener: a file viewer, public link dialog or table
      // open inside (Files & data) marks the Esc it used to close itself.
      setTimeout(() => { if (!e.defaultPrevented) onClose() }, 0)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return createPortal(
    <div
      className={`fixed inset-0 z-50 flex justify-center bg-black/60 wide:items-center wide:p-4 ${size === 'sm' || size === 'wide' ? 'items-end' : 'items-stretch'}`}
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-label={`${title} — ${agentName}`}
        className={`relative flex w-full flex-col overflow-hidden bg-zinc-950 pb-[env(safe-area-inset-bottom)] shadow-2xl wide:rounded-2xl wide:pb-0 ${DIALOG_SIZE[size]}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex h-14 shrink-0 items-center justify-between gap-2 border-b border-zinc-800 pl-5 pr-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <Icon className="h-4 w-4 shrink-0 text-violet-400" />
            <div className="min-w-0">
              <h2 className="truncate text-sm font-semibold text-zinc-100">{title}</h2>
              <div className="truncate text-[11px] text-zinc-500">{agentName}</div>
            </div>
          </div>
          <button onClick={onClose} title="Close" aria-label="Close" className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-5 w-5 wide:h-4 wide:w-4" />
          </button>
        </div>
        {children}
      </div>
    </div>,
    document.body,
  )
}

// ── Info: intentions and topics ──

type InfoTab = 'intentions' | 'topics'
const INFO_TABS: { id: InfoTab; label: string; icon: typeof Info }[] = [
  { id: 'intentions', label: 'Intentions', icon: Target },
  { id: 'topics', label: 'Topics', icon: Tags },
]

export function KioskAgentInfoDialog({ agent, phone, onClose }: { agent: KioskAgent; phone: boolean; onClose: () => void }) {
  const [tab, setTab] = useState<InfoTab>('intentions')
  // Both read the agent itself, through Flight Deck: a stopped one has nothing to say.
  const up = agent.reachable && agent.port > 0

  return (
    <AgentDialog title="Info" icon={Info} agentName={agent.name} size="lg" onClose={onClose}>
      <div className="shrink-0 border-b border-zinc-800 p-2">
        <div className="grid grid-cols-2 rounded-xl bg-zinc-900 p-1 wide:max-w-xs" role="tablist">
          {INFO_TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              role="tab"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={`flex h-9 items-center justify-center gap-1.5 rounded-lg text-sm font-medium transition-colors ${
                tab === id ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              <Icon className={`h-4 w-4 ${id === 'intentions' ? 'text-amber-400' : 'text-sky-400'}`} />
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">
        {!up ? (
          <div className="flex h-full items-center justify-center px-8 text-center text-sm text-zinc-500">
            Power {agent.name} on to see its intentions and topics.
          </div>
        ) : tab === 'intentions' ? (
          <IntentionsPanel key={agent.id} embedded host={agent.host} port={agent.port} auth={agent.auth} agentName={agent.name} />
        ) : (
          <TopicsPanel key={agent.id} embedded phone={phone} host={agent.host} port={agent.port} auth={agent.auth} agentName={agent.name} />
        )}
      </div>
    </AgentDialog>
  )
}

// ── Settings: model, vision, cognitive mode and reasoning effort ──

const VISION_CHOICES = VISION_OPTIONS.map((o) => {
  const [title, detail = ''] = o.label.split(' — ')
  return { value: o.value, title, detail: detail.charAt(0).toUpperCase() + detail.slice(1) }
})

// Providers that sign in on their own — no key, no endpoint.
const SIGN_IN_PROVIDERS = new Set(['antigravity-cli', 'antigravity', 'google-subscription'])

/** The endpoint a provider gets when nobody names one — as the server writes it. */
function defaultEndpoint(provider: string): string {
  return provider === 'ollama' ? 'http://127.0.0.1:11434' : ''
}

interface ModelSetting { provider: string; model: string; baseUrl: string }

export function KioskAgentSettingsDialog({ agent, restarting, onRestart, onClose }: {
  agent: KioskAgent & { kind: 'docker' | 'process' }
  /** Being restarted right now (or switched on/off). */
  restarting: boolean
  /** Restart it so a start-up setting (model, vision) applies. Resolves once it's on its way back. */
  onRestart: () => Promise<boolean>
  onClose: () => void
}) {
  const { kind, startKey: key } = agent
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [mode, setMode] = useState('neutra')
  const [modeSaved, setModeSaved] = useState(false)
  const [vision, setVision] = useState('auto')
  // What its config holds, and the change being typed.
  const [model, setModel] = useState<ModelSetting>({ provider: '', model: '', baseUrl: '' })
  const [draft, setDraft] = useState({ provider: '', model: '', baseUrl: '' })
  const [apiKey, setApiKey] = useState('')
  // The endpoint follows the provider until it is typed in.
  const [urlTouched, setUrlTouched] = useState(false)
  const [modelSaved, setModelSaved] = useState(false)
  const [saving, setSaving] = useState<'' | 'mode' | 'vision' | 'model'>('')
  // Model and vision are read when the agent starts: changed since it last did.
  const [pendingRestart, setPendingRestart] = useState(false)
  const [restarted, setRestarted] = useState(false)
  const running = agent.state === 'running'

  useEffect(() => {
    let live = true
    Promise.all([getAgentMode(kind, key), getAgentConfig(kind, key)])
      .then(([m, cfg]) => {
        if (!live) return
        const info = parseModelFromYaml(cfg.config_yaml || '')
        setMode(m.mode || 'neutra')
        setVision(info.vision)
        setModel({ provider: info.provider, model: info.model, baseUrl: info.base_url })
        setDraft({ provider: info.provider, model: info.model, baseUrl: info.base_url })
      })
      .catch((e) => { if (live) setError(`Couldn't load its settings: ${e instanceof Error ? e.message : String(e)}`) })
      .finally(() => { if (live) setLoading(false) })
    return () => { live = false }
  }, [kind, key])

  const provider = draft.provider.trim()
  const switched = provider !== model.provider
  const modelDirty = switched || draft.model.trim() !== model.model || draft.baseUrl.trim() !== model.baseUrl || !!apiKey.trim()
  const signIn = SIGN_IN_PROVIDERS.has(provider)

  const pickProvider = (next: string) => {
    setDraft((d) => ({ ...d, provider: next, baseUrl: urlTouched ? d.baseUrl : next.trim() === model.provider ? model.baseUrl : defaultEndpoint(next.trim()) }))
  }

  const saveModel = async () => {
    if (!modelDirty || !provider || !draft.model.trim() || saving) return
    setSaving('model'); setError(''); setModelSaved(false)
    const next = { provider, model: draft.model.trim(), baseUrl: signIn ? '' : draft.baseUrl.trim() }
    try {
      await setAgentModel(kind, key, {
        provider: next.provider, model: next.model,
        ...(signIn ? {} : { base_url: next.baseUrl, ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}) }),
      })
      // The agent list shows the new model at once (and no older label of it hides it).
      if (kind === 'process') useProcessStore.getState().setModelOverride(key, next.provider, next.model)
      setModel(next)
      setDraft(next)
      setApiKey(''); setUrlTouched(false)
      setPendingRestart(true); setRestarted(false)
      setModelSaved(true)
      setTimeout(() => setModelSaved(false), 2000)
    } catch (e) {
      setError(`Couldn't change its model: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setSaving('')
    }
  }

  const undoModel = () => {
    setDraft({ provider: model.provider, model: model.model, baseUrl: model.baseUrl })
    setApiKey(''); setUrlTouched(false)
  }

  // The mode applies on its next reply — no restart.
  const pickMode = async (next: string) => {
    if (next === mode || saving) return
    const was = mode
    setMode(next); setSaving('mode'); setError(''); setModeSaved(false)
    try {
      await setAgentMode(kind, key, next)
      // …and the agent cards' copy of it.
      const cards = kind === 'docker' ? useContainerStore.getState() : useProcessStore.getState()
      cards.setCognitiveMode(key, next)
      setModeSaved(true)
      setTimeout(() => setModeSaved(false), 2000)
    } catch (e) {
      setMode(was)
      setError(`Couldn't change how it thinks: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setSaving('')
    }
  }

  const pickVision = async (next: string) => {
    if (next === vision || saving) return
    const was = vision
    setVision(next); setSaving('vision'); setError('')
    try {
      await setAgentVision(kind, key, next)
      setPendingRestart(true)
      setRestarted(false)
    } catch (e) {
      setVision(was)
      setError(`Couldn't change vision: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setSaving('')
    }
  }

  const restart = async () => {
    if (await onRestart()) { setPendingRestart(false); setRestarted(true) }
  }

  // A model change being typed isn't thrown away without asking.
  const close = () => {
    if (modelDirty && !confirm('Discard the model change you have not saved?')) return
    onClose()
  }

  const textOnly = SIGN_IN_PROVIDERS.has(model.provider)
  const current = MODES.find((m) => m.id === mode) ?? MODES[0]
  const providerLabel = LLM_PROVIDERS.find((p) => p.value === provider)?.label || provider || 'the new provider'
  const keyHint = switched
    ? `The key for ${providerLabel}. Leave it empty to use your team's key, if your deck has one.`
    : 'Leave it empty to keep the key it uses now.'

  return (
    <AgentDialog title="Agent settings" icon={SlidersHorizontal} agentName={agent.name} size="wide" onClose={close}>
      <div className="min-h-0 flex-1 space-y-6 overflow-y-auto overscroll-contain p-5 lg:p-6">
        {loading ? (
          <div className="flex items-center justify-center py-10 text-sm text-zinc-500">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading…
          </div>
        ) : (
          // Two columns on a big screen: what it runs on (model, vision) on
          // the left, how it thinks (mode, reasoning effort) on the right.
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 lg:gap-8">
            <div className="min-w-0 space-y-6">
              <section>
                <SectionTitle icon={Cpu} title="Model" saving={saving === 'model'} saved={modelSaved} />
                <p className="mb-2.5 text-[11px] leading-relaxed text-zinc-500">
                  The AI model it runs on, and the key that pays for it. Applies once the agent restarts.
                </p>
                <div className="grid grid-cols-1 gap-2 wide:grid-cols-[2fr_3fr]">
                  <label className="block min-w-0">
                    <span className="mb-1 block text-[11px] font-medium text-zinc-400">Provider</span>
                    <select
                      value={draft.provider}
                      onChange={(e) => pickProvider(e.target.value)}
                      className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-sm text-zinc-200 focus:border-violet-500/60 focus:outline-none"
                    >
                      {/* What it runs on now, even if it isn't one of the list. */}
                      {!LLM_PROVIDERS.some((p) => p.value === model.provider) && (
                        <option value={model.provider}>{model.provider || '— choose —'}</option>
                      )}
                      {LLM_PROVIDERS.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
                    </select>
                  </label>
                  <TextField label="Model" value={draft.model} onChange={(v) => setDraft((d) => ({ ...d, model: v }))} placeholder="e.g. gpt-4.1-mini" />
                </div>
                {signIn ? (
                  <p className="mt-2 text-[11px] text-zinc-500">Local Google sign-in — no API key or endpoint.</p>
                ) : (
                  <div className="mt-2 space-y-2">
                    <TextField
                      label="API key" type="password" value={apiKey} onChange={setApiKey}
                      placeholder={switched ? 'sk-…' : 'Unchanged'} hint={keyHint}
                    />
                    <TextField
                      label="Endpoint (optional)" value={draft.baseUrl}
                      onChange={(v) => { setUrlTouched(true); setDraft((d) => ({ ...d, baseUrl: v })) }}
                      placeholder="The provider's own" hint="Only for a self-hosted or local server, e.g. http://127.0.0.1:8000/v1"
                    />
                  </div>
                )}
                {modelDirty && (
                  <div className="mt-2.5 flex items-center justify-end gap-2">
                    <button
                      onClick={undoModel}
                      disabled={saving === 'model'}
                      className="rounded-lg px-3 py-1.5 text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
                    >
                      Undo
                    </button>
                    <button
                      onClick={saveModel}
                      disabled={!!saving || !provider || !draft.model.trim()}
                      className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
                    >
                      {saving === 'model' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
                      Save model
                    </button>
                  </div>
                )}
              </section>

              <section>
                <SectionTitle icon={Eye} title="Vision" saving={saving === 'vision'} />
                <p className="mb-2.5 text-[11px] leading-relaxed text-zinc-500">
                  Whether its own model looks at the pictures you send.
                  {model.provider && <> It runs on <span className="text-zinc-400">{model.provider}{model.model ? ` / ${model.model}` : ''}</span>.</>}
                </p>
                {textOnly ? (
                  <p className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2.5 text-xs text-zinc-400">
                    This model reads text only.
                  </p>
                ) : (
                  <div role="radiogroup" aria-label="Vision" className="space-y-1.5">
                    {VISION_CHOICES.map((o) => {
                      const on = o.value === vision
                      return (
                        <button
                          key={o.value}
                          role="radio"
                          aria-checked={on}
                          onClick={() => pickVision(o.value)}
                          disabled={!!saving}
                          className={`flex w-full items-center gap-2.5 rounded-lg border px-3 py-2 text-left transition-colors disabled:cursor-wait ${
                            on ? 'border-violet-500/50 bg-violet-500/10' : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900'
                          }`}
                        >
                          <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full border ${on ? 'border-violet-400 bg-violet-500' : 'border-zinc-600'}`}>
                            {on && <span className="h-1.5 w-1.5 rounded-full bg-white" />}
                          </span>
                          <span className="min-w-0">
                            <span className="block text-xs font-medium text-zinc-200">{o.title}</span>
                            <span className="block text-[11px] text-zinc-500">{o.detail}</span>
                          </span>
                        </button>
                      )
                    })}
                  </div>
                )}
              </section>
            </div>
            <div className="min-w-0 space-y-6">
              <section>
                <SectionTitle icon={BrainCog} title="How it thinks" saving={saving === 'mode'} saved={modeSaved} />
                <p className="mb-2.5 text-[11px] leading-relaxed text-zinc-500">
                  Its cognitive mode — the way it reasons. Applies from its next reply.
                </p>
                <div role="radiogroup" aria-label="Cognitive mode" className="grid grid-cols-2 gap-1.5">
                  {MODES.map((m) => {
                    const on = m.id === mode
                    return (
                      <button
                        key={m.id}
                        role="radio"
                        aria-checked={on}
                        onClick={() => pickMode(m.id)}
                        disabled={!!saving}
                        className={`flex min-w-0 items-start gap-2 rounded-lg border px-2.5 py-2 text-left transition-colors disabled:cursor-wait ${
                          on ? `border-violet-500/50 ${m.bgActive}` : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900'
                        }`}
                      >
                        <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${m.color}`} />
                        <span className="min-w-0">
                          <span className={`block truncate text-xs font-medium ${on ? m.textColor : 'text-zinc-200'}`}>{m.label}</span>
                          <span className="block truncate text-[10px] text-zinc-500">{m.character}</span>
                        </span>
                      </button>
                    )
                  })}
                </div>
                <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">{current.description}</p>
              </section>

              <KioskReasoningSettings kind={kind} agentKey={key} />
            </div>
          </div>
        )}
        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {error}
          </div>
        )}
      </div>
      {(pendingRestart || restarted) && (
        <div className="shrink-0 border-t border-zinc-800 px-5 py-3">
          {pendingRestart ? (
            <div className="flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
              <span className="min-w-0 flex-1">
                {running ? 'Saved. It applies once the agent restarts.' : 'Saved. It applies the next time the agent starts.'}
              </span>
              {running && (
                <button
                  onClick={restart}
                  disabled={restarting}
                  className="flex shrink-0 items-center gap-1 rounded-md bg-amber-500/20 px-2 py-1 font-medium hover:bg-amber-500/30 disabled:opacity-60"
                >
                  {restarting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCw className="h-3.5 w-3.5" />}
                  Restart now
                </button>
              )}
            </div>
          ) : (
            <p className="flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400">
              <Check className="h-3.5 w-3.5" /> Restarting — it's back in a moment.
            </p>
          )}
        </div>
      )}
    </AgentDialog>
  )
}

function TextField({ label, value, onChange, placeholder, hint, type = 'text' }: {
  label: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
  hint?: string
  type?: 'text' | 'password'
}) {
  return (
    <label className="block min-w-0">
      <span className="mb-1 block text-[11px] font-medium text-zinc-400">{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        spellCheck={false}
        autoComplete={type === 'password' ? 'new-password' : 'off'}
        className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
      />
      {hint && <span className="mt-1 block text-[11px] leading-snug text-zinc-500">{hint}</span>}
    </label>
  )
}

function SectionTitle({ icon: Icon, title, saving, saved = false }: { icon: typeof Info; title: string; saving: boolean; saved?: boolean }) {
  return (
    <h3 className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-400">
      <Icon className="h-3.5 w-3.5" />
      {title}
      {saving && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
      {saved && <span className="flex items-center gap-0.5 normal-case tracking-normal text-emerald-600 dark:text-emerald-400"><Check className="h-3 w-3" />saved</span>}
    </h3>
  )
}

// ── Instructions & insights ──
//
// Two tabs. Instructions: the owner's standing instructions, added to the
// agent's system prompt in every conversation — saved where the agent cards
// keep them; open chats are told at once, so they apply from its next reply.
// Insights: what the agent remembers between conversations, read from the
// agent itself (KioskInsightsPanel). Both stay mounted, so switching keeps
// unsaved text and a search.

const MAX_INSTRUCTIONS = 64_000  // the server's limit

type MindTab = 'instructions' | 'insights'

export function KioskAgentInstructionsDialog({ agent, initialTab = 'instructions', onClose }: {
  agent: KioskAgent & { kind: 'docker' | 'process' }
  initialTab?: MindTab
  onClose: () => void
}) {
  const { kind, startKey: key } = agent
  const cards = () => (kind === 'docker' ? useContainerStore.getState() : useProcessStore.getState())
  const [saved] = useState(() => cards().getFleetInstructions(key))
  const [text, setText] = useState(saved)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [tab, setTab] = useState<MindTab>(initialTab)
  const [total, setTotal] = useState<number | null>(null)
  const dirty = text.trim() !== saved.trim()
  // Insights live in the agent: a stopped one has none to show.
  const up = agent.reachable && agent.port > 0
  const address = useMemo(() => (up ? { host: agent.host, port: agent.port, auth: agent.auth } : null),
    [up, agent.host, agent.port, agent.auth])
  const onTotal = useCallback((n: number) => setTotal(n), [])

  const save = async () => {
    setSaving(true); setError('')
    try {
      await cards().setFleetInstructions(key, text.trim())
      useChatStore.getState().resendFleetIdentity(agent.id)
      useNotificationStore.getState().add('info', 'Instructions saved', `${agent.name} follows them from its next reply.`)
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setSaving(false)
    }
  }

  const close = () => {
    if (saving) return
    if (dirty && !confirm('Discard the changes to its instructions?')) return
    onClose()
  }

  return (
    <AgentDialog title="Instructions & insights" icon={BookOpen} agentName={agent.name} size="lg" onClose={close}>
      <div className="shrink-0 border-b border-zinc-800 p-2">
        <div className="grid grid-cols-2 rounded-xl bg-zinc-900 p-1 wide:max-w-sm" role="tablist">
          <MindTabButton on={tab === 'instructions'} onClick={() => setTab('instructions')} icon={BookOpen} iconClass="text-violet-400" label="Instructions">
            {dirty && <span className="h-1.5 w-1.5 rounded-full bg-amber-500" title="Not saved yet" />}
          </MindTabButton>
          <MindTabButton on={tab === 'insights'} onClick={() => setTab('insights')} icon={Brain} iconClass="text-amber-400" label="Insights">
            {total !== null && (
              <span className="rounded-full bg-zinc-700/70 px-1.5 text-[10px] font-semibold tabular-nums text-zinc-300">{total}</span>
            )}
          </MindTabButton>
        </div>
      </div>

      <div role="tabpanel" className={tab === 'instructions' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
        <div className="flex min-h-0 flex-1 flex-col gap-3 px-5 pb-4 pt-4">
          <p className="shrink-0 text-xs leading-relaxed text-zinc-500">
            What {agent.name} should always keep in mind — tone, format, house rules, or things about
            you and your work. They go with every conversation and apply from its next reply.
          </p>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            maxLength={MAX_INSTRUCTIONS}
            autoFocus={initialTab === 'instructions'}
            placeholder={'For example:\n- Answer in British English\n- Keep answers short, with bullet points\n- Cite your sources\n- I run a small bakery in Zagreb'}
            className="min-h-[12rem] w-full flex-1 resize-none rounded-xl border border-zinc-700 bg-zinc-950 px-4 py-3 text-sm leading-relaxed text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none wide:font-mono wide:text-[13px]"
          />
          {text.length > MAX_INSTRUCTIONS * 0.9 && (
            <p className="shrink-0 text-right text-[11px] text-amber-700 dark:text-amber-400">
              {text.length.toLocaleString()} / {MAX_INSTRUCTIONS.toLocaleString()} characters
            </p>
          )}
          {error && (
            <div className="flex shrink-0 items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> Couldn't save them: {error}
            </div>
          )}
        </div>
        <div className="flex shrink-0 items-center justify-end gap-2 border-t border-zinc-800 px-5 py-3.5">
          <span className="mr-auto hidden text-[11px] text-zinc-500 wide:inline">
            {dirty ? 'Not saved yet' : `${text.trim().length.toLocaleString()} characters`}
          </span>
          <button
            onClick={close}
            disabled={saving}
            className="rounded-lg px-3.5 py-2 text-sm font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={save}
            disabled={saving || !dirty}
            className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
          >
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
            Save
          </button>
        </div>
      </div>

      <div role="tabpanel" className={tab === 'insights' ? 'min-h-0 flex-1' : 'hidden'}>
        <KioskInsightsPanel address={address} agentName={agent.name} onTotal={onTotal} />
      </div>
    </AgentDialog>
  )
}

function MindTabButton({ on, onClick, icon: Icon, iconClass, label, children }: {
  on: boolean
  onClick: () => void
  icon: typeof Info
  iconClass: string
  label: string
  children?: ReactNode
}) {
  return (
    <button
      role="tab"
      aria-selected={on}
      onClick={onClick}
      className={`flex h-9 items-center justify-center gap-1.5 rounded-lg text-sm font-medium transition-colors ${
        on ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
      }`}
    >
      <Icon className={`h-4 w-4 ${iconClass}`} />
      {label}
      {children}
    </button>
  )
}

// ── Remove ──

export function KioskRemoveAgentDialog({ agent, busy, onRemove, onOpenLibrary, onOpenData, onClose }: {
  agent: KioskAgent
  /** At work right now (in a chat open here). */
  busy: boolean
  /** Removes it; rejects with what went wrong. */
  onRemove: () => Promise<void>
  /** Opens the File library on its files — to keep some before they go. */
  onOpenLibrary?: () => void
  /** Opens the Data library on its tables — likewise. */
  onOpenData?: () => void
  onClose: () => void
}) {
  const [removing, setRemoving] = useState(false)
  const [error, setError] = useState('')
  const cancelRef = useRef<HTMLButtonElement>(null)
  useEffect(() => { cancelRef.current?.focus() }, [])

  const remove = async () => {
    setRemoving(true); setError('')
    try {
      await onRemove()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setRemoving(false)
    }
  }

  return (
    <AgentDialog title="Remove agent" icon={Trash2} agentName={agent.name} onClose={removing ? () => {} : onClose}>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto overscroll-contain p-5">
        <div className="flex items-start gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-red-500/10 text-red-600 dark:text-red-400">
            <AlertTriangle className="h-[18px] w-[18px]" />
          </span>
          <div className="min-w-0 text-sm text-zinc-300">
            <p className="font-medium text-zinc-100">Remove {agent.name} for good?</p>
            <p className="mt-1 text-zinc-400">
              {busy ? 'It is working right now — that stops. ' : ''}
              It is switched off and deleted from this deck, with everything it keeps:
            </p>
            <ul className="mt-2 list-disc space-y-0.5 pl-5 text-[13px] text-zinc-400">
              <li>its conversations and what it remembers</li>
              <li>its files and datastore tables</li>
              <li>its settings, and the scheduled tasks that run it</li>
            </ul>
            <p className="mt-2 font-medium text-red-600 dark:text-red-400">This can't be undone.</p>
          </div>
        </div>
        {(onOpenLibrary || onOpenData) && (
          <div className="space-y-2">
            {onOpenLibrary && (
              <KeepLink icon={LibraryBig} iconClass="text-violet-600 dark:text-violet-400" onClick={onOpenLibrary} disabled={removing}>
                Want to keep a file? Download it from the <span className="font-medium text-zinc-200">File library</span> first.
              </KeepLink>
            )}
            {onOpenData && (
              <KeepLink icon={Database} iconClass="text-emerald-600 dark:text-emerald-400" onClick={onOpenData} disabled={removing}>
                Want to keep a table? Download it from the <span className="font-medium text-zinc-200">Data library</span> first.
              </KeepLink>
            )}
          </div>
        )}
        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> Couldn't remove it: {error}
          </div>
        )}
      </div>
      <div className="flex shrink-0 items-center justify-end gap-2 border-t border-zinc-800 px-5 py-3.5">
        <button
          ref={cancelRef}
          onClick={onClose}
          disabled={removing}
          className="rounded-lg px-3.5 py-2 text-sm font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          onClick={remove}
          disabled={removing}
          className="flex items-center gap-1.5 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-500 disabled:opacity-60"
        >
          {removing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
          {removing ? 'Removing…' : 'Remove agent'}
        </button>
      </div>
    </AgentDialog>
  )
}

/** A way from the removal to the library that holds something of the agent's. */
function KeepLink({ icon: Icon, iconClass, onClick, disabled, children }: {
  icon: typeof Info
  iconClass: string
  onClick: () => void
  disabled: boolean
  children: ReactNode
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="flex w-full items-center gap-2.5 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3.5 py-2.5 text-left text-xs text-zinc-400 transition-colors hover:border-violet-500/40 hover:bg-zinc-800/60 disabled:opacity-50"
    >
      <Icon className={`h-4 w-4 shrink-0 ${iconClass}`} />
      <span>{children}</span>
    </button>
  )
}
