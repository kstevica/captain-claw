import { useEffect, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  Bot,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Compass,
  Cpu,
  Eye,
  Info,
  Layers,
  Loader2,
  Plug,
  UserRound,
  X,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { getMyInfo, type InfoTier, type MyInfo } from '../../services/profile'
import { fmtCtxTokens } from '../../services/tierConfig'
import { PlanUsageMeters } from '../common/PlanUsage'
import { APP_VERSION, BUILD_DATE } from '../../version'

// ── Kiosk: Info ──────────────────────────────────────────────────────
//
// Everything a kiosk user may want to know about their corner of the deck,
// read-only: which models their agents are created on (the tier set — theirs
// or the team default), their agents, the archetypes they can pick from, what
// their agents know about them (deck-wide defaults + their own profile, and
// the exact text the agents receive), their plan and their connections.
// Full-screen on a phone, a panel on a larger screen.

export interface InfoAgent {
  id: string
  name: string
  running: boolean
  shared: boolean
  ownerName?: string
  model?: string
  provider?: string
}

const SOURCE_TEXT: Record<MyInfo['tiers']['source'], (name: string) => string> = {
  own: (name) => `Your own model set${name ? ` “${name}”` : ''}.`,
  team: (name) => `The team's default set${name ? ` “${name}”` : ''}, set up by your admin.`,
  none: () => "No model set yet: new agents use the deck's built-in models.",
  byok: () => 'Your own API keys (Settings → Your own AI keys) — no daily limits.',
}

const LIMIT_LABELS: [keyof MyInfo['account']['limits'], string][] = [
  ['max_agents', 'Agents'],
  ['spawns_per_hour', 'New agents / hour'],
  ['requests_per_minute', 'Requests / min'],
  ['max_storage_mb', 'Storage (MB)'],
]

function ctx(t: InfoTier): string {
  if (!t.input_ctx && !t.output_ctx) return ''
  return [t.input_ctx ? `${fmtCtxTokens(t.input_ctx)} in` : '', t.output_ctx ? `${fmtCtxTokens(t.output_ctx)} out` : '']
    .filter(Boolean).join(' · ')
}

export function KioskInfoDialog({ agents, onClose, onOpenProfile, onOpenConnections, onOpenGuide }: {
  agents: InfoAgent[]
  onClose: () => void
  onOpenProfile: () => void
  onOpenConnections: () => void
  onOpenGuide: () => void
}) {
  const branding = useAuthStore((s) => s.branding)
  const [info, setInfo] = useState<MyInfo | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let live = true
    getMyInfo()
      .then((d) => { if (live) setInfo(d) })
      .catch((e) => { if (live) setError(e instanceof Error ? e.message : String(e)) })
    return () => { live = false }
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const running = agents.filter((a) => a.running).length
  const own = agents.filter((a) => !a.shared).length
  const maxAgents = info?.account.limits.max_agents

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label="Info"
        className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 shadow-2xl wide:h-[85vh] wide:max-w-2xl wide:rounded-2xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-zinc-800 px-4 wide:px-5">
          <h2 className="flex items-center gap-2 text-base font-semibold text-zinc-100">
            <Info className="h-4 w-4 text-violet-400" />
            Info
          </h2>
          <button onClick={onClose} aria-label="Close" className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto overscroll-contain p-4 pb-[max(1rem,env(safe-area-inset-bottom))] wide:p-5">
          {/* The deck */}
          <div className="flex items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
            {branding.logo
              ? <img src={branding.logo} alt="" className="h-11 w-11 shrink-0 rounded-lg object-contain" />
              : <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-violet-500/15 text-violet-400"><Bot className="h-5 w-5" /></span>}
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-zinc-100">{branding.name}</div>
              {branding.description && <div className="line-clamp-2 text-xs text-zinc-500">{branding.description}</div>}
              <div className="mt-0.5 text-[11px] text-zinc-600">Flight Deck v{APP_VERSION} · {BUILD_DATE}</div>
            </div>
          </div>

          {/* The guide every kiosk user gets on their first visit */}
          <button
            onClick={onOpenGuide}
            className="flex w-full items-center gap-3 rounded-xl border border-violet-500/30 bg-violet-500/5 p-4 text-left transition-colors hover:bg-violet-500/10"
          >
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-violet-500/15 text-violet-400"><Compass className="h-[18px] w-[18px]" /></span>
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-medium text-zinc-100">Getting started</span>
              <span className="block text-xs text-zinc-500">How to create an agent, connect Google, Microsoft and Telegram, and more</span>
            </span>
            <ChevronRight className="h-4 w-4 shrink-0 text-zinc-500" />
          </button>

          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-700 dark:text-red-300">
              Couldn't load your details: {error}
            </div>
          )}
          {!info && !error && (
            <div className="flex justify-center py-10"><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></div>
          )}

          {info && (
            <>
              {/* Account */}
              <Section icon={UserRound} title="Your account">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
                  <span className="font-medium text-zinc-200">{info.account.display_name || info.account.email}</span>
                  {info.account.display_name && <span className="text-zinc-500">{info.account.email}</span>}
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  <Pill>{info.account.usage?.plan_label || info.account.plan} plan</Pill>
                  {info.account.role === 'admin' && <Pill>admin</Pill>}
                  {info.account.kiosk && <Pill>simple chat</Pill>}
                  {info.account.created_at && <Pill>since {new Date(info.account.created_at).toLocaleDateString()}</Pill>}
                </div>
                <dl className="mt-3 grid grid-cols-2 gap-2">
                  {LIMIT_LABELS.filter(([k]) => info.account.limits[k] != null).map(([k, label]) => (
                    <div key={k} className="rounded-lg bg-zinc-900 px-3 py-2">
                      <dt className="text-[11px] text-zinc-500">{label}</dt>
                      <dd className="text-sm font-medium text-zinc-200">
                        {k === 'max_agents' ? `${own} of ${info.account.limits[k]}` : info.account.limits[k]}
                      </dd>
                    </div>
                  ))}
                </dl>
                <PlanUsageMeters initial={info.account.usage} />
              </Section>

              {/* Agents */}
              <Section icon={Bot} title="Your agents" note={`${agents.length} agent${agents.length === 1 ? '' : 's'} · ${running} running`}>
                {agents.length === 0 ? (
                  <p className="text-sm text-zinc-500">No agents yet. Create one with “New agent”.</p>
                ) : (
                  <ul className="divide-y divide-zinc-800/60">
                    {agents.map((a) => (
                      <li key={a.id} className="flex items-center gap-2.5 py-2">
                        <span className={`h-2 w-2 shrink-0 rounded-full ${a.running ? 'bg-emerald-500 dark:bg-emerald-400' : 'bg-zinc-600'}`} />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm text-zinc-200">{a.name}</div>
                          <div className="truncate text-[11px] text-zinc-500">
                            {a.shared ? `Shared by ${a.ownerName || 'another user'}` : a.model ? `${a.model}${a.provider ? ` · ${a.provider}` : ''}` : a.running ? 'Running' : 'Stopped'}
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
                {maxAgents != null && own >= maxAgents && (
                  <p className="mt-2 text-xs text-amber-700 dark:text-amber-400">You're at your plan's agent limit.</p>
                )}
              </Section>

              {/* Models */}
              <Section icon={Cpu} title="Models (tiers)">
                <p className="mb-3 text-xs text-zinc-500">
                  {SOURCE_TEXT[info.tiers.source](info.tiers.set_name)} Each archetype names a tier; a new agent gets that tier's model.
                </p>
                <ul className="space-y-2">
                  {info.tiers.tiers.map((t) => (
                    <li key={t.id} className="rounded-lg bg-zinc-900 px-3 py-2.5">
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                        <span className="text-sm font-medium text-zinc-200">{t.label}</span>
                        {t.use && <span className="text-[11px] text-zinc-500">{t.use}</span>}
                        {t.archetypes > 0 && (
                          <span className="ml-auto text-[11px] text-zinc-500">{t.archetypes} archetype{t.archetypes === 1 ? '' : 's'}</span>
                        )}
                      </div>
                      <div className="mt-1 break-all font-mono text-xs text-zinc-300">{t.model || '—'}</div>
                      <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-zinc-500">
                        <span>{t.provider}{t.endpoint ? ` · ${t.endpoint}` : ''}</span>
                        {ctx(t) && <span>{ctx(t)}</span>}
                        {t.vision && <span className="flex items-center gap-0.5 text-violet-400"><Eye className="h-3 w-3" />sees images</span>}
                        {!t.from_set && <span className="italic">deck default</span>}
                      </div>
                    </li>
                  ))}
                </ul>
              </Section>

              {/* Archetypes */}
              <Section icon={Layers} title="Archetypes" note={String(info.archetypes.total)}>
                <p className="text-sm text-zinc-400">
                  {info.archetypes.total} to create agents from: {info.archetypes.base} built in
                  {info.archetypes.own > 0 && `, ${info.archetypes.own} of yours`}
                  {info.archetypes.shared > 0 && `, ${info.archetypes.shared} shared with you`}.
                </p>
              </Section>

              {/* What the agents know */}
              {info.profile && (
                <Section icon={BookOpen} title="What your agents know about you">
                  <Field label="Deck-wide company (from your admin)" text={info.profile.deck.company}
                    note={info.profile.own.company ? 'Your own company text replaces this.' : undefined} />
                  <Field label="Deck-wide instructions (from your admin)" text={info.profile.deck.instructions} />
                  <Field label="About you" text={info.profile.own.about_me} empty="Not filled in." />
                  <Field label="Your company" text={info.profile.own.company} empty="Not filled in: the deck-wide text applies." />
                  <Field label="Your preferences" text={info.profile.own.instructions} empty="Not filled in." />
                  <button
                    onClick={onOpenProfile}
                    className="mt-1 rounded-lg border border-zinc-700 px-3 py-2 text-xs font-medium text-zinc-300 hover:bg-zinc-800"
                  >
                    Edit your profile
                  </button>
                  <Received full={info.profile.preview.full} compact={info.profile.preview.compact} />
                </Section>
              )}

              {/* Connections */}
              <Section icon={Plug} title="Connections">
                <ul className="divide-y divide-zinc-800/60">
                  <ConnRow name="Google" state={connState(info.connections.google)} />
                  <ConnRow name="Microsoft 365" state={connState(info.connections.microsoft)} />
                  <ConnRow name="Telegram" state={telegramState(info.connections.telegram)} />
                  <ConnRow name="Jev" state={info.connections.jev ? (info.connections.jev.connected
                    ? { text: info.connections.jev.source === 'shared' ? 'Team key' : 'Connected', ok: true }
                    : { text: 'Not connected', ok: false }) : null} />
                  <ConnRow name="Soniox (voice)" state={info.connections.soniox ? (info.connections.soniox.connected
                    ? { text: info.connections.soniox.source === 'own' ? 'Connected' : 'Team key', ok: true }
                    : { text: 'Not connected', ok: false }) : null} />
                </ul>
                <button
                  onClick={onOpenConnections}
                  className="mt-2 rounded-lg border border-zinc-700 px-3 py-2 text-xs font-medium text-zinc-300 hover:bg-zinc-800"
                >
                  Manage connections
                </button>
              </Section>
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}

function Section({ icon: Icon, title, note, children }: {
  icon: typeof Info
  title: string
  note?: string
  children: ReactNode
}) {
  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-zinc-200">
        <Icon className="h-4 w-4 text-violet-400" />
        {title}
        {note && <span className="ml-auto text-xs font-normal text-zinc-500">{note}</span>}
      </h3>
      {children}
    </section>
  )
}

function Pill({ children }: { children: ReactNode }) {
  return <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] capitalize text-zinc-400">{children}</span>
}

function Field({ label, text, empty = 'None set.', note }: { label: string; text: string; empty?: string; note?: string }) {
  return (
    <div className="mb-3">
      <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">{label}</div>
      {text
        ? <p className="mt-0.5 whitespace-pre-wrap break-words text-sm text-zinc-300">{text}</p>
        : <p className="mt-0.5 text-sm italic text-zinc-600">{empty}</p>}
      {note && <p className="mt-0.5 text-[11px] text-zinc-500">{note}</p>}
    </div>
  )
}

// The composed text exactly as agents get it: the full block, and the short
// one small-model (eco) agents get.
function Received({ full, compact }: { full: string; compact: string }) {
  const [open, setOpen] = useState(false)
  const [which, setWhich] = useState<'full' | 'compact'>('full')
  if (!full && !compact) return null
  const text = which === 'full' ? full : compact
  return (
    <div className="mt-3 rounded-lg border border-zinc-800">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-3 py-2.5 text-left text-xs font-medium text-zinc-300"
      >
        Exact text your agents receive
        <ChevronDown className={`h-4 w-4 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="border-t border-zinc-800 p-3">
          <div className="mb-2 inline-flex rounded-lg border border-zinc-700 p-0.5" role="tablist">
            {(['full', 'compact'] as const).map((k) => (
              <button
                key={k}
                role="tab"
                aria-selected={which === k}
                onClick={() => setWhich(k)}
                className={`rounded-md px-2.5 py-1 text-[11px] font-medium ${which === k ? 'bg-violet-600 text-white' : 'text-zinc-400'}`}
              >
                {k === 'full' ? 'Full' : 'Eco agents'}
              </button>
            ))}
          </div>
          <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md bg-zinc-900 p-3 text-[11px] leading-relaxed text-zinc-300">{text || '—'}</pre>
          <p className="mt-1.5 text-[11px] text-zinc-500">{text.length} characters</p>
        </div>
      )}
    </div>
  )
}

type ConnState = { text: string; ok: boolean } | null

function connState(c?: { configured: boolean; connected: boolean }): ConnState {
  if (!c) return null
  if (c.connected) return { text: 'Connected', ok: true }
  return { text: c.configured ? 'Not connected' : 'Not set up on this deck', ok: false }
}

function telegramState(t?: { configured: boolean; status: string; linked: boolean; agent: string }): ConnState {
  if (!t) return null
  if (t.status === 'approved' && t.linked) return { text: t.agent ? `Connected to ${t.agent}` : 'Connected', ok: true }
  if (t.status === 'approved') return { text: 'Approved: open the bot to finish', ok: false }
  if (t.status === 'pending') return { text: 'Waiting for approval', ok: false }
  if (t.status === 'rejected') return { text: 'Not approved', ok: false }
  return { text: t.configured ? 'Not connected' : 'Not set up on this deck', ok: false }
}

function ConnRow({ name, state }: { name: string; state: ConnState }) {
  if (!state) return null
  return (
    <li className="flex items-center justify-between gap-3 py-2">
      <span className="text-sm text-zinc-300">{name}</span>
      <span className={`flex items-center gap-1.5 text-right text-xs ${state.ok ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-500'}`}>
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${state.ok ? 'bg-emerald-500' : 'bg-zinc-600'}`} />
        {state.text}
      </span>
    </li>
  )
}
