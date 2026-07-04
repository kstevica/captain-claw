import { useEffect, useMemo, useState } from 'react'
import {
  MessagesSquare, Search, Rocket, Loader2, Crown, Gauge, MonitorUp, Monitor,
  Trash2, MessageSquare, Check, AlertTriangle, Plus, X,
} from 'lucide-react'
import { useTierConfig, type Archetype } from '../services/tierConfig'
import { spawnProcess, type SpawnConfig } from '../services/docker'
import { useProcessStore } from '../stores/processStore'
import { useChatStore } from '../stores/chatStore'
import { useDesktopPrefsStore } from '../stores/desktopPrefsStore'
import { useQuickChatStore, type QuickChatSession } from '../stores/quickChatStore'
import { useUIStore } from '../stores/uiStore'

export function QuickChatPage() {
  const { tiers, registry, envVars } = useTierConfig()
  const {
    processes, fetchProcesses, setFleetInstructions, setDescription, setNameOverride, removeProcess,
  } = useProcessStore()
  const openChat = useChatStore((s) => s.openChat)
  const switchChat = useChatStore((s) => s.switchChat)
  const disconnectChat = useChatStore((s) => s.disconnectChat)
  const chatSessions = useChatStore((s) => s.sessions)
  const setAgentHidden = useDesktopPrefsStore((s) => s.setAgentHidden)
  const { sessions: quickSessions, add: quickAdd, remove: quickRemove, setPromoted } = useQuickChatStore()
  const setView = useUIStore((s) => s.setView)

  const [picking, setPicking] = useState(false)
  const [search, setSearch] = useState('')
  const [spawningId, setSpawningId] = useState<string | null>(null)
  const [error, setError] = useState('')

  // Keep the process list fresh on entry.
  useEffect(() => { fetchProcesses() }, [fetchProcesses])

  const procFor = (slug: string) => processes.find((p) => p.slug === slug)

  // Slugs whose agent web server has answered a probe — safe to open a chat.
  const [readySlugs, setReadySlugs] = useState<string[]>([])
  // Slug to auto-open once ready (the agent we just spawned).
  const [autoOpenSlug, setAutoOpenSlug] = useState<string | null>(null)

  const isReady = (slug: string) => readySlugs.includes(slug)
  const allReady = quickSessions.every((qs) => readySlugs.includes(qs.slug))

  // Keep polling the process list while any agent is still coming up.
  useEffect(() => {
    if (quickSessions.length === 0 || allReady) return
    const t = setInterval(() => { void fetchProcesses() }, 1500)
    return () => clearInterval(t)
  }, [quickSessions.length, allReady, fetchProcesses])

  // Probe each agent's web server; only mark it ready — and auto-open the chat —
  // once it actually answers, so we never connect into a not-yet-listening port
  // (which is what surfaces the "Connection failed" bubble).
  useEffect(() => {
    let cancelled = false
    const pending = quickSessions.filter((qs) => {
      const p = processes.find((x) => x.slug === qs.slug)
      return p && p.web_port && !readySlugs.includes(qs.slug)
    })
    if (pending.length === 0) return
    void (async () => {
      for (const qs of pending) {
        const p = processes.find((x) => x.slug === qs.slug)
        if (!p || !p.web_port) continue
        let ok = false
        try {
          const res = await fetch(`/fd/probe?host=localhost&port=${p.web_port}`)
          ok = res.ok && !!(await res.json()).ok
        } catch { ok = false }
        if (cancelled) return
        if (ok) {
          setReadySlugs((cur) => (cur.includes(qs.slug) ? cur : [...cur, qs.slug]))
          if (autoOpenSlug === qs.slug) {
            openChat(`proc-${qs.slug}`, qs.role, 'localhost', p.web_port, p.web_auth)
            setAutoOpenSlug(null)
          }
        }
      }
    })()
    return () => { cancelled = true }
  }, [quickSessions, processes, readySlugs, autoOpenSlug, openChat])

  const spawn = async (a: Archetype) => {
    const tc = tiers[a.tier]
    if (!tc || !tc.model.trim()) {
      setError(`The "${a.tier}" tier has no model — set it under Library → Model Tiers first.`)
      return
    }
    setError(''); setSpawningId(a.id)
    const uniq = Math.random().toString(36).slice(2, 6)
    const payload: SpawnConfig = {
      name: `${a.id}-${uniq}`,
      description: a.description,
      hostname: 'captain-claw',
      image: 'kstevica/captain-claw:latest',
      provider: tc.provider,
      model: tc.model,
      tier: '',
      temperature: 0.7,
      max_tokens: tc.output_ctx > 0 ? tc.output_ctx : 32768,
      max_context: tc.input_ctx > 0 ? tc.input_ctx : 0,
      provider_api_key: tc.api_key,
      base_url: tc.base_url,
      botport_enabled: false,
      botport_url: '',
      botport_instance_name: '',
      botport_key: '',
      botport_secret: '',
      botport_max_concurrent: 5,
      tools: a.tools,
      cognitive_mode: a.cognitive_mode || 'neutra',
      web_enabled: true,
      web_port: 0,
      web_auth_token: '',
      telegram_enabled: false,
      telegram_bot_token: '',
      discord_enabled: false,
      discord_bot_token: '',
      slack_enabled: false,
      slack_bot_token: '',
      network_mode: 'host',
      restart_policy: 'unless-stopped',
      extra_volumes: [],
      env_vars: envVars.filter((ev) => ev.key.trim() && ev.value.trim()),
    }
    try {
      const result = await spawnProcess(payload)
      if (!result.ok) throw new Error(result.message)
      const slug = result.slug
      // Persona metadata, same as a Library spawn.
      setFleetInstructions(slug, a.fleet_instructions)
      setDescription(slug, a.description)
      setNameOverride(slug, a.lead ? `${a.role} [Lead]` : a.role)
      // Spawn hidden — the agent lives on the desktop but off-canvas until promoted.
      setAgentHidden(`proc-${slug}`, true)
      quickAdd({ slug, role: a.role, name: a.role, tier: a.tier, promoted: false, createdAt: Date.now() })
      setPicking(false)
      // Don't open the chat yet — wait for the readiness probe to auto-open it.
      setAutoOpenSlug(slug)
      void fetchProcesses()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSpawningId(null)
    }
  }

  const openSession = (qs: QuickChatSession) => {
    const id = `proc-${qs.slug}`
    if (chatSessions.get(id)) { switchChat(id); return }
    if (!isReady(qs.slug)) return
    const p = procFor(qs.slug)
    if (p && p.web_port) openChat(id, qs.role, 'localhost', p.web_port, p.web_auth)
  }

  const promote = (qs: QuickChatSession) => { setAgentHidden(`proc-${qs.slug}`, false); setPromoted(qs.slug, true) }
  const demote = (qs: QuickChatSession) => { setAgentHidden(`proc-${qs.slug}`, true); setPromoted(qs.slug, false) }

  const removeSession = async (qs: QuickChatSession) => {
    if (!confirm(`Stop and remove "${qs.role}"?`)) return
    const id = `proc-${qs.slug}`
    disconnectChat(id)
    setAgentHidden(id, false)
    try { await removeProcess(qs.slug) } catch { /* already gone */ }
    quickRemove(qs.slug)
  }

  const statusOf = (qs: QuickChatSession): { label: string; color: string } => {
    const p = procFor(qs.slug)
    if (!p) return { label: 'starting…', color: 'bg-amber-400' }
    if (p.status !== 'running') return { label: 'stopped', color: 'bg-zinc-600' }
    if (!isReady(qs.slug)) return { label: 'warming up…', color: 'bg-amber-400' }
    return { label: 'ready', color: 'bg-emerald-400' }
  }

  const families = useMemo(() => (registry ? [...new Set(registry.archetypes.map((a) => a.family))] : []), [registry])
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    const list = registry?.archetypes || []
    if (!q) return list
    return list.filter((a) =>
      a.role.toLowerCase().includes(q) ||
      a.family.toLowerCase().includes(q) ||
      (a.description || '').toLowerCase().includes(q) ||
      (a.keywords || []).some((k) => k.toLowerCase().includes(q)),
    )
  }, [registry, search])

  const showPicker = picking || quickSessions.length === 0

  return (
    <div className="h-full overflow-auto p-4 md:p-6">
      <div className="mb-6 flex items-start justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-lg font-semibold">
            <MessagesSquare className="h-5 w-5 text-violet-400" /> Quick chat
          </h1>
          <p className="text-xs text-zinc-500 sm:text-sm">
            Pick an archetype, chat with it instantly. It stays off the Agent Desktop until you promote it.
          </p>
        </div>
        {quickSessions.length > 0 && (
          <button
            onClick={() => setPicking((v) => !v)}
            className="flex shrink-0 items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-2 text-sm font-medium text-white hover:bg-violet-500"
          >
            {picking ? <X className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
            {picking ? 'Close' : 'New quick chat'}
          </button>
        )}
      </div>

      {error && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/[0.06] px-3 py-2 text-xs text-red-400">
          <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
        </div>
      )}

      {/* ── Archetype picker ── */}
      {showPicker && (
        <div className="mb-6 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <div className="mb-3 flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-600" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search archetypes…"
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 py-2 pl-9 pr-3 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              />
            </div>
          </div>

          {!registry ? (
            <p className="text-[11px] text-zinc-600">Loading archetypes…</p>
          ) : filtered.length === 0 ? (
            <p className="py-6 text-center text-xs text-zinc-600">No archetypes match “{search}”.</p>
          ) : (
            <div className="space-y-4">
              {families.map((family) => {
                const items = filtered.filter((a) => a.family === family)
                if (items.length === 0) return null
                return (
                  <div key={family}>
                    <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">{family}</p>
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
                      {items.map((a) => {
                        const configured = !!tiers[a.tier]?.model?.trim()
                        const busy = spawningId === a.id
                        return (
                          <button
                            key={a.id}
                            onClick={() => spawn(a)}
                            disabled={busy || !configured}
                            title={configured ? `Start a quick chat with ${a.role}` : `Configure the "${a.tier}" tier first`}
                            className="group rounded-lg border border-zinc-800 bg-zinc-950/50 p-3 text-left transition-colors hover:border-violet-500/40 hover:bg-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            <div className="mb-1 flex items-center justify-between gap-2">
                              <span className="flex items-center gap-1 truncate text-sm font-medium text-zinc-200">
                                {a.lead && <Crown className="h-3 w-3 shrink-0 text-amber-400" />}{a.role}
                              </span>
                              {busy
                                ? <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-violet-400" />
                                : <Rocket className="h-3.5 w-3.5 shrink-0 text-zinc-600 group-hover:text-violet-400" />}
                            </div>
                            <p className="mb-2 text-[11px] leading-snug text-zinc-500 line-clamp-2">{a.description}</p>
                            <div className="flex flex-wrap items-center gap-1.5">
                              <span className="inline-flex items-center gap-1 rounded border border-cyan-500/25 bg-cyan-600/15 px-1.5 py-0.5 text-[10px] font-medium text-cyan-400">
                                <Gauge className="h-2.5 w-2.5" />{a.tier}
                              </span>
                              <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">{a.cognitive_mode}</span>
                              {!configured && <span className="text-[10px] text-amber-500/80">tier unset</span>}
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* ── Active quick chats ── */}
      {quickSessions.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
            Active ({quickSessions.length})
          </p>
          <div className="space-y-2">
            {quickSessions.map((qs) => {
              const st = statusOf(qs)
              const hasChat = !!chatSessions.get(`proc-${qs.slug}`)
              return (
                <div
                  key={qs.slug}
                  className="flex flex-wrap items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900/50 p-3"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium text-zinc-200">{qs.role}</span>
                      <span className="inline-flex items-center gap-1 rounded border border-cyan-500/25 bg-cyan-600/15 px-1.5 py-0.5 text-[10px] font-medium text-cyan-400">
                        <Gauge className="h-2.5 w-2.5" />{qs.tier}
                      </span>
                      {qs.promoted && (
                        <span className="inline-flex items-center gap-1 rounded border border-emerald-500/25 bg-emerald-600/15 px-1.5 py-0.5 text-[10px] font-medium text-emerald-400">
                          <Monitor className="h-2.5 w-2.5" /> on desktop
                        </span>
                      )}
                    </div>
                    <div className="mt-1 flex items-center gap-1.5 text-[11px] text-zinc-500">
                      <span className={`h-1.5 w-1.5 rounded-full ${st.color}`} />
                      {st.label}
                    </div>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => openSession(qs)}
                      disabled={!hasChat && !isReady(qs.slug)}
                      className="flex items-center gap-1.5 rounded-lg bg-zinc-100 px-3 py-1.5 text-sm font-medium text-zinc-900 hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {hasChat || isReady(qs.slug)
                        ? <><MessageSquare className="h-3.5 w-3.5" /> {hasChat ? 'Open chat' : 'Chat'}</>
                        : <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Starting…</>}
                    </button>
                    {qs.promoted ? (
                      <>
                        <button
                          onClick={() => setView('desktop')}
                          title="View on the Agent Desktop"
                          className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1.5 text-xs text-zinc-300 hover:border-zinc-600 hover:text-zinc-100"
                        >
                          View
                        </button>
                        <button
                          onClick={() => demote(qs)}
                          title="Hide again from the Agent Desktop"
                          className="rounded-lg border border-zinc-700 px-2.5 py-1.5 text-xs text-zinc-400 hover:border-zinc-600 hover:text-zinc-200"
                        >
                          Hide
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={() => promote(qs)}
                        title="Make this agent visible on the Agent Desktop"
                        className="flex items-center gap-1.5 rounded-lg border border-violet-500/40 bg-violet-500/10 px-3 py-1.5 text-sm font-medium text-violet-700 hover:bg-violet-500/20 dark:text-violet-200"
                      >
                        <MonitorUp className="h-3.5 w-3.5" /> Promote to desktop
                      </button>
                    )}
                    <button
                      onClick={() => removeSession(qs)}
                      title="Stop and remove"
                      className="rounded-lg p-1.5 text-zinc-600 hover:bg-zinc-800 hover:text-red-400"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
          <p className="mt-3 flex items-center gap-1.5 text-[11px] text-zinc-600">
            <Check className="h-3 w-3" /> The conversation opens in the chat panel — everything you can do with a normal agent works here.
          </p>
        </div>
      )}
    </div>
  )
}
