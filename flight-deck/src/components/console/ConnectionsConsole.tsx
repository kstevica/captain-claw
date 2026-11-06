import { useEffect, useState, type ComponentType, type ReactNode } from 'react'
import {
  Plug, Mail, LayoutGrid, Send, Scale, Mic, Sparkles, MessageCircle, Database, ChevronRight, ChevronLeft, Cable, Smartphone, Bell, Globe, AtSign, Brain } from 'lucide-react'
import GoogleConnection from '../connections/GoogleConnection'
import MicrosoftConnection from '../connections/MicrosoftConnection'
import ImapConnection from '../connections/ImapConnection'
import { useImapMailStore } from '../../stores/imapMailStore'
import TelegramConnection from '../connections/TelegramConnection'
import WhatsAppWebConnection from '../connections/WhatsAppWebConnection'
import CodexConnection from '../connections/CodexConnection'
import AntigravityConnection from '../connections/AntigravityConnection'
import MCPConnection from '../connections/MCPConnection'
import MCPAgentAccess from '../connections/MCPAgentAccess'
import AgentLinksConnection, { type LinksStatus } from '../connections/AgentLinksConnection'
import BrowserConnection, { type BrowsersStatus } from '../connections/BrowserConnection'
import JevConnection from '../connections/JevConnection'
import SonioxConnection from '../connections/SonioxConnection'
import ExternalMemoryConnection from '../connections/ExternalMemoryConnection'
import ExternalMemoryDeckConnection from '../connections/ExternalMemoryDeckConnection'
import WhatsAppNudgeConnection from '../connections/WhatsAppNudgeConnection'
import NotificationsConnection from '../connections/NotificationsConnection'
import TypesenseConnection from '../connections/TypesenseConnection'
import { SharedTelegramBotAdmin } from '../admin/SharedTelegramBotAdmin'
import { JevTeamKey, SonioxTeamKey, SharedMCPServers } from '../admin/SharedCredentialsAdmin'
import {
  getOAuthClientConfig, getAdminTelegram, getSharedJevKey, getSharedSonioxKey,
} from '../../services/sharedIntegrations'
import { getWhatsAppWebAdmin } from '../../services/whatsappWeb'
import { useMCPStore } from '../../stores/mcpStore'
import { useCodexAuthStore } from '../../stores/codexAuthStore'
import { useConsoleStore } from '../../stores/consoleStore'
import { selectConnectionsAvailable, useAuthStore } from '../../stores/authStore'
import { Page, PageHeader, Dot } from './parts'

// Connections: the services agents reach, with the deck-wide part (the OAuth
// client, the team bot, the team keys, shared MCP servers) next to the admin's
// own connection. The same cards as Connections and Admin → Integrations; all
// stay mounted so sign-in popups and half-typed forms survive switching.

type ServiceId = 'google' | 'microsoft' | 'email' | 'telegram' | 'whatsapp_web' | 'mcp' | 'systems' | 'browser' | 'jev' | 'soniox' | 'memory' | 'codex' | 'antigravity' | 'notify' | 'whatsapp' | 'typesense'

interface Service {
  id: ServiceId
  label: string
  hint: string
  icon: ComponentType<{ className?: string }>
  // Set up once for everyone (vs. this deck's own engines).
  everyone: boolean
}

const SERVICES: Service[] = [
  { id: 'google', label: 'Google', hint: 'Gmail, Calendar and Drive — each person signs in with their own account', icon: Mail, everyone: true },
  { id: 'microsoft', label: 'Microsoft 365', hint: 'Outlook, Calendar and OneDrive — each person signs in', icon: LayoutGrid, everyone: true },
  { id: 'email', label: 'Email (IMAP)', hint: 'Any IMAP/SMTP mailbox with a username and password — each person connects their own, as many as they need', icon: AtSign, everyone: true },
  { id: 'telegram', label: 'Telegram', hint: 'One team bot people talk to their agents through', icon: Send, everyone: true },
  { id: 'whatsapp_web', label: 'WhatsApp Web', hint: 'Each person links their own WhatsApp as a device (unofficial client)', icon: Smartphone, everyone: true },
  { id: 'mcp', label: 'MCP servers', hint: 'Extra tools for agents from MCP servers', icon: Plug, everyone: true },
  // The admin's own links; everyone links their own systems from Connections.
  { id: 'systems', label: 'Systems', hint: 'Other systems that send your agents messages and files', icon: Cable, everyone: true },
  // Captain for Chrome: everyone connects their own browser from Connections.
  { id: 'browser', label: 'Browser', hint: 'Captain for Chrome — send the page you are on to an agent', icon: Globe, everyone: true },
  { id: 'jev', label: 'Jev', hint: 'Legal research (TypeSafe) — a team key or each person\'s own', icon: Scale, everyone: true },
  { id: 'soniox', label: 'Soniox', hint: 'Speech-to-text and voices — a team key or each person\'s own', icon: Mic, everyone: true },
  { id: 'memory', label: 'External memory', hint: 'Membase, Hindsight — one server for everyone (a bank per person) or each person their own; all of their agents share it', icon: Brain, everyone: true },
  { id: 'codex', label: 'ChatGPT', hint: 'A ChatGPT subscription as a model provider', icon: Sparkles, everyone: false },
  { id: 'antigravity', label: 'Antigravity', hint: 'A Google subscription as a model provider', icon: Sparkles, everyone: false },
  { id: 'notify', label: 'Notifications', hint: 'Where your nudges and waiting proposals reach you — WhatsApp, Telegram, devices', icon: Bell, everyone: false },
  { id: 'whatsapp', label: 'WhatsApp', hint: 'Nudges from autonomous work over WhatsApp', icon: MessageCircle, everyone: false },
  { id: 'typesense', label: 'Deep memory', hint: 'The search server behind agents\' long-term memory', icon: Database, everyone: false },
]

function serviceFor(id: string | undefined): ServiceId | undefined {
  return SERVICES.find((s) => s.id === id)?.id
}

type Status = { tone: 'on' | 'pending' | 'off'; text: string }

// What each row says: the deck-wide part where there is one.
function useStatuses(accounts: boolean, oauth: boolean): Partial<Record<ServiceId, Status>> {
  const [out, setOut] = useState<Partial<Record<ServiceId, Status>>>({})
  const mcpShared = useMCPStore((s) => s.servers.filter((x) => x.deck_wide).length)
  const codex = useCodexAuthStore((s) => s.status)
  const inboxes = useImapMailStore((s) => s.accounts)

  useEffect(() => {
    let cancelled = false
    const put = (id: ServiceId, st: Status) => { if (!cancelled) setOut((o) => ({ ...o, [id]: st })) }
    for (const p of ['google', 'microsoft'] as const) {
      // Their sign-in needs accounts — or the desktop app's deck (selectConnectionsAvailable).
      if (!oauth) { put(p, { tone: 'off', text: 'Needs sign-in' }); continue }
      getOAuthClientConfig(p)
        // Microsoft's secret is optional: none for a "Mobile and desktop" registration.
        .then((c) => put(p, c.client_id_set && (c.client_secret_set || p === 'microsoft')
          ? { tone: 'on', text: accounts ? 'Ready for everyone' : 'Ready' }
          : { tone: 'pending', text: 'Needs the OAuth client' }))
        .catch(() => {})
    }
    // The team bot and the team keys come with accounts.
    if (!accounts) return () => { cancelled = true }
    getAdminTelegram()
      .then((t) => put('telegram', !t.config.configured
        ? { tone: 'off', text: 'No team bot' }
        : !t.config.enabled
          ? { tone: 'pending', text: 'Bot paused' }
          : t.status.running ? { tone: 'on', text: `@${t.config.bot_username || 'bot'} running` } : { tone: 'pending', text: 'Bot not running' }))
      .catch(() => {})
    getWhatsAppWebAdmin()
      .then((w) => put('whatsapp_web', w.kill_switch
        ? { tone: 'off', text: 'Switched off (FD_WHATSAPP_WEB)' }
        : !w.installed
          ? { tone: 'off', text: 'Not installed' }
          : w.allowed ? { tone: 'on', text: w.service ? 'Allowed · service running' : 'Allowed' } : { tone: 'off', text: 'Off' }))
      .catch(() => {})
    getSharedJevKey()
      .then((k) => put('jev', k.key_set ? { tone: k.verified === false ? 'pending' : 'on', text: 'Team key shared' } : { tone: 'off', text: 'No team key' }))
      .catch(() => {})
    getSharedSonioxKey()
      .then((k) => put('soniox', k.key_set
        ? { tone: k.verified === false ? 'pending' : 'on', text: 'Team key shared' }
        : k.env_key_set ? { tone: 'on', text: 'Server key' } : { tone: 'off', text: 'No team key' }))
      .catch(() => {})
    return () => { cancelled = true }
  }, [accounts, oauth])

  const mcp: Status = mcpShared > 0 ? { tone: 'on', text: `${mcpShared} shared with everyone` } : { tone: 'off', text: 'None shared' }
  return {
    ...out,
    ...(accounts ? { mcp } : {}),
    ...(codex?.connected && !codex.stale ? { codex: { tone: 'on' as const, text: 'Connected' } } : {}),
    email: !oauth ? { tone: 'off', text: 'Needs sign-in' }
      : inboxes?.length ? { tone: 'on', text: `${inboxes.length} of yours connected` }
        : { tone: 'off', text: 'Available — none of yours yet' },
  }
}

export function ConnectionsConsole() {
  // Sent here for one service (the Overview, a card's link): open on it.
  const [active, setActive] = useState<ServiceId>(() => serviceFor(useConsoleStore.getState().intent.connection) ?? 'google')
  // Phone: the list, or one service.
  const [detail, setDetail] = useState(() => !!serviceFor(useConsoleStore.getState().intent.connection))
  useEffect(() => {
    useConsoleStore.getState().takeIntent()
    // …or sent here while already on this page.
    return useConsoleStore.subscribe((s) => {
      const id = serviceFor(s.intent.connection)
      if (!id) return
      setActive(id)
      setDetail(true)
      useConsoleStore.getState().takeIntent()
    })
  }, [])
  const accounts = useAuthStore((s) => s.authEnabled !== false)
  const oauth = useAuthStore(selectConnectionsAvailable)
  const [systems, setSystems] = useState<LinksStatus | null>(null)
  const [browsers, setBrowsers] = useState<BrowsersStatus | null>(null)
  const statuses: Partial<Record<ServiceId, Status>> = {
    ...useStatuses(accounts, oauth),
    ...(systems ? {
      systems: systems.connected > 0
        ? { tone: 'on' as const, text: `${systems.connected} connected` }
        : { tone: 'off' as const, text: systems.links ? `${systems.links} link${systems.links === 1 ? '' : 's'}, none connected` : 'No links' },
    } : {}),
    ...(browsers ? {
      browser: browsers.connected > 0
        ? { tone: 'on' as const, text: `${browsers.connected} connected` }
        : { tone: 'off' as const, text: 'None connected' },
    } : {}),
  }
  const current = SERVICES.find((s) => s.id === active)!

  const pick = (id: ServiceId) => { setActive(id); setDetail(true) }

  const panel = (id: ServiceId, team: ReactNode | null, own: ReactNode) => (
    <div hidden={active !== id} className="space-y-6">
      {team && (
        <section className="space-y-3">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">For everyone on this deck</h3>
          {team}
        </section>
      )}
      <section className="space-y-3">
        {team && <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Your own</h3>}
        {own}
      </section>
    </div>
  )

  const row = (s: Service) => {
    const st = statuses[s.id]
    const on = s.id === active
    return (
      <button
        key={s.id}
        onClick={() => pick(s.id)}
        className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors ${
          on ? 'bg-zinc-800/80 text-zinc-100 wide:bg-violet-500/10' : 'text-zinc-300 hover:bg-zinc-800/50'
        }`}
      >
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800 text-zinc-300">
          <s.icon className="h-4 w-4" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium">{s.label}</span>
          {st && <span className="mt-0.5 flex items-center gap-1.5 truncate text-[11px] text-zinc-500"><Dot tone={st.tone} />{st.text}</span>}
        </span>
        <ChevronRight className="h-4 w-4 shrink-0 text-zinc-600 wide:hidden" />
      </button>
    )
  }

  return (
    <Page>
      <PageHeader
        icon={Plug}
        title="Connections"
        description="The services agents reach. Most are set up once here for everyone — people then just press Connect with their own account."
      />

      <div className="flex flex-col gap-6 wide:flex-row wide:items-start">
        <nav className={`${detail ? 'hidden' : 'block'} space-y-4 wide:sticky wide:top-0 wide:block wide:w-64 wide:shrink-0`} aria-label="Services">
          <div className="space-y-0.5">
            <div className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">{accounts ? 'For everyone' : 'Services'}</div>
            {SERVICES.filter((s) => s.everyone).map(row)}
          </div>
          <div className="space-y-0.5">
            <div className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">This deck</div>
            {SERVICES.filter((s) => !s.everyone).map(row)}
          </div>
        </nav>

        <div className={`${detail ? 'block' : 'hidden'} min-w-0 flex-1 wide:block`}>
          <button onClick={() => setDetail(false)} className="mb-4 flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-200 wide:hidden">
            <ChevronLeft className="h-4 w-4" /> All connections
          </button>
          <div className="mb-5 flex items-start gap-3">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
              <current.icon className="h-5 w-5" />
            </span>
            <div>
              <h2 className="text-lg font-semibold text-zinc-100">{current.label}</h2>
              <p className="text-sm text-zinc-500">{current.hint}</p>
            </div>
          </div>

          {panel('google', null, <GoogleConnection />)}
          {panel('microsoft', null, <MicrosoftConnection />)}
          {panel('email', null, <ImapConnection />)}
          {panel('telegram', accounts ? <SharedTelegramBotAdmin /> : null, <TelegramConnection />)}
          {panel('whatsapp_web', null, <WhatsAppWebConnection />)}
          {panel('mcp', accounts ? <SharedMCPServers refreshTick={0} inline /> : null, <><MCPConnection /><MCPAgentAccess /></>)}
          {panel('systems', null, <AgentLinksConnection active={active === 'systems'} onStatus={setSystems} />)}
          {panel('browser', null, <BrowserConnection active={active === 'browser'} onStatus={setBrowsers} />)}
          {panel('jev', accounts ? <JevTeamKey refreshTick={0} /> : null, <JevConnection />)}
          {panel('soniox', accounts ? <SonioxTeamKey refreshTick={0} /> : null, <SonioxConnection />)}
          {panel('memory', accounts ? <ExternalMemoryDeckConnection /> : null, <ExternalMemoryConnection />)}
          {panel('codex', null, <CodexConnection />)}
          {panel('antigravity', null, <AntigravityConnection />)}
          {panel('notify', null, <NotificationsConnection />)}
          {panel('whatsapp', null, <WhatsAppNudgeConnection />)}
          {panel('typesense', null, <TypesenseConnection />)}
        </div>
      </div>
    </Page>
  )
}
