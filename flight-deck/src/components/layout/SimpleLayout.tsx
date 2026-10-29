import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  ChevronLeft,
  ChevronRight,
  LayoutDashboard,
  Search,
  Plus,
  Box,
  Cpu,
  Server,
  Power,
  Loader2,
  Sun,
  Moon,
  RefreshCw,
  LogOut,
  MessagesSquare,
  FolderOpen,
  Database,
  Folders,
  Settings,
  Plug,
  IdCard,
  Users,
  Globe,
  X,
  Info,
  Compass,
  AlarmClock,
  LibraryBig,
  ListChecks,
  SlidersHorizontal,
  Trash2,
  BookOpen,
  GraduationCap,
  ShieldCheck,
  Smartphone,
  Activity,
  KeyRound,
} from 'lucide-react'
import { useUIStore } from '../../stores/uiStore'
import { openConsole, leaveKioskView, selectConsoleAdmin } from '../../stores/consoleStore'
import { isAppShell, hasAndroidAppSettings, openAndroidAppSettings } from '../../services/desktopShell'
import { useAgentStore } from '../../stores/agentStore'
import { useAuthStore, logoutUser } from '../../stores/authStore'
import { useChatStore, parseLaneKey, clearAgentSlices } from '../../stores/chatStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { useContainerStore } from '../../stores/containerStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { useProcessStore } from '../../stores/processStore'
import { useThemeStore } from '../../stores/themeStore'
import { useOnboardingStore } from '../../stores/onboardingStore'
import { useSharedAgentStore } from '../../stores/sharedAgentStore'
import type { SharedAgent } from '../../services/sharedAgents'
import { useScheduledStore } from '../../stores/scheduledStore'
import { ScheduledTasks } from '../scheduler/ScheduledTasks'
import { useFileLibraryStore } from '../../stores/fileLibraryStore'
import { FileLibrary } from '../library/FileLibrary'
import { useDataLibraryStore } from '../../stores/dataLibraryStore'
import { DataLibrary } from '../library/DataLibrary'
import { useActivityStore } from '../../stores/activityStore'
import { ToolCallRegister } from '../audit/ToolCallRegister'
import { useTaskboardStore } from '../../stores/taskboardStore'
import { TaskBoard } from '../taskboard/TaskBoard'
import { SharedFolders } from '../vfs/SharedFolders'
import { DialogFullscreen } from '../library/dialogFullscreen'
import { isManagedAgent } from '../../utils/managedAgents'
import { SHARED_PREFIX, sharedCaps, sharedContainerId } from '../../utils/sharedAgent'
import { CHAT_ONLY_TEXT, workspaceVisible } from '../../utils/sharedWorkspace'
import { usePersistedSize } from '../../hooks/usePersistedSize'
import { ChatPanel } from '../agents/ChatPanel'
import { AgentFilesPanel } from '../agents/AgentFilesPanel'
import { AgentDatastorePanel } from '../agents/AgentDatastorePanel'
import { SharedFilesPanel } from '../agents/SharedFilesPanel'
import { SharedDatastorePanel } from '../agents/SharedDatastorePanel'
import { AgentConfigEditor } from '../agents/AgentConfigEditor'
import { AgentSetupDialog } from './AgentSetupDialog'
import { OnboardingDialog, type OnboardingTab } from './OnboardingDialog'
import { ArchetypeSpawnDialog, type SetupRequest } from './ArchetypeSpawnDialog'
import { KioskByokDialog } from './KioskByokDialog'
import { useByokStore } from '../../stores/byokStore'
import { useQuotaStore } from '../../stores/quotaStore'
import ConnectionsPage, { type TabId as ConnectionsTab } from '../../pages/ConnectionsPage'
import { ProfilePage } from '../../pages/ProfilePage'
import { NotificationBell } from '../common/NotificationCenter'
import { BrandMark } from '../common/BrandMark'
import { KioskInfoDialog, type InfoAgent } from './KioskInfoDialog'
import { KioskOnboardingDialog, type OnboardingAction } from './KioskOnboardingDialog'
import {
  AgentDialog, AgentMenu, KioskAgentInfoDialog, KioskAgentInstructionsDialog, KioskAgentSettingsDialog, KioskRemoveAgentDialog,
  type AgentMenuItem,
} from './KioskAgentDialogs'
import { useIsMobile } from '../../hooks/useMediaQuery'
import { NetworkBadge, networkShareAction } from '../agents/NetworkShare'
import { dockerShareKey, processShareKey, useNetworkShareStore } from '../../stores/networkShareStore'

// ── Simple layout ────────────────────────────────────────────────────
//
// A chat-first arrangement of the same Flight Deck: agents on the left, the
// conversation in the middle, the active agent's files and datastore on the
// right (the kiosk has the File and Data libraries, and each agent's "⋯",
// instead). No nav, no pages, no director — those all live in the full layout,
// one click away. Both side columns collapse to a thin rail and remember
// their state.

type AgentState = 'running' | 'starting' | 'stopped' | 'unknown'

interface SimpleAgent {
  /** Chat id: docker container id, `proc-<slug>`, the local agent id, or
   *  `shared:<agent_ref>` for an agent another user shared with you. */
  id: string
  kind: 'docker' | 'process' | 'local' | 'shared'
  name: string
  description: string
  state: AgentState
  /** Running AND listening on a web port — a chat can open right now. */
  reachable: boolean
  host: string
  port: number
  auth: string
  /** What the power button needs: the container id or the process slug. */
  startKey: string
  /** Flight Deck runs this one itself (a run's worker, a being's body): not the user's to switch. */
  managed: boolean
  /** Stopped for being idle, not by the user: still on — opening its chat wakes it. */
  asleep?: boolean
  /** Shared with you: its agent_ref, and whose it is. */
  sharedRef?: string
  ownerId?: string
  ownerName?: string
}

const STATE_RANK: Record<AgentState, number> = { running: 0, starting: 1, unknown: 2, stopped: 3 }

/** How the BotPort share store names this agent ("" for one it can't share). */
function shareKeyOf(a: SimpleAgent): string {
  if (a.managed) return ''
  return a.kind === 'process' ? processShareKey(a.startKey) : a.kind === 'docker' ? dockerShareKey(a.name) : ''
}

function dockerState(status: string): AgentState {
  if (/running/i.test(status)) return 'running'
  if (/created|restarting/i.test(status)) return 'starting'
  if (/exited|stopped|dead|paused/i.test(status)) return 'stopped'
  return 'unknown'
}

/** Under an agent's name in its menu: what it does while it runs, else its state. */
function menuSubtitle(a: SimpleAgent): string {
  if (a.kind === 'shared') return `Shared by ${a.ownerName || 'another user'}`
  return a.state === 'running' && a.reachable && a.description ? a.description : stateLabel(a)
}

function stateLabel(a: SimpleAgent): string {
  if (a.asleep) return 'Asleep — wakes when you open it'
  if (a.state === 'running') return a.reachable ? 'Running' : 'Running — no web port'
  if (a.state === 'starting') return 'Starting…'
  if (a.state === 'stopped') return a.kind === 'local' ? 'Offline' : 'Stopped'
  return a.kind === 'local' ? 'Checking…' : 'Unknown'
}

function initials(name: string): string {
  const words = name.trim().split(/[\s_-]+/).filter(Boolean)
  if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase()
  return name.trim().slice(0, 2).toUpperCase() || '?'
}

/** The File and Data libraries index the user's own process agents — from
 *  their folder on disk, so stopped ones too — not one Flight Deck runs itself. */
function inLibraries(a: SimpleAgent): boolean {
  return a.kind === 'process' && !a.managed
}

/** What an agent's files-and-data view holds: its own files and datastore;
 *  for one shared with the user, the member panels — only on a deck that
 *  serves them and only those the agent offers (a process agent); chat only
 *  for a shared agent with none (a Docker agent, an older Flight Deck, a row
 *  not listed yet). */
type AgentPanels =
  | { region: 'own' }
  | { region: 'member'; row: SharedAgent; files: boolean; datastore: boolean }
  | { region: 'chat' }

function agentPanels(agentId: string, memberWorkspace: boolean, sharedAgents: SharedAgent[]): AgentPanels {
  if (!agentId.startsWith(SHARED_PREFIX)) return { region: 'own' }
  const row = sharedAgents.find((a) => sharedContainerId(a.agent_ref) === agentId)
  const vis = workspaceVisible(memberWorkspace, sharedCaps(row))
  return row && (vis.files || vis.datastore) ? { region: 'member', row, ...vis } : { region: 'chat' }
}

function useAgentPanels(agentId: string): AgentPanels {
  const memberWorkspace = useSharedAgentStore((s) => s.memberWorkspace)
  const sharedAgents = useSharedAgentStore((s) => s.agents)
  return agentPanels(agentId, memberWorkspace, sharedAgents)
}

const iconBtn = 'rounded p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300'

// The Getting started guide opened by itself this page load (the desktop
// admin comes back to the kiosk without a reload: it doesn't open again).
let guideAutoShown = false

// `adminSwitch`: the desktop app's admin in the users' kiosk (locked) — the
// Admin | User switch takes them back to the Admin console.
export function SimpleLayout({ locked = false, adminSwitch = false }: { locked?: boolean; adminSwitch?: boolean }) {
  const {
    containers, fetchContainers, checkHealth, startContainer, stopContainer, restartContainer, removeContainer,
    descriptionOverrides: dockerDesc,
  } = useContainerStore()
  const {
    processes, fetchProcesses, startProcess, stopProcess, restartProcess, removeProcess, descriptionOverrides: procDesc,
  } = useProcessStore()
  const { agents: localAgents, probeAll, probeAgent } = useLocalAgentStore()
  // Do NOT subscribe to the whole `sessions` Map — its identity churns on every
  // streaming event, which would re-render this always-mounted layout (and the
  // ChatPanel it hosts) on every token. Subscribe instead to a PRIMITIVE
  // signature that changes only when a per-agent busy/unread flag flips, and to
  // whether the active session exists; read the Map non-reactively where needed.
  const activeChatId = useChatStore((s) => s.activeChatId)
  const hasActiveSession = useChatStore((s) => (s.activeChatId ? s.sessions.has(s.activeChatId) : false))
  const activeSessionName = useChatStore((s) => (s.activeChatId ? s.sessions.get(s.activeChatId)?.containerName ?? '' : ''))
  const sessionSig = useChatStore((s) => {
    let sig = ''
    for (const x of s.sessions.values()) sig += `${x.containerId} ${x.busy ? 1 : 0} ${x.unread ? 1 : 0}\n`
    return sig
  })
  const openChat = useChatStore((s) => s.openChat)
  const openSharedChat = useChatStore((s) => s.openSharedChat)
  // Narrow selectors: the list keeps its identity across polls when nothing changed.
  const sharingEnabled = useSharedAgentStore((s) => s.enabled)
  const sharedAgents = useSharedAgentStore((s) => s.agents)
  const memberWorkspace = useSharedAgentStore((s) => s.memberWorkspace)
  const fetchShared = useSharedAgentStore((s) => s.fetch)
  const leaveShared = useSharedAgentStore((s) => s.leave)
  const setLayoutMode = useUIStore((s) => s.setLayoutMode)
  const setView = useUIStore((s) => s.setView)
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)

  // Agents being powered on or off right now.
  const [switching, setSwitching] = useState<Set<string>>(() => new Set())
  const [optionsAgent, setOptionsAgent] = useState<SimpleAgent | null>(null)
  // Kiosk "New agent" picker.
  const [archetypeOpen, setArchetypeOpen] = useState(false)
  // A new agent whose "way of working" setup opens right after it is created.
  const [setupFor, setSetupFor] = useState<SetupRequest | null>(null)
  // An agent's knowledge, to browse (its ⋯ menu → Knowledge).
  // An agent's onboarding page; Way of working opened from it comes back to it when it closes.
  const [onboardingFor, setOnboardingFor] = useState<{ slug: string; name: string; tab: OnboardingTab } | null>(null)
  const [setupBack, setSetupBack] = useState<{ slug: string; name: string; tab: OnboardingTab } | null>(null)
  // Agents just spawned or powered on: Flight Deck lists one as running the
  // moment it launches, seconds before its web server binds — a chat opened
  // then fails. They stay "starting" until they answer.
  const [booting, setBooting] = useState<Set<string>>(() => new Set())
  // The one booting agent whose chat opens by itself once it answers: a new
  // agent always (`force`), a powered-on one only if nothing else is open then.
  const autoOpen = useRef<{ id: string; force: boolean } | null>(null)
  const bootTries = useRef(new Map<string, number>())
  // Kiosk Connections dialog — the full layout's Connections page, which the
  // lock otherwise puts out of reach.
  const [connectionsOpen, setConnectionsOpen] = useState(false)
  // The tab it opens on: the guide's "Connect Google" and the like; otherwise
  // the one this viewer last had open.
  const [connectionsTab, setConnectionsTab] = useState<ConnectionsTab | undefined>(undefined)
  const openConnections = (tab?: ConnectionsTab) => { setConnectionsTab(tab); setConnectionsOpen(true) }
  // Kiosk Profile dialog — the owner profile every agent of theirs receives.
  const [profileOpen, setProfileOpen] = useState(false)
  // Kiosk Settings dialog — theme, connections, profile, refresh, sign-out.
  const [settingsOpen, setSettingsOpen] = useState(false)
  // "Your own AI keys": from Settings, or the plan bar when the day runs out.
  const byokOpen = useByokStore((s) => s.open)
  const openByok = useByokStore((s) => s.openDialog)
  const closeByok = useByokStore((s) => s.close)
  const setByokHost = useByokStore((s) => s.setHost)
  useEffect(() => {
    setByokHost(locked)
    return () => setByokHost(false)
  }, [locked, setByokHost])
  // Kiosk Info dialog — models, agents, archetypes, profile, plan, connections.
  const [infoOpen, setInfoOpen] = useState(false)
  // Kiosk agent dialogs from an agent's "⋯": its Info (intentions, topics),
  // Instructions, Settings and its removal. By id — they follow the live agent row.
  const [agentInfoId, setAgentInfoId] = useState<string | null>(null)
  const [agentInstructionsId, setAgentInstructionsId] = useState<string | null>(null)
  const [agentSettingsId, setAgentSettingsId] = useState<string | null>(null)
  const [agentRemoveId, setAgentRemoveId] = useState<string | null>(null)
  // …and its files and data, read live from the agent (see agentMenuItems).
  const [filesDataId, setFilesDataId] = useState<string | null>(null)
  // Kiosk Getting started guide. It opens by itself once per page load (so
  // once per person on a shared kiosk: sign-out reloads) until its "Do not
  // show again" is ticked — which is per user, so it waits for their settings.
  const [guideOpen, setGuideOpen] = useState(false)
  const [guideDecided, setGuideDecided] = useState(false)
  const settingsHydrated = useOnboardingStore((s) => s.hydrated)
  const guideOptedOut = useOnboardingStore((s) => s.completed.kiosk)
  if (locked && settingsHydrated && !guideDecided) {
    setGuideDecided(true)
    if (!guideOptedOut && !guideAutoShown) setGuideOpen(true)
  }
  useEffect(() => { if (guideOpen) guideAutoShown = true }, [guideOpen])
  // A phone gets its own layout: the agent list, or one chat, full screen.
  const { isMobile } = useIsMobile()
  // Scheduled tasks: the dialog (opened from the agent column, Settings, or a
  // task's result in the chat) and the task list its buttons count.
  const scheduledOpen = useScheduledStore((s) => s.open)
  const scheduledFocus = useScheduledStore((s) => s.focusTaskId)
  const scheduledNewFor = useScheduledStore((s) => s.newForAgent)
  const openScheduled = useScheduledStore((s) => s.openTasks)
  const closeScheduled = useScheduledStore((s) => s.close)
  const fetchScheduled = useScheduledStore((s) => s.fetchJobs)
  const scheduledCount = useScheduledStore((s) => s.jobs.filter((j) => j.enabled === 1).length)
  const scheduledRunning = useScheduledStore((s) => s.jobs.some((j) => j.running))
  // File library: every file the agents made, summarised (agent column,
  // Settings, the phone header) — or one agent's (its "⋯", its phone chat).
  const libraryOpen = useFileLibraryStore((s) => s.open)
  const libraryAgent = useFileLibraryStore((s) => s.agentFilter)
  const libraryFocus = useFileLibraryStore((s) => s.focus)
  const openLibrary = useFileLibraryStore((s) => s.openLibrary)
  const closeLibrary = useFileLibraryStore((s) => s.close)
  // Data library: every datastore table the agents keep, summarised.
  const dataOpen = useDataLibraryStore((s) => s.open)
  const dataAgent = useDataLibraryStore((s) => s.agentFilter)
  const openData = useDataLibraryStore((s) => s.openData)
  const closeData = useDataLibraryStore((s) => s.close)
  // Activity: the tool-call register for the user's own agents.
  const activityOpen = useActivityStore((s) => s.open)
  const activitySlug = useActivityStore((s) => s.agentSlug)
  const activityRef = useActivityStore((s) => s.agentRef)
  const openActivity = useActivityStore((s) => s.openActivity)
  const closeActivity = useActivityStore((s) => s.close)
  // Taskboard: every task the agents got or sent, and how many need the user.
  const taskboardOpen = useTaskboardStore((s) => s.open)
  const taskboardAgent = useTaskboardStore((s) => s.agentFilter)
  const taskboardFocus = useTaskboardStore((s) => s.focus)
  const openTaskboard = useTaskboardStore((s) => s.openTaskboard)
  const closeTaskboard = useTaskboardStore((s) => s.close)
  const fetchTaskboardSummary = useTaskboardStore((s) => s.fetchSummary)
  const taskboardCount = useTaskboardStore((s) => s.needsMe)
  // Shared folders: the files the agents keep together (vfs:<folder>/…).
  const [foldersOpen, setFoldersOpen] = useState(false)
  const openFolders = () => setFoldersOpen(true)
  // Linking a folder from the deck's own disk is the deck operator's: an
  // admin, or a deck nobody signs in to (the desktop app).
  const canLinkFolders = useAuthStore((s) => s.authEnabled === false || s.user?.role === 'admin')
  useEffect(() => {
    if (authEnabled === true && !isAuthenticated) return
    fetchScheduled()
    const t = setInterval(fetchScheduled, 30000)
    return () => clearInterval(t)
  }, [fetchScheduled, authEnabled, isAuthenticated])
  // The Taskboard's Needs me count, for the badge on its buttons.
  useEffect(() => {
    if (authEnabled === true && !isAuthenticated) return
    fetchTaskboardSummary()
    const t = setInterval(fetchTaskboardSummary, 30000)
    return () => clearInterval(t)
  }, [fetchTaskboardSummary, authEnabled, isAuthenticated])

  // Keep the agent list fresh — the Desktop page normally does this polling,
  // and it isn't mounted here. Same auth guard as there: with auth on and no
  // session yet, every request would only 401.
  useEffect(() => {
    if (authEnabled === true && !isAuthenticated) return
    checkHealth()
    fetchContainers()
    fetchProcesses()
    fetchShared()
    probeAll()
    const interval = setInterval(() => { fetchContainers(); fetchProcesses(); fetchShared() }, 10000)
    return () => clearInterval(interval)
  }, [checkHealth, fetchContainers, fetchProcesses, fetchShared, probeAll, authEnabled, isAuthenticated])

  const agents: SimpleAgent[] = useMemo(() => {
    const list: SimpleAgent[] = []
    for (const c of containers) {
      const state = dockerState(c.status)
      list.push({
        id: c.id, kind: 'docker',
        name: c.agent_name || c.name,
        description: dockerDesc[c.id] || c.description || '',
        state, reachable: state === 'running' && !!c.web_port,
        host: 'localhost', port: c.web_port ?? 0, auth: c.web_auth || '',
        startKey: c.id, managed: false,
      })
    }
    for (const p of processes) {
      const state: AgentState = p.status === 'running' ? 'running' : 'stopped'
      // Asleep: its chat opens (Flight Deck holds it and wakes the agent).
      list.push({
        id: `proc-${p.slug}`, kind: 'process',
        name: p.name || p.slug,
        description: procDesc[p.slug] || p.description || '',
        state, reachable: (state === 'running' || !!p.asleep) && !!p.web_port,
        host: 'localhost', port: p.web_port, auth: p.web_auth || '',
        startKey: p.slug, managed: isManagedAgent(p.slug, p.description || ''),
        asleep: !!p.asleep,
      })
    }
    for (const a of localAgents) {
      const state: AgentState = a.status === 'online' ? 'running' : a.status === 'offline' ? 'stopped' : 'unknown'
      list.push({
        id: a.id, kind: 'local',
        name: a.name, description: a.description || '',
        state, reachable: state === 'running',
        host: a.host, port: a.port, auth: a.authToken || '',
        startKey: a.id, managed: false,
      })
    }
    // Shared with you: chat only, through Flight Deck — no host, port or token.
    if (sharingEnabled) {
      for (const sa of sharedAgents) {
        const running = sa.status === 'running'
        list.push({
          id: sharedContainerId(sa.agent_ref), kind: 'shared',
          name: sa.name || sa.slug, description: sa.description || '',
          state: running ? 'running' : 'stopped', reachable: running,
          host: '', port: 0, auth: '', startKey: '', managed: false,
          sharedRef: sa.agent_ref, ownerId: sa.owner_id, ownerName: sa.owner_name,
        })
      }
    }
    // A sleeping agent keeps its place among the running ones.
    const rank = (a: SimpleAgent) => (a.asleep ? STATE_RANK.running : STATE_RANK[a.state])
    list.sort((x, y) => rank(x) - rank(y) || x.name.localeCompare(y.name))
    return list
  }, [containers, processes, localAgents, dockerDesc, procDesc, sharingEnabled, sharedAgents])

  // Per-agent chat state, folded across lanes: busy if any lane is, unread if
  // any lane finished something while the user looked elsewhere.
  const sessionInfo = useMemo(() => {
    const m = new Map<string, { busy: boolean; unread: boolean }>()
    for (const s of useChatStore.getState().sessions.values()) {
      const cur = m.get(s.containerId) ?? { busy: false, unread: false }
      cur.busy = cur.busy || s.busy
      cur.unread = cur.unread || !!s.unread
      m.set(s.containerId, cur)
    }
    return m
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionSig])

  const activeAgentId = hasActiveSession ? parseLaneKey(activeChatId!).containerId : null
  const activeAgent = activeAgentId ? agents.find((a) => a.id === activeAgentId) ?? null : null

  // What the Info dialog lists: each agent, and the model a process agent runs on.
  const infoAgents: InfoAgent[] = useMemo(() => agents.map((a) => {
    const p = a.kind === 'process' ? processes.find((x) => `proc-${x.slug}` === a.id) : undefined
    return {
      id: a.id, name: a.name, running: a.state === 'running', shared: a.kind === 'shared',
      ownerName: a.ownerName, model: p?.model || undefined, provider: p?.provider || undefined,
    }
  }), [agents, processes])

  // Chats opened so far: a phone shows the chat for each, even the one it
  // left for the list (its id doesn't change then).
  const [chatOpens, setChatOpens] = useState(0)
  const handleOpen = (a: SimpleAgent) => {
    // An existing session survives the agent stopping; openChat just
    // re-activates it (stale host/port are ignored for a known key).
    if (!a.reachable && !useChatStore.getState().sessions.has(a.id)) return
    setChatOpens((n) => n + 1)
    if (a.kind === 'shared') { openSharedChat(a.sharedRef!, a.name, a.ownerName || ''); return }
    openChat(a.id, a.name, a.host, a.port, a.auth)
  }

  // Give up an agent somebody shared with you. Its chats close here.
  const handleLeave = async (a: SimpleAgent) => {
    if (a.kind !== 'shared' || !a.sharedRef) return
    const owner = a.ownerName || 'its owner'
    if (!confirm(`Leave ${a.name}? You'll lose access until ${owner} shares it again.`)) return
    try {
      await leaveShared(a.sharedRef, a.ownerId || '')
    } catch (e) {
      useNotificationStore.getState().add('error', 'Could not leave agent',
        `${a.name}: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const startBoot = (id: string, force: boolean) => {
    // A new agent's promise to open is not given up for a power-on.
    if (force || !autoOpen.current?.force) {
      autoOpen.current = force || !useChatStore.getState().activeChatId ? { id, force } : autoOpen.current
    }
    bootTries.current.delete(id)
    setBooting((prev) => new Set(prev).add(id))
  }
  const endBoot = (id: string) => {
    if (autoOpen.current?.id === id) autoOpen.current = null
    bootTries.current.delete(id)
    setBooting((prev) => { if (!prev.has(id)) return prev; const n = new Set(prev); n.delete(id); return n })
  }

  // Its chats go first: a session left open reconnects to an agent that is no
  // longer there and fills with "Connection failed" — already while the agent
  // is shutting down. The conversation is the agent's; it comes back when the
  // chat is opened again. A queue on Auto is parked — on every lane, open
  // here or not: what was interrupted must not re-run by itself when the chat
  // next connects.
  const releaseChats = (id: string) => {
    const chat = useChatStore.getState()
    chat.parkQueues(id)
    for (const [key, session] of [...chat.sessions]) {
      if (session.containerId === id) chat.disconnectChat(key)
    }
  }

  // Power an agent on or off — the one switch a kiosk user has over an agent
  // of theirs (Flight Deck only lets them act on their own). A local agent
  // isn't ours to start or stop: "on" just checks whether it's back.
  const handlePower = async (a: SimpleAgent, on: boolean) => {
    if (!on && a.kind === 'local') return
    if (!on) {
      // This tab only knows what ITS open chats show — an agent can be at
      // work with none of them open — so powering off is always confirmed.
      const chat = useChatStore.getState()
      const busy = [...chat.sessions.values()].some((x) => x.containerId === a.id && x.busy)
      const parks = chat.parkQueues(a.id, true)
      if (!confirm(
        (busy ? `${a.name} is working right now. Power it off anyway?` : `Power off ${a.name}?`)
        + ' Anything it is doing will be interrupted.'
        + (parks ? ' Its queue will wait — switch Auto back on to continue it.' : ''),
      )) return
    }
    setSwitching((prev) => new Set(prev).add(a.id))
    // Whether its chat was the one in front — to bring it back if the stop fails.
    const wasActive = !on && useChatStore.getState().activeChatId != null
      && parseLaneKey(useChatStore.getState().activeChatId!).containerId === a.id
    try {
      if (!on) {
        endBoot(a.id)
        releaseChats(a.id)
      }
      if (a.kind === 'docker') await (on ? startContainer : stopContainer)(a.startKey)
      else if (a.kind === 'process') await (on ? startProcess : stopProcess)(a.startKey)
      else await probeAgent(a.startKey)
      if (on && a.kind !== 'local') startBoot(a.id, false)
      // The list before the switch is released — until then the row still
      // shows the state the agent just left.
      if (a.kind === 'docker') await fetchContainers()
      else if (a.kind === 'process') await fetchProcesses()
    } catch (e) {
      useNotificationStore.getState().add('error', on ? 'Could not start agent' : 'Could not stop agent',
        `${a.name}: ${e instanceof Error ? e.message : String(e)}`)
      // Still running: its chat comes back (a parked queue stays parked — Auto
      // is the user's to switch on again).
      if (wasActive) handleOpen(a)
    } finally {
      setSwitching((prev) => { const n = new Set(prev); n.delete(a.id); return n })
    }
  }

  // Restart an agent so what it reads at start-up (vision) applies. Its chats
  // close as for a power-off and the one in front opens again once it answers.
  const handleRestart = async (a: SimpleAgent): Promise<boolean> => {
    if (a.kind !== 'docker' && a.kind !== 'process') return false
    const chat = useChatStore.getState()
    const busy = [...chat.sessions.values()].some((x) => x.containerId === a.id && x.busy)
    if (!confirm(
      (busy ? `${a.name} is working right now. Restart it anyway?` : `Restart ${a.name}?`)
      + ' Anything it is doing will be interrupted.'
      + (chat.parkQueues(a.id, true) ? ' Its queue will wait — switch Auto back on to continue it.' : ''),
    )) return false
    setSwitching((prev) => new Set(prev).add(a.id))
    try {
      endBoot(a.id)
      releaseChats(a.id)
      await (a.kind === 'docker' ? restartContainer : restartProcess)(a.startKey)
      startBoot(a.id, false)
      return true
    } catch (e) {
      useNotificationStore.getState().add('error', 'Could not restart agent',
        `${a.name}: ${e instanceof Error ? e.message : String(e)}`)
      return false
    } finally {
      setSwitching((prev) => { const n = new Set(prev); n.delete(a.id); return n })
    }
  }

  // Remove an agent for good: Flight Deck stops it and deletes it with its
  // folder, its chat log and the scheduled tasks that run it; here its chats
  // close and their queues go, so nothing of it comes back under its name.
  const handleRemove = async (a: SimpleAgent) => {
    if (a.kind !== 'docker' && a.kind !== 'process') return
    setSwitching((prev) => new Set(prev).add(a.id))
    try {
      endBoot(a.id)
      releaseChats(a.id)
      const result = a.kind === 'docker' ? await removeContainer(a.startKey, true) : await removeProcess(a.startKey, true)
      clearAgentSlices(a.id)
      setAgentRemoveId(null)
      if (result.ok) useNotificationStore.getState().add('info', 'Agent removed', `${a.name} and its folder are gone.`)
      else useNotificationStore.getState().add('warning', 'Agent removed — not everything went', result.message)
      fetchScheduled()
    } catch (e) {
      // Still listed (and maybe switched off by now): the list says how it is.
      if (a.kind === 'docker') fetchContainers()
      else fetchProcesses()
      throw e
    } finally {
      setSwitching((prev) => { const n = new Set(prev); n.delete(a.id); return n })
    }
  }

  // Sharing one's own agents on this deck's BotPort network (when it has one):
  // the "⋯" menu offers it, the list shows a "network" badge.
  const netAgents = useNetworkShareStore((s) => s.agents)
  const netCanShare = useNetworkShareStore((s) => s.canShare)
  const netAdmin = useNetworkShareStore((s) => s.admin)
  const netDeck = useNetworkShareStore((s) => s.deckName)
  const netSetShared = useNetworkShareStore((s) => s.setShared)
  const netAskShare = useNetworkShareStore((s) => s.askShare)
  const shareKeys = useMemo(() => agents.map(shareKeyOf).filter(Boolean).join('|'), [agents])
  useEffect(() => {
    const st = useNetworkShareStore.getState()
    void st.fetch().then(() => {
      for (const k of shareKeys.split('|')) if (k) useNetworkShareStore.getState().ensure(k)
    })
  }, [shareKeys])

  // The kiosk's "⋯" on an agent. Its files and data live in the Library and
  // Data buttons (bottom left) for the agents those index — the library's
  // "Everything it keeps" opens the agent's own live view. An agent they
  // don't index keeps "Files & data" here, its only way to them: one shared
  // with the user (when its owner's deck gives members some) or a Docker one.
  // Instructions, Settings and Remove are for the user's own agents only —
  // not one shared with them, a local one, or one Flight Deck runs itself.
  const agentMenuItems = (a: SimpleAgent): AgentMenuItem[] => {
    if (a.kind === 'shared') {
      return agentPanels(a.id, memberWorkspace, sharedAgents).region === 'member'
        ? [{ icon: FolderOpen, label: 'Files & data', hint: 'What everyone who uses it can open', tone: 'amber', onSelect: () => setFilesDataId(a.id) }]
        : []
    }
    const up = a.reachable && !a.asleep && !booting.has(a.id) && !switching.has(a.id)
    const items: AgentMenuItem[] = [{
      icon: Info, label: 'Info', hint: up ? 'Its intentions and topics' : 'Power it on to see its intentions and topics',
      tone: 'sky', onSelect: () => setAgentInfoId(a.id), disabled: !up,
    }]
    // Read from the agent itself: a stopped one has none to show.
    if (!inLibraries(a)) {
      items.push({
        icon: FolderOpen, label: 'Files & data', tone: 'amber',
        hint: up ? 'Its files and datastore tables' : 'Its files and tables show while it runs',
        onSelect: () => setFilesDataId(a.id), disabled: !up,
      })
    }
    if (a.kind === 'docker' || a.kind === 'process') {
      items.push({
        icon: Activity, label: 'Activity', hint: 'Every tool it used, and with what', tone: 'emerald',
        onSelect: () => openActivity(a.kind === 'process' ? { slug: a.startKey } : { ref: a.startKey }),
      })
    }
    if ((a.kind === 'docker' || a.kind === 'process') && !a.managed) {
      items.push(
        { icon: BookOpen, label: 'Instructions, insights & skills', hint: 'What it keeps in mind, has learned, and knows how to do',
          tone: 'violet', onSelect: () => setAgentInstructionsId(a.id) },
      )
      // Your documents and way of working → its rules, skills and knowledge (process agents: FD writes its folder).
      if (a.kind === 'process') {
        items.push({ icon: GraduationCap, label: 'Onboarding', hint: 'Its documents, the lessons it was taught, and how well it knows them',
          tone: 'fuchsia', onSelect: () => setOnboardingFor({ slug: a.startKey, name: a.name, tab: 'overview' }) })
      }
      items.push(
        { icon: SlidersHorizontal, label: 'Settings', hint: 'Model, how it thinks, vision', tone: 'zinc', onSelect: () => setAgentSettingsId(a.id) },
      )
      const net = networkShareAction(netAgents[shareKeyOf(a)], {
        canShare: netCanShare, admin: netAdmin, deckName: netDeck, displayName: a.name, setShared: netSetShared,
        ask: netAskShare,
        onError: (m) => useNotificationStore.getState().add('error', 'BotPort network', m),
      })
      if (net.show) items.push({ icon: Globe, label: net.label, hint: net.hint, tone: 'cyan', onSelect: net.onClick, disabled: net.disabled })
      items.push(
        { icon: Trash2, label: 'Remove agent', hint: 'Delete it and its folder', danger: true,
          onSelect: () => setAgentRemoveId(a.id), disabled: switching.has(a.id) },
      )
    }
    return items
  }
  const menuFor = locked ? agentMenuItems : undefined
  const agentInfo = agentInfoId ? agents.find((a) => a.id === agentInfoId) : undefined
  const agentInstructions = agentInstructionsId ? agents.find((a) => a.id === agentInstructionsId) : undefined
  const agentSettings = agentSettingsId ? agents.find((a) => a.id === agentSettingsId) : undefined
  const agentRemove = agentRemoveId ? agents.find((a) => a.id === agentRemoveId) : undefined
  const filesData = filesDataId ? agents.find((a) => a.id === filesDataId) : undefined

  const refresh = () => { fetchContainers(); fetchProcesses(); fetchShared(); probeAll() }

  // "Continue in chat" (Scheduled tasks, the Taskboard): the task's agent
  // opens on the task's own session (the agent's /session switch, as a
  // Topics resume) once its chat is up.
  const openTaskChat = (slug: string, sessionId: string) => {
    const a = agents.find((x) => x.id === `proc-${slug}`)
    if (!a || (!a.reachable && !useChatStore.getState().sessions.has(a.id))) {
      useNotificationStore.getState().add('warning', 'Agent is off', 'Power the agent on to open this task in its chat.')
      return
    }
    closeScheduled()
    closeTaskboard()
    handleOpen(a)
    let tries = 0
    const attempt = () => {
      const s = useChatStore.getState().sessions.get(a.id)
      if (s?.connected && !s.busy) { useChatStore.getState().resumeTopic(a.id, sessionId); return }
      // A sleeping agent takes a while to wake.
      if (++tries < (a.asleep ? 240 : 40)) setTimeout(attempt, 250)
    }
    setTimeout(attempt, 250)
  }
  // A library file's "Agent chat": its agent's chat, in front.
  const openLibraryAgent = (slug: string) => {
    const a = agents.find((x) => x.id === `proc-${slug}`)
    if (!a || (!a.reachable && !useChatStore.getState().sessions.has(a.id))) {
      useNotificationStore.getState().add('warning', 'Agent is off', 'Power the agent on to chat with it.')
      return
    }
    closeLibrary()
    closeData()
    handleOpen(a)
  }
  // The library's "Everything it keeps": the agent's own files and data
  // (the Files & data dialog), in place of the library.
  const openLibraryLive = (slug: string) => {
    const a = agents.find((x) => x.id === `proc-${slug}`)
    const notify = useNotificationStore.getState().add
    if (!a || !a.reachable) { notify('warning', 'Agent is off', 'Power the agent on to see everything it keeps.'); return }
    if (a.asleep) { notify('info', 'Agent is asleep', `Open ${a.name}'s chat to wake it, then try again.`); return }
    if (booting.has(a.id)) { notify('info', 'Agent is starting', `${a.name} is still starting. Try again once it's up.`); return }
    if (switching.has(a.id)) { notify('info', 'One moment', `${a.name} is being switched on or off. Try again in a moment.`); return }
    closeLibrary()
    setFilesDataId(a.id)
  }
  // Whether that opens now — the library offers it only then.
  const libraryLiveReady = (slug: string) => {
    const a = agents.find((x) => x.id === `proc-${slug}`)
    return !!a && a.reachable && !a.asleep && !booting.has(a.id) && !switching.has(a.id)
  }
  // What a phone's Back closes before it leaves a chat (or Flight Deck).
  const phoneOverlayOpen = libraryOpen || dataOpen || foldersOpen || activityOpen || taskboardOpen || !!filesData
  const closePhoneOverlays = () => {
    closeLibrary(); closeData(); closeActivity(); closeTaskboard(); setFoldersOpen(false); setFilesDataId(null)
  }
  // Drive and OneDrive folders want the user's Google or Microsoft account:
  // Connections, on that tab.
  const foldersConnections = (service: 'google' | 'microsoft') => {
    setFoldersOpen(false)
    if (locked) { openConnections(service); return }
    useChatStore.setState({ chatFullscreen: false }); setLayoutMode('full'); setView('connections')
  }
  const scheduledConnections = () => {
    closeScheduled()
    if (locked) { openConnections(); return }
    useChatStore.setState({ chatFullscreen: false }); setLayoutMode('full'); setView('connections')
  }
  // A held reply the Taskboard can't show waits under Connections → Google.
  const taskboardConnections = () => {
    closeTaskboard()
    if (locked) { openConnections('google'); return }
    useChatStore.setState({ chatFullscreen: false }); setLayoutMode('full'); setView('connections')
  }

  // A guide step's button: the guide gives way to that part of the kiosk.
  const guideAction = (action: OnboardingAction) => {
    setGuideOpen(false)
    if (action === 'new-agent') setArchetypeOpen(true)
    else if (action === 'profile') setProfileOpen(true)
    else openConnections(action === 'connections-google' ? 'google' : action === 'connections-microsoft' ? 'microsoft' : 'telegram')
  }

  // The full Spawner leaves the chat surface, which the kiosk lock can't — so
  // there spawning means the archetype picker instead.
  const goSpawn = () => {
    if (locked) { setArchetypeOpen(true); return }
    useChatStore.setState({ chatFullscreen: false }); setLayoutMode('full'); setView('spawner')
  }

  // Watch the booting agents until each one answers (or ~90s pass), then let
  // it be opened — and open the one that was promised.
  const agentsRef = useRef(agents)
  agentsRef.current = agents
  const bootingRef = useRef(booting)
  bootingRef.current = booting
  const anyBooting = booting.size > 0
  useEffect(() => {
    if (!anyBooting) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const tick = async () => {
      for (const id of [...bootingRef.current]) {
        const a = agentsRef.current.find((x) => x.id === id)
        let up = false
        if (a?.reachable) {
          const { token } = useAuthStore.getState()
          up = await fetch(`/fd/probe?host=${encodeURIComponent(a.host)}&port=${a.port}`, {
            headers: token ? { Authorization: `Bearer ${token}` } : {},
            credentials: 'include',
          }).then((r) => (r.ok ? r.json() : null)).then((d) => !!d?.ok).catch(() => false)
          if (cancelled) return
        }
        const tries = (bootTries.current.get(id) ?? 0) + 1
        bootTries.current.set(id, tries)
        if (!up && tries < 60) continue
        const promised = autoOpen.current?.id === id ? autoOpen.current : null
        endBoot(id)
        // Whatever the user opened in the meantime stays in front of a
        // powered-on agent; a newly created one opens as announced.
        if (up && a && promised && (promised.force || !useChatStore.getState().activeChatId)) handleOpen(a)
      }
      if (!cancelled) timer = setTimeout(tick, 1500)
    }
    timer = setTimeout(tick, 1500)
    return () => { cancelled = true; clearTimeout(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anyBooting])

  const dialogs = (
    <>
      {archetypeOpen && (
        <ArchetypeSpawnDialog
          takenSlugs={new Set(processes.map((p) => p.slug))}
          onClose={() => setArchetypeOpen(false)}
          onSpawned={(slug, setup) => {
            setArchetypeOpen(false)
            startBoot(`proc-${slug}`, true)
            if (setup) setSetupFor(setup)
            else useNotificationStore.getState().add('info', 'Agent starting', `${slug} will open here as soon as it's up.`)
          }}
        />
      )}
      {setupFor && (
        <AgentSetupDialog agent={setupFor} onClose={() => {
          setSetupFor(null)
          if (setupBack) { setOnboardingFor(setupBack); setSetupBack(null) }
        }} />
      )}
      {onboardingFor && (
        <OnboardingDialog
          key={onboardingFor.slug}
          agent={onboardingFor}
          initialTab={onboardingFor.tab}
          phone={isMobile}
          onClose={() => setOnboardingFor(null)}
          onSetup={(readDocs) => {
            setSetupBack({ ...onboardingFor, tab: 'overview' })
            setSetupFor({ slug: onboardingFor.slug, name: onboardingFor.name, role: '', readDocs })
            setOnboardingFor(null)
          }}
        />
      )}

      {locked && settingsOpen && (
        <KioskSettingsDialog
          onAdmin={adminSwitch ? () => { setSettingsOpen(false); leaveKioskView() } : undefined}
          onClose={() => setSettingsOpen(false)}
          onRefresh={() => { refresh(); setSettingsOpen(false) }}
          onConnections={() => { setSettingsOpen(false); openConnections() }}
          onProfile={() => { setSettingsOpen(false); setProfileOpen(true) }}
          onGuide={() => { setSettingsOpen(false); setGuideOpen(true) }}
          onTaskboard={() => { setSettingsOpen(false); openTaskboard() }}
          onScheduled={() => { setSettingsOpen(false); openScheduled() }}
          onLibrary={() => { setSettingsOpen(false); openLibrary() }}
          onData={() => { setSettingsOpen(false); openData() }}
          onFolders={() => { setSettingsOpen(false); openFolders() }}
          onActivity={() => { setSettingsOpen(false); openActivity() }}
          onByok={() => { setSettingsOpen(false); openByok() }}
        />
      )}
      {locked && byokOpen && <KioskByokDialog onClose={closeByok} />}
      {locked && infoOpen && (
        <KioskInfoDialog
          agents={infoAgents}
          onClose={() => setInfoOpen(false)}
          onOpenProfile={() => { setInfoOpen(false); setProfileOpen(true) }}
          onOpenConnections={() => { setInfoOpen(false); openConnections() }}
          onOpenGuide={() => { setInfoOpen(false); setGuideOpen(true) }}
        />
      )}
      {locked && guideOpen && (
        <KioskOnboardingDialog phone={isMobile} onClose={() => setGuideOpen(false)} onAction={guideAction} />
      )}
      {scheduledOpen && (
        <KioskWideDialog label="Scheduled tasks" phone={isMobile} onClose={closeScheduled}>
          <ScheduledTasks
            phone={isMobile}
            initialTaskId={scheduledFocus}
            newForAgent={scheduledNewFor}
            defaultAgentSlug={activeAgent?.kind === 'process' ? activeAgent.startKey : undefined}
            onClose={closeScheduled}
            onOpenChat={openTaskChat}
            onOpenConnections={scheduledConnections}
          />
        </KioskWideDialog>
      )}
      {libraryOpen && (
        <KioskWideDialog label="File library" phone={isMobile} onClose={closeLibrary} widest>
          <FileLibrary
            phone={isMobile}
            initialAgent={libraryAgent}
            initialFile={libraryFocus}
            onClose={closeLibrary}
            onOpenAgent={openLibraryAgent}
            onOpenLive={locked ? openLibraryLive : undefined}
            liveReady={locked ? libraryLiveReady : undefined}
          />
        </KioskWideDialog>
      )}
      {taskboardOpen && (
        <KioskWideDialog label="Taskboard" phone={isMobile} onClose={closeTaskboard} widest>
          <TaskBoard
            phone={isMobile}
            initialAgent={taskboardAgent}
            initialTask={taskboardFocus}
            onClose={closeTaskboard}
            onOpenScheduled={(jobId) => { closeTaskboard(); openScheduled(jobId) }}
            onOpenChat={openTaskChat}
            onOpenConnections={taskboardConnections}
          />
        </KioskWideDialog>
      )}
      {dataOpen && (
        <KioskWideDialog label="Data library" phone={isMobile} onClose={closeData} widest>
          <DataLibrary phone={isMobile} initialAgent={dataAgent} onClose={closeData} onOpenAgent={openLibraryAgent} />
        </KioskWideDialog>
      )}
      {activityOpen && (
        <KioskWideDialog label="Activity" phone={isMobile} onClose={closeActivity} widest>
          <ToolCallRegister
            key={`${activitySlug}|${activityRef}`}
            scope="mine"
            phone={isMobile}
            initialAgent={activitySlug}
            initialRef={activityRef}
            onClose={closeActivity}
          />
        </KioskWideDialog>
      )}
      {foldersOpen && (
        <KioskWideDialog label="Shared folders" phone={isMobile} onClose={() => setFoldersOpen(false)} widest>
          <SharedFolders phone={isMobile} canLink={canLinkFolders} onClose={() => setFoldersOpen(false)} onConnect={foldersConnections} />
        </KioskWideDialog>
      )}
      {locked && agentInfo && (
        <KioskAgentInfoDialog agent={agentInfo} phone={isMobile} onClose={() => setAgentInfoId(null)} />
      )}
      {locked && agentInstructions && (agentInstructions.kind === 'docker' || agentInstructions.kind === 'process') && (
        <KioskAgentInstructionsDialog
          agent={{ ...agentInstructions, kind: agentInstructions.kind }}
          onClose={() => setAgentInstructionsId(null)}
        />
      )}
      {locked && agentSettings && (agentSettings.kind === 'docker' || agentSettings.kind === 'process') && (
        <KioskAgentSettingsDialog
          agent={{ ...agentSettings, kind: agentSettings.kind }}
          restarting={switching.has(agentSettings.id)}
          onRestart={() => handleRestart(agentSettings)}
          onClose={() => setAgentSettingsId(null)}
        />
      )}
      {locked && agentRemove && (
        <KioskRemoveAgentDialog
          agent={agentRemove}
          busy={!!sessionInfo.get(agentRemove.id)?.busy}
          onRemove={() => handleRemove(agentRemove)}
          onOpenLibrary={inLibraries(agentRemove)
            ? () => { setAgentRemoveId(null); openLibrary(agentRemove.startKey) }
            : undefined}
          onOpenData={inLibraries(agentRemove)
            ? () => { setAgentRemoveId(null); openData(agentRemove.startKey) }
            : undefined}
          onClose={() => setAgentRemoveId(null)}
        />
      )}
      {/* A phone has the full-screen sheet its chat header opens. */}
      {locked && filesData && (isMobile
        ? <PhoneFilesSheet agentId={filesData.id} agentName={filesData.name} onClose={() => setFilesDataId(null)} />
        : <KioskFilesDataDialog agentId={filesData.id} agentName={filesData.name} onClose={() => setFilesDataId(null)} />)}
      {locked && connectionsOpen && <KioskConnectionsDialog initialTab={connectionsTab} onClose={() => setConnectionsOpen(false)} />}
      {locked && profileOpen && <KioskProfileDialog onClose={() => setProfileOpen(false)} />}

      {!locked && optionsAgent && optionsAgent.kind !== 'shared' && createPortal(
        <AgentConfigEditor
          kind={optionsAgent.kind}
          identifier={optionsAgent.startKey}
          agentName={optionsAgent.name}
          onClose={() => setOptionsAgent(null)}
        />,
        document.body,
      )}
    </>
  )

  if (isMobile) {
    return (
      <>
        <PhoneLayout
          agents={agents}
          activeAgent={activeAgent}
          activeAgentId={activeAgentId}
          activeSessionName={activeSessionName}
          sessionInfo={sessionInfo}
          switching={switching}
          booting={booting}
          onOpen={handleOpen}
          onPower={handlePower}
          onLeave={handleLeave}
          onSpawn={goSpawn}
          onSettings={locked ? () => setSettingsOpen(true) : undefined}
          onInfo={locked ? () => setInfoOpen(true) : undefined}
          onTaskboard={() => openTaskboard()}
          taskboardCount={taskboardCount}
          onScheduled={() => openScheduled()}
          scheduledRunning={scheduledRunning}
          onLibrary={() => openLibrary()}
          onData={() => openData()}
          onFolders={openFolders}
          menuFor={menuFor}
          locked={locked}
          adminSwitch={adminSwitch}
          overlayOpen={phoneOverlayOpen}
          onCloseOverlays={closePhoneOverlays}
          chatOpens={chatOpens}
        />
        {dialogs}
      </>
    )
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <AgentsColumn
        agents={agents}
        activeAgentId={activeAgentId}
        sessionInfo={sessionInfo}
        switching={switching}
        booting={booting}
        onOpen={handleOpen}
        onPower={handlePower}
        onLeave={handleLeave}
        onRefresh={refresh}
        onSpawn={goSpawn}
        onOptions={setOptionsAgent}
        onSettings={locked ? () => setSettingsOpen(true) : undefined}
        onInfo={locked ? () => setInfoOpen(true) : undefined}
        onTaskboard={() => openTaskboard()}
        taskboardCount={taskboardCount}
        onScheduled={() => openScheduled()}
        scheduledCount={scheduledCount}
        scheduledRunning={scheduledRunning}
        onLibrary={() => openLibrary()}
        onData={() => openData()}
        onFolders={openFolders}
        menuFor={menuFor}
        locked={locked}
        adminSwitch={adminSwitch}
      />

      <main className="flex min-w-0 flex-1 flex-col overflow-hidden" style={{ minWidth: 320 }}>
        {hasActiveSession
          ? <ChatPanel variant="simple" />
          : <EmptyChat hasAgents={agents.length > 0} onSpawn={goSpawn} locked={locked} onGuide={locked ? () => setGuideOpen(true) : undefined} />}
      </main>

      {/* The kiosk's files and data are the left column's Library and Data —
          the agents they index, with the same actions — and an agent's "⋯"
          (the libraries filtered to it, and this column's panels in a
          dialog), so it has no right column. */}
      {!locked && (
        <ContextColumn
          agentId={activeAgentId}
          agentName={activeSessionName}
          onOptions={activeAgent && activeAgent.kind !== 'shared' ? () => setOptionsAgent(activeAgent) : undefined}
        />
      )}

      {dialogs}
    </div>
  )
}

// ── Kiosk: Settings ──────────────────────────────────────────────────
//
// The kiosk keeps its whole footer behind one Settings button: this dialog
// holds the theme switch, Connections, Profile, a refresh of the agent list,
// the Getting started guide, and sign-out (how a shared kiosk hands the
// screen to the next person).

function KioskSettingsDialog({ onAdmin, onClose, onRefresh, onConnections, onProfile, onGuide, onTaskboard, onScheduled, onLibrary, onData, onFolders, onActivity, onByok }: {
  /** The desktop app's admin in the users' kiosk: back to the Admin console. */
  onAdmin?: () => void
  onClose: () => void
  onRefresh: () => void
  onConnections: () => void
  onProfile: () => void
  onGuide: () => void
  onTaskboard: () => void
  onScheduled: () => void
  onLibrary: () => void
  onData: () => void
  onFolders: () => void
  onActivity: () => void
  onByok: () => void
}) {
  const { theme, toggle: toggleTheme } = useThemeStore()
  // Offered where the deck allows own keys (the plan status says so).
  const quota = useQuotaStore((s) => s.status)
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const authUser = useAuthStore((s) => s.user)
  const signingOut = useAuthStore((s) => s.signingOut)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const dark = theme === 'dark'
  const tiles: { icon: typeof Plug; title: string; detail: string; onClick: () => void }[] = [
    ...(onAdmin ? [{ icon: ShieldCheck, title: 'Switch to admin view', detail: 'Models, archetypes, connections and settings', onClick: onAdmin }] : []),
    { icon: dark ? Sun : Moon, title: dark ? 'Light theme' : 'Dark theme', detail: `Switch to the ${dark ? 'light' : 'dark'} look`, onClick: toggleTheme },
    { icon: Plug, title: 'Connections', detail: 'Google, Microsoft, MCP and Jev', onClick: onConnections },
    ...(quota?.byok_allowed || quota?.byok ? [{
      icon: KeyRound, title: 'Your own AI keys',
      detail: quota.byok ? 'On — your agents run on your keys, no daily limits' : 'Run your agents on your own API keys, with no daily limits',
      onClick: onByok,
    }] : []),
    { icon: ListChecks, title: 'Taskboard', detail: 'Every task your agents got or sent — and what needs you', onClick: onTaskboard },
    { icon: AlarmClock, title: 'Scheduled tasks', detail: 'Your agents on a timer — results by chat or email', onClick: onScheduled },
    { icon: LibraryBig, title: 'File library', detail: 'Every file your agents made, summarised and searchable', onClick: onLibrary },
    { icon: Database, title: 'Data library', detail: 'Every table your agents keep, summarised and easy to read', onClick: onData },
    { icon: Folders, title: 'Shared folders', detail: 'The folders your agents work in together — open, search, add files', onClick: onFolders },
    { icon: Activity, title: 'Activity', detail: 'Every tool your agents used — when, with what, how it went', onClick: onActivity },
    { icon: IdCard, title: 'Profile', detail: 'About you, your company, your preferences', onClick: onProfile },
    // The Android app: back to its start screen — the saved decks and accounts.
    ...(hasAndroidAppSettings() ? [{ icon: Smartphone, title: 'Switch deck', detail: 'Your Flight Decks and accounts on this phone', onClick: openAndroidAppSettings }] : []),
    { icon: RefreshCw, title: 'Refresh agents', detail: 'Check every agent again', onClick: onRefresh },
    { icon: Compass, title: 'Getting started', detail: 'A short tour: agents, connections, Telegram and more', onClick: onGuide },
  ]

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label="Settings"
        className="relative max-h-[100dvh] w-full max-w-md overflow-y-auto rounded-t-2xl border border-zinc-800 bg-zinc-950 pb-[env(safe-area-inset-bottom)] shadow-2xl wide:rounded-2xl wide:pb-0"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
          <h2 className="flex items-center gap-2 text-base font-semibold text-zinc-100">
            <Settings className="h-4 w-4 text-violet-400" />
            Settings
          </h2>
          <button onClick={onClose} title="Close" aria-label="Close" className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 wide:p-1">
            <X className="h-5 w-5 wide:h-4 wide:w-4" />
          </button>
        </div>

        {authEnabled && authUser && (
          <div className="flex items-center gap-3 px-5 pt-4">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-violet-500/15 text-sm font-semibold text-violet-400">
              {initials(authUser.display_name || authUser.email)}
            </span>
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-zinc-200">{authUser.display_name || authUser.email}</div>
              {authUser.display_name && <div className="truncate text-xs text-zinc-500">{authUser.email}</div>}
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2.5 p-5">
          {tiles.map(({ icon: Icon, title, detail, onClick }, i) => (
            <button
              key={title}
              onClick={onClick}
              // An odd one out at the end takes the whole row.
              className={`flex flex-col items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 text-left transition-colors hover:border-violet-500/50 hover:bg-zinc-800/60 ${
                i === tiles.length - 1 && tiles.length % 2 ? 'col-span-2' : ''
              }`}
            >
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-violet-500/10 text-violet-400">
                <Icon className="h-[18px] w-[18px]" />
              </span>
              <span>
                <span className="block text-sm font-medium text-zinc-200">{title}</span>
                <span className="mt-0.5 block text-[11px] leading-snug text-zinc-500">{detail}</span>
              </span>
            </button>
          ))}
        </div>

        {authEnabled && authUser && (
          <div className="border-t border-zinc-800 p-5 pt-4">
            <button
              onClick={logoutUser}
              disabled={signingOut}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-zinc-800 px-4 py-2.5 text-sm font-medium text-zinc-300 transition-colors hover:border-red-500/40 hover:bg-red-500/10 hover:text-red-600 disabled:opacity-60 dark:hover:text-red-400"
            >
              {signingOut ? <Loader2 className="h-4 w-4 animate-spin" /> : <LogOut className="h-4 w-4" />}
              {signingOut ? 'Signing out…' : 'Sign out'}
            </button>
          </div>
        )}
      </div>
    </div>,
    document.body,
  )
}

// ── Kiosk: the Connections page in a dialog ──────────────────────────
//
// The same page the full layout shows, minus the deck-setup cards (see
// ConnectionsPage's `kiosk`), so a kiosk user can link their own Google
// account and the MCP servers without leaving the chat surface.

function KioskConnectionsDialog({ initialTab, onClose }: { initialTab?: ConnectionsTab; onClose: () => void }) {
  return (
    <KioskDialog label="Connections" onClose={onClose}>
      <ConnectionsPage kiosk initialTab={initialTab} />
    </KioskDialog>
  )
}

// ── Kiosk: the Profile page in a dialog ──────────────────────────────
//
// About me / my company / my standing preferences — what every agent working
// for this user receives. Long-form text, so closing asks before it drops an
// unsaved edit.

function KioskProfileDialog({ onClose }: { onClose: () => void }) {
  const dirty = useRef(false)
  const close = useCallback(() => {
    if (dirty.current && !window.confirm('Discard your unsaved profile changes?')) return
    onClose()
  }, [onClose])
  return (
    <KioskDialog label="Profile" onClose={close}>
      <ProfilePage kiosk onDirtyChange={(d) => { dirty.current = d }} />
    </KioskDialog>
  )
}

function KioskDialog({ label, onClose, children }: { label: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label={label}
        className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 shadow-2xl wide:h-[85vh] wide:max-w-3xl wide:rounded-xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          title="Close"
          aria-label="Close"
          className="absolute right-2 top-2 z-10 rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 wide:right-3 wide:top-3 wide:p-1"
        >
          <X className="h-5 w-5 wide:h-4 wide:w-4" />
        </button>
        <div className="min-h-0 flex-1">
          {children}
        </div>
      </div>
    </div>,
    document.body,
  )
}

// ── Scheduled tasks / the libraries / shared folders in a dialog ─────
//
// Wider than the other kiosk dialogs: a list beside the selected item's
// detail (the library also has its filters). Escape is the component's
// (Scheduled tasks keeps an unsaved edit open; the library closes its viewer
// first). Off a phone it can fill the window — the libraries and shared
// folders put the switch in their header (FullscreenButton).

function KioskWideDialog({ label, phone, widest = false, onClose, children }: {
  label: string; phone: boolean; widest?: boolean; onClose: () => void; children: ReactNode
}) {
  const [fullscreen, setFullscreen] = useState(false)
  const host = useMemo(() => (phone ? null : { fullscreen, toggle: () => setFullscreen((v) => !v) }), [phone, fullscreen])
  const big = fullscreen && !phone
  return createPortal(
    <div className={`fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center ${big ? '' : 'wide:p-4'}`} onClick={phone ? undefined : onClose}>
      <div
        role="dialog"
        aria-label={label}
        className={`relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 pb-[env(safe-area-inset-bottom)] shadow-2xl wide:pb-0 ${big ? ''
          : `wide:h-[88vh] wide:rounded-xl wide:border wide:border-zinc-800 ${widest ? 'wide:max-w-6xl' : 'wide:max-w-5xl'}`}`}
        onClick={(e) => e.stopPropagation()}
      >
        <DialogFullscreen.Provider value={host}>{children}</DialogFullscreen.Provider>
      </div>
    </div>,
    document.body,
  )
}

const toolTile = 'relative flex min-w-0 flex-col items-center gap-1 rounded-lg px-1 py-2 text-[11px] font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100'
// Equal tiles, none narrower than its name: a narrow column gives the longer
// names ("Scheduled") their width before the rest share what's left.
const TILE_COLS: Record<number, string> = {
  2: 'grid-cols-[repeat(2,minmax(max-content,1fr))]',
  3: 'grid-cols-[repeat(3,minmax(max-content,1fr))]',
  4: 'grid-cols-[repeat(4,minmax(max-content,1fr))]',
  5: 'grid-cols-[repeat(5,minmax(max-content,1fr))]',
}

/** The agent column's row of tools — the Taskboard (how many tasks need
 *  you), Scheduled tasks (how many are active, a pulse while one runs), the
 *  File library, the Data library and the shared folders — as tiles, an
 *  icon over its name. */
function ToolTiles({ onTaskboard, taskboardCount, onScheduled, scheduledCount, scheduledRunning, onLibrary, onData, onFolders }: {
  onTaskboard?: () => void; taskboardCount: number
  onScheduled?: () => void; scheduledCount: number; scheduledRunning: boolean
  onLibrary?: () => void; onData?: () => void; onFolders?: () => void
}) {
  const n = [onTaskboard, onScheduled, onLibrary, onData, onFolders].filter(Boolean).length
  return (
    <div className={`grid gap-0.5 border-t border-zinc-800 p-2 pb-0 ${TILE_COLS[n] ?? 'grid-cols-1'}`}>
      {onTaskboard && (
        <button onClick={onTaskboard} className={toolTile} title="Taskboard — every task your agents got or sent, and what needs you">
          <span className="relative">
            <ListChecks className="h-[18px] w-[18px] text-teal-600 dark:text-teal-400" />
            {taskboardCount > 0 && (
              <span className="absolute -right-2.5 -top-1.5 min-w-[16px] rounded-full bg-teal-600 px-1 text-center text-[9px] font-semibold leading-4 text-white">
                {taskboardCount > 99 ? '99+' : taskboardCount}
              </span>
            )}
          </span>
          <span className="max-w-full truncate">Tasks</span>
        </button>
      )}
      {onScheduled && (
        <button onClick={onScheduled} className={toolTile} title="Scheduled tasks — your agents on a timer">
          <span className="relative">
            <AlarmClock className="h-[18px] w-[18px] text-amber-600 dark:text-amber-400" />
            {scheduledRunning && <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-pulse rounded-full bg-amber-500 ring-2 ring-zinc-900 dark:bg-amber-400" />}
            {scheduledCount > 0 && !scheduledRunning && (
              <span className="absolute -right-2.5 -top-1.5 min-w-[16px] rounded-full bg-zinc-700 px-1 text-center text-[9px] font-semibold leading-4 text-zinc-100">{scheduledCount}</span>
            )}
          </span>
          <span className="max-w-full truncate">Scheduled</span>
        </button>
      )}
      {onLibrary && (
        <button onClick={onLibrary} className={toolTile} title="File library — every file your agents made, summarised and searchable">
          <LibraryBig className="h-[18px] w-[18px] text-violet-600 dark:text-violet-400" />
          <span className="max-w-full truncate">Library</span>
        </button>
      )}
      {onData && (
        <button onClick={onData} className={toolTile} title="Data library — every table your agents keep, summarised and easy to read">
          <Database className="h-[18px] w-[18px] text-emerald-600 dark:text-emerald-400" />
          <span className="max-w-full truncate">Data</span>
        </button>
      )}
      {onFolders && (
        <button onClick={onFolders} className={toolTile} title="Shared folders — the folders your agents work in together">
          <Folders className="h-[18px] w-[18px] text-sky-600 dark:text-sky-400" />
          <span className="max-w-full truncate">Folders</span>
        </button>
      )}
    </div>
  )
}

// ── Left: agents ─────────────────────────────────────────────────────

function AgentsColumn({
  agents, activeAgentId, sessionInfo, switching, booting, onOpen, onPower, onLeave, onRefresh, onSpawn, onOptions, onSettings, onInfo,
  onTaskboard, taskboardCount = 0,
  onScheduled, scheduledCount = 0, scheduledRunning = false, onLibrary, onData, onFolders, menuFor, locked = false, adminSwitch = false,
}: {
  agents: SimpleAgent[]
  activeAgentId: string | null
  sessionInfo: Map<string, { busy: boolean; unread: boolean }>
  switching: Set<string>
  /** Started moments ago and not answering yet — not openable until they do. */
  booting: Set<string>
  onOpen: (a: SimpleAgent) => void
  /** Power an agent on (or, for a local one, check it again) or off. */
  onPower: (a: SimpleAgent, on: boolean) => void
  /** Leave an agent somebody shared with you. */
  onLeave: (a: SimpleAgent) => void
  onRefresh: () => void
  onSpawn: () => void
  onOptions: (a: SimpleAgent) => void
  /** Kiosk only: the footer is one Settings button opening this dialog. */
  onSettings?: () => void
  /** Kiosk only: the Info button beside Settings. */
  onInfo?: () => void
  /** Opens the Taskboard; its badge is the Needs me count. */
  onTaskboard?: () => void
  taskboardCount?: number
  /** Opens Scheduled tasks. */
  onScheduled?: () => void
  scheduledCount?: number
  scheduledRunning?: boolean
  /** Opens the File library. */
  onLibrary?: () => void
  /** Opens the Data library. */
  onData?: () => void
  /** Opens the shared folders. */
  onFolders?: () => void
  /** Kiosk only: what an agent's "⋯" offers (Info, Settings, Remove). */
  menuFor?: (a: SimpleAgent) => AgentMenuItem[]
  locked?: boolean
  /** The desktop app's admin in the kiosk: the way back to Admin. */
  adminSwitch?: boolean
}) {
  const open = useUIStore((s) => s.simpleLeftOpen)
  const setOpen = useUIStore((s) => s.setSimpleLeftOpen)
  const setLayoutMode = useUIStore((s) => s.setLayoutMode)
  const wsConnected = useAgentStore((s) => s.wsConnected)
  const { authEnabled, user: authUser, signingOut } = useAuthStore()
  const consoleAdmin = useAuthStore(selectConsoleAdmin) && !locked
  const appShell = isAppShell()
  const { theme, toggle: toggleTheme } = useThemeStore()
  const width = usePersistedSize('fd:simple-left-width', 260, 200, 440, 'x')
  const [search, setSearch] = useState('')
  // The search box only renders past a threshold; gate the filter on the SAME
  // condition so a stale query can't strand the list once the box is gone.
  const searchable = agents.length > 6

  const shown = useMemo(() => {
    const q = searchable ? search.trim().toLowerCase() : ''
    if (!q) return agents
    return agents.filter((a) => a.name.toLowerCase().includes(q) || a.description.toLowerCase().includes(q))
  }, [agents, search, searchable])

  // ── Collapsed rail ──
  if (!open) {
    return (
      <aside className="flex w-14 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/50">
        <div className="flex flex-col items-center gap-0.5 border-b border-zinc-800 py-2">
          <button onClick={() => setOpen(true)} className={iconBtn} title="Show agents">
            <ChevronRight className="h-4 w-4" />
          </button>
          {!locked && (
            <button onClick={() => setLayoutMode('full')} className={iconBtn} title="Full view — all pages and panels">
              <LayoutDashboard className="h-4 w-4" />
            </button>
          )}
          {(consoleAdmin || adminSwitch) && (
            <button onClick={adminSwitch ? leaveKioskView : () => openConsole()} className={iconBtn} title={adminSwitch ? 'Switch to admin view' : 'Admin console — models, archetypes, connections, users, settings'}>
              <ShieldCheck className="h-4 w-4" />
            </button>
          )}
          {/* Bell lives at the TOP so its downward dropdown isn't clipped by
              the h-screen overflow-hidden root when the rail is collapsed. */}
          <NotificationBell align="left" />
        </div>
        <div className="flex flex-1 flex-col items-center gap-1.5 overflow-y-auto py-2">
          {agents.map((a) => {
            const info = sessionInfo.get(a.id)
            const active = a.id === activeAgentId
            const hasChat = sessionInfo.has(a.id)
            const isBooting = booting.has(a.id) && a.state !== 'stopped'
            const openable = !switching.has(a.id) && ((a.reachable && !isBooting) || hasChat)
            const canStart = a.kind === 'local' ? a.state !== 'running' && !locked
              : a.kind !== 'shared' && a.state === 'stopped' && !a.managed
            const railStart = canStart && !switching.has(a.id)
            return (
              <button
                key={a.id}
                onClick={() => openable ? onOpen(a) : railStart ? onPower(a, true) : undefined}
                aria-disabled={!openable && !railStart}
                title={openable
                  ? `Chat with ${a.name}`
                  : railStart
                    ? `${a.name} — ${stateLabel(a)} · click to ${a.kind === 'local' ? 'check again' : 'start'}`
                    : `${a.name} — ${isBooting ? 'Starting…' : stateLabel(a)}`}
                className={`relative flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-[11px] font-semibold transition-colors ${
                  active
                    ? 'bg-violet-600/20 text-violet-300 ring-1 ring-violet-500/40'
                    : openable
                      ? 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                      : 'bg-zinc-900 text-zinc-500 dark:text-zinc-400 hover:bg-zinc-800'
                }`}
              >
                {switching.has(a.id) || isBooting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : initials(a.name)}
                <StateDot agent={a} className="absolute -bottom-0.5 -right-0.5 ring-2 ring-zinc-900" />
                {info?.busy && (
                  <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-pulse rounded-full bg-amber-500 ring-2 ring-zinc-900 dark:bg-amber-400" />
                )}
                {info?.unread && !active && !info.busy && (
                  <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-sky-400 ring-2 ring-zinc-900" />
                )}
              </button>
            )
          })}
          <button onClick={onSpawn} className={`${iconBtn} mt-1`} title={locked ? 'New agent from an archetype' : 'Spawn a new agent'}>
            <Plus className="h-4 w-4" />
          </button>
        </div>
        {(onTaskboard || onScheduled || onLibrary || onData || onFolders) && (
          <div className="flex flex-col items-center gap-0.5 border-t border-zinc-800 py-2">
            {onTaskboard && (
              <button
                onClick={onTaskboard}
                title={taskboardCount > 0 ? `Taskboard — ${taskboardCount} need${taskboardCount === 1 ? 's' : ''} you` : 'Taskboard'}
                aria-label="Taskboard"
                className="relative flex h-10 w-10 items-center justify-center rounded-lg text-teal-600 transition-colors hover:bg-zinc-800 dark:text-teal-400"
              >
                <ListChecks className="h-5 w-5" />
                {taskboardCount > 0 && (
                  <span className="absolute right-0.5 top-0.5 min-w-[16px] rounded-full bg-teal-600 px-1 text-center text-[9px] font-semibold leading-4 text-white">
                    {taskboardCount > 99 ? '99+' : taskboardCount}
                  </span>
                )}
              </button>
            )}
            {onScheduled && (
              <button
                onClick={onScheduled}
                title="Scheduled tasks"
                aria-label="Scheduled tasks"
                className="relative flex h-10 w-10 items-center justify-center rounded-lg text-amber-600 transition-colors hover:bg-zinc-800 dark:text-amber-400"
              >
                <AlarmClock className="h-5 w-5" />
                {scheduledRunning && <span className="absolute right-1.5 top-1.5 h-2 w-2 animate-pulse rounded-full bg-amber-500 ring-2 ring-zinc-900 dark:bg-amber-400" />}
              </button>
            )}
            {onLibrary && (
              <button
                onClick={onLibrary}
                title="File library"
                aria-label="File library"
                className="flex h-10 w-10 items-center justify-center rounded-lg text-violet-600 transition-colors hover:bg-zinc-800 dark:text-violet-400"
              >
                <LibraryBig className="h-5 w-5" />
              </button>
            )}
            {onData && (
              <button
                onClick={onData}
                title="Data library"
                aria-label="Data library"
                className="flex h-10 w-10 items-center justify-center rounded-lg text-emerald-600 transition-colors hover:bg-zinc-800 dark:text-emerald-400"
              >
                <Database className="h-5 w-5" />
              </button>
            )}
            {onFolders && (
              <button
                onClick={onFolders}
                title="Shared folders"
                aria-label="Shared folders"
                className="flex h-10 w-10 items-center justify-center rounded-lg text-sky-600 transition-colors hover:bg-zinc-800 dark:text-sky-400"
              >
                <Folders className="h-5 w-5" />
              </button>
            )}
          </div>
        )}
        {onSettings ? (
          <div className="flex flex-col items-center gap-0.5 border-t border-zinc-800 py-2">
            {onInfo && (
              <button
                onClick={onInfo}
                title="Info"
                aria-label="Info"
                className="flex h-10 w-10 items-center justify-center rounded-lg text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100"
              >
                <Info className="h-5 w-5" />
              </button>
            )}
            <button
              onClick={onSettings}
              title="Settings"
              aria-label="Settings"
              className="flex h-10 w-10 items-center justify-center rounded-lg text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100"
            >
              <Settings className="h-5 w-5" />
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-0.5 border-t border-zinc-800 py-2">
            <button onClick={toggleTheme} className={iconBtn} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
              {theme === 'dark' ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
            </button>
            <button onClick={onRefresh} className={iconBtn} title="Refresh agents">
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
            {authEnabled && authUser && (
              <button onClick={logoutUser} disabled={signingOut} className={iconBtn} title={signingOut ? 'Signing out…' : `Sign out ${authUser.display_name || authUser.email}`}>
                {signingOut ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <LogOut className="h-3.5 w-3.5" />}
              </button>
            )}
          </div>
        )}
      </aside>
    )
  }

  // ── Expanded column ──
  return (
    <aside
      className="relative flex shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/50"
      style={{ width: width.size }}
    >
      {/* Header */}
      <div className="flex h-14 items-center justify-between border-b border-zinc-800 px-3">
        <BrandMark connected={wsConnected} caption={adminSwitch ? 'User view' : undefined} />
        <div className="flex items-center gap-0.5">
          <NotificationBell align="left" />
          {!locked && (
            <button onClick={() => setLayoutMode('full')} className={iconBtn} title="Full view — all pages and panels">
              <LayoutDashboard className="h-4 w-4" />
            </button>
          )}
          {consoleAdmin && !appShell && (
            <button onClick={() => openConsole()} className={iconBtn} title="Admin console — models, archetypes, connections, users, settings">
              <ShieldCheck className="h-4 w-4" />
            </button>
          )}
          <button onClick={() => setOpen(false)} className={iconBtn} title="Hide agents">
            <ChevronLeft className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* List header */}
      <div className="flex items-center justify-between px-3 pt-3 pb-1">
        <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
          Agents ({agents.length})
        </span>
        <button onClick={onSpawn} className="rounded p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300" title={locked ? 'New agent from an archetype' : 'Spawn a new agent'}>
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Search (only once the list is long enough to need it) */}
      {searchable && (
        <div className="px-2 pb-1.5">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-600" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search agents…"
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 py-1 pl-7 pr-2 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
            />
          </div>
        </div>
      )}

      {/* Agents */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {agents.length === 0 && (
          <p className="px-2.5 py-6 text-center text-xs text-zinc-600">No agents yet.</p>
        )}
        {agents.length > 0 && shown.length === 0 && (
          <p className="px-2.5 py-6 text-center text-xs text-zinc-600">No agents match your search.</p>
        )}
        <ul className="flex flex-col gap-0.5">
          {shown.map((a) => {
            const info = sessionInfo.get(a.id)
            const active = a.id === activeAgentId
            const busy = !!info?.busy
            const KindIcon = a.kind === 'docker' ? Box : a.kind === 'process' ? Cpu : a.kind === 'shared' ? Users : Server
            const shared = a.kind === 'shared'
            const hasChat = sessionInfo.has(a.id)
            const isBooting = booting.has(a.id) && a.state !== 'stopped'
            const working = switching.has(a.id)
            const openable = !working && ((a.reachable && !isBooting) || hasChat)
            // A power switch for the agents Flight Deck runs; a local one can
            // only be checked again (and not from the kiosk).
            const powered = a.state === 'running' || !!a.asleep
            const canPower = a.kind !== 'local' && !shared && !a.managed && (powered || a.state === 'stopped')
            const canRecheck = a.kind === 'local' && a.state !== 'running' && !locked
            const menu = menuFor?.(a) ?? []
            return (
              <li key={a.id} className="group flex items-center gap-0.5">
                <button
                  onClick={() => onOpen(a)}
                  disabled={!openable}
                  title={openable ? `Chat with ${a.name}` : `${a.name} — ${isBooting ? 'Starting…' : stateLabel(a)}`}
                  className={`flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors ${
                    active
                      ? 'bg-zinc-800 text-zinc-100'
                      : busy
                        ? 'bg-amber-500/10 text-zinc-100 hover:bg-amber-500/20'
                        : openable
                          ? 'text-zinc-300 hover:bg-zinc-800/50 hover:text-zinc-100'
                          : 'text-zinc-500'
                  }`}
                >
                  <StateDot agent={a} className="shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="min-w-0 truncate font-medium">{a.name}</span>
                      {busy && (
                        <span
                          className="flex shrink-0 items-center gap-1 rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-medium leading-none text-amber-700 dark:text-amber-400"
                          title="Working"
                        >
                          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500 dark:bg-amber-400" />
                          Busy
                        </span>
                      )}
                      {info?.unread && !active && !busy && (
                        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400" title="New reply" />
                      )}
                      {shared && (
                        <span
                          className="flex shrink-0 items-center gap-0.5 rounded border border-sky-500/25 bg-sky-500/15 px-1 py-0.5 text-[9px] font-medium leading-none text-sky-700 dark:text-sky-300"
                          title={`Shared by ${a.ownerName || 'another user'}`}
                        >
                          <Users className="h-2.5 w-2.5" />shared
                        </span>
                      )}
                      {shareKeyOf(a) && <NetworkBadge shareKey={shareKeyOf(a)} />}
                    </div>
                    <div className="truncate text-[11px] text-zinc-500">
                      {isBooting
                        ? 'Starting…'
                        : shared && a.state === 'running'
                          ? `Shared by ${a.ownerName || 'another user'}`
                          : a.state === 'running' && a.reachable && a.description ? a.description : stateLabel(a)}
                    </div>
                  </div>
                  <KindIcon className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
                </button>
                {!locked && !shared && (
                  <button
                    onClick={() => onOptions(a)}
                    title={`Options — ${a.name}`}
                    className="shrink-0 rounded p-1.5 text-zinc-500 opacity-0 transition-colors hover:bg-zinc-800 hover:text-zinc-200 focus:opacity-100 group-hover:opacity-100"
                  >
                    <Settings className="h-3.5 w-3.5" />
                  </button>
                )}
                {menu.length > 0 && <AgentMenu agentName={a.name} items={menu} subtitle={menuSubtitle(a)} status={<StateDot agent={a} />} />}
                {canPower && (
                  <button
                    onClick={() => onPower(a, !powered)}
                    disabled={working}
                    role="switch"
                    aria-checked={powered}
                    aria-label={`${a.name} power`}
                    title={working ? (powered ? 'Powering off…' : 'Powering on…') : powered ? `Power off ${a.name}` : `Power on ${a.name}`}
                    className={`shrink-0 rounded p-1.5 transition-colors hover:bg-zinc-800 disabled:opacity-60 ${
                      powered
                        ? 'text-emerald-600 hover:text-red-600 dark:text-emerald-400 dark:hover:text-red-400'
                        : 'text-zinc-500 hover:text-emerald-600 dark:hover:text-emerald-400'
                    }`}
                  >
                    {working ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Power className="h-3.5 w-3.5" />}
                  </button>
                )}
                {shared && (
                  <button
                    onClick={() => onLeave(a)}
                    title={`Leave ${a.name}`}
                    aria-label={`Leave ${a.name}`}
                    className="shrink-0 rounded p-1.5 text-zinc-500 opacity-0 transition-colors hover:bg-zinc-800 hover:text-red-600 focus:opacity-100 group-hover:opacity-100 dark:hover:text-red-400"
                  >
                    <LogOut className="h-3.5 w-3.5" />
                  </button>
                )}
                {canRecheck && (
                  <button
                    onClick={() => onPower(a, true)}
                    disabled={working}
                    title="Check again"
                    className="shrink-0 rounded p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-emerald-600 dark:hover:text-emerald-400 disabled:opacity-60"
                  >
                    {working ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                  </button>
                )}
              </li>
            )
          })}
        </ul>
      </div>

      {(onTaskboard || onScheduled || onLibrary || onData || onFolders) && (
        <ToolTiles
          onTaskboard={onTaskboard}
          taskboardCount={taskboardCount}
          onScheduled={onScheduled}
          scheduledCount={scheduledCount}
          scheduledRunning={scheduledRunning}
          onLibrary={onLibrary}
          onData={onData}
          onFolders={onFolders}
        />
      )}

      {/* Footer — the kiosk's is one Settings button (KioskSettingsDialog). */}
      {onSettings ? (
        <div className="flex items-center gap-1 border-t border-zinc-800 p-2">
          <button
            onClick={onSettings}
            className="flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-3 py-2.5 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800 hover:text-zinc-100"
          >
            <Settings className="h-[18px] w-[18px] text-zinc-400" />
            Settings
            {authEnabled && authUser && (
              <span
                title={authUser.display_name || authUser.email}
                className="ml-auto flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet-500/15 text-[10px] font-semibold text-violet-400"
              >
                {initials(authUser.display_name || authUser.email)}
              </span>
            )}
          </button>
          {onInfo && (
            <button
              onClick={onInfo}
              title="Info — models, agents, profile, plan"
              aria-label="Info"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100"
            >
              <Info className="h-[18px] w-[18px]" />
            </button>
          )}
          {adminSwitch && (
            <button
              onClick={leaveKioskView}
              title="Switch to admin view — models, archetypes, connections, settings"
              aria-label="Switch to admin view"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-violet-600 transition-colors hover:bg-violet-500/10 dark:text-violet-400"
            >
              <ShieldCheck className="h-[18px] w-[18px]" />
            </button>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-between gap-2 border-t border-zinc-800 px-2 py-1.5">
          <div className="flex items-center gap-0.5">
            <button onClick={toggleTheme} className={iconBtn} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
              {theme === 'dark' ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
            </button>
            <button onClick={onRefresh} className={iconBtn} title="Refresh agents">
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
          {authEnabled && authUser && (
            <div className="flex min-w-0 items-center gap-1">
              <span className="truncate text-xs text-zinc-500">{authUser.display_name || authUser.email}</span>
              <button onClick={logoutUser} disabled={signingOut} className={iconBtn} title={signingOut ? 'Signing out…' : 'Sign out'}>
                {signingOut ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <LogOut className="h-3.5 w-3.5" />}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Right edge: drag to resize */}
      <div
        onMouseDown={width.onResizeStart}
        title="Drag to resize"
        className="absolute right-0 top-0 z-20 h-full w-1.5 cursor-col-resize transition-colors hover:bg-violet-500/30 active:bg-violet-500/40"
      />
    </aside>
  )
}

function StateDot({ agent, className = '' }: { agent: SimpleAgent; className?: string }) {
  const cls = agent.asleep
    ? 'bg-indigo-400/80 dark:bg-indigo-400/70'
    : agent.state === 'running'
    ? (agent.reachable ? 'bg-emerald-600 dark:bg-emerald-400' : 'bg-amber-500 dark:bg-amber-400')
    : agent.state === 'starting'
      ? 'animate-pulse bg-amber-400'
      : agent.state === 'unknown'
        ? 'bg-zinc-500'
        : 'bg-zinc-600'
  return <span className={`h-2 w-2 rounded-full ${cls} ${className}`} />
}

// ── Centre: nothing open yet ─────────────────────────────────────────

function EmptyChat({ hasAgents, onSpawn, locked = false, onGuide }: {
  hasAgents: boolean
  onSpawn: () => void
  locked?: boolean
  /** Kiosk only: open the Getting started guide. */
  onGuide?: () => void
}) {
  return (
    <div className="flex h-full items-center justify-center bg-zinc-950/80 p-6">
      <div className="w-full max-w-sm rounded-2xl border border-zinc-800 bg-zinc-900/60 p-8 text-center">
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-violet-600/20 text-violet-400">
          <MessagesSquare className="h-6 w-6" />
        </div>
        <h2 className="mb-1.5 text-lg font-semibold text-zinc-100">
          {hasAgents ? 'Pick an agent' : 'No agents available'}
        </h2>
        <p className="text-sm text-zinc-400">
          {hasAgents
            ? locked
              ? 'Choose an agent on the left to start a conversation. What it saves for you lands in Library, Data and Folders, bottom left.'
              : 'Choose an agent on the left to start a conversation. Its files and data show up on the right.'
            : locked
              ? 'Create an agent from an archetype to start chatting.'
              : 'Spawn your first agent to start chatting.'}
        </p>
        {!hasAgents && (
          <button
            onClick={onSpawn}
            className="mt-6 rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500"
          >
            {locked ? 'New agent' : 'Spawn an agent'}
          </button>
        )}
        {onGuide && (
          <button
            onClick={onGuide}
            className="mx-auto mt-4 flex items-center gap-1.5 text-xs font-medium text-zinc-500 transition-colors hover:text-violet-600 dark:hover:text-violet-400"
          >
            <Compass className="h-3.5 w-3.5" />
            New here? Getting started
          </button>
        )}
      </div>
    </div>
  )
}

// ── Right: files + datastore of the active agent ─────────────────────

function ContextColumn({ agentId, agentName, onOptions, menu }: {
  agentId: string | null
  agentName: string
  onOptions?: () => void
  /** Kiosk only: the active agent's "⋯". */
  menu?: ReactNode
}) {
  const open = useUIStore((s) => s.simpleRightOpen)
  const setOpen = useUIStore((s) => s.setSimpleRightOpen)
  // The handle is on the LEFT edge of a right-docked column, so dragging left
  // makes it wider.
  const width = usePersistedSize('fd:simple-right-width', 320, 240, 640, 'x', 'backward')

  if (!open) {
    return (
      <aside className="flex w-12 shrink-0 flex-col items-center gap-0.5 border-l border-zinc-800 bg-zinc-900/50 py-2">
        <button onClick={() => setOpen(true)} className={iconBtn} title="Show files and data">
          <ChevronLeft className="h-4 w-4" />
        </button>
        {onOptions && (
          <button onClick={onOptions} className={iconBtn} title="Agent options">
            <Settings className="h-4 w-4" />
          </button>
        )}
        <button onClick={() => setOpen(true)} className={iconBtn} title="Files">
          <FolderOpen className="h-4 w-4 text-violet-400" />
        </button>
        <button onClick={() => setOpen(true)} className={iconBtn} title="Datastore">
          <Database className="h-4 w-4 text-emerald-400" />
        </button>
      </aside>
    )
  }

  return (
    <aside
      className="relative flex shrink-0 flex-col border-l border-zinc-800 bg-zinc-950/40"
      style={{ width: width.size }}
    >
      {/* Left edge: drag to resize */}
      <div
        onMouseDown={width.onResizeStart}
        title="Drag to resize"
        className="absolute left-0 top-0 z-20 h-full w-1.5 cursor-col-resize transition-colors hover:bg-violet-500/30 active:bg-violet-500/40"
      />

      {/* Header */}
      <div className="flex h-10 shrink-0 items-center justify-between border-b border-zinc-800 px-3">
        <span className="min-w-0 truncate text-xs font-medium text-zinc-400" title={agentName || undefined}>
          {agentId ? agentName : 'Files & data'}
        </span>
        <div className="flex items-center gap-0.5">
          {onOptions && (
            <button onClick={onOptions} className={iconBtn} title="Agent options">
              <Settings className="h-3.5 w-3.5" />
            </button>
          )}
          {agentId && menu}
          <button onClick={() => setOpen(false)} className={iconBtn} title="Hide files and data">
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      {agentId ? (
        <FilesDataPanels agentId={agentId} agentName={agentName} />
      ) : (
        <div className="flex flex-1 items-center justify-center px-6 text-center text-xs text-zinc-500">
          Open a chat to see that agent's files and datastore here.
        </div>
      )}
    </aside>
  )
}

// One agent's files over its datastore — the right column's body, and the
// kiosk's "Files & data" dialog. A shared agent: on a process agent, on a deck
// that serves them, the member's commons panels (the agent's saved/ files and
// datastore, read through Flight Deck's member routes); otherwise chat only —
// a Docker agent, an older Flight Deck.
function FilesDataPanels({ agentId, agentName }: { agentId: string; agentName: string }) {
  const panels = useAgentPanels(agentId)
  // Measure the height available to the files+datastore split so the files
  // pane can never be taller than (region − a datastore minimum).
  const regionRef = useRef<HTMLDivElement>(null)
  const [regionH, setRegionH] = useState(0)
  // Which region is mounted (own split, a member's panels, chat-only): the
  // observer follows it when the chat or the member panels change.
  useEffect(() => {
    const el = regionRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => setRegionH(el.clientHeight))
    ro.observe(el)
    setRegionH(el.clientHeight)
    return () => ro.disconnect()
  }, [panels.region])
  const DATASTORE_MIN = 140
  const filesLiveMax = regionH > 0 ? Math.max(120, regionH - DATASTORE_MIN) : undefined
  const filesH = usePersistedSize('fd:simple-right-files-height', 360, 120, 1400, 'y', 'forward', filesLiveMax)

  if (panels.region === 'chat') {
    return (
      <div className="flex flex-1 items-center justify-center px-6 text-center text-xs text-zinc-500">
        {CHAT_ONLY_TEXT}
      </div>
    )
  }

  if (panels.region === 'member') {
    const { row, files, datastore } = panels
    const owner = row.owner_name || row.owner_email
    return (
      <div ref={regionRef} className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {files && datastore ? (
          <>
            {/* Same split as the owner's: files on top, datastore below. */}
            <div className="min-h-[120px] overflow-hidden" style={{ height: filesH.size }}>
              <SharedFilesPanel key={agentId} agentRef={row.agent_ref} agentName={agentName} ownerName={owner} />
            </div>
            <div
              onMouseDown={filesH.onResizeStart}
              title="Drag to resize"
              className="h-1 shrink-0 cursor-row-resize border-y border-zinc-800 bg-zinc-900 transition-colors hover:bg-violet-500/40"
            />
            <div className="min-h-[120px] flex-1 overflow-hidden">
              <SharedDatastorePanel key={agentId} agentRef={row.agent_ref} agentName={agentName} ownerName={owner} />
            </div>
          </>
        ) : files ? (
          <div className="min-h-0 flex-1 overflow-hidden">
            <SharedFilesPanel key={agentId} agentRef={row.agent_ref} agentName={agentName} ownerName={owner} />
          </div>
        ) : (
          <div className="min-h-0 flex-1 overflow-hidden">
            <SharedDatastorePanel key={agentId} agentRef={row.agent_ref} agentName={agentName} ownerName={owner} />
          </div>
        )}
      </div>
    )
  }

  return (
    <div ref={regionRef} className="flex min-h-0 flex-1 flex-col overflow-hidden">
      {/* Top: files. Shrinkable with a floor so a stale oversized height
          can't push the datastore off-screen. */}
      <div className="min-h-[120px] overflow-hidden" style={{ height: filesH.size }}>
        <AgentFilesPanel key={agentId} containerId={agentId} />
      </div>
      {/* Divider (drag to resize files vs datastore) */}
      <div
        onMouseDown={filesH.onResizeStart}
        title="Drag to resize"
        className="h-1 shrink-0 cursor-row-resize border-y border-zinc-800 bg-zinc-900 transition-colors hover:bg-violet-500/40"
      />
      {/* Bottom: datastore tables — keeps a floor so it never hits 0. */}
      <div className="min-h-[120px] flex-1 overflow-hidden">
        <AgentDatastorePanel key={agentId} containerId={agentId} />
      </div>
    </div>
  )
}

// ── Kiosk: an agent's files and data in a dialog ─────────────────────
//
// The kiosk has no right column: any agent's "⋯" → Files & data shows the
// column's panels here instead. They read the agent itself, so they hold
// what the File and Data libraries don't: the agents they don't index (a
// Docker or local one, one Flight Deck runs itself, one shared with the
// user) and, for one they do, the files they leave out.

function KioskFilesDataDialog({ agentId, agentName, onClose }: { agentId: string; agentName: string; onClose: () => void }) {
  return (
    <AgentDialog title="Files & data" icon={FolderOpen} agentName={agentName} size="md" onClose={onClose}>
      <FilesDataPanels agentId={agentId} agentName={agentName} />
    </AgentDialog>
  )
}

// ── Phone ────────────────────────────────────────────────────────────
//
// One screen at a time: the agent list, or one chat filling the screen with a
// back arrow to the list (the phone's own Back gesture does the same). The
// active agent's files and datastore open as a full-screen sheet (in the
// kiosk, the File library for an agent it indexes). The phone's Back closes a
// files view first (the sheet, a library, an agent's Files & data), then the
// chat. Touch targets are at least 40px, the layout follows the visible
// viewport (dvh), and the bottom bar keeps clear of the home indicator.

function PhoneLayout({
  agents, activeAgent, activeAgentId, activeSessionName, sessionInfo, switching, booting,
  onOpen, onPower, onLeave, onSpawn, onSettings, onInfo, onTaskboard, taskboardCount = 0, onScheduled, scheduledRunning = false,
  onLibrary, onData, onFolders, menuFor, locked, adminSwitch = false, overlayOpen = false, onCloseOverlays, chatOpens = 0,
}: {
  agents: SimpleAgent[]
  activeAgent: SimpleAgent | null
  activeAgentId: string | null
  activeSessionName: string
  sessionInfo: Map<string, { busy: boolean; unread: boolean }>
  switching: Set<string>
  booting: Set<string>
  onOpen: (a: SimpleAgent) => void
  onPower: (a: SimpleAgent, on: boolean) => void
  onLeave: (a: SimpleAgent) => void
  onSpawn: () => void
  onSettings?: () => void
  onInfo?: () => void
  onTaskboard?: () => void
  taskboardCount?: number
  onScheduled?: () => void
  scheduledRunning?: boolean
  onLibrary?: () => void
  onData?: () => void
  onFolders?: () => void
  /** Kiosk only: what an agent's "⋯" offers (Info, Settings, Remove). */
  menuFor?: (a: SimpleAgent) => AgentMenuItem[]
  locked: boolean
  adminSwitch?: boolean
  /** The Taskboard, the File or Data library, the shared folders or an
   *  agent's Files & data is open (they live a level up): Back closes them
   *  before anything else. */
  overlayOpen?: boolean
  onCloseOverlays?: () => void
  /** Counts the chats opened (see SimpleLayout's handleOpen). */
  chatOpens?: number
}) {
  const wsConnected = useAgentStore((s) => s.wsConnected)
  const activeChatId = useChatStore((s) => s.activeChatId)
  const [search, setSearch] = useState('')
  const [filesOpen, setFilesOpen] = useState(false)
  // Back to the list keeps the chat open; opening any chat shows it again: a
  // tap, a new agent that just came up, a library's "Agent chat" — also the
  // one left for the list, whose id stays the same (`chatOpens` still
  // counts it). Set while rendering, not in an effect after: a library that
  // closes as the chat opens leaves no render with neither in front, which
  // would give up the Back entry and set it again at once.
  const [atList, setAtList] = useState(false)
  const [opened, setOpened] = useState({ chat: activeChatId, n: chatOpens })
  if (opened.chat !== activeChatId || opened.n !== chatOpens) {
    setOpened({ chat: activeChatId, n: chatOpens })
    setAtList(false)
  }
  const showChat = !!activeAgentId && !atList

  // The phone's Back: a files view in front closes first (staying in the
  // chat), then an open chat returns to the list — instead of leaving Flight
  // Deck. One history entry stands for both; it is set again when Back has
  // closed a files view and the chat is still there to go back from.
  const overlay = filesOpen || overlayOpen
  const layered = showChat || overlay
  const pushed = useRef(false)
  // Our own Back is on its way (the entry given up once nothing is in front;
  // the timer stands in should no popstate come): that popstate isn't the
  // user's, and an entry needed again meanwhile waits for it — pushed before
  // that Back lands, the browser takes it away too.
  const unwinding = useRef<ReturnType<typeof setTimeout> | null>(null)
  const popRef = useRef({ layered, overlay, showChat, onCloseOverlays })
  useEffect(() => { popRef.current = { layered, overlay, showChat, onCloseOverlays } })
  // Our Back has landed: whatever came to the front meanwhile gets its entry.
  const unwound = useCallback(() => {
    if (unwinding.current) clearTimeout(unwinding.current)
    unwinding.current = null
    if (popRef.current.layered && !pushed.current) {
      window.history.pushState({ fdPhoneChat: true }, '')
      pushed.current = true
    }
  }, [])
  useEffect(() => {
    if (unwinding.current) return
    if (layered && !pushed.current) {
      window.history.pushState({ fdPhoneChat: true }, '')
      pushed.current = true
    } else if (!layered && pushed.current) {
      pushed.current = false
      if (window.history.state?.fdPhoneChat) {
        unwinding.current = setTimeout(unwound, 1500)
        window.history.back()
      }
    }
  }, [layered, unwound])
  useEffect(() => {
    const onPop = () => {
      if (unwinding.current) { unwound(); return }
      if (!pushed.current) return
      pushed.current = false
      const cur = popRef.current
      if (cur.overlay) {
        setFilesOpen(false)
        cur.onCloseOverlays?.()
        if (cur.showChat) {
          window.history.pushState({ fdPhoneChat: true }, '')
          pushed.current = true
        }
        return
      }
      setAtList(true)
    }
    window.addEventListener('popstate', onPop)
    return () => {
      window.removeEventListener('popstate', onPop)
      if (unwinding.current) clearTimeout(unwinding.current)
    }
  }, [unwound])

  const searchable = agents.length > 6
  const shown = useMemo(() => {
    const q = searchable ? search.trim().toLowerCase() : ''
    if (!q) return agents
    return agents.filter((a) => a.name.toLowerCase().includes(q) || a.description.toLowerCase().includes(q))
  }, [agents, search, searchable])

  const phoneBtn = 'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-zinc-400 transition-colors active:bg-zinc-800 hover:bg-zinc-800 hover:text-zinc-100'
  // The agent list's header carries every tool: a little narrower, so seven
  // of them and the bell still leave the logo its room on a 375 px phone.
  const toolBtn = 'flex h-10 w-9 shrink-0 items-center justify-center rounded-lg text-zinc-400 transition-colors active:bg-zinc-800 hover:bg-zinc-800 hover:text-zinc-100'

  if (showChat) {
    const a = activeAgent
    const info = activeAgentId ? sessionInfo.get(activeAgentId) : undefined
    const menu = a && menuFor ? menuFor(a) : []
    // The kiosk keeps an agent's files in the File library when it indexes
    // that agent: the folder opens there, filtered to it (its "Everything it
    // keeps" and the "⋯" → Files & data open the sheet). Any other agent
    // (and every one outside the kiosk) keeps the sheet.
    const filesInLibrary = locked && !!a && inLibraries(a)
    const openFiles = () => {
      if (filesInLibrary) useFileLibraryStore.getState().openLibrary(a.startKey)
      else setFilesOpen(true)
    }
    return (
      <div className="flex h-[100dvh] flex-col overflow-hidden bg-zinc-950">
        <header className="flex h-14 shrink-0 items-center gap-1 border-b border-zinc-800 px-1.5">
          <button onClick={() => setAtList(true)} className={phoneBtn} aria-label="Back to agents" title="Agents">
            <ChevronLeft className="h-6 w-6" />
          </button>
          <div className="flex min-w-0 flex-1 items-center gap-2">
            {a && <StateDot agent={a} className="shrink-0" />}
            <div className="min-w-0">
              <div className="truncate text-[15px] font-semibold text-zinc-100">{a?.name || activeSessionName}</div>
              <div className="truncate text-[11px] text-zinc-500">
                {info?.busy ? 'Working…' : a ? (a.kind === 'shared' ? `Shared by ${a.ownerName || 'another user'}` : stateLabel(a)) : ''}
              </div>
            </div>
          </div>
          <button
            onClick={openFiles}
            className={phoneBtn}
            aria-label={filesInLibrary ? 'Files in the File library' : 'Files and data'}
            title={filesInLibrary ? 'Files in the File library' : 'Files and data'}
          >
            <FolderOpen className="h-5 w-5" />
          </button>
          {a && menu.length > 0 && <AgentMenu agentName={a.name} items={menu} subtitle={menuSubtitle(a)} status={<StateDot agent={a} />} phone />}
        </header>
        <main className="flex min-h-0 flex-1 flex-col overflow-hidden pb-[env(safe-area-inset-bottom)]">
          <ChatPanel variant="simple" />
        </main>
        {filesOpen && activeAgentId && (
          <PhoneFilesSheet agentId={activeAgentId} agentName={a?.name || activeSessionName} onClose={() => setFilesOpen(false)} />
        )}
      </div>
    )
  }

  return (
    <div className="flex h-[100dvh] flex-col overflow-hidden bg-zinc-950">
      <header className="flex h-14 shrink-0 items-center gap-0.5 border-b border-zinc-800 pl-3 pr-1.5">
        <div className="@container min-w-0 flex-1"><BrandMark connected={wsConnected} caption={adminSwitch ? 'User view' : undefined} compact /></div>
        {adminSwitch && (
          <button onClick={leaveKioskView} className={toolBtn} aria-label="Switch to admin view" title="Switch to admin view">
            <ShieldCheck className="h-5 w-5 text-violet-600 dark:text-violet-400" />
          </button>
        )}
        <NotificationBell />
        {onTaskboard && (
          <button onClick={onTaskboard} className={`${toolBtn} relative`} aria-label="Taskboard" title="Taskboard">
            <ListChecks className="h-5 w-5 text-teal-600 dark:text-teal-400" />
            {taskboardCount > 0 && (
              <span className="absolute right-0 top-1 min-w-[16px] rounded-full bg-teal-600 px-1 text-center text-[9px] font-semibold leading-4 text-white">
                {taskboardCount > 99 ? '99+' : taskboardCount}
              </span>
            )}
          </button>
        )}
        {onScheduled && (
          <button onClick={onScheduled} className={`${toolBtn} relative`} aria-label="Scheduled tasks" title="Scheduled tasks">
            <AlarmClock className="h-5 w-5 text-amber-600 dark:text-amber-400" />
            {scheduledRunning && <span className="absolute right-1.5 top-2 h-2 w-2 animate-pulse rounded-full bg-amber-500 ring-2 ring-zinc-950 dark:bg-amber-400" />}
          </button>
        )}
        {onLibrary && (
          <button onClick={onLibrary} className={toolBtn} aria-label="File library" title="File library">
            <LibraryBig className="h-5 w-5 text-violet-600 dark:text-violet-400" />
          </button>
        )}
        {onData && (
          <button onClick={onData} className={toolBtn} aria-label="Data library" title="Data library">
            <Database className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
          </button>
        )}
        {onFolders && (
          <button onClick={onFolders} className={toolBtn} aria-label="Shared folders" title="Shared folders">
            <Folders className="h-5 w-5 text-sky-600 dark:text-sky-400" />
          </button>
        )}
        {onInfo && (
          <button onClick={onInfo} className={toolBtn} aria-label="Info" title="Info">
            <Info className="h-5 w-5" />
          </button>
        )}
        {onSettings && (
          <button onClick={onSettings} className={toolBtn} aria-label="Settings" title="Settings">
            <Settings className="h-5 w-5" />
          </button>
        )}
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 pb-4 pt-3">
        <div className="mb-2 flex items-center justify-between px-1">
          <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">Agents ({agents.length})</span>
        </div>
        {searchable && (
          <div className="relative mb-3">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search agents…"
              className="h-11 w-full rounded-xl border border-zinc-800 bg-zinc-900 pl-9 pr-3 text-base text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
            />
          </div>
        )}

        {agents.length === 0 && (
          <div className="mt-6 rounded-2xl border border-zinc-800 bg-zinc-900/60 p-6 text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-violet-600/20 text-violet-400">
              <MessagesSquare className="h-6 w-6" />
            </div>
            <h2 className="text-base font-semibold text-zinc-100">No agents yet</h2>
            <p className="mt-1 text-sm text-zinc-400">
              {locked ? 'Create an agent from an archetype to start chatting.' : 'Spawn your first agent to start chatting.'}
            </p>
          </div>
        )}
        {agents.length > 0 && shown.length === 0 && (
          <p className="py-8 text-center text-sm text-zinc-500">No agents match your search.</p>
        )}

        <ul className="flex flex-col gap-2">
          {shown.map((a) => {
            const info = sessionInfo.get(a.id)
            const busy = !!info?.busy
            const shared = a.kind === 'shared'
            const hasChat = sessionInfo.has(a.id)
            const isBooting = booting.has(a.id) && a.state !== 'stopped'
            const working = switching.has(a.id)
            const openable = !working && ((a.reachable && !isBooting) || hasChat)
            const powered = a.state === 'running' || !!a.asleep
            const canPower = a.kind !== 'local' && !shared && !a.managed && (powered || a.state === 'stopped')
            const menu = menuFor?.(a) ?? []
            return (
              <li
                key={a.id}
                className={`flex items-center gap-1 rounded-xl border pr-1 transition-colors ${
                  a.id === activeAgentId ? 'border-violet-500/40 bg-violet-500/10' : busy ? 'border-amber-500/30 bg-amber-500/10' : 'border-zinc-800 bg-zinc-900/60'
                }`}
              >
                <button
                  onClick={() => onOpen(a)}
                  disabled={!openable}
                  className="flex min-w-0 flex-1 items-center gap-3 py-3.5 pl-3.5 text-left disabled:opacity-60"
                >
                  <span className="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-zinc-800 text-xs font-semibold text-zinc-300">
                    {working || isBooting ? <Loader2 className="h-4 w-4 animate-spin" /> : initials(a.name)}
                    <StateDot agent={a} className="absolute -bottom-0.5 -right-0.5 ring-2 ring-zinc-900" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-1.5">
                      <span className="truncate text-[15px] font-medium text-zinc-100">{a.name}</span>
                      {busy && <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-amber-500 dark:bg-amber-400" title="Working" />}
                      {info?.unread && !busy && <span className="h-2 w-2 shrink-0 rounded-full bg-sky-400" title="New reply" />}
                      {shareKeyOf(a) && <NetworkBadge shareKey={shareKeyOf(a)} />}
                    </span>
                    <span className="block truncate text-xs text-zinc-500">
                      {isBooting ? 'Starting…'
                        : busy ? 'Working…'
                          : shared && a.state === 'running' ? `Shared by ${a.ownerName || 'another user'}`
                            : a.state === 'running' && a.reachable && a.description ? a.description : stateLabel(a)}
                    </span>
                  </span>
                </button>
                {canPower && (
                  <button
                    onClick={() => onPower(a, !powered)}
                    disabled={working}
                    role="switch"
                    aria-checked={powered}
                    aria-label={`${a.name} power`}
                    title={powered ? `Power off ${a.name}` : `Power on ${a.name}`}
                    className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-lg transition-colors active:bg-zinc-800 disabled:opacity-60 ${
                      powered ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-500'
                    }`}
                  >
                    {working ? <Loader2 className="h-5 w-5 animate-spin" /> : <Power className="h-5 w-5" />}
                  </button>
                )}
                {menu.length > 0 && <AgentMenu agentName={a.name} items={menu} subtitle={menuSubtitle(a)} status={<StateDot agent={a} />} phone />}
                {shared && (
                  <button
                    onClick={() => onLeave(a)}
                    aria-label={`Leave ${a.name}`}
                    title={`Leave ${a.name}`}
                    className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg text-zinc-500 active:bg-zinc-800"
                  >
                    <LogOut className="h-5 w-5" />
                  </button>
                )}
              </li>
            )
          })}
        </ul>
      </div>

      <div className="shrink-0 border-t border-zinc-800 bg-zinc-950 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
        <button
          onClick={onSpawn}
          className="flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-violet-600 text-[15px] font-medium text-white transition-colors active:bg-violet-700 hover:bg-violet-500"
        >
          <Plus className="h-5 w-5" />
          New agent
        </button>
      </div>
    </div>
  )
}

// The active agent's files and datastore, full screen, one at a time.
function PhoneFilesSheet({ agentId, agentName, onClose }: { agentId: string; agentName: string; onClose: () => void }) {
  const [tab, setTab] = useState<'files' | 'data'>('files')
  const panels = useAgentPanels(agentId)
  const hasFiles = panels.region === 'own' || (panels.region === 'member' && panels.files)
  const hasData = panels.region === 'own' || (panels.region === 'member' && panels.datastore)
  const current = tab === 'files' ? (hasFiles ? 'files' : 'data') : (hasData ? 'data' : 'files')
  const owner = panels.region === 'member' ? panels.row.owner_name || panels.row.owner_email : ''

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // As AgentDialog: a viewer, link dialog or table open inside marks the
      // Esc it used to close itself.
      setTimeout(() => { if (!e.defaultPrevented) onClose() }, 0)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return createPortal(
    <div role="dialog" aria-label="Files and data" className="fixed inset-0 z-50 flex h-[100dvh] flex-col bg-zinc-950">
      <header className="flex h-14 shrink-0 items-center gap-2 border-b border-zinc-800 pl-4 pr-1.5">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[15px] font-semibold text-zinc-100">Files &amp; data</div>
          <div className="truncate text-[11px] text-zinc-500">{agentName}</div>
        </div>
        <button onClick={onClose} aria-label="Close" className="flex h-10 w-10 items-center justify-center rounded-lg text-zinc-400 active:bg-zinc-800">
          <X className="h-5 w-5" />
        </button>
      </header>
      {hasFiles && hasData && (
        <div className="shrink-0 border-b border-zinc-800 p-2">
          <div className="grid grid-cols-2 rounded-xl bg-zinc-900 p-1" role="tablist">
            {(['files', 'data'] as const).map((k) => (
              <button
                key={k}
                role="tab"
                aria-selected={current === k}
                onClick={() => setTab(k)}
                className={`flex h-9 items-center justify-center gap-1.5 rounded-lg text-sm font-medium transition-colors ${
                  current === k ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500'
                }`}
              >
                {k === 'files' ? <FolderOpen className="h-4 w-4 text-violet-400" /> : <Database className="h-4 w-4 text-emerald-400" />}
                {k === 'files' ? 'Files' : 'Datastore'}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-hidden pb-[env(safe-area-inset-bottom)]">
        {!hasFiles && !hasData ? (
          <div className="flex h-full items-center justify-center px-8 text-center text-sm text-zinc-500">{CHAT_ONLY_TEXT}</div>
        ) : panels.region === 'member' ? (
          current === 'files'
            ? <SharedFilesPanel key={agentId} agentRef={panels.row.agent_ref} agentName={agentName} ownerName={owner} />
            : <SharedDatastorePanel key={agentId} agentRef={panels.row.agent_ref} agentName={agentName} ownerName={owner} />
        ) : current === 'files' ? (
          <AgentFilesPanel key={agentId} containerId={agentId} />
        ) : (
          <AgentDatastorePanel key={agentId} containerId={agentId} />
        )}
      </div>
    </div>,
    document.body,
  )
}
