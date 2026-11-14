import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  ChevronLeft,
  ChevronRight,
  EyeOff,
  FolderLock,
  Wrench,
  LayoutDashboard,
  Search,
  Plus,
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
  Goal,
  AlarmClock,
  LibraryBig,
  ListChecks,
  SlidersHorizontal,
  Smile,
  Trash2,
  Archive,
  BookCheck,
  BookOpen,
  GraduationCap,
  ShieldCheck,
  Smartphone,
  Activity,
  KeyRound,
  Ship,
  FolderInput,
  FolderPlus,
  Pin,
  PinOff,
  Target,
  LayoutGrid,
  Network,
  Flame,
} from 'lucide-react'
import { useUIStore } from '../../stores/uiStore'
import { openConsole, leaveKioskView, selectConsoleAdmin } from '../../stores/consoleStore'
import { isAppShell, hasAndroidAppSettings, openAndroidAppSettings } from '../../services/desktopShell'
import { useAgentStore } from '../../stores/agentStore'
import { useLessonInbox } from '../../stores/lessonInboxStore'
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
import { useGroupStore } from '../../stores/groupStore'
import { agentSections, isFolder } from '../../utils/agentGroups'
import { useFileLibraryStore } from '../../stores/fileLibraryStore'
import { folderSourceId } from '../../services/fileLibraryApi'
import { FileLibrary } from '../library/FileLibrary'
import { useDataLibraryStore } from '../../stores/dataLibraryStore'
import { DataLibrary } from '../library/DataLibrary'
import { useActivityStore } from '../../stores/activityStore'
import { ToolCallRegister } from '../audit/ToolCallRegister'
import { takeTaskboardLink, useTaskboardStore } from '../../stores/taskboardStore'
import { TaskBoard } from '../taskboard/TaskBoard'
import { CrewBoard } from '../crew/CrewBoard'
import { useCrewAllowed, useCrewStore } from '../../stores/crewStore'
import { useCustomToolsOn, useCustomToolsStore } from '../../stores/customToolsStore'
import { CustomToolsPage } from '../customTools/CustomToolsPage'
import { useAnonymizeStore } from '../../stores/anonymizeStore'
import { AnonymizePage } from '../anonymize/AnonymizePage'
import { useRoomsStore } from '../../stores/roomsStore'
import { RoomsPage } from '../rooms/RoomsPage'
import { useCaptainSwarmStore, useCanSwarm } from '../../stores/captainSwarmStore'
import { CaptainSwarmPage } from '../../pages/CaptainSwarmPage'
import { useTeamWorkStore } from '../../stores/teamWorkStore'
import { BasnaPage } from '../../pages/BasnaPage'
import { CouncilPage } from '../../pages/CouncilPage'
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
import { AgentBackupDialog, AgentImportDialog } from './AgentBundleDialogs'
import { AgentBackupsDialog } from './AgentBackupsDialog'
import { ForgeAgentsDialog } from '../forge/ForgeAgentsDialog'
import { KioskByokDialog } from './KioskByokDialog'
import { useByokStore } from '../../stores/byokStore'
import { useQuotaStore } from '../../stores/quotaStore'
import ConnectionsPage, { type TabId as ConnectionsTab } from '../../pages/ConnectionsPage'
import { ProfilePage } from '../../pages/ProfilePage'
import { CoursePage } from '../../pages/CoursePage'
import { OPEN_COURSE_EVENT } from '../../services/course'
import { OPEN_PROFILE_EVENT } from '../../utils/profileSources'
import { NotificationBell } from '../common/NotificationCenter'
import { LinePanel } from '../line/LinePanel'
import { OPEN_LINE_EVENT } from '../../services/line'
import { useLineStore } from '../../stores/lineStore'
import { BrandMark } from '../common/BrandMark'
import { KioskInfoDialog, type InfoAgent } from './KioskInfoDialog'
import { KioskOnboardingDialog, type OnboardingAction } from './KioskOnboardingDialog'
import { KioskAgentIdentityDialog } from './KioskAgentIdentityDialog'
import { KioskAgentPictureDialog } from './KioskAgentPicture'
import { AgentHome } from './AgentHome'
import { AgentCards } from './AgentCards'
import { AgentPicture } from '../agents/AgentPicture'
import { useAgentAvatarStore } from '../../stores/agentAvatarStore'
import { MenuSheet, type MenuSection } from './KioskMenuSheet'
import {
  AgentDialog, AgentMenu, KioskAgentInfoDialog, KioskAgentInstructionsDialog, KioskAgentSettingsDialog, KioskRemoveAgentDialog,
  type AgentMenuItem,
} from './KioskAgentDialogs'
import { AGENT_DRAG_TYPE, AgentGroupDialog, AgentGroupPicker, AgentSections, RailGroupMark, type RowParts } from './KioskAgentGroups'
import { SimpleNav } from './SimpleNav'
import { navEntries, type NavEntry, type NavTarget } from './navDestinations'
import { PeoplePanel } from './PeoplePanel'
import { useIsMobile } from '../../hooks/useMediaQuery'
import { NetworkBadge, networkShareAction } from '../agents/NetworkShare'
import { dockerShareKey, processShareKey, useNetworkShareStore } from '../../stores/networkShareStore'

// ── Simple layout ────────────────────────────────────────────────────
//
// A chat-first arrangement of the same Flight Deck: on the left the outcome
// nav (Today, Agents, Inbox, Tasks, People, Automations, Activity —
// SimpleNav) over the person's team of agents, and the conversation (or, with
// none open, Today or the Agents cards) on one rounded panel. No right column:
// an agent's files and data are the File and Data libraries (bottom left) and
// its "⋯" → Files & data. No pages, no director — those live in the full
// layout, one click away. The left column folds to a thin rail and remembers
// it. The look comes from the shell tokens and .fd-* classes in index.css.

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

/** A group to edit (`id`), or a new one with these agents in it. */
interface GroupRequest { id?: string; agentIds?: string[] }

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

/** Under an agent's name in the list: one word for its state (the tooltip says more). */
function rowState(a: SimpleAgent): string {
  if (a.asleep) return 'Asleep'
  if (a.state === 'stopped') return a.kind === 'local' ? 'Offline' : 'Off'
  return stateLabel(a)
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

const iconBtn = 'rounded-full p-2 text-[var(--fd-ink-faint)] transition-colors hover:bg-[var(--fd-hover)] hover:text-[var(--fd-ink)]'
/** A button on an agent's row in the column: one size for all of them, so they line up. */
const rowBtn = 'flex h-7 w-7 shrink-0 items-center justify-center rounded-md transition-colors'

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
  const [forgeOpen, setForgeOpen] = useState(false)
  // Agent bundles: an agent's ⋯ → Back up / export, and New agent → From a file.
  const [backupFor, setBackupFor] = useState<{ slug: string; name: string } | null>(null)
  // Open: '' for a new agent, or the slug of the agent a file goes into.
  const [importOpen, setImportOpen] = useState<string | null>(null)
  // Settings → Agent backups (schedule, backups kept, signed files).
  const [backupsOpen, setBackupsOpen] = useState(false)
  // A new agent whose "way of working" setup opens right after it is created.
  const [setupFor, setSetupFor] = useState<SetupRequest | null>(null)
  // An agent's knowledge, to browse (its ⋯ menu → Knowledge).
  // An agent's onboarding page; Way of working opened from it comes back to it when it closes.
  const [onboardingFor, setOnboardingFor] = useState<{ slug: string; name: string; tab: OnboardingTab } | null>(null)
  // Lessons to approve: what each agent noticed in the owner's chats (its menu shows its count).
  const lessonInbox = useLessonInbox((s) => s.data)
  const showLessons = useLessonInbox((s) => s.show)
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
  // The Line: what the deck and the agents said on their own (its row above
  // the agents, and the bell's link to it, open it).
  const [lineOpen, setLineOpen] = useState(false)
  useEffect(() => {
    const open = () => setLineOpen(true)
    window.addEventListener(OPEN_LINE_EVENT, open)
    return () => window.removeEventListener(OPEN_LINE_EVENT, open)
  }, [])
  // Kiosk Course dialog — the goals the deck, the user and their agents work
  // toward. A "Still on course?" item (the Line, the bell) opens it.
  const [courseOpen, setCourseOpen] = useState(false)
  useEffect(() => {
    const open = () => { setLineOpen(false); setCourseOpen(true) }
    window.addEventListener(OPEN_COURSE_EVENT, open)
    return () => window.removeEventListener(OPEN_COURSE_EVENT, open)
  }, [])
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
  // …its identity as a colleague: role, own inbox, the email it picks up, who it hands work to.
  const [agentIdentityId, setAgentIdentityId] = useState<string | null>(null)
  // …and its files and data, read live from the agent (see agentMenuItems).
  const [filesDataId, setFilesDataId] = useState<string | null>(null)
  // …and the picture its square shows instead of its initials.
  const [agentPictureId, setAgentPictureId] = useState<string | null>(null)
  const avatars = useAgentAvatarStore((s) => s.avatars)
  const forgetAvatar = useAgentAvatarStore((s) => s.forgetAgent)
  // Agent groups: a group being made or edited, and the agent whose "⋯" → Group is open.
  const groups = useGroupStore((s) => s.groups)
  const forgetAgent = useGroupStore((s) => s.forgetAgent)
  const toggleAgentPin = useGroupStore((s) => s.toggleAgentPin)
  const [groupDialog, setGroupDialog] = useState<GroupRequest | null>(null)
  const [groupPickerId, setGroupPickerId] = useState<string | null>(null)
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
  // Flight Deck's tasks and the jobs agents set up for themselves (their cron).
  const scheduledCount = useScheduledStore((s) =>
    s.jobs.filter((j) => j.enabled === 1).length + s.agentJobs.filter((j) => j.enabled).length)
  const scheduledRunning = useScheduledStore((s) => s.jobs.some((j) => j.running) || s.agentJobs.some((j) => j.running))
  // File library: every file the agents made, summarised (agent column,
  // Settings, the phone header) — or one agent's (its "⋯", its phone chat).
  const libraryOpen = useFileLibraryStore((s) => s.open)
  const libraryAgent = useFileLibraryStore((s) => s.agentFilter)
  const libraryFocus = useFileLibraryStore((s) => s.focus)
  const openLibrary = useFileLibraryStore((s) => s.openLibrary)
  const showLibraryFile = useFileLibraryStore((s) => s.showFile)
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
  const showTaskboardTask = useTaskboardStore((s) => s.showTask)
  const closeTaskboard = useTaskboardStore((s) => s.close)
  const fetchTaskboardSummary = useTaskboardStore((s) => s.fetchSummary)
  // Crew: coding agents working for this user, when an admin approved it.
  const crewOpen = useCrewStore((s) => s.open)
  const closeCrew = useCrewStore((s) => s.close)
  // Custom tools: the person's own programs and the ones offered to them.
  const toolsOpen = useCustomToolsStore((s) => s.open)
  const closeTools = useCustomToolsStore((s) => s.close)
  const anonOpen = useAnonymizeStore((s) => s.open)
  const closeAnon = useAnonymizeStore((s) => s.close)
  const roomsOpen = useRoomsStore((s) => s.open)
  const closeRooms = useRoomsStore((s) => s.close)
  const refreshCustomTools = useCustomToolsStore((s) => s.refreshStatus)
  useEffect(() => { refreshCustomTools() }, [refreshCustomTools])
  // Captain Swarm: many attempts at one task, documents and research, each checked by Captain.
  const swarmOpen = useCaptainSwarmStore((s) => s.open)
  const closeSwarm = useCaptainSwarmStore((s) => s.close)
  const canSwarm = useCanSwarm()
  // Vatra, Basna, Council: agents working as a team, opened from More.
  const teamWork = useTeamWorkStore((s) => s.open)
  const closeTeamWork = useTeamWorkStore((s) => s.close)
  const taskboardCount = useTaskboardStore((s) => s.needsMe)
  // People: who is on the Taskboard's tasks (the nav's People).
  const [peopleOpen, setPeopleOpen] = useState(false)
  // With no chat open, the centre shows Today or the Agents cards (the nav picks).
  const [homeView, setHomeView] = useState<'today' | 'agents'>('today')
  // The Line's counts, for the nav's Inbox.
  const lineUnread = useLineStore((s) => s.unread)
  const lineWaiting = useLineStore((s) => s.waiting)
  // Shared folders: the files the agents keep together (vfs:<folder>/…).
  const [foldersOpen, setFoldersOpen] = useState(false)
  const openFolders = () => setFoldersOpen(true)
  // A place Shared folders is asked to show (a library's Show in folder);
  // `n` is new for every ask.
  const [foldersAt, setFoldersAt] = useState<{ folder: string; path: string; n: number } | null>(null)
  // The libraries' way back to a shared folder: the library gives way to it.
  const showInFolder = (folder: string, path = '') => {
    closeLibrary()
    closeData()
    setFoldersAt({ folder, path, n: Date.now() })
    setFoldersOpen(true)
  }
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
  // Taskboard v2: a link with ?taskboard=<task id> (a nudge, a bell item, a
  // notification) opens the board on that task, once the user is signed in.
  useEffect(() => {
    if (authEnabled === null || (authEnabled && !isAuthenticated)) return
    const taskId = takeTaskboardLink()
    if (taskId) showTaskboardTask(taskId)
  }, [showTaskboardTask, authEnabled, isAuthenticated])

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
      forgetAgent(a.id)
      forgetAvatar(a.id)
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
  const groupMenuItem = (a: SimpleAgent): AgentMenuItem => {
    const inGroups = groups.filter((g) => isFolder(g) && g.agentIds.includes(a.id))
    return {
      icon: FolderInput, label: 'Group', tone: 'sky', group: 'place',
      hint: inGroups.length ? `In ${inGroups.map((g) => g.name).join(', ')}` : 'Keep it with others like it',
      onSelect: () => setGroupPickerId(a.id),
    }
  }
  // In a group its "⋯" also pins it to the top of that group (all a phone has).
  const pinMenuItems = (a: SimpleAgent): AgentMenuItem[] => {
    const g = groups.find((x) => isFolder(x) && x.agentIds.includes(a.id))
    if (!g) return []
    const pinned = !!g.pinnedAgentIds?.includes(a.id)
    return [{
      icon: pinned ? PinOff : Pin, label: pinned ? 'Unpin' : 'Pin', tone: 'violet', direct: true, group: 'place',
      hint: pinned ? `Back among the others in ${g.name}` : `To the top of ${g.name}`,
      onSelect: () => toggleAgentPin(g.id, a.id),
    }]
  }
  // Its picture in the list: an icon or one of the person's own, for any agent they see.
  const pictureMenuItem = (a: SimpleAgent): AgentMenuItem => {
    const pic = avatars[a.id]
    return {
      icon: Smile, label: 'Picture', tone: 'fuchsia', group: 'who',
      hint: pic ? ('emoji' in pic ? `Shows ${pic.emoji} instead of its initials` : 'Shows your picture instead of its initials')
        : 'An icon or a picture of yours, instead of its initials',
      onSelect: () => setAgentPictureId(a.id),
    }
  }
  // A process agent of yours as a colleague: who it is to the team and its email.
  const identityMenuItems = (a: SimpleAgent): AgentMenuItem[] => (a.kind === 'process' && !a.managed ? [{
    icon: IdCard, label: 'Identity & email', tone: 'cyan', group: 'who',
    hint: 'Its role, its own inbox, the email it picks up, who it hands work to',
    onSelect: () => setAgentIdentityId(a.id),
  }] : [])
  // Its files and datastore tables, in a dialog. The kiosk offers it for an
  // agent the Library and Data (bottom left) don't index — one shared with the
  // user (when its owner's deck gives members some), a Docker one; outside the
  // kiosk (`always`) for every agent, as the right column used to show them.
  // Read from the agent itself: a stopped one has none to show.
  const filesDataMenuItems = (a: SimpleAgent, always = false): AgentMenuItem[] => {
    if (a.kind === 'shared') {
      return agentPanels(a.id, memberWorkspace, sharedAgents).region === 'member'
        ? [{ icon: FolderOpen, label: 'Files & data', hint: 'What everyone who uses it can open', tone: 'amber', group: 'work', onSelect: () => setFilesDataId(a.id) }]
        : []
    }
    if (!always && inLibraries(a)) return []
    const up = a.reachable && !a.asleep && !booting.has(a.id) && !switching.has(a.id)
    return [{
      icon: FolderOpen, label: 'Files & data', tone: 'amber', group: 'work',
      hint: up ? 'Its files and datastore tables' : 'Its files and tables show while it runs',
      onSelect: () => setFilesDataId(a.id), disabled: !up,
    }]
  }
  const agentMenuItems = (a: SimpleAgent): AgentMenuItem[] => {
    if (a.kind === 'shared') {
      return [
        ...filesDataMenuItems(a),
        pictureMenuItem(a),
        groupMenuItem(a),
        ...pinMenuItems(a),
      ]
    }
    const up = a.reachable && !a.asleep && !booting.has(a.id) && !switching.has(a.id)
    const items: AgentMenuItem[] = [{
      icon: Info, label: 'Info', hint: up ? 'Its intentions and topics' : 'Power it on to see its intentions and topics',
      tone: 'sky', group: 'work', onSelect: () => setAgentInfoId(a.id), disabled: !up,
    }]
    items.push(...filesDataMenuItems(a))
    if (a.kind === 'docker' || a.kind === 'process') {
      items.push({
        icon: Activity, label: 'Activity', hint: 'Every tool it used, and with what', tone: 'emerald', group: 'work',
        onSelect: () => openActivity(a.kind === 'process' ? { slug: a.startKey } : { ref: a.startKey }),
      })
    }
    // Taskboard v2: its tasks — what it was asked and sent, and what needs you.
    if (a.kind === 'process' && !a.managed) {
      items.push({
        icon: ListChecks, label: 'Tasks', hint: 'What it was asked and sent, and what needs you', tone: 'teal', group: 'work',
        onSelect: () => openTaskboard(a.startKey),
      })
    }
    if ((a.kind === 'docker' || a.kind === 'process') && !a.managed) {
      items.push(
        { icon: BookOpen, label: 'Instructions, insights & skills', hint: 'What it keeps in mind, has learned, and knows how to do',
          tone: 'violet', group: 'knows', onSelect: () => setAgentInstructionsId(a.id) },
      )
      // Your documents and way of working → its rules, skills and knowledge (process agents: FD writes its folder).
      if (a.kind === 'process') {
        items.push({ icon: GraduationCap, label: 'Onboarding', hint: 'Its documents, the lessons it was taught, and how well it knows them',
          tone: 'fuchsia', group: 'knows', onSelect: () => setOnboardingFor({ slug: a.startKey, name: a.name, tab: 'overview' }) })
        // What it noticed in your chats, waiting for your yes — only its own (Lessons to approve, scoped).
        const waitingLessons = lessonInbox?.agents.find((x) => x.slug === a.startKey)?.waiting.length ?? 0
        items.push({ icon: BookCheck, label: 'Lessons to approve', tone: waitingLessons ? 'amber' : 'violet', group: 'knows',
          hint: waitingLessons ? `${waitingLessons} ${waitingLessons === 1 ? 'lesson waits' : 'lessons wait'} for you — noticed in your chats`
            : 'What it noticed in your chats, for you to approve',
          onSelect: () => showLessons(a.startKey, { name: a.name }) })
      }
      items.push(
        ...identityMenuItems(a),
        { icon: SlidersHorizontal, label: 'Settings', hint: 'Model, how it thinks, vision', tone: 'zinc', group: 'who', onSelect: () => setAgentSettingsId(a.id) },
        pictureMenuItem(a),
      )
      const net = networkShareAction(netAgents[shareKeyOf(a)], {
        canShare: netCanShare, admin: netAdmin, deckName: netDeck, displayName: a.name, setShared: netSetShared,
        ask: netAskShare,
        onError: (m) => useNotificationStore.getState().add('error', 'BotPort network', m),
      })
      if (net.show) items.push({ icon: Globe, label: net.label, hint: net.hint, tone: 'cyan', group: 'who', onSelect: net.onClick, disabled: net.disabled })
    }
    // A local agent, or one Flight Deck runs: its picture is all of "who it is" here.
    if (!((a.kind === 'docker' || a.kind === 'process') && !a.managed)) items.push(pictureMenuItem(a))
    items.push(groupMenuItem(a), ...pinMenuItems(a))
    // The whole agent in one file — Flight Deck writes it from the agent's folder (process agents).
    if (a.kind === 'process' && !a.managed) {
      items.push({ icon: Archive, label: 'Back up / export', hint: 'Save it to a file, to restore it here or on another deck',
        tone: 'teal', group: 'place', onSelect: () => setBackupFor({ slug: a.startKey, name: a.name }) })
    }
    if ((a.kind === 'docker' || a.kind === 'process') && !a.managed) {
      items.push(
        { icon: Trash2, label: 'Remove agent', hint: 'Delete it and its folder', danger: true,
          onSelect: () => setAgentRemoveId(a.id), disabled: switching.has(a.id) },
      )
    }
    return items
  }
  // Off the kiosk its "⋯" holds only the agent's group and pin (the gear has the rest).
  const menuFor = locked ? agentMenuItems
    : (a: SimpleAgent) => [...filesDataMenuItems(a, true), groupMenuItem(a), ...identityMenuItems(a), pictureMenuItem(a), ...pinMenuItems(a)]
  const groupPickerAgent = groupPickerId ? agents.find((a) => a.id === groupPickerId) : undefined
  const agentInfo = agentInfoId ? agents.find((a) => a.id === agentInfoId) : undefined
  const agentInstructions = agentInstructionsId ? agents.find((a) => a.id === agentInstructionsId) : undefined
  const agentSettings = agentSettingsId ? agents.find((a) => a.id === agentSettingsId) : undefined
  const agentRemove = agentRemoveId ? agents.find((a) => a.id === agentRemoveId) : undefined
  const agentIdentity = agentIdentityId ? agents.find((a) => a.id === agentIdentityId) : undefined
  const filesData = filesDataId ? agents.find((a) => a.id === filesDataId) : undefined
  const agentPicture = agentPictureId ? agents.find((a) => a.id === agentPictureId) : undefined

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
    setPeopleOpen(false)
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
  const phoneOverlayOpen = libraryOpen || dataOpen || foldersOpen || activityOpen || taskboardOpen || crewOpen || swarmOpen || toolsOpen
    || anonOpen || roomsOpen || !!teamWork || !!filesData || peopleOpen
  const closePhoneOverlays = () => {
    if (taskboardOpen && peopleOpen) { closeTaskboard(); return }
    closeLibrary(); closeData(); closeActivity(); closeTaskboard(); setFoldersOpen(false); setFilesDataId(null); closeAnon(); closeRooms()
    closeTeamWork(); setPeopleOpen(false)
  }

  // The outcome nav. Each destination opens the view it always did (the
  // Line, the Taskboard, Scheduled tasks, Activity) — one at a time, so the
  // one picked is the one in front; Today closes them and the chat, leaving
  // the home. A chat's session stays connected: its agent opens it again.
  const closeDestinations = () => {
    setLineOpen(false); closeTaskboard(); setPeopleOpen(false); closeScheduled(); closeActivity()
  }
  const goTo = (open: () => void) => () => { closeDestinations(); open() }
  const goHome = (view: 'today' | 'agents') => () => {
    closeDestinations()
    setHomeView(view)
    if (useChatStore.getState().activeChatId) useChatStore.setState({ activeChatId: null, chatOpen: false })
  }
  const nav: NavEntry[] = navEntries({
    onToday: goHome('today'),
    onAgents: goHome('agents'),
    onInbox: goTo(() => setLineOpen(true)),
    inboxUnread: lineUnread,
    inboxWaiting: lineWaiting,
    onTasks: goTo(() => openTaskboard()),
    tasksNeedMe: taskboardCount,
    onPeople: goTo(() => setPeopleOpen(true)),
    onAutomations: goTo(() => openScheduled()),
    automationsCount: scheduledCount,
    automationsRunning: scheduledRunning,
    onActivity: goTo(() => openActivity()),
    // The Agent store is a page of its own; a locked account never reaches it.
    onStore: locked ? undefined : () => { window.location.assign('/store') },
  })
  const navCurrent: NavTarget | null = lineOpen ? 'inbox'
    : taskboardOpen ? 'tasks'
      : peopleOpen ? 'people'
        : scheduledOpen ? 'automations'
          : activityOpen ? 'activity'
            : hasActiveSession ? null : homeView
  // People's links into the Taskboard: it opens over People, which waits
  // under it — closing the board comes back to the same person.
  const peopleToTasks = () => openTaskboard()
  const peopleToTask = (taskId: string) => showTaskboardTask(taskId)
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
  // A held reply the Taskboard can't show waits under Connections → Google;
  // its settings' "Connect Jev" opens Connections → Jev.
  const taskboardConnections = (tab: ConnectionsTab = 'google') => {
    closeTaskboard()
    setPeopleOpen(false)
    if (locked) { openConnections(tab); return }
    // Taskboard: the full Connections page opens on the tab it remembers — this one.
    try { localStorage.setItem('fd:connections-tab', tab) } catch { /* storage blocked: its last tab */ }
    useChatStore.setState({ chatFullscreen: false }); setLayoutMode('full'); setView('connections')
  }

  // The bell: a finished profile research run opens the Profile dialog here
  // (the full layout shows its page instead).
  useEffect(() => {
    if (!locked) return
    const open = () => { setSettingsOpen(false); setProfileOpen(true) }
    window.addEventListener(OPEN_PROFILE_EVENT, open)
    return () => window.removeEventListener(OPEN_PROFILE_EVENT, open)
  }, [locked])

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
          onConnect={(service) => { setArchetypeOpen(false); openConnections(service) }}
          onSpawned={(slug, setup) => {
            setArchetypeOpen(false)
            startBoot(`proc-${slug}`, true)
            if (setup) setSetupFor(setup)
            else useNotificationStore.getState().add('info', 'Agent starting', `${slug} will open here as soon as it's up.`)
          }}
          onForge={() => { setArchetypeOpen(false); setForgeOpen(true) }}
          onImport={() => { setArchetypeOpen(false); setImportOpen('') }}
        />
      )}
      {importOpen !== null && (
        <AgentImportDialog
          agents={agents.filter((a) => a.kind === 'process' && !a.managed).map((a) => ({ slug: a.startKey, name: a.name }))}
          into={importOpen || undefined}
          onClose={() => setImportOpen(null)}
          onConnect={() => openConnections()}
          onImported={(r) => {
            useProcessStore.getState().fetchProcesses()
            // A new agent opens once it's up; one a file went into restarts if it was running.
            if (!('into' in r)) startBoot(`proc-${r.slug}`, true)
            else if (r.restarted) startBoot(`proc-${r.slug}`, false)
          }}
        />
      )}
      {backupFor && (
        <AgentBackupDialog agent={backupFor} onClose={() => setBackupFor(null)}
          onBringIn={() => { setImportOpen(backupFor.slug); setBackupFor(null) }}
          onSchedule={() => { setBackupFor(null); setBackupsOpen(true) }} />
      )}
      {backupsOpen && (
        <AgentBackupsDialog
          agents={agents.filter((a) => a.kind === 'process' && !a.managed).map((a) => ({ slug: a.startKey, name: a.name }))}
          onClose={() => setBackupsOpen(false)}
          onRestored={(slug, restarted) => {
            useProcessStore.getState().fetchProcesses()
            if (restarted) startBoot(`proc-${slug}`, false)
          }}
        />
      )}
      {forgeOpen && (
        <ForgeAgentsDialog
          onClose={() => setForgeOpen(false)}
          onCreated={(slugs) => { for (const slug of slugs) startBoot(`proc-${slug}`, false) }}
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
          phone={isMobile}
          onAdmin={adminSwitch ? () => { setSettingsOpen(false); leaveKioskView() } : undefined}
          onClose={() => setSettingsOpen(false)}
          onRefresh={() => { refresh(); setSettingsOpen(false) }}
          onConnections={() => { setSettingsOpen(false); openConnections() }}
          onProfile={() => { setSettingsOpen(false); setProfileOpen(true) }}
          onCourse={() => { setSettingsOpen(false); setCourseOpen(true) }}
          onGuide={() => { setSettingsOpen(false); setGuideOpen(true) }}
          onTaskboard={() => { setSettingsOpen(false); openTaskboard() }}
          onScheduled={() => { setSettingsOpen(false); openScheduled() }}
          onLibrary={() => { setSettingsOpen(false); openLibrary() }}
          onData={() => { setSettingsOpen(false); openData() }}
          onFolders={() => { setSettingsOpen(false); openFolders() }}
          onActivity={() => { setSettingsOpen(false); openActivity() }}
          onByok={() => { setSettingsOpen(false); openByok() }}
          onBackups={() => { setSettingsOpen(false); setBackupsOpen(true) }}
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
            onOpenTaskboard={(jobId) => { closeScheduled(); showTaskboardTask(`sched:job:${jobId}`) }}
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
            onOpenLive={openLibraryLive}
            liveReady={libraryLiveReady}
            onShowInFolder={showInFolder}
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
            onOpenScheduled={(jobId) => { closeTaskboard(); setPeopleOpen(false); openScheduled(jobId) }}
            onOpenChat={openTaskChat}
            onOpenConnections={taskboardConnections}
          />
        </KioskWideDialog>
      )}
      {crewOpen && (
        <KioskWideDialog label="Crew" phone={isMobile} onClose={closeCrew} roomy>
          <CrewBoard phone={isMobile} onClose={closeCrew} />
        </KioskWideDialog>
      )}
      {toolsOpen && (
        <KioskWideDialog label="Custom tools" phone={isMobile} onClose={closeTools} roomy>
          <CustomToolsPage phone={isMobile} onClose={closeTools} />
        </KioskWideDialog>
      )}
      {anonOpen && (
        <KioskWideDialog label="Anonymize" phone={isMobile} onClose={closeAnon} widest>
          <AnonymizePage phone={isMobile} onClose={closeAnon} />
        </KioskWideDialog>
      )}
      {roomsOpen && (
        <KioskWideDialog label="Data rooms" phone={isMobile} onClose={closeRooms} widest>
          <RoomsPage phone={isMobile} onClose={closeRooms} />
        </KioskWideDialog>
      )}
      {(teamWork === 'vatra' || teamWork === 'basna') && (
        <KioskWideDialog label={teamWork === 'vatra' ? 'Vatra' : 'Basna'} phone={isMobile} onClose={closeTeamWork} roomy>
          <BasnaPage key={teamWork} mode={teamWork} onClose={closeTeamWork} />
        </KioskWideDialog>
      )}
      {teamWork === 'council' && (
        <KioskWideDialog label="Council" phone={isMobile} onClose={closeTeamWork} roomy>
          <CouncilPage onClose={closeTeamWork} />
        </KioskWideDialog>
      )}
      {swarmOpen && canSwarm && (
        <KioskWideDialog label="Captain Swarm" phone={isMobile} onClose={closeSwarm} roomy>
          <div className="flex h-full min-h-0 flex-col">
            <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-3 py-2">
              <span className="flex items-center gap-2 text-sm font-medium text-zinc-200">
                <Target className="h-4 w-4 text-amber-400" /> Captain Swarm
              </span>
              <button onClick={closeSwarm} aria-label="Close" className="rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1"><CaptainSwarmPage /></div>
          </div>
        </KioskWideDialog>
      )}
      {dataOpen && (
        <KioskWideDialog label="Data library" phone={isMobile} onClose={closeData} widest>
          <DataLibrary
            phone={isMobile}
            initialAgent={dataAgent}
            onClose={closeData}
            onOpenAgent={openLibraryAgent}
            onShowInFolder={(folder) => showInFolder(folder)}
          />
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
        <KioskWideDialog label="Shared folders" phone={isMobile} onClose={() => { setFoldersOpen(false); setFoldersAt(null) }} widest>
          {/* A folder's files and tables open in the File and Data libraries,
              over this dialog — Escape there closes only them. */}
          <SharedFolders
            phone={isMobile}
            canLink={canLinkFolders}
            onClose={() => { setFoldersOpen(false); setFoldersAt(null) }}
            onConnect={foldersConnections}
            onOpenLibrary={(folder) => openLibrary(folderSourceId(folder))}
            onOpenFileInLibrary={(folder, path, name) => showLibraryFile({ agent: folderSourceId(folder), relPath: path, filename: name })}
            onOpenTables={(folder) => openData(folderSourceId(folder))}
            at={foldersAt}
            covered={libraryOpen || dataOpen}
          />
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
      {agentIdentity && agentIdentity.kind === 'process' && (
        <KioskAgentIdentityDialog slug={agentIdentity.startKey} agentName={agentIdentity.name} onClose={() => setAgentIdentityId(null)} />
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
      {filesData && (isMobile
        ? <PhoneFilesSheet agentId={filesData.id} agentName={filesData.name} onClose={() => setFilesDataId(null)} />
        : <FilesDataDialog agentId={filesData.id} agentName={filesData.name} onClose={() => setFilesDataId(null)} />)}
      {groupDialog && (
        <AgentGroupDialog
          key={groupDialog.id ?? 'new'}
          groupId={groupDialog.id}
          agentIds={groupDialog.agentIds}
          agents={agents}
          phone={isMobile}
          onClose={() => setGroupDialog(null)}
        />
      )}
      {agentPicture && (
        <KioskAgentPictureDialog agentId={agentPicture.id} agentName={agentPicture.name} onClose={() => setAgentPictureId(null)} />
      )}
      {groupPickerAgent && (
        <AgentGroupPicker agentId={groupPickerAgent.id} agentName={groupPickerAgent.name} phone={isMobile} onClose={() => setGroupPickerId(null)} />
      )}
      {locked && connectionsOpen && <KioskConnectionsDialog initialTab={connectionsTab} onClose={() => setConnectionsOpen(false)} />}
      {locked && profileOpen && <KioskProfileDialog onClose={() => setProfileOpen(false)} />}
      {locked && courseOpen && (
        <KioskDialog label="Course" widest onClose={() => setCourseOpen(false)}>
          <CoursePage kiosk />
        </KioskDialog>
      )}
      {peopleOpen && (
        <KioskWideDialog label="People" phone={isMobile} onClose={() => setPeopleOpen(false)}>
          <PeoplePanel onClose={() => setPeopleOpen(false)} onOpenTasks={peopleToTasks} onOpenTask={peopleToTask}
            covered={taskboardOpen} />
        </KioskWideDialog>
      )}
      {lineOpen && (
        <KioskWideDialog label="Inbox" phone={isMobile} widest onClose={() => setLineOpen(false)}>
          <LinePanel onClose={() => setLineOpen(false)} />
        </KioskWideDialog>
      )}

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
          onGroup={setGroupDialog}
          onSettings={locked ? () => setSettingsOpen(true) : undefined}
          onInfo={locked ? () => setInfoOpen(true) : undefined}
          nav={nav}
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

  // With every agent off, the centre offers to switch the first one on.
  const firstOff = agents.find((a) => a.state === 'stopped' && !a.managed && (a.kind === 'docker' || a.kind === 'process'))
  // No chat open: Today or the Agents cards, the same agents and the same empty card.
  const homeProps = {
    agents, sessionInfo, booting, switching, onOpen: handleOpen, locked,
    onGuide: locked ? () => setGuideOpen(true) : undefined,
    empty: (
      <EmptyChat
        hasAgents={agents.length > 0}
        onSpawn={goSpawn}
        locked={locked}
        onGuide={locked ? () => setGuideOpen(true) : undefined}
        startName={firstOff?.name}
        onStart={firstOff && !switching.has(firstOff.id) ? () => handlePower(firstOff, true) : undefined}
      />
    ),
  }

  return (
    <div className="fd-shell flex h-screen overflow-hidden">
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
        onGroup={setGroupDialog}
        onOptions={setOptionsAgent}
        onSettings={locked ? () => setSettingsOpen(true) : undefined}
        onInfo={locked ? () => setInfoOpen(true) : undefined}
        nav={nav}
        navCurrent={navCurrent}
        onLibrary={() => openLibrary()}
        onData={() => openData()}
        onFolders={openFolders}
        menuFor={menuFor}
        locked={locked}
        adminSwitch={adminSwitch}
      />

      {/* The centre: one rounded panel floating on the shell. */}
      <main className="fd-main-panel my-2 mr-2 flex min-w-0 flex-1 flex-col overflow-hidden" style={{ minWidth: 320 }}>
        {hasActiveSession
          ? <ChatPanel variant="simple" />
          : homeView === 'agents' ? <AgentCards {...homeProps} /> : <AgentHome {...homeProps} />}
      </main>

      {dialogs}
    </div>
  )
}

// ── Kiosk: Settings ──────────────────────────────────────────────────
//
// The kiosk keeps its whole footer behind one Settings button: this menu (the
// same MenuSheet as an agent's "⋯") holds what the agents are doing and keep,
// the user's own things, the quick ones — theme, a refresh of the agent list,
// the Getting started guide — and sign-out (how a shared kiosk hands the screen
// to the next person).

function KioskSettingsDialog({ phone, onAdmin, onClose, onRefresh, onConnections, onProfile, onCourse, onGuide, onTaskboard, onScheduled, onLibrary, onData, onFolders, onActivity, onByok, onBackups }: {
  /** The phone layout: two tiles to a row, not three. */
  phone: boolean
  /** The desktop app's admin in the users' kiosk: back to the Admin console. */
  onAdmin?: () => void
  onClose: () => void
  onRefresh: () => void
  onConnections: () => void
  onProfile: () => void
  onCourse: () => void
  onGuide: () => void
  onTaskboard: () => void
  onScheduled: () => void
  onLibrary: () => void
  onData: () => void
  onFolders: () => void
  onActivity: () => void
  onByok: () => void
  onBackups: () => void
}) {
  const { theme, toggle: toggleTheme } = useThemeStore()
  // Offered where the deck allows own keys (the plan status says so).
  const quota = useQuotaStore((s) => s.status)
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const authUser = useAuthStore((s) => s.user)
  const signingOut = useAuthStore((s) => s.signingOut)
  const signedIn = authEnabled && authUser
  const crewAllowed = useCrewAllowed()
  const openCrew = useCrewStore((s) => s.openCrew)
  const toolsOn = useCustomToolsOn()
  const openTools = useCustomToolsStore((s) => s.openTools)
  const openSwarm = useCaptainSwarmStore((s) => s.openSwarm)
  const canSwarm = useCanSwarm()
  const openAnonymize = useAnonymizeStore((s) => s.openAnonymize)
  const openRooms = useRoomsStore((s) => s.openRooms)
  const openTeamWork = useTeamWorkStore((s) => s.openTeamWork)

  // Closed without choosing: the focus goes back to the Settings button.
  const [opener] = useState(() => document.activeElement as HTMLElement | null)
  const close = useCallback(() => {
    onClose()
    opener?.focus()
  }, [onClose, opener])

  const dark = theme === 'dark'
  const sections: MenuSection[] = [
    ...(onAdmin ? [{ items: [{ icon: ShieldCheck, label: 'Switch to admin view', hint: 'Models, archetypes, connections and settings', onSelect: onAdmin, tone: 'amber' as const }] }] : []),
    {
      title: 'Your agents at work',
      items: [
        { icon: ListChecks, label: 'Tasks', hint: 'The Taskboard: asked, sent, and what needs you', onSelect: onTaskboard, tone: 'teal' },
        ...(crewAllowed ? [{ icon: Ship, label: 'Crew', hint: 'Coding agents working for you', onSelect: () => { onClose(); openCrew() }, tone: 'teal' as const }] : []),
        ...(toolsOn ? [{ icon: Wrench, label: 'Custom tools', hint: 'Programs your agents run, yours and ones offered to you', onSelect: () => { onClose(); openTools() }, tone: 'violet' as const }] : []),
        ...(canSwarm ? [{ icon: Target, label: 'Captain Swarm', hint: 'Many tries at a task, document or research; Captain checks each', onSelect: () => { onClose(); openSwarm() }, tone: 'amber' as const }] : []),
        { icon: AlarmClock, label: 'Automations', hint: 'Scheduled tasks: on a timer, results by chat or email', onSelect: onScheduled, tone: 'amber' },
        { icon: Archive, label: 'Agent backups', hint: 'Your agents backed up on a schedule, and restored', onSelect: onBackups, tone: 'teal' },
        { icon: Activity, label: 'Activity', hint: 'Every tool they used, and how it went', onSelect: onActivity, tone: 'emerald' },
      ],
    },
    // Also in the agent column's More; a phone has no column, so this is its way in.
    {
      title: 'Agents as a team',
      items: [
        { icon: Flame, label: 'Vatra', hint: 'A lead splits one job; the team builds one result', onSelect: () => { onClose(); openTeamWork('vatra') }, tone: 'violet' },
        { icon: Network, label: 'Basna', hint: 'Several agents answer; the best of each merged into one', onSelect: () => { onClose(); openTeamWork('basna') }, tone: 'sky' },
        { icon: MessagesSquare, label: 'Council', hint: 'Your agents discuss a topic in rounds, then sum it up', onSelect: () => { onClose(); openTeamWork('council') }, tone: 'amber' },
      ],
    },
    {
      title: 'What they keep',
      items: [
        { icon: LibraryBig, label: 'File library', hint: 'Every file they made, easy to find', onSelect: onLibrary, tone: 'violet' },
        { icon: Database, label: 'Data library', hint: 'Every table they keep, easy to read', onSelect: onData, tone: 'cyan' },
        { icon: Folders, label: 'Shared folders', hint: 'The folders they work in together', onSelect: onFolders, tone: 'sky' },
        { icon: EyeOff, label: 'Anonymize', hint: 'Names and numbers out of a file; the real ones kept for you', onSelect: () => { onClose(); openAnonymize() }, tone: 'fuchsia' },
        { icon: FolderLock, label: 'Data rooms', hint: 'A data room anonymized for Claude; results restored here', onSelect: () => { onClose(); openRooms() }, tone: 'fuchsia' },
      ],
    },
    {
      title: 'You',
      items: [
        { icon: IdCard, label: 'Profile', hint: 'You, your company, your preferences', onSelect: onProfile, tone: 'fuchsia' },
        { icon: Goal, label: 'Course', hint: 'What you and your agents work toward', onSelect: onCourse, tone: 'violet' },
        { icon: Plug, label: 'Connections', hint: 'Google, Microsoft, MCP and Jev', onSelect: onConnections, tone: 'sky' },
        ...(quota?.byok_allowed || quota?.byok ? [{
          icon: KeyRound, label: 'Your own AI keys',
          hint: quota.byok ? 'On: your keys, no daily limits' : 'Your own API keys, no daily limits',
          onSelect: onByok, tone: 'amber' as const,
        }] : []),
      ],
    },
    {
      compact: true,
      items: [
        { icon: dark ? Moon : Sun, label: 'Dark theme', hint: `Switch to the ${dark ? 'light' : 'dark'} look`, onSelect: toggleTheme, on: dark },
        { icon: RefreshCw, label: 'Refresh agents', hint: 'Check every agent again', onSelect: onRefresh },
        { icon: Compass, label: 'Getting started', hint: 'A short tour: agents, connections, Telegram and more', onSelect: onGuide },
        // The Android app: back to its start screen — the saved decks and accounts.
        ...(hasAndroidAppSettings() ? [{ icon: Smartphone, label: 'Switch deck', hint: 'Your Flight Decks and accounts on this phone', onSelect: openAndroidAppSettings }] : []),
      ],
    },
  ]

  return (
    <MenuSheet
      label="Settings"
      picture={signedIn ? initials(authUser.display_name || authUser.email) : <Settings className="h-6 w-6" />}
      title={signedIn ? authUser.display_name || authUser.email : 'Settings'}
      subtitle={signedIn && authUser.display_name ? authUser.email : ''}
      sections={sections}
      apart={signedIn ? [{
        icon: signingOut ? Loader2 : LogOut,
        label: signingOut ? 'Signing out…' : 'Sign out',
        hint: 'End your session on this device',
        onSelect: logoutUser,
        disabled: signingOut,
        busy: signingOut,
        direct: true,
      }] : []}
      columns={phone ? 2 : 3}
      onClose={close}
    />
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
// unsaved edit. As wide as the widest dialogs: the page puts its panels side by
// side when it has the room.

function KioskProfileDialog({ onClose }: { onClose: () => void }) {
  const dirty = useRef(false)
  const close = useCallback(() => {
    if (dirty.current && !window.confirm('Discard your unsaved profile changes?')) return
    onClose()
  }, [onClose])
  return (
    <KioskDialog label="Profile" widest onClose={close}>
      <ProfilePage kiosk onDirtyChange={(d) => { dirty.current = d }} />
    </KioskDialog>
  )
}

function KioskDialog({ label, widest = false, onClose, children }: {
  label: string; widest?: boolean; onClose: () => void; children: ReactNode
}) {
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
        className={`fd-dialog relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 wide:rounded-[var(--fd-radius-panel)] wide:border ${widest ? 'wide:h-[88vh] wide:max-w-6xl' : 'wide:h-[85vh] wide:max-w-3xl'}`}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          title="Close"
          aria-label="Close"
          className="absolute right-2 top-2 z-10 rounded-full p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 wide:right-4 wide:top-4 wide:p-1.5"
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

function KioskWideDialog({ label, phone, widest = false, roomy = false, onClose, children }: {
  label: string; phone: boolean; widest?: boolean; roomy?: boolean; onClose: () => void; children: ReactNode
}) {
  const [fullscreen, setFullscreen] = useState(false)
  const host = useMemo(() => (phone ? null : { fullscreen, toggle: () => setFullscreen((v) => !v) }), [phone, fullscreen])
  const big = fullscreen && !phone
  return createPortal(
    <div className={`fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center ${big ? '' : 'wide:p-4'}`} onClick={phone ? undefined : onClose}>
      <div
        role="dialog"
        aria-label={label}
        className={`relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 pb-[env(safe-area-inset-bottom)] wide:pb-0 ${big ? ''
          : `fd-dialog wide:rounded-[var(--fd-radius-panel)] wide:border ${roomy ? 'wide:h-[90vh] wide:w-[90vw]' : `wide:h-[88vh] ${widest ? 'wide:max-w-6xl' : 'wide:max-w-5xl'}`}`}`}
        onClick={(e) => e.stopPropagation()}
      >
        <DialogFullscreen.Provider value={host}>{children}</DialogFullscreen.Provider>
      </div>
    </div>,
    document.body,
  )
}

const toolTile = 'relative flex min-w-0 flex-col items-center gap-1 rounded-[var(--fd-radius-control)] px-1 py-2 text-[11px] font-medium text-[var(--fd-ink-soft)] transition-colors hover:bg-[var(--fd-hover)] hover:text-[var(--fd-ink)]'
const toolIcon = 'h-[18px] w-[18px] text-[var(--fd-ink-faint)]'

/** The agent column's row of what the agents keep, as quiet tiles — an icon
 *  over its name: the File library, the Data library, the shared folders,
 *  and More — a menu with the rest (Crew, Swarm, Vatra, Basna, Council,
 *  Anonymize, Data rooms). The destinations people come for are the nav's. */
function ToolTiles({ onLibrary, onData, onFolders }: { onLibrary?: () => void; onData?: () => void; onFolders?: () => void }) {
  const [moreOpen, setMoreOpen] = useState(false)
  const moreRef = useRef<HTMLButtonElement>(null)
  // More is always there: Anonymize and Data rooms always are.
  const n = [onLibrary, onData, onFolders, true].filter(Boolean).length
  return (
    <div className="grid gap-0.5 px-2 pt-1" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
      {onLibrary && (
        <button onClick={onLibrary} className={toolTile} title="File library — every file your agents made, summarised and searchable">
          <LibraryBig className={toolIcon} />
          <span className="max-w-full truncate">Library</span>
        </button>
      )}
      {onData && (
        <button onClick={onData} className={toolTile} title="Data library — every table your agents keep, summarised and easy to read">
          <Database className={toolIcon} />
          <span className="max-w-full truncate">Data</span>
        </button>
      )}
      {onFolders && (
        <button onClick={onFolders} className={toolTile} title="Shared folders — the folders your agents work in together">
          <Folders className={toolIcon} />
          <span className="max-w-full truncate">Folders</span>
        </button>
      )}
      <button
        ref={moreRef}
        onClick={() => setMoreOpen(true)}
        className={toolTile}
        aria-haspopup="dialog"
        aria-label="More tools"
        title="More — Crew, Vatra, Basna, Council, Anonymize, Data rooms"
      >
        <LayoutGrid className={toolIcon} />
        <span className="max-w-full truncate">More</span>
      </button>
      {moreOpen && (
        <MoreToolsDialog
          onPicked={() => setMoreOpen(false)}
          onClose={() => { setMoreOpen(false); moreRef.current?.focus() }}
        />
      )}
    </div>
  )
}

/** More, from the agent column: the tools that are neither nav destinations
 *  nor tiles there. A phone, which has no tiles, also gets the libraries and
 *  the shared folders here. Choosing one closes the menu and opens it;
 *  closing it without a choice puts the focus back on More. */
function MoreToolsDialog({ onLibrary, onData, onFolders, onPicked, onClose }: {
  onLibrary?: () => void
  onData?: () => void
  onFolders?: () => void
  onPicked: () => void
  onClose: () => void
}) {
  const { isMobile } = useIsMobile()
  const crewAllowed = useCrewAllowed()
  const openCrew = useCrewStore((s) => s.openCrew)
  const canSwarm = useCanSwarm()
  const openSwarm = useCaptainSwarmStore((s) => s.openSwarm)
  const openAnonymize = useAnonymizeStore((s) => s.openAnonymize)
  const openRooms = useRoomsStore((s) => s.openRooms)
  const openTeamWork = useTeamWorkStore((s) => s.openTeamWork)
  const pick = (open: () => void) => () => { onPicked(); open() }

  const atWork: MenuSection['items'] = [
    ...(crewAllowed ? [{ icon: Ship, label: 'Crew', hint: 'Coding agents working for you', onSelect: pick(openCrew), tone: 'teal' as const }] : []),
    ...(canSwarm ? [{ icon: Target, label: 'Captain Swarm', hint: 'Many tries at a task, document or research; Captain checks each', onSelect: pick(openSwarm), tone: 'amber' as const }] : []),
  ]
  const sections: MenuSection[] = [
    ...(atWork.length ? [{ title: 'Your agents at work', items: atWork }] : []),
    {
      title: 'Agents as a team',
      items: [
        { icon: Flame, label: 'Vatra', hint: 'A lead splits one job; the team builds one result', onSelect: pick(() => openTeamWork('vatra')), tone: 'violet' },
        { icon: Network, label: 'Basna', hint: 'Several agents answer; the best of each merged into one', onSelect: pick(() => openTeamWork('basna')), tone: 'sky' },
        { icon: MessagesSquare, label: 'Council', hint: 'Your agents discuss a topic in rounds, then sum it up', onSelect: pick(() => openTeamWork('council')), tone: 'amber' },
      ],
    },
    {
      title: 'What they keep',
      items: [
        ...(onLibrary ? [{ icon: LibraryBig, label: 'File library', hint: 'Every file they made, easy to find', onSelect: pick(onLibrary), tone: 'violet' as const }] : []),
        ...(onData ? [{ icon: Database, label: 'Data library', hint: 'Every table they keep, easy to read', onSelect: pick(onData), tone: 'cyan' as const }] : []),
        ...(onFolders ? [{ icon: Folders, label: 'Shared folders', hint: 'The folders they work in together', onSelect: pick(onFolders), tone: 'sky' as const }] : []),
        { icon: EyeOff, label: 'Anonymize', hint: 'Names and numbers out of a file; the real ones kept for you', onSelect: pick(() => openAnonymize()), tone: 'fuchsia' },
        { icon: FolderLock, label: 'Data rooms', hint: 'A data room anonymized for Claude; results restored here', onSelect: pick(() => openRooms()), tone: 'fuchsia' },
      ],
    },
  ]

  return (
    <MenuSheet
      label="More tools"
      picture={<LayoutGrid className="h-6 w-6" />}
      title="More tools"
      subtitle="Pick one to open it"
      sections={sections}
      columns={isMobile ? 2 : 3}
      onClose={onClose}
    />
  )
}

// ── Left: the nav over the team of agents ────────────────────────────

function AgentsColumn({
  agents, activeAgentId, sessionInfo, switching, booting, onOpen, onPower, onLeave, onRefresh, onSpawn, onGroup, onOptions, onSettings, onInfo,
  nav, navCurrent, onLibrary, onData, onFolders, menuFor, locked = false, adminSwitch = false,
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
  /** Open the group sheet: a group to edit, or a new one. */
  onGroup: (req: GroupRequest) => void
  onOptions: (a: SimpleAgent) => void
  /** Kiosk only: the footer is one Settings button opening this dialog. */
  onSettings?: () => void
  /** Kiosk only: the Info button beside Settings. */
  onInfo?: () => void
  /** The outcome nav (Today, Inbox, Tasks…) and the one in front. */
  nav: NavEntry[]
  navCurrent: NavTarget | null
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
  const groups = useGroupStore((s) => s.groups)
  const collapsedGroups = useGroupStore((s) => s.collapsed)
  const railSections = useMemo(() => agentSections(agents, groups), [agents, groups])
  const canGroup = agents.length > 1 || groups.some(isFolder)
  // An agent being dragged: "New group" becomes where it can be dropped.
  const [agentDrag, setAgentDragState] = useState(false)
  const [overNew, setOverNew] = useState(false)
  const setAgentDrag = useCallback((on: boolean) => {
    setAgentDragState(on)
    if (!on) setOverNew(false)
  }, [])
  // The folded rail's More (the open column has its tiles).
  const [railMore, setRailMore] = useState(false)
  const railMoreRef = useRef<HTMLButtonElement>(null)

  // ── Collapsed rail ──
  if (!open) {
    const railGrouped = groups.some(isFolder)
    const railTile = (a: SimpleAgent) => {
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
          className={`relative flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-[11px] font-semibold transition-colors ${
            active
              ? 'bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)] ring-2 ring-[var(--fd-accent)]/40'
              : openable
                ? 'bg-[var(--fd-selected)] text-[var(--fd-ink)] hover:bg-[var(--fd-hover)]'
                : 'bg-[var(--fd-hover)] text-[var(--fd-ink-faint)] hover:bg-[var(--fd-selected)]'
          }`}
        >
          {switching.has(a.id) || isBooting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <AgentPicture agentId={a.id} name={a.name} size="md" dim={!openable && !active} />}
          <StateDot agent={a} className="absolute -bottom-0.5 -right-0.5 ring-2 ring-[var(--fd-shell)]" />
          {info?.busy && (
            <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-pulse rounded-full bg-amber-500 ring-2 ring-[var(--fd-shell)] dark:bg-amber-400" />
          )}
          {info?.unread && !active && !info.busy && (
            <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-sky-400 ring-2 ring-[var(--fd-shell)]" />
          )}
        </button>
      )
    }
    return (
      <aside className="flex w-16 shrink-0 flex-col">
        <div className="flex flex-col items-center gap-0.5 py-3">
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
        <div className="py-1">
          <SimpleNav entries={nav} current={navCurrent} rail />
        </div>
        <div className="mx-4 my-2 border-t fd-divider" />
        <div className="flex flex-1 flex-col items-center gap-1.5 overflow-y-auto py-2">
          {railSections.map((s) => {
            if (!s.group && s.agents.length === 0) return null
            const folded = railGrouped && collapsedGroups.includes(s.id)
            return (
              <Fragment key={s.id}>
                {railGrouped && <RailGroupMark section={s} sessionInfo={sessionInfo} />}
                {(folded ? s.agents.filter((a) => a.id === activeAgentId) : s.agents).map(railTile)}
              </Fragment>
            )
          })}
          <button onClick={onSpawn} className={`${iconBtn} mt-1`} title={locked ? 'New agent from an archetype' : 'Spawn a new agent'}>
            <Plus className="h-4 w-4" />
          </button>
        </div>
        <div className="flex flex-col items-center gap-0.5 py-2">
          {onLibrary && (
            <button onClick={onLibrary} title="File library" aria-label="File library" className="fd-nav-item fd-nav-rail">
              <LibraryBig className="fd-nav-icon" />
            </button>
          )}
          {onData && (
            <button onClick={onData} title="Data library" aria-label="Data library" className="fd-nav-item fd-nav-rail">
              <Database className="fd-nav-icon" />
            </button>
          )}
          {onFolders && (
            <button onClick={onFolders} title="Shared folders" aria-label="Shared folders" className="fd-nav-item fd-nav-rail">
              <Folders className="fd-nav-icon" />
            </button>
          )}
          <button ref={railMoreRef} onClick={() => setRailMore(true)} title="More tools" aria-label="More tools" aria-haspopup="dialog" className="fd-nav-item fd-nav-rail">
            <LayoutGrid className="fd-nav-icon" />
          </button>
          {railMore && (
            <MoreToolsDialog onPicked={() => setRailMore(false)} onClose={() => { setRailMore(false); railMoreRef.current?.focus() }} />
          )}
        </div>
        {onSettings ? (
          <div className="flex flex-col items-center gap-0.5 pb-3 pt-1">
            {onInfo && (
              <button onClick={onInfo} title="Info" aria-label="Info" className="fd-nav-item fd-nav-rail">
                <Info className="fd-nav-icon" />
              </button>
            )}
            <button onClick={onSettings} title="Settings" aria-label="Settings" className="fd-nav-item fd-nav-rail">
              <Settings className="fd-nav-icon" />
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-0.5 pb-3 pt-1">
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
      className="relative flex shrink-0 flex-col"
      style={{ width: width.size }}
    >
      {/* Header */}
      <div className="flex h-16 items-center justify-between pl-4 pr-2">
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

      {/* What people come for, in plain words (the Line is the Inbox). */}
      <div className="px-3">
        <SimpleNav entries={nav} current={navCurrent} />
      </div>

      {/* The team: the agents, the two ways to add to it, and the search */}
      <div className="@container space-y-2 px-4 pb-2 pt-3">
        <div className="flex items-center gap-2">
          <h2 className="fd-section-label">Team</h2>
          <span className="fd-count">{agents.length}</span>
        </div>
        {/* While an agent is dragged the row is one drop target, the same height: nothing below moves. */}
        {agentDrag && canGroup ? (
          <button
            onClick={() => onGroup({})}
            aria-label="Drop here for a new group"
            onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; setOverNew(true) }}
            onDragLeave={() => setOverNew(false)}
            onDrop={(e) => {
              e.preventDefault()
              setOverNew(false)
              const id = e.dataTransfer.getData(AGENT_DRAG_TYPE)
              if (id) onGroup({ agentIds: [id] })
            }}
            className={`flex h-9 w-full items-center justify-center gap-1.5 rounded-full border border-dashed px-2 text-xs font-medium transition-colors ${
              overNew ? 'border-violet-400 bg-violet-500/10 text-zinc-100' : 'border-zinc-600 text-zinc-300'}`}
          >
            <FolderPlus className="h-4 w-4 shrink-0" />
            <span className="truncate @max-[14rem]:hidden">Drop here for a new group</span>
            <span className="hidden truncate @max-[14rem]:inline">Drop for a new group</span>
          </button>
        ) : (
          <div className="flex gap-1.5">
            <button
              onClick={onSpawn}
              aria-label="New agent"
              title={locked ? 'New agent from an archetype' : 'Spawn a new agent'}
              className="fd-btn-primary h-9 min-w-0 flex-1 px-3 text-xs"
            >
              <Plus className="h-4 w-4 shrink-0" />
              {/* A narrow column keeps the word that matters, not "New …". */}
              <span className="truncate @max-[14rem]:hidden">New agent</span>
              <span className="hidden truncate @max-[14rem]:inline">Agent</span>
            </button>
            {canGroup && (
              <button
                onClick={() => onGroup({})}
                aria-label="New group"
                title="New group"
                className="fd-btn-quiet h-9 w-9 shrink-0"
              >
                <FolderPlus className="h-4 w-4 shrink-0 text-[var(--fd-ink-faint)]" />
              </button>
            )}
          </div>
        )}
        {/* Search (only once the list is long enough to need it) */}
        {searchable && (
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Escape' && search) { e.stopPropagation(); setSearch('') } }}
              placeholder="Search agents…"
              aria-label="Search agents"
              className="h-9 w-full rounded-full border border-[var(--fd-line)] bg-[var(--fd-surface)] pl-8 pr-7 text-xs text-zinc-200 placeholder-zinc-500 transition-colors focus:border-violet-500/60 focus:outline-none"
            />
            {search && (
              <button
                onClick={() => setSearch('')}
                aria-label="Clear the search"
                title="Clear the search"
                className="absolute right-1 top-1/2 -translate-y-1/2 rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        )}
      </div>

      {/* Agents */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {agents.length === 0 && (
          <div className="mx-1 mt-2 rounded-[var(--fd-radius-card)] border border-dashed border-[var(--fd-line)] px-4 py-5 text-center">
            <p className="text-sm font-semibold text-[var(--fd-ink)]">Your team is empty</p>
            <p className="mt-1 text-xs leading-relaxed text-[var(--fd-ink-soft)]">Add an agent for a job you'd hand to a helper — inbox, research, reports.</p>
            <button onClick={onSpawn} className="mt-3 text-xs font-semibold text-[var(--fd-accent-text)] hover:underline">
              Add your first agent
            </button>
          </div>
        )}
        <AgentSections
          agents={agents}
          query={searchable ? search : ''}
          activeAgentId={activeAgentId}
          sessionInfo={sessionInfo}
          onEdit={(id) => onGroup({ id })}
          onAgentDrag={setAgentDrag}
          renderRow={(a: SimpleAgent, { drag, pin }: RowParts) => {
            const info = sessionInfo.get(a.id)
            const active = a.id === activeAgentId
            const busy = !!info?.busy
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
            const unread = !!info?.unread && !active && !busy
            const subtitle = isBooting ? 'Starting…'
              : busy ? 'Working…'
                : shared && a.state === 'running' ? `Shared by ${a.ownerName || 'another user'}`
                  : a.state === 'running' && a.reachable && a.description ? a.description : rowState(a)
            const pinButton = pin && (
              <button
                onClick={pin.onToggle}
                aria-pressed={pin.pinned}
                aria-label={`${pin.pinned ? 'Unpin' : 'Pin'} ${a.name}`}
                title={pin.pinned ? `Back among the others in ${pin.groupName}` : `Pin to the top of ${pin.groupName}`}
                className={`${rowBtn} ${pin.pinned ? 'text-violet-600 hover:bg-zinc-700/60 dark:text-violet-400' : 'text-zinc-500 hover:bg-zinc-700/60 hover:text-zinc-200'}`}
              >
                <Pin className={`h-3.5 w-3.5 ${pin.pinned ? 'fill-current' : ''}`} />
              </button>
            )
            return (
              <li
                key={a.id}
                {...drag}
                className={`group/row group relative flex items-center gap-0.5 rounded-[var(--fd-radius-control)] pr-1 transition-colors ${active ? 'bg-[var(--fd-selected)]' : 'hover:bg-[var(--fd-hover)]'}`}
              >
                <button
                  onClick={() => onOpen(a)}
                  disabled={!openable}
                  title={openable
                    ? a.asleep ? `Chat with ${a.name} — asleep, it wakes when you open it` : `Chat with ${a.name}`
                    : `${a.name} — ${isBooting ? 'Starting…' : stateLabel(a)}`}
                  className="flex min-w-0 flex-1 items-center gap-2.5 rounded-[var(--fd-radius-control)] py-1.5 pl-1.5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40"
                >
                  {/* Its picture (its initials unless one was picked), its state on the corner, working or a new reply on top. */}
                  <span className={`relative flex h-8 w-8 shrink-0 items-center justify-center rounded-xl text-[11px] font-semibold transition-colors ${
                    active ? 'bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)]'
                      : powered ? 'bg-[var(--fd-selected)] text-[var(--fd-ink)]' : 'bg-[var(--fd-hover)] text-[var(--fd-ink-faint)]'}`}
                  >
                    {working || isBooting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <AgentPicture agentId={a.id} name={a.name} dim={!powered && !active} />}
                    <StateDot agent={a} className="absolute -bottom-0.5 -right-0.5 ring-2 ring-[var(--fd-shell)]" />
                    {busy && <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-pulse rounded-full bg-amber-500 ring-2 ring-[var(--fd-shell)] dark:bg-amber-400" title="Working" />}
                    {unread && <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-sky-400 ring-2 ring-[var(--fd-shell)]" title="New reply" />}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-1.5">
                      <span className={`min-w-0 truncate text-[13px] font-medium ${active || unread ? 'text-zinc-100' : openable ? 'text-zinc-200' : 'text-zinc-400'}`}>{a.name}</span>
                      {shared && (
                        <span
                          className="flex shrink-0 items-center gap-0.5 rounded border border-sky-500/25 bg-sky-500/15 px-1 py-0.5 text-[9px] leading-none text-sky-700 dark:text-sky-300"
                          title={`Shared by ${a.ownerName || 'another user'}`}
                        >
                          <Users className="h-2.5 w-2.5" />shared
                        </span>
                      )}
                      {shareKeyOf(a) && <NetworkBadge shareKey={shareKeyOf(a)} />}
                    </span>
                    <span className={`block truncate text-[11px] ${busy ? 'text-amber-700 dark:text-amber-400' : 'text-zinc-500'}`}>{subtitle}</span>
                  </span>
                </button>
                {/* What else there is to do with it: out of the way (no width at all)
                    until the row is pointed at or focused; always there on a touch screen. */}
                <div className="flex max-w-0 shrink-0 items-center overflow-hidden transition-[max-width] duration-150 group-focus-within/row:max-w-32 group-hover/row:max-w-32 motion-reduce:transition-none pointer-coarse:max-w-none">
                  {!locked && !shared && (
                    <button onClick={() => onOptions(a)} title={`Options — ${a.name}`} aria-label={`Options — ${a.name}`} className={`${rowBtn} text-zinc-500 hover:bg-zinc-700/60 hover:text-zinc-200`}>
                      <Settings className="h-3.5 w-3.5" />
                    </button>
                  )}
                  {menu.length > 0 && <AgentMenu agentId={a.id} agentName={a.name} items={menu} subtitle={menuSubtitle(a)} status={<StateDot agent={a} />} shown />}
                  {shared && (
                    <button
                      onClick={() => onLeave(a)}
                      title={`Leave ${a.name}`}
                      aria-label={`Leave ${a.name}`}
                      className={`${rowBtn} text-zinc-500 hover:bg-zinc-700/60 hover:text-red-600 dark:hover:text-red-400`}
                    >
                      <LogOut className="h-3.5 w-3.5" />
                    </button>
                  )}
                  {/* Last, where it stays once pinned: it doesn't jump when clicked. */}
                  {pin && !pin.pinned && pinButton}
                </div>
                {/* Pinned: the pin stays, so it says so. */}
                {pin?.pinned && pinButton}
                {canPower && (
                  <button
                    onClick={() => onPower(a, !powered)}
                    disabled={working}
                    role="switch"
                    aria-checked={powered}
                    aria-label={`${a.name} power`}
                    title={working ? (powered ? 'Powering off…' : 'Powering on…') : powered ? `On — power off ${a.name}` : `Off — power on ${a.name}`}
                    className={`${rowBtn} disabled:opacity-60 ${
                      powered
                        ? 'text-emerald-600 hover:bg-red-500/10 hover:text-red-600 dark:text-emerald-400 dark:hover:text-red-400'
                        : 'text-zinc-600 hover:bg-emerald-500/10 hover:text-emerald-600 group-hover/row:text-zinc-400 dark:hover:text-emerald-400'
                    }`}
                  >
                    {working ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Power className="h-3.5 w-3.5" />}
                  </button>
                )}
                {canRecheck && (
                  <button
                    onClick={() => onPower(a, true)}
                    disabled={working}
                    title="Check again"
                    aria-label={`Check ${a.name} again`}
                    className={`${rowBtn} text-zinc-500 hover:bg-zinc-700/60 hover:text-emerald-600 disabled:opacity-60 dark:hover:text-emerald-400`}
                  >
                    {working ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                  </button>
                )}
              </li>
            )
          }}
        />
      </div>

      <ToolTiles onLibrary={onLibrary} onData={onData} onFolders={onFolders} />

      {/* Footer — the kiosk's is one Settings button (KioskSettingsDialog). */}
      {onSettings ? (
        <div className="flex items-center gap-1 px-3 pb-3 pt-1">
          <button onClick={onSettings} className="fd-nav-item min-w-0 flex-1">
            <Settings className="fd-nav-icon" />
            Settings
            {authEnabled && authUser && (
              <span
                title={authUser.display_name || authUser.email}
                className="ml-auto flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--fd-accent-soft)] text-[10px] font-semibold text-[var(--fd-accent-text)]"
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
              className="fd-nav-item fd-nav-rail shrink-0"
            >
              <Info className="fd-nav-icon" />
            </button>
          )}
          {adminSwitch && (
            <button
              onClick={leaveKioskView}
              title="Switch to admin view — models, archetypes, connections, settings"
              aria-label="Switch to admin view"
              className="fd-nav-item fd-nav-rail shrink-0 !text-[var(--fd-accent-text)]"
            >
              <ShieldCheck className="h-[18px] w-[18px]" />
            </button>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-between gap-2 px-3 pb-3 pt-1">
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

function EmptyChat({ hasAgents, onSpawn, locked = false, onGuide, startName, onStart }: {
  hasAgents: boolean
  onSpawn: () => void
  locked?: boolean
  /** Kiosk only: open the Getting started guide. */
  onGuide?: () => void
  /** With agents but none on: the first one that can be switched on, and how. */
  startName?: string
  onStart?: () => void
}) {
  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="w-full max-w-md text-center">
        <div className="fd-empty-mark mx-auto mb-6">
          <MessagesSquare className="h-6 w-6" />
        </div>
        <h2 className="fd-empty-title mb-2 text-2xl">
          {hasAgents ? 'Your team is taking a break' : 'Let\'s set up your first agent'}
        </h2>
        <p className="fd-empty-text text-[15px] leading-relaxed">
          {hasAgents
            ? 'None of your agents is on right now. Switch one on in Team on the left and say hello. What it saves for you lands in Library, Data and Folders, bottom left.'
            : locked
              ? 'An agent is a helper for one job — your inbox, research, a weekly report. Pick one from a ready-made role and it\'s ready in a minute.'
              : 'An agent is a helper for one job — your inbox, research, a weekly report. Spawn one and start chatting.'}
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          {hasAgents && onStart && startName ? (
            <button onClick={onStart} className="fd-btn-primary h-11 px-6 text-sm">
              <Power className="h-4 w-4" />
              Switch on {startName}
            </button>
          ) : !hasAgents ? (
            <button onClick={onSpawn} className="fd-btn-primary h-11 px-6 text-sm">
              <Plus className="h-4 w-4" />
              {locked ? 'New agent' : 'Spawn an agent'}
            </button>
          ) : null}
          {onGuide && (
            <button onClick={onGuide} className="fd-btn-quiet h-11 px-5 text-sm">
              <Compass className="h-4 w-4" />
              Getting started
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// One agent's files over its datastore — the "Files & data" dialog's body. A shared agent: on a process agent, on a deck
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
// An agent's "⋯" → Files & data (and the library's "Everything it keeps"):
// its files and datastore in a dialog. They read the agent itself, so they
// hold what the File and Data libraries don't: the agents they don't index (a
// Docker or local one, one Flight Deck runs itself, one shared with the
// user) and, for one they do, the files they leave out.

function FilesDataDialog({ agentId, agentName, onClose }: { agentId: string; agentName: string; onClose: () => void }) {
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
  onOpen, onPower, onLeave, onSpawn, onGroup, onSettings, onInfo, nav,
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
  onGroup: (req: GroupRequest) => void
  onSettings?: () => void
  onInfo?: () => void
  /** The outcome nav: tiles at the top of the list (the list itself is Today). */
  nav: NavEntry[]
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
  const groups = useGroupStore((s) => s.groups)
  const [search, setSearch] = useState('')
  const [filesOpen, setFilesOpen] = useState(false)
  // More, from the tiles at the top of the list.
  const [phoneMore, setPhoneMore] = useState(false)
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
  const canGroup = agents.length > 1 || groups.some(isFolder)

  const phoneBtn = 'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-zinc-400 transition-colors active:bg-zinc-800 hover:bg-zinc-800 hover:text-zinc-100'
  // The agent list's header: the bell, Info and Settings (the tools are tiles below it).
  const toolBtn = 'flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-[var(--fd-ink-soft)] transition-colors active:bg-[var(--fd-selected)] hover:bg-[var(--fd-hover)] hover:text-[var(--fd-ink)]'

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
          {a && menu.length > 0 && <AgentMenu agentId={a.id} agentName={a.name} items={menu} subtitle={menuSubtitle(a)} status={<StateDot agent={a} />} phone />}
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

  // The tiles at the top of the list: the nav's destinations (the list is
  // Today and Agents), then what the agents keep and More.
  const tiles: { key: string; label: string; hint: string; icon: NavEntry['icon']; count?: number; hot?: boolean; dot?: boolean; onSelect: () => void }[] = [
    ...nav.filter((e) => e.id !== 'today' && e.id !== 'agents'),
    ...(onLibrary ? [{ key: 'library', label: 'Library', hint: 'File library', icon: LibraryBig, onSelect: onLibrary }] : []),
    ...(onData ? [{ key: 'data', label: 'Data', hint: 'Data library', icon: Database, onSelect: onData }] : []),
    ...(onFolders ? [{ key: 'folders', label: 'Folders', hint: 'Shared folders', icon: Folders, onSelect: onFolders }] : []),
    { key: 'more', label: 'More', hint: 'More tools', icon: LayoutGrid, onSelect: () => setPhoneMore(true) },
  ].map((t) => ('id' in t ? { ...t, key: t.id } : t))

  return (
    <div className="fd-shell flex h-[100dvh] flex-col overflow-hidden">
      <header className="flex h-14 shrink-0 items-center gap-0.5 pl-4 pr-1.5">
        <div className="@container min-w-0 flex-1"><BrandMark connected={wsConnected} caption={adminSwitch ? 'User view' : undefined} compact /></div>
        {adminSwitch && (
          <button onClick={leaveKioskView} className={toolBtn} aria-label="Switch to admin view" title="Switch to admin view">
            <ShieldCheck className="h-5 w-5 text-[var(--fd-accent-text)]" />
          </button>
        )}
        <NotificationBell />
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

      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 pb-4 pt-2">
        <nav aria-label="Main" className="mb-6 grid grid-cols-3 gap-2">
          {tiles.map((t) => {
            const Icon = t.icon
            const count = t.count && t.count > 0 ? (t.count > 99 ? '99+' : String(t.count)) : ''
            return (
              <button key={t.key} onClick={t.onSelect} title={t.hint} aria-label={count && t.hot ? `${t.label} — ${t.hint}` : t.label}
                className="fd-card relative flex h-[76px] flex-col items-center justify-center gap-1.5 px-1 text-[13px] font-medium text-[var(--fd-ink)] shadow-none transition-colors active:bg-[var(--fd-selected)]">
                <Icon className="h-5 w-5 text-[var(--fd-ink-soft)]" />
                <span className="max-w-full truncate">{t.label}</span>
                {t.dot ? <span className="absolute right-2.5 top-2.5 h-2 w-2 animate-pulse rounded-full bg-[var(--fd-accent)]" />
                  : count ? <span className={`fd-count absolute right-1.5 top-1.5 ${t.hot ? 'fd-count-hot' : ''}`}>{count}</span> : null}
              </button>
            )
          })}
        </nav>
        {phoneMore && <MoreToolsDialog onPicked={() => setPhoneMore(false)} onClose={() => setPhoneMore(false)} />}
        <div className="mb-3 flex items-center justify-between gap-2 px-1">
          <div className="flex min-w-0 items-center gap-2">
            <h2 className="text-lg font-bold tracking-tight text-[var(--fd-ink)]">Team</h2>
            <span className="fd-count">{agents.length}</span>
          </div>
          {canGroup && (
            <button
              onClick={() => onGroup({})}
              className="fd-btn-quiet h-10 shrink-0 px-4 text-sm"
            >
              <FolderPlus className="h-[18px] w-[18px] text-[var(--fd-ink-faint)]" />
              New group
            </button>
          )}
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
          <div className="fd-card mt-2 p-6 text-center shadow-none">
            <div className="fd-empty-mark mx-auto mb-4">
              <MessagesSquare className="h-6 w-6" />
            </div>
            <h2 className="fd-empty-title text-lg">Your team is empty</h2>
            <p className="fd-empty-text mt-1 text-sm leading-relaxed">
              {locked
                ? 'An agent is a helper for one job — your inbox, research, a weekly report. Tap New agent below to pick a ready-made role.'
                : 'An agent is a helper for one job — your inbox, research, a weekly report. Tap New agent below to spawn one.'}
            </p>
          </div>
        )}
        <AgentSections
          phone
          agents={agents}
          query={searchable ? search : ''}
          activeAgentId={activeAgentId}
          sessionInfo={sessionInfo}
          onEdit={(id) => onGroup({ id })}
          renderRow={(a: SimpleAgent, { pin }: RowParts) => {
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
                className={`flex items-center gap-1 rounded-[var(--fd-radius-card)] border pr-1 transition-colors ${
                  a.id === activeAgentId ? 'border-violet-500/40 bg-[var(--fd-accent-soft)]' : busy ? 'border-amber-500/30 bg-amber-500/10' : 'border-[var(--fd-line)] bg-[var(--fd-surface-raised)]'
                }`}
              >
                <button
                  onClick={() => onOpen(a)}
                  disabled={!openable}
                  className="flex min-w-0 flex-1 items-center gap-3 py-3.5 pl-3.5 text-left disabled:opacity-60"
                >
                  <span className="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--fd-selected)] text-xs font-semibold text-[var(--fd-ink)]">
                    {working || isBooting ? <Loader2 className="h-4 w-4 animate-spin" /> : <AgentPicture agentId={a.id} name={a.name} size="md" />}
                    <StateDot agent={a} className="absolute -bottom-0.5 -right-0.5 ring-2 ring-[var(--fd-shell)]" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-1.5">
                      <span className="truncate text-[15px] font-medium text-zinc-100">{a.name}</span>
                      {pin?.pinned && <Pin className="h-3 w-3 shrink-0 fill-current text-violet-600 dark:text-violet-400" aria-label="Pinned" />}
                      {busy && <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-amber-500 dark:bg-amber-400" title="Working" />}
                      {info?.unread && !busy && <span className="h-2 w-2 shrink-0 rounded-full bg-sky-400" title="New reply" />}
                      {shareKeyOf(a) && <NetworkBadge shareKey={shareKeyOf(a)} />}
                    </span>
                    <span className="block truncate text-xs text-zinc-500">
                      {isBooting ? 'Starting…'
                        : busy ? 'Working…'
                          : shared && a.state === 'running' ? `Shared by ${a.ownerName || 'another user'}`
                            : a.state === 'running' && a.reachable && a.description ? a.description : rowState(a)}
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
                {menu.length > 0 && <AgentMenu agentId={a.id} agentName={a.name} items={menu} subtitle={menuSubtitle(a)} status={<StateDot agent={a} />} phone />}
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
          }}
        />
      </div>

      <div className="shrink-0 px-4 pt-2 pb-[max(1rem,env(safe-area-inset-bottom))]">
        <button
          onClick={onSpawn}
          className="fd-btn-primary h-12 w-full text-[15px]"
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
