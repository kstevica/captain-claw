import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Gamepad2,
  Loader2,
  Play,
  Pause,
  RefreshCw,
  Trash2,
  StepForward,
  Send,
  CheckCircle2,
  Wand2,
  ChevronRight,
  ChevronLeft,
  Zap,
  Layers,
  Info,
  RotateCcw,
  Users,
  Brain,
  MessageSquare,
  Dices,
  ArrowRightLeft,
  Maximize2,
  Minimize2,
} from 'lucide-react'
import { useContainerStore } from '../stores/containerStore'
import { useProcessStore } from '../stores/processStore'
import { useLocalAgentStore } from '../stores/localAgentStore'
import { useAuthStore } from '../stores/authStore'

// ── Types ──────────────────────────────────────────────────────────

interface AgentTarget {
  id: string
  name: string
  kind: 'docker' | 'process' | 'local'
  host: string
  port: number
  auth: string
}

interface WorldSummary {
  id: string
  title: string
  summary: string
  characters: { id: string; name: string; glyph: string }[]
}

interface GameCharacter {
  id: string
  name: string
  glyph: string
  description?: string
  objective?: string
  start_room?: string
}

interface GameRoom {
  id: string
  name: string
}

interface GameSeat {
  character: string
  kind: string
  cognitive_mode?: string
  remote_agent?: { host: string; port: number; name?: string }
}

interface GameSummary {
  game_id: string
  world_id: string
  title: string
  summary: string
  tick: number
  terminal: boolean
  win: boolean
  seed: number
  characters: GameCharacter[]
  rooms?: GameRoom[]
  seats: GameSeat[]
}

interface AgentThought {
  tick: number
  reasoning: string
  action: string
}

interface ConversationEntry {
  tick: number
  actor: string
  actor_name: string
  text: string
  room: string
  room_name: string
  audience: string[]
  kind?: 'say' | 'talk'
  target?: string
}

interface GameView {
  tick: number
  terminal: boolean
  win: boolean
  character: { id: string; name: string; glyph: string; objective: string }
  current_room?: {
    id: string
    name: string
    description: string
    ascii_tile?: string[]
    entities?: { id: string; name: string; glyph: string; takeable: boolean; examinable?: boolean; examined?: boolean }[]
    others_here?: { id: string; name: string; glyph: string }[]
    exits?: Record<string, string>
    locked_exits?: Record<string, boolean>
  }
  inventory?: { id: string; name: string }[]
  heard?: { actor: string; actor_name: string; text: string; kind?: 'say' | 'talk' }[]
  action_results?: { kind: string; text: string; entity_id?: string; entity_name?: string }[]
  discoveries?: { tick: number; kind: string; text: string; entity_id?: string; entity_name?: string; item_id?: string; target_id?: string }[]
}

interface FullGame extends GameSummary {
  views: Record<string, GameView>
  rendered: Record<string, string>
  agent_thoughts?: Record<string, AgentThought[]>
  conversation_log?: ConversationEntry[]
  game_images?: Record<string, boolean>
}

interface ReplayResult {
  ok: boolean
  ticks_replayed: number
  matches_live: boolean
  final_tick: number
}

type MainView = 'landing' | 'play' | 'generate' | 'assign-seats'
type SeatKind = 'scripted' | 'human' | 'agent'

const COGNITIVE_MODES = [
  { id: 'neutra',     label: 'Neutra',     color: 'bg-zinc-500',    short: 'balanced' },
  { id: 'ionian',     label: 'Ionian',     color: 'bg-amber-400',   short: 'convergent solver' },
  { id: 'dorian',     label: 'Dorian',     color: 'bg-teal-400',    short: 'empathetic pragmatist' },
  { id: 'phrygian',   label: 'Phrygian',   color: 'bg-red-400',     short: 'adversarial analyst' },
  { id: 'lydian',     label: 'Lydian',     color: 'bg-violet-400',  short: 'creative explorer' },
  { id: 'mixolydian', label: 'Mixolydian', color: 'bg-orange-400',  short: 'iterative builder' },
  { id: 'aeolian',    label: 'Aeolian',    color: 'bg-blue-400',    short: 'deep researcher' },
  { id: 'locrian',    label: 'Locrian',    color: 'bg-fuchsia-400', short: 'deconstructionist' },
] as const

// ── Helpers ────────────────────────────────────────────────────────

function fdAuthHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = {}
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

function gamesUrl(sub?: string): string {
  return sub ? `/fd/games/${sub}` : '/fd/games'
}

function agentProvider(target: AgentTarget): { host: string; port: number; auth: string } {
  return { host: target.host, port: target.port, auth: target.auth }
}

async function fetchJSON<T>(url: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...fdAuthHeaders(), ...(init.headers || {}) },
    credentials: 'include',
  })
  if (!res.ok) {
    const text = await res.text()
    // Try to extract a clean error message from JSON responses
    try {
      const json = JSON.parse(text)
      const msg = json?.error || json?.detail?.error || json?.detail || text
      throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
    } catch (e) {
      if (e instanceof Error && !e.message.startsWith('HTTP')) throw e
      throw new Error(`HTTP ${res.status}: ${text}`)
    }
  }
  return res.json() as Promise<T>
}

// ── Constants ──────────────────────────────────────────────────────

const SIZE_INFO: Record<string, { rooms: string; entities: string; npcs: string; desc: string }> = {
  small:  { rooms: '4-6',     entities: '3-5',    npcs: '0-1', desc: 'Quick session, tight map. Good for testing ideas or short play.' },
  medium: { rooms: '8-12',    entities: '6-10',   npcs: '1-3', desc: 'Standard adventure with room to explore and multiple paths.' },
  large:  { rooms: '15-20',   entities: '10-15',  npcs: '3-5', desc: 'Sprawling world with branching routes, many items, and NPCs.' },
  xl:     { rooms: '35-40',   entities: '25-30',  npcs: '2-5', desc: 'Extended world with dense rooms and rich puzzle chains.' },
  huge:   { rooms: '70-80',   entities: '50-60',  npcs: '2-5', desc: 'Massive world. Multi-pass generation. Expect longer creation time.' },
  epic:   { rooms: '120-150', entities: '80-100', npcs: '2-5', desc: 'The largest possible world. Many batched LLM calls. Be patient.' },
}

const WIZARD_STEPS = ['Story', 'World', 'Generate'] as const

// Shared Tailwind fragments
const inputCls = 'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500 focus:outline-none'
const selectCls = 'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500 focus:outline-none'
const labelCls = 'block text-xs font-medium text-zinc-400 mb-1'
const hintCls = 'mt-1 text-[11px] text-zinc-600 leading-snug'

// ── Page ────────────────────────────────────────────────────────────

export function GamesPage() {
  const containers = useContainerStore((s) => s.containers)
  const processes = useProcessStore((s) => s.processes)
  const localAgents = useLocalAgentStore((s) => s.agents)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [worlds, setWorlds] = useState<WorldSummary[]>([])
  const [games, setGames] = useState<GameSummary[]>([])
  const [activeGame, setActiveGame] = useState<FullGame | null>(null)
  const [mainView, setMainView] = useState<MainView>('landing')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [replay, setReplay] = useState<ReplayResult | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [focusedChar, setFocusedChar] = useState<string | null>(null)

  // Per-character intent draft inputs
  const [intentDrafts, setIntentDrafts] = useState<Record<string, string>>({})
  // Agent panel tab: 'thoughts' | 'memory'
  const [agentTab, setAgentTab] = useState<Record<string, string>>({})
  // Seat assignments for demo world create
  const [seatPick, setSeatPick] = useState<Record<string, SeatKind>>({})
  const [modePick, setModePick] = useState<Record<string, string>>({})
  const [agentTargetPick, setAgentTargetPick] = useState<Record<string, string>>({})
  const [pickedWorld, setPickedWorld] = useState<string>('')

  // ── Generator form state ──
  const [wizardStep, setWizardStep] = useState(0) // 0=Story, 1=World, 2=Generate
  const [genTitle, setGenTitle] = useState('')
  const [genGoal, setGenGoal] = useState('')
  const [genDescription, setGenDescription] = useState('')
  const [genGenre, setGenGenre] = useState('exploration')
  const [genTone, setGenTone] = useState('neutral')
  const [genSize, setGenSize] = useState<'small' | 'medium' | 'large' | 'xl' | 'huge' | 'epic'>('small')
  const [genSeats, setGenSeats] = useState(2)
  const [genConstraints, setGenConstraints] = useState('')
  const [genMode, setGenMode] = useState<'fast' | 'pipeline'>('fast')
  const [genDefaultSeat, setGenDefaultSeat] = useState<SeatKind>('scripted')
  const [generating, setGenerating] = useState(false)
  const [randomizing, setRandomizing] = useState(false)

  // Post-generation seat assignment
  const [pendingGame, setPendingGame] = useState<FullGame | null>(null)
  const [pendingSeatPick, setPendingSeatPick] = useState<Record<string, SeatKind>>({})
  const [pendingModePick, setPendingModePick] = useState<Record<string, string>>({})
  const [pendingAgentTarget, setPendingAgentTarget] = useState<Record<string, string>>({})

  // Tick-in-progress (LLM working)
  const [ticking, setTicking] = useState(false)

  // Image provider selection
  const [imageProviders, setImageProviders] = useState<{ id: string; label: string; kind: string; available: boolean; active: boolean }[]>([])
  const [activeImageProvider, setActiveImageProvider] = useState<string>('')

  // Image zoom modal
  const [zoomImage, setZoomImage] = useState<{ src: string; alt: string } | null>(null)

  // Item action menu: "charId:entityId" -> open
  const [itemMenu, setItemMenu] = useState<string | null>(null)

  // Character interaction menu: "charId:targetId" -> open
  const [charMenu, setCharMenu] = useState<string | null>(null)

  // Transfer modal
  const [transferring, setTransferring] = useState(false)
  const [showTransfer, setShowTransfer] = useState(false)
  const [transferTarget, setTransferTarget] = useState<string | null>(null)

  // Auto-tick
  const [playing, setPlaying] = useState(false)
  const [playInterval, setPlayInterval] = useState(1500)
  const playingRef = useRef(false)
  useEffect(() => { playingRef.current = playing }, [playing])

  const targets: AgentTarget[] = useMemo(() => {
    const out: AgentTarget[] = []
    for (const c of containers) {
      if (c.web_port) {
        out.push({
          id: c.id, name: c.agent_name || c.name, kind: 'docker',
          host: 'localhost', port: c.web_port, auth: c.web_auth || '',
        })
      }
    }
    for (const p of processes) {
      if (p.web_port) {
        out.push({
          id: p.slug, name: p.name, kind: 'process',
          host: 'localhost', port: p.web_port, auth: p.web_auth || '',
        })
      }
    }
    for (const a of localAgents) {
      out.push({
        id: a.id, name: a.name, kind: 'local',
        host: a.host, port: a.port, auth: a.authToken || '',
      })
    }
    return out
  }, [containers, processes, localAgents])

  const selected = useMemo(
    () => targets.find((t) => t.id === selectedId) || targets[0] || null,
    [targets, selectedId]
  )

  useEffect(() => {
    if (!selectedId && targets.length > 0) setSelectedId(targets[0].id)
  }, [targets, selectedId])

  // ── Loaders ──

  const loadWorldsAndGames = useCallback(async (_target?: AgentTarget) => {
    setLoading(true)
    setError(null)
    try {
      const [worldsRes, gamesRes, imgRes] = await Promise.all([
        fetchJSON<{ worlds: WorldSummary[] }>(gamesUrl('worlds')),
        fetchJSON<{ games: GameSummary[] }>(gamesUrl()),
        fetchJSON<{ providers: typeof imageProviders; active: string }>(gamesUrl('image-providers')),
      ])
      setWorlds(worldsRes.worlds || [])
      setGames(gamesRes.games || [])
      if (imgRes.providers) {
        setImageProviders(imgRes.providers)
        setActiveImageProvider(imgRes.active || '')
      }
      if (!pickedWorld && worldsRes.worlds.length > 0) {
        setPickedWorld(worldsRes.worlds[0].id)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [pickedWorld])

  useEffect(() => {
    if (selected) void loadWorldsAndGames()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.id])

  // Trigger on-demand image generation for missing images
  const ensureGameImages = useCallback((game: FullGame) => {
    if (!game.game_images) return
    const needed = new Set<string>()
    for (const cid of Object.keys(game.views || {})) {
      const view = game.views[cid] as GameView | undefined
      const rid = view?.current_room?.id
      if (rid && !game.game_images[`room:${rid}`]) needed.add(`room:${rid}`)
      // Character portrait
      if (!game.game_images[`char:${cid}`]) needed.add(`char:${cid}`)
      // Entities in current room
      for (const e of view?.current_room?.entities || []) {
        if (!game.game_images[`entity:${e.id}`]) needed.add(`entity:${e.id}`)
      }
      // Inventory items
      for (const e of view?.inventory || []) {
        if (!game.game_images[`entity:${e.id}`]) needed.add(`entity:${e.id}`)
      }
    }
    for (const key of Array.from(needed)) {
      const [kind, objId] = key.split(':')
      fetchJSON<{ ok: boolean }>(
        gamesUrl(`${game.game_id}/images/${kind}/${objId}`),
        { method: 'POST', body: '{}' }
      ).then((res) => {
        if (res.ok) {
          setActiveGame((prev) =>
            prev && prev.game_id === game.game_id
              ? { ...prev, game_images: { ...prev.game_images, [key]: true } }
              : prev
          )
        }
      }).catch(() => {})
    }
  }, [])

  const loadGame = useCallback(async (_target: AgentTarget | null, gameId: string) => {
    setPlaying(false)
    setReplay(null)
    try {
      const res = await fetchJSON<{ ok: boolean; game: FullGame }>(gamesUrl(gameId))
      if (res.ok) {
        setActiveGame(res.game)
        ensureGameImages(res.game)
        setMainView('play')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [ensureGameImages])

  // ── Actions ──

  const createGame = useCallback(async () => {
    if (!selected || !pickedWorld) return
    setLoading(true)
    setError(null)
    setReplay(null)
    try {
      // Build seats payload: "agent:ionian" format for agent seats with cognitive modes
      const seatsPayload: Record<string, string> = {}
      for (const [cid, kind] of Object.entries(seatPick)) {
        const mode = modePick[cid]
        seatsPayload[cid] = kind === 'agent' && mode && mode !== 'neutra' ? `${kind}:${mode}` : kind
      }
      // Build agent_targets — FD has no local LLM, so ALL agent seats need a remote provider.
      // Default to `selected` agent; only override if user explicitly picked a different one.
      const agentTargets: Record<string, { host: string; port: number; auth: string; name: string }> = {}
      for (const [cid, kind] of Object.entries(seatPick)) {
        if (kind !== 'agent') continue
        const overrideId = agentTargetPick[cid]
        const provider = (overrideId && overrideId !== selected.id)
          ? targets.find((a) => a.id === overrideId) || selected
          : selected
        agentTargets[cid] = { host: provider.host, port: provider.port, auth: provider.auth, name: provider.name }
      }
      const body: Record<string, any> = { world_id: pickedWorld, seats: seatsPayload }
      if (Object.keys(agentTargets).length > 0) body.agent_targets = agentTargets
      const res = await fetchJSON<{ ok: boolean; game: FullGame }>(gamesUrl(), {
        method: 'POST',
        body: JSON.stringify(body),
      })
      if (res.ok) {
        setActiveGame(res.game)
        setMainView('play')
        await loadWorldsAndGames()
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [selected, pickedWorld, seatPick, modePick, agentTargetPick, targets, loadWorldsAndGames])

  const generateGame = useCallback(async () => {
    if (!selected) return
    if (!genTitle.trim() || !genGoal.trim()) {
      setError('Title and goal are required')
      return
    }
    setGenerating(true)
    setError(null)
    setReplay(null)
    try {
      const spec = {
        title: genTitle.trim(),
        goal: genGoal.trim(),
        summary: genDescription.trim(),
        seats: genSeats,
        genre: genGenre,
        tone: genTone,
        size: genSize,
        constraints: genConstraints.split(',').map((s) => s.trim()).filter(Boolean),
        seat_mode: 'party',
      }
      const res = await fetchJSON<{ ok: boolean; game: FullGame; error?: string }>(
        gamesUrl('generate'),
        { method: 'POST', body: JSON.stringify({ spec, mode: genMode, provider: agentProvider(selected) }) }
      )
      if (res.ok) {
        setPendingGame(res.game)
        const defaults: Record<string, SeatKind> = {}
        const modeDefaults: Record<string, string> = {}
        for (const c of res.game.characters) {
          defaults[c.id] = genDefaultSeat
          modeDefaults[c.id] = 'neutra'
        }
        setPendingSeatPick(defaults)
        setPendingModePick(modeDefaults)
        setPendingAgentTarget({})  // reset — selected agent is the default for all
        setMainView('assign-seats')
        await loadWorldsAndGames()
      } else {
        setError(res.error || 'generation failed')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setGenerating(false)
    }
  }, [selected, genTitle, genGoal, genDescription, genSeats, genGenre, genTone, genSize, genConstraints, genMode, genDefaultSeat, loadWorldsAndGames])

  const randomizeIdea = useCallback(async () => {
    if (!selected) return
    setRandomizing(true)
    setError(null)
    try {
      const res = await fetchJSON<{ ok: boolean; title: string; goal: string; description: string; error?: string }>(
        gamesUrl('random-idea'),
        { method: 'POST', body: JSON.stringify({ provider: agentProvider(selected) }) }
      )
      if (res.ok) {
        setGenTitle(res.title)
        setGenGoal(res.goal)
        setGenDescription(res.description)
      } else {
        setError(res.error || 'failed to generate idea')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRandomizing(false)
    }
  }, [selected])

  const confirmSeats = useCallback(async () => {
    if (!selected || !pendingGame) return
    setError(null)
    try {
      // Build seats payload with cognitive mode
      const seatsPayload: Record<string, string> = {}
      for (const [cid, kind] of Object.entries(pendingSeatPick)) {
        const mode = pendingModePick[cid]
        seatsPayload[cid] = kind === 'agent' && mode && mode !== 'neutra' ? `${kind}:${mode}` : kind
      }
      // Build agent_targets — FD has no local LLM, so ALL agent seats need a remote provider.
      // Default to `selected` agent; only override if user explicitly picked a different one.
      const agentTargets: Record<string, { host: string; port: number; auth: string; name: string }> = {}
      for (const [cid, kind] of Object.entries(pendingSeatPick)) {
        if (kind !== 'agent') continue
        const overrideId = pendingAgentTarget[cid]
        const provider = (overrideId && overrideId !== selected.id)
          ? targets.find((a) => a.id === overrideId) || selected
          : selected
        agentTargets[cid] = { host: provider.host, port: provider.port, auth: provider.auth, name: provider.name }
      }
      const body: Record<string, any> = { seats: seatsPayload }
      if (Object.keys(agentTargets).length > 0) body.agent_targets = agentTargets
      const res = await fetchJSON<{ ok: boolean; game: FullGame }>(
        gamesUrl(`${pendingGame.game_id}/seats`),
        { method: 'POST', body: JSON.stringify(body) }
      )
      if (res.ok) {
        setActiveGame(res.game)
        setPendingGame(null)
        setMainView('play')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [selected, pendingGame, pendingSeatPick, pendingModePick, pendingAgentTarget, targets])

  const tickGame = useCallback(async (): Promise<FullGame | null> => {
    if (!selected || !activeGame) return null
    setTicking(true)
    try {
      const res = await fetchJSON<{ ok: boolean; game: FullGame }>(
        gamesUrl(`${activeGame.game_id}/tick`),
        { method: 'POST', body: '{}' }
      )
      if (res.ok) {
        setActiveGame(res.game)
        ensureGameImages(res.game)
        setGames((prev) =>
          prev.map((g) =>
            g.game_id === res.game.game_id
              ? { ...g, tick: res.game.tick, terminal: res.game.terminal, win: res.game.win }
              : g
          )
        )
        return res.game
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setTicking(false)
    }
    return null
  }, [selected, activeGame, ensureGameImages])

  useEffect(() => {
    if (!playing || !activeGame || activeGame.terminal) return
    let cancelled = false
    const loop = async () => {
      while (!cancelled && playingRef.current) {
        const next = await tickGame()
        if (!next || next.terminal) { setPlaying(false); break }
        await new Promise((r) => setTimeout(r, playInterval))
      }
    }
    void loop()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, activeGame?.game_id])

  const replayGame = useCallback(async () => {
    if (!selected || !activeGame) return
    try {
      const res = await fetchJSON<ReplayResult>(
        gamesUrl(`${activeGame.game_id}/replay`),
        { method: 'POST', body: '{}' }
      )
      setReplay(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [selected, activeGame])

  const restartGame = useCallback(async () => {
    if (!selected || !activeGame) return
    setPlaying(false)
    setReplay(null)
    setError(null)
    try {
      const res = await fetchJSON<{ ok: boolean; game: FullGame }>(
        gamesUrl(`${activeGame.game_id}/restart`),
        { method: 'POST', body: '{}' }
      )
      if (res.ok) {
        setActiveGame(res.game)
        await loadWorldsAndGames()
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [selected, activeGame, loadWorldsAndGames])

  const openSeatReassign = useCallback(async () => {
    if (!selected || !activeGame) return
    // If game has progressed, restart first
    if (activeGame.tick > 0) {
      setPlaying(false)
      setReplay(null)
      setError(null)
      try {
        const res = await fetchJSON<{ ok: boolean; game: FullGame }>(
          gamesUrl(`${activeGame.game_id}/restart`),
          { method: 'POST', body: '{}' }
        )
        if (res.ok) {
          setActiveGame(res.game)
          setPendingGame(res.game)
          const initial: Record<string, SeatKind> = {}
          for (const s of res.game.seats) initial[s.character] = s.kind as SeatKind
          setPendingSeatPick(initial)
          setMainView('assign-seats')
          await loadWorldsAndGames()
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      }
    } else {
      // Already at tick 0 — just open reassign
      setPendingGame(activeGame)
      const initial: Record<string, SeatKind> = {}
      for (const s of activeGame.seats) initial[s.character] = s.kind as SeatKind
      setPendingSeatPick(initial)
      setMainView('assign-seats')
    }
  }, [selected, activeGame, loadWorldsAndGames])

  const deleteGame = useCallback(async (gameId: string) => {
    if (!selected) return
    try {
      await fetchJSON(gamesUrl(gameId), { method: 'DELETE' })
      if (activeGame?.game_id === gameId) { setActiveGame(null); setMainView('landing') }
      await loadWorldsAndGames()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [selected, activeGame, loadWorldsAndGames])

  const transferGame = useCallback(async () => {
    if (!activeGame || !transferTarget) return
    const dest = targets.find((t) => t.id === transferTarget)
    if (!dest) return
    setTransferring(true)
    setError(null)
    try {
      // 1) Export game bundle
      const exportRes = await fetchJSON<{ ok: boolean; bundle: any }>(
        gamesUrl(`${activeGame.game_id}/export`),
        { method: 'POST' },
      )
      if (!exportRes.bundle) throw new Error('export returned no bundle')

      // 2) Re-import as a copy (with new agent targets)
      await fetchJSON<{ ok: boolean; game: FullGame }>(
        gamesUrl('import'),
        { method: 'POST', body: JSON.stringify({ bundle: exportRes.bundle }) },
      )

      setShowTransfer(false)
      setTransferTarget(null)
      await loadWorldsAndGames()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setTransferring(false)
    }
  }, [selected, activeGame, transferTarget, targets, loadWorldsAndGames])

  const submitDraft = useCallback(async (charId: string) => {
    if (!selected || !activeGame) return
    const raw = (intentDrafts[charId] || '').trim()
    if (!raw) return
    setError(null)
    setTicking(true)
    try {
      await fetchJSON(gamesUrl(`${activeGame.game_id}/natural`), {
        method: 'POST',
        body: JSON.stringify({ actor: charId, text: raw, provider: agentProvider(selected) }),
      })
      setIntentDrafts((prev) => ({ ...prev, [charId]: '' }))
      // Auto-tick after queuing so the human sees the result immediately
      await tickGame()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setTicking(false)
    }
  }, [selected, activeGame, intentDrafts, tickGame])

  // Quick action helper for human seats (exits, items, etc.)
  const quickAction = useCallback(async (charId: string, text: string) => {
    if (!selected || !activeGame || ticking) return
    setTicking(true)
    setItemMenu(null)
    setCharMenu(null)
    try {
      await fetchJSON(gamesUrl(`${activeGame.game_id}/natural`), {
        method: 'POST',
        body: JSON.stringify({ actor: charId, text, provider: agentProvider(selected) }),
      })
      await tickGame()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setTicking(false)
    }
  }, [selected, activeGame, ticking, tickGame])

  useEffect(() => {
    const w = worlds.find((w) => w.id === pickedWorld)
    if (!w) return
    setSeatPick((prev) => {
      const next = { ...prev }
      for (const c of w.characters) { if (!next[c.id]) next[c.id] = 'scripted' }
      return next
    })
  }, [pickedWorld, worlds])

  // ── Wizard validation ──
  const canAdvanceStep = wizardStep === 0
    ? genTitle.trim().length > 0 && genGoal.trim().length > 0
    : true

  // ── Render ──

  if (targets.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-zinc-500">
        <div className="text-center">
          <Gamepad2 className="mx-auto mb-3 h-10 w-10 opacity-40" />
          <p>No connected agents. Start a Captain Claw agent first.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full overflow-hidden bg-zinc-950 text-zinc-200">
      {/* ═══ Left rail ═══ */}
      <div className={`flex-shrink-0 overflow-y-auto border-r border-zinc-800 transition-all duration-200 ${sidebarOpen ? 'w-72 p-4' : 'w-10 py-4 px-1.5'}`}>
        {!sidebarOpen ? (
          <div className="flex flex-col items-center gap-3">
            <button onClick={() => setSidebarOpen(true)} className="text-zinc-500 hover:text-zinc-300" title="Expand sidebar">
              <ChevronRight className="h-4 w-4" />
            </button>
            <Gamepad2 className="h-4 w-4 text-zinc-600" />
          </div>
        ) : (
        <>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-zinc-400">
            <Gamepad2 className="h-4 w-4" /> Captain Claw Game
          </h2>
          <button onClick={() => setSidebarOpen(false)} className="text-zinc-600 hover:text-zinc-300" title="Collapse sidebar">
            <ChevronLeft className="h-4 w-4" />
          </button>
        </div>

        {/* Agent picker (LLM provider) */}
        <div className="mb-4">
          <label className={labelCls}>LLM Provider (Agent)</label>
          <select
            value={selected?.id || ''}
            onChange={(e) => setSelectedId(e.target.value)}
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200"
          >
            {targets.map((t) => (
              <option key={t.id} value={t.id}>{t.name} ({t.kind})</option>
            ))}
          </select>
        </div>

        {/* Image generation model */}
        {imageProviders.length > 0 && (
          <div className="mb-4">
            <label className={labelCls}>Image Generation</label>
            <select
              value={activeImageProvider}
              onChange={async (e) => {
                const id = e.target.value
                try {
                  const res = await fetchJSON<{ ok: boolean; active: string }>(
                    gamesUrl(`image-providers/${id}`),
                    { method: 'POST', body: '{}' }
                  )
                  if (res.ok) setActiveImageProvider(res.active)
                } catch (err) {
                  setError(err instanceof Error ? err.message : String(err))
                }
              }}
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200"
            >
              {imageProviders.filter((p) => p.available).map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
          </div>
        )}

        {/* Active games */}
        <div className="mb-4">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-400">Active games</span>
            <button
              onClick={() => selected && loadWorldsAndGames()}
              className="text-zinc-500 hover:text-zinc-300"
              title="Refresh"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
          {games.length === 0 && (
            <p className="text-xs italic text-zinc-600">No games yet.</p>
          )}
          {games.map((g) => (
            <div
              key={g.game_id}
              className={`mb-1 cursor-pointer rounded-md border p-2 text-xs transition ${
                activeGame?.game_id === g.game_id && mainView === 'play'
                  ? 'border-violet-500/50 bg-violet-500/10'
                  : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700'
              }`}
              onClick={() => selected && loadGame(selected, g.game_id)}
            >
              <div className="flex items-center justify-between">
                <span className="flex-1 truncate font-medium text-zinc-200">{g.title}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteGame(g.game_id) }}
                  className="ml-2 text-zinc-600 hover:text-red-400"
                  title="Delete"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
              <div className="mt-1 flex items-center gap-2 text-zinc-500">
                <span>tick {g.tick}</span>
                <span className="text-zinc-700">|</span>
                <span>seed {g.seed}</span>
                {g.terminal && (
                  <span className="ml-auto inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/15 px-2 py-0.5 text-[10px] text-emerald-500">
                    <CheckCircle2 className="h-2.5 w-2.5" /> {g.win ? 'WIN' : 'END'}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Quick-create from demo world */}
        <div className="mb-4 border-t border-zinc-800 pt-4">
          <label className={labelCls}>Quick start (demo world)</label>
          <select
            value={pickedWorld}
            onChange={(e) => setPickedWorld(e.target.value)}
            className="mb-2 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200"
          >
            {worlds.map((w) => (
              <option key={w.id} value={w.id}>{w.title}</option>
            ))}
          </select>

          {pickedWorld && (() => {
            const w = worlds.find((x) => x.id === pickedWorld)
            if (!w) return null
            return (
              <div className="mb-2 rounded-md border border-zinc-800 bg-zinc-900/60 p-2 text-xs">
                <p className="mb-2 italic text-zinc-500">{w.summary}</p>
                {w.characters.map((c) => (
                  <div key={c.id} className="mb-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-300">{c.glyph} {c.name}</span>
                      <select
                        value={seatPick[c.id] || 'scripted'}
                        onChange={(e) =>
                          setSeatPick((prev) => ({ ...prev, [c.id]: e.target.value as SeatKind }))
                        }
                        className="rounded-md border border-zinc-700 bg-zinc-950 px-1 py-0.5 text-xs text-zinc-200"
                      >
                        <option value="scripted">scripted</option>
                        <option value="human">human</option>
                        <option value="agent">agent (LLM)</option>
                      </select>
                    </div>
                    {(seatPick[c.id] || 'scripted') === 'agent' && (
                      <>
                        <div className="mt-1 ml-5 flex items-center gap-1.5">
                          <Brain className="h-3 w-3 text-zinc-500 flex-shrink-0" />
                          <select
                            value={modePick[c.id] || 'neutra'}
                            onChange={(e) =>
                              setModePick((prev) => ({ ...prev, [c.id]: e.target.value }))
                            }
                            className="rounded-md border border-zinc-700 bg-zinc-950 px-1 py-0.5 text-[10px] text-zinc-300 flex-1"
                          >
                            {COGNITIVE_MODES.map((m) => (
                              <option key={m.id} value={m.id}>
                                {m.label} — {m.short}
                              </option>
                            ))}
                          </select>
                        </div>
                        {targets.length > 1 && (
                          <div className="mt-1 ml-5 flex items-center gap-1.5">
                            <Gamepad2 className="h-3 w-3 text-zinc-500 flex-shrink-0" />
                            <select
                              value={agentTargetPick[c.id] || selected?.id || ''}
                              onChange={(e) =>
                                setAgentTargetPick((prev) => ({ ...prev, [c.id]: e.target.value }))
                              }
                              className="rounded-md border border-zinc-700 bg-zinc-950 px-1 py-0.5 text-[10px] text-zinc-300 flex-1"
                            >
                              {targets.map((t) => (
                                <option key={t.id} value={t.id}>
                                  {t.name}{t.id === selected?.id ? ' (local)' : ` (:${t.port})`}
                                </option>
                              ))}
                            </select>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                ))}
              </div>
            )
          })()}

          <button
            onClick={createGame}
            disabled={loading || !pickedWorld}
            className="flex w-full items-center justify-center gap-1 rounded-md bg-violet-600 px-2 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-violet-500 disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Create
          </button>
        </div>

        {/* Generate button */}
        <div className="border-t border-zinc-800 pt-4">
          <button
            onClick={() => { setMainView('generate'); setWizardStep(0) }}
            className={`flex w-full items-center justify-center gap-2 rounded-md border px-2 py-2 text-sm font-medium transition ${
              mainView === 'generate'
                ? 'border-violet-500/50 bg-violet-500/10 text-violet-300'
                : 'border-zinc-700 bg-zinc-900/60 text-zinc-300 hover:border-zinc-600 hover:text-zinc-100'
            }`}
          >
            <Wand2 className="h-4 w-4" />
            Generate New World
          </button>
        </div>

        {error && (
          <div className="mt-4 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-400">
            {error}
          </div>
        )}
        </>
        )}
      </div>

      {/* ═══ Main content area ═══ */}
      <div className="flex-1 overflow-y-auto">

        {/* ─── Landing ─── */}
        {mainView === 'landing' && (
          <div className="flex h-full items-center justify-center">
            <div className="max-w-md text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-md border border-zinc-800 bg-zinc-900/60">
                <Gamepad2 className="h-6 w-6 text-zinc-500" />
              </div>
              <h2 className="mb-2 text-lg font-semibold text-zinc-100">Captain Claw Game</h2>
              <p className="mb-6 text-sm text-zinc-500">
                Multiplayer text adventures for agents and humans.
                Select an active game from the sidebar, create one from a demo world,
                or generate a brand new world.
              </p>
              <button
                onClick={() => { setMainView('generate'); setWizardStep(0) }}
                className="inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900/60 px-4 py-2 text-sm font-medium text-zinc-200 hover:border-zinc-600 hover:text-white"
              >
                <Wand2 className="h-4 w-4" />
                Generate New World
                <ChevronRight className="h-4 w-4 text-zinc-500" />
              </button>
            </div>
          </div>
        )}

        {/* ─── World Generator Wizard ─── */}
        {mainView === 'generate' && (
          <div className="mx-auto max-w-3xl px-6 py-8 pb-16">
            {/* Header */}
            <div className="mb-6 flex items-start gap-3">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-md border border-zinc-800 bg-zinc-900/60">
                <Wand2 className="h-5 w-5 text-zinc-400" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-zinc-100">Generate a New World</h1>
                <p className="mt-0.5 text-sm text-zinc-500">
                  Describe the world you want to play in. The selected agent's LLM will generate
                  rooms, characters, items, and a win condition.
                </p>
              </div>
            </div>

            {/* Step indicator */}
            <div className="mb-6 flex items-center gap-1">
              {WIZARD_STEPS.map((label, i) => (
                <div key={label} className="flex items-center gap-1">
                  {i > 0 && <div className="mx-1 h-px w-6 bg-zinc-800" />}
                  <button
                    onClick={() => setWizardStep(i)}
                    className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition ${
                      i === wizardStep
                        ? 'border-violet-500/50 bg-violet-500/15 text-violet-400'
                        : i < wizardStep
                          ? 'border-zinc-700 bg-zinc-900/60 text-zinc-300'
                          : 'border-zinc-800 bg-transparent text-zinc-600'
                    }`}
                  >
                    <span className={`flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold ${
                      i === wizardStep
                        ? 'bg-violet-500/30 text-violet-300'
                        : i < wizardStep
                          ? 'bg-zinc-700 text-zinc-300'
                          : 'bg-zinc-800 text-zinc-600'
                    }`}>
                      {i < wizardStep ? '\u2713' : i + 1}
                    </span>
                    {label}
                  </button>
                </div>
              ))}
            </div>

            {/* Step 0: Story */}
            {wizardStep === 0 && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-5">
                <div className="mb-1 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-zinc-100">Your Story</h3>
                  <button
                    onClick={randomizeIdea}
                    disabled={randomizing}
                    className="inline-flex items-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-800/80 px-2.5 py-1 text-[11px] font-medium text-zinc-300 transition hover:border-violet-500/40 hover:text-violet-300 disabled:opacity-50"
                    title="Generate a random world idea using AI"
                  >
                    {randomizing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Dices className="h-3 w-3" />}
                    {randomizing ? 'Thinking…' : 'Random idea'}
                  </button>
                </div>
                <p className="mb-5 text-xs text-zinc-500">
                  Give your adventure a name and describe what the players are trying to do.
                  The description is optional but helps the generator create a richer, more coherent world.
                </p>

                <div className="space-y-4">
                  <div>
                    <label className={labelCls}>Title</label>
                    <input
                      type="text"
                      placeholder="e.g. The Sunken Library"
                      value={genTitle}
                      onChange={(e) => setGenTitle(e.target.value)}
                      className={inputCls}
                    />
                    <p className={hintCls}>
                      The name of your adventure. Sets the tone and appears as the game title.
                    </p>
                  </div>

                  <div>
                    <label className={labelCls}>Goal</label>
                    <input
                      type="text"
                      placeholder="e.g. Escape the flooded ruins before the tide rises"
                      value={genGoal}
                      onChange={(e) => setGenGoal(e.target.value)}
                      className={inputCls}
                    />
                    <p className={hintCls}>
                      The shared objective for all players. Each character also gets a private
                      sub-objective generated from this.
                    </p>
                  </div>

                  <div>
                    <label className={labelCls}>Description (optional)</label>
                    <textarea
                      rows={4}
                      placeholder="A longer description of the world, its backstory, atmosphere, or any specific details you want the generator to incorporate..."
                      value={genDescription}
                      onChange={(e) => setGenDescription(e.target.value)}
                      className={`${inputCls} resize-y`}
                    />
                    <p className={hintCls}>
                      Expand on the story with context, lore, or specific scene details. The generator
                      uses this to shape room descriptions, character backgrounds, and item placement.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Step 1: World Settings */}
            {wizardStep === 1 && (
              <div className="space-y-4">
                <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-5">
                  <h3 className="mb-1 text-sm font-semibold text-zinc-100">World Settings</h3>
                  <p className="mb-5 text-xs text-zinc-500">
                    Configure the genre, tone, and scale of the generated world.
                  </p>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className={labelCls}>Genre</label>
                      <select value={genGenre} onChange={(e) => setGenGenre(e.target.value)} className={selectCls}>
                        <option value="exploration">Exploration</option>
                        <option value="cozy-mystery">Cozy Mystery</option>
                        <option value="escape">Escape</option>
                        <option value="puzzle">Puzzle</option>
                        <option value="horror">Horror</option>
                      </select>
                      <p className={hintCls}>Influences room descriptions, item types, and atmosphere.</p>
                    </div>
                    <div>
                      <label className={labelCls}>Tone</label>
                      <input
                        type="text"
                        placeholder="e.g. whimsical, grim, comedic"
                        value={genTone}
                        onChange={(e) => setGenTone(e.target.value)}
                        className={inputCls}
                      />
                      <p className={hintCls}>Free-form writing tone. Affects descriptions and dialogue style.</p>
                    </div>
                    <div>
                      <label className={labelCls}>Characters (1-6)</label>
                      <input
                        type="number"
                        min={1} max={6}
                        value={genSeats}
                        onChange={(e) => setGenSeats(Math.max(1, Math.min(6, Number(e.target.value) || 2)))}
                        className={inputCls}
                      />
                      <p className={hintCls}>Each gets a unique start position and private objective.</p>
                    </div>
                    <div>
                      <label className={labelCls}>Default seat type</label>
                      <select value={genDefaultSeat} onChange={(e) => setGenDefaultSeat(e.target.value as SeatKind)} className={selectCls}>
                        <option value="scripted">Scripted (autopilot)</option>
                        <option value="human">Human (you play)</option>
                        <option value="agent">Agent (LLM decides)</option>
                      </select>
                      <p className={hintCls}>Pre-selects who controls each character. Changeable after generation.</p>
                    </div>
                  </div>

                  <div className="mt-4">
                    <label className={labelCls}>Constraints (optional)</label>
                    <input
                      type="text"
                      placeholder="e.g. no locked doors, include a cat, all rooms underwater"
                      value={genConstraints}
                      onChange={(e) => setGenConstraints(e.target.value)}
                      className={inputCls}
                    />
                    <p className={hintCls}>Comma-separated rules the generator must follow.</p>
                  </div>
                </div>

                {/* Size selector */}
                <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-5">
                  <h3 className="mb-1 text-sm font-semibold text-zinc-100">World Size</h3>
                  <p className="mb-4 text-xs text-zinc-500">
                    Controls how many rooms, items, and NPCs the generator creates.
                  </p>
                  <div className="grid grid-cols-3 gap-3">
                    {(['small', 'medium', 'large', 'xl', 'huge', 'epic'] as const).map((s) => {
                      const info = SIZE_INFO[s]
                      const active = genSize === s
                      return (
                        <button
                          key={s}
                          onClick={() => setGenSize(s)}
                          className={`rounded-md border p-3 text-left transition ${
                            active
                              ? 'border-violet-500/50 bg-violet-500/10'
                              : 'border-zinc-800 bg-zinc-950/60 hover:border-zinc-700'
                          }`}
                        >
                          <div className="mb-1 text-sm font-semibold capitalize text-zinc-200">{s}</div>
                          <div className="space-y-0.5 text-[11px] text-zinc-500">
                            <div>{info.rooms} rooms</div>
                            <div>{info.entities} items</div>
                            <div>{info.npcs} NPCs</div>
                          </div>
                          <p className="mt-1.5 text-[11px] leading-snug text-zinc-600">{info.desc}</p>
                        </button>
                      )
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Step 2: Generation Mode */}
            {wizardStep === 2 && (
              <div className="space-y-4">
                <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-5">
                  <h3 className="mb-1 text-sm font-semibold text-zinc-100">Generation Mode</h3>
                  <p className="mb-4 text-xs text-zinc-500">
                    Choose how the world is generated. Fast mode is great for quick experiments;
                    Pipeline mode produces higher-quality worlds with a solvability guarantee.
                  </p>

                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => setGenMode('fast')}
                      className={`rounded-md border p-4 text-left transition ${
                        genMode === 'fast'
                          ? 'border-violet-500/50 bg-violet-500/10'
                          : 'border-zinc-800 bg-zinc-950/60 hover:border-zinc-700'
                      }`}
                    >
                      <div className="mb-2 flex items-center gap-2">
                        <Zap className={`h-5 w-5 ${genMode === 'fast' ? 'text-violet-400' : 'text-zinc-500'}`} />
                        <span className="text-sm font-semibold text-zinc-200">Fast</span>
                        <span className="rounded-full border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">M1a</span>
                      </div>
                      <p className="text-xs leading-relaxed text-zinc-400">
                        Single LLM call generates the entire world at once. Quick
                        (~5-15s) but no solvability repair.
                      </p>
                      <div className="mt-3 rounded-md border border-zinc-800 bg-zinc-950/50 px-2.5 py-1.5 text-[11px] text-zinc-500">
                        <Info className="mr-1 inline h-3 w-3 align-text-bottom" />
                        1 API call. If the win condition is unreachable, the game may be unwinnable.
                      </div>
                    </button>

                    <button
                      onClick={() => setGenMode('pipeline')}
                      className={`rounded-md border p-4 text-left transition ${
                        genMode === 'pipeline'
                          ? 'border-violet-500/50 bg-violet-500/10'
                          : 'border-zinc-800 bg-zinc-950/60 hover:border-zinc-700'
                      }`}
                    >
                      <div className="mb-2 flex items-center gap-2">
                        <Layers className={`h-5 w-5 ${genMode === 'pipeline' ? 'text-violet-400' : 'text-zinc-500'}`} />
                        <span className="text-sm font-semibold text-zinc-200">Pipeline</span>
                        <span className="rounded-full border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">M1b</span>
                      </div>
                      <p className="text-xs leading-relaxed text-zinc-400">
                        8-stage pipeline: premise, rooms, cast, inventory, clues,
                        character sheets, ASCII art, win condition.
                      </p>
                      <div className="mt-3 rounded-md border border-zinc-800 bg-zinc-950/50 px-2.5 py-1.5 text-[11px] text-zinc-500">
                        <Info className="mr-1 inline h-3 w-3 align-text-bottom" />
                        8+ API calls, ~30-90s. Includes solvability check with auto-repair (up to 2 retries).
                      </div>
                    </button>
                  </div>
                </div>

                {/* Summary before generating */}
                <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-4 py-3">
                  <div className="grid grid-cols-4 gap-3 text-xs">
                    <div>
                      <span className="text-zinc-500">Title</span>
                      <div className="mt-0.5 font-medium text-zinc-200">{genTitle || '(none)'}</div>
                    </div>
                    <div>
                      <span className="text-zinc-500">Size</span>
                      <div className="mt-0.5 font-medium capitalize text-zinc-200">{genSize}</div>
                    </div>
                    <div>
                      <span className="text-zinc-500">Genre</span>
                      <div className="mt-0.5 font-medium capitalize text-zinc-200">{genGenre}</div>
                    </div>
                    <div>
                      <span className="text-zinc-500">Characters</span>
                      <div className="mt-0.5 font-medium text-zinc-200">{genSeats}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Navigation */}
            <div className="mt-6 flex items-center justify-between">
              <div>
                {wizardStep > 0 && (
                  <button
                    onClick={() => setWizardStep((s) => s - 1)}
                    className="inline-flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900/60 px-3.5 py-2 text-sm text-zinc-300 hover:border-zinc-600 hover:text-zinc-100"
                  >
                    <ChevronLeft className="h-4 w-4" /> Back
                  </button>
                )}
              </div>
              <div className="flex items-center gap-3">
                {wizardStep < 2 ? (
                  <button
                    onClick={() => setWizardStep((s) => s + 1)}
                    disabled={!canAdvanceStep}
                    className="inline-flex items-center gap-1 rounded-md bg-violet-600 px-3.5 py-2 text-sm font-medium text-white shadow-sm hover:bg-violet-500 disabled:opacity-50"
                  >
                    Next <ChevronRight className="h-4 w-4" />
                  </button>
                ) : (
                  <>
                    <button
                      onClick={generateGame}
                      disabled={generating || !genTitle.trim() || !genGoal.trim()}
                      className="inline-flex items-center gap-2 rounded-md bg-violet-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-violet-500 disabled:opacity-50"
                    >
                      {generating
                        ? <Loader2 className="h-4 w-4 animate-spin" />
                        : <Wand2 className="h-4 w-4" />}
                      {generating ? 'Generating...' : 'Generate World'}
                    </button>
                    {generating && (
                      <span className="text-xs text-zinc-500">
                        {genMode === 'pipeline'
                          ? 'Running 8-stage pipeline...'
                          : 'Single-call generation...'}
                      </span>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ─── Seat assignment (post-generation) ─── */}
        {mainView === 'assign-seats' && pendingGame && (
          <div className="mx-auto max-w-3xl px-6 py-8 pb-16">
            {/* Header */}
            <div className="mb-6 flex items-start gap-3">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-md border border-emerald-500/30 bg-emerald-500/10">
                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-zinc-100">{pendingGame.title}</h1>
                <p className="mt-0.5 text-sm text-zinc-500">World generated successfully</p>
              </div>
            </div>

            {/* Story & setting */}
            <div className="mb-5 rounded-lg border border-zinc-800 bg-zinc-900/60 p-5">
              <h3 className="mb-2 text-sm font-semibold text-zinc-100">The Story</h3>
              {pendingGame.summary ? (
                <p className="text-sm leading-relaxed text-zinc-400">{pendingGame.summary}</p>
              ) : (
                <p className="text-sm italic text-zinc-600">No summary available.</p>
              )}
              {pendingGame.rooms && pendingGame.rooms.length > 0 && (
                <div className="mt-3 rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2">
                  <span className="text-[11px] font-medium text-zinc-500">
                    {pendingGame.rooms.length} locations:
                  </span>{' '}
                  <span className="text-[11px] text-zinc-600">
                    {pendingGame.rooms.map((r) => r.name).join(' \u2022 ')}
                  </span>
                </div>
              )}
            </div>

            {/* Characters + seat assignment */}
            <div className="mb-5 rounded-lg border border-zinc-800 bg-zinc-900/60 p-5">
              <h3 className="mb-1 text-sm font-semibold text-zinc-100">Characters &amp; Seats</h3>
              <p className="mb-4 text-xs text-zinc-500">
                Choose who controls each character. All agent seats use the LLM provider of the
                selected agent ({selected?.name || 'none'}).
              </p>

              <div className="space-y-3">
                {pendingGame.characters.map((c) => {
                  const startRoom = pendingGame.rooms?.find((r) => r.id === c.start_room)
                  return (
                    <div key={c.id} className="rounded-md border border-zinc-800 bg-zinc-950/60 p-4">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex items-start gap-3">
                          <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-md border border-zinc-700 bg-zinc-900 font-mono text-lg text-zinc-300">
                            {c.glyph}
                          </span>
                          <div className="min-w-0">
                            <div className="text-sm font-semibold text-zinc-200">{c.name}</div>
                            {c.description && (
                              <p className="mt-0.5 text-xs leading-relaxed text-zinc-400">
                                {c.description}
                              </p>
                            )}
                            <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-zinc-600">
                              {c.objective && (
                                <span>
                                  <span className="text-zinc-500">Objective:</span> {c.objective}
                                </span>
                              )}
                              {startRoom && (
                                <span>
                                  <span className="text-zinc-500">Starts in:</span> {startRoom.name}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="flex flex-col gap-2 flex-shrink-0 items-end">
                          <select
                            value={pendingSeatPick[c.id] || 'scripted'}
                            onChange={(e) =>
                              setPendingSeatPick((prev) => ({ ...prev, [c.id]: e.target.value as SeatKind }))
                            }
                            className="rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500 focus:outline-none"
                          >
                            <option value="scripted">Scripted (autopilot)</option>
                            <option value="human">Human (you play)</option>
                            <option value="agent">Agent (LLM decides)</option>
                          </select>
                          {(pendingSeatPick[c.id] || 'scripted') === 'agent' && (
                            <>
                              <div className="flex items-center gap-1.5 w-full">
                                <Brain className="h-3.5 w-3.5 text-zinc-500 flex-shrink-0" />
                                <select
                                  value={pendingModePick[c.id] || 'neutra'}
                                  onChange={(e) =>
                                    setPendingModePick((prev) => ({ ...prev, [c.id]: e.target.value }))
                                  }
                                  className="flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-300 focus:border-violet-500 focus:outline-none"
                                >
                                  {COGNITIVE_MODES.map((m) => (
                                    <option key={m.id} value={m.id}>
                                      {m.label} — {m.short}
                                    </option>
                                  ))}
                                </select>
                              </div>
                              {targets.length > 1 && (
                                <div className="flex items-center gap-1.5 w-full">
                                  <Gamepad2 className="h-3.5 w-3.5 text-zinc-500 flex-shrink-0" />
                                  <select
                                    value={pendingAgentTarget[c.id] || selected?.id || ''}
                                    onChange={(e) =>
                                      setPendingAgentTarget((prev) => ({ ...prev, [c.id]: e.target.value }))
                                    }
                                    className="flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-300 focus:border-violet-500 focus:outline-none"
                                  >
                                    {targets.map((t) => (
                                      <option key={t.id} value={t.id}>
                                        {t.name}{t.id === selected?.id ? ' (local)' : ` (${t.kind} · :${t.port})`}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                              )}
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Info note about agent selection */}
            <div className="mb-6 rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2 text-[11px] text-zinc-500">
              <Info className="mr-1 inline h-3 w-3 align-text-bottom" />
              Agent-controlled characters use the LLM of the hosting agent (<span className="text-zinc-400">{selected?.name}</span>).
              Each agent can have a different <span className="text-zinc-400">cognitive mode</span> that shapes its reasoning style — from convergent problem-solving (Ionian) to adversarial analysis (Phrygian) to creative exploration (Lydian).
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={confirmSeats}
                className="inline-flex items-center gap-2 rounded-md bg-violet-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-violet-500"
              >
                <Play className="h-4 w-4" /> Start Game
              </button>
              <button
                onClick={() => {
                  setActiveGame(pendingGame)
                  setPendingGame(null)
                  setMainView('play')
                }}
                className="rounded-md border border-zinc-700 bg-zinc-900/60 px-3.5 py-2 text-sm text-zinc-400 hover:text-zinc-200"
              >
                Skip (use defaults)
              </button>
            </div>
          </div>
        )}

        {/* ─── Active game view ─── */}
        {mainView === 'play' && activeGame && (
          <div className="p-4">
            {/* ── World banner ── */}
            {activeGame.game_images?.banner && (
              <div className="mb-4 overflow-hidden rounded-lg border border-zinc-800">
                <img
                  src={`/fd/games/${activeGame.game_id}/images/banner`}
                  alt={activeGame.title}
                  className="h-32 w-full cursor-pointer object-cover transition hover:brightness-110"
                  loading="lazy"
                  onClick={() => setZoomImage({ src: `/fd/games/${activeGame.game_id}/images/banner`, alt: activeGame.title })}
                />
              </div>
            )}
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h1 className="text-lg font-semibold text-zinc-100">{activeGame.title}</h1>
                <p className="text-xs text-zinc-500">
                  {activeGame.game_id} | seed {activeGame.seed} | tick {activeGame.tick}
                  {activeGame.terminal && (
                    <span className="ml-2 inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/15 px-2 py-0.5 text-[10px] text-emerald-500">
                      {activeGame.win ? 'WIN' : 'END'}
                    </span>
                  )}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <select
                  value={playInterval}
                  onChange={(e) => setPlayInterval(Number(e.target.value))}
                  className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-200"
                  title="Auto-tick interval"
                >
                  <option value={500}>0.5s</option>
                  <option value={1000}>1s</option>
                  <option value={1500}>1.5s</option>
                  <option value={3000}>3s</option>
                  <option value={5000}>5s</option>
                </select>
                <button
                  onClick={() => setPlaying((p) => !p)}
                  disabled={activeGame.terminal}
                  className={`flex items-center gap-1 rounded-md border px-3 py-1.5 text-sm disabled:opacity-50 ${
                    playing
                      ? 'border-amber-500/30 bg-amber-500/10 text-amber-400 hover:bg-amber-500/15'
                      : 'border-zinc-700 bg-zinc-900/60 text-zinc-300 hover:border-zinc-600'
                  }`}
                  title={playing ? 'Pause auto-tick' : 'Auto-tick until terminal'}
                >
                  {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  {playing ? 'Pause' : 'Play'}
                </button>
                <button
                  onClick={() => void tickGame()}
                  disabled={activeGame.terminal || playing || ticking}
                  className="flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-sm text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
                >
                  {ticking && !playing ? <Loader2 className="h-4 w-4 animate-spin" /> : <StepForward className="h-4 w-4" />}
                  {ticking && !playing ? 'Thinking…' : 'Tick'}
                </button>
                <button
                  onClick={replayGame}
                  className="flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-sm text-zinc-300 hover:border-zinc-600"
                >
                  <RefreshCw className="h-4 w-4" /> Replay
                </button>
                <div className="mx-1 h-6 w-px bg-zinc-700" />
                <button
                  onClick={() => void restartGame()}
                  className="flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-sm text-zinc-300 hover:border-zinc-600"
                  title="Reset game to tick 0"
                >
                  <RotateCcw className="h-4 w-4" /> Restart
                </button>
                <button
                  onClick={() => void openSeatReassign()}
                  className="flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-sm text-zinc-300 hover:border-zinc-600"
                  title="Change seat assignments (restarts game)"
                >
                  <Users className="h-4 w-4" /> Seats
                </button>
                {targets.length > 1 && (
                  <button
                    onClick={() => { setShowTransfer(true); setTransferTarget(null) }}
                    className="flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-sm text-zinc-300 hover:border-zinc-600"
                    title="Transfer game to another agent"
                  >
                    <ArrowRightLeft className="h-4 w-4" /> Transfer
                  </button>
                )}
              </div>
            </div>

            {replay && (
              <div className={`mb-4 rounded-md border px-3 py-2 text-xs ${
                replay.matches_live
                  ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                  : 'border-amber-500/30 bg-amber-500/10 text-amber-400'
              }`}>
                Replay: {replay.ticks_replayed} ticks |{' '}
                {replay.matches_live
                  ? 'matches live state byte-for-byte'
                  : 'DIVERGED from live state'}
              </div>
            )}

            {/* ── Transfer modal ── */}
            {showTransfer && (
              <div className="mb-4 rounded-lg border border-zinc-700 bg-zinc-900/90 p-4">
                <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-zinc-200">
                  <ArrowRightLeft className="h-4 w-4" /> Transfer game to another agent
                </h3>
                <p className="mb-3 text-xs text-zinc-500">
                  Copies this game (world, history, conversations, thoughts) to the selected agent.
                  The original stays here.
                </p>
                <div className="mb-3">
                  <select
                    value={transferTarget || ''}
                    onChange={(e) => setTransferTarget(e.target.value || null)}
                    className={selectCls}
                  >
                    <option value="">Select destination agent…</option>
                    {targets.filter((t) => t.id !== selected?.id).map((t) => (
                      <option key={t.id} value={t.id}>{t.name} ({t.kind} · :{t.port})</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => void transferGame()}
                    disabled={!transferTarget || transferring}
                    className="flex items-center gap-1 rounded-md bg-violet-600 px-3 py-1.5 text-sm text-white hover:bg-violet-500 disabled:opacity-50"
                  >
                    {transferring ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowRightLeft className="h-4 w-4" />}
                    {transferring ? 'Transferring…' : 'Transfer'}
                  </button>
                  <button
                    onClick={() => setShowTransfer(false)}
                    className="rounded-md border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-sm text-zinc-400 hover:text-zinc-200"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            <div className={`grid gap-4 ${focusedChar ? '' : 'lg:grid-cols-2'}`}>
              {activeGame.characters.filter((c) => !focusedChar || c.id === focusedChar).map((c) => {
                const seat = activeGame.seats.find((s) => s.character === c.id)
                const view = activeGame.views[c.id] as GameView | undefined
                const isHuman = seat?.kind === 'human'
                const isAgent = seat?.kind === 'agent'
                const thoughts = isAgent ? (activeGame.agent_thoughts?.[c.id] || []) : []
                const curTab = agentTab[c.id] || 'thoughts'
                const convLog = activeGame.conversation_log || []
                // Conversations this character was part of (heard or said)
                const charConv = convLog.filter(
                  (e) => e.actor === c.id || e.audience.includes(c.id)
                )
                const cr = view?.current_room

                return (
                  <div key={c.id} className={`flex flex-col overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/60 ${focusedChar === c.id ? 'text-lg' : ''}`}>
                    {/* ── Header ── */}
                    <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
                      <div className="flex items-center gap-2">
                        {activeGame.game_images?.[`char:${c.id}`] ? (
                          <img
                            src={`/fd/games/${activeGame.game_id}/images/char/${c.id}`}
                            alt={c.name}
                            className="h-8 w-8 cursor-pointer rounded-full border border-zinc-700 object-cover transition hover:border-zinc-500"
                            onClick={() => setZoomImage({ src: `/fd/games/${activeGame.game_id}/images/char/${c.id}`, alt: c.name })}
                          />
                        ) : (
                          <span className="font-mono text-zinc-400">[{c.glyph}]</span>
                        )}
                        <span className="font-medium text-zinc-200">{c.name}</span>
                        <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] ${
                          isAgent
                            ? 'border-blue-500/30 bg-blue-500/15 text-blue-400'
                            : isHuman
                              ? 'border-violet-500/30 bg-violet-500/15 text-violet-400'
                              : 'border-zinc-700 bg-zinc-800 text-zinc-400'
                        }`}>
                          {seat?.kind}
                        </span>
                        {isAgent && seat?.cognitive_mode && seat.cognitive_mode !== 'neutra' && (() => {
                          const modeInfo = COGNITIVE_MODES.find((m) => m.id === seat.cognitive_mode)
                          return modeInfo ? (
                            <span className="inline-flex items-center gap-1 rounded-full border border-zinc-700 bg-zinc-800/80 px-2 py-0.5 text-[10px] text-zinc-400">
                              <span className={`h-1.5 w-1.5 rounded-full ${modeInfo.color}`} />
                              {modeInfo.label}
                            </span>
                          ) : null
                        })()}
                        {isAgent && seat?.remote_agent && seat.remote_agent.name && (
                          <span className="inline-flex items-center gap-1 rounded-full border border-violet-500/30 bg-violet-500/10 px-2 py-0.5 text-[10px] text-violet-400">
                            <ArrowRightLeft className="h-2.5 w-2.5" />
                            {seat.remote_agent.name}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        {ticking && (isAgent || playing) && (
                          <span className="inline-flex items-center gap-1 text-[10px] text-blue-400/70">
                            <Loader2 className="h-3 w-3 animate-spin" /> thinking
                          </span>
                        )}
                        <button
                          onClick={() => setFocusedChar(focusedChar === c.id ? null : c.id)}
                          className={`rounded p-0.5 transition ${focusedChar === c.id ? 'text-violet-400 hover:text-violet-300' : 'text-zinc-600 hover:text-zinc-300'}`}
                          title={focusedChar === c.id ? 'Exit focus' : 'Focus'}
                        >
                          {focusedChar === c.id ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
                        </button>
                        <span className="text-[10px] text-zinc-600">tick {view?.tick ?? '?'}</span>
                      </div>
                    </div>

                    {/* ── Body ── */}
                    <div className={`flex flex-1 ${isAgent ? 'divide-x divide-zinc-800' : ''}`} style={{ minHeight: focusedChar === c.id ? 400 : 220 }}>
                      {/* ── Game state panel (structured) ── */}
                      <div className="flex-1 overflow-y-auto p-3">
                        {view?.character && (
                          <p className="mb-2 text-sm italic text-zinc-400">{view.character.objective}</p>
                        )}
                        {cr && (
                          <>
                            <h4 className="mb-1 text-lg font-semibold text-zinc-100">{cr.name}</h4>
                            {/* ── Room image ── */}
                            {activeGame.game_images?.[`room:${cr.id}`] && (
                              <img
                                src={`/fd/games/${activeGame.game_id}/images/room/${cr.id}`}
                                alt={cr.name}
                                className="mb-2 w-full cursor-pointer rounded-md border border-zinc-800 object-cover transition hover:border-zinc-600"
                                style={{ maxHeight: focusedChar === c.id ? 320 : 180 }}
                                loading="lazy"
                                onClick={() => setZoomImage({ src: `/fd/games/${activeGame.game_id}/images/room/${cr.id}`, alt: cr.name })}
                              />
                            )}
                            {cr.ascii_tile && !activeGame.game_images?.[`room:${cr.id}`] && (
                              <pre className="mb-2 font-mono text-xs leading-none text-emerald-500/70">{cr.ascii_tile.join('\n')}</pre>
                            )}
                            <p className="mb-3 text-sm leading-snug text-zinc-400">{cr.description}</p>
                            {cr.entities && cr.entities.length > 0 && (
                              <div className="mb-2">
                                <span className="text-xs font-semibold uppercase tracking-wide text-amber-600/70">Items</span>
                                {cr.entities.map((e) => {
                                  const menuKey = `${c.id}:${e.id}`
                                  const menuOpen = itemMenu === menuKey
                                  return (
                                    <div key={e.id} className="relative mt-0.5">
                                      <div
                                        className={`flex items-center gap-1.5 text-sm ${isHuman ? 'cursor-pointer rounded px-1 -mx-1 transition hover:bg-zinc-800/60' : ''}`}
                                        onClick={isHuman ? () => { setItemMenu(menuOpen ? null : menuKey); setCharMenu(null) } : undefined}
                                      >
                                        {activeGame.game_images?.[`entity:${e.id}`] ? (
                                          <img
                                            src={`/fd/games/${activeGame.game_id}/images/entity/${e.id}`}
                                            alt={e.name}
                                            className="h-5 w-5 rounded border border-zinc-700 object-cover"
                                            onClick={(ev) => { ev.stopPropagation(); setZoomImage({ src: `/fd/games/${activeGame.game_id}/images/entity/${e.id}`, alt: e.name }) }}
                                          />
                                        ) : (
                                          <span className="text-amber-500/50">{e.glyph}</span>
                                        )}
                                        <span className="text-zinc-300">{e.name}</span>
                                        {e.takeable && <span className="text-xs text-emerald-600/50">(take)</span>}
                                        {e.examinable && !e.examined && <span className="text-xs text-purple-500/60">(examine)</span>}
                                        {e.examined && <span className="text-xs text-purple-400/40">(examined)</span>}
                                        <span className="font-mono text-[11px] text-zinc-600">{e.id}</span>
                                      </div>
                                      {/* Item action menu */}
                                      {menuOpen && isHuman && (
                                        <div className="absolute left-0 z-20 mt-0.5 flex gap-1 rounded-md border border-zinc-700 bg-zinc-900 px-1.5 py-1 shadow-lg">
                                          {e.examinable && !e.examined && (
                                            <button
                                              onClick={() => quickAction(c.id, `examine ${e.id}`)}
                                              className="rounded px-2 py-0.5 text-xs text-purple-400 transition hover:bg-purple-900/30"
                                            >
                                              Examine
                                            </button>
                                          )}
                                          <button
                                            onClick={() => quickAction(c.id, `take ${e.id}`)}
                                            disabled={!e.takeable}
                                            className={`rounded px-2 py-0.5 text-xs transition ${e.takeable ? 'text-emerald-400 hover:bg-emerald-900/30' : 'cursor-not-allowed text-zinc-600'}`}
                                          >
                                            Take
                                          </button>
                                          <button
                                            onClick={() => quickAction(c.id, `use ${e.id}`)}
                                            className="rounded px-2 py-0.5 text-xs text-amber-400 transition hover:bg-amber-900/30"
                                          >
                                            Use
                                          </button>
                                          <button
                                            onClick={() => setItemMenu(null)}
                                            className="rounded px-1.5 py-0.5 text-xs text-zinc-500 transition hover:bg-zinc-800"
                                          >
                                            &times;
                                          </button>
                                        </div>
                                      )}
                                    </div>
                                  )
                                })}
                              </div>
                            )}
                            {cr.others_here && cr.others_here.length > 0 && (
                              <div className="mb-2">
                                <span className="text-xs font-semibold uppercase tracking-wide text-cyan-600/70">Present</span>
                                {cr.others_here.map((o) => {
                                  const cMenuKey = `${c.id}:${o.id}`
                                  const cMenuOpen = charMenu === cMenuKey
                                  return (
                                    <div key={o.id} className="relative mt-0.5">
                                      <div
                                        className={`flex items-center gap-1.5 text-sm ${isHuman ? 'cursor-pointer rounded px-1 -mx-1 transition hover:bg-zinc-800/60' : ''}`}
                                        onClick={isHuman ? () => { setCharMenu(cMenuOpen ? null : cMenuKey); setItemMenu(null) } : undefined}
                                      >
                                        {activeGame.game_images?.[`char:${o.id}`] ? (
                                          <img
                                            src={`/fd/games/${activeGame.game_id}/images/char/${o.id}`}
                                            alt={o.name}
                                            className="h-6 w-6 rounded-full border border-zinc-700 object-cover"
                                            onClick={(ev) => { ev.stopPropagation(); setZoomImage({ src: `/fd/games/${activeGame.game_id}/images/char/${o.id}`, alt: o.name }) }}
                                          />
                                        ) : (
                                          <span className="text-cyan-500/50">{o.glyph}</span>
                                        )}
                                        <span className="text-zinc-300">{o.name}</span>
                                      </div>
                                      {/* Character interaction menu */}
                                      {cMenuOpen && isHuman && (
                                        <div className="absolute left-0 z-20 mt-0.5 flex gap-1 rounded-md border border-zinc-700 bg-zinc-900 px-1.5 py-1 shadow-lg">
                                          <button
                                            onClick={() => {
                                              setCharMenu(null)
                                              setIntentDrafts((prev) => ({ ...prev, [c.id]: `talk ${o.name} ` }))
                                            }}
                                            className="rounded px-2 py-0.5 text-xs text-cyan-400 transition hover:bg-cyan-900/30"
                                          >
                                            Talk
                                          </button>
                                          <button
                                            onClick={() => {
                                              setCharMenu(null)
                                              setIntentDrafts((prev) => ({ ...prev, [c.id]: `say ` }))
                                            }}
                                            className="rounded px-2 py-0.5 text-xs text-blue-400 transition hover:bg-blue-900/30"
                                          >
                                            Say
                                          </button>
                                          <button
                                            onClick={() => setCharMenu(null)}
                                            className="rounded px-1.5 py-0.5 text-xs text-zinc-500 transition hover:bg-zinc-800"
                                          >
                                            &times;
                                          </button>
                                        </div>
                                      )}
                                    </div>
                                  )
                                })}
                              </div>
                            )}
                            {cr.exits && (
                              <div className="mb-2">
                                <span className="text-xs font-semibold uppercase tracking-wide text-zinc-600">Exits</span>
                                <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-1">
                                  {Object.entries(cr.exits).sort().map(([dir, roomId]) => {
                                    const isLocked = cr.locked_exits?.[dir]
                                    const canClick = isHuman && !isLocked && !ticking
                                    return (
                                      <span
                                        key={dir}
                                        className={`text-xs ${canClick ? 'cursor-pointer' : ''}`}
                                        onClick={canClick ? () => quickAction(c.id, `go ${dir}`) : undefined}
                                        title={canClick ? `Move ${dir}` : undefined}
                                      >
                                        <span className={`rounded px-1.5 py-0.5 font-mono ${isLocked ? 'bg-red-900/30 text-red-400' : canClick ? 'bg-zinc-800 text-zinc-300 transition hover:bg-violet-900/40 hover:text-violet-300' : 'bg-zinc-800 text-zinc-300'}`}>{dir}{isLocked ? ' 🔒' : ''}</span>
                                        <span className="ml-1 text-zinc-600">{roomId.replace(/_/g, ' ')}</span>
                                      </span>
                                    )
                                  })}
                                </div>
                              </div>
                            )}
                          </>
                        )}
                        {/* ── Inventory ── */}
                        <div className="mt-2 rounded border border-zinc-800/60 bg-zinc-950/40 px-2.5 py-2">
                          <span className="text-xs font-semibold uppercase tracking-wide text-amber-700/50">Inventory</span>
                          {(view?.inventory ?? []).length === 0 ? (
                            <p className="mt-0.5 text-sm italic text-zinc-600">(empty)</p>
                          ) : (
                            <div className="mt-1 flex flex-wrap gap-1.5">
                              {view!.inventory!.map((e) => (
                                <span key={e.id} className="inline-flex items-center gap-1 rounded bg-amber-900/20 px-2 py-0.5 text-sm text-amber-600/60">
                                  {activeGame.game_images?.[`entity:${e.id}`] && (
                                    <img
                                      src={`/fd/games/${activeGame.game_id}/images/entity/${e.id}`}
                                      alt={e.name}
                                      className="h-4 w-4 cursor-pointer rounded object-cover"
                                      onClick={() => setZoomImage({ src: `/fd/games/${activeGame.game_id}/images/entity/${e.id}`, alt: e.name })}
                                    />
                                  )}
                                  {e.name}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                        {/* ── Heard this tick ── */}
                        {view?.heard && view.heard.length > 0 && (
                          <div className="mt-2">
                            {view.heard.map((h, i) => (
                              <div key={i} className={`mt-1 rounded border-l-2 px-2.5 py-1.5 text-sm ${
                                h.kind === 'talk'
                                  ? 'border-orange-700/50 bg-orange-950/30'
                                  : 'border-cyan-500/40 bg-cyan-500/5'
                              }`}>
                                {h.kind === 'talk' && (
                                  <span className="mr-1.5 text-[10px] font-medium text-orange-600/70">DIRECT</span>
                                )}
                                <span className={`font-medium ${h.kind === 'talk' ? 'text-orange-400' : 'text-cyan-400'}`}>
                                  {h.actor_name}:
                                </span>{' '}
                                <span className="italic text-zinc-300">&ldquo;{h.text}&rdquo;</span>
                              </div>
                            ))}
                          </div>
                        )}
                        {/* ── Action results this tick (examine, use, errors) ── */}
                        {view?.action_results && view.action_results.length > 0 && (
                          <div className="mt-2">
                            {view.action_results.map((r, i) => (
                              <div key={i} className={`mt-1 rounded border-l-2 px-2.5 py-1.5 text-sm ${
                                r.kind === 'error'
                                  ? 'border-red-500/40 bg-red-500/5'
                                  : r.kind === 'reflection'
                                    ? 'border-sky-500/30 bg-sky-500/5'
                                    : 'border-purple-500/40 bg-purple-500/5'
                              }`}>
                                {r.kind === 'examine_result' && (
                                  <>
                                    <span className="font-medium text-purple-400">Examined {r.entity_name || r.entity_id}:</span>{' '}
                                    <span className="text-zinc-300">{r.text}</span>
                                  </>
                                )}
                                {r.kind === 'use_result' && (
                                  <span className="text-zinc-300">{r.text}</span>
                                )}
                                {r.kind === 'reflection' && (
                                  <span className="italic text-sky-300">{r.text}</span>
                                )}
                                {r.kind === 'error' && (
                                  <span className="text-red-400">{r.text}</span>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                        {/* ── Accumulated discoveries ── */}
                        {view?.discoveries && view.discoveries.length > 0 && (
                          <div className="mt-2 rounded border border-zinc-800/60 bg-zinc-950/40 px-2.5 py-2">
                            <span className="text-xs font-semibold uppercase tracking-wide text-purple-600/60">Discoveries</span>
                            {view.discoveries.map((d, i) => (
                              <div key={i} className="mt-1 text-sm leading-snug">
                                {d.kind === 'examine_result' && (
                                  <div className="flex gap-1.5">
                                    <span className="shrink-0 text-purple-500/50">🔍</span>
                                    <span><span className="font-medium text-purple-400/70">{d.entity_name || d.entity_id}:</span>{' '}<span className="text-zinc-400">{d.text}</span></span>
                                  </div>
                                )}
                                {d.kind === 'use_result' && (
                                  <div className="flex gap-1.5">
                                    <span className="shrink-0 text-amber-500/50">⚡</span>
                                    <span className="text-zinc-400">{d.text}</span>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                        {view?.terminal && (
                          <div className="mt-3 rounded border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-2 text-center text-base font-bold text-emerald-400">
                            GAME OVER {view.win ? '(WIN)' : ''}
                          </div>
                        )}
                      </div>

                      {/* ── Agent side panel with tabs ── */}
                      {isAgent && (
                        <div className={`flex flex-col overflow-hidden ${focusedChar === c.id ? 'w-1/2' : 'w-2/5'}`}>
                          {/* Tab bar */}
                          <div className="flex border-b border-zinc-800">
                            <button
                              onClick={() => setAgentTab((p) => ({ ...p, [c.id]: 'thoughts' }))}
                              className={`flex flex-1 items-center justify-center gap-1 px-2 py-1.5 text-[10px] font-medium uppercase tracking-wide transition ${
                                curTab === 'thoughts'
                                  ? 'border-b-2 border-blue-500 text-blue-400'
                                  : 'text-zinc-500 hover:text-zinc-300'
                              }`}
                            >
                              <Brain className="h-3 w-3" /> Thoughts
                            </button>
                            <button
                              onClick={() => setAgentTab((p) => ({ ...p, [c.id]: 'memory' }))}
                              className={`flex flex-1 items-center justify-center gap-1 px-2 py-1.5 text-[10px] font-medium uppercase tracking-wide transition ${
                                curTab === 'memory'
                                  ? 'border-b-2 border-cyan-500 text-cyan-400'
                                  : 'text-zinc-500 hover:text-zinc-300'
                              }`}
                            >
                              <MessageSquare className="h-3 w-3" /> Memory
                            </button>
                          </div>
                          {/* Tab content */}
                          <div className={`flex-1 overflow-y-auto ${focusedChar === c.id ? 'p-4' : 'p-2'}`}>
                            {curTab === 'thoughts' && (() => {
                              const focused = focusedChar === c.id
                              const tsSmall = focused ? 'text-xs' : 'text-[10px]'
                              const tsBody = focused ? 'text-sm leading-relaxed' : 'text-[11px] leading-snug'
                              const tsBubble = focused ? 'text-sm' : 'text-xs'
                              const tsDm = focused ? 'text-[10px]' : 'text-[9px]'
                              return (
                              <>
                                {thoughts.length === 0 ? (
                                  <p className={`${tsBody} italic text-zinc-600`}>No thoughts yet — tick to see reasoning.</p>
                                ) : (
                                  [...thoughts].reverse().slice(0, focused ? 10 : 3).map((t, i) => {
                                    const isSay = t.action.startsWith('say ')
                                    const isTalk = t.action.startsWith('talk ')
                                    const talkParts = isTalk ? t.action.slice(5).split(' ') : []
                                    const talkTarget = talkParts[0] || ''
                                    const talkText = talkParts.slice(1).join(' ')
                                    const sayText = isSay ? t.action.slice(4) : ''
                                    const isSpeech = isSay || isTalk
                                    return (
                                      <div key={i} className={`mb-2 rounded border border-zinc-800 bg-zinc-950/60 ${focused ? 'p-3' : 'p-2'}`}>
                                        <div className="mb-1 flex items-center justify-between">
                                          <span className={`${tsSmall} font-medium text-zinc-500`}>Tick {t.tick}</span>
                                          {!isSpeech && (
                                            <span className={`rounded-full bg-blue-500/15 px-1.5 py-0.5 ${tsSmall} font-mono text-blue-400`}>{t.action}</span>
                                          )}
                                        </div>
                                        {isSay && (
                                          <div className={`mb-1.5 rounded-lg rounded-tl-none border border-violet-500/20 bg-violet-500/8 px-2.5 py-1.5 ${tsBubble} italic text-violet-300`}>
                                            &ldquo;{sayText}&rdquo;
                                          </div>
                                        )}
                                        {isTalk && (
                                          <div className={`mb-1.5 rounded-lg rounded-tl-none border border-amber-500/20 bg-amber-500/8 px-2.5 py-1.5 ${tsBubble}`}>
                                            <span className={`mr-1.5 ${tsDm} font-medium text-amber-500/60`}>DM to {talkTarget}</span>
                                            <span className="italic text-zinc-100">&ldquo;{talkText}&rdquo;</span>
                                          </div>
                                        )}
                                        <p className={`${tsBody} text-zinc-400`}>{t.reasoning || '(no reasoning)'}</p>
                                      </div>
                                    )
                                  })
                                )}
                              </>
                              )
                            })()}
                            {curTab === 'memory' && (() => {
                              const focused = focusedChar === c.id
                              const tsSmall = focused ? 'text-xs' : 'text-[10px]'
                              const tsDm = focused ? 'text-[10px]' : 'text-[9px]'
                              const tsBody = focused ? 'text-sm leading-relaxed' : 'text-[11px] leading-snug'
                              return (
                              <>
                                {charConv.length === 0 ? (
                                  <p className={`${tsBody} italic text-zinc-600`}>No conversations yet.</p>
                                ) : (
                                  [...charConv].reverse().map((e, i) => {
                                    const isTalk = e.kind === 'talk'
                                    const isSelf = e.actor === c.id
                                    return (
                                      <div key={i} className={`mb-2 rounded border ${focused ? 'p-3' : 'p-2'} ${
                                        isTalk
                                          ? 'border-orange-700/20 bg-orange-950/20'
                                          : (isSelf ? 'border-blue-500/20 bg-blue-500/5' : 'border-cyan-500/20 bg-cyan-500/5')
                                      }`}>
                                        <div className="mb-0.5 flex items-center justify-between">
                                          <span className={`${tsSmall} font-medium ${
                                            isTalk
                                              ? 'text-orange-400'
                                              : (isSelf ? 'text-blue-400' : 'text-cyan-400')
                                          }`}>
                                            {isTalk && <span className={`mr-1 ${tsDm} text-orange-600/60`}>DM</span>}
                                            {isSelf ? (isTalk ? 'Whispered' : 'Said') : e.actor_name}
                                          </span>
                                          <span className={`${tsDm} text-zinc-600`}>tick {e.tick} · {e.room_name || e.room}</span>
                                        </div>
                                        <p className={`${tsBody} italic text-zinc-300`}>&ldquo;{e.text}&rdquo;</p>
                                      </div>
                                    )
                                  })
                                )}
                              </>
                              )
                            })()}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* ── Human input (sticky bottom) ── */}
                    {isHuman && !activeGame.terminal && (
                      <div className="mt-auto border-t border-zinc-800 bg-zinc-900/80 p-2">
                        <div className="flex gap-1">
                          <input
                            type="text"
                            value={intentDrafts[c.id] || ''}
                            onChange={(e) =>
                              setIntentDrafts((prev) => ({ ...prev, [c.id]: e.target.value }))
                            }
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' && !ticking) void submitDraft(c.id)
                            }}
                            disabled={ticking}
                            placeholder="go north / grab lantern / examine journal / use key on door / give key to Ben"
                            className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 font-mono text-xs text-zinc-200 focus:border-violet-500 focus:outline-none disabled:opacity-50"
                          />
                          <button
                            onClick={() => submitDraft(c.id)}
                            disabled={ticking}
                            className="rounded-md bg-violet-600 px-2 py-1 text-white hover:bg-violet-500 disabled:opacity-50"
                            title="Submit & tick"
                          >
                            {ticking ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
                          </button>
                        </div>
                        <p className="mt-1 text-[10px] text-zinc-600">
                          Type naturally — the agent interprets your intent. Auto-ticks on submit.
                        </p>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>

      {/* ── Image zoom modal ── */}
      {zoomImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
          onClick={() => setZoomImage(null)}
        >
          <div className="relative max-h-[90vh] max-w-[90vw]" onClick={(e) => e.stopPropagation()}>
            <img
              src={zoomImage.src}
              alt={zoomImage.alt}
              className="max-h-[90vh] max-w-[90vw] rounded-lg border border-zinc-700 object-contain"
            />
            <button
              onClick={() => setZoomImage(null)}
              className="absolute -right-3 -top-3 flex h-7 w-7 items-center justify-center rounded-full bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200"
            >
              &times;
            </button>
            <p className="mt-2 text-center text-sm text-zinc-400">{zoomImage.alt}</p>
          </div>
        </div>
      )}
    </div>
  )
}
