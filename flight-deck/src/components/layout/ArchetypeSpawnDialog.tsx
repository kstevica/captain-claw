import { useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  ArrowLeft, ArrowRight, Bot, Brain, Calendar, ChartColumn, Check, Clock, Code2, Contact, Crop, FolderUp, Link2,
  Database, Eye, FileSpreadsheet, FileText, Gauge, Globe, Hammer, HeartPulse, Image as ImageIcon, LayoutGrid, Library, Lightbulb, ListTodo,
  Mail, Map as MapIcon, Mic, Monitor, Presentation, Scale, Search, SearchX, ScanText, Sparkles, Star,
  TerminalSquare, Wand2, X, Zap,
  FileUp,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import {
  KINDS, archetypeCategory, archetypeKind, archetypeLook, familyLook, kindLook, kindName, rememberTeamLooks, shelves,
  type ArchetypeKind, type FamilyLook, type KindLook,
} from './archetypeLook'
import { KindTabs, type KindFilter } from './KindTabs'
import { tileInNextRow } from './tileNav'
import { Button, EmptyState, Switch, inputCls, labelCls } from '../console/parts'
import { ErrorNote } from '../forge/forgeParts'
import { DialogFullscreen } from '../library/dialogFullscreen'
import { FullscreenButton } from '../library/libraryParts'
import { listArchetypes } from '../../services/archetypes'
import { MODES } from '../../services/cognitiveModes'
import { spawnArchetypeProcess } from '../../services/docker'
import type { Archetype, ArchetypeCategory, ArchetypeRegistry } from '../../services/tierConfig'
import { useProcessStore } from '../../stores/processStore'
import { probeGoogle, probeMicrosoft } from '../../stores/connectionsStore'
import { useIsMobile } from '../../hooks/useMediaQuery'

// ── Kiosk "New agent" picker ─────────────────────────────────────────
//
// The locked Simple layout has no Spawner, so this is the one way a kiosk user
// creates an agent: pick a Library archetype, name it, done. Everything else —
// tools, cognitive mode, model — is resolved by FD from the archetype and the
// account's tier set (or the admin's team default), so no model config or API
// key is exposed here.
//
// Layout, in the Admin console's look: the kinds (Company, Generalists,
// Specialists, each in its own colour) over the search, the families down the
// left under their kind and shelf (chips on a narrower window), the cards in the
// middle under a band per kind that stays in view while they scroll, and on the right
// the detail of the card in hand — what it does, how it thinks, which model
// class it runs on, what it can reach — with the name and Create button under
// it. The detail is never empty: it shows the first card until another is
// chosen, and follows the search as it narrows, so typing, Enter and Enter again
// makes an agent. On a phone the two are steps: the gallery, then the detail
// with a Back arrow.
//
// "Set up its way of working next" (optional, under the name): the agent is
// created as usual, then the caller opens AgentSetupDialog on it — the
// person's instructions and documents, read by their model into rules,
// skills, facts and knowledge they review before it reaches the agent.
//
// "Forge agents" (under the families, when the caller offers it): the other way
// in, for someone who knows what they need and not which agent — ForgeAgentsDialog.

/** Mirrors the backend `_slug`: a process agent's name is its directory. */
function agentSlug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/^-+|-+$/g, '') || 'cc-agent'
}

function uniqueName(base: string, taken: Set<string>): string {
  if (!taken.has(agentSlug(base))) return base
  for (let n = 2; ; n++) {
    const candidate = `${base} ${n}`
    if (!taken.has(agentSlug(candidate))) return candidate
  }
}

// ── Presentation metadata ────────────────────────────────────────────

/** The model class an archetype runs on, said for people. The registry's own
 *  tier label is the fallback for tiers this map doesn't know. */
const TIER_LOOKS: Record<string, { icon: LucideIcon; label: string; blurb: string }> = {
  reason: { icon: Brain, label: 'Deep reasoning', blurb: 'Takes its time and thinks things through.' },
  balanced: { icon: Gauge, label: 'Balanced', blurb: 'Everyday knowledge work at a steady pace.' },
  fast: { icon: Zap, label: 'Fast', blurb: 'Quick answers, built for volume.' },
  coding: { icon: Code2, label: 'Coding', blurb: 'Tuned for code and multi-file edits.' },
  vision: { icon: Eye, label: 'Vision', blurb: 'Understands images, screenshots and scans.' },
  longctx: { icon: FileText, label: 'Long documents', blurb: 'Reads very large documents in one go.' },
  micro: { icon: Sparkles, label: 'Small local model', blurb: 'Runs on a small model, fully local.' },
}

/** Tools every archetype has — not worth a chip of their own. */
const BASE_TOOLS = new Set([
  'shell', 'read', 'write', 'glob', 'edit', 'web_fetch', 'web_search', 'personality', 'playbooks', 'scripts',
  'pdf_extract', 'docx_extract', 'xlsx_extract', 'pptx_extract', 'insights', 'contacts', 'apis', 'todo',
])

const CAPABILITIES: Record<string, { icon: LucideIcon; label: string }> = {
  browser: { icon: Globe, label: 'Browse the web' },
  docx: { icon: FileText, label: 'Edit Word files' },
  xlsx: { icon: FileSpreadsheet, label: 'Build spreadsheets' },
  pptx: { icon: Presentation, label: 'Build slide decks' },
  visualize: { icon: ChartColumn, label: 'Make charts' },
  pdf_extract: { icon: FileText, label: 'Read PDFs' },
  docx_extract: { icon: FileText, label: 'Read Word files' },
  xlsx_extract: { icon: FileSpreadsheet, label: 'Read spreadsheets' },
  pptx_extract: { icon: Presentation, label: 'Read slide decks' },
  summarize_files: { icon: FileText, label: 'Summarise documents' },
  datastore: { icon: Database, label: 'Keep data tables' },
  insights: { icon: Lightbulb, label: 'Remember insights' },
  cron: { icon: Clock, label: 'Run on a schedule' },
  send_mail: { icon: Mail, label: 'Send email' },
  google_mail: { icon: Mail, label: 'Gmail' },
  google_calendar: { icon: Calendar, label: 'Google Calendar' },
  microsoft_mail: { icon: Mail, label: 'Outlook mail' },
  microsoft_calendar: { icon: Calendar, label: 'Outlook calendar' },
  microsoft_onedrive: { icon: FileText, label: 'OneDrive' },
  image_gen: { icon: Wand2, label: 'Generate images' },
  image_manipulation: { icon: Crop, label: 'Edit images' },
  image_vision: { icon: ImageIcon, label: 'Look at images' },
  image_ocr: { icon: ScanText, label: 'Read text in images' },
  cv: { icon: Eye, label: 'Local vision' },
  audio: { icon: Mic, label: 'Transcribe audio' },
  contacts: { icon: Contact, label: 'Contacts' },
  todo: { icon: ListTodo, label: 'To-do lists' },
  apis: { icon: Globe, label: 'Call APIs' },
  direct_api: { icon: Globe, label: 'Call APIs' },
  codemap: { icon: MapIcon, label: 'Map a codebase' },
  desktop_action: { icon: Monitor, label: 'Desktop actions' },
  terminal: { icon: TerminalSquare, label: 'Terminal' },
  flight_deck: { icon: Library, label: 'Manage the deck' },
}

function capabilities(a: Archetype): { key: string; icon: LucideIcon; label: string }[] {
  const seen = new Set<string>()
  const out: { key: string; icon: LucideIcon; label: string }[] = []
  for (const t of a.tools || []) {
    if (BASE_TOOLS.has(t)) continue
    const cap = CAPABILITIES[t] || { icon: Sparkles, label: t.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase()) }
    if (seen.has(cap.label)) continue
    seen.add(cap.label)
    out.push({ key: t, ...cap })
  }
  return out
}

function modeInfo(id: string) {
  return MODES.find((m) => m.id === id) || MODES[0]
}

/** The first sentence of a mode's long description: enough for a card. */
function firstSentence(text: string): string {
  const m = text.match(/^[^.!?]+[.!?]/)
  return (m ? m[0] : text).trim()
}

function matchesQuery(a: Archetype, q: string): boolean {
  return !q || a.role.toLowerCase().includes(q) || a.id.toLowerCase().includes(q)
    || a.family.toLowerCase().includes(q) || a.description.toLowerCase().includes(q)
    || (a.keywords || []).some((k) => k.toLowerCase().includes(q))
    || capabilities(a).some((c) => c.label.toLowerCase().includes(q))
}

// ── Component ────────────────────────────────────────────────────────

/** What a caller needs to open AgentSetupDialog on the new agent. */
export interface SetupRequest { slug: string; name: string; role: string; readDocs?: string[] }

export function ArchetypeSpawnDialog({ takenSlugs, onClose, onSpawned, onForge, onImport, asUser, ownerLabel, onConnect }: {
  /** Slugs of the agents this user can see — the default name avoids them. */
  takenSlugs: Set<string>
  onClose: () => void
  /** `setup`: the person ticked "set up its way of working next". */
  onSpawned: (slug: string, setup?: SetupRequest) => void
  /** Say what's needed and let Captain plan the team instead. */
  onForge?: () => void
  /** "From a file": make it of a .captain backup or export instead (AgentImportDialog). */
  onImport?: () => void
  /** Admin creating the agent FOR that user: their gallery, their ownership,
   *  their tier set and plan. */
  asUser?: string
  ownerLabel?: string
  /** Open Connections on that service (the kiosk passes it): an archetype that
   *  works in mail or a calendar says so before it's created, not on its first
   *  turn ("Triage my inbox" → blocked, cohort001). */
  onConnect?: (service: 'google' | 'microsoft') => void
}) {
  const [registry, setRegistry] = useState<ArchetypeRegistry | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const [query, setQuery] = useState('')
  const [kind, setKind] = useState<KindFilter>('all')
  // A family is picked on its shelf: the company's "Legal" isn't the built-in one.
  const [family, setFamily] = useState<FamilyPick | null>(null)
  const [picked, setPicked] = useState<Archetype | null>(null)
  // What the person typed as its name; until then, the name that goes with the card.
  const [typedName, setTypedName] = useState<string | null>(null)
  const [nameTurn, setNameTurn] = useState(0)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [setupNext, setSetupNext] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  const nameRef = useRef<HTMLInputElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const { isMobile } = useIsMobile()
  const archetypes = registry?.archetypes ?? null
  const fullscreenHost = useMemo(() => (isMobile ? null : { fullscreen, toggle: () => setFullscreen((v) => !v) }), [isMobile, fullscreen])
  const big = fullscreen && !isMobile
  // Are the person's mail accounts connected? Only their own count: an admin
  // creating for someone else can't see theirs, so no note then. null = unknown.
  const [mail, setMail] = useState<{ connected: boolean; microsoft: boolean } | null>(null)
  useEffect(() => {
    if (asUser) return
    let live = true
    Promise.all([probeGoogle(), probeMicrosoft()]).then(([g, m]) => {
      if (!live || g.health === 'gray') return  // sign-in off: no accounts to connect
      const up = (c: { health: string } | null) => !!c && (c.health === 'green' || c.health === 'yellow')
      setMail({ connected: up(g) || up(m), microsoft: m !== null })
    }).catch(() => {})
    return () => { live = false }
  }, [asUser])

  useEffect(() => {
    listArchetypes(asUser, true)
      .then((reg) => {
        const registry = { ...reg, archetypes: reg.archetypes || [] }
        rememberTeamLooks(registry)
        setRegistry(registry)
      })
      .catch((e) => setLoadErr(e instanceof Error ? e.message : String(e)))
  }, [asUser])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || busy) return
      // On a phone the detail is a step of its own: Escape walks back first.
      if (isMobile && picked) { setPicked(null); return }
      onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onClose, isMobile, picked])

  // Each kind, its shelves, their families and how many each holds, whatever the search says.
  const onDeck = useMemo(() => byKind(archetypes || []).map((k) => ({
    ...k,
    shelves: k.shelves.map((sh) => ({ ...sh, families: sh.families.map(([fam, list]) => [fam, list.length] as const) })),
  })), [archetypes])
  const inView = onDeck.filter((k) => kind === 'all' || k.id === kind)
  const inViewTotal = inView.reduce((n, k) => n + k.total, 0)
  const familiesInView = inView.flatMap((k) => k.shelves.flatMap((sh) => sh.families.map(([name, n]) => ({ shelf: sh.id, name, n }))))
  const kindCounts = useMemo(() => Object.fromEntries(onDeck.map((k) => [k.id, k.total])) as Partial<Record<ArchetypeKind, number>>, [onDeck])
  // A shelf gets a heading when its name says more than its kind's does.
  const titled = useMemo(() => new Set(onDeck.filter((k) => k.id === 'company' || k.shelves.length > 1).map((k) => k.id)), [onDeck])

  const q = query.trim().toLowerCase()
  const shown = useMemo(() => (archetypes || []).filter((a) => {
    if (kind !== 'all' && archetypeKind(a) !== kind) return false
    if (family && (archetypeCategory(a) !== family.shelf || (a.family || 'Other') !== family.name)) return false
    return matchesQuery(a, q)
  }), [archetypes, q, kind, family])
  // A search that finds nothing here may still find something of another kind.
  const elsewhere = useMemo(() => (q && !shown.length ? (archetypes || []).filter((a) => matchesQuery(a, q)).length : 0), [archetypes, q, shown])

  const groups = useMemo(() => byKind(shown), [shown])

  const inOrder = useMemo(() => groups.flatMap((k) => k.shelves.flatMap((sh) => sh.families.flatMap(([, list]) => list))), [groups])
  // The search's answer: a card with a word of its name starting that way ("fact"
  // is the Fact Checker, not the Refactorer), then one with it inside its name
  // ("bug" is the Debugger, though the Code Reviewer has it as a keyword), else
  // the first one listed.
  const best = useMemo(() => {
    const q = query.trim().toLowerCase()
    const named = (a: Archetype) => ` ${a.role.toLowerCase()}`
    return (q && (inOrder.find((a) => named(a).includes(` ${q}`)) || inOrder.find((a) => named(a).includes(q)))) || inOrder[0] || null
  }, [inOrder, query])
  // The card in hand: what was chosen while it's still listed, else that answer.
  // A phone opens the detail with a tap, so nothing is in hand until then.
  const current = isMobile ? picked : (picked && inOrder.includes(picked) ? picked : best)
  const name = typedName ?? (current ? uniqueName(current.role || current.id, takenSlugs) : '')

  const pick = (a: Archetype) => {
    setPicked(a)
    setErr('')
    if (!typedName?.trim()) setTypedName(null)
    setNameTurn((n) => n + 1)
  }

  // Focus the name once the pick has re-rendered it, so Enter creates. Not on
  // a phone: the keyboard would cover the detail the user just opened.
  useEffect(() => {
    if (nameTurn && !isMobile) {
      nameRef.current?.focus()
      nameRef.current?.select()
    }
  }, [nameTurn, isMobile])

  const chooseFamily = (fam: FamilyPick | null) => {
    setFamily(fam && family && family.shelf === fam.shelf && family.name === fam.name ? null : fam)
    setPicked(null)
    listRef.current?.scrollTo(0, 0)
  }

  const chooseKind = (next: KindFilter) => {
    setKind(next)
    setFamily(null)
    setPicked(null)
    listRef.current?.scrollTo(0, 0)
  }

  // Arrows move between the cards, as in the kiosk's menus, and up out of the
  // first row to the search. The card with the focus shows in the detail.
  const onCardKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const across = ({ ArrowRight: 1, ArrowLeft: -1 } as Record<string, number>)[e.key]
    const down = ({ ArrowDown: 1, ArrowUp: -1 } as Record<string, number>)[e.key]
    if (!across && !down) return
    const cards = [...e.currentTarget.querySelectorAll<HTMLButtonElement>('[data-tile]:not(:disabled)')]
    const at = cards.indexOf(document.activeElement as HTMLButtonElement)
    if (at < 0) return
    e.preventDefault()
    const next = across ? cards[at + across] : tileInNextRow(cards, cards[at], down)
    if (next) next.focus()
    else if (down === -1) searchRef.current?.focus()
  }

  const create = async () => {
    if (!current || !name.trim() || busy) return
    setBusy(true)
    setErr('')
    let finalName = name.trim()
    try {
      let res
      try {
        res = await spawnArchetypeProcess(current.id, finalName, current.description, asUser)
      } catch (e) {
        // Taken by an agent this user can't see (another account's): try once
        // more with a short suffix rather than make a kiosk user guess.
        const msg = e instanceof Error ? e.message : String(e)
        if (!/already (exists|running)/i.test(msg)) throw e
        finalName = `${finalName} ${Math.random().toString(36).slice(2, 6)}`
        res = await spawnArchetypeProcess(current.id, finalName, current.description, asUser)
      }
      if (!res.ok) throw new Error(res.message)
      const slug = res.slug || agentSlug(finalName)
      // FD records the instructions for the owner; mirror what it stored —
      // and only when the agent is the viewer's own.
      if (!asUser) {
        const procs = useProcessStore.getState()
        const stored = res.fleet_instructions || current.fleet_instructions
        if (stored) procs.setFleetInstructions(slug, stored)
        procs.fetchProcesses()
      }
      onSpawned(slug, setupNext ? { slug, name: finalName, role: current.role || current.id } : undefined)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  const phoneDetail = isMobile && !!picked

  return createPortal(
    <div className={`fd-fade fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 backdrop-blur-[2px] wide:items-center ${big ? '' : 'wide:p-4'}`} onClick={() => { if (!busy) onClose() }}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="New agent from an archetype"
        className={`fd-sheet fd-card-wide flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 shadow-2xl ${
          big ? '' : 'wide:h-[min(92vh,62rem)] wide:max-w-6xl wide:rounded-2xl wide:border wide:border-zinc-800'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        <DialogFullscreen.Provider value={fullscreenHost}>
          {/* Header */}
          <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
            {phoneDetail ? (
              <button onClick={() => setPicked(null)} disabled={busy} aria-label="Back to all archetypes" className="-ml-2 rounded-lg p-2 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
                <ArrowLeft className="h-5 w-5" />
              </button>
            ) : (
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
                <Bot className="h-[18px] w-[18px]" />
              </span>
            )}
            <div className="min-w-0 flex-1">
              <h3 className="truncate text-base font-semibold text-zinc-100">New agent{asUser && ownerLabel ? ` for ${ownerLabel}` : ''}</h3>
              <p className="truncate text-xs text-zinc-500">
                {phoneDetail ? 'Give it a name and create it.' : isMobile ? 'Pick one and name it.' : 'Pick an archetype and give it a name. Its role, tools and model come with it.'}
              </p>
            </div>
            {onForge && !phoneDetail && (
              <button
                onClick={onForge}
                disabled={busy}
                title="Say what you need and attach files; Captain plans the agents"
                className={`flex shrink-0 items-center gap-1.5 rounded-lg border border-violet-500/40 bg-violet-500/10 px-2.5 py-1.5 text-xs font-medium text-violet-700 transition-colors hover:bg-violet-500/20 disabled:opacity-50 dark:text-violet-200 ${isMobile ? '' : 'lg:hidden'}`}
              >
                <Hammer className="h-3.5 w-3.5" /> Forge agents
              </button>
            )}
            {onImport && !phoneDetail && (
              <button
                onClick={onImport}
                disabled={busy}
                title="Restore a backup, or bring in an agent someone exported for you (.captain file)"
                className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1.5 text-xs font-medium text-zinc-300 transition-colors hover:bg-zinc-800 disabled:opacity-50"
              >
                <FileUp className="h-3.5 w-3.5" /> From a file
              </button>
            )}
            <div className="flex shrink-0 items-center gap-0.5">
              <FullscreenButton />
              <button onClick={onClose} disabled={busy} aria-label="Close" title="Close" className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
                <X className="h-[18px] w-[18px]" />
              </button>
            </div>
          </div>

          <div className="flex min-h-0 flex-1 flex-col wide:flex-row">
            {/* ── Families ── */}
            {!isMobile && (
              <aside className="hidden w-56 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/40 lg:flex">
                <nav className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-2" aria-label="Archetype families">
                  <RailItem
                    active={family === null}
                    onClick={() => chooseFamily(null)}
                    icon={kind === 'all' ? LayoutGrid : kindLook(kind).icon}
                    iconClass={kind === 'all' ? 'text-violet-600 dark:text-violet-400' : kindLook(kind).text}
                    label={kind === 'all' ? 'All' : `All ${kindLook(kind).label.toLowerCase()}`}
                    count={inViewTotal}
                  />
                  {inView.map((k) => (
                    <div key={k.id} className="mt-4">
                      {kind === 'all' && (
                        <p className={`flex items-center gap-1.5 px-2.5 pb-1 text-[11px] font-semibold uppercase tracking-wider ${k.text}`}>
                          <k.icon className="h-3.5 w-3.5" /> {k.label}
                        </p>
                      )}
                      {k.shelves.map((sh) => (
                        <div key={sh.id} className="space-y-0.5">
                          {titled.has(k.id) && <p className="px-2.5 pb-0.5 pt-1.5 text-[11px] font-medium text-zinc-500">{sh.label}</p>}
                          {sh.families.map(([fam, n]) => {
                            const look = familyLook(fam)
                            return <RailItem key={fam} active={family?.shelf === sh.id && family.name === fam} onClick={() => chooseFamily({ shelf: sh.id, name: fam })} icon={look.icon} iconClass={look.text} label={fam} count={n} />
                          })}
                        </div>
                      ))}
                    </div>
                  ))}
                </nav>
                {onForge && (
                  <div className="shrink-0 p-2">
                    <button
                      onClick={onForge}
                      disabled={busy}
                      className="group w-full rounded-2xl border border-violet-500/20 bg-gradient-to-br from-violet-500/[0.12] via-transparent to-sky-500/[0.06] p-3 text-left transition-colors hover:border-violet-500/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 disabled:opacity-50"
                    >
                      <span className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                        <Hammer className="h-4 w-4 shrink-0 text-violet-600 dark:text-violet-400" /> Forge agents
                      </span>
                      <span className="mt-1 block text-xs leading-relaxed text-zinc-500">Not sure which one? Say what you need and Captain plans the team.</span>
                      <span className="mt-2 flex items-center gap-1 text-xs font-medium text-violet-700 dark:text-violet-300">
                        Describe it <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
                      </span>
                    </button>
                  </div>
                )}
              </aside>
            )}

            {/* ── Gallery ── */}
            {!phoneDetail && (
              <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                <div className="shrink-0 space-y-2.5 px-4 pt-3 wide:px-5 wide:pt-4">
                  {onDeck.length > 1 && <KindTabs value={kind} onChange={chooseKind} counts={kindCounts} hints={big} compact={isMobile} />}
                  <div className="relative">
                    <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
                    <input
                      ref={searchRef}
                      // Not on a phone: the keyboard would cover the list first thing.
                      autoFocus={!isMobile}
                      value={query}
                      onChange={(e) => { setQuery(e.target.value); setPicked(null); listRef.current?.scrollTo(0, 0) }}
                      onKeyDown={(e) => {
                        // Enter on a narrowed list takes the search's answer.
                        if (e.key === 'Enter' && best && query.trim()) { e.preventDefault(); pick(best) }
                        if (e.key === 'ArrowDown') {
                          e.preventDefault()
                          const list = listRef.current
                          const card = list?.querySelector<HTMLButtonElement>('[data-tile][aria-pressed="true"]') ?? list?.querySelector<HTMLButtonElement>('[data-tile]')
                          card?.focus()
                        }
                      }}
                      placeholder="Search by role, task or skill…"
                      aria-label="Search archetypes"
                      className={`${inputCls} !py-2.5 !pl-10 !pr-9 !text-base wide:!text-sm`}
                    />
                    {query && (
                      <button onClick={() => { setQuery(''); searchRef.current?.focus() }} aria-label="Clear search" className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                        <X className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                  {familiesInView.length > 1 && (
                    <div className={`-mx-4 flex gap-1.5 overflow-x-auto px-4 py-0.5 [mask-image:linear-gradient(to_right,black_calc(100%-1.5rem),transparent)] [scrollbar-width:none] wide:-mx-5 wide:px-5 [&::-webkit-scrollbar]:hidden ${isMobile ? '' : 'lg:hidden'}`} role="tablist" aria-label="Archetype families">
                      <FamilyChip active={family === null} onClick={() => chooseFamily(null)} label="All" count={inViewTotal} />
                      {familiesInView.map((f) => (
                        <FamilyChip key={`${f.shelf}/${f.name}`} active={family?.shelf === f.shelf && family.name === f.name} onClick={() => chooseFamily({ shelf: f.shelf, name: f.name })} label={f.name} count={f.n} look={familyLook(f.name)} />
                      ))}
                    </div>
                  )}
                </div>

                <div ref={listRef} onKeyDown={onCardKey} className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 pb-5 pt-4 wide:px-5">
                  {loadErr && <ErrorNote>Couldn't load archetypes: {loadErr}</ErrorNote>}
                  {!loadErr && archetypes === null && (
                    <div role="status" aria-label="Loading archetypes" className={CARD_GRID}>
                      {Array.from({ length: 8 }, (_, i) => <div key={i} className="fd-shimmer h-28 rounded-2xl" />)}
                    </div>
                  )}
                  {archetypes && groups.length === 0 && (
                    archetypes.length === 0 ? (
                      <EmptyState icon={Bot} title="No archetypes available">Ask an administrator to add some.</EmptyState>
                    ) : (
                      <EmptyState icon={SearchX} title={`Nothing matches “${query.trim()}”${family ? ` in ${family.name}` : kind !== 'all' ? ` in ${kindLook(kind).label}` : ''}`}>
                        <span className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1">
                          {elsewhere > 0 ? (
                            <button onClick={() => chooseKind('all')} className="font-medium text-violet-700 hover:underline dark:text-violet-300">
                              See {elsewhere} match{elsewhere === 1 ? '' : 'es'} everywhere else
                            </button>
                          ) : (
                            <button onClick={() => { setQuery(''); chooseKind('all') }} className="font-medium text-violet-700 hover:underline dark:text-violet-300">Show everything</button>
                          )}
                          {onForge && <button onClick={onForge} className="font-medium text-violet-700 hover:underline dark:text-violet-300">Describe it to Captain instead</button>}
                        </span>
                      </EmptyState>
                    )
                  )}
                  <div key={`${kind}/${family ? `${family.shelf}/${family.name}` : 'all'}`} className="fd-fade space-y-8">
                    {groups.map((k) => (
                      <section key={k.id} aria-label={k.label} className="space-y-5">
                        {!family && <KindBand k={k} count={k.total} />}
                        {k.shelves.map((sh) => (
                          <section key={sh.id} aria-label={titled.has(k.id) ? sh.label : undefined} className="space-y-5">
                            {!family && titled.has(k.id) && (
                              <div className="px-0.5">
                                <h4 className="text-sm font-semibold text-zinc-200">{sh.label}</h4>
                                <p className="text-xs text-zinc-500">{sh.hint}</p>
                              </div>
                            )}
                            {sh.families.map(([fam, list]) => {
                              const look = familyLook(fam)
                              const FamIcon = look.icon
                              return (
                                <section key={fam} aria-label={fam}>
                                  <h5 className="mb-2 flex items-center gap-1.5 px-0.5 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                                    <FamIcon className={`h-3.5 w-3.5 ${look.text}`} /> {fam}
                                    <span className="font-medium tabular-nums text-zinc-600">{list.length}</span>
                                  </h5>
                                  <ul className={CARD_GRID}>
                                    {list.map((a) => (
                                      <li key={a.id}>
                                        <ArchetypeCard a={a} active={current?.id === a.id} disabled={busy} tier={tierLook(a, registry)} onPick={() => pick(a)} onShow={isMobile ? undefined : () => setPicked(a)} />
                                      </li>
                                    ))}
                                  </ul>
                                </section>
                              )
                            })}
                          </section>
                        ))}
                      </section>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* ── Detail ── */}
            {(phoneDetail || !isMobile) && (
              <div className={`flex min-h-0 flex-1 flex-col ${isMobile ? '' : 'wide:w-80 wide:flex-none wide:border-l wide:border-zinc-800 xl:w-[22rem]'}`}>
                {current ? (
                  <ArchetypeDetail key={current.id} a={current} registry={registry}
                    needsConnection={mail?.connected === false && worksInMail(current)}
                    microsoft={!!mail?.microsoft} onConnect={onConnect} />
                ) : !loadErr && archetypes === null ? (
                  <div role="presentation" className="flex-1 space-y-3 p-5">
                    <div className="fd-shimmer h-12 w-12 rounded-2xl" />
                    <div className="fd-shimmer h-6 w-2/3 rounded-lg" />
                    <div className="fd-shimmer h-14 rounded-xl" />
                    <div className="fd-shimmer h-36 rounded-2xl" />
                  </div>
                ) : (
                  <div className="flex flex-1 flex-col items-center justify-center px-8 text-center">
                    <span className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-zinc-800/80 text-zinc-500">
                      <Bot className="h-7 w-7" />
                    </span>
                    <p className="text-sm font-medium text-zinc-300">Pick an archetype</p>
                    <p className="mt-1 text-xs leading-relaxed text-zinc-500">
                      It brings its own role, tools and model. You'll see what it does and what it can reach here, then give it a name.
                    </p>
                  </div>
                )}

                {current && (
                  <form className="shrink-0 space-y-2.5 border-t border-zinc-800 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:px-5" onSubmit={(e) => { e.preventDefault(); create() }}>
                    {err && <ErrorNote>{err}</ErrorNote>}
                    <div>
                      <label className={labelCls} htmlFor="new-agent-name">Agent name</label>
                      <input
                        id="new-agent-name"
                        ref={nameRef}
                        value={name}
                        onChange={(e) => setTypedName(e.target.value)}
                        disabled={busy}
                        placeholder="Agent name"
                        aria-label="Agent name"
                        className={`${inputCls} !text-base wide:!text-sm`}
                      />
                    </div>
                    <label className={`block cursor-pointer rounded-xl border p-3 transition-colors ${
                      setupNext ? 'border-violet-500/50 bg-violet-500/[0.07]' : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700'
                    } ${busy ? 'pointer-events-none opacity-60' : ''}`}>
                      <span className="flex items-center justify-between gap-3">
                        <span className="min-w-0 text-sm font-medium text-zinc-200">Set up its way of working next</span>
                        <Switch on={setupNext} busy={busy} label="Set up its way of working next" onChange={setSetupNext} />
                      </span>
                      <span className={`mt-1 block text-xs leading-snug text-zinc-500 ${SHORT_WINDOW_HIDDEN}`}>
                        Your instructions and documents become its rules, skills and knowledge.
                      </span>
                    </label>
                    <div>
                      <Button type="submit" variant="primary" icon={setupNext ? Sparkles : Check} busy={busy} disabled={!name.trim()} className="w-full !py-2.5">
                        {busy ? 'Creating…' : createLabel(current.role, setupNext)}
                        {!busy && !isMobile && <kbd className="ml-1 rounded bg-white/15 px-1 font-sans text-[10px] font-medium">↵</kbd>}
                      </Button>
                      <p className={`mt-1.5 text-center text-[11px] text-zinc-500 ${SHORT_WINDOW_HIDDEN}`}>
                        {setupNext ? 'It starts right away; its setup opens as soon as it is created.'
                          : 'Starts in a few seconds and opens its chat by itself.'}
                      </p>
                    </div>
                  </form>
                )}
              </div>
            )}
          </div>
        </DialogFullscreen.Provider>
      </div>
    </div>,
    document.body,
  )
}

// ── Pieces ───────────────────────────────────────────────────────────

interface FamilyPick { shelf: ArchetypeCategory; name: string }

/** The archetypes under their kind, shelf and family, in the gallery's order; empty ones left out. */
function byKind(list: Archetype[]) {
  return KINDS.map((k) => {
    const inKind = shelves().filter((sh) => sh.kind === k.id).map((sh) => {
      const byFamily = new Map<string, Archetype[]>()
      for (const a of list) {
        if (archetypeCategory(a) !== sh.id) continue
        const fam = a.family || 'Other'
        byFamily.set(fam, [...(byFamily.get(fam) || []), a])
      }
      return { ...sh, families: [...byFamily.entries()].sort(([x], [y]) => x.localeCompare(y)) }
    }).filter((sh) => sh.families.length > 0)
    return { ...k, shelves: inKind, total: inKind.reduce((n, sh) => n + sh.families.reduce((m, [, l]) => m + l.length, 0), 0) }
  }).filter((k) => k.total > 0)
}

// As many columns as the room gives: the dialog can go fullscreen.
const CARD_GRID = 'grid grid-cols-[repeat(auto-fill,minmax(14.5rem,1fr))] gap-2.5'

// On a short window the form's small print gives its room to what the agent can do.
const SHORT_WINDOW_HIDDEN = '[@media(max-height:800px)]:hidden'

/** The Create button's words: a long role would wrap it to two lines, and its name is right above. */
function createLabel(role: string, setup: boolean): string {
  const verb = setup ? 'Create and set up' : 'Create'
  return role && `${verb} ${role}`.length <= 30 ? `${verb} ${role}` : `${verb} agent`
}

function tierLook(a: Archetype, registry: ArchetypeRegistry | null) {
  const known = TIER_LOOKS[a.tier]
  if (known) return known
  return { icon: Gauge, label: registry?.tiers?.[a.tier]?.label || a.tier || 'Default', blurb: registry?.tiers?.[a.tier]?.use || '' }
}

/** A kind's heading over its cards: it stays in view while they scroll by. Its
 *  backing reaches up over the list's top padding, so no card shows above it. */
function KindBand({ k, count }: { k: KindLook; count: number }) {
  return (
    <div className="sticky -top-4 z-10 -mx-1 -mt-4 bg-zinc-950 px-1 pb-1 pt-4">
      <div className={`flex items-center gap-3 rounded-2xl border border-zinc-800 bg-gradient-to-r ${k.wash} to-transparent px-3 py-2.5`}>
        <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ring-1 ${k.bg} ${k.text} ${k.ring}`}>
          <k.icon className="h-[18px] w-[18px]" />
        </span>
        <div className="min-w-0">
          <h4 className="flex items-baseline gap-2 text-base font-semibold tracking-tight text-zinc-100">
            {k.label} <span className="text-xs font-medium tabular-nums text-zinc-500">{count}</span>
          </h4>
          <p className="truncate text-xs text-zinc-500">{k.hint}</p>
        </div>
      </div>
    </div>
  )
}

function RailItem({ active, onClick, icon: Icon, iconClass, label, count }: {
  active: boolean; onClick: () => void; icon: LucideIcon; iconClass: string; label: string; count: number
}) {
  return (
    <button
      onClick={onClick}
      aria-current={active ? 'true' : undefined}
      className={`flex w-full items-center gap-2 rounded-xl px-2.5 py-2 text-left text-[13px] leading-snug transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 ${
        active ? 'bg-violet-500/10 font-medium text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200'
      }`}
    >
      <Icon className={`h-4 w-4 shrink-0 ${iconClass}`} />
      <span className="min-w-0 flex-1">{label}</span>
      <span className={`shrink-0 text-[11px] tabular-nums ${active ? 'text-violet-600 dark:text-violet-300' : 'text-zinc-500'}`}>{count}</span>
    </button>
  )
}

function FamilyChip({ active, onClick, label, count, look }: { active: boolean; onClick: () => void; label: string; count: number; look?: FamilyLook }) {
  const Icon = look?.icon
  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 wide:py-1 ${
        active
          ? 'border-violet-500/50 bg-violet-500/15 font-medium text-violet-700 dark:text-violet-200'
          : 'border-zinc-800 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
      }`}
    >
      {Icon && <Icon className={`h-3 w-3 ${active ? 'text-violet-600 dark:text-violet-300' : look?.text}`} />}
      {label}
      <span className={`text-[10px] tabular-nums ${active ? 'text-violet-600/70 dark:text-violet-300/70' : 'text-zinc-500'}`}>{count}</span>
    </button>
  )
}

function ArchetypeCard({ a, active, disabled, tier, onPick, onShow }: {
  a: Archetype; active: boolean; disabled: boolean; tier: { icon: LucideIcon; label: string }; onPick: () => void; onShow?: () => void
}) {
  const look = archetypeLook(a.id, a.family || 'Other')
  const Icon = look.icon
  const TierIcon = tier.icon
  return (
    <button
      data-tile
      onClick={onPick}
      onFocus={onShow}
      disabled={disabled}
      aria-pressed={active}
      className={`flex h-full w-full flex-col rounded-2xl border p-3.5 text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40 disabled:opacity-60 ${
        active
          ? 'border-violet-500/60 bg-violet-500/[0.07] shadow-sm shadow-violet-500/10'
          : 'border-zinc-800 bg-zinc-900/40 hover:-translate-y-px hover:border-violet-500/40 hover:bg-zinc-900/70 hover:shadow-lg hover:shadow-black/10'
      }`}
    >
      <span className="flex w-full items-center gap-3">
        <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ring-1 ${look.bg} ${look.text} ${look.ring}`}>
          <Icon className="h-5 w-5" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="line-clamp-2 text-sm font-semibold leading-snug text-zinc-100">
            {a.role || a.id}
            {a.lead && <Star className="ml-1 inline h-3 w-3 -translate-y-px fill-amber-400 text-amber-400" aria-label="Can lead a team" />}
          </span>
          <span className="mt-0.5 flex items-center gap-1 text-[11px] text-zinc-500">
            <TierIcon className="h-3 w-3 shrink-0" /> <span className="truncate">{tier.label}</span>
          </span>
        </span>
        {active && (
          <span className="flex h-5 w-5 shrink-0 items-center justify-center self-start rounded-full bg-violet-600 text-white" aria-hidden>
            <Check className="h-3 w-3" />
          </span>
        )}
      </span>
      {a.description && <span className="mt-2.5 line-clamp-3 text-xs leading-relaxed text-zinc-500">{a.description}</span>}
    </button>
  )
}

const MAIL_TOOLS = new Set(['google_mail', 'google_calendar', 'microsoft_mail', 'microsoft_calendar'])

/** It does its work in someone's mail or calendar. */
function worksInMail(a: Archetype): boolean {
  return (a.tools || []).some((t) => MAIL_TOOLS.has(t))
}

/** It works on code (people asked hosted ones to "explore the repository"). */
function worksOnCode(a: Archetype): boolean {
  return a.family === 'Engineering' || (a.tools || []).includes('codemap')
}

function ArchetypeDetail({ a, registry, needsConnection, microsoft, onConnect }: {
  a: Archetype
  registry: ArchetypeRegistry | null
  needsConnection?: boolean
  /** Microsoft is set up on this deck (otherwise only Google is offered). */
  microsoft?: boolean
  onConnect?: (service: 'google' | 'microsoft') => void
}) {
  const look = archetypeLook(a.id, a.family || 'Other')
  const Icon = look.icon
  const kind = kindLook(archetypeKind(a))
  const tier = tierLook(a, registry)
  const TierIcon = tier.icon
  const mode = modeInfo(a.cognitive_mode)
  const disclaimer = registry?.disclaimers?.[a.disclaimer || '']
  const caps = [
    { key: 'web_search', icon: Search, label: 'Search the web' },
    { key: 'files', icon: FileText, label: 'Read and write documents' },
    ...capabilities(a),
  ]
  return (
    <div className="fd-rise min-h-0 flex-1 overflow-y-auto overscroll-contain">
      <div className={`bg-gradient-to-b ${look.wash} to-transparent px-4 pb-4 pt-5 wide:px-5`}>
        <div className="flex items-center gap-3.5">
          <span className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl ring-1 ${look.bg} ${look.text} ${look.ring}`}>
            <Icon className="h-6 w-6" />
          </span>
          <div className="min-w-0">
            <p className={`text-[11px] font-semibold uppercase tracking-wider ${look.text}`}>{a.family || 'Other'}</p>
            <h4 className="text-lg font-semibold leading-tight tracking-tight text-zinc-100">{a.role || a.id}</h4>
          </div>
        </div>
        <p className={`mt-3 inline-flex max-w-full items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ${kind.bg} ${kind.text} ${kind.ring}`}>
          <kind.icon className="h-3 w-3 shrink-0" /> <span className="truncate">{kindName(a)}</span>
        </p>
        {a.description && <p className="mt-3 text-sm leading-relaxed text-zinc-400">{a.description}</p>}
      </div>

      <div className="space-y-4 px-4 pb-5 wide:px-5">
        {needsConnection && (
          <div role="note" className="rounded-xl border border-sky-500/30 bg-sky-500/5 px-3 py-2.5 text-xs leading-relaxed text-sky-800 dark:text-sky-200">
            <p className="flex items-start gap-2">
              <Link2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span className="min-w-0">
                It works in your mail and calendar, and {microsoft ? 'neither Google nor Microsoft is' : 'Google isn\'t'} connected
                yet. Connect {microsoft ? 'one' : 'it'} first, or it can't read your inbox.
              </span>
            </p>
            {onConnect && (
              <div className="mt-2 flex flex-wrap gap-2 pl-5">
                <Button className="!px-2.5 !py-1 !text-xs" onClick={() => onConnect('google')}>Connect Google</Button>
                {microsoft && <Button className="!px-2.5 !py-1 !text-xs" onClick={() => onConnect('microsoft')}>Connect Microsoft</Button>}
              </div>
            )}
          </div>
        )}
        {worksOnCode(a) && (
          <p role="note" className="flex items-start gap-2 rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-2.5 text-xs leading-relaxed text-zinc-400">
            <FolderUp className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span className="min-w-0">It works on code you give it: attach files or a .zip to a chat message. For a folder on your own computer, use the Captain desktop app.</span>
          </p>
        )}
        {disclaimer && (
          <p role="note" className="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/5 px-3 py-2.5 text-xs leading-relaxed text-amber-800 dark:text-amber-200">
            {a.disclaimer === 'legal' ? <Scale className="mt-0.5 h-3.5 w-3.5 shrink-0" /> : <HeartPulse className="mt-0.5 h-3.5 w-3.5 shrink-0" />}
            <span className="min-w-0">{disclaimer}</span>
          </p>
        )}
        <dl className="divide-y divide-zinc-800/70 rounded-2xl border border-zinc-800 bg-zinc-900/40">
          <DetailRow icon={<TierIcon className="h-4 w-4" />} label="Runs on" title={tier.label} sub={tier.blurb} />
          <DetailRow
            icon={<span className={`block h-2.5 w-2.5 rounded-full ${mode.color}`} />}
            label={`How it thinks · ${mode.label}`}
            title={mode.character}
            // Modes are described for work: "team dynamics" says nothing about a journal.
            sub={archetypeCategory(a) === 'personal' ? '' : firstSentence(mode.description)}
          />
        </dl>

        <section aria-label="What it can do">
          <h5 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">What it can do</h5>
          <ul className="grid grid-cols-2 gap-x-3 gap-y-2">
            {caps.map((c) => (
              <li key={c.key} className="flex min-w-0 items-start gap-2 text-xs leading-snug text-zinc-300">
                <c.icon className="mt-px h-3.5 w-3.5 shrink-0 text-zinc-500" />
                <span className="min-w-0">{c.label}</span>
              </li>
            ))}
          </ul>
          {a.lead && (
            <p className="mt-3 flex items-center gap-1.5 text-xs text-zinc-500">
              <Star className="h-3 w-3 shrink-0 fill-amber-400 text-amber-400" /> Can lead other agents when they work as a team.
            </p>
          )}
        </section>
      </div>
    </div>
  )
}

function DetailRow({ icon, label, title, sub }: { icon: ReactNode; label: string; title: string; sub: string }) {
  return (
    <div className="flex items-start gap-3 p-3">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800/80 text-zinc-300">{icon}</span>
      <div className="min-w-0">
        <dt className="text-[11px] text-zinc-500">{label}</dt>
        <dd className="text-sm font-medium text-zinc-100">{title}</dd>
        {sub && <dd className="mt-0.5 text-xs leading-snug text-zinc-500">{sub}</dd>}
      </div>
    </div>
  )
}
