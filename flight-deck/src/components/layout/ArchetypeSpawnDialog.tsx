import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  ArrowLeft, BarChart3, Bot, Brain, Briefcase, Calendar, Check, ChevronRight, ClipboardList, Clock, Code2, Contact,
  Database, Eye, FileSpreadsheet, FileText, Gauge, Globe, Image as ImageIcon, Library, Lightbulb, ListTodo,
  Loader2, Mail, Map as MapIcon, Mic, Monitor, PenLine, Presentation, Search, ScanText, SlidersHorizontal, Sparkles,
  Star, Telescope, TerminalSquare, Wand2, X, Zap,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { prepareAgentPreset } from '../../services/agentPreset'
import { listArchetypes } from '../../services/archetypes'
import { MODES } from '../../services/cognitiveModes'
import { spawnArchetypeProcess } from '../../services/docker'
import type { Archetype, ArchetypeRegistry } from '../../services/tierConfig'
import { useProcessStore } from '../../stores/processStore'
import { useIsMobile } from '../../hooks/useMediaQuery'
import { AdvancedEditor, PresetReview, keptCount, reviewedPreset, startReview, type Review } from './ArchetypeAdvancedSetup'

// ── Kiosk "New agent" picker ─────────────────────────────────────────
//
// The locked Simple layout has no Spawner, so this is the one way a kiosk user
// creates an agent: pick a Library archetype, name it, done. Everything else —
// tools, cognitive mode, model — is resolved by FD from the archetype and the
// account's tier set (or the admin's team default), so no model config or API
// key is exposed here.
//
// Layout: a gallery of archetype cards (search + family filter) on the left,
// and a detail panel on the right that explains the pick — what it does, how
// it thinks, which model class it runs on, what it can reach — with the name
// and Create button under it. On a phone the two are steps: the gallery, then
// the detail with a Back arrow.
//
// Advanced (optional, under the name): extra instructions and documents. Your
// model reads them into a preset — rules, the requests it handles, facts,
// when to ask you, templates, a glossary, the documents themselves — which
// takes the whole dialog for review before Create; the spawn seeds what was
// kept (see ArchetypeAdvancedSetup, backend agent_preset).

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

interface FamilyLook { icon: LucideIcon; text: string; bg: string; ring: string }

const FAMILY_LOOKS: Record<string, FamilyLook> = {
  'Research & Intelligence': { icon: Telescope, text: 'text-sky-400', bg: 'bg-sky-500/10', ring: 'ring-sky-500/30' },
  'Writing & Comms': { icon: PenLine, text: 'text-rose-400', bg: 'bg-rose-500/10', ring: 'ring-rose-500/30' },
  'Engineering': { icon: Code2, text: 'text-emerald-400', bg: 'bg-emerald-500/10', ring: 'ring-emerald-500/30' },
  'Data & Analysis': { icon: BarChart3, text: 'text-amber-400', bg: 'bg-amber-500/10', ring: 'ring-amber-500/30' },
  'Ops & Coordination': { icon: ClipboardList, text: 'text-orange-400', bg: 'bg-orange-500/10', ring: 'ring-orange-500/30' },
  'Investment & VC': { icon: Briefcase, text: 'text-indigo-400', bg: 'bg-indigo-500/10', ring: 'ring-indigo-500/30' },
  'Vision & Multimodal': { icon: Eye, text: 'text-fuchsia-400', bg: 'bg-fuchsia-500/10', ring: 'ring-fuchsia-500/30' },
}
const DEFAULT_LOOK: FamilyLook = { icon: Bot, text: 'text-violet-400', bg: 'bg-violet-500/10', ring: 'ring-violet-500/30' }

function familyLook(family: string): FamilyLook {
  return FAMILY_LOOKS[family] || DEFAULT_LOOK
}

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
const BASE_TOOLS = new Set(['shell', 'read', 'write', 'glob', 'edit', 'web_fetch', 'web_search', 'personality', 'playbooks', 'scripts'])

const CAPABILITIES: Record<string, { icon: LucideIcon; label: string }> = {
  browser: { icon: Globe, label: 'Browse the web' },
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

// ── Component ────────────────────────────────────────────────────────

export function ArchetypeSpawnDialog({ takenSlugs, onClose, onSpawned, asUser, ownerLabel }: {
  /** Slugs of the agents this user can see — the default name avoids them. */
  takenSlugs: Set<string>
  onClose: () => void
  onSpawned: (slug: string) => void
  /** Admin creating the agent FOR that user: their gallery, their ownership,
   *  their tier set and plan. */
  asUser?: string
  ownerLabel?: string
}) {
  const [registry, setRegistry] = useState<ArchetypeRegistry | null>(null)
  const [loadErr, setLoadErr] = useState('')
  const [query, setQuery] = useState('')
  const [family, setFamily] = useState<string | null>(null)
  const [picked, setPicked] = useState<Archetype | null>(null)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  // Advanced: what the person adds, the review once their model has read it,
  // and whether the review is on screen (Back keeps it, an edit drops it).
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [advInstructions, setAdvInstructions] = useState('')
  const [advFiles, setAdvFiles] = useState<File[]>([])
  const [review, setReview] = useState<Review | null>(null)
  const [reviewing, setReviewing] = useState(false)
  const [preparing, setPreparing] = useState(false)
  const nameRef = useRef<HTMLInputElement>(null)
  const { isMobile } = useIsMobile()
  const archetypes = registry?.archetypes ?? null

  useEffect(() => {
    listArchetypes(asUser)
      .then((reg) => setRegistry({ ...reg, archetypes: reg.archetypes || [] }))
      .catch((e) => setLoadErr(e instanceof Error ? e.message : String(e)))
  }, [asUser])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || busy || preparing) return
      if (reviewing) { setReviewing(false); return }
      // On a phone the detail is a step of its own: Escape walks back first.
      if (isMobile && picked) { setPicked(null); return }
      onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, preparing, reviewing, onClose, isMobile, picked])

  const families = useMemo(() => {
    const counts = new Map<string, number>()
    for (const a of archetypes || []) {
      const fam = a.family || 'Other'
      counts.set(fam, (counts.get(fam) || 0) + 1)
    }
    return [...counts.entries()].sort(([x], [y]) => x.localeCompare(y))
  }, [archetypes])

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    return (archetypes || []).filter((a) => {
      if (family && (a.family || 'Other') !== family) return false
      if (!q) return true
      return a.role.toLowerCase().includes(q) || a.id.toLowerCase().includes(q)
        || a.family.toLowerCase().includes(q) || a.description.toLowerCase().includes(q)
        || (a.keywords || []).some((k) => k.toLowerCase().includes(q))
        || capabilities(a).some((c) => c.label.toLowerCase().includes(q))
    })
  }, [archetypes, query, family])

  const groups = useMemo(() => {
    const byFamily = new Map<string, Archetype[]>()
    for (const a of shown) {
      const fam = a.family || 'Other'
      byFamily.set(fam, [...(byFamily.get(fam) || []), a])
    }
    return [...byFamily.entries()].sort(([x], [y]) => x.localeCompare(y))
  }, [shown])

  const pick = (a: Archetype) => {
    setPicked(a)
    setName(uniqueName(a.role || a.id, takenSlugs))
    setErr('')
    // The person's material stays; what was read for another archetype doesn't.
    if (picked?.id !== a.id) setReview(null)
  }

  const hasAdvanced = !!advInstructions.trim() || advFiles.length > 0
  const editAdvanced = (instructions: string, files: File[]) => {
    setAdvInstructions(instructions)
    setAdvFiles(files)
    setReview(null)
  }

  const prepare = async () => {
    if (!picked || preparing || busy) return
    setPreparing(true)
    setErr('')
    try {
      const res = await prepareAgentPreset({ archetype: picked.id, instructions: advInstructions, files: advFiles, asUser })
      setReview(startReview(res))
      setReviewing(true)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setPreparing(false)
    }
  }

  // Focus the name once the pick has re-rendered it, so Enter creates. Not on
  // a phone: the keyboard would cover the detail the user just opened.
  useEffect(() => { if (picked && !isMobile) nameRef.current?.select() }, [picked, isMobile])

  const create = async () => {
    if (!picked || !name.trim() || busy || preparing) return
    setBusy(true)
    setErr('')
    let finalName = name.trim()
    const preset = review && hasAdvanced ? reviewedPreset(review) : undefined
    try {
      let res
      try {
        res = await spawnArchetypeProcess(picked.id, finalName, picked.description, asUser, preset)
      } catch (e) {
        // Taken by an agent this user can't see (another account's): try once
        // more with a short suffix rather than make a kiosk user guess.
        const msg = e instanceof Error ? e.message : String(e)
        if (!/already (exists|running)/i.test(msg)) throw e
        finalName = `${finalName} ${Math.random().toString(36).slice(2, 6)}`
        res = await spawnArchetypeProcess(picked.id, finalName, picked.description, asUser, preset)
      }
      if (!res.ok) throw new Error(res.message)
      const slug = res.slug || agentSlug(finalName)
      // FD records the instructions for the owner (the archetype's, plus the
      // Advanced setup's rules); mirror what it stored — never the archetype's
      // own text over it — and only when the agent is the viewer's own.
      if (!asUser) {
        const procs = useProcessStore.getState()
        const stored = res.fleet_instructions || picked.fleet_instructions
        if (stored) procs.setFleetInstructions(slug, stored)
        procs.fetchProcesses()
      }
      onSpawned(slug)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  const phoneDetail = isMobile && !!picked
  const showReview = reviewing && !!review && !!picked
  const working = busy || preparing
  const subtitle = showReview
    ? 'Check what it starts with, then create it.'
    : phoneDetail ? 'Give it a name and create it.'
      : 'Pick an archetype — its role, tools and model come with it. You only give it a name.'

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={() => { if (!working) onClose() }}>
      <div
        role="dialog"
        aria-label="New agent from an archetype"
        className="flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-900 shadow-2xl wide:h-[min(88vh,52rem)] wide:max-w-5xl wide:rounded-2xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
          {showReview || phoneDetail ? (
            <button
              onClick={() => (showReview ? setReviewing(false) : setPicked(null))}
              disabled={working}
              aria-label={showReview ? 'Back to the setup' : 'Back to all archetypes'}
              className="-ml-2 rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
          ) : (
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
              <Library className="h-4 w-4" />
            </span>
          )}
          <div className="min-w-0">
            <h3 className="truncate text-base font-semibold text-zinc-100">
              {showReview ? `What ${name.trim() || picked?.role || 'it'} starts with` : `New agent${asUser && ownerLabel ? ` for ${ownerLabel}` : ''}`}
            </h3>
            <p className="truncate text-xs text-zinc-500">{subtitle}</p>
          </div>
          <button onClick={onClose} disabled={working} aria-label="Close" className="ml-auto rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50" title="Close">
            <X className="h-5 w-5" />
          </button>
        </div>

        {showReview && review ? (
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 wide:px-6">
              <div className="mx-auto max-w-3xl">
                <PresetReview review={review} disabled={busy} onChange={setReview} />
              </div>
            </div>
            <div className="shrink-0 border-t border-zinc-800 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:px-6">
              <div className="mx-auto flex max-w-3xl flex-col gap-2 wide:flex-row wide:items-center">
                {err && <p className="text-xs text-red-600 dark:text-red-400 wide:mr-auto">{err}</p>}
                <button
                  onClick={() => setReviewing(false)}
                  disabled={busy}
                  className="rounded-xl border border-zinc-700 px-4 py-2.5 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800 disabled:opacity-50 wide:ml-auto"
                >
                  Back to the setup
                </button>
                <button
                  onClick={create}
                  disabled={!name.trim() || busy}
                  className="flex items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
                >
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                  {busy ? 'Creating…' : `Create ${picked?.role || 'agent'} with ${keptCount(review)} item${keptCount(review) === 1 ? '' : 's'}`}
                </button>
              </div>
            </div>
          </div>
        ) : (
        <div className="flex min-h-0 flex-1 flex-col wide:flex-row">
          {/* ── Gallery ── */}
          {!phoneDetail && (
            <div className="flex min-h-0 flex-1 flex-col wide:border-r wide:border-zinc-800">
              <div className="shrink-0 space-y-2.5 px-4 pt-3 wide:px-5">
                <div className="relative">
                  <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-600" />
                  <input
                    // Not on a phone: the keyboard would cover the list first thing.
                    autoFocus={!isMobile}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                      // Enter on a narrowed list picks its first card.
                      if (e.key === 'Enter' && shown.length > 0 && query.trim()) { e.preventDefault(); pick(shown[0]) }
                    }}
                    placeholder="Search by role, task or skill…"
                    aria-label="Search archetypes"
                    className="w-full rounded-xl border border-zinc-700 bg-zinc-950 py-2.5 pl-10 pr-9 text-base text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none wide:py-2 wide:text-sm"
                  />
                  {query && (
                    <button onClick={() => setQuery('')} aria-label="Clear search" className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                      <X className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
                {families.length > 1 && (
                  <div className="-mx-4 flex gap-1.5 overflow-x-auto px-4 pb-1 wide:-mx-5 wide:flex-wrap wide:px-5 wide:pb-0 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden" role="tablist" aria-label="Archetype families">
                    <FamilyChip active={family === null} onClick={() => setFamily(null)} label="All" count={archetypes?.length || 0} />
                    {families.map(([fam, n]) => (
                      <FamilyChip key={fam} active={family === fam} onClick={() => setFamily(family === fam ? null : fam)} label={fam} count={n} look={familyLook(fam)} />
                    ))}
                  </div>
                )}
              </div>

              <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 wide:px-5">
                {loadErr && <p className="py-10 text-center text-sm text-red-600 dark:text-red-400">Couldn't load archetypes: {loadErr}</p>}
                {!loadErr && archetypes === null && (
                  <p className="flex items-center justify-center gap-2 py-10 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</p>
                )}
                {archetypes && groups.length === 0 && (
                  <div className="py-10 text-center">
                    <p className="text-sm text-zinc-400">
                      {archetypes.length === 0 ? 'No archetypes available. Ask an administrator to add some.' : `Nothing matches “${query.trim()}”${family ? ` in ${family}` : ''}.`}
                    </p>
                    {archetypes.length > 0 && (
                      <button onClick={() => { setQuery(''); setFamily(null) }} className="mt-2 text-xs text-violet-400 hover:underline">Show everything</button>
                    )}
                  </div>
                )}
                {groups.map(([fam, list]) => {
                  const look = familyLook(fam)
                  const FamIcon = look.icon
                  return (
                    <div key={fam} className="mb-5 last:mb-0">
                      {!family && (
                        <div className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                          <FamIcon className={`h-3 w-3 ${look.text}`} /> {fam}
                        </div>
                      )}
                      <ul className="grid grid-cols-1 gap-2 md:grid-cols-2">
                        {list.map((a) => (
                          <li key={a.id}>
                            <ArchetypeCard a={a} active={picked?.id === a.id} disabled={busy} look={look} tierLabel={tierLook(a, registry).label} onPick={() => pick(a)} />
                          </li>
                        ))}
                      </ul>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* ── Detail ── */}
          {(phoneDetail || !isMobile) && (
            <div className={`flex min-h-0 flex-1 flex-col bg-zinc-950/40 ${isMobile ? '' : 'wide:w-[22rem] wide:flex-none lg:w-[24rem]'}`}>
              {!picked ? (
                <div className="flex flex-1 flex-col items-center justify-center px-8 text-center">
                  <span className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-zinc-800/80 text-zinc-500">
                    <Bot className="h-7 w-7" />
                  </span>
                  <p className="text-sm font-medium text-zinc-300">Pick an archetype</p>
                  <p className="mt-1 text-xs leading-relaxed text-zinc-500">
                    It brings its own role, tools and model. You'll see what it does and what it can reach here, then give it a name.
                  </p>
                </div>
              ) : advancedOpen ? (
                <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 wide:px-5">
                  <AdvancedEditor
                    instructions={advInstructions}
                    files={advFiles}
                    disabled={working}
                    onInstructions={(v) => editAdvanced(v, advFiles)}
                    onFiles={(f) => editAdvanced(advInstructions, f)}
                  />
                </div>
              ) : (
                <ArchetypeDetail a={picked} registry={registry} />
              )}

              {picked && (
                <div className="shrink-0 border-t border-zinc-800 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:px-5">
                  {err && <p className="mb-2 text-xs text-red-600 dark:text-red-400">{err}</p>}
                  <form className="space-y-2" onSubmit={(e) => {
                    e.preventDefault()
                    if (hasAdvanced && !review) prepare()
                    else if (hasAdvanced && review) setReviewing(true)
                    else create()
                  }}>
                    <label className="block">
                      <span className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Agent name</span>
                      <input
                        ref={nameRef}
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        disabled={working}
                        placeholder="Agent name"
                        aria-label="Agent name"
                        className="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2.5 text-base text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none disabled:opacity-60 wide:py-2 wide:text-sm"
                      />
                    </label>
                    <button
                      type="button"
                      onClick={() => setAdvancedOpen((o) => !o)}
                      disabled={working}
                      aria-expanded={advancedOpen}
                      className={`flex w-full items-center gap-2 rounded-xl border px-3 py-2 text-left text-sm transition-colors disabled:opacity-60 ${
                        advancedOpen || hasAdvanced
                          ? 'border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-200'
                          : 'border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
                      }`}
                    >
                      <SlidersHorizontal className="h-4 w-4 shrink-0" />
                      <span className="min-w-0 flex-1 truncate">
                        {advancedOpen ? 'Back to the archetype' : 'Advanced — your way of working'}
                      </span>
                      {!advancedOpen && hasAdvanced && (
                        <span className="shrink-0 text-[11px] opacity-80">
                          {advFiles.length > 0 ? `${advFiles.length} doc${advFiles.length === 1 ? '' : 's'}` : 'instructions'}
                        </span>
                      )}
                      <ChevronRight className={`h-4 w-4 shrink-0 transition-transform ${advancedOpen ? 'rotate-180' : ''}`} />
                    </button>
                    <button
                      type="submit"
                      disabled={!name.trim() || working}
                      className="flex w-full items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
                    >
                      {working ? <Loader2 className="h-4 w-4 animate-spin" /> : hasAdvanced ? <Sparkles className="h-4 w-4" /> : <Check className="h-4 w-4" />}
                      {preparing ? 'Reading your material…'
                        : busy ? 'Creating…'
                          : hasAdvanced ? (review ? 'Review the setup' : 'Read it and review')
                            : `Create ${picked.role || 'agent'}`}
                    </button>
                    <p className="text-center text-[11px] text-zinc-600">
                      {preparing ? 'Your model is reading it — this can take a minute or two.'
                        : hasAdvanced ? 'You\'ll see what it learned before the agent is created.'
                          : 'Starts in a few seconds and opens its chat by itself.'}
                    </p>
                  </form>
                </div>
              )}
            </div>
          )}
        </div>
        )}
      </div>
    </div>,
    document.body,
  )
}

// ── Pieces ───────────────────────────────────────────────────────────

function tierLook(a: Archetype, registry: ArchetypeRegistry | null) {
  const known = TIER_LOOKS[a.tier]
  if (known) return known
  return { icon: Gauge, label: registry?.tiers?.[a.tier]?.label || a.tier || 'Default', blurb: registry?.tiers?.[a.tier]?.use || '' }
}

function FamilyChip({ active, onClick, label, count, look }: { active: boolean; onClick: () => void; label: string; count: number; look?: FamilyLook }) {
  const Icon = look?.icon
  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition-colors wide:py-1 ${
        active
          ? 'border-violet-500/50 bg-violet-500/15 text-violet-700 dark:text-violet-200'
          : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
      }`}
    >
      {Icon && <Icon className={`h-3 w-3 ${active ? 'text-violet-600 dark:text-violet-300' : look?.text}`} />}
      {label}
      <span className={`text-[10px] ${active ? 'text-violet-600/70 dark:text-violet-300/70' : 'text-zinc-600'}`}>{count}</span>
    </button>
  )
}

function ArchetypeCard({ a, active, disabled, look, tierLabel, onPick }: {
  a: Archetype; active: boolean; disabled: boolean; look: FamilyLook; tierLabel: string; onPick: () => void
}) {
  const Icon = look.icon
  const mode = modeInfo(a.cognitive_mode)
  return (
    <button
      onClick={onPick}
      disabled={disabled}
      aria-pressed={active}
      className={`group flex h-full w-full gap-3 rounded-xl border p-3 text-left transition-all ${
        active
          ? 'border-violet-500/60 bg-violet-500/10 ring-1 ring-violet-500/40'
          : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700 hover:bg-zinc-800/60'
      } disabled:opacity-60`}
    >
      <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${look.bg} ${look.text}`}>
        <Icon className="h-5 w-5" />
      </span>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-sm font-medium text-zinc-100">{a.role || a.id}</span>
          {a.lead && <Star className="h-3 w-3 shrink-0 text-amber-400" aria-label="Can lead a team" />}
          {active && <Check className="ml-auto h-4 w-4 shrink-0 text-violet-400" aria-hidden />}
        </div>
        {a.description && <p className="mt-0.5 line-clamp-2 text-xs leading-snug text-zinc-500">{a.description}</p>}
        <div className="mt-auto flex flex-wrap items-center gap-1 pt-2">
          <span className="rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">{tierLabel}</span>
          <span className="flex items-center gap-1 rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
            <span className={`h-1.5 w-1.5 rounded-full ${mode.color}`} /> {mode.character}
          </span>
        </div>
      </div>
    </button>
  )
}

function ArchetypeDetail({ a, registry }: { a: Archetype; registry: ArchetypeRegistry | null }) {
  const look = familyLook(a.family || 'Other')
  const Icon = look.icon
  const tier = tierLook(a, registry)
  const TierIcon = tier.icon
  const mode = modeInfo(a.cognitive_mode)
  const caps = capabilities(a)
  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 wide:px-5">
      <div className="flex items-start gap-3">
        <span className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl ring-1 ${look.bg} ${look.text} ${look.ring}`}>
          <Icon className="h-6 w-6" />
        </span>
        <div className="min-w-0">
          <p className={`text-[11px] font-semibold uppercase tracking-wider ${look.text}`}>{a.family || 'Other'}</p>
          <h4 className="flex items-center gap-1.5 text-lg font-semibold leading-tight text-zinc-100">
            {a.role || a.id}
            {a.lead && <Star className="h-3.5 w-3.5 shrink-0 text-amber-400" aria-label="Can lead a team" />}
          </h4>
        </div>
      </div>
      {a.description && <p className="mt-3 text-sm leading-relaxed text-zinc-300">{a.description}</p>}

      <dl className="mt-4 space-y-2">
        <DetailRow icon={<TierIcon className="h-4 w-4" />} title={tier.label} sub={tier.blurb} label="Runs on" />
        <DetailRow
          icon={<span className={`block h-2.5 w-2.5 rounded-full ${mode.color}`} />}
          title={`${mode.label} · ${mode.character}`}
          sub={firstSentence(mode.description)}
          label="Thinks"
        />
      </dl>

      <div className="mt-4">
        <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500">What it can do</p>
        <div className="flex flex-wrap gap-1.5">
          <Chip icon={Search} label="Search the web" muted />
          <Chip icon={FileText} label="Work with files" muted />
          {caps.map((c) => <Chip key={c.key} icon={c.icon} label={c.label} />)}
        </div>
        {a.lead && (
          <p className="mt-2 flex items-center gap-1.5 text-[11px] text-zinc-500">
            <Star className="h-3 w-3 text-amber-400" /> Can lead other agents when they work as a team.
          </p>
        )}
      </div>
    </div>
  )
}

function DetailRow({ icon, title, sub, label }: { icon: ReactNode; title: string; sub: string; label: string }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2.5">
      <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center text-zinc-400">{icon}</span>
      <div className="min-w-0">
        <dt className="text-[10px] font-medium uppercase tracking-wider text-zinc-600">{label}</dt>
        <dd className="text-sm font-medium text-zinc-200">{title}</dd>
        {sub && <dd className="text-xs leading-snug text-zinc-500">{sub}</dd>}
      </div>
    </div>
  )
}

function Chip({ icon: Icon, label, muted }: { icon: LucideIcon; label: string; muted?: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-xs ${
      muted ? 'border-zinc-800/80 text-zinc-500' : 'border-zinc-700/80 bg-zinc-800/60 text-zinc-300'
    }`}>
      <Icon className="h-3 w-3" /> {label}
    </span>
  )
}
