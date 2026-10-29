import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertTriangle, ArrowLeft, BookA, BookOpen, Check, ChevronLeft, ChevronRight, CircleDashed, FileText, Layers,
  ListTree, Loader2, Map as MapIcon, ScrollText, Search, Sparkles, Table2, Wand2, X,
} from 'lucide-react'
import {
  agentKnowledgeApi, type KnowledgeDoc, type KnowledgeDocDetail, type KnowledgeHit, type KnowledgeOverview,
  type KnowledgeSection, type KnowledgeSectionText,
} from '../../services/agentKnowledge'
import type { ReadState } from '../../services/agentPreset'
import { SearchBox } from '../admin/ListFilters'
import { fmtBytes } from '../layout/agentSetup'
import { FullscreenButton, Highlight, libraryIconBtn as iconBtn } from './libraryParts'

// ── An agent's knowledge, to browse ───────────────────────────────────
//
// The Knowledge item in the kiosk's agent menu opens this on one agent: the
// documents its owner's setup gave it (its knowledge/ folder), read through
// /fd/agent-knowledge/{slug} — so it works while the agent is off.
//
//   documents — each with how much of it the model read (a meter), then the
//               glossary and the owner's rules;
//   sections  — the document's headings as a tree, each marked read by the
//               model or not, folded when there are many, with a filter;
//   reader    — a document's landing page, or one section's own text: where
//               it is in the files, its subsections, the next and previous.
//
// Search looks through all of it (or one document) and opens the section a
// passage is in, scrolled to it. On a phone it is one pane at a time.

type Pane = 'docs' | 'sections' | 'reader'
/** A request's answer — or why it failed — and which request it was (``key``). */
type Got<T> = { key: string; data?: T; error?: string }
type Picked = { kind: 'doc'; id: string } | { kind: 'extra'; id: string }

const FOLD_OVER = 80          // a document with more sections opens with its deeper ones folded
const FILTER_ROWS = 300

const kChars = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}k` : `${n}`)
const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`
const message = (e: unknown) => (e instanceof Error ? e.message : String(e))
/** A heading as words: its links' text, without pictures, badges (one cut short too), HTML, code ticks or emphasis. */
const plain = (title: string) => title
  .replace(/<[^>]*>?/g, '')
  .replace(/\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)/g, '')
  .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
  .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
  .replace(/\[?!\[.*$/, '')
  .replace(/`+([^`]+)`+/g, '$1')
  .replace(/(^|[\s(])(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\2(?=$|[\s.,;:!?)])/g, '$1$3')
  .replace(/\s+/g, ' ').trim() || title

// The owner's documents: links open in a new tab; pictures from elsewhere aren't fetched.
const MD: Components = {
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer noopener" className="text-violet-600 underline underline-offset-2 dark:text-violet-400">{children}</a>
  ),
  img: ({ alt }) => (alt ? <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[11px] text-zinc-500">[picture: {alt}]</span> : null),
}

/** How much of a document the model read, as a bar and a few words. */
function readLook(read: ReadState): { pct: number; bar: string; text: string; tone: string } {
  const pct = read.of ? Math.round((read.sections / read.of) * 100) : read.state === 'full' ? 100 : 0
  switch (read.state) {
    case 'full': return { pct: 100, bar: 'bg-emerald-500', text: 'All read', tone: 'text-emerald-700 dark:text-emerald-400' }
    case 'partial': return { pct: Math.max(3, pct), bar: 'bg-amber-500', text: `${pct}% read`, tone: 'text-amber-700 dark:text-amber-400' }
    case 'none': return { pct: 0, bar: 'bg-zinc-600', text: 'Not read', tone: 'text-zinc-500' }
    default: return { pct: 0, bar: 'bg-zinc-600', text: 'Read before tracking', tone: 'text-zinc-500' }
  }
}

function ReadMeter({ read, wide = false }: { read: ReadState; wide?: boolean }) {
  const look = readLook(read)
  return (
    <span className={`flex items-center gap-2 ${wide ? 'w-full' : ''}`}>
      <span className={`h-1 overflow-hidden rounded-full bg-zinc-800 ${wide ? 'flex-1' : 'w-14 shrink-0'}`}>
        <span className={`block h-full rounded-full ${look.bar}`} style={{ width: `${look.pct}%` }} />
      </span>
      <span className={`shrink-0 text-[11px] ${look.tone}`}>{look.text}</span>
    </span>
  )
}

/** A section's read mark: read by the model, not read, or nothing (a bare heading, or not tracked). */
function ReadDot({ read }: { read: boolean | null }) {
  if (read === null) return <span className="h-1.5 w-1.5 shrink-0" />
  return read
    ? <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" title="Read by the model" />
    : <span className="h-1.5 w-1.5 shrink-0 rounded-full ring-1 ring-inset ring-zinc-600" title="Not read — searchable" />
}

export function KnowledgeBrowser({ agent, phone = false, embedded = false, onClose, onSetup }: {
  agent: { slug: string; name: string }
  phone?: boolean
  /** Inside another dialog (the Onboarding page's Materials): no header of its own, and Escape doesn't close it. */
  embedded?: boolean
  onClose: () => void
  /** Open the agent's Way of working — to give it documents, or read more of them. */
  onSetup?: () => void
}) {
  const [overview, setOverview] = useState<KnowledgeOverview | null>(null)
  const [error, setError] = useState('')
  const [tries, setTries] = useState(0)
  const [picked, setPicked] = useState<Picked | null>(null)
  const [sectionI, setSectionI] = useState<number | null>(null)
  const [page, setPage] = useState(1)
  // What came back, each kept with the request it answers ("loading" is a request without one yet).
  const [docGot, setDocGot] = useState<Got<KnowledgeDocDetail> | null>(null)
  const [sectionGot, setSectionGot] = useState<Got<KnowledgeSectionText> | null>(null)
  const [extraGot, setExtraGot] = useState<Got<{ title: string; text: string }> | null>(null)
  const [hitsGot, setHitsGot] = useState<Got<KnowledgeHit[]> | null>(null)
  const [folded, setFolded] = useState<Set<number>>(new Set())
  const [filter, setFilter] = useState('')
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [scope, setScope] = useState<'all' | 'doc'>('all')
  const [focus, setFocus] = useState('')
  const [pane, setPane] = useState<Pane>('docs')
  const searchBox = useRef<HTMLDivElement>(null)
  const readerBox = useRef<HTMLDivElement>(null)
  const outlineBox = useRef<HTMLDivElement>(null)

  // ── Loading ─────────────────────────────────────────────────────────

  useEffect(() => {
    let live = true
    agentKnowledgeApi.overview(agent.slug).then((o) => {
      if (!live) return
      setOverview(o)
      // A wide screen opens on the first document; a phone on the list.
      if (!phone && o.documents.length) setPicked((cur) => cur ?? { kind: 'doc', id: o.documents[0].id })
    }).catch((e) => { if (live) setError(message(e)) })
    return () => { live = false }
  }, [agent.slug, phone, tries])
  const retry = () => { setError(''); setTries((n) => n + 1) }

  const docId = picked?.kind === 'doc' ? picked.id : ''
  useEffect(() => {
    if (!docId) return
    let live = true
    agentKnowledgeApi.doc(agent.slug, docId).then((d) => {
      if (!live) return
      // Many sections: the deeper ones folded, so the chapters show at a glance.
      const deep = new Set<number>()
      if (d.sections.length > FOLD_OVER) {
        d.sections.forEach((s, n) => {
          const next = d.sections[n + 1]
          if (s.level >= 2 && next && next.level > s.level) deep.add(s.i)
        })
      }
      setFolded(deep)
      setDocGot({ key: docId, data: d })
    }).catch((e) => { if (live) setDocGot({ key: docId, error: message(e) }) })
    return () => { live = false }
  }, [agent.slug, docId])
  const detail = docGot?.key === docId ? docGot.data ?? null : null
  const detailError = docGot?.key === docId ? docGot.error ?? '' : ''

  const sectionKey = docId && sectionI !== null ? `${docId}|${sectionI}|${page}` : ''
  useEffect(() => {
    if (!sectionKey || sectionI === null) return
    let live = true
    agentKnowledgeApi.section(agent.slug, docId, sectionI, page)
      .then((s) => { if (live) setSectionGot({ key: sectionKey, data: s }) })
      .catch((e) => { if (live) setSectionGot({ key: sectionKey, error: message(e) }) })
    return () => { live = false }
  }, [agent.slug, docId, sectionI, page, sectionKey])
  // While the next section loads, the one shown stays (dimmed) — no flash of nothing.
  const section = sectionGot?.data && sectionKey && sectionGot.data.id === docId ? sectionGot.data : null

  const extraId = picked?.kind === 'extra' ? picked.id : ''
  useEffect(() => {
    if (!extraId) return
    let live = true
    agentKnowledgeApi.extra(agent.slug, extraId)
      .then((x) => { if (live) setExtraGot({ key: extraId, data: x }) })
      .catch((e) => { if (live) setExtraGot({ key: extraId, error: message(e) }) })
    return () => { live = false }
  }, [agent.slug, extraId])
  const extra = extraGot?.key === extraId ? extraGot.data ?? null : null

  const readerLoading = extraId ? extraGot?.key !== extraId : !!sectionKey && sectionGot?.key !== sectionKey
  const readerError = extraId ? (extraGot?.key === extraId ? extraGot.error ?? '' : '')
    : sectionGot?.key === sectionKey ? sectionGot.error ?? '' : ''

  // ── Search ──────────────────────────────────────────────────────────

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), query ? 350 : 0)
    return () => clearTimeout(t)
  }, [query])
  const searchIn = scope === 'doc' && docId ? docId : ''
  const searchOn = debounced.length >= 2
  const searchKey = searchOn ? `${searchIn}|${debounced}` : ''
  useEffect(() => {
    if (!searchKey) return
    let live = true
    agentKnowledgeApi.search(agent.slug, debounced, searchIn)
      .then((r) => { if (live) setHitsGot({ key: searchKey, data: r.results }) })
      .catch((e) => { if (live) setHitsGot({ key: searchKey, error: message(e) }) })
    return () => { live = false }
  }, [agent.slug, debounced, searchIn, searchKey])
  const searching = !!searchKey && hitsGot?.key !== searchKey
  const hits = hitsGot?.key === searchKey ? hitsGot.data ?? null : null
  const searchError = hitsGot?.key === searchKey ? hitsGot.error ?? '' : ''
  const terms = useMemo(() => debounced.toLowerCase().split(/\s+/).filter((t) => t.length > 1), [debounced])

  // ── Moving around ───────────────────────────────────────────────────

  const openDoc = (id: string) => {
    setPicked({ kind: 'doc', id })
    setSectionI(null)
    setPage(1)
    setFilter('')
    setFocus('')
    setPane('sections')
  }
  const openExtra = (id: string) => {
    setPicked({ kind: 'extra', id })
    setSectionI(null)
    setFocus('')
    setPane('reader')
  }
  /** Open section *i* — its folded parents unfolded, so it shows in the tree. */
  const openSection = (i: number, scrollTo = '') => {
    setSectionI(i)
    setPage(1)
    setFocus(scrollTo)
    setPane('reader')
    if (detail) {
      const s = detail.sections[i]
      if (s) {
        setFolded((prev) => {
          const next = new Set(prev)
          for (let n = i - 1, level = s.level; n >= 0 && level > 1; n--) {
            if (detail.sections[n].level < level) { next.delete(n); level = detail.sections[n].level }
          }
          return next
        })
      }
    }
  }
  const openHit = (h: KnowledgeHit) => {
    const scrollTo = h.text.split('\n').find((l) => l.trim().length > 12)?.trim().slice(0, 60) ?? ''
    if (h.extra) { openExtra(h.extra); setFocus(scrollTo); return }
    if (!h.id || h.i === undefined) return
    if (h.id !== docId) {
      setPicked({ kind: 'doc', id: h.id })
      setFilter('')
    }
    setSectionI(h.i)
    setPage(1)
    setFocus(scrollTo)
    setPane('reader')
  }

  // The reader scrolls to the top for a new section, or to the passage a search found, marked a moment.
  useEffect(() => {
    const box = readerBox.current
    if (!box || readerLoading) return
    if (!focus) { box.scrollTop = 0; return }
    const want = focus.toLowerCase().replace(/[*_`#>[\]]/g, '').replace(/\s+/g, ' ').slice(0, 40)
    const walker = document.createTreeWalker(box, NodeFilter.SHOW_ELEMENT)
    let found: HTMLElement | null = null
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const el = n as HTMLElement
      if (!['P', 'LI', 'TD', 'PRE', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'BLOCKQUOTE', 'CODE'].includes(el.tagName)) continue
      if ((el.textContent ?? '').toLowerCase().replace(/\s+/g, ' ').includes(want)) { found = el; break }
    }
    if (!found) { box.scrollTop = 0; return }
    found.scrollIntoView({ block: 'center' })
    found.classList.add('fd-kb-flash')
    const t = setTimeout(() => found?.classList.remove('fd-kb-flash'), 2200)
    return () => clearTimeout(t)
  }, [section, extra, focus, readerLoading])

  // The tree as shown: folded sections' insides hidden, or — filtering — the matches.
  const visible = useMemo(() => {
    if (!detail) return []
    const f = filter.trim().toLowerCase()
    if (f) return detail.sections.filter((s) => plain(s.title).toLowerCase().includes(f)).slice(0, FILTER_ROWS)
    const out: KnowledgeSection[] = []
    let hideBelow = Infinity
    for (const s of detail.sections) {
      if (s.level > hideBelow) continue
      hideBelow = folded.has(s.i) ? s.level : Infinity
      out.push(s)
    }
    return out
  }, [detail, folded, filter])
  const hasKids = (s: KnowledgeSection) => !!detail && detail.sections[s.i + 1]?.level > s.level

  // Keys: / to search, ↑ ↓ through the sections, Escape out of a search, a pane, the dialog.
  const state = useRef({ query, pane, visible, sectionI, phone, embedded })
  useEffect(() => { state.current = { query, pane, visible, sectionI, phone, embedded } })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement
      const s = state.current
      if (e.key === 'Escape') {
        if (s.query) { setQuery(''); return }
        if (s.phone && s.pane === 'reader') { setPane('sections'); return }
        if (s.phone && s.pane === 'sections') { setPane('docs'); return }
        if (!s.embedded) onClose()
        return
      }
      if (typing) return
      if (e.key === '/') {
        e.preventDefault()
        searchBox.current?.querySelector('input')?.focus()
      } else if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && s.visible.length) {
        e.preventDefault()
        const at = s.visible.findIndex((v) => v.i === s.sectionI)
        const next = s.visible[Math.max(0, Math.min(s.visible.length - 1, at + (e.key === 'ArrowDown' ? 1 : -1)))]
        if (next) {
          setSectionI(next.i)
          setPage(1)
          setFocus('')
          outlineBox.current?.querySelector(`[data-i="${next.i}"]`)?.scrollIntoView({ block: 'nearest' })
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // The tree follows the open section (opened from a search, or next/previous).
  useEffect(() => {
    if (sectionI === null) return
    outlineBox.current?.querySelector(`[data-i="${sectionI}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [sectionI, visible])

  // ── Pieces ──────────────────────────────────────────────────────────

  const totals = overview?.totals
  const header = (
    <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-sky-500/15 text-sky-600 dark:text-sky-400">
        <BookOpen className="h-[18px] w-[18px]" />
      </span>
      <div className="min-w-0 flex-1">
        <h2 className="truncate text-base font-semibold text-zinc-100">{agent.name}'s knowledge</h2>
        <p className="truncate text-xs text-zinc-500">
          {!totals ? 'The documents it was given, to search and read'
            : totals.documents === 0 ? 'No documents yet'
              : `${plural(totals.documents, 'document')} · ${fmtBytes(totals.bytes)}`
                + (totals.sections ? ` · the model read ${totals.read.toLocaleString()} of ${plural(totals.sections, 'section')}` : '')}
        </p>
      </div>
      <FullscreenButton />
      <button onClick={onClose} className={iconBtn} title="Close" aria-label="Close"><X className="h-4 w-4" /></button>
    </div>
  )

  const hasDocs = !!overview && (overview.documents.length > 0 || overview.extras.length > 0)
  const toolbar = hasDocs && (
    <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2.5 wide:px-4">
      <div ref={searchBox} className="flex min-w-0 flex-1">
        <SearchBox value={query} onChange={setQuery}
          placeholder={phone ? 'Search its knowledge…' : 'Search every document — words, a phrase, a code…  ( / )'} />
      </div>
      {docId && detail && (
        <div role="radiogroup" aria-label="Search where" className="flex shrink-0 rounded-lg border border-zinc-800 p-0.5 text-xs">
          {(['all', 'doc'] as const).map((s) => (
            <button key={s} role="radio" aria-checked={scope === s} onClick={() => setScope(s)}
              className={`rounded-md px-2.5 py-1 transition-colors ${scope === s ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}>
              {s === 'all' ? 'All' : phone ? 'This one' : 'This document'}
            </button>
          ))}
        </div>
      )}
    </div>
  )

  const docsList = overview && (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        <p className="px-2 pb-1.5 pt-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Documents</p>
        <ul className="flex flex-col gap-0.5">
          {overview.documents.map((d) => (
            <DocRow key={d.id} doc={d} active={picked?.kind === 'doc' && picked.id === d.id} onOpen={() => openDoc(d.id)} />
          ))}
        </ul>
        {overview.extras.length > 0 && (
          <>
            <p className="px-2 pb-1.5 pt-4 text-[11px] font-medium uppercase tracking-wider text-zinc-500">From the setup</p>
            <ul className="flex flex-col gap-0.5">
              {overview.extras.map((x) => (
                <li key={x.id}>
                  <button onClick={() => openExtra(x.id)}
                    className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-[13px] transition-colors ${
                      picked?.kind === 'extra' && picked.id === x.id ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-300 hover:bg-zinc-800/60'}`}>
                    {x.id === 'glossary.md' ? <BookA className="h-4 w-4 shrink-0 text-zinc-500" /> : <ScrollText className="h-4 w-4 shrink-0 text-zinc-500" />}
                    <span className="truncate">{x.title}</span>
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
      {onSetup && overview.documents.some((d) => d.read.state !== 'full') && (
        <div className="border-t border-zinc-800 p-3">
          <button onClick={onSetup}
            className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-2 text-xs text-zinc-300 transition-colors hover:border-violet-500/50 hover:text-zinc-100">
            <Wand2 className="h-3.5 w-3.5" /> Read more of them…
          </button>
          <p className="mt-1.5 text-center text-[10px] text-zinc-600">In Way of working: pick what isn't read yet.</p>
        </div>
      )}
    </div>
  )

  const docHead = detail && (
    <div className="border-b border-zinc-800 px-3 py-2.5">
      <p className="truncate text-sm font-semibold text-zinc-100" title={detail.title}>{detail.title}</p>
      <p className="mt-0.5 text-[11px] text-zinc-500">{plural(detail.sections.length, 'section')} · {fmtBytes(detail.bytes)}</p>
      <div className="mt-2"><ReadMeter read={detail.read} wide /></div>
      <div className="relative mt-2.5">
        <ListTree className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
        <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter sections"
          aria-label="Filter sections"
          className="w-full rounded-lg border border-zinc-800 bg-zinc-900 py-1.5 pl-7 pr-2 text-base text-zinc-200 placeholder-zinc-500 focus:border-sky-500/60 focus:outline-none wide:text-xs" />
      </div>
    </div>
  )

  const tree = (
    <div className="flex h-full flex-col">
      {phone && (
        <button onClick={() => setPane('docs')} className="flex items-center gap-1.5 border-b border-zinc-800 px-3 py-2 text-xs text-zinc-400">
          <ArrowLeft className="h-3.5 w-3.5" /> Documents
        </button>
      )}
      {detailError ? <Notice tone="error" text={detailError} />
        : !detail ? <Spinner />
          : (
            <>
              {docHead}
              <div ref={outlineBox} className="min-h-0 flex-1 overflow-y-auto py-1.5" role="tree" aria-label={`${detail.title}: its sections`}>
                {visible.length === 0 && <p className="px-4 py-6 text-center text-xs text-zinc-500">{filter ? 'No section is called that.' : 'It has no headings — open it to read it.'}</p>}
                {visible.map((s) => {
                  const kids = !filter && hasKids(s)
                  const open = !folded.has(s.i)
                  return (
                    <div key={s.i} data-i={s.i} role="treeitem" aria-selected={s.i === sectionI} aria-expanded={kids ? open : undefined}
                      className={`group flex items-center gap-1 pr-2 text-[13px] ${s.i === sectionI
                        ? 'bg-sky-500/10 text-zinc-100' : 'text-zinc-300 hover:bg-zinc-800/60'}`}
                      style={{ paddingLeft: filter ? 10 : 6 + Math.min(6, s.level - 1) * 14 }}>
                      {kids ? (
                        <button onClick={() => setFolded((prev) => { const n = new Set(prev); if (n.has(s.i)) n.delete(s.i); else n.add(s.i); return n })}
                          aria-label={open ? 'Fold' : 'Unfold'} className="rounded p-0.5 text-zinc-500 hover:text-zinc-200">
                          <ChevronRight className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-90' : ''}`} />
                        </button>
                      ) : <span className="w-[18px] shrink-0" />}
                      <button onClick={() => openSection(s.i)} className="flex min-w-0 flex-1 items-center gap-2 py-1.5 text-left">
                        <ReadDot read={s.read} />
                        <span className="min-w-0 flex-1 truncate" title={[...s.path, s.title].map(plain).join(' › ')}>
                          {filter ? <Highlight text={plain(s.title)} terms={[filter.toLowerCase()]} /> : plain(s.title)}
                          {filter && s.path.length > 0 && <span className="ml-1.5 text-[11px] text-zinc-500">{plain(s.path[s.path.length - 1])}</span>}
                        </span>
                        <span className="shrink-0 text-[10px] tabular-nums text-zinc-600 opacity-0 group-hover:opacity-100">{kChars(s.chars)}</span>
                      </button>
                    </div>
                  )
                })}
              </div>
              {detail.read.state !== 'unknown' && (
                <div className="flex items-center gap-3 border-t border-zinc-800 px-3 py-2 text-[10px] text-zinc-500">
                  <span className="flex items-center gap-1.5"><ReadDot read /> Read by the model</span>
                  <span className="flex items-center gap-1.5"><ReadDot read={false} /> Searchable only</span>
                </div>
              )}
            </>
          )}
    </div>
  )

  const results = (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2 text-xs text-zinc-400">
        {searching ? <Loader2 className="h-3.5 w-3.5 animate-spin text-sky-500" /> : <Search className="h-3.5 w-3.5 text-zinc-500" />}
        <span className="min-w-0 flex-1 truncate">
          {searching ? 'Searching…' : hits ? `${plural(hits.length, 'passage')} for “${debounced}”` : ''}
          {searchIn && detail ? ` in ${detail.title}` : ''}
        </span>
        <button onClick={() => setQuery('')} className="text-[11px] text-sky-700 hover:underline dark:text-sky-400">Done</button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {searchError && <Notice tone="error" text={searchError} />}
        {hits && hits.length === 0 && !searching && (
          <div className="px-4 py-10 text-center">
            <Search className="mx-auto mb-2 h-5 w-5 text-zinc-600" />
            <p className="text-sm text-zinc-300">Nothing matches “{debounced}”.</p>
            <p className="mt-1 text-xs text-zinc-500">Try other words — or a phrase in quotes, to require it.</p>
          </div>
        )}
        <ul className="flex flex-col gap-1.5">
          {(hits ?? []).map((h, n) => (
            <li key={`${h.file}:${h.first}:${n}`}>
              <button onClick={() => openHit(h)}
                className={`w-full rounded-xl border px-3 py-2.5 text-left transition-colors ${
                  h.id === docId && h.i === sectionI ? 'border-sky-500/40 bg-sky-500/5' : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700'}`}>
                <p className="flex items-center gap-1.5 truncate text-[11px] text-zinc-500">
                  {h.extra ? <BookA className="h-3 w-3 shrink-0" /> : <FileText className="h-3 w-3 shrink-0" />}
                  <span className="truncate">{h.where.split(' › ').map(plain).join(' › ')}</span>
                </p>
                <p className="mt-1 line-clamp-4 whitespace-pre-line text-xs leading-relaxed text-zinc-300">
                  <Highlight text={h.text.replace(/\n{2,}/g, '\n')} terms={terms} />
                </p>
                <p className="mt-1 text-[10px] text-zinc-600">{h.file} · lines {h.first}–{h.last}</p>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )

  const reader = (
    <div className="flex h-full flex-col">
      {phone && (
        <button onClick={() => setPane(searchOn ? 'sections' : picked?.kind === 'extra' ? 'docs' : 'sections')}
          className="flex items-center gap-1.5 border-b border-zinc-800 px-3 py-2 text-xs text-zinc-400">
          <ArrowLeft className="h-3.5 w-3.5" /> {searchOn ? 'Results' : picked?.kind === 'extra' ? 'Documents' : 'Sections'}
        </button>
      )}
      <div ref={readerBox} className="min-h-0 flex-1 overflow-y-auto">
        {readerError ? <Notice tone="error" text={readerError} />
          : readerLoading && !section && !extra ? <Spinner />
            : picked?.kind === 'extra' && extra ? (
              <article className="mx-auto max-w-3xl px-5 py-5 wide:px-8">
                <h1 className="mb-4 text-xl font-semibold text-zinc-100">{extra.title}</h1>
                <div className="fd-file-markdown"><Markdown remarkPlugins={[remarkGfm]} components={MD}>{extra.text}</Markdown></div>
              </article>
            ) : sectionI !== null && section ? (
              <SectionView section={section} doc={detail} loading={readerLoading} onOpen={(i) => openSection(i)}
                onPage={(p) => { setPage(p); setFocus('') }} />
            ) : detail ? (
              <DocLanding doc={detail} onStart={(i) => openSection(i)} />
            ) : picked ? <Spinner /> : (
              <Empty icon={<BookOpen className="h-6 w-6" />} title="Pick a document" text="Its sections show here to read." />
            )}
      </div>
    </div>
  )

  // ── Layout ──────────────────────────────────────────────────────────

  let body: ReactNode
  if (error) {
    body = <div className="p-6"><Notice tone="error" text={error} action={{ label: 'Try again', onClick: retry }} /></div>
  } else if (!overview) {
    body = <Spinner />
  } else if (!hasDocs) {
    body = (
      <Empty icon={<Layers className="h-6 w-6" />} title={`${agent.name} has no documents yet`}
        text="Give it your policies, manuals and price lists in Way of working: they become knowledge it can search — and read for rules and skills."
        action={onSetup ? { label: 'Open Way of working', onClick: onSetup } : undefined} />
    )
  } else if (phone) {
    body = pane === 'docs' && !searchOn ? docsList
      : pane === 'reader' ? reader
        : searchOn ? results : tree
  } else {
    body = (
      <div className="flex h-full min-h-0">
        <aside className="w-60 shrink-0 border-r border-zinc-800 bg-zinc-950 lg:w-64">{docsList}</aside>
        <section className="w-72 shrink-0 border-r border-zinc-800 lg:w-80" aria-label={searchOn ? 'Search results' : 'Sections'}>
          {searchOn ? results : picked?.kind === 'doc' ? tree
            : <Empty icon={<ListTree className="h-5 w-5" />} title="" text="A document's sections show here." />}
        </section>
        <main className="min-w-0 flex-1 bg-zinc-950">{reader}</main>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col bg-zinc-950">
      {!embedded && header}
      {toolbar}
      <div className="min-h-0 flex-1">{body}</div>
    </div>
  )
}

function DocRow({ doc, active, onOpen }: { doc: KnowledgeDoc; active: boolean; onOpen: () => void }) {
  return (
    <li>
      <button onClick={onOpen} title={doc.summary || doc.title}
        className={`w-full rounded-lg px-2.5 py-2 text-left transition-colors ${active ? 'bg-zinc-800' : 'hover:bg-zinc-800/60'}`}>
        <span className="flex items-center gap-2">
          <FileText className={`h-4 w-4 shrink-0 ${active ? 'text-sky-500 dark:text-sky-400' : 'text-zinc-500'}`} />
          <span className={`min-w-0 flex-1 truncate text-[13px] ${active ? 'font-medium text-zinc-100' : 'text-zinc-300'}`}>{doc.title}</span>
        </span>
        <span className="mt-1 flex items-center gap-2 pl-6 text-[11px] text-zinc-500">
          <span className="shrink-0">{fmtBytes(doc.bytes)}{doc.parts > 1 ? ` · ${doc.parts} parts` : ''}</span>
        </span>
        <span className="mt-1 block pl-6"><ReadMeter read={doc.read} /></span>
      </button>
    </li>
  )
}

/** A document opened: what it is, how much the model read, and its chapters. */
function DocLanding({ doc, onStart }: { doc: KnowledgeDocDetail; onStart: (i: number) => void }) {
  const first = doc.sections.find((s) => s.chars > 40) ?? doc.sections[0]
  const top = doc.sections.filter((s) => s.level === 1)
  const chapters = top.length > 1 ? top : doc.sections.filter((s) => s.level === 2)
  const read = readLook(doc.read)
  const chip = 'inline-flex items-center gap-1 rounded-full border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-400'
  return (
    <article className="mx-auto max-w-3xl px-5 py-6 wide:px-8">
      <p className="text-[11px] font-medium uppercase tracking-wider text-sky-700 dark:text-sky-400">Document</p>
      <h1 className="mt-1 text-2xl font-semibold text-zinc-100">{doc.title}</h1>
      {doc.summary && <p className="mt-2 text-sm leading-relaxed text-zinc-400">{doc.summary}</p>}
      <div className="mt-3 flex flex-wrap gap-1.5">
        <span className={chip}>{fmtBytes(doc.bytes)}</span>
        <span className={chip}>{plural(doc.sections.length, 'section')}</span>
        {doc.parts > 1 && <span className={chip}><Layers className="h-3 w-3" />{doc.parts} parts</span>}
        {doc.map && <span className={chip}><MapIcon className="h-3 w-3" />section map</span>}
        {doc.use && <span className={chip}>{doc.use}</span>}
        {doc.tables.length > 0 && <span className={chip}><Table2 className="h-3 w-3" />{plural(doc.tables.length, 'table')}</span>}
        {doc.doc !== doc.id && <span className={chip} title="The name it was given under">{doc.doc}</span>}
      </div>
      <div className="mt-5 rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
        <div className="flex items-center gap-2">
          {doc.read.state === 'full' ? <Check className="h-4 w-4 text-emerald-500" />
            : doc.read.state === 'unknown' ? <CircleDashed className="h-4 w-4 text-zinc-500" />
              : <Sparkles className={`h-4 w-4 ${read.tone}`} />}
          <p className="text-sm font-medium text-zinc-200">
            {doc.read.state === 'full' ? 'The model read all of it'
              : doc.read.state === 'partial'
                ? `The model read ${doc.read.sections.toLocaleString()} of the ${plural(doc.read.of, 'section')} that hold text`
                : doc.read.state === 'none' ? 'Kept as knowledge — not read by the model'
                  : 'Given before reads were tracked'}
          </p>
        </div>
        <div className="mt-2.5"><ReadMeter read={doc.read} wide /></div>
        <p className="mt-2 text-xs leading-relaxed text-zinc-500">
          {doc.read.state === 'full'
            ? 'Its rules, procedures and facts went into the setup; all of it is searchable too.'
            : 'All of it is searchable: the agent finds any part when it needs it. Sections the model read gave rules, skills and facts — the rest can be read later, in Way of working.'}
        </p>
      </div>
      {first && (
        <button onClick={() => onStart(first.i)}
          className="mt-5 inline-flex items-center gap-2 rounded-xl bg-sky-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-sky-500">
          <BookOpen className="h-4 w-4" /> Start reading
        </button>
      )}
      {chapters.length > 1 && (
        <div className="mt-7">
          <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Contents</p>
          <ol className="grid gap-1 wide:grid-cols-2">
            {chapters.slice(0, 40).map((s) => (
              <li key={s.i}>
                <button onClick={() => onStart(s.i)}
                  className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] text-zinc-300 transition-colors hover:bg-zinc-800/60 hover:text-zinc-100">
                  <ReadDot read={s.read} /><span className="truncate">{plain(s.title)}</span>
                </button>
              </li>
            ))}
          </ol>
          {chapters.length > 40 && <p className="mt-1.5 px-2.5 text-[11px] text-zinc-500">…and {chapters.length - 40} more in the sections.</p>}
        </div>
      )}
    </article>
  )
}

/** One section: where it is, its text, its subsections, the ones around it. */
function SectionView({ section, doc, loading, onOpen, onPage }: {
  section: KnowledgeSectionText
  doc: KnowledgeDocDetail | null
  loading: boolean
  onOpen: (i: number) => void
  onPage: (p: number) => void
}) {
  // Its own text with its heading line taken off: the page shows the heading itself.
  const body = section.text.replace(/^#{1,6}[ \t]+[^\n]*\n?/, '').trim()
  return (
    <article className={`mx-auto max-w-3xl px-5 pb-8 wide:px-8 ${loading ? 'opacity-60' : ''}`}>
      <div className="sticky top-0 z-10 -mx-5 border-b border-zinc-800/80 bg-zinc-950/95 px-5 py-2.5 backdrop-blur wide:-mx-8 wide:px-8">
        <p className="flex min-w-0 items-center gap-1 text-[11px] text-zinc-500">
          <span className="shrink-0 truncate font-medium text-zinc-400">{doc?.title ?? section.title}</span>
          {section.path.map((p, n) => (
            <span key={n} className="flex min-w-0 items-center gap-1"><ChevronRight className="h-3 w-3 shrink-0" /><span className="truncate">{plain(p)}</span></span>
          ))}
        </p>
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
          <h1 className="min-w-0 flex-1 text-lg font-semibold leading-snug text-zinc-100">{plain(section.section)}</h1>
          {section.read !== null && (
            <span className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[11px] ${section.read
              ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400' : 'bg-zinc-800 text-zinc-400'}`}>
              {section.read ? <><Check className="h-3 w-3" />Read by the model</> : <><Search className="h-3 w-3" />Searchable, not read</>}
            </span>
          )}
        </div>
        <p className="mt-0.5 font-mono text-[10px] text-zinc-600">knowledge/{section.file} · line {section.line.toLocaleString()} · {kChars(section.chars)} characters</p>
      </div>
      {body ? (
        <div className="fd-file-markdown mt-4"><Markdown remarkPlugins={[remarkGfm]} components={MD}>{body}</Markdown></div>
      ) : (
        <p className="mt-5 text-sm text-zinc-500">{section.children.length ? 'A heading over the sections below.' : 'Nothing under this heading.'}</p>
      )}
      {section.pages > 1 && (
        <div className="mt-6 flex items-center justify-center gap-2 text-xs text-zinc-400">
          <button disabled={section.page <= 1} onClick={() => onPage(section.page - 1)} className={iconBtn} aria-label="Previous page"><ChevronLeft className="h-4 w-4" /></button>
          <span>Page {section.page} of {section.pages}</span>
          <button disabled={section.page >= section.pages} onClick={() => onPage(section.page + 1)} className={iconBtn} aria-label="Next page"><ChevronRight className="h-4 w-4" /></button>
        </div>
      )}
      {section.children.length > 0 && (
        <div className="mt-6">
          <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">In this section</p>
          <ul className="grid gap-1 wide:grid-cols-2">
            {section.children.map((c) => (
              <li key={c.i}>
                <button onClick={() => onOpen(c.i)}
                  className="flex w-full items-center gap-2 rounded-lg border border-zinc-800 px-3 py-2 text-left text-[13px] text-zinc-300 transition-colors hover:border-zinc-700 hover:text-zinc-100">
                  <ReadDot read={doc?.sections[c.i]?.read ?? null} /><span className="truncate">{plain(c.title)}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      <nav className="mt-8 grid grid-cols-2 gap-2 border-t border-zinc-800 pt-4" aria-label="Sections around it">
        {section.prev ? (
          <button onClick={() => onOpen(section.prev!.i)} className="group flex min-w-0 flex-col items-start rounded-xl border border-zinc-800 px-3 py-2 text-left hover:border-zinc-700">
            <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-zinc-500"><ChevronLeft className="h-3 w-3" />Previous</span>
            <span className="w-full truncate text-[13px] text-zinc-300 group-hover:text-zinc-100">{plain(section.prev.title)}</span>
          </button>
        ) : <span />}
        {section.next ? (
          <button onClick={() => onOpen(section.next!.i)} className="group flex min-w-0 flex-col items-end rounded-xl border border-zinc-800 px-3 py-2 text-right hover:border-zinc-700">
            <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-zinc-500">Next<ChevronRight className="h-3 w-3" /></span>
            <span className="w-full truncate text-[13px] text-zinc-300 group-hover:text-zinc-100">{plain(section.next.title)}</span>
          </button>
        ) : <span />}
      </nav>
    </article>
  )
}

function Spinner() {
  return <div className="flex h-full min-h-32 items-center justify-center"><Loader2 className="h-5 w-5 animate-spin text-zinc-500" /></div>
}

function Empty({ icon, title, text, action }: {
  icon: ReactNode; title: string; text: string; action?: { label: string; onClick: () => void }
}) {
  return (
    <div className="flex h-full min-h-48 flex-col items-center justify-center px-6 py-10 text-center">
      <span className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-sky-500/10 text-sky-600 dark:text-sky-400">{icon}</span>
      {title && <p className="text-sm font-medium text-zinc-200">{title}</p>}
      <p className="mt-1 max-w-sm text-xs leading-relaxed text-zinc-500">{text}</p>
      {action && (
        <button onClick={action.onClick} className="mt-4 inline-flex items-center gap-1.5 rounded-xl bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500">
          <Wand2 className="h-4 w-4" /> {action.label}
        </button>
      )}
    </div>
  )
}

function Notice({ tone, text, action }: { tone: 'error'; text: string; action?: { label: string; onClick: () => void } }) {
  return (
    <div className={`m-3 flex items-start gap-2 rounded-xl border px-3 py-2.5 text-xs ${tone === 'error'
      ? 'border-red-500/30 bg-red-500/5 text-red-700 dark:text-red-400' : ''}`}>
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <span className="min-w-0 flex-1">{text}</span>
      {action && <button onClick={action.onClick} className="shrink-0 underline">{action.label}</button>}
    </div>
  )
}
