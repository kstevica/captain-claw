import {
  BookA, BookOpen, FileSpreadsheet, FileText, Hand, LayoutTemplate, Lightbulb, ListChecks, Presentation, ScrollText,
  Sparkles,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { AgentPreset, DocUsage, PreparedPreset, ReviewedPreset } from '../../services/agentPreset'

// What AgentSetupDialog and its review share: how a document is to be used
// (the person's pick, which the backend's USAGES tells the model), what kind
// of file it is, and the review's state — the groups the model fills, what
// was kept, and what Apply sends.

export const USAGE_ORDER: DocUsage[] = ['auto', 'rules', 'procedures', 'reference', 'templates']

export const USAGE_LOOKS: Record<DocUsage, { icon: LucideIcon; label: string; blurb: string }> = {
  auto: { icon: Sparkles, label: 'Let it decide', blurb: 'Your model works out what this document is for.' },
  rules: { icon: ScrollText, label: 'Rules to follow', blurb: 'Policies, limits, tone — it follows them every time.' },
  procedures: { icon: ListChecks, label: 'Procedures', blurb: 'How requests are handled, step by step — each becomes a skill.' },
  reference: { icon: BookOpen, label: 'Reference', blurb: 'Facts to look up — prices, FAQs, product details. No new rules.' },
  templates: { icon: LayoutTemplate, label: 'Templates', blurb: 'Formats and example answers it reuses.' },
}

/** The documents the setup reads (the backend reads the same). */
export const SETUP_ACCEPT = '.pdf,.docx,.xlsx,.pptx,.txt,.md,.csv,.html,.htm'
export const SETUP_EXTS = new Set(SETUP_ACCEPT.split(','))

export function extOf(name: string): string {
  const m = name.toLowerCase().match(/\.[a-z0-9]+$/)
  return m ? m[0] : ''
}

export function fileLook(name: string): { icon: LucideIcon; tag: string; tone: string } {
  const ext = extOf(name)
  switch (ext) {
    case '.pdf': return { icon: FileText, tag: 'PDF', tone: 'bg-red-500/10 text-red-600 dark:text-red-400' }
    case '.docx': return { icon: FileText, tag: 'Word', tone: 'bg-sky-500/10 text-sky-600 dark:text-sky-400' }
    case '.xlsx': return { icon: FileSpreadsheet, tag: 'Excel', tone: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400' }
    case '.csv': return { icon: FileSpreadsheet, tag: 'CSV', tone: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400' }
    case '.pptx': return { icon: Presentation, tag: 'Slides', tone: 'bg-orange-500/10 text-orange-600 dark:text-orange-400' }
    case '.md': return { icon: FileText, tag: 'MD', tone: 'bg-zinc-800 text-zinc-400' }
    default: return { icon: FileText, tag: ext ? ext.slice(1).toUpperCase() : 'File', tone: 'bg-zinc-800 text-zinc-400' }
  }
}

export function fmtBytes(n: number): string {
  return n >= 1024 * 1024 ? `${(n / (1024 * 1024)).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`
}

// ── The review ───────────────────────────────────────────────────────

export type Group = 'rules' | 'intents' | 'facts' | 'escalation' | 'templates' | 'glossary' | 'knowledge'
export const GROUP_ORDER: Group[] = ['rules', 'intents', 'facts', 'escalation', 'templates', 'glossary', 'knowledge']

export const GROUPS: Record<Group, { icon: LucideIcon; title: string; blurb: string }> = {
  rules: { icon: ScrollText, title: 'Rules', blurb: 'Standing rules it follows — those always in mind in every conversation, the rest when a message is about them.' },
  intents: { icon: ListChecks, title: 'What it handles', blurb: 'Requests it knows a procedure for — one skill each.' },
  facts: { icon: Lightbulb, title: 'Facts', blurb: 'Things it’s reminded of when a message is about them.' },
  escalation: { icon: Hand, title: 'When to ask you', blurb: 'It stops and checks with you first.' },
  templates: { icon: LayoutTemplate, title: 'Templates', blurb: 'Formats it uses for its answers.' },
  glossary: { icon: BookA, title: 'Glossary', blurb: 'Your terms and what they mean.' },
  knowledge: { icon: BookOpen, title: 'Documents', blurb: 'It keeps these and looks things up in them.' },
}

/** The review's state: the preset as edited, and which items are left out. */
export interface Review { prepared: PreparedPreset; preset: AgentPreset; off: Set<string> }

export function startReview(prepared: PreparedPreset): Review {
  return { prepared, preset: structuredClone(prepared.preset), off: new Set() }
}

/** What Apply sends: the items kept, as edited. */
export function reviewedPreset(r: Review): ReviewedPreset {
  const kept = <T,>(g: Group, items: T[]) => items.filter((_, i) => !r.off.has(`${g}:${i}`))
  const p = r.preset
  return {
    id: r.prepared.id,
    rules: kept('rules', p.rules),
    intents: kept('intents', p.intents),
    facts: kept('facts', p.facts),
    escalation: kept('escalation', p.escalation),
    templates: kept('templates', p.templates),
    glossary: kept('glossary', p.glossary),
    knowledge: kept('knowledge', p.knowledge),
  }
}

export function keptIn(r: Review, g: Group): number {
  return r.preset[g].filter((_, i) => !r.off.has(`${g}:${i}`)).length
}

export function keptCount(r: Review): number {
  return GROUP_ORDER.reduce((n, g) => n + keptIn(r, g), 0)
}

// ── The block a setup puts in the agent's instructions ────────────────

const MARKED_BLOCK = /\s*<!-- owner-setup -->[\s\S]*?<!-- \/owner-setup -->/g
// One written before it was marked: its heading to the next "## " or the end.
const BARE_BLOCK = /\s*^## Your owner's rules\n\nSet up by your owner[\s\S]*?(?=^## |(?![\s\S]))/gm

/** The instructions without the block an earlier setup put in them — what
 *  a deck without accounts (the browser keeps them) replaces itself; the
 *  backend's agent_preset.strip_block. */
export function stripOwnerBlock(text: string): string {
  const unmarked = text.replace(MARKED_BLOCK, '')
  return unmarked.replace(BARE_BLOCK, (m, offset: number) => (offset + m.length >= unmarked.length ? '' : '\n\n')).trimEnd()
}
