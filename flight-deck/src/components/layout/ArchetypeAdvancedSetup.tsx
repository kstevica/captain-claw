import { useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  AlertTriangle, BookA, BookOpen, FileText, Hand, LayoutTemplate, Lightbulb, ListChecks, ScrollText, Trash2, Upload,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { AgentPreset, PreparedPreset, PresetSource, ReviewedPreset } from '../../services/agentPreset'

// ── The New agent dialog's "Advanced" setup ──────────────────────────
//
// Two pieces ArchetypeSpawnDialog places: the editor (extra instructions and
// documents, in the detail panel) and the review of what the owner's model
// pulled out of them (every item kept or dropped, editable, with its source),
// which takes the whole dialog body before Create.

export const ADVANCED_ACCEPT = '.pdf,.docx,.xlsx,.pptx,.txt,.md,.csv'
const MAX_FILE_BYTES = 25 * 1024 * 1024
const MAX_FILES = 12
// Prepare reads at most this much text (the backend's caps) — what the token
// estimate is bounded by.
const MAX_CHARS_PER_FILE = 40000
const MAX_CHARS = 160000
const TEXT_TYPES = /\.(txt|md|csv)$/i

function fmtBytes(n: number): string {
  return n >= 1024 * 1024 ? `${(n / (1024 * 1024)).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`
}

/** A rough upper bound of the tokens Prepare spends: the text it can read
 *  (a text file's size; about a third of a PDF's or an Office file's), the
 *  instructions, the prompt and the answer. */
export function estimateTokens(files: File[], instructions: string): number {
  let chars = 0
  for (const f of files) chars += Math.min(MAX_CHARS_PER_FILE, TEXT_TYPES.test(f.name) ? f.size : f.size / 3)
  chars = Math.min(MAX_CHARS, chars) + instructions.length
  return Math.round(chars / 4 + 2500 + 8000)
}

export function AdvancedEditor({ instructions, files, disabled, onInstructions, onFiles }: {
  instructions: string
  files: File[]
  disabled: boolean
  onInstructions: (v: string) => void
  onFiles: (files: File[]) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [note, setNote] = useState('')

  const add = (list: FileList | null) => {
    if (!list) return
    const incoming = Array.from(list)
    const tooBig = incoming.filter((f) => f.size > MAX_FILE_BYTES)
    const seen = new Set(files.map((f) => f.name + f.size))
    const next = [...files, ...incoming.filter((f) => f.size <= MAX_FILE_BYTES && !seen.has(f.name + f.size))]
    setNote([
      tooBig.length ? `Left out ${tooBig.length} file${tooBig.length === 1 ? '' : 's'} over 25 MB.` : '',
      next.length > MAX_FILES ? `Up to ${MAX_FILES} documents.` : '',
    ].filter(Boolean).join(' '))
    onFiles(next.slice(0, MAX_FILES))
  }
  const total = files.reduce((n, f) => n + f.size, 0)

  return (
    <div className="space-y-3">
      <p className="text-xs leading-relaxed text-zinc-500">
        Tell it how you work, and add your documents — policies, FAQs, procedures, price lists, templates.
        Your model reads them and shows you what the agent will start with before you create it.
      </p>
      <label className="block">
        <span className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Instructions</span>
        <textarea
          value={instructions}
          onChange={(e) => onInstructions(e.target.value)}
          disabled={disabled}
          rows={5}
          placeholder="e.g. Answer customers in Croatian. Never promise a delivery date. Ask me before refunding more than 200 EUR."
          className="w-full resize-y rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-base text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none disabled:opacity-60 wide:text-sm"
        />
      </label>
      <div>
        <span className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Documents</span>
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); if (!disabled) add(e.dataTransfer.files) }}
          className="flex w-full flex-col items-center justify-center gap-1 rounded-xl border border-dashed border-zinc-700 bg-zinc-950/50 px-3 py-4 text-center transition-colors hover:border-violet-500/50 hover:bg-zinc-900 disabled:opacity-60"
        >
          <Upload className="h-4 w-4 text-zinc-500" />
          <span className="text-xs text-zinc-400">Drop files here or click to add</span>
          <span className="text-[10px] text-zinc-600">PDF, Word, Excel, PowerPoint, text, Markdown, CSV · up to 25 MB each</span>
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ADVANCED_ACCEPT}
          className="hidden"
          onChange={(e) => { add(e.target.files); e.target.value = '' }}
        />
        {note && <p className="mt-1.5 text-[11px] text-amber-600 dark:text-amber-400">{note}</p>}
        {files.length > 0 && (
          <ul className="mt-2 space-y-1">
            {files.map((f, i) => (
              <li key={f.name + f.size} className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950/50 px-2.5 py-1.5">
                <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                <span className="min-w-0 flex-1 truncate text-xs text-zinc-300">{f.name}</span>
                <span className="text-[10px] text-zinc-600">{fmtBytes(f.size)}</span>
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => onFiles(files.filter((_, j) => j !== i))}
                  aria-label={`Remove ${f.name}`}
                  className="rounded p-1 text-zinc-600 hover:text-red-600 disabled:opacity-50 dark:hover:text-red-400"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      {(files.length > 0 || instructions.trim()) && (
        <p className="text-[11px] text-zinc-500">
          {files.length > 0
            ? `${files.length} document${files.length === 1 ? '' : 's'}, ${fmtBytes(total)} · reading them`
            : 'Reading it'}{' '}
          takes roughly up to {Math.ceil(estimateTokens(files, instructions) / 1000)}k tokens of your daily allowance.
        </p>
      )}
    </div>
  )
}

// ── The review ───────────────────────────────────────────────────────

type Group = 'rules' | 'intents' | 'facts' | 'escalation' | 'templates' | 'glossary' | 'knowledge'
const GROUP_ORDER: Group[] = ['rules', 'intents', 'facts', 'escalation', 'templates', 'glossary', 'knowledge']

const GROUPS: Record<Group, { icon: LucideIcon; title: string; blurb: string }> = {
  rules: { icon: ScrollText, title: 'Rules', blurb: 'Standing rules it always follows.' },
  intents: { icon: ListChecks, title: 'What it handles', blurb: 'Requests it knows a procedure for — one skill each.' },
  facts: { icon: Lightbulb, title: 'Facts', blurb: 'Things it remembers from the start.' },
  escalation: { icon: Hand, title: 'When to ask you', blurb: 'It stops and checks with you first.' },
  templates: { icon: LayoutTemplate, title: 'Templates', blurb: 'Formats it uses for its answers.' },
  glossary: { icon: BookA, title: 'Glossary', blurb: 'Your terms and what they mean.' },
  knowledge: { icon: BookOpen, title: 'Knowledge', blurb: 'Documents it keeps and looks things up in.' },
}

/** The review's state: the preset as edited, and which items are left out. */
export interface Review { prepared: PreparedPreset; preset: AgentPreset; off: Set<string> }

export function startReview(prepared: PreparedPreset): Review {
  return { prepared, preset: structuredClone(prepared.preset), off: new Set() }
}

/** What Create sends: the items kept, as edited. */
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

export function keptCount(r: Review): number {
  return GROUP_ORDER.reduce((n, g) => n + r.preset[g].filter((_, i) => !r.off.has(`${g}:${i}`)).length, 0)
}

const fieldCls = 'w-full rounded-lg border border-zinc-800 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none disabled:opacity-60'

export function PresetReview({ review, disabled, onChange }: {
  review: Review
  disabled: boolean
  onChange: (r: Review) => void
}) {
  const { preset, off, prepared } = review
  const toggle = (key: string) => {
    const next = new Set(off)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    onChange({ ...review, off: next })
  }
  function edit<G extends Group>(g: G, i: number, patch: Partial<AgentPreset[G][number]>) {
    const list = preset[g].map((item, j) => (j === i ? { ...item, ...patch } : item))
    onChange({ ...review, preset: { ...preset, [g]: list } })
  }
  const notices = [...preset.conflicts.map((t) => ({ t, conflict: true })), ...preset.warnings.map((t) => ({ t, conflict: false }))]
  const empty = GROUP_ORDER.every((g) => preset[g].length === 0)

  return (
    <div className="space-y-5">
      <p className="text-xs text-zinc-500">
        Read by {prepared.model || 'your model'}
        {prepared.docs.length > 0 && ` from ${prepared.docs.length} document${prepared.docs.length === 1 ? '' : 's'}`}.
        Untick anything it shouldn't start with, and fix what it got wrong.
      </p>

      {notices.length > 0 && (
        <ul className="space-y-1.5 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2.5">
          {notices.map(({ t, conflict }, i) => (
            <li key={i} className="flex items-start gap-2 text-xs text-amber-700 dark:text-amber-300">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{conflict && <span className="font-medium">Conflict: </span>}{t}</span>
            </li>
          ))}
        </ul>
      )}

      {empty && <p className="py-6 text-center text-sm text-zinc-500">Nothing to set up came out of this. You can still create the agent.</p>}

      {GROUP_ORDER.filter((g) => preset[g].length > 0).map((g) => {
        const { icon: Icon, title, blurb } = GROUPS[g]
        const kept = preset[g].filter((_, i) => !off.has(`${g}:${i}`)).length
        return (
          <section key={g}>
            <div className="mb-2 flex items-baseline gap-2">
              <Icon className="h-3.5 w-3.5 shrink-0 self-center text-violet-400" />
              <h4 className="text-sm font-semibold text-zinc-100">{title}</h4>
              <span className="text-[11px] text-zinc-500">{kept} of {preset[g].length}</span>
              <span className="hidden truncate text-[11px] text-zinc-600 wide:inline">— {blurb}</span>
            </div>
            <ul className="space-y-2">
              {preset[g].map((item, i) => {
                const key = `${g}:${i}`
                const on = !off.has(key)
                return (
                  <li key={key} className={`flex gap-2.5 rounded-xl border px-3 py-2.5 transition-colors ${on ? 'border-zinc-800 bg-zinc-900/60' : 'border-zinc-800/60 bg-transparent opacity-60'}`}>
                    <input
                      type="checkbox"
                      checked={on}
                      disabled={disabled}
                      onChange={() => toggle(key)}
                      aria-label={on ? 'Leave this out' : 'Keep this'}
                      className="mt-1.5 h-4 w-4 shrink-0 accent-violet-600"
                    />
                    <div className="min-w-0 flex-1 space-y-1.5">
                      <ItemFields group={g} item={item} disabled={disabled || !on} onEdit={(patch) => edit(g, i, patch as never)} />
                    </div>
                  </li>
                )
              })}
            </ul>
          </section>
        )
      })}
    </div>
  )
}

function SourceLine({ source }: { source?: PresetSource }) {
  if (!source || (!source.doc && !source.section)) return null
  const from = source.doc === 'instructions' ? 'your instructions' : source.doc
  return <p className="truncate text-[10px] text-zinc-600">from {from}{source.section ? ` · ${source.section}` : ''}</p>
}

function Lines({ value, disabled, rows, placeholder, onChange }: {
  value: string[]; disabled: boolean; rows: number; placeholder: string; onChange: (v: string[]) => void
}) {
  return (
    <textarea
      value={value.join('\n')}
      disabled={disabled}
      rows={rows}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value.split('\n'))}
      onBlur={(e) => onChange(e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))}
      className={`${fieldCls} resize-y`}
    />
  )
}

function ItemFields({ group, item, disabled, onEdit }: {
  group: Group
  item: AgentPreset[Group][number]
  disabled: boolean
  onEdit: (patch: Record<string, unknown>) => void
}): ReactNode {
  const it = item as unknown as Record<string, unknown>
  const text = (key: string, placeholder: string, rows = 1) => (
    rows > 1
      ? <textarea value={String(it[key] ?? '')} disabled={disabled} rows={rows} placeholder={placeholder}
          onChange={(e) => onEdit({ [key]: e.target.value })} className={`${fieldCls} resize-y`} />
      : <input value={String(it[key] ?? '')} disabled={disabled} placeholder={placeholder}
          onChange={(e) => onEdit({ [key]: e.target.value })} className={fieldCls} />
  )
  const source = <SourceLine source={it.source as PresetSource | undefined} />
  switch (group) {
    case 'rules':
    case 'escalation':
      return <>{text('text', 'The rule', 2)}{source}</>
    case 'facts':
      return <>{text('content', 'The fact', 2)}{source}</>
    case 'intents':
      return (
        <>
          <div className="flex flex-col gap-1.5 wide:flex-row">
            <div className="wide:w-40 wide:shrink-0">{text('name', 'name')}</div>
            <div className="min-w-0 flex-1">{text('when', 'When it applies')}</div>
          </div>
          <Lines value={(it.steps as string[]) || []} disabled={disabled} rows={3} placeholder="One step per line"
            onChange={(steps) => onEdit({ steps })} />
          {source}
        </>
      )
    case 'templates':
      return <>{text('name', 'Template name')}{text('body', 'The format', 4)}{source}</>
    case 'glossary':
      return (
        <>
          <div className="flex flex-col gap-1.5 wide:flex-row">
            <div className="wide:w-40 wide:shrink-0">{text('term', 'Term')}</div>
            <div className="min-w-0 flex-1">{text('meaning', 'What it means')}</div>
          </div>
          {source}
        </>
      )
    case 'knowledge':
      return (
        <>
          {text('title', 'Title')}
          {text('summary', 'What is in it, and when to look there', 2)}
          <p className="truncate text-[10px] text-zinc-600">{String(it.doc ?? '')}</p>
        </>
      )
  }
}
