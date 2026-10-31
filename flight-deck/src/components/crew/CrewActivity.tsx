import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  AlertTriangle, Bot, CheckCircle2, ChevronRight, FilePen, FilePlus2, FileText, Flag, Globe, ListChecks, MessageSquare,
  Search, Terminal, Wrench, type LucideIcon,
} from 'lucide-react'
import type { CrewEvent } from '../../services/crewApi'
import { CrewMarkdown, STATUS } from './crewParts'

// ── What a worker did ──
//
// A timeline a person can read: what it was told, what it said, and its steps folded
// into one line each ("Wrote 3 files · Ran 1 command"). Each CLI names its tools its
// own way, so a tool is known by the shape of its name.

const TOOLS: { match: RegExp; icon: LucideIcon; verb: string; unit: string; units: string }[] = [
  { match: /web|fetch|brows/i, icon: Globe, verb: 'Looked up', unit: 'page', units: 'pages' },
  { match: /write|create/i, icon: FilePlus2, verb: 'Wrote', unit: 'file', units: 'files' },
  { match: /edit|patch|replace/i, icon: FilePen, verb: 'Edited', unit: 'file', units: 'files' },
  { match: /read|view|open/i, icon: FileText, verb: 'Read', unit: 'file', units: 'files' },
  { match: /grep|glob|search|find|list|^ls$/i, icon: Search, verb: 'Searched', unit: 'time', units: 'times' },
  { match: /bash|shell|exec|command|terminal|run/i, icon: Terminal, verb: 'Ran', unit: 'command', units: 'commands' },
  { match: /todo|plan/i, icon: ListChecks, verb: 'Planned', unit: 'time', units: 'times' },
  { match: /task|agent|spawn/i, icon: Bot, verb: 'Handed off', unit: 'job', units: 'jobs' },
]

function toolOf(name?: string) {
  return TOOLS.find((t) => t.match.test(name || ''))
}

/** Paths as the worker's own folder sees them, not where that folder is on the server. */
const shortPaths = (text: string) => text.replace(/\S*\/worktrees\/[^/\s]+\//g, '')

interface Step { n: number; tool?: string; text: string; output: string }
type Block =
  | { type: 'steps'; n: number; steps: Step[] }
  | { type: 'event'; n: number; event: CrewEvent }

function toBlocks(events: CrewEvent[]): Block[] {
  const blocks: Block[] = []
  let group: Extract<Block, { type: 'steps' }> | null = null
  let waiting: Step[] = []
  const open = (n: number) => {
    if (!group) {
      group = { type: 'steps', n, steps: [] }
      blocks.push(group)
    }
    return group
  }
  for (const e of events) {
    if (e.kind === 'tool') {
      const step: Step = { n: e.n, tool: e.tool, text: shortPaths(e.text), output: '' }
      open(e.n).steps.push(step)
      waiting.push(step)
    } else if (e.kind === 'result') {
      if (!e.text.trim()) continue
      // Answers come in the order the calls went out.
      const step = waiting.shift()
      if (step) step.output = shortPaths(e.text)
      else open(e.n).steps.push({ n: e.n, text: shortPaths(e.text.split('\n')[0]), output: shortPaths(e.text) })
    } else {
      // Its CREW-STATUS line is for the watcher: what it says shows as the status below.
      const text = e.kind === 'text' ? e.text.replace(/^[\s*_`>]*CREW-STATUS:.*$/gim, '').trim() : e.text
      if (!text.trim()) continue
      group = null
      waiting = []
      // A failure often arrives three times over: as text, as an error and as the status.
      const last = blocks[blocks.length - 1]
      if (last?.type === 'event' && last.event.kind !== 'turn' && last.event.text.trim() === text.trim()) blocks.pop()
      blocks.push({ type: 'event', n: e.n, event: { ...e, text } })
    }
  }
  return blocks
}

function summary(steps: Step[]): string {
  const counts = new Map<string, { verb: string; unit: string; units: string; n: number }>()
  for (const s of steps) {
    const t = toolOf(s.tool)
    const key = t?.verb ?? 'Did'
    const row = counts.get(key) ?? { verb: key, unit: t?.unit ?? 'step', units: t?.units ?? 'steps', n: 0 }
    row.n += 1
    counts.set(key, row)
  }
  return [...counts.values()].map((c) => `${c.verb} ${c.n} ${c.n === 1 ? c.unit : c.units}`).join(' · ')
}

function Line({ icon: Icon, tone, children }: { icon: LucideIcon; tone: string; children: ReactNode }) {
  return (
    <li className="relative pl-7">
      <span className={`absolute left-0 top-0.5 flex h-5 w-5 items-center justify-center rounded-full ring-4 ring-zinc-900 ${tone}`}>
        <Icon className="h-3 w-3" />
      </span>
      {children}
    </li>
  )
}

function StepRow({ step }: { step: Step }) {
  const t = toolOf(step.tool)
  const Icon = t?.icon ?? Wrench
  const first = step.text.split('\n')[0]
  const more = step.text.includes('\n') || !!step.output
  const label = t?.verb ?? step.tool ?? 'Output'
  const row = (
    <>
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-500" />
      <span className="shrink-0 text-xs font-medium text-zinc-400">{label}</span>
      <span className="min-w-0 truncate font-mono text-[11.5px] text-zinc-300">{first}</span>
    </>
  )
  if (!more) return <div className="flex items-start gap-2 rounded-md px-2 py-1">{row}</div>
  return (
    <details className="group/step rounded-md">
      <summary className="flex cursor-pointer list-none items-start gap-2 rounded-md px-2 py-1 transition-colors hover:bg-zinc-800/60 [&::-webkit-details-marker]:hidden">
        {row}
        <ChevronRight className="ml-auto mt-0.5 h-3 w-3 shrink-0 text-zinc-600 transition-transform group-open/step:rotate-90" />
      </summary>
      <div className="mx-2 mb-1.5 mt-0.5 space-y-1.5 rounded-md bg-zinc-950/60 p-2.5 font-mono text-[11px] leading-relaxed">
        {step.text.includes('\n') && <pre className="whitespace-pre-wrap break-words text-zinc-300">{step.text}</pre>}
        {step.output && <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words text-zinc-500">{step.output}</pre>}
      </div>
    </details>
  )
}

function Steps({ steps, live }: { steps: Step[]; live: boolean }) {
  const [open, setOpen] = useState(live)
  useEffect(() => setOpen(live), [live])
  return (
    <Line icon={Wrench} tone="bg-zinc-800 text-zinc-400">
      <button type="button" onClick={() => setOpen((v) => !v)} className="flex w-full items-center gap-1.5 text-left text-xs text-zinc-400 transition-colors hover:text-zinc-200">
        <ChevronRight className={`h-3 w-3 shrink-0 transition-transform ${open ? 'rotate-90' : ''}`} />
        <span className="min-w-0 truncate">{summary(steps)}</span>
      </button>
      {open && <div className="mt-1.5 space-y-0.5 rounded-lg border border-zinc-800 p-1">{steps.map((s) => <StepRow key={s.n} step={s} />)}</div>}
    </Line>
  )
}

function Brief({ event }: { event: CrewEvent }) {
  const [more, setMore] = useState(false)
  const long = event.text.length > 420
  return (
    <Line icon={Flag} tone="bg-teal-500/15 text-teal-700 dark:text-teal-300">
      <div className="rounded-xl border border-zinc-800 bg-zinc-800/30 px-3 py-2">
        <p className="mb-0.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">{event.turn <= 1 ? 'The task' : 'You told it'}</p>
        <p className="whitespace-pre-wrap break-words text-[13px] text-zinc-200">{long && !more ? `${event.text.slice(0, 420)}…` : event.text}</p>
        {long && <button type="button" onClick={() => setMore((v) => !v)} className="mt-1 text-xs text-teal-700 hover:underline dark:text-teal-300">{more ? 'Show less' : 'Show all of it'}</button>}
      </div>
    </Line>
  )
}

function EventLine({ event: e }: { event: CrewEvent }) {
  if (e.kind === 'turn') return <Brief event={e} />
  if (e.kind === 'text') {
    return <Line icon={MessageSquare} tone="bg-sky-500/15 text-sky-600 dark:text-sky-300"><CrewMarkdown text={e.text} /></Line>
  }
  if (e.kind === 'error') {
    return (
      <Line icon={AlertTriangle} tone="bg-red-500/15 text-red-700 dark:text-red-300">
        <pre className="whitespace-pre-wrap break-words rounded-lg bg-red-500/10 p-2.5 font-mono text-xs text-red-700 dark:text-red-300">{e.text}</pre>
      </Line>
    )
  }
  if (e.kind === 'status') {
    const s = e.status ? STATUS[e.status as keyof typeof STATUS] : undefined
    return (
      <Line icon={e.status === 'failed' || e.status === 'waiting' ? AlertTriangle : CheckCircle2} tone={s?.tone ?? STATUS.ready.tone}>
        <p className="break-words pt-0.5 text-[13px] font-medium text-zinc-200">{e.text}</p>
      </Line>
    )
  }
  return <Line icon={Wrench} tone="bg-zinc-800 text-zinc-500"><p className="break-words pt-0.5 text-xs text-zinc-500">{e.text}</p></Line>
}

export function CrewActivity({ events, live }: { events: CrewEvent[]; live: boolean }) {
  const end = useRef<HTMLDivElement>(null)
  const box = useRef<HTMLDivElement>(null)
  const blocks = useMemo(() => toBlocks(events), [events])
  useEffect(() => {
    const el = box.current
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 240) end.current?.scrollIntoView({ block: 'end' })
  }, [events.length])
  if (!blocks.length) return <p className="p-5 text-sm text-zinc-500">Nothing yet.</p>
  return (
    <div ref={box} className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
      <ul className="relative space-y-4 before:absolute before:bottom-2 before:left-[9.5px] before:top-2 before:w-px before:bg-zinc-800">
        {blocks.map((b, i) => (b.type === 'steps'
          ? <Steps key={b.n} steps={b.steps} live={live && i === blocks.length - 1} />
          : <EventLine key={b.n} event={b.event} />))}
      </ul>
      <div ref={end} />
    </div>
  )
}
