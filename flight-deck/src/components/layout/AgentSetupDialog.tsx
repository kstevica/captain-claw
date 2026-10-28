import { useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent, ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  ArrowLeft, BookA, BookOpen, Check, CheckCircle2, FolderOpen, Hand, LayoutTemplate, Library, Lightbulb,
  ListChecks, Loader2, Plus, ScrollText, Search, Sparkles, Table2, Trash2, Upload, Wand2, X,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import {
  applyAgentPreset, cancelRead, getCurrentSetup, getRead, getReader, stageDocument, startRead,
  type AppliedPreset, type ApplyMode, type CurrentSetup, type DocUsage, type ReadJob, type Reader, type SetupCounts,
  type StagedDoc,
} from '../../services/agentPreset'
import { fileLibraryApi, type LibraryItem } from '../../services/fileLibraryApi'
import type { QuotaStatus } from '../../services/profile'
import { useChatStore } from '../../stores/chatStore'
import { useProcessStore } from '../../stores/processStore'
import { quotaLimited, useQuotaStore } from '../../stores/quotaStore'
import { PresetReview } from './AgentSetupReview'
import {
  SETUP_ACCEPT, SETUP_EXTS, USAGE_LOOKS, USAGE_ORDER, extOf, fileLook, fmtBytes, keptCount, reviewedPreset, startReview,
  stripOwnerBlock, type Review,
} from './agentSetup'

// ── "Your way of working" — set up right after the agent is created ───
//
// The New agent dialog's "set up its way of working next" box opens this on
// the agent it just created. Four steps:
//
//   material — instructions, and documents (uploaded, or picked from the File
//              library), each read in full as it is added (POST
//              /fd/agent-preset/documents: its pages, and what reading it
//              costs), with what it is, how the agent should use it, and
//              whether the model reads it or it is kept as knowledge only;
//   reading  — the owner's model reads it in the background, in steps sized
//              to its context window, then merges the parts (POST
//              /fd/agent-preset/prepare, followed by GET …/prepare/{id});
//   review   — every rule, skill, fact… kept or left out, editable
//              (AgentSetupReview);
//   done     — what the agent now has (POST /fd/agent-preset/apply). Every
//              document reaches it whole, read or not; a spreadsheet's
//              sheets become datastore tables too, with no model involved.
//
// The agent already exists, so closing at any step costs nothing but the
// setup: it works as its archetype made it. The agent's ⋯ menu opens it on
// an agent later, too: one an earlier setup gave rules, skills and documents
// gets the new ones added to them — or in their place (GET …/current/{slug}
// says what it has).

const MAX_FILE_BYTES = 25 * 1024 * 1024
const MAX_FILES = 12
// The backend's: the most one setup reads, and a step's prompt and answer
// around its material — what the estimate counts.
const MAX_READ_TOKENS = 1_000_000
const STEP_OVERHEAD = 3000 + 4000
const POLL_MS = 1500

interface Doc {
  key: string
  name: string
  size: number
  description: string
  usage: DocUsage
  /** Read it for rules, skills and facts — or keep it as knowledge only. */
  read: boolean
  status: 'staging' | 'ready' | 'error'
  staged?: StagedDoc
  error?: string
  /** An upload… */
  file?: File
  /** …or a File library file. */
  pick?: LibraryItem
}

type Step = 'material' | 'reading' | 'review' | 'done'

let seq = 0
const nextKey = () => `doc-${++seq}`

/** What reading the material takes, roughly: its steps (each at most the
 *  reader's step size), and the tokens — the material, each step's prompt
 *  and answer, and merging the parts. */
function estimateRead(docs: Doc[], instructions: string, stepTokens: number): { steps: number; tokens: number } {
  const material = Math.min(MAX_READ_TOKENS, docs.reduce((n, d) => n + (d.read && d.staged ? d.staged.tokens : 0), 0))
    + Math.ceil(instructions.trim().length / 4)
  if (!material || !stepTokens) return { steps: 0, tokens: 0 }
  const steps = Math.max(1, Math.ceil(material / stepTokens))
  const merging = steps > 1 ? steps * 3000 + 8000 * Math.max(1, Math.ceil((steps * 3000) / stepTokens)) : 0
  return { steps, tokens: material + steps * STEP_OVERHEAD + merging }
}

const kTokens = (n: number) => (n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : `${Math.max(1, Math.round(n / 1000))}k`)
const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`
const message = (e: unknown) => (e instanceof Error ? e.message : String(e))

/** Starting points for the instructions: the bracketed part is selected for typing over. */
const STARTERS = [
  { label: 'Language', text: 'Always answer in [English].' },
  { label: 'Tone', text: 'Keep a [friendly but professional] tone.' },
  { label: 'Never', text: 'Never [promise a delivery date].' },
  { label: 'Ask me first', text: 'Ask me before [refunding more than 200 EUR].' },
]

const STEPS: { id: Step; label: string }[] = [
  { id: 'material', label: 'Your material' },
  { id: 'review', label: 'Review' },
  { id: 'done', label: 'Done' },
]

export function AgentSetupDialog({ agent, asUser, ownerLabel, onClose }: {
  /** The agent just created: its slug, display name and archetype role. */
  agent: { slug: string; name: string; role?: string }
  /** Admin setting up an agent FOR that user (their model, their library). */
  asUser?: string
  ownerLabel?: string
  onClose: () => void
}) {
  const [step, setStep] = useState<Step>('material')
  const [instructions, setInstructions] = useState('')
  const [docs, setDocs] = useState<Doc[]>([])
  const [notice, setNotice] = useState('')
  const [picking, setPicking] = useState(false)
  const [reader, setReader] = useState<Reader | null>(null)
  const [current, setCurrent] = useState<CurrentSetup | null>(null)
  const [mode, setMode] = useState<ApplyMode>('add')
  const [job, setJob] = useState<ReadJob | null>(null)
  const [review, setReview] = useState<Review | null>(null)
  const [applied, setApplied] = useState<AppliedPreset | null>(null)
  const [applying, setApplying] = useState(false)
  const [err, setErr] = useState('')
  const [dragging, setDragging] = useState(false)
  const [startedAt, setStartedAt] = useState(0)
  const [now, setNow] = useState(() => Date.now())
  const dragDepth = useRef(0)
  const fileRef = useRef<HTMLInputElement>(null)
  const textRef = useRef<HTMLTextAreaElement>(null)
  const quota = useQuotaStore((s) => s.status)
  const name = agent.name

  useEffect(() => { if (!asUser) void useQuotaStore.getState().refresh() }, [asUser])
  useEffect(() => {
    getReader(agent.slug, asUser).then(setReader).catch(() => setReader({ model: '', step_tokens: 0 }))
  }, [agent.slug, asUser])
  useEffect(() => {
    getCurrentSetup(agent.slug, asUser).then(setCurrent).catch(() => setCurrent(null))
  }, [agent.slug, asUser])

  useEffect(() => {
    if (step !== 'reading') return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [step])

  // Follow the read until it ends.
  useEffect(() => {
    if (step !== 'reading' || !job || job.state !== 'running') return
    let live = true
    const t = setTimeout(() => {
      getRead(job.id, asUser).then((next) => {
        if (!live) return
        setJob(next)
        if (next.state === 'done' && next.result) {
          setReview(startReview(next.result))
          setStep('review')
        } else if (next.state === 'error') {
          setErr(next.error || 'Reading failed.')
          setStep('material')
        } else if (next.state === 'cancelled') {
          setStep('material')
        }
        if (next.state !== 'running' && !asUser) useQuotaStore.getState().refreshSoon()
      }).catch((e) => {
        if (!live) return
        setErr(message(e))
        setStep('material')
      })
    }, POLL_MS)
    return () => { live = false; clearTimeout(t) }
  }, [step, job, asUser])

  // Anything changed in the material: what was read from it no longer holds.
  const updateDocs = (fn: (prev: Doc[]) => Doc[]) => { setDocs(fn); setReview(null) }
  const editInstructions = (v: string) => { setInstructions(v); setReview(null) }
  const patchDoc = (key: string, patch: Partial<Doc>) => updateDocs((prev) => prev.map((d) => (d.key === key ? { ...d, ...patch } : d)))

  /** Read a document in full on the server: its pages and what reading it costs. */
  const stage = (doc: Doc) => {
    stageDocument(doc.file ? { file: doc.file } : { libraryId: doc.pick!.id }, asUser)
      .then(({ docs: got, warnings }) => patchDoc(doc.key, got[0]
        ? { status: 'ready', staged: got[0] }
        : { status: 'error', error: warnings.join(' ') || 'It could not be read.' }))
      .catch((e) => patchDoc(doc.key, { status: 'error', error: message(e) }))
  }

  const addFiles = (list: FileList | File[] | null) => {
    if (!list) return
    const incoming = Array.from(list)
    const unreadable = incoming.filter((f) => !SETUP_EXTS.has(extOf(f.name)))
    const tooBig = incoming.filter((f) => SETUP_EXTS.has(extOf(f.name)) && f.size > MAX_FILE_BYTES)
    const seen = new Set(docs.filter((d) => d.file).map((d) => `${d.name}:${d.size}`))
    const fresh = incoming.filter((f) => SETUP_EXTS.has(extOf(f.name)) && f.size <= MAX_FILE_BYTES && !seen.has(`${f.name}:${f.size}`))
    const room = MAX_FILES - docs.length
    setNotice([
      unreadable.length ? `Left out ${unreadable.map((f) => f.name).join(', ')} — not a document it can read.` : '',
      tooBig.length ? `Left out ${tooBig.length} file${tooBig.length === 1 ? '' : 's'} over 25 MB.` : '',
      fresh.length > room ? `Up to ${MAX_FILES} documents.` : '',
    ].filter(Boolean).join(' '))
    if (fresh.length && room > 0) {
      const added: Doc[] = fresh.slice(0, room).map((file) => ({
        key: nextKey(), name: file.name, size: file.size, description: '', usage: 'auto', read: true, status: 'staging', file,
      }))
      updateDocs((prev) => [...prev, ...added])
      added.forEach(stage)
    }
  }

  const addPicks = (items: LibraryItem[]) => {
    const have = new Set(docs.filter((d) => d.pick).map((d) => d.pick!.id))
    const fresh = items.filter((i) => !have.has(i.id)).slice(0, MAX_FILES - docs.length)
    setNotice('')
    if (fresh.length) {
      const added: Doc[] = fresh.map((pick) => ({
        key: nextKey(), name: pick.filename, size: pick.size, usage: 'auto', read: true, status: 'staging', pick,
        // The library's own title is a good first answer to "what is it?".
        description: pick.status === 'done' && pick.title ? pick.title : '',
      }))
      updateDocs((prev) => [...prev, ...added])
      added.forEach(stage)
    }
    setPicking(false)
  }

  const insertStarter = (text: string) => {
    const base = instructions.trimEnd()
    const lead = base ? `${base}\n` : ''
    editInstructions(lead + text)
    requestAnimationFrame(() => {
      const el = textRef.current
      if (!el) return
      el.focus()
      const open = lead.length + text.indexOf('[')
      const close = lead.length + text.indexOf(']') + 1
      if (open >= lead.length && close > open) el.setSelectionRange(open, close)
    })
  }

  const ready = docs.filter((d) => d.status === 'ready')
  const staging = docs.some((d) => d.status === 'staging')
  const hasMaterial = ready.length > 0 || !!instructions.trim()
  const stepTokens = reader?.step_tokens || 0
  const estimate = useMemo(() => estimateRead(docs, instructions, stepTokens), [docs, instructions, stepTokens])
  const needsModel = !!instructions.trim() || ready.some((d) => d.read)
  const noModel = needsModel && reader !== null && !reader.model
  const canRead = hasMaterial && !staging && !noModel && (reader !== null || !needsModel)

  const read = async () => {
    if (!canRead || step === 'reading') return
    if (review) { setStep('review'); return }
    setErr('')
    setJob(null)
    setStartedAt(Date.now())
    setNow(Date.now())
    setStep('reading')
    try {
      setJob(await startRead({
        agent: agent.slug,
        instructions,
        docs: ready.map((d) => ({ id: d.staged!.id, description: d.description, usage: d.usage, read: d.read })),
        asUser,
      }))
    } catch (e) {
      setErr(message(e))
      setStep('material')
    }
  }

  const stopReading = () => {
    if (job) void cancelRead(job.id, asUser)
    setJob(null)
    setStep('material')
  }

  const apply = async () => {
    if (!review || applying) return
    setApplying(true)
    setErr('')
    try {
      const res = await applyAgentPreset(agent.slug, reviewedPreset(review), current?.has ? mode : 'add', asUser)
      // The agent's chat sends its instructions: mirror what FD stored (or,
      // without accounts, put the block in this browser's copy, in place of
      // an earlier one) and re-send. Not for an agent an admin set up for
      // someone else.
      if (!asUser) {
        const procs = useProcessStore.getState()
        const had = procs.getFleetInstructions(agent.slug)
        const next = res.stored ? res.instructions : [stripOwnerBlock(had), res.block].filter(Boolean).join('\n\n')
        if (next !== had) {
          void procs.setFleetInstructions(agent.slug, next)
            .then(() => useChatStore.getState().resendFleetIdentity(`proc-${agent.slug}`))
        }
      }
      setApplied(res)
      setStep('done')
    } catch (e) {
      setErr(message(e))
    } finally {
      setApplying(false)
    }
  }

  const close = () => {
    if (applying) return
    const unsaved = step === 'reading' || step === 'review' || (step === 'material' && (hasMaterial || docs.length > 0))
    if (unsaved && !confirm(`Close without setting up ${name}? It works as it is, but what you added here is lost.`)) return
    if (step === 'reading' && job) void cancelRead(job.id, asUser)
    onClose()
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (picking) setPicking(false)
      else close()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // Files dropped anywhere on the dialog while the material is on screen.
  const dropProps = step === 'material' && !picking ? {
    onDragEnter: (e: DragEvent) => { if (e.dataTransfer.types.includes('Files')) { dragDepth.current++; setDragging(true) } },
    onDragOver: (e: DragEvent) => { if (e.dataTransfer.types.includes('Files')) e.preventDefault() },
    onDragLeave: () => { dragDepth.current = Math.max(0, dragDepth.current - 1); if (!dragDepth.current) setDragging(false) },
    onDrop: (e: DragEvent) => { e.preventDefault(); dragDepth.current = 0; setDragging(false); addFiles(e.dataTransfer.files) },
  } : {}

  const stepIndex = step === 'reading' ? 0 : STEPS.findIndex((s) => s.id === step)
  const readCount = ready.filter((d) => d.read).length
  const readLabel = staging ? 'Reading pages…'
    : review ? 'Back to the review'
      : estimate.steps === 0 && ready.length ? 'Continue — knowledge only'
        : readCount ? `Read ${plural(readCount, 'document')}${estimate.steps > 1 ? ` in ${estimate.steps} steps` : ''}`
          : 'Read it'
  const footHint = review ? 'Nothing changed since it was read.'
    : staging ? 'Reading the documents’ pages — no tokens yet.'
      : noModel ? (reader?.note || 'No model is set up to read with. Keep the documents as knowledge only, or ask your admin.')
        : hasMaterial ? 'You review everything before it reaches the agent.'
          : 'Add instructions or a document to start.'

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={close}>
      <div
        role="dialog"
        aria-label={`Set up ${name}`}
        className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-900 shadow-2xl wide:h-[min(88vh,52rem)] wide:max-w-5xl wide:rounded-2xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}
        {...dropProps}
      >
        {/* Header */}
        <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
          {step === 'review' ? (
            <button onClick={() => setStep('material')} disabled={applying} aria-label="Back to your material"
              className="-ml-2 rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
              <ArrowLeft className="h-5 w-5" />
            </button>
          ) : (
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
              <Wand2 className="h-4 w-4" />
            </span>
          )}
          <div className="min-w-0">
            <h3 className="truncate text-base font-semibold text-zinc-100">
              {step === 'review' ? `What ${name} starts with` : `Set up ${name}`}
            </h3>
            <p className="truncate text-xs text-zinc-500">
              {step === 'review' ? 'Keep what is right, fix what is not — then apply.'
                : `Your way of working${agent.role ? ` · ${agent.role}` : ''}${asUser && ownerLabel ? ` · for ${ownerLabel}` : ''}`}
            </p>
          </div>
          <ol className="ml-auto hidden items-center gap-1.5 text-[11px] md:flex" aria-label="Steps">
            {STEPS.map((s, i) => (
              <li key={s.id} className="flex items-center gap-1.5">
                {i > 0 && <span className="h-px w-4 bg-zinc-700" />}
                <span className={`flex items-center gap-1.5 rounded-full px-2 py-0.5 ${
                  i === stepIndex ? 'bg-violet-500/15 text-violet-700 dark:text-violet-200' : i < stepIndex ? 'text-zinc-400' : 'text-zinc-600'
                }`}>
                  <span className={`flex h-4 w-4 items-center justify-center rounded-full text-[9px] font-semibold ${
                    i < stepIndex ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400' : i === stepIndex ? 'bg-violet-600 text-white' : 'bg-zinc-800 text-zinc-500'
                  }`}>{i < stepIndex ? <Check className="h-2.5 w-2.5" /> : i + 1}</span>
                  {s.label}
                </span>
              </li>
            ))}
          </ol>
          <button onClick={close} disabled={applying} aria-label="Close" title="Close"
            className="ml-auto rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50 md:ml-2">
            <X className="h-5 w-5" />
          </button>
        </div>

        {step === 'material' && (
          <>
            {current?.has && <EarlierSetup name={name} current={current} mode={mode} onMode={(m) => { setMode(m); setReview(null) }} />}
            <div className="flex min-h-0 flex-1 flex-col overflow-y-auto wide:flex-row wide:overflow-hidden">
              {/* Documents */}
              <section className="order-2 shrink-0 px-4 py-4 wide:order-1 wide:min-h-0 wide:flex-1 wide:shrink wide:overflow-y-auto wide:px-6">
                <div className="mb-3 flex flex-col gap-2 wide:flex-row wide:items-center">
                  <div className="min-w-0 flex-1">
                    <h4 className="text-sm font-semibold text-zinc-100">
                      Documents {docs.length > 0 && <span className="font-normal text-zinc-500">· {docs.length} of {MAX_FILES}</span>}
                    </h4>
                    <p className="text-xs text-zinc-500">Policies, FAQs, procedures, price lists, templates — whatever {name} should go by.</p>
                  </div>
                  <div className="flex gap-1.5">
                    <button onClick={() => fileRef.current?.click()} disabled={docs.length >= MAX_FILES}
                      className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-50">
                      <Upload className="h-3.5 w-3.5" /> Upload
                    </button>
                    {!asUser && (
                      <button onClick={() => setPicking(true)} disabled={docs.length >= MAX_FILES}
                        className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-50">
                        <Library className="h-3.5 w-3.5" /> From your library
                      </button>
                    )}
                  </div>
                </div>
                <input ref={fileRef} type="file" multiple accept={SETUP_ACCEPT} className="hidden"
                  onChange={(e) => { addFiles(e.target.files); e.target.value = '' }} />

                {docs.length === 0 ? (
                  <button onClick={() => fileRef.current?.click()}
                    className="flex w-full flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-zinc-700 bg-zinc-950/40 px-6 py-10 text-center transition-colors hover:border-violet-500/50 hover:bg-zinc-900">
                    <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-violet-500/10 text-violet-600 dark:text-violet-300">
                      <FolderOpen className="h-6 w-6" />
                    </span>
                    <span className="text-sm font-medium text-zinc-200">Drop your documents here, or choose them</span>
                    <span className="text-xs text-zinc-500">PDF, Word, Excel, PowerPoint, text, Markdown, CSV · up to 25 MB each · up to {MAX_FILES}</span>
                    <span className="mt-1 text-[11px] text-zinc-600">Then say what each one is and how {name} should use it.</span>
                  </button>
                ) : (
                  <>
                    <ul className="space-y-2.5">
                      {docs.map((d) => (
                        <DocCard key={d.key} doc={d} agentName={name} stepTokens={stepTokens}
                          onChange={(patch) => patchDoc(d.key, patch)}
                          onRemove={() => updateDocs((prev) => prev.filter((x) => x.key !== d.key))} />
                      ))}
                    </ul>
                    {docs.length < MAX_FILES && (
                      <button onClick={() => fileRef.current?.click()}
                        className="mt-2.5 flex w-full items-center justify-center gap-1.5 rounded-xl border border-dashed border-zinc-700 px-3 py-2.5 text-xs text-zinc-500 transition-colors hover:border-violet-500/50 hover:text-zinc-300">
                        <Plus className="h-3.5 w-3.5" /> Drop more files here, or click to add
                      </button>
                    )}
                  </>
                )}
                {notice && <p className="mt-2 text-[11px] text-amber-600 dark:text-amber-400">{notice}</p>}
              </section>

              {/* On a phone, after the documents: they come first there. */}
              <div className="order-3 shrink-0 space-y-5 border-t border-zinc-800 px-4 py-4 wide:hidden">
                <HowItWorks name={name} />
                <UsageNote estimate={estimate} reader={reader} name={name} quota={asUser ? null : quota} knowledgeOnly={ready.some((d) => !d.read)} />
              </div>

              {/* Instructions, how it works, usage */}
              <aside className="order-1 shrink-0 space-y-5 border-b border-zinc-800 px-4 py-4 wide:order-2 wide:min-h-0 wide:w-[22rem] wide:overflow-y-auto wide:border-b-0 wide:border-l wide:bg-zinc-950/40 wide:px-5 lg:w-[24rem]">
                <div>
                  <label htmlFor="setup-instructions" className="text-sm font-semibold text-zinc-100">How you work</label>
                  <p className="mb-2 text-xs text-zinc-500">Anything {name} should always keep in mind.</p>
                  <textarea
                    id="setup-instructions"
                    ref={textRef}
                    value={instructions}
                    onChange={(e) => editInstructions(e.target.value)}
                    rows={6}
                    placeholder="e.g. Answer customers in Croatian. Never promise a delivery date. Ask me before refunding more than 200 EUR."
                    className="w-full resize-y rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-base text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none wide:text-sm"
                  />
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {STARTERS.map((s) => (
                      <button key={s.label} onClick={() => insertStarter(s.text)}
                        className="flex items-center gap-1 rounded-full border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-500 hover:border-zinc-700 hover:text-zinc-300">
                        <Plus className="h-2.5 w-2.5" /> {s.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="hidden space-y-5 wide:block">
                  <HowItWorks name={name} />
                  <UsageNote estimate={estimate} reader={reader} name={name} quota={asUser ? null : quota} knowledgeOnly={ready.some((d) => !d.read)} />
                </div>
              </aside>
            </div>

            <div className="shrink-0 border-t border-zinc-800 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:px-5">
              {err && <p className="mb-2 text-xs text-red-600 dark:text-red-400">{err}</p>}
              <div className="flex flex-col-reverse gap-2 wide:flex-row wide:items-center">
                <button onClick={close} className="rounded-xl px-3 py-2 text-sm text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                  Skip for now
                </button>
                <span className={`text-[11px] wide:ml-auto ${noModel ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-600'}`}>
                  {footHint}
                </span>
                <button onClick={read} disabled={!canRead && !review}
                  className="flex items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50">
                  {staging ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                  {readLabel}
                </button>
              </div>
            </div>
          </>
        )}

        {step === 'reading' && (
          <ReadingView job={job} docs={docs} name={name} now={now} elapsed={now - startedAt} onStop={stopReading} />
        )}

        {step === 'review' && review && (
          <>
            <PresetReview review={review} disabled={applying} onChange={setReview} />
            <div className="shrink-0 border-t border-zinc-800 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:px-5">
              {err && <p className="mb-2 text-xs text-red-600 dark:text-red-400">{err}</p>}
              <div className="flex flex-col-reverse gap-2 wide:flex-row wide:items-center">
                <button onClick={() => setStep('material')} disabled={applying}
                  className="rounded-xl border border-zinc-700 px-4 py-2.5 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800 disabled:opacity-50">
                  Back to your material
                </button>
                <span className="text-[11px] text-zinc-500 wide:ml-auto">
                  Keeping {keptCount(review)} item{keptCount(review) === 1 ? '' : 's'}
                </span>
                <button onClick={apply} disabled={applying || keptCount(review) === 0}
                  className="flex items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50">
                  {applying ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                  {applying ? 'Applying…' : current?.has && mode === 'replace' ? `Replace ${name}'s setup` : `Apply to ${name}`}
                </button>
              </div>
            </div>
          </>
        )}

        {step === 'done' && applied && (
          <>
            <Done name={name} applied={applied} hadSetup={!!current?.has} />
            <div className="shrink-0 border-t border-zinc-800 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:px-5">
              <div className="flex justify-end">
                <button onClick={onClose} className="flex w-full items-center justify-center gap-2 rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-violet-500 wide:w-auto">
                  <Check className="h-4 w-4" /> Done
                </button>
              </div>
            </div>
          </>
        )}

        {dragging && (
          <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center bg-violet-950/60 backdrop-blur-[1px]">
            <div className="flex flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-violet-400/70 bg-zinc-900/90 px-10 py-8 text-center">
              <Upload className="h-6 w-6 text-violet-500 dark:text-violet-300" />
              <p className="text-sm font-medium text-zinc-100">Drop to add to {name}'s documents</p>
            </div>
          </div>
        )}

        {picking && (
          <LibraryPicker
            have={new Set(docs.filter((d) => d.pick).map((d) => d.pick!.id))}
            room={MAX_FILES - docs.length}
            onCancel={() => setPicking(false)}
            onAdd={addPicks}
          />
        )}
      </div>
    </div>,
    document.body,
  )
}

function clock(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

// ── Pieces ───────────────────────────────────────────────────────────

function DocCard({ doc, agentName, stepTokens, onChange, onRemove }: {
  doc: Doc
  agentName: string
  /** How much one reading step takes (0 until known). */
  stepTokens: number
  onChange: (patch: Partial<Doc>) => void
  onRemove: () => void
}) {
  const look = fileLook(doc.name)
  const Icon = look.icon
  const st = doc.staged
  const steps = st && stepTokens ? Math.max(1, Math.ceil(st.tokens / stepTokens)) : 0
  return (
    <li className={`rounded-xl border bg-zinc-900/60 p-3 ${doc.status === 'error' ? 'border-red-500/40' : 'border-zinc-800'}`}>
      <div className="flex items-start gap-3">
        <span className={`flex h-10 w-10 shrink-0 flex-col items-center justify-center rounded-lg ${look.tone}`}>
          <Icon className="h-4 w-4" />
          <span className="mt-0.5 text-[8px] font-semibold uppercase tracking-wide">{look.tag}</span>
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-zinc-100" title={doc.name}>{doc.name}</p>
          <p className="flex min-w-0 items-center gap-1 text-[11px] text-zinc-500">
            {doc.pick
              ? <><Library className="h-3 w-3 shrink-0" /><span className="truncate">From your library · {doc.pick.agent_name}</span></>
              : <><Upload className="h-3 w-3 shrink-0" /><span>Uploaded</span></>}
            <span className="shrink-0">· {fmtBytes(doc.size)}</span>
          </p>
          {doc.status === 'staging' && (
            <p className="mt-0.5 flex items-center gap-1 text-[11px] text-zinc-500">
              <Loader2 className="h-3 w-3 animate-spin" /> Reading its pages…
            </p>
          )}
          {doc.status === 'error' && <p className="mt-0.5 text-[11px] text-red-600 dark:text-red-400">{doc.error}</p>}
          {st && (
            <p className="mt-0.5 text-[11px] text-zinc-400">
              {st.pages_exact ? plural(st.pages, 'page') : `about ${plural(st.pages, 'page')}`}
              {' · '}{kTokens(st.tokens)} tokens to read in full
              {st.truncated && <span className="text-amber-600 dark:text-amber-400"> · only its first {kTokens(st.chars)} characters are kept</span>}
            </p>
          )}
          {!!st?.tables && (
            <p className="mt-0.5 flex items-center gap-1 text-[11px] text-emerald-700 dark:text-emerald-400">
              <Table2 className="h-3 w-3 shrink-0" />
              {st.tables === 1
                ? `Also a table ${agentName} can query`
                : `Also ${st.tables} tables ${agentName} can query, one per sheet`}
              <span className="text-zinc-500">· exact values, no tokens</span>
            </p>
          )}
        </div>
        <button onClick={onRemove} aria-label={`Remove ${doc.name}`} title="Remove"
          className="rounded-lg p-1.5 text-zinc-600 hover:bg-zinc-800 hover:text-red-600 dark:hover:text-red-400">
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
      {doc.status !== 'error' && (
        <div className="mt-3 space-y-2.5 wide:pl-[3.25rem]">
          <label className="block">
            <span className="mb-1 block text-[11px] font-medium text-zinc-500">What is it?</span>
            <input
              value={doc.description}
              maxLength={500}
              onChange={(e) => onChange({ description: e.target.value })}
              placeholder="e.g. Our refund policy for EU customers, updated March 2026"
              className="w-full rounded-lg border border-zinc-800 bg-zinc-950 px-2.5 py-1.5 text-base text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none wide:text-sm"
            />
          </label>
          <div>
            <span className="mb-1 block text-[11px] font-medium text-zinc-500">How should {agentName} use it?</span>
            <div role="radiogroup" aria-label={`How ${agentName} should use ${doc.name}`} className="flex flex-wrap gap-1.5">
              {USAGE_ORDER.map((u) => {
                const { icon: UIcon, label } = USAGE_LOOKS[u]
                const on = doc.usage === u
                return (
                  // Reference material is looked up, not read for rules: knowledge only by default.
                  <button key={u} role="radio" aria-checked={on} onClick={() => onChange({ usage: u, read: u !== 'reference' })}
                    className={`flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs transition-colors ${
                      on
                        ? 'border-violet-500/50 bg-violet-500/15 text-violet-700 dark:text-violet-200'
                        : 'border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
                    }`}>
                    <UIcon className="h-3 w-3" /> {label}
                  </button>
                )
              })}
            </div>
            <p className="mt-1 text-[11px] text-zinc-500">{USAGE_LOOKS[doc.usage].blurb}</p>
          </div>
          <button
            role="switch"
            aria-checked={doc.read}
            onClick={() => onChange({ read: !doc.read })}
            className="flex w-full items-start gap-2.5 rounded-lg border border-zinc-800 px-2.5 py-2 text-left hover:border-zinc-700"
          >
            <span className={`mt-0.5 flex h-4 w-7 shrink-0 items-center rounded-full p-0.5 transition-colors ${doc.read ? 'bg-violet-600' : 'bg-zinc-700'}`}>
              <span className={`h-3 w-3 rounded-full bg-white transition-transform ${doc.read ? 'translate-x-3' : ''}`} />
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex flex-wrap items-baseline gap-x-2 text-xs text-zinc-200">
                {doc.read ? 'Read it for rules, skills and facts' : 'Knowledge only'}
                <span className="text-[11px] text-zinc-500">
                  {doc.read ? (st ? `~${kTokens(st.tokens)} tokens${steps > 1 ? ` · ${steps} steps` : ''}` : '') : 'no tokens'}
                </span>
              </span>
              <span className="block text-[11px] leading-snug text-zinc-500">
                {doc.read
                  ? `${agentName} also keeps the whole document to look things up in.`
                  : `Nothing is read now: ${agentName} keeps the whole document and looks things up in it when it needs to.`}
              </span>
            </span>
          </button>
        </div>
      )}
    </li>
  )
}

/** "6 rules, 1 skill, 3 documents" — the groups a setup gave, those it has. */
function setupSummary(c: Pick<SetupCounts, 'rules' | 'escalation' | 'templates' | 'glossary' | 'intents' | 'facts' | 'knowledge' | 'tables'>): string {
  const parts = [
    c.rules && plural(c.rules, 'rule'), c.intents && plural(c.intents, 'skill'), c.knowledge && plural(c.knowledge, 'document'),
    c.tables && plural(c.tables, 'table'), c.facts && plural(c.facts, 'fact'), c.escalation && `${c.escalation} ask-you-first`, c.templates && plural(c.templates, 'template'),
    c.glossary && plural(c.glossary, 'term'),
  ].filter(Boolean)
  return parts.length ? parts.join(', ') : 'nothing yet'
}

/** The setup an agent already has, and whether the new one adds to it or replaces it. */
function EarlierSetup({ name, current, mode, onMode }: {
  name: string
  current: CurrentSetup
  mode: ApplyMode
  onMode: (m: ApplyMode) => void
}) {
  const summary = setupSummary({
    ...current, intents: current.skills.length, knowledge: current.knowledge.length, tables: current.tables?.length ?? 0,
  })
  const options: { id: ApplyMode; label: string }[] = [{ id: 'add', label: 'Add to it' }, { id: 'replace', label: 'Replace it' }]
  return (
    <div className="shrink-0 border-b border-zinc-800 bg-violet-500/5 px-4 py-3 wide:px-6">
      <div className="flex flex-col gap-2 wide:flex-row wide:items-center">
        <p className="min-w-0 flex-1 text-xs text-zinc-400">
          <span className="font-medium text-zinc-200">{name} already has a setup:</span> {summary}.
          {current.knowledge.length > 0 && (
            <span className="text-zinc-500"> Documents: {current.knowledge.map((k) => k.title).join(', ')}.</span>
          )}
        </p>
        <div role="radiogroup" aria-label="Add to its setup or replace it" className="flex shrink-0 rounded-lg border border-zinc-800 p-0.5">
          {options.map((o) => (
            <button key={o.id} role="radio" aria-checked={mode === o.id} onClick={() => onMode(o.id)}
              className={`rounded-md px-3 py-1 text-xs transition-colors ${
                mode === o.id ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:text-zinc-200'
              }`}>
              {o.label}
            </button>
          ))}
        </div>
      </div>
      <p className="mt-1 text-[11px] text-zinc-500">
        {mode === 'add'
          ? 'What you add now joins it. A document you give again replaces its earlier copy (a spreadsheet, its tables too); newer rules come first.'
          : 'Its earlier rules, skills, documents and tables are taken out and this setup goes in their place. Facts it already learned stay.'}
      </p>
    </div>
  )
}

function HowItWorks({ name }: { name: string }) {
  const steps: { icon: LucideIcon; title: string; text: string }[] = [
    { icon: Upload, title: 'You add', text: 'Your instructions and documents — and, for each document, what it is and how to use it.' },
    { icon: Sparkles, title: 'Your model reads it', text: 'In steps it can take in, it turns it into rules, skills, facts, templates and a glossary. This uses your daily allowance; knowledge-only documents don’t.' },
    { icon: CheckCircle2, title: 'You review', text: `Keep what's right, fix what isn't. Nothing reaches ${name} until you apply.` },
  ]
  return (
    <div>
      <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">How it works</p>
      <ol className="space-y-2.5">
        {steps.map(({ icon: Icon, title, text }, i) => (
          <li key={title} className="flex gap-2.5">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-zinc-800 text-zinc-400">
              <Icon className="h-3 w-3" />
            </span>
            <div className="min-w-0">
              <p className="text-xs font-medium text-zinc-300">{i + 1}. {title}</p>
              <p className="text-[11px] leading-snug text-zinc-500">{text}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}

/** What reading the material costs, against what is left today. */
function UsageNote({ estimate, reader, name, quota, knowledgeOnly }: {
  estimate: { steps: number; tokens: number }
  reader: Reader | null
  /** The agent being set up. */
  name: string
  quota: QuotaStatus | null
  /** Some documents are kept as knowledge only. */
  knowledgeOnly: boolean
}) {
  const { steps, tokens } = estimate
  const whose = reader?.via === 'agent' ? `, ${name}'s own model` : ''
  // A paced read takes at least its tokens over the pace — say so when that's minutes.
  const minutes = reader?.pace_tpm ? Math.ceil(tokens / reader.pace_tpm) : 0
  const slow = minutes >= 2 ? `, about ${minutes} minutes or more — paced to stay under the provider's limits` : ''
  const how = reader?.model ? ` in ${plural(steps, 'step')} by ${reader.model}${whose}${slow}` : ''
  let body: ReactNode
  if (!tokens) {
    body = (
      <p className="text-[11px] text-zinc-500">
        {knowledgeOnly
          ? 'Documents kept as knowledge only cost no tokens.'
          : 'Reading your material takes tokens from your daily allowance — how many depends on what you add.'}
      </p>
    )
  } else if (quota?.byok) {
    body = <p className="text-[11px] text-zinc-500">About {kTokens(tokens)} tokens{how}, on your own AI keys.</p>
  } else if (quotaLimited(quota) && quota.tokens.limit > 0) {
    const { used, limit, remaining } = quota.tokens
    const left = remaining ?? Math.max(0, limit - used)
    const usedPct = Math.min(100, (used / limit) * 100)
    const estPct = Math.min(100 - usedPct, (tokens / limit) * 100)
    body = (
      <>
        <div className="flex h-1.5 overflow-hidden rounded-full bg-zinc-800" aria-hidden>
          <span className="bg-zinc-600" style={{ width: `${usedPct}%` }} />
          <span className={tokens > left ? 'bg-amber-500' : 'bg-violet-500'} style={{ width: `${estPct}%` }} />
        </div>
        <p className="mt-1.5 text-[11px] text-zinc-500">
          About <span className="text-zinc-300">{kTokens(tokens)}</span> tokens{how}, of the {kTokens(left)} you have left today.
        </p>
        {tokens > left && (
          <p className="mt-1 text-[11px] text-amber-600 dark:text-amber-400">
            That is more than you have left. Reading stops when the allowance runs out and keeps what it read —
            or keep the long documents as knowledge only, which costs nothing.
          </p>
        )}
      </>
    )
  } else {
    body = <p className="text-[11px] text-zinc-500">About {kTokens(tokens)} tokens{how}.</p>
  }
  return (
    <div>
      <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Usage</p>
      {body}
      {reader?.note && reader.model && <p className="mt-1 text-[11px] text-zinc-500">{reader.note}</p>}
    </div>
  )
}

/** The read under way: which part, how far, what it has spent. */
function ReadingView({ job, docs, name, now, elapsed, onStop }: {
  job: ReadJob | null
  docs: Doc[]
  /** The agent being set up. */
  name: string
  now: number
  elapsed: number
  onStop: () => void
}) {
  const wait = job?.waiting
  const left = wait && 'until' in wait ? Math.max(0, Math.ceil(wait.until - now / 1000)) : 0
  const waitLine = !wait || (wait.why === 'pacing' && left < 3) ? ''
    : wait.why === 'waking' ? `Waking ${wait.name}…`
      : wait.why === 'pacing' ? `Taking it slowly so the provider's limits aren't hit — next part in ${left} s`
        : `${wait.model} is busy (a rate limit, or overloaded) — trying again in ${left} s`
  const by = job?.model ? `By ${job.model}${job.via === 'agent' ? `, ${name}'s own model` : ''}` : ''
  const merging = job?.phase === 'merging'
  const steps = job?.steps || 0
  const pct = steps ? Math.round(((merging ? steps : Math.max(0, (job?.step || 1) - 1)) / steps) * 100) : 0
  const reading = docs.filter((d) => d.status === 'ready' && d.read)
  const kept = docs.filter((d) => d.status === 'ready' && !d.read)
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center overflow-y-auto px-6 py-10 text-center">
      <div className="relative mb-5 flex h-16 w-16 items-center justify-center">
        <span className="absolute inset-0 animate-ping rounded-2xl bg-violet-500/15" />
        <span className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-violet-500/15 text-violet-600 dark:text-violet-300">
          <Sparkles className="h-7 w-7" />
        </span>
      </div>
      <h4 className="text-base font-semibold text-zinc-100">{merging ? 'Putting the parts together' : 'Reading your material'}</h4>
      <p className="mt-1 max-w-md text-sm leading-relaxed text-zinc-500">
        {!job ? 'Starting…'
          : merging ? 'Merging what each part found into one setup — duplicates out, conflicts flagged.'
            : steps > 1 ? <>Part <span className="text-zinc-300">{job.step}</span> of {steps}: {job.current}</>
              : 'Pulling out rules, procedures, facts and terms.'}
      </p>
      <div className="mt-4 h-1.5 w-full max-w-sm overflow-hidden rounded-full bg-zinc-800" role="progressbar"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={steps > 1 && !merging ? pct : undefined}>
        {steps > 1 && !merging
          ? <span className="block h-full rounded-full bg-violet-500 transition-all" style={{ width: `${Math.max(4, pct)}%` }} />
          : <span className="block h-full w-1/3 animate-pulse rounded-full bg-violet-500/70" />}
      </div>
      <p className="mt-2 font-mono text-xs tabular-nums text-zinc-500">
        {clock(elapsed)}{job && job.tokens > 0 ? ` · ${kTokens(job.tokens)} tokens so far` : ''}
      </p>
      {by && <p className="mt-1 text-[11px] text-zinc-500">{by}</p>}
      {waitLine && (
        <p className={`mt-2 max-w-sm text-[11px] ${wait?.why === 'busy' ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-500'}`}
          role="status">{waitLine}</p>
      )}
      {steps > 3 && <p className="mt-1 text-[11px] text-zinc-600">Long documents take a while — keep this open until it is done.</p>}
      {(reading.length > 0 || kept.length > 0) && (
        <ul className="mt-5 w-full max-w-sm space-y-1.5 text-left">
          {[...reading, ...kept].map((d) => {
            const look = fileLook(d.name)
            const Icon = look.icon
            return (
              <li key={d.key} className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/60 px-2.5 py-1.5">
                <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded ${look.tone}`}><Icon className="h-3 w-3" /></span>
                <span className="min-w-0 flex-1 truncate text-xs text-zinc-300">{d.name}</span>
                {d.read
                  ? <Loader2 className="h-3 w-3 shrink-0 animate-spin text-zinc-500" />
                  : <span className="shrink-0 text-[10px] text-zinc-500">knowledge only</span>}
              </li>
            )
          })}
        </ul>
      )}
      <button onClick={onStop} className="mt-6 rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-300 hover:bg-zinc-800">
        Stop reading
      </button>
    </div>
  )
}

function Done({ name, applied, hadSetup }: { name: string; applied: AppliedPreset; hadSetup: boolean }) {
  const c = applied.counts
  const tiles: { icon: LucideIcon; n: number; label: string }[] = [
    { icon: ScrollText, n: c.rules, label: c.rules === 1 ? 'rule' : 'rules' },
    { icon: ListChecks, n: c.intents, label: c.intents === 1 ? 'skill' : 'skills' },
    { icon: Lightbulb, n: c.facts, label: c.facts === 1 ? 'fact' : 'facts' },
    { icon: Hand, n: c.escalation, label: 'ask you first' },
    { icon: BookOpen, n: c.knowledge, label: c.knowledge === 1 ? 'document' : 'documents' },
    { icon: Table2, n: c.tables ?? 0, label: c.tables === 1 ? 'table' : 'tables' },
    { icon: LayoutTemplate, n: c.templates, label: c.templates === 1 ? 'template' : 'templates' },
    { icon: BookA, n: c.glossary, label: c.glossary === 1 ? 'term' : 'terms' },
  ].filter((t) => t.n > 0)
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center overflow-y-auto px-6 py-10 text-center">
      <span className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
        <Check className="h-7 w-7" />
      </span>
      <h4 className="text-lg font-semibold text-zinc-100">{name} now works your way</h4>
      <p className="mt-1 text-sm text-zinc-500">It goes by this from its next reply.</p>
      {tiles.length > 0 && (
        <ul className="mt-6 grid w-full max-w-lg grid-cols-2 gap-2 wide:grid-cols-3">
          {tiles.map(({ icon: Icon, n, label }) => (
            <li key={label} className="flex items-center gap-2.5 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2.5 text-left">
              <Icon className="h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />
              <span className="text-lg font-semibold tabular-nums text-zinc-100">{n}</span>
              <span className="text-xs text-zinc-500">{label}</span>
            </li>
          ))}
        </ul>
      )}
      {hadSetup && applied.mode === 'add' && (
        <p className="mt-4 text-xs text-zinc-400">Added to its earlier setup — in all it now has {setupSummary(applied.totals)}.</p>
      )}
      {c.facts > 0 && (
        <p className="mt-4 text-xs text-zinc-500">
          {applied.running ? 'Its facts are being added now.' : `Its facts are added as soon as ${name} is up.`}
        </p>
      )}
      {c.knowledge > 0 && (
        <p className="mt-1 text-xs text-zinc-500">
          The documents are in its workspace, in the knowledge folder
          {applied.parts > 0 ? ` — the long ones in ${applied.parts} parts, so it can search all of them` : ''}.
        </p>
      )}
      {!!applied.tables?.length && (
        <p className="mt-1 text-xs text-zinc-500">
          {applied.running ? 'The spreadsheets are going into its datastore now' : `The spreadsheets go into its datastore as soon as ${name} is up`}
          {' '}as {plural(applied.tables.length, 'table')} it can query: {applied.tables.join(', ')}.
        </p>
      )}
      {!!applied.warnings?.length && (
        <ul className="mt-3 max-w-lg space-y-0.5 text-left text-xs text-amber-700 dark:text-amber-400">
          {applied.warnings.map((w) => <li key={w}>{w}</li>)}
        </ul>
      )}
    </div>
  )
}

/** Documents already on the deck: what the person's agents made or were
 *  given, from the File library. */
function LibraryPicker({ have, room, onCancel, onAdd }: {
  have: Set<string>
  room: number
  onCancel: () => void
  onAdd: (items: LibraryItem[]) => void
}) {
  const [q, setQ] = useState('')
  const [items, setItems] = useState<LibraryItem[] | null>(null)
  const [err, setErr] = useState('')
  const [sel, setSel] = useState<Map<string, LibraryItem>>(() => new Map())

  useEffect(() => {
    let live = true
    const t = setTimeout(() => {
      fileLibraryApi.files({ q: q.trim(), limit: 300, sort: 'recent' })
        .then((page) => { if (live) { setItems(page.items.filter((i) => SETUP_EXTS.has(extOf(i.filename)))); setErr('') } })
        .catch((e) => { if (live) setErr(e instanceof Error ? e.message : String(e)) })
    }, q ? 250 : 0)
    return () => { live = false; clearTimeout(t) }
  }, [q])

  const toggle = (item: LibraryItem) => {
    const next = new Map(sel)
    if (next.has(item.id)) next.delete(item.id)
    else if (next.size < room) next.set(item.id, item)
    setSel(next)
  }

  return (
    <div className="absolute inset-0 z-10 flex flex-col bg-zinc-900">
      <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
        <button onClick={onCancel} aria-label="Back to your material" className="-ml-2 rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
          <ArrowLeft className="h-5 w-5" />
        </button>
        <div className="min-w-0">
          <h3 className="truncate text-base font-semibold text-zinc-100">From your library</h3>
          <p className="truncate text-xs text-zinc-500">Documents your agents made or were given · up to {room} more</p>
        </div>
      </div>
      <div className="shrink-0 px-4 pt-3 wide:px-5">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-600" />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search by name, title or what's in it…"
            aria-label="Search your library"
            className="w-full rounded-xl border border-zinc-700 bg-zinc-950 py-2.5 pl-10 pr-3 text-base text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none wide:py-2 wide:text-sm"
          />
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 wide:px-5">
        {err && <p className="py-10 text-center text-sm text-red-600 dark:text-red-400">Couldn't load your library: {err}</p>}
        {!err && items === null && (
          <p className="flex items-center justify-center gap-2 py-10 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</p>
        )}
        {items && items.length === 0 && (
          <p className="py-10 text-center text-sm text-zinc-500">
            {q.trim() ? `No documents match “${q.trim()}”.` : 'No documents in your library yet. Files your agents make or are given show up here once the Library has looked for them.'}
          </p>
        )}
        {items && items.length > 0 && (
          <ul className="grid grid-cols-1 gap-2 md:grid-cols-2">
            {items.map((item) => {
              const added = have.has(item.id)
              const on = sel.has(item.id)
              const full = !on && sel.size >= room
              const look = fileLook(item.filename)
              const Icon = look.icon
              return (
                <li key={item.id}>
                  <button
                    onClick={() => toggle(item)}
                    disabled={added || full}
                    aria-pressed={on}
                    className={`flex h-full w-full gap-3 rounded-xl border p-3 text-left transition-colors disabled:opacity-50 ${
                      on ? 'border-violet-500/60 bg-violet-500/10 ring-1 ring-violet-500/40' : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700'
                    }`}
                  >
                    <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${look.tone}`}><Icon className="h-4 w-4" /></span>
                    <span className="flex min-w-0 flex-1 flex-col">
                      <span className="flex items-center gap-1.5">
                        <span className="truncate text-sm font-medium text-zinc-100">{item.title || item.filename}</span>
                        {(on || added) && <Check className="ml-auto h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />}
                      </span>
                      <span className="truncate text-[11px] text-zinc-500">
                        {item.title ? `${item.filename} · ` : ''}{item.agent_name} · {fmtBytes(item.size)}
                      </span>
                      {item.summary && <span className="mt-1 line-clamp-2 text-xs leading-snug text-zinc-500">{item.summary}</span>}
                      {added && <span className="mt-1 text-[10px] text-zinc-500">Already added</span>}
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </div>
      <div className="shrink-0 border-t border-zinc-800 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:px-5">
        <div className="flex flex-col-reverse gap-2 wide:flex-row wide:items-center wide:justify-end">
          <button onClick={onCancel} className="rounded-xl border border-zinc-700 px-4 py-2.5 text-sm font-medium text-zinc-300 hover:bg-zinc-800">Cancel</button>
          <button onClick={() => onAdd([...sel.values()])} disabled={sel.size === 0}
            className="flex items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50">
            <Plus className="h-4 w-4" /> {sel.size ? `Add ${sel.size} document${sel.size === 1 ? '' : 's'}` : 'Add documents'}
          </button>
        </div>
      </div>
    </div>
  )
}
