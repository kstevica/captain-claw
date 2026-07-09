import { useRef, useState } from 'react'
import {
  Wand2, X, Loader2, Sparkles, Upload, FileText, Trash2, Check,
  AlertTriangle, Maximize2,
} from 'lucide-react'
import { forgeArchetypes, createArchetype, type ArchetypeInput } from '../../services/archetypes'
import type { TierMap } from '../../services/tierConfig'

// Archetype tiers accepted by the backend validator (coding/vision are catalog-only).
const ARCH_TIERS = ['reason', 'balanced', 'fast', 'longctx']
const ACCEPT = '.pdf,.docx,.xlsx,.pptx,.txt,.md,.csv,.json'
const MAX_FILE_BYTES = 25 * 1024 * 1024

function slugify(s: string): string {
  return (s || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
}
function fmtBytes(n: number): string {
  return n >= 1024 * 1024 ? `${(n / (1024 * 1024)).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`
}

type Phase = 'input' | 'review' | 'done'

// Forge a set of reusable archetypes from instructions + optional documents, then
// review and save the chosen ones to the caller's Library.
export function ForgeArchetypesModal({ tiers, forgeTier, existingIds, toolPalette, onClose, onSaved }: {
  tiers: TierMap
  forgeTier: string
  existingIds: string[]
  toolPalette: string[]
  onClose: () => void
  onSaved: (count: number) => void
}) {
  const [phase, setPhase] = useState<Phase>('input')
  const [instructions, setInstructions] = useState('')
  const [count, setCount] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [drafts, setDrafts] = useState<ArchetypeInput[]>([])
  const [selected, setSelected] = useState<Set<number>>(new Set())
  // Which draft is shown in the right-hand detail pane (master-detail layout).
  const [activeIdx, setActiveIdx] = useState<number | null>(null)
  const [savedCount, setSavedCount] = useState(0)
  // Index of the draft whose SOP is being edited in the large pop-out editor.
  const [sopEdit, setSopEdit] = useState<number | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const ft = tiers[forgeTier]
  const inputCls = 'w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none'
  const labelCls = 'mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500'

  const addFiles = (list: FileList | File[] | null) => {
    if (!list) return
    const incoming = Array.from(list)
    const tooBig = incoming.filter((f) => f.size > MAX_FILE_BYTES)
    const ok = incoming.filter((f) => f.size <= MAX_FILE_BYTES)
    if (tooBig.length) setError(`Skipped ${tooBig.length} file(s) over 25 MB.`)
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name + f.size))
      return [...prev, ...ok.filter((f) => !names.has(f.name + f.size))]
    })
  }
  const removeFile = (i: number) => setFiles((prev) => prev.filter((_, idx) => idx !== i))

  const toggleIn = (setter: (u: (p: Set<number>) => Set<number>) => void, i: number) =>
    setter((prev) => { const n = new Set(prev); n.has(i) ? n.delete(i) : n.add(i); return n })
  const updateDraft = (i: number, patch: Partial<ArchetypeInput>) =>
    setDrafts((prev) => prev.map((d, idx) => (idx === i ? { ...d, ...patch } : d)))
  const toggleTool = (i: number, t: string) =>
    setDrafts((prev) => prev.map((d, idx) => (idx === i
      ? { ...d, tools: d.tools.includes(t) ? d.tools.filter((x) => x !== t) : [...d.tools, t] }
      : d)))

  const allSel = drafts.length > 0 && selected.size === drafts.length
  const toggleAll = () => setSelected(allSel ? new Set() : new Set(drafts.map((_, i) => i)))

  const doForge = async () => {
    if (!ft || !ft.model.trim()) {
      setError(`Configure the "${forgeTier}" tier model in Model Tiers first.`)
      return
    }
    if (!instructions.trim() && files.length === 0) {
      setError('Add instructions or at least one document.')
      return
    }
    setLoading(true); setError('')
    try {
      const result = await forgeArchetypes({
        instructions: instructions.trim(),
        files,
        provider: ft.provider, model: ft.model, apiKey: ft.api_key, baseUrl: ft.base_url,
        maxTokens: ft.output_ctx > 0 ? ft.output_ctx : 0,
        count: count ? parseInt(count, 10) || 0 : 0,
      })
      if (result.length === 0) { setError('No archetypes were produced. Add more detail and retry.'); return }
      setDrafts(result)
      setSelected(new Set(result.map((_, i) => i)))
      setActiveIdx(0)
      setPhase('review')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const doSave = async () => {
    setSaving(true); setError('')
    const used = new Set(existingIds)
    let saved = 0
    for (let i = 0; i < drafts.length; i++) {
      if (!selected.has(i)) continue
      const d = drafts[i]
      const base = slugify(d.archetype_id || d.role) || 'archetype'
      let id = base, n = 2
      while (used.has(id)) { id = `${base}-${n}`; n++ }
      used.add(id)
      try {
        await createArchetype({ ...d, archetype_id: id })
        saved++
      } catch (e) {
        console.warn('Forge archetypes: save failed', id, e)
      }
    }
    setSavedCount(saved)
    setSaving(false)
    onSaved(saved)
    setPhase('done')
  }

  // Back from review → keep the original forging task (instructions, files,
  // count) so the user can tweak and re-forge without re-entering everything.
  // Only the generated drafts and their selection are discarded.
  const goBack = () => {
    setPhase('input'); setDrafts([]); setSelected(new Set()); setActiveIdx(null); setError('')
  }

  // Fresh start ("Forge more" after a successful save) — clear the task too.
  const resetToInput = () => {
    goBack()
    setInstructions(''); setFiles([]); setCount(''); setSavedCount(0)
  }

  // Palette shown per card: known tools unioned with the draft's own (so a
  // generated tool not in the base set still appears).
  const paletteFor = (d: ArchetypeInput) => [...new Set([...toolPalette, ...d.tools])].sort()

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
        <div className="flex h-[80vh] w-[80vw] flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl" onClick={(e) => e.stopPropagation()}>
          {/* Header */}
          <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-4 py-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-300">
              <Wand2 className="h-3.5 w-3.5" />
            </span>
            <div>
              <h3 className="text-sm font-semibold text-zinc-200">Forge archetypes</h3>
              <p className="text-[11px] text-zinc-500">Describe the roles you need — add documents for context — and design a reusable set.</p>
            </div>
            <button onClick={onClose} className="ml-auto rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
              <X className="h-4 w-4" />
            </button>
          </div>

          {/* Scrollable body */}
          <div className="min-h-0 flex-1 overflow-auto p-4">
            {/* ── Input phase — fills the height: big instructions left, docs + options right ── */}
            {phase === 'input' && (
              <div className="flex h-full flex-col gap-4">
                <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-5">
                  {/* Left: instructions (grows to fill) */}
                  <div className="flex min-h-0 flex-col lg:col-span-3">
                    <label className={labelCls}>Instructions</label>
                    <textarea
                      value={instructions}
                      onChange={(e) => setInstructions(e.target.value)}
                      placeholder="Describe the domain, team, or process — the roles you want as reusable agents. e.g. 'A due-diligence bench for VC deals: a market scanner, a financials analyst, a founder-reference checker, and an IC-memo writer.'"
                      className={`${inputCls} min-h-0 flex-1 resize-none leading-relaxed`}
                      autoFocus
                    />
                    <p className="mt-1.5 text-[11px] text-zinc-500">
                      Designs on the <strong className="text-zinc-400">{forgeTier}</strong> tier
                      {ft?.model ? ` · ${ft.provider}/${ft.model}` : ' (unset)'}.
                    </p>
                  </div>

                  {/* Right: documents (grows) + count */}
                  <div className="flex min-h-0 flex-col gap-4 lg:col-span-2">
                    <div className="flex min-h-0 flex-1 flex-col">
                      <label className={labelCls}>Reference documents <span className="font-normal normal-case text-zinc-600">— optional</span></label>
                      <div
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => { e.preventDefault(); addFiles(e.dataTransfer.files) }}
                        className="flex min-h-[120px] flex-1 cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed border-zinc-700 bg-zinc-950/50 px-3 py-6 text-center hover:border-violet-500/50 hover:bg-zinc-900"
                      >
                        <Upload className="h-5 w-5 text-zinc-500" />
                        <p className="text-[12px] text-zinc-400">Drop files here or click to browse</p>
                        <p className="text-[10px] text-zinc-600">PDF, DOCX, XLSX, PPTX, TXT, MD — their contents ground the archetypes</p>
                      </div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept={ACCEPT}
                        className="hidden"
                        onChange={(e) => { addFiles(e.target.files); e.target.value = '' }}
                      />
                      {files.length > 0 && (
                        <div className="mt-2 max-h-40 space-y-1 overflow-auto">
                          {files.map((f, i) => (
                            <div key={f.name + i} className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950/50 px-2.5 py-1.5">
                              <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                              <span className="flex-1 truncate text-[12px] text-zinc-300">{f.name}</span>
                              <span className="text-[10px] text-zinc-600">{fmtBytes(f.size)}</span>
                              <button onClick={() => removeFile(i)} className="rounded p-0.5 text-zinc-600 hover:text-red-600 dark:hover:text-red-400">
                                <Trash2 className="h-3 w-3" />
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <div>
                      <label className={labelCls}>How many? <span className="font-normal normal-case text-zinc-600">optional — blank lets the model decide</span></label>
                      <input
                        type="number" min={1} max={20}
                        value={count}
                        onChange={(e) => setCount(e.target.value)}
                        placeholder="auto"
                        className={`${inputCls} w-32`}
                      />
                    </div>
                  </div>
                </div>

                {error && (
                  <div className="flex shrink-0 items-center gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-[12px] text-red-700 dark:text-red-400">
                    <AlertTriangle className="h-3.5 w-3.5 shrink-0" /> {error}
                  </div>
                )}
              </div>
            )}

            {/* ── Review phase — master-detail: pick list on the left, the active
                 archetype's full editor on the right ── */}
            {phase === 'review' && (() => {
              const active = activeIdx !== null ? drafts[activeIdx] : null
              return (
                <div className="flex h-full flex-col gap-3">
                  <div className="flex shrink-0 items-center gap-2">
                    <label className="flex cursor-pointer select-none items-center gap-2 text-xs text-zinc-400">
                      <input type="checkbox" checked={allSel} onChange={toggleAll} className="accent-violet-500" />
                      {selected.size} of {drafts.length} selected
                    </label>
                    <span className="ml-auto text-[11px] text-zinc-500">Pick an archetype to review its tools &amp; instructions, and choose what to keep</span>
                  </div>

                  <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-[300px_1fr]">
                    {/* Left: archetype list */}
                    <div className="flex min-h-0 flex-col overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950/40">
                      <div className="min-h-0 flex-1 overflow-auto">
                        {drafts.map((d, i) => {
                          const sel = selected.has(i)
                          const isActive = i === activeIdx
                          return (
                            <div
                              key={i}
                              onClick={() => setActiveIdx(i)}
                              className={`flex cursor-pointer items-start gap-2 border-b border-zinc-800/60 px-3 py-2.5 last:border-b-0 ${
                                isActive ? 'bg-violet-100 dark:bg-violet-500/15' : 'hover:bg-zinc-800/10'
                              }`}
                            >
                              <input
                                type="checkbox"
                                checked={sel}
                                onClick={(e) => e.stopPropagation()}
                                onChange={() => toggleIn(setSelected, i)}
                                className="mt-0.5 shrink-0 accent-violet-500"
                              />
                              <div className="min-w-0 flex-1">
                                <p className={`truncate text-[13px] font-medium ${isActive ? 'text-violet-800 dark:text-violet-200' : 'text-zinc-200'}`}>
                                  {d.role || 'Untitled'}
                                </p>
                                <p className="truncate text-[11px] text-zinc-500">{d.family} · {d.tier}</p>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>

                    {/* Right: full editor for the active archetype */}
                    {active && activeIdx !== null && (
                      <div className="flex min-h-0 flex-col gap-3 overflow-auto rounded-lg border border-zinc-800 bg-zinc-950/20 p-4">
                        <div className="flex items-center gap-3">
                          <input
                            value={active.role}
                            onChange={(e) => updateDraft(activeIdx, { role: e.target.value })}
                            placeholder="Role"
                            className="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm font-medium text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          />
                          <label className="flex shrink-0 cursor-pointer select-none items-center gap-1.5 text-[11px] text-zinc-400">
                            <input type="checkbox" checked={selected.has(activeIdx)} onChange={() => toggleIn(setSelected, activeIdx)} className="accent-violet-500" />
                            Keep
                          </label>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className={labelCls}>Family</label>
                            <input value={active.family} onChange={(e) => updateDraft(activeIdx, { family: e.target.value })} className={inputCls} />
                          </div>
                          <div>
                            <label className={labelCls}>Tier</label>
                            <select
                              value={ARCH_TIERS.includes(active.tier) ? active.tier : 'balanced'}
                              onChange={(e) => updateDraft(activeIdx, { tier: e.target.value })}
                              className={inputCls}
                            >
                              {ARCH_TIERS.map((t) => <option key={t} value={t}>{t}</option>)}
                            </select>
                          </div>
                        </div>

                        <div>
                          <label className={labelCls}>Description</label>
                          <input value={active.description} onChange={(e) => updateDraft(activeIdx, { description: e.target.value })} className={inputCls} />
                        </div>

                        <div>
                          <label className={labelCls}>Tools — click to toggle{active.tools.length ? ` (${active.tools.length} selected)` : ''}</label>
                          <div className="flex flex-wrap gap-1.5 rounded-lg border border-zinc-700 bg-zinc-950/40 p-2">
                            {paletteFor(active).map((t) => {
                              const on = active.tools.includes(t)
                              return (
                                <button
                                  key={t}
                                  type="button"
                                  onClick={() => toggleTool(activeIdx, t)}
                                  className={`rounded border px-2 py-0.5 font-mono text-[11px] transition-colors ${
                                    on
                                      ? 'border-violet-500/40 bg-violet-500/15 text-violet-700 dark:text-violet-200'
                                      : 'border-zinc-700 bg-zinc-900/40 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300'
                                  }`}
                                >
                                  {t}
                                </button>
                              )
                            })}
                          </div>
                        </div>

                        <div className="flex min-h-[220px] flex-1 flex-col">
                          <div className="mb-1 flex items-center justify-between">
                            <label className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">Fleet instructions (SOP)</label>
                            <button
                              onClick={() => setSopEdit(activeIdx)}
                              className="flex items-center gap-1 text-[11px] text-violet-700 hover:text-violet-800 dark:text-violet-300 dark:hover:text-violet-200"
                            >
                              <Maximize2 className="h-3 w-3" /> Expand editor
                            </button>
                          </div>
                          <textarea
                            value={active.fleet_instructions}
                            onChange={(e) => updateDraft(activeIdx, { fleet_instructions: e.target.value })}
                            className={`${inputCls} min-h-0 flex-1 resize-none font-mono text-[12px] leading-relaxed`}
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {error && (
                    <div className="flex shrink-0 items-center gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-[12px] text-red-700 dark:text-red-400">
                      <AlertTriangle className="h-3.5 w-3.5 shrink-0" /> {error}
                    </div>
                  )}
                </div>
              )
            })()}

            {/* ── Done phase ── */}
            {phase === 'done' && (
              <div className="mx-auto max-w-3xl">
                <div className="flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-3 text-sm text-emerald-700 dark:text-emerald-300">
                  <Check className="h-4 w-4 shrink-0" />
                  Saved {savedCount} archetype{savedCount !== 1 ? 's' : ''} to your Library.
                </div>
              </div>
            )}
          </div>

          {/* Sticky footer */}
          <div className="flex shrink-0 items-center gap-2 border-t border-zinc-800 px-4 py-3">
            {phase === 'input' && (
              <>
                <div className="flex-1" />
                <button onClick={onClose} className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-300 hover:text-zinc-100">
                  Cancel
                </button>
                <button
                  onClick={doForge}
                  disabled={loading}
                  className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50"
                >
                  {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                  {loading ? 'Forging…' : 'Forge archetypes'}
                </button>
              </>
            )}
            {phase === 'review' && (
              <>
                <button onClick={goBack} className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-300 hover:text-zinc-100">
                  Back
                </button>
                <div className="flex-1" />
                <button
                  onClick={doSave}
                  disabled={saving || selected.size === 0}
                  className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50"
                >
                  {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
                  {saving ? 'Saving…' : `Save ${selected.size} archetype${selected.size !== 1 ? 's' : ''}`}
                </button>
              </>
            )}
            {phase === 'done' && (
              <>
                <div className="flex-1" />
                <button onClick={resetToInput} className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-300 hover:text-zinc-100">
                  Forge more
                </button>
                <button onClick={onClose} className="rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500">
                  Done
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Large pop-out SOP editor */}
      {sopEdit !== null && drafts[sopEdit] && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-6" onClick={() => setSopEdit(null)}>
          <div className="flex h-[85vh] w-[85vw] flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-4 py-3">
              <h3 className="text-sm font-semibold text-zinc-200">Fleet instructions — {drafts[sopEdit].role || 'archetype'}</h3>
              <button onClick={() => setSopEdit(null)} className="ml-auto rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 p-4">
              <textarea
                value={drafts[sopEdit].fleet_instructions}
                onChange={(e) => updateDraft(sopEdit, { fleet_instructions: e.target.value })}
                className="h-full w-full resize-none rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 font-mono text-[13px] leading-relaxed text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                autoFocus
              />
            </div>
            <div className="flex shrink-0 items-center justify-end border-t border-zinc-800 px-4 py-3">
              <button onClick={() => setSopEdit(null)} className="rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500">
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
