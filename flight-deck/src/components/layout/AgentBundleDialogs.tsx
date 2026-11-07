/** Agent bundles in the kiosk: an agent's ⋯ → Back up / export, and New agent
 *  → From a file. A bundle is one agent in a .captain file. */

import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  Archive, Brain, CheckCircle2, ChevronDown, ChevronRight, Download, FileUp, GraduationCap, KeyRound, Loader2,
  Plug, ShieldCheck, TriangleAlert,
} from 'lucide-react'
import { AgentDialog } from './KioskAgentDialogs'
import {
  countChips, discardUpload, downloadExport, exportAgent, formatBytes, importUpload, myTiers, reviewAgent, uploadBundle,
  type AgentReview, type BundleCategory, type Connection, type ExportResult, type ImportResult, type ModelTier,
  type Preset, type UploadResult,
} from '../../services/agentBundle'

const btn = 'inline-flex h-10 items-center justify-center gap-2 rounded-xl px-4 text-sm font-medium transition-colors disabled:opacity-50'
const primary = `${btn} bg-violet-600 text-white hover:bg-violet-500`
const quiet = `${btn} border border-zinc-800 text-zinc-300 hover:bg-zinc-900`
const field = 'h-10 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 text-sm text-zinc-100 outline-none focus:border-violet-500'

function Chips({ items }: { items: string[] }) {
  if (!items.length) return null
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((c) => (
        <span key={c} className="rounded-full border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-xs text-zinc-300">{c}</span>
      ))}
    </div>
  )
}

function Notes({ notes }: { notes: string[] }) {
  if (!notes.length) return null
  return (
    <ul className="space-y-1.5 rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-xs leading-relaxed text-amber-800 dark:text-amber-200">
      {notes.map((n) => (
        <li key={n} className="flex gap-2"><TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />{n}</li>
      ))}
    </ul>
  )
}

function Problem({ text }: { text: string }) {
  return <p role="alert" className="rounded-xl border border-red-500/30 bg-red-500/5 p-3 text-sm text-red-700 dark:text-red-300">{text}</p>
}

function Connections({ items, title, onConnect }: { items: Connection[]; title: string; onConnect?: () => void }) {
  if (!items.length) return null
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
      <p className="flex items-center gap-2 text-xs font-medium text-zinc-300"><Plug className="h-3.5 w-3.5 text-sky-500" />{title}</p>
      <ul className="mt-2 space-y-1 text-xs text-zinc-400">
        {items.map((c) => <li key={`${c.kind}:${c.label}`} className="flex gap-2"><span className="text-zinc-600">•</span>{c.label}</li>)}
      </ul>
      {onConnect && (
        <button onClick={onConnect} className="mt-3 text-xs font-medium text-violet-700 hover:underline dark:text-violet-300">Open Connections</button>
      )}
    </div>
  )
}

const STAYS_OUT = 'Keys, tokens and its connections (mail, Telegram, MCP servers) never go in the file.'

const SectionTitle = ({ children }: { children: string }) => (
  <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">{children}</p>
)

// ── Back up / export ─────────────────────────────────────────────────────────

const PRESETS: { id: Preset; label: string; icon: typeof Archive; hint: string; categories: BundleCategory[] }[] = [
  { id: 'template', label: 'Template', icon: GraduationCap, hint: 'Who it is and what it was taught. For a fresh copy someone else starts from.',
    categories: ['identity', 'onboarding'] },
  { id: 'trained', label: 'Trained', icon: Brain, hint: 'And what it learned since — without its conversations and files.',
    categories: ['identity', 'onboarding', 'memory', 'data', 'schedules'] },
  { id: 'full', label: 'Full backup', icon: Archive, hint: 'Everything, to restore it later: conversations and files too.',
    categories: ['identity', 'onboarding', 'memory', 'data', 'conversations', 'files', 'schedules'] },
]

const CATEGORIES: { id: BundleCategory; label: string; hint: string }[] = [
  { id: 'identity', label: 'Identity', hint: 'Name, role, instructions, model, how it thinks' },
  { id: 'onboarding', label: 'Onboarding', hint: 'Documents, lessons, skills, practice and exams' },
  { id: 'memory', label: 'What it learned', hint: 'Insights, intentions, reflections, playbooks' },
  { id: 'data', label: 'Data tables', hint: 'Its datastore' },
  { id: 'conversations', label: 'Conversations', hint: 'Sessions, topics and the chat log' },
  { id: 'files', label: 'Files', hint: 'Its workspace' },
  { id: 'schedules', label: 'Scheduled tasks', hint: 'They arrive switched off' },
]

function Check({ checked, onChange, disabled, label, hint, right }: {
  checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; label: string; hint?: string; right?: ReactNode
}) {
  return (
    <label className={`flex items-start gap-3 rounded-lg px-2 py-1.5 ${disabled ? 'opacity-60' : 'cursor-pointer hover:bg-zinc-900'}`}>
      <input type="checkbox" className="mt-0.5 h-4 w-4 accent-violet-600" checked={checked} disabled={disabled}
        onChange={(e) => onChange(e.target.checked)} />
      <span className="min-w-0 flex-1">
        <span className="block text-sm text-zinc-200">{label}</span>
        {hint && <span className="block text-xs text-zinc-500">{hint}</span>}
      </span>
      {right}
    </label>
  )
}

export function AgentBackupDialog({ agent, asUser, onClose }: {
  agent: { slug: string; name: string }
  asUser?: string
  onClose: () => void
}) {
  const [preset, setPreset] = useState<Preset>('full')
  const [cats, setCats] = useState<Set<BundleCategory>>(() => new Set(PRESETS[2].categories))
  const [review, setReview] = useState<AgentReview | null>(null)
  const [reviewErr, setReviewErr] = useState('')
  const [insightCats, setInsightCats] = useState<Set<string>>(new Set())
  const [excluded, setExcluded] = useState<Set<string>>(new Set())
  const [open, setOpen] = useState<string | null>(null)
  const [contacts, setContacts] = useState(false)
  const [withKeys, setWithKeys] = useState(false)
  const [pass, setPass] = useState('')
  const [pass2, setPass2] = useState('')
  const [busy, setBusy] = useState<'' | 'packing' | 'saving'>('')
  const [made, setMade] = useState<ExportResult | null>(null)
  const [err, setErr] = useState('')

  const full = preset === 'full'
  const reviewing = !full && cats.has('memory')
  useEffect(() => {
    if (!reviewing || review) return
    let gone = false
    reviewAgent(agent.slug, asUser).then((r) => {
      if (gone) return
      setReview(r)
      setInsightCats(new Set(r.insights.filter((c) => c.default).map((c) => c.id)))
    }).catch((e) => { if (!gone) setReviewErr(e instanceof Error ? e.message : String(e)) })
    return () => { gone = true }
  }, [reviewing, review, agent.slug, asUser])

  const choosePreset = (p: Preset) => {
    setPreset(p)
    setCats(new Set(PRESETS.find((x) => x.id === p)!.categories))
    if (p !== 'full') setWithKeys(false)
  }
  const toggleCat = (c: BundleCategory, on: boolean) => setCats((s) => {
    const n = new Set(s)
    if (on) n.add(c)
    else n.delete(c)
    return n
  })
  const passOk = !withKeys || (pass.length >= 8 && pass === pass2)

  const make = async () => {
    setErr('')
    setBusy('packing')
    try {
      const m = await exportAgent(agent.slug, {
        preset,
        categories: [...cats],
        ...(reviewing && review ? { insight_categories: [...insightCats], exclude_insights: [...excluded], contacts } : {}),
        ...(full && withKeys ? { passphrase: pass } : {}),
      }, asUser)
      setMade(m)
      setBusy('saving')
      await downloadExport(m, asUser)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  const again = async () => {
    if (!made) return
    setErr('')
    setBusy('saving')
    try { await downloadExport(made, asUser) } catch (e) { setErr(e instanceof Error ? e.message : String(e)) } finally { setBusy('') }
  }

  return (
    <AgentDialog title="Back up / export" icon={Archive} agentName={agent.name} size="md" onClose={onClose}>
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-5">
        {made ? (
          <div className="space-y-3">
            <div className="flex items-center gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3">
              <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-500" />
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-zinc-100">{made.file_name}</p>
                <p className="text-xs text-zinc-500">
                  {formatBytes(made.size)} · saved to your downloads{made.manifest.secrets ? ' · its keys sealed with your passphrase' : ''}
                </p>
              </div>
            </div>
            <Chips items={countChips(made.manifest.counts)} />
            <Notes notes={made.manifest.notes} />
            <Connections items={made.manifest.connections} title="Whoever brings it in sets these up again:" />
          </div>
        ) : (
          <>
            <div className="space-y-2">
              <SectionTitle>What goes in</SectionTitle>
              <div className="grid gap-2 sm:grid-cols-3" role="radiogroup">
                {PRESETS.map((p) => (
                  <button key={p.id} role="radio" aria-checked={preset === p.id} onClick={() => choosePreset(p.id)} disabled={!!busy}
                    className={`rounded-xl border p-3 text-left transition-colors ${preset === p.id
                      ? 'border-violet-500 bg-violet-500/10' : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900/60'}`}>
                    <span className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                      <p.icon className="h-4 w-4 text-violet-500" />{p.label}
                    </span>
                    <span className="mt-1 block text-xs leading-relaxed text-zinc-500">{p.hint}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1">
              <SectionTitle>Includes</SectionTitle>
              <div className="grid sm:grid-cols-2">
                {CATEGORIES.map((c) => (
                  <Check key={c.id} checked={cats.has(c.id)} disabled={c.id === 'identity' || !!busy}
                    onChange={(v) => toggleCat(c.id, v)} label={c.label} hint={c.hint} />
                ))}
              </div>
            </div>

            {reviewing && (
              <div className="space-y-2">
                <SectionTitle>What it learned — look it over</SectionTitle>
                <p className="text-xs leading-relaxed text-zinc-500">
                  Unticked are usually about the person it worked with, or other clients. Open a group to leave single ones out.
                </p>
                {reviewErr && <Problem text={reviewErr} />}
                {!review && !reviewErr && <p className="flex items-center gap-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" />Reading what it learned…</p>}
                {review && review.insights.length === 0 && <p className="text-xs text-zinc-500">It has no insights yet.</p>}
                {review && (
                  <div className="divide-y divide-zinc-800 rounded-xl border border-zinc-800">
                    {review.insights.map((c) => {
                      const on = insightCats.has(c.id)
                      const left = c.items.filter((i) => excluded.has(i.id)).length
                      return (
                        <div key={c.id}>
                          <div className="flex items-center gap-1 pr-2">
                            <div className="min-w-0 flex-1">
                              <Check checked={on} onChange={(v) => setInsightCats((s) => { const n = new Set(s); if (v) n.add(c.id); else n.delete(c.id); return n })}
                                label={c.label} hint={`${c.count} insight${c.count === 1 ? '' : 's'}${on && left ? ` · ${left} left out` : ''}`} />
                            </div>
                            <button onClick={() => setOpen(open === c.id ? null : c.id)} aria-label={`Show ${c.label}`}
                              className="rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
                              {open === c.id ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                            </button>
                          </div>
                          {open === c.id && (
                            <ul className="max-h-56 space-y-0.5 overflow-y-auto border-t border-zinc-800 bg-zinc-900/40 p-2">
                              {c.items.map((i) => (
                                <li key={i.id}>
                                  <Check checked={on && !excluded.has(i.id)} disabled={!on} label={i.content}
                                    onChange={(v) => setExcluded((s) => { const n = new Set(s); if (v) n.delete(i.id); else n.add(i.id); return n })} />
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
                {review && review.contacts > 0 && (
                  <div className="rounded-xl border border-zinc-800">
                    <Check checked={contacts} onChange={setContacts} label={`People it knows (${review.contacts})`}
                      hint="Its contacts — names and addresses" />
                  </div>
                )}
              </div>
            )}

            {full && (
              <div className="space-y-2">
                <SectionTitle>Its keys</SectionTitle>
                <div className="rounded-xl border border-zinc-800">
                  <Check checked={withKeys} onChange={setWithKeys} label="Include its keys, sealed with a passphrase"
                    hint="Its model key and the keys of the services it learned to call. Without the passphrase nobody can open them — Flight Deck doesn't keep it."
                    right={<KeyRound className="mt-0.5 h-4 w-4 text-amber-500" />} />
                  {withKeys && (
                    <div className="grid gap-2 border-t border-zinc-800 p-3 sm:grid-cols-2">
                      <input type="password" value={pass} onChange={(e) => setPass(e.target.value)} placeholder="Passphrase (8+ characters)"
                        autoComplete="new-password" className={field} />
                      <input type="password" value={pass2} onChange={(e) => setPass2(e.target.value)} placeholder="Once more"
                        autoComplete="new-password" className={field} />
                      {pass2 && pass !== pass2 && <p className="text-xs text-red-600 dark:text-red-400 sm:col-span-2">The two don't match.</p>}
                    </div>
                  )}
                </div>
              </div>
            )}

            {!(full && withKeys) && (
              <p className="flex gap-2 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3 text-xs leading-relaxed text-zinc-400">
                <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />{STAYS_OUT} The owner's own profile never does either.
              </p>
            )}
          </>
        )}
        {err && <Problem text={err} />}
      </div>
      <div className="flex shrink-0 justify-end gap-2 border-t border-zinc-800 p-3">
        {!made ? (
          <>
            <button className={quiet} onClick={onClose} disabled={!!busy}>Cancel</button>
            <button className={primary} onClick={make} disabled={!!busy || !passOk || (reviewing && !review && !reviewErr)}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
              {busy === 'packing' ? 'Packing…' : busy === 'saving' ? 'Saving…' : full ? 'Make backup' : 'Make file'}
            </button>
          </>
        ) : (
          <>
            <button className={quiet} onClick={again} disabled={!!busy}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />} Download again
            </button>
            <button className={primary} onClick={onClose}>Done</button>
          </>
        )}
      </div>
    </AgentDialog>
  )
}

// ── From a file ──────────────────────────────────────────────────────────────

function madeOn(ts: number): string {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function AgentImportDialog({ asUser, ownerLabel, onClose, onImported, onConnect }: {
  asUser?: string
  ownerLabel?: string
  onClose: () => void
  onImported: (result: ImportResult) => void
  /** Open the Connections dialog (what the new agent was connected to before). */
  onConnect?: () => void
}) {
  const [upload, setUpload] = useState<UploadResult | null>(null)
  const [name, setName] = useState('')
  const [tiers, setTiers] = useState<ModelTier[]>([])
  const [mode, setMode] = useState<'keep' | 'tier'>('keep')
  const [tier, setTier] = useState('')
  const [pass, setPass] = useState('')
  const [busy, setBusy] = useState<'' | 'reading' | 'making'>('')
  const [err, setErr] = useState('')
  const [done, setDone] = useState<ImportResult | null>(null)
  const [drag, setDrag] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  // An upload nobody imported is thrown away when the dialog closes.
  const pending = useRef<string | null>(null)
  useEffect(() => () => { if (pending.current) void discardUpload(pending.current, asUser) }, [asUser])
  useEffect(() => { myTiers(asUser).then(setTiers).catch(() => setTiers([])) }, [asUser])

  const read = async (file: File | undefined) => {
    if (!file) return
    setErr('')
    setBusy('reading')
    try {
      if (pending.current) { void discardUpload(pending.current, asUser); pending.current = null }
      const u = await uploadBundle(file, asUser)
      pending.current = u.upload_id
      setUpload(u)
      setName(u.suggested_name)
      setPass('')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }
  // Its own tier where this person has it; else its own model.
  useEffect(() => {
    if (!upload) return
    const same = tiers.find((t) => t.tier === upload.model_tier)
    setMode(same ? 'tier' : 'keep')
    setTier(same?.tier ?? tiers[0]?.tier ?? '')
  }, [upload, tiers])
  const ownModel = useMemo(() => upload ? [upload.model.provider, upload.model.model].filter(Boolean).join(' · ') : '', [upload])

  const make = async () => {
    if (!upload) return
    setErr('')
    setBusy('making')
    try {
      const r = await importUpload(upload.upload_id, {
        name: name.trim(),
        model: mode === 'tier' && tier ? { mode: 'tier', tier } : { mode: 'keep' },
        ...(upload.has_secrets && pass ? { passphrase: pass } : {}),
      }, asUser)
      pending.current = null
      setDone(r)
      onImported(r)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  return (
    <AgentDialog title={`New agent from a file${asUser && ownerLabel ? ` for ${ownerLabel}` : ''}`} icon={FileUp}
      agentName="A .captain backup or export" size={upload && !done ? 'md' : 'sm'} onClose={onClose}>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
        {done ? (
          <div className="space-y-3">
            <div className="flex items-center gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3">
              <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-500" />
              <p className="text-sm text-zinc-100"><span className="font-medium">{done.name}</span> is starting — its chat opens as soon as it's up.</p>
            </div>
            <Notes notes={done.notes} />
            <Connections items={done.connections} title="Where it came from it was connected to — set these up here if it needs them:"
              onConnect={onConnect ? () => { onClose(); onConnect() } : undefined} />
          </div>
        ) : !upload ? (
          <>
            <button
              type="button"
              onClick={() => input.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); void read(e.dataTransfer.files?.[0]) }}
              disabled={!!busy}
              className={`flex w-full flex-col items-center gap-2 rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${
                drag ? 'border-violet-500 bg-violet-500/10' : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900/60'}`}
            >
              {busy === 'reading' ? <Loader2 className="h-7 w-7 animate-spin text-violet-400" /> : <FileUp className="h-7 w-7 text-violet-400" />}
              <span className="text-sm font-medium text-zinc-100">{busy === 'reading' ? 'Reading the file…' : 'Choose a .captain file'}</span>
              <span className="text-xs text-zinc-500">or drop it here</span>
            </button>
            <input ref={input} type="file" accept=".captain,application/zip" className="hidden"
              onChange={(e) => { void read(e.target.files?.[0]); e.target.value = '' }} />
            <p className="text-xs leading-relaxed text-zinc-500">
              You'll see what is in it before anything is made. The new agent is yours; {STAYS_OUT.charAt(0).toLowerCase() + STAYS_OUT.slice(1)}
            </p>
          </>
        ) : (
          <div className="space-y-4">
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
              <p className="text-base font-semibold text-zinc-100">{upload.agent.name}</p>
              {upload.agent.description && <p className="mt-0.5 text-sm text-zinc-400">{upload.agent.description}</p>}
              <p className="mt-2 text-xs text-zinc-500">
                {[PRESETS.find((p) => p.id === upload.preset)?.label, madeOn(upload.made), upload.made_by && `by ${upload.made_by}`,
                  formatBytes(upload.size)].filter(Boolean).join(' · ')}
              </p>
            </div>
            <Chips items={countChips(upload.counts)} />
            <Notes notes={upload.notes} />
            <label className="block">
              <span className="text-xs font-medium text-zinc-400">Name</span>
              <input value={name} onChange={(e) => setName(e.target.value)} maxLength={80} className={`mt-1 ${field}`} />
            </label>
            <div className="space-y-1.5">
              <span className="text-xs font-medium text-zinc-400">Model</span>
              <label className={`flex cursor-pointer items-start gap-3 rounded-xl border p-3 ${mode === 'keep' ? 'border-violet-500 bg-violet-500/10' : 'border-zinc-800'}`}>
                <input type="radio" className="mt-0.5 accent-violet-600" checked={mode === 'keep'} onChange={() => setMode('keep')} />
                <span className="min-w-0">
                  <span className="block text-sm text-zinc-200">Its own model</span>
                  <span className="block truncate text-xs text-zinc-500">{ownModel || 'As it was made'} — needs a key for it here</span>
                </span>
              </label>
              <label className={`flex items-start gap-3 rounded-xl border p-3 ${tiers.length ? 'cursor-pointer' : 'opacity-60'} ${mode === 'tier' ? 'border-violet-500 bg-violet-500/10' : 'border-zinc-800'}`}>
                <input type="radio" className="mt-0.5 accent-violet-600" checked={mode === 'tier'} disabled={!tiers.length} onChange={() => setMode('tier')} />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm text-zinc-200">One of your models</span>
                  {tiers.length ? (
                    <select value={tier} onChange={(e) => { setTier(e.target.value); setMode('tier') }} className={`mt-1.5 ${field}`}>
                      {tiers.map((t) => <option key={t.tier} value={t.tier}>{t.tier} — {t.model}</option>)}
                    </select>
                  ) : <span className="block text-xs text-zinc-500">You have no models set up yet.</span>}
                </span>
              </label>
            </div>
            {upload.has_secrets && (
              <label className="block">
                <span className="flex items-center gap-1.5 text-xs font-medium text-zinc-400"><KeyRound className="h-3.5 w-3.5 text-amber-500" />Passphrase for its keys</span>
                <input type="password" value={pass} onChange={(e) => setPass(e.target.value)} autoComplete="off"
                  placeholder="Leave empty to bring it in without them" className={`mt-1 ${field}`} />
              </label>
            )}
            <Connections items={upload.connections} title="Where it came from it was connected to (none of it is in the file):" />
          </div>
        )}
        {err && <Problem text={err} />}
      </div>
      <div className="flex shrink-0 justify-end gap-2 border-t border-zinc-800 p-3">
        {done ? (
          <button className={primary} onClick={onClose}>Done</button>
        ) : (
          <>
            {upload && <button className={quiet} onClick={() => { void discardUpload(upload.upload_id, asUser); pending.current = null; setUpload(null) }} disabled={!!busy}>Another file</button>}
            <button className={quiet} onClick={onClose} disabled={busy === 'making'}>Cancel</button>
            {upload && (
              <button className={primary} onClick={make} disabled={!!busy || !name.trim() || (mode === 'tier' && !tier)}>
                {busy === 'making' ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
                {busy === 'making' ? 'Making it…' : 'Create agent'}
              </button>
            )}
          </>
        )}
      </div>
    </AgentDialog>
  )
}
