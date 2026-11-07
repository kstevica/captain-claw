/** Agent bundles in the kiosk: an agent's ⋯ → Back up / export, and New agent
 *  → Import from a file. A bundle is one agent in a .captain file. */

import { useEffect, useRef, useState } from 'react'
import { Archive, CheckCircle2, Download, FileUp, Loader2, ShieldCheck, TriangleAlert } from 'lucide-react'
import { AgentDialog } from './KioskAgentDialogs'
import {
  countChips, discardUpload, downloadExport, exportAgent, formatBytes, importUpload, uploadBundle,
  type ExportResult, type ImportResult, type UploadResult,
} from '../../services/agentBundle'

const btn = 'inline-flex h-10 items-center justify-center gap-2 rounded-xl px-4 text-sm font-medium transition-colors disabled:opacity-50'
const primary = `${btn} bg-violet-600 text-white hover:bg-violet-500`
const quiet = `${btn} border border-zinc-800 text-zinc-300 hover:bg-zinc-900`

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

const STAYS_OUT = 'Keys, tokens and its connections (mail, Telegram, MCP servers) never go in the file.'

// ── Back up / export ─────────────────────────────────────────────────────────

export function AgentBackupDialog({ agent, asUser, onClose }: {
  agent: { slug: string; name: string }
  asUser?: string
  onClose: () => void
}) {
  const [busy, setBusy] = useState<'' | 'packing' | 'saving'>('')
  const [made, setMade] = useState<ExportResult | null>(null)
  const [err, setErr] = useState('')

  const make = async () => {
    setErr('')
    setBusy('packing')
    try {
      const m = await exportAgent(agent.slug, 'full', asUser)
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
    <AgentDialog title="Back up / export" icon={Archive} agentName={agent.name} onClose={onClose}>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
        {!made ? (
          <>
            <p className="text-sm leading-relaxed text-zinc-300">
              A full backup puts <span className="font-medium text-zinc-100">{agent.name}</span> in one <code className="rounded bg-zinc-900 px-1 text-xs">.captain</code> file:
              its instructions, onboarding, what it learned, its data tables, conversations, files and scheduled tasks.
            </p>
            <p className="text-sm leading-relaxed text-zinc-300">
              To bring it back, use <span className="font-medium text-zinc-100">New agent → From a file</span>, here or on another Flight Deck.
            </p>
            <p className="flex gap-2 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3 text-xs leading-relaxed text-zinc-400">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />{STAYS_OUT}
            </p>
          </>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3">
              <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-500" />
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-zinc-100">{made.file_name}</p>
                <p className="text-xs text-zinc-500">{formatBytes(made.size)} · saved to your downloads</p>
              </div>
            </div>
            <Chips items={countChips(made.manifest.counts)} />
            <Notes notes={made.manifest.notes} />
          </div>
        )}
        {err && <Problem text={err} />}
      </div>
      <div className="flex shrink-0 justify-end gap-2 border-t border-zinc-800 p-3">
        {!made ? (
          <>
            <button className={quiet} onClick={onClose} disabled={!!busy}>Cancel</button>
            <button className={primary} onClick={make} disabled={!!busy}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
              {busy === 'packing' ? 'Packing…' : busy === 'saving' ? 'Saving…' : 'Make backup'}
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

// ── Import from a file ───────────────────────────────────────────────────────

function made(ts: number): string {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function AgentImportDialog({ asUser, ownerLabel, onClose, onImported }: {
  asUser?: string
  ownerLabel?: string
  onClose: () => void
  onImported: (result: ImportResult) => void
}) {
  const [upload, setUpload] = useState<UploadResult | null>(null)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState<'' | 'reading' | 'making'>('')
  const [err, setErr] = useState('')
  const [done, setDone] = useState<ImportResult | null>(null)
  const [drag, setDrag] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  // An upload nobody imported is thrown away when the dialog closes.
  const pending = useRef<string | null>(null)
  useEffect(() => () => { if (pending.current) void discardUpload(pending.current, asUser) }, [asUser])

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
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  const make = async () => {
    if (!upload) return
    setErr('')
    setBusy('making')
    try {
      const r = await importUpload(upload.upload_id, name.trim(), asUser)
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
      agentName="A .captain backup or export" onClose={onClose}>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
        {done ? (
          <div className="space-y-3">
            <div className="flex items-center gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3">
              <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-500" />
              <p className="text-sm text-zinc-100"><span className="font-medium">{done.name}</span> is starting — its chat opens as soon as it's up.</p>
            </div>
            <Notes notes={done.notes} />
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
                {[made(upload.made), upload.made_by && `by ${upload.made_by}`, formatBytes(upload.size),
                  upload.model.model && `model ${upload.model.model}`].filter(Boolean).join(' · ')}
              </p>
            </div>
            <Chips items={countChips(upload.counts)} />
            <Notes notes={upload.notes} />
            <label className="block">
              <span className="text-xs font-medium text-zinc-400">Name</span>
              <input value={name} onChange={(e) => setName(e.target.value)} maxLength={80}
                className="mt-1 h-10 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 text-sm text-zinc-100 outline-none focus:border-violet-500" />
            </label>
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
              <button className={primary} onClick={make} disabled={!!busy || !name.trim()}>
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
