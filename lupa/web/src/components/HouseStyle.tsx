import { useState } from 'react'
import { Loader2, Sparkles, Check } from 'lucide-react'
import { post, postForm } from '../api'

interface Draft {
  id: string
  role?: string
  instructions?: string
  tier?: string
  [k: string]: unknown
}

/** "Your methodology, encoded": instructions + documents → a reusable house
 * cast. Forge drafts (FD, unpersisted) → review → save the keepers. */
export default function HouseStyle() {
  const [open, setOpen] = useState(false)
  const [instructions, setInstructions] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [drafts, setDrafts] = useState<Draft[] | null>(null)
  const [saved, setSaved] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const forge = async () => {
    setBusy(true); setError(''); setDrafts(null); setSaved(new Set())
    try {
      const fd = new FormData()
      fd.append('instructions', instructions.trim())
      for (const f of files) fd.append('files', f)
      const out = await postForm<{ drafts: Draft[] }>('/api/forge', fd)
      setDrafts(out.drafts ?? [])
    } catch (e) { setError(e instanceof Error ? e.message : 'forge failed') }
    finally { setBusy(false) }
  }

  const save = async (d: Draft) => {
    await post('/api/archetypes', d)
    setSaved((s) => new Set(s).add(d.id))
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="w-full rounded-xl border border-dashed border-[var(--lp-border)] bg-[var(--lp-surface)] px-4 py-3 text-sm text-[var(--lp-text-dim)] hover:border-[var(--lp-accent)] hover:text-[var(--lp-text)] flex items-center justify-center gap-2"
      >
        <Sparkles size={14} /> Forge your house cast from documents
      </button>
    )
  }

  return (
    <div className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Sparkles size={15} style={{ color: 'var(--lp-accent)' }} />
        <span className="font-semibold text-sm">House style</span>
        <span className="text-xs text-[var(--lp-text-dim)]">
          Turn your playbooks into a reusable research team.
        </span>
        <div className="flex-1" />
        <button onClick={() => setOpen(false)}
                className="text-xs text-[var(--lp-text-dim)] hover:text-[var(--lp-text)]">
          close
        </button>
      </div>
      <textarea
        value={instructions}
        onChange={(e) => setInstructions(e.target.value)}
        placeholder="Describe your methodology, output style, and the roles you want (e.g. a rigorous fact-checker, a terse memo writer)…"
        rows={3}
        className="w-full rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2 text-sm outline-none focus:border-[var(--lp-accent)] resize-y"
      />
      <div className="flex items-center gap-2 flex-wrap">
        <label className="text-xs px-2.5 py-1.5 rounded-lg border border-[var(--lp-border)] cursor-pointer hover:border-[var(--lp-accent)]">
          Add documents
          <input type="file" multiple className="hidden"
                 onChange={(e) => setFiles(Array.from(e.target.files ?? []))} />
        </label>
        {files.length > 0 && (
          <span className="text-xs text-[var(--lp-text-dim)]">
            {files.map((f) => f.name).join(', ')}
          </span>
        )}
        <div className="flex-1" />
        <button
          onClick={() => void forge()}
          disabled={busy || (!instructions.trim() && files.length === 0)}
          className="rounded-lg px-4 py-1.5 text-sm font-semibold text-black disabled:opacity-40 flex items-center gap-1.5"
          style={{ background: 'var(--lp-accent)' }}
        >
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />} Forge
        </button>
      </div>
      {error && <div className="text-sm text-red-400">{error}</div>}
      {drafts && (
        <div className="space-y-2 pt-1">
          <div className="text-xs text-[var(--lp-text-dim)]">
            {drafts.length} drafts — save the ones you want, then pin them per stream in settings.
          </div>
          {drafts.map((d) => {
            const isSaved = saved.has(d.id)
            return (
              <div key={d.id}
                   className="rounded-lg border border-[var(--lp-border)] bg-[var(--lp-bg)] px-3 py-2 flex items-start gap-3">
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium">{d.role ?? d.id}</div>
                  <div className="text-xs text-[var(--lp-text-dim)] line-clamp-2">{d.instructions}</div>
                </div>
                <button
                  onClick={() => void save(d)} disabled={isSaved}
                  className={`text-xs px-2.5 py-1 rounded border whitespace-nowrap flex items-center gap-1 ${
                    isSaved ? 'border-emerald-700/60 text-emerald-400'
                            : 'border-[var(--lp-border)] hover:border-[var(--lp-accent)] hover:text-[var(--lp-accent)]'}`}
                >
                  {isSaved ? <><Check size={12} /> Saved</> : 'Save'}
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
