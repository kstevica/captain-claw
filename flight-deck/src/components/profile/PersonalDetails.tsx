import { useId, useRef, useState, type DragEvent } from 'react'
import { Building2, FileText, Link2, Loader2, Plus, Trash2, Upload, UserRound, X } from 'lucide-react'
import {
  deleteProfileDocument,
  uploadProfileDocuments,
  type ProfileLink,
  type ProfileSources,
} from '../../services/profile'
import { dateLabel, docKindsLabel, keptLinks, labelForUrl, type DetailsDraft } from '../../utils/profileSources'

// Profile → Personal: who the person is, as material — their name, their
// links (LinkedIn, GitHub, a personal site), their company's name and links,
// and their documents (a CV, a bio, a one-pager). The researcher below it
// drafts the three texts from this; the texts themselves are edited further
// down the page. The name and the links are a draft the page saves; a
// document is kept the moment it is uploaded.

const INPUT = 'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none disabled:opacity-50'
const LABEL = 'mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500'

export function PersonalDetails({
  sources, draft, onDraft, dirty, saving, notice, error, onSave, onRevert, onSources,
}: {
  sources: ProfileSources
  draft: DetailsDraft
  onDraft: (d: DetailsDraft) => void
  dirty: boolean
  saving: boolean
  notice: string | null
  error: string | null
  onSave: () => void
  onRevert: () => void
  /** The server's material after an upload or a removal. */
  onSources: (s: ProfileSources) => void
}) {
  const nameId = useId()
  const companyId = useId()
  const fileInput = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [warnings, setWarnings] = useState<string[]>([])
  const [docError, setDocError] = useState<string | null>(null)
  const [removing, setRemoving] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  const caps = sources.caps
  const set = <K extends keyof DetailsDraft>(k: K, v: DetailsDraft[K]) => onDraft({ ...draft, [k]: v })

  const addFiles = async (files: FileList | File[] | null) => {
    const list = Array.from(files ?? []).slice(0, caps.documents)
    if (!list.length || uploading) return
    setUploading(true); setDocError(null); setWarnings([])
    try {
      const r = await uploadProfileDocuments(list)
      onSources(r)
      setWarnings(r.warnings ?? [])
    } catch (e) {
      setDocError(e instanceof Error ? e.message : String(e))
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const remove = async (id: string) => {
    if (removing) return
    setRemoving(id); setDocError(null)
    try {
      onSources(await deleteProfileDocument(id))
    } catch (e) {
      setDocError(e instanceof Error ? e.message : String(e))
    } finally {
      setRemoving(null)
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    void addFiles(e.dataTransfer?.files ?? null)
  }

  const room = caps.documents - sources.documents.length
  const kinds = docKindsLabel(sources.kinds)

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-5 space-y-5">
      <div className="flex items-start gap-2">
        <UserRound className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
        <div>
          <h3 className="text-sm font-semibold text-zinc-200">Who you are</h3>
          <p className="mt-0.5 text-xs text-zinc-500">
            Your name, where to find you and your company, and the documents that describe you. Flight Deck can
            draft the texts below from these.
          </p>
        </div>
      </div>

      {/* The account's name — how agents and teammates see the person. */}
      {sources.can_rename && (
        <div>
          <label htmlFor={nameId} className={LABEL}>Your name</label>
          <input
            id={nameId}
            value={draft.name}
            onChange={(e) => set('name', e.target.value)}
            maxLength={120}
            autoComplete="name"
            className={`${INPUT} sm:max-w-sm`}
          />
          <p className="mt-1 text-[11px] text-zinc-500">
            How your agents address you{sources.email ? ` — the account is ${sources.email}` : ''}.
          </p>
        </div>
      )}

      {/* Where to find the person. */}
      <div>
        <div className={LABEL}>Your links</div>
        <LinkList
          links={draft.links}
          onChange={(links) => set('links', links)}
          cap={caps.links}
          labelCap={caps.label}
          urlCap={caps.url}
          placeholder="linkedin.com/in/you, github.com/you, your site…"
          addLabel="Add a link"
        />
        <p className="mt-1 text-[11px] text-zinc-500">
          LinkedIn, GitHub, X, a personal site, a speaker page — anything public about you.
        </p>
      </div>

      {/* The company. */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Building2 className="h-4 w-4 shrink-0 text-violet-500" />
          <h4 className="text-sm font-semibold text-zinc-200">Your company</h4>
        </div>
        <div>
          <label htmlFor={companyId} className={LABEL}>Company name</label>
          <input
            id={companyId}
            value={draft.company_name}
            onChange={(e) => set('company_name', e.target.value)}
            maxLength={caps.company_name}
            placeholder="The name it goes by"
            autoComplete="organization"
            className={`${INPUT} sm:max-w-sm`}
          />
        </div>
        <div>
          <div className={LABEL}>Company links</div>
          <LinkList
            links={draft.company_links}
            onChange={(links) => set('company_links', links)}
            cap={caps.company_links}
            labelCap={caps.label}
            urlCap={caps.url}
            placeholder="company.com, its LinkedIn page, a product page…"
            addLabel="Add a company link"
          />
          <p className="mt-1 text-[11px] text-zinc-500">The website first — its About page is read too.</p>
        </div>
      </div>

      {/* Documents: kept the moment they are uploaded. */}
      <div
        onDragOver={(e) => { e.preventDefault(); if (room > 0) setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={`rounded-lg border border-dashed p-3 transition-colors ${dragOver ? 'border-violet-500/60 bg-violet-500/5' : 'border-zinc-800'}`}
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 shrink-0 text-violet-500" />
            <h4 className="text-sm font-semibold text-zinc-200">Documents</h4>
            <span className="text-[11px] text-zinc-600">{sources.documents.length} / {caps.documents}</span>
          </div>
          <input
            ref={fileInput}
            type="file"
            multiple
            accept={sources.kinds.join(',')}
            className="hidden"
            onChange={(e) => void addFiles(e.target.files)}
          />
          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            disabled={uploading || room <= 0}
            className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
          >
            {uploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
            {uploading ? 'Reading…' : 'Upload'}
          </button>
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">
          A CV, a bio, a company one-pager, a pitch deck — {kinds}, up to 25 MB each. Drop files here, too.
        </p>
        {sources.documents.length > 0 && (
          <ul className="mt-3 space-y-1.5">
            {sources.documents.map((d) => (
              <li key={d.id} className="flex items-center gap-2 rounded-md border border-zinc-800 bg-zinc-950/60 px-2.5 py-1.5">
                <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                <span className="min-w-0 flex-1 truncate text-xs text-zinc-200" title={d.name}>{d.name}</span>
                <span className="shrink-0 text-[11px] text-zinc-600">
                  {d.pages} page{d.pages === 1 ? '' : 's'}{d.added ? ` · ${dateLabel(d.added)}` : ''}
                </span>
                <button
                  type="button"
                  onClick={() => void remove(d.id)}
                  disabled={!!removing}
                  title="Remove"
                  aria-label={`Remove ${d.name}`}
                  className="shrink-0 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-red-500 disabled:opacity-50 dark:hover:text-red-400"
                >
                  {removing === d.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                </button>
              </li>
            ))}
          </ul>
        )}
        {warnings.length > 0 && (
          <ul className="mt-2 space-y-0.5 text-[11px] text-amber-700 dark:text-amber-300">
            {warnings.map((w) => <li key={w}>{w}</li>)}
          </ul>
        )}
        {docError && (
          <div className="mt-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
            {docError}
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onSave}
          disabled={saving || !dirty}
          className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
        >
          {saving && <Loader2 className="h-3 w-3 animate-spin" />}
          {saving ? 'Saving…' : 'Save details'}
        </button>
        {dirty && (
          <button
            type="button"
            onClick={onRevert}
            disabled={saving}
            className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
          >
            Revert
          </button>
        )}
        <span className="text-[11px] text-zinc-600">Your agents don't see these — only the texts below.</span>
      </div>
      {notice && (
        <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-300">
          {notice}
        </div>
      )}
      {error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
          {error}
        </div>
      )}
    </div>
  )
}

// A short list of links: a label and an address a row; the label fills itself
// in from the address (LinkedIn, GitHub, the host) when left empty.
function LinkList({ links, onChange, cap, labelCap, urlCap, placeholder, addLabel }: {
  links: ProfileLink[]
  onChange: (links: ProfileLink[]) => void
  cap: number
  labelCap: number
  urlCap: number
  placeholder: string
  addLabel: string
}) {
  const update = (i: number, patch: Partial<ProfileLink>) =>
    onChange(links.map((l, n) => (n === i ? { ...l, ...patch } : l)))
  const removeAt = (i: number) => onChange(links.filter((_, n) => n !== i))
  const add = () => onChange([...links, { url: '', label: '' }])
  const full = keptLinks(links).length >= cap || links.length >= cap

  return (
    <div className="space-y-2">
      {links.map((l, i) => (
        <div key={i} className="flex items-center gap-2">
          <Link2 className="hidden h-3.5 w-3.5 shrink-0 text-zinc-600 sm:block" aria-hidden />
          <input
            value={l.label}
            onChange={(e) => update(i, { label: e.target.value })}
            maxLength={labelCap}
            placeholder="Label"
            aria-label="Link label"
            className={`${INPUT} w-24 shrink-0 sm:w-32`}
          />
          <input
            value={l.url}
            onChange={(e) => update(i, { url: e.target.value })}
            onBlur={() => { if (!l.label.trim() && l.url.trim()) update(i, { label: labelForUrl(l.url) }) }}
            maxLength={urlCap}
            placeholder={placeholder}
            aria-label="Link address"
            inputMode="url"
            autoComplete="url"
            spellCheck={false}
            className={`${INPUT} min-w-0 flex-1`}
          />
          <button
            type="button"
            onClick={() => removeAt(i)}
            title="Remove"
            aria-label="Remove link"
            className="shrink-0 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ))}
      {!full && (
        <button
          type="button"
          onClick={add}
          className="flex items-center gap-1 rounded-md border border-dashed border-zinc-700 px-2.5 py-1 text-xs text-zinc-400 hover:border-zinc-600 hover:bg-zinc-800/60 hover:text-zinc-200"
        >
          <Plus className="h-3.5 w-3.5" />
          {addLabel}
        </button>
      )}
    </div>
  )
}
