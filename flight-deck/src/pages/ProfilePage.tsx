import { useCallback, useEffect, useRef, useState } from 'react'
import { IdCard, Loader2, RefreshCw } from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import { useUIStore } from '../stores/uiStore'
import {
  getProfile,
  saveProfile,
  agentsUpdatedLabel,
  charCount,
  deckDefaultsEditable,
  guardUnload,
  rebaseDraft,
  DEFAULT_PREVIEW_MODE,
  ECO_SHORTENED_HINT,
  EMPTY_PROFILE,
  PREVIEW_AUDIENCE,
  type PreviewMode,
  type ProfileFields,
  type ProfileResponse,
} from '../services/profile'
import { CappedTextarea } from '../components/profile/CappedTextarea'
import { DeckProfileDefaults } from '../components/profile/DeckProfileDefaults'

const FIELDS: (keyof ProfileFields)[] = ['about_me', 'company', 'instructions']
const PREVIEW_MODES: PreviewMode[] = ['compact', 'full']

// The owner profile: who the user is, their company, and standing preferences.
// Flight Deck merges it with the deck-wide defaults and hands the result to
// every agent working for the user, next to the agent's own instructions.
//
// `kiosk` — shown in the locked Simple layout's dialog, where the Admin page
// (and so the "edit deck defaults" link) is out of reach, and the deck-wide
// defaults are read-only even on an auth-off deck. `onDirtyChange` lets that
// dialog ask before throwing away unsaved text.
export function ProfilePage({ kiosk = false, onDirtyChange }: {
  kiosk?: boolean
  onDirtyChange?: (dirty: boolean) => void
}) {
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const isAdmin = useAuthStore((s) => s.user?.role === 'admin')
  const setView = useUIStore((s) => s.setView)
  const [data, setData] = useState<ProfileResponse | null>(null)
  const [draft, setDraft] = useState<ProfileFields>(EMPTY_PROFILE)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [previewMode, setPreviewMode] = useState<PreviewMode>(DEFAULT_PREVIEW_MODE)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // The saved profile the draft is based on — what a reload rebases against.
  const baseRef = useRef<ProfileFields | null>(null)

  // `rebase` — refresh without dropping unsaved text: fields the user changed
  // keep their text, untouched ones take the server's (possibly newer) value.
  const load = useCallback(async (rebase = false) => {
    setLoading(true)
    try {
      const r = await getProfile()
      const base = baseRef.current
      baseRef.current = r.profile
      setData(r)
      setDraft((d) => (rebase && base ? rebaseDraft(d, base, r.profile) : r.profile))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const dirty = !!data && FIELDS.some((k) => draft[k] !== data.profile[k])
  useEffect(() => { onDirtyChange?.(dirty) }, [dirty, onDirtyChange])
  // Closing or reloading the browser tab asks before dropping unsaved text.
  useEffect(() => guardUnload(window, dirty), [dirty])

  const caps = data?.caps
  const tooLong = !!caps && FIELDS.some((k) => charCount(draft[k]) > caps[k])

  const set = (k: keyof ProfileFields) => (v: string) => {
    setDraft((d) => ({ ...d, [k]: v }))
    setNotice(null)
  }

  const save = async () => {
    if (!data) return
    setSaving(true); setError(null); setNotice(null)
    // Partial merge: send only what changed.
    const patch: Partial<ProfileFields> = {}
    for (const k of FIELDS) if (draft[k] !== data.profile[k]) patch[k] = draft[k]
    try {
      const r = await saveProfile(patch)
      baseRef.current = r.profile
      setData(r)
      setDraft(r.profile)
      setNotice(`Saved — ${agentsUpdatedLabel(r.agents_updated)}.`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const deck = data?.deck
  const deckEmpty = !deck || (!deck.company.trim() && !deck.instructions.trim())
  const usingDeckCompany = !!deck?.company.trim() && !draft.company.trim()
  const preview = data ? data.preview[previewMode] : ''
  const card = 'rounded-lg border border-zinc-800 bg-zinc-900/60 p-5 space-y-4'

  // Admin is another page: leaving drops the draft, so ask first.
  const openAdmin = () => {
    if (dirty && !window.confirm('Discard your unsaved profile changes?')) return
    setView('admin')
  }

  return (
    <div className="h-full overflow-y-auto bg-zinc-950 text-zinc-200">
      <div className="max-w-3xl mx-auto px-6 py-8 pb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="h-10 w-10 shrink-0 rounded-md bg-zinc-900 border border-zinc-800 flex items-center justify-center">
            <IdCard className="h-5 w-5 text-violet-500" />
          </div>
          <div className="min-w-0 flex-1">
            <h1 className="text-xl font-semibold text-zinc-100">Profile</h1>
            <p className="text-sm text-zinc-500">
              Tell your agents who you are, what your company does and how you like things done. Every agent
              working for you receives this alongside its own instructions.
            </p>
          </div>
          {!kiosk && (
            <button
              onClick={() => load(true)}
              disabled={loading}
              title="Reload"
              className="shrink-0 rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
          )}
        </div>

        {!data && loading && (
          <div className="flex justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
          </div>
        )}
        {!data && !loading && error && (
          <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
            {error}
          </div>
        )}

        {data && caps && (
          <div className="space-y-4">
            {/* Your profile */}
            <div className={card}>
              <p className="text-[11px] text-zinc-500">{ECO_SHORTENED_HINT}</p>
              <CappedTextarea
                label="About me"
                value={draft.about_me}
                onChange={set('about_me')}
                cap={caps.about_me}
                rows={4}
                placeholder="Your role, what you work on, your expertise, how you like to work…"
                hint="Who you are, so your agents can pitch their work at the right level."
              />
              <CappedTextarea
                label="My company"
                value={draft.company}
                onChange={set('company')}
                cap={caps.company}
                rows={6}
                placeholder="What your company does, its products and customers, the voice it writes in…"
                hint={usingDeckCompany
                  ? "Empty — your agents get the deck's company description below. Anything you write here replaces it."
                  : "Replaces the deck's company description for your agents."}
              />
              <CappedTextarea
                label="My standing preferences"
                value={draft.instructions}
                onChange={set('instructions')}
                cap={caps.instructions}
                rows={5}
                placeholder="e.g. Answer in Croatian. Prefer short bullet points. Always cite your sources."
                hint="Your agents apply these unless a task, role or output format says otherwise."
              />

              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={save}
                  disabled={saving || !dirty || tooLong}
                  className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
                >
                  {saving && <Loader2 className="h-3 w-3 animate-spin" />}
                  {saving ? 'Saving…' : 'Save'}
                </button>
                {dirty && (
                  <button
                    onClick={() => { setDraft(data.profile); setError(null) }}
                    disabled={saving}
                    className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
                  >
                    Revert
                  </button>
                )}
                <span className="text-[11px] text-zinc-600">Agents pick it up on their next turn — no restart.</span>
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

            {/* Deck-wide defaults: read-only here; an auth-off deck's only user
                is its admin and has no Admin page, so they edit them here —
                but never from the kiosk: a visitor there must not change what
                every agent on the deck receives. */}
            {deckDefaultsEditable(authEnabled, kiosk) ? (
              <DeckProfileDefaults onSaved={() => load(true)} />
            ) : (
              <div className={card}>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-zinc-200">Deck-wide defaults</h3>
                    <p className="text-xs text-zinc-500 mt-0.5">
                      Applies to everyone; your company text replaces the deck's. The deck's instructions
                      apply alongside your preferences and win if they conflict.
                    </p>
                  </div>
                  {isAdmin && !kiosk && (
                    <button
                      onClick={openAdmin}
                      className="shrink-0 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                      title="Admin → Settings → Deck-wide profile defaults"
                    >
                      Edit in Admin → Settings
                    </button>
                  )}
                </div>
                {deckEmpty ? (
                  <p className="text-xs text-zinc-600">No deck-wide defaults are set.</p>
                ) : (
                  <div className="space-y-3">
                    {deck?.company.trim() && (
                      <ReadOnlyBlock label="Company" text={deck.company} muted={!usingDeckCompany} />
                    )}
                    {deck?.instructions.trim() && (
                      <ReadOnlyBlock label="Instructions for everyone" text={deck.instructions} />
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Preview: what the server composes from the SAVED profile */}
            <div className={card}>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-zinc-200">What your agents receive</h3>
                  <p className="text-xs text-zinc-500 mt-0.5">
                    Added to the system prompt of {PREVIEW_AUDIENCE[previewMode]}.
                  </p>
                </div>
                <div className="flex shrink-0 rounded-md border border-zinc-700 p-0.5 text-[11px]">
                  {PREVIEW_MODES.map((m) => (
                    <button
                      key={m}
                      onClick={() => setPreviewMode(m)}
                      className={`rounded px-2 py-0.5 capitalize transition-colors ${
                        previewMode === m ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>
              {dirty && (
                <p className="text-[11px] text-amber-700 dark:text-amber-400">
                  Showing the saved profile — save to see your changes here.
                </p>
              )}
              {preview.trim() ? (
                <pre className="max-h-96 overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-zinc-800 bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-300">
                  {preview}
                </pre>
              ) : (
                <p className="rounded-md border border-dashed border-zinc-800 px-3 py-4 text-center text-xs text-zinc-600">
                  Nothing yet — your agents get no owner background until you or your admin fill something in.
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function ReadOnlyBlock({ label, text, muted = false }: { label: string; text: string; muted?: boolean }) {
  return (
    <div>
      <div className="mb-1 flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        {label}
        {muted && <span className="normal-case tracking-normal text-zinc-600">(replaced by yours)</span>}
      </div>
      <div
        className={`max-h-48 overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-2 text-xs leading-relaxed ${
          muted ? 'text-zinc-600' : 'text-zinc-300'
        }`}
      >
        {text}
      </div>
    </div>
  )
}
