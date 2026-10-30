import { useCallback, useEffect, useRef, useState } from 'react'
import { Building2, IdCard, KeyRound, Loader2, MessageSquareText, RefreshCw, UserRound } from 'lucide-react'
import { useAuthStore, updateProfile } from '../stores/authStore'
import {
  getProfile,
  saveProfile,
  getProfileSources,
  saveProfileSources,
  getProfileResearch,
  startProfileResearch,
  stopProfileResearch,
  agentsUpdatedLabel,
  charCount,
  deckDefaultsEditable,
  guardUnload,
  rebaseDraft,
  DEFAULT_PREVIEW_MODE,
  ECO_SHORTENED_HINT,
  EMPTY_PROFILE,
  IDLE_RESEARCH,
  PREVIEW_AUDIENCE,
  RESEARCH_POLL_MS,
  type PreviewMode,
  type ProfileFields,
  type ProfileResearch as Research,
  type ProfileResponse,
  type ProfileSources,
} from '../services/profile'
import { CappedTextarea } from '../components/profile/CappedTextarea'
import { DeckProfileDefaults } from '../components/profile/DeckProfileDefaults'
import { MyContextPacks } from '../components/profile/MyContextPacks'
import { ChangePasswordCard } from '../components/profile/ChangePasswordCard'
import { PersonalDetails } from '../components/profile/PersonalDetails'
import { ProfileResearch } from '../components/profile/ProfileResearch'
import { ProfileTabs, type ProfileTab, type ProfileTabSpec } from '../components/profile/ProfileTabs'
import { useSharedAgentStore } from '../stores/sharedAgentStore'
import type { MyPackRow } from '../services/contextPacks'
import { MY_PACKS_HEADING, MY_PACKS_ID, profilePackAgents, profileSharedNote } from '../utils/contextPacks'
import {
  detailsDirty as isDetailsDirty,
  detailsOf,
  detailsPatch,
  nameChanged,
  sourcesEmpty,
  EMPTY_DETAILS,
  type DetailsDraft,
} from '../utils/profileSources'

const FIELDS: (keyof ProfileFields)[] = ['about_me', 'company', 'instructions']
const PREVIEW_MODES: PreviewMode[] = ['compact', 'full']
const TEXTS_ID = 'profile-texts'

// The owner profile, in three tabs:
//
// * Personal — who the user is as material (name, links, company, documents),
//   the researcher that drafts the three texts from it, the texts themselves
//   (what every agent working for the user receives, next to its own
//   instructions), the preview of what the agents get, and what the user
//   shares with agents' people;
// * Deck-wide — the admin's defaults every user's agents receive (editable by
//   an admin here, read-only for everyone else and in the kiosk);
// * Password — on a deck with accounts.
//
// `kiosk` — shown in the locked Simple layout's dialog; the deck-wide defaults
// are read-only there even on an auth-off deck. `onDirtyChange` lets that
// dialog ask before throwing away unsaved text.
export function ProfilePage({ kiosk = false, onDirtyChange, initialTab = 'personal' }: {
  kiosk?: boolean
  onDirtyChange?: (dirty: boolean) => void
  initialTab?: ProfileTab
}) {
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const isAdmin = useAuthStore((s) => s.user?.role === 'admin')
  // Context packs: what I share with agents' people (only on a deck that offers it).
  const contextPacks = useSharedAgentStore((s) => s.contextPacks)
  // Known only once the shared-agent list has loaded: before that a deck with
  // sharing on would look like one with it off.
  const sharedLoaded = useSharedAgentStore((s) => s.loaded)
  // My packs, as the "Shared with agents’ people" card last loaded them.
  const [myPacks, setMyPacks] = useState<MyPackRow[]>([])
  const [tab, setTab] = useState<ProfileTab>(initialTab)

  // The three texts.
  const [data, setData] = useState<ProfileResponse | null>(null)
  const [draft, setDraft] = useState<ProfileFields>(EMPTY_PROFILE)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [previewMode, setPreviewMode] = useState<PreviewMode>(DEFAULT_PREVIEW_MODE)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  // The saved profile the draft is based on — what a reload rebases against.
  const baseRef = useRef<ProfileFields | null>(null)

  // The material: name, links, company, documents.
  const [sources, setSources] = useState<ProfileSources | null>(null)
  const [sourcesError, setSourcesError] = useState<string | null>(null)
  const [details, setDetails] = useState<DetailsDraft>(EMPTY_DETAILS)
  const [detailsSaving, setDetailsSaving] = useState(false)
  const [detailsNotice, setDetailsNotice] = useState<string | null>(null)
  const [detailsError, setDetailsError] = useState<string | null>(null)
  const detailsDirtyRef = useRef(false)

  // The researcher.
  const [research, setResearch] = useState<Research>(IDLE_RESEARCH)
  const [researchBusy, setResearchBusy] = useState(false)
  const [researchError, setResearchError] = useState<string | null>(null)

  // `rebase` — refresh without dropping unsaved text: fields the user changed
  // keep their text, untouched ones take the server's (possibly newer) value.
  const load = useCallback(async (rebase = false) => {
    setLoading(true)
    const [profile, material, run] = await Promise.allSettled([getProfile(), getProfileSources(), getProfileResearch()])
    if (profile.status === 'fulfilled') {
      const r = profile.value
      const base = baseRef.current
      baseRef.current = r.profile
      setData(r)
      setDraft((d) => (rebase && base ? rebaseDraft(d, base, r.profile) : r.profile))
      setError(null)
    } else {
      setError(profile.reason instanceof Error ? profile.reason.message : String(profile.reason))
    }
    if (material.status === 'fulfilled') {
      setSources(material.value)
      // An edit under way keeps its text; otherwise the details follow the server.
      if (!(rebase && detailsDirtyRef.current)) setDetails(detailsOf(material.value))
      setSourcesError(null)
    } else {
      setSourcesError(material.reason instanceof Error ? material.reason.message : String(material.reason))
    }
    if (run.status === 'fulfilled') setResearch(run.value)
    setLoading(false)
  }, [])

  useEffect(() => { load() }, [load])
  // The shared-agent list is polled by the Agent Desktop and the simple
  // layout; opened straight onto this page, ask once whether the deck offers
  // context packs.
  useEffect(() => {
    const shared = useSharedAgentStore.getState()
    if (!shared.loaded) void shared.fetch()
  }, [])

  // A run under way: follow it until it ends.
  useEffect(() => {
    if (research.state !== 'running') return
    const timer = window.setInterval(() => {
      getProfileResearch().then(setResearch).catch(() => { /* asked again next tick */ })
    }, RESEARCH_POLL_MS)
    return () => window.clearInterval(timer)
  }, [research.state])

  const profileDirty = !!data && FIELDS.some((k) => draft[k] !== data.profile[k])
  const detailsDirty = isDetailsDirty(details, sources)
  detailsDirtyRef.current = detailsDirty
  const dirty = profileDirty || detailsDirty
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

  // The details: the name goes to the account, the rest to the material.
  const saveDetails = useCallback(async (): Promise<boolean> => {
    if (!sources) return false
    setDetailsSaving(true); setDetailsError(null); setDetailsNotice(null)
    try {
      let next = sources
      const renamed = nameChanged(details, sources)
      if (renamed) {
        await updateProfile({ display_name: details.name.trim() })
        next = { ...next, name: details.name.trim() }
      }
      const patch = detailsPatch(details, sources)
      if (Object.keys(patch).length) next = { ...(await saveProfileSources(patch)), name: next.name }
      setSources(next)
      setDetails(detailsOf(next))
      setDetailsNotice('Saved.')
      // The agents' block names the owner: a new name shows in the preview.
      if (renamed) void load(true)
      return true
    } catch (e) {
      setDetailsError(e instanceof Error ? e.message : String(e))
      return false
    } finally {
      setDetailsSaving(false)
    }
  }, [details, sources, load])

  const startResearch = async () => {
    setResearchError(null); setResearchBusy(true)
    try {
      if (detailsDirty && !(await saveDetails())) return
      setResearch(await startProfileResearch())
    } catch (e) {
      setResearchError(e instanceof Error ? e.message : String(e))
    } finally {
      setResearchBusy(false)
    }
  }

  const stopResearch = async () => {
    setResearchBusy(true)
    try {
      await stopProfileResearch()
      setResearch(await getProfileResearch())
    } catch (e) {
      setResearchError(e instanceof Error ? e.message : String(e))
    } finally {
      setResearchBusy(false)
    }
  }

  // A draft into its text box: the user reads it there, edits, and saves.
  const useDrafts = (fields: Partial<ProfileFields>) => {
    setDraft((d) => ({ ...d, ...fields }))
    setNotice(null)
    document.getElementById(TEXTS_ID)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const deck = data?.deck
  const deckEmpty = !deck || (!deck.company.trim() && !deck.instructions.trim())
  const usingDeckCompany = !!deck?.company.trim() && !draft.company.trim()
  const preview = data ? data.preview[previewMode] : ''
  // About me and My company also go to every agent my profile is shared on.
  const sharedNote = contextPacks ? profileSharedNote(profilePackAgents(myPacks)) : ''
  const showMyPacks = () => {
    document.getElementById(MY_PACKS_ID)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  const card = 'rounded-lg border border-zinc-800 bg-zinc-900/60 p-5 space-y-4'

  const tabs: ProfileTabSpec[] = [
    { id: 'personal', label: 'Personal', icon: UserRound, hint: 'You, your company, and what your agents are told' },
    { id: 'deck', label: 'Deck-wide', icon: Building2, hint: 'What every agent on this deck receives' },
    ...(authEnabled === true
      ? [{ id: 'password' as const, label: 'Password', icon: KeyRound, hint: 'Change the password you sign in with' }]
      : []),
  ]
  const shownTab: ProfileTab = tabs.some((t) => t.id === tab) ? tab : 'personal'

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

        <ProfileTabs tabs={tabs} tab={shownTab} onChange={setTab} />

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

        {/* ── Personal ── */}
        {shownTab === 'personal' && data && caps && (
          <div role="tabpanel" id="profile-panel-personal" aria-labelledby="profile-tab-personal" className="space-y-4">
            {sources ? (
              <PersonalDetails
                sources={sources}
                draft={details}
                onDraft={(d) => { setDetails(d); setDetailsNotice(null) }}
                dirty={detailsDirty}
                saving={detailsSaving}
                notice={detailsNotice}
                error={detailsError}
                onSave={() => void saveDetails()}
                onRevert={() => { setDetails(detailsOf(sources)); setDetailsError(null) }}
                onSources={(s) => { setSources(s); setDetails((d) => ({ ...d, name: s.name })) }}
              />
            ) : sourcesError ? (
              <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
                Your links and documents couldn't be loaded: {sourcesError}
              </div>
            ) : null}

            {sources && (
              <ProfileResearch
                research={research}
                canStart={!sourcesEmpty(details, sources)}
                busy={researchBusy}
                error={researchError}
                caps={caps}
                current={draft}
                onStart={() => void startResearch()}
                onStop={() => void stopResearch()}
                onUse={useDrafts}
              />
            )}

            {/* The three texts */}
            <div id={TEXTS_ID} className={card}>
              <div className="flex items-start gap-2">
                <MessageSquareText className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
                <div>
                  <h3 className="text-sm font-semibold text-zinc-200">What your agents are told</h3>
                  <p className="mt-0.5 text-xs text-zinc-500">Three short texts, in your words. Every agent of yours gets them.</p>
                </div>
              </div>
              <p className="text-[11px] text-zinc-500">{ECO_SHORTENED_HINT}</p>
              {sharedNote && (
                <div className="rounded-md border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-[11px] leading-relaxed text-amber-800 dark:text-amber-200">
                  {sharedNote}{' '}
                  <button
                    type="button"
                    onClick={showMyPacks}
                    className="font-medium underline underline-offset-2 hover:no-underline"
                  >
                    {`See “${MY_PACKS_HEADING}”`}
                  </button>
                </div>
              )}
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
                  ? "Empty — your agents get the deck's company description (see Deck-wide). Anything you write here replaces it."
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
                  disabled={saving || !profileDirty || tooLong}
                  className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
                >
                  {saving && <Loader2 className="h-3 w-3 animate-spin" />}
                  {saving ? 'Saving…' : 'Save'}
                </button>
                {profileDirty && (
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
              {profileDirty && (
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

            {/* What I share with agents' people (context packs), on every
                agent — also with sharing off (paused, still stoppable) while
                Flight Deck lists any; hidden only on a deck without packs. */}
            <MyContextPacks
              sharingOff={sharedLoaded && !contextPacks}
              showEmpty={contextPacks}
              onPacks={setMyPacks}
            />
          </div>
        )}

        {/* ── Deck-wide ── */}
        {shownTab === 'deck' && data && (
          <div role="tabpanel" id="profile-panel-deck" aria-labelledby="profile-tab-deck" className="space-y-4">
            {/* Editable by an admin (and the auth-off deck's one user) — never
                from the kiosk: a visitor there must not change what every agent
                on the deck receives. Everyone else reads them. */}
            {deckDefaultsEditable(authEnabled, kiosk, isAdmin) ? (
              <DeckProfileDefaults onSaved={() => load(true)} />
            ) : (
              <div className={card}>
                <div className="flex items-start gap-2">
                  <Building2 className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
                  <div>
                    <h3 className="text-sm font-semibold text-zinc-200">Deck-wide defaults</h3>
                    <p className="text-xs text-zinc-500 mt-0.5">
                      Set by the deck's admin for everyone. Your company text replaces the deck's; the deck's
                      instructions apply alongside your preferences and win if they conflict.
                    </p>
                  </div>
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
                {isAdmin && kiosk && (
                  <p className="text-[11px] text-zinc-600">
                    To change them, switch to the admin view: Settings → Switch to admin view → Profile → Deck-wide.
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Password ── */}
        {shownTab === 'password' && authEnabled === true && (
          <div role="tabpanel" id="profile-panel-password" aria-labelledby="profile-tab-password">
            <ChangePasswordCard />
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
