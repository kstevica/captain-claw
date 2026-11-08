import { useMemo, useState } from 'react'
import {
  Library, Plus, Wand2, Search, Crown, Pencil, Trash2, Copy, Share2, Users, Sparkles,
  Check, Wrench, Gauge, Layers, Lock, Building2,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { useTierConfig, TIER_ORDER, type Archetype, type ArchetypeCategory, type ArchetypeRegistry } from '../../services/tierConfig'
import { CATEGORIES, archetypeCategory, archetypeKind, companyShelves, kindLook, rememberTeamLooks, shelves, type ArchetypeKind } from '../layout/archetypeLook'
import { KindTabs, type KindFilter } from '../layout/KindTabs'
import { TeamPackDialog } from './TeamPackDialog'
import {
  DISCLAIMER_CHOICES,
  createArchetype, updateArchetype, deleteArchetype, deleteTeamArchetype, deleteTeamGroup, generateArchetype, updateTeamArchetype,
  publishTeamPack,
  type ArchetypeInput,
} from '../../services/archetypes'
import { MODES } from '../../services/cognitiveModes'
import { ForgeArchetypesModal } from '../library/ForgeArchetypesModal'
import { ShareModal } from '../common/ShareModal'
import { Switch } from '../admin/Switch'
import { matchesQuery } from '../admin/deckConfig'
import {
  Page, PageHeader, Pill, Button, Sheet, Field, Notice, Segmented, EmptyState, inputCls,
} from './parts'

// Archetypes: the ready-made agents people start from (the kiosk's "New
// agent" picker lists the same gallery). Built-in ones come with Captain Claw;
// yours can be edited, shared with other users, or forged in bulk from
// documents. A built-in or shared one is copied before it's changed. What the
// admin publishes for everyone is the Company kind, in groups of its own.

type Source = 'all' | 'user' | 'base' | 'shared'

function slugify(s: string): string {
  return s.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
}

function toInput(a: Archetype): ArchetypeInput {
  return {
    archetype_id: a.id, role: a.role, family: a.family || 'Custom', category: archetypeCategory(a), disclaimer: a.disclaimer || '', icon: a.icon || '', description: a.description || '',
    cognitive_mode: a.cognitive_mode || 'neutra', tier: a.tier || 'balanced', tools: a.tools || [],
    fleet_instructions: a.fleet_instructions || '', keywords: a.keywords || [], lead: !!a.lead,
    reliability_seed: a.reliability_seed ?? 0.7, runtime: a.runtime || '',
  }
}

function freeId(registry: ArchetypeRegistry | null, base: string): string {
  const taken = new Set((registry?.archetypes || []).map((x) => x.id))
  let id = `${base}-copy`
  for (let n = 2; taken.has(id); n += 1) id = `${base}-copy-${n}`
  return id
}

export function ArchetypesPage() {
  // No starting set here either (see ModelsPage); drafting needs one.
  const { registry, refreshRegistry, tiers, forgeTier, sets, bootstrapped } = useTierConfig({ seed: false })
  const noModels = bootstrapped && sets.length === 0
  const [query, setQuery] = useState('')
  const [source, setSource] = useState<Source>('all')
  const [family, setFamily] = useState('')
  const [kind, setKind] = useState<KindFilter>('all')
  // A shelf under the kind: the company's groups, or specialists for work or for life.
  const [shelf, setShelf] = useState<ArchetypeCategory | ''>('')
  // The editor: a draft plus the id it saves over (null = a new archetype).
  const [draft, setDraft] = useState<{ input: ArchetypeInput; id: string | null; team?: boolean } | null>(null)
  const [viewing, setViewing] = useState<Archetype | null>(null)
  // One of the admin's own archetypes, on its way into a company group.
  const [promoting, setPromoting] = useState<Archetype | null>(null)
  const [forgeOpen, setForgeOpen] = useState(false)
  const [packOpen, setPackOpen] = useState(false)
  const [shareTarget, setShareTarget] = useState<Archetype | null>(null)
  const [error, setError] = useState('')
  // Sharing needs other users — none on a deck without sign-in.
  const accounts = useAuthStore((s) => s.authEnabled !== false)

  rememberTeamLooks(registry)
  const all = useMemo(() => registry?.archetypes || [], [registry])
  const counts = useMemo(() => ({
    all: all.length,
    user: all.filter((a) => a.source === 'user').length,
    base: all.filter((a) => (a.source || 'base') === 'base').length,
    shared: all.filter((a) => a.source === 'shared').length,
  }), [all])
  const kindCounts = useMemo(() => {
    const out: Partial<Record<ArchetypeKind, number>> = {}
    for (const a of all) out[archetypeKind(a)] = (out[archetypeKind(a)] || 0) + 1
    return out
  }, [all])
  const inKind = (a: Archetype) => (kind === 'all' || archetypeKind(a) === kind) && (!shelf || archetypeCategory(a) === shelf)
  // The shelves under the chosen kind, worth a row of their own when there's a choice,
  // and always for the company's groups: removing one starts there.
  const kindShelves = kind === 'all' ? [] : shelves()
    .filter((c) => c.kind === kind)
    .map((c) => ({ ...c, count: all.filter((a) => archetypeCategory(a) === c.id).length }))
    .filter((c) => c.count > 0)
  // The company's families first, as its archetypes are.
  const companyFamilies = new Set(all.filter((a) => archetypeKind(a) === 'company').map((a) => a.family || 'Custom'))
  const families = [...new Set(all.filter(inKind).map((a) => a.family || 'Custom'))]
    .sort((x, y) => Number(companyFamilies.has(y)) - Number(companyFamilies.has(x)) || x.localeCompare(y))

  const shown = all.filter((a) =>
    (source === 'all' || (a.source || 'base') === source)
    && inKind(a)
    && (!family || (a.family || 'Custom') === family)
    && matchesQuery(query, a.role, a.id, a.description, a.family, (a.keywords || []).join(' ')),
  )

  // Every tool the gallery uses, for the editor's palette.
  const toolPalette = useMemo(() => {
    const s = new Set<string>(registry?.base_tools || [])
    all.forEach((a) => a.tools?.forEach((t) => s.add(t)))
    return [...s].sort()
  }, [registry, all])

  const blank = (): ArchetypeInput => ({
    archetype_id: '', role: '', family: 'Custom', category: 'specialist', description: '', cognitive_mode: 'neutra', tier: 'balanced',
    runtime: '', tools: registry?.base_tools ? [...registry.base_tools] : [], fleet_instructions: '',
    keywords: [], lead: false, reliability_seed: 0.7,
  })

  // A copy is the person's own, so it leaves the company's groups.
  const copyOf = (a: Archetype): ArchetypeInput => ({
    ...toInput(a), archetype_id: freeId(registry, slugify(a.id || a.role || 'archetype')), role: `${a.role} (copy)`,
    ...(archetypeKind(a) === 'company' ? { category: 'specialist' } : {}),
  })

  const remove = async (a: Archetype) => {
    const team = a.source === 'team'
    if (!confirm(team
      ? `Remove "${a.role}" for everyone on this deck? Agents already made from it keep working.`
      : `Delete the archetype "${a.role}"?${a.overrides ? ' The built-in version comes back.' : ''}`)) return
    setError('')
    try {
      await (team ? deleteTeamArchetype(a.id) : deleteArchetype(a.id))
      refreshRegistry()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const removeGroup = async () => {
    const group = companyShelves().find((c) => c.id === shelf)
    const inIt = all.filter((a) => a.source === 'team' && a.category === shelf).length
    if (!group || !confirm(`Remove "${group.label}" and its ${inIt} archetype${inIt === 1 ? '' : 's'} for everyone on this deck? Agents already made from them keep working.`)) return
    setError('')
    try {
      await deleteTeamGroup(group.id)
      setShelf('')
      setFamily('')
      refreshRegistry()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const open = (a: Archetype) => {
    if (a.source === 'user' || a.source === 'team') setDraft({ input: toInput(a), id: a.id, team: a.source === 'team' })
    else setViewing(a)
  }

  return (
    <Page>
      <PageHeader
        icon={Library}
        title="Archetypes"
        description="Ready-made agents people pick to start a new agent: a role, instructions, tools and a model tier. Company archetypes are for everyone on this deck: Generate a group makes them, and any of yours can become one. Until then, yours are only for you."
        actions={<>
          <Button icon={Layers} onClick={() => setPackOpen(true)} disabled={noModels} title={noModels ? 'Create a tier set on Models first.' : 'A new group and family of archetypes, for everyone on this deck'}>Generate a group</Button>
          <Button icon={Wand2} onClick={() => setForgeOpen(true)} disabled={noModels} title={noModels ? 'Create a tier set on Models first — forging uses its model' : 'Draft several archetypes from instructions and documents'}>Forge from documents</Button>
          <Button variant="primary" icon={Plus} onClick={() => setDraft({ input: blank(), id: null })}>New archetype</Button>
        </>}
      />

      {error && <Notice tone="red">{error}</Notice>}

      <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by role, description or keyword…"
            className={`${inputCls} pl-9`}
          />
        </div>
        <Segmented
          value={source}
          onChange={setSource}
          label="Which archetypes"
          options={[
            { value: 'all', label: 'All', count: counts.all },
            { value: 'user', label: 'Yours', count: counts.user },
            { value: 'base', label: 'Built-in', count: counts.base },
            ...(counts.shared ? [{ value: 'shared' as const, label: 'Shared with you', count: counts.shared }] : []),
          ]}
        />
      </div>

      <KindTabs value={kind} onChange={(next) => { setKind(next); setShelf(''); setFamily('') }} counts={kindCounts} hints />
      {(kindShelves.length > 1 || kind === 'company') && (
        <div className="-mt-2 flex flex-wrap items-center gap-1.5">
          {[{ id: '' as const, label: `All ${kindLook(kind as ArchetypeKind).label.toLowerCase()}`, count: kindShelves.reduce((n, c) => n + c.count, 0) }, ...kindShelves].map((c) => (
            <button
              key={c.id || 'all'}
              onClick={() => { setShelf(c.id); setFamily('') }}
              className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors ${
                shelf === c.id
                  ? `${kindLook(kind as ArchetypeKind).active} text-zinc-100`
                  : 'border-zinc-800 text-zinc-500 hover:border-zinc-600 hover:text-zinc-200'
              }`}
            >
              {c.label} <span className="tabular-nums text-zinc-500">{c.count}</span>
            </button>
          ))}
        </div>
      )}
      {shelf.startsWith('group-') && (
        <div className="-mt-2 flex flex-wrap items-center gap-2 text-xs text-zinc-500">
          <span>{companyShelves().find((c) => c.id === shelf)?.hint}</span>
          <Button variant="danger" icon={Trash2} className="!px-2 !py-1 !text-xs" onClick={removeGroup}>Remove this group for everyone</Button>
        </div>
      )}

      {families.length > 1 && (
        <div className="-mt-2 flex flex-wrap gap-1.5">
          {['', ...families].map((f) => (
            <button
              key={f || 'all'}
              onClick={() => setFamily(f)}
              className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                family === f
                  ? 'border-violet-500/50 bg-violet-500/15 text-violet-700 dark:text-violet-200'
                  : 'border-zinc-800 text-zinc-500 hover:border-zinc-600 hover:text-zinc-200'
              }`}
            >
              {f || 'Every family'}
            </button>
          ))}
        </div>
      )}

      {!registry ? (
        <p className="text-sm text-zinc-500">Loading archetypes…</p>
      ) : shown.length === 0 ? (
        <EmptyState icon={Library} title={all.length ? 'Nothing matches' : 'No archetypes yet'}>
          {all.length ? 'Try another search or filter.' : 'Create one, or forge a set from your documents.'}
        </EmptyState>
      ) : (
        <div className="grid grid-cols-1 gap-3 wide:grid-cols-2 lg:grid-cols-3">
          {shown.map((a) => (
            <ArchetypeCard
              key={a.id}
              a={a}
              tierLabel={registry.tiers[a.tier]?.label || a.tier}
              onOpen={() => open(a)}
              onCopy={() => setDraft({ input: copyOf(a), id: null })}
              onShare={accounts ? () => setShareTarget(a) : undefined}
              onPromote={a.source === 'user' && !a.overrides ? () => setPromoting(a) : undefined}
              onDelete={() => remove(a)}
            />
          ))}
        </div>
      )}

      {promoting && (
        <PromoteSheet
          a={promoting}
          onClose={() => setPromoting(null)}
          onDone={() => { setPromoting(null); refreshRegistry() }}
        />
      )}

      {draft && (
        <ArchetypeSheet
          initial={draft.input}
          editingId={draft.id}
          team={!!draft.team}
          registry={registry}
          toolPalette={toolPalette}
          canDraft={!noModels}
          generate={(prompt) => {
            const ft = tiers[forgeTier]
            return generateArchetype(prompt, ft?.provider || '', ft?.model || '', ft?.api_key || '', ft?.base_url || '', ft?.output_ctx || 0)
          }}
          onClose={() => setDraft(null)}
          onSaved={() => { refreshRegistry(); setDraft(null) }}
        />
      )}

      {viewing && (
        <ArchetypeView
          a={viewing}
          tierLabel={registry?.tiers[viewing.tier]?.label || viewing.tier}
          disclaimers={registry?.disclaimers}
          onClose={() => setViewing(null)}
          onCopy={() => { setDraft({ input: copyOf(viewing), id: null }); setViewing(null) }}
        />
      )}

      {packOpen && (
        <TeamPackDialog
          existingFamilies={[...new Set(all.map((a) => a.family).filter(Boolean))].sort()}
          onClose={() => setPackOpen(false)}
          onPublished={() => refreshRegistry()}
        />
      )}

      {forgeOpen && (
        <ForgeArchetypesModal
          tiers={tiers}
          forgeTier={forgeTier}
          existingIds={all.map((a) => a.id)}
          toolPalette={toolPalette}
          onClose={() => setForgeOpen(false)}
          onSaved={() => refreshRegistry()}
        />
      )}

      {shareTarget && (
        <ShareModal
          resourceType="archetype"
          resourceId={shareTarget.id}
          resourceName={shareTarget.role}
          allowEdit={false}
          onClose={() => setShareTarget(null)}
        />
      )}
    </Page>
  )
}

function ArchetypeCard({ a, tierLabel, onOpen, onCopy, onShare, onPromote, onDelete }: {
  a: Archetype
  tierLabel: string
  onOpen: () => void
  onCopy: () => void
  onShare?: () => void
  onPromote?: () => void
  onDelete: () => void
}) {
  const mine = a.source === 'user' || a.source === 'team'
  const iconBtn = 'rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200'
  return (
    <div className="group flex flex-col rounded-2xl border border-zinc-800 bg-zinc-900/40 transition-colors hover:border-zinc-700">
      <button onClick={onOpen} className="flex flex-1 flex-col p-4 pb-2 text-left">
        <div className="flex items-start justify-between gap-2">
          <span className="flex min-w-0 items-center gap-1.5 text-sm font-semibold text-zinc-100">
            {a.lead && <Crown className="h-3.5 w-3.5 shrink-0 text-amber-500" />}
            <span className="truncate">{a.role}</span>
          </span>
          {a.source === 'team'
            ? <Pill tone="emerald" icon={Building2} title="Published for everyone on this deck">company</Pill>
            : mine
            ? <Pill tone="violet">{a.overrides ? 'yours · replaces built-in' : 'yours'}</Pill>
            : a.source === 'shared'
              ? <Pill tone="sky" icon={Users} title={`Shared by ${a.shared_owner_name || a.shared_owner_email || 'another user'}`}>shared</Pill>
              : <Pill icon={Lock}>built-in</Pill>}
        </div>
        <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-zinc-500">{a.description || 'No description.'}</p>
        <div className="mt-auto flex flex-wrap gap-1 pt-3">
          <Pill tone="cyan" icon={Gauge}>{tierLabel}</Pill>
          <Pill>{a.family || 'Custom'}</Pill>
          <Pill icon={Wrench}>{a.tools.length} tool{a.tools.length === 1 ? '' : 's'}</Pill>
          {a.runtime === 'mrav' && <Pill tone="amber">micro</Pill>}
        </div>
      </button>
      <div className="flex items-center justify-end gap-0.5 px-3 pb-2.5">
        {mine ? (
          <>
            {onPromote && <button onClick={onPromote} className={iconBtn} title="Make it a company archetype, for everyone on this deck"><Building2 className="h-4 w-4" /></button>}
            {onShare && a.source === 'user' && <button onClick={onShare} className={iconBtn} title="Share with other users"><Share2 className="h-4 w-4" /></button>}
            <button onClick={onOpen} className={iconBtn} title="Edit"><Pencil className="h-4 w-4" /></button>
            <button onClick={onDelete} className={`${iconBtn} hover:text-red-500`} title={a.overrides ? 'Delete (the built-in comes back)' : 'Delete'}><Trash2 className="h-4 w-4" /></button>
          </>
        ) : (
          <button onClick={onCopy} className={`${iconBtn} flex items-center gap-1 text-xs`} title="An editable copy in your archetypes">
            <Copy className="h-4 w-4" /> Make a copy
          </button>
        )}
      </div>
    </div>
  )
}

// ── One of yours, made a company archetype ──
// It joins a company group (an existing one, or a new one named here) through
// the same publish as Generate a group, and leaves your own archetypes.

const NEW_GROUP = '__new__'

function PromoteSheet({ a, onClose, onDone }: { a: Archetype; onClose: () => void; onDone: () => void }) {
  const groups = companyShelves()
  const [group, setGroup] = useState<string>(groups[0]?.id ?? NEW_GROUP)
  const [label, setLabel] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const ready = group !== NEW_GROUP || !!label.trim()

  const publish = async () => {
    setBusy(true)
    setError('')
    try {
      const input = toInput(a)
      await publishTeamPack({
        group: group === NEW_GROUP ? { label: label.trim() } : { id: group },
        family: { name: input.family, icon: '', color: '' },
        archetypes: [{ ...input, archetype_id: input.archetype_id, family: input.family }],
      })
      await deleteArchetype(a.id)
      onDone()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  return (
    <Sheet
      title="Make it a company archetype"
      subtitle={a.role}
      icon={Building2}
      onClose={onClose}
      footer={<>
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button variant="primary" icon={Building2} busy={busy} disabled={!ready} onClick={() => { void publish() }}>Publish to everyone</Button>
      </>}
    >
      <div className="space-y-4">
        <p className="text-sm leading-relaxed text-zinc-400">
          Everyone on this deck can then start an agent from it, under Company in their New agent picker.
          It moves out of your own archetypes. Only admins can change it.
        </p>
        {error && <Notice tone="red">{error}</Notice>}
        <Field label="Company group">
          <select value={group} onChange={(e) => setGroup(e.target.value)} className={inputCls}>
            {groups.map((g) => <option key={g.id} value={g.id}>{g.label}</option>)}
            <option value={NEW_GROUP}>A new group</option>
          </select>
        </Field>
        {group === NEW_GROUP && (
          <Field label="Its name">
            <input value={label} onChange={(e) => setLabel(e.target.value)} maxLength={40} className={inputCls} autoFocus />
          </Field>
        )}
      </div>
    </Sheet>
  )
}

// ── Read-only: a built-in or shared archetype ──

function ArchetypeView({ a, tierLabel, disclaimers, onClose, onCopy }: { a: Archetype; tierLabel: string; disclaimers?: Record<string, string>; onClose: () => void; onCopy: () => void }) {
  const mode = MODES.find((m) => m.id === a.cognitive_mode)
  return (
    <Sheet
      title={a.role}
      subtitle={a.source === 'shared' ? `Shared by ${a.shared_owner_name || a.shared_owner_email || 'another user'}` : 'Built-in — copy it to change it'}
      icon={Library}
      onClose={onClose}
      wide
      footer={<>
        <Button variant="ghost" onClick={onClose}>Close</Button>
        <Button variant="primary" icon={Copy} onClick={onCopy}>Make an editable copy</Button>
      </>}
    >
      <div className="space-y-5">
        <p className="text-sm leading-relaxed text-zinc-300">{a.description}</p>
        {a.disclaimer && <Notice tone="amber">{disclaimers?.[a.disclaimer] || a.disclaimer}</Notice>}
        <dl className="grid grid-cols-2 gap-4 text-sm">
          <div><dt className="text-xs text-zinc-500">Model tier</dt><dd className="mt-0.5 text-zinc-200">{tierLabel}</dd></div>
          <div><dt className="text-xs text-zinc-500">Thinking style</dt><dd className="mt-0.5 text-zinc-200">{mode ? `${mode.label} — ${mode.character}` : a.cognitive_mode}</dd></div>
          <div><dt className="text-xs text-zinc-500">Family</dt><dd className="mt-0.5 text-zinc-200">{a.family}</dd></div>
          <div><dt className="text-xs text-zinc-500">Runtime</dt><dd className="mt-0.5 text-zinc-200">{a.runtime === 'mrav' ? 'Micro (8k)' : 'Classic'}</dd></div>
        </dl>
        <div>
          <div className="mb-1.5 text-xs text-zinc-500">Tools</div>
          <div className="flex flex-wrap gap-1">{a.tools.map((t) => <Pill key={t}>{t}</Pill>)}</div>
        </div>
        {a.fleet_instructions && (
          <div>
            <div className="mb-1.5 text-xs text-zinc-500">Instructions</div>
            <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-xl border border-zinc-800 bg-zinc-900/60 p-3 font-sans text-xs leading-relaxed text-zinc-300">{a.fleet_instructions}</pre>
          </div>
        )}
      </div>
    </Sheet>
  )
}

// ── The editor ──

function ArchetypeSheet({ initial, editingId, team, registry, toolPalette, canDraft, generate, onClose, onSaved }: {
  initial: ArchetypeInput
  editingId: string | null
  // A team archetype: saving changes it for everyone on the deck.
  team: boolean
  registry: ArchetypeRegistry | null
  toolPalette: string[]
  // A tier set exists to draft with.
  canDraft: boolean
  generate: (prompt: string) => Promise<ArchetypeInput>
  onClose: () => void
  onSaved: () => void
}) {
  const [d, setD] = useState<ArchetypeInput>(initial)
  const [prompt, setPrompt] = useState('')
  const [generating, setGenerating] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [toolQuery, setToolQuery] = useState('')
  const [keywords, setKeywords] = useState(initial.keywords.join(', '))

  const set = (patch: Partial<ArchetypeInput>) => setD((prev) => ({ ...prev, ...patch }))
  const tools = [...new Set([...toolPalette, ...d.tools])].sort()
    .filter((t) => !toolQuery.trim() || t.toLowerCase().includes(toolQuery.trim().toLowerCase()))

  const runGenerate = async () => {
    if (!prompt.trim()) return
    setGenerating(true); setError('')
    try {
      const out = await generate(prompt.trim())
      // Editing keeps its id; a new one keeps an id typed already.
      const id = editingId || d.archetype_id || out.archetype_id
      setD({ ...out, archetype_id: id, runtime: out.runtime ?? d.runtime })
      setKeywords(out.keywords.join(', '))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setGenerating(false)
    }
  }

  const save = async () => {
    const body: ArchetypeInput = {
      ...d,
      archetype_id: editingId || slugify(d.archetype_id || d.role),
      keywords: keywords.split(',').map((s) => s.trim()).filter(Boolean),
    }
    if (!body.role.trim()) { setError('Give it a role — that is its name.'); return }
    if (!body.archetype_id) { setError('An id (or a role) is needed.'); return }
    setSaving(true); setError('')
    try {
      if (editingId && team) await updateTeamArchetype(editingId, body)
      else if (editingId) await updateArchetype(editingId, body)
      else await createArchetype(body)
      onSaved()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <Sheet
      title={editingId ? `Edit ${initial.role || 'archetype'}` : 'New archetype'}
      subtitle={team ? `Changes reach everyone on this deck. id: ${editingId}` : editingId ? `id: ${editingId}` : 'Describe it and let the model draft it, or fill it in.'}
      icon={Library}
      onClose={onClose}
      wide
      footer={<>
        {error && <span className="mr-auto text-xs text-red-600 dark:text-red-400">{error}</span>}
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button variant="primary" icon={Check} busy={saving} onClick={save}>{editingId ? 'Save changes' : 'Create archetype'}</Button>
      </>}
    >
      <div className="space-y-7">
        <section className="space-y-2 rounded-2xl border border-violet-500/20 bg-violet-500/[0.04] p-4">
          <div className="flex items-center gap-2 text-sm font-medium text-zinc-200">
            <Sparkles className="h-4 w-4 text-violet-600 dark:text-violet-400" /> Draft it from a description
          </div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={2}
            placeholder="e.g. A careful contract reviewer that flags risky clauses in vendor agreements"
            className={inputCls}
          />
          <div className="flex items-center justify-between gap-3">
            <span className="text-[11px] text-zinc-500">
              {canDraft ? 'Fills in the form below — check it before saving.' : 'Create a tier set on Models first — the draft is written by its model.'}
            </span>
            <Button icon={generating ? undefined : Sparkles} busy={generating} disabled={!prompt.trim() || !canDraft} onClick={runGenerate}>
              {generating ? 'Drafting…' : 'Draft'}
            </Button>
          </div>
        </section>

        <section className="space-y-4">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Basics</h3>
          <div className="grid grid-cols-1 gap-4 wide:grid-cols-2">
            <Field label="Role" hint="Its name, as people see it.">
              <input value={d.role} onChange={(e) => set({ role: e.target.value })} placeholder="Contract Reviewer" className={inputCls} />
            </Field>
            <Field label="Id" hint={editingId ? 'Fixed once created.' : 'Made from the role if left blank.'}>
              <input
                value={editingId || d.archetype_id}
                disabled={!!editingId}
                onChange={(e) => set({ archetype_id: e.target.value })}
                placeholder={slugify(d.role) || 'contract-reviewer'}
                className={`${inputCls} font-mono`}
              />
            </Field>
          </div>
          <Field label="Description" hint="One sentence on what it does — shown in the picker.">
            <input value={d.description} onChange={(e) => set({ description: e.target.value })} className={inputCls} />
          </Field>
          <Field label="Kind" hint={team ? 'The company group it sits in, for everyone.' : 'Where it sits in the picker.'}>
            <select value={archetypeCategory(d)} onChange={(e) => set({ category: e.target.value })} className={inputCls}>
              {(team ? companyShelves() : CATEGORIES).map((c) => <option key={c.id} value={c.id}>{c.full}</option>)}
            </select>
          </Field>
          <Field label="Disclaimer" hint="For legal or health work: the agent says it, and the deck shows it in the picker and under the chat.">
            <select value={d.disclaimer || ''} onChange={(e) => set({ disclaimer: e.target.value })} className={inputCls}>
              {DISCLAIMER_CHOICES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </Field>
          <Field label="Family" hint="Groups archetypes in the picker.">
            <input value={d.family} onChange={(e) => set({ family: e.target.value })} placeholder="Custom" className={inputCls} list="fd-console-families" />
            <datalist id="fd-console-families">
              {[...new Set((registry?.archetypes || []).map((a) => a.family))].map((f) => <option key={f} value={f} />)}
            </datalist>
          </Field>
        </section>

        <section className="space-y-4">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">How it works</h3>
          <div className="grid grid-cols-1 gap-4 wide:grid-cols-2">
            <Field label="Model tier" hint="Which model of the tier set it runs on.">
              <select value={d.tier} onChange={(e) => set({ tier: e.target.value })} className={inputCls}>
                {TIER_ORDER.map((t) => <option key={t} value={t}>{registry?.tiers[t]?.label || t}</option>)}
              </select>
            </Field>
            <Field label="Thinking style">
              <select value={d.cognitive_mode} onChange={(e) => set({ cognitive_mode: e.target.value })} className={inputCls}>
                {MODES.map((m) => <option key={m.id} value={m.id}>{m.label} — {m.character}</option>)}
              </select>
            </Field>
            <Field label="Runtime" hint="Micro runs the whole loop in 8k tokens — for small local models.">
              <select value={d.runtime || ''} onChange={(e) => set({ runtime: e.target.value })} className={inputCls}>
                <option value="">Classic (default)</option>
                <option value="mrav">Micro — 8k cap</option>
              </select>
            </Field>
            <div className="flex items-start justify-between gap-4 rounded-xl border border-zinc-800 px-3 py-2.5">
              <div>
                <div className="text-sm font-medium text-zinc-200">Team lead</div>
                <div className="mt-0.5 text-xs text-zinc-500">Coordinates the others in a multi-agent run.</div>
              </div>
              <Switch on={d.lead} onToggle={() => set({ lead: !d.lead })} label="Team lead" />
            </div>
          </div>
        </section>

        <section className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
              Tools <span className="normal-case tracking-normal text-zinc-600">— {d.tools.length} on</span>
            </h3>
            <input value={toolQuery} onChange={(e) => setToolQuery(e.target.value)} placeholder="Filter tools" className={`${inputCls} w-44 py-1.5 text-xs`} />
          </div>
          <div className="flex max-h-52 flex-wrap gap-1.5 overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-900/40 p-2.5">
            {tools.map((t) => {
              const on = d.tools.includes(t)
              return (
                <button
                  key={t}
                  type="button"
                  onClick={() => set({ tools: on ? d.tools.filter((x) => x !== t) : [...d.tools, t] })}
                  className={`rounded-lg border px-2 py-0.5 font-mono text-[11px] transition-colors ${
                    on
                      ? 'border-violet-500/40 bg-violet-500/15 text-violet-700 dark:text-violet-200'
                      : 'border-zinc-800 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300'
                  }`}
                >
                  {t}
                </button>
              )
            })}
            {tools.length === 0 && <span className="px-1 text-xs text-zinc-500">No tool matches.</span>}
          </div>
        </section>

        <section className="space-y-4">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Instructions</h3>
          <Field label="Operating manual" hint="What it is, what it does, and how — every agent made from it starts with this.">
            <textarea
              value={d.fleet_instructions}
              onChange={(e) => set({ fleet_instructions: e.target.value })}
              rows={10}
              placeholder="You are a … Your job is to … Work like this: 1) …"
              className={`${inputCls} text-[13px] leading-relaxed`}
            />
          </Field>
          <Field label="Keywords" hint="Comma-separated — they help route work to it in multi-agent runs.">
            <input value={keywords} onChange={(e) => setKeywords(e.target.value)} placeholder="contracts, legal, risk" className={inputCls} />
          </Field>
        </section>
      </div>
    </Sheet>
  )
}
