import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Gauge, Plus, Copy, Trash2, Users, Check, Wand2, KeyRound, Eye, EyeOff, Zap, Pencil,
  AlertTriangle, Layers, Server, Image as ImageIcon, Brain, Hammer, Upload,
} from 'lucide-react'
import {
  useTierConfig, TIER_ORDER, LLM_PROVIDERS, INPUT_CTX_OPTIONS, OUTPUT_CTX_OPTIONS, fmtCtxTokens,
  REASONING_EFFORTS, REASONING_EFFORT_PROVIDERS, splitReasoningEffort, withReasoningEffort,
  type TierSet, type TierConfig, type TierDef, type EnvVar,
} from '../../services/tierConfig'
import { publishSharedTierSets, type PublishResult } from '../../services/sharedTierSets'
import { fetchTierPresets, type TierPresetCatalog } from '../../services/tierPresets'
import { useAuthStore } from '../../stores/authStore'
import { useConsoleStore } from '../../stores/consoleStore'
import { Switch } from '../admin/Switch'
import { TEAM_KEY_PROVIDERS, fetchProviderKeys, saveProviderKey } from './consoleApi'
import { QuickStartForm } from './QuickStartForm'
import {
  Page, PageHeader, Card, Pill, Button, Sheet, Dialog, Field, ChipPicker, Notice, Segmented,
  inputCls, type Tone,
} from './parts'

// Models: the admin's tier sets (useTierConfig — the same sets Library edits)
// as cards, each tier a summary that opens an editor, publishing a set to the
// team (what teammates without models of their own run on), and the team's
// API keys per provider. Selecting a set also makes it the admin's own active
// set, as in Library.

const SHORT: Record<string, string> = {
  anthropic: 'Anthropic', openai: 'OpenAI', ollama: 'Ollama', gemini: 'Gemini',
  'antigravity-cli': 'Antigravity', xai: 'xAI', openrouter: 'OpenRouter', litert: 'LiteRT',
  browser: 'Browser', deepseek: 'DeepSeek',
}
const providerName = (p: string) => SHORT[p] || p || 'No provider'

const PROVIDER_TONE: Record<string, Tone> = {
  anthropic: 'amber', openai: 'emerald', gemini: 'sky', openrouter: 'violet', ollama: 'cyan',
  'antigravity-cli': 'sky', deepseek: 'sky',
}

// What a published copy keeps of a set (the server's _mask_shared_sets): a
// tier key becomes "@system", a blank one stays blank.
function maskSet(s: TierSet): TierSet {
  return {
    ...s,
    tiers: Object.fromEntries(Object.entries(s.tiers).map(([k, t]) => [k, { ...t, api_key: t.api_key?.trim() ? '@system' : '' }])),
  }
}

// Everything that reaches a teammate, to tell whether the set changed since it
// was published. A key counts by being there (the copy only says "@system").
function setSignature(s: TierSet): string {
  const tier = (t?: TierConfig) => t
    ? [t.provider, t.model.trim(), (t.base_url || '').trim(), t.input_ctx, t.output_ctx, t.weak ? 1 : 0, t.vision ? 1 : 0, t.api_key?.trim() ? 1 : 0].join('|')
    : '-'
  return [
    s.name.trim(), s.forgeTier,
    ...TIER_ORDER.map((k) => tier(s.tiers[k])),
    ...s.envVars.filter((e) => e.key.trim() && e.value.trim()).map((e) => `${e.key.trim()}=${e.value}`),
  ].join('\n')
}

function providersOf(s: TierSet): string[] {
  return [...new Set(TIER_ORDER.map((k) => s.tiers[k]?.provider).filter(Boolean) as string[])]
}

function keyHint(v: string): string {
  return v.length >= 4 ? `····${v.slice(-4)}` : '····'
}

export function ModelsPage() {
  // The console never makes a starting set: the admin creates the first one.
  const tc = useTierConfig({ seed: false })
  const {
    tiers, setTiers, forgeTier, setForgeTier, envVars, setEnvVars, registry, updateTier, setupSet, setupPresetSet, bootstrapped,
    sets, activeSetId, setActiveSet, addSet, duplicateSet, renameSet, deleteSet, sharedSets, adoptSharedSet,
  } = tc

  // A deck without sign-in (the desktop app) has no team to publish to.
  const accounts = useAuthStore((s) => s.authEnabled !== false)
  // The team's published copy. Loaded with the sets; replaced when you publish.
  const [teamSets, setTeamSets] = useState<TierSet[]>([])
  useEffect(() => { setTeamSets(sharedSets) }, [sharedSets])
  const teamSet = accounts ? teamSets[0] || null : null

  const [editingTier, setEditingTier] = useState<string | null>(null)
  // Sent here to create the first tier set (the Overview, the launch): open "New tier set".
  const [newOpen, setNewOpen] = useState(() => !!useConsoleStore.getState().intent.createSet)
  useEffect(() => {
    useConsoleStore.getState().takeIntent()
    return useConsoleStore.subscribe((s) => {
      if (!s.intent.createSet) return
      setNewOpen(true)
      useConsoleStore.getState().takeIntent()
    })
  }, [])
  const [oneModelOpen, setOneModelOpen] = useState(false)
  // Quick starts (provider + key → its current models). null while loading.
  const [catalog, setCatalog] = useState<TierPresetCatalog | null>(null)
  useEffect(() => { fetchTierPresets().then(setCatalog) }, [])
  const [publishing, setPublishing] = useState(false)
  const [publishNote, setPublishNote] = useState<PublishResult | null>(null)
  const [publishErr, setPublishErr] = useState('')
  const [keys, setKeys] = useState<Record<string, string>>({})
  const reloadKeys = useCallback(() => { fetchProviderKeys().then(setKeys) }, [])
  useEffect(() => { reloadKeys() }, [reloadKeys])

  const active = sets.find((s) => s.id === activeSetId) || sets[0] || null
  const isTeam = (s: TierSet) => !!teamSet && teamSet.id === s.id
  const changedSincePublish = !!active && isTeam(active) && setSignature(active) !== setSignature(teamSet!)

  const publish = async () => {
    if (!active) return
    const replacing = teamSet && teamSet.id !== active.id ? ` It replaces "${teamSet.name}".` : ''
    // The set's extra keys (Brave, Tavily …) travel with it too.
    const extra = (active.envVars || []).filter((v) => v.key?.trim()).map((v) => v.key.trim())
    const extraLine = extra.length
      ? ` Its extra keys for agents (${extra.join(', ')}) go to teammates' agents too.`
      : ''
    if (!confirm(
      `Publish "${active.name}" to the team?${replacing}\n\n` +
      'Teammates who haven\'t set up models of their own will run on it. The API keys in this set become ' +
      `the team's keys for those models — they are written into teammates' agents.${extraLine}`,
    )) return
    setPublishing(true); setPublishErr(''); setPublishNote(null)
    try {
      setPublishNote(await publishSharedTierSets([active], active.id))
      setTeamSets([maskSet(active)])
      reloadKeys()
    } catch (e) {
      setPublishErr(e instanceof Error ? e.message : String(e))
    } finally {
      setPublishing(false)
    }
  }

  const removeSet = () => {
    if (!active || sets.length <= 1) return
    const note = isTeam(active) ? '\n\nThe team keeps running on the copy you published.' : ''
    if (confirm(`Delete the tier set "${active.name}"?${note}`)) deleteSet(active.id)
  }

  // The team's set isn't one of yours (another admin published it, or you
  // deleted your copy): copy it in to change it.
  const foreignTeamSet = teamSet && !sets.some((s) => s.id === teamSet.id) ? teamSet : null

  // Providers the team's models use on their own endpoint with no team key.
  const teamNeedsKey = useMemo(() => {
    if (!teamSet) return new Set<string>()
    const out = new Set<string>()
    for (const k of TIER_ORDER) {
      const t = teamSet.tiers[k]
      if (t?.model && !t.base_url?.trim() && TEAM_KEY_PROVIDERS.some((p) => p.value === t.provider) && !keys[t.provider]) {
        out.add(t.provider)
      }
    }
    return out
  }, [teamSet, keys])

  const editing = editingTier && tiers[editingTier] ? editingTier : null

  const foreignNotice = foreignTeamSet && (
    <Notice tone="zinc">
      <div className="flex flex-wrap items-center gap-3">
        <Users className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
        <span className="min-w-0 flex-1">
          The team runs on <b className="text-zinc-200">{foreignTeamSet.name}</b>, which isn't one of your sets.
          Copy it in to change it, then publish your copy.
        </span>
        <Button icon={Copy} onClick={() => adoptSharedSet(foreignTeamSet.id)}>Copy to my sets</Button>
      </div>
    </Notice>
  )

  return (
    <Page>
      <PageHeader
        icon={Gauge}
        title="Models"
        description={<>
          A tier set gives each kind of work — reasoning, everyday, fast, long documents, code, images — a model.
          {accounts && ' Publish one to the team and everyone who hasn\'t set up models of their own runs on it.'}
        </>}
        actions={<Button variant="primary" icon={Plus} onClick={() => setNewOpen(true)}>New tier set</Button>}
      />

      {!bootstrapped ? (
        <Card><p className="text-sm text-zinc-500">Loading your tier sets…</p></Card>
      ) : !active ? (
        <>
          <section className="flex flex-col items-center rounded-2xl border border-dashed border-violet-500/40 bg-violet-500/[0.04] px-6 py-12 text-center">
            <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-violet-500/15 text-violet-600 dark:text-violet-400">
              <Layers className="h-6 w-6" />
            </span>
            <h2 className="mt-4 text-lg font-semibold text-zinc-100">Create your first tier set</h2>
            <p className="mt-1.5 max-w-md text-sm leading-relaxed text-zinc-500">
              Agents need models before they can work. Pick your AI provider and paste its API key: every kind of work
              gets that provider's current models. Fine-tune each tier later.
            </p>
            <Button variant="primary" icon={Plus} className="mt-5" onClick={() => setNewOpen(true)}>Create a tier set</Button>
          </section>
          {foreignNotice}
          <TeamKeysCard keys={keys} onSaved={setKeys} needsKey={teamNeedsKey} team={accounts} />
        </>
      ) : (
        <>
          {/* ── The sets ── */}
          <div className="grid grid-cols-1 gap-3 wide:grid-cols-2 lg:grid-cols-3">
            {sets.map((s) => (
              <SetCard
                key={s.id}
                set={s}
                selected={s.id === active.id}
                team={isTeam(s)}
                stale={isTeam(s) && setSignature(s) !== setSignature(teamSet!)}
                onSelect={() => { setActiveSet(s.id); setPublishNote(null); setPublishErr('') }}
              />
            ))}
            <button
              onClick={() => setNewOpen(true)}
              className="flex min-h-[7.5rem] flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-zinc-700 text-sm text-zinc-500 transition-colors hover:border-violet-500/50 hover:text-zinc-200"
            >
              <Plus className="h-5 w-5" /> New tier set
            </button>
          </div>

          {foreignNotice}

          {/* ── The selected set ── */}
          <Card bodyClassName="p-5 space-y-5">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
              <div className="min-w-0 flex-1">
                <label className="sr-only" htmlFor="fd-console-set-name">Set name</label>
                <input
                  id="fd-console-set-name"
                  value={active.name}
                  onChange={(e) => renameSet(active.id, e.target.value)}
                  placeholder="Name this set"
                  className="w-full rounded-lg border border-transparent bg-transparent px-2 py-1 -ml-2 text-xl font-semibold tracking-tight text-zinc-100 hover:border-zinc-800 focus:border-violet-500/50 focus:outline-none"
                />
                <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                  <Pill tone="violet" icon={Check} title="Your own agents, Agent Forge and Basna use this set">In use by you</Pill>
                  {isTeam(active) && (changedSincePublish
                    ? <Pill tone="amber" icon={AlertTriangle}>Changes not published</Pill>
                    : <Pill tone="emerald" icon={Users}>The team's models</Pill>)}
                  {accounts && !isTeam(active) && <Pill>Only you</Pill>}
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {accounts && (!isTeam(active) || changedSincePublish) && (
                  <Button variant="primary" icon={Upload} busy={publishing} onClick={publish}>
                    {isTeam(active) ? 'Publish changes' : 'Publish to team'}
                  </Button>
                )}
                <Button icon={Wand2} onClick={() => setOneModelOpen(true)} title="Give every tier the same model">One model for all</Button>
                <Button icon={Copy} onClick={() => duplicateSet(active.id)} title="Duplicate this set">Duplicate</Button>
                <Button
                  variant="danger"
                  icon={Trash2}
                  onClick={removeSet}
                  disabled={sets.length <= 1}
                  title={sets.length <= 1 ? 'Keep at least one set' : 'Delete this set'}
                  aria-label="Delete this set"
                />
              </div>
            </div>

            {publishErr && <Notice tone="red">Couldn't publish: {publishErr}</Notice>}
            {publishNote && <PublishNotes note={publishNote} />}

            <div className="grid grid-cols-1 gap-3 wide:grid-cols-2">
              {TIER_ORDER.filter((k) => tiers[k]).map((k) => (
                <TierCard key={k} tierKey={k} def={registry?.tiers[k]} tc={tiers[k]} onEdit={() => setEditingTier(k)} />
              ))}
            </div>
            {Object.keys(tiers).length === 0 && <p className="text-sm text-zinc-500">Loading the tiers…</p>}

            {/* Less-used settings of the set: the Forge tier lines up with the
                left tier column, the extra keys get the whole width below it. */}
            <div className="space-y-5 border-t border-zinc-800/80 pt-5">
              <div className="wide:w-[calc(50%-0.375rem)]">
                <Field
                  label={<span className="flex items-center gap-1.5"><Hammer className="h-3.5 w-3.5" /> Designs teams in Agent Forge</span>}
                  hint="The tier whose model plans a team of agents from a goal."
                >
                  <select value={forgeTier} onChange={(e) => setForgeTier(e.target.value)} className={inputCls}>
                    {TIER_ORDER.filter((t) => tiers[t]).map((t) => (
                      <option key={t} value={t}>{registry?.tiers[t]?.label || t} — {tiers[t].model || 'no model'}</option>
                    ))}
                  </select>
                </Field>
              </div>
              <ExtraKeys envVars={envVars} setEnvVars={setEnvVars} />
            </div>
          </Card>

          <TeamKeysCard keys={keys} onSaved={setKeys} needsKey={teamNeedsKey} team={accounts} />
        </>
      )}

      {editing && (
        <TierSheet
          tierKey={editing}
          def={registry?.tiers[editing]}
          tc={tiers[editing]}
          onChange={(patch) => updateTier(editing, patch)}
          onUseEverywhere={() => {
            const t = tiers[editing]
            if (!confirm(`Give every tier in "${active?.name}" the model ${providerName(t.provider)} · ${t.model || '(none)'}?`)) return
            const apply: Partial<TierConfig> = {
              provider: t.provider, model: t.model, api_key: t.api_key, base_url: t.base_url,
              input_ctx: t.input_ctx, output_ctx: t.output_ctx, vision: t.vision, weak: t.weak,
            }
            setTiers((prev) => Object.fromEntries(Object.entries(prev).map(([k, v]) => [k, { ...v, ...apply }])))
          }}
          onClose={() => setEditingTier(null)}
        />
      )}

      {newOpen && (
        <NewSetDialog
          copyName={active?.name || ''}
          catalog={catalog}
          tierLabel={(k) => registry?.tiers[k]?.label || k}
          existingNames={sets.map((s) => s.name)}
          onClose={() => setNewOpen(false)}
          onQuick={(name, perTier) => { setupPresetSet(name, perTier, { replaceSeed: false }); setNewOpen(false) }}
          onOneModel={(name, apply) => { setupSet(name, apply, { replaceSeed: false }); setNewOpen(false) }}
          onDefaults={(name) => { addSet(name); setNewOpen(false) }}
          onCopy={(name) => {
            if (!active) return
            const id = duplicateSet(active.id)
            if (id && name.trim()) renameSet(id, name.trim())
            setNewOpen(false)
          }}
        />
      )}

      {oneModelOpen && active && (
        <Dialog
          title={`One model for every tier of "${active.name}"`}
          icon={Wand2}
          onClose={() => setOneModelOpen(false)}
        >
          <OneModelForm
            submitLabel="Apply to every tier"
            initial={tiers.balanced || Object.values(tiers)[0]}
            onSubmit={(apply) => {
              setTiers((prev) => Object.fromEntries(Object.entries(prev).map(([k, v]) => [k, { ...v, ...apply }])))
              setOneModelOpen(false)
            }}
          />
        </Dialog>
      )}
    </Page>
  )
}

// ── A set ──

function SetCard({ set, selected, team, stale, onSelect }: {
  set: TierSet
  selected: boolean
  team: boolean
  stale: boolean
  onSelect: () => void
}) {
  const providers = providersOf(set)
  const main = set.tiers.balanced?.model || set.tiers.reason?.model || ''
  return (
    <button
      onClick={onSelect}
      aria-pressed={selected}
      className={`flex min-h-[7.5rem] flex-col rounded-2xl border p-4 text-left transition-colors ${
        selected
          ? 'border-violet-500/60 bg-violet-500/[0.06] ring-1 ring-violet-500/30'
          : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-900/70'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="flex min-w-0 items-center gap-2">
          <Layers className={`h-4 w-4 shrink-0 ${selected ? 'text-violet-600 dark:text-violet-400' : 'text-zinc-500'}`} />
          <span className="truncate text-sm font-semibold text-zinc-100">{set.name || 'Untitled set'}</span>
        </span>
        {team && (stale
          ? <Pill tone="amber" title="Changed since you published it">Team · changed</Pill>
          : <Pill tone="emerald" icon={Users}>Team</Pill>)}
      </div>
      <div className="mt-3 flex flex-wrap gap-1">
        {providers.length === 0 && <Pill>no models yet</Pill>}
        {providers.slice(0, 3).map((p) => <Pill key={p} tone={PROVIDER_TONE[p] || 'zinc'}>{providerName(p)}</Pill>)}
        {providers.length > 3 && <Pill>+{providers.length - 3}</Pill>}
      </div>
      <div className="mt-auto pt-3 truncate font-mono text-[11px] text-zinc-500">{main || '—'}</div>
    </button>
  )
}

function PublishNotes({ note }: { note: PublishResult }) {
  const list = (xs: string[]) => xs.map(providerName).join(', ')
  return (
    <Notice tone={note.teamKeysMissing.length || note.teamKeysConflict.length ? 'amber' : 'emerald'}>
      <div className="space-y-1">
        <div className="font-medium">Published — the team now runs on this set.</div>
        {note.teamKeysAdded.length > 0 && <div>Your {list(note.teamKeysAdded)} key is now the team key.</div>}
        {note.teamKeysUpdated.length > 0 && <div>The team key for {list(note.teamKeysUpdated)} was replaced with the one in this set.</div>}
        {note.teamEndpointsShared.length > 0 && <div>The key for {note.teamEndpointsShared.join(', ')} is shared — used only for that endpoint.</div>}
        {note.agentsGivenKey > 0 && <div>{note.agentsGivenKey === 1 ? '1 agent' : `${note.agentsGivenKey} agents`} without a model key got one; running ones restart.</div>}
        {note.teamKeysConflict.length > 0 && <div>This set holds more than one key for {list(note.teamKeysConflict)} — teammates' agents use just one.</div>}
        {note.teamKeysMissing.length > 0 && <div>No team key for {list(note.teamKeysMissing)} yet — add one under Team API keys below.</div>}
      </div>
    </Notice>
  )
}

// ── A tier ──

function TierCard({ tierKey, def, tc, onEdit }: { tierKey: string; def?: TierDef; tc: TierConfig; onEdit: () => void }) {
  const { effort } = splitReasoningEffort(tc.model)
  const reasoning = REASONING_EFFORT_PROVIDERS.includes(tc.provider) && effort
  return (
    <button
      onClick={onEdit}
      className="group flex flex-col rounded-xl border border-zinc-800 bg-zinc-950/50 p-4 text-left transition-colors hover:border-violet-500/40 hover:bg-zinc-900/60"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-semibold text-zinc-100">{def?.label || tierKey}</span>
        <Pencil className="h-3.5 w-3.5 text-zinc-600 transition-colors group-hover:text-violet-500" />
      </div>
      {def?.use && <p className="mt-0.5 line-clamp-1 text-xs text-zinc-500">{def.use}</p>}
      <div className="mt-3 flex min-w-0 items-center gap-2">
        <Pill tone={PROVIDER_TONE[tc.provider] || 'zinc'}>{providerName(tc.provider)}</Pill>
        {tc.model
          ? <span className="truncate font-mono text-[13px] text-zinc-200" title={tc.model}>{tc.model}</span>
          : <span className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400"><AlertTriangle className="h-3.5 w-3.5" /> No model</span>}
      </div>
      <div className="mt-2.5 flex flex-wrap gap-1">
        <Pill title="Input context">{fmtCtxTokens(tc.input_ctx)} in</Pill>
        <Pill title="Output context">{fmtCtxTokens(tc.output_ctx)} out</Pill>
        {reasoning && <Pill icon={Brain}>{effort}</Pill>}
        {tc.vision && <Pill tone="sky" icon={ImageIcon}>sees images</Pill>}
        {tc.weak && <Pill tone="amber" icon={Zap}>fast model</Pill>}
        {tc.base_url?.trim() && <Pill icon={Server} title={tc.base_url}>own endpoint</Pill>}
        {tc.api_key?.trim() && <Pill icon={KeyRound}>{tc.api_key === '@system' ? 'team key' : 'own key'}</Pill>}
      </div>
    </button>
  )
}

function SectionLabel({ children }: { children: string }) {
  return <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">{children}</h3>
}

function TierSheet({ tierKey, def, tc, onChange, onUseEverywhere, onClose }: {
  tierKey: string
  def?: TierDef
  tc: TierConfig
  onChange: (patch: Partial<TierConfig>) => void
  onUseEverywhere: () => void
  onClose: () => void
}) {
  const [showKey, setShowKey] = useState(false)
  const effortable = REASONING_EFFORT_PROVIDERS.includes(tc.provider)
  const effort = splitReasoningEffort(tc.model).effort
  const cli = tc.provider === 'antigravity-cli'

  return (
    <Sheet
      title={def?.label || tierKey}
      subtitle={def?.use || 'Edits save as you type.'}
      icon={Gauge}
      onClose={onClose}
      footer={<>
        <Button variant="ghost" icon={Copy} onClick={onUseEverywhere} className="mr-auto">Use for every tier</Button>
        <Button variant="primary" onClick={onClose}>Done</Button>
      </>}
    >
      <div className="space-y-7">
        <section className="space-y-4">
          <SectionLabel>Model</SectionLabel>
          <Field label="Provider">
            <select
              value={tc.provider}
              onChange={(e) => onChange(e.target.value === 'antigravity-cli'
                ? { provider: e.target.value, api_key: '', base_url: '' }
                : { provider: e.target.value })}
              className={inputCls}
            >
              {!LLM_PROVIDERS.some((p) => p.value === tc.provider) && <option value={tc.provider}>{tc.provider || '—'}</option>}
              {LLM_PROVIDERS.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
            </select>
          </Field>
          <Field label="Model" hint="The provider's model id, as its docs spell it.">
            <input
              value={tc.model}
              onChange={(e) => onChange({ model: e.target.value })}
              placeholder="e.g. claude-sonnet-4-5, gpt-5, qwen3:32b"
              className={`${inputCls} font-mono`}
              autoFocus
            />
          </Field>
          {effortable && (
            <Field label="Reasoning effort" hint="How hard the model thinks before answering. “none” turns it off (some models need that to use tools).">
              <ChipPicker
                value={effort}
                options={['', ...REASONING_EFFORTS]}
                format={(v) => v || 'default'}
                onChange={(v) => onChange({ model: withReasoningEffort(tc.model, v) })}
              />
            </Field>
          )}
        </section>

        <section className="space-y-4">
          <SectionLabel>Context window</SectionLabel>
          <Field label="Reads up to" hint="The most the model takes in at once (tokens).">
            <ChipPicker value={tc.input_ctx} options={INPUT_CTX_OPTIONS} format={fmtCtxTokens} onChange={(n) => onChange({ input_ctx: n })} />
          </Field>
          <Field label="Writes up to" hint="The longest single reply (tokens).">
            <ChipPicker value={tc.output_ctx} options={OUTPUT_CTX_OPTIONS} format={fmtCtxTokens} onChange={(n) => onChange({ output_ctx: n })} />
          </Field>
        </section>

        <section className="space-y-1">
          <SectionLabel>Abilities</SectionLabel>
          <div className="divide-y divide-zinc-800/80">
            <ToggleRow
              title="Sees images"
              hint="The model is multimodal: agents on this tier look at pictures themselves. Needed for local builds (e.g. Qwen); GPT, Claude and Gemini are recognised anyway."
              on={!!tc.vision}
              onToggle={() => onChange({ vision: !tc.vision })}
            />
            <ToggleRow
              title="Fast / small model"
              hint="A small or fast model (a Flash tier, a local model): multi-agent runs lean harder on their safety checks for it."
              on={!!tc.weak}
              onToggle={() => onChange({ weak: !tc.weak })}
            />
          </div>
        </section>

        <section className="space-y-4">
          <SectionLabel>Connection</SectionLabel>
          {cli ? (
            <p className="text-sm leading-relaxed text-zinc-500">
              Runs through the Antigravity CLI with a Google sign-in on this machine: text only, no API key or endpoint.
            </p>
          ) : (
            <>
              <Field
                label="API key"
                hint={tc.api_key === '@system'
                  ? 'Uses the team key for this provider (Team API keys, below).'
                  : 'Optional. Blank uses the team key for this provider, or the server\'s.'}
              >
                <div className="relative">
                  <input
                    type={showKey ? 'text' : 'password'}
                    value={tc.api_key}
                    onChange={(e) => onChange({ api_key: e.target.value })}
                    placeholder="sk-…"
                    autoComplete="off"
                    className={`${inputCls} pr-10 font-mono`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowKey(!showKey)}
                    aria-label={showKey ? 'Hide key' : 'Show key'}
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-zinc-500 hover:text-zinc-200"
                  >
                    {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </Field>
              <Field label="Endpoint (base URL)" hint="Only for a gateway or a local server (Ollama, LM Studio, vLLM…). Blank is the provider's own.">
                <input
                  value={tc.base_url}
                  onChange={(e) => onChange({ base_url: e.target.value })}
                  placeholder="https://… or http://localhost:11434"
                  className={`${inputCls} font-mono`}
                />
              </Field>
            </>
          )}
        </section>
      </div>
    </Sheet>
  )
}

function ToggleRow({ title, hint, on, onToggle }: { title: string; hint: string; on: boolean; onToggle: () => void }) {
  return (
    <div className="flex items-start justify-between gap-4 py-3">
      <div className="min-w-0">
        <div className="text-sm font-medium text-zinc-200">{title}</div>
        <div className="mt-0.5 text-xs leading-relaxed text-zinc-500">{hint}</div>
      </div>
      <Switch on={on} onToggle={onToggle} label={title} />
    </div>
  )
}

// ── New set / one model everywhere ──

type OneModel = Pick<TierConfig, 'provider' | 'model' | 'api_key' | 'base_url' | 'input_ctx' | 'output_ctx'>

function OneModelForm({ initial, submitLabel, onSubmit, name }: {
  initial?: TierConfig
  submitLabel: string
  onSubmit: (apply: OneModel) => void
  // A new set's name, when the form creates one.
  name?: { value: string; onChange: (v: string) => void }
}) {
  const [provider, setProvider] = useState(initial?.provider || 'anthropic')
  const [model, setModel] = useState(initial?.model || '')
  const [apiKey, setApiKey] = useState(initial?.api_key || '')
  const [baseUrl, setBaseUrl] = useState(initial?.base_url || '')
  const [inputCtx, setInputCtx] = useState(initial?.input_ctx || 262144)
  const [outputCtx, setOutputCtx] = useState(initial?.output_ctx || 32768)
  const cli = provider === 'antigravity-cli'
  const valid = model.trim().length > 0 && (!name || name.value.trim().length > 0)

  return (
    <form
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault()
        if (!valid) return
        onSubmit({
          provider, model: model.trim(), api_key: cli ? '' : apiKey, base_url: cli ? '' : baseUrl.trim(),
          input_ctx: inputCtx, output_ctx: outputCtx,
        })
      }}
    >
      {name && (
        <Field label="Set name">
          <input value={name.value} onChange={(e) => name.onChange(e.target.value)} placeholder="e.g. All Anthropic, Local Ollama" className={inputCls} />
        </Field>
      )}
      <div className="grid grid-cols-1 gap-4 wide:grid-cols-2">
        <Field label="Provider">
          <select value={provider} onChange={(e) => setProvider(e.target.value)} className={inputCls}>
            {LLM_PROVIDERS.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
          </select>
        </Field>
        <Field label="Model">
          <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="model id" className={`${inputCls} font-mono`} autoFocus={!name} />
        </Field>
      </div>
      {!cli && (
        <div className="grid grid-cols-1 gap-4 wide:grid-cols-2">
          <Field label="API key" hint="Optional — blank uses the team key.">
            <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-…" autoComplete="off" className={`${inputCls} font-mono`} />
          </Field>
          <Field label="Endpoint" hint="Optional — a gateway or local server.">
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="default" className={`${inputCls} font-mono`} />
          </Field>
        </div>
      )}
      <Field label="Reads up to">
        <ChipPicker value={inputCtx} options={INPUT_CTX_OPTIONS} format={fmtCtxTokens} onChange={setInputCtx} />
      </Field>
      <Field label="Writes up to">
        <ChipPicker value={outputCtx} options={OUTPUT_CTX_OPTIONS} format={fmtCtxTokens} onChange={setOutputCtx} />
      </Field>
      <div className="flex justify-end pt-1">
        <button
          type="submit"
          disabled={!valid}
          className="inline-flex items-center gap-1.5 rounded-xl bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50"
        >
          <Check className="h-4 w-4" /> {submitLabel}
        </button>
      </div>
    </form>
  )
}

type NewKind = 'quick' | 'one' | 'defaults' | 'copy'

function NewSetDialog({ copyName, catalog, tierLabel, existingNames, onClose, onQuick, onOneModel, onDefaults, onCopy }: {
  copyName: string
  // Quick starts; null while they load.
  catalog: TierPresetCatalog | null
  tierLabel: (tier: string) => string
  existingNames: string[]
  onClose: () => void
  onQuick: (name: string, perTier: Record<string, Partial<TierConfig>>) => void
  onOneModel: (name: string, apply: OneModel) => void
  onDefaults: (name: string) => void
  onCopy: (name: string) => void
}) {
  // No quick starts on this deck: start on One model.
  const noQuick = catalog !== null && catalog.presets.length === 0
  const [picked, setKind] = useState<NewKind>('quick')
  const kind: NewKind = picked === 'quick' && noQuick ? 'one' : picked
  const [name, setName] = useState('')
  const options: { value: NewKind; label: string }[] = [
    ...(noQuick ? [] : [{ value: 'quick' as const, label: 'Quick start' }]),
    { value: 'one', label: 'One model' },
    { value: 'defaults', label: 'Built-in defaults' },
    ...(copyName ? [{ value: 'copy' as const, label: 'Copy current' }] : []),
  ]
  const fallbackName = kind === 'copy' ? `${copyName} copy` : 'New set'

  return (
    <Dialog title="New tier set" icon={Layers} onClose={onClose}>
      <div className="space-y-5">
        <Segmented value={kind} options={options} onChange={setKind} label="How to start" />
        {kind === 'quick' ? (
          <>
            <p className="text-xs leading-relaxed text-zinc-500">
              Pick your AI provider and paste its API key. Every tier gets that provider's current models, and the set
              is named after it.
            </p>
            {catalog
              ? <QuickStartForm presets={catalog.presets} checked={catalog.checked} tierLabel={tierLabel} existingNames={existingNames} onCreate={onQuick} />
              : <p className="text-sm text-zinc-500">Loading the providers…</p>}
          </>
        ) : kind === 'one' ? (
          <>
            <p className="text-xs leading-relaxed text-zinc-500">One model powers every tier. Fine-tune each tier afterwards.</p>
            <OneModelForm
              name={{ value: name, onChange: setName }}
              submitLabel="Create tier set"
              onSubmit={(apply) => onOneModel(name.trim(), apply)}
            />
          </>
        ) : (
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault()
              const n = name.trim() || fallbackName
              if (kind === 'copy') onCopy(n)
              else onDefaults(n)
            }}
          >
            <p className="text-xs leading-relaxed text-zinc-500">
              {kind === 'copy'
                ? `A copy of “${copyName}”, keys included, to change without touching the original.`
                : 'Each tier starts on the model Captain Claw suggests for it; add keys and change models after.'}
            </p>
            <Field label="Set name">
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder={fallbackName} className={inputCls} autoFocus />
            </Field>
            <div className="flex justify-end">
              <button type="submit" className="inline-flex items-center gap-1.5 rounded-xl bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500">
                <Check className="h-4 w-4" /> Create tier set
              </button>
            </div>
          </form>
        )}
      </div>
    </Dialog>
  )
}

// ── Extra keys of a set (env vars every agent it spawns gets) ──

// The rows' columns: name, value, remove. A phone stacks the value under the name.
const KEY_ROW_COLS = 'grid-cols-[minmax(0,1fr)_2.25rem] sm:grid-cols-[minmax(0,2fr)_minmax(0,3fr)_2.25rem]'

function ExtraKeys({ envVars, setEnvVars }: { envVars: EnvVar[]; setEnvVars: React.Dispatch<React.SetStateAction<EnvVar[]>> }) {
  // Rows whose value is shown in the clear (by index; a removal hides them all).
  const [shown, setShown] = useState<Set<number>>(new Set())
  const set = (i: number, patch: Partial<EnvVar>) => setEnvVars((prev) => prev.map((v, j) => (j === i ? { ...v, ...patch } : v)))
  const remove = (i: number) => {
    setEnvVars((prev) => prev.filter((_, j) => j !== i))
    setShown(new Set())
  }
  const toggle = (i: number) => setShown((prev) => {
    const next = new Set(prev)
    if (next.has(i)) next.delete(i)
    else next.add(i)
    return next
  })
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="flex items-center gap-1.5 text-xs font-medium text-zinc-400">
          <KeyRound className="h-3.5 w-3.5" /> Extra keys for agents
          {envVars.length > 0 && <span className="text-zinc-600">({envVars.length})</span>}
        </span>
        <span className="text-[11px] leading-relaxed text-zinc-500">
          Other services' keys — web search and the like — that every agent this set starts gets as environment variables.
        </span>
      </div>
      {envVars.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-zinc-800">
          <div className={`hidden ${KEY_ROW_COLS} gap-2 border-b border-zinc-800 bg-zinc-900/50 px-3 py-1.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500 sm:grid`}>
            <span>Name</span>
            <span>Value</span>
            <span />
          </div>
          <ul className="divide-y divide-zinc-800">
            {envVars.map((ev, i) => (
              <li key={i} className={`grid ${KEY_ROW_COLS} items-center gap-2 px-3 py-2`}>
                <input
                  value={ev.key}
                  onChange={(e) => set(i, { key: e.target.value })}
                  placeholder="BRAVE_API_KEY"
                  aria-label="Key name"
                  spellCheck={false}
                  className={`${inputCls} font-mono text-xs`}
                />
                <div className="relative order-last col-span-2 sm:order-none sm:col-span-1">
                  <input
                    type={shown.has(i) ? 'text' : 'password'}
                    value={ev.value}
                    onChange={(e) => set(i, { value: e.target.value })}
                    placeholder="Value"
                    aria-label={`Value of ${ev.key || 'this key'}`}
                    autoComplete="off"
                    spellCheck={false}
                    className={`${inputCls} pr-9 font-mono text-xs`}
                  />
                  <button
                    type="button"
                    onClick={() => toggle(i)}
                    aria-label={shown.has(i) ? 'Hide the value' : 'Show the value'}
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-zinc-500 hover:text-zinc-200"
                  >
                    {shown.has(i) ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                  </button>
                </div>
                <button
                  type="button"
                  onClick={() => remove(i)}
                  aria-label={`Remove ${ev.key || 'this key'}`}
                  className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-red-500"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      <button
        type="button"
        onClick={() => setEnvVars((prev) => [...prev, { key: '', value: '' }])}
        className={`${envVars.length > 0 ? 'mt-2' : ''} inline-flex items-center gap-1.5 rounded-lg border border-dashed border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:border-zinc-500 hover:text-zinc-200`}
      >
        <Plus className="h-3.5 w-3.5" /> Add a key
      </button>
    </div>
  )
}

// ── Team API keys (fd:provider-keys) ──

function TeamKeysCard({ keys, onSaved, needsKey, team }: {
  keys: Record<string, string>
  onSaved: (keys: Record<string, string>) => void
  needsKey: Set<string>
  // A deck with accounts: these are the whole team's keys.
  team: boolean
}) {
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const rows = [
    ...TEAM_KEY_PROVIDERS,
    ...Object.keys(keys).filter((k) => !TEAM_KEY_PROVIDERS.some((p) => p.value === k)).map((k) => ({ value: k, label: providerName(k) })),
  ]

  const save = async (provider: string, value: string) => {
    setBusy(provider); setError('')
    try {
      onSaved(await saveProviderKey(provider, value))
      setEditing(null); setDraft('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  return (
    <Card
      icon={KeyRound}
      title={team ? 'Team API keys' : 'API keys'}
      hint={team
        ? 'One key per provider, used by everyone\'s agents that run on that provider without a key of their own. Publishing a set fills these in from its keys. Nobody else sees them.'
        : 'One key per provider, used by every agent on that provider that has no key of its own.'}
      bodyClassName="divide-y divide-zinc-800/80"
    >
      {rows.map((p) => {
        const value = keys[p.value] || ''
        const open = editing === p.value
        return (
          <div key={p.value} className="px-5 py-3">
            <div className="flex flex-wrap items-center gap-3">
              <span className="min-w-[8rem] flex-1 text-sm font-medium text-zinc-200">{p.label}</span>
              {value
                ? <Pill tone="emerald" icon={Check}>Set · {keyHint(value)}</Pill>
                : needsKey.has(p.value)
                  ? <Pill tone="amber" icon={AlertTriangle} title="The team's models use this provider">Needed by the team's models</Pill>
                  : <Pill>Not set</Pill>}
              {!open && (
                <div className="flex gap-1.5">
                  <Button onClick={() => { setEditing(p.value); setDraft(''); setError('') }}>{value ? 'Replace' : 'Add key'}</Button>
                  {value && (
                    <Button
                      variant="danger"
                      busy={busy === p.value}
                      onClick={() => { if (confirm(`Remove the team's ${p.label} key? Agents that rely on it lose access to ${p.label}.`)) save(p.value, '') }}
                    >
                      Remove
                    </Button>
                  )}
                </div>
              )}
            </div>
            {open && (
              <form
                className="mt-3 flex flex-col gap-2 wide:flex-row"
                onSubmit={(e) => { e.preventDefault(); if (draft.trim()) save(p.value, draft.trim()) }}
              >
                <input
                  type="password"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  placeholder={`${p.label} API key`}
                  autoComplete="off"
                  autoFocus
                  className={`${inputCls} flex-1 font-mono`}
                />
                <div className="flex gap-2">
                  <Button type="submit" variant="primary" busy={busy === p.value} disabled={!draft.trim()}>Save</Button>
                  <Button variant="ghost" onClick={() => setEditing(null)}>Cancel</Button>
                </div>
              </form>
            )}
          </div>
        )
      })}
      {error && <div className="px-5 py-3"><Notice tone="red">{error}</Notice></div>}
    </Card>
  )
}
