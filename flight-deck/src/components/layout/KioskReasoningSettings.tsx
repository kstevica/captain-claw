import { useEffect, useState } from 'react'
import { AlertTriangle, Brain, Check, Loader2, Waypoints } from 'lucide-react'
import { getAgentReasoning, setAgentReasoning, type AgentReasoning, type AgentReasoningSettings } from '../../services/docker'
import {
  LEVEL_HINT, LEVEL_LABEL, clampLevel, parseProfile, sortLevels,
  type ReasoningLevel, type ReasoningProfile,
} from '../../utils/reasoningEffort'

// ── Settings: reasoning effort by where a message comes from ──
//
// The web chat has its own picker in the chat box. Everything else — WhatsApp,
// Telegram, the app, glasses, a device, a scheduled task, another agent —
// takes these: one default, and a level per source where it should differ.
// The agent reads them afresh, so a change applies from its next message.

const AUTO = 'auto'
const INHERIT = ''

/** A stored value as the person reads it on this model. */
function levelText(value: string, profile: ReasoningProfile | null): string {
  if (value === AUTO || !value) return profile ? `Model default · ${LEVEL_LABEL[profile.default]}` : 'Model default'
  const lvl = value as ReasoningLevel
  if (!profile || profile.levels.includes(lvl)) return LEVEL_LABEL[lvl] ?? value
  return `${LEVEL_LABEL[lvl] ?? value} → ${LEVEL_LABEL[clampLevel(lvl, profile)]} on this model`
}

/** The level a source set to Default runs at. */
function defaultLevel(value: string, profile: ReasoningProfile): string {
  if (value === AUTO || !value) return LEVEL_LABEL[profile.default]
  return LEVEL_LABEL[clampLevel(value as ReasoningLevel, profile)]
}

export function KioskReasoningSettings({ kind, agentKey }: { kind: 'process' | 'docker'; agentKey: string }) {
  const [data, setData] = useState<AgentReasoning | null>(null)
  const [settings, setSettings] = useState<AgentReasoningSettings>({ channels: {} })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState('')
  const [saved, setSaved] = useState('')

  useEffect(() => {
    let live = true
    getAgentReasoning(kind, agentKey)
      .then((r) => { if (live) { setData(r); setSettings(r.settings || { channels: {} }) } })
      .catch((e) => { if (live) setError(`Couldn't load its reasoning settings: ${e instanceof Error ? e.message : String(e)}`) })
      .finally(() => { if (live) setLoading(false) })
    return () => { live = false }
  }, [kind, agentKey])

  const profile = parseProfile(data?.profile)
  const levels = profile ? sortLevels(profile.levels) : []

  const save = async (key: string, change: Parameters<typeof setAgentReasoning>[2], optimistic: AgentReasoningSettings) => {
    const was = settings
    setSettings(optimistic); setSaving(key); setError(''); setSaved('')
    try {
      const res = await setAgentReasoning(kind, agentKey, change)
      setSettings(res.settings)
      setSaved(key)
      setTimeout(() => setSaved(''), 2000)
    } catch (e) {
      setSettings(was)
      setError(`Couldn't save: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setSaving('')
    }
  }

  const pickDefault = (value: string) => {
    const next = { ...settings }
    if (value === AUTO) delete next.default
    else next.default = value
    void save('default', value === AUTO ? { clear_default: true } : { default: value }, next)
  }

  const pickSource = (source: string, value: string) => {
    const channels = { ...settings.channels }
    if (value === INHERIT) delete channels[source]
    else channels[source] = value
    void save(source, { channels: { [source]: value === INHERIT ? null : value } }, { ...settings, channels })
  }

  const defaultValue = settings.default || AUTO
  const defaultChoices = profile ? [
    { value: AUTO, title: 'Model default', hint: `${LEVEL_LABEL[profile.default]} on this model` },
    ...levels.map((l) => ({ value: l as string, title: LEVEL_LABEL[l], hint: LEVEL_HINT[l] })),
    // Picked on another model, and not one of this one's levels.
    ...(defaultValue !== AUTO && !levels.includes(defaultValue as ReasoningLevel)
      ? [{ value: defaultValue, title: LEVEL_LABEL[defaultValue as ReasoningLevel] ?? defaultValue, hint: `${defaultLevel(defaultValue, profile)} on this model` }] : []),
  ] : []

  return (
    <div className="space-y-6">
      <section>
        <SectionTitle icon={Brain} title="Default level" saving={saving === 'default'} saved={saved === 'default'} />
        <p className="mb-2.5 text-[11px] leading-relaxed text-zinc-500">
          How hard it thinks before answering messages that don't come from the web chat — the chat box has its own
          picker. Applies from its next message.
        </p>

        {loading ? (
          <div className="flex items-center py-4 text-xs text-zinc-500"><Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" /> Loading…</div>
        ) : !profile ? (
          <p className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2.5 text-xs text-zinc-400">
            {data?.model.model
              ? <>Its model ({data.model.provider} / {data.model.model}) has no reasoning levels to choose from.</>
              : <>Its model has no reasoning levels to choose from.</>}
          </p>
        ) : (
          <>
            <div role="radiogroup" aria-label="Default reasoning effort" className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 lg:grid-cols-4">
              {defaultChoices.map((o) => {
                const on = o.value === defaultValue
                return (
                  <button
                    key={o.value}
                    role="radio"
                    aria-checked={on}
                    onClick={() => { if (!on) pickDefault(o.value) }}
                    disabled={!!saving}
                    className={`flex min-w-0 items-center gap-2.5 rounded-lg border px-3 py-2 text-left transition-colors disabled:cursor-wait ${
                      on ? 'border-violet-500/50 bg-violet-500/10' : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900'
                    }`}
                  >
                    <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full border ${on ? 'border-violet-400 bg-violet-500' : 'border-zinc-600'}`}>
                      {on && <span className="h-1.5 w-1.5 rounded-full bg-white" />}
                    </span>
                    <span className="min-w-0">
                      <span className="block text-xs font-medium text-zinc-200">{o.title}</span>
                      <span className="block text-[10px] leading-snug text-zinc-500">{o.hint}</span>
                    </span>
                  </button>
                )
              })}
            </div>
            {profile.note && <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">{profile.note}</p>}
          </>
        )}
      </section>

      {!loading && profile && (
        <section>
          <SectionTitle icon={Waypoints} title="By source" saving={!!saving && saving !== 'default'} saved={!!saved && saved !== 'default'} />
          <p className="mb-2.5 text-[11px] leading-relaxed text-zinc-500">
            A level of its own for a source that should differ from the default.
          </p>
          <div className="lg:columns-2 lg:gap-8">
            {(data?.catalog ?? []).map((group) => (
              <div key={group.title} className="mb-3 break-inside-avoid last:mb-0">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">{group.title}</div>
                <div className="divide-y divide-zinc-800/70 rounded-lg border border-zinc-800">
                  {group.sources.map((src) => {
                    const value = settings.channels[src.key] ?? INHERIT
                    return (
                      <div key={src.key} className="px-3 py-1.5">
                        <LevelRow
                          label={src.label}
                          value={value}
                          onChange={(v) => pickSource(src.key, v)}
                          disabled={!!saving}
                          options={[
                            { value: INHERIT, text: `Default · ${defaultLevel(defaultValue, profile)}` },
                            { value: AUTO, text: levelText(AUTO, profile) },
                            ...levels.map((l) => ({ value: l, text: LEVEL_LABEL[l] })),
                            ...(value && value !== AUTO && !levels.includes(value as ReasoningLevel)
                              ? [{ value, text: levelText(value, profile) }] : []),
                          ]}
                        />
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {error}
        </div>
      )}
    </div>
  )
}

function SectionTitle({ icon: Icon, title, saving, saved }: { icon: typeof Brain; title: string; saving: boolean; saved: boolean }) {
  return (
    <h3 className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-400">
      <Icon className="h-3.5 w-3.5" />
      {title}
      {saving && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
      {saved && <span className="flex items-center gap-0.5 normal-case tracking-normal text-emerald-600 dark:text-emerald-400"><Check className="h-3 w-3" />saved</span>}
    </h3>
  )
}

function LevelRow({ label, value, options, onChange, disabled }: {
  label: string
  value: string
  options: { value: string; text: string }[]
  onChange: (value: string) => void
  disabled?: boolean
}) {
  return (
    <label className="flex items-center gap-3">
      <span className="min-w-0 flex-1 text-xs leading-snug text-zinc-200">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-40 shrink-0 rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none disabled:cursor-wait wide:w-52"
      >
        {options.map((o) => <option key={o.value || 'inherit'} value={o.value}>{o.text}</option>)}
      </select>
    </label>
  )
}
