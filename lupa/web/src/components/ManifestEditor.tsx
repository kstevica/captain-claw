import { useMemo, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Braces, LayoutGrid, Plus, X } from 'lucide-react'

type Manifest = Record<string, unknown>

interface Props {
  value: Manifest
  onChange: (m: Manifest) => void
}

// ── immutable helpers ────────────────────────────────────────────────

function set(obj: Manifest, key: string, val: unknown): Manifest {
  return { ...obj, [key]: val }
}
function setIn(obj: Manifest, k1: string, k2: string, val: unknown): Manifest {
  const inner = { ...((obj[k1] as Record<string, unknown>) ?? {}), [k2]: val }
  return { ...obj, [k1]: inner }
}

const THEME_KEYS = ['accent', 'accent_soft', 'bg', 'surface', 'border', 'text', 'text_dim']
const HEX = /^#[0-9a-fA-F]{6}$/

// ── small field primitives ───────────────────────────────────────────

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-[11px] uppercase tracking-wide text-[var(--lp-text-dim)]">{label}</span>
      <div className="mt-0.5">{children}</div>
    </label>
  )
}

const inputCls =
  'w-full rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2 text-sm outline-none focus:border-[var(--lp-accent)]'

function Text({ value, onChange, placeholder }:
  { value: string; onChange: (v: string) => void; placeholder?: string }) {
  return <input value={value} placeholder={placeholder}
                onChange={(e) => onChange(e.target.value)} className={inputCls} />
}

function Section({ title, hint, children }:
  { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-[var(--lp-border)] bg-[var(--lp-bg)] p-3 space-y-2.5">
      <div>
        <div className="text-sm font-semibold">{title}</div>
        {hint && <div className="text-xs text-[var(--lp-text-dim)]">{hint}</div>}
      </div>
      {children}
    </div>
  )
}

// ── the editor ───────────────────────────────────────────────────────

export default function ManifestEditor({ value, onChange }: Props) {
  const [mode, setMode] = useState<'visual' | 'json'>('visual')
  const [rawText, setRawText] = useState('')
  const [rawError, setRawError] = useState('')

  const theme = (value.theme as Record<string, string>) ?? {}
  const vocab = (value.vocabulary as Record<string, string>) ?? {}
  const roi = (value.roi as Record<string, unknown>) ?? {}
  const quality = (value.quality as Record<string, unknown>) ?? {}
  const briefs = ((value.briefs as { presets?: { id: string; label: string; hours: number }[] })?.presets) ?? []
  const intakeTypes = ((value.intake as { types?: Record<string, unknown>[] })?.types) ?? []
  const evals = (value.evals as { brief?: string }[]) ?? []

  const enterJson = () => {
    setRawText(JSON.stringify(value, null, 2)); setRawError(''); setMode('json')
  }
  const applyJson = (text: string) => {
    setRawText(text)
    try { onChange(JSON.parse(text)); setRawError('') }
    catch { setRawError('Not valid JSON — fix to apply.') }
  }

  const preview = useMemo(() => ({
    background: theme.surface || 'var(--lp-surface)',
    borderColor: theme.border || 'var(--lp-border)',
    color: theme.text || 'var(--lp-text)',
  }), [theme])

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="flex rounded-lg border border-[var(--lp-border)] overflow-hidden text-xs">
          <button onClick={() => setMode('visual')}
                  className={`px-2.5 py-1 flex items-center gap-1 ${mode === 'visual'
                    ? 'bg-[var(--lp-border)] text-[var(--lp-text)]' : 'text-[var(--lp-text-dim)]'}`}>
            <LayoutGrid size={12} /> Visual
          </button>
          <button onClick={enterJson}
                  className={`px-2.5 py-1 flex items-center gap-1 ${mode === 'json'
                    ? 'bg-[var(--lp-border)] text-[var(--lp-text)]' : 'text-[var(--lp-text-dim)]'}`}>
            <Braces size={12} /> JSON
          </button>
        </div>
      </div>

      {mode === 'json' ? (
        <>
          <textarea value={rawText} onChange={(e) => applyJson(e.target.value)}
                    rows={18} spellCheck={false}
                    className="w-full rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2 text-xs font-mono outline-none focus:border-[var(--lp-accent)] resize-y" />
          {rawError && <div className="text-sm text-red-400">{rawError}</div>}
        </>
      ) : (
        <div className="space-y-3">
          {/* Identity */}
          <Section title="Identity">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <Field label="Desk name">
                <Text value={String(value.name ?? '')}
                      onChange={(x) => onChange(set(value, 'name', x))} />
              </Field>
              <Field label="Tagline">
                <Text value={String(value.tagline ?? '')}
                      onChange={(x) => onChange(set(value, 'tagline', x))} />
              </Field>
            </div>
          </Section>

          {/* Theme */}
          <Section title="Theme" hint="The desk's colors — the swatch previews them.">
            <div className="flex gap-3 items-start flex-wrap">
              <div className="grid grid-cols-2 gap-2 flex-1 min-w-64">
                {THEME_KEYS.map((k) => {
                  const val = theme[k] ?? ''
                  const valid = HEX.test(val)
                  return (
                    <div key={k} className="flex items-center gap-2">
                      <input type="color" value={valid ? val : '#000000'}
                             onChange={(e) => onChange(setIn(value, 'theme', k, e.target.value))}
                             className="w-7 h-7 rounded border border-[var(--lp-border)] bg-transparent cursor-pointer shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="text-[10px] text-[var(--lp-text-dim)]">{k}</div>
                        <input value={val}
                               onChange={(e) => onChange(setIn(value, 'theme', k, e.target.value))}
                               className="w-full rounded bg-[var(--lp-bg)] border border-[var(--lp-border)] px-1.5 py-0.5 text-xs font-mono outline-none focus:border-[var(--lp-accent)]" />
                      </div>
                    </div>
                  )
                })}
              </div>
              <div className="rounded-lg border p-3 w-44" style={preview}>
                <div className="text-sm font-bold" style={{ color: theme.accent }}>
                  {String(value.name ?? 'Desk')}
                </div>
                <div className="text-xs" style={{ color: theme.text_dim }}>
                  {String(value.tagline ?? 'preview')}
                </div>
                <div className="mt-2 text-xs px-2 py-1 rounded text-center font-semibold"
                     style={{ background: theme.accent, color: theme.bg }}>
                  Commission
                </div>
              </div>
            </div>
          </Section>

          {/* Vocabulary */}
          <Section title="Vocabulary" hint="Every user-facing label. Keys are fixed; edit the values.">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-3 gap-y-1.5">
              {Object.keys(vocab).sort().map((k) => (
                <div key={k} className="flex items-center gap-2">
                  <span className="text-[11px] text-[var(--lp-text-dim)] w-32 shrink-0 truncate" title={k}>{k}</span>
                  <input value={vocab[k] ?? ''}
                         onChange={(e) => onChange(setIn(value, 'vocabulary', k, e.target.value))}
                         className="flex-1 rounded bg-[var(--lp-bg)] border border-[var(--lp-border)] px-2 py-1 text-xs outline-none focus:border-[var(--lp-accent)]" />
                </div>
              ))}
            </div>
          </Section>

          {/* Intake + quality + ROI */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Section title="Run defaults">
              <Field label="Quality profile">
                <select value={String(quality.profile ?? 'thorough')}
                        onChange={(e) => onChange(set(value, 'quality', { ...quality, profile: e.target.value }))}
                        className={inputCls}>
                  <option value="thorough">Thorough</option>
                  <option value="balanced">Balanced</option>
                  <option value="off">Off</option>
                </select>
              </Field>
              {intakeTypes[0] && (
                <Field label="Default max agents">
                  <input type="number" min={1} max={10}
                         value={Number(intakeTypes[0].default_max_agents ?? 6)}
                         onChange={(e) => {
                           const types = intakeTypes.map((t, i) =>
                             i === 0 ? { ...t, default_max_agents: Number(e.target.value) } : t)
                           onChange(set(value, 'intake', { ...(value.intake as object), types }))
                         }}
                         className={inputCls} />
                </Field>
              )}
            </Section>
            <Section title="ROI framing" hint="The “≈N× below …” chip on every report.">
              <Field label="Analyst hourly rate (USD)">
                <input type="number" min={0} value={Number(roi.analyst_hourly_usd ?? 60)}
                       onChange={(e) => onChange(set(value, 'roi', { ...roi, analyst_hourly_usd: Number(e.target.value) }))}
                       className={inputCls} />
              </Field>
              <Field label="Comparison label">
                <Text value={String(roi.analyst_label ?? 'a human analyst')}
                      onChange={(x) => onChange(set(value, 'roi', { ...roi, analyst_label: x }))} />
              </Field>
            </Section>
          </div>

          {/* Standing-brief cadences */}
          <Section title="Standing-brief cadences" hint="The schedule options offered on a stream.">
            <div className="space-y-1.5">
              {briefs.map((p, i) => (
                <div key={i} className="flex items-center gap-2">
                  <input value={p.label ?? ''} placeholder="label"
                         onChange={(e) => {
                           const next = briefs.map((x, j) => j === i ? { ...x, label: e.target.value } : x)
                           onChange(set(value, 'briefs', { presets: next }))
                         }}
                         className="flex-1 rounded bg-[var(--lp-bg)] border border-[var(--lp-border)] px-2 py-1 text-xs outline-none focus:border-[var(--lp-accent)]" />
                  <input type="number" min={1} value={Number(p.hours ?? 24)}
                         onChange={(e) => {
                           const next = briefs.map((x, j) => j === i ? { ...x, hours: Number(e.target.value) } : x)
                           onChange(set(value, 'briefs', { presets: next }))
                         }}
                         className="w-20 rounded bg-[var(--lp-bg)] border border-[var(--lp-border)] px-2 py-1 text-xs outline-none focus:border-[var(--lp-accent)]" />
                  <span className="text-[11px] text-[var(--lp-text-dim)]">hrs</span>
                  <button onClick={() => onChange(set(value, 'briefs', { presets: briefs.filter((_, j) => j !== i) }))}
                          className="p-1 text-[var(--lp-text-dim)] hover:text-red-400"><X size={13} /></button>
                </div>
              ))}
              <button onClick={() => onChange(set(value, 'briefs',
                { presets: [...briefs, { id: `c${briefs.length}`, label: 'New', hours: 24 }] }))}
                      className="text-xs flex items-center gap-1 text-[var(--lp-text-dim)] hover:text-[var(--lp-accent)]">
                <Plus size={12} /> add cadence
              </button>
            </div>
          </Section>

          {/* Golden task — the ship-gate input */}
          <Section title="Golden task"
                   hint="The commission the ship-gate runs. It must pass its own receipts to publish.">
            <textarea value={evals[0]?.brief ?? ''}
                      onChange={(e) => {
                        const next = [{ ...(evals[0] ?? {}), brief: e.target.value }, ...evals.slice(1)]
                        onChange(set(value, 'evals', next))
                      }}
                      rows={3}
                      placeholder="A representative task a customer of this desk would commission…"
                      className={inputCls} />
          </Section>

          {/* Onboarding, with preview */}
          <Section title="Onboarding" hint="Shown on an empty desk. Markdown.">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <textarea value={String(value.onboarding_md ?? '')}
                        onChange={(e) => onChange(set(value, 'onboarding_md', e.target.value))}
                        rows={8}
                        className="w-full rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2 text-xs font-mono outline-none focus:border-[var(--lp-accent)] resize-y" />
              <div className="lp-prose text-sm rounded-lg border border-[var(--lp-border)] bg-[var(--lp-bg)] px-3 py-2 overflow-y-auto max-h-56">
                <Markdown remarkPlugins={[remarkGfm]}>
                  {String(value.onboarding_md ?? '_nothing yet_')}
                </Markdown>
              </div>
            </div>
          </Section>
        </div>
      )}
    </div>
  )
}
