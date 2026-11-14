import { useState } from 'react'
import { ChevronDown, Gauge } from 'lucide-react'
import { LEVERS, COST_STYLE, applyPreset, setFlag } from '../services/quality'
import type { QualityProfile, Scope, BoolFlag } from '../services/quality'
import { Switch as SwitchControl } from './console/parts'

// Display labels. The underlying preset value stays 'off' (load-bearing across the
// backend + preset logic); only the shown text is "Basic".
const PRESET_LABEL: Record<string, string> = {
  off: 'Basic',
  balanced: 'Balanced',
  thorough: 'Thorough',
}

// A preset's colour, as a dot beside its name.
const PRESET_DOT: Record<string, string> = {
  off: 'bg-zinc-500',
  balanced: 'bg-sky-500',
  thorough: 'bg-amber-500',
}

function Switch({ on, onClick, label }: { on: boolean; onClick: () => void; label: string }) {
  return <SwitchControl on={on} label={label} onChange={() => onClick()} />
}

const numCls = 'rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-100 focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/15'
const selectCls = 'min-w-0 rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-100 focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/15'
const badgeCls = 'rounded-full border px-1.5 text-[10px] font-semibold uppercase leading-4'

/**
 * Opt-in quality/cost levers for a run (research = Basna/Vatra, code = Code).
 * Preset picker + an expandable advanced panel of individual toggles. Everything
 * off == the systems' current behaviour, so this can never regress a run.
 */
export function QualityControls({
  value, onChange, scope, saving,
}: {
  value: QualityProfile
  onChange: (q: QualityProfile) => void
  scope: Scope
  saving?: boolean
}) {
  const [open, setOpen] = useState(false)
  const levers = LEVERS.filter((l) => l.scope === scope)
  const activeCount = levers.filter((l) => value[l.flag]).length

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/40">
      {/* header row: preset picker + summary */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-3.5 py-3">
        <span className="flex items-center gap-1.5">
          <Gauge className="h-3.5 w-3.5 text-zinc-500" />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Quality</span>
        </span>
        <div className="inline-flex max-w-full flex-wrap gap-1 rounded-xl border border-zinc-800 bg-zinc-950/60 p-1">
          {(['off', 'balanced', 'thorough'] as const).map((p) => {
            const sel = value.profile === p
            return (
              <button
                key={p}
                onClick={() => onChange(applyPreset(p, value))}
                aria-pressed={sel}
                title={p === 'off' ? 'No extra work — exactly current behaviour'
                  : p === 'balanced' ? 'Free / token-saving levers only'
                  : 'Everything cheap-and-safe (Deep build stays a separate opt-in)'}
                className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                  sel ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-500 hover:text-zinc-200'}`}
              >
                <span className={`h-1.5 w-1.5 rounded-full ${PRESET_DOT[p]} ${sel ? '' : 'opacity-50'}`} />
                {PRESET_LABEL[p]}
              </button>
            )
          })}
          {value.profile === 'custom' && (
            <span className="flex items-center rounded-lg bg-violet-500/15 px-2.5 py-1 text-xs font-medium text-violet-700 dark:text-violet-300">Custom</span>
          )}
        </div>
        {!value.honesty_guard && (
          <span className={`${badgeCls} border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300`}
                title="The anti-fabrication guard is switched off for this run">
            guard off
          </span>
        )}
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="ml-auto flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
        >
          {activeCount > 0 ? `${activeCount} on` : 'Advanced'}
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
          {saving && <span className="ml-1 text-[10px] text-zinc-600">saving…</span>}
        </button>
      </div>

      {/* advanced: individual levers */}
      {open && (
        <div className="space-y-4 border-t border-zinc-800 px-3.5 py-4">
          {/* Calibration posture (research): the default-on honesty guard + the
              output mode. Preset-independent — pinned above the preset levers. */}
          {scope === 'research' && (
            <div className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/40 px-3.5 py-3">
              <div className="flex items-start gap-3.5">
                <div className="pt-0.5">
                  <Switch on={value.honesty_guard} label="Honesty guard"
                          onClick={() => onChange({ ...value, honesty_guard: !value.honesty_guard })} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-medium text-zinc-200">Honesty guard</span>
                    <span className={`${badgeCls} border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300`}>
                      default on
                    </span>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-zinc-500">
                    Anti-fabrication counterweight to the “finish it completely” pressure: unconfirmable
                    specifics get qualified (never invented), unknowns become labeled placeholders, estimates
                    carry their basis, and genuinely unresolved disagreements surface in an “Unresolved &amp;
                    assumptions” section instead of being silently absorbed. Free, prompt-only; switching it
                    off restores the previous prompts exactly.
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2 border-t border-zinc-800/80 pt-3">
                <span className="text-xs text-zinc-400">Output mode</span>
                <select
                  value={value.output_mode}
                  onChange={(e) => onChange({ ...value, output_mode: e.target.value as QualityProfile['output_mode'] })}
                  className={selectCls}
                >
                  <option value="">Default</option>
                  <option value="complete">Complete — full draft, every unknown labeled</option>
                  <option value="conservative">Conservative — review copy, fabricates nothing</option>
                </select>
              </div>
            </div>
          )}
          {/* Calibration posture (code): the same default-on honesty guard, but
              about claiming work done — plus the code output mode. Applies only
              within an active profile, so a bare off-profile run is unchanged. */}
          {scope === 'code' && (
            <div className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/40 px-3.5 py-3">
              <div className="flex items-start gap-3.5">
                <div className="pt-0.5">
                  <Switch on={value.honesty_guard} label="Honesty guard"
                          onClick={() => onChange({ ...value, honesty_guard: !value.honesty_guard })} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-medium text-zinc-200">Honesty guard</span>
                    <span className={`${badgeCls} border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300`}>
                      default on
                    </span>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-zinc-500">
                    Stops an agent claiming work it didn’t do — “tests pass”, “build green”, “done” — without
                    actually running the command, and stops it describing an edit it never wrote to disk. The
                    prompt-side counterweight to the deterministic build/fix guards. Free; only applies within an
                    active profile, and switching it off restores the previous prompts exactly.
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2 border-t border-zinc-800/80 pt-3">
                <span className="text-xs text-zinc-400">Output mode</span>
                <select
                  value={value.output_mode}
                  onChange={(e) => onChange({ ...value, output_mode: e.target.value as QualityProfile['output_mode'] })}
                  className={selectCls}
                >
                  <option value="">Default</option>
                  <option value="complete">Full build — implement everything, label placeholders</option>
                  <option value="conservative">Cautious — no fakes/stubs to pass checks</option>
                </select>
              </div>
            </div>
          )}
          {levers.map((l) => {
            const on = value[l.flag as BoolFlag]
            return (
              <div key={l.flag}>
                <div className="flex items-start gap-3.5">
                  <div className="pt-0.5"><Switch on={on} label={l.label} onClick={() => onChange(setFlag(value, l.flag, !on))} /></div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[13px] font-medium text-zinc-200">{l.label}</span>
                      <span className={`${badgeCls} ${COST_STYLE[l.cost]}`}>
                        {l.cost}
                      </span>
                      <span className="font-mono text-[10px] text-zinc-600">{l.code}</span>
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-zinc-500">{l.blurb}</p>
                    {/* deep_build sample count */}
                    {l.flag === 'deep_build' && on && (
                      <label className="mt-2 flex items-center gap-2 text-xs text-zinc-400">
                        Attempts
                        <input
                          type="number" min={2} max={6} value={value.deep_build_samples}
                          onChange={(e) => onChange({ ...value, deep_build_samples: Math.max(2, Math.min(6, Number(e.target.value))) })}
                          className={`${numCls} w-16`}
                        />
                      </label>
                    )}
                    {/* claim_check: how many top claims to verify */}
                    {l.flag === 'claim_check' && on && (
                      <label className="mt-2 flex items-center gap-2 text-xs text-zinc-400">
                        Claims to verify
                        <input
                          type="number" min={3} max={20} value={value.claim_check_max}
                          onChange={(e) => onChange({ ...value, claim_check_max: Math.max(3, Math.min(20, Number(e.target.value))) })}
                          className={`${numCls} w-16`}
                        />
                      </label>
                    )}
                    {/* parallel_build: max decomposition slices */}
                    {l.flag === 'parallel_build' && on && (
                      <label className="mt-2 flex items-center gap-2 text-xs text-zinc-400">
                        Max slices
                        <input
                          type="number" min={2} max={8} value={value.parallel_build_max_slices}
                          onChange={(e) => onChange({ ...value, parallel_build_max_slices: Math.max(2, Math.min(8, Number(e.target.value))) })}
                          className={`${numCls} w-16`}
                        />
                      </label>
                    )}
                    {/* block_on_critical: revise-recheck rounds cap */}
                    {l.flag === 'block_on_critical' && on && (
                      <label className="mt-2 flex items-center gap-2 text-xs text-zinc-400">
                        Max rounds
                        <input
                          type="number" min={1} max={5} value={value.block_max_rounds}
                          onChange={(e) => onChange({ ...value, block_max_rounds: Math.max(1, Math.min(5, Number(e.target.value))) })}
                          className={`${numCls} w-16`}
                        />
                      </label>
                    )}
                  </div>
                </div>
              </div>
            )
          })}

          {/* token budget — shared cost ceiling */}
          <div className="flex flex-wrap items-center gap-2 border-t border-zinc-800/80 pt-3.5">
            <span className="text-xs text-zinc-400">Token budget</span>
            <input
              type="number" min={0} step={50000} value={value.token_budget}
              onChange={(e) => onChange({ ...value, token_budget: Math.max(0, Number(e.target.value)) })}
              placeholder="0"
              className={`${numCls} w-28`}
            />
            <span className="text-[11px] text-zinc-500">
              {value.token_budget > 0 ? `caps the paid levers at ~${Math.round(value.token_budget / 1000)}K out tok` : '0 = unbounded'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
