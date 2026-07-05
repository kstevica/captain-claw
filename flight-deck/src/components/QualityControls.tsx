import { useState } from 'react'
import { ChevronDown, ChevronRight, Gauge } from 'lucide-react'
import { LEVERS, COST_STYLE, applyPreset, setFlag } from '../services/quality'
import type { QualityProfile, Scope, BoolFlag } from '../services/quality'

const PRESET_STYLE: Record<string, string> = {
  off: 'bg-slate-500 text-white',   // fixed neutral (zinc + text-white washes out in light theme)
  balanced: 'bg-sky-600 text-white',
  thorough: 'bg-amber-500 text-zinc-950',
}

function Switch({ on, onClick, disabled }: { on: boolean; onClick: () => void; disabled?: boolean }) {
  return (
    <button
      type="button" role="switch" aria-checked={on} onClick={onClick} disabled={disabled}
      className={`relative h-4 w-7 shrink-0 rounded-full transition-colors ${
        disabled ? 'cursor-not-allowed opacity-40' : ''} ${on ? 'bg-sky-500' : 'bg-zinc-700'}`}
    >
      <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${
        on ? 'translate-x-3.5' : 'translate-x-0.5'}`} />
    </button>
  )
}

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
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/30">
      {/* header row: preset picker + summary */}
      <div className="flex items-center gap-2 px-3 py-2">
        <Gauge className="h-3.5 w-3.5 text-zinc-500" />
        <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Quality</span>
        <div className="inline-flex rounded-lg border border-zinc-700 bg-zinc-900/50 p-0.5">
          {(['off', 'balanced', 'thorough'] as const).map((p) => {
            const sel = value.profile === p
            return (
              <button
                key={p}
                onClick={() => onChange(applyPreset(p, value))}
                title={p === 'off' ? 'No extra work — exactly current behaviour'
                  : p === 'balanced' ? 'Free / token-saving levers only'
                  : 'Everything cheap-and-safe (Deep build stays a separate opt-in)'}
                className={`rounded-md px-2.5 py-1 text-xs font-medium capitalize transition-colors ${
                  sel ? PRESET_STYLE[p] : 'text-zinc-400 hover:text-zinc-200'}`}
              >
                {p}
              </button>
            )
          })}
          {value.profile === 'custom' && (
            <span className="rounded-md bg-violet-600 px-2.5 py-1 text-xs font-medium text-white">Custom</span>
          )}
        </div>
        <button
          onClick={() => setOpen((o) => !o)}
          className="ml-auto flex items-center gap-1 rounded-md px-2 py-1 text-xs text-zinc-500 hover:text-zinc-300"
        >
          {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          {activeCount > 0 ? `${activeCount} on` : 'Advanced'}
          {saving && <span className="ml-1 text-[10px] text-zinc-600">saving…</span>}
        </button>
      </div>

      {/* advanced: individual levers */}
      {open && (
        <div className="space-y-3 border-t border-zinc-800 px-3 py-3">
          {levers.map((l) => {
            const on = value[l.flag as BoolFlag]
            return (
              <div key={l.flag}>
                <div className="flex items-start gap-3.5">
                  <div className="pt-0.5"><Switch on={on} onClick={() => onChange(setFlag(value, l.flag, !on))} /></div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-zinc-200">{l.label}</span>
                      <span className={`rounded border px-1 py-px text-[9px] font-semibold uppercase ${COST_STYLE[l.cost]}`}>
                        {l.cost}
                      </span>
                      <span className="text-[9px] font-mono text-zinc-600">{l.code}</span>
                    </div>
                    <p className="mt-1 text-[11px] leading-snug text-zinc-500">{l.blurb}</p>
                    {/* deep_build sample count */}
                    {l.flag === 'deep_build' && on && (
                      <label className="mt-1.5 flex items-center gap-2 text-[11px] text-zinc-400">
                        Attempts
                        <input
                          type="number" min={2} max={6} value={value.deep_build_samples}
                          onChange={(e) => onChange({ ...value, deep_build_samples: Math.max(2, Math.min(6, Number(e.target.value))) })}
                          className="w-14 rounded border border-rose-800/50 bg-zinc-950/60 px-2 py-0.5 text-zinc-200 focus:border-rose-500 focus:outline-none"
                        />
                      </label>
                    )}
                  </div>
                </div>
              </div>
            )
          })}

          {/* token budget — shared cost ceiling */}
          <div className="mt-2 flex items-center gap-2 border-t border-zinc-800/70 pt-2.5">
            <span className="text-[11px] text-zinc-400">Token budget</span>
            <input
              type="number" min={0} step={50000} value={value.token_budget}
              onChange={(e) => onChange({ ...value, token_budget: Math.max(0, Number(e.target.value)) })}
              placeholder="0"
              className="w-28 rounded border border-zinc-700 bg-zinc-950/60 px-2 py-0.5 text-xs text-zinc-200 focus:border-sky-500 focus:outline-none"
            />
            <span className="text-[10px] text-zinc-600">
              {value.token_budget > 0 ? `caps the paid levers at ~${Math.round(value.token_budget / 1000)}K out tok` : '0 = unbounded'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
