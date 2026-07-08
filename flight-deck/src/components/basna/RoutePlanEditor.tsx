import { useState } from 'react'
import { SlidersHorizontal } from 'lucide-react'
import type { RoutePlan, RouteSelected } from '../../stores/basnaStore'
import type { TierMap, ArchetypeRegistry } from '../../services/tierConfig'
import { TIER_ORDER, PROVIDERS } from '../../services/tierConfig'
import { Badge, DIFFICULTY_COLOR, WeightBar } from './shared'

const COGNITIVE_MODES = ['neutra', 'ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian']

// ── Basna route plan: the selected team, each agent editable before Run ──────

export function RoutePlanEditor({ routePlan, tiers, registry, onUpdateSelected }: {
  routePlan: RoutePlan
  tiers: TierMap
  registry: ArchetypeRegistry | null
  onUpdateSelected: (idx: number, patch: Partial<RouteSelected>) => void
}) {
  const [editing, setEditing] = useState<Record<number, boolean>>({})
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Route plan</span>
        <Badge className="text-zinc-300">{routePlan.domain}</Badge>
        <Badge className={DIFFICULTY_COLOR[routePlan.difficulty] || 'text-zinc-300'}>{routePlan.difficulty}</Badge>
        <Badge className="text-sky-700 dark:text-sky-300">{routePlan.merge_kind}</Badge>
        {routePlan.source && <Badge className="text-zinc-500">{routePlan.source}</Badge>}
        <span className="ml-auto text-[11px] text-zinc-600">{routePlan.selected.length} agent(s)</span>
      </div>
      {routePlan.rationale && <p className="mb-3 text-xs text-zinc-400">{routePlan.rationale}</p>}
      <div className="space-y-2">
        {routePlan.selected.map((sel, idx) => {
          const tc = tiers[sel.tier]
          const arch = registry?.archetypes.find((a) => a.id === sel.archetype_id)
          const dispProvider = sel.provider ?? tc?.provider ?? ''
          const dispModel = sel.model ?? tc?.model ?? ''
          const isOpen = !!editing[idx]
          const fld = 'w-full rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-xs text-zinc-200 focus:border-sky-600 focus:outline-none'
          const lbl = 'mb-0.5 block text-[10px] font-medium text-zinc-500'
          return (
          <div key={sel.archetype_id} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium text-zinc-200">{sel.role || sel.archetype_id}</span>
              <div className="flex items-center gap-1.5">
                <Badge className="text-sky-700 dark:text-sky-300">{sel.tier}</Badge>
                <button
                  onClick={() => setEditing((e) => ({ ...e, [idx]: !e[idx] }))}
                  title="Edit agent"
                  className={`rounded p-1 ${isOpen ? 'text-sky-400' : 'text-zinc-500 hover:text-zinc-200'}`}
                >
                  <SlidersHorizontal className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
            <p className="mt-0.5 font-mono text-[11px] text-zinc-600">
              {dispModel ? `${dispProvider}/${dispModel}` : `${sel.tier} tier (model from server)`}
            </p>
            {sel.why && <p className="mt-1 text-xs text-zinc-500">{sel.why}</p>}
            <div className="mt-2 flex items-center gap-2">
              <span className="w-20 shrink-0 text-[11px] text-zinc-500">prior {(sel.prior_weight ?? 0).toFixed(2)}</span>
              <WeightBar value={sel.prior_weight ?? 0} />
            </div>

            {isOpen && (
              <div className="mt-3 space-y-2 border-t border-zinc-800 pt-3">
                <div>
                  <label className={lbl}>Role</label>
                  <input className={fld} value={sel.role}
                    onChange={(e) => onUpdateSelected(idx, { role: e.target.value })} />
                </div>
                <div>
                  <label className={lbl}>Tier</label>
                  <div className="flex flex-wrap gap-1.5">
                    {TIER_ORDER.filter((t) => tiers[t]).map((t) => (
                      <button
                        key={t}
                        onClick={() => onUpdateSelected(idx, {
                          tier: t, provider: undefined, model: undefined, api_key: undefined,
                          base_url: undefined, max_context: undefined, max_tokens: undefined,
                        })}
                        className={`rounded-full border px-2.5 py-0.5 text-[11px] ${
                          sel.tier === t ? 'border-sky-500 bg-sky-500/15 text-sky-300'
                            : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'
                        }`}
                      >
                        {registry?.tiers[t]?.label || t}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={lbl}>Provider</label>
                    <select className={fld} value={dispProvider}
                      onChange={(e) => onUpdateSelected(idx, { provider: e.target.value })}>
                      {dispProvider && !PROVIDERS.includes(dispProvider) && <option value={dispProvider}>{dispProvider}</option>}
                      {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className={lbl}>Model</label>
                    <input className={fld} value={dispModel} placeholder="(tier)"
                      onChange={(e) => onUpdateSelected(idx, { model: e.target.value })} />
                  </div>
                </div>
                <div>
                  <label className={lbl}>API key</label>
                  <input className={fld} type="password" value={sel.api_key ?? ''}
                    placeholder="leave blank to use the tier key"
                    onChange={(e) => onUpdateSelected(idx, { api_key: e.target.value })} />
                </div>
                <div>
                  <label className={lbl}>Base URL</label>
                  <input className={fld} value={sel.base_url ?? tc?.base_url ?? ''} placeholder="(tier)"
                    onChange={(e) => onUpdateSelected(idx, { base_url: e.target.value })} />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={lbl}>Input ctx</label>
                    <input className={fld} type="number" value={sel.max_context ?? tc?.input_ctx ?? 0}
                      onChange={(e) => onUpdateSelected(idx, { max_context: Number(e.target.value) || 0 })} />
                  </div>
                  <div>
                    <label className={lbl}>Output ctx</label>
                    <input className={fld} type="number" value={sel.max_tokens ?? tc?.output_ctx ?? 0}
                      onChange={(e) => onUpdateSelected(idx, { max_tokens: Number(e.target.value) || 0 })} />
                  </div>
                </div>
                <div>
                  <label className={lbl}>Cognitive mode</label>
                  {(() => {
                    const cm = sel.cognitive_mode ?? arch?.cognitive_mode ?? 'neutra'
                    return (
                      <select className={fld} value={cm}
                        onChange={(e) => onUpdateSelected(idx, { cognitive_mode: e.target.value })}>
                        {cm && !COGNITIVE_MODES.includes(cm) && <option value={cm}>{cm}</option>}
                        {COGNITIVE_MODES.map((m) => <option key={m} value={m}>{m}</option>)}
                      </select>
                    )
                  })()}
                </div>
                <div>
                  <label className={lbl}>Fleet instructions (system prompt)</label>
                  <textarea className={`${fld} resize-y font-mono`} rows={16}
                    value={sel.fleet_instructions ?? arch?.fleet_instructions ?? ''}
                    onChange={(e) => onUpdateSelected(idx, { fleet_instructions: e.target.value })} />
                </div>
                <div>
                  <label className={lbl}>Extra task instructions (appended to the prompt)</label>
                  <textarea className={`${fld} resize-y`} rows={8} value={sel.extra ?? ''}
                    placeholder="optional — e.g. focus areas, output format, constraints"
                    onChange={(e) => onUpdateSelected(idx, { extra: e.target.value })} />
                </div>
              </div>
            )}
          </div>
          )
        })}
      </div>
    </div>
  )
}
