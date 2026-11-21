import { useEffect, useState } from 'react'
import { Loader2, Plus, RefreshCw, Save, Trash2 } from 'lucide-react'
import { forgetRecipe, getRecipe, saveRecipe, type PinnedRecipe, type RecipeFilter, type RecipeRule } from '../../services/firstMate'
import { errText, Pill } from './ui'

// A list factory's recipe: what its runs follow. Built once per charter by the first
// run, then the same for every run; the owner can change its rules, the columns a row
// must fill and its filters, or have the next run build it again.

const input = 'w-full rounded-lg border border-[var(--fd-line)] bg-zinc-950 px-2.5 py-1.5 text-[13px] text-zinc-100 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none'

function filterText(f: RecipeFilter): string {
  const col = f.column === '*' ? 'anything in its listing' : f.column
  const v = Array.isArray(f.value) ? f.value.join(', ') : String(f.value ?? '')
  const words: Record<string, string> = {
    regex: 'must match', not_regex: 'must not match', in: 'is one of', not_in: 'is none of',
    min: 'is at least', max: 'is at most', size_max: 'has at most (people)', size_min: 'has at least (people)',
  }
  const only = f.groups?.length ? ` (${f.groups.join(', ')} only)` : ''
  return `${col} ${words[f.op] ?? f.op} ${v}${only}`
}

export function FactoryRecipe({ factoryId, onNotice, onChanged }: { factoryId: string; onNotice: (text: string) => void; onChanged: () => void }) {
  const [data, setData] = useState<PinnedRecipe | null>(null)
  const [rules, setRules] = useState<RecipeRule[]>([])
  const [required, setRequired] = useState<string[]>([])
  const [filters, setFilters] = useState<RecipeFilter[]>([])
  const [shortage, setShortage] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [busy, setBusy] = useState('')
  const [asking, setAsking] = useState(false)
  const [error, setError] = useState('')

  const take = (d: PinnedRecipe) => {
    setData(d)
    setRules(d.recipe?.rules ?? [])
    setRequired(d.recipe?.required ?? [])
    setFilters(d.recipe?.filters ?? [])
    setShortage(!!d.recipe?.shortage_ok)
    setDirty(false)
  }
  useEffect(() => {
    let stop = false
    getRecipe(factoryId).then((d) => { if (!stop) take(d) }).catch((e) => { if (!stop) setError(errText(e)) })
    return () => { stop = true }
  }, [factoryId])

  const edit = <T,>(set: (v: T) => void) => (v: T) => { set(v); setDirty(true) }
  const setRule = (i: number, patch: Partial<RecipeRule>) =>
    edit(setRules)(rules.map((r, n) => (n === i ? { ...r, ...patch } : r)))
  const save = async () => {
    setBusy('save'); setError('')
    try {
      take(await saveRecipe(factoryId, { rules: rules.filter((r) => r.rule.trim()), required, filters, shortage_ok: shortage }))
      onNotice('Saved: its runs follow the recipe as you changed it')
      onChanged()
    } catch (e) { setError(errText(e)) } finally { setBusy('') }
  }
  const forget = async () => {
    setBusy('forget'); setError('')
    try {
      take(await forgetRecipe(factoryId))
      setAsking(false)
      onNotice('Forgotten: the next run builds a new recipe')
      onChanged()
    } catch (e) { setError(errText(e)) } finally { setBusy('') }
  }

  if (!data) {
    return error ? <p role="alert" className="text-xs text-red-700 dark:text-red-300">{error}</p>
      : <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />
  }
  const r = data.recipe
  if (!r) {
    return (
      <p className="rounded-xl border border-dashed border-[var(--fd-line)] px-4 py-6 text-center text-[13px] text-zinc-500">
        {data.stale ? 'The charter or its goals changed since the recipe was built. The next run builds a new one.'
          : 'No recipe yet. The first run builds it from the charter, and every later run follows the same one.'}
      </p>
    )
  }
  const sources = r.sources.map((s) => (s.path ? s.path : `${s.queries?.length ?? 0} searches${s.group ? ` for ${s.group}` : ''}`))
  const reads = r.gather.map((g) => (g.kind === 'fetch' ? `its own site${g.paths?.filter(Boolean).length ? ` (${g.paths.filter(Boolean).join(', ')})` : ''}` : `a search: ${g.query ?? ''}`))
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <p className="min-w-0 flex-1 text-xs text-zinc-500">
          What every run of this charter follows. Built {data.built_at ?? 'earlier'}{data.edited ? `, edited by hand ${data.edited_at ?? ''}` : ''}.
          It is built again when the charter or its goals change.
        </p>
        {data.edited && <Pill tone="violet">Edited</Pill>}
        {asking ? (
          <>
            <button onClick={() => void forget()} disabled={!!busy}
              className="inline-flex h-8 items-center gap-1.5 rounded-full bg-red-600 px-3 text-xs font-semibold text-white hover:bg-red-500 disabled:opacity-60">
              {busy === 'forget' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}Forget it?
            </button>
            <button onClick={() => setAsking(false)} disabled={!!busy} className="fd-btn-quiet h-8 px-3 text-xs">Keep</button>
          </>
        ) : (
          <button onClick={() => setAsking(true)} className="fd-btn-quiet inline-flex h-8 items-center gap-1.5 px-3 text-xs"
            title="The next run writes a new recipe from the charter">
            <RefreshCw className="h-3.5 w-3.5" />Build again next run
          </button>
        )}
      </div>

      <section>
        <h3 className="text-[13px] font-semibold text-zinc-100">Rules</h3>
        <p className="mt-0.5 text-xs text-zinc-500">
          A must is never bent. “If the fact isn’t found” is what holds then (left empty: the item only comes close).
        </p>
        <ul className="mt-2 space-y-2">
          {rules.map((rule, i) => (
            <li key={i} className="rounded-xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] p-3">
              <div className="flex items-start gap-2">
                <span className="mt-1.5 w-5 shrink-0 text-right text-xs tabular-nums text-zinc-500">{i + 1}.</span>
                <textarea value={rule.rule} rows={2} aria-label={`Rule ${i + 1}`} onChange={(e) => setRule(i, { rule: e.target.value })}
                  className={`${input} resize-none`} />
                <button onClick={() => edit(setRules)(rules.filter((_, n) => n !== i))} aria-label={`Remove rule ${i + 1}`}
                  className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
              <div className="mt-2 grid gap-2 pl-7 wide:grid-cols-[8rem_1fr_1fr]">
                <select value={rule.kind} aria-label={`Rule ${i + 1} kind`} onChange={(e) => setRule(i, { kind: e.target.value as RecipeRule['kind'] })}
                  className={input}>
                  <option value="must">Must</option>
                  <option value="prefer">Prefer</option>
                </select>
                <input value={rule.if_not_shown ?? ''} placeholder="If the fact isn’t found…" aria-label={`Rule ${i + 1}, if the fact isn’t found`}
                  onChange={(e) => setRule(i, { if_not_shown: e.target.value })} className={input} />
                <input value={rule.looser ?? ''} placeholder={rule.kind === 'prefer' ? 'Its looser reading (a prefer rule needs one)' : 'Its looser reading (optional)'}
                  aria-label={`Rule ${i + 1}, looser reading`} onChange={(e) => setRule(i, { looser: e.target.value })} className={input} />
              </div>
            </li>
          ))}
        </ul>
        <button onClick={() => edit(setRules)([...rules, { rule: '', kind: 'must' }])}
          className="fd-btn-quiet mt-2 inline-flex h-8 items-center gap-1.5 px-3 text-xs">
          <Plus className="h-3.5 w-3.5" />Add a rule
        </button>
      </section>

      <section>
        <h3 className="text-[13px] font-semibold text-zinc-100">Columns a row must fill</h3>
        <p className="mt-0.5 text-xs text-zinc-500">An item missing one of these comes close; it isn’t put on the list.</p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {r.columns.map((c) => {
            const on = required.includes(c)
            return (
              <button key={c} role="switch" aria-checked={on}
                onClick={() => edit(setRequired)(on ? required.filter((x) => x !== c) : [...required, c])}
                className={`h-7 rounded-full border px-2.5 text-xs transition-colors ${on
                  ? 'border-violet-500/40 bg-violet-500/15 text-violet-200' : 'border-[var(--fd-line)] text-zinc-400 hover:text-zinc-200'}`}>
                {c}
              </button>
            )
          })}
        </div>
      </section>

      <section>
        <h3 className="text-[13px] font-semibold text-zinc-100">Ruled out before reading</h3>
        {filters.length ? (
          <ul className="mt-2 divide-y divide-[var(--fd-line)] overflow-hidden rounded-xl border border-[var(--fd-line)]">
            {filters.map((flt, i) => (
              <li key={i} className="flex items-center gap-2 px-3 py-1.5">
                <span className="min-w-0 flex-1 break-words text-[13px] text-zinc-300">{filterText(flt)}</span>
                <button onClick={() => edit(setFilters)(filters.filter((_, n) => n !== i))} aria-label={`Remove filter ${i + 1}`}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </li>
            ))}
          </ul>
        ) : <p className="mt-1 text-xs text-zinc-500">Nothing: every item found is read and judged.</p>}
      </section>

      <label className="flex cursor-pointer items-start gap-2.5 text-[13px] text-zinc-300">
        <input type="checkbox" checked={shortage} onChange={(e) => edit(setShortage)(e.target.checked)} className="mt-0.5" />
        <span>A short list is fine when the summary says why. No rule is then read loosely to fill it.</span>
      </label>

      <section className="rounded-xl border border-[var(--fd-line)] p-3 text-xs text-zinc-500">
        <p><span className="font-medium text-zinc-300">The list:</span> {r.output}, one {r.item} per row{Object.keys(r.groups).length
          ? `; ${Object.entries(r.groups).map(([g, n]) => `${g}${n ? ` ${n}` : ''}`).join(', ')}` : ''}.</p>
        <p className="mt-1"><span className="font-medium text-zinc-300">Found in:</span> {sources.join('; ') || 'nothing'}.</p>
        <p className="mt-1"><span className="font-medium text-zinc-300">Read for each:</span> {reads.join('; ') || 'only its listing'}.</p>
        {r.checks.length > 0 && (
          <p className="mt-1"><span className="font-medium text-zinc-300">Checked by code:</span> {r.checks.map((c) => `${c.column} (${c.op.replace('_', ' ')})`).join(', ')}.</p>
        )}
      </section>

      {error && <p role="alert" className="text-xs text-red-700 dark:text-red-300">{error}</p>}
      {dirty && (
        <div className="sticky bottom-0 flex items-center justify-end gap-2 border-t border-[var(--fd-line)] bg-[var(--fd-surface)]/95 py-3 backdrop-blur">
          <button onClick={() => take(data)} disabled={!!busy} className="fd-btn-quiet h-8 px-3 text-xs">Discard</button>
          <button onClick={() => void save()} disabled={!!busy}
            className="inline-flex h-8 items-center gap-1.5 rounded-full bg-violet-600 px-3 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-60">
            {busy === 'save' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}Save
          </button>
        </div>
      )}
    </div>
  )
}
