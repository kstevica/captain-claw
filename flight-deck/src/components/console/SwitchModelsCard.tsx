import { useEffect, useMemo, useState } from 'react'
import { Plus, Shuffle, Trash2 } from 'lucide-react'
import { LLM_PROVIDERS } from '../../services/tierConfig'
import { fetchSwitchEndpoints, saveSwitchEndpoints, type SwitchEndpoint } from './consoleApi'
import { Button, Card, Field, Notice, inputCls } from './parts'

// Models agents can switch to (captain_claw/llm/model_switch.py): an agent may
// move to another model of its own provider — same key, same endpoint — from
// the chat box's model picker, when asked, or on its own (Agent settings). The
// big providers' current models come from the tier presets catalog; for any
// other endpoint (a gateway, a local server, OpenRouter) the admin lists them
// here. Tier sets are not touched.

// Providers an agent can run on with a model list of its own.
const PROVIDERS = LLM_PROVIDERS.filter((p) => p.value !== 'browser' && p.value !== 'antigravity-cli')

interface Row {
  key: number
  provider: string
  base_url: string
  /** One model per line, a note after " — " (or " - ", " | "). */
  text: string
}

let nextKey = 1

function toText(models: SwitchEndpoint['models']): string {
  return models.map((m) => (m.note ? `${m.model} — ${m.note}` : m.model)).join('\n')
}

function toRow(e: SwitchEndpoint): Row {
  return { key: nextKey++, provider: e.provider, base_url: e.base_url, text: toText(e.models) }
}

function parseModels(text: string): SwitchEndpoint['models'] {
  return text.split('\n').map((line) => line.trim()).filter(Boolean).map((line) => {
    const m = line.match(/^(\S+)\s+(?:—|–|-|\|)\s+(.*)$/)
    return m ? { model: m[1], note: m[2].trim() } : { model: line.split(/\s+/)[0], note: '' }
  })
}

function sameEndpoint(a: { provider: string; base_url: string }, b: { provider: string; base_url: string }): boolean {
  const url = (u: string) => u.trim().replace(/\/+$/, '').toLowerCase().replace('://localhost', '://127.0.0.1')
  return a.provider === b.provider && url(a.base_url) === url(b.base_url)
}

const signature = (rows: Row[]) => JSON.stringify(rows.map((r) => [r.provider, r.base_url.trim(), r.text.trim()]))

export function SwitchModelsCard({ suggestions }: {
  /** Endpoints the deck's tier sets use that no catalog covers. */
  suggestions: { provider: string; base_url: string }[]
}) {
  const [rows, setRows] = useState<Row[] | null>(null)
  const [saved, setSaved] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [done, setDone] = useState(false)

  useEffect(() => {
    fetchSwitchEndpoints()
      .then((list) => { const r = list.map(toRow); setRows(r); setSaved(signature(r)) })
      .catch((e) => { setRows([]); setSaved(signature([])); setError(e instanceof Error ? e.message : String(e)) })
  }, [])

  const offered = useMemo(
    () => suggestions.filter((s) => !(rows || []).some((r) => sameEndpoint(r, s))),
    [suggestions, rows],
  )
  const dirty = rows !== null && signature(rows) !== saved

  const update = (key: number, patch: Partial<Row>) => {
    setRows((rs) => (rs || []).map((r) => (r.key === key ? { ...r, ...patch } : r)))
    setDone(false)
  }
  const add = (s?: { provider: string; base_url: string }) => {
    setRows((rs) => [...(rs || []), { key: nextKey++, provider: s?.provider || 'openai', base_url: s?.base_url || '', text: '' }])
    setDone(false)
  }
  const remove = (key: number) => { setRows((rs) => (rs || []).filter((r) => r.key !== key)); setDone(false) }

  const save = async () => {
    if (!rows) return
    setBusy(true); setError(''); setDone(false)
    try {
      const stored = await saveSwitchEndpoints(rows.map((r) => ({
        provider: r.provider, base_url: r.base_url.trim(), models: parseModels(r.text),
      })))
      const next = stored.map(toRow)
      setRows(next); setSaved(signature(next)); setDone(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card
      icon={Shuffle}
      title="Models agents can switch to"
      hint={<>
        An agent can move to another model of its own provider — same key, same endpoint — from the chat box, when asked,
        or on its own if its settings allow. For Anthropic, OpenAI, Gemini, Grok and DeepSeek it offers their current models.
        For a gateway, a local server or OpenRouter, list the models here. Tier sets stay as they are.
      </>}
      action={<Button icon={Plus} onClick={() => add()}>Add endpoint</Button>}
      bodyClassName="p-5 space-y-4"
    >
      {rows === null ? (
        <p className="text-sm text-zinc-500">Loading…</p>
      ) : rows.length === 0 && offered.length === 0 ? (
        <p className="text-sm text-zinc-500">No custom endpoints listed. Agents on the big providers already have their models.</p>
      ) : null}

      {rows?.map((r) => (
        <div key={r.key} className="space-y-2 rounded-xl border border-zinc-800 p-3">
          <div className="grid grid-cols-1 gap-2 wide:grid-cols-[12rem_1fr_auto] wide:items-end">
            <Field label="Provider">
              <select value={r.provider} onChange={(e) => update(r.key, { provider: e.target.value })} className={inputCls}>
                {!PROVIDERS.some((p) => p.value === r.provider) && <option value={r.provider}>{r.provider}</option>}
                {PROVIDERS.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
              </select>
            </Field>
            <Field label="Endpoint (base URL)">
              <input
                value={r.base_url}
                onChange={(e) => update(r.key, { base_url: e.target.value })}
                placeholder="The provider's own"
                className={`${inputCls} font-mono`}
              />
            </Field>
            <Button variant="ghost" icon={Trash2} aria-label="Remove this endpoint" onClick={() => remove(r.key)}>Remove</Button>
          </div>
          <Field label="Models" hint="One per line, as the endpoint names them. A note after a dash: qwen3.6-35b — fast, cheap">
            <textarea
              value={r.text}
              onChange={(e) => update(r.key, { text: e.target.value })}
              rows={Math.min(8, Math.max(3, r.text.split('\n').length + 1))}
              placeholder={'qwen3.6-35b — fast, cheap\nkimi-k3 — smarter'}
              className={`${inputCls} font-mono text-[13px]`}
            />
          </Field>
        </div>
      ))}

      {offered.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
          <span>Your tier sets use:</span>
          {offered.map((s) => (
            <button
              key={`${s.provider}|${s.base_url}`}
              type="button"
              onClick={() => add(s)}
              className="flex items-center gap-1 rounded-full border border-zinc-700 px-2.5 py-1 font-mono text-[11px] text-zinc-300 hover:border-violet-500/50 hover:bg-violet-500/10"
            >
              <Plus className="h-3 w-3" /> {s.provider}{s.base_url ? ` · ${s.base_url}` : ''}
            </button>
          ))}
        </div>
      )}

      {error && <Notice tone="red">{error}</Notice>}
      {(dirty || done) && (
        <div className="flex items-center justify-end gap-3">
          {done && !dirty && <span className="text-xs text-emerald-600 dark:text-emerald-400">Saved — agents pick it up within minutes.</span>}
          {dirty && <Button variant="primary" busy={busy} onClick={save}>Save</Button>}
        </div>
      )}
    </Card>
  )
}
