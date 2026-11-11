import { useEffect, useState } from 'react'
import { EyeOff, Save, Trash2, FlaskConical, CheckCircle2, XCircle } from 'lucide-react'
import {
  anonymizeApi, type TrustedModel, type TrustedModelInput, type TrustedTestCase,
} from '../../services/anonymizeApi'
import { Card, Button, Field, Notice, Pill, Segmented, SettingRow, Switch, inputCls } from '../console/parts'

// Admin → Settings → Anonymization: the trusted model behind the LLM lane.
// "Trusted" means trusted with the raw document — a model on this machine or
// a private network as it is, a cloud model only when the admin vouches for
// it. Never the agent's own model by default.

const LOCAL_HINT: Record<string, string> = {
  ollama: 'http://localhost:11434',
  litert: '',
  openai: 'http://localhost:8000/v1 (vLLM, LM Studio, llama.cpp …)',
}

export function TrustedModelCard() {
  const [model, setModel] = useState<TrustedModel | null>(null)
  const [providers, setProviders] = useState<string[]>([])
  const [form, setForm] = useState<TrustedModelInput>({
    provider: 'ollama', model: '', base_url: '', attested: false, on_failure: 'refuse', enabled: true,
  })
  const [apiKey, setApiKey] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [saved, setSaved] = useState('')
  const [test, setTest] = useState<{ model: string; cases: TrustedTestCase[] } | null>(null)

  const take = (m: TrustedModel | null) => {
    setModel(m)
    if (m) setForm({ provider: m.provider, model: m.model, base_url: m.base_url, attested: m.attested,
      on_failure: m.on_failure, enabled: m.enabled })
  }

  useEffect(() => {
    anonymizeApi.trustedGet()
      .then((r) => { take(r.model); setProviders(r.providers) })
      .catch(() => setError("Couldn't load the anonymization settings"))
      .finally(() => setLoaded(true))
  }, [])

  const set = <K extends keyof TrustedModelInput>(k: K, v: TrustedModelInput[K]) => setForm((f) => ({ ...f, [k]: v }))
  const local = form.provider === 'ollama' || form.provider === 'litert' || (form.provider === 'openai' && !!form.base_url)

  const save = async () => {
    setBusy('save'); setError(''); setSaved('')
    try {
      const body: TrustedModelInput = { ...form, ...(apiKey ? { api_key: apiKey } : {}) }
      const r = await anonymizeApi.trustedPut(body)
      take(r.model)
      setApiKey('')
      setSaved(r.model.basis === 'local'
        ? 'Saved. It runs on this machine or your network, so it is trusted as it is.'
        : 'Saved. You vouched for this provider: it will receive the documents people anonymize.')
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't save")
    } finally {
      setBusy('')
    }
  }

  const remove = async () => {
    setBusy('remove'); setError(''); setSaved(''); setTest(null)
    try {
      const r = await anonymizeApi.trustedDelete()
      take(r.model)
      if (!r.model) setForm({ provider: 'ollama', model: '', base_url: '', attested: false, on_failure: 'refuse', enabled: true })
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't remove")
    } finally {
      setBusy('')
    }
  }

  const runTest = async () => {
    setBusy('test'); setError(''); setTest(null)
    try {
      setTest(await anonymizeApi.trustedTest())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The check failed')
    } finally {
      setBusy('')
    }
  }

  return (
    <Card
      icon={EyeOff}
      title="Anonymization"
      hint="A trusted model also looks when people and agents anonymize: it finds names with nothing around them and descriptions that point to one person. Code always looks too."
      action={model && <Pill tone={model.usable ? 'emerald' : 'zinc'}>{model.usable ? `On · ${model.label}` : 'Off'}</Pill>}
      bodyClassName="space-y-5 p-5"
    >
      {!loaded ? null : (
        <>
          {model?.source === 'env' && (
            <Notice>Set in this deck's environment (FD_ANONYMIZE_*). Saving here takes its place; removing it here goes back to the environment's.</Notice>
          )}
          <div className="grid gap-4 wide:grid-cols-2">
            <Field label="Provider">
              <select value={form.provider} onChange={(e) => set('provider', e.target.value)} className={inputCls}>
                {(providers.length ? providers : ['ollama', 'litert', 'openai', 'anthropic', 'gemini', 'xai', 'openrouter']).map((p) => (
                  <option key={p} value={p}>{p === 'openai' ? 'openai (or a compatible server)' : p}</option>
                ))}
              </select>
            </Field>
            <Field label="Model" hint={form.provider === 'ollama' ? 'A model you pulled, e.g. qwen3:8b. Not a "-cloud" one: those run at Ollama.' : undefined}>
              <input value={form.model} onChange={(e) => set('model', e.target.value)} className={inputCls}
                placeholder={form.provider === 'anthropic' ? 'claude-haiku-5-5' : 'qwen3:8b'} />
            </Field>
            <Field label="Address" hint="Leave empty for the provider's usual one.">
              <input value={form.base_url} onChange={(e) => set('base_url', e.target.value)} className={inputCls}
                placeholder={LOCAL_HINT[form.provider] ?? ''} />
            </Field>
            <Field label="API key" hint={model?.has_key ? 'A key is saved; type a new one to replace it.' : 'Only if the provider needs one.'}>
              <input type="password" autoComplete="off" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
                className={inputCls} placeholder={model?.has_key ? '•••••••• (saved)' : ''} />
            </Field>
          </div>

          {!local && (
            <label className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-amber-700 dark:text-amber-300">
              <input type="checkbox" checked={form.attested} onChange={(e) => set('attested', e.target.checked)} className="mt-1" />
              <span>
                This model doesn't run on this machine or your network, so it receives the raw documents. Tick this only if you trust
                the provider with them — zero data retention, a data-processing agreement, a region you accept.
              </span>
            </label>
          )}

          <div className="divide-y divide-zinc-800/80">
            <SettingRow title="If the model fails" hint="Refusing returns nothing, so nobody gets a result that missed what only the model would find.">
              <Segmented label="If the model fails" value={form.on_failure} onChange={(v) => set('on_failure', v)}
                options={[{ value: 'refuse', label: 'Refuse' }, { value: 'deterministic', label: 'Code only, marked' }]} />
            </SettingRow>
            <SettingRow title="On" hint="Off: only code looks, everywhere.">
              <Switch on={form.enabled} label="Trusted model on" onChange={(v) => set('enabled', v)} />
            </SettingRow>
          </div>

          {error && <Notice tone="red">{error}</Notice>}
          {saved && <Notice tone="emerald">{saved}</Notice>}

          <div className="flex flex-wrap items-center justify-end gap-2">
            {model?.source === 'settings' && <Button variant="danger" icon={Trash2} busy={busy === 'remove'} onClick={remove} className="mr-auto">Remove</Button>}
            {model?.usable && <Button icon={FlaskConical} busy={busy === 'test'} onClick={runTest}>Check on samples</Button>}
            <Button variant="primary" icon={Save} busy={busy === 'save'} disabled={!form.model.trim()} onClick={save}>Save</Button>
          </div>

          {test && <TestTable result={test} />}
        </>
      )}
    </Card>
  )
}

function TestTable({ result }: { result: { model: string; cases: TrustedTestCase[] } }) {
  return (
    <div className="space-y-2">
      <p className="text-xs text-zinc-500">
        Made-up documents with made-up people, run through code alone and through code + {result.model}. What each left behind:
      </p>
      <div className="overflow-x-auto rounded-xl border border-zinc-800">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-zinc-500">
            <tr className="border-b border-zinc-800">
              <th className="px-3 py-2 font-medium">Sample</th>
              <th className="px-3 py-2 font-medium">Code alone missed</th>
              <th className="px-3 py-2 font-medium">With the model missed</th>
              <th className="px-3 py-2 font-medium">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/80">
            {result.cases.map((c) => {
              const ok = !c.error && (c.both_missed || []).length === 0 && (c.both_over_hid || []).length === 0
              return (
                <tr key={c.id} className="align-top">
                  <td className="whitespace-nowrap px-3 py-2 text-zinc-300">
                    <span className="inline-flex items-center gap-1.5">
                      {ok ? <CheckCircle2 className="h-4 w-4 text-emerald-500" /> : <XCircle className="h-4 w-4 text-red-500" />}
                      {c.id}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs text-zinc-400">{c.code_missed.length ? c.code_missed.join(', ') : 'nothing'}</td>
                  <td className="px-3 py-2 text-xs text-zinc-400">
                    {c.error ? <span className="text-red-600 dark:text-red-400">{c.error}</span>
                      : (c.both_missed || []).length ? c.both_missed!.join(', ') : 'nothing'}
                    {!c.error && (c.both_over_hid || []).length > 0 && <div className="text-amber-600 dark:text-amber-400">also hid: {c.both_over_hid!.join(', ')}</div>}
                  </td>
                  <td className="whitespace-nowrap px-3 py-2 text-xs tabular-nums text-zinc-500">{c.seconds}s</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
