import { useCallback, useEffect, useMemo, useState } from 'react'
import { Wrench, Play, KeyRound, UploadCloud, History, Share2, BarChart3, Trash2, Hammer, ChevronDown, ChevronRight, X, Workflow, Plus } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import {
  customToolsApi, type JsonSchema, type MyAgent, type RunResult, type Sharing, type ToolDetail, type Usage,
} from '../../services/customToolsApi'
import { Sheet, Button, Notice, Pill, Switch, Field, inputCls } from '../console/parts'
import { ShareModal } from '../common/ShareModal'
import { DeclaresLine, statusPill, usesWords } from './toolParts'
import { HarnessPicker } from './NewToolDialog'

// One tool: try it, set your secrets for it, and for your own tools publish,
// go back to a kept version, have Crew change it, offer it, see how it's used.

type FieldValue = string | boolean

function fieldsOf(schema: JsonSchema | undefined) {
  return Object.entries(schema?.properties || {}).map(([key, prop]) => {
    const fileList = prop.type === 'array' && prop.items?.format === 'file'
    const kind = prop.format === 'file' ? 'file' : fileList ? 'files' : prop.enum ? 'enum'
      : prop.type === 'boolean' ? 'boolean' : prop.type === 'integer' || prop.type === 'number' ? 'number'
        : prop.type === 'string' ? 'string' : 'json'
    return { key, prop, kind, required: (schema?.required || []).includes(key) }
  })
}

function buildInput(fields: ReturnType<typeof fieldsOf>, values: Record<string, FieldValue>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const f of fields) {
    const v = values[f.key]
    if (v === undefined || v === '') continue
    if (f.kind === 'boolean') out[f.key] = !!v
    else if (f.kind === 'number') out[f.key] = Number(v)
    else if (f.kind === 'files') out[f.key] = String(v).split('\n').map((s) => s.trim()).filter(Boolean)
    else if (f.kind === 'json') out[f.key] = JSON.parse(String(v))
    else out[f.key] = v
  }
  return out
}

function Section({ icon: Icon, title, children, action }: {
  icon: typeof Wrench; title: string; children: React.ReactNode; action?: React.ReactNode
}) {
  return (
    <section className="space-y-3 border-t border-zinc-800/80 pt-4 first:border-t-0 first:pt-0">
      <div className="flex items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-zinc-200"><Icon className="h-4 w-4 text-zinc-400" />{title}</h3>
        {action}
      </div>
      {children}
    </section>
  )
}

export function ToolSheet({ slug, owner, mine, onClose, onChanged, onDeleted }: {
  slug: string
  owner?: string
  mine: boolean
  onClose: () => void
  onChanged: () => void
  onDeleted: () => void
}) {
  const authEnabled = useAuthStore((s) => s.authEnabled !== false)
  const [tool, setTool] = useState<ToolDetail | null>(null)
  const [sharing, setSharing] = useState<Sharing | null>(null)
  const [usage, setUsage] = useState<Usage | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [values, setValues] = useState<Record<string, FieldValue>>({})
  const [asJson, setAsJson] = useState(false)
  const [jsonText, setJsonText] = useState('{}')
  const [result, setResult] = useState<RunResult | null>(null)
  const [secretDraft, setSecretDraft] = useState<Record<string, string>>({})
  const [shareOpen, setShareOpen] = useState(false)
  const [steps, setSteps] = useState(false)
  const [what, setWhat] = useState('')
  const [harness, setHarness] = useState('')
  const [note, setNote] = useState('')
  // Phase 7c: a tool that calls back into an agent runs as one of the person's agents.
  const [agents, setAgents] = useState<MyAgent[] | null>(null)
  const [runAs, setRunAs] = useState('')

  const load = useCallback(async () => {
    try {
      const t = await customToolsApi.get(slug, mine ? undefined : owner)
      setTool(t)
      if (mine) {
        setUsage(await customToolsApi.usage(slug))
        if (authEnabled) setSharing(await customToolsApi.sharing(slug))
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't load it")
    }
  }, [slug, owner, mine, authEnabled])

  useEffect(() => { load() }, [load])

  // Publishing: look again until it's done.
  useEffect(() => {
    if (tool?.publish?.state !== 'running') return
    const timer = window.setInterval(() => { load(); onChanged() }, 2500)
    return () => window.clearInterval(timer)
  }, [tool?.publish?.state, load, onChanged])

  const fields = useMemo(() => fieldsOf(tool?.manifest?.input), [tool?.manifest?.input])
  const callsBack = (tool?.manifest?.uses || []).length > 0

  useEffect(() => {
    if (!callsBack || agents) return
    customToolsApi.agents().then((r) => {
      setAgents(r.agents)
      const first = r.agents.find((a) => a.status !== 'stopped')
      if (first) setRunAs((v) => v || first.slug)
    }).catch(() => setAgents([]))
  }, [callsBack, agents])

  const act = async (key: string, fn: () => Promise<unknown>, after?: () => void) => {
    setBusy(key)
    setError('')
    try {
      await fn()
      after?.()
      await load()
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setBusy('')
    }
  }

  const run = async () => {
    setBusy('run')
    setError('')
    setResult(null)
    try {
      const input = asJson ? JSON.parse(jsonText || '{}') : buildInput(fields, values)
      setResult(await customToolsApi.run(slug, input, mine ? undefined : owner, callsBack ? runAs : undefined))
      if (mine) setUsage(await customToolsApi.usage(slug))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setBusy('')
    }
  }

  const saveSecrets = () => act('secrets', () => customToolsApi.setSecrets(slug, secretDraft, mine ? undefined : owner),
    () => setSecretDraft({}))

  if (!tool) {
    return (
      <Sheet title={slug} icon={Wrench} onClose={onClose} wide>
        {error ? <Notice tone="red">{error}</Notice> : <p className="text-sm text-zinc-500">Loading…</p>}
      </Sheet>
    )
  }

  const m = tool.manifest
  const subtitle = mine ? `${slug}${tool.live ? ` · version ${tool.live.version.slice(0, 7)}` : ''}`
    : `by ${tool.owner_person?.name || 'someone'}${tool.added ? ' · added' : ' · not added'}`

  return (
    <Sheet title={tool.title} subtitle={subtitle} icon={Wrench} onClose={onClose} wide>
      <div className="space-y-5">
        {error && <Notice tone="red">{error}</Notice>}
        {note && <Notice tone="emerald">{note}</Notice>}
        <div>
          <div className="flex flex-wrap items-center gap-2">{statusPill(tool)}</div>
          {tool.description && <p className="mt-2 text-sm leading-relaxed text-zinc-400">{tool.description}</p>}
          {m && <DeclaresLine d={{ version: tool.live?.version || '', network: m.network, secrets: m.secrets.map((s) => s.name),
            timeout_s: m.timeout_s, memory_mb: m.memory_mb, ...usesWords(m.uses) }} />}
          {m && (m.uses || []).length > 0 && (
            <ul className="mt-2 space-y-0.5 text-[11px] text-zinc-500">
              {(m.uses || []).map((u, i) => (
                <li key={i}><span className="font-mono text-zinc-400">{u.tool || u.custom_tool}</span> — {u.why}</li>
              ))}
            </ul>
          )}
        </div>

        {!mine && !tool.added && (
          <Notice tone="amber">Add it first to let your agents use it: it runs as you, with your secrets and files.</Notice>
        )}

        {(tool.depends_on || []).length > 0 && (
          <Section icon={Workflow} title="Tools it runs">
            <p className="text-xs text-zinc-500">
              {mine ? 'It runs these of your tools while it works. People you offer it to need them too.'
                : 'It runs these tools while it works, as you. Add them too, or it stops when it reaches one.'}
            </p>
            <ul className="space-y-1.5">
              {(tool.depends_on || []).map((d) => (
                <li key={d.slug} className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-zinc-800 px-3 py-2 text-xs">
                  <span className="min-w-0"><span className="text-zinc-200">{d.title}</span> <span className="font-mono text-zinc-500">{d.slug}</span></span>
                  {d.state === 'yours' ? (d.live ? <Pill tone="emerald">Live</Pill> : <Pill tone="amber">Not published</Pill>)
                    : d.state === 'added' ? <Pill tone="emerald">Added</Pill>
                      : d.state === 'offered' ? (
                        <Button icon={Plus} busy={busy === `dep-${d.slug}`} onClick={() => act(`dep-${d.slug}`, () => customToolsApi.add(d.owner, d.slug))}>Add</Button>
                      ) : <Pill tone="amber">Not offered to you</Pill>}
                </li>
              ))}
            </ul>
            {!mine && (tool.depends_on || []).some((d) => d.state === 'not_offered') && (
              <p className="text-xs text-amber-600 dark:text-amber-400">Its maker hasn't offered you every tool it runs, so part of it won't work for you yet.</p>
            )}
          </Section>
        )}

        {m && (mine || tool.added) && (
          <Section icon={Play} title="Try it" action={
            <button className="text-xs text-zinc-500 underline-offset-2 hover:underline" onClick={() => {
              if (!asJson) setJsonText(JSON.stringify((() => { try { return buildInput(fields, values) } catch { return {} } })(), null, 2))
              setAsJson(!asJson)
            }}>{asJson ? 'Use the form' : 'Edit as JSON'}</button>
          }>
            {asJson ? (
              <textarea value={jsonText} onChange={(e) => setJsonText(e.target.value)} rows={6} spellCheck={false}
                className={`${inputCls} font-mono text-xs`} />
            ) : fields.length === 0 ? (
              <p className="text-xs text-zinc-500">It takes no input.</p>
            ) : (
              <div className="space-y-3">
                {fields.map((f) => (
                  <Field key={f.key} label={<>{f.key}{f.required ? '' : <span className="text-zinc-600"> (optional)</span>}</>}
                    hint={f.prop.description || (f.kind === 'file' ? 'A file in your VFS, like vfs:reports/march.csv' : f.kind === 'files' ? 'VFS files, one per line' : undefined)}>
                    {f.kind === 'boolean' ? (
                      <input type="checkbox" checked={!!values[f.key]} onChange={(e) => setValues({ ...values, [f.key]: e.target.checked })} />
                    ) : f.kind === 'enum' ? (
                      <select className={inputCls} value={String(values[f.key] ?? '')} onChange={(e) => setValues({ ...values, [f.key]: e.target.value })}>
                        <option value="">—</option>
                        {(f.prop.enum || []).map((v) => <option key={String(v)} value={String(v)}>{String(v)}</option>)}
                      </select>
                    ) : f.kind === 'files' || f.kind === 'json' ? (
                      <textarea rows={3} className={`${inputCls} font-mono text-xs`} value={String(values[f.key] ?? '')}
                        placeholder={f.kind === 'json' ? 'JSON' : 'vfs:folder/file'} onChange={(e) => setValues({ ...values, [f.key]: e.target.value })} />
                    ) : (
                      <input className={inputCls} type={f.kind === 'number' ? 'number' : 'text'} value={String(values[f.key] ?? '')}
                        placeholder={f.kind === 'file' ? 'vfs:folder/file' : ''} onChange={(e) => setValues({ ...values, [f.key]: e.target.value })} />
                    )}
                  </Field>
                ))}
              </div>
            )}
            {callsBack && (
              <Field label="Run as" hint="While it runs it asks this agent to use its tools, as you. Its requests go to a session of their own, not your chat.">
                {agents && agents.length === 0 ? (
                  <p className="text-xs text-amber-600 dark:text-amber-400">You have no agent to run it with.</p>
                ) : (
                  <select aria-label="Run as" className={inputCls} value={runAs} onChange={(e) => setRunAs(e.target.value)}>
                    {(agents || []).map((a) => (
                      <option key={a.slug} value={a.slug} disabled={a.status === 'stopped'}>
                        {a.name}{a.status === 'asleep' ? ' (asleep: it wakes)' : a.status === 'stopped' ? ' (off)' : ''}
                      </option>
                    ))}
                  </select>
                )}
              </Field>
            )}
            <Button variant="primary" icon={Play} busy={busy === 'run'} onClick={run} disabled={!tool.live || (callsBack && !runAs)}>Run</Button>
            {result && (
              result.ok ? (
                <div className="space-y-2">
                  <div className="text-xs text-zinc-500">Ran in {result.seconds} s (version {result.version.slice(0, 7)})</div>
                  <pre className="max-h-72 overflow-auto rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs text-zinc-200">{JSON.stringify(result.output, null, 2)}</pre>
                  {(result.files || []).length > 0 && (
                    <ul className="space-y-1 text-xs text-zinc-400">
                      {(result.files || []).map((f) => <li key={f.path} className="font-mono">{f.path} <span className="text-zinc-600">({f.size} bytes)</span></li>)}
                    </ul>
                  )}
                </div>
              ) : (
                <Notice tone="red">
                  <div className="font-medium">{result.error?.message}</div>
                  {result.stderr && <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap text-[11px] opacity-80">{result.stderr}</pre>}
                </Notice>
              )
            )}
          </Section>
        )}

        {tool.secrets.length > 0 && (mine || tool.added) && (
          <Section icon={KeyRound} title="Your secrets for it">
            <p className="text-xs text-zinc-500">Kept sealed in your account and given to this tool only when you (or your agents) run it. Nobody else sees them, the tool’s maker included.</p>
            {tool.secrets.map((s) => (
              <Field key={s.name} label={<span className="font-mono">{s.name} {s.set ? <Pill tone="emerald">set</Pill> : <Pill tone="amber">not set</Pill>}</span>} hint={s.why}>
                <div className="flex gap-2">
                  <input type="password" autoComplete="off" className={inputCls} placeholder={s.set ? 'Enter a new value to replace it' : 'Value'}
                    value={secretDraft[s.name] || ''} onChange={(e) => setSecretDraft({ ...secretDraft, [s.name]: e.target.value })} />
                  {s.set && <Button variant="ghost" icon={X} aria-label={`Clear ${s.name}`}
                    onClick={() => act(`clear-${s.name}`, () => customToolsApi.setSecrets(slug, { [s.name]: '' }, mine ? undefined : owner))} />}
                </div>
              </Field>
            ))}
            <Button icon={KeyRound} busy={busy === 'secrets'} disabled={!Object.values(secretDraft).some(Boolean)} onClick={saveSecrets}>Save secrets</Button>
          </Section>
        )}

        {mine && (
          <Section icon={UploadCloud} title="Publishing" action={tool.can_build ? (
            <Button icon={UploadCloud} busy={busy === 'publish' || tool.publish?.state === 'running'}
              onClick={() => act('publish', () => customToolsApi.publish(slug))}>Publish main</Button>
          ) : undefined}>
            <p className="text-xs text-zinc-500">A version goes live only when its tests and every example pass. Crew publishes after you approve a merge; this publishes what’s on main now.</p>
            {tool.publish && (
              <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-3 text-xs">
                <button className="flex w-full items-center gap-2 text-left text-zinc-300" onClick={() => setSteps(!steps)}>
                  {steps ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                  Last publish: <span className={tool.publish.state === 'failed' ? 'text-red-500' : tool.publish.state === 'passed' ? 'text-emerald-500' : 'text-amber-500'}>{tool.publish.state}</span>
                  {tool.publish.error && <span className="truncate text-zinc-500">— {tool.publish.error}</span>}
                </button>
                {steps && (
                  <ol className="mt-2 space-y-2">
                    {(tool.publish.steps || []).map((s, i) => (
                      <li key={i}>
                        <div className={s.ok ? 'text-zinc-300' : 'text-red-500'}>{s.ok ? '✓' : '✗'} {s.step}</div>
                        {s.detail && <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-[11px] text-zinc-500">{s.detail}</pre>}
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            )}
          </Section>
        )}

        {mine && (tool.versions || []).length > 0 && (
          <Section icon={History} title="Versions">
            <ul className="space-y-1.5">
              {(tool.versions || []).map((v) => (
                <li key={v.version} className="flex items-center justify-between gap-2 text-xs">
                  <span className="font-mono text-zinc-300">{v.version.slice(0, 7)} {v.built_at ? <span className="text-zinc-600">· {new Date(v.built_at * 1000).toLocaleString()}</span> : null}</span>
                  {v.live ? <Pill tone="emerald">live</Pill> : (
                    <Button variant="ghost" busy={busy === `live-${v.version}`} onClick={() => act(`live-${v.version}`, () => customToolsApi.makeLive(slug, v.version))}>Make live</Button>
                  )}
                </li>
              ))}
            </ul>
          </Section>
        )}

        {mine && tool.can_build && (
          <Section icon={Hammer} title="Change it">
            <p className="text-xs text-zinc-500">A Crew worker changes it in its own repository. It goes live after you approve the merge on the Crew page and its tests and examples pass.</p>
            <textarea rows={3} className={inputCls} placeholder="What to change, or the failure to fix" value={what} onChange={(e) => setWhat(e.target.value)} />
            <div className="flex flex-wrap items-center gap-2">
              <HarnessPicker value={harness} onChange={setHarness} />
              <Button icon={Hammer} busy={busy === 'improve'} disabled={what.trim().length < 10 || !harness}
                onClick={() => act('improve', () => customToolsApi.improve(slug, { what, harness }), () => {
                  setWhat('')
                  setNote('A Crew task is changing it. Approve its merge on the Crew page when it’s done.')
                })}>Start</Button>
            </div>
          </Section>
        )}

        {mine && authEnabled && sharing && (
          <Section icon={Share2} title="Offer it" action={<Button icon={Share2} onClick={() => setShareOpen(true)} disabled={!tool.live}>People…</Button>}>
            <p className="text-xs text-zinc-500">People you offer it to can use it, never change it. Each of them adds it first, and it runs as them, with their secrets and files.</p>
            {sharing.people.length > 0 && (
              <div className="flex flex-wrap gap-1.5">{sharing.people.map((p) => <Pill key={p.grantee_id}>{p.grantee_name || p.grantee_email}</Pill>)}</div>
            )}
            <div className="flex items-center justify-between gap-3 rounded-xl border border-zinc-800 p-3">
              <div className="min-w-0">
                <div className="text-sm text-zinc-200">Everybody on this deck</div>
                <div className="text-xs text-zinc-500">
                  {!sharing.everybody ? (sharing.needs_approval ? 'An admin approves it first.' : 'Everyone is offered it at once.')
                    : sharing.everybody.status === 'rejected' ? 'An admin declined it. Turn it off and on to ask again.'
                      : sharing.everybody.live ? 'Offered to everybody.' : 'Waiting for an admin to approve it.'}
                </div>
              </div>
              <Switch on={!!sharing.everybody} busy={busy === 'everybody'} label="Offer to everybody"
                onChange={(on) => act('everybody', () => customToolsApi.everybody(slug, on))} />
            </div>
            <p className="text-xs text-zinc-500">{sharing.added_by === 1 ? '1 person has added it.' : `${sharing.added_by} people have added it.`}</p>
            {(tool.depends_on || []).length > 0 && (sharing.people.length > 0 || !!sharing.everybody) && (
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-zinc-800 p-3">
                <span className="min-w-0 text-xs text-zinc-500">It runs {(tool.depends_on || []).map((d) => d.title).join(', ')}: offer {(tool.depends_on || []).length === 1 ? 'it' : 'them'} to the same people so it works for them.</span>
                <Button icon={Share2} busy={busy === 'deps'} onClick={() => act('deps', () => customToolsApi.offerDependencies(slug), () => setNote('Offered the tools it runs the same way.'))}>Offer them too</Button>
              </div>
            )}
          </Section>
        )}

        {mine && usage && (
          <Section icon={BarChart3} title="How it’s used">
            <p className="text-xs text-zinc-500">
              {usage.calls === 1 ? '1 call' : `${usage.calls} calls`} in the last 30 days{usage.failed ? `, ${usage.failed} failed` : ''}, by {usage.people} {usage.people === 1 ? 'person' : 'people'}.
              {Object.keys(usage.failures).length > 0 && ` Failures: ${Object.entries(usage.failures).map(([k, n]) => `${k} ${n}`).join(', ')}.`}
              {' '}You never see anyone else’s inputs or results.
            </p>
            {usage.days.length > 0 && (
              <div className="flex h-16 items-end gap-1">
                {usage.days.slice(-14).map((d) => {
                  const top = Math.max(...usage.days.slice(-14).map((x) => x.calls), 1)
                  return <div key={d.day} title={`${d.day}: ${d.calls} calls, ${d.failed} failed`} className="flex-1 rounded-t bg-violet-500/60"
                    style={{ height: `${Math.max(6, (d.calls / top) * 100)}%` }} />
                })}
              </div>
            )}
          </Section>
        )}

        {!mine && tool.added && (
          <Button icon={X} busy={busy === 'unadd'} onClick={() => act('unadd', () => customToolsApi.unadd(owner || '', slug))}>Remove from my tools</Button>
        )}

        {mine && (
          <div className="border-t border-zinc-800/80 pt-4">
            <Button variant="danger" icon={Trash2} busy={busy === 'delete'} onClick={() => {
              const others = sharing?.added_by ? ` ${sharing.added_by} other ${sharing.added_by === 1 ? 'person uses' : 'people use'} it.` : ''
              if (!confirm(`Delete “${tool.title}”, its versions and its repository?${others} This can't be undone.`)) return
              setBusy('delete')
              customToolsApi.remove(slug).then(onDeleted).catch((e) => { setError(e instanceof Error ? e.message : 'Failed'); setBusy('') })
            }}>Delete tool</Button>
          </div>
        )}
      </div>

      {shareOpen && (
        <ShareModal resourceType="external_tool" resourceId={`${tool.owner}/${slug}`} resourceName={tool.title} allowEdit={false}
          note="They can use it, never change it, and add it before their agents run it."
          onClose={() => { setShareOpen(false); load() }} />
      )}
    </Sheet>
  )
}
