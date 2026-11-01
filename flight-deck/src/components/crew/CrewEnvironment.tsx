import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Boxes, CheckCircle2, ChevronRight, CircleDashed, Save, Sparkles, TerminalSquare } from 'lucide-react'
import { Card, Pill, labelCls } from '../console/parts'
import { crewApi, type CrewEnv, type CrewMe } from '../../services/crewApi'
import { CopyChip, ErrorLine, errorOf, fieldClass, primaryBtn, quietBtn } from './crewParts'

// ── What a project's work runs in ──
//
// What's installed, for any kind of project: the deck only reports what it finds, and
// what isn't a common layout comes from these settings. It's prepared in the project's
// main folder, by the person in a terminal or over SSH, or by a coding agent told to.

const varsText = (vars: Record<string, string>) => Object.entries(vars).map(([k, v]) => `${k}=${v}`).join('\n')

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-0.5 text-sm">
      <dt className="shrink-0 text-xs text-zinc-500">{label}</dt>
      <dd className="min-w-0 break-words text-zinc-200">{children}</dd>
    </div>
  )
}

const mono = 'font-mono text-[13px]'

function Settings({ env, onSaved }: { env: CrewEnv; onSaved: (env: CrewEnv) => void }) {
  const [test, setTest] = useState(env.test)
  const [run, setRun] = useState(env.run)
  const [copy, setCopy] = useState(env.copy.join(' '))
  const [vars, setVars] = useState(varsText(env.vars))
  const [activate, setActivate] = useState(env.activate)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  const save = async () => {
    setBusy(true); setError(''); setSaved(false)
    try {
      onSaved(await crewApi.saveEnv(env.id, { test, run, copy_paths: copy, vars, activate }))
      setSaved(true)
    } catch (e) {
      setError(errorOf(e))
    }
    setBusy(false)
  }

  return (
    <div className="mt-3 space-y-3">
      <div className="grid gap-3 sm:grid-cols-2">
        <label className={labelCls}>
          Command that runs its tests
          <input value={test} onChange={(e) => setTest(e.target.value)} aria-label="Test command" className={`${fieldClass} mt-1.5 font-normal ${mono}`} />
        </label>
        <label className={labelCls}>
          Command that starts it
          <input value={run} onChange={(e) => setRun(e.target.value)} aria-label="Start command" className={`${fieldClass} mt-1.5 font-normal ${mono}`} />
        </label>
      </div>
      <label className={labelCls}>
        Other folders or files each task needs a copy of
        <input value={copy} onChange={(e) => setCopy(e.target.value)} placeholder="From the project's root, separated by spaces" aria-label="Also copy" className={`${fieldClass} mt-1.5 font-normal ${mono}`} />
      </label>
      <label className={labelCls}>
        Variables
        <textarea value={vars} onChange={(e) => setVars(e.target.value)} rows={3} placeholder="NAME=value, one on each line" aria-label="Variables" className={`${fieldClass} mt-1.5 resize-y font-normal ${mono}`} />
      </label>
      <label className={labelCls}>
        Script to run before each turn
        <textarea value={activate} onChange={(e) => setActivate(e.target.value)} rows={3} placeholder="For a version manager, say" aria-label="Script" className={`${fieldClass} mt-1.5 resize-y font-normal ${mono}`} />
      </label>
      <p className="text-xs leading-relaxed text-zinc-500">
        These stay on this computer, outside the project, as you typed them: they aren't encrypted, and its workers can read them.
      </p>
      <ErrorLine text={error} />
      <div className="flex items-center gap-3">
        <button className={quietBtn} disabled={busy} onClick={save}><Save className="h-4 w-4" /> Save</button>
        {saved && <span className="text-xs text-emerald-700 dark:text-emerald-300">Saved. New tasks get this.</span>}
      </div>
    </div>
  )
}

export function EnvironmentCard({ env, me, onChanged, onOpenTask, onTerminal }: {
  env: CrewEnv
  me: CrewMe | null
  onChanged: () => void
  onOpenTask: (id: string) => void
  onTerminal: (env: CrewEnv) => void
}) {
  const usable = useMemo(() => (me?.harnesses ?? []).filter((h) => h.installed && h.login.ok), [me?.harnesses])
  const [harness, setHarness] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => { setHarness((old) => old || usable[0]?.id || '') }, [usable])

  const prepare = async () => {
    setBusy(true); setError('')
    try {
      onOpenTask((await crewApi.dispatch({ harness, brief: '', kind: 'setup', ...env.place })).id)
    } catch (e) {
      setError(errorOf(e))
      setBusy(false)
    }
  }
  const markCurrent = async () => {
    setBusy(true); setError('')
    try {
      await crewApi.envCurrent(env.id)
      onChanged()
    } catch (e) {
      setError(errorOf(e))
    }
    setBusy(false)
  }

  const standing = env.stale.length ? <Pill tone="amber" icon={AlertTriangle}>May be out of date</Pill>
    : env.ready && env.missing.length ? <Pill tone="amber" icon={AlertTriangle}>Partly prepared</Pill>
    : env.ready ? <Pill tone="emerald" icon={CheckCircle2}>Prepared</Pill>
      : <Pill icon={CircleDashed}>Not prepared</Pill>

  return (
    <Card title={`Environment${env.place.folder && env.place.folder !== env.project_name ? `: ${env.place.folder}` : ''}`} icon={Boxes} action={standing} bodyClassName="space-y-4 p-4">
      <p className="text-sm leading-relaxed text-zinc-400">
        {env.ready
          ? `What its tasks build, run and test with. New tasks get it by themselves.${env.missing.length ? ' Something it looks like it needs isn\'t there yet.' : ''}`
          : 'Nothing is installed yet, so its tasks can edit code but may not be able to run it. Set it up in this folder the way you would any project, or have a coding agent do it. Every new task then gets it.'}
      </p>

      <dl className="space-y-2">
        <Fact label="Folder"><CopyChip text={env.repo} /></Fact>
        {env.kinds.length > 0 && <Fact label="Looks like">{env.kinds.join(', ')}</Fact>}
        {env.python.map((p) => (
          <Fact key={p.path} label="Python">
            <span className={mono}>{p.path}</span>
            <span className="text-zinc-500"> · {p.version ? `${p.version} · ` : ''}{p.packages} packages · shared by its tasks</span>
          </Fact>
        ))}
        {env.folders.length > 0 && (
          <Fact label="Packages">
            <span className={mono}>{env.folders.map((f) => f.path).join(', ')}</span>
            <span className="text-zinc-500"> · each task gets its own copy</span>
          </Fact>
        )}
        {env.copy.length > 0 && <Fact label="Also copied"><span className={mono}>{env.copy.join(', ')}</span></Fact>}
        {env.missing.length > 0 && <Fact label="Not there yet"><span className="text-amber-700 dark:text-amber-300">{env.missing.join(', ')}</span></Fact>}
        {env.test && <Fact label="Tests"><span className={mono}>{env.test}</span></Fact>}
        {env.run && <Fact label="Start"><span className={mono}>{env.run}</span></Fact>}
      </dl>

      {env.stale.length > 0 && (
        <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-amber-500/30 bg-amber-500/[0.06] px-4 py-3 text-sm text-amber-800 dark:text-amber-200">
          <span className="min-w-0 flex-1 basis-56 break-words">{env.stale.join(', ')} changed after it was prepared, so what's installed may not match.</span>
          <button className={quietBtn} disabled={busy} onClick={markCurrent}>It's up to date</button>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {me?.terminal && (
          <button className={quietBtn} disabled={!env.exists} onClick={() => onTerminal(env)}><TerminalSquare className="h-4 w-4" /> Open a terminal here</button>
        )}
        {usable.length > 0 && (
          <span className="inline-flex items-center gap-2">
            <button className={env.ready ? quietBtn : primaryBtn} disabled={busy || !harness} onClick={prepare}>
              <Sparkles className="h-4 w-4" /> {env.ready ? 'Update it with' : 'Prepare it with'}
            </button>
            <select value={harness} onChange={(e) => setHarness(e.target.value)} aria-label="Which coding agent" className={`${fieldClass} !w-auto`}>
              {usable.map((h) => <option key={h.id} value={h.id}>{h.label}</option>)}
            </select>
          </span>
        )}
      </div>
      {!me?.terminal && (
        <p className="text-xs leading-relaxed text-zinc-500">To set it up by hand, reach this folder over SSH, or ask an admin to turn on the terminal for you.</p>
      )}
      <ErrorLine text={error} />

      <details className="group/env text-xs">
        <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-zinc-500 transition-colors hover:text-zinc-300 [&::-webkit-details-marker]:hidden">
          <ChevronRight className="h-3.5 w-3.5 transition-transform group-open/env:rotate-90" /> Settings for this project
        </summary>
        <Settings key={`${env.id}|${env.test}|${env.run}|${env.copy.join()}`} env={env} onSaved={() => onChanged()} />
      </details>
    </Card>
  )
}
