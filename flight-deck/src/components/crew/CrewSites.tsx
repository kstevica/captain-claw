import { useEffect, useMemo, useState } from 'react'
import {
  Activity, ArrowLeft, Check, ExternalLink, FileCode2, FolderOpen, Globe, Loader2, Play, Plus, Server, Square, Terminal, Trash2,
} from 'lucide-react'
import { EmptyState, Pill, Segmented, Sheet, labelCls } from '../console/parts'
import { AppLogs, SiteVisits } from '../../pages/HostingPage'
import { crewApi, type CrewMe, type CrewTask } from '../../services/crewApi'
import { hostingApi, type HostedSite } from '../../services/hostingApi'
import {
  ChoiceTile, CopyChip, ErrorLine, dangerBtn, errorOf, fieldClass, fullUrl, hasPage, primaryBtn, quietBtn, sureDangerBtn, taskOfSite,
} from './crewParts'
import { libraryIconBtn } from '../library/libraryParts'

// ── Live sites ──
//
// What this person has put on the web, whoever put it there: them, their agent, or
// a task's Files tab. The same two verbs for a page and an app: a stopped one
// keeps its address and comes back when it's started.

export const isLive = (s: HostedSite) => s.running && !s.missing

function SitePill({ site }: { site: HostedSite }) {
  if (site.missing) return <Pill tone="amber">Files gone</Pill>
  if (!site.running) return <Pill tone="zinc">Stopped</Pill>
  return (
    <Pill tone="emerald">
      <span className="relative flex h-1.5 w-1.5">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
        <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
      </span>
      Live
    </Pill>
  )
}

export function SiteIcon({ site }: { site: Pick<HostedSite, 'kind'> }) {
  const Icon = site.kind === 'app' ? Server : FileCode2
  return (
    <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${site.kind === 'app' ? 'bg-cyan-500/10 text-cyan-700 dark:text-cyan-300' : 'bg-violet-500/10 text-violet-700 dark:text-violet-300'}`}>
      <Icon className="h-5 w-5" />
    </span>
  )
}

function SiteCard({ site, task, canRunApps, onChanged, onOpenTask }: {
  site: HostedSite; task?: CrewTask; canRunApps: boolean; onChanged: () => void; onOpenTask: (id: string) => void
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [more, setMore] = useState(false)
  const [sure, setSure] = useState(false)
  useEffect(() => {
    if (!sure) return
    const t = setTimeout(() => setSure(false), 6000)
    return () => clearTimeout(t)
  }, [sure])
  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError('')
    try {
      await fn()
      await onChanged()
    } catch (e) {
      setError(errorOf(e))
    }
    setBusy(false)
  }
  const app = site.kind === 'app'
  const startable = !site.missing && (!app || canRunApps)
  const source = site.subdir ? `${site.project}/${site.subdir}` : site.project
  return (
    <section className="fd-rise rounded-2xl border border-zinc-800 bg-zinc-900/40">
      <div className="flex flex-wrap items-start gap-3 p-4">
        <SiteIcon site={site} />
        <div className="min-w-0 flex-1 basis-56">
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="min-w-0 truncate text-[15px] font-semibold text-zinc-100">{site.name}</h4>
            <SitePill site={site} />
            <span className="text-xs text-zinc-500">{app ? 'App' : 'Page'}{site.visit_count > 0 ? ` · ${site.visit_count} visit${site.visit_count === 1 ? '' : 's'}` : ''}</span>
          </div>
          <div className="mt-1.5"><CopyChip text={fullUrl(site.url)} /></div>
          {task ? (
            <p className="mt-1.5 flex min-w-0 items-baseline gap-1 text-xs text-zinc-500">
              <span className="shrink-0">From</span>
              <button onClick={() => onOpenTask(task.id)} title={task.title} className="min-w-0 truncate text-zinc-300 underline-offset-2 hover:underline">{task.title}</button>
            </p>
          ) : <p className="mt-1.5 truncate text-xs text-zinc-500" title={source}>From your folder {source}</p>}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {isLive(site) && <a href={site.url} target="_blank" rel="noreferrer" className={quietBtn}><ExternalLink className="h-4 w-4" /> Open</a>}
          {site.running && !site.missing && <button disabled={busy} onClick={() => act(() => hostingApi.stop(site.name))} className={quietBtn}>{busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />} Stop</button>}
          {!site.running && startable && <button disabled={busy} onClick={() => act(() => hostingApi.start(site.name))} className={primaryBtn}>{busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />} Start</button>}
          <button
            onClick={() => setMore((v) => !v)} aria-expanded={more} title={app ? 'Its output' : 'Its visits'} aria-label={app ? 'Its output' : 'Its visits'}
            className={`rounded-xl border p-2 transition-colors ${more ? 'border-violet-500/40 text-violet-600 dark:text-violet-300' : 'border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-200'}`}
          >
            {app ? <Terminal className="h-4 w-4" /> : <Activity className="h-4 w-4" />}
          </button>
          {sure ? (
            <button disabled={busy} onClick={() => { setSure(false); act(() => hostingApi.remove(site.name)) }} className={sureDangerBtn}><Check className="h-4 w-4" /> Yes, remove it</button>
          ) : (
            <button disabled={busy} onClick={() => setSure(true)} className={`${dangerBtn} !border-zinc-800 !p-2`} title="Remove it: its address goes too" aria-label="Remove it"><Trash2 className="h-4 w-4" /></button>
          )}
        </div>
        {!startable && !site.running && !site.missing && <p className="basis-full text-xs text-zinc-500">Only an admin can start an app on this Flight Deck.</p>}
        {site.missing && <p className="basis-full text-xs text-amber-700 dark:text-amber-300">The folder it served is gone, so there's nothing to show. Remove it, or publish something new.</p>}
        {error && <div className="basis-full"><ErrorLine text={error} /></div>}
      </div>
      {more && <div className="border-t border-zinc-800/80 p-4">{app ? <AppLogs name={site.name} /> : <SiteVisits name={site.name} />}</div>}
    </section>
  )
}

export function CrewSites({ sites, tasks, me, onBack, onChanged, onOpenTask, onPublish }: {
  sites: HostedSite[] | null; tasks: CrewTask[]; me: CrewMe | null; onBack?: () => void
  onChanged: () => void; onOpenTask: (id: string) => void; onPublish: () => void
}) {
  const list = sites ?? []
  const online = list.filter(isLive)
  const rest = list.filter((s) => !isLive(s))
  const card = (s: HostedSite) => (
    <SiteCard key={s.name} site={s} task={taskOfSite(s, tasks)} canRunApps={!!me?.trusted} onChanged={onChanged} onOpenTask={onOpenTask} />
  )
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl space-y-5 p-4 wide:p-6">
        <div className="flex flex-wrap items-start gap-3">
          {onBack && <button onClick={onBack} className={libraryIconBtn} aria-label="Back"><ArrowLeft className="h-4 w-4" /></button>}
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-violet-500/10 text-violet-600 dark:text-violet-400"><Globe className="h-5 w-5" /></span>
          <div className="min-w-0 flex-1 basis-56">
            <h3 className="text-xl font-semibold tracking-tight text-zinc-100">Live sites</h3>
            <p className="mt-0.5 text-sm text-zinc-500">What you've put on the web. Anyone with an address can open it, with no sign-in.</p>
          </div>
          <button onClick={onPublish} className={primaryBtn}><Plus className="h-4 w-4" /> Publish</button>
        </div>
        {sites === null && <div className="flex justify-center p-10 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>}
        {sites !== null && !list.length && (
          <EmptyState icon={Globe} title="Nothing on the web yet">
            When a task builds a page, publish it and you get an address to send. Your agent can do it too: "put that page online".
          </EmptyState>
        )}
        {online.length > 0 && <div className="space-y-3">{online.map(card)}</div>}
        {rest.length > 0 && (
          <div className="space-y-3">
            {online.length > 0 && <h4 className="px-1 pt-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Not online</h4>}
            {rest.map(card)}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Publishing something new ─────────────────────────────────────────────

const toName = (text: string) => text.toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/^-+/, '').slice(0, 63)

export function PublishSheet({ tasks, sites, me, onClose, onDone }: {
  tasks: CrewTask[]; sites: HostedSite[]; me: CrewMe | null; onClose: () => void; onDone: () => void
}) {
  const hosted = useMemo(() => new Set(sites.map((s) => s.name)), [sites])
  const pages = useMemo(() => tasks.filter((t) => hasPage(t.results) && !hosted.has(t.results?.link?.name ?? '')), [tasks, hosted])
  const [source, setSource] = useState<'task' | 'folder'>(pages.length ? 'task' : 'folder')
  const [taskId, setTaskId] = useState(pages[0]?.id ?? '')
  const [projects, setProjects] = useState<string[]>([])
  const [project, setProject] = useState('')
  const [folders, setFolders] = useState<string[]>([''])
  const [folder, setFolder] = useState('')
  const [kind, setKind] = useState<'static' | 'app'>('static')
  const [startCmd, setStartCmd] = useState('')
  const [name, setName] = useState('')
  const [named, setNamed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    hostingApi.projects().then((r) => {
      const names = r.projects.map((p) => p.name)
      setProjects(names)
      setProject((p) => p || names[0] || '')
    }).catch((e) => setError(errorOf(e)))
  }, [])
  useEffect(() => {
    if (source !== 'folder' || !project) return
    setFolder('')
    hostingApi.folders(project).then((r) => setFolders(r.folders.length ? r.folders : [''])).catch(() => setFolders(['']))
  }, [source, project])

  // A name to start from, until they type their own.
  const chosen = pages.find((t) => t.id === taskId)
  useEffect(() => {
    if (named) return
    const word = source === 'task' ? chosen?.project_name : (folder.split('/').pop() || project)
    setName(toName(word || '').replace(/-+$/, ''))
  }, [named, source, chosen?.project_name, folder, project])

  const taken = hosted.has(name)
  const app = source === 'folder' && kind === 'app'
  const ready = !!name && !taken && (source === 'task' ? !!chosen : !!project) && (!app || !!startCmd.trim())

  const submit = async () => {
    setBusy(true); setError('')
    try {
      if (source === 'task') {
        await crewApi.link(taskId, name)
      } else {
        await hostingApi.publish({ name, kind, project, subdir: folder, start_cmd: startCmd.trim() })
        if (app) await hostingApi.start(name)
      }
      onDone()
    } catch (e) {
      setError(errorOf(e))
      setBusy(false)
    }
  }

  const prefix = `${window.location.origin}/${app ? 'vfs-apps' : 'vfs'}/`
  return (
    <Sheet
      icon={Globe} title="Publish a site" subtitle="Put a page or an app on the web, at an address of its own." onClose={onClose}
      footer={(
        <>
          <button onClick={onClose} className={quietBtn}>Cancel</button>
          <button onClick={submit} disabled={busy || !ready} className={primaryBtn}>{busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Globe className="h-4 w-4" />} Publish</button>
        </>
      )}
    >
      <div className="space-y-5">
        <div>
          <p className={labelCls}>What to publish</p>
          <Segmented
            label="What to publish" value={source} onChange={setSource}
            options={[{ value: 'task', label: 'What a task made', count: pages.length }, { value: 'folder', label: 'One of your folders' }]}
          />
        </div>

        {source === 'task' && (pages.length ? (
          <div className="space-y-2">
            {pages.map((t) => <ChoiceTile key={t.id} icon={FileCode2} title={<span className="line-clamp-2">{t.title}</span>} hint={t.project_name} on={taskId === t.id} onClick={() => setTaskId(t.id)} />)}
          </div>
        ) : (
          <p className="rounded-2xl border border-dashed border-zinc-800 p-5 text-center text-sm text-zinc-500">
            No task has made a page yet: it needs an <code className="rounded bg-zinc-800 px-1 text-xs">index.html</code>.
          </p>
        ))}

        {source === 'folder' && (
          <>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className={labelCls}>Project
                <select value={project} onChange={(e) => setProject(e.target.value)} className={`${fieldClass} mt-1.5 font-normal`}>
                  {!projects.length && <option value="">No projects yet</option>}
                  {projects.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </label>
              <label className={labelCls}>Folder
                <select value={folder} onChange={(e) => setFolder(e.target.value)} className={`${fieldClass} mt-1.5 font-normal`}>
                  {folders.map((f) => <option key={f || '_top'} value={f}>{f || 'The whole project'}</option>)}
                </select>
              </label>
            </div>
            {me?.trusted && (
              <div className="grid gap-2 sm:grid-cols-2">
                <ChoiceTile icon={FolderOpen} title="A page" hint="Its files, as they are. Needs an index.html." on={kind === 'static'} onClick={() => setKind('static')} />
                <ChoiceTile icon={Server} title="An app that runs" hint="Starts a program here, and passes visitors to it." on={kind === 'app'} onClick={() => setKind('app')} />
              </div>
            )}
            {app && (
              <label className={labelCls}>How to start it
                <input value={startCmd} onChange={(e) => setStartCmd(e.target.value)} placeholder="npm run start" spellCheck={false} className={`${fieldClass} mt-1.5 font-mono font-normal`} />
                <span className="mt-1.5 block font-normal leading-relaxed text-zinc-500">It has to listen on the port in <code className="rounded bg-zinc-800 px-1">PORT</code>.</span>
              </label>
            )}
          </>
        )}

        <label className={labelCls}>Its address
          <input
            value={name} onChange={(e) => { setName(toName(e.target.value)); setNamed(true) }}
            placeholder="my-site" spellCheck={false} autoComplete="off" className={`${fieldClass} mt-1.5 font-normal`}
          />
          <span className={`mt-1.5 block break-all font-normal ${taken ? 'text-red-600 dark:text-red-400' : 'text-zinc-500'}`}>
            {taken ? `You already have a site called ${name}.` : <>{prefix}<span className="text-zinc-300">{name || 'my-site'}</span>/</>}
          </span>
        </label>
        <ErrorLine text={error} />
      </div>
    </Sheet>
  )
}
