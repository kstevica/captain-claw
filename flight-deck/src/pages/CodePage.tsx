import { useEffect, useRef, useState } from 'react'
import {
  Plus, FolderGit2, Send, GitCommit, Loader2, Bot, User, ClipboardList, CheckCircle2,
  ShieldAlert, Wrench, RotateCcw, X, Download, ChevronRight, ChevronDown, Link2, Lock,
  MessageSquarePlus, FolderPlus, FolderTree, Trash2, Map, Square, Eraser,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useCodeStore } from '../stores/codeStore'
import { useVFSStore } from '../stores/vfsStore'

const SEV_COLOR: Record<string, string> = {
  blocking: 'text-red-400', major: 'text-amber-400', minor: 'text-zinc-400',
}

function diffLineClass(line: string): string {
  if (line.startsWith('+') && !line.startsWith('+++')) return 'text-emerald-400'
  if (line.startsWith('-') && !line.startsWith('---')) return 'text-red-400'
  if (line.startsWith('@@')) return 'text-sky-400'
  if (line.startsWith('diff ') || line.startsWith('index ')) return 'text-zinc-500'
  return 'text-zinc-400'
}

export function CodePage() {
  const {
    projects, activeProject, activeSession, messages, commits, progress, status, sending, error,
    loadProjects, createProject, addFolder, linkFolder, createSession, deleteSession,
    setSessionFolder, selectSession, send, approvePlan, cancelPlan, showCommit, rollback, exportProcess,
    loadMap, searchMap, buildMap, stopRun, cleanupAgents,
  } = useCodeStore()
  const browseFs = useVFSStore((s) => s.browseFs)

  const [tab, setTab] = useState<'chat' | 'map'>('chat')
  const [map, setMap] = useState<import('../stores/codeStore').CodeMap | null>(null)
  const [mapQ, setMapQ] = useState('')
  const [mapHits, setMapHits] = useState<import('../stores/codeStore').CodeMapHit[]>([])
  const [mapBusy, setMapBusy] = useState(false)

  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [foldersFor, setFoldersFor] = useState<string>('')     // project whose folders panel is open
  const [newVfsFolder, setNewVfsFolder] = useState('')
  const [cleanupMsg, setCleanupMsg] = useState('')
  // New-session modal: project → folder → session in one flow.
  const [modal, setModal] = useState<null | {
    project: string          // existing project name; '' = create new
    newProject: string
    folderMode: 'existing' | 'new' | 'link'
    folder: string           // existing folder pick
    newFolder: string
    linkPath: string
    linkMode: 'rw' | 'ro'
    title: string
    busy: boolean
  }>(null)
  const [input, setInput] = useState('')
  const [planDraft, setPlanDraft] = useState('')
  const [diff, setDiff] = useState<{ sha: string; text: string } | null>(null)
  // Folder browser (for linking)
  const [fs, setFs] = useState<{ project: string; path: string; parent: string;
    dirs: { name: string; hidden: boolean; is_git: boolean }[] } | null>(null)
  const chatEnd = useRef<HTMLDivElement>(null)

  useEffect(() => { loadProjects() }, [loadProjects])
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, progress])
  useEffect(() => {
    if (status === 'awaiting_plan') {
      const plan = [...messages].reverse().find((m) => m.kind === 'plan')
      setPlanDraft(plan?.text || '')
    }
  }, [status, messages])
  // Load the code map when the Map tab is open (or session/status changes).
  useEffect(() => {
    if (tab === 'map' && activeSession) loadMap().then(setMap)
  }, [tab, activeSession, status, loadMap])

  const onMapSearch = async (q: string) => {
    setMapQ(q)
    setMapHits(q.trim() ? await searchMap(q) : [])
  }
  const onRebuildMap = async () => {
    setMapBusy(true)
    try {
      await buildMap()
      setMap(await loadMap())   // reload the freshly-written overview/models/ui
    } finally { setMapBusy(false) }
  }

  const activeProj = projects.find((p) => p.name === activeProject)
  const activeSess = activeProj?.sessions.find((s) => s.id === activeSession)
  const activeFolders = activeProj?.folders || []

  const toggle = (name: string) =>
    setCollapsed((s) => { const n = new Set(s); n.has(name) ? n.delete(name) : n.add(name); return n })

  const openDiff = async (sha: string) => {
    setDiff({ sha, text: 'Loading…' })
    setDiff({ sha, text: (await showCommit(sha)) || '(no changes)' })
  }
  const onRollback = async (sha: string, message: string) => {
    if (!window.confirm(`Roll back to "${message}"?\nThis discards everything after this commit.`)) return
    await rollback(sha); setDiff(null)
  }
  const onSend = async () => {
    const text = input.trim()
    if (!text || sending) return
    setInput(''); await send(text)
  }
  const openModal = (project = '') => {
    const p = projects.find((x) => x.name === project) || projects.find((x) => x.name === activeProject)
    const hasFolders = !!p?.folders.length
    setModal({
      project: p?.name || '',
      newProject: '',
      folderMode: hasFolders ? 'existing' : 'new',
      folder: p?.folders[0]?.name || '',
      newFolder: '',
      linkPath: '',
      linkMode: 'rw',
      title: '',
      busy: false,
    })
  }
  const onModalProject = (name: string) => {
    if (!modal) return
    const p = projects.find((x) => x.name === name)
    const hasFolders = !!p?.folders.length
    setModal({
      ...modal, project: name,
      folderMode: hasFolders ? 'existing' : (modal.folderMode === 'existing' ? 'new' : modal.folderMode),
      folder: p?.folders[0]?.name || '',
    })
  }
  const linkNameOf = (path: string) =>
    (path.split('/').pop() || 'folder').replace(/[^a-zA-Z0-9._-]/g, '-')
  const modalValid = !!modal && !modal.busy
    && !!(modal.project || modal.newProject.trim())
    && !!(modal.folderMode === 'existing' ? modal.folder
        : modal.folderMode === 'new' ? modal.newFolder.trim()
        : modal.linkPath)
  const submitModal = async () => {
    if (!modal || !modalValid) return
    const projName = modal.project || modal.newProject.trim()
    setModal({ ...modal, busy: true })
    try {
      if (!modal.project) await createProject(projName)
      let folderName = modal.folder
      if (modal.folderMode === 'new') {
        folderName = modal.newFolder.trim()
        await addFolder(projName, folderName)
      } else if (modal.folderMode === 'link') {
        folderName = linkNameOf(modal.linkPath)
        await linkFolder(projName, folderName, modal.linkPath, modal.linkMode)
      }
      await createSession(projName, modal.title, folderName)
      setModal(null)
    } catch {
      setModal((m) => (m ? { ...m, busy: false } : m))
    }
  }
  const onCleanup = async () => {
    const n = await cleanupAgents()
    setCleanupMsg(n ? `Removed ${n} leftover agent${n === 1 ? '' : 's'}.` : 'No leftover agents.')
    setTimeout(() => setCleanupMsg(''), 4000)
  }
  const fsGo = async (project: string, path: string) => {
    const r = await browseFs(path)
    setFs({ project, path: r.path, parent: r.parent, dirs: r.dirs })
  }
  const useFolderForLink = async () => {
    if (!fs) return
    if (fs.project === '@modal') {
      // Browsing from the new-session modal: capture the path, don't link yet.
      const path = fs.path
      setFs(null)
      setModal((m) => (m ? { ...m, linkPath: path } : m))
      return
    }
    const name = linkNameOf(fs.path)
    const proj = fs.project, path = fs.path
    setFs(null)
    await linkFolder(proj, name, path, 'rw')
  }

  return (
    <div className="flex h-full overflow-hidden text-zinc-100">
      {/* ── Left rail: project → sessions ── */}
      <aside className="flex w-72 shrink-0 flex-col border-r border-zinc-800 bg-zinc-950/40">
        <div className="border-b border-zinc-800 p-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-300">
            <FolderGit2 size={16} /> Code Projects
            <button onClick={onCleanup} title="Clean up leftover code agents (after crashes/stops)"
              className="ml-auto rounded px-1.5 py-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
              <Eraser size={14} />
            </button>
          </div>
          <button onClick={() => openModal()}
            className="mt-2 flex w-full items-center justify-center gap-1.5 rounded bg-zinc-100 px-2 py-1.5 text-xs font-medium text-zinc-900 hover:bg-zinc-200">
            <MessageSquarePlus size={14} /> New session
          </button>
          {cleanupMsg && <div className="mt-2 text-[11px] text-zinc-500">{cleanupMsg}</div>}
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {projects.length === 0 && <div className="px-2 py-4 text-xs text-zinc-500">No projects yet. Create one to start.</div>}
          {projects.map((p) => {
            const open = !collapsed.has(p.name)
            return (
              <div key={p.name} className="group/proj mb-1">
                <div className="flex items-center gap-1 rounded px-1 py-1 hover:bg-zinc-900">
                  <button onClick={() => toggle(p.name)} className="flex min-w-0 flex-1 items-center gap-1 text-left text-xs font-semibold uppercase tracking-wide text-zinc-400">
                    {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                    <span className="truncate">{p.name}</span>
                    <span className="text-[10px] font-normal text-zinc-600">· {p.sessions.length}</span>
                  </button>
                  <button onClick={() => openModal(p.name)} title="New session in this project"
                    className="shrink-0 text-zinc-500 opacity-0 hover:text-zinc-200 group-hover/proj:opacity-100"><MessageSquarePlus size={13} /></button>
                  <button onClick={() => setFoldersFor((f) => f === p.name ? '' : p.name)} title="Folders"
                    className={`shrink-0 hover:text-zinc-200 ${foldersFor === p.name ? 'text-zinc-200' : 'text-zinc-500 opacity-0 group-hover/proj:opacity-100'}`}><FolderPlus size={13} /></button>
                </div>

                {open && (
                  <>
                    {/* sessions */}
                    {p.sessions.map((s) => (
                      <div key={s.id}
                        className={`group/sess ml-3 mb-0.5 flex w-[calc(100%-0.75rem)] items-start gap-1 rounded px-2 py-1.5 text-sm ${
                          p.name === activeProject && s.id === activeSession ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-300 hover:bg-zinc-900'}`}>
                        <button onClick={() => selectSession(p.name, s.id)} className="min-w-0 flex-1 text-left">
                          <span className="flex items-center gap-1 truncate font-medium">
                            {s.status === 'running' && <Loader2 size={11} className="shrink-0 animate-spin text-amber-400" />}
                            {s.title}
                          </span>
                          <span className="truncate text-[11px] text-zinc-500">{s.folder} · {s.messages} msgs</span>
                        </button>
                        <button onClick={() => { if (confirm(`Delete session "${s.title}"?`)) deleteSession(p.name, s.id) }}
                          title="Delete session" className="shrink-0 text-zinc-600 opacity-0 hover:text-red-400 group-hover/sess:opacity-100"><Trash2 size={12} /></button>
                      </div>
                    ))}

                    {/* folders panel */}
                    {foldersFor === p.name && (
                      <div className="ml-3 mb-1 space-y-1 rounded bg-zinc-900/60 p-2">
                        <div className="text-[10px] font-semibold uppercase text-zinc-500">Folders</div>
                        {p.folders.map((f) => (
                          <div key={f.name} className="flex items-center gap-1 text-[11px] text-zinc-400">
                            {f.linked ? <Link2 size={11} className="text-emerald-400" /> : <FolderGit2 size={11} className="text-zinc-500" />}
                            <span className="truncate">{f.name}</span>
                            {f.mode === 'ro' && <Lock size={10} className="text-zinc-500" />}
                            {f.missing && <span className="text-red-400">missing</span>}
                            <span className="ml-auto text-zinc-600">{f.files}f</span>
                          </div>
                        ))}
                        <div className="flex gap-1 pt-1">
                          <input value={foldersFor === p.name ? newVfsFolder : ''} onChange={(e) => setNewVfsFolder(e.target.value)}
                            onKeyDown={(e) => { if (e.key === 'Enter' && newVfsFolder.trim()) { addFolder(p.name, newVfsFolder); setNewVfsFolder('') } }}
                            placeholder="new folder" className="min-w-0 flex-1 rounded bg-zinc-900 px-2 py-1 text-[11px] text-zinc-100 outline-none ring-1 ring-zinc-700" />
                          <button onClick={() => { if (newVfsFolder.trim()) { addFolder(p.name, newVfsFolder); setNewVfsFolder('') } }}
                            title="Add VFS folder" className="rounded bg-zinc-800 px-1.5 text-zinc-200 hover:bg-zinc-700"><Plus size={12} /></button>
                          <button onClick={() => fsGo(p.name, '')} title="Link a local folder"
                            className="flex items-center gap-1 rounded bg-zinc-800 px-1.5 text-[11px] text-zinc-200 hover:bg-zinc-700"><Link2 size={12} /></button>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )
          })}
        </div>
      </aside>

      {/* ── Center: chat ── */}
      <main className="flex min-w-0 flex-1 flex-col">
        {!activeSess ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-sm text-zinc-500">
            <div>Select a session on the left, or start a new one.</div>
            <button onClick={() => openModal()}
              className="flex items-center gap-1.5 rounded bg-zinc-100 px-3 py-1.5 text-sm font-medium text-zinc-900 hover:bg-zinc-200">
              <MessageSquarePlus size={15} /> New session
            </button>
          </div>
        ) : (
          <>
            <header className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2 text-sm">
              <span className="text-zinc-500">{activeProject}</span>
              <span className="text-zinc-600">/</span>
              <span className="font-semibold">{activeSess.title}</span>
              <span className="ml-2 flex items-center gap-1 text-xs text-zinc-500">
                <FolderGit2 size={13} />
                <select value={activeSess.folder} onChange={(e) => setSessionFolder(e.target.value)}
                  disabled={status === 'running'}
                  className="rounded bg-zinc-900 px-1.5 py-0.5 text-xs text-zinc-200 outline-none ring-1 ring-zinc-700 disabled:opacity-50">
                  {activeFolders.map((f) => <option key={f.name} value={f.name}>{f.name}{f.kind === 'link' ? ' (linked)' : ''}</option>)}
                </select>
              </span>
              <div className="ml-auto flex items-center gap-1 rounded bg-zinc-900 p-0.5 text-xs ring-1 ring-zinc-700">
                <button onClick={() => setTab('chat')}
                  className={`rounded px-2 py-0.5 ${tab === 'chat' ? 'bg-zinc-950 text-zinc-100 shadow-sm dark:bg-zinc-700 dark:shadow-none' : 'text-zinc-400 hover:text-zinc-200'}`}>Chat</button>
                <button onClick={() => setTab('map')}
                  className={`flex items-center gap-1 rounded px-2 py-0.5 ${tab === 'map' ? 'bg-zinc-950 text-zinc-100 shadow-sm dark:bg-zinc-700 dark:shadow-none' : 'text-zinc-400 hover:text-zinc-200'}`}>
                  <Map size={12} /> Map
                </button>
              </div>
              {(status === 'running' || sending) && (
                <button onClick={() => stopRun()}
                  title="Stop the current coding run (agents wind down at the next step)"
                  className="flex items-center gap-1.5 rounded bg-red-500/15 px-2 py-1 text-xs font-medium text-red-300 ring-1 ring-red-500/40 hover:bg-red-500/25">
                  <Square size={11} fill="currentColor" /> Stop
                </button>
              )}
              <button onClick={() => exportProcess()} disabled={!messages.length}
                title="Export the coding process (tools, narration, outputs) as Markdown"
                className="flex items-center gap-1.5 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40">
                <Download size={13} /> Export
              </button>
            </header>

            {tab === 'map' && (
              <div className="flex-1 overflow-y-auto px-4 py-4">
                <div className="mb-3 flex items-center gap-3 text-xs text-zinc-500">
                  <span>{map?.stats ? `${map.stats.files} files · ${map.stats.symbols} symbols · ${map.stats.summarized} summarized` : 'No map yet.'}</span>
                  <button onClick={onRebuildMap} disabled={mapBusy || status === 'running'}
                    className="ml-auto flex items-center gap-1.5 rounded bg-zinc-800 px-2 py-1 text-zinc-200 hover:bg-zinc-700 disabled:opacity-40">
                    {mapBusy ? <Loader2 size={13} className="animate-spin" /> : <Map size={13} />} Rebuild map
                  </button>
                </div>
                {mapBusy && (
                  <div className="mb-3 space-y-1 rounded border border-zinc-800 bg-zinc-900/40 p-3">
                    {progress.slice(-8).map((e) => (
                      <div key={e.i} className="truncate text-[11px] text-zinc-500">
                        {e.stage === 'phase' ? <span className="text-zinc-300">▸ {e.message}</span> : <span>{e.message}</span>}
                      </div>
                    ))}
                    <div className="flex items-center gap-1.5 text-xs text-zinc-400"><Loader2 size={12} className="animate-spin" /> mapping the codebase…</div>
                  </div>
                )}
                <input value={mapQ} onChange={(e) => onMapSearch(e.target.value)}
                  placeholder="Search symbols, files, models…"
                  className="mb-3 w-full rounded bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none ring-1 ring-zinc-700 focus:ring-zinc-500" />
                {mapQ && (
                  <div className="mb-4 space-y-1">
                    {mapHits.length === 0 && <div className="text-xs text-zinc-500">No matches.</div>}
                    {mapHits.map((h, i) => (
                      <div key={i} className="rounded bg-zinc-900/50 px-2 py-1 text-xs">
                        <span className="text-zinc-500">{h.kind}</span>{' '}
                        <span className="font-medium text-zinc-200">{h.signature || h.name}</span>{' '}
                        <span className="text-zinc-600">— {h.file}:{h.line}</span>
                        {h.summary && <span className="text-zinc-500"> · {h.summary}</span>}
                      </div>
                    ))}
                  </div>
                )}
                {!mapQ && (
                  <>
                    {map?.overview
                      ? <div className="fd-markdown text-sm text-zinc-200"><Markdown remarkPlugins={[remarkGfm]}>{map.overview}</Markdown></div>
                      : <div className="text-sm text-zinc-500">No architecture overview yet. Click <b>Rebuild map</b> to have the cartographer map this folder.</div>}
                    {map?.models && (
                      <div className="mt-4">
                        <div className="mb-1 text-xs font-semibold uppercase text-zinc-500">Data models</div>
                        <pre className="overflow-auto rounded bg-zinc-900/50 p-3 text-xs text-zinc-300">{JSON.stringify(map.models, null, 2)}</pre>
                      </div>
                    )}
                    {map?.ui && (
                      <div className="mt-4">
                        <div className="mb-1 text-xs font-semibold uppercase text-zinc-500">UI map</div>
                        <pre className="overflow-auto rounded bg-zinc-900/50 p-3 text-xs text-zinc-300">{JSON.stringify(map.ui, null, 2)}</pre>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {tab === 'chat' && (<>
            <div className="flex-1 overflow-y-auto px-4 py-4">
              {messages.length === 0 && (
                <div className="mt-10 text-center text-sm text-zinc-500">
                  Describe what to build. The router picks a quick edit or a full build automatically.
                </div>
              )}
              {messages.map((m) => {
                const KindIcon = m.kind === 'plan' ? ClipboardList
                  : m.kind === 'review' ? (m.needs_fix ? ShieldAlert : CheckCircle2)
                  : m.kind === 'fix' ? Wrench : Bot
                return (
                  <div key={m.id} className="mb-4 flex gap-3">
                    <div className="mt-0.5 shrink-0 text-zinc-500">
                      {m.role === 'user' ? <User size={16} /> : <KindIcon size={16} />}
                    </div>
                    <div className="min-w-0 flex-1">
                      {m.role === 'assistant' && (m.archetype || m.size || m.kind) && (
                        <div className="mb-1 flex flex-wrap items-center gap-2 text-[11px] text-zinc-500">
                          {m.kind && m.kind !== 'note' && <span className="rounded bg-zinc-800 px-1.5 py-0.5 capitalize">{m.kind}{m.round ? ` r${m.round}` : ''}</span>}
                          {m.size && <span className="rounded bg-zinc-800 px-1.5 py-0.5">{m.size}</span>}
                          {m.archetype && <span className="text-zinc-400">{m.archetype}</span>}
                          {m.commit && <span className="flex items-center gap-1 text-emerald-500/80"><GitCommit size={11} /> {m.commit.slice(0, 7)}</span>}
                          {m.usage && <span className="text-zinc-600">{m.usage}</span>}
                          {m.ok === false && <span className="text-red-400">failed</span>}
                        </div>
                      )}
                      {m.role === 'user' ? (
                        <div className="whitespace-pre-wrap break-words text-sm text-zinc-200">{m.text}</div>
                      ) : (
                        <div className={`fd-markdown break-words text-sm text-zinc-200 ${m.kind === 'plan' ? 'rounded border border-zinc-800 bg-zinc-900/40 p-3' : ''}`}>
                          <Markdown remarkPlugins={[remarkGfm]}>{m.text}</Markdown>
                        </div>
                      )}
                      {!!m.findings?.length && (
                        <ul className="mt-1.5 space-y-0.5">
                          {m.findings.map((f, i) => (
                            <li key={i} className="text-[11px] text-zinc-400">
                              <span className={`font-medium uppercase ${SEV_COLOR[f.severity] || 'text-zinc-400'}`}>{f.severity}</span>
                              {' · '}{f.title}{f.file ? <span className="text-zinc-600"> ({f.file})</span> : null}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </div>
                )
              })}

              {sending && (
                <div className="mb-4 flex gap-3">
                  <div className="mt-0.5 shrink-0 text-zinc-500"><Bot size={16} /></div>
                  <div className="min-w-0 flex-1 space-y-1">
                    {progress.slice(-8).map((e) => (
                      <div key={e.i} className="truncate text-[11px] text-zinc-500">
                        {e.stage === 'phase' ? <span className="text-zinc-300">▸ {e.message}</span> : <span>{e.message}</span>}
                      </div>
                    ))}
                    <div className="flex items-center gap-1.5 text-xs text-zinc-400"><Loader2 size={12} className="animate-spin" /> working…</div>
                  </div>
                </div>
              )}
              <div ref={chatEnd} />
            </div>

            {error && <div className="px-4 py-1 text-xs text-red-400">{error}</div>}

            {status === 'awaiting_plan' ? (
              <div className="border-t border-amber-500/30 bg-amber-500/5 p-3">
                <div className="mb-2 flex items-center gap-2 text-sm text-amber-200/90"><ClipboardList size={15} /> Plan ready — edit if needed, then build.</div>
                <textarea value={planDraft} onChange={(e) => setPlanDraft(e.target.value)} rows={8}
                  className="mb-2 w-full resize-y rounded bg-zinc-900 px-3 py-2 font-mono text-[13px] text-zinc-200 outline-none ring-1 ring-zinc-700 focus:ring-zinc-500" />
                <div className="flex justify-end gap-2">
                  <button onClick={() => cancelPlan()} disabled={sending}
                    className="flex h-9 items-center gap-1.5 rounded px-3 text-sm font-medium text-zinc-400 ring-1 ring-zinc-700 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40">
                    <Trash2 size={15} /> Discard
                  </button>
                  <button onClick={() => approvePlan(planDraft)} disabled={sending}
                    className="flex h-9 items-center gap-1.5 rounded bg-amber-400 px-3 text-sm font-medium text-zinc-900 hover:bg-amber-300 disabled:opacity-40">
                    {sending ? <Loader2 size={15} className="animate-spin" /> : <CheckCircle2 size={15} />} Approve & Build
                  </button>
                </div>
              </div>
            ) : (
              <div className="border-t border-zinc-800 p-3">
                <div className="flex items-end gap-2">
                  <textarea value={input} onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend() } }}
                    placeholder={status === 'running' ? 'Build in progress…' : 'Build, edit, or fix something…  (Enter to send, Shift+Enter for newline)'}
                    rows={2} disabled={status === 'running'}
                    className="min-w-0 flex-1 resize-none rounded bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none ring-1 ring-zinc-700 focus:ring-zinc-500 disabled:opacity-50" />
                  <button onClick={onSend} disabled={sending || status === 'running' || !input.trim()}
                    className="flex h-10 items-center gap-1.5 rounded bg-zinc-100 px-3 text-sm font-medium text-zinc-900 hover:bg-zinc-200 disabled:opacity-40">
                    {sending ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
                  </button>
                </div>
              </div>
            )}
            </>)}
          </>
        )}
      </main>

      {/* ── Right: history (session's folder repo) ── */}
      {activeSess && (
        <aside className="hidden w-64 shrink-0 flex-col border-l border-zinc-800 bg-zinc-950/40 lg:flex">
          <div className="border-b border-zinc-800 p-3 text-sm font-semibold text-zinc-300">History</div>
          <div className="flex-1 overflow-y-auto p-2">
            {commits.length === 0 && <div className="px-2 py-4 text-xs text-zinc-500">No commits yet.</div>}
            {commits.map((c, idx) => (
              <div key={c.sha} className={`group mb-1 flex items-start gap-2 rounded px-2 py-1.5 text-xs hover:bg-zinc-900 ${diff?.sha === c.sha ? 'bg-zinc-900' : ''}`}>
                <GitCommit size={13} className="mt-0.5 shrink-0 text-zinc-600" />
                <button onClick={() => openDiff(c.sha)} className="min-w-0 flex-1 text-left">
                  <div className="truncate text-zinc-300">{c.message}</div>
                  <div className="text-[10px] text-zinc-600">{c.short}</div>
                </button>
                {idx > 0 && status !== 'running' && (
                  <button onClick={() => onRollback(c.sha, c.message)} title="Roll back to this commit"
                    className="shrink-0 text-zinc-600 opacity-0 hover:text-amber-400 group-hover:opacity-100"><RotateCcw size={13} /></button>
                )}
              </div>
            ))}
          </div>
        </aside>
      )}

      {/* ── Diff overlay ── */}
      {diff && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-8" onClick={() => setDiff(null)}>
          <div className="flex max-h-full w-full max-w-4xl flex-col rounded-lg border border-zinc-800 bg-zinc-950" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
              <div className="flex items-center gap-2 text-sm text-zinc-300"><GitCommit size={14} /> {diff.sha.slice(0, 8)}</div>
              <button onClick={() => setDiff(null)} className="text-zinc-500 hover:text-zinc-200"><X size={16} /></button>
            </div>
            <pre className="flex-1 overflow-auto p-4 font-mono text-[12px] leading-relaxed">
              {diff.text.split('\n').map((line, i) => <div key={i} className={diffLineClass(line)}>{line || ' '}</div>)}
            </pre>
          </div>
        </div>
      )}

      {/* ── New-session modal: project → folder → session ── */}
      {modal && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-6"
          onClick={() => !modal.busy && setModal(null)}>
          <div className="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 shadow-2xl"
            onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 border-b border-zinc-800 px-5 py-3">
              <MessageSquarePlus size={16} className="text-zinc-300" />
              <span className="text-sm font-semibold text-zinc-100">Start a coding session</span>
              <button onClick={() => setModal(null)} disabled={modal.busy}
                className="ml-auto text-zinc-500 hover:text-zinc-200"><X size={16} /></button>
            </div>

            <div className="space-y-5 px-5 py-4">
              {/* 1 · Project */}
              <div>
                <div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-zinc-500">1 · Project</div>
                <select value={modal.project} onChange={(e) => onModalProject(e.target.value)}
                  className="w-full rounded bg-zinc-900 px-2.5 py-2 text-sm text-zinc-100 outline-none ring-1 ring-zinc-700 focus:ring-zinc-500">
                  {projects.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
                  <option value="">＋ Create a new project…</option>
                </select>
                {!modal.project && (
                  <input value={modal.newProject} autoFocus
                    onChange={(e) => setModal({ ...modal, newProject: e.target.value })}
                    placeholder="new project name"
                    className="mt-2 w-full rounded bg-zinc-900 px-2.5 py-2 text-sm text-zinc-100 outline-none ring-1 ring-zinc-700 focus:ring-zinc-500" />
                )}
              </div>

              {/* 2 · Folder */}
              <div>
                <div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-zinc-500">
                  2 · Folder <span className="font-normal normal-case text-zinc-600">— the repo agents work in</span>
                </div>
                <div className="mb-2 flex gap-1 rounded bg-zinc-900 p-0.5 text-xs ring-1 ring-zinc-700">
                  {([['existing', 'Existing'], ['new', 'New folder'], ['link', 'Link local folder']] as const).map(([k, label]) => (
                    <button key={k} onClick={() => setModal({ ...modal, folderMode: k })}
                      disabled={k === 'existing' && !(projects.find((p) => p.name === modal.project)?.folders.length)}
                      className={`flex-1 rounded px-2 py-1 ${modal.folderMode === k
                        ? 'bg-zinc-950 text-zinc-100 shadow-sm dark:bg-zinc-700 dark:shadow-none'
                        : 'text-zinc-400 hover:text-zinc-200 disabled:opacity-30'}`}>
                      {label}
                    </button>
                  ))}
                </div>
                {modal.folderMode === 'existing' && (
                  <select value={modal.folder} onChange={(e) => setModal({ ...modal, folder: e.target.value })}
                    className="w-full rounded bg-zinc-900 px-2.5 py-2 text-sm text-zinc-100 outline-none ring-1 ring-zinc-700 focus:ring-zinc-500">
                    {(projects.find((p) => p.name === modal.project)?.folders || []).map((f) => (
                      <option key={f.name} value={f.name}>{f.name}{f.kind === 'link' ? ' (linked)' : ''}</option>
                    ))}
                  </select>
                )}
                {modal.folderMode === 'new' && (
                  <input value={modal.newFolder} onChange={(e) => setModal({ ...modal, newFolder: e.target.value })}
                    placeholder="folder name (becomes a fresh git repo)"
                    className="w-full rounded bg-zinc-900 px-2.5 py-2 text-sm text-zinc-100 outline-none ring-1 ring-zinc-700 focus:ring-zinc-500" />
                )}
                {modal.folderMode === 'link' && (
                  <div className="space-y-2">
                    <div className="flex gap-2">
                      <input value={modal.linkPath} readOnly placeholder="pick a folder on this machine…"
                        className="min-w-0 flex-1 rounded bg-zinc-900 px-2.5 py-2 font-mono text-xs text-zinc-300 outline-none ring-1 ring-zinc-700" />
                      <button onClick={() => fsGo('@modal', modal.linkPath || '')}
                        className="flex shrink-0 items-center gap-1.5 rounded bg-zinc-800 px-3 text-xs text-zinc-200 hover:bg-zinc-700">
                        <FolderTree size={13} /> Browse…
                      </button>
                    </div>
                    <label className="flex items-center gap-2 text-xs text-zinc-400">
                      <input type="checkbox" checked={modal.linkMode === 'ro'}
                        onChange={(e) => setModal({ ...modal, linkMode: e.target.checked ? 'ro' : 'rw' })}
                        className="accent-zinc-400" />
                      Read-only (agents can look but never modify)
                    </label>
                  </div>
                )}
              </div>

              {/* 3 · Session */}
              <div>
                <div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-zinc-500">
                  3 · Session name <span className="font-normal normal-case text-zinc-600">— optional</span>
                </div>
                <input value={modal.title} onChange={(e) => setModal({ ...modal, title: e.target.value })}
                  onKeyDown={(e) => { if (e.key === 'Enter') submitModal() }}
                  placeholder="e.g. RTS game, auth refactor…"
                  className="w-full rounded bg-zinc-900 px-2.5 py-2 text-sm text-zinc-100 outline-none ring-1 ring-zinc-700 focus:ring-zinc-500" />
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 border-t border-zinc-800 px-5 py-3">
              <button onClick={() => setModal(null)} disabled={modal.busy}
                className="rounded px-3 py-1.5 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">Cancel</button>
              <button onClick={submitModal} disabled={!modalValid}
                className="flex items-center gap-1.5 rounded bg-zinc-100 px-3 py-1.5 text-sm font-medium text-zinc-900 hover:bg-zinc-200 disabled:opacity-40">
                {modal.busy ? <Loader2 size={14} className="animate-spin" /> : <MessageSquarePlus size={14} />}
                Start session
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Folder browser (for linking into a project) ── */}
      {fs && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-8" onClick={() => setFs(null)}>
          <div className="flex max-h-full w-full max-w-2xl flex-col rounded-lg border border-zinc-800 bg-zinc-950" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2">
              <FolderTree size={15} className="text-violet-400" />
              <span className="truncate font-mono text-xs text-zinc-300">{fs.path || '~'}</span>
              <span className="ml-auto text-[11px] text-zinc-500">
                {fs.project === '@modal' ? 'pick a folder to link' : <>link into <b className="text-zinc-300">{fs.project}</b></>}
              </span>
              <button onClick={() => setFs(null)} className="text-zinc-500 hover:text-zinc-200"><X size={16} /></button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto p-2">
              {fs.parent && (
                <button onClick={() => fsGo(fs.project, fs.parent)} className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm text-zinc-400 hover:bg-zinc-900">
                  <ChevronRight size={14} className="rotate-180" /> ..
                </button>
              )}
              {fs.dirs.filter((d) => !d.hidden).map((d) => (
                <button key={d.name} onClick={() => fsGo(fs.project, `${fs.path}/${d.name}`)}
                  className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm text-zinc-200 hover:bg-zinc-900">
                  <FolderGit2 size={14} className="shrink-0 text-violet-400" />
                  <span className="truncate">{d.name}</span>
                  {d.is_git && <span className="ml-auto rounded border border-emerald-500/40 bg-emerald-500/10 px-1 text-[9px] uppercase text-emerald-300">git</span>}
                </button>
              ))}
            </div>
            <div className="flex items-center justify-between gap-2 border-t border-zinc-800 px-4 py-2">
              <span className="truncate font-mono text-[11px] text-zinc-500">{fs.path}</span>
              <button onClick={useFolderForLink} className="shrink-0 rounded bg-emerald-600/80 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-600">
                {fs.project === '@modal' ? 'Use this folder' : 'Link this folder'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
