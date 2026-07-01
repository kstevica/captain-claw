import { useEffect, useRef, useState } from 'react'
import { Plus, FolderGit2, Send, GitCommit, Loader2, Bot, User, ClipboardList, CheckCircle2, ShieldAlert, Wrench, RotateCcw, X, Download, ChevronRight, ChevronDown, Link2 } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useCodeStore } from '../stores/codeStore'

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
    projects, activeFolder, messages, commits, progress, status, sending, error,
    loadProjects, createFolder, selectFolder, send, approvePlan, showCommit, rollback, exportProcess,
  } = useCodeStore()
  const [newProject, setNewProject] = useState('')
  const [newFolder, setNewFolder] = useState('')
  const [showNew, setShowNew] = useState(false)
  const [creating, setCreating] = useState(false)
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [input, setInput] = useState('')
  const [planDraft, setPlanDraft] = useState('')
  const [diff, setDiff] = useState<{ sha: string; text: string } | null>(null)
  const chatEnd = useRef<HTMLDivElement>(null)

  useEffect(() => { loadProjects() }, [loadProjects])
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, progress])

  // Pre-fill the editable plan from the latest plan message when the gate opens.
  useEffect(() => {
    if (status === 'awaiting_plan') {
      const plan = [...messages].reverse().find((m) => m.kind === 'plan')
      setPlanDraft(plan?.text || '')
    }
  }, [status, messages])

  const openDiff = async (sha: string) => {
    setDiff({ sha, text: 'Loading…' })
    setDiff({ sha, text: (await showCommit(sha)) || '(no changes)' })
  }

  const onRollback = async (sha: string, message: string) => {
    if (!window.confirm(`Roll back to "${message}"?\nThis discards everything after this commit.`)) return
    await rollback(sha)
    setDiff(null)
  }

  const onCreate = async () => {
    const project = newProject.trim()
    const folder = newFolder.trim()
    if (!project || !folder) return
    setCreating(true)
    try {
      await createFolder(project, folder)
      setNewFolder(''); setShowNew(false)
    } finally { setCreating(false) }
  }

  const toggle = (name: string) =>
    setCollapsed((s) => { const n = new Set(s); n.has(name) ? n.delete(name) : n.add(name); return n })

  const activeProjectName = activeFolder.includes('/') ? activeFolder.split('/')[0] : activeFolder
  const activeFolderName = activeFolder.includes('/') ? activeFolder.split('/').slice(1).join('/') : activeFolder

  const onSend = async () => {
    const text = input.trim()
    if (!text || sending) return
    setInput('')
    await send(text)
  }

  return (
    <div className="flex h-full overflow-hidden text-zinc-100">
      {/* ── Left rail: project → folder tree ── */}
      <aside className="flex w-64 shrink-0 flex-col border-r border-zinc-800 bg-zinc-950/40">
        <div className="border-b border-zinc-800 p-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-300">
            <FolderGit2 size={16} /> Code Projects
            <button
              onClick={() => setShowNew((v) => !v)}
              className="ml-auto rounded bg-zinc-800 px-1.5 py-1 text-zinc-200 hover:bg-zinc-700"
              title="New folder"
            >
              <Plus size={14} />
            </button>
          </div>
          {showNew && (
            <div className="mt-2 space-y-1">
              <input
                value={newProject}
                onChange={(e) => setNewProject(e.target.value)}
                list="code-projects"
                placeholder="project (group)"
                className="w-full rounded bg-zinc-900 px-2 py-1 text-xs text-zinc-100 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600"
              />
              <datalist id="code-projects">
                {projects.map((p) => <option key={p.name} value={p.name} />)}
              </datalist>
              <div className="flex gap-1">
                <input
                  value={newFolder}
                  onChange={(e) => setNewFolder(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && onCreate()}
                  placeholder="folder (sub-project)"
                  className="min-w-0 flex-1 rounded bg-zinc-900 px-2 py-1 text-xs text-zinc-100 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600"
                />
                <button
                  onClick={onCreate}
                  disabled={creating || !newProject.trim() || !newFolder.trim()}
                  className="rounded bg-zinc-800 px-2 text-zinc-200 hover:bg-zinc-700 disabled:opacity-40"
                  title="Create folder"
                >
                  {creating ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
                </button>
              </div>
            </div>
          )}
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {projects.length === 0 && (
            <div className="px-2 py-4 text-xs text-zinc-500">No projects yet. Create a folder to start.</div>
          )}
          {projects.map((p) => {
            const open = !collapsed.has(p.name)
            return (
              <div key={p.name} className="mb-1">
                <button
                  onClick={() => toggle(p.name)}
                  className="flex w-full items-center gap-1 rounded px-1 py-1 text-left text-xs font-semibold uppercase tracking-wide text-zinc-400 hover:bg-zinc-900"
                >
                  {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                  <span className="truncate">{p.name}</span>
                  <span className="ml-auto text-[10px] font-normal text-zinc-600">{p.folders.length}</span>
                </button>
                {open && p.folders.map((f) => (
                  <button
                    key={f.id}
                    onClick={() => selectFolder(f.id)}
                    className={`mb-0.5 ml-3 flex w-[calc(100%-0.75rem)] flex-col rounded px-2 py-1.5 text-left text-sm ${
                      f.id === activeFolder ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-300 hover:bg-zinc-900'
                    }`}
                  >
                    <span className="flex items-center gap-1 truncate font-medium">
                      {f.status === 'running' && <Loader2 size={11} className="shrink-0 animate-spin text-amber-400" />}
                      {f.linked && <Link2 size={11} className="shrink-0 text-emerald-400" />}
                      {f.name}
                    </span>
                    <span className="truncate text-[11px] text-zinc-500">
                      {f.files} files · {f.messages} msgs{f.linked ? ` · linked${f.mode === 'ro' ? ' · read-only' : ''}` : ''}
                    </span>
                  </button>
                ))}
              </div>
            )
          })}
        </div>
      </aside>

      {/* ── Center: chat ── */}
      <main className="flex min-w-0 flex-1 flex-col">
        {!activeFolder ? (
          <div className="flex h-full items-center justify-center text-sm text-zinc-500">
            Select or create a folder to start coding.
          </div>
        ) : (
          <>
            <header className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2 text-sm">
              <FolderGit2 size={15} className="text-zinc-400" />
              <span className="text-zinc-500">{activeProjectName}</span>
              <span className="text-zinc-600">/</span>
              <span className="font-semibold">{activeFolderName}</span>
              <button
                onClick={() => exportProcess()}
                disabled={!messages.length}
                title="Export the coding process (tools, narration, outputs) as Markdown"
                className="ml-auto flex items-center gap-1.5 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40"
              >
                <Download size={13} /> Export
              </button>
            </header>

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
                          {m.kind && m.kind !== 'note' && (
                            <span className="rounded bg-zinc-800 px-1.5 py-0.5 capitalize">
                              {m.kind}{m.round ? ` r${m.round}` : ''}
                            </span>
                          )}
                          {m.size && <span className="rounded bg-zinc-800 px-1.5 py-0.5">{m.size}</span>}
                          {m.archetype && <span className="text-zinc-400">{m.archetype}</span>}
                          {m.commit && (
                            <span className="flex items-center gap-1 text-emerald-500/80">
                              <GitCommit size={11} /> {m.commit.slice(0, 7)}
                            </span>
                          )}
                          {m.ok === false && <span className="text-red-400">failed</span>}
                        </div>
                      )}
                      {m.role === 'user' ? (
                        <div className="whitespace-pre-wrap break-words text-sm text-zinc-200">{m.text}</div>
                      ) : (
                        <div className={`fd-markdown break-words text-sm text-zinc-200 ${
                          m.kind === 'plan' ? 'rounded border border-zinc-800 bg-zinc-900/40 p-3' : ''
                        }`}>
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
                        {e.stage === 'phase' ? (
                          <span className="text-zinc-300">▸ {e.message}</span>
                        ) : (
                          <span>{e.message}</span>
                        )}
                      </div>
                    ))}
                    <div className="flex items-center gap-1.5 text-xs text-zinc-400">
                      <Loader2 size={12} className="animate-spin" /> working…
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEnd} />
            </div>

            {error && <div className="px-4 py-1 text-xs text-red-400">{error}</div>}

            {status === 'awaiting_plan' ? (
              <div className="border-t border-amber-500/30 bg-amber-500/5 p-3">
                <div className="mb-2 flex items-center gap-2 text-sm text-amber-200/90">
                  <ClipboardList size={15} /> Plan ready — edit if needed, then build.
                </div>
                <textarea
                  value={planDraft}
                  onChange={(e) => setPlanDraft(e.target.value)}
                  rows={8}
                  className="mb-2 w-full resize-y rounded bg-zinc-900 px-3 py-2 font-mono text-[13px] text-zinc-200 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600"
                />
                <div className="flex justify-end">
                  <button
                    onClick={() => approvePlan(planDraft)}
                    disabled={sending}
                    className="flex h-9 items-center gap-1.5 rounded bg-amber-400 px-3 text-sm font-medium text-zinc-900 hover:bg-amber-300 disabled:opacity-40"
                  >
                    {sending ? <Loader2 size={15} className="animate-spin" /> : <CheckCircle2 size={15} />} Approve & Build
                  </button>
                </div>
              </div>
            ) : (
              <div className="border-t border-zinc-800 p-3">
                <div className="flex items-end gap-2">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend() }
                    }}
                    placeholder={status === 'running' ? 'Build in progress…' : 'Build, edit, or fix something…  (Enter to send, Shift+Enter for newline)'}
                    rows={2}
                    disabled={status === 'running'}
                    className="min-w-0 flex-1 resize-none rounded bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600 disabled:opacity-50"
                  />
                  <button
                    onClick={onSend}
                    disabled={sending || status === 'running' || !input.trim()}
                    className="flex h-10 items-center gap-1.5 rounded bg-zinc-100 px-3 text-sm font-medium text-zinc-900 hover:bg-white disabled:opacity-40"
                  >
                    {sending ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </main>

      {/* ── Right: phase timeline (click a commit → diff; roll back) ── */}
      {activeFolder && (
        <aside className="hidden w-64 shrink-0 flex-col border-l border-zinc-800 bg-zinc-950/40 lg:flex">
          <div className="border-b border-zinc-800 p-3 text-sm font-semibold text-zinc-300">History</div>
          <div className="flex-1 overflow-y-auto p-2">
            {commits.length === 0 && <div className="px-2 py-4 text-xs text-zinc-500">No commits yet.</div>}
            {commits.map((c, idx) => (
              <div
                key={c.sha}
                className={`group mb-1 flex items-start gap-2 rounded px-2 py-1.5 text-xs hover:bg-zinc-900 ${
                  diff?.sha === c.sha ? 'bg-zinc-900' : ''
                }`}
              >
                <GitCommit size={13} className="mt-0.5 shrink-0 text-zinc-600" />
                <button onClick={() => openDiff(c.sha)} className="min-w-0 flex-1 text-left">
                  <div className="truncate text-zinc-300">{c.message}</div>
                  <div className="text-[10px] text-zinc-600">{c.short}</div>
                </button>
                {idx > 0 && status !== 'running' && (
                  <button
                    onClick={() => onRollback(c.sha, c.message)}
                    title="Roll back to this commit"
                    className="shrink-0 text-zinc-600 opacity-0 hover:text-amber-400 group-hover:opacity-100"
                  >
                    <RotateCcw size={13} />
                  </button>
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
              <div className="flex items-center gap-2 text-sm text-zinc-300">
                <GitCommit size={14} /> {diff.sha.slice(0, 8)}
              </div>
              <button onClick={() => setDiff(null)} className="text-zinc-500 hover:text-zinc-200"><X size={16} /></button>
            </div>
            <pre className="flex-1 overflow-auto p-4 font-mono text-[12px] leading-relaxed">
              {diff.text.split('\n').map((line, i) => (
                <div key={i} className={diffLineClass(line)}>{line || ' '}</div>
              ))}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}
