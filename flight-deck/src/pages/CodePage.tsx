import { useEffect, useRef, useState } from 'react'
import { Plus, FolderGit2, Send, GitCommit, Loader2, Bot, User } from 'lucide-react'
import { useCodeStore } from '../stores/codeStore'

export function CodePage() {
  const {
    projects, activeProject, messages, commits, progress, sending, error,
    loadProjects, createProject, selectProject, send,
  } = useCodeStore()
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [input, setInput] = useState('')
  const chatEnd = useRef<HTMLDivElement>(null)

  useEffect(() => { loadProjects() }, [loadProjects])
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, progress])

  const onCreate = async () => {
    const name = newName.trim()
    if (!name) return
    setCreating(true)
    try { await createProject(name); setNewName('') } finally { setCreating(false) }
  }

  const onSend = async () => {
    const text = input.trim()
    if (!text || sending) return
    setInput('')
    await send(text)
  }

  return (
    <div className="flex h-full overflow-hidden text-zinc-100">
      {/* ── Left rail: projects + new folder ── */}
      <aside className="flex w-64 shrink-0 flex-col border-r border-zinc-800 bg-zinc-950/40">
        <div className="border-b border-zinc-800 p-3">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-zinc-300">
            <FolderGit2 size={16} /> Code Projects
          </div>
          <div className="flex gap-1">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && onCreate()}
              placeholder="new-folder-name"
              className="min-w-0 flex-1 rounded bg-zinc-900 px-2 py-1 text-xs text-zinc-100 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600"
            />
            <button
              onClick={onCreate}
              disabled={creating || !newName.trim()}
              className="rounded bg-zinc-800 px-2 text-zinc-200 hover:bg-zinc-700 disabled:opacity-40"
              title="Create project folder"
            >
              {creating ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {projects.length === 0 && (
            <div className="px-2 py-4 text-xs text-zinc-500">No projects yet. Create a folder to start.</div>
          )}
          {projects.map((p) => (
            <button
              key={p.name}
              onClick={() => selectProject(p.name)}
              className={`mb-1 w-full rounded px-2 py-2 text-left text-sm ${
                p.name === activeProject ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-300 hover:bg-zinc-900'
              }`}
            >
              <div className="truncate font-medium">{p.name}</div>
              <div className="truncate text-[11px] text-zinc-500">
                {p.files} files · {p.messages} msgs{p.status === 'running' ? ' · running…' : ''}
              </div>
            </button>
          ))}
        </div>
      </aside>

      {/* ── Center: chat ── */}
      <main className="flex min-w-0 flex-1 flex-col">
        {!activeProject ? (
          <div className="flex h-full items-center justify-center text-sm text-zinc-500">
            Select or create a project folder to start coding.
          </div>
        ) : (
          <>
            <header className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2 text-sm">
              <FolderGit2 size={15} className="text-zinc-400" />
              <span className="font-semibold">{activeProject}</span>
              <span className="text-zinc-500">— agentic coding</span>
            </header>

            <div className="flex-1 overflow-y-auto px-4 py-4">
              {messages.length === 0 && (
                <div className="mt-10 text-center text-sm text-zinc-500">
                  Describe what to build. The router picks a quick edit or a full build automatically.
                </div>
              )}
              {messages.map((m) => (
                <div key={m.id} className="mb-4 flex gap-3">
                  <div className="mt-0.5 shrink-0 text-zinc-500">
                    {m.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  <div className="min-w-0 flex-1">
                    {m.role === 'assistant' && (m.archetype || m.size) && (
                      <div className="mb-1 flex items-center gap-2 text-[11px] text-zinc-500">
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
                    <div className="whitespace-pre-wrap break-words text-sm text-zinc-200">{m.text}</div>
                  </div>
                </div>
              ))}

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

            <div className="border-t border-zinc-800 p-3">
              <div className="flex items-end gap-2">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend() }
                  }}
                  placeholder="Build, edit, or fix something…  (Enter to send, Shift+Enter for newline)"
                  rows={2}
                  className="min-w-0 flex-1 resize-none rounded bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none ring-1 ring-zinc-800 focus:ring-zinc-600"
                />
                <button
                  onClick={onSend}
                  disabled={sending || !input.trim()}
                  className="flex h-10 items-center gap-1.5 rounded bg-zinc-100 px-3 text-sm font-medium text-zinc-900 hover:bg-white disabled:opacity-40"
                >
                  {sending ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
                </button>
              </div>
            </div>
          </>
        )}
      </main>

      {/* ── Right: commit timeline ── */}
      {activeProject && (
        <aside className="hidden w-60 shrink-0 flex-col border-l border-zinc-800 bg-zinc-950/40 lg:flex">
          <div className="border-b border-zinc-800 p-3 text-sm font-semibold text-zinc-300">History</div>
          <div className="flex-1 overflow-y-auto p-2">
            {commits.length === 0 && <div className="px-2 py-4 text-xs text-zinc-500">No commits yet.</div>}
            {commits.map((c) => (
              <div key={c.sha} className="mb-2 flex gap-2 px-1 text-xs">
                <GitCommit size={13} className="mt-0.5 shrink-0 text-zinc-600" />
                <div className="min-w-0">
                  <div className="truncate text-zinc-300">{c.message}</div>
                  <div className="text-[10px] text-zinc-600">{c.short}</div>
                </div>
              </div>
            ))}
          </div>
        </aside>
      )}
    </div>
  )
}
