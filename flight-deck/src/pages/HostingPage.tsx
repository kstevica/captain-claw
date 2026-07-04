import { useEffect, useState } from 'react'
import {
  Globe, Plus, Trash2, Play, Square, ExternalLink, Loader2, X,
  AlertTriangle, RefreshCw, FileCode2, Server, Info, Pencil,
} from 'lucide-react'
import { useAuthStore } from '../stores/authStore'

interface HostingEntry {
  name: string
  kind: 'static' | 'app'
  project: string
  subdir: string
  start_cmd: string
  running: boolean
  port: number | null
  url: string
}

function authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

const req = (url: string, init: RequestInit = {}) =>
  fetch(url, { ...init, headers: { ...authHeaders(), ...(init.headers || {}) }, credentials: 'include' })

export function HostingPage() {
  const [entries, setEntries] = useState<HostingEntry[]>([])
  const [projects, setProjects] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<string | null>(null)

  const [showForm, setShowForm] = useState(false)
  const [name, setName] = useState('')
  const [kind, setKind] = useState<'static' | 'app'>('static')
  const [project, setProject] = useState('')
  const [subdir, setSubdir] = useState('')
  const [startCmd, setStartCmd] = useState('')
  const [formErr, setFormErr] = useState('')
  const [publishing, setPublishing] = useState(false)
  // null = publishing a new entry; a name = editing that existing entry.
  const [editing, setEditing] = useState<string | null>(null)

  const load = async () => {
    setError('')
    try {
      const [er, pr] = await Promise.all([req('/fd/hosting'), req('/fd/vfs/projects')])
      if (!er.ok) throw new Error(await er.text())
      const ed = await er.json()
      setEntries(ed.entries || [])
      if (pr.ok) {
        const pd = await pr.json()
        const names: string[] = (pd.projects || []).map((p: { name: string }) => p.name)
        setProjects(names)
        if (!project && names.length) setProject(names[0])
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  // Poll while any app is running so status stays fresh.
  useEffect(() => {
    if (!entries.some((e) => e.kind === 'app')) return
    const t = setInterval(load, 4000)
    return () => clearInterval(t)
  }, [entries])  // eslint-disable-line react-hooks/exhaustive-deps

  const openNew = () => {
    setEditing(null); setName(''); setSubdir(''); setStartCmd(''); setKind('static')
    setFormErr(''); setShowForm(true)
  }
  const openEdit = (e: HostingEntry) => {
    setEditing(e.name); setName(e.name); setKind(e.kind); setProject(e.project)
    setSubdir(e.subdir); setStartCmd(e.start_cmd); setFormErr(''); setShowForm(true)
  }

  const submit = async () => {
    setFormErr('')
    if (!editing && !name.trim()) { setFormErr('Pick a name.'); return }
    if (!project) { setFormErr('Pick a VFS project.'); return }
    if (kind === 'app' && !startCmd.trim()) { setFormErr('Apps need a start command.'); return }
    setPublishing(true)
    try {
      const payload = { kind, project, subdir: subdir.trim(), start_cmd: startCmd.trim() }
      const res = editing
        ? await req(`/fd/hosting/${encodeURIComponent(editing)}`, { method: 'PUT', body: JSON.stringify(payload) })
        : await req('/fd/hosting', { method: 'POST', body: JSON.stringify({ name: name.trim().toLowerCase(), ...payload }) })
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || await res.text())
      setShowForm(false); setEditing(null)
      setName(''); setSubdir(''); setStartCmd(''); setKind('static')
      await load()
    } catch (e) {
      setFormErr(e instanceof Error ? e.message : String(e))
    } finally {
      setPublishing(false)
    }
  }

  const act = async (name: string, action: 'start' | 'stop') => {
    setBusy(name)
    try {
      const res = await req(`/fd/hosting/${encodeURIComponent(name)}/${action}`, { method: 'POST' })
      if (!res.ok) setError((await res.json().catch(() => ({}))).detail || `Failed to ${action}`)
      await load()
    } finally { setBusy(null) }
  }

  const unpublish = async (name: string) => {
    if (!confirm(`Unpublish "${name}"? (stops its app if running)`)) return
    setBusy(name)
    try {
      await req(`/fd/hosting/${encodeURIComponent(name)}`, { method: 'DELETE' })
      await load()
    } finally { setBusy(null) }
  }

  return (
    <div className="h-full overflow-auto p-4 md:p-6">
      <div className="mb-6 flex items-start justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-lg font-semibold">
            <Globe className="h-5 w-5 text-violet-400" /> Hosting
          </h1>
          <p className="text-xs text-zinc-500 sm:text-sm">
            Publish a VFS folder as a static site at <code className="text-zinc-400">/vfs/&lt;name&gt;</code>, or run a built app routed through Flight Deck at <code className="text-zinc-400">/vfs-apps/&lt;name&gt;</code>.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button onClick={load} title="Refresh" className="rounded-lg border border-zinc-700 p-2 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200">
            <RefreshCw className="h-4 w-4" />
          </button>
          <button
            onClick={() => { if (showForm) { setShowForm(false); setEditing(null) } else { openNew() } }}
            className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-2 text-sm font-medium text-white hover:bg-violet-500"
          >
            {showForm ? <X className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
            {showForm ? 'Close' : 'Publish'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/[0.06] px-3 py-2 text-xs text-red-400">
          <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
        </div>
      )}

      {/* ── Publish form ── */}
      {showForm && (
        <div className="mb-6 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <div className="mb-3 flex w-fit gap-1 rounded-lg bg-zinc-900 p-0.5 text-xs ring-1 ring-zinc-700">
            {(['static', 'app'] as const).map((k) => (
              <button
                key={k}
                onClick={() => setKind(k)}
                className={`flex items-center gap-1.5 rounded px-3 py-1.5 ${kind === k ? 'bg-zinc-950 text-zinc-100 shadow-sm dark:bg-zinc-700 dark:shadow-none' : 'text-zinc-400 hover:text-zinc-200'}`}
              >
                {k === 'static' ? <FileCode2 className="h-3.5 w-3.5" /> : <Server className="h-3.5 w-3.5" />}
                {k === 'static' ? 'Static site' : 'Built app'}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">Public name</label>
              <div className="flex items-center gap-1.5">
                <span className="font-mono text-xs text-zinc-600">/{kind === 'app' ? 'vfs-apps' : 'vfs'}/</span>
                <input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-site" disabled={!!editing}
                  title={editing ? 'The name is the public URL and cannot be changed — unpublish to rename' : undefined}
                  className="flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none disabled:opacity-60" />
              </div>
            </div>
            <div>
              <label className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">VFS project</label>
              <select value={project} onChange={(e) => setProject(e.target.value)}
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none">
                {projects.length === 0 && <option value="">(no projects)</option>}
                {projects.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                Subfolder <span className="font-normal normal-case text-zinc-600">— optional (e.g. dist)</span>
              </label>
              <input value={subdir} onChange={(e) => setSubdir(e.target.value)} placeholder="dist"
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none" />
            </div>
            {kind === 'app' && (
              <div>
                <label className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">Start command</label>
                <input value={startCmd} onChange={(e) => setStartCmd(e.target.value)} placeholder="npm run start"
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 font-mono text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none" />
              </div>
            )}
          </div>

          <div className="mt-3 flex items-start gap-1.5 rounded-lg border border-zinc-800 bg-zinc-950/40 px-3 py-2 text-[11px] text-zinc-500">
            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-600" />
            {kind === 'static'
              ? <span>A built SPA must set its base path to <code className="text-zinc-400">/vfs/{name || '<name>'}/</code> (e.g. Vite <code className="text-zinc-400">base</code>) so its assets resolve.</span>
              : <span>The app must bind the <code className="text-zinc-400">PORT</code> env var Flight Deck assigns, on <code className="text-zinc-400">127.0.0.1</code>. Set its base path to <code className="text-zinc-400">/vfs-apps/{name || '<name>'}/</code> for assets/routing.</span>}
          </div>

          {formErr && <p className="mt-2 text-[11px] text-red-400">{formErr}</p>}

          <div className="mt-3 flex justify-end">
            <button onClick={submit} disabled={publishing}
              className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50">
              {publishing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Globe className="h-3.5 w-3.5" />} {editing ? 'Save changes' : 'Publish'}
            </button>
          </div>
        </div>
      )}

      {/* ── Published list ── */}
      {loading ? (
        <p className="text-[11px] text-zinc-600">Loading…</p>
      ) : entries.length === 0 ? (
        <p className="rounded-xl border border-dashed border-zinc-800 py-10 text-center text-sm text-zinc-600">
          Nothing published yet. Click <span className="text-zinc-400">Publish</span> to serve a VFS folder.
        </p>
      ) : (
        <div className="space-y-2">
          {entries.map((e) => (
            <div key={e.name} className="flex flex-wrap items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
              <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md ${e.kind === 'app' ? 'bg-cyan-500/10 text-cyan-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
                {e.kind === 'app' ? <Server className="h-4 w-4" /> : <FileCode2 className="h-4 w-4" />}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <a href={e.url} target="_blank" rel="noreferrer" className="truncate font-mono text-sm font-medium text-zinc-100 hover:text-violet-300">{e.url}</a>
                  {e.kind === 'app' && (
                    <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${e.running ? 'border border-emerald-500/25 bg-emerald-600/15 text-emerald-400' : 'border border-zinc-700 text-zinc-500'}`}>
                      <span className={`h-1.5 w-1.5 rounded-full ${e.running ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
                      {e.running ? `running · :${e.port}` : 'stopped'}
                    </span>
                  )}
                </div>
                <div className="mt-0.5 truncate text-[11px] text-zinc-500">
                  vfs:{e.project}{e.subdir ? `/${e.subdir}` : ''}{e.kind === 'app' && e.start_cmd ? ` · ${e.start_cmd}` : ''}
                </div>
              </div>
              <div className="flex items-center gap-1.5">
                {e.kind === 'app' && (
                  e.running
                    ? <button onClick={() => act(e.name, 'stop')} disabled={busy === e.name}
                        className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1.5 text-xs text-zinc-300 hover:border-red-500/40 hover:text-red-400 disabled:opacity-50">
                        {busy === e.name ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Square className="h-3.5 w-3.5" />} Stop
                      </button>
                    : <button onClick={() => act(e.name, 'start')} disabled={busy === e.name}
                        className="flex items-center gap-1.5 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-500/20 dark:text-emerald-300 disabled:opacity-50">
                        {busy === e.name ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />} Start
                      </button>
                )}
                <button onClick={() => openEdit(e)} title="Edit"
                  className="rounded-lg border border-zinc-700 p-1.5 text-zinc-400 hover:border-zinc-600 hover:text-violet-300">
                  <Pencil className="h-4 w-4" />
                </button>
                <a href={e.url} target="_blank" rel="noreferrer" title="Open"
                  className="rounded-lg border border-zinc-700 p-1.5 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200">
                  <ExternalLink className="h-4 w-4" />
                </a>
                <button onClick={() => unpublish(e.name)} disabled={busy === e.name} title="Unpublish"
                  className="rounded-lg p-1.5 text-zinc-600 hover:bg-zinc-800 hover:text-red-400 disabled:opacity-50">
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
