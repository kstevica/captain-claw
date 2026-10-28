import { useEffect, useMemo, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertTriangle,
  ArrowLeft,
  Blocks,
  Check,
  FileText,
  Loader2,
  Pencil,
  Search,
  Sparkles,
  UserRound,
  Wand2,
  X,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { AgentAddress } from '../../services/agentInsights'
import { listOwnSkills, readSkill, saveSkill, switchSkill, type OwnSkill, type SkillFile } from '../../services/agentOwnSkills'

// ── Skills: the procedures a kiosk agent knows ─────────────────────────
//
// The third tab of "Instructions, insights & skills": the agent's skills
// (its /api/skills, through Flight Deck) — the ones its owner's setup gave
// it ("What it handles"), the ones it wrote or installed itself, and any
// built in. Pick one to read what it does; switch it on or off; correct one
// the agent keeps itself (built-in ones are read-only).

type Group = 'setup' | 'own' | 'builtin'
const OWN_SOURCES = new Set(['captain-managed', 'captain-workspace', 'agents-skills-personal', 'agents-skills-project'])

const GROUPS: Record<Group, { label: string; one: string; icon: LucideIcon; tone: string }> = {
  setup: { label: 'From your setup', one: 'From your setup', icon: Wand2, tone: 'bg-violet-500/10 text-violet-600 dark:text-violet-300' },
  own: { label: 'Its own', one: 'Its own', icon: Sparkles, tone: 'bg-sky-500/10 text-sky-700 dark:text-sky-300' },
  builtin: { label: 'Built in', one: 'Built in', icon: Blocks, tone: 'bg-zinc-500/10 text-zinc-300' },
}
const GROUP_ORDER: Group[] = ['setup', 'own', 'builtin']

function groupOf(s: OwnSkill): Group {
  if (s.from_owner) return 'setup'
  return (s.editable ?? OWN_SOURCES.has(s.source)) ? 'own' : 'builtin'
}

/** "weekly-sales-report" → "Weekly sales report". */
const titleOf = (name: string) => {
  const t = name.replace(/[-_]+/g, ' ').trim()
  return t ? t[0].toUpperCase() + t.slice(1) : name
}

const isOn = (s: OwnSkill) => s.enabled !== false

const SETUP_SOURCE = /^Source: (.+?) \(from your owner, when (?:they set you up|you were created)\)\s*$/m

/** A SKILL.md as shown: no frontmatter, and the setup's "Source:" line as a chip instead. */
function bodyOf(content: string): { body: string; source: string } {
  let text = content.replace(/\r\n?/g, '\n')
  if (text.startsWith('---')) {
    const end = text.indexOf('\n---', 3)
    if (end >= 0) text = text.slice(text.indexOf('\n', end + 1) + 1)
  }
  const m = text.match(SETUP_SOURCE)
  return { body: (m ? text.replace(m[0], '') : text).trim(), source: m ? m[1] : '' }
}

export function KioskSkillsPanel({ address, agentName, onTotal }: {
  /** The running agent; null when it's off. */
  address: AgentAddress | null
  agentName: string
  onTotal: (n: number) => void
}) {
  const [skills, setSkills] = useState<OwnSkill[] | null>(null)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [group, setGroup] = useState<Group | ''>('')
  const [picked, setPicked] = useState('')
  const [editing, setEditing] = useState(false)
  // Each skill's text, read once per dialog.
  const [files] = useState(() => new Map<string, SkillFile>())

  useEffect(() => {
    if (!address) return
    let gone = false
    listOwnSkills(address)
      .then((xs) => {
        if (gone) return
        setSkills(xs); setError(''); onTotal(xs.length)
      })
      .catch((e) => { if (!gone) setError(e instanceof Error ? e.message : String(e)) })
    return () => { gone = true }
  }, [address, onTotal])

  const counts = useMemo(() => {
    const c = new Map<Group, number>()
    for (const s of skills ?? []) c.set(groupOf(s), (c.get(groupOf(s)) ?? 0) + 1)
    return c
  }, [skills])
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    return (skills ?? [])
      .filter((s) => !group || groupOf(s) === group)
      .filter((s) => !q || `${s.name} ${s.description}`.toLowerCase().includes(q))
      .sort((a, b) => GROUP_ORDER.indexOf(groupOf(a)) - GROUP_ORDER.indexOf(groupOf(b)) || a.name.localeCompare(b.name))
  }, [skills, group, query])
  const current = (skills ?? []).find((s) => s.name === picked) ?? null

  if (!address) {
    return <Empty icon={Wand2} title={`${agentName} is off`}>Power it on to see the skills it has.</Empty>
  }

  const pick = (name: string) => {
    if (editing && !confirm('Discard the changes to this skill?')) return
    setEditing(false)
    setPicked(name)
  }
  const switched = (name: string, on: boolean) =>
    setSkills((xs) => (xs ?? []).map((s) => (s.name === name ? { ...s, enabled: on ? null : false } : s)))

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 space-y-3 border-b border-zinc-800 px-5 pb-3 pt-4">
        <p className="text-xs leading-relaxed text-zinc-500">
          The procedures {agentName} follows when a request fits — the ones your way of working gave it, the ones it
          wrote or installed itself, and any built in. Switch one off and it stops using it.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex h-9 min-w-[12rem] flex-1 items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-950 px-3 focus-within:border-violet-500/60">
            <Search className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search its skills"
              aria-label="Search skills"
              className="min-w-0 flex-1 bg-transparent text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none"
            />
            {query && (
              <button onClick={() => setQuery('')} aria-label="Clear search" className="rounded p-0.5 text-zinc-500 hover:text-zinc-300">
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </label>
          {(skills?.length ?? 0) > 0 && (
            <div className="-mx-1 flex items-center gap-1.5 overflow-x-auto px-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
              role="tablist" aria-label="Where they come from">
              <Chip on={group === ''} onClick={() => setGroup('')}>All <Count n={skills?.length ?? 0} /></Chip>
              {GROUP_ORDER.filter((g) => counts.get(g)).map((g) => (
                <Chip key={g} on={group === g} onClick={() => setGroup(group === g ? '' : g)}>
                  {GROUPS[g].label} <Count n={counts.get(g) ?? 0} />
                </Chip>
              ))}
            </div>
          )}
        </div>
      </div>

      {error ? (
        <div className="m-5 flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> Couldn't read its skills: {error}
        </div>
      ) : skills === null ? (
        <div className="space-y-2 p-5">
          {[0, 1, 2].map((n) => <div key={n} className="h-16 animate-pulse rounded-xl border border-zinc-800 bg-zinc-900/50" />)}
        </div>
      ) : skills.length === 0 ? (
        <Empty icon={Wand2} title="No skills yet">
          Give {agentName} your procedures through its way of working — each request it should handle becomes a
          skill — or it writes its own as it learns.
        </Empty>
      ) : (
        <div className="flex min-h-0 flex-1">
          <ul className={`${current ? 'hidden' : 'flex'} min-h-0 w-full flex-col gap-1.5 overflow-y-auto p-3 wide:flex wide:w-80 wide:shrink-0 wide:border-r wide:border-zinc-800`}>
            {shown.length === 0 && (
              <li className="px-3 py-8 text-center text-xs text-zinc-500">Nothing matches.</li>
            )}
            {shown.map((s) => {
              const g = GROUPS[groupOf(s)]
              const Icon = g.icon
              const on = s.name === picked
              return (
                <li key={s.name}>
                  <button
                    onClick={() => pick(s.name)}
                    aria-current={on}
                    className={`flex w-full items-start gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors ${
                      on ? 'border-violet-500/50 bg-violet-500/10' : 'border-transparent hover:border-zinc-800 hover:bg-zinc-900/60'
                    }`}
                  >
                    <span className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-sm ${g.tone}`}>
                      {s.emoji || <Icon className="h-4 w-4" />}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-1.5">
                        <span className={`truncate text-sm font-medium ${isOn(s) ? 'text-zinc-100' : 'text-zinc-500'}`}>{titleOf(s.name)}</span>
                        {!isOn(s) && <span className="shrink-0 rounded bg-zinc-800 px-1.5 text-[10px] font-medium uppercase tracking-wide text-zinc-400">Off</span>}
                      </span>
                      <span className="mt-0.5 line-clamp-2 text-xs leading-snug text-zinc-500">{s.description}</span>
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
          <section className={`${current ? 'flex' : 'hidden'} min-w-0 flex-1 flex-col wide:flex`}>
            {current ? (
              <SkillDetail
                key={current.name}
                skill={current}
                address={address}
                agentName={agentName}
                files={files}
                editing={editing}
                setEditing={setEditing}
                onBack={() => pick('')}
                onSwitched={switched}
              />
            ) : (
              <Empty icon={Wand2} title="Pick a skill">See what it does, switch it off, or correct it.</Empty>
            )}
          </section>
        </div>
      )}
    </div>
  )
}

function SkillDetail({ skill: s, address, agentName, files, editing, setEditing, onBack, onSwitched }: {
  skill: OwnSkill
  address: AgentAddress
  agentName: string
  files: Map<string, SkillFile>
  editing: boolean
  setEditing: (v: boolean) => void
  onBack: () => void
  onSwitched: (name: string, on: boolean) => void
}) {
  const [file, setFile] = useState<SkillFile | null>(files.get(s.name) ?? null)
  const [error, setError] = useState('')
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [switching, setSwitching] = useState(false)
  const g = GROUPS[groupOf(s)]
  const Icon = g.icon
  const editable = file?.editable ?? s.editable ?? OWN_SOURCES.has(s.source)

  useEffect(() => {
    if (files.has(s.name)) return
    let gone = false
    readSkill(address, s.name)
      .then((f) => { files.set(s.name, f); if (!gone) setFile(f) })
      .catch((e) => {
        if (gone) return
        const msg = e instanceof Error ? e.message : String(e)
        // An agent from before it could show a skill's text.
        setError(/404|Not Found/i.test(msg) ? `Restart ${agentName} to read its skills here.` : msg)
      })
    return () => { gone = true }
  }, [address, agentName, files, s.name])

  const shown = useMemo(() => (file ? bodyOf(file.content) : null), [file])

  const toggle = async () => {
    setSwitching(true); setError('')
    try {
      await switchSkill(address, s.skill_key, !isOn(s))
      onSwitched(s.name, !isOn(s))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSwitching(false)
    }
  }
  const startEdit = () => { if (file) { setDraft(file.content); setEditing(true) } }
  const cancel = () => {
    if (file && draft !== file.content && !confirm('Discard the changes to this skill?')) return
    setEditing(false)
  }
  const save = async () => {
    if (!file || busy) return
    setBusy(true); setError('')
    try {
      await saveSkill(address, s.name, draft)
      const next = { ...file, content: draft.trimEnd() + '\n' }
      files.set(s.name, next)
      setFile(next)
      setEditing(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  const onKey = (e: ReactKeyboardEvent) => {
    if (e.key === 'Escape') { e.preventDefault(); cancel() }
    if (e.key === 's' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); void save() }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="shrink-0 border-b border-zinc-800 px-5 py-4">
        <button onClick={onBack} className="mb-3 flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200 wide:hidden">
          <ArrowLeft className="h-3.5 w-3.5" /> All skills
        </button>
        <div className="flex items-start gap-3">
          <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-base ${g.tone}`}>
            {s.emoji || <Icon className="h-5 w-5" />}
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-base font-semibold text-zinc-100">{titleOf(s.name)}</h3>
            <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-zinc-500">
              <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-zinc-400">{s.user_invocable ? `/${s.name}` : s.name}</span>
              <span className="flex items-center gap-1">{s.from_owner ? <UserRound className="h-3 w-3" /> : null}{g.one}</span>
              {shown?.source && (
                <span className="flex max-w-[18rem] items-center gap-1 rounded-md bg-sky-500/10 px-1.5 py-0.5 text-sky-700 dark:text-sky-300" title={`From ${shown.source}`}>
                  <FileText className="h-3 w-3 shrink-0" /><span className="truncate">{shown.source}</span>
                </span>
              )}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {editable && !editing && (
              <button onClick={startEdit} disabled={!file}
                className="flex h-8 items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-40">
                <Pencil className="h-3.5 w-3.5" /> <span className="hidden wide:inline">Correct it</span>
              </button>
            )}
            <button
              role="switch"
              aria-checked={isOn(s)}
              onClick={toggle}
              disabled={switching}
              title={isOn(s) ? 'On — switch it off' : 'Off — switch it on'}
              className={`relative h-6 w-11 shrink-0 rounded-full transition-colors disabled:opacity-60 ${isOn(s) ? 'bg-violet-600' : 'bg-zinc-700'}`}
            >
              <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${isOn(s) ? 'left-[1.375rem]' : 'left-0.5'}`} />
            </button>
          </div>
        </div>
        <p className="mt-3 text-sm leading-relaxed text-zinc-300">{s.description}</p>
        {isOn(s) && !s.active && (
          <p className="mt-2 flex items-center gap-1.5 text-xs text-amber-700 dark:text-amber-400">
            <AlertTriangle className="h-3.5 w-3.5" /> {agentName} can't use it right now — something it needs is missing.
          </p>
        )}
        {error && (
          <p className="mt-2 flex items-start gap-1.5 text-xs text-red-700 dark:text-red-300">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {error}
          </p>
        )}
      </div>

      {editing ? (
        <div className="flex min-h-0 flex-1 flex-col gap-3 p-5" onKeyDown={onKey}>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            autoFocus
            spellCheck={false}
            className="min-h-[14rem] w-full flex-1 resize-none rounded-xl border border-zinc-700 bg-zinc-950 px-4 py-3 font-mono text-[13px] leading-relaxed text-zinc-200 focus:border-violet-500/60 focus:outline-none"
          />
          <div className="flex shrink-0 flex-wrap items-center gap-2">
            <span className="mr-auto text-[11px] text-zinc-500">
              Keep the <code className="font-mono text-zinc-400">name: {s.name}</code> line at the top. {agentName} uses the change from its next request.
            </span>
            <button onClick={cancel} disabled={busy}
              className="rounded-lg px-3 py-1.5 text-sm font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
              Cancel
            </button>
            <button onClick={save} disabled={busy || !file || draft === file.content}
              className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-1.5 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50">
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />} Save
            </button>
          </div>
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto">
          {shown ? (
            <div className="fd-file-markdown px-5 py-4 [&_ol]:list-decimal [&_ul]:list-disc">
              <Markdown remarkPlugins={[remarkGfm]}>{shown.body}</Markdown>
            </div>
          ) : !error ? (
            <div className="space-y-2 px-5 py-5">
              {[90, 70, 80, 50].map((w) => <div key={w} className="h-3 animate-pulse rounded bg-zinc-800" style={{ width: `${w}%` }} />)}
            </div>
          ) : null}
        </div>
      )}
    </div>
  )
}

function Chip({ on, onClick, children }: { on: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      role="tab"
      aria-selected={on}
      onClick={onClick}
      className={`flex h-7 shrink-0 items-center gap-1.5 rounded-full border px-2.5 text-xs transition-colors ${
        on ? 'border-violet-500/50 bg-violet-500/15 text-zinc-100' : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
      }`}
    >
      {children}
    </button>
  )
}

function Count({ n }: { n: number }) {
  return <span className="tabular-nums text-zinc-500">{n}</span>
}

function Empty({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children: ReactNode }) {
  return (
    <div className="flex h-full min-h-[14rem] flex-1 flex-col items-center justify-center px-8 text-center">
      <span className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-900 text-zinc-500">
        <Icon className="h-6 w-6" />
      </span>
      <p className="text-sm font-medium text-zinc-300">{title}</p>
      <p className="mt-1 max-w-sm text-xs leading-relaxed text-zinc-500">{children}</p>
    </div>
  )
}
