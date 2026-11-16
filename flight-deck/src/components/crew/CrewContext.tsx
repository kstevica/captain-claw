import { useEffect, useMemo, useRef, useState, type ChangeEvent, type ReactNode } from 'react'
import { ArrowLeft, Bot, Check, ChevronRight, FileText, Loader2, MessagesSquare, Paperclip, Plus, Search, Upload, X } from 'lucide-react'
import {
  crewApi,
  type CrewAgentFile, type CrewContextAgent, type CrewConversation, type CrewConversationMessage,
  type CrewConversationSession, type CrewStaged,
} from '../../services/crewApi'
import { EmptyState, Sheet, labelCls } from '../console/parts'
import { ErrorLine, ago, errorOf, fieldClass, primaryBtn, quietBtn } from './crewParts'

// ── Context from the person's own agents ──
//
// Files an agent made, and a stretch of what was said with one. Both are kept by the deck
// like uploads, so they go with the task or the message exactly as an attached file does.

const sizeText = (size: number) => (size >= 1_000_000 ? `${(size / 1_000_000).toFixed(1)} MB` : `${Math.max(1, Math.round(size / 1000))} KB`)

function when(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const sameYear = d.getFullYear() === new Date().getFullYear()
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', ...(sameYear ? {} : { year: 'numeric' }), hour: '2-digit', minute: '2-digit' })
}

const laneLabel = (c: Pick<CrewConversation, 'agent' | 'lane'>) => (c.lane && c.lane !== 'A' ? `${c.agent} · chat ${c.lane}` : c.agent)

type Which = 'files' | 'conversation' | null

/** What can go with a task or a message: files from here, an agent's files, a conversation.
 *  `compact`: one paperclip opening a menu, for a message box; else three buttons in a row. */
export function AttachChoices({ onPick, onStaged, compact = false }: {
  onPick: (ev: ChangeEvent<HTMLInputElement>) => void
  onStaged: (files: CrewStaged[]) => void
  compact?: boolean
}) {
  const input = useRef<HTMLInputElement>(null)
  const [open, setOpen] = useState<Which>(null)
  const [menu, setMenu] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!menu) return
    const away = (ev: MouseEvent) => { if (!menuRef.current?.contains(ev.target as Node)) setMenu(false) }
    const esc = (ev: KeyboardEvent) => { if (ev.key === 'Escape') setMenu(false) }
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', esc)
    return () => { document.removeEventListener('mousedown', away); document.removeEventListener('keydown', esc) }
  }, [menu])

  const choose = (which: Which | 'upload') => {
    setMenu(false)
    if (which === 'upload') input.current?.click()
    else setOpen(which)
  }

  const inline = 'inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200'
  const item = 'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm text-zinc-200 transition-colors hover:bg-zinc-800'

  return (
    <>
      <input ref={input} type="file" multiple onChange={onPick} className="hidden" aria-hidden="true" tabIndex={-1} />
      {compact ? (
        <div ref={menuRef} className="relative shrink-0">
          <button
            type="button" onClick={() => setMenu((m) => !m)} title="Attach" aria-label="Attach" aria-haspopup="menu" aria-expanded={menu}
            className="flex h-9 w-9 items-center justify-center rounded-xl text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
          >
            <Paperclip className="h-4 w-4" />
          </button>
          {menu && (
            <div role="menu" className="absolute bottom-full left-0 z-20 mb-2 w-64 rounded-xl border border-zinc-800 bg-zinc-950 p-1 shadow-xl">
              <button type="button" role="menuitem" onClick={() => choose('upload')} className={item}><Upload className="h-4 w-4 text-zinc-500" />Files from this computer</button>
              <button type="button" role="menuitem" onClick={() => choose('files')} className={item}><Bot className="h-4 w-4 text-zinc-500" />Files from an agent</button>
              <button type="button" role="menuitem" onClick={() => choose('conversation')} className={item}><MessagesSquare className="h-4 w-4 text-zinc-500" />A conversation with an agent</button>
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-1">
          <button type="button" onClick={() => choose('upload')} className={inline}><Paperclip className="h-3.5 w-3.5" />Attach files</button>
          <button type="button" onClick={() => choose('files')} className={inline}><Bot className="h-3.5 w-3.5" />From an agent</button>
          <button type="button" onClick={() => choose('conversation')} className={inline}><MessagesSquare className="h-3.5 w-3.5" />A conversation</button>
        </div>
      )}
      {open === 'files' && <AgentFilesSheet onClose={() => setOpen(null)} onStaged={onStaged} />}
      {open === 'conversation' && <ConversationSheet onClose={() => setOpen(null)} onStaged={onStaged} />}
    </>
  )
}

// ── Files an agent made ──

function AgentFilesSheet({ onClose, onStaged }: { onClose: () => void; onStaged: (files: CrewStaged[]) => void }) {
  const [agents, setAgents] = useState<CrewContextAgent[] | null>(null)
  const [agent, setAgent] = useState('')
  const [files, setFiles] = useState<Record<string, { files: CrewAgentFile[]; more: boolean } | undefined>>({})
  const [picked, setPicked] = useState<Record<string, string[]>>({})
  const [filter, setFilter] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    crewApi.contextAgents().then((list) => {
      setAgents(list)
      if (list[0]) setAgent(list[0].id)
    }).catch((e) => setError(errorOf(e)))
  }, [])

  useEffect(() => {
    if (!agent || files[agent]) return
    crewApi.agentFiles(agent)
      .then((got) => setFiles((old) => ({ ...old, [agent]: got })))
      .catch((e) => setError(errorOf(e)))
  }, [agent, files])

  const here = files[agent]
  const shown = useMemo(() => {
    const words = filter.trim().toLowerCase()
    const list = here?.files ?? []
    return words ? list.filter((f) => f.path.toLowerCase().includes(words)) : list
  }, [here, filter])
  const chosenHere = new Set(picked[agent] ?? [])
  const total = Object.values(picked).reduce((n, list) => n + list.length, 0)
  const nameOf = (id: string) => agents?.find((a) => a.id === id)?.name ?? id

  const toggle = (path: string) => setPicked((old) => {
    const list = old[agent] ?? []
    return { ...old, [agent]: list.includes(path) ? list.filter((p) => p !== path) : [...list, path] }
  })

  const attach = async () => {
    setBusy(true); setError('')
    const made: CrewStaged[] = []
    try {
      for (const [id, paths] of Object.entries(picked)) {
        if (paths.length) made.push(...await crewApi.attachAgentFiles(id, paths))
      }
      onStaged(made)
      onClose()
    } catch (e) {
      // What was copied before the refusal still goes: it's on the list, and can be taken off.
      if (made.length) onStaged(made)
      setError(errorOf(e))
      setBusy(false)
    }
  }

  return (
    <Sheet
      wide icon={Bot} title="Files from an agent" onClose={onClose}
      subtitle="Copies go with this. The agent keeps its own, and nothing it hasn't saved in its folder is here."
      footer={(
        <>
          <button type="button" onClick={onClose} className={quietBtn}>Cancel</button>
          <button type="button" onClick={attach} disabled={busy || !total} className={primaryBtn}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Paperclip className="h-4 w-4" />}
            {total ? `Attach ${total} file${total === 1 ? '' : 's'}` : 'Attach'}
          </button>
        </>
      )}
    >
      <div className="space-y-4">
        {error && <ErrorLine text={error} />}
        {agents === null && !error && <Loading />}
        {agents?.length === 0 && <EmptyState icon={Bot} title="You have no agents here">Files can come from agents of your own on this deck.</EmptyState>}
        {!!agents?.length && (
          <>
            <label className="block min-w-0">
              <span className={labelCls}>Agent</span>
              <select value={agent} onChange={(e) => { setAgent(e.target.value); setFilter('') }} className={`${fieldClass} font-normal`}>
                {agents.map((a) => {
                  const n = picked[a.id]?.length ?? 0
                  return <option key={a.id} value={a.id}>{a.name}{n ? ` (${n} chosen)` : ''}</option>
                })}
              </select>
            </label>
            {total > 0 && Object.keys(picked).some((id) => id !== agent && picked[id]?.length) && (
              <p className="text-xs text-zinc-500">
                Also chosen: {Object.entries(picked).filter(([id, list]) => id !== agent && list.length).map(([id, list]) => `${list.length} from ${nameOf(id)}`).join(', ')}
              </p>
            )}
            {!here && !error && <Loading />}
            {here && (here.files.length ? (
              <>
                <div className="relative">
                  <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
                  <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Find by name or folder" aria-label="Find a file" className={`${fieldClass} !pl-9 font-normal`} />
                </div>
                <ul className="divide-y divide-zinc-800/70 overflow-hidden rounded-xl border border-zinc-800" aria-label="Files">
                  {shown.map((f) => {
                    const on = chosenHere.has(f.path)
                    const folder = f.path.includes('/') ? f.path.slice(0, f.path.lastIndexOf('/')) : ''
                    return (
                      <li key={f.path}>
                        <button
                          type="button" role="checkbox" aria-checked={on} onClick={() => toggle(f.path)}
                          className={`flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors ${on ? 'bg-violet-500/10' : 'hover:bg-zinc-900'}`}
                        >
                          <Tick on={on} />
                          <FileText className="h-4 w-4 shrink-0 text-zinc-500" />
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-sm text-zinc-200">{f.name}</span>
                            {folder && <span className="block truncate text-[11px] text-zinc-500">{folder}</span>}
                          </span>
                          <span className="shrink-0 text-right text-[11px] text-zinc-500">{sizeText(f.size)}<br />{ago(f.modified)}</span>
                        </button>
                      </li>
                    )
                  })}
                  {!shown.length && <li className="px-3 py-6 text-center text-xs text-zinc-500">Nothing matches.</li>}
                </ul>
                {here.more && <p className="text-xs text-zinc-500">Only the newest {here.files.length} are listed. Find an older one by its name.</p>}
              </>
            ) : <EmptyState icon={FileText} title="No files yet">This agent hasn't saved anything in its folder.</EmptyState>)}
          </>
        )}
      </div>
    </Sheet>
  )
}

// ── What was said with an agent ──

interface Chosen {
  key: string
  lane: string
  label: string
  session: number
  sessionTitle: string
  first: number
  last: number
  count: number
}

function ConversationSheet({ onClose, onStaged }: { onClose: () => void; onStaged: (files: CrewStaged[]) => void }) {
  const [chats, setChats] = useState<{ list: CrewConversation[]; kept: boolean } | null>(null)
  const [chat, setChat] = useState<CrewConversation | null>(null)
  const [sessions, setSessions] = useState<CrewConversationSession[] | null>(null)
  const [session, setSession] = useState<CrewConversationSession | null>(null)
  const [msgs, setMsgs] = useState<{ messages: CrewConversationMessage[]; more: boolean } | null>(null)
  // The range, as indexes into msgs: everything to start with; a tap starts a new one, the next ends it.
  const [range, setRange] = useState<{ a: number; b: number; ending: boolean }>({ a: 0, b: 0, ending: false })
  const [basket, setBasket] = useState<Chosen[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    crewApi.conversations().then((r) => setChats({ list: r.conversations, kept: r.kept })).catch((e) => setError(errorOf(e)))
  }, [])

  const openChat = (c: CrewConversation) => {
    setChat(c); setSessions(null); setError('')
    crewApi.conversationSessions(c.id).then((list) => {
      setSessions(list)
      if (list.length === 1) openSession(c, list[0])
    }).catch((e) => setError(errorOf(e)))
  }

  const openSession = (c: CrewConversation, s: CrewConversationSession) => {
    setSession(s); setMsgs(null); setError('')
    crewApi.conversationMessages(c.id, s.id).then((got) => {
      setMsgs(got)
      setRange({ a: 0, b: Math.max(0, got.messages.length - 1), ending: false })
    }).catch((e) => setError(errorOf(e)))
  }

  const back = () => {
    setError('')
    if (session && sessions && sessions.length > 1) { setSession(null); setMsgs(null) } else { setChat(null); setSession(null); setSessions(null); setMsgs(null) }
  }

  const tap = (i: number) => setRange((r) => (r.ending ? { a: r.a, b: i, ending: false } : { a: i, b: i, ending: true }))
  const lo = Math.min(range.a, range.b)
  const hi = Math.max(range.a, range.b)
  const list = msgs?.messages ?? []
  const current: Chosen | null = chat && session && list.length ? {
    key: `${chat.id}:${session.id}:${list[lo].id}:${list[hi].id}`,
    lane: chat.id, label: laneLabel(chat), session: session.id, sessionTitle: session.title,
    first: list[lo].id, last: list[hi].id, count: hi - lo + 1,
  } : null
  const inBasket = !!current && basket.some((c) => c.key === current.key)
  const toSend = [...basket, ...(current && !inBasket ? [current] : [])]

  const addAnother = () => {
    if (current && !inBasket) setBasket((old) => [...old, current])
    setChat(null); setSession(null); setSessions(null); setMsgs(null)
  }

  const attach = async () => {
    setBusy(true); setError('')
    const made: CrewStaged[] = []
    try {
      for (const c of toSend) made.push(await crewApi.attachConversation(c.lane, c.session, c.first, c.last))
      onStaged(made)
      onClose()
    } catch (e) {
      if (made.length) onStaged(made)
      setBasket((old) => old.slice(made.length))
      setError(errorOf(e))
      setBusy(false)
    }
  }

  return (
    <Sheet
      wide icon={MessagesSquare} title="A conversation with an agent" onClose={onClose}
      subtitle="Your messages and the agent's final answers, as a file the worker reads. Its steps and tool calls are left out."
      footer={(
        <>
          {current && !inBasket && (
            <button type="button" onClick={addAnother} className={`${quietBtn} mr-auto`}><Plus className="h-4 w-4" />Add another</button>
          )}
          <button type="button" onClick={onClose} className={quietBtn}>Cancel</button>
          <button type="button" onClick={attach} disabled={busy || !toSend.length} className={primaryBtn}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Paperclip className="h-4 w-4" />}
            {toSend.length > 1 ? `Attach ${toSend.length} conversations` : 'Attach'}
          </button>
        </>
      )}
    >
      <div className="space-y-4">
        {basket.length > 0 && (
          <div>
            <p className={labelCls}>To attach</p>
            <ul className="flex flex-wrap gap-2">
              {basket.map((c) => (
                <li key={c.key} className="flex max-w-full items-center gap-2 rounded-xl border border-violet-500/30 bg-violet-500/10 py-1 pl-2.5 pr-1 text-xs text-zinc-200">
                  <span className="min-w-0 truncate"><span className="font-medium">{c.label}</span> · {c.count} message{c.count === 1 ? '' : 's'}</span>
                  <button type="button" onClick={() => setBasket((old) => old.filter((x) => x.key !== c.key))} aria-label={`Take ${c.label} off`} className="rounded-md p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                    <X className="h-3.5 w-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
        {error && <ErrorLine text={error} />}

        {!chat && (
          chats === null ? (!error && <Loading />)
            : !chats.kept ? <EmptyState icon={MessagesSquare} title="Conversations aren't kept here">Flight Deck keeps your chats only when sign-in is on.</EmptyState>
              : !chats.list.length ? <EmptyState icon={MessagesSquare} title="No conversations yet">Chat with one of your agents, and it can be handed on from here.</EmptyState>
                : (
                  <Rows label="Agents">
                    {chats.list.map((c) => (
                      <Row key={c.id} onClick={() => openChat(c)} title={laneLabel(c)} hint={c.updated_at ? `Last message ${when(c.updated_at)}` : ''} />
                    ))}
                  </Rows>
                )
        )}

        {chat && !session && (
          <>
            <BackLine onBack={back} label="All agents" title={laneLabel(chat)} />
            {sessions === null ? (!error && <Loading />) : !sessions.length ? (
              <EmptyState icon={MessagesSquare} title="Nothing said yet" />
            ) : (
              <Rows label="Sessions">
                {sessions.map((s) => (
                  <Row key={s.id} onClick={() => openSession(chat, s)} title={s.title} hint={[s.started_at && when(s.started_at), s.count !== null && `${s.count} messages`].filter(Boolean).join(' · ')} />
                ))}
              </Rows>
            )}
          </>
        )}

        {chat && session && (
          <>
            <BackLine onBack={back} label={sessions && sessions.length > 1 ? 'Sessions' : 'All agents'} title={`${laneLabel(chat)} · ${session.title}`} />
            {msgs === null ? (!error && <Loading />) : !list.length ? (
              <EmptyState icon={MessagesSquare} title="Nothing to send here">This session has no messages of yours or final answers.</EmptyState>
            ) : (
              <>
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-zinc-500">
                  <span aria-live="polite">
                    {range.ending ? 'Now tap the last message.' : `${hi - lo + 1} of ${list.length} chosen. Tap the first message, then the last, to choose fewer.`}
                  </span>
                  {(lo > 0 || hi < list.length - 1 || range.ending) && (
                    <button type="button" onClick={() => setRange({ a: 0, b: list.length - 1, ending: false })} className="font-medium text-violet-700 hover:underline dark:text-violet-300">All of it</button>
                  )}
                </div>
                <ol className="space-y-1.5" aria-label="Messages">
                  {list.map((m, i) => {
                    const on = i >= lo && i <= hi
                    const edge = i === lo || i === hi
                    return (
                      <li key={m.id}>
                        <button
                          type="button" onClick={() => tap(i)} aria-pressed={on}
                          className={`w-full rounded-xl border px-3 py-2 text-left transition-colors ${on
                            ? `${edge ? 'border-violet-500/60' : 'border-violet-500/25'} bg-violet-500/[0.07]`
                            : 'border-zinc-800 opacity-60 hover:opacity-100'}`}
                        >
                          <span className="flex items-center gap-2 text-[11px] text-zinc-500">
                            <span className={`font-semibold ${m.role === 'user' ? 'text-zinc-300' : 'text-violet-700 dark:text-violet-300'}`}>{m.role === 'user' ? 'You' : chat.agent}</span>
                            <span>{when(m.at)}</span>
                            {on && edge && <Check className="ml-auto h-3.5 w-3.5 text-violet-600 dark:text-violet-400" />}
                          </span>
                          <span className="mt-0.5 line-clamp-3 whitespace-pre-line break-words text-sm text-zinc-200">
                            {m.text}{m.chars > m.text.length ? '…' : ''}
                          </span>
                        </button>
                      </li>
                    )
                  })}
                </ol>
                {msgs.more && <p className="text-xs text-zinc-500">Only the first {list.length} messages of this session are listed.</p>}
              </>
            )}
          </>
        )}
      </div>
    </Sheet>
  )
}

// ── Small parts ──

function Loading() {
  return <div className="flex justify-center py-10 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
}

function Tick({ on }: { on: boolean }) {
  return (
    <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border ${on ? 'border-violet-500 bg-violet-600 text-white' : 'border-zinc-600'}`}>
      {on && <Check className="h-3 w-3" />}
    </span>
  )
}

function Rows({ label, children }: { label: string; children: ReactNode }) {
  return <ul aria-label={label} className="divide-y divide-zinc-800/70 overflow-hidden rounded-xl border border-zinc-800">{children}</ul>
}

function Row({ title, hint, onClick }: { title: string; hint?: string; onClick: () => void }) {
  return (
    <li>
      <button type="button" onClick={onClick} className="flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors hover:bg-zinc-900">
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm text-zinc-200">{title}</span>
          {hint && <span className="block truncate text-[11px] text-zinc-500">{hint}</span>}
        </span>
        <ChevronRight className="h-4 w-4 shrink-0 text-zinc-600" />
      </button>
    </li>
  )
}

function BackLine({ onBack, label, title }: { onBack: () => void; label: string; title: string }) {
  return (
    <div className="flex min-w-0 items-center gap-2">
      <button type="button" onClick={onBack} className="inline-flex shrink-0 items-center gap-1 rounded-lg px-1.5 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
        <ArrowLeft className="h-3.5 w-3.5" />{label}
      </button>
      <span className="min-w-0 truncate text-sm font-medium text-zinc-200">{title}</span>
    </div>
  )
}
