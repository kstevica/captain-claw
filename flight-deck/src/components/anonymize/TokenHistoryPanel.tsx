import { useEffect, useState } from 'react'
import { ArrowDownLeft, ArrowUpRight, Sparkles } from 'lucide-react'
import { anonymizeApi, type TokenHistory, type TokenUse } from '../../services/anonymizeApi'

// One token's history: what put it in the vault, and every time it went out
// to a model (in a new text) and came back and had to be put back. Contexts
// are as they were sent — tokens; hover one to see its real value.

const TOKEN_SPLIT = /(⟦[A-Z][A-Z0-9_]*?_\d{1,6}⟧)/

const FOUND_BY: Record<string, string> = {
  email: 'an email address', 'phone-intl': 'a phone number', 'phone-label': 'a phone label',
  iban: 'an IBAN', card: 'a card number', oib: 'an OIB (its check digit)', 'oib-label': 'an OIB label',
  jmbg: 'a JMBG', 'jmbg-label': 'a JMBG label', 'mb-label': 'an MB label', 'id-label': 'an ID label',
  ip: 'an IP address', 'profile-url': 'a profile link', 'person-label': 'a name label',
  'org-label': 'a company label', 'address-label': 'an address label', 'dob-label': 'a date of birth label',
  ner: 'the name model', hint: 'a column name', known: 'a value it already knew',
  'known-digits': 'a number it already knew',
}

const CHANNEL_OUT: Record<string, string> = {
  'gate-chat': 'the chat', gate: 'a tool result', text: 'anonymized text', vfs: 'a file', 'vfs-folder': 'a folder',
  library: 'a Library file', browser: 'a web page in Chrome', screen: 'a screenshot in Chrome', names: 'folder names',
  file: 'a file',
}

const FRAME: Record<string, string> = {
  chat_message: 'a reply', thinking: 'a narration line', narration: 'a narration line', tool_call: 'a tool step',
  tool_result: 'a tool step',
}

function who(actor: string): string {
  if (!actor || actor === 'person') return 'you'
  if (actor.startsWith('agent:')) return `agent ${actor.slice(6)}`
  if (actor.startsWith('browser:')) return 'Chrome'
  return actor
}

function foundBy(u: TokenUse): string {
  if (u.found_by === 'extra') return u.model ? `the trusted model (${u.model})` : 'your list of names'
  if (u.found_by === 'known-slug') return `a spelling of ${u.related || 'a known name'} (in a link or a file name)`
  return FOUND_BY[u.found_by] || (u.found_by ? `code (${u.found_by})` : 'code')
}

function purpose(label: string): string {
  if (!label) return ''
  if (label.startsWith('tool:')) return `tool ${label.slice(5)}`
  if (label === 'message') return 'your message'
  if (label === 'borrowed') return 'a Flight Deck feature'
  if (label === 'task-name') return "the task's name"
  return label.replace(/_/g, ' ')
}

function describe(u: TokenUse): string {
  if (u.direction === 'added') {
    const where = CHANNEL_OUT[u.channel] || u.channel || 'a text'
    return `Added from ${where}${u.label ? ` (${purpose(u.label)})` : ''} — found by ${foundBy(u)}`
  }
  if (u.direction === 'out') {
    const where = CHANNEL_OUT[u.channel] || u.channel || 'a text'
    return `Went out in ${where}${u.model ? ` to ${u.model}` : ''}${u.label ? ` · ${purpose(u.label)}` : ''}`
  }
  if (u.channel === 'chat') return `Put back in your chat — ${FRAME[u.label] || u.label || 'a message'}`
  if (u.channel === 'browser') return 'Put back in Chrome'
  if (u.channel === 'restore') return 'Put back in a text you restored'
  const p = purpose(u.label)
  if (u.label.startsWith('tool:')) return `Put back in what ${u.model || 'the model'} sent to ${p}`
  return `Put back for ${p || u.channel}${u.model ? ` (from ${u.model})` : ''}`
}

function Context({ text, token, values }: { text: string; token: string; values: Map<string, string> }) {
  if (!text) return null
  return (
    <p className="mt-1 break-words rounded-lg bg-zinc-900/60 px-2 py-1 font-mono text-[11px] leading-relaxed text-zinc-300">
      {text.split(TOKEN_SPLIT).map((part, i) => TOKEN_SPLIT.test(part) ? (
        <span key={i} title={values.get(part) || ''}
          className={part === token ? 'rounded bg-violet-500/25 px-0.5 text-violet-200' : 'text-zinc-400 underline decoration-dotted'}>
          {part}
        </span>
      ) : <span key={i}>{part}</span>)}
    </p>
  )
}

function Line({ u, token, values }: { u: TokenUse; token: string; values: Map<string, string> }) {
  const Icon = u.direction === 'added' ? Sparkles : u.direction === 'out' ? ArrowUpRight : ArrowDownLeft
  const tone = u.direction === 'added' ? 'text-emerald-400' : u.direction === 'out' ? 'text-amber-400' : 'text-sky-400'
  return (
    <li className="flex gap-2">
      <Icon className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${tone}`} />
      <div className="min-w-0 flex-1">
        <div className="text-xs text-zinc-300">
          {describe(u)}
          {u.count > 1 && <span className="text-zinc-500"> · {u.count}×</span>}
        </div>
        <div className="text-[11px] text-zinc-500">{new Date(u.at * 1000).toLocaleString()} · {who(u.actor)}</div>
        <Context text={u.context} token={token} values={values} />
      </div>
    </li>
  )
}

export function TokenHistoryPanel({ vaultId, token, values }: { vaultId: string; token: string; values: Map<string, string> }) {
  const [h, setH] = useState<TokenHistory | null>(null)
  const [error, setError] = useState('')
  const [all, setAll] = useState(false)
  const [loading, setLoading] = useState(false)

  const older = async () => {
    if (!h || !h.uses.length) return
    setLoading(true)
    try {
      const page = await anonymizeApi.tokenHistory(vaultId, token, h.uses[h.uses.length - 1].id)
      setH({ ...h, uses: [...h.uses, ...page.uses], more: page.more })
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't load more")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let live = true
    anonymizeApi.tokenHistory(vaultId, token)
      .then((r) => { if (live) setH(r) })
      .catch((e) => { if (live) setError(e instanceof Error ? e.message : "Couldn't load its history") })
    return () => { live = false }
  }, [vaultId, token])

  if (error) return <p className="text-xs text-red-400">{error}</p>
  if (!h) return <p className="text-xs text-zinc-500">Loading…</p>
  const uses = all ? h.uses : h.uses.slice(0, 12)
  return (
    <div className="space-y-2">
      <p className="text-xs text-zinc-400">
        Went out in {h.stats.sent} text{h.stats.sent === 1 ? '' : 's'} · put back {h.stats.restored} time{h.stats.restored === 1 ? '' : 's'}
      </p>
      <ul className="space-y-2.5">
        {h.added
          ? <Line u={h.added} token={token} values={values} />
          : <li className="text-xs text-zinc-500">Added before its history was kept.</li>}
        {uses.map((u, i) => <Line key={i} u={u} token={token} values={values} />)}
      </ul>
      {h.uses.length > uses.length && (
        <button type="button" onClick={() => setAll(true)} className="text-xs font-medium text-violet-600 hover:underline dark:text-violet-400">
          Show {h.uses.length - uses.length} more
        </button>
      )}
      {h.uses.length === uses.length && h.more && (
        <button type="button" onClick={older} disabled={loading} className="text-xs font-medium text-violet-600 hover:underline disabled:opacity-60 dark:text-violet-400">
          {loading ? 'Loading…' : 'Load older'}
        </button>
      )}
      {!h.uses.length && h.added && <p className="text-xs text-zinc-500">Not sent or put back since.</p>}
    </div>
  )
}
