// One usage_logs row in plain words, for Recent activity (the console's Usage
// page and the Admin page). Rows store their detail as JSON with user ids and
// field names; an admin reading the list wants "Lena Park gave Tomás Ruiz a
// free reset", not "count: 1 · user ids: […]". Unknown events still read
// sensibly: their fields are listed, with any user id replaced by the name.

export interface ActivityRow {
  /** Short tag for the row, such as "Free resets". */
  label: string
  /** The sentence, starting with who did it. */
  text: string
}

type Detail = Record<string, unknown>

const PROVIDERS: Record<string, string> = {
  anthropic: 'Anthropic', openai: 'OpenAI', gemini: 'Gemini', google: 'Gemini',
  openrouter: 'OpenRouter', xai: 'xAI',
}
const MAILBOX: Record<string, string> = { google: 'Gmail', microsoft: 'Outlook' }
const DEFAULT_FIELDS: Record<string, string> = { company: 'company profile', instructions: 'instructions' }

const str = (v: unknown) => (typeof v === 'string' ? v : v == null ? '' : String(v))
const num = (v: unknown) => (typeof v === 'number' ? v : Number(v) || 0)
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`
const andList = (xs: string[]) => (xs.length <= 1 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`)

/** "proposal-writer" → "Proposal Writer". */
export function agentName(slug: string): string {
  return slug.split(/[-_]+/).filter(Boolean).map((w) => w[0].toUpperCase() + w.slice(1)).join(' ') || 'an agent'
}

function tokens(n: number): string {
  return n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1000 ? `${Math.round(n / 1000)}k` : String(n)
}

function bytes(n: number): string {
  return n >= 1_048_576 ? `${(n / 1_048_576).toFixed(1)} MB` : n >= 1024 ? `${Math.round(n / 1024)} KB` : `${n} B`
}

/** What an admin did for someone (X-FD-Act-As), from the request it made: "started Proposal Writer for Maya Novak". */
function actedFor(method: string, path: string, whom: string): string {
  const agent = path.match(/^\/fd\/processes\/([^/]+)\/(start|stop|restart)$/)
  if (agent) return `${agent[2] === 'stop' ? 'stopped' : agent[2] === 'start' ? 'started' : 'restarted'} ${agentName(agent[1])} for ${whom}`
  const removed = path.match(/^\/fd\/processes\/([^/]+)$/)
  if (removed && method === 'DELETE') return `removed ${agentName(removed[1])} for ${whom}`
  if (path === '/fd/spawn-process' || path === '/fd/spawn') return `started a new agent for ${whom}`
  const verb = method === 'DELETE' ? 'removed something' : method === 'GET' ? 'looked at something' : 'changed something'
  return `${verb} for ${whom} (${path})`
}

export function activityText(eventType: string, rawDetail: string, actor: string, nameOf: (id: string) => string): ActivityRow {
  let d: Detail = {}
  try {
    const parsed = JSON.parse(rawDetail || '{}')
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) d = parsed as Detail
  } catch { /* not JSON: shown as it is below */ }
  const who = (key: string) => nameOf(str(d[key]))
  const by = d.acting_admin ? `, for them by ${who('acting_admin')}` : ''
  const failed = d.ok === false ? ', which failed' : ''

  switch (eventType) {
    case 'agent_spawn':
      return { label: 'Agent started', text: `${actor} started ${agentName(str(d.agent))}${d.type === 'container' ? ' in Docker' : ''}${by}` }
    case 'admin_act_as':
      return { label: 'Acted for', text: `${actor} ${actedFor(str(d.method), str(d.path), who('target_user'))}` }
    case 'usage_reset_today':
      return { label: 'Day reset', text: `${actor} gave ${who('user_id')} today's messages and tokens back` }
    case 'usage_reset_used':
      return { label: 'Free reset', text: `${actor} used a free reset (${num(d.left)} left)` }
    case 'usage_resets_granted': {
      const count = num(d.count), cap = num(d.cap)
      const resets = count === 1 ? 'a free reset' : `${count} free resets`
      const ids = Array.isArray(d.user_ids) ? d.user_ids.map((id) => nameOf(str(id))) : []
      const whom = d.all ? 'everyone' : ids.length && ids.length === num(d.users) ? andList(ids) : plural(num(d.users), 'person', 'people')
      const full = num(d.users) - num(d.given)
      return {
        label: 'Free resets',
        text: `${actor} gave ${whom} ${resets}, at most ${cap} each${full > 0 ? ` (${full} already held the most)` : ''}`,
      }
    }
    case 'usage_resets_set':
      return { label: 'Free resets', text: `${actor} set ${who('user_id')}'s free resets to ${num(d.balance)}` }
    case 'feedback_requested':
      return { label: 'Feedback', text: `${actor} asked ${who('user_id')} for feedback${MAILBOX[str(d.via)] ? ` by ${MAILBOX[str(d.via)]}` : ''}` }
    case 'profile_defaults_update': {
      const fields = Array.isArray(d.fields) ? d.fields.map((f) => DEFAULT_FIELDS[str(f)] || str(f).replace(/_/g, ' ')) : []
      const cleared = num(d.company_chars) === 0 && num(d.instructions_chars) === 0
      return {
        label: 'Deck-wide defaults',
        text: `${actor} ${cleared ? 'cleared' : 'changed'} the deck-wide ${andList(fields) || 'defaults'} (${plural(num(d.agents_updated), 'agent')} updated)`,
      }
    }
    case 'course_settings_update':
      return { label: 'Settings', text: `${actor} changed the Course settings` }
    case 'byok_key_saved':
      return { label: 'Own keys', text: `${actor} added their own ${PROVIDERS[str(d.provider)] || str(d.provider)} key` }
    case 'byok_key_removed':
      return { label: 'Own keys', text: `${actor} removed their own ${PROVIDERS[str(d.provider)] || str(d.provider)} key` }
    case 'byok_on':
      return { label: 'Own keys', text: `${actor} turned on their own API keys` }
    case 'byok_off':
      return {
        label: 'Own keys',
        text: d.by_admin ? `${who('by_admin')} moved ${actor} back to the deck's models` : `${actor} turned off their own API keys`,
      }
    case 'browser_connected':
      return { label: 'Browser', text: `${actor} connected ${str(d.browser) || 'a browser'}` }
    case 'browser_clip':
      return { label: 'Clip', text: `${actor} clipped a page${d.site ? ` from ${str(d.site)}` : ''} to ${agentName(str(d.agent))}` }
    case 'screen_clip':
      return { label: 'Clip', text: `${actor} clipped ${str(d.app) || 'the screen'} to ${agentName(str(d.agent))}` }
    case 'file_upload':
      return { label: 'File', text: `${actor} uploaded ${str(d.filename) || 'a file'}${d.size ? ` (${bytes(num(d.size))})` : ''}` }
    case 'file_save':
      return { label: 'File', text: `${actor} saved ${str(d.path) || 'a file'}` }
    case 'shared_agent_file_upload':
      return { label: 'Shared agent', text: `${actor} gave a shared agent ${str(d.filename) || 'a file'}` }
    case 'shared_agent_file_delete':
      return { label: 'Shared agent', text: `${actor} removed a file from a shared agent` }
    case 'jev_call': {
      const used = num(d.input_tokens) + num(d.output_tokens)
      return {
        label: 'Jev',
        text: `${d.agent ? `${agentName(str(d.agent))} (${actor})` : actor} asked Jev ${plural(num(d.questions) || 1, 'question')}${used ? `, ${tokens(used)} tokens` : ''}${failed}`,
      }
    }
    case 'soniox_call':
      return { label: 'Soniox', text: `${actor} used Soniox${d.kind ? ` (${str(d.kind).replace(/_/g, ' ')})` : ''}${failed}` }
    case 'external_memory_call':
      return {
        label: 'Memory',
        text: `${d.agent ? `${agentName(str(d.agent))} (${actor})` : actor}: ${str(d.action) || 'a call'} on ${str(d.provider) || 'external memory'}${failed}`,
      }
    case 'gmail_autoreply':
      return { label: 'Auto-reply', text: `${actor}'s Gmail auto-reply (${str(d.stage) || 'a step'}): ${tokens(num(d.tokens))} tokens${failed}` }
  }

  // Anything else: its fields, with user ids as names.
  const fields = Object.entries(d).map(([k, v]) => {
    const key = k.replace(/_/g, ' ')
    if (typeof v === 'string' && /(^|_)(user|user_id|admin|by_admin|acting_admin|target_user)$/.test(k)) return `${key}: ${nameOf(v)}`
    return `${key}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`
  })
  const label = eventType.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())
  return { label, text: [actor, ...(fields.length ? fields : rawDetail && rawDetail !== '{}' && !Object.keys(d).length ? [rawDetail] : [])].join(' · ') }
}
