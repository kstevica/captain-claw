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

// ── An agent's tool calls, as a colleague would narrate them ──
//
// The chat shows what an agent is doing as "Captain is checking your
// calendar", not "google_calendar". toolActivity gives the words after "is":
// present-continuous, lower-case, no subject, so the caller puts the agent's
// (or a peer's) name in front. A tool not listed here is "working on it"; the
// raw tool name stays in the chat's "See what happened".

type Args = Record<string, unknown> | undefined

const WRITES = /^(send|reply|forward|create|insert|add|update|edit|set|delete|remove|move|write|upsert|draft|schedule|cancel)/
const action = (args: Args) => str(args?.action ?? args?.operation ?? args?.op ?? args?.command).toLowerCase()
const writes = (args: Args) => WRITES.test(action(args))

function mail(args: Args): string {
  const a = action(args)
  if (/^draft/.test(a)) return 'drafting an email'
  if (/^(send|reply|forward)/.test(a)) return 'sending an email'
  return 'checking your email'
}

const TOOL_ACTIVITY: Record<string, string | ((args: Args) => string)> = {
  google_mail: mail, microsoft_mail: mail, imap_mail: mail,
  send_mail: 'sending an email',
  google_calendar: (a) => (writes(a) ? 'updating your calendar' : 'checking your calendar'),
  microsoft_calendar: (a) => (writes(a) ? 'updating your calendar' : 'checking your calendar'),
  google_drive: 'looking through your files', microsoft_onedrive: 'looking through your files',
  datastore: (a) => (writes(a) ? 'updating your records' : 'looking through your records'),
  contacts: (a) => (writes(a) ? 'updating your contacts' : 'looking up a contact'),
  read: 'reading a file', workspace_read: 'reading a file', vfs: 'looking through your files',
  glob: 'looking through your files', grep: 'searching your files', summarize_files: 'reading your files',
  pdf_extract: 'reading a PDF', ocr_pdf: 'reading a PDF', docx_extract: 'reading a document',
  xlsx_extract: 'reading a spreadsheet', pptx_extract: 'reading a slide deck', ocr_image: 'reading text in a picture',
  write: 'saving a file', edit: 'editing a file', workspace_write: 'saving a file',
  docx: 'putting a document together', xlsx: 'building a spreadsheet', pptx: 'putting slides together',
  web_search: 'searching the web', web_fetch: 'reading a web page', web_fetch_batch: 'reading a few web pages',
  web_get: 'reading a web page', browser: 'browsing the web', pinchtab: 'browsing the web',
  consult_peer: 'checking with a teammate', colleagues: 'checking with a teammate', crew: 'working with the crew',
  captain_swarm: 'working with the crew', botport: 'checking with another agent',
  shell: 'running a quick task', terminal: 'running a quick task', code: 'running a quick task',
  scripts: 'running a script', app_runner: 'running an app', termux: 'running a quick task',
  image_gen: 'making a picture', image_manipulation: 'touching up a picture',
  image_vision: 'looking at a picture', cv: 'looking at a picture', video_vision: 'watching a video',
  audio: 'listening to the audio', pocket_tts: 'recording a voice note',
  knowledge: 'checking its notes', facts: 'checking its notes', project_memory: 'checking its notes',
  external_memory: 'checking its notes', history: 'looking back through past chats',
  insights: 'checking its notes', intentions: 'checking its plans',
  todo: 'updating its to-do list', cron: 'setting up a schedule',
  whatsapp_send_file: 'sending a file on WhatsApp', whatsapp_web: 'checking WhatsApp',
  telegram_inbox: 'checking Telegram', line: 'checking LINE', twitter: 'checking X',
  payments: 'preparing a payment', spend: 'preparing a payment',
  ask_user: 'asking you something', ask_human: 'asking you something',
  playbooks: 'pulling up a playbook', visualize: 'drawing a chart', synthesize_flow: 'sketching a workflow',
  personality: 'adjusting its settings', model_switch: 'adjusting its settings', reasoning_effort: 'adjusting its settings', drive_level: 'adjusting its settings',
  sessions: 'switching topics', topics: 'switching topics', clipboard: 'using the clipboard',
  hosting: 'publishing a page', public_link: 'making a share link',
  custom_tools: 'using one of your tools', apis: 'using a connected service', direct_api: 'using a connected service',
  linked_system: 'using a connected service', anonymize: 'hiding personal details',
  watch_background: 'keeping an eye on a background job', self_check: 'double-checking its work',
  flight_deck: 'checking Flight Deck', course: 'checking the course', desktop_action: 'working on your desktop',
  ocr: 'reading text in a picture',
}

/** What a tool call is doing, after "<agent> is": "checking your calendar". */
export function toolActivity(tool: string | undefined, args?: Args): string {
  const t = TOOL_ACTIVITY[(tool || '').trim()]
  if (!t) return 'working on it'
  return typeof t === 'function' ? t(args) : t
}

/** The agent's status line ("Using web_search...", "Calling LLM (…)") the
 *  same way, or null for a status that is plain words already. */
export function statusActivity(status: string | undefined): string | null {
  const s = (status || '').trim()
  const using = s.match(/^Using\s+([\w-]+)\s*(?:\.\.\.|…)?$/)
  if (using) return toolActivity(using[1])
  if (/^(Calling LLM\b|Thinking|Working)/i.test(s)) return 'thinking'
  if (/^Stopping/i.test(s)) return 'stopping'
  if (/^Waking up/i.test(s)) return 'waking up'
  return null
}
