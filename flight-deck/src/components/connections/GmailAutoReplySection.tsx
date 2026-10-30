import { useEffect, useRef, useState } from 'react'
import {
  Bot,
  Loader2,
  Check,
  AlertCircle,
  Save,
  RefreshCw,
  Pause,
  Plus,
  Trash2,
  ChevronDown,
  ChevronRight,
  FlaskConical,
  Send,
  Copy,
  ExternalLink,
  Play,
  Folder,
  CornerUpLeft,
  X,
  Undo2,
} from 'lucide-react'
import { useAuthStore, selectKioskLocked } from '../../stores/authStore'
import { useGoogleAuthStore } from '../../stores/googleAuthStore'
import {
  AUTO_REPLY_LIMITS,
  autoReplyOptInLocked,
  foldText,
  gmailThreadUrl,
  listFolderProjects,
  listSubfolders,
  multiLine,
  normalizeFolder,
  normalizeSenders,
  oneLine,
  requirementText,
  senderAllowed,
  TRY_IT_SAMPLE_FROM,
  tryItSender,
  useGmailAutoReplyStore,
  validateTopic,
  type AutoReplyActivity,
  type AutoReplyChecker,
  type AutoReplyDryRunResult,
  type AutoReplyFinding,
  type AutoReplyPatch,
  type AutoReplyTestResult,
  type AutoReplyTopic,
  type AutoReplyTopicErrors,
  type AutoReplyTopicInput,
  type AutoReplyView,
} from '../../stores/gmailAutoReplyStore'

// Topic auto-replies (Connections → Google): the user's chosen agent answers
// emails that are only about one of their narrow topics, without asking them.
// Off until the user turns it on here; every requirement is the server's to
// check (gmail_autoreply_routes.py). Rendered right after Email sending, for
// the signed-in user's own account only.


const pill = 'inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border'
const field = 'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500 focus:outline-none disabled:opacity-60'
const label = 'block text-xs font-medium text-zinc-400 mb-1'
const hint = 'mt-1 text-[11px] text-zinc-500 leading-snug'
const fieldError = 'mt-1 text-[11px] text-red-500'
const smallBtn = 'inline-flex items-center gap-1.5 rounded-md border border-zinc-700 hover:bg-zinc-800 px-2.5 py-1 text-xs text-zinc-300 disabled:opacity-50 disabled:cursor-not-allowed'

function fmtTime(ts: string): string {
  if (!ts) return ''
  const d = new Date(ts)
  return isNaN(d.getTime()) ? ts : d.toLocaleString()
}

// ── Words for the server's codes ────────────────────────────────────

const TIER_NAMES: Record<string, string> = { balanced: 'Balanced', reason: 'Reason' }

// "Jev" / "your Reason model (anthropic/claude-…)" / "nothing".
function checkerLabel(c: AutoReplyChecker | undefined | null): string {
  if (!c || !c.source) return 'nothing'
  if (c.source === 'jev') {
    return c.fallback?.source ? `Jev (else ${checkerLabel(c.fallback)})` : 'Jev'
  }
  const tier = TIER_NAMES[c.tier] || c.tier || 'Library'
  const model = [c.provider, c.model].filter(Boolean).join('/')
  return `your ${tier} model${model ? ` (${model})` : ''}`
}

// Who reads the mail and the drafted replies for the checks.
function checkerParties(...checkers: (AutoReplyChecker | undefined)[]): string[] {
  const out: string[] = []
  const add = (c: AutoReplyChecker | null | undefined) => {
    if (!c || !c.source) return
    const who = c.source === 'jev' ? 'Jev' : c.provider || 'your model provider'
    if (!out.includes(who)) out.push(who)
    add(c.fallback)
  }
  checkers.forEach(add)
  return out
}

const DISABLED_REASONS: Record<string, string> = {
  paused_by_user: 'You paused topic auto-replies.',
  google_disconnected: 'Turned off because Google was disconnected.',
  google_account_changed: 'Turned off because a different Google account was connected.',
  auto_paused:
    'Paused automatically: several replies were blocked for sensitive data within a day. ' +
    'Check the activity below before turning them on again.',
}

function lastErrorText(code: string): string {
  if (!code) return ''
  if (code.startsWith('missing_scopes:')) {
    return `Google doesn't allow ${code.slice('missing_scopes:'.length).split(',').join(' and ')}.`
  }
  if (code === 'gmail_unreachable') return "Gmail couldn't be reached — retrying."
  if (/^gmail_\d+$/.test(code)) return `Gmail answered ${code.slice(6)} — retrying.`
  const known: Record<string, string> = {
    google_not_connected: "Google isn't connected.",
    no_agent: 'No agent is chosen, so nothing is answered.',
    no_topics: 'No topic is on, so nothing is answered.',
    burst_truncated: 'A burst of mail arrived: only the newest was considered.',
    history_expired: 'Flight Deck was away too long: older mail was skipped.',
    state_missing: "On, but not running — turn it off and on again.",
  }
  return known[code] || code
}

const AGENT_REASONS: Record<string, string> = {
  no_agent: 'none chosen',
  agent_not_found: 'not one of your agents',
  agent_managed: 'a worker Flight Deck runs for itself',
  agent_no_token: 'no access token',
  agent_no_port: 'no port',
  agent_stopped: 'stopped',
  agent_unavailable: "Flight Deck can't read its agent records right now",
}

// What a local-scan finding is (outbound_guard.KIND_LABELS).
const KIND_LABELS: Record<string, string> = {
  secret_key: 'an API key or token',
  jwt: 'a login token',
  private_key: 'a private key',
  url_credentials: 'a link with a password in it',
  token_param: 'a link carrying an access token',
  seed_phrase: 'a recovery phrase',
  credential_phrase: 'a password or PIN',
  high_entropy: 'a secret-looking code',
  card_number: 'a card number',
  internal_host: 'an internal network address',
  local_path: 'a file path',
  canary: 'its own instructions',
  prompt_echo: 'its own instructions',
  private_copy: 'text from your private profile',
  never_disclose: 'one of your never-mention terms',
  guard_error: "something the safety check couldn't read",
  format_control: 'hidden formatting characters',
  iban: 'a bank account number',
  oib: 'an OIB',
  jmbg: 'a JMBG',
  ssn: 'a social security number',
  long_number: 'a long number',
  email_address: 'an email address',
  phone_number: 'a phone number',
  amount: 'an amount',
  url: 'a link',
  html: 'HTML',
  markdown_image: 'an image',
}

function kindLabel(kind: string): string {
  return KIND_LABELS[kind] || kind.replace(/_/g, ' ')
}

// Why a ledger row ended where it did, in a few words.
const REASON_LABELS: Record<string, string> = {
  labels: 'not plain inbox mail',
  before_enable: 'arrived before auto-replies were on',
  stale: 'too old when picked up',
  from_invalid: "sender's address unreadable",
  own_mail: 'from you',
  automated_sender: 'automated sender',
  auto_or_list: 'list or automated mail',
  unauthenticated: 'sender not verified',
  reply_to_divergent: 'Reply-To points elsewhere',
  not_addressed_to_me: 'not addressed to you',
  group_mail: 'sent to a group',
  empty: 'empty email',
  too_long: 'too long',
  check_error: 'a check failed',
  no_match: 'not about a topic',
  partial_match: 'only partly about a topic',
  mixed_topic: 'asks for more than one topic covers',
  sender_not_in_topic: "sender not in the topic's list",
  classifier_unavailable: "couldn't be sorted",
  judge_unavailable: 'the reply check was unavailable',
  gone: 'gone from Gmail',
  fetch: "couldn't be read from Gmail",
  burst: 'part of a burst of mail',
  cap_thread: 'thread limit reached',
  cap_sender: 'sender limit reached',
  cap_flood: 'too much mail from this sender',
  flood_suppressed: 'sender paused after a flood of mail',
  classified_automated: 'reads as automated mail',
  from_not_plain: "sender's address isn't a plain address",
  thread_has_unanswered: 'an earlier email in the thread is still unanswered',
  brief_changed: "the topic's brief changed while it was being written",
  cap_daily: 'daily limit reached',
  cap_classify: 'daily sorting budget used up',
  topic_off: 'topic switched off',
  draft_in_thread: 'you have a draft in the thread',
  owner_in_thread: 'you already replied in the thread',
  thread_unreadable: "thread couldn't be read",
  agent_offline: 'the agent was offline for an hour',
  agent_stopped: 'the agent is stopped',
  agent_unavailable: "agent records unreadable",
  guard_error: "the safety check couldn't read the reply",
  no_signature: 'no agentic signature set',
  turned_off: 'turned off meanwhile',
  already_answered: 'already answered',
  already_handled: 'already handled',
  build_error: "the email couldn't be built",
  send_error: 'Gmail refused the send',
  send_unconfirmed: 'Gmail never confirmed the send',
  internal_error: 'internal error',
  expired: 'not sent within 7 days',
  writer_skip: 'the agent chose not to answer',
  writer_format: "the agent's reply was malformed",
  writer_unavailable: "the agent's model didn't answer",
  no_sender: 'no sender to answer',
  format: 'the reply had hidden formatting',
}

function reasonLabel(code: string): string {
  if (!code) return ''
  if (code.startsWith('guard:')) {
    const kinds = code.slice(6).split(',').filter(Boolean).map(kindLabel)
    return `the safety scan found ${[...new Set(kinds)].join(', ') || 'something sensitive'}`
  }
  if (code.startsWith('judge:')) {
    return `the reply check flagged: ${code.slice(6).split(',').join(', ').replace(/_/g, ' ')}`
  }
  return REASON_LABELS[code] || code.replace(/_/g, ' ')
}

const STATE_BADGES: Record<string, { text: string; cls: string; title?: string }> = {
  sent: { text: 'sent', cls: 'bg-emerald-500/15 text-emerald-500 border-emerald-500/30' },
  held: { text: 'waiting for you', cls: 'bg-amber-500/15 text-amber-600 border-amber-500/30' },
  blocked: {
    text: 'blocked',
    cls: 'bg-red-500/15 text-red-500 border-red-500/30',
    title: 'The reply contained something sensitive. Nothing was sent and its text was not kept.',
  },
  skipped: { text: 'skipped', cls: 'bg-zinc-800 text-zinc-400 border-zinc-700' },
  failed: { text: 'failed', cls: 'bg-red-500/10 text-red-500 border-red-500/20' },
  unknown: {
    text: 'may have sent',
    cls: 'bg-amber-500/15 text-amber-600 border-amber-500/30',
    title: 'Gmail never confirmed this send — it may or may not have gone out. Check Gmail Sent.',
  },
  dismissed: { text: 'dismissed', cls: 'bg-zinc-800 text-zinc-500 border-zinc-700' },
  waiting_agent: { text: 'waiting for the agent', cls: 'bg-zinc-800 text-zinc-400 border-zinc-700' },
}
const WORKING_BADGE: { text: string; cls: string; title?: string } = {
  text: 'working…',
  cls: 'bg-blue-500/15 text-blue-500 border-blue-500/30',
}

// A breadth test's result, coloured by the gate: at most 25 % may match,
// 10–25 % passes with a warning, under 5 eligible emails can't tell.
function rateBadge(eligible: number, matched: number): { text: string; cls: string } {
  if (eligible < 5) {
    return { text: `${matched}/${eligible} — too little mail to judge`, cls: 'bg-zinc-800 text-zinc-400 border-zinc-700' }
  }
  const rate = matched / eligible
  const text = `matched ${Math.round(rate * 100)}% (${matched}/${eligible})`
  if (rate > 0.25) return { text, cls: 'bg-red-500/15 text-red-500 border-red-500/30' }
  if (rate >= 0.1) return { text, cls: 'bg-amber-500/15 text-amber-600 border-amber-500/30' }
  return { text, cls: 'bg-emerald-500/15 text-emerald-500 border-emerald-500/30' }
}

// ── Topic drafts (the editor's form) ────────────────────────────────

interface TopicDraft {
  key: string          // the saved id, or 'new-<n>' until the server gives one
  id: string           // '' for a topic not saved yet
  name: string
  enabled: boolean
  about: string
  notThis: string
  examplesText: string // one per line
  counterText: string  // one per line
  sendersText: string  // one per line, commas work too
  brief: string
  folder: string
  insights: boolean
  playbooks: boolean
  webSearch: boolean
}

function draftFromTopic(t: AutoReplyTopic): TopicDraft {
  return {
    key: t.id,
    id: t.id,
    name: t.name,
    enabled: t.enabled,
    about: t.about,
    notThis: t.not_this,
    examplesText: t.examples.join('\n'),
    counterText: t.counter_examples.join('\n'),
    sendersText: t.senders.join('\n'),
    brief: t.brief,
    folder: t.sources.folder,
    insights: t.sources.insights,
    playbooks: t.sources.playbooks,
    webSearch: t.sources.web_search,
  }
}

function blankDraft(key: string): TopicDraft {
  return {
    key, id: '', name: '', enabled: false, about: '', notThis: '', examplesText: '',
    counterText: '', sendersText: '', brief: '', folder: '',
    insights: false, playbooks: true, webSearch: false,
  }
}

const lines = (text: string) => text.split('\n').map((l) => l.trim()).filter(Boolean)

// What the draft says, as the PUT takes it.
function draftInput(d: TopicDraft): AutoReplyTopicInput {
  return {
    ...(d.id ? { id: d.id } : {}),
    name: d.name,
    enabled: d.enabled,
    about: d.about,
    not_this: d.notThis,
    examples: lines(d.examplesText),
    counter_examples: lines(d.counterText),
    senders: d.sendersText.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean),
    brief: d.brief,
    sources: { folder: d.folder, insights: d.insights, playbooks: d.playbooks, web_search: d.webSearch },
  }
}

// A topic as the server would store it, in a fixed key order — so a draft
// nobody touched compares equal to its saved topic, and re-typing a saved
// value in another spacing or case doesn't count as a change.
function canonicalTopic(t: AutoReplyTopicInput): AutoReplyTopicInput {
  return {
    ...(t.id ? { id: t.id } : {}),
    name: oneLine(t.name),
    enabled: !!t.enabled,
    about: oneLine(t.about),
    not_this: oneLine(t.not_this),
    examples: t.examples.map(oneLine).filter(Boolean),
    counter_examples: t.counter_examples.map(oneLine).filter(Boolean),
    senders: normalizeSenders(t.senders).senders,
    brief: multiLine(t.brief),
    sources: {
      folder: normalizeFolder(t.sources.folder) ?? t.sources.folder.trim(),
      insights: !!t.sources.insights,
      playbooks: !!t.sources.playbooks,
      web_search: !!t.sources.web_search,
    },
  }
}

// The fields a breadth test is tied to (the server's match hash): editing
// one of them needs a fresh test before the topic can be on.
function matchKey(t: AutoReplyTopicInput): string {
  return JSON.stringify([t.name, t.about, t.not_this, t.examples, t.counter_examples, t.senders])
}

// The never-disclose box: one term per line, repeats (case and accents
// folded) dropped — as the server stores them.
function parseTerms(text: string): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of text.split('\n')) {
    const term = oneLine(raw)
    const key = foldText(term)
    if (term && !seen.has(key)) {
      seen.add(key)
      out.push(term)
    }
  }
  return out
}

function termsError(terms: string[]): string {
  const [lo, hi] = AUTO_REPLY_LIMITS.neverDiscloseChars
  const bad = terms.find((t) => [...t].length < lo || [...t].length > hi)
  if (bad !== undefined) return `Each term must be ${lo} to ${hi} characters (got ${[...bad].length}).`
  if (terms.length > AUTO_REPLY_LIMITS.neverDisclose) {
    return `At most ${AUTO_REPLY_LIMITS.neverDisclose} terms (got ${terms.length}).`
  }
  return ''
}

interface SavedKeys {
  enabled: string
  agent: string
  limit: string
  terms: string
  topics: string
}

// The saved settings, field group by group: a group's form fields re-seed
// only when IT changed — a breadth test, a pause or a Refresh keeps what's
// being typed elsewhere.
function savedKeys(v: AutoReplyView): SavedKeys {
  return {
    enabled: String(v.enabled),
    agent: v.agent_ref,
    limit: String(v.daily_limit),
    terms: JSON.stringify(v.never_disclose),
    topics: JSON.stringify(v.topics.map((t) => canonicalTopic(draftInput(draftFromTopic(t))))),
  }
}

// ── The section ─────────────────────────────────────────────────────

export default function GmailAutoReplySection() {
  // The kiosk shows only an off switch, and only once they're on.
  const kioskLocked = useAuthStore(selectKioskLocked)
  return kioskLocked ? <KioskAutoReplyLine /> : <AutoReplyEditor />
}

function KioskAutoReplyLine() {
  const view = useGmailAutoReplyStore((s) => s.config)
  const error = useGmailAutoReplyStore((s) => s.error)
  const fetchConfig = useGmailAutoReplyStore((s) => s.fetchConfig)
  const saveConfig = useGmailAutoReplyStore((s) => s.saveConfig)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void fetchConfig()
  }, [fetchConfig])

  if (!view?.enabled) return null
  const turnOff = async () => {
    setBusy(true)
    await saveConfig({ enabled: false })
    setBusy(false)
  }
  return (
    <div className="flex items-center gap-2 flex-wrap rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2 text-xs">
      <Bot className="h-3.5 w-3.5 text-zinc-400" />
      <span className="text-zinc-300">
        Topic auto-replies: <span className="text-emerald-500">on</span> —
      </span>
      <button
        type="button"
        onClick={turnOff}
        disabled={busy}
        className="inline-flex items-center gap-1 text-xs text-violet-500 hover:underline disabled:opacity-50"
      >
        {busy && <Loader2 className="h-3 w-3 animate-spin" />}
        Turn off
      </button>
      {error && <span className="text-red-500 break-words">{error}</span>}
    </div>
  )
}

function AutoReplyEditor() {
  const view = useGmailAutoReplyStore((s) => s.config)
  const error = useGmailAutoReplyStore((s) => s.error)
  const agents = useGmailAutoReplyStore((s) => s.agents)
  const fetchConfig = useGmailAutoReplyStore((s) => s.fetchConfig)
  const saveConfig = useGmailAutoReplyStore((s) => s.saveConfig)
  const pause = useGmailAutoReplyStore((s) => s.pause)
  const fetchAgents = useGmailAutoReplyStore((s) => s.fetchAgents)
  const fetchActivity = useGmailAutoReplyStore((s) => s.fetchActivity)

  const [enabled, setEnabled] = useState(false)
  const [agentRef, setAgentRef] = useState('')
  const [limitText, setLimitText] = useState('')
  const [termsText, setTermsText] = useState('')
  const [drafts, setDrafts] = useState<TopicDraft[]>([])
  const [newTopics, setNewTopics] = useState(0)
  const [saving, setSaving] = useState(false)
  const [pausing, setPausing] = useState(false)
  const [reloading, setReloading] = useState(true)
  const [notice, setNotice] = useState('')
  const [showAll, setShowAll] = useState(false)

  const reload = async () => {
    setReloading(true)
    await Promise.all([fetchConfig(), fetchAgents(), fetchActivity(showAll)])
    setReloading(false)
  }

  useEffect(() => {
    reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // The signature is set in Email sending, just above: once it's saved there,
  // re-read what turning on still needs.
  const signature = useGoogleAuthStore((s) => s.gmailSend?.signature)
  const lastSignature = useRef(signature)
  useEffect(() => {
    if (lastSignature.current !== undefined && signature !== undefined && signature !== lastSignature.current) {
      void fetchConfig()
    }
    lastSignature.current = signature
  }, [signature, fetchConfig])

  const syncDraft = (v: AutoReplyView, only?: (keyof SavedKeys)[]) => {
    const all = !only
    if (all || only.includes('enabled')) setEnabled(v.enabled)
    if (all || only.includes('agent')) setAgentRef(v.agent_ref)
    if (all || only.includes('limit')) setLimitText(String(v.daily_limit))
    if (all || only.includes('terms')) setTermsText(v.never_disclose.join('\n'))
    if (all || only.includes('topics')) {
      // A topic keeps its card (open, with its breadth test) across a save:
      // a new one is matched by position, since it only now has an id.
      setDrafts((prev) => v.topics.map((t, i) => {
        const old = prev.find((d) => d.id === t.id) ?? (prev[i] && !prev[i].id ? prev[i] : undefined)
        return old ? { ...draftFromTopic(t), key: old.key } : draftFromTopic(t)
      }))
    }
  }

  // Re-seed a field group only when its saved value changes. Done while
  // rendering, not in an effect, so the form never paints a frame empty.
  const keys = view ? savedKeys(view) : null
  const [seeded, setSeeded] = useState<SavedKeys | null>(null)
  if (view && keys) {
    const changed = (Object.keys(keys) as (keyof SavedKeys)[]).filter((k) => !seeded || seeded[k] !== keys[k])
    if (changed.length > 0) {
      setSeeded(keys)
      syncDraft(view, changed)
    }
  }

  const limit = Number(limitText)
  const [minLimit, maxLimit] = AUTO_REPLY_LIMITS.dailyLimit
  const limitValid = limitText.trim() !== '' && Number.isInteger(limit) && limit >= minLimit && limit <= maxLimit
  const terms = parseTerms(termsText)
  const termsProblem = termsError(terms)
  const topicInputs = drafts.map((d) => canonicalTopic(draftInput(d)))
  const topicErrors = drafts.map((d) => validateTopic(draftInput(d)))
  const topicsValid = topicErrors.every((e) => Object.keys(e).length === 0)
  const enabledCount = drafts.filter((d) => d.enabled).length
  const tooManyOn = enabledCount > AUTO_REPLY_LIMITS.maxEnabledTopics

  const enabledDirty = !!view && enabled !== view.enabled
  const agentDirty = !!view && agentRef !== view.agent_ref
  const limitDirty = !!view && (!limitValid || limit !== view.daily_limit)
  const termsDirty = !!view && terms.join('\n') !== view.never_disclose.join('\n')
  const topicsDirty = !!view && JSON.stringify(topicInputs) !== keys?.topics
  const dirty = enabledDirty || agentDirty || limitDirty || termsDirty || topicsDirty
  const canSave = !!view && !saving && dirty && limitValid && !termsProblem
    && (!topicsDirty || (topicsValid && !tooManyOn))

  const chosen = agents.find((a) => a.ref === agentRef)
  const savedAgent = view?.status.agent
  const agentName = chosen?.name || (savedAgent && savedAgent.ref === agentRef ? savedAgent.name : '') || ''
  const agentShared = chosen ? chosen.shared : !!(savedAgent && savedAgent.ref === agentRef && savedAgent.shared)
  const who = agentName || 'the agent you choose'

  const handleSave = async () => {
    if (!view || !canSave) return
    const patch: AutoReplyPatch = {}
    if (agentDirty) patch.agent_ref = agentRef
    if (limitDirty) patch.daily_limit = limit
    if (termsDirty) patch.never_disclose = terms
    if (topicsDirty) patch.topics = topicInputs
    if (enabledDirty) patch.enabled = enabled
    if (patch.enabled === true && !confirm(
      'Turn on topic auto-replies?\n\n' +
      `${agentName || 'Your agent'} will answer emails that are only about one of your topics ` +
      'right away, without asking you, to anyone whose address is verified. Anything in a ' +
      "topic's brief or knowledge folder can end up in a reply to a stranger.",
    )) return
    setSaving(true)
    const { ok, notice: changed } = await saveConfig(patch)
    setSaving(false)
    if (!ok) {
      // The box shows what's saved, never what was asked for: a refused
      // turn-on wrote nothing (the error says what's missing); a turn-off goes
      // first on its own (saveConfig), so it is off unless even that failed —
      // then it reads ON again beside the error. The other changes stay to
      // save again.
      if (patch.enabled !== undefined) setEnabled(!!useGmailAutoReplyStore.getState().config?.enabled)
      return
    }
    setNotice(changed
      ? 'A topic whose wording changed was switched off — re-run its breadth test, then turn it on again.'
      : '')
    // The backend may write a value differently (a bare domain as '@b.c'),
    // which leaves the saved keys as they were — show what it stored regardless.
    const saved = useGmailAutoReplyStore.getState().config
    if (saved) syncDraft(saved)
  }

  const handlePause = async () => {
    setPausing(true)
    await pause()
    setPausing(false)
  }

  const updateDraft = (key: string, patch: Partial<TopicDraft>) => {
    setDrafts((prev) => prev.map((d) => (d.key === key ? { ...d, ...patch } : d)))
  }

  const addTopic = () => {
    setDrafts((prev) => [...prev, blankDraft(`new-${newTopics + 1}`)])
    setNewTopics((n) => n + 1)
  }

  const optInLocked = autoReplyOptInLocked(view)
  const status = view?.status
  const deckOff = !!status?.deck_disabled || !!status?.send_deck_disabled

  const saveBar = view && (
    <div className="flex items-center gap-2 flex-wrap">
      <button
        onClick={handleSave}
        disabled={!canSave}
        className="inline-flex items-center gap-2 rounded-md bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-sm text-white disabled:opacity-50 shadow-sm"
      >
        {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
        Save
      </button>
      {dirty && !saving && <span className="text-xs text-zinc-500">Unsaved changes</span>}
      {topicsDirty && !topicsValid && (
        <span className="text-xs text-red-500">Fix the topic errors to save.</span>
      )}
      {tooManyOn && (
        <span className="text-xs text-red-500">
          At most {AUTO_REPLY_LIMITS.maxEnabledTopics} topics can be on at once.
        </span>
      )}
    </div>
  )

  return (
    <div className="space-y-3 rounded-md border border-zinc-800 bg-zinc-950/50 p-3">
      <div className="flex items-center gap-2 flex-wrap">
        <Bot className="h-3.5 w-3.5 text-zinc-400" />
        <span className="text-xs font-medium text-zinc-300">Topic auto-replies</span>
        {view && status && (
          status.deck_disabled ? (
            <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Off on this deck</span>
          ) : view.enabled && status.state === 'on' ? (
            <span className={`${pill} bg-emerald-500/15 text-emerald-500 border-emerald-500/30`}>
              <Check className="h-2.5 w-2.5" /> On
            </span>
          ) : view.enabled ? (
            <span className={`${pill} bg-amber-500/15 text-amber-600 border-amber-500/30`}>
              <AlertCircle className="h-2.5 w-2.5" /> Needs attention
            </span>
          ) : (
            <span className={`${pill} bg-zinc-800 text-zinc-400 border-zinc-700`}>Off</span>
          )
        )}
        <div className="flex-1" />
        <button
          type="button"
          onClick={reload}
          disabled={reloading}
          className="inline-flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300 disabled:opacity-50"
          title="Reload the auto-reply settings, status and activity"
        >
          <RefreshCw className={'h-3 w-3 ' + (reloading ? 'animate-spin' : '')} />
          Refresh
        </button>
      </div>

      <div className="text-xs text-zinc-500 leading-snug">
        Off by default. When an email that is only about one of your topics arrives,{' '}
        <span className="text-zinc-300">{who}</span> answers right away — without asking you — to
        anyone whose address is verified. It uses the topic's brief and knowledge folder, and (if
        you allow) what it has learned, its playbooks and web search.{' '}
        <span className="text-amber-600">
          Everything in a brief or folder can end up in a reply to a stranger: treat it as public.
        </span>{' '}
        Replies are checked for sensitive data first; anything doubtful waits here for you. Works
        even when “Let my agents send email” is off. Gmail only.
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-500">
          <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
          <span className="break-words">{error}</span>
        </div>
      )}

      {notice && (
        <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-600">
          <span className="flex-1">{notice}</span>
          <button type="button" onClick={() => setNotice('')} className="shrink-0 hover:opacity-70" title="Dismiss">
            <X className="h-3 w-3" />
          </button>
        </div>
      )}

      {!view || !status ? (
        !error && (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…
          </div>
        )
      ) : (
        <>
          {/* The switch — saved with Save, like the rest; Pause all is at once. */}
          <div className="flex items-start gap-3 flex-wrap">
            <label
              className={
                'flex flex-1 min-w-[14rem] items-start gap-2 ' +
                (optInLocked.length > 0 && !enabled ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer')
              }
            >
              <input
                type="checkbox"
                checked={enabled}
                disabled={saving || (optInLocked.length > 0 && !enabled)}
                onChange={(e) => setEnabled(e.target.checked)}
                className="mt-0.5 accent-violet-600"
              />
              <div className="flex-1 min-w-0">
                <div className="text-sm text-zinc-200">Answer emails about my topics automatically</div>
                <div className="text-xs text-zinc-500 mt-0.5 leading-snug">
                  Turning on starts from now: mail already in your inbox is never answered.
                  {enabledDirty && <span className="text-zinc-300"> Save to apply.</span>}
                </div>
              </div>
            </label>
            {view.enabled && (
              <button
                type="button"
                onClick={handlePause}
                disabled={pausing}
                className="inline-flex items-center gap-1.5 rounded-md border border-amber-500/30 hover:bg-amber-500/10 px-3 py-1.5 text-xs text-amber-600 disabled:opacity-50"
                title="Turn topic auto-replies off now. Turning them on again starts from that moment."
              >
                {pausing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Pause className="h-3 w-3" />}
                Pause all
              </button>
            )}
          </div>

          {!view.enabled && DISABLED_REASONS[view.disabled_reason] && (
            <div className="rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-xs text-zinc-400">
              {DISABLED_REASONS[view.disabled_reason]}
            </div>
          )}

          {optInLocked.length > 0 && (
            <div className="rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-xs text-zinc-400">
              <div className="text-zinc-300 mb-1">To turn them on:</div>
              <ul className="list-disc pl-4 space-y-0.5">
                {optInLocked.map((item) => <li key={item}>{item}</li>)}
              </ul>
            </div>
          )}

          {view.enabled && deckOff && (
            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-600">
              {status.deck_disabled
                ? 'The administrator has turned topic auto-replies off on this Flight Deck (FD_GMAIL_AUTOREPLY=off).'
                : 'The administrator has turned email sending off on this Flight Deck (FD_GMAIL_SEND=off).'}
              {' '}Nothing is sent. You can still turn yours off here.
            </div>
          )}

          <StatusLine view={view} agentName={agentName || savedAgent?.name || ''} />

          {/* Agent, limit, never-disclose terms */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="sm:col-span-2">
              <label className={label}>Agent that writes the replies</label>
              <select
                value={agentRef}
                onChange={(e) => setAgentRef(e.target.value)}
                disabled={saving}
                className={field}
              >
                <option value="">Choose an agent…</option>
                {agents.map((a) => (
                  <option key={a.ref} value={a.ref} disabled={!a.eligible && a.ref !== agentRef}>
                    {a.name || a.ref}
                    {a.eligible && !a.running ? ' — stopped' : ''}
                    {a.shared ? ' — shared — insights/playbooks unavailable' : ''}
                    {!a.eligible ? ` — ${AGENT_REASONS[a.reason] || a.reason}` : ''}
                  </option>
                ))}
                {agentRef && !chosen && (
                  <option value={agentRef}>
                    {(savedAgent?.ref === agentRef && savedAgent.name) || agentRef} — not in your list
                  </option>
                )}
              </select>
              <div className={hint}>
                {agents.length === 0 && !reloading && 'You have no agents yet — start one first. '}
                One of your own agents. Its model writes each reply from what Flight Deck looks up
                for it — it never gets to send, and a stopped agent's replies wait for it (up to an
                hour).
              </div>
            </div>
            <div>
              <label className={label}>Daily limit</label>
              <input
                type="number"
                min={minLimit}
                max={maxLimit}
                step={1}
                value={limitText}
                onChange={(e) => setLimitText(e.target.value)}
                disabled={saving}
                className={field}
              />
              {!limitValid && (
                <div className={fieldError}>A whole number from {minLimit} to {maxLimit}.</div>
              )}
              <div className={hint}>
                Auto-replies in the last 24h:{' '}
                <span className={status.sent_last_24h >= view.daily_limit ? 'text-amber-600' : 'text-zinc-300'}>
                  {status.sent_last_24h} / {view.daily_limit}
                </span>
              </div>
            </div>
          </div>

          <div>
            <label className={label}>
              Never mention <span className="text-zinc-600">(optional)</span>
            </label>
            <textarea
              value={termsText}
              onChange={(e) => setTermsText(e.target.value)}
              rows={3}
              disabled={saving}
              placeholder={'Project Falcon\nour bank name'}
              className={field}
            />
            {termsProblem && <div className={fieldError}>{termsProblem}</div>}
            <div className={hint}>
              One term per line ({AUTO_REPLY_LIMITS.neverDiscloseChars[0]}–{AUTO_REPLY_LIMITS.neverDiscloseChars[1]}{' '}
              characters, up to {AUTO_REPLY_LIMITS.neverDisclose}). A reply that contains one is
              blocked. They are only checked here — never shown to a model.
            </div>
          </div>

          {saveBar}

          {/* Topics */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-zinc-300">Topics</span>
              <span className="text-[11px] text-zinc-500">
                {drafts.length}/{AUTO_REPLY_LIMITS.maxTopics} · {enabledCount} on (at most {AUTO_REPLY_LIMITS.maxEnabledTopics})
              </span>
              <div className="flex-1" />
              <button
                type="button"
                onClick={addTopic}
                disabled={saving || drafts.length >= AUTO_REPLY_LIMITS.maxTopics}
                className={smallBtn}
              >
                <Plus className="h-3 w-3" /> Add topic
              </button>
            </div>
            {drafts.length === 0 && (
              <div className="text-xs text-zinc-500">
                No topics yet. A topic is one narrow kind of email — “when and where is the meetup”,
                not “customer questions”.
              </div>
            )}
            {drafts.map((d, i) => (
              <TopicCard
                key={d.key}
                index={i}
                draft={d}
                saved={d.id ? view.topics.find((t) => t.id === d.id) : undefined}
                errors={topicErrors[i]}
                agentName={agentName || 'the agent'}
                agentShared={agentShared}
                canTurnOnMore={enabledCount < AUTO_REPLY_LIMITS.maxEnabledTopics}
                disabled={saving}
                onChange={(patch) => updateDraft(d.key, patch)}
                onRemove={() => setDrafts((prev) => prev.filter((x) => x.key !== d.key))}
              />
            ))}
            {drafts.length > 0 && saveBar}
          </div>

          <TryItBox topics={view.topics} />

          <ActivityList
            accountEmail={view.account_email || status.account_email}
            showAll={showAll}
            onShowAll={(all) => {
              setShowAll(all)
              void fetchActivity(all)
            }}
          />
        </>
      )}
    </div>
  )
}

// ── Status ──────────────────────────────────────────────────────────

function StatusLine({ view, agentName }: { view: AutoReplyView; agentName: string }) {
  const s = view.status
  const parties = checkerParties(s.classifier, s.judge)
  const attention: string[] = []
  if (view.enabled) {
    if (!s.bound_ok) {
      attention.push("The connected Google account isn't the one auto-replies were turned on for — turn them off and on again.")
    }
    const err = lastErrorText(s.last_error)
    if (err) attention.push(err)
    // A topic past its breadth test's 30 days keeps running: the gate only
    // stands in the way of turning on.
    for (const code of s.missing || []) {
      if (code !== 'topic_gate') attention.push(requirementText(code, s.missing_scopes || []))
    }
  }
  const agent = s.agent
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-[11px] text-zinc-500 space-y-1 leading-snug">
      <div>
        Emails are sorted by <span className="text-zinc-300">{checkerLabel(s.classifier)}</span>; replies
        are checked by <span className="text-zinc-300">{checkerLabel(s.judge)}</span> + local scan.
        {parties.length > 0 && (
          <> Each incoming email and drafted reply goes to {parties.join(' and ')} for these checks
            {agentName ? `, and to ${agentName}'s own model to write the reply` : ''}.</>
        )}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5">
        {agent.ref && (
          <span>
            Agent: <span className="text-zinc-300">{agent.name || agent.ref}</span>{' '}
            {agent.eligible
              ? agent.running
                ? <span className="text-emerald-500">running</span>
                : <span className="text-amber-600">stopped — replies wait for it</span>
              : <span className="text-red-500">{AGENT_REASONS[agent.reason] || agent.reason}</span>}
            {agent.shared && ' · shared: insights and playbooks unavailable'}
          </span>
        )}
        {view.enabled && (
          <span>Last checked Gmail: <span className="text-zinc-300">{fmtTime(s.last_poll_at) || 'not yet'}</span></span>
        )}
        <span>Waiting for you: <span className={s.held_open > 0 ? 'text-amber-600' : 'text-zinc-300'}>{s.held_open}</span></span>
        {view.enabled && view.account_email && (
          <span>Answering as <span className="text-zinc-300">{view.account_email}</span></span>
        )}
      </div>
      {attention.length > 0 && (
        <ul className="list-disc pl-4 text-amber-600">
          {attention.map((a) => <li key={a}>{a}</li>)}
        </ul>
      )}
    </div>
  )
}

// ── A topic card ────────────────────────────────────────────────────

interface TopicCardProps {
  index: number
  draft: TopicDraft
  saved: AutoReplyTopic | undefined
  errors: AutoReplyTopicErrors
  agentName: string
  agentShared: boolean
  canTurnOnMore: boolean
  disabled: boolean
  onChange: (patch: Partial<TopicDraft>) => void
  onRemove: () => void
}

function TopicCard({
  index, draft, saved, errors, agentName, agentShared, canTurnOnMore, disabled, onChange, onRemove,
}: TopicCardProps) {
  const dryRun = useGmailAutoReplyStore((s) => s.dryRun)
  const [open, setOpen] = useState(!saved)
  const [testing, setTesting] = useState(false)
  const [result, setResult] = useState<AutoReplyDryRunResult | null>(null)
  const [testError, setTestError] = useState('')

  const input = canonicalTopic(draftInput(draft))
  const savedInput = saved ? canonicalTopic(draftInput(draftFromTopic(saved))) : null
  const dirty = !savedInput || JSON.stringify(input) !== JSON.stringify(savedInput)
  const matchDirty = !savedInput || matchKey(input) !== matchKey(savedInput)
  const gate = saved?.gate
  const canTurnOn = !!saved && !matchDirty && !!gate?.ok && canTurnOnMore
  const switchTitle = draft.enabled ? 'Turn this topic off'
    : !saved ? 'Save the topic, then test its breadth on your recent mail'
    : matchDirty ? 'Save your changes, then re-run the breadth test'
    : !gate?.ok ? `Can't be turned on yet: ${gate?.reason || 'run the breadth test'}`
    : !canTurnOnMore ? `At most ${AUTO_REPLY_LIMITS.maxEnabledTopics} topics can be on at once`
    : 'Turn this topic on'
  // The last test's rate, while it still counts: it passed, or it says the
  // topic is too broad (an edited or month-old test needs a new one).
  const dry = saved?.dry_run
  const tooBroad = !!dry && dry.eligible >= 5 && dry.matched / dry.eligible > 0.25
  const badge = dry?.at && !matchDirty && (gate?.ok || tooBroad) ? rateBadge(dry.eligible, dry.matched) : null
  const folderUnchanged = !!saved && normalizeFolder(draft.folder) === saved.sources.folder
  const errorCount = Object.keys(errors).length

  const runTest = async () => {
    if (!saved) return
    setTesting(true)
    setTestError('')
    const out = await dryRun(saved.id)
    setTesting(false)
    if (out.ok) setResult(out.data)
    else setTestError(out.error)
  }

  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/40">
      <div className="flex items-center gap-2 px-3 py-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex flex-1 min-w-0 items-center gap-1.5 text-left"
          aria-expanded={open}
        >
          {open ? <ChevronDown className="h-3.5 w-3.5 text-zinc-500 shrink-0" /> : <ChevronRight className="h-3.5 w-3.5 text-zinc-500 shrink-0" />}
          <span className="text-sm text-zinc-200 truncate">{oneLine(draft.name) || `New topic ${index + 1}`}</span>
          {dirty && <span className="text-[10px] text-zinc-500 shrink-0">unsaved</span>}
          {errorCount > 0 && !open && <AlertCircle className="h-3 w-3 text-red-500 shrink-0" />}
        </button>
        {badge ? (
          <span className={`${pill} ${badge.cls} shrink-0`} title="Its last breadth test on your recent mail">{badge.text}</span>
        ) : (
          <span className={`${pill} bg-zinc-800 text-zinc-500 border-zinc-700 shrink-0`}>
            {saved && !matchDirty && !dry?.at ? 'not tested' : 'needs a test'}
          </span>
        )}
        <label
          className={'flex items-center gap-1.5 text-xs shrink-0 ' + (!draft.enabled && !canTurnOn ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer')}
          title={switchTitle}
        >
          <input
            type="checkbox"
            checked={draft.enabled}
            disabled={disabled || (!draft.enabled && !canTurnOn)}
            onChange={(e) => onChange({ enabled: e.target.checked })}
            className="accent-violet-600"
          />
          <span className="text-zinc-300">On</span>
        </label>
      </div>

      {open && (
        <div className="space-y-3 border-t border-zinc-800 px-3 py-3">
          <div>
            <label className={label}>Name</label>
            <input
              type="text"
              value={draft.name}
              onChange={(e) => onChange({ name: e.target.value })}
              disabled={disabled}
              maxLength={120}
              placeholder="Meetup logistics"
              className={field}
            />
            {errors.name && <div className={fieldError}>{errors.name}</div>}
          </div>

          <div>
            <label className={label}>Emails that…</label>
            <textarea
              value={draft.about}
              onChange={(e) => onChange({ about: e.target.value })}
              disabled={disabled}
              rows={2}
              placeholder="ask when or where the Zagreb AI Meetup on 14 Nov is, or how to register"
              className={field}
            />
            {errors.about && <div className={fieldError}>{errors.about}</div>}
            <div className={hint}>
              One concrete sentence naming the subject <em>and</em> the kind of request. Good:{' '}
              <span className="text-zinc-300">“Emails asking when/where the Zagreb AI Meetup on 14 Nov is, or how to register”</span>.
              Too broad: <span className="text-zinc-300">“Customer questions”</span>.
            </div>
          </div>

          <div>
            <label className={label}>Not this… <span className="text-zinc-600">(optional)</span></label>
            <textarea
              value={draft.notThis}
              onChange={(e) => onChange({ notThis: e.target.value })}
              disabled={disabled}
              rows={2}
              placeholder="Sponsorship, speaking slots, invoices."
              className={field}
            />
            {errors.not_this && <div className={fieldError}>{errors.not_this}</div>}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className={label}>Examples that should match</label>
              <textarea
                value={draft.examplesText}
                onChange={(e) => onChange({ examplesText: e.target.value })}
                disabled={disabled}
                rows={3}
                placeholder={'Where is the meetup?\nHow do I register for the 14 Nov meetup?'}
                className={field}
              />
              {errors.examples && <div className={fieldError}>{errors.examples}</div>}
              <div className={hint}>One per line, 1–{AUTO_REPLY_LIMITS.maxExamples}.</div>
            </div>
            <div>
              <label className={label}>Near-misses that should not <span className="text-zinc-600">(optional)</span></label>
              <textarea
                value={draft.counterText}
                onChange={(e) => onChange({ counterText: e.target.value })}
                disabled={disabled}
                rows={3}
                placeholder="Can I speak at the meetup?"
                className={field}
              />
              {errors.counter_examples && <div className={fieldError}>{errors.counter_examples}</div>}
              <div className={hint}>One per line, up to {AUTO_REPLY_LIMITS.maxCounterExamples}.</div>
            </div>
          </div>

          <div>
            <label className={label}>Only from these senders <span className="text-zinc-600">(optional)</span></label>
            <textarea
              value={draft.sendersText}
              onChange={(e) => onChange({ sendersText: e.target.value })}
              disabled={disabled}
              rows={2}
              placeholder={'alice@example.com\n@partner.com'}
              className={`${field} font-mono`}
            />
            {errors.senders && <div className={fieldError}>{errors.senders}</div>}
            <div className={hint}>Empty = any sender whose address is verified.</div>
          </div>

          <div>
            <label className={label}>What the agent may say</label>
            <textarea
              value={draft.brief}
              onChange={(e) => onChange({ brief: e.target.value })}
              disabled={disabled}
              rows={5}
              placeholder="The meetup is on 14 Nov at 18:00 at the Hub, Ilica 1, Zagreb. Registration is free at https://example.com/meetup."
              className={field}
            />
            {errors.brief && <div className={fieldError}>{errors.brief}</div>}
            <div className={hint}>
              <span className="text-amber-600">Treated as public.</span> The only place for facts
              cleared for sharing — prices, links, the company IBAN. Anything here can end up in a
              reply to a stranger.
            </div>
          </div>

          <div>
            <label className={label}>Knowledge folder <span className="text-zinc-600">(optional — also public)</span></label>
            <FolderPicker
              value={draft.folder}
              onChange={(folder) => onChange({ folder })}
              disabled={disabled}
            />
            {errors.folder && <div className={fieldError}>{errors.folder}</div>}
            {folderUnchanged && saved?.folder_error && (
              <div className={fieldError}>This folder isn't in your files any more — choose another.</div>
            )}
            {folderUnchanged && !!saved?.hidden_passages.length && (
              <div className={hint}>
                <span className="text-amber-600">
                  {saved.hidden_passages.reduce((n, p) => n + p.hidden, 0)} passage(s) hidden
                </span>{' '}
                — they look sensitive, so {agentName} never sees them:{' '}
                {saved.hidden_passages.map((p) => `${p.file} (${p.hidden})`).join(', ')}
              </div>
            )}
          </div>

          <div>
            <div className={label}>Also let it look things up in</div>
            <div className="space-y-1">
              <label className="flex items-start gap-2 text-xs text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={draft.playbooks}
                  onChange={(e) => onChange({ playbooks: e.target.checked })}
                  disabled={disabled}
                  className="mt-0.5 accent-violet-600"
                />
                <span>
                  {agentName}'s playbooks
                  {agentShared && <span className="text-zinc-500"> — unavailable: the agent is shared</span>}
                </span>
              </label>
              <label className="flex items-start gap-2 text-xs text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={draft.insights}
                  onChange={(e) => onChange({ insights: e.target.checked })}
                  disabled={disabled}
                  className="mt-0.5 accent-violet-600"
                />
                <span>
                  What {agentName} has learned — specifics are checked and may be held
                  {agentShared && <span className="text-zinc-500"> — unavailable: the agent is shared</span>}
                </span>
              </label>
              <label className="flex items-start gap-2 text-xs text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={draft.webSearch}
                  onChange={(e) => onChange({ webSearch: e.target.checked })}
                  disabled={disabled}
                  className="mt-0.5 accent-violet-600"
                />
                <span>Web search — the search words go to the search provider</span>
              </label>
            </div>
          </div>

          {/* The breadth test: the topic alone against your recent inbox. */}
          <div className="space-y-2 rounded-md border border-zinc-800 bg-zinc-950/50 p-2.5">
            <div className="flex items-center gap-2 flex-wrap">
              <button
                type="button"
                onClick={runTest}
                disabled={testing || disabled || !saved || matchDirty}
                className={smallBtn}
                title={!saved || matchDirty ? 'Save the topic first' : 'Sort your last 30 days of inbox mail against this topic alone'}
              >
                {testing ? <Loader2 className="h-3 w-3 animate-spin" /> : <FlaskConical className="h-3 w-3" />}
                Test breadth on my recent mail
              </button>
              {(!saved || matchDirty) && <span className="text-[11px] text-zinc-500">Save the topic first.</span>}
              {saved && !matchDirty && gate && (
                gate.ok ? (
                  <span className="text-[11px] text-emerald-500">
                    Can be turned on{gate.warning ? '' : gate.note ? ` — ${gate.note}` : ''}.
                  </span>
                ) : (
                  <span className="text-[11px] text-amber-600">{gate.reason}</span>
                )
              )}
            </div>
            {saved && !matchDirty && gate?.warning && (
              <div className="text-[11px] text-amber-600">{gate.warning}</div>
            )}
            {saved?.enabled && matchDirty && (
              <div className="text-[11px] text-amber-600">
                Saving this wording switches the topic off until its breadth test is re-run.
              </div>
            )}
            {testError && <div className="text-[11px] text-red-500 break-words">{testError}</div>}
            <div className="text-[11px] text-zinc-500 leading-snug">
              Reads up to 30 emails from your inbox's last 30 days and asks whether each is about
              this topic. It may be on only if at most 25% match. Nothing is answered.
            </div>
            {result && result.topic_id === saved?.id && !matchDirty && (
              <DryRunTable result={result} />
            )}
          </div>

          <div className="flex justify-end">
            <button
              type="button"
              onClick={onRemove}
              disabled={disabled}
              className="inline-flex items-center gap-1.5 rounded-md border border-red-500/30 hover:bg-red-500/10 px-2.5 py-1 text-xs text-red-500 disabled:opacity-50"
            >
              <Trash2 className="h-3 w-3" /> Remove topic
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function DryRunTable({ result }: { result: AutoReplyDryRunResult }) {
  const badge = rateBadge(result.eligible, result.matched)
  return (
    <div className="space-y-1.5">
      <span className={`${pill} ${badge.cls}`}>{badge.text}</span>
      {result.rows.length > 0 && (
        <div className="max-h-64 overflow-y-auto rounded-md border border-zinc-800 divide-y divide-zinc-800">
          {result.rows.map((r, i) => (
            <div key={i} className="px-2.5 py-1.5 text-[11px]">
              <div className="flex items-center gap-2 min-w-0">
                <span
                  className={
                    'shrink-0 ' +
                    (r.verdict === 'match' ? 'text-emerald-500' : r.verdict.startsWith('not eligible') ? 'text-zinc-600' : 'text-zinc-400')
                  }
                >
                  {r.verdict === 'match' ? 'matches' : r.verdict.startsWith('not eligible') ? 'not eligible' : reasonLabel(r.verdict)}
                </span>
                {r.fit && <span className="text-zinc-600 shrink-0">fit {r.fit}</span>}
                {typeof r.confidence === 'number' && (
                  <span className="text-zinc-600 shrink-0">{Math.round(r.confidence * 100)}%</span>
                )}
                <span className="ml-auto text-zinc-500 truncate">{r.from}</span>
              </div>
              <div className="text-zinc-300 truncate">{r.subject || '(no subject)'}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// A folder in the user's files (/fd/vfs): a project, then down its folders.
function FolderPicker({ value, onChange, disabled }: {
  value: string
  onChange: (folder: string) => void
  disabled: boolean
}) {
  const [projects, setProjects] = useState<string[] | null>(null)
  // The folders inside the one listed last; shown only while it's the
  // current one, so a stale listing never offers the wrong path.
  const [listing, setListing] = useState<{ at: string; dirs: string[] }>({ at: '', dirs: [] })
  const folder = normalizeFolder(value) || ''
  const project = folder ? folder.slice(4).split('/')[0] : ''
  const path = folder ? folder.slice(4 + project.length + 1).replace(/\/+$/, '') : ''
  const at = `${project}/${path}`
  const subfolders = project && listing.at === at ? listing.dirs : []

  useEffect(() => {
    let live = true
    void listFolderProjects().then((p) => { if (live) setProjects(p) })
    return () => { live = false }
  }, [])

  useEffect(() => {
    if (!project || disabled) return
    let live = true
    void listSubfolders(project, path).then((dirs) => {
      if (live) setListing({ at: `${project}/${path}`, dirs })
    })
    return () => { live = false }
  }, [project, path, disabled])

  const into = (name: string) => onChange(`vfs:${project}/${path ? `${path}/` : ''}${name}`)
  const up = () => onChange(`vfs:${project}/${path.split('/').slice(0, -1).join('/')}`)

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2 flex-wrap">
        <select
          value={project}
          onChange={(e) => onChange(e.target.value ? `vfs:${e.target.value}/` : '')}
          disabled={disabled}
          className={`${field} sm:w-auto`}
        >
          <option value="">No folder</option>
          {(projects || []).map((p) => <option key={p} value={p}>{p}</option>)}
          {project && projects && !projects.includes(project) && <option value={project}>{project}</option>}
        </select>
        {project && (
          <code className="text-[11px] text-zinc-300 bg-zinc-900 border border-zinc-800 rounded px-1.5 py-0.5 break-all">
            {folder}
          </code>
        )}
        {project && path && (
          <button type="button" onClick={up} disabled={disabled} className={smallBtn} title="Up one folder">
            <CornerUpLeft className="h-3 w-3" /> Up
          </button>
        )}
      </div>
      {project && subfolders.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {subfolders.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => into(name)}
              disabled={disabled}
              className="inline-flex items-center gap-1 rounded border border-zinc-800 bg-zinc-900 hover:bg-zinc-800 px-1.5 py-0.5 text-[11px] text-zinc-400"
            >
              <Folder className="h-3 w-3" /> {name}
            </button>
          ))}
        </div>
      )}
      {projects !== null && projects.length === 0 && (
        <div className={hint}>No folders in your files yet — add one under Files.</div>
      )}
    </div>
  )
}

// ── Try it ──────────────────────────────────────────────────────────

const OUTCOMES: Record<AutoReplyTestResult['outcome'], { text: string; cls: string }> = {
  would_send: { text: 'Would send', cls: 'bg-emerald-500/15 text-emerald-500 border-emerald-500/30' },
  held: { text: 'Would wait for you', cls: 'bg-amber-500/15 text-amber-600 border-amber-500/30' },
  blocked: { text: 'Blocked — nothing would be sent', cls: 'bg-red-500/15 text-red-500 border-red-500/30' },
  skipped: { text: 'Would not answer', cls: 'bg-zinc-800 text-zinc-400 border-zinc-700' },
  waiting: { text: 'Would wait for the agent to run', cls: 'bg-zinc-800 text-zinc-400 border-zinc-700' },
  failed: { text: 'Failed', cls: 'bg-red-500/10 text-red-500 border-red-500/20' },
}

function TryItBox({ topics }: { topics: AutoReplyTopic[] }) {
  const tryIt = useGmailAutoReplyStore((s) => s.tryIt)
  const [open, setOpen] = useState(false)
  const [from, setFrom] = useState('')
  const [subject, setSubject] = useState('')
  const [body, setBody] = useState('')
  const [topicId, setTopicId] = useState('')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<AutoReplyTestResult | null>(null)
  const [resultFrom, setResultFrom] = useState('')
  const [runError, setRunError] = useState('')

  // Who the made-up email is from: the typed address, else the made-up guest.
  const sender = tryItSender(from)
  const picked = topics.find((t) => t.id === topicId)
  // Topics narrowed to their own senders, which this one isn't: a real email
  // from it would never be answered by them.
  const notTheirs = (picked ? [picked] : topics.filter((t) => t.enabled))
    .filter((t) => !senderAllowed(t.senders, sender))

  const run = async () => {
    setRunning(true)
    setRunError('')
    setResult(null)
    const out = await tryIt({
      subject,
      body,
      from: sender,
      ...(topicId ? { topic_id: topicId } : {}),
    })
    setRunning(false)
    if (out.ok) {
      setResult(out.data)
      setResultFrom(sender)
    } else setRunError(out.error)
  }

  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-1.5 px-3 py-2 text-left"
        aria-expanded={open}
      >
        {open ? <ChevronDown className="h-3.5 w-3.5 text-zinc-500" /> : <ChevronRight className="h-3.5 w-3.5 text-zinc-500" />}
        <span className="text-xs font-medium text-zinc-300">Try it</span>
        <span className="text-[11px] text-zinc-500">— a made-up email through every step; nothing is sent</span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-zinc-800 px-3 py-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <input
              type="email"
              value={from}
              onChange={(e) => setFrom(e.target.value)}
              placeholder={`From — ${TRY_IT_SAMPLE_FROM} if left blank`}
              className={field}
            />
            <select value={topicId} onChange={(e) => setTopicId(e.target.value)} className={field}>
              <option value="">Any topic that is on</option>
              {topics.map((t) => <option key={t.id} value={t.id}>Only: {t.name}</option>)}
            </select>
          </div>
          {notTheirs.length > 0 && (
            <div className={hint}>
              {notTheirs.map((t) => t.name).join(', ')} only {notTheirs.length === 1 ? 'answers its' : 'answer their'}{' '}
              listed senders, and {sender} isn't one — a real email from it wouldn't be answered. Put
              one of those senders in From to try it as one.
            </div>
          )}
          <input
            type="text"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            placeholder="Subject"
            className={field}
          />
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            rows={4}
            placeholder="Hi! Where is the meetup on the 14th, and do I need to register?"
            className={field}
          />
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              onClick={run}
              disabled={running || !body.trim()}
              className="inline-flex items-center gap-2 rounded-md bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-sm text-white disabled:opacity-50 shadow-sm"
            >
              {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              Try it
            </button>
            <span className="text-[11px] text-zinc-500">
              Uses your checkers and the agent's model, like a real email. Up to 20 an hour.
            </span>
          </div>
          {runError && <div className="text-xs text-red-500 break-words">{runError}</div>}
          {result && <TryItResult result={result} from={resultFrom} />}
        </div>
      )}
    </div>
  )
}

function TryItResult({ result, from }: { result: AutoReplyTestResult; from: string }) {
  const outcome = OUTCOMES[result.outcome] || OUTCOMES.failed
  const m = result.match
  const w = result.writer
  return (
    <div className="space-y-2 text-xs">
      <div className="flex items-center gap-2 flex-wrap">
        <span className={`${pill} ${outcome.cls}`}>{outcome.text}</span>
        {from && <span className="text-zinc-500">as an email from <span className="text-zinc-300">{from}</span></span>}
      </div>
      <div className="text-zinc-400">
        <span className="text-zinc-500">Sorting: </span>
        {result.topic && !result.forced
          ? <>about <span className="text-zinc-200">{result.topic.name}</span></>
          : m ? reasonLabel(m.reason) || 'no topic' : 'not run'}
        {m && (m.source || m.model) && (
          <span className="text-zinc-600">
            {' '}· {m.source === 'jev' ? 'Jev' : m.model || m.source}
            {typeof m.confidence === 'number' ? ` · ${Math.round(m.confidence * 100)}%` : ''}
            {m.fit ? ` · fit ${m.fit}` : ''}
          </span>
        )}
      </div>
      {result.forced && result.topic && (
        <div className="text-zinc-500">
          You picked <span className="text-zinc-300">{result.topic.name}</span>, so the reply was
          written anyway — a real email like this would not be answered.
        </div>
      )}
      {w && (
        <div className="text-zinc-400">
          <span className="text-zinc-500">Writer: </span>
          {w.state === 'reply' ? 'wrote a reply' : reasonLabel(w.reason) || w.state}
          {w.detail ? <span className="text-zinc-600"> · {w.detail}</span> : null}
          {!!w.sources_used?.length && (
            <span className="text-zinc-600"> · used {w.sources_used.join(', ')}</span>
          )}
        </div>
      )}
      {w?.reply && (
        <pre className="whitespace-pre-wrap break-words rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-2 text-xs text-zinc-200 font-sans">
          {w.reply}
        </pre>
      )}
      {result.guard && result.guard.findings.length > 0 && (
        <FindingsList findings={result.guard.findings} />
      )}
      {result.judge && (
        <div className="text-zinc-400">
          <span className="text-zinc-500">Reply check: </span>
          {result.judge.unavailable
            ? 'unavailable — the reply would wait for you'
            : result.judge.passed
              ? <span className="text-emerald-500">passed</span>
              : <span className="text-amber-600">flagged {result.judge.failing.join(', ').replace(/_/g, ' ')}</span>}
          {result.judge.note && <span className="text-zinc-600"> · {result.judge.note}</span>}
        </div>
      )}
    </div>
  )
}

function FindingsList({ findings }: { findings: AutoReplyFinding[] }) {
  return (
    <ul className="space-y-0.5 text-[11px]">
      {findings.map((f, i) => (
        <li key={i} className={f.level === 'block' ? 'text-red-500' : 'text-amber-600'}>
          {f.level === 'block' ? 'Blocks: ' : 'Holds: '}
          {kindLabel(f.kind)}
          {f.hint && <span className="font-mono"> ({f.hint})</span>}
          {f.echo && <span className="text-zinc-500"> — it was in their email</span>}
        </li>
      ))}
    </ul>
  )
}

// ── Activity ────────────────────────────────────────────────────────

function ActivityList({ accountEmail, showAll, onShowAll }: {
  accountEmail: string
  showAll: boolean
  onShowAll: (all: boolean) => void
}) {
  const activity = useGmailAutoReplyStore((s) => s.activity)
  const fetchActivity = useGmailAutoReplyStore((s) => s.fetchActivity)
  const fetchConfig = useGmailAutoReplyStore((s) => s.fetchConfig)
  const [openId, setOpenId] = useState('')
  const [flash, setFlash] = useState('')

  // A held card was acted on (or refused): the row and the counts moved.
  const refresh = async (message: string) => {
    setFlash(message)
    await Promise.all([fetchActivity(showAll), fetchConfig()])
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-xs text-zinc-500">Recent auto-reply activity</span>
        <div className="flex-1" />
        <label className="flex items-center gap-1.5 text-[11px] text-zinc-500 cursor-pointer">
          <input
            type="checkbox"
            checked={showAll}
            onChange={(e) => onShowAll(e.target.checked)}
            className="accent-violet-600"
          />
          Show all
        </label>
      </div>
      {flash && (
        <div className="mb-1.5 flex items-start gap-2 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-500">
          <span className="flex-1">{flash}</span>
          <button type="button" onClick={() => setFlash('')} className="shrink-0 hover:opacity-70" title="Dismiss">
            <X className="h-3 w-3" />
          </button>
        </div>
      )}
      {activity.length === 0 ? (
        <div className="text-xs text-zinc-500">
          {showAll ? 'No emails looked at yet.' : 'Nothing yet. Emails that were simply not for a topic show with “Show all”.'}
        </div>
      ) : (
        <div className="rounded-md border border-zinc-800 bg-zinc-950/60 divide-y divide-zinc-800">
          {/* Everything the server sent: the newest rows plus every open
              held card, however old — cutting it back would hide those. */}
          {activity.map((row) => (
            <ActivityRow
              key={row.id}
              row={row}
              accountEmail={accountEmail}
              open={openId === row.id}
              onToggle={() => setOpenId((v) => (v === row.id ? '' : row.id))}
              onDone={refresh}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function ActivityRow({ row, accountEmail, open, onToggle, onDone }: {
  row: AutoReplyActivity
  accountEmail: string
  open: boolean
  onToggle: () => void
  onDone: (message: string) => Promise<void>
}) {
  const badge = STATE_BADGES[row.state] || WORKING_BADGE
  const expandable = !!row.reply_text || row.findings.length > 0 || !!row.judge?.failing?.length
  const gmailUrl = gmailThreadUrl(accountEmail, row.thread_id)
  return (
    <div className="px-3 py-2 text-xs">
      <button
        type="button"
        onClick={expandable ? onToggle : undefined}
        className={'w-full text-left ' + (expandable ? 'cursor-pointer' : 'cursor-default')}
        aria-expanded={expandable ? open : undefined}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-zinc-500 shrink-0 tabular-nums">{fmtTime(row.received_at || row.created_at)}</span>
          <span className={`${pill} ${badge.cls} shrink-0`} title={badge.title}>{badge.text}</span>
          {row.topic_name && <span className="text-zinc-400 truncate">{row.topic_name}</span>}
          {row.state === 'held' && !open && (
            <span className="ml-auto shrink-0 text-[11px] text-violet-500">Review</span>
          )}
          {expandable && row.state !== 'held' && (
            open
              ? <ChevronDown className="ml-auto h-3.5 w-3.5 text-zinc-500 shrink-0" />
              : <ChevronRight className="ml-auto h-3.5 w-3.5 text-zinc-500 shrink-0" />
          )}
        </div>
        <div className="flex items-baseline gap-1.5 min-w-0 mt-0.5">
          <span className="text-zinc-500 shrink-0">From</span>
          <span className="text-zinc-300 truncate">{row.from || '(unknown)'}</span>
        </div>
        <div className="text-zinc-200 truncate mt-0.5">{row.subject || '(no subject)'}</div>
        {(row.reason || row.sources_used.length > 0) && (
          <div className="text-zinc-500 truncate mt-0.5">
            {reasonLabel(row.reason)}
            {row.reason && row.sources_used.length > 0 ? ' · ' : ''}
            {row.sources_used.length > 0 && `used ${row.sources_used.join(', ')}`}
          </div>
        )}
      </button>
      {open && (
        row.state === 'held'
          ? <HeldCard row={row} gmailUrl={gmailUrl} onDone={onDone} />
          : (
            <div className="mt-2 space-y-1.5">
              {row.reply_text && (
                <pre className="whitespace-pre-wrap break-words rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-2 text-xs text-zinc-300 font-sans">
                  {row.reply_text}
                </pre>
              )}
              {row.findings.length > 0 && <FindingsList findings={row.findings} />}
              {gmailUrl && (
                <a href={gmailUrl} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-[11px] text-violet-500 hover:underline">
                  <ExternalLink className="h-3 w-3" /> Open in Gmail
                </a>
              )}
            </div>
          )
      )}
    </div>
  )
}

// A reply that waits for the owner: send it as it is, edit and send, or
// dismiss it. Flight Deck re-checks it for sensitive data and that nobody
// answered in the thread meanwhile; a second click is refused. The
// Taskboard shows the same card on the held reply's task.
export function HeldCard({ row, gmailUrl, onDone }: {
  row: AutoReplyActivity
  gmailUrl: string
  onDone: (message: string) => Promise<void>
}) {
  const sendHeld = useGmailAutoReplyStore((s) => s.sendHeld)
  const dismissHeld = useGmailAutoReplyStore((s) => s.dismissHeld)
  const [text, setText] = useState(row.reply_text)
  const [busy, setBusy] = useState<'' | 'send' | 'dismiss'>('')
  const [actionError, setActionError] = useState('')
  const [copied, setCopied] = useState(false)
  const edited = text !== row.reply_text
  const failing = row.judge?.failing || []

  const send = async (withEdit: boolean) => {
    if (!confirm(`Send this reply to ${row.from || 'the sender'} now?`)) return
    setBusy('send')
    setActionError('')
    const out = await sendHeld(row.id, withEdit ? text : undefined)
    setBusy('')
    if (!out.ok) {
      setActionError(out.error)
      await onDone('')
      return
    }
    await onDone(`Sent to ${out.data.to || row.from}.`)
  }

  const dismiss = async () => {
    setBusy('dismiss')
    setActionError('')
    const out = await dismissHeld(row.id)
    setBusy('')
    if (!out.ok) {
      setActionError(out.error)
      await onDone('')
      return
    }
    await onDone('Dismissed — nothing was sent.')
  }

  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }).catch(() => {})
  }

  return (
    <div className="mt-2 space-y-2">
      {(row.findings.length > 0 || failing.length > 0 || row.reason) && (
        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-2.5 py-1.5 space-y-0.5">
          <div className="text-[11px] text-amber-600">Why it waits: {reasonLabel(row.reason) || 'it needs your look'}</div>
          {row.findings.length > 0 && <FindingsList findings={row.findings} />}
          {failing.length > 0 && (
            <div className="text-[11px] text-amber-600">
              Reply check flagged: {failing.join(', ').replace(/_/g, ' ')}
              {row.judge?.note ? ` — ${row.judge.note}` : ''}
            </div>
          )}
        </div>
      )}
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={8}
        disabled={!!busy}
        className={field}
      />
      {actionError && <div className="text-xs text-red-500 break-words">{actionError}</div>}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          type="button"
          onClick={() => send(false)}
          disabled={!!busy || edited}
          className="inline-flex items-center gap-1.5 rounded-md bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-xs text-white disabled:opacity-50 shadow-sm"
          title={edited ? 'You edited the text — use Edit & send, or Reset' : 'Send the reply as it is'}
        >
          {busy === 'send' && !edited ? <Loader2 className="h-3 w-3 animate-spin" /> : <Send className="h-3 w-3" />}
          Send
        </button>
        <button
          type="button"
          onClick={() => send(true)}
          disabled={!!busy || !edited || !text.trim()}
          className="inline-flex items-center gap-1.5 rounded-md border border-violet-500/40 hover:bg-violet-500/10 px-3 py-1.5 text-xs text-violet-500 disabled:opacity-50"
          title="Send your edited text instead"
        >
          {busy === 'send' && edited ? <Loader2 className="h-3 w-3 animate-spin" /> : <Send className="h-3 w-3" />}
          Edit &amp; send
        </button>
        {edited && (
          <button type="button" onClick={() => setText(row.reply_text)} disabled={!!busy} className={smallBtn}>
            <Undo2 className="h-3 w-3" /> Reset
          </button>
        )}
        <button type="button" onClick={dismiss} disabled={!!busy} className={smallBtn}>
          {busy === 'dismiss' ? <Loader2 className="h-3 w-3 animate-spin" /> : <X className="h-3 w-3" />}
          Dismiss
        </button>
        <button type="button" onClick={copy} className={smallBtn}>
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
        {gmailUrl && (
          <a href={gmailUrl} target="_blank" rel="noopener noreferrer" className={smallBtn}>
            <ExternalLink className="h-3 w-3" /> Open in Gmail
          </a>
        )}
      </div>
      <div className="text-[11px] text-zinc-500">
        Sending re-checks the text for sensitive data and that nobody answered in the thread
        meanwhile. Held replies can be sent for 7 days.
      </div>
    </div>
  )
}
