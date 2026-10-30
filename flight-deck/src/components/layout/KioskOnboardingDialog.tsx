import { useCallback, useEffect, useMemo, useRef, useState, type ComponentType, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Compass,
  IdCard,
  Lightbulb,
  Mail,
  Maximize2,
  MessagesSquare,
  Plus,
  Send,
  Sparkles,
  X,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { useOnboardingStore } from '../../stores/onboardingStore'
import { useSharedAgentStore } from '../../stores/sharedAgentStore'
import { useThemeStore } from '../../stores/themeStore'
import { getMyInfo, type MyInfo } from '../../services/profile'
import overviewDark from '../../assets/onboarding/overview-dark.webp'
import overviewLight from '../../assets/onboarding/overview-light.webp'
import newAgentDark from '../../assets/onboarding/new-agent-dark.webp'
import newAgentLight from '../../assets/onboarding/new-agent-light.webp'
import chatDark from '../../assets/onboarding/chat-dark.webp'
import chatLight from '../../assets/onboarding/chat-light.webp'
import googleDark from '../../assets/onboarding/google-dark.webp'
import googleLight from '../../assets/onboarding/google-light.webp'
import telegramDark from '../../assets/onboarding/telegram-dark.webp'
import telegramLight from '../../assets/onboarding/telegram-light.webp'
import profileDark from '../../assets/onboarding/profile-dark.webp'
import profileLight from '../../assets/onboarding/profile-light.webp'
import settingsDark from '../../assets/onboarding/settings-dark.webp'
import settingsLight from '../../assets/onboarding/settings-light.webp'
import overviewPhoneDark from '../../assets/onboarding/overview-phone-dark.webp'
import overviewPhoneLight from '../../assets/onboarding/overview-phone-light.webp'
import newAgentPhoneDark from '../../assets/onboarding/new-agent-phone-dark.webp'
import newAgentPhoneLight from '../../assets/onboarding/new-agent-phone-light.webp'
import chatPhoneDark from '../../assets/onboarding/chat-phone-dark.webp'
import chatPhoneLight from '../../assets/onboarding/chat-phone-light.webp'
import googlePhoneDark from '../../assets/onboarding/google-phone-dark.webp'
import googlePhoneLight from '../../assets/onboarding/google-phone-light.webp'
import telegramPhoneDark from '../../assets/onboarding/telegram-phone-dark.webp'
import telegramPhoneLight from '../../assets/onboarding/telegram-phone-light.webp'
import profilePhoneDark from '../../assets/onboarding/profile-phone-dark.webp'
import profilePhoneLight from '../../assets/onboarding/profile-phone-light.webp'
import settingsPhoneDark from '../../assets/onboarding/settings-phone-dark.webp'
import settingsPhoneLight from '../../assets/onboarding/settings-phone-light.webp'

// ── Kiosk: Getting started ───────────────────────────────────────────
//
// The guide a kiosk user gets the first time they land on the deck: what it
// is, how to create an agent, how to give it work, how to connect Google /
// Microsoft and Telegram, the profile, and where everything else lives. It
// opens by itself until "Do not show again" is ticked (a per-user setting,
// so it holds on a shared machine and across devices); Info and Settings
// open it again. Steps for a connection the deck doesn't offer are left out.
// The screenshots are real ones (e2e/onboarding regenerates them), of the
// layout the viewer has (phone or larger) in their theme. Full-screen on a
// phone, where a swipe turns the page; a panel on a larger screen.

/** What a step's button asks the kiosk to open once the guide is closed. */
export type OnboardingAction =
  | 'new-agent'
  | 'connections-google'
  | 'connections-microsoft'
  | 'connections-telegram'
  | 'profile'

/** One screen, shot in both layouts and both themes. */
interface Shot {
  dark: string
  light: string
  phoneDark: string
  phoneLight: string
  alt: string
  /** When the phone screen shows something else than the larger one. */
  phoneAlt?: string
}

interface Step {
  id: string
  nav: string
  icon: ComponentType<{ className?: string }>
  title: string
  lead: ReactNode
  points: ReactNode[]
  numbered?: boolean
  note?: ReactNode
  status?: ReactNode
  shot?: Shot
  actions?: { label: string; action: OnboardingAction }[]
}

const SHOTS = {
  overview: {
    dark: overviewDark, light: overviewLight, phoneDark: overviewPhoneDark, phoneLight: overviewPhoneLight,
    alt: 'The deck: agents on the left with Library and Data at the bottom, the chat in the middle',
    phoneAlt: 'Your agents, one per row, with New agent at the bottom',
  },
  newAgent: {
    dark: newAgentDark, light: newAgentLight, phoneDark: newAgentPhoneDark, phoneLight: newAgentPhoneLight,
    alt: 'New agent: pick an archetype, name it, Create agent',
  },
  chat: {
    dark: chatDark, light: chatLight, phoneDark: chatPhoneDark, phoneLight: chatPhoneLight,
    alt: 'At work: a conversation, two tasks in the queue beside it, the next request typed',
    phoneAlt: 'A chat: the conversation, and the next request typed in the message box',
  },
  google: {
    dark: googleDark, light: googleLight, phoneDark: googlePhoneDark, phoneLight: googlePhoneLight,
    alt: 'Settings → Connections, Google tab, with the Connect Google button',
  },
  telegram: {
    dark: telegramDark, light: telegramLight, phoneDark: telegramPhoneDark, phoneLight: telegramPhoneLight,
    alt: 'Settings → Connections, Telegram tab: pick an agent and connect',
  },
  profile: {
    dark: profileDark, light: profileLight, phoneDark: profilePhoneDark, phoneLight: profilePhoneLight,
    alt: 'Profile: about me, my company, my standing preferences',
  },
  settings: {
    dark: settingsDark, light: settingsLight, phoneDark: settingsPhoneDark, phoneLight: settingsPhoneLight,
    alt: 'Settings: your agents at work, what they keep, Profile and Connections, the theme, Refresh agents, Getting started and Sign out',
  },
} satisfies Record<string, Shot>

const B = ({ children }: { children: ReactNode }) => <span className="font-medium text-zinc-100">{children}</span>

export function KioskOnboardingDialog({ phone, onClose, onAction }: {
  /** The phone layout: its screenshots, and its words for where things are. */
  phone: boolean
  onClose: () => void
  /** A step's button: the guide closes, then that part of the kiosk opens. */
  onAction: (action: OnboardingAction) => void
}) {
  const branding = useAuthStore((s) => s.branding)
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const sharingEnabled = useSharedAgentStore((s) => s.enabled)
  const theme = useThemeStore((s) => s.theme)
  const optedOut = useOnboardingStore((s) => s.completed.kiosk)
  const setOptedOut = useOnboardingStore((s) => s.setStepCompleted)
  const [info, setInfo] = useState<MyInfo | null>(null)
  const [index, setIndex] = useState(0)
  const [zoom, setZoom] = useState<Shot | null>(null)
  // Where a touch on the page started: a sideways swipe turns the page.
  const touch = useRef<{ x: number; y: number } | null>(null)

  // Which connections this deck offers, and the user's own state of each.
  // Best effort: until (or unless) it answers, every step shows.
  useEffect(() => {
    let live = true
    getMyInfo().then((d) => { if (live) setInfo(d) }).catch(() => {})
    return () => { live = false }
  }, [])

  const steps = useMemo(
    () => buildSteps({ name: branding.name, phone, authEnabled: authEnabled !== false, sharingEnabled, info }),
    [branding.name, phone, authEnabled, sharingEnabled, info],
  )
  // A step can drop out once /fd/info answers: stay within the list.
  const at = Math.min(index, steps.length - 1)
  const step = steps[at]
  const last = at === steps.length - 1

  const go = useCallback((i: number) => setIndex(Math.max(0, Math.min(i, steps.length - 1))), [steps.length])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { if (zoom) setZoom(null); else onClose(); return }
      if (zoom) return
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return
      if (e.key === 'ArrowRight') go(at + 1)
      else if (e.key === 'ArrowLeft') go(at - 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [at, go, onClose, zoom])

  const src = (s: Shot) => (phone
    ? (theme === 'light' ? s.phoneLight : s.phoneDark)
    : (theme === 'light' ? s.light : s.dark))
  const alt = (s: Shot) => (phone && s.phoneAlt) || s.alt
  const Icon = step.icon

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Getting started"
        className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 shadow-2xl wide:h-[min(88vh,820px)] wide:max-w-4xl wide:rounded-2xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex h-14 shrink-0 items-center gap-2 border-b border-zinc-800 px-4 wide:px-5">
          <Compass className="h-4 w-4 shrink-0 text-violet-400" />
          <h2 className="text-base font-semibold text-zinc-100">Getting started</h2>
          <span className="text-xs text-zinc-500">{at + 1} of {steps.length}</span>
          <button onClick={onClose} aria-label="Close" title="Close" className="ml-auto rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-5 w-5" />
          </button>
        </div>
        {/* Progress — on a phone the only map of the steps */}
        <div className="h-0.5 shrink-0 bg-zinc-900 wide:hidden">
          <div className="h-full bg-violet-500 transition-all" style={{ width: `${((at + 1) / steps.length) * 100}%` }} />
        </div>

        <div className="flex min-h-0 flex-1">
          {/* Contents — a larger screen lists every topic */}
          <nav aria-label="Topics" className="hidden w-52 shrink-0 flex-col gap-0.5 overflow-y-auto border-r border-zinc-800 p-3 wide:flex">
            {steps.map((s, i) => {
              const NavIcon = s.icon
              const current = i === at
              return (
                <button
                  key={s.id}
                  onClick={() => go(i)}
                  aria-current={current ? 'step' : undefined}
                  className={`flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors ${
                    current ? 'bg-violet-500/10 font-medium text-zinc-100' : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200'
                  }`}
                >
                  <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md ${
                    current ? 'bg-violet-500/15 text-violet-400' : i < at ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-500'
                  }`}>
                    {i < at ? <Check className="h-3.5 w-3.5" /> : <NavIcon className="h-3.5 w-3.5" />}
                  </span>
                  <span className="min-w-0 truncate">{s.nav}</span>
                </button>
              )
            })}
          </nav>

          {/* The step */}
          <div
            key={step.id}
            className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-4 wide:p-6"
            onTouchStart={(e) => { touch.current = { x: e.touches[0].clientX, y: e.touches[0].clientY } }}
            onTouchEnd={(e) => {
              const from = touch.current
              touch.current = null
              if (!from) return
              const dx = e.changedTouches[0].clientX - from.x
              const dy = e.changedTouches[0].clientY - from.y
              // Clearly sideways, not a scroll that drifted.
              if (Math.abs(dx) > 60 && Math.abs(dx) > 2 * Math.abs(dy)) go(at + (dx < 0 ? 1 : -1))
            }}
          >
            {step.shot && (
              <button
                onClick={() => setZoom(step.shot!)}
                title="Enlarge"
                aria-label="Enlarge screenshot"
                className={`group relative mb-5 flex w-full items-center justify-center overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/60 p-2 ${
                  // A phone on its side: the screen beside the text, not above it,
                  // with a strip under it for the badge (the box hugs the image there).
                  phone ? '[@media(max-height:500px)]:float-right [@media(max-height:500px)]:mb-2 [@media(max-height:500px)]:ml-4 [@media(max-height:500px)]:w-auto [@media(max-height:500px)]:pb-8' : ''
                }`}
              >
                <img
                  src={src(step.shot)}
                  alt={alt(step.shot)}
                  decoding="async"
                  // Small enough on a short screen to leave the step's title
                  // and first lines in view; a phone's own (portrait) screen
                  // gets a little more height.
                  className={`w-auto max-w-full object-contain ${phone
                    ? 'max-h-[min(36dvh,340px)] rounded-xl border border-zinc-800 [@media(max-height:500px)]:max-h-[50dvh]'
                    : 'max-h-[28dvh] rounded-lg wide:max-h-[min(270px,30vh)]'}`}
                />
                <span className="absolute bottom-2 right-2 flex items-center gap-1 rounded-md bg-black/60 p-1 text-[11px] text-white opacity-80 transition-opacity group-hover:opacity-100 wide:px-1.5">
                  <Maximize2 className="h-3 w-3" />
                  <span className="hidden wide:inline">Enlarge</span>
                </span>
              </button>
            )}

            <div className="mb-2 flex items-center gap-2.5">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-500/15 text-violet-400">
                <Icon className="h-4 w-4" />
              </span>
              <h3 className="text-lg font-semibold leading-tight text-zinc-100">{step.title}</h3>
            </div>
            <p className="text-sm leading-relaxed text-zinc-400">{step.lead}</p>
            {step.status && <div className="mt-3 flex flex-wrap gap-1.5">{step.status}</div>}

            {step.numbered ? (
              <ol className="mt-4 space-y-2.5">
                {step.points.map((p, i) => (
                  <li key={i} className="flex gap-3 text-sm leading-relaxed text-zinc-300">
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet-600 text-xs font-semibold text-white">{i + 1}</span>
                    <span className="min-w-0 pt-0.5">{p}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <ul className="mt-4 space-y-2.5">
                {step.points.map((p, i) => (
                  <li key={i} className="flex gap-3 text-sm leading-relaxed text-zinc-300">
                    <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-violet-400" />
                    <span className="min-w-0">{p}</span>
                  </li>
                ))}
              </ul>
            )}

            {step.actions && step.actions.length > 0 && (
              <div className="mt-5 flex flex-wrap gap-2">
                {step.actions.map((a) => (
                  <button
                    key={a.action}
                    onClick={() => onAction(a.action)}
                    className="flex h-11 items-center gap-1.5 rounded-lg border border-violet-500/40 bg-violet-500/10 px-4 text-sm font-medium text-violet-700 transition-colors hover:bg-violet-500/20 dark:text-violet-300 wide:h-auto wide:px-3.5 wide:py-2"
                  >
                    {a.label}
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                ))}
              </div>
            )}

            {step.note && (
              <p className="mt-5 flow-root rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2.5 text-xs leading-relaxed text-zinc-400">{step.note}</p>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-t border-zinc-800 px-4 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:px-5">
          <label className="flex cursor-pointer select-none items-center gap-2 py-2 text-sm text-zinc-400 wide:py-1">
            <input
              type="checkbox"
              checked={optedOut}
              onChange={(e) => setOptedOut('kiosk', e.target.checked)}
              className="h-4 w-4 accent-violet-600"
            />
            Do not show again
          </label>
          <div className="ml-auto flex items-center gap-2">
            {at > 0 && (
              <button
                onClick={() => go(at - 1)}
                aria-label="Back"
                title="Back"
                // Just the arrow on a phone: the footer stays one row on a narrow screen.
                className="flex h-11 items-center gap-1.5 rounded-lg border border-zinc-700 px-3 text-sm font-medium text-zinc-300 transition-colors hover:bg-zinc-800 wide:h-auto wide:px-3.5 wide:py-2"
              >
                <ArrowLeft className="h-4 w-4" />
                <span className="hidden wide:inline">Back</span>
              </button>
            )}
            <button
              onClick={() => (last ? onClose() : go(at + 1))}
              className="flex h-11 items-center gap-1.5 rounded-lg bg-violet-600 px-5 text-sm font-medium text-white transition-colors hover:bg-violet-500 wide:h-auto wide:px-4 wide:py-2"
            >
              {last ? 'Done' : 'Next'}
              {last ? <Check className="h-4 w-4" /> : <ArrowRight className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>

      {zoom && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/85 p-3 wide:p-8"
          onClick={(e) => { e.stopPropagation(); setZoom(null) }}
        >
          <img src={src(zoom)} alt={alt(zoom)} className="max-h-full max-w-full rounded-lg object-contain shadow-2xl" />
          <button
            onClick={(e) => { e.stopPropagation(); setZoom(null) }}
            aria-label="Close screenshot"
            title="Close"
            className="absolute right-3 top-3 rounded-lg bg-black/50 p-2 text-white hover:bg-black/70"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
      )}
    </div>,
    document.body,
  )
}

// ── The steps ────────────────────────────────────────────────────────

type Conn = { configured: boolean; connected: boolean } | undefined

function Chip({ ok, children }: { ok: boolean; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs ${
      ok
        ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
        : 'border-zinc-700 bg-zinc-900 text-zinc-400'
    }`}>
      {ok ? <Check className="h-3 w-3" /> : <span className="h-1.5 w-1.5 rounded-full bg-zinc-500" />}
      {children}
    </span>
  )
}

function buildSteps({ name, phone, authEnabled, sharingEnabled, info }: {
  name: string
  phone: boolean
  /** Sign-in on: Google, Microsoft and the team bot each belong to a signed-in user. */
  authEnabled: boolean
  sharingEnabled: boolean
  info: MyInfo | null
}): Step[] {
  const c = info?.connections
  // Unknown (not loaded, or the deck didn't say) counts as offered.
  const offered = (x: Conn) => authEnabled && x?.configured !== false
  const google = offered(c?.google)
  const microsoft = offered(c?.microsoft)
  const telegram = authEnabled && c?.telegram?.configured !== false
  const steps: Step[] = []
  // Where Settings and Info are: the top of a phone's agent list, the foot of the agents column.
  const settingsAt = phone ? <>Tap <B>Settings</B> (the gear, top right)</> : <>Open <B>Settings</B> (bottom left)</>

  steps.push({
    id: 'welcome',
    nav: 'What is this',
    icon: Sparkles,
    title: `Welcome to ${name}`,
    lead: <>{name} is where your AI agents live. An agent is an assistant with its own role, tools, memory and files: you ask in plain words, and it researches, writes and gets the work done for you.</>,
    points: phone
      ? [
          <><B>Your agents</B> are the first screen: tap one to chat. Green means running, amber working, blue a new reply.</>,
          <><B>In a chat</B>, the folder icon at the top opens the files the agent saved for you.</>,
          <><B>Back</B> returns to the list: the agent keeps working meanwhile.</>,
        ]
      : [
          <><B>Left: your agents.</B> Click one to chat. Green means running, amber working, blue a new reply.</>,
          <><B>Middle: the conversation.</B> Ask, attach files, and follow its work as it goes.</>,
          <><B>Library, Data and Folders</B> (bottom left) gather what the agents you create here save: documents, spreadsheets, tables, and the folders they share.</>,
        ],
    shot: SHOTS.overview,
  })

  steps.push({
    id: 'agent',
    nav: 'Create an agent',
    icon: Plus,
    title: 'Create your first agent',
    lead: 'Agents start from archetypes: ready-made roles, like a researcher or a writer, that come with the right tools and model. There is nothing to configure.',
    numbered: true,
    points: [
      phone
        ? <>Press <B>New agent</B> at the bottom of the screen.</>
        : <>Press <B>+</B> next to “Agents” on the left (or <B>New agent</B> on an empty deck).</>,
      'Pick an archetype from the list. Search by role or by what it does.',
      <>Give it a name and press <B>Create agent</B>. It starts in a few seconds, and its chat opens by itself.</>,
    ],
    note: <>The power button next to an agent turns it off when you don't need it, and back on later. Its conversations and files are kept.</>,
    shot: SHOTS.newAgent,
    actions: [{ label: 'Create an agent', action: 'new-agent' }],
  })

  steps.push({
    id: 'chat',
    nav: 'Give it work',
    icon: MessagesSquare,
    title: 'Give it work',
    lead: "Write what you need the way you'd brief a colleague: the goal, the context, and what you want back.",
    points: [
      phone
        ? <><B>Attach files</B> with the paperclip: documents, spreadsheets, photos.</>
        : <><B>Attach files</B> with the paperclip, or drop or paste them into the message: documents, spreadsheets, images.</>,
      phone
        ? <><B>Line up several tasks</B> in the queue: the list icon on the row above the chat switches to it. On <B>Auto</B>, the agent works through them one after another.</>
        : <><B>Line up several tasks</B> in the queue beside the chat (the list icon at the top shows it). On <B>Auto</B>, the agent works through them one after another.</>,
      <><B>Change topic</B> with <B>{phone ? 'New' : 'New topic'}</B>: the agent starts with a fresh context, and everything said so far stays above the divider. It keeps its memory and files.</>,
      <><B>Topics</B> (next to it) lists earlier topics: tap one to jump back to it, or resume it so the agent picks that context up again.</>,
      <><B>What it makes</B> lands in the <B>Library</B> (files), <B>Data</B> (tables) and <B>Folders</B> (what agents share), {phone ? 'at the top of your agent list (and behind the folder icon at the top of the chat)' : 'bottom left'}. Open, preview or download it there.</>,
    ],
    note: 'It keeps working while you chat with another agent: a blue dot tells you a reply is waiting.',
    shot: SHOTS.chat,
  })

  if (google || microsoft) {
    const both = google && microsoft
    const which = both ? 'Google or Microsoft' : google ? 'Google' : 'Microsoft'
    const status: ReactNode[] = []
    if (google && c?.google) status.push(<Chip key="g" ok={c.google.connected}>Google: {c.google.connected ? 'connected' : 'not connected yet'}</Chip>)
    if (microsoft && c?.microsoft) status.push(<Chip key="m" ok={c.microsoft.connected}>Microsoft: {c.microsoft.connected ? 'connected' : 'not connected yet'}</Chip>)
    const actions: Step['actions'] = []
    if (google) actions.push({ label: 'Connect Google', action: 'connections-google' })
    if (microsoft) actions.push({ label: 'Connect Microsoft', action: 'connections-microsoft' })
    steps.push({
      id: 'google',
      nav: both ? 'Google & Microsoft' : which,
      icon: Mail,
      title: `Connect ${which}`,
      lead: `Connected, your agents work with your own email, calendar and documents: read and draft mail, find a free slot, pull a file from ${both ? 'Drive or OneDrive' : google ? 'Drive' : 'OneDrive'}.`,
      numbered: true,
      points: [
        <>{settingsAt}, then <B>Connections</B>.</>,
        both
          ? <>Choose the <B>Google</B> or <B>Microsoft</B> tab and press <B>Connect Google</B> or <B>Connect Microsoft</B>.</>
          : <>On the <B>{which}</B> tab, press <B>Connect {which}</B>.</>,
        'Sign in with your own account in the window that opens, and allow access.',
      ],
      note: "It's your account, and only your agents use it: everyone on this deck connects their own. Disconnect any time, in the same place.",
      status: status.length ? status : undefined,
      shot: SHOTS.google,
      actions,
    })
  }

  if (telegram) {
    const t = c?.telegram
    let status: ReactNode
    if (t && t.status === 'approved' && t.linked) status = <Chip ok>Telegram: connected{t.agent ? ` to ${t.agent}` : ''}</Chip>
    else if (t?.status === 'pending') status = <Chip ok={false}>Telegram: waiting for approval</Chip>
    else if (t?.status === 'approved') status = <Chip ok={false}>Telegram: approved, open the bot to finish</Chip>
    else if (t) status = <Chip ok={false}>Telegram: not connected yet</Chip>
    steps.push({
      id: 'telegram',
      nav: 'Telegram',
      icon: Send,
      title: 'Chat from Telegram',
      lead: "Talk to one of your agents from your phone, through this deck's Telegram bot: handy when you're away from the screen.",
      numbered: true,
      points: [
        <>{settingsAt}, then <B>Connections</B> → <B>Telegram</B>.</>,
        <>Pick the agent that should answer you and press <B>Connect</B> (or <B>Request access</B>, where an admin approves new people).</>,
        <>Press <B>Open … in Telegram</B> and tap <B>Start</B> in the bot. The link works once: it ties your Telegram to your account here.</>,
      ],
      note: 'The agent has to be running to answer. Waiting for approval? The bot messages you once you are in.',
      status,
      shot: SHOTS.telegram,
      actions: [{ label: 'Set up Telegram', action: 'connections-telegram' }],
    })
  }

  const own = info?.profile?.own
  const filled = !!own && !!(own.about_me || own.company || own.instructions)
  steps.push({
    id: 'profile',
    nav: 'Your profile',
    icon: IdCard,
    title: 'Tell your agents about you',
    lead: 'Your profile is a short note every agent of yours receives, so you never repeat yourself.',
    points: [
      <><B>About me:</B> your role and what you work on.</>,
      <><B>My company:</B> what it does and who it serves.</>,
      <><B>My standing preferences:</B> language, tone, format. “Answer in Croatian”, “keep it short”, “tables over prose”.</>,
    ],
    note: <>It's in <B>Settings</B> → <B>Profile</B>. <B>Info</B> shows the exact text your agents get.</>,
    status: info?.profile ? <Chip ok={filled}>{filled ? 'Profile filled in' : 'Profile not filled in yet'}</Chip> : undefined,
    shot: SHOTS.profile,
    actions: [{ label: 'Edit my profile', action: 'profile' }],
  })

  const tips: ReactNode[] = [
    <><B>Info</B> (the ⓘ {phone ? 'at the top' : 'next to Settings'}) shows your plan and limits, the models your agents run on, and what they know about you.</>,
    <><B>Settings</B> holds the light and dark theme, Connections, Profile{authEnabled ? ' and Sign out' : ''}.</>,
    <><B>The bell</B> tells you when an agent finishes something or needs you.</>,
  ]
  if (authEnabled) tips.push(<><B>Daily limits:</B> your plan allows a number of messages and some AI usage each day. The line above the message box shows what's left; it starts over every day at midnight UTC.</>)
  if (sharingEnabled) tips.push(<><B>Shared with you:</B> agents a colleague shares show “Shared by …” in your list. You can leave one any time.</>)
  if (authEnabled) tips.push(<><B>On a shared screen</B>, sign out when you're done (<B>Settings</B> → <B>Sign out</B>): the next person starts clean, and your agents stay yours.</>)
  tips.push(<><B>This guide</B> is always one click away: <B>Info</B> or <B>Settings</B> → <B>Getting started</B>.</>)
  steps.push({
    id: 'tips',
    nav: 'Good to know',
    icon: Lightbulb,
    title: 'Good to know',
    lead: 'A few more things that make the deck easier to use.',
    points: tips,
    shot: SHOTS.settings,
  })

  return steps
}
