import { useEffect, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  AlertTriangle, ArrowRight, Bot, Check, CheckCircle2, CircleDot, Copy, Download, Globe, Loader2, Send, ShieldCheck, X, XCircle, Zap,
} from 'lucide-react'
import { EXTENSION_ZIP, type ChromeActivity, type ChromeActNow } from '../../services/browserClip'
import { timeAgo } from '../console/format'

// ── Captain for Chrome in the sidebar ────────────────────────────────
//
// The Browser group's row: whether the person's Chrome is connected to this
// Flight Deck and, when it is, what it's doing for their agents right now
// (browser_clip.browser_activity on the deck). Not connected — a warning
// pill; the row opens a dialog with the download, how to put it in Chrome
// and what it's for. Connected — the dialog shows the browsers, what runs
// now and what ran last. The deck is asked again every few seconds, quicker
// while something runs or the dialog is open, never while the tab is hidden.

type Tone = 'off' | 'ready' | 'busy' | 'needs'

interface Status {
  tone: Tone
  /** The warning pill (not connected, or an update). */
  pill?: string
  /** The second line: what it's doing, or how it is. */
  line?: string
  /** Said in full: the tooltip and the screen reader's label. */
  words: string
}

function doingLine(a: ChromeActNow): string {
  return a.site && !a.doing.includes(a.site) ? `${a.doing} · ${a.site}` : a.doing
}

function chromeStatus(d: ChromeActivity): Status {
  if (d.state !== 'connected') {
    const words = d.state === 'offline'
      ? `Paired with ${d.paired.map((p) => p.name).join(', ')}, but not connected now — open Chrome. Click for help.`
      : "Not connected — your agents can't use your Chrome yet. Click to get the extension."
    return { tone: 'off', pill: 'Not connected', words }
  }
  const waiting = d.now.find((a) => a.needs_you)
  if (waiting) {
    const line = `${waiting.agent} needs you · robot check${waiting.site ? ` on ${waiting.site}` : ''}`
    return { tone: 'needs', line, words: `Connected — ${line}. It's open in your Chrome.` }
  }
  if (d.now.length) {
    const latest = d.now[d.now.length - 1]
    const more = d.now.length > 1 ? ` +${d.now.length - 1}` : ''
    const line = `${latest.agent} · ${doingLine(latest)}${more}`
    return { tone: 'busy', line, words: `Connected — ${line}` }
  }
  const update = d.browsers.some((b) => b.version_state === 'update')
  const pill = update ? 'Update' : undefined
  if (!d.lent) {
    return { tone: 'ready', pill, line: "Agents can't use it yet",
      words: 'Connected, but "Agents can use this browser" is off in its Settings.' }
  }
  const last = d.recent[0]
  const line = last ? `Ready · used ${timeAgo(last.at)}` : 'Ready for your agents'
  return { tone: 'ready', pill, line, words: `Connected — ${line}${update ? ' · a newer extension is ready' : ''}` }
}

const DOT: Record<Tone, string> = {
  off: 'bg-amber-500',
  ready: 'bg-emerald-500',
  busy: 'bg-amber-500 animate-pulse dark:bg-amber-400',
  needs: 'bg-[var(--fd-accent)] animate-pulse',
}

const LINE: Record<Tone, string> = {
  off: 'text-zinc-500',
  ready: 'text-zinc-500',
  busy: 'text-amber-700 dark:text-amber-400',
  needs: 'font-medium text-[var(--fd-accent-text)]',
}

const PILL = 'shrink-0 rounded-full border border-amber-500/30 bg-amber-500/15 px-1.5 text-[10px] font-semibold leading-[18px] text-amber-700 dark:text-amber-300'

/** The Browser group's row in the open column: its name, and under it the
 *  warning pill (not connected, an update) and what it's doing. */
export function ChromeNavRow({ data, onOpen, open }: { data: ChromeActivity; onOpen: () => void; open: boolean }) {
  const s = chromeStatus(data)
  return (
    <button onClick={onOpen} title={s.words} aria-label={`Chrome extension — ${s.words}`} aria-haspopup="dialog"
      aria-current={open ? 'page' : undefined} className="fd-nav-item py-1.5">
      <span className="relative shrink-0 self-start pt-0.5">
        <Globe className="fd-nav-icon" />
        <span className={`absolute -right-0.5 bottom-0 h-2 w-2 rounded-full ring-2 ring-[var(--fd-shell)] ${DOT[s.tone]}`} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate">Chrome extension</span>
        {(s.pill || s.line) && (
          <span className="mt-0.5 flex min-w-0 items-center gap-1.5">
            {s.pill && <span className={PILL}>{s.pill}</span>}
            {s.line && <span className={`min-w-0 truncate text-[11px] leading-4 ${LINE[s.tone]}`}>{s.line}</span>}
          </span>
        )}
      </span>
    </button>
  )
}

/** The same on the folded rail: the icon, its state on the corner. */
export function ChromeRailButton({ data, onOpen, open }: { data: ChromeActivity; onOpen: () => void; open: boolean }) {
  const s = chromeStatus(data)
  return (
    <button onClick={onOpen} title={`Chrome extension — ${s.words}`} aria-label={`Chrome extension — ${s.words}`}
      aria-haspopup="dialog" aria-current={open ? 'page' : undefined} className="fd-nav-item fd-nav-rail">
      <Globe className="fd-nav-icon" />
      <span className={`absolute right-1.5 top-1.5 h-2 w-2 rounded-full ring-2 ring-[var(--fd-shell)] ${DOT[s.tone]}`} />
    </button>
  )
}

// ── The dialog ───────────────────────────────────────────────────────

function CopyText({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }).catch(() => {})
  }
  return (
    <span className="inline-flex max-w-full items-center gap-1 align-middle">
      <code className="truncate rounded bg-zinc-800 px-1.5 py-0.5 text-xs text-zinc-100">{text}</code>
      <button onClick={copy} aria-label={`Copy ${label}`} title={`Copy ${label}`}
        className="inline-flex shrink-0 items-center gap-1 rounded px-1 py-0.5 text-xs text-violet-700 hover:bg-violet-500/10 dark:text-violet-300">
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}{copied ? 'Copied' : 'Copy'}
      </button>
    </span>
  )
}

function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  )
}

const USES: { icon: typeof Globe; title: string; text: ReactNode }[] = [
  { icon: Send, title: 'Send a page to an agent',
    text: <>Click the Captain icon on any page (or <b className="text-zinc-300">Alt+Shift+C</b>, or right-click) and pick an agent: summarize it, grab its files, pull out the data, watch it for changes.</> },
  { icon: Bot, title: 'Let agents use your Chrome',
    text: <>With <b className="text-zinc-300">Agents can use this browser</b> on, your agents work in tabs of their own tab group, signed in as you. You watch them, and Stop any of them from the panel.</> },
  { icon: Zap, title: 'Run playbooks in it',
    text: <>Ready-made web tasks — prices, bookings, orders — run in your Chrome whenever it's connected. They stop before anything that pays or sends.</> },
  { icon: ShieldCheck, title: 'Get past sites that block bots',
    text: <>When a site turns the server's browser away, agents carry on in yours. Web searches and page reads can go through it too: <b className="text-zinc-300">Profile → Web searches</b> and <b className="text-zinc-300">Web pages</b>.</> },
  { icon: CircleDot, title: 'Record a process once',
    text: <>Record what you do in the panel; an agent does it again with new inputs, as often as you like.</> },
]

function HowToUse() {
  return (
    <ul className="grid gap-2 wide:grid-cols-2">
      {USES.map(({ icon: Icon, title, text }) => (
        <li key={title} className="flex gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
          <Icon className="mt-0.5 h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />
          <div className="min-w-0">
            <div className="text-sm font-medium text-zinc-100">{title}</div>
            <p className="mt-0.5 text-xs leading-relaxed text-zinc-400">{text}</p>
          </div>
        </li>
      ))}
    </ul>
  )
}

function GetIt({ data }: { data: ChromeActivity }) {
  const deck = window.location.origin
  const step = 'flex gap-3'
  const num = 'flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet-500/15 text-xs font-semibold text-violet-700 dark:text-violet-300'
  return (
    <ol className="space-y-3 text-sm text-zinc-300">
      <li className={step}>
        <span className={num}>1</span>
        <div className="min-w-0 space-y-1.5">
          {data.download ? (
            <>
              <a href={EXTENSION_ZIP} download className="fd-btn-primary inline-flex h-9 items-center gap-1.5 px-3.5 text-xs">
                <Download className="h-4 w-4" />Download Captain for Chrome{data.extension_version ? ` ${data.extension_version}` : ''}
              </a>
              <p className="text-xs text-zinc-500">Unzip it somewhere it can stay — Chrome runs it from that folder.</p>
            </>
          ) : (
            <p className="text-xs text-zinc-400">This Flight Deck has no copy of the extension to hand out — ask whoever runs it.</p>
          )}
        </div>
      </li>
      <li className={step}>
        <span className={num}>2</span>
        <div className="min-w-0 space-y-1">
          <p>Open <CopyText text="chrome://extensions" label="the extensions address" /> in Chrome, turn on <b>Developer mode</b> (top right), press <b>Load unpacked</b> and pick the <code className="rounded bg-zinc-800 px-1 py-0.5 text-xs">captain-for-chrome</code> folder.</p>
          <p className="text-xs text-zinc-500">Pin the Captain icon from the puzzle-piece menu. Edge, Brave and Arc take it too.</p>
        </div>
      </li>
      <li className={step}>
        <span className={num}>3</span>
        <div className="min-w-0 space-y-1">
          <p>Click the Captain icon, type this Flight Deck's address and press <b>Connect this browser</b>:</p>
          <CopyText text={deck} label="this Flight Deck's address" />
          <p className="text-xs text-zinc-500">A tab opens here with a code — check it matches the panel and approve it.</p>
        </div>
      </li>
      <li className={step}>
        <span className={num}>4</span>
        <p className="min-w-0">In the panel's <b>Settings</b>, turn on <b>Agents can use this browser</b> — and <b>Work on every site</b> if agents should read pages there.</p>
      </li>
    </ol>
  )
}

function Waiting() {
  return (
    <p className="flex items-center gap-2 rounded-lg border border-dashed border-zinc-700 px-3 py-2 text-xs text-zinc-400" role="status">
      <Loader2 className="h-3.5 w-3.5 animate-spin" />Watching for your Chrome — this turns green once it connects.
    </p>
  )
}

function NotConnected({ data }: { data: ChromeActivity }) {
  if (data.state === 'offline') {
    return (
      <>
        <div className="flex gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400" />
          <div className="min-w-0">
            <div className="text-sm font-semibold text-zinc-100">Paired, but not connected right now</div>
            <p className="mt-1 text-xs leading-relaxed text-zinc-400">
              {data.paired.map((p) => p.name).join(', ')} {data.paired.length === 1 ? 'is' : 'are'} set up for your
              agents{data.paired[0]?.last_used_at ? ` (last used ${timeAgo(data.paired[0].last_used_at)})` : ''}, but
              none is talking to Flight Deck now. Until one is, agents work in the server's browser.
            </p>
          </div>
        </div>
        <Section title="Bring it back">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-zinc-300">
            <li>Open Chrome — it connects while Chrome runs.</li>
            <li>Click the Captain icon to open its panel; that wakes it if Chrome put it to sleep.</li>
            <li>In the panel's <b>Settings</b>, check <b>Agents can use this browser</b> is on — when it's off, and web searches and page reads are too, it doesn't connect at all.</li>
            <li>Still nothing? It may have been disconnected — set it up again below.</li>
          </ul>
          <Waiting />
        </Section>
        <details className="group rounded-xl border border-zinc-800 px-4 py-3">
          <summary className="cursor-pointer text-sm font-medium text-zinc-200">Set it up again, or in another browser</summary>
          <div className="mt-3"><GetIt data={data} /></div>
        </details>
      </>
    )
  }
  return (
    <>
      <div className="flex gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400" />
        <div className="min-w-0">
          <div className="text-sm font-semibold text-zinc-100">Not connected</div>
          <p className="mt-1 text-xs leading-relaxed text-zinc-400">
            Your agents browse with a browser on the server, which many sites block or keep signed out. Captain for
            Chrome lets them work in your Chrome instead — signed in as you, in tabs you can watch and stop.
          </p>
        </div>
      </div>
      <Section title="Get it — two minutes">
        <GetIt data={data} />
        <Waiting />
      </Section>
    </>
  )
}

function Connected({ data }: { data: ChromeActivity }) {
  return (
    <>
      {data.browsers.map((b) => (
        <div key={b.name} className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-600 dark:text-emerald-400" />
            <div className="min-w-0 flex-1 truncate text-sm font-semibold text-zinc-100">Connected · {b.name}</div>
            {b.version && <span className="shrink-0 text-[11px] text-zinc-500">extension {b.version}</span>}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5 text-[11px]">
            <span className={`rounded-full border px-2 py-0.5 ${b.agents_can_use
              ? 'border-emerald-500/30 bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
              : 'border-amber-500/30 bg-amber-500/15 text-amber-700 dark:text-amber-300'}`}>
              {b.agents_can_use ? 'Agents can use it' : "Agents can't use it"}
            </span>
            {b.searches.length > 0 && (
              <span className="rounded-full border border-zinc-700 px-2 py-0.5 text-zinc-300">Web searches on {b.searches.join(', ')}</span>
            )}
            {b.reads_pages && <span className="rounded-full border border-zinc-700 px-2 py-0.5 text-zinc-300">Reads pages for agents</span>}
          </div>
          {!b.agents_can_use && (
            <p className="mt-2 text-xs text-zinc-400">Turn on <b className="text-zinc-200">Settings → Agents can use this browser</b> in the Captain panel so agents can work in it.</p>
          )}
          {b.version_state === 'update' && (
            <p className="mt-2 text-xs text-amber-700 dark:text-amber-400">
              Version {data.extension_version} is ready —{' '}
              <a href={EXTENSION_ZIP} download className="underline hover:no-underline">download it</a>, unzip it over the folder
              this browser loaded, then press ↻ on Captain in chrome://extensions.
            </p>
          )}
        </div>
      ))}

      <Section title="Now" aside={data.now.length > 0 ? <span className="fd-count">{data.now.length}</span> : undefined}>
        {data.now.length === 0 ? (
          <p className="text-sm text-zinc-500">Nothing right now. Your agents use it when a task needs your signed-in browser.</p>
        ) : (
          <ul className="divide-y divide-zinc-800 rounded-xl border border-zinc-800">
            {[...data.now].reverse().map((a, i) => (
              <li key={`${a.slug}-${i}`} className="flex items-center gap-3 px-3 py-2.5">
                <span className={`h-2 w-2 shrink-0 animate-pulse rounded-full ${a.needs_you ? 'bg-[var(--fd-accent)]' : 'bg-amber-500 dark:bg-amber-400'}`} />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm text-zinc-100"><b className="font-semibold">{a.agent}</b> · {doingLine(a)}</div>
                  {a.needs_you && <div className="text-xs font-medium text-[var(--fd-accent-text)]">A robot check is open in your Chrome — answer it there and the agent carries on.</div>}
                </div>
                <span className="shrink-0 text-[11px] tabular-nums text-zinc-500">{a.for_s}s</span>
              </li>
            ))}
          </ul>
        )}
      </Section>

      {data.recent.length > 0 && (
        <Section title="Recently">
          <ul className="space-y-1">
            {data.recent.slice(0, 8).map((r, i) => (
              <li key={`${r.at}-${i}`} className="flex items-center gap-2.5 text-sm">
                {r.ok ? <Check className="h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                  : <XCircle className="h-3.5 w-3.5 shrink-0 text-red-500 dark:text-red-400" />}
                <span className="min-w-0 flex-1 truncate text-zinc-300">
                  <span className="text-zinc-100">{r.agent}</span> · {r.site && !r.did.includes(r.site) ? `${r.did} · ${r.site}` : r.did}
                  {!r.ok && <span className="text-zinc-500"> — didn't work</span>}
                </span>
                <span className="shrink-0 text-[11px] text-zinc-500">{timeAgo(r.at)}</span>
              </li>
            ))}
          </ul>
        </Section>
      )}
    </>
  )
}

/** Captain for Chrome: get it and how to use it, or what it's doing. */
export function ChromeExtensionDialog({ data, onClose, onConnections }: {
  data: ChromeActivity
  onClose: () => void
  /** Connections → Browser: every browser and Mac, recordings, disconnect. */
  onConnections: () => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const connected = data.state === 'connected'
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 wide:items-center wide:p-4" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-label="Captain for Chrome"
        className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 shadow-2xl wide:h-auto wide:max-h-[85vh] wide:max-w-2xl wide:rounded-2xl wide:border wide:border-zinc-800"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-zinc-800 px-4 wide:px-5">
          <h2 className="flex min-w-0 items-center gap-2 text-base font-semibold text-zinc-100">
            <Globe className="h-4 w-4 shrink-0 text-violet-400" />
            <span className="truncate">Captain for Chrome</span>
            {connected
              ? <span className="shrink-0 rounded-full border border-emerald-500/30 bg-emerald-500/15 px-2 text-[11px] font-semibold leading-5 text-emerald-700 dark:text-emerald-300">Connected</span>
              : <span className={`${PILL} px-2 text-[11px] leading-5`}>Not connected</span>}
          </h2>
          <button onClick={onClose} aria-label="Close" className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto overscroll-contain p-4 wide:p-5">
          {connected ? <Connected data={data} /> : <NotConnected data={data} />}
          {connected ? (
            <details className="rounded-xl border border-zinc-800 px-4 py-3">
              <summary className="cursor-pointer text-sm font-medium text-zinc-200">How to use it</summary>
              <div className="mt-3"><HowToUse /></div>
            </details>
          ) : (
            <Section title="How to use it"><HowToUse /></Section>
          )}
        </div>

        <div className="flex shrink-0 items-center justify-between gap-2 border-t border-zinc-800 px-4 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:px-5">
          <button onClick={onConnections} className="inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-xs font-medium text-violet-700 hover:bg-violet-500/10 dark:text-violet-300">
            Browsers, recordings, disconnect<ArrowRight className="h-3.5 w-3.5" />
          </button>
          <button onClick={onClose} className="fd-btn-quiet h-9 px-4 text-xs">Done</button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
