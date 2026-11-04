import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, Check, Copy, Download, Globe, Loader2, Monitor, Trash2 } from 'lucide-react'
import { EXTENSION_ZIP, MAC_APP_ZIP, listConnections, removeBrowser, type ConnectedBrowser } from '../../services/browserClip'

// Connections → Browser: Captain for Chrome — how to put it in Chrome — and
// Captain for Mac (the same for any app on a Mac), with the browsers and Macs
// connected to this account (each holds a token of its own).

export interface BrowsersStatus {
  connected: number
}

function when(ts: string | null): string {
  if (!ts) return 'never'
  try { return new Date(ts).toLocaleString() } catch { return ts }
}

const ACTIONS: [string, string][] = [
  ['Handle it', 'the agent decides what is useful'],
  ['Summarize', 'gist, key points, what to act on'],
  ['Save to space', 'the whole page and its files, with notes'],
  ['Grab files', 'the documents it links to, downloaded and named'],
  ['Extract data', 'tables, prices, people and dates as CSV'],
  ['Dig deeper · Fact-check', 'beyond the page, with sources'],
  ['Draft a reply · Make tasks', 'what to send, what to do next'],
  ['Watch it · Remember', 'a daily check; facts kept for later'],
]

export default function BrowserConnection({ active = true, onStatus }: { active?: boolean; onStatus?: (s: BrowsersStatus) => void }) {
  const [browsers, setBrowsers] = useState<ConnectedBrowser[] | null>(null)
  const [macApp, setMacApp] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)
  const [removing, setRemoving] = useState('')
  const deck = window.location.origin

  const load = useCallback(async () => {
    try {
      const { browsers: list, macApp: hasMac } = await listConnections()
      setBrowsers(list)
      setMacApp(hasMac)
      setError('')
      onStatus?.({ connected: list.length })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load browsers')
    }
  }, [onStatus])

  // A browser connects from another tab: the list follows while it shows.
  useEffect(() => {
    load()
    if (!active) return
    const t = setInterval(load, 10000)
    return () => clearInterval(t)
  }, [active, load])

  const copy = () => {
    navigator.clipboard?.writeText(deck).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }).catch(() => {})
  }

  const remove = async (b: ConnectedBrowser) => {
    const what = b.kind === 'desktop' ? 'send what is on its screen' : 'send pages'
    if (!confirm(`Disconnect “${b.name}”? It will need to be connected again to ${what}.`)) return
    setRemoving(b.id)
    try {
      await removeBrowser(b.id)
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not disconnect')
    }
    setRemoving('')
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
      <div className="flex items-center gap-3 px-5 py-4 border-b border-zinc-800">
        <div className="h-10 w-10 rounded-md bg-gradient-to-br from-violet-500/20 to-sky-500/20 border border-zinc-800 flex items-center justify-center">
          <Globe className="h-5 w-5 text-zinc-200" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-zinc-100">Captain for Chrome</h3>
            {browsers && browsers.length > 0 && (
              <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30">
                {browsers.length} connected
              </span>
            )}
          </div>
          <p className="text-xs text-zinc-500 mt-0.5">
            Send the page you're on to one of your agents — read in your own browser, signed in, with the files it links to.
            It lands in your <b className="text-zinc-300">web-clips</b> shared folder and in the agent's chat.
          </p>
        </div>
      </div>

      <div className="px-5 py-4 space-y-5">
        {error && (
          <div className="flex items-center gap-2 text-xs text-red-500 dark:text-red-400"><AlertCircle className="h-3.5 w-3.5" />{error}</div>
        )}

        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Put it in Chrome</h4>
          <ol className="space-y-2 text-sm text-zinc-300 list-decimal pl-5">
            <li>
              <a
                href={EXTENSION_ZIP}
                download
                className="inline-flex items-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-100 hover:bg-zinc-700"
              >
                <Download className="h-3.5 w-3.5" /> Download the extension
              </a>
              <span className="text-zinc-500"> and unzip it.</span>
            </li>
            <li>
              Open <code className="rounded bg-zinc-800 px-1 py-0.5 text-xs">chrome://extensions</code>, turn on <b>Developer mode</b>,
              press <b>Load unpacked</b> and pick the <code className="rounded bg-zinc-800 px-1 py-0.5 text-xs">captain-for-chrome</code> folder.
              Pin the Captain icon. (Edge, Brave and Arc take it too.)
            </li>
            <li>
              Click the icon, type this deck's address and approve the code that shows up here:
              <span className="mt-1.5 flex items-center gap-2">
                <code className="rounded bg-zinc-800 px-2 py-1 text-xs text-zinc-100">{deck}</code>
                <button onClick={copy} className="inline-flex items-center gap-1 text-xs text-violet-700 dark:text-violet-300 hover:underline">
                  {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}{copied ? 'Copied' : 'Copy'}
                </button>
              </span>
            </li>
          </ol>
          <p className="text-xs text-zinc-500">Open it with the toolbar icon, <b>Alt+Shift+C</b>, or right-click a page, a selection, a link or a picture.</p>
        </section>

        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">What it asks your agent</h4>
          <ul className="grid gap-x-4 gap-y-1 sm:grid-cols-2 text-xs">
            {ACTIONS.map(([name, what]) => (
              <li key={name} className="text-zinc-400"><span className="text-zinc-200 font-medium">{name}</span> — {what}</li>
            ))}
          </ul>
          <p className="text-xs text-zinc-500">
            Each can be read as an investor, a buyer, a lawyer, a skeptic, a competitor, an engineer, a newcomer, your team — or your own angle.
          </p>
        </section>

        <section className="space-y-2 rounded-md border border-zinc-800 bg-zinc-950/40 px-4 py-3">
          <div className="flex items-center gap-2">
            <Monitor className="h-4 w-4 text-zinc-300" />
            <h4 className="text-sm font-semibold text-zinc-100">Captain for Mac</h4>
            <span className="text-[10px] uppercase tracking-wider text-zinc-500">any app, not just the browser</span>
          </div>
          <p className="text-xs text-zinc-400">
            A menu-bar app: hold <b className="text-zinc-200">fn</b> for a few seconds (or click its icon) and it reads what's on screen —
            the app's text through Accessibility, your selection first, what's under the pointer, a screenshot if you turn it on — and
            sends it, with what you type or say, to one of your agents. It lands in your <b className="text-zinc-300">screen-clips</b> shared
            folder and in the agent's chat; the answer can be pasted back where you were.
          </p>
          <ol className="space-y-1.5 text-sm text-zinc-300 list-decimal pl-5">
            <li>
              {macApp ? (
                <>
                  <a
                    href={MAC_APP_ZIP}
                    download
                    className="inline-flex items-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-100 hover:bg-zinc-700"
                  >
                    <Download className="h-3.5 w-3.5" /> Download Captain for Mac
                  </a>
                  <span className="text-zinc-500"> — unzip it and move it to Applications. It isn't notarized yet: if macOS won't open it, go to System Settings → Privacy &amp; Security and press <b>Open Anyway</b>.</span>
                </>
              ) : (
                <span className="text-zinc-400">
                  This deck has no build of it yet — on a Mac, run{' '}
                  <code className="rounded bg-zinc-800 px-1 py-0.5 text-xs">captain-for-mac/build.sh</code> in the Captain checkout.
                </span>
              )}
            </li>
            <li>Open it, allow <b>Accessibility</b> (and Screen Recording or the microphone if you want them), type this deck's address, press <b>Connect</b> and approve the code here.</li>
          </ol>
        </section>

        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Connected browsers and Macs</h4>
          {browsers === null ? (
            <div className="flex items-center gap-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" />Loading…</div>
          ) : browsers.length === 0 ? (
            <p className="text-xs text-zinc-500">None yet.</p>
          ) : (
            <ul className="divide-y divide-zinc-800 rounded-md border border-zinc-800">
              {browsers.map((b) => (
                <li key={b.id} className="flex items-center gap-3 px-3 py-2">
                  {b.kind === 'desktop'
                    ? <Monitor className="h-4 w-4 text-zinc-500 shrink-0" />
                    : <Globe className="h-4 w-4 text-zinc-500 shrink-0" />}
                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-zinc-200 truncate">{b.name}</div>
                    <div className="text-[11px] text-zinc-500">Connected {when(b.created_at)} · last used {when(b.last_used_at)}</div>
                  </div>
                  <button
                    onClick={() => remove(b)}
                    disabled={removing === b.id}
                    className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-red-600 dark:text-red-400 hover:bg-red-500/10 disabled:opacity-50"
                  >
                    {removing === b.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                    Disconnect
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  )
}
