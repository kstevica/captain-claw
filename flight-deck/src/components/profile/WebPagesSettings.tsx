import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, CheckCircle2, Globe, Loader2 } from 'lucide-react'
import { getChromeFetch, setChromeFetch, type ChromeFetchMode, type ChromeFetchStatus } from '../../services/browserClip'

// Profile → Personal → Web pages: the person's agents read web pages in their
// own Chrome (Captain for Chrome, a background tab, signed in as them) — for
// sites that turn Flight Deck away (LinkedIn, Facebook, sign-in walls, bot
// checks), or for every page. Off by default; optionally only listed sites.

const MODES: { id: ChromeFetchMode; label: string }[] = [
  { id: '', label: "Off — Flight Deck reads pages itself" },
  { id: 'blocked', label: 'Only pages that turn Flight Deck away' },
  { id: 'always', label: 'Every page' },
]

function sitesText(sites: string[]): string {
  return sites.join(', ')
}

function parseSites(text: string): string[] {
  return text.split(/[\s,;]+/).map((s) => s.trim()).filter(Boolean)
}

export function WebPagesSettings() {
  const [st, setSt] = useState<ChromeFetchStatus | null>(null)
  const [sites, setSites] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(async (keepSites = true) => {
    try {
      const got = await getChromeFetch()
      setSt(got)
      if (!keepSites) setSites(sitesText(got.sites))
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load')
    }
  }, [])

  useEffect(() => {
    load(false)
    const t = setInterval(() => load(true), 15000)
    return () => clearInterval(t)
  }, [load])

  const save = async (mode: ChromeFetchMode, siteText: string) => {
    setSaving(true)
    try {
      const got = await setChromeFetch(mode, parseSites(siteText))
      setSt(got)
      setSites(sitesText(got.sites))
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save')
    }
    setSaving(false)
  }

  const ready = st?.browsers.filter((b) => b.ready) ?? []
  const sitesDirty = !!st && sitesText(parseSites(sites)) !== sitesText(st.sites)

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-5 space-y-3">
      <div className="flex items-start gap-2">
        <Globe className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-zinc-200">Web pages</h3>
          <p className="mt-0.5 text-xs text-zinc-500">
            Your agents can read web pages in your own Chrome, signed in as you — so LinkedIn, Facebook and other sites that
            turn Flight Deck away open for them. The page opens in a background tab that closes once it's read. Needs Captain
            for Chrome with “Work on every site”. When your Chrome is closed, Flight Deck reads pages itself as before.
          </p>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-xs text-red-500 dark:text-red-400"><AlertCircle className="h-3.5 w-3.5" />{error}</div>
      )}

      {!st ? (
        !error && <div className="flex items-center gap-2 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" />Loading…</div>
      ) : (
        <>
          <label className="flex flex-wrap items-center gap-2 text-sm text-zinc-300">
            <span>Read pages through my Chrome</span>
            <select
              value={st.mode}
              disabled={saving}
              onChange={(e) => void save(e.target.value as ChromeFetchMode, sites)}
              className="rounded-md border border-zinc-700 bg-zinc-800 px-2 py-1 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-violet-500/40"
            >
              {MODES.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
          </label>

          {st.mode && (
            <div className="space-y-2 text-xs">
              <div className="space-y-1">
                <label className="flex flex-wrap items-center gap-2 text-zinc-400">
                  <span>Only these sites</span>
                  <input
                    value={sites}
                    onChange={(e) => setSites(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && sitesDirty) void save(st.mode, sites) }}
                    placeholder="any site — or e.g. linkedin.com, facebook.com"
                    className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-800 px-2 py-1 text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-violet-500/40"
                  />
                  {sitesDirty && (
                    <button
                      onClick={() => void save(st.mode, sites)}
                      disabled={saving}
                      className="rounded-md bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50"
                    >
                      Save
                    </button>
                  )}
                </label>
                <p className="text-zinc-500">A site covers its subdomains. Empty = any site.</p>
              </div>

              {!st.extension_set_up ? (
                <p className="text-amber-700 dark:text-amber-400">
                  Captain for Chrome isn't connected to this deck yet — put it in Chrome from Connections → Browser.
                </p>
              ) : ready.length > 0 ? (
                <p className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-400">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Reading pages in {ready.map((b) => b.browser).join(', ')}.
                </p>
              ) : (
                <p className="text-amber-700 dark:text-amber-400">
                  {st.browsers.length > 0
                    ? 'Waiting for your Chrome: turn on “Work on every site” in Captain for Chrome (or press Allow on its “Web pages” card).'
                    : 'No Chrome of yours is open right now. Open Captain for Chrome once and allow it; after that, pages are read whenever Chrome is open.'}
                </p>
              )}
              <p className="text-zinc-500">
                Your agents see what you see on those sites, as you. A page an agent reads can try to send it to others — list
                only the sites you need. Never Flight Deck itself or addresses on your own network.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  )
}
