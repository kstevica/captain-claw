import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, CheckCircle2, Loader2, Search } from 'lucide-react'
import { getChromeSearch, setChromeSearch, type ChromeSearchStatus } from '../../services/browserClip'

// Profile → Personal → Web searches: the person's agents search the web in
// their own Chrome (Captain for Chrome, a background tab) on the engine picked
// here, instead of the deck's search API. Off by default; the extension asks
// once, in its panel, to read that engine's pages. When their Chrome is closed
// or the engine asks for a robot check, the deck's own engine answers.

export function WebSearchSettings() {
  const [st, setSt] = useState<ChromeSearchStatus | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    try {
      setSt(await getChromeSearch())
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load')
    }
  }, [])

  // The extension allows it in another window: the status follows while this shows.
  useEffect(() => {
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [load])

  const pick = async (engine: string) => {
    setSaving(true)
    try {
      setSt(await setChromeSearch(engine))
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save')
    }
    setSaving(false)
  }

  const engine = st?.engines.find((e) => e.id === st.engine)
  const ready = st?.browsers.filter((b) => b.ready) ?? []
  const waiting = st?.browsers.filter((b) => !b.ready) ?? []

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-5 space-y-3">
      <div className="flex items-start gap-2">
        <Search className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-zinc-200">Web searches</h3>
          <p className="mt-0.5 text-xs text-zinc-500">
            Your agents can search the web in your own Chrome — on the engine you pick, in a background tab that closes
            once the results are in — instead of through Flight Deck's search service. Needs Captain for Chrome
            (Connections → Browser). When your Chrome is closed, Flight Deck's own search answers as before.
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
            <span>Search through my Chrome with</span>
            <select
              value={st.engine}
              disabled={saving}
              onChange={(e) => void pick(e.target.value)}
              className="rounded-md border border-zinc-700 bg-zinc-800 px-2 py-1 text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-violet-500/40"
            >
              <option value="">Off — Flight Deck's search service</option>
              {st.engines.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
            </select>
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
          </label>

          {engine && (
            <div className="space-y-1.5 text-xs">
              {!st.extension_set_up ? (
                <p className="text-amber-700 dark:text-amber-400">
                  Captain for Chrome isn't connected to this deck yet — put it in Chrome from Connections → Browser.
                  Until then your agents search as before.
                </p>
              ) : ready.length > 0 ? (
                <p className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-400">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Searching on {engine.name} in {ready.map((b) => b.browser).join(', ')}.
                </p>
              ) : (
                <p className="text-amber-700 dark:text-amber-400">
                  {waiting.length > 0
                    ? `Waiting for your Chrome: open Captain for Chrome and press Allow on its “${engine.name}” card (or Settings → Web searches).`
                    : `No Chrome of yours is open right now. Open Captain for Chrome once and press Allow on its “${engine.name}” card; after that, searches run whenever Chrome is open.`}
                </p>
              )}
              <p className="text-zinc-500">
                Searches show in that browser's history{engine.id === 'google' || engine.id === 'bing' ? `, and in your ${engine.name} activity if you're signed in there` : ''}.
                If {engine.name} asks to confirm you're not a robot, answer it there once — meanwhile Flight Deck's search answers.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  )
}
