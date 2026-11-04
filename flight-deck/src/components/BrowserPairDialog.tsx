import { useEffect, useState } from 'react'
import { AlertTriangle, Check, Globe, Loader2, Monitor, X } from 'lucide-react'
import { decidePair, pairCodeFromLocation, pairInfo, PAIR_PATH, type PairInfo } from '../services/browserClip'

// <deck>/connect-browser#<code>: Captain for Chrome (or Captain for Mac)
// asks to be connected to this account. The person checks the code matches the one in the extension
// and approves (or turns it away). Read at load, so a sign-in in between
// doesn't lose it.

const CODE = pairCodeFromLocation()

function leave() {
  if (window.location.pathname === PAIR_PATH) window.history.replaceState(null, '', '/')
}

export default function BrowserPairDialog() {
  const [open, setOpen] = useState(!!CODE)
  const [info, setInfo] = useState<PairInfo | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState<'approved' | 'denied' | ''>('')

  useEffect(() => {
    if (!CODE) return
    pairInfo(CODE).then(setInfo).catch((e) => setError(e instanceof Error ? e.message : 'Could not load the request'))
  }, [])

  if (!open) return null

  const mac = info?.kind === 'desktop'
  const product = mac ? 'Captain for Mac' : 'Captain for Chrome'

  const close = () => {
    setOpen(false)
    leave()
  }

  const decide = async (approve: boolean) => {
    setBusy(true)
    setError('')
    try {
      const r = await decidePair(CODE, approve)
      setDone(r.status === 'approved' ? 'approved' : 'denied')
      leave()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not answer the request')
    }
    setBusy(false)
  }

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal="true" aria-labelledby="browser-pair-title">
      <div className="w-full max-w-md rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl">
        <div className="flex items-center gap-3 border-b border-zinc-800 px-5 py-4">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-violet-500/10 text-violet-600 dark:text-violet-400">
            {mac ? <Monitor className="h-5 w-5" /> : <Globe className="h-5 w-5" />}
          </div>
          <h2 id="browser-pair-title" className="flex-1 text-base font-semibold text-zinc-100">Connect {product}?</h2>
          <button onClick={close} className="rounded-md p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-4 px-5 py-4 text-sm">
          {done === 'approved' ? (
            <div className="flex items-start gap-3">
              <Check className="mt-0.5 h-5 w-5 text-emerald-500" />
              <div>
                <div className="font-medium text-zinc-100">Connected</div>
                <p className="text-zinc-400">
                  {mac
                    ? 'Go back to Captain for Mac — it is ready: hold fn or click its menu-bar icon. You can close this tab.'
                    : 'Go back to the page you were on — Captain is ready in the side panel. You can close this tab.'}
                </p>
              </div>
            </div>
          ) : done === 'denied' ? (
            <p className="text-zinc-400">Turned away — {mac ? 'that Mac' : 'that browser'} was not connected.</p>
          ) : error && !info ? (
            <div className="flex items-start gap-2 text-red-600 dark:text-red-400"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{error}</div>
          ) : !info ? (
            <div className="flex items-center gap-2 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" />Loading…</div>
          ) : (
            <>
              <p className="text-zinc-300">
                {mac ? (
                  <>
                    <b className="text-zinc-100">{info.browser}</b> wants to send what's on that Mac's screen to your agents — the text of
                    the app you're in, your selection and, if you turn it on, a screenshot. It will be able to list your agents, send them
                    that, and read their answers — as you.
                  </>
                ) : (
                  <>
                    <b className="text-zinc-100">{info.browser}</b> wants to send pages to your agents. It will be able to list your agents,
                    send them pages and files, and read their answers — as you.
                  </>
                )}
              </p>
              <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 px-4 py-3 text-center">
                <div className="text-[11px] uppercase tracking-wider text-zinc-500">{mac ? 'The app shows' : 'The extension shows'}</div>
                <div className="mt-1 font-mono text-2xl font-semibold tracking-[0.2em] text-violet-700 dark:text-violet-300">{info.code}</div>
              </div>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
                <dt className="text-zinc-500">Asked from</dt>
                <dd className="text-zinc-300">{info.ip || 'unknown'}{info.same_network ? ' (same network as you)' : ''}</dd>
                <dt className="text-zinc-500">When</dt>
                <dd className="text-zinc-300">{new Date(info.created_at).toLocaleTimeString()}</dd>
              </dl>
              {!info.same_network && (
                <div className="flex items-start gap-2 rounded-md bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  This request came from another network than yours. Only approve it if you just pressed Connect {mac ? 'in Captain for Mac' : 'in your own browser'}.
                </div>
              )}
              {error && <div className="text-xs text-red-600 dark:text-red-400">{error}</div>}
            </>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t border-zinc-800 px-5 py-3">
          {done || (error && !info) ? (
            <button onClick={close} className="rounded-md bg-zinc-800 px-3 py-1.5 text-sm text-zinc-100 hover:bg-zinc-700">Close</button>
          ) : (
            <>
              <button onClick={() => decide(false)} disabled={busy || !info} className="rounded-md px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800 disabled:opacity-50">
                Don't connect
              </button>
              <button onClick={() => decide(true)} disabled={busy || !info} className="inline-flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50">
                {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}Connect
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
