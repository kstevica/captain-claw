import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fdAuthHeaders, useAppCodeStore, type CodeAppSummary } from '../stores/appCodeStore'
import { useAuthStore } from '../stores/authStore'

/**
 * Sandboxed iframe shell for agent-coded apps.
 *
 * A "code-app" is an agent-authored directory under
 * ``~/.captain-claw-fd/apps/<slug>/`` containing ``backend.py`` +
 * ``frontend.html``. The backend is served by a per-app subprocess
 * managed by the Captain Claw runtime; this page only embeds the
 * frontend HTML in a sandboxed iframe and offers a tiny status
 * header.
 *
 * The iframe's ``src`` is ``/fd/code-apps/{slug}/page`` — that route
 * serves the on-disk ``frontend.html`` with a hardened CSP. The
 * iframe makes API calls relative to itself (``./api/...``), which
 * routes back through the FD proxy and into the subprocess.
 *
 * App list + selection live in :file:`stores/appCodeStore.ts` so the
 * sidebar can share state with this page. Per-app actions (restart,
 * logs) stay local because they're only useful while looking at the
 * iframe.
 */

interface CodeAppLogs {
  stderr: string[]
  stdout: string[]
  last_error: string | null
}

export function AppCodePage() {
  const apps = useAppCodeStore((s) => s.apps)
  const selectedSlug = useAppCodeStore((s) => s.selectedSlug)
  const loading = useAppCodeStore((s) => s.loading)
  const storeError = useAppCodeStore((s) => s.error)
  const iframeNonce = useAppCodeStore((s) => s.iframeNonce)
  const refresh = useAppCodeStore((s) => s.refresh)
  const selectSlug = useAppCodeStore((s) => s.selectSlug)
  const bumpIframe = useAppCodeStore((s) => s.bumpIframe)

  // The auth token is needed in the iframe's ``src`` (via ``?fd_token=``)
  // because the browser will not attach ``Authorization`` headers to an
  // iframe load. The ``/page`` route accepts the query param, sets a
  // path-scoped ``fd_app_token`` cookie on the response, and every
  // subsequent ``./api/*`` call inside the iframe authenticates via that
  // cookie automatically. Without this, the very first iframe load 401s.
  const token = useAuthStore((s) => s.token)
  const [error, setError] = useState<string | null>(null)
  const [logsOpen, setLogsOpen] = useState(false)
  const [logs, setLogs] = useState<CodeAppLogs | null>(null)
  const [logsLoading, setLogsLoading] = useState(false)
  const iframeRef = useRef<HTMLIFrameElement | null>(null)

  // Initial load + poll every 15s so the running/idle indicators stay
  // fresh without being chatty.
  useEffect(() => { refresh() }, [refresh])
  useEffect(() => {
    const id = setInterval(refresh, 15000)
    return () => clearInterval(id)
  }, [refresh])

  const selected: CodeAppSummary | null =
    apps.find((a) => a.slug === selectedSlug) ?? null

  // Build the iframe ``src`` with a one-time ``fd_token`` query so the
  // page handler can mint the path-scoped auth cookie. Memoized on
  // (slug, token, nonce) so a Reload / Restart re-emits a fresh URL,
  // which both invalidates the cookie's max-age and forces a full
  // page reload.
  const iframeSrc = useMemo(() => {
    if (!selected) return ''
    const base = `/fd/code-apps/${encodeURIComponent(selected.slug)}/page`
    if (!token) return base
    const sep = base.includes('?') ? '&' : '?'
    return `${base}${sep}fd_token=${encodeURIComponent(token)}`
  }, [selected, token, iframeNonce])

  const onSelect = useCallback((slug: string) => {
    selectSlug(slug)
    setLogsOpen(false)
    setError(null)
  }, [selectSlug])

  const onReload = useCallback(() => bumpIframe(), [bumpIframe])

  const onRestart = useCallback(async () => {
    if (!selectedSlug) return
    setError(null)
    try {
      const r = await fetch(`/fd/code-apps/${encodeURIComponent(selectedSlug)}/restart`, {
        method: 'POST',
        headers: fdAuthHeaders(),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      bumpIframe()
      refresh()
    } catch (e) {
      setError((e as Error).message || 'Restart failed')
    }
  }, [selectedSlug, bumpIframe, refresh])

  const onShowLogs = useCallback(async () => {
    if (!selectedSlug) return
    const willOpen = !logsOpen
    setLogsOpen(willOpen)
    if (!willOpen) return
    setLogsLoading(true)
    setError(null)
    try {
      const r = await fetch(`/fd/code-apps/${encodeURIComponent(selectedSlug)}/logs?n=200`, {
        headers: fdAuthHeaders(),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      setLogs(await r.json())
    } catch (e) {
      setError((e as Error).message || 'Failed to load logs')
    } finally {
      setLogsLoading(false)
    }
  }, [selectedSlug, logsOpen])

  if (loading && apps.length === 0) {
    return <CenterPanel>Loading code-apps…</CenterPanel>
  }

  if (apps.length === 0) {
    return (
      <CenterPanel>
        <h2 className="mb-2 text-lg font-semibold text-zinc-200">No code-apps yet</h2>
        <p className="text-sm text-zinc-500">
          Ask Captain Claw to scaffold one — for example, “build me a notes app”.
          The agent will write <code>backend.py</code> and <code>frontend.html</code>;
          they'll appear here automatically.
        </p>
        {(storeError || error) && (
          <p className="mt-3 text-xs text-red-400">{storeError || error}</p>
        )}
      </CenterPanel>
    )
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800 bg-zinc-950 px-4 py-2">
        <div className="flex items-center gap-3">
          <select
            value={selectedSlug ?? ''}
            onChange={(e) => onSelect(e.target.value)}
            className="cursor-pointer appearance-none rounded border border-zinc-800 bg-zinc-900 py-1 pl-2 pr-2 text-sm font-medium text-zinc-100 hover:border-zinc-700 focus:border-violet-500 focus:outline-none"
          >
            {apps.map((a) => {
              const name = String((a.manifest?.['name'] as string) || a.slug)
              return (
                <option key={a.slug} value={a.slug}>
                  {name} {a.running ? '●' : '○'}
                </option>
              )
            })}
          </select>
          {selected && (
            <span className="text-[11px] text-zinc-500">
              {selected.running ? (
                <>
                  pid {selected.pid}
                  {typeof selected.idle_seconds === 'number' && (
                    <> · idle {Math.round(selected.idle_seconds)}s</>
                  )}
                </>
              ) : (
                <>not running (will spawn on first request)</>
              )}
              {selected.has_error && (
                <span className="ml-2 rounded bg-red-900/40 px-1.5 py-0.5 text-red-300">last error</span>
              )}
              {!selected.has_backend && (
                <span className="ml-2 rounded bg-amber-900/40 px-1.5 py-0.5 text-amber-200">no backend.py</span>
              )}
            </span>
          )}
        </div>
        <div className="flex gap-1">
          <button
            onClick={onReload}
            className="rounded border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-zinc-800"
            title="Reload the iframe (no backend restart)"
          >
            Reload
          </button>
          <button
            onClick={onRestart}
            className="rounded border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-zinc-800"
            title="Restart the subprocess and reload the iframe"
          >
            Restart
          </button>
          <button
            onClick={onShowLogs}
            className={
              logsOpen
                ? 'rounded bg-violet-600/20 px-2 py-0.5 text-[11px] font-medium text-violet-300'
                : 'rounded border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-zinc-800'
            }
            title="Show subprocess stderr / stdout tail"
          >
            Logs
          </button>
        </div>
      </header>

      {(storeError || error) && (
        <div className="border-b border-red-900/40 bg-red-950/40 px-4 py-1.5 text-xs text-red-300">
          {storeError || error}
        </div>
      )}

      <main className="relative flex flex-1 overflow-hidden">
        <div className="flex-1">
          {selected ? (
            // The iframe is sandboxed with the minimum it needs to be
            // useful. ``allow-same-origin`` is required so cookie-based
            // auth flows to the proxy; without it, the iframe sees a
            // null origin and FastAPI rejects the request. The CSP on
            // the page response is the real security boundary.
            <iframe
              key={`${selected.slug}-${iframeNonce}`}
              ref={iframeRef}
              src={iframeSrc}
              sandbox="allow-scripts allow-forms allow-same-origin allow-popups allow-modals"
              className="h-full w-full border-0 bg-white"
              title={`${selected.slug} app`}
            />
          ) : (
            <CenterPanel>
              <p className="text-sm text-zinc-500">Pick an app from the dropdown.</p>
            </CenterPanel>
          )}
        </div>

        {logsOpen && selected && (
          <aside className="flex h-full w-[420px] shrink-0 flex-col border-l border-zinc-800 bg-zinc-950">
            <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-1.5">
              <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">
                Logs · {selected.slug}
              </span>
              <button
                onClick={onShowLogs}
                className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                title="Close"
              >
                ×
              </button>
            </div>
            <div className="flex-1 overflow-auto px-3 py-2 font-mono text-[11px] leading-snug text-zinc-300">
              {logsLoading && <p className="text-zinc-500">Loading…</p>}
              {!logsLoading && logs && (
                <>
                  {logs.last_error && (
                    <div className="mb-3 rounded border border-red-900/60 bg-red-950/40 p-2 text-red-300">
                      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-red-400">
                        Last error
                      </div>
                      <pre className="whitespace-pre-wrap break-words">{logs.last_error}</pre>
                    </div>
                  )}
                  {logs.stderr.length > 0 && (
                    <>
                      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-amber-400">
                        stderr ({logs.stderr.length})
                      </div>
                      <pre className="mb-3 whitespace-pre-wrap break-words text-amber-200/80">
                        {logs.stderr.join('\n')}
                      </pre>
                    </>
                  )}
                  {logs.stdout.length > 0 && (
                    <>
                      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                        stdout ({logs.stdout.length})
                      </div>
                      <pre className="whitespace-pre-wrap break-words text-zinc-400">
                        {logs.stdout.join('\n')}
                      </pre>
                    </>
                  )}
                  {logs.stderr.length === 0 && logs.stdout.length === 0 && !logs.last_error && (
                    <p className="text-zinc-500">No logs yet.</p>
                  )}
                </>
              )}
            </div>
          </aside>
        )}
      </main>
    </div>
  )
}

function CenterPanel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="max-w-md text-center">{children}</div>
    </div>
  )
}
