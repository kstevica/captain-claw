import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, Loader2, Power, RefreshCw, TerminalSquare } from 'lucide-react'
import { FitAddon } from '@xterm/addon-fit'
import { Terminal } from '@xterm/xterm'
import '@xterm/xterm/css/xterm.css'
import { libraryIconBtn } from '../library/libraryParts'
import { crewApi, type CrewTerminalTarget } from '../../services/crewApi'
import { refreshAccessToken } from '../../stores/authStore'
import { ErrorLine, quietBtn } from './crewParts'

// ── A shell in a project or a task's folder ──
//
// The shell lives on the deck and outlives this page: leaving, or losing the
// connection, comes back to the same one with what it printed meanwhile. Loaded on
// its own, since nobody who never opens one should carry a terminal in the bundle.

type Link = 'connecting' | 'open' | 'lost' | 'ended' | 'refused'

/** The keys a phone's keyboard hasn't got. */
const PHONE_KEYS: [string, string][] = [['Tab', '\t'], ['Esc', '\x1b'], ['Ctrl-C', '\x03'], ['Ctrl-D', '\x04'], ['↑', '\x1b[A'], ['↓', '\x1b[B']]

export default function CrewTerminal({ target, title, onBack }: { target: CrewTerminalTarget; title: string; onBack: () => void }) {
  const host = useRef<HTMLDivElement>(null)
  const socket = useRef<WebSocket | null>(null)
  const [link, setLink] = useState<Link>('connecting')
  const [why, setWhy] = useState('')
  const [attempt, setAttempt] = useState(0)

  const send = (event: object) => {
    if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify(event))
  }

  useEffect(() => {
    if (!host.current) return
    let refreshed = false
    const term = new Terminal({
      cursorBlink: true, fontSize: 13, scrollback: 10000, allowProposedApi: false,
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
      theme: { background: '#09090b', foreground: '#e4e4e7', cursor: '#a78bfa', selectionBackground: '#5b21b680' },
    })
    const fit = new FitAddon()
    term.loadAddon(fit)
    term.open(host.current)
    fit.fit()
    setLink('connecting')
    setWhy('')

    const ws = new WebSocket(crewApi.terminalUrl(target, term.cols, term.rows))
    ws.binaryType = 'arraybuffer'
    socket.current = ws
    ws.onopen = () => { setLink('open'); term.focus() }
    ws.onmessage = (event) => {
      if (typeof event.data !== 'string') { term.write(new Uint8Array(event.data as ArrayBuffer)); return }
      try {
        if (JSON.parse(event.data).type === 'exit') setLink('ended')
      } catch { /* only exit comes as text */ }
    }
    ws.onclose = (event) => {
      if (socket.current !== ws) return
      if (event.code === 4001 && !refreshed) {
        // Its sign-in ran out while the page sat open: once more, with a fresh one.
        refreshed = true
        void refreshAccessToken().then((ok) => { if (ok) setAttempt((n) => n + 1); else { setLink('refused'); setWhy('Sign in again to use the terminal.') } })
        return
      }
      if (event.code >= 4000) { setLink('refused'); setWhy(event.reason || 'The terminal was refused.'); return }
      setLink((old) => (old === 'ended' ? old : 'lost'))
    }
    const typed = term.onData((data) => send({ type: 'input', data }))
    const fitted = new ResizeObserver(() => {
      fit.fit()
      send({ type: 'resize', cols: term.cols, rows: term.rows })
    })
    fitted.observe(host.current)
    return () => {
      socket.current = null
      fitted.disconnect()
      typed.dispose()
      ws.close()
      term.dispose()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- a target is a new shell; send reads the ref
  }, [target.env, target.task, attempt])

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3">
        <button onClick={onBack} className={libraryIconBtn} aria-label="Back"><ArrowLeft className="h-4 w-4" /></button>
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/15 text-violet-700 dark:text-violet-300"><TerminalSquare className="h-[18px] w-[18px]" /></span>
        <div className="mr-auto min-w-0">
          <h3 className="truncate text-sm font-semibold text-zinc-100">{title}</h3>
          <p className="truncate text-xs text-zinc-500">
            {link === 'connecting' && 'Connecting…'}
            {link === 'open' && 'It keeps running if you leave this page.'}
            {link === 'lost' && 'The connection dropped. The shell is still there.'}
            {link === 'ended' && 'This shell has ended.'}
            {link === 'refused' && 'Not available.'}
          </p>
        </div>
        {link === 'connecting' && <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />}
        {(link === 'lost' || link === 'ended') && (
          <button className={quietBtn} onClick={() => setAttempt((n) => n + 1)}><RefreshCw className="h-4 w-4" /> {link === 'ended' ? 'New shell' : 'Reconnect'}</button>
        )}
        {link === 'open' && (
          <button className={quietBtn} title="End this shell and whatever it's running" onClick={() => send({ type: 'close' })}><Power className="h-4 w-4" /> End</button>
        )}
      </div>
      {why && <div className="shrink-0 p-3"><ErrorLine text={why} /></div>}
      <div ref={host} className="min-h-0 flex-1 bg-[#09090b] p-2" />
      <div className="flex shrink-0 gap-1.5 overflow-x-auto border-t border-zinc-800 bg-zinc-900/60 p-2 wide:hidden" aria-label="Keys">
        {PHONE_KEYS.map(([label, data]) => (
          <button key={label} onClick={() => send({ type: 'input', data })} className="shrink-0 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-200">{label}</button>
        ))}
      </div>
    </div>
  )
}
