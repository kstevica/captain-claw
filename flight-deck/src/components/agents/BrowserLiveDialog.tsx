import { useCallback, useEffect, useRef, useState } from 'react'
import { CornerDownLeft, Hand, MonitorPlay, Send, Undo2 } from 'lucide-react'
import { api } from '../../services/fileLibraryApi'
import { Button, Dialog, Notice, inputCls } from '../console/parts'

// An agent's own browser (captain_claw/tools/browser_live.py): what its page
// looks like now, refreshed about once a second while this is open. Take
// control for a sign-in, a CAPTCHA or a step it's stuck on — its own browser
// actions wait meanwhile — then hand it back; it hears who had control and
// where the page is now. Frames are never kept.

interface Live {
  alive: boolean
  held: boolean
  since: number | null
  by: string
  url?: string
  title?: string
  width?: number
  height?: number
  frame?: string
  error?: string
}

const KEYS = ['Enter', 'Tab', 'Backspace', 'Escape'] as const

export function BrowserLiveDialog({ agent, onClose }: { agent: { slug: string; name: string }; onClose: () => void }) {
  const base = `/fd/agents/${encodeURIComponent(agent.slug)}/browser`
  const [live, setLive] = useState<Live | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [text, setText] = useState('')
  const img = useRef<HTMLImageElement>(null)
  const alive = useRef(true)

  const load = useCallback(async () => {
    try {
      setLive(await api<Live>(`${base}/live`))
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    }
  }, [base])

  useEffect(() => {
    alive.current = true
    let timer = 0
    const tick = async () => {
      if (!alive.current) return
      if (document.visibilityState === 'visible') await load()
      timer = window.setTimeout(tick, 1000)
    }
    void tick()
    return () => { alive.current = false; window.clearTimeout(timer) }
  }, [load])

  const control = async (take: boolean) => {
    setBusy(true)
    try {
      await api(`${base}/control`, { method: 'POST', body: JSON.stringify({ take }) })
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setBusy(false)
    }
  }

  // Close with control still held: hand it back, so the agent isn't left waiting.
  const close = async () => {
    if (live?.held) await control(false)
    onClose()
  }

  const send = async (body: Record<string, unknown>) => {
    try {
      await api(`${base}/input`, { method: 'POST', body: JSON.stringify(body) })
      window.setTimeout(load, 250)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    }
  }

  const click = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!live?.held || !img.current || !live.width || !live.height) return
    const box = img.current.getBoundingClientRect()
    const x = ((e.clientX - box.left) / box.width) * live.width
    const y = ((e.clientY - box.top) / box.height) * live.height
    void send({ kind: 'click', x: Math.round(x), y: Math.round(y) })
  }

  const held = !!live?.held
  return (
    <Dialog title={`${agent.name}'s browser`} icon={MonitorPlay} onClose={close} large footer={
      held ? (
        <Button variant="primary" icon={Undo2} busy={busy} onClick={() => control(false)}>Hand it back</Button>
      ) : (
        <Button variant="secondary" icon={Hand} busy={busy} disabled={!live?.alive} onClick={() => control(true)}>Take control</Button>
      )
    }>
      <div className="space-y-3">
        {error && <Notice tone="red">{error}</Notice>}
        {!live ? (
          <p className="text-sm text-zinc-500">Looking…</p>
        ) : !live.alive ? (
          <p className="text-sm text-zinc-500">Its browser isn't open. It shows here once {agent.name} opens a page.</p>
        ) : (
          <>
            <p className="truncate text-xs text-zinc-500" title={live.url}>
              {live.title ? <span className="text-zinc-300">{live.title}</span> : null}{live.title ? ' · ' : ''}{live.url}
            </p>
            {held ? (
              <Notice tone="amber">
                You have control{live.by ? ` (${live.by})` : ''}: click on the page, type below, then hand it back.
                {` ${agent.name}`}'s own browser steps wait until you do.
              </Notice>
            ) : (
              <p className="text-xs text-zinc-500">Take control to click and type in it yourself — for a sign-in, a check, or a step it's stuck on.</p>
            )}
            {live.frame ? (
              <img ref={img} alt="The agent's browser page" src={`data:image/jpeg;base64,${live.frame}`}
                onClick={click}
                onWheel={(e) => { if (held) void send({ kind: 'scroll', dy: e.deltaY }) }}
                className={`w-full rounded-lg border border-zinc-800 ${held ? 'cursor-crosshair ring-2 ring-amber-500/50' : ''}`} />
            ) : (
              <p className="text-xs text-zinc-500">{live.error || 'No picture right now.'}</p>
            )}
            {held && (
              <div className="flex flex-wrap items-center gap-2">
                <input className={`${inputCls} min-w-0 flex-1`} placeholder="Type into the page…" value={text}
                  autoComplete="off" onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && text) { e.preventDefault(); void send({ kind: 'type', text }); setText('') } }} />
                <Button variant="secondary" icon={Send} disabled={!text} onClick={() => { void send({ kind: 'type', text }); setText('') }}>Type</Button>
                {KEYS.map((k) => (
                  <Button key={k} variant="ghost" icon={k === 'Enter' ? CornerDownLeft : undefined} onClick={() => send({ kind: 'key', key: k })}>{k}</Button>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </Dialog>
  )
}
