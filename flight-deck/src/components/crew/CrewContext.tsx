import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import { Database, Paperclip, Upload } from 'lucide-react'
import type { CrewStaged } from '../../services/crewApi'
import { AddDataPanel } from '../context/AddData'

// ── What goes with a task or a message ──
//
// Files from this computer straight away, or anything else through the shared Add data
// panel: something pasted, an agent's files, files from Flight Deck's folders, a stretch
// of a conversation with an agent. The panel keeps them like uploads, in the same place
// Crew does, so they go with the task or the message exactly as an attached file does.

/** `compact`: one paperclip opening a menu, for a message box; else two buttons in a row. */
export function AttachChoices({ onPick, onStaged, compact = false }: {
  onPick: (ev: ChangeEvent<HTMLInputElement>) => void
  onStaged: (files: CrewStaged[]) => void
  compact?: boolean
}) {
  const input = useRef<HTMLInputElement>(null)
  const [panel, setPanel] = useState(false)
  const [menu, setMenu] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!menu) return
    const away = (ev: MouseEvent) => { if (!menuRef.current?.contains(ev.target as Node)) setMenu(false) }
    const esc = (ev: KeyboardEvent) => { if (ev.key === 'Escape') setMenu(false) }
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', esc)
    return () => { document.removeEventListener('mousedown', away); document.removeEventListener('keydown', esc) }
  }, [menu])

  const choose = (which: 'upload' | 'panel') => {
    setMenu(false)
    if (which === 'upload') input.current?.click()
    else setPanel(true)
  }

  const item = 'flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm text-zinc-200 transition-colors hover:bg-zinc-800'

  return (
    <>
      <input ref={input} type="file" multiple onChange={onPick} className="hidden" aria-hidden="true" tabIndex={-1} />
      {compact ? (
        <div ref={menuRef} className="relative shrink-0">
          <button
            type="button" onClick={() => setMenu((m) => !m)} title="Attach" aria-label="Attach" aria-haspopup="menu" aria-expanded={menu}
            className="flex h-9 w-9 items-center justify-center rounded-xl text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
          >
            <Paperclip className="h-4 w-4" />
          </button>
          {menu && (
            <div role="menu" className="absolute bottom-full left-0 z-20 mb-2 w-72 rounded-xl border border-zinc-800 bg-zinc-950 p-1 shadow-xl">
              <button type="button" role="menuitem" onClick={() => choose('upload')} className={item}>
                <Upload className="mt-0.5 h-4 w-4 shrink-0 text-zinc-500" />Files from this computer
              </button>
              <button type="button" role="menuitem" onClick={() => choose('panel')} className={item}>
                <Database className="mt-0.5 h-4 w-4 shrink-0 text-violet-600 dark:text-violet-300" />
                <span>Add data…<span className="block text-[11px] text-zinc-500">Paste, agents' files, Flight Deck, a conversation</span></span>
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-1.5">
          <button type="button" onClick={() => choose('upload')}
            className="inline-flex items-center gap-1.5 rounded-full border border-zinc-800 px-3 py-1 text-xs font-medium text-zinc-400 transition-colors hover:border-zinc-700 hover:text-zinc-200">
            <Paperclip className="h-3.5 w-3.5" />Attach files
          </button>
          <button type="button" onClick={() => choose('panel')}
            className="inline-flex items-center gap-1.5 rounded-full border border-violet-500/30 bg-violet-500/[0.07] px-3 py-1 text-xs font-medium text-violet-700 transition-colors hover:border-violet-500/50 hover:bg-violet-500/10 dark:text-violet-300">
            <Database className="h-3.5 w-3.5" />Add data…
          </button>
          <span className="text-[11px] text-zinc-500">Paste, agents' files, Flight Deck, a conversation</span>
        </div>
      )}
      {panel && (
        <AddDataPanel title="Add data for the worker" start="paste"
          subtitle="It finds these in its working folder, told they're information, not instructions."
          action={(n) => (n ? `Attach ${n}` : 'Attach')}
          onDone={(items) => onStaged(items)}
          onClose={() => setPanel(false)} />
      )}
    </>
  )
}
