import { useState, useRef, useEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { ExternalLink, ChevronDown, Monitor } from 'lucide-react'

export function OpenDropdown({ host, port, auth }: { host: string; port: number; auth?: string }) {
  const [open, setOpen] = useState(false)
  const btnRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ top: 0, left: 0 })

  const MENU_W = 140

  const updatePos = useCallback(() => {
    if (!btnRef.current) return
    const r = btnRef.current.getBoundingClientRect()
    // Right-align the menu to the button, clamped to the viewport.
    const left = Math.max(8, Math.min(r.right - MENU_W, window.innerWidth - MENU_W - 8))
    setPos({ top: r.bottom + 4, left })
  }, [])

  useEffect(() => {
    if (!open) return
    updatePos()
    const handler = (e: MouseEvent) => {
      const t = e.target as Node
      if (menuRef.current?.contains(t) || btnRef.current?.contains(t)) return
      setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    window.addEventListener('scroll', updatePos, true)
    window.addEventListener('resize', updatePos)
    return () => {
      document.removeEventListener('mousedown', handler)
      window.removeEventListener('scroll', updatePos, true)
      window.removeEventListener('resize', updatePos)
    }
  }, [open, updatePos])

  const qs = auth ? `?token=${encodeURIComponent(auth)}` : ''
  const base = `http://${host}:${port}`

  return (
    <>
      <button
        ref={btnRef}
        onClick={() => { updatePos(); setOpen(!open) }}
        className="flex items-center gap-1 rounded-lg px-2 py-0.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
      >
        <ExternalLink className="h-3 w-3" />
        Open
        <ChevronDown className="h-3 w-3" />
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          className="fixed z-[100] min-w-[140px] rounded-lg border border-zinc-700 bg-zinc-900 py-1 shadow-xl shadow-black/40"
          style={{ top: pos.top, left: pos.left }}
        >
          <button
            onClick={() => { window.open(`${base}/chat${qs}`, '_blank'); setOpen(false) }}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
          >
            <ExternalLink className="h-3 w-3" />
            Agent
          </button>
          <button
            onClick={() => { window.open(`${base}/computer${qs}`, '_blank'); setOpen(false) }}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
          >
            <Monitor className="h-3 w-3" />
            Computer
          </button>
        </div>,
        document.body,
      )}
    </>
  )
}
