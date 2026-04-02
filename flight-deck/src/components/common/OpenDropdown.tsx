import { useState, useRef, useEffect } from 'react'
import { ExternalLink, ChevronDown, Monitor } from 'lucide-react'

export function OpenDropdown({ host, port, auth }: { host: string; port: number; auth?: string }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const qs = auth ? `?token=${encodeURIComponent(auth)}` : ''
  const base = `http://${host}:${port}`

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 rounded-lg px-2 py-0.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
      >
        <ExternalLink className="h-3 w-3" />
        Open
        <ChevronDown className="h-3 w-3" />
      </button>
      {open && (
        <div className="absolute right-0 top-full z-50 mt-1 min-w-[120px] rounded-lg border border-zinc-700 bg-zinc-900 py-1 shadow-xl">
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
        </div>
      )}
    </div>
  )
}
