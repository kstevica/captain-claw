import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ChevronLeft, ChevronRight, Download, ExternalLink, X } from 'lucide-react'

export interface LightboxItem {
  /** Where the picture is seen: the deck's proxy to the agent, a shared
   *  file, or a data URL for one still in the composer. */
  src: string
  name: string
}

// Black and white, not zinc: a picture viewer stays dark in the light theme
// too, and the light theme turns the zinc scale around.
const ICON_BTN = 'flex h-10 w-10 items-center justify-center rounded-full text-white/80 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/60'

/** A picture you sent (or are about to), over the page: the whole picture,
 *  its name, open in a new tab, download. Arrows (and ← →) step through the
 *  other pictures of the same message; Escape, × or a click beside the
 *  picture closes it. */
export function ImageLightbox({ items, index, onClose }: {
  items: LightboxItem[]
  index: number
  onClose: () => void
}) {
  const [at, setAt] = useState(() => Math.min(Math.max(index, 0), items.length - 1))
  const closeRef = useRef<HTMLButtonElement>(null)
  const item = items[at]
  const many = items.length > 1
  const step = useCallback((d: number) => setAt((i) => (i + d + items.length) % items.length), [items.length])

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); onClose() }
      else if (e.key === 'ArrowRight' && many) { e.preventDefault(); step(1) }
      else if (e.key === 'ArrowLeft' && many) { e.preventDefault(); step(-1) }
    }
    window.addEventListener('keydown', onKey, true)
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey, true)
      document.body.style.overflow = overflow
      opener?.focus?.()
    }
  }, [onClose, many, step])

  if (!item) return null
  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label={item.name}
      data-fd-lightbox
      className="fixed inset-0 z-[100] flex flex-col bg-black/85 backdrop-blur-sm"
      onClick={onClose}
    >
      <div className="flex shrink-0 items-center gap-2 px-3 py-2 pt-[max(0.5rem,env(safe-area-inset-top))] text-white" onClick={(e) => e.stopPropagation()}>
        <div className="min-w-0 flex-1 pl-1">
          <div className="truncate text-sm font-medium">{item.name}</div>
          {many && <div className="text-xs text-white/60">{at + 1} of {items.length}</div>}
        </div>
        <a href={item.src} target="_blank" rel="noopener noreferrer" title="Open in a new tab" aria-label="Open in a new tab" className={ICON_BTN}>
          <ExternalLink className="h-[18px] w-[18px]" />
        </a>
        <a href={item.src} download={item.name} title="Download" aria-label="Download" className={ICON_BTN}>
          <Download className="h-[18px] w-[18px]" />
        </a>
        <button ref={closeRef} type="button" onClick={onClose} title="Close (Esc)" aria-label="Close" className={ICON_BTN}>
          <X className="h-5 w-5" />
        </button>
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center px-4 pb-6 sm:px-16">
        <img
          key={item.src}
          src={item.src}
          alt={item.name}
          onClick={(e) => e.stopPropagation()}
          className="max-h-full max-w-full rounded-lg object-contain shadow-2xl"
        />
        {many && (
          <>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); step(-1) }}
              aria-label="Previous picture"
              className={`${ICON_BTN} absolute left-2 top-1/2 -translate-y-1/2 bg-black/40 sm:left-4`}
            >
              <ChevronLeft className="h-6 w-6" />
            </button>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); step(1) }}
              aria-label="Next picture"
              className={`${ICON_BTN} absolute right-2 top-1/2 -translate-y-1/2 bg-black/40 sm:right-4`}
            >
              <ChevronRight className="h-6 w-6" />
            </button>
          </>
        )}
      </div>
    </div>,
    document.body,
  )
}
