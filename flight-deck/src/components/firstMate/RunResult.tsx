import { useLayoutEffect, useRef, useState } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { untrustedUrlTransform } from '../../utils/untrustedContent'

// A factory run's result, as its Bat run wrote it: Markdown, tables included.
// It's agent output, so it loads nothing (no images) and links open in a new
// tab through the deck's own check. Long results show their start until opened.

const COMPONENTS: Components = {
  img: () => null,
  a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer nofollow">{children}</a>,
  // A wide table scrolls inside the card instead of squeezing its columns.
  table: ({ children }) => <div className="my-2 overflow-x-auto"><table className="min-w-full">{children}</table></div>,
}

const COLLAPSED_PX = 192

export function RunResult({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  const [long, setLong] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (el) setLong(el.scrollHeight > COLLAPSED_PX + 24)
  }, [text])
  return (
    <div className="mt-2">
      <div ref={ref} className={`relative overflow-hidden ${open || !long ? '' : 'max-h-48'}`}>
        <div className="fd-markdown text-[13px] leading-relaxed text-zinc-300 [&_ol]:list-decimal [&_ul]:list-disc [&_li]:pl-0.5">
          <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={COMPONENTS}>{text}</Markdown>
        </div>
        {long && !open && (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-[var(--fd-surface-raised)] to-transparent" />
        )}
      </div>
      {long && (
        <button onClick={() => setOpen(!open)} className="mt-1 text-[11px] text-zinc-500 hover:text-zinc-300">
          {open ? 'Show less' : 'Show all'}
        </button>
      )}
    </div>
  )
}
