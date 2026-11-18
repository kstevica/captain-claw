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

/** A worker's progress note (Markdown it keeps as it goes), small: headings no bigger
 *  than the text. The start shows until it's opened; a step being worked on opens it. */
export function NoteText({ text, startOpen = false }: { text: string; startOpen?: boolean }) {
  const [open, setOpen] = useState(startOpen)
  const [long, setLong] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (el) setLong(el.scrollHeight > 96 + 16)
  }, [text])
  return (
    <div className="mt-2 rounded-xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] px-3 py-2">
      <div ref={ref} className={`relative ${open ? 'max-h-96 overflow-y-auto' : 'max-h-24 overflow-hidden'}`}>
        <div className="fd-markdown text-xs leading-relaxed text-zinc-400 [&_h1]:text-xs [&_h1]:font-semibold [&_h1]:text-zinc-300 [&_h2]:text-xs [&_h2]:font-semibold [&_h2]:text-zinc-300 [&_h3]:text-xs [&_h3]:text-zinc-300 [&_h1]:my-1 [&_h2]:my-1 [&_h3]:my-1 [&_p]:my-1 [&_ol]:list-decimal [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:pl-4">
          <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform} components={COMPONENTS}>{text}</Markdown>
        </div>
        {long && !open && (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-[var(--fd-surface-raised)] to-transparent" />
        )}
      </div>
      {long && (
        <button onClick={() => setOpen(!open)} className="mt-1 text-[11px] font-medium text-violet-700 hover:underline dark:text-violet-300">
          {open ? 'Show less' : 'Show the whole note'}
        </button>
      )}
    </div>
  )
}
