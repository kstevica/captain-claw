// Markdown for the HUD: file previews (`blocks`) and chat replies.
//
// react-markdown + remark-gfm only. Everything renders as React text: no raw
// HTML (skipHtml, never rehype-raw; rehypeHud keeps `<br>` breaks and the
// visible text of HTML blocks as text), no KaTeX, no syntax highlighting. Links
// are text — following one would leave the app inside the glasses webview —
// and images become an alt-text chip: never load remote images (bandwidth,
// and a tracking pixel in someone else's file). Tables and reading blocks are
// shaped by rehypeHud (see rehypeHud.ts).
//
// blocks=true: each top-level block is a focusable `.hud-block` the D-pad
// walks. blocks=false: no focusable descendants (a chat message is already
// one focus stop).

import { memo } from 'react'
import Markdown, { type Components, type Options } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { rehypeHud } from './rehypeHud'
import './markdown.css'

const REMARK_PLUGINS: Options['remarkPlugins'] = [remarkGfm]
const REHYPE_INLINE: Options['rehypePlugins'] = [rehypeHud]
const REHYPE_BLOCKS: Options['rehypePlugins'] = [[rehypeHud, { blocks: true }]]

const COMPONENTS: Components = {
  a: ({ children }) => <span className="hud-md-link">{children}</span>,
  img: ({ alt }) => (
    <span className="hud-md-img">
      <span aria-hidden="true">🖼️</span> {alt?.trim() || 'image'}
    </span>
  ),
  // GFM task-list checkboxes: a glyph, never a (focusable) form control.
  input: ({ type, checked }) =>
    type === 'checkbox' ? <span className="hud-md-check">{checked ? '☑' : '☐'}</span> : null,
}

function HudMarkdownView(props: { text: string; blocks?: boolean; className?: string }) {
  const cls = `hud-md${props.blocks ? ' hud-md--blocks' : ''}${props.className ? ' ' + props.className : ''}`
  return (
    <div className={cls}>
      <Markdown
        remarkPlugins={REMARK_PLUGINS}
        rehypePlugins={props.blocks ? REHYPE_BLOCKS : REHYPE_INLINE}
        components={COMPONENTS}
        skipHtml
      >
        {props.text}
      </Markdown>
    </div>
  )
}

/** `<HudMarkdown text={md} blocks? className? />` — memoized (parsing is the
 *  expensive part on the glasses' slow CPU). */
export const HudMarkdown = memo(HudMarkdownView)
