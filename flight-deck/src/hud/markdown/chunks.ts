// Split a long markdown document into pieces that parse on their own, so a
// big file can be rendered progressively. Parsing is the expensive part
// (~7 µs per character on a laptop; the glasses' CPU is ~12× slower), so a
// 200 KB file parsed in one go would freeze the display for many seconds.
//
// Cuts are made only at a blank line, outside fenced code / raw HTML blocks,
// before a line that starts a new top-level block — never before an indented
// line (list continuation, indented code) or a list item (a list stays in one
// piece, so numbering and tight/loose spacing survive). Link-reference and
// footnote definitions are document-wide in markdown, so their (first) lines
// are appended to every piece.

const FENCE_OPEN = /^ {0,3}(`{3,}|~{3,})/
const FENCE_CLOSE = /^ {0,3}(`{3,}|~{3,})[ \t]*$/
const LIST_ITEM = /^ {0,3}(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$)/
const DEFINITION = /^ {0,3}\[[^\]]+\]:[ \t]*\S/
const HTML_COMMENT_OPEN = /^ {0,3}<!--/
const HTML_RAW_OPEN = /^ {0,3}<(pre|script|style|textarea)(?:[\s>]|$)/i
const MAX_DEFS_CHARS = 8_000

function startsTopBlock(line: string): boolean {
  return line.trim() !== '' && !/^[ \t]/.test(line) && !LIST_ITEM.test(line)
}

/** Pieces of roughly `target` characters (a piece may be longer when no safe
 *  cut exists, e.g. inside one huge table). Short texts come back whole. */
export function splitMarkdown(text: string, target = 4_000): string[] {
  if (text.length <= target * 1.5) return [text]
  const lines = text.split('\n')
  const pieces: string[] = []
  const defs: string[] = []
  let start = 0
  let size = 0
  let fence: string | null = null
  let rawEnd: RegExp | null = null
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    size += line.length + 1
    if (fence) {
      const m = FENCE_CLOSE.exec(line)
      if (m && m[1][0] === fence[0] && m[1].length >= fence.length) fence = null
      continue
    }
    if (rawEnd) {
      if (rawEnd.test(line)) rawEnd = null
      continue
    }
    const f = FENCE_OPEN.exec(line)
    if (f) { fence = f[1]; continue }
    if (HTML_COMMENT_OPEN.test(line)) {
      if (!line.includes('-->')) rawEnd = /-->/
      continue
    }
    const h = HTML_RAW_OPEN.exec(line)
    if (h) {
      const end = new RegExp(`</${h[1]}>`, 'i')
      if (!end.test(line)) rawEnd = end
      continue
    }
    if (DEFINITION.test(line)) defs.push(line.trim())
    if (size >= target && line.trim() === '' && i + 1 < lines.length && startsTopBlock(lines[i + 1])) {
      pieces.push(lines.slice(start, i + 1).join('\n'))
      start = i + 1
      size = 0
    }
  }
  pieces.push(lines.slice(start).join('\n'))
  if (pieces.length > 1 && defs.length) {
    const tail = `\n\n${defs.join('\n')}\n`
    if (tail.length <= MAX_DEFS_CHARS) {
      // Not into a last piece that ends inside an unclosed fence / HTML block.
      const last = fence || rawEnd ? pieces.length - 1 : pieces.length
      for (let k = 0; k < last; k++) pieces[k] += tail
    }
  }
  return pieces
}
