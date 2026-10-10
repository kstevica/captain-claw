// Split a long markdown document into pieces that parse on their own, so a
// big file can be rendered progressively. Parsing is the expensive part
// (~7 µs per character on a laptop; the glasses' CPU is ~12× slower), so a
// 200 KB file parsed in one go would freeze the display for many seconds.
// Piece size is weighed by cost, not only characters: each list item and
// table cell counts extra (4 KB of a table renders ~6× slower than prose).
//
// Preferred cuts are at a blank line, outside fenced code / HTML blocks,
// before a line that starts a new top-level block — never before an indented
// line (list continuation, indented code) or a list item (a list stays in one
// piece, so numbering and tight/loose spacing survive). A piece that grows
// past HARD_FACTOR × target without such a cut (one huge list, table or
// paragraph) is cut at the next line boundary that is still safe: before a
// top-level list item (an ordered list restarts at that item's own number),
// before a table row (the next piece repeats the header and delimiter rows)
// or between two paragraph lines (the soft break becomes a paragraph break) —
// still never inside a fence or an HTML block.
//
// Definitions are document-wide in markdown, pieces are parsed alone:
//  - a (single-line) link-reference definition is appended to every other
//    piece that uses its label;
//  - footnotes are numbered once, in order of first reference, like the whole
//    document would be: references become superscript numbers and the notes
//    are listed in pieces of their own at the end (GFM would otherwise number
//    and list them per piece, or drop them where the definition is missing).

/** Backtick fences: no backtick in the info string (else it is inline code). */
const FENCE_OPEN = /^ {0,3}(`{3,}(?=[^`]*$)|~{3,})/
const FENCE_CLOSE = /^ {0,3}(`{3,}|~{3,})[ \t]*$/
const LIST_ITEM = /^ {0,3}(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$)/
const ATX_HEADING = /^ {0,3}#{1,6}(?:[ \t]|$)/
/** Setext underline (or a `---` rule): cutting before it would change it. */
const SETEXT = /^ {0,3}(?:=+|-+)[ \t]*$/
const THEMATIC = /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:_[ \t]*){3,}|(?:-[ \t]*){3,})$/
const TABLE_DELIM = /^ {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$/
/** A line that starts some other block (ends a table). */
const BLOCK_START = /^ {0,3}(?:#{1,6}(?:[ \t]|$)|>|`{3,}|~{3,}|<[a-zA-Z/?!]|(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$))/
const HTML_COMMENT_OPEN = /^ {0,3}<!--/
const HTML_RAW_OPEN = /^ {0,3}<(pre|script|style|textarea)(?:[\s>]|$)/i
/** Any other HTML block: runs to the next blank line. */
const HTML_BLOCK_OPEN = /^ {0,3}<[a-zA-Z/?!]/
/** A complete one-line link-reference definition (CommonMark: destination,
 *  then an optional quoted title, then nothing). */
const LINK_DEF = /^ {0,3}\[(?!\^)((?:[^\\[\]]|\\.){1,999})\]:[ \t]*(?:<[^<>\n]*>|[^\s<]\S*)(?:[ \t]+(?:"[^"]*"|'[^']*'|\([^()]*\)))?[ \t]*$/
/** GFM footnote definition (labels have no whitespace); its text follows. */
const FOOTNOTE_DEF = /^ {0,3}\[\^([^\]\s]{1,999})\]:[ \t]*(.*)$/
const INDENTED = /^(?: {4}|\t)/
/** A code span (left alone) or a footnote reference. */
const FOOTNOTE_REF = /(`+)[^`]*?\1(?!`)|(?<!\\)\[\^([^\]\s]+)\]/g
/** Bracketed text: candidate reference labels. */
const BRACKETED = /\[((?:[^\\[\]]|\\.)+)\]/g
const HARD_FACTOR = 2
/** Extra size per list item / per table cell (render cost, in characters). */
const ITEM_WEIGHT = 100
const CELL_WEIGHT = 40
const ANY_LIST_ITEM = /^[ \t]*(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$)/
const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹'

function startsTopBlock(line: string): boolean {
  return line.trim() !== '' && !/^[ \t]/.test(line) && !LIST_ITEM.test(line)
}

/** A line boundary a forced cut may use (we are outside fences / HTML). */
function canForceCut(line: string, next: string): boolean {
  if (next.trim() === '' || /^[ \t]/.test(next)) return false
  if (line.trim() === '') return true // before any top-level block, list items included
  // Mid-paragraph / between list items: not into a quote, a setext heading
  // or a table header.
  if (next.trimStart().startsWith('>') || SETEXT.test(next)) return false
  return !(next.includes('|') && TABLE_DELIM.test(next))
}

function normLabel(label: string): string {
  return label.trim().replace(/\s+/g, ' ').toLowerCase()
}

function superscript(n: number): string {
  return String(n).replace(/\d/g, (d) => SUPERSCRIPT[Number(d)])
}

/** Lines [start, end); `head`: the table header + delimiter line numbers to
 *  repeat first (a table cut in two). */
interface Bounds { start: number; end: number; head: number | null }

/** Pieces of roughly `target` characters, weighed by cost (up to HARD_FACTOR
 *  × target, more only for a single huge line / fence / HTML block). Short
 *  texts come back whole. Line endings are normalised to `\n`. */
export function splitMarkdown(input: string, target = 4_000): string[] {
  const text = input.replace(/\r\n?/g, '\n')
  if (text.length <= target * 1.5) return [text]
  const lines = text.split('\n')
  const hardMax = target * HARD_FACTOR
  /** Inside a fence / raw HTML block: never rewritten. */
  const code = new Uint8Array(lines.length)
  /** Footnote definition lines: listed at the end instead. */
  const drop = new Uint8Array(lines.length)
  const linkDefs = new Map<string, { line: number; text: string }>()
  const notes = new Map<string, string[]>()
  const bounds: Bounds[] = []
  let start = 0
  let head: number | null = null
  let size = 0
  let fence: string | null = null
  let rawEnd: RegExp | null = null
  /** In an HTML block that ends at the next blank line. */
  let html = false
  /** Line number of the header row of the GFM table we are in. */
  let table: number | null = null
  /** No paragraph is open: a link definition may start here. */
  let defOk = true
  /** The footnote definition being collected. */
  let note: string[] | null = null

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    size += line.length + 1
    if (fence) {
      code[i] = 1
      const m = FENCE_CLOSE.exec(line)
      if (m && m[1][0] === fence[0] && m[1].length >= fence.length) { fence = null; defOk = true }
      continue
    }
    if (rawEnd) {
      code[i] = 1
      if (rawEnd.test(line)) { rawEnd = null; defOk = true }
      continue
    }
    const blank = line.trim() === ''
    if (note) {
      // GFM: a footnote continues through blank lines and indented lines,
      // and lazily through paragraph lines.
      const last = note[note.length - 1]
      if (blank && i + 1 < lines.length && (lines[i + 1].trim() === '' || INDENTED.test(lines[i + 1]))) {
        note.push(''); drop[i] = 1; continue
      }
      if (!blank && INDENTED.test(line)) {
        note.push(line.replace(/^(?: {1,4}|\t)/, '')); drop[i] = 1; continue
      }
      if (!blank && last !== '' && !FOOTNOTE_DEF.test(line) && !BLOCK_START.test(line) && !THEMATIC.test(line)) {
        note.push(line); drop[i] = 1; continue
      }
      note = null
    }
    if (blank) {
      html = false
      table = null
      defOk = true
    } else {
      const f = FENCE_OPEN.exec(line)
      if (f) { fence = f[1]; code[i] = 1; continue }
      if (HTML_COMMENT_OPEN.test(line)) {
        code[i] = 1
        if (line.includes('-->')) defOk = true
        else rawEnd = /-->/
        continue
      }
      const h = HTML_RAW_OPEN.exec(line)
      if (h) {
        code[i] = 1
        const end = new RegExp(`</${h[1]}>`, 'i')
        if (end.test(line)) defOk = true
        else rawEnd = end
        continue
      }
      const fn = html ? null : FOOTNOTE_DEF.exec(line)
      if (fn) {
        const label = normLabel(fn[1])
        note = [fn[2]]
        drop[i] = 1
        if (!notes.has(label)) notes.set(label, note)
        defOk = true
        table = null
        continue
      }
      if (HTML_BLOCK_OPEN.test(line)) html = true
      // A definition cannot interrupt a paragraph.
      const d = html || !defOk ? null : LINK_DEF.exec(line)
      if (d) {
        const label = normLabel(d[1])
        if (!linkDefs.has(label)) linkDefs.set(label, { line: i, text: line.trim() })
      } else {
        defOk = !html && (ATX_HEADING.test(line) || SETEXT.test(line) || THEMATIC.test(line))
      }
      if (table === null) {
        if (i > 0 && line.includes('|') && TABLE_DELIM.test(line) && lines[i - 1].trim() !== '' && !code[i - 1] && !drop[i - 1]) {
          table = i - 1
        }
      } else if (BLOCK_START.test(line)) {
        table = null
      }
      if (table !== null) size += CELL_WEIGHT * Math.max(1, line.split('|').length - 1)
      else if (ANY_LIST_ITEM.test(line)) size += ITEM_WEIGHT
    }
    if (i + 1 >= lines.length) continue
    const next = lines[i + 1]
    let cut = false
    let nextHead: number | null = null
    if (size >= target && blank && startsTopBlock(next)) {
      cut = true
    } else if (size >= hardMax && !html) {
      if (table !== null) {
        // Before a body row (not right after the delimiter row): every piece
        // of the table repeats its header.
        cut = i > table + 1 && next.trim() !== '' && !BLOCK_START.test(next)
        nextHead = table
      } else {
        cut = canForceCut(line, next)
      }
    }
    if (cut) {
      bounds.push({ start, end: i + 1, head })
      start = i + 1
      head = nextHead
      size = head === null ? 0 : lines[head].length + lines[head + 1].length + 2
    }
  }
  if (!bounds.length) return [text]
  bounds.push({ start, end: lines.length, head })
  // The last piece may end inside an unclosed fence / HTML block: nothing is
  // appended to it.
  const openEnd = fence !== null || rawEnd !== null

  // Footnote numbers, in order of first reference (body first, then notes).
  const numbers = new Map<string, number>()
  const rewrite = (line: string): string => line.replace(FOOTNOTE_REF, (m, tick: string | undefined, label: string | undefined) => {
    if (tick || !label) return m
    const key = normLabel(label)
    if (!notes.has(key)) return m // undefined: stays literal, as in GFM
    let n = numbers.get(key)
    if (!n) { n = numbers.size + 1; numbers.set(key, n) }
    return superscript(n)
  })

  const pieces = bounds.map((b, k) => {
    const kept: string[] = []
    const keep = (i: number) => { kept.push(notes.size && !code[i] ? rewrite(lines[i]) : lines[i]) }
    if (b.head !== null) { keep(b.head); keep(b.head + 1) }
    for (let i = b.start; i < b.end; i++) if (!drop[i]) keep(i)
    let piece = kept.join('\n')
    if (linkDefs.size && !(openEnd && k === bounds.length - 1)) {
      const tail: string[] = []
      const seen = new Set<string>()
      for (const m of piece.matchAll(BRACKETED)) {
        const label = normLabel(m[1])
        const def = linkDefs.get(label)
        if (!def || seen.has(label) || (def.line >= b.start && def.line < b.end)) continue
        seen.add(label)
        tail.push(def.text)
      }
      if (tail.length) piece += `\n\n${tail.join('\n\n')}\n`
    }
    return piece
  }).filter((piece) => /\S/.test(piece)) // e.g. a piece of footnote definitions only

  // The notes, as one numbered list (notes may reference further notes).
  if (numbers.size) {
    let piece = '---\n\n#### Footnotes\n\n'
    size = piece.length
    // `numbers` grows while we go when a note references a further note.
    for (const [key, n] of numbers) {
      const [first, ...rest] = notes.get(key)!.map(rewrite)
      while (rest.length && rest[rest.length - 1] === '') rest.pop()
      const indent = ' '.repeat(String(n).length + 2)
      const item = [`${n}. ${first}`, ...rest.map((l) => (l ? indent + l : ''))].join('\n')
      if (size >= target) {
        pieces.push(piece) // the next piece's list starts at its own number
        piece = ''
        size = 0
      }
      piece += `${item}\n`
      size += item.length + 1 + ITEM_WEIGHT
    }
    pieces.push(piece)
  }
  return pieces
}
