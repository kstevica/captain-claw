// CSV/TSV read and written whole, for the table editor (CsvTable). Quote-aware:
// a quoted cell may hold the separator, doubled quotes and line breaks. What it
// writes back reads the same in a spreadsheet and in Python's csv module.

const SEPARATORS = [',', ';', '\t'] as const

/** The separator the text uses: the one that splits the first lines (outside
 *  quotes) the most consistently. A .tsv file is tab-separated. */
export function sniffSeparator(text: string, extension = ''): string {
  if (extension.toLowerCase() === '.tsv') return '\t'
  const lines: string[] = []
  let cur = ''
  let quoted = false
  for (let i = 0; i < text.length && lines.length < 10; i++) {
    const ch = text[i]
    if (ch === '"') quoted = !quoted
    if ((ch === '\n' || ch === '\r') && !quoted) {
      if (cur.trim()) lines.push(cur)
      cur = ''
      continue
    }
    cur += ch
  }
  if (cur.trim() && lines.length < 10) lines.push(cur)
  let best = ','
  let bestScore = -1
  for (const sep of SEPARATORS) {
    const counts = lines.map((l) => countOutsideQuotes(l, sep))
    if (!counts.length || counts[0] === 0) continue
    const same = counts.filter((c) => c === counts[0]).length
    const score = same * 100 + counts[0]
    if (score > bestScore) { best = sep; bestScore = score }
  }
  return best
}

function countOutsideQuotes(line: string, sep: string): number {
  let n = 0
  let quoted = false
  for (const ch of line) {
    if (ch === '"') quoted = !quoted
    else if (ch === sep && !quoted) n++
  }
  return n
}

/** Every row of the text. A cell that opens with a quote runs to its closing
 *  quote ("" is a quote inside it); CRLF and LF both end a row; blank lines are
 *  dropped. A byte-order mark is ignored. */
export function parseCsv(text: string, sep: string): string[][] {
  const src = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  let started = false // the cell has content (a quote only opens an empty cell)
  const endCell = () => { row.push(cell); cell = ''; started = false }
  const endRow = () => {
    endCell()
    if (row.length > 1 || row[0] !== '') rows.push(row)
    row = []
  }
  for (let i = 0; i < src.length; i++) {
    const ch = src[i]
    if (quoted) {
      if (ch === '"') {
        if (src[i + 1] === '"') { cell += '"'; i++ } else quoted = false
      } else cell += ch
      continue
    }
    if (ch === '"' && !started) { quoted = true; started = true; continue }
    if (ch === sep) { endCell(); continue }
    if (ch === '\r') { if (src[i + 1] === '\n') i++; endRow(); continue }
    if (ch === '\n') { endRow(); continue }
    cell += ch
    started = true
  }
  if (cell !== '' || row.length) endRow()
  return rows
}

/** One cell as written: quoted when it holds the separator, a quote, a line
 *  break, or space at either end. */
function writeCell(value: string, sep: string): string {
  const v = value ?? ''
  return v.includes(sep) || v.includes('"') || v.includes('\n') || v.includes('\r') || v !== v.trim()
    ? `"${v.replace(/"/g, '""')}"` : v
}

/** The rows as CSV text, one row a line (LF), ending with a line break. */
export function toCsv(rows: string[][], sep: string): string {
  return rows.map((r) => r.map((c) => writeCell(c, sep)).join(sep)).join('\n') + '\n'
}
