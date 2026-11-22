// An Excel workbook (.xlsx, .xlsm) read in the browser, for the table views:
// each visible sheet's cells as text, the way Excel shows them closely enough
// to read, sort and filter — numbers as numbers, dates as ISO dates (from the
// cell's number format), percentages as percentages, a formula's last result.
// No dependency: an .xlsx is a zip of XML files, unzipped here with the
// browser's own DecompressionStream and read with DOMParser.

export interface Sheet {
  name: string
  /** Every row with something in it, the first one usually the header. */
  rows: string[][]
  /** More rows than `maxRows` asked for: the rest were left out. */
  cut: boolean
}

export interface Workbook {
  sheets: Sheet[]
  /** How many sheets it shows in Excel (more than `sheets` when `maxSheets` cut it short). */
  total: number
  /** Sheets left out because they are hidden in Excel. */
  hidden: number
}

/** Files the table views read as a workbook. */
export const SHEET_EXTS = ['.xlsx', '.xlsm']

export function isSheetExt(ext: string): boolean {
  return SHEET_EXTS.includes(ext.toLowerCase())
}

// ── Zip ──

interface ZipEntry { method: number; compSize: number; offset: number }

function readZip(buf: ArrayBuffer): Map<string, ZipEntry> {
  const dv = new DataView(buf)
  const len = buf.byteLength
  // The end-of-central-directory record, searched for from the end (a
  // comment of up to 64 KiB may follow it).
  let eocd = -1
  for (let i = len - 22; i >= Math.max(0, len - 22 - 0xffff); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break }
  }
  if (eocd < 0) throw new Error("It isn't an Excel workbook (.xlsx).")
  let count = dv.getUint16(eocd + 10, true)
  let cdOffset = dv.getUint32(eocd + 16, true)
  // Zip64: the real numbers are in the zip64 record its locator points at.
  if ((count === 0xffff || cdOffset === 0xffffffff) && eocd >= 20 && dv.getUint32(eocd - 20, true) === 0x07064b50) {
    const rec = Number(dv.getBigUint64(eocd - 12, true))
    if (dv.getUint32(rec, true) === 0x06064b50) {
      count = Number(dv.getBigUint64(rec + 32, true))
      cdOffset = Number(dv.getBigUint64(rec + 48, true))
    }
  }
  const names = new TextDecoder()
  const out = new Map<string, ZipEntry>()
  let p = cdOffset
  for (let n = 0; n < count && p + 46 <= len; n++) {
    if (dv.getUint32(p, true) !== 0x02014b50) break
    const method = dv.getUint16(p + 10, true)
    let compSize = dv.getUint32(p + 20, true)
    const uncompSize = dv.getUint32(p + 24, true)
    const nameLen = dv.getUint16(p + 28, true)
    const extraLen = dv.getUint16(p + 30, true)
    const commentLen = dv.getUint16(p + 32, true)
    let offset = dv.getUint32(p + 42, true)
    const name = names.decode(new Uint8Array(buf, p + 46, nameLen))
    // Zip64 sizes and offset, in that order, for the fields that overflowed.
    let e = p + 46 + nameLen
    const end = e + extraLen
    while (e + 4 <= end) {
      const id = dv.getUint16(e, true)
      const size = dv.getUint16(e + 2, true)
      if (id === 0x0001) {
        let q = e + 4
        if (uncompSize === 0xffffffff) q += 8
        if (compSize === 0xffffffff) { compSize = Number(dv.getBigUint64(q, true)); q += 8 }
        if (offset === 0xffffffff) offset = Number(dv.getBigUint64(q, true))
      }
      e += 4 + size
    }
    out.set(name.replace(/^\/+/, ''), { method, compSize, offset })
    p = end + commentLen
  }
  return out
}

async function unzipText(buf: ArrayBuffer, entry: ZipEntry): Promise<string> {
  const dv = new DataView(buf)
  const at = entry.offset
  if (dv.getUint32(at, true) !== 0x04034b50) throw new Error('The workbook is damaged.')
  const start = at + 30 + dv.getUint16(at + 26, true) + dv.getUint16(at + 28, true)
  const data = new Uint8Array(buf, start, entry.compSize)
  if (entry.method === 0) return new TextDecoder().decode(data)
  if (entry.method !== 8) throw new Error('The workbook is packed in a way this viewer can\'t read.')
  const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'))
  return new Response(stream).text()
}

// ── XML ──

function parseXml(text: string): Document {
  const doc = new DOMParser().parseFromString(text, 'application/xml')
  if (doc.getElementsByTagName('parsererror').length) throw new Error('The workbook is damaged.')
  return doc
}

/** Elements by local name, whatever prefix the writer gave them (x:row …). */
const all = (root: Document | Element, local: string) => Array.from(root.getElementsByTagNameNS('*', local))
const kids = (el: Element, local: string) => Array.from(el.children).filter((c) => c.localName === local)
const kid = (el: Element, local: string) => Array.from(el.children).find((c) => c.localName === local)

/** A string item's text: its own <t>, or its runs' (phonetic hints left out). */
function stringText(si: Element): string {
  const t = kid(si, 't')
  if (t) return t.textContent ?? ''
  return kids(si, 'r').map((r) => kid(r, 't')?.textContent ?? '').join('')
}

// ── Number formats ──

const DATE_FORMAT_IDS = new Set([14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 45, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58])
const PERCENT_FORMAT_IDS = new Set([9, 10])

type Look = 'date' | 'percent' | null

/** What a number format makes of a number: a date or time, a percentage, or plain. */
function lookOf(id: number, code: string | undefined): Look {
  if (DATE_FORMAT_IDS.has(id)) return 'date'
  if (PERCENT_FORMAT_IDS.has(id)) return 'percent'
  if (!code) return null
  // Quoted text, [colours]/[$currency], escaped and padded characters aside.
  const bare = code.replace(/"[^"]*"/g, '').replace(/\[[^\]]*\]/g, '').replace(/\\.|_.|\*./g, '')
  if (/%/.test(bare)) return 'percent'
  if (/[dmyhs]/i.test(bare) && !/^general$/i.test(bare.trim())) return 'date'
  return null
}

const pad = (n: number) => String(n).padStart(2, '0')

/** An Excel serial date as ISO text: a date, a date and time, or a time. */
function serialDate(serial: number, date1904: boolean): string {
  let days = serial
  if (date1904) days += 1462
  // Excel counts 29 February 1900, which didn't exist.
  const base = Date.UTC(1899, 11, days < 60 ? 31 : 30)
  const ms = Math.round((base + days * 86_400_000) / 1000) * 1000
  const d = new Date(ms)
  if (Number.isNaN(d.getTime())) return String(serial)
  const date = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`
  const secs = d.getUTCSeconds()
  const time = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}${secs ? `:${pad(secs)}` : ''}`
  if (serial < 1 && serial >= 0) return time
  return Number.isInteger(serial) || (d.getUTCHours() === 0 && d.getUTCMinutes() === 0 && !secs) ? date : `${date} ${time}`
}

/** A number as people write it: no float noise (0.1 + 0.2), no exponent for everyday sizes. */
function numberText(n: number): string {
  if (!Number.isFinite(n)) return String(n)
  return String(Number.parseFloat(n.toPrecision(15)))
}

// ── Cells ──

/** "BC12" → 54 (0-based column). */
function columnOf(ref: string): number {
  let n = 0
  for (const ch of ref) {
    const c = ch.charCodeAt(0)
    if (c >= 65 && c <= 90) n = n * 26 + (c - 64)
    else if (c >= 97 && c <= 122) n = n * 26 + (c - 96)
    else break
  }
  return n - 1
}

function readSheet(doc: Document, shared: string[], looks: Look[], date1904: boolean, maxRows: number): { rows: string[][]; cut: boolean } {
  const rows: string[][] = []
  let cut = false
  const data = all(doc, 'sheetData')[0]
  if (!data) return { rows, cut }
  for (const row of kids(data, 'row')) {
    const cells: string[] = []
    let next = 0
    for (const c of kids(row, 'c')) {
      const ref = c.getAttribute('r')
      const col = ref ? columnOf(ref) : next
      next = col + 1
      const t = c.getAttribute('t') ?? 'n'
      const v = kid(c, 'v')?.textContent ?? ''
      let text: string
      if (t === 's') text = shared[Number(v)] ?? ''
      else if (t === 'inlineStr') { const is = kid(c, 'is'); text = is ? stringText(is) : '' }
      else if (t === 'b') text = v === '1' ? 'TRUE' : v === '0' ? 'FALSE' : v
      else if (t === 'str' || t === 'e' || v === '') text = v
      else {
        const n = Number(v)
        const look = looks[Number(c.getAttribute('s') ?? 0)] ?? null
        text = Number.isNaN(n) ? v
          : look === 'date' ? serialDate(n, date1904)
            : look === 'percent' ? `${numberText(n * 100)}%`
              : numberText(n)
      }
      if (col >= 0 && col < 16_384) cells[col] = text
    }
    // Holes (cells never written) read as empty; empty rows are dropped.
    const filled = Array.from({ length: cells.length }, (_, i) => cells[i] ?? '')
    if (!filled.some((x) => x.trim() !== '')) continue
    if (rows.length >= maxRows) { cut = true; break }
    rows.push(filled)
  }
  // Every row as wide as the widest, empty columns at the right end dropped.
  let width = 0
  for (const r of rows) for (let i = r.length - 1; i >= width; i--) if (r[i].trim() !== '') { width = i + 1; break }
  return { rows: rows.map((r) => Array.from({ length: width }, (_, i) => r[i] ?? '')), cut }
}

/** Where a relationship target lives in the zip, from xl/. */
function resolveTarget(target: string): string {
  if (target.startsWith('/')) return target.slice(1)
  const parts = `xl/${target}`.split('/')
  const out: string[] = []
  for (const p of parts) {
    if (p === '..') out.pop()
    else if (p !== '.' && p !== '') out.push(p)
  }
  return out.join('/')
}

/** The workbook's visible sheets (the first `maxSheets`), each up to `maxRows` rows. */
export async function readXlsx(buf: ArrayBuffer, { maxRows = 50_000, maxSheets = 50 } = {}): Promise<Workbook> {
  const zip = readZip(buf)
  const text = async (path: string) => {
    const e = zip.get(path)
    return e ? unzipText(buf, e) : null
  }
  const wbXml = await text('xl/workbook.xml')
  if (!wbXml) throw new Error("It isn't an Excel workbook (.xlsx).")
  const wb = parseXml(wbXml)
  const date1904 = ['1', 'true'].includes(all(wb, 'workbookPr')[0]?.getAttribute('date1904') ?? '')

  const targets = new Map<string, string>()
  const relsXml = await text('xl/_rels/workbook.xml.rels')
  if (relsXml) {
    for (const r of all(parseXml(relsXml), 'Relationship')) {
      targets.set(r.getAttribute('Id') ?? '', resolveTarget(r.getAttribute('Target') ?? ''))
    }
  }

  const shared: string[] = []
  const ssXml = await text('xl/sharedStrings.xml')
  if (ssXml) for (const si of all(parseXml(ssXml), 'si')) shared.push(stringText(si))

  const looks: Look[] = []
  const stylesXml = await text('xl/styles.xml')
  if (stylesXml) {
    const st = parseXml(stylesXml)
    const codes = new Map<number, string>()
    for (const f of all(st, 'numFmt')) codes.set(Number(f.getAttribute('numFmtId')), f.getAttribute('formatCode') ?? '')
    const xfs = all(st, 'cellXfs')[0]
    if (xfs) {
      for (const xf of kids(xfs, 'xf')) {
        const id = Number(xf.getAttribute('numFmtId') ?? 0)
        looks.push(lookOf(id, codes.get(id)))
      }
    }
  }

  const listed = all(wb, 'sheet').map((s) => ({
    name: s.getAttribute('name') ?? 'Sheet',
    state: s.getAttribute('state') ?? 'visible',
    path: targets.get(s.getAttributeNS('http://schemas.openxmlformats.org/officeDocument/2006/relationships', 'id')
      ?? s.getAttribute('r:id') ?? '') ?? '',
  })).filter((s) => s.path.includes('worksheets/'))
  const visible = listed.filter((s) => s.state === 'visible')
  const shown = (visible.length ? visible : listed).slice(0, maxSheets)

  const sheets: Sheet[] = []
  for (const s of shown) {
    const xml = await text(s.path)
    if (xml === null) continue
    const { rows, cut } = readSheet(parseXml(xml), shared, looks, date1904, maxRows)
    sheets.push({ name: s.name, rows, cut })
  }
  if (!sheets.length) throw new Error('The workbook has no sheets to show.')
  return { sheets, total: (visible.length ? visible : listed).length, hidden: visible.length ? listed.length - visible.length : 0 }
}
