// rehype plugin that shapes markdown for the 600×600 HUD (see HudMarkdown).
//
// Runs on the hast tree react-markdown builds (remark-gfm → remark-rehype),
// before it is turned into React elements:
//
//   1. drops embedded raw HTML (react-markdown's skipHtml would drop it after
//      us anyway — doing it first keeps "is this cell empty?" honest);
//   2. tables: narrow ones (≤ 3 columns, short headers) stay real tables in a
//      `.hud-md-table` wrapper; wider ones become one card per row — a
//      horizontal scroller is unusable with a D-pad and Meta QA flags any
//      horizontal overflow;
//   3. `blocks: true` (file preview): every top-level block becomes a
//      focusable reading block (`.hud-block`, tabIndex 0) so the wearer can
//      walk the document with swipes — paragraphs, headings, quotes, code,
//      each top-level list item, each table card / narrow table.
//
// Hand-written walkers on purpose: unist-util-visit & co. are only transitive
// dependencies here.

import type { Element, ElementContent, Root, RootContent, Text } from 'hast'

export interface RehypeHudOptions {
  /** Make top-level blocks focusable reading blocks (file preview). */
  blocks?: boolean
}

type Parent = Root | Element
type HNode = RootContent | ElementContent

/** Narrow-table limits: more columns or longer headers → cards. */
const NARROW_MAX_COLS = 3
const NARROW_MAX_HEADER_CHARS = 40
const EMPTY_CELL = '—'

// ── tiny hast helpers ──

function isEl(n: HNode | undefined, tag?: string): n is Element {
  return !!n && n.type === 'element' && (tag === undefined || n.tagName === tag)
}

function text(value: string): Text {
  return { type: 'text', value }
}

function el(tagName: string, className: string | null, children: ElementContent[]): Element {
  return { type: 'element', tagName, properties: className ? { className: [className] } : {}, children }
}

function classList(e: Element): string[] {
  const c = e.properties.className
  if (Array.isArray(c)) return c.map(String)
  if (typeof c === 'string') return c.split(/\s+/).filter(Boolean)
  return []
}

function hasClass(e: Element, name: string): boolean {
  return classList(e).includes(name)
}

function addClass(e: Element, name: string): void {
  const list = classList(e)
  if (!list.includes(name)) e.properties.className = [...list, name]
}

function textOf(n: HNode): string {
  if (n.type === 'text') return n.value
  if (n.type === 'element') {
    let out = ''
    for (const c of n.children) out += textOf(c)
    return out
  }
  return ''
}

/** Visible content: non-blank text, or an image / checkbox (rendered as chips). */
function hasContent(n: HNode): boolean {
  if (n.type === 'text') return n.value.trim() !== ''
  if (n.type !== 'element') return false
  if (n.tagName === 'img' || n.tagName === 'input') return true
  return n.children.some(hasContent)
}

function cloneNode(n: ElementContent): ElementContent {
  if (n.type !== 'element') return { ...n }
  const properties: Element['properties'] = {}
  for (const [k, v] of Object.entries(n.properties)) properties[k] = Array.isArray(v) ? [...v] : v
  return { ...n, properties, children: n.children.map(cloneNode) }
}

/** Cell children, or an em dash when the cell shows nothing. */
function filled(children: ElementContent[] | undefined): ElementContent[] {
  return children && children.some(hasContent) ? children : [text(EMPTY_CELL)]
}

// ── 1. raw HTML ──

function stripRaw(parent: Parent): void {
  // `raw` nodes come from remark-rehype's allowDangerousHtml (react-markdown
  // always sets it); they are not part of the core hast types.
  const kids = parent.children as HNode[]
  for (let i = kids.length - 1; i >= 0; i--) {
    const c = kids[i]
    if ((c as { type: string }).type === 'raw') kids.splice(i, 1)
    else if (c.type === 'element') stripRaw(c)
  }
}

// ── 2. tables ──

function tableRows(table: Element): { head: Element | null; body: Element[] } {
  let head: Element | null = null
  const all: Element[] = []
  for (const c of table.children) {
    if (!isEl(c)) continue
    if (c.tagName === 'tr') { all.push(c); continue }
    if (c.tagName === 'thead' || c.tagName === 'tbody' || c.tagName === 'tfoot') {
      for (const r of c.children) {
        if (!isEl(r, 'tr')) continue
        if (c.tagName === 'thead' && !head) head = r
        all.push(r)
      }
    }
  }
  // No header row: the first row is the header.
  if (!head) head = all[0] ?? null
  return { head, body: all.filter((r) => r !== head) }
}

function rowCells(tr: Element): Element[] {
  return tr.children.filter((c): c is Element => isEl(c) && (c.tagName === 'td' || c.tagName === 'th'))
}

function emDashEmptyCells(rows: Element[]): void {
  for (const tr of rows) {
    for (const cell of rowCells(tr)) {
      if (!cell.children.some(hasContent)) cell.children = [text(EMPTY_CELL)]
    }
  }
}

function tableToCards(head: Element, body: Element[]): Element {
  const headers = rowCells(head)
  const cols = headers.length
  if (!body.length) {
    // Header only: one card listing the column names.
    const names = headers.map((h) => textOf(h).trim()).filter(Boolean).join(' · ')
    return el('div', 'hud-md-cards', [el('div', 'hud-md-card', [el('div', 'hud-md-card-title', [text(names || EMPTY_CELL)])])])
  }
  const cards = body.map((tr) => {
    const cells = rowCells(tr)
    const n = Math.max(cols, cells.length)
    const kvs: Element[] = []
    for (let i = 1; i < n; i++) {
      const h = headers[i]
      const label = h && h.children.some(hasContent) ? h.children.map(cloneNode) : [text(`Column ${i + 1}`)]
      kvs.push(el('div', 'hud-md-kv', [el('dt', null, label), el('dd', null, filled(cells[i]?.children))]))
    }
    const title = el('div', 'hud-md-card-title', filled(cells[0]?.children))
    return el('div', 'hud-md-card', kvs.length ? [title, el('dl', null, kvs)] : [title])
  })
  return el('div', 'hud-md-cards', cards)
}

function transformTable(table: Element): Element | null {
  const { head, body } = tableRows(table)
  if (!head) return null // an empty table shows nothing
  const headers = rowCells(head)
  const headerChars = headers.reduce((sum, h) => sum + textOf(h).trim().length, 0)
  if (headers.length <= NARROW_MAX_COLS && headerChars <= NARROW_MAX_HEADER_CHARS) {
    emDashEmptyCells(body)
    return el('div', 'hud-md-table', [table])
  }
  return tableToCards(head, body)
}

function transformTables(parent: Parent): void {
  const kids = parent.children as HNode[]
  for (let i = kids.length - 1; i >= 0; i--) {
    const c = kids[i]
    if (!isEl(c)) continue
    if (c.tagName === 'table') {
      const out = transformTable(c)
      if (out) kids[i] = out
      else kids.splice(i, 1)
    } else {
      transformTables(c)
    }
  }
}

// ── 3. reading blocks ──

function makeBlock(e: Element): void {
  addClass(e, 'hud-block')
  e.properties.tabIndex = 0
}

/** Top-level list: each item is a block. Item markers are drawn by us (an
 *  outside ::marker would sit outside the item's focus highlight). */
function blockifyList(list: Element): void {
  addClass(list, 'hud-md-list')
  const ordered = list.tagName === 'ol'
  const start = Number(list.properties.start)
  let n = Number.isFinite(start) ? start : 1
  for (const li of list.children) {
    if (!isEl(li, 'li')) continue
    if (!hasClass(li, 'task-list-item')) {
      const marker = el('span', 'hud-md-marker', [text(ordered ? `${n}.` : '•')])
      marker.properties.ariaHidden = 'true'
      li.children.unshift(marker)
    }
    n++
    if (li.children.some(hasContent)) makeBlock(li)
  }
}

function blockify(parent: Parent): void {
  for (const c of parent.children) {
    if (!isEl(c)) continue // whitespace text, comments
    const tag = c.tagName
    if (tag === 'hr') continue
    if (tag === 'ul' || tag === 'ol') { blockifyList(c); continue }
    if (tag === 'section') { blockify(c); continue } // GFM footnotes: heading + list
    if (tag === 'div' && hasClass(c, 'hud-md-cards')) {
      for (const card of c.children) if (isEl(card)) makeBlock(card)
      continue
    }
    if (hasContent(c)) makeBlock(c)
  }
}

/** The plugin. `[rehypeHud, { blocks: true }]` for file previews. */
export function rehypeHud(options?: RehypeHudOptions) {
  const blocks = !!options?.blocks
  return (tree: Root): void => {
    stripRaw(tree)
    transformTables(tree)
    if (blocks) blockify(tree)
  }
}
