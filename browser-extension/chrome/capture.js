// Captain for Chrome — reads the page it is injected into.
//
// Injected by the side panel with chrome.scripting.executeScript; the value
// of the last expression (this function's) is what the panel gets back:
// the page's details, its main text as Markdown, the selection, its tables,
// the files it links to and its links. Nothing leaves the page from here.

(() => {
  const MAX_MD = 400000
  const MAX_NODES = 60000
  const MAX_LINKS = 400
  const MAX_FILES = 250
  const MAX_IMAGES = 60
  const MAX_TABLES = 40
  const MAX_ROWS = 3000
  const MAX_COLS = 60
  const MAX_LD = 20000

  const oneLine = (s) => String(s || '').replace(/\s+/g, ' ').trim()
  const abs = (u) => {
    try { return new URL(u, document.baseURI).href } catch { return '' }
  }
  const metaContent = (sel) => {
    const el = document.querySelector(sel)
    return el ? oneLine(el.getAttribute('content') || el.getAttribute('href') || '') : ''
  }
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s)

  // ── the page's details ──
  const contentType = document.contentType || ''
  const isPdf = contentType === 'application/pdf' || /\.pdf$/i.test(location.pathname)
  const timeEl = document.querySelector('time[datetime]')
  const meta = {
    url: location.href,
    title: oneLine(document.title) || metaContent('meta[property="og:title"]'),
    canonical: abs(metaContent('link[rel="canonical"]')) || '',
    description: metaContent('meta[name="description"]') || metaContent('meta[property="og:description"]'),
    site_name: metaContent('meta[property="og:site_name"]'),
    author: metaContent('meta[name="author"]') || metaContent('meta[property="article:author"]'),
    published: metaContent('meta[property="article:published_time"]') || metaContent('meta[name="date"]')
      || (timeEl ? oneLine(timeEl.getAttribute('datetime')) : ''),
    lang: document.documentElement.lang || '',
    image: abs(metaContent('meta[property="og:image"]')) || '',
    content_type: contentType,
  }

  // JSON-LD: a product's price, a job's salary, an event's date…
  const ld = []
  const ldTypes = new Set()
  for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
    const raw = (s.textContent || '').trim()
    if (!raw) continue
    try {
      const data = JSON.parse(raw)
      const walk = (v) => {
        if (Array.isArray(v)) v.forEach(walk)
        else if (v && typeof v === 'object') {
          const t = v['@type']
          if (t) (Array.isArray(t) ? t : [t]).forEach((x) => ldTypes.add(String(x)))
          if (v['@graph']) walk(v['@graph'])
        }
      }
      walk(data)
      ld.push(JSON.stringify(data, null, 1))
    } catch { /* not JSON after all */ }
  }
  const structured = clip(ld.join('\n'), MAX_LD)

  // ── what counts as a file ──
  const EXT_KIND = {}
  const kinds = {
    document: 'pdf doc docx dot dotx odt rtf pages epub mobi xls xlsx xlsm xlsb ods numbers csv tsv ppt pptx pps ppsx odp key txt md',
    archive: 'zip rar 7z tar gz tgz bz2 xz',
    data: 'json xml ics vcf kml kmz gpx geojson ipynb sql yaml yml',
    media: 'mp3 m4a wav flac ogg aac mp4 m4v mov webm mkv avi',
    image: 'png jpg jpeg gif webp avif bmp tif tiff heic svg',
    design: 'psd ai eps sketch fig dwg dxf stl step stp',
  }
  for (const [kind, list] of Object.entries(kinds)) for (const e of list.split(' ')) EXT_KIND[e] = kind

  const extOf = (name) => {
    const m = /\.([a-z0-9]{1,6})$/i.exec(String(name || ''))
    return m ? m[1].toLowerCase() : ''
  }
  const baseName = (url) => {
    try {
      const p = new URL(url).pathname
      return decodeURIComponent(p.slice(p.lastIndexOf('/') + 1)) || ''
    } catch { return '' }
  }

  const files = []
  const fileSeen = new Set()
  const pageOrigin = location.origin
  const addFile = (url, name, context, source) => {
    if (!url || files.length >= MAX_FILES) return
    let u
    try { u = new URL(url) } catch { return }
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return
    u.hash = ''
    const key = u.href
    if (fileSeen.has(key)) return
    let n = oneLine(name || baseName(key))
    let ext = extOf(n) || extOf(baseName(key))
    if (!ext) {
      const fmt = (u.searchParams.get('format') || u.searchParams.get('type') || '').toLowerCase()
      if (EXT_KIND[fmt]) ext = fmt
    }
    const kind = EXT_KIND[ext] || (source === 'download' ? 'document' : '')
    if (!kind) return
    if (!n) n = 'file'
    if (ext && !extOf(n)) n = `${n}.${ext}`
    fileSeen.add(key)
    files.push({ url: key, name: clip(n, 120), ext, kind, context: clip(oneLine(context), 90),
                 same_origin: u.origin === pageOrigin, source })
  }

  if (isPdf) addFile(location.href, baseName(location.href) || 'document.pdf', 'This page is a PDF', 'page')

  // ── visibility ──
  const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG', 'CANVAS', 'IFRAME', 'OBJECT', 'EMBED',
    'BUTTON', 'SELECT', 'OPTION', 'INPUT', 'TEXTAREA', 'DIALOG', 'HEAD', 'LINK', 'META', 'MAP', 'AUDIO', 'VIDEO'])
  const CHROME_TAGS = new Set(['NAV', 'ASIDE', 'FOOTER'])
  const INLINE = new Set(['A', 'B', 'STRONG', 'I', 'EM', 'CODE', 'SMALL', 'SUB', 'SUP', 'MARK', 'ABBR', 'CITE', 'Q', 'U', 'S', 'TIME', 'LABEL', 'IMG', 'BR'])
  const hidden = (el) => {
    if (el.hidden || el.getAttribute('aria-hidden') === 'true') return true
    if (INLINE.has(el.tagName)) return false
    const cs = getComputedStyle(el)
    return cs.display === 'none' || cs.visibility === 'hidden'
  }

  // ── where the page's own text is ──
  const textLen = (el) => (el && el.innerText ? el.innerText.length : 0)
  const bodyLen = textLen(document.body)
  const pickRoot = () => {
    let best = null
    let bestLen = 0
    for (const c of document.querySelectorAll('article, main, [role="main"], [itemprop="articleBody"]')) {
      if (hidden(c)) continue
      const l = textLen(c)
      if (l > bestLen) { best = c; bestLen = l }
    }
    if (best && bestLen > 400 && bestLen > bodyLen * 0.2) return best
    // Paragraph density: the block holding most of the page's prose.
    const score = new Map()
    for (const p of document.querySelectorAll('p, pre, li, td')) {
      const t = (p.innerText || '').trim()
      if (t.length < 50) continue
      let el = p.parentElement
      let weight = 1
      for (let up = 0; el && up < 3; up++, el = el.parentElement, weight /= 2) {
        score.set(el, (score.get(el) || 0) + t.length * weight)
      }
    }
    let top = null
    let topScore = 0
    for (const [el, s] of score) if (s > topScore && el !== document.documentElement) { top = el; topScore = s }
    return top && topScore > 800 && top !== document.body ? top : document.body
  }

  // ── tables ──
  const tables = []
  const tableIndex = new Map()
  for (const t of document.querySelectorAll('table')) {
    if (tables.length >= MAX_TABLES) break
    if (t.querySelector('table') || hidden(t)) continue
    const rows = []
    let width = 0
    for (const tr of t.rows) {
      if (rows.length >= MAX_ROWS) break
      const cells = []
      for (const c of tr.cells) {
        if (cells.length >= MAX_COLS) break
        cells.push(oneLine(c.innerText))
        const span = Math.min(Number(c.colSpan) || 1, 20)
        for (let k = 1; k < span && cells.length < MAX_COLS; k++) cells.push('')
      }
      if (cells.some((c) => c)) rows.push(cells)
      width = Math.max(width, cells.length)
    }
    if (rows.length < 2 || width < 2) continue
    let caption = t.caption ? oneLine(t.caption.innerText) : ''
    if (!caption) {
      let prev = t.previousElementSibling
      for (let k = 0; prev && k < 3; k++, prev = prev.previousElementSibling) {
        if (/^H[1-6]$/.test(prev.tagName)) { caption = oneLine(prev.innerText); break }
      }
    }
    tableIndex.set(t, tables.length + 1)
    tables.push({ caption: clip(caption, 120), rows, width })
  }

  // ── Markdown ──
  let nodes = 0
  const tableMd = (t) => {
    const n = tableIndex.get(t)
    if (!n) return ''
    const tb = tables[n - 1]
    if (tb.rows.length > 120 || tb.width > 10) {
      return `(Table ${n}${tb.caption ? ` — ${tb.caption}` : ''}: ${tb.rows.length} rows × ${tb.width} columns, in tables/table-${n}.csv)`
    }
    const pad = (r) => Array.from({ length: tb.width }, (_, i) => (r[i] || '').replace(/\|/g, '\\|'))
    const lines = [`| ${pad(tb.rows[0]).join(' | ')} |`, `| ${Array(tb.width).fill('---').join(' | ')} |`]
    for (const r of tb.rows.slice(1)) lines.push(`| ${pad(r).join(' | ')} |`)
    return lines.join('\n')
  }

  const md = (node, ctx, root) => {
    if (++nodes > MAX_NODES) return ''
    if (node.nodeType === 3) return ctx.pre ? node.nodeValue : node.nodeValue.replace(/\s+/g, ' ')
    if (node.nodeType !== 1) return ''
    const el = node
    const tag = el.tagName
    if (SKIP.has(tag)) return ''
    if (el !== root && (CHROME_TAGS.has(tag) || el.getAttribute('role') === 'navigation')) return ''
    if (el !== root && hidden(el)) return ''
    const kids = (c = ctx) => {
      let s = ''
      for (const ch of el.childNodes) s += md(ch, c, root)
      return s
    }
    switch (tag) {
      case 'H1': case 'H2': case 'H3': case 'H4': case 'H5': case 'H6': {
        const t = oneLine(kids())
        return t ? `\n\n${'#'.repeat(Number(tag[1]))} ${t}\n\n` : ''
      }
      case 'P': return `\n\n${kids().trim()}\n\n`
      case 'BR': return '\n'
      case 'HR': return '\n\n---\n\n'
      case 'B': case 'STRONG': { const t = kids().trim(); return t ? `**${t}**` : '' }
      case 'I': case 'EM': { const t = kids().trim(); return t ? `*${t}*` : '' }
      case 'CODE': return ctx.pre ? el.textContent : '`' + (el.textContent || '').replace(/`/g, "'") + '`'
      case 'PRE': return '\n\n```\n' + (el.textContent || '').replace(/\n+$/, '') + '\n```\n\n'
      case 'A': {
        const href = el.getAttribute('href') || ''
        const t = oneLine(kids())
        if (!href || href.startsWith('#') || /^javascript:/i.test(href)) return t
        const u = abs(href)
        if (!t) return ''
        return `[${t.replace(/[[\]]/g, '')}](${u})`
      }
      case 'IMG': {
        const src = el.currentSrc || el.getAttribute('src') || ''
        const alt = oneLine(el.getAttribute('alt'))
        const w = el.naturalWidth || el.width || 0
        if (!src || src.startsWith('data:') || (w && w < 64 && !alt)) return ''
        return `![${alt.replace(/[[\]]/g, '')}](${abs(src)})`
      }
      case 'UL': case 'OL': {
        const depth = ctx.depth || 0
        const lines = []
        let i = 1
        for (const li of el.children) {
          if (li.tagName !== 'LI') continue
          let body = ''
          for (const ch of li.childNodes) body += md(ch, { ...ctx, depth: depth + 1 }, root)
          body = body.trim().replace(/\n{2,}/g, '\n')
          if (!body) continue
          const marker = tag === 'OL' ? `${i++}.` : '-'
          lines.push('  '.repeat(depth) + marker + ' ' + body.replace(/\n/g, '\n' + '  '.repeat(depth + 1)))
        }
        return lines.length ? `\n\n${lines.join('\n')}\n\n` : ''
      }
      case 'BLOCKQUOTE': {
        const t = kids().trim()
        return t ? '\n\n' + t.split('\n').map((l) => '> ' + l).join('\n') + '\n\n' : ''
      }
      case 'TABLE': return `\n\n${tableMd(el) || kids()}\n\n`
      case 'FIGCAPTION': { const t = kids().trim(); return t ? `\n\n*${t}*\n\n` : '' }
      case 'DT': return `\n\n**${kids().trim()}**\n`
      case 'DD': return `${kids().trim()}\n`
      case 'DIV': case 'SECTION': case 'ARTICLE': case 'MAIN': case 'HEADER': case 'FIGURE': case 'FORM':
      case 'FIELDSET': case 'DETAILS': case 'SUMMARY': case 'DL': case 'ADDRESS': case 'TR': case 'LI':
        return `\n${kids()}\n`
      case 'TD': case 'TH': return ` ${kids()} `
      default: return kids()
    }
  }
  const tidy = (s) => s.replace(/[ \t]+\n/g, '\n').replace(/\n[ \t]+/g, '\n').replace(/\n{3,}/g, '\n\n').trim()

  const root = pickRoot()
  let markdown = isPdf ? '' : tidy(md(root, {}, root))
  if (!isPdf && root !== document.body && markdown.length < 1500 && bodyLen > markdown.length * 2) {
    nodes = 0
    const rest = tidy(md(document.body, {}, document.body))
    if (rest.length > markdown.length) markdown = `${markdown}\n\n## Everything else on the page\n\n${rest}`
  }
  markdown = clip(markdown, MAX_MD)

  // ── selection ──
  let selection = null
  const sel = window.getSelection ? window.getSelection() : null
  if (sel && sel.rangeCount && !sel.isCollapsed) {
    const box = document.createElement('div')
    box.appendChild(sel.getRangeAt(0).cloneContents())
    nodes = 0
    const text = String(sel.toString() || '').slice(0, 200000)
    const selMd = tidy(md(box, {}, box)) || text
    if (text.trim()) selection = { text, markdown: clip(selMd, 200000) }
  }

  // ── files and links ──
  const links = []
  const linkSeen = new Set()
  for (const a of document.querySelectorAll('a[href]')) {
    const href = a.getAttribute('href') || ''
    if (!href || href.startsWith('#') || /^(javascript|mailto|tel):/i.test(href)) continue
    const u = abs(href)
    if (!/^https?:/i.test(u)) continue
    const text = oneLine(a.innerText || a.getAttribute('aria-label') || a.title || '')
    if (a.hasAttribute('download')) addFile(u, a.getAttribute('download') || '', text, 'download')
    else addFile(u, '', text, 'link')
    const key = u.split('#')[0]
    if (!linkSeen.has(key) && links.length < MAX_LINKS && !fileSeen.has(key)) {
      linkSeen.add(key)
      links.push({ url: key, text: clip(text, 120) })
    }
  }
  for (const el of document.querySelectorAll('embed[src], object[data], iframe[src]')) {
    const u = abs(el.getAttribute('src') || el.getAttribute('data') || '')
    if (/\.pdf($|[?#])/i.test(u)) addFile(u, baseName(u), 'Shown in the page', 'embed')
  }
  for (const el of document.querySelectorAll('video[src], audio[src], video source[src], audio source[src]')) {
    const u = abs(el.getAttribute('src') || '')
    addFile(u, baseName(u), 'Played in the page', 'media')
  }
  const images = []
  const imgSeen = new Set()
  for (const img of document.images) {
    if (images.length >= MAX_IMAGES) break
    const src = img.currentSrc || img.src || ''
    if (!/^https?:/i.test(src) || imgSeen.has(src)) continue
    if ((img.naturalWidth || 0) < 500 && (img.naturalHeight || 0) < 500) continue
    imgSeen.add(src)
    let name = baseName(src) || 'image'
    if (!extOf(name)) name += '.jpg'
    images.push({ url: src, name: clip(name, 120), ext: extOf(name), kind: 'image',
                  context: clip(oneLine(img.alt || ''), 90), same_origin: new URL(src).origin === pageOrigin,
                  source: 'img', width: img.naturalWidth, height: img.naturalHeight })
  }

  const words = (markdown.match(/\S+/g) || []).length
  return {
    ok: true,
    meta,
    pdf: isPdf,
    markdown,
    words,
    selection,
    tables,
    files,
    images,
    links,
    structured,
    structured_types: Array.from(ldTypes).slice(0, 20),
    scroll: { y: Math.round(window.scrollY), height: Math.round(document.documentElement.scrollHeight) },
  }
})()
