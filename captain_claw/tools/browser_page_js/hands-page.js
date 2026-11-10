// Captain for Chrome — page helpers for an agent's browser tool.
//
// The service worker (hands.js) evaluates this through the debugger in an
// isolated world of the page, so the page's own scripts can neither see nor
// change any of it. It describes the page with refs (ref_12) an agent can
// point at, finds things, fills fields and says where an element is, so the
// worker can click it with real input events. It defines
// globalThis.__captainHands once per document; refs last until the page
// navigates.

(() => {
  if (globalThis.__captainHands) return

  const refs = new Map()      // 'ref_12' → WeakRef(element)
  const ids = new WeakMap()   // element → 'ref_12'
  let next = 1

  const MAX_WALK = 12000
  const SKIP_TAGS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'HEAD', 'META', 'LINK', 'BR', 'WBR', 'PATH', 'DEFS', 'SYMBOL', 'USE'])
  const INTERACTIVE_ROLES = new Set(['button', 'link', 'checkbox', 'radio', 'textbox', 'searchbox', 'combobox', 'listbox',
    'option', 'slider', 'spinbutton', 'switch', 'tab', 'menuitem', 'menuitemcheckbox', 'menuitemradio', 'treeitem'])
  const TEXT_ROLES = new Set(['heading', 'paragraph', 'listitem', 'cell', 'columnheader', 'rowheader', 'label', 'caption', 'term', 'definition', 'status', 'alert'])
  const SHOWN_ROLES = new Set(['navigation', 'main', 'banner', 'contentinfo', 'complementary', 'form', 'dialog', 'alertdialog', 'search',
    'table', 'row', 'list', 'img', 'iframe', 'region', 'tablist', 'menu', 'menubar', 'toolbar', 'grid', 'tree', 'group', 'figure'])

  const oneLine = (s) => String(s == null ? '' : s).replace(/\s+/g, ' ').trim()
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s)
  const q = (s) => JSON.stringify(s)

  function refOf(el) {
    let id = ids.get(el)
    if (!id) {
      id = `ref_${next++}`
      ids.set(el, id)
      refs.set(id, new WeakRef(el))
    }
    return id
  }

  function byRef(ref) {
    const id = String(ref || '').trim()
    if (!id) throw new Error('Give a ref (like ref_12) from read_page or find.')
    const el = refs.get(id)?.deref()
    if (!el || !el.isConnected) throw new Error(`${id} isn't on the page any more (it changed or navigated) — read the page again for fresh refs.`)
    return el
  }

  function roleOf(el) {
    const explicit = (el.getAttribute('role') || '').trim().split(/\s+/)[0].toLowerCase()
    if (explicit && explicit !== 'presentation' && explicit !== 'none' && explicit !== 'generic') return explicit
    const tag = el.tagName
    switch (tag) {
      case 'A': case 'AREA': return el.hasAttribute('href') ? 'link' : ''
      case 'BUTTON': case 'SUMMARY': return 'button'
      case 'SELECT': return el.multiple || el.size > 1 ? 'listbox' : 'combobox'
      case 'TEXTAREA': return 'textbox'
      case 'OPTION': return 'option'
      case 'IMG': return el.getAttribute('alt') === '' ? '' : 'img'
      case 'H1': case 'H2': case 'H3': case 'H4': case 'H5': case 'H6': return 'heading'
      case 'NAV': return 'navigation'
      case 'MAIN': return 'main'
      case 'HEADER': return el.closest('article,section,main,aside,nav') ? '' : 'banner'
      case 'FOOTER': return el.closest('article,section,main,aside,nav') ? '' : 'contentinfo'
      case 'ASIDE': return 'complementary'
      case 'FORM': return 'form'
      case 'DIALOG': return 'dialog'
      case 'TABLE': return 'table'
      case 'TR': return 'row'
      case 'TD': return 'cell'
      case 'TH': return el.scope === 'row' ? 'rowheader' : 'columnheader'
      case 'CAPTION': return 'caption'
      case 'UL': case 'OL': return 'list'
      case 'LI': return 'listitem'
      case 'DT': return 'term'
      case 'DD': return 'definition'
      case 'P': return 'paragraph'
      case 'LABEL': return 'label'
      case 'FIGURE': return 'figure'
      case 'FIELDSET': return 'group'
      case 'IFRAME': case 'FRAME': return 'iframe'
      case 'INPUT': {
        const t = (el.type || 'text').toLowerCase()
        if (t === 'hidden') return ''
        if (t === 'button' || t === 'submit' || t === 'reset' || t === 'image') return 'button'
        if (t === 'checkbox') return 'checkbox'
        if (t === 'radio') return 'radio'
        if (t === 'range') return 'slider'
        if (t === 'number') return 'spinbutton'
        if (t === 'search') return 'searchbox'
        if (t === 'file') return 'file'
        return 'textbox'
      }
    }
    if (el.isContentEditable && el.hasAttribute('contenteditable')) return 'textbox'
    return ''
  }

  function isInteractive(el, role) {
    if (INTERACTIVE_ROLES.has(role) || role === 'file') return true
    if (el.tagName === 'INPUT' && el.type !== 'hidden') return true
    if (el.hasAttribute('onclick')) return true
    const ti = el.getAttribute('tabindex')
    return ti !== null && Number(ti) >= 0 && el !== el.ownerDocument.body
  }

  function styleOf(el) {
    try { return el.ownerDocument.defaultView.getComputedStyle(el) } catch { return null }
  }

  // Clickable for a person even without a role: a pointer cursor its parent doesn't have.
  function looksClickable(el, style) {
    if (!style || style.cursor !== 'pointer') return false
    const ps = el.parentElement ? styleOf(el.parentElement) : null
    return !ps || ps.cursor !== 'pointer'
  }

  function textOfIds(el, attr) {
    const doc = el.ownerDocument
    return oneLine((el.getAttribute(attr) || '').split(/\s+/).map((id) => doc.getElementById(id)?.innerText || '').join(' '))
  }

  function nameOf(el, role) {
    const aria = oneLine(el.getAttribute('aria-label'))
    if (aria) return clip(aria, 140)
    if (el.hasAttribute('aria-labelledby')) {
      const t = textOfIds(el, 'aria-labelledby')
      if (t) return clip(t, 140)
    }
    const tag = el.tagName
    if (el.labels && el.labels.length) {
      const t = oneLine(el.labels[0].innerText)
      if (t) return clip(t, 140)
    }
    if (tag === 'IMG' || tag === 'AREA') return clip(oneLine(el.getAttribute('alt') || el.getAttribute('title')), 140)
    if (tag === 'INPUT') {
      const t = (el.type || '').toLowerCase()
      if (t === 'submit' || t === 'button' || t === 'reset') return clip(oneLine(el.value) || (t === 'submit' ? 'Submit' : t === 'reset' ? 'Reset' : ''), 140)
      if (t === 'image') return clip(oneLine(el.alt || el.value || 'Submit'), 140)
    }
    if (tag === 'INPUT' || tag === 'TEXTAREA' || role === 'textbox' || role === 'searchbox' || role === 'combobox') {
      const t = oneLine(el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('name'))
      if (t) return clip(t, 140)
    }
    if (tag === 'SELECT') return clip(oneLine(el.getAttribute('title') || el.getAttribute('name')), 140)
    if (tag === 'IFRAME' || tag === 'FRAME') return clip(oneLine(el.getAttribute('title') || el.getAttribute('name')), 140)
    const title = oneLine(el.getAttribute('title'))
    const text = oneLine(el.innerText || el.textContent)
    if (text) return clip(text, TEXT_ROLES.has(role) ? 240 : 140)
    return clip(title, 140)
  }

  function shortHref(el) {
    const raw = el.getAttribute('href') || ''
    if (!raw || raw.startsWith('javascript:')) return ''
    try {
      const u = new URL(raw, el.ownerDocument.baseURI)
      const here = new URL(el.ownerDocument.baseURI)
      return clip(u.origin === here.origin ? (u.pathname + u.search + u.hash) : u.href, 140)
    } catch { return clip(raw, 140) }
  }

  function extras(el, role) {
    const out = []
    const tag = el.tagName
    if (role === 'link') {
      const h = shortHref(el)
      if (h) out.push(`href=${q(h)}`)
      if (el.target === '_blank') out.push('new-tab')
    }
    if (role === 'heading') out.push(`level=${/^H[1-6]$/.test(tag) ? tag[1] : (el.getAttribute('aria-level') || '2')}`)
    if (tag === 'INPUT' || tag === 'TEXTAREA') {
      const t = (el.type || 'text').toLowerCase()
      if (tag === 'INPUT' && !['text', 'checkbox', 'radio', 'submit', 'button', 'search'].includes(t)) out.push(`type=${t}`)
      if (t === 'checkbox' || t === 'radio') out.push(el.checked ? 'checked' : 'unchecked')
      else if (t === 'password') out.push(el.value ? 'value=(filled, hidden)' : 'empty')
      else if (t === 'file') out.push(el.files && el.files.length ? `files=${q(Array.from(el.files).map((f) => f.name).join(', '))}` : 'no file')
      else if (!['submit', 'button', 'reset', 'image'].includes(t)) {
        out.push(el.value ? `value=${q(clip(oneLine(el.value), 120))}` : 'empty')
        const ph = oneLine(el.getAttribute('placeholder'))
        if (ph && ph !== nameOf(el, role)) out.push(`placeholder=${q(clip(ph, 60))}`)
      }
    } else if (tag === 'SELECT') {
      const sel = Array.from(el.selectedOptions || []).map((o) => oneLine(o.text)).join(', ')
      out.push(`selected=${q(clip(sel, 80))}`)
      const opts = Array.from(el.options || []).slice(0, 12).map((o) => oneLine(o.text))
      if (opts.length) out.push(`options=${q(clip(opts.join(' | '), 240))}${el.options.length > 12 ? ` (+${el.options.length - 12} more)` : ''}`)
    } else if (role === 'textbox' && el.isContentEditable) {
      const t = oneLine(el.innerText)
      out.push(t ? `text=${q(clip(t, 120))}` : 'empty')
    }
    const ac = el.getAttribute('aria-checked')
    if (ac && !(tag === 'INPUT')) out.push(ac === 'true' ? 'checked' : ac === 'mixed' ? 'mixed' : 'unchecked')
    const ae = el.getAttribute('aria-expanded')
    if (ae) out.push(ae === 'true' ? 'expanded' : 'collapsed')
    if (el.getAttribute('aria-selected') === 'true') out.push('selected')
    if (el.getAttribute('aria-current') && el.getAttribute('aria-current') !== 'false') out.push('current')
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') out.push('disabled')
    if (el.required || el.getAttribute('aria-required') === 'true') out.push('required')
    if (el.readOnly) out.push('readonly')
    if (role === 'iframe') {
      let other = false
      try { other = !el.contentDocument } catch { other = true }
      const src = el.getAttribute('src') || ''
      if (src) { try { out.push(`src=${q(new URL(src, el.ownerDocument.baseURI).host)}`) } catch { /* not a URL */ } }
      if (other) out.push('(another site: not read — take a screenshot to see it, click by coordinate)')
    }
    return out.join(' ')
  }

  // Where *el* is in the top page's viewport (same-origin frames added up).
  function rectInTop(el) {
    const r = el.getBoundingClientRect()
    let x = r.left
    let y = r.top
    let win = el.ownerDocument.defaultView
    while (win && win !== win.top) {
      let fe = null
      try { fe = win.frameElement } catch { fe = null }
      if (!fe) break
      const fr = fe.getBoundingClientRect()
      x += fr.left + (fe.clientLeft || 0)
      y += fr.top + (fe.clientTop || 0)
      win = fe.ownerDocument.defaultView
    }
    return { left: x, top: y, width: r.width, height: r.height }
  }

  function inViewport(r) {
    return r.top < innerHeight && r.left < innerWidth && r.top + r.height > 0 && r.left + r.width > 0
  }

  function kids(el) {
    if (el.shadowRoot) return Array.from(el.shadowRoot.children)
    if (el.tagName === 'SLOT') {
      const assigned = el.assignedElements ? el.assignedElements({ flatten: true }) : []
      return assigned.length ? assigned : Array.from(el.children)
    }
    if (el.tagName === 'IFRAME' || el.tagName === 'FRAME') {
      try {
        const body = el.contentDocument && el.contentDocument.body
        return body ? [body] : []
      } catch { return [] }
    }
    return Array.from(el.children)
  }

  function ownText(el) {
    const nodes = el.shadowRoot ? el.shadowRoot.childNodes : el.childNodes
    let s = ''
    for (const n of nodes) if (n.nodeType === 3) s += ' ' + n.nodeValue
    return oneLine(s)
  }

  const HAS_INTERACTIVE = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[role=button],[role=link],[role=checkbox],[role=radio],[role=tab],[role=menuitem],[role=switch],[role=textbox],[role=combobox],[contenteditable],[onclick]'

  function header() {
    const sh = Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0)
    return [
      `Page: ${clip(oneLine(document.title) || '(no title)', 160)} — ${location.href}`,
      `Viewport ${innerWidth}×${innerHeight} CSS px, scrolled to ${Math.round(scrollY)} of ${Math.round(sh)}${scrollX ? `, x ${Math.round(scrollX)}` : ''}`,
    ]
  }

  /** The page as a tree of what matters, every element with a ref. */
  function tree({ filter = 'interactive', refId = '', maxChars = 40000 } = {}) {
    const onlyInteractive = filter !== 'all'
    const lines = header()
    let size = lines.join('\n').length
    let walked = 0
    let cut = false
    let shown = 0
    const push = (s) => {
      if (cut) return
      if (size + s.length + 1 > maxChars) { cut = true; return }
      lines.push(s)
      size += s.length + 1
    }
    const start = refId ? byRef(refId) : document.body
    if (!start) return lines.join('\n') + '\n(the page has no body yet — it may still be loading)'
    if (refId) push(`Inside ${refId}:`)

    const walk = (el, depth) => {
      if (cut || walked > MAX_WALK) { cut = true; return }
      walked++
      if (SKIP_TAGS.has(el.tagName) || el.getAttribute('aria-hidden') === 'true') return
      const style = styleOf(el)
      if (style && style.display === 'none') return
      const hiddenHere = style && (style.visibility === 'hidden' || style.visibility === 'collapse')
      const role = roleOf(el)
      const r = el.getBoundingClientRect()
      const sized = r.width > 0 || r.height > 0 || (style && style.display === 'contents')
      const interactive = !hiddenHere && sized && (isInteractive(el, role) || looksClickable(el, style))
      if (onlyInteractive) {
        if (interactive) {
          const name = nameOf(el, role)
          const off = inViewport(rectInTop(el)) ? '' : ' (off-screen)'
          push(`- ${role || 'clickable'}${name ? ' ' + q(name) : ''} [${refOf(el)}]${extras(el, role) ? ' ' + extras(el, role) : ''}${off}`)
          shown++
          if (el.tagName === 'SELECT' || el.tagName === 'TEXTAREA') return
        }
        for (const c of kids(el)) walk(c, 0)
        return
      }
      let shownHere = false
      if (!hiddenHere && sized) {
        if (interactive) {
          const name = nameOf(el, role)
          push(`${'  '.repeat(depth)}- ${role || 'clickable'}${name ? ' ' + q(name) : ''} [${refOf(el)}]${extras(el, role) ? ' ' + extras(el, role) : ''}`)
          shownHere = true
          shown++
          if (el.tagName === 'SELECT' || el.tagName === 'TEXTAREA' || el.tagName === 'BUTTON' || role === 'link') {
            if (!el.querySelector(HAS_INTERACTIVE)) return
          }
        } else if (TEXT_ROLES.has(role) && !el.querySelector(HAS_INTERACTIVE)) {
          const name = nameOf(el, role)
          if (name) push(`${'  '.repeat(depth)}- ${role} ${q(name)} [${refOf(el)}]${role === 'heading' ? ' ' + extras(el, role) : ''}`)
          return
        } else if (SHOWN_ROLES.has(role) || TEXT_ROLES.has(role)) {
          const name = role === 'img' || role === 'iframe' || role === 'dialog' || role === 'region' || role === 'form' ? nameOf(el, role) : ''
          if (role !== 'img' || name) {
            push(`${'  '.repeat(depth)}- ${role}${name ? ' ' + q(name) : ''} [${refOf(el)}]${role === 'iframe' ? ' ' + extras(el, role) : ''}`)
            shownHere = true
          }
        } else {
          const t = ownText(el)
          if (t && t.length > 1) push(`${'  '.repeat(depth)}- text ${q(clip(t, 240))}`)
        }
      }
      for (const c of kids(el)) walk(c, depth + (shownHere ? 1 : 0))
    }
    walk(start, 0)
    if (shown === 0 && onlyInteractive) lines.push('(nothing to click or fill here — try filter "all" or a screenshot)')
    if (cut) lines.push(`… (cut at ${maxChars.toLocaleString()} characters — read one part with ref_id, or use find)`)
    return lines.join('\n')
  }

  /** Elements matching *query* (words in their role, name, text, placeholder, value…). */
  function find({ query = '', max = 20 } = {}) {
    const words = oneLine(query).toLowerCase().split(' ').filter(Boolean)
    if (!words.length) throw new Error('Say what to find (query).')
    const hits = []
    let walked = 0
    const visit = (el) => {
      if (walked++ > MAX_WALK) return
      if (SKIP_TAGS.has(el.tagName) || el.getAttribute('aria-hidden') === 'true') return
      const style = styleOf(el)
      if (style && style.display === 'none') return
      const role = roleOf(el)
      const interactive = isInteractive(el, role) || looksClickable(el, style)
      const leaf = !el.firstElementChild
      if (interactive || role || leaf) {
        const r = rectInTop(el)
        if (r.width > 0 || r.height > 0) {
          const name = nameOf(el, role)
          const hay = [role, name, el.getAttribute('placeholder'), el.value && typeof el.value === 'string' ? el.value : '',
            el.getAttribute('title'), el.getAttribute('name'), el.id, el.getAttribute('aria-label'),
            el.tagName === 'A' ? el.getAttribute('href') : ''].map((s) => oneLine(s).toLowerCase()).join(' ')
          const matched = words.filter((w) => hay.includes(w)).length
          if (matched) {
            const all = matched === words.length
            hits.push({ el, role, name, score: matched + (all ? 10 : 0) + (interactive ? 2 : 0) + (inViewport(r) ? 0.5 : 0) - (name.length > 120 ? 1 : 0) })
          }
        }
      }
      for (const c of kids(el)) visit(c)
    }
    if (document.body) visit(document.body)
    // A container whose child matches just as well is noise.
    const kept = hits.filter((h) => !hits.some((o) => o !== h && o.score >= h.score && h.el.contains(o.el)))
    kept.sort((a, b) => b.score - a.score)
    const best = kept.length && kept[0].score >= 10 ? kept.filter((h) => h.score >= 10) : kept
    const out = best.slice(0, Math.max(1, Math.min(50, max))).map((h) => {
      const off = inViewport(rectInTop(h.el)) ? '' : ' (off-screen)'
      const ex = extras(h.el, h.role)
      return `- ${h.role || h.el.tagName.toLowerCase()}${h.name ? ' ' + q(h.name) : ''} [${refOf(h.el)}]${ex ? ' ' + ex : ''}${off}`
    })
    if (!out.length) return `Nothing on the page matches ${q(query)}.`
    return `${out.length} match${out.length === 1 ? '' : 'es'} for ${q(query)}${best.length > out.length ? ` (of ${best.length})` : ''}:\n${out.join('\n')}`
  }

  function describe(el) {
    const role = roleOf(el)
    const name = nameOf(el, role)
    return `${role || el.tagName.toLowerCase()}${name ? ' ' + q(clip(name, 80)) : ''}`
  }

  function centerOf(el) {
    let target = el
    let r = rectInTop(el)
    if ((r.width < 1 || r.height < 1) && el.labels && el.labels.length) {
      target = el.labels[0]
      r = rectInTop(target)
    }
    if ((r.width < 1 || r.height < 1) && el.parentElement) {
      target = el.parentElement
      r = rectInTop(target)
    }
    return { target, r }
  }

  /** Where to click *el*: scrolled into view, its middle, and whether something covers it. */
  function pointAt(el) {
    let { r } = centerOf(el)
    const vw = innerWidth
    const vh = innerHeight
    if (!(r.top >= 0 && r.left >= 0 && r.top + r.height <= vh && r.left + r.width <= vw)) {
      el.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' })
      r = centerOf(el).r
    }
    const x = Math.round(Math.min(Math.max(r.left + r.width / 2, 1), vw - 1))
    const y = Math.round(Math.min(Math.max(r.top + r.height / 2, 1), vh - 1))
    let covered = ''
    if (el.ownerDocument === document) {
      const hit = document.elementFromPoint(x, y)
      if (hit && hit !== el && !el.contains(hit) && !hit.contains(el) && !(el.labels && Array.from(el.labels).some((l) => l.contains(hit)))) {
        covered = describe(hit)
      }
    }
    return { x, y, width: Math.round(r.width), height: Math.round(r.height), what: describe(el), ref: refOf(el), covered,
             visible: r.width > 0 && r.height > 0 }
  }

  function point({ ref } = {}) {
    return pointAt(byRef(ref))
  }

  /** The old way of pointing: a CSS selector, visible text, or a role and its name, nth for the n-th. */
  function pointBy({ selector = '', text = '', role = '', nth = null } = {}) {
    let found = []
    const want = oneLine(text).toLowerCase()
    if (selector) {
      try { found = Array.from(document.querySelectorAll(selector)) } catch { throw new Error(`${q(selector)} isn't a CSS selector.`) }
    } else if (role) {
      const all = document.body ? document.body.querySelectorAll('*') : []
      for (const el of all) {
        if (roleOf(el) !== role.toLowerCase()) continue
        if (want && !nameOf(el, role).toLowerCase().includes(want)) continue
        found.push(el)
      }
    } else if (want) {
      const all = document.body ? document.body.querySelectorAll('*') : []
      const exact = []
      const partial = []
      for (const el of all) {
        if (SKIP_TAGS.has(el.tagName)) continue
        const r = el.getBoundingClientRect()
        if (!r.width && !r.height) continue
        const t = oneLine(el.innerText || el.value || el.getAttribute('aria-label') || '').toLowerCase()
        if (!t || t.length > 400) continue
        if (t === want) exact.push(el)
        else if (t.includes(want)) partial.push(el)
      }
      const leafy = (list) => list.filter((el) => !list.some((o) => o !== el && el.contains(o)))
      found = leafy(exact).length ? leafy(exact) : leafy(partial)
      found.sort((a, b) => Number(isInteractive(b, roleOf(b))) - Number(isInteractive(a, roleOf(a))))
    } else {
      throw new Error('Point at something: ref (best), coordinate, selector, text, or role.')
    }
    const visible = found.filter((el) => { const r = el.getBoundingClientRect(); return r.width > 0 || r.height > 0 })
    const list = visible.length ? visible : found
    if (!list.length) throw new Error(`Nothing on the page matches ${selector ? q(selector) : role ? `role=${role}${want ? ` named ${q(text)}` : ''}` : q(text)}.`)
    const i = nth === null || nth === undefined ? 0 : Number(nth)
    if (i < 0 || i >= list.length) throw new Error(`There are ${list.length} matches; nth=${i} is out of range.`)
    const p = pointAt(list[i])
    if (list.length > 1 && (nth === null || nth === undefined)) p.note = `${list.length} matched; used the first (nth=0).`
    return p
  }

  function scrollTo({ ref } = {}) {
    const el = byRef(ref)
    el.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' })
    return `Scrolled to ${describe(el)} [${refOf(el)}]; the page is at ${Math.round(scrollY)} px.`
  }

  function truthy(v) {
    if (typeof v === 'boolean') return v
    return !['false', '0', 'no', 'off', 'unchecked', ''].includes(String(v).trim().toLowerCase())
  }

  function fire(el, kind, data) {
    const ev = kind === 'input'
      ? new InputEvent('input', { bubbles: true, composed: true, inputType: 'insertReplacementText', data: data ?? null })
      : new Event(kind, { bubbles: true, composed: true })
    el.dispatchEvent(ev)
  }

  /** Set a field to *value*: a text field, a select (by value or label), a checkbox or radio, a contenteditable. */
  function formInput({ ref, value } = {}) {
    const el = byRef(ref)
    const tag = el.tagName
    const role = roleOf(el)
    if (el.disabled) throw new Error(`${describe(el)} is disabled.`)
    if (tag === 'SELECT') {
      const want = Array.isArray(value) ? value.map((v) => oneLine(v).toLowerCase()) : [oneLine(value).toLowerCase()]
      const opts = Array.from(el.options)
      const pick = (w) => opts.find((o) => o.value.toLowerCase() === w) || opts.find((o) => oneLine(o.text).toLowerCase() === w)
        || opts.find((o) => oneLine(o.text).toLowerCase().includes(w))
      const chosen = want.map(pick).filter(Boolean)
      if (!chosen.length) throw new Error(`No option ${q(String(value))} — the options are: ${opts.slice(0, 30).map((o) => oneLine(o.text)).join(' | ')}`)
      el.focus()
      if (el.multiple) for (const o of opts) o.selected = chosen.includes(o)
      else el.value = chosen[0].value
      fire(el, 'input')
      fire(el, 'change')
      return `Picked ${chosen.map((o) => q(oneLine(o.text))).join(', ')} in ${describe(el)}.`
    }
    if (tag === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio')) {
      const want = truthy(value)
      if (el.checked !== want) el.click()
      return `${describe(el)} is now ${el.checked ? 'checked' : 'unchecked'}.`
    }
    if (el.hasAttribute('aria-checked') && (role === 'checkbox' || role === 'switch' || role === 'radio' || role === 'menuitemcheckbox')) {
      const want = truthy(value)
      if ((el.getAttribute('aria-checked') === 'true') !== want) el.click()
      return `${describe(el)} is now ${el.getAttribute('aria-checked') === 'true' ? 'on' : 'off'}.`
    }
    if (tag === 'INPUT' && el.type === 'file') throw new Error('That is a file field: use upload_file with a path.')
    if (tag === 'INPUT' || tag === 'TEXTAREA') {
      if (el.readOnly) throw new Error(`${describe(el)} is read-only.`)
      const text = String(value ?? '')
      el.focus()
      el.value = text
      fire(el, 'input', text)
      fire(el, 'change')
      return el.type === 'password' ? `Filled ${describe(el)}.` : `${describe(el)} is now ${q(clip(el.value, 200))}.`
    }
    if (el.isContentEditable) {
      el.focus()
      const sel = el.ownerDocument.getSelection()
      const range = el.ownerDocument.createRange()
      range.selectNodeContents(el)
      sel.removeAllRanges()
      sel.addRange(range)
      el.ownerDocument.execCommand('insertText', false, String(value ?? ''))
      return `${describe(el)} now reads ${q(clip(oneLine(el.innerText), 200))}.`
    }
    throw new Error(`${describe(el)} [${refOf(el)}] isn't something to fill in — click it, or type after clicking.`)
  }

  /** Put a file the agent sent into a file field. */
  function upload({ ref, name = 'file', type = 'application/octet-stream', data = '' } = {}) {
    const el = byRef(ref)
    if (!(el.tagName === 'INPUT' && el.type === 'file')) throw new Error(`${describe(el)} isn't a file field (input type=file). Find the file field with read_page.`)
    const bin = atob(data)
    const bytes = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
    const dt = new DataTransfer()
    for (const f of Array.from(el.multiple ? el.files || [] : [])) dt.items.add(f)
    dt.items.add(new File([bytes], name, { type }))
    el.files = dt.files
    fire(el, 'input')
    fire(el, 'change')
    return `Put ${q(name)} (${bytes.length.toLocaleString()} bytes) in ${describe(el)}.`
  }

  function info() {
    return {
      url: location.href,
      title: oneLine(document.title),
      vw: innerWidth,
      vh: innerHeight,
      sx: Math.round(scrollX),
      sy: Math.round(scrollY),
      sh: Math.round(Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0)),
      dpr: devicePixelRatio || 1,
      ready: document.readyState,
    }
  }

  globalThis.__captainHands = { tree, find, point, pointBy, scrollTo, formInput, upload, info }
})()
