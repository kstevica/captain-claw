// Captain for Chrome — the recorder, in the page.
//
// Injected (every frame it may reach) while the person records what they do:
// it tells the service worker each thing done — a click, text typed into a
// field, a choice in a list, a tick, a file picked, Enter or Escape — with
// what it was done to, named the way a person would name it (its role and
// label), plus a selector and what it sits in as hints. Passwords and other
// secret fields are noted, never their values. Nothing is read from the page
// beyond the element acted on.
//
// Idempotent: injected again (a navigation, a new frame) it only resumes.

(() => {
  if (globalThis.__captainRecorder) {
    globalThis.__captainRecorder.resume()
    return 'resumed'
  }

  let on = true
  let pending = null            // text being typed: {el, value}
  const sent = new WeakMap()    // field → the value last told
  const TOP = window.top === window

  const oneLine = (s) => String(s == null ? '' : s).replace(/\s+/g, ' ').trim()
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s)

  // ── naming an element ──

  function roleOf(el) {
    const explicit = (el.getAttribute('role') || '').trim().split(/\s+/)[0].toLowerCase()
    if (explicit && explicit !== 'presentation' && explicit !== 'none' && explicit !== 'generic') return explicit
    switch (el.tagName) {
      case 'A': case 'AREA': return el.hasAttribute('href') ? 'link' : ''
      case 'BUTTON': case 'SUMMARY': return 'button'
      case 'SELECT': return el.multiple || el.size > 1 ? 'listbox' : 'combobox'
      case 'TEXTAREA': return 'textbox'
      case 'OPTION': return 'option'
      case 'IMG': return 'img'
      case 'INPUT': {
        const t = (el.type || 'text').toLowerCase()
        if (t === 'button' || t === 'submit' || t === 'reset' || t === 'image') return 'button'
        if (t === 'checkbox') return 'checkbox'
        if (t === 'radio') return 'radio'
        if (t === 'range') return 'slider'
        if (t === 'search') return 'searchbox'
        if (t === 'file') return 'file'
        return 'textbox'
      }
    }
    if (el.isContentEditable) return 'textbox'
    return ''
  }

  function textOfIds(el, attr) {
    const root = el.getRootNode()
    return oneLine((el.getAttribute(attr) || '').split(/\s+/)
      .map((id) => (root.getElementById ? root.getElementById(id) : document.getElementById(id))?.innerText || '').join(' '))
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
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || role === 'textbox' || role === 'searchbox' || role === 'combobox') {
      return clip(oneLine(el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('name') || el.id), 140)
    }
    const text = oneLine(el.innerText || el.textContent)
    if (text) return clip(text, 140)
    const title = oneLine(el.getAttribute('title'))
    if (title) return clip(title, 140)
    // A picture button: what its picture is called.
    const pic = el.querySelector?.('img[alt],svg[aria-label],[aria-label],[title]')
    return pic ? clip(oneLine(pic.getAttribute('alt') || pic.getAttribute('aria-label') || pic.getAttribute('title')), 140) : ''
  }

  const cssId = (s) => (globalThis.CSS && CSS.escape ? CSS.escape(s) : s.replace(/[^\w-]/g, '\\$&'))
  // An id a person wrote, not one a framework counted out ("r:1a", "input-3817", a hash).
  const steady = (id) => !!id && id.length < 50 && !/\d{3,}|[:.]|^[0-9a-f-]{16,}$/i.test(id)

  // A hint for finding it again, never the only way: short, by its own marks
  // first, by its place under the nearest marked ancestor otherwise.
  function selectorOf(el) {
    const parts = []
    let node = el
    for (let depth = 0; node && node.nodeType === 1 && depth < 5; depth++) {
      const tag = node.tagName.toLowerCase()
      const test = node.getAttribute('data-testid') || node.getAttribute('data-test') || node.getAttribute('data-qa')
      if (test) { parts.unshift(`[data-testid="${test.replace(/"/g, '\\"')}"]`); break }
      if (steady(node.id)) { parts.unshift(`#${cssId(node.id)}`); break }
      const name = node.getAttribute('name')
      if (name && /^(input|select|textarea|button)$/.test(tag)) { parts.unshift(`${tag}[name="${name.replace(/"/g, '\\"')}"]`); break }
      let part = tag
      const parent = node.parentElement
      if (parent) {
        const same = [...parent.children].filter((c) => c.tagName === node.tagName)
        if (same.length > 1) part += `:nth-of-type(${same.indexOf(node) + 1})`
      }
      parts.unshift(part)
      if (tag === 'body' || tag === 'html') break
      node = parent
    }
    return clip(parts.join(' > '), 300)
  }

  function headingOf(box) {
    const aria = oneLine(box.getAttribute('aria-label'))
    if (aria) return aria
    if (box.hasAttribute('aria-labelledby')) {
      const t = textOfIds(box, 'aria-labelledby')
      if (t) return t
    }
    const h = box.querySelector(':scope > legend, :scope > h1, :scope > h2, :scope > h3, :scope > h4, :scope > header h1, :scope > header h2, :scope > header h3, h1, h2, h3, [role=heading]')
    return h ? oneLine(h.innerText) : ''
  }

  // What it sits in, when that tells one from another: a dialog, a row of a
  // list or table (which "Delete" was it?), a named part of the page.
  function nearOf(el, name) {
    const dialog = el.closest('dialog,[role=dialog],[role=alertdialog]')
    if (dialog) {
      const t = headingOf(dialog)
      return clip(t ? `in the dialog “${t}”` : 'in the dialog', 160)
    }
    const row = el.closest('tr,[role=row],li,[role=listitem],article')
    if (row && row !== el) {
      const t = oneLine(row.innerText)
      const rest = name && t.includes(name) ? oneLine(t.replace(name, ' ')) : t
      if (rest && rest.length > 1) return clip(`in the row “${clip(rest, 90)}”`, 160)
    }
    const box = el.closest('form,fieldset,section,nav,aside,header,footer,[role=region],[role=navigation],[role=menu],[role=tablist],[role=toolbar]')
    if (box && box !== el) {
      const t = headingOf(box)
      if (t && t !== name) return clip(`in “${clip(t, 90)}”`, 160)
      const role = roleOf(box) || box.tagName.toLowerCase()
      if (/^(nav|navigation|menu|toolbar|tablist|header|footer|aside)$/.test(role)) {
        return `in the ${role === 'nav' ? 'navigation' : role === 'tablist' ? 'tabs' : role === 'aside' ? 'side bar' : role}`
      }
    }
    return ''
  }

  function describe(el) {
    const role = roleOf(el)
    const name = nameOf(el, role)
    const out = { role, name, tag: el.tagName.toLowerCase(), selector: selectorOf(el) }
    if (el.tagName === 'INPUT') out.type = (el.type || 'text').toLowerCase()
    const near = nearOf(el, name)
    if (near) out.near = near
    const href = el.tagName === 'A' ? el.getAttribute('href') || '' : ''
    if (href && !/^javascript:/i.test(href)) {
      try { out.href = clip(new URL(href, document.baseURI).href, 300) } catch { /* not an address */ }
    }
    return out
  }

  // ── telling the service worker ──

  function stop() {
    on = false
    pending = null
    badge(false)
  }

  function tell(step) {
    if (!on) return
    try {
      chrome.runtime.sendMessage({ type: 'captain-rec-step', step }).then(
        (answer) => { if (answer && answer.off) stop() },
        () => { /* the worker was asleep and dropped it, or the extension is gone */ })
    } catch {
      stop()   // the extension was reloaded: this copy is an orphan
    }
  }

  // ── typing ──

  const TEXT_INPUTS = new Set(['text', 'search', 'email', 'url', 'tel', 'number', 'password', 'date', 'datetime-local', 'month',
    'week', 'time', 'color', 'range', ''])

  function isTextField(el) {
    if (!el || el.nodeType !== 1) return false
    if (el.tagName === 'TEXTAREA') return true
    if (el.tagName === 'INPUT') return TEXT_INPUTS.has((el.type || '').toLowerCase())
    return !!el.isContentEditable
  }

  // The editable box itself, not the paragraph inside it the caret is in.
  function fieldOf(el) {
    if (!el || el.nodeType !== 1) return null
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') return el
    if (el.isContentEditable) {
      let node = el
      while (node.parentElement && node.parentElement.isContentEditable) node = node.parentElement
      return node
    }
    return null
  }

  const SECRET_NAME = /pass(word|wd)?|pwd|secret|otp|one.?time|cvv|cvc|card.?(number|no)|iban|ssn|security.?code|pin\b/i

  function isSecret(el) {
    if (el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA') return false
    if ((el.type || '').toLowerCase() === 'password') return true
    const ac = (el.getAttribute('autocomplete') || '').toLowerCase()
    if (/password|one-time-code|cc-(number|csc|exp)/.test(ac)) return true
    return SECRET_NAME.test(`${el.name || ''} ${el.id || ''}`)
  }

  const valueOf = (el) => (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' ? String(el.value) : String(el.innerText || '').trim())

  function flush() {
    const p = pending
    pending = null
    if (!p) return
    const value = p.value
    if (sent.get(p.el) === value) return
    sent.set(p.el, value)
    const step = { kind: 'type', target: describe(p.el) }
    if (isSecret(p.el)) step.secret = true
    else step.value = clip(value, 2000)
    tell(step)
  }

  function onInput(ev) {
    if (!on) return
    const el = fieldOf(ev.composedPath()[0])
    if (!el || !isTextField(el)) return
    if (pending && pending.el !== el) flush()
    pending = { el, value: valueOf(el) }
  }

  // ── choices ──

  function onChange(ev) {
    if (!on) return
    const el = ev.composedPath()[0]
    if (!el || el.nodeType !== 1) return
    if (el.tagName === 'SELECT') {
      flush()
      const picked = [...el.selectedOptions].map((o) => oneLine(o.label || o.text)).filter(Boolean)
      tell({ kind: 'select', target: describe(el), value: clip(picked.join(', '), 500) })
    } else if (el.tagName === 'INPUT') {
      const type = (el.type || '').toLowerCase()
      if (type === 'checkbox' || type === 'radio') {
        flush()
        tell({ kind: 'check', target: describe(el), checked: !!el.checked })
      } else if (type === 'file') {
        flush()
        const names = [...(el.files || [])].map((f) => f.name)
        if (names.length) tell({ kind: 'upload', target: describe(el), value: clip(names.join(', '), 500) })
      } else if (isTextField(el)) {
        pending = { el, value: valueOf(el) }
        flush()
      }
    } else if (el.tagName === 'TEXTAREA') {
      pending = { el, value: valueOf(el) }
      flush()
    }
  }

  // ── clicks ──

  const CLICKABLE = 'a[href],button,summary,select,option,input,textarea,label,[role=button],[role=link],[role=tab],[role=menuitem],' +
    '[role=menuitemcheckbox],[role=menuitemradio],[role=option],[role=checkbox],[role=radio],[role=switch],[role=treeitem],' +
    '[role=combobox],[onclick],[contenteditable=""],[contenteditable=true],[contenteditable=plaintext-only]'

  function clickTarget(el) {
    if (!el || el.nodeType !== 1) el = el?.parentElement
    if (!el) return null
    const hit = el.closest(CLICKABLE)
    if (hit) return hit
    // No role, yet made to be clicked: the outermost thing with the hand cursor.
    let node = el
    let best = null
    for (let i = 0; node && node !== document.body && i < 6; i++, node = node.parentElement) {
      let cursor = ''
      try { cursor = getComputedStyle(node).cursor } catch { /* gone */ }
      if (cursor === 'pointer') best = node
      else if (best) break
    }
    return best || el
  }

  function onPointerDown() {
    if (on) flush()   // what was typed comes before what is clicked next
  }

  function onClick(ev) {
    if (!on || ev.button !== 0) return
    const el = clickTarget(ev.composedPath()[0])
    if (!el || el === document.body || el === document.documentElement) return
    const tag = el.tagName
    if (tag === 'SELECT' || tag === 'OPTION' || tag === 'TEXTAREA') return        // its change says it
    if (tag === 'INPUT') {
      const t = (el.type || '').toLowerCase()
      if (!(t === 'submit' || t === 'button' || t === 'reset' || t === 'image')) return   // a tick, a file, or just the caret
    }
    if (tag === 'LABEL' && el.control) return                                     // the field it names says it
    if (el.isContentEditable) return
    flush()
    const step = { kind: 'click', target: describe(el) }
    if (ev.detail === 2) step.double = true
    tell(step)
  }

  // ── keys ──

  function onKeyDown(ev) {
    if (!on || ev.isComposing) return
    if (ev.key === 'Escape') {
      flush()
      tell({ kind: 'key', key: 'Escape' })
      return
    }
    if (ev.key !== 'Enter') return
    const el = fieldOf(ev.composedPath()[0])
    if (!el) return   // Enter on a button is its click
    if (el.tagName === 'INPUT') {
      if (!isTextField(el)) return
      pending = { el, value: valueOf(el) }
      flush()
      tell({ kind: 'key', key: 'Enter', target: describe(el) })
      return
    }
    // A text box of many lines: Enter is a new line — unless it sends (a chat
    // box empties itself), or it came with Ctrl/⌘.
    const before = valueOf(el)
    const target = describe(el)
    const withMod = ev.ctrlKey || ev.metaKey
    setTimeout(() => {
      if (!on) return
      if (!withMod && !(before.trim() && !valueOf(el).trim())) return
      if (sent.get(el) !== before) {
        sent.set(el, before)
        tell({ kind: 'type', target, value: clip(before, 2000) })
      }
      pending = null
      sent.delete(el)   // the same words typed again are a new step
      tell({ kind: 'key', key: withMod ? (ev.metaKey ? '⌘+Enter' : 'Ctrl+Enter') : 'Enter', target })
    }, 80)
  }

  // ── a frame that arrived later ──

  function onLoad(ev) {
    if (!on || !TOP) return
    const el = ev.target
    if (el && (el.tagName === 'IFRAME' || el.tagName === 'FRAME')) {
      try { chrome.runtime.sendMessage({ type: 'captain-rec-frames' }).catch(() => {}) } catch { stop() }
    }
  }

  // ── "recording" on the page ──

  let badgeEl = null
  function badge(show) {
    if (!TOP) return
    if (!show) { badgeEl?.remove(); badgeEl = null; return }
    if (badgeEl?.isConnected) return
    try {
      badgeEl = document.createElement('captain-recording')
      badgeEl.setAttribute('aria-hidden', 'true')
      const root = badgeEl.attachShadow({ mode: 'closed' })
      root.innerHTML = '<style>:host{all:initial;position:fixed;left:12px;bottom:12px;z-index:2147483647;pointer-events:none}' +
        'div{display:flex;align-items:center;gap:6px;font:600 11px/1 system-ui,sans-serif;color:#fff;background:rgba(24,24,27,.86);' +
        'padding:6px 9px;border-radius:999px;box-shadow:0 1px 4px rgba(0,0,0,.3)}' +
        'i{width:8px;height:8px;border-radius:50%;background:#ef4444;animation:b 1.4s infinite}@keyframes b{50%{opacity:.25}}</style>' +
        '<div><i></i>Captain is recording</div>'
      document.documentElement.appendChild(badgeEl)
    } catch { badgeEl = null }
  }

  const opts = { capture: true, passive: true }
  window.addEventListener('input', onInput, opts)
  window.addEventListener('change', onChange, opts)
  window.addEventListener('pointerdown', onPointerDown, opts)
  window.addEventListener('click', onClick, opts)
  window.addEventListener('keydown', onKeyDown, opts)
  window.addEventListener('focusout', () => { if (on) flush() }, opts)
  window.addEventListener('pagehide', () => { if (on) flush() }, opts)
  window.addEventListener('blur', () => { if (on) flush() })   // off to the side panel (to stop, to add a note)
  document.addEventListener('load', onLoad, true)

  try {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg && msg.type === 'captain-rec') {
        if (msg.on) { on = true; badge(true) } else { flush(); stop() }
      }
    })
  } catch { /* no extension to listen to */ }

  globalThis.__captainRecorder = { resume() { on = true; badge(true) }, stop }
  badge(true)
  return 'on'
})()
