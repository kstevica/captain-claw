// Waiting for a page to be drawn before it's read.
//
// "Loaded" isn't "there": pages like LinkedIn draw their header first and
// each section as its data arrives — some only once they're scrolled to.
// Both functions run INSIDE the page (chrome.scripting.executeScript's func,
// or as source through the debugger: `(${settleInPage})(opts)`), so they must
// stay self-contained: no imports, nothing from outside their own body.

/** Resolves once the page has stopped changing: loaded, no change to its DOM
 * and no data request finishing for *quietMs*, nothing marked busy, and at
 * least *minMs* gone (a page waiting for its data looks quiet meanwhile) — or
 * after *maxMs* whatever it looks like. */
export function settleInPage({ maxMs = 8000, quietMs = 800, minMs = 0 } = {}) {
  return new Promise((resolve) => {
    const start = Date.now()
    let last = start
    let changes = 0
    const busy = () => {
      try { return !!document.querySelector('[aria-busy="true"]') } catch { return false }
    }
    const watch = new MutationObserver((list) => { changes += list.length; last = Date.now() })
    try {
      watch.observe(document.documentElement || document, { childList: true, subtree: true, characterData: true })
    } catch { /* no document yet */ }
    // Data arriving (fetch, XHR, a script loaded late) is the page still at work.
    let data = null
    try {
      data = new PerformanceObserver((list) => {
        if (list.getEntries().some((e) => ['fetch', 'xmlhttprequest', 'script'].includes(e.initiatorType))) last = Date.now()
      })
      data.observe({ type: 'resource', buffered: false })
    } catch { data = null }
    const tick = () => {
      const now = Date.now()
      const quiet = now - last >= quietMs && now - start >= minMs && document.readyState === 'complete' && !busy()
      if (quiet || now - start >= maxMs) {
        watch.disconnect()
        data?.disconnect()
        resolve({ waited: now - start, changes, quiet, text: (document.body?.innerText || '').length })
      } else {
        setTimeout(tick, 100)
      }
    }
    setTimeout(tick, 100)
  })
}

/** Scrolls the page down a screen at a time — waiting at each step for what
 * that brings in (sections drawn when scrolled to, more of a list) — then back
 * to where it was. The page's own scroller, or its biggest scrolling box when
 * the window itself doesn't scroll (apps that scroll inside a panel). */
export function scrollThrough({ steps = 10, quietMs = 600, stepMaxMs = 2500, totalMs = 15000 } = {}) {
  const quietFor = (ms, max) => new Promise((resolve) => {
    const start = Date.now()
    let last = start
    const watch = new MutationObserver(() => { last = Date.now() })
    try { watch.observe(document.documentElement, { childList: true, subtree: true, characterData: true }) } catch { /* gone */ }
    const tick = () => {
      const now = Date.now()
      if (now - last >= ms || now - start >= max) { watch.disconnect(); resolve() } else setTimeout(tick, 100)
    }
    setTimeout(tick, 100)
  })
  const scroller = () => {
    const doc = document.scrollingElement || document.documentElement
    if (doc.scrollHeight > window.innerHeight + 50) return null   // the window scrolls
    let best = null
    const all = document.body ? document.body.getElementsByTagName('*') : []
    for (let i = 0; i < all.length && i < 6000; i++) {
      const el = all[i]
      if (el.scrollHeight <= el.clientHeight + 50 || el.clientHeight < 200) continue
      const oy = getComputedStyle(el).overflowY
      if (oy !== 'auto' && oy !== 'scroll') continue
      if (!best || el.scrollHeight > best.scrollHeight) best = el
    }
    return best
  }
  return (async () => {
    const start = Date.now()
    const box = scroller()
    const doc = document.scrollingElement || document.documentElement
    const pos = () => (box ? box.scrollTop : window.scrollY)
    const height = () => (box ? box.scrollHeight : doc.scrollHeight)
    const view = () => (box ? box.clientHeight : window.innerHeight)
    const x0 = window.scrollX
    const y0 = pos()
    let n = 0
    let grew = 0
    let h = height()
    while (n < steps && Date.now() - start < totalMs) {
      const before = pos()
      const by = Math.max(Math.round(view() * 0.85), 300)
      if (box) box.scrollTop = before + by
      else window.scrollTo(x0, before + by)
      n++
      await quietFor(quietMs, stepMaxMs)
      if (height() > h) { grew++; h = height() }
      if (pos() <= before) break   // the bottom, and nothing more came
    }
    if (box) box.scrollTop = y0
    else window.scrollTo(x0, y0)
    return { steps: n, grew, height: h, inBox: !!box }
  })()
}
