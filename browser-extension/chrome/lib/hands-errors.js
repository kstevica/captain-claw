// Captain for Chrome — what a browser act that went wrong says to the agent.
//
// Chrome keeps an extension's debugger off a tab that holds a frame of ANOTHER
// extension (a helper button some extension put into a page): attaching fails,
// and an attached debugger is cut off when such a frame appears. hands.js then
// works on that page without the debugger (chrome.scripting, chrome.tabs);
// these are the words for it. Nothing here touches chrome.*, so it's tested
// outside Chrome too (tests/test_flight_deck/test_browser_hands.py).

/** Chrome refused the debugger: the tab holds another extension's frame. */
export const OTHER_EXTENSION = /Cannot access a chrome-extension:\/\/ URL of (a )?different extension/i

/** Chrome cut the debugger off while a command was on its way. */
export const LOST_DEBUGGER = /Detached while handling command|Debugger is not attached/i

/** What still works on such a page, and what doesn't. */
export const PLAIN_CAN = 'navigate, read_page, find, page_text, page_files, form_input, click (by ref, selector or text) and ' +
  'type (by ref, or into the field that has the keyboard) — made of page events, not real input —, scroll, scroll_to, ' +
  'screenshot and zoom (the tab comes to the front), download, save_page, upload_file and resize'
export const PLAIN_CANNOT = 'press_key, hover, drag, double/right/triple clicks, clicks at a coordinate, javascript, console and network_requests'

const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s)

/** Where the person stops that other extension on this site. */
export function pauseHint(ids = []) {
  const where = ids.length ? `chrome://extensions/?id=${ids[0]}` : 'chrome://extensions → its Details'
  return `the person can set that extension's Site access to “On click” (${where}) and reload the page`
}

function why(ids = []) {
  return `Another Chrome extension put a frame of its own into this page${ids.length ? ` (extension ${ids[0]})` : ''}, ` +
    "and Chrome keeps Captain's debugger off a tab that holds one"
}

/** *what* (an act) needs the debugger, and this page won't have it. */
export function needsDebugger(what, page = {}) {
  if (page.scriptable === false) return noScripting(page)
  return `${why(page.ext)}, so ${what} can't work on this page. Without the debugger these still work here: ${PLAIN_CAN}. ` +
    `For ${what}, ${pauseHint(page.ext)}.`
}

/** Not even chrome.scripting may run there: Captain may not "Work on every site". */
export function noScripting(page = {}) {
  return `${why(page.ext)}. Without the debugger Captain can still navigate this tab and save_page, but reading and using the page ` +
    "also needs Captain to “Work on every site” — ask the person to turn that on in Captain's settings, or " +
    `${pauseHint(page.ext)}.`
}

/** Said after a page loads where the debugger can't go. */
export function plainNote(page = {}) {
  if (page.scriptable === false) return `Note: ${noScripting(page)}`
  return `Note: ${why(page.ext)}. Captain works on this page without it: ${PLAIN_CAN} work; ${PLAIN_CANNOT} ` +
    `need the debugger (for those, ${pauseHint(page.ext)}).`
}

export function cleanError(e) {
  let s = String(e?.message || e || 'Something went wrong.')
  s = s.replace(/^(Uncaught )?(Error|TypeError): /, '').split('\n    at ')[0]
  if (OTHER_EXTENSION.test(s)) {
    return `${why()} (this isn't a chrome:// page). Try again: reading, navigating, filling fields and clicks by ref ` +
      `work there without the debugger. For the rest, ${pauseHint()}.`
  }
  if (LOST_DEBUGGER.test(s)) {
    return "Chrome took Captain's debugger off this tab in the middle of that — most often because another extension " +
      'just put a frame of its own into the page (or the person pressed Cancel on Chrome\'s debugging bar). It may or ' +
      'may not have happened: check with read_page or a screenshot before trying again.'
  }
  if (/Cannot access a chrome|cannot be scripted|Cannot attach to this target|chrome-extension:\/\//i.test(s)) {
    return "Chrome doesn't let extensions work on this page (a chrome:// or extension page, or the Web Store)."
  }
  if (/Another debugger is already attached/i.test(s)) {
    return 'DevTools (or another debugger) is open on that tab — the person has to close it first.'
  }
  if (/No tab with id/i.test(s)) return 'That tab was closed.'
  return clip(s, 2000)
}
