// Captain for Chrome — the service worker.
//
// It only opens the side panel (toolbar button, Alt+Shift+C, right-click
// menu) and tells it what a right-click pointed at. The panel does the rest:
// reading the page, fetching its files, sending them to Flight Deck.

const MENU = [
  { id: 'captain-page', title: 'Send this page to Captain', contexts: ['page'] },
  { id: 'captain-selection', title: 'Send selection to Captain', contexts: ['selection'] },
  { id: 'captain-link', title: 'Send link to Captain', contexts: ['link'] },
  { id: 'captain-image', title: 'Send image to Captain', contexts: ['image'] },
]

function setUp() {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {})
  chrome.contextMenus.removeAll(() => {
    for (const item of MENU) chrome.contextMenus.create(item, () => void chrome.runtime.lastError)
  })
}

chrome.runtime.onInstalled.addListener(setUp)
chrome.runtime.onStartup.addListener(setUp)

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!tab || tab.id === undefined) return
  // Opening the panel must happen in the click itself (a user gesture).
  chrome.sidePanel.open({ tabId: tab.id }).catch(() => chrome.sidePanel.open({ windowId: tab.windowId }).catch(() => {}))
  const focus = { kind: info.menuItemId.replace('captain-', ''), tabId: tab.id, at: Date.now() }
  if (info.selectionText) focus.selection = String(info.selectionText).slice(0, 20000)
  if (info.linkUrl) focus.link = info.linkUrl
  if (info.srcUrl) focus.src = info.srcUrl
  chrome.storage.session.set({ focus }).catch(() => {})
})
