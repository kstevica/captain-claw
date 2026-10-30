import { create } from 'zustand'
import type { ViewMode } from '../types'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'

type MobilePanel = 'none' | 'director' | 'chat' | 'tool'

/**
 * Two ways to lay out the desktop:
 *  - `full`   — the existing Flight Deck: nav sidebar, top bar, director,
 *               pages, tool panels, docked chat. Untouched.
 *  - `simple` — a chat-first three-column view: agents on the left, the
 *               chat in the middle, the active agent's files + datastore on
 *               the right. Both side columns collapse to a thin rail.
 *  - `console` — the Admin console: the deck's admin work (models,
 *               archetypes, connections, users, usage, settings) and
 *               nothing else. Admins only; anyone else gets `full`.
 */
export type LayoutMode = 'full' | 'simple' | 'console'

function asLayoutMode(v: string | null | undefined): LayoutMode {
  return v === 'simple' || v === 'console' ? v : 'full'
}

const LAYOUT_MODE_KEY = 'fd:layout-mode'
const SIMPLE_LEFT_OPEN_KEY = 'fd:simple-left-open'
const SIMPLE_RIGHT_OPEN_KEY = 'fd:simple-right-open'
const SIMPLE_QUEUE_OPEN_KEY = 'fd:simple-queue-open'

function loadLayoutMode(): LayoutMode {
  try {
    return asLayoutMode(localStorage.getItem(LAYOUT_MODE_KEY))
  } catch { return 'full' }
}
function loadBool(key: string, fallback: boolean): boolean {
  try {
    const v = localStorage.getItem(key)
    if (v === null) return fallback
    return v === 'true'
  } catch { return fallback }
}
// Mirror the value into localStorage in every case so the very next page
// load (before server hydration finishes) already renders the right layout;
// the server copy is what carries it across devices.
function persist(key: string, val: string) {
  try { localStorage.setItem(key, val) } catch { /* private mode */ }
  if (useAuthStore.getState().authEnabled) queueSave(key, val)
}

interface UIStore {
  view: ViewMode
  sidebarOpen: boolean
  panelOpen: boolean
  mobilePanel: MobilePanel
  sidebarDrawerOpen: boolean
  forgeProjectId: string
  layoutMode: LayoutMode
  /** Simple layout: is the agents column expanded (vs. a thin rail)? */
  simpleLeftOpen: boolean
  /** Simple layout: is the files + datastore column expanded? */
  simpleRightOpen: boolean
  /** Simple layout: is the chat's queue column shown? Hidden until asked for. */
  simpleQueueOpen: boolean
  setView: (v: ViewMode) => void
  toggleSidebar: () => void
  togglePanel: () => void
  setPanelOpen: (v: boolean) => void
  setMobilePanel: (panel: MobilePanel) => void
  toggleMobilePanel: (panel: MobilePanel) => void
  setSidebarDrawerOpen: (v: boolean) => void
  setForgeProjectId: (id: string) => void
  setLayoutMode: (mode: LayoutMode) => void
  /** Show a layout for this visit without choosing it: never saved, so one
   *  followed link doesn't change what opens on every device from then on. */
  visitLayout: (mode: LayoutMode) => void
  toggleLayoutMode: () => void
  setSimpleLeftOpen: (v: boolean) => void
  setSimpleRightOpen: (v: boolean) => void
  setSimpleQueueOpen: (v: boolean) => void
}

/**
 * An admin who never picked a layout starts in the Admin console. The
 * signed-in settings say so for an admin (the hydrator below); a deck without
 * sign-in (the desktop app), whose user is its admin, asks this on start.
 * Not saved: the first layout they pick is.
 */
// A layout picked while signed-in settings are loading wins over the loaded one.
let layoutChosenSinceHydrate = false
// A layout shown for this visit only (a ?taskboard= link): settings loading
// meanwhile leave it on screen, and nothing saves it.
let layoutVisited = false

/** Call right before hydrating the signed-in settings. */
export function beginLayoutHydrate() {
  layoutChosenSinceHydrate = false
}

export function startInConsoleUnlessChosen() {
  try { if (localStorage.getItem(LAYOUT_MODE_KEY) !== null) return } catch { /* storage blocked */ }
  useUIStore.setState({ layoutMode: 'console' })
}

export const useUIStore = create<UIStore>((set, get) => ({
  view: 'desktop',
  sidebarOpen: true,
  panelOpen: false,
  mobilePanel: 'none',
  sidebarDrawerOpen: false,
  forgeProjectId: '',
  layoutMode: loadLayoutMode(),
  simpleLeftOpen: loadBool(SIMPLE_LEFT_OPEN_KEY, true),
  simpleRightOpen: loadBool(SIMPLE_RIGHT_OPEN_KEY, true),
  simpleQueueOpen: loadBool(SIMPLE_QUEUE_OPEN_KEY, false),

  setView: (view) => set({ view }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  togglePanel: () => set((s) => ({ panelOpen: !s.panelOpen })),
  setPanelOpen: (panelOpen) => set({ panelOpen }),
  setMobilePanel: (mobilePanel) => set({ mobilePanel }),
  toggleMobilePanel: (panel) => set((s) => ({ mobilePanel: s.mobilePanel === panel ? 'none' : panel })),
  setSidebarDrawerOpen: (sidebarDrawerOpen) => set({ sidebarDrawerOpen }),
  setForgeProjectId: (forgeProjectId) => set({ forgeProjectId }),

  setLayoutMode: (layoutMode) => {
    // Idempotent: a redundant call (e.g. a page shortcut fired while already
    // in the full layout) must not write localStorage or PUT /fd/settings.
    if (get().layoutMode === layoutMode) return
    layoutChosenSinceHydrate = true
    layoutVisited = false
    persist(LAYOUT_MODE_KEY, layoutMode)
    set({ layoutMode })
  },
  visitLayout: (layoutMode) => {
    if (get().layoutMode === layoutMode) return
    layoutVisited = true
    set({ layoutMode })
  },
  toggleLayoutMode: () => {
    const mode = get().layoutMode
    if (mode === 'console') return  // left from the console's own sidebar
    get().setLayoutMode(mode === 'simple' ? 'full' : 'simple')
  },
  setSimpleLeftOpen: (simpleLeftOpen) => {
    if (get().simpleLeftOpen === simpleLeftOpen) return
    persist(SIMPLE_LEFT_OPEN_KEY, String(simpleLeftOpen))
    set({ simpleLeftOpen })
  },
  setSimpleRightOpen: (simpleRightOpen) => {
    if (get().simpleRightOpen === simpleRightOpen) return
    persist(SIMPLE_RIGHT_OPEN_KEY, String(simpleRightOpen))
    set({ simpleRightOpen })
  },
  setSimpleQueueOpen: (simpleQueueOpen) => {
    if (get().simpleQueueOpen === simpleQueueOpen) return
    persist(SIMPLE_QUEUE_OPEN_KEY, String(simpleQueueOpen))
    set({ simpleQueueOpen })
  },
}))

// Server-side settings win on login. The hydrator only runs when authenticated
// (hydrateAllStores), so treat the server as authoritative: a key that's ABSENT
// there resets the field to its default (the layout's: the Admin console for
// an admin, the full layout for anyone else). Otherwise a previous user's mode/rails
// would survive in the module-level store on a shared browser whenever the tab
// isn't reloaded between sessions (sign-out reloads — this is defense in
// depth). hydrateAllStores() clears the localStorage mirror for absent
// keys itself, so we only re-mirror keys the server actually has.
registerHydrator((settings) => {
  const mode = settings[LAYOUT_MODE_KEY]
  const bool = (key: string, dflt: boolean) => {
    const v = settings[key]
    return v === 'true' ? true : v === 'false' ? false : dflt
  }
  const admin = useAuthStore.getState().user?.role === 'admin'
  const patch = {
    layoutMode: layoutChosenSinceHydrate || layoutVisited ? useUIStore.getState().layoutMode
      : mode === undefined && admin ? 'console' as const : asLayoutMode(mode),
    simpleLeftOpen: bool(SIMPLE_LEFT_OPEN_KEY, true),
    simpleRightOpen: bool(SIMPLE_RIGHT_OPEN_KEY, true),
    simpleQueueOpen: bool(SIMPLE_QUEUE_OPEN_KEY, false),
  }
  for (const key of [LAYOUT_MODE_KEY, SIMPLE_LEFT_OPEN_KEY, SIMPLE_RIGHT_OPEN_KEY, SIMPLE_QUEUE_OPEN_KEY]) {
    const v = settings[key]
    try { if (v !== undefined) localStorage.setItem(key, v) } catch { /* ignore */ }
  }
  useUIStore.setState(patch)
})
