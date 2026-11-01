import { useEffect, useState } from 'react'
import {
  LayoutGrid, Gauge, Library, Plug, Network, UserCog, BarChart3, Settings2, LayoutDashboard, MessagesSquare, ScrollText,
  RefreshCw, Sun, Moon, LogOut, Loader2, Menu, X, type LucideIcon,
} from 'lucide-react'
import { useAuthStore, logoutUser, DEFAULT_LOGO } from '../../stores/authStore'
import { useThemeStore } from '../../stores/themeStore'
import { useUIStore } from '../../stores/uiStore'
import {
  useConsoleStore, leaveConsole, enterKioskView, pageFromPath, CONSOLE_PATH, type ConsolePage,
} from '../../stores/consoleStore'
import { APP_VERSION } from '../../version'
import { isAppShell } from '../../services/desktopShell'
import { hasTierSets, TIER_SETS_EVENT } from '../../services/tierConfig'
import { fetchWaitlistPending } from './consoleApi'
import { OverviewPage } from './OverviewPage'
import { ModelsPage } from './ModelsPage'
import { ArchetypesPage } from './ArchetypesPage'
import { ConnectionsConsole } from './ConnectionsConsole'
import { NetworkConsole } from './NetworkConsole'
import { UsersConsole } from './UsersConsole'
import { UsageConsole } from './UsageConsole'
import { AuditConsole } from './AuditConsole'
import { UpdatesConsole } from './UpdatesConsole'
import { SettingsConsole } from './SettingsConsole'

// The Admin console's frame: the pages down the left (a menu on a phone),
// the page on the right. Everything an admin sets up for the deck and nothing
// else — the agents and their chats stay in Flight Deck, one click away.

// `accounts`: only on a deck with sign-in — without it (the desktop app)
// there is one user, no plans and no one to invite.
type NavItem = { id: ConsolePage; icon: LucideIcon; label: string; hint: string; accounts?: boolean; soloHint?: string }

const NAV: NavItem[] = [
  { id: 'overview', icon: LayoutGrid, label: 'Overview', hint: 'How the deck is doing' },
  { id: 'models', icon: Gauge, label: 'Models', hint: 'Tier sets and team keys' },
  { id: 'archetypes', icon: Library, label: 'Archetypes', hint: 'Agents people start from' },
  { id: 'connections', icon: Plug, label: 'Connections', hint: 'Google, Telegram, MCP…' },
  { id: 'network', icon: Network, label: 'BotPort', hint: 'Connect this deck to other Flight Decks' },
  { id: 'users', icon: UserCog, label: 'Users', hint: 'Accounts, invitations, waitlist', accounts: true },
  { id: 'usage', icon: BarChart3, label: 'Usage & limits', hint: 'Today, spend and plans', accounts: true },
  { id: 'audit', icon: ScrollText, label: 'Audit', hint: 'Every tool call agents made' },
  { id: 'updates', icon: RefreshCw, label: 'Updates', hint: 'Pull a new version and restart' },
  { id: 'settings', icon: Settings2, label: 'Settings', hint: 'Branding, sign-up, kiosk', soloHint: 'Branding, agents, profile' },
]

function initials(s: string): string {
  const parts = s.trim().split(/[\s@._-]+/).filter(Boolean)
  return ((parts[0]?.[0] || '') + (parts[1]?.[0] || '')).toUpperCase() || '?'
}

export function ConsoleLayout() {
  const asked = useConsoleStore((s) => s.page)
  const go = useConsoleStore((s) => s.go)
  const layoutMode = useUIStore((s) => s.layoutMode)
  const accounts = useAuthStore((s) => s.authEnabled !== false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [waitlist, setWaitlist] = useState(0)
  // No tier set yet: Models is marked as the place to start.
  const [models, setModels] = useState(hasTierSets)
  useEffect(() => {
    const check = () => setModels(hasTierSets())
    window.addEventListener(TIER_SETS_EVENT, check)
    return () => window.removeEventListener(TIER_SETS_EVENT, check)
  }, [])
  const nav = NAV.filter((n) => accounts || !n.accounts).map((n) => (!accounts && n.soloHint ? { ...n, hint: n.soloHint } : n))
  const current = nav.find((n) => n.id === asked) || nav[0]
  const page = current.id

  // Opened from the address bar (or the saved layout): keep the layout saved as
  // the console, even if the signed-in settings land later with another one.
  useEffect(() => {
    if (layoutMode !== 'console') useUIStore.getState().setLayoutMode('console')
  }, [layoutMode])

  // On screen: links in the cards it shows open console pages (consoleStore).
  useEffect(() => {
    useConsoleStore.setState({ shown: true })
    return () => useConsoleStore.setState({ shown: false })
  }, [])

  // The address shows the page (Back / Forward: syncConsoleFromUrl, in App).
  useEffect(() => {
    if (pageFromPath(window.location.pathname) !== page) {
      window.history.replaceState(null, '', `${CONSOLE_PATH}/${page}`)
    }
  }, [page])

  // Applications waiting for approval, on the Users entry.
  useEffect(() => {
    if (!accounts) return
    let cancelled = false
    fetchWaitlistPending().then((n) => { if (!cancelled) setWaitlist(n) }).catch(() => {})
    return () => { cancelled = true }
  }, [page, accounts])

  useEffect(() => { document.querySelector('#fd-console-main')?.scrollTo(0, 0) }, [page])

  const open = (id: ConsolePage) => { setMenuOpen(false); go(id) }

  return (
    <div className="flex h-[100dvh] overflow-hidden bg-zinc-950 text-zinc-200">
      <aside className="hidden w-64 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/40 wide:flex">
        <Brand />
        <Nav items={nav} page={page} waitlist={waitlist} needsModels={!models} onOpen={open} />
        <Footer />
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Phone: the page's name and a menu */}
        <header className="flex h-14 shrink-0 items-center gap-3 border-b border-zinc-800 bg-zinc-900/40 px-4 wide:hidden">
          <button onClick={() => setMenuOpen(true)} aria-label="Menu" className="-ml-1.5 rounded-lg p-2 text-zinc-400 hover:bg-zinc-800">
            <Menu className="h-5 w-5" />
          </button>
          <current.icon className="h-4 w-4 text-violet-600 dark:text-violet-400" />
          <span className="truncate text-sm font-semibold text-zinc-100">{current.label}</span>
          {waitlist > 0 && page !== 'users' && (
            <button onClick={() => go('users', { usersTab: 'waitlist' })} className="ml-auto rounded-full bg-violet-600 px-2 py-0.5 text-[11px] font-semibold text-white">
              {waitlist} waiting
            </button>
          )}
        </header>

        <main id="fd-console-main" className="flex-1 overflow-y-auto">
          {page === 'overview' && <OverviewPage />}
          {page === 'models' && <ModelsPage />}
          {page === 'archetypes' && <ArchetypesPage />}
          {page === 'connections' && <ConnectionsConsole />}
          {page === 'network' && <NetworkConsole />}
          {page === 'users' && <UsersConsole onWaitlistCount={setWaitlist} />}
          {page === 'usage' && <UsageConsole />}
          {page === 'audit' && <AuditConsole />}
          {page === 'updates' && <UpdatesConsole />}
          {page === 'settings' && <SettingsConsole />}
        </main>
      </div>

      {menuOpen && (
        <div className="fixed inset-0 z-50 flex bg-black/60 wide:hidden" onClick={() => setMenuOpen(false)}>
          <div className="flex h-full w-[min(18rem,85vw)] flex-col border-r border-zinc-800 bg-zinc-950" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center">
              <div className="min-w-0 flex-1"><Brand /></div>
              <button onClick={() => setMenuOpen(false)} aria-label="Close" className="mr-3 rounded-lg p-2 text-zinc-500 hover:bg-zinc-800">
                <X className="h-4 w-4" />
              </button>
            </div>
            <Nav items={nav} page={page} waitlist={waitlist} needsModels={!models} onOpen={open} />
            <Footer />
          </div>
        </div>
      )}
    </div>
  )
}

function Brand() {
  const name = useAuthStore((s) => s.branding.name)
  const logo = useAuthStore((s) => s.branding.logo)
  return (
    <div className="flex items-center gap-3 px-5 pb-4 pt-5">
      <img src={logo || DEFAULT_LOGO} alt="" className="h-9 w-9 shrink-0 rounded-xl object-contain" />
      <div className="min-w-0">
        <div className="truncate text-sm font-semibold tracking-tight text-zinc-100" title={name}>{name}</div>
        {/* The desktop app has two views of the deck: say which one this is. */}
        {isAppShell()
          ? <div className="text-[11px] font-medium text-violet-600 dark:text-violet-400">Admin view</div>
          : <div className="text-[11px] text-zinc-500">Admin console</div>}
      </div>
    </div>
  )
}


function Nav({ items, page, waitlist, needsModels, onOpen }: {
  items: NavItem[]
  page: ConsolePage
  waitlist: number
  needsModels: boolean
  onOpen: (p: ConsolePage) => void
}) {
  return (
    <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 py-2" aria-label="Admin console">
      {items.map(({ id, icon: Icon, label, hint }) => {
        const on = id === page
        return (
          <button
            key={id}
            onClick={() => onOpen(id)}
            aria-current={on ? 'page' : undefined}
            className={`group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors ${
              on ? 'bg-violet-500/10 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200'
            }`}
          >
            <Icon className={`h-[18px] w-[18px] shrink-0 ${on ? 'text-violet-600 dark:text-violet-400' : 'text-zinc-500 group-hover:text-zinc-300'}`} />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-medium">{label}</span>
              <span className="block truncate text-[11px] text-zinc-500">{hint}</span>
            </span>
            {id === 'models' && needsModels && (
              <span title="No tier set yet — start here" className="rounded-full bg-amber-500 px-1.5 py-0.5 text-[10px] font-semibold leading-none text-zinc-950">
                Start
              </span>
            )}
            {id === 'users' && waitlist > 0 && (
              <span title={`${waitlist} waiting for approval`} className="rounded-full bg-violet-600 px-1.5 py-0.5 text-[10px] font-semibold leading-none text-white">
                {waitlist}
              </span>
            )}
          </button>
        )
      })}
    </nav>
  )
}

function Footer() {
  const user = useAuthStore((s) => s.user)
  const signingOut = useAuthStore((s) => s.signingOut)
  const { theme, toggle } = useThemeStore()
  const who = user?.display_name || user?.email || ''
  const row = 'flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm text-zinc-400 transition-colors hover:bg-zinc-800/60 hover:text-zinc-200'
  return (
    <div className="space-y-0.5 border-t border-zinc-800 p-3">
      {/* The desktop app has no full Flight Deck: its other side is the users' kiosk. */}
      {isAppShell() ? (
        <button onClick={enterKioskView} className={row} title="The kiosk your users get: their agents and chats">
          <MessagesSquare className="h-[18px] w-[18px] text-zinc-500" /> Switch to user view
        </button>
      ) : (
        <button onClick={() => leaveConsole('desktop')} className={row} title="The agents, their chats and every Flight Deck page">
          <LayoutDashboard className="h-[18px] w-[18px] text-zinc-500" /> Open Flight Deck
        </button>
      )}
      <button onClick={toggle} className={row}>
        {theme === 'dark' ? <Sun className="h-[18px] w-[18px] text-zinc-500" /> : <Moon className="h-[18px] w-[18px] text-zinc-500" />}
        {theme === 'dark' ? 'Light theme' : 'Dark theme'}
      </button>
      {user && (
        <div className="mt-2 flex items-center gap-3 rounded-xl px-3 py-2">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-violet-500/15 text-xs font-semibold text-violet-700 dark:text-violet-300">
            {initials(who)}
          </span>
          <div className="min-w-0 flex-1">
            <div className="truncate text-xs font-medium text-zinc-200">{who}</div>
            <div className="text-[10px] text-zinc-600">v{APP_VERSION}</div>
          </div>
          <button
            onClick={logoutUser}
            disabled={signingOut}
            title={signingOut ? 'Signing out…' : 'Sign out'}
            aria-label="Sign out"
            className="rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
          >
            {signingOut ? <Loader2 className="h-4 w-4 animate-spin" /> : <LogOut className="h-4 w-4" />}
          </button>
        </div>
      )}
    </div>
  )
}
