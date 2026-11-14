import { Flame, Network, type LucideIcon } from 'lucide-react'

// The Basna / Vatra pages' style values: tones, the two modes' look, and the
// class strings of their compact controls. The components that use them are in
// look.tsx (kept apart so Fast Refresh sees a file of components only).

export type LookTone = 'violet' | 'sky' | 'emerald' | 'amber' | 'red' | 'zinc'

// The kiosk menu's tinted squares (icon colour + soft fill + ring).
export const SQUARE_TONE: Record<LookTone, string> = {
  violet: 'bg-violet-500/15 text-violet-600 ring-violet-500/20 dark:text-violet-300',
  sky: 'bg-sky-500/15 text-sky-600 ring-sky-500/20 dark:text-sky-300',
  emerald: 'bg-emerald-500/15 text-emerald-600 ring-emerald-500/20 dark:text-emerald-300',
  amber: 'bg-amber-500/15 text-amber-600 ring-amber-500/20 dark:text-amber-300',
  red: 'bg-red-500/15 text-red-600 ring-red-500/20 dark:text-red-300',
  zinc: 'bg-zinc-500/15 text-zinc-600 ring-zinc-500/20 dark:text-zinc-300',
}

// The two ways to run a team: name, icon, colour and a line on what each does.
export const MODE_LOOK: Record<'basna' | 'vatra', {
  name: string; icon: LucideIcon; tone: LookTone; blurb: string; short: string
}> = {
  basna: {
    name: 'Basna',
    icon: Network,
    tone: 'sky',
    blurb: 'Several agents answer on their own; the best of each is merged into one answer.',
    short: 'Agents answer on their own, merged by reliability',
  },
  vatra: {
    name: 'Vatra',
    icon: Flame,
    tone: 'violet',
    blurb: 'A lead splits one job, specialists work on it together, and a reporter assembles one result.',
    short: 'A lead splits the work, a reporter assembles it',
  },
}

// A compact input for the dense spots (numbers, small selects).
export const smallInputCls =
  'rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/15 disabled:opacity-50'

// A segmented control's frame and its buttons, for the spots that need their
// own (icons in colour, counts, small sizes).
export const segFrameCls = 'inline-flex max-w-full flex-wrap gap-1 rounded-xl border border-zinc-800 bg-zinc-950/60 p-1'
export const segBtnCls = (on: boolean) =>
  `flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
    on ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-500 hover:text-zinc-200'}`

// A small icon-only action (fullscreen, export, delete …).
export const iconBtnCls = 'rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40'

// The panel shell of the run views (board, progress, files …).
export const panelCls = 'rounded-2xl border border-zinc-800 bg-zinc-900/40'

// A Vatra panel: the card with a faint violet wash.
export const vatraPanelCls = 'rounded-2xl border border-violet-500/25 bg-violet-500/[0.04] p-4 wide:p-5'

// The run button in the mode's colour: sky for Basna, violet for Vatra.
export const ctaCls = (tone: 'violet' | 'sky') =>
  `inline-flex items-center justify-center gap-1.5 rounded-xl px-3.5 py-2 text-sm font-medium text-white shadow-sm transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
    tone === 'sky' ? 'bg-sky-600 shadow-sky-900/20 hover:bg-sky-500' : 'bg-violet-600 shadow-violet-900/20 hover:bg-violet-500'}`

// A small on/off button (a disclosure that's open, a panel that's shown).
export const toggleBtnCls = (on: boolean) =>
  `inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors ${
    on ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`

// A button that opens a picker, tinted once something in it is picked.
export const chipToggleCls = (picked: boolean, tone: 'violet' | 'emerald') =>
  `inline-flex items-center gap-2 rounded-xl border px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
    !picked ? 'border-zinc-700 bg-zinc-900/60 text-zinc-300 hover:border-zinc-600 hover:bg-zinc-800'
      : tone === 'violet' ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-300'
        : 'border-emerald-500/50 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'}`

export const checkCls = 'h-4 w-4 shrink-0 rounded border-zinc-600 accent-violet-600'

// A mode tile when it's the one picked.
export const MODE_TILE_ON: Record<LookTone, string> = {
  sky: 'border-sky-500/50 bg-sky-500/[0.07] ring-1 ring-sky-500/20',
  violet: 'border-violet-500/50 bg-violet-500/[0.07] ring-1 ring-violet-500/20',
  emerald: 'border-emerald-500/50 bg-emerald-500/[0.07] ring-1 ring-emerald-500/20',
  amber: 'border-amber-500/50 bg-amber-500/[0.07] ring-1 ring-amber-500/20',
  red: 'border-red-500/50 bg-red-500/[0.07] ring-1 ring-red-500/20',
  zinc: 'border-zinc-600 bg-zinc-800/50',
}

// The header's icon-only buttons (fullscreen).
export const headerIconBtnCls = 'rounded-xl px-2.5 py-2 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200'

// A small icon-only delete.
export const dangerIconBtnCls = 'rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-red-500/10 hover:text-red-600 disabled:opacity-40 dark:hover:text-red-400'
