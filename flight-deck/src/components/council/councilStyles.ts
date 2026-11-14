import {
  Bug, ClipboardCheck, Lightbulb, Map, MessageCircleQuestion, MessagesSquare, ShieldAlert, Swords,
  type LucideIcon,
} from 'lucide-react'
import type { SessionType } from '../../stores/councilStore'

// Council's style values: its tones, a colour per session type and per agent,
// and a small-button class. The components that use them are in councilLook.tsx
// (kept apart so Fast Refresh sees a file of components only).

export type CouncilTone = 'violet' | 'fuchsia' | 'rose' | 'amber' | 'orange' | 'emerald' | 'teal' | 'sky' | 'cyan' | 'zinc'

/** An icon's tinted square (KioskMenuSheet's tiles). */
export const TONE_SQUARE: Record<CouncilTone, string> = {
  violet: 'bg-violet-500/15 text-violet-600 ring-violet-500/20 dark:text-violet-300',
  fuchsia: 'bg-fuchsia-500/15 text-fuchsia-600 ring-fuchsia-500/20 dark:text-fuchsia-300',
  rose: 'bg-rose-500/15 text-rose-600 ring-rose-500/20 dark:text-rose-300',
  amber: 'bg-amber-500/15 text-amber-600 ring-amber-500/20 dark:text-amber-300',
  orange: 'bg-orange-500/15 text-orange-600 ring-orange-500/20 dark:text-orange-300',
  emerald: 'bg-emerald-500/15 text-emerald-600 ring-emerald-500/20 dark:text-emerald-300',
  teal: 'bg-teal-500/15 text-teal-600 ring-teal-500/20 dark:text-teal-300',
  sky: 'bg-sky-500/15 text-sky-600 ring-sky-500/20 dark:text-sky-300',
  cyan: 'bg-cyan-500/15 text-cyan-600 ring-cyan-500/20 dark:text-cyan-300',
  zinc: 'bg-zinc-500/15 text-zinc-600 ring-zinc-500/20 dark:text-zinc-300',
}

/** A soft pill in a tone (beyond the console's Pill tones). */
export const TONE_PILL: Record<CouncilTone, string> = {
  violet: 'bg-violet-500/10 text-violet-700 dark:text-violet-300',
  fuchsia: 'bg-fuchsia-500/10 text-fuchsia-700 dark:text-fuchsia-300',
  rose: 'bg-rose-500/10 text-rose-700 dark:text-rose-300',
  amber: 'bg-amber-500/10 text-amber-700 dark:text-amber-300',
  orange: 'bg-orange-500/10 text-orange-700 dark:text-orange-300',
  emerald: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
  teal: 'bg-teal-500/10 text-teal-700 dark:text-teal-300',
  sky: 'bg-sky-500/10 text-sky-700 dark:text-sky-300',
  cyan: 'bg-cyan-500/10 text-cyan-700 dark:text-cyan-300',
  zinc: 'bg-zinc-800 text-zinc-400',
}

// ── Session types ──

export const TYPE_LOOK: Record<SessionType, { label: string; icon: LucideIcon; tone: CouncilTone }> = {
  debate: { label: 'Debate', icon: Swords, tone: 'rose' },
  brainstorm: { label: 'Brainstorm', icon: Lightbulb, tone: 'amber' },
  review: { label: 'Review', icon: ClipboardCheck, tone: 'sky' },
  planning: { label: 'Planning', icon: Map, tone: 'emerald' },
  interview: { label: 'Interview', icon: MessageCircleQuestion, tone: 'cyan' },
  troubleshoot: { label: 'Troubleshoot', icon: Bug, tone: 'orange' },
  critique: { label: 'Critique', icon: ShieldAlert, tone: 'fuchsia' },
  freeform: { label: 'Freeform', icon: MessagesSquare, tone: 'violet' },
}

export function typeLook(t: string) {
  return TYPE_LOOK[t as SessionType] || TYPE_LOOK.brainstorm
}

// ── Agents ──

const AVATAR_TONES: CouncilTone[] = ['violet', 'sky', 'emerald', 'amber', 'fuchsia', 'cyan', 'rose', 'teal', 'orange']

function hash(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
  return Math.abs(h)
}

export function agentTone(key: string): CouncilTone {
  return AVATAR_TONES[hash(key || '?') % AVATAR_TONES.length]
}

/** A small button in a panel's head (the console's Button, a size smaller). */
export const SMALL_BTN = 'px-2.5! py-1! text-xs! [&>svg]:h-3.5! [&>svg]:w-3.5!'
