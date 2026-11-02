import {
  BarChart3, Bot, Briefcase, ClipboardList, Code2, Eye, PenLine, Telescope, type LucideIcon,
} from 'lucide-react'

// How an archetype's family looks: the New agent gallery and Forge agents.

export interface FamilyLook { icon: LucideIcon; text: string; bg: string; ring: string }

const FAMILY_LOOKS: Record<string, FamilyLook> = {
  'Research & Intelligence': { icon: Telescope, text: 'text-sky-400', bg: 'bg-sky-500/10', ring: 'ring-sky-500/30' },
  'Writing & Comms': { icon: PenLine, text: 'text-rose-400', bg: 'bg-rose-500/10', ring: 'ring-rose-500/30' },
  'Engineering': { icon: Code2, text: 'text-emerald-400', bg: 'bg-emerald-500/10', ring: 'ring-emerald-500/30' },
  'Data & Analysis': { icon: BarChart3, text: 'text-amber-400', bg: 'bg-amber-500/10', ring: 'ring-amber-500/30' },
  'Ops & Coordination': { icon: ClipboardList, text: 'text-orange-400', bg: 'bg-orange-500/10', ring: 'ring-orange-500/30' },
  'Investment & VC': { icon: Briefcase, text: 'text-indigo-400', bg: 'bg-indigo-500/10', ring: 'ring-indigo-500/30' },
  'Vision & Multimodal': { icon: Eye, text: 'text-fuchsia-400', bg: 'bg-fuchsia-500/10', ring: 'ring-fuchsia-500/30' },
}
const DEFAULT_LOOK: FamilyLook = { icon: Bot, text: 'text-violet-400', bg: 'bg-violet-500/10', ring: 'ring-violet-500/30' }

export function familyLook(family: string): FamilyLook {
  return FAMILY_LOOKS[family] || DEFAULT_LOOK
}
