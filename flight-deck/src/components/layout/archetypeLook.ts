import {
  BarChart3, Binoculars, Blocks, Bot, Briefcase, Bug, ChartPie, ClipboardList, Code2, ConciergeBell,
  DraftingCompass, Eye, Feather, FileChartColumn, Filter, FlaskConical, GitBranch, GitPullRequest, Inbox,
  Map as MapIcon, Megaphone, Microscope, Milestone, NotebookPen, Palette, PenLine, Radar, Rocket, ScanEye,
  ScanText, Scissors, ScrollText, SearchCheck, Send, ShieldCheck, Signpost, Telescope, TrendingUp, Workflow,
  type LucideIcon,
} from 'lucide-react'

// How an archetype looks: the New agent gallery and Forge agents. The family
// gives the colour, the archetype its own icon; one an admin added, which isn't
// listed here, wears its family's.

export interface FamilyLook { icon: LucideIcon; text: string; bg: string; ring: string; wash: string }

const FAMILY_LOOKS: Record<string, FamilyLook> = {
  'Research & Intelligence': { icon: Telescope, text: 'text-sky-600 dark:text-sky-300', bg: 'bg-sky-500/15', ring: 'ring-sky-500/20', wash: 'from-sky-500/[0.12]' },
  'Writing & Comms': { icon: PenLine, text: 'text-rose-600 dark:text-rose-300', bg: 'bg-rose-500/15', ring: 'ring-rose-500/20', wash: 'from-rose-500/[0.12]' },
  'Engineering': { icon: Code2, text: 'text-emerald-600 dark:text-emerald-300', bg: 'bg-emerald-500/15', ring: 'ring-emerald-500/20', wash: 'from-emerald-500/[0.12]' },
  'Data & Analysis': { icon: BarChart3, text: 'text-amber-600 dark:text-amber-300', bg: 'bg-amber-500/15', ring: 'ring-amber-500/20', wash: 'from-amber-500/[0.12]' },
  'Ops & Coordination': { icon: ClipboardList, text: 'text-orange-600 dark:text-orange-300', bg: 'bg-orange-500/15', ring: 'ring-orange-500/20', wash: 'from-orange-500/[0.12]' },
  'Investment & VC': { icon: Briefcase, text: 'text-indigo-600 dark:text-indigo-300', bg: 'bg-indigo-500/15', ring: 'ring-indigo-500/20', wash: 'from-indigo-500/[0.12]' },
  'Vision & Multimodal': { icon: Eye, text: 'text-fuchsia-600 dark:text-fuchsia-300', bg: 'bg-fuchsia-500/15', ring: 'ring-fuchsia-500/20', wash: 'from-fuchsia-500/[0.12]' },
}
const DEFAULT_LOOK: FamilyLook = { icon: Bot, text: 'text-violet-600 dark:text-violet-300', bg: 'bg-violet-500/15', ring: 'ring-violet-500/20', wash: 'from-violet-500/[0.12]' }

const ARCHETYPE_ICONS: Record<string, LucideIcon> = {
  'data-analyst': ChartPie,
  'report-builder': FileChartColumn,
  'code-cartographer': MapIcon,
  'code-reviewer': GitPullRequest,
  'debugger': Bug,
  'git-operator': GitBranch,
  'light-planner': Feather,
  'long-horizon-planner': Milestone,
  'qa-engineer': FlaskConical,
  'quick-dirty': Rocket,
  'simplifier': Scissors,
  'security-reviewer': ShieldCheck,
  'architect': DraftingCompass,
  'code-implementer': Blocks,
  'deal-screener': Filter,
  'ic-prep': ScrollText,
  'portfolio-watch': TrendingUp,
  'concierge': ConciergeBell,
  'inbox-manager': Inbox,
  'monitor-watchdog': Radar,
  'project-coordinator': Workflow,
  'triage-router': Signpost,
  'deep-researcher': Microscope,
  'fact-checker': SearchCheck,
  'market-scanner': Binoculars,
  'brand-design-reviewer': Palette,
  'ui-reviewer': ScanEye,
  'visual-extractor': ScanText,
  'editor-writer': NotebookPen,
  'comms-outbound': Send,
  'social-repurposer': Megaphone,
}

export function familyLook(family: string): FamilyLook {
  return FAMILY_LOOKS[family] || DEFAULT_LOOK
}

export function archetypeLook(id: string, family: string): FamilyLook {
  const look = familyLook(family)
  return { ...look, icon: ARCHETYPE_ICONS[id] ?? look.icon }
}
