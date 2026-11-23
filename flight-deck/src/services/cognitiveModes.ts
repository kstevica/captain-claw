// The cognitive modes an agent can think in (captain_claw/cognitive_mode.py),
// with how Flight Deck shows them — the agent cards' selector and the kiosk's
// agent Settings.

export interface ModeInfo {
  id: string
  label: string
  character: string
  description: string
  color: string      // tailwind bg class for the dot
  textColor: string  // tailwind text class for active state
  bgActive: string   // tailwind bg class for active row
}

export const MODES: ModeInfo[] = [
  { id: 'neutra',     label: 'Neutra',     character: 'Balanced generalist',
    description: 'No mode. The agent works the way its role and instructions say, with no added leaning.',
    color: 'bg-zinc-500',    textColor: 'text-zinc-400',    bgActive: 'bg-zinc-500/10' },
  { id: 'ionian',     label: 'Ionian',     character: 'Convergent problem-solving',
    description: 'Works toward closure: fixes what done looks like, takes the most direct route, gives one clear recommendation and a short answer. Direct about what it knows and about what it could not confirm.',
    color: 'bg-amber-400',   textColor: 'text-amber-400',   bgActive: 'bg-amber-500/10' },
  { id: 'dorian',     label: 'Dorian',     character: 'Empathetic pragmatism',
    description: 'Works from your real situation: who will use the result and under which limits. Names the real trade-off, picks the best realistic option, and notes what was left for later.',
    color: 'bg-teal-400',    textColor: 'text-teal-400',    bgActive: 'bg-teal-500/10' },
  { id: 'phrygian',   label: 'Phrygian',   character: 'Adversarial analysis',
    description: 'Looks for what would make the work fail: wrong figures, outdated sources, unhandled cases. Checks instead of guessing, looks hardest where the stakes are high, and says what was and was not verified.',
    color: 'bg-red-400',     textColor: 'text-red-400',     bgActive: 'bg-red-500/10' },
  { id: 'lydian',     label: 'Lydian',     character: 'Creative exploration',
    description: 'Widens the field on open questions: a few directions that differ in kind, each weighed, then one recommendation taken far enough to use. Closed tasks are simply done.',
    color: 'bg-violet-400',  textColor: 'text-violet-400',  bgActive: 'bg-violet-500/10' },
  { id: 'mixolydian', label: 'Mixolydian', character: 'Iterative building',
    description: 'Gets a small working version first, tries it for real, and improves in small steps. Delivers something usable as it stands and notes the next improvement.',
    color: 'bg-orange-400',  textColor: 'text-orange-400',  bgActive: 'bg-orange-500/10' },
  { id: 'aeolian',    label: 'Aeolian',    character: 'Deep research',
    description: 'Reads the primary material before concluding, traces causes, and shows where each important claim comes from. Depth goes into the work; a simple question still gets a short answer.',
    color: 'bg-blue-400',    textColor: 'text-blue-400',    bgActive: 'bg-blue-500/10' },
  { id: 'locrian',    label: 'Locrian',    character: 'Deconstruction',
    description: 'Asks whether this is the right problem and whether less would do. Looks for what can be removed, says the uncomfortable option once, then does the work unless the premise is plainly broken.',
    color: 'bg-fuchsia-400', textColor: 'text-fuchsia-400', bgActive: 'bg-fuchsia-500/10' },
]
