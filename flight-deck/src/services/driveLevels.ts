// Drive: how far Captain pushes to finish a turn (captain_claw/drive.py) —
// the chat box's Drive picker and the agent's Settings. Reasoning effort is
// how hard the model thinks; this is how hard the harness works around it.

export type DriveLevel = 'relaxed' | 'diligent' | 'persistent' | 'ambitious'
export type DrivePick = DriveLevel | 'auto'

export interface DriveInfo {
  id: DriveLevel
  label: string
  /** One line: what you get, and what it costs. */
  hint: string
  description: string
}

export const DRIVE_LEVELS: DriveInfo[] = [
  { id: 'relaxed', label: 'Relaxed', hint: 'Takes the first complete answer. Fastest, cheapest.',
    description: 'Fewer steps and no second rounds: the first complete answer is the answer. For quick questions and cheap routine jobs.' },
  { id: 'diligent', label: 'Diligent', hint: 'The everyday level.',
    description: 'Works until the task is done, within the normal limits. What every agent did before this setting existed.' },
  { id: 'persistent', label: 'Persistent', hint: 'Keeps going longer. Slower, costs more.',
    description: 'More steps before it gives up and a weak answer is sent back more often. For work that has to be finished.' },
  { id: 'ambitious', label: 'Ambitious', hint: 'As Persistent, and may add the obvious next step. Costs the most.',
    description: 'The limits of Persistent. When the asked work is done it may add one obvious next step nobody asked for — reading and writing files only — and says what it added under the answer.' },
]

export const DRIVE_LABEL: Record<DriveLevel, string> = Object.fromEntries(
  DRIVE_LEVELS.map((l) => [l.id, l.label]),
) as Record<DriveLevel, string>

export function parseDrive(raw: unknown): DrivePick {
  return DRIVE_LEVELS.some((l) => l.id === raw) ? (raw as DriveLevel) : 'auto'
}

// What a cognitive mode suggests when the agent has no default of its own
// (cognitive_mode.py `suggested_drive`).
const MODE_SUGGESTS: Record<string, DriveLevel> = { ionian: 'persistent', phrygian: 'persistent' }

export function suggestedDriveFor(mode: string): DriveLevel | '' {
  return MODE_SUGGESTS[mode] ?? ''
}
