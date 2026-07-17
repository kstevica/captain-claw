// A building holds a slice of every iskra's home (village-world map): the
// Garden their garden/, the Library their reading reports, and so on. Shared
// by the parent map (authed files) and the public /village map (public files).
import type { VillagePlace } from '../../services/beings'

export const PLACE_FOLDER: Record<string, { folder: string; label: string; excl?: string }> = {
  tend: { folder: 'garden/', label: 'the gardens', excl: 'garden/reports/' },
  create: { folder: 'skills/', label: 'what was made here' },
  read: { folder: 'garden/reports/', label: 'the reading room' },
  remember: { folder: 'self/', label: 'who they are' },
}

export const folderFor = (p: VillagePlace) =>
  p.affordances.map((a) => PLACE_FOLDER[a]).find(Boolean) ?? null

// A home path → its bare file name, e.g. "garden/sea-poem.md" → "sea-poem".
export const shortName = (p: string) => p.split('/').pop()?.replace(/\.md$/, '') ?? p

// The noise files every home carries that a browser should skip.
export const isBoilerplate = (path: string) => /README|\.keep/.test(path)
