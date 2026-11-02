import type { Archetype } from '../../services/tierConfig'

export const errorOf = (e: unknown) => (e instanceof Error ? e.message : String(e))

export function roleOf(archetype: string, archetypes: Archetype[]): string {
  return archetypes.find((a) => a.id === archetype)?.role || archetype
}

export function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
