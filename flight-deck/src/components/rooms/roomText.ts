import type { Room } from '../../services/roomsApi'

// Plain helpers the data room's views share (no components here, so fast
// refresh keeps working in the files that hold them).

export function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

export function count(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`
}

/** "company" → "companies", "address" → "addresses", "person" → "people". */
export function plural(word: string, n: number): string {
  if (n === 1) return word
  const special: Record<string, string> = { person: 'people', 'date of birth': 'dates of birth' }
  if (special[word]) return special[word]
  if (/[^aeiou]y$/i.test(word)) return `${word.slice(0, -1)}ies`
  if (/(s|x|z|ch|sh)$/i.test(word)) return `${word}es`
  return `${word}s`
}

export function tokenKind(token: string): string {
  return token.replace(/^⟦|⟧$/g, '').replace(/_\d+(:[a-z0-9]+)?$/, '')
}

/** "⟦ORG_7:2⟧" → "⟦ORG_7⟧": the entity a spelling belongs to. */
export function baseToken(token: string): string {
  return token.replace(/:[a-z0-9]{1,3}⟧$/, '⟧')
}

export type Range = readonly [number, number]

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Where *needles* appear in *text*, case-insensitively, longest first, no overlaps. */
export function findAll(text: string, needles: string[]): Range[] {
  const words = [...new Set(needles.map((n) => n.trim()).filter((n) => n.length > 0))].sort((a, b) => b.length - a.length)
  if (!words.length) return []
  const re = new RegExp(words.map(esc).join('|'), 'giu')
  const out: Range[] = []
  for (const m of text.matchAll(re)) {
    if (!m[0]) continue
    out.push([m.index ?? 0, (m.index ?? 0) + m[0].length])
  }
  return out
}

export function findRe(text: string, re: RegExp): Range[] {
  const g = new RegExp(re.source, re.flags.includes('g') ? re.flags : `${re.flags}g`)
  return [...text.matchAll(g)].filter((m) => m[0]).map((m) => [m.index ?? 0, (m.index ?? 0) + m[0].length] as Range)
}

export const STATE_LABEL: Record<Room['state'], string> = {
  setup: 'Setting up', running: 'Running', review: 'Ready to check', released: 'Released',
}
export const STATE_TONE: Record<Room['state'], 'zinc' | 'sky' | 'amber' | 'emerald'> = {
  setup: 'zinc', running: 'sky', review: 'amber', released: 'emerald',
}
