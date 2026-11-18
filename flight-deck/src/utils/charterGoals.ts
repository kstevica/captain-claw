// A factory's goals as the charter editor holds them: each on (checked on every run)
// or off (kept with the version, never checked).

export interface Goal { id: string; text: string; on: boolean }

let seq = 0
export const newGoal = (text: string, on = true): Goal => ({ id: `g${++seq}`, text, on })

/** The goals of a version: what it checks, then what was switched off. */
export function goalsOf(checks: { contract: string[]; off?: string[] }): Goal[] {
  return [...checks.contract.map((t) => newGoal(t, true)), ...(checks.off ?? []).map((t) => newGoal(t, false))]
}

/** Back to the version's fields: on goals are checked, off ones only kept. */
export function checksOf(goals: Goal[]): { contract: string[]; off: string[] } {
  const clean = (g: Goal) => g.text.replace(/\s+/g, ' ').trim()
  return { contract: goals.filter((g) => g.on).map(clean).filter(Boolean), off: goals.filter((g) => !g.on).map(clean).filter(Boolean) }
}

/** A goal Captain can count or test itself (rows, columns, links, words, sections, tests)
 *  or one the reviewers judge. A hint from its words: the run decides which checks it gets. */
const COUNTS = /\b(at least|at most|exactly|no more than|fewer than|more than|up to|between)\s+\d|\d[\d,.]*\s*(%|percent\b|rows?\b|items?\b|entries\b|leads\b|records\b|words\b|pages\b|columns?\b|sections?\b|files?\b)/i
const SHAPES = /\bcolumns?\b|\bunique\b|\bno (repeated|duplicate)|\bduplicates?\b|\b(is|are) (a )?links?\b|\blinks? to\b|role inbox|info@|free[- ]mail|\bsections?\b|tests? pass|\.(csv|tsv|md|json|xlsx|docx)\b|\bheaders?\b|\bempty\b/i

export function goalKind(text: string): 'countable' | 'judgement' {
  return COUNTS.test(text) || SHAPES.test(text) ? 'countable' : 'judgement'
}
