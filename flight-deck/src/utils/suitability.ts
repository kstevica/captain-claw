// Map an agent's self-rated council suitability (0.0–1.0) to a short human label.
export function suitabilityLabel(score: number): string {
  if (score >= 0.96) return 'Perfect fit'
  if (score >= 0.86) return 'On topic'
  if (score >= 0.70) return 'Relevant'
  if (score >= 0.50) return 'Loosely related'
  if (score >= 0.30) return 'Tangential'
  return 'Off topic'
}
