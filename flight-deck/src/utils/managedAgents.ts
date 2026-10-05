// Agents Flight Deck spawns and stops itself — the workers of a Basna / Vatra /
// Council / Dubina run and an Iskra being's body — by the names it generates
// for them (not a bare prefix: "Council notes" is somebody's own agent).
//
// Mirrored server-side by agent_sharing.is_managed_agent: such workers can't
// be shared, and have no power switch of their own.
const MANAGED_AGENT = /^(?:(?:basna|vatra)-[0-9a-f]{8}-|council-[0-9a-f]{6}-|iskra-.+-[0-9a-f]{4}$)/

export function isManagedAgent(slug: string, description: string): boolean {
  return MANAGED_AGENT.test(slug) || (slug.startsWith('dubina-') && description.startsWith('Dubina ephemeral'))
}

/** A docker agent's slug as Flight Deck derives it (server `_slug`):
 *  lower-cased, anything outside `[a-z0-9-]` becomes `-`, dashes trimmed. */
export function dockerSlug(name: string): string {
  return (name || 'cc-agent').toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/^-+|-+$/g, '') || 'cc-agent'
}
