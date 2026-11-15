// The Agent store's Featured row. Run: node --test src/components/store/storeFeatured.test.mjs
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { FEATURED_IDS, featured } from './storeFeatured.ts'

const agent = (id) => ({ id, role: id })
const catalog = ['market-scanner', 'editor-writer', 'fact-checker', 'data-analyst', 'inbox-manager', 'deep-researcher', 'debugger']
  .map(agent)

test('the four picks lead, in their own order, wherever the catalog has them', () => {
  assert.deepEqual(featured(catalog).map((a) => a.id), [...FEATURED_IDS])
})

test('a pick the catalog does not offer is skipped and the row fills up in catalog order', () => {
  const without = catalog.filter((a) => a.id !== 'inbox-manager')
  assert.deepEqual(featured(without).map((a) => a.id), ['deep-researcher', 'data-analyst', 'editor-writer', 'market-scanner'])
})

test('a catalog with none of them shows its first four, and a short one all it has', () => {
  const other = ['a', 'b', 'c', 'd', 'e'].map(agent)
  assert.deepEqual(featured(other).map((a) => a.id), ['a', 'b', 'c', 'd'])
  assert.deepEqual(featured(other.slice(0, 2)).map((a) => a.id), ['a', 'b'])
  assert.deepEqual(featured([]), [])
})

test('every pick exists in the built-in registry', async () => {
  const { readFile } = await import('node:fs/promises')
  const registry = JSON.parse(await readFile(new URL('../../../../captain_claw/instructions/archetypes.json', import.meta.url), 'utf8'))
  const ids = new Set(registry.archetypes.map((a) => a.id))
  for (const id of FEATURED_IDS) assert.ok(ids.has(id), `${id} is not in archetypes.json`)
})
