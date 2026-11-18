// The deck's Bat settings (/fd/admin/bat/settings, admins): how Bat runs here.
// The FD_BAT_* environment values are only the defaults.

import { fdFetch } from './notifications'

export interface BatSettings {
  /** Steps of a run working at the same time. */
  max_parallel: number
  /** The most steps one worker turn may take while it makes progress. */
  step_ceiling: number
  /** How long a step's worker may stay quiet before the step stops. */
  idle_minutes: number
}

export interface BatSettingsView extends BatSettings {
  defaults: BatSettings
  ranges: Record<keyof BatSettings, [number, number]>
}

async function call(init?: RequestInit): Promise<BatSettingsView> {
  const r = await fdFetch('/fd/admin/bat/settings', init)
  if (!r.ok) {
    const body = await r.json().catch(() => ({}))
    throw new Error(typeof body?.detail === 'string' ? body.detail : `HTTP ${r.status}`)
  }
  return r.json()
}

export const getBatSettings = () => call()
export const saveBatSettings = (patch: Partial<BatSettings>) =>
  call({ method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) })
