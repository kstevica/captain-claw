import { create } from 'zustand'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'
import { parseDrive, type DrivePick } from '../services/driveLevels'

// The Drive level picked in the chat box, per agent ("auto" is not stored: no
// entry means the agent's own default). Synced with the user's deck settings,
// so the pick follows them to another device.

const STORAGE_KEY = 'fd:drive'

function clean(raw: unknown): Record<string, DrivePick> {
  const out: Record<string, DrivePick> = {}
  if (raw && typeof raw === 'object') {
    for (const [agent, pick] of Object.entries(raw as Record<string, unknown>)) {
      const p = parseDrive(pick)
      if (p !== 'auto') out[agent] = p
    }
  }
  return out
}
function load(): Record<string, DrivePick> {
  try { return clean(JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')) } catch { return {} }
}
function save(byAgent: Record<string, DrivePick>) {
  const val = JSON.stringify(byAgent)
  try { localStorage.setItem(STORAGE_KEY, val) } catch { /* private mode */ }
  if (useAuthStore.getState().authEnabled) queueSave(STORAGE_KEY, val)
}

interface DriveStore {
  byAgent: Record<string, DrivePick>
  setDrive: (agentId: string, pick: DrivePick) => void
}

export const useDriveStore = create<DriveStore>((set, get) => ({
  byAgent: load(),

  setDrive: (agentId, pick) => {
    if (!agentId) return
    const byAgent = { ...get().byAgent }
    if (pick === 'auto') delete byAgent[agentId]
    else byAgent[agentId] = pick
    save(byAgent)
    set({ byAgent })
  },
}))

export function driveFor(agentId: string): DrivePick {
  return useDriveStore.getState().byAgent[agentId] ?? 'auto'
}

// Absent key → reset, so a previous user's picks don't carry into the next
// account (see pinnedStore).
registerHydrator((settings) => {
  const raw = settings[STORAGE_KEY]
  if (!raw) {
    useDriveStore.setState({ byAgent: {} })
    return
  }
  try { useDriveStore.setState({ byAgent: clean(JSON.parse(raw)) }) } catch { /* ignore */ }
})
