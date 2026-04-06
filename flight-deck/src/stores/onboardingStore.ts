import { create } from 'zustand'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'

const STORAGE_KEY = 'fd:onboarding'

export interface OnboardingState {
  desktop: boolean
  forge: boolean
  council: boolean
}

interface OnboardingStore {
  completed: OnboardingState
  dismissedHints: string[]
  completeStep: (step: keyof OnboardingState) => void
  dismissHint: (hintId: string) => void
  isHintDismissed: (hintId: string) => boolean
  resetOnboarding: () => void
}

function load(): { completed: OnboardingState; dismissedHints: string[] } {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    return {
      completed: {
        desktop: raw.completed?.desktop ?? false,
        forge: raw.completed?.forge ?? false,
        council: raw.completed?.council ?? false,
      },
      dismissedHints: raw.dismissedHints ?? [],
    }
  } catch {
    return { completed: { desktop: false, forge: false, council: false }, dismissedHints: [] }
  }
}

function save(state: { completed: OnboardingState; dismissedHints: string[] }) {
  const val = JSON.stringify(state)
  if (useAuthStore.getState().authEnabled) {
    queueSave(STORAGE_KEY, val)
  } else {
    localStorage.setItem(STORAGE_KEY, val)
  }
}

export const useOnboardingStore = create<OnboardingStore>((set, get) => {
  const initial = load()
  return {
    completed: initial.completed,
    dismissedHints: initial.dismissedHints,

    completeStep: (step) => {
      const completed = { ...get().completed, [step]: true }
      const state = { completed, dismissedHints: get().dismissedHints }
      save(state)
      set({ completed })
    },

    dismissHint: (hintId) => {
      if (get().dismissedHints.includes(hintId)) return
      const dismissedHints = [...get().dismissedHints, hintId]
      const state = { completed: get().completed, dismissedHints }
      save(state)
      set({ dismissedHints })
    },

    isHintDismissed: (hintId) => get().dismissedHints.includes(hintId),

    resetOnboarding: () => {
      const state = { completed: { desktop: false, forge: false, council: false }, dismissedHints: [] }
      save(state)
      set(state)
    },
  }
})

registerHydrator((settings) => {
  const raw = settings[STORAGE_KEY]
  if (raw) {
    try {
      const parsed = JSON.parse(raw)
      useOnboardingStore.setState({
        completed: {
          desktop: parsed.completed?.desktop ?? false,
          forge: parsed.completed?.forge ?? false,
          council: parsed.completed?.council ?? false,
        },
        dismissedHints: parsed.dismissedHints ?? [],
      })
    } catch { /* ignore */ }
  }
})
