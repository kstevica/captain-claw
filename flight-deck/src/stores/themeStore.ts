import { create } from 'zustand'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'

type Theme = 'dark' | 'light'

const STORAGE_KEY = 'fd:theme'

function load(): Theme {
  return (localStorage.getItem(STORAGE_KEY) as Theme) || 'dark'
}

function persist(theme: Theme) {
  if (useAuthStore.getState().authEnabled) {
    queueSave(STORAGE_KEY, theme)
  } else {
    localStorage.setItem(STORAGE_KEY, theme)
  }
}

interface ThemeStore {
  theme: Theme
  setTheme: (t: Theme) => void
  toggle: () => void
}

export const useThemeStore = create<ThemeStore>((set, get) => ({
  theme: load(),

  setTheme: (theme) => {
    persist(theme)
    applyTheme(theme)
    set({ theme })
  },

  toggle: () => {
    const next = get().theme === 'dark' ? 'light' : 'dark'
    get().setTheme(next)
  },
}))

function applyTheme(theme: Theme) {
  const root = document.documentElement
  if (theme === 'light') {
    root.classList.add('light')
    root.classList.remove('dark')
  } else {
    root.classList.add('dark')
    root.classList.remove('light')
  }
}

// Apply on load
applyTheme(load())

// Register hydrator for server-side settings
registerHydrator((settings) => {
  const theme = settings[STORAGE_KEY] as Theme
  if (theme && (theme === 'dark' || theme === 'light')) {
    applyTheme(theme)
    useThemeStore.setState({ theme })
  }
})
