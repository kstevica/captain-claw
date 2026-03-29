import { create } from 'zustand'

type Theme = 'dark' | 'light'

const STORAGE_KEY = 'fd:theme'

function load(): Theme {
  return (localStorage.getItem(STORAGE_KEY) as Theme) || 'dark'
}

interface ThemeStore {
  theme: Theme
  setTheme: (t: Theme) => void
  toggle: () => void
}

export const useThemeStore = create<ThemeStore>((set, get) => ({
  theme: load(),

  setTheme: (theme) => {
    localStorage.setItem(STORAGE_KEY, theme)
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
