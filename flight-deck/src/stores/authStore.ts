import { create } from 'zustand'

export interface User {
  id: string
  email: string
  display_name: string
  role: string
}

interface AuthStore {
  user: User | null
  token: string
  isAuthenticated: boolean
  authEnabled: boolean | null  // null = not yet checked

  setAuth: (user: User, token: string) => void
  clearAuth: () => void
  setAuthEnabled: (enabled: boolean) => void
  setToken: (token: string) => void
}

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  token: '',
  isAuthenticated: false,
  authEnabled: null,

  setAuth: (user, token) => set({ user, token, isAuthenticated: true }),
  clearAuth: () => set({ user: null, token: '', isAuthenticated: false }),
  setAuthEnabled: (enabled) => set({ authEnabled: enabled }),
  setToken: (token) => set({ token }),
}))

// ── Auth API calls ──

const FD = '/fd'

export async function checkAuthStatus(): Promise<boolean> {
  try {
    const res = await fetch(`${FD}/auth/status`)
    const data = await res.json()
    const enabled = data.auth_enabled === true
    useAuthStore.getState().setAuthEnabled(enabled)
    return enabled
  } catch {
    useAuthStore.getState().setAuthEnabled(false)
    return false
  }
}

export async function loginUser(email: string, password: string): Promise<User> {
  const res = await fetch(`${FD}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: 'Login failed' }))
    throw new Error(body.detail || 'Login failed')
  }
  const data = await res.json()
  useAuthStore.getState().setAuth(data.user, data.access_token)
  return data.user
}

export async function registerUser(email: string, password: string, displayName: string): Promise<User> {
  const res = await fetch(`${FD}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, password, display_name: displayName }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: 'Registration failed' }))
    throw new Error(body.detail || 'Registration failed')
  }
  const data = await res.json()
  useAuthStore.getState().setAuth(data.user, data.access_token)
  return data.user
}

export async function refreshAccessToken(): Promise<boolean> {
  try {
    const res = await fetch(`${FD}/auth/refresh`, {
      method: 'POST',
      credentials: 'include',
    })
    if (!res.ok) return false
    const data = await res.json()
    useAuthStore.getState().setAuth(data.user, data.access_token)
    return true
  } catch {
    return false
  }
}

export async function logoutUser(): Promise<void> {
  try {
    await fetch(`${FD}/auth/logout`, {
      method: 'POST',
      credentials: 'include',
    })
  } catch { /* ignore */ }
  useAuthStore.getState().clearAuth()
}

export async function updateProfile(data: { display_name?: string; password?: string; current_password?: string }): Promise<User> {
  const { token } = useAuthStore.getState()
  const res = await fetch(`${FD}/auth/me`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: 'Update failed' }))
    throw new Error(body.detail || 'Update failed')
  }
  const user = await res.json()
  useAuthStore.getState().setAuth(user, token)
  return user
}
