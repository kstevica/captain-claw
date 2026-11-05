import { create } from 'zustand'
import { useAuthStore, registerSignOutTeardown, selectConnectionsAvailable } from './authStore'
import { fdFetch } from '../services/notifications'

// The signed-in user's IMAP/SMTP inboxes (/fd/imap/*): any number of them,
// each with its own settings, sending policy, signature and agent access.
// Passwords go to the deck once and never come back.

export type ImapSecurity = 'ssl' | 'starttls' | 'none'

export interface ImapSending {
  enabled: boolean
  allowed_recipients: string[]
  daily_limit: number
}

export interface ImapAccount {
  id: string
  label: string
  email: string
  username: string
  display_name: string
  imap_host: string
  imap_port: number
  imap_security: ImapSecurity
  smtp_host: string
  smtp_port: number
  smtp_security: ImapSecurity
  smtp_username: string
  save_sent: boolean
  can_send: boolean
  separate_smtp_login: boolean
  agents: string[]
  sending: ImapSending
  signature: string
  sent_last_24h: number
  created_at: string
  updated_at: string
  last_check: { ok?: boolean; at?: string; roles?: string[] }
  provider: string
}

export interface ImapSuggestion {
  source: 'preset' | 'autoconfig' | 'guess'
  provider: string
  username: string
  imap_host: string
  imap_port: number
  imap_security: ImapSecurity
  smtp_host: string
  smtp_port: number
  smtp_security: ImapSecurity
  save_sent: boolean
  hint: string
  microsoft: boolean
}

// The other standard port, tried when the typed one didn't answer.
export interface ImapAlternative {
  port: number
  security: ImapSecurity
  ok: boolean
  error?: string
}

export interface ImapServerCheck {
  ok: boolean
  error?: string
  unreachable?: boolean
  alternative?: ImapAlternative
}

export interface ImapTestResult {
  ok: boolean
  imap: ImapServerCheck & { folders?: number; roles?: string[] }
  smtp: ImapServerCheck & { skipped?: boolean }
}

export interface ImapSend {
  id: string
  status: 'sent' | 'unknown'
  account_id: string
  account_email: string
  agent: string
  to: string
  cc: string
  bcc: string
  subject: string
  message_id: string
  draft_id: string
  attachments: string[]
  created_at: string
}

// What the add / edit forms send. Passwords only when typed.
export interface ImapAccountPatch {
  label?: string
  email?: string
  username?: string
  password?: string
  display_name?: string
  imap_host?: string
  imap_port?: number
  imap_security?: ImapSecurity
  smtp_host?: string
  smtp_port?: number
  smtp_security?: ImapSecurity
  smtp_username?: string
  smtp_password?: string
  save_sent?: boolean
  agents?: string[]
  sending?: Partial<ImapSending>
  signature?: string
  provider?: string
  account_id?: string
}

interface ImapMailStore {
  accounts: ImapAccount[] | null
  maxAccounts: number
  deckSendDisabled: boolean
  cryptoAvailable: boolean
  loading: boolean
  error: string | null
  sends: Record<string, ImapSend[]>
  refresh: () => Promise<void>
  suggest: (email: string) => Promise<ImapSuggestion | null>
  test: (body: ImapAccountPatch) => Promise<{ result?: ImapTestResult; error?: string }>
  // adjusted: a port Flight Deck switched because the typed one didn't answer.
  add: (body: ImapAccountPatch) => Promise<{ account?: ImapAccount; error?: string; adjusted?: string[] }>
  update: (id: string, patch: ImapAccountPatch) => Promise<{ account?: ImapAccount; error?: string; adjusted?: string[] }>
  remove: (id: string) => Promise<string | null>
  fetchSends: (accountId: string, limit?: number) => Promise<void>
}

async function detailOf(r: Response): Promise<string> {
  try {
    const data = await r.json()
    const d = data?.detail
    if (typeof d === 'string') return d
    if (Array.isArray(d)) return d.map((x) => x?.msg || String(x)).join('; ')
  } catch { /* not JSON */ }
  return `${r.status} ${r.statusText}`
}

const AUTH_OFF = "Email (IMAP) isn't available on this deck: Flight Deck sign-in is turned off."

function jsonInit(method: string, body: unknown): RequestInit {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}

export const useImapMailStore = create<ImapMailStore>((set, get) => ({
  accounts: null,
  maxAccounts: 20,
  deckSendDisabled: false,
  cryptoAvailable: true,
  loading: false,
  error: null,
  sends: {},

  refresh: async () => {
    if (!selectConnectionsAvailable(useAuthStore.getState())) {
      set({ accounts: [], loading: false, error: null })
      return
    }
    set({ loading: true, error: null })
    try {
      const r = await fdFetch('/fd/imap/accounts')
      if (!r.ok) throw new Error(await detailOf(r))
      const data = await r.json()
      set({
        accounts: Array.isArray(data?.accounts) ? data.accounts : [],
        maxAccounts: Number(data?.max_accounts) || 20,
        deckSendDisabled: !!data?.deck_send_disabled,
        cryptoAvailable: data?.crypto_available !== false,
        loading: false,
      })
    } catch (exc) {
      set({ loading: false, error: exc instanceof Error ? exc.message : String(exc) })
    }
  },

  suggest: async (email) => {
    try {
      const r = await fdFetch(`/fd/imap/suggest?email=${encodeURIComponent(email)}`)
      if (!r.ok) return null
      return (await r.json()) as ImapSuggestion
    } catch {
      return null
    }
  },

  test: async (body) => {
    if (!selectConnectionsAvailable(useAuthStore.getState())) return { error: AUTH_OFF }
    try {
      const r = await fdFetch('/fd/imap/test', jsonInit('POST', body))
      if (!r.ok) return { error: await detailOf(r) }
      return { result: (await r.json()) as ImapTestResult }
    } catch (exc) {
      return { error: exc instanceof Error ? exc.message : String(exc) }
    }
  },

  add: async (body) => {
    if (!selectConnectionsAvailable(useAuthStore.getState())) return { error: AUTH_OFF }
    try {
      const r = await fdFetch('/fd/imap/accounts', jsonInit('POST', body))
      if (!r.ok) return { error: await detailOf(r) }
      const data = await r.json()
      const account = data?.account as ImapAccount
      set({ accounts: [...(get().accounts || []), account] })
      return { account, adjusted: Array.isArray(data?.adjusted) ? data.adjusted : [] }
    } catch (exc) {
      return { error: exc instanceof Error ? exc.message : String(exc) }
    }
  },

  update: async (id, patch) => {
    try {
      const r = await fdFetch(`/fd/imap/accounts/${encodeURIComponent(id)}`, jsonInit('PUT', patch))
      if (!r.ok) return { error: await detailOf(r) }
      const data = await r.json()
      const account = data?.account as ImapAccount
      set({ accounts: (get().accounts || []).map((a) => (a.id === id ? account : a)) })
      return { account, adjusted: Array.isArray(data?.adjusted) ? data.adjusted : [] }
    } catch (exc) {
      return { error: exc instanceof Error ? exc.message : String(exc) }
    }
  },

  remove: async (id) => {
    try {
      const r = await fdFetch(`/fd/imap/accounts/${encodeURIComponent(id)}`, { method: 'DELETE' })
      if (!r.ok) return await detailOf(r)
      set({ accounts: (get().accounts || []).filter((a) => a.id !== id) })
      return null
    } catch (exc) {
      return exc instanceof Error ? exc.message : String(exc)
    }
  },

  fetchSends: async (accountId, limit = 10) => {
    try {
      const r = await fdFetch(`/fd/imap/sends?account_id=${encodeURIComponent(accountId)}&limit=${limit}`)
      if (!r.ok) return
      const data = await r.json()
      set({ sends: { ...get().sends, [accountId]: Array.isArray(data?.sends) ? data.sends : [] } })
    } catch { /* keep what we had */ }
  },
}))

registerSignOutTeardown(() => useImapMailStore.setState({ accounts: null, sends: {}, error: null }))
