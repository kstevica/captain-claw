import { adminHeaders } from '../admin/deckConfig'
import type { MailOptions, Provider } from '../admin/InvitesAdmin'

// The Activity page's endpoints (captain_claw/flight_deck/user_activity.py):
// who is active, who an admin wrote to, the admin's own Gmail, and writing.

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: adminHeaders(), credentials: 'include' })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(typeof body.detail === 'string' ? body.detail : `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export type ActivityStatus = 'active' | 'quiet' | 'never'

/** feedback / message: an email the deck sent; noted: the admin wrote another way. */
export interface Contact {
  id: string
  kind: 'feedback' | 'message' | 'noted'
  channel: string
  subject: string
  note: string
  created_at: string
  admin_id: string
  admin_name: string
}

export interface ActivityDay {
  day: string       // YYYY-MM-DD, UTC
  messages: number  // typed by them
  tokens: number    // spent by their agents
}

export interface ActivityUser {
  id: string
  email: string
  display_name: string
  role: string
  kiosk: boolean
  created_at: string
  last_login_at: string
  status: ActivityStatus
  typed: {
    first_day: string
    last_day: string
    days_all: number
    messages_all: number
    messages_week: number
    messages: number   // in the window
    days: number       // in the window
  }
  agents: { last_day: string; tokens: number; days: number }
  days: ActivityDay[]
  contacts: Contact[]
  contacts_total: number
}

export interface ActivityReport {
  today: string
  since: string
  days: number
  active_days: number
  users: ActivityUser[]
}

export interface MailHit { at: string; subject: string }

export interface Mailbox {
  provider: 'google'
  address: string
  available: boolean
  reason: string
  partial: boolean
  checked_at: string
  users: Record<string, { sent: MailHit | null; received: MailHit | null }>
}

export interface MessageDraft {
  to: string
  subject: string
  body: string
  mail: MailOptions
  asked_at?: string
}

export interface SendOutcome {
  ok: boolean
  email: { from: string; error: string }
}

export function fetchActivity(days: number): Promise<ActivityReport> {
  return call(`/fd/admin/activity?days=${days}`)
}

export function fetchMailbox(refresh = false): Promise<Mailbox> {
  return call(`/fd/admin/activity/mailbox${refresh ? '?refresh=true' : ''}`)
}

export type Template = 'message' | 'feedback'

function draftUrl(userId: string, template: Template): string {
  return template === 'feedback'
    ? `/fd/admin/users/${userId}/feedback-request`
    : `/fd/admin/activity/${userId}/message`
}

export function fetchDraft(userId: string, template: Template): Promise<MessageDraft> {
  return call(draftUrl(userId, template))
}

/** Ask for feedback also marks them asked on the Users page. */
export function sendEmail(userId: string, template: Template, subject: string, body: string, via: Provider): Promise<SendOutcome> {
  return call(draftUrl(userId, template), { method: 'POST', body: JSON.stringify({ subject, body, send_via: via }) })
}

export function noteContact(userId: string, channel: string, note: string): Promise<{ ok: boolean; contact: Contact }> {
  return call(`/fd/admin/activity/${userId}/contacts`, { method: 'POST', body: JSON.stringify({ channel, note }) })
}

export function deleteContact(contactId: string): Promise<{ ok: boolean }> {
  return call(`/fd/admin/activity/contacts/${contactId}`, { method: 'DELETE' })
}
