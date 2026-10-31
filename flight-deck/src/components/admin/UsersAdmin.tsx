import { useCallback, useEffect, useMemo, useState } from 'react'
import { isAppShell } from '../../services/desktopShell'
import {
  UserPlus,
  Trash2,
  ChevronDown,
  ChevronRight,
  Crown,
  Loader2,
  Bot,
  MessageSquare,
  KeyRound,
  Mail,
  RotateCcw,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { useUIStore } from '../../stores/uiStore'
import { MIN_PASSWORD_LENGTH } from '../profile/ChangePasswordCard'
import { UserAgentsDialog } from '../agents/UserAgentsDialog'
import { timeAgo } from '../console/format'
import {
  adminHeaders, fetchPlans, kioskModeNote, matchesQuery, type KioskMode, type PlanLimits, type SystemConfig,
} from './deckConfig'
import { FeedbackRequestPanel } from './FeedbackRequestPanel'
import { FilterSelect, FilterSummary, SearchBox } from './ListFilters'

// Users → Users: every account on the deck — add one, change a plan, role or
// kiosk lock, set a password, manage their agents, delete them, give free
// resets of the daily allowances (to one, the selected or every user, up to
// the deck's cap), see when they last logged in and what they spent, ask them
// for feedback. Search and filters narrow the list in the browser (the newest
// USERS_MAX accounts).

interface UserInfo {
  id: string
  email: string
  display_name: string
  role: string
  created_at: string
  updated_at: string
  metadata: string
  last_login_at?: string  // blank: no login on record
  feedback_asked_at?: string
}

function userKioskFlag(user: UserInfo): boolean {
  try { return JSON.parse(user.metadata || '{}').kiosk === true } catch { return false }
}

// An admin set their password and they haven't chosen their own yet.
function userMustChangePassword(user: UserInfo): boolean {
  try { return JSON.parse(user.metadata || '{}').must_change_password === true } catch { return false }
}

// What the server resolves as the user's lock (auth_routes._user_payload).
function kioskEffective(role: string, flag: boolean, mode: KioskMode): boolean {
  if (role === 'admin') return false
  return mode === 'all' || (mode === 'selected' && flag)
}

function userPlan(user: UserInfo): string {
  try {
    const p = JSON.parse(user.metadata || '{}').plan || 'free'
    return p === 'pro' ? 'premium' : p  // "pro" is "premium" now
  } catch { return 'free' }
}

// The most the list loads (the server's cap); search and filters work on these.
const USERS_MAX = 1000

async function fetchUsers(limit = 100, offset = 0): Promise<{ users: UserInfo[]; total: number }> {
  const res = await fetch(`/fd/admin/users?limit=${limit}&offset=${offset}`, {
    headers: adminHeaders(), credentials: 'include',
  })
  if (!res.ok) throw new Error('Failed to fetch users')
  return res.json()
}

async function createUser(payload: {
  email: string
  password: string
  display_name?: string
  role?: string
  kiosk?: boolean
  must_change_password?: boolean
}): Promise<void> {
  const res = await fetch('/fd/admin/users', {
    method: 'POST', headers: adminHeaders(), credentials: 'include',
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || 'Failed to create user')
  }
}

async function updateUser(userId: string, patch: Record<string, unknown>): Promise<void> {
  const res = await fetch(`/fd/admin/users/${userId}`, {
    method: 'PUT', headers: adminHeaders(), credentials: 'include',
    body: JSON.stringify(patch),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(typeof data.detail === 'string' ? data.detail : 'Failed to update user')
  }
}

async function deleteUser(userId: string): Promise<void> {
  const res = await fetch(`/fd/admin/users/${userId}`, {
    method: 'DELETE', headers: adminHeaders(), credentials: 'include',
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || 'Failed to delete user')
  }
}

// What each user spent today against their plan's daily allowances, the free
// resets each holds (spending one starts their day over) and the most a user
// may hold.
interface DayUsage { messages: number; tokens: number }
interface UsageToday { usage: Record<string, DayUsage>; resets: Record<string, number>; reset_cap: number }

const NO_USAGE: UsageToday = { usage: {}, resets: {}, reset_cap: 3 }

async function fetchUsageToday(): Promise<UsageToday> {
  const res = await fetch('/fd/admin/usage/today', { headers: adminHeaders(), credentials: 'include' })
  if (!res.ok) throw new Error("Failed to fetch today's usage")
  const body = await res.json()
  return { usage: body.usage ?? {}, resets: body.resets ?? {}, reset_cap: body.reset_cap ?? NO_USAGE.reset_cap }
}

interface GrantResult { cap: number; users: number; given: number }

/** Give users *count* more free resets, none past *cap* (saved as the deck's). */
async function grantResets(payload: {
  count: number; cap?: number; all?: boolean; user_ids?: string[]
}): Promise<GrantResult> {
  const res = await fetch('/fd/admin/usage/resets', {
    method: 'POST', headers: adminHeaders(), credentials: 'include',
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(typeof data.detail === 'string' ? data.detail : 'Failed to give resets')
  }
  return res.json()
}

async function setResets(userId: string, balance: number): Promise<void> {
  const res = await fetch(`/fd/admin/users/${userId}/usage/resets`, {
    method: 'PUT', headers: adminHeaders(), credentials: 'include',
    body: JSON.stringify({ balance }),
  })
  if (!res.ok) throw new Error('Failed to change resets')
}

const MAX_RESETS = 100  // usage_quota.MAX_RESETS
const nResets = (n: number) => `${n} free reset${n === 1 ? '' : 's'}`

async function resetUsageToday(userId: string): Promise<void> {
  const res = await fetch(`/fd/admin/users/${userId}/usage/today`, {
    method: 'DELETE', headers: adminHeaders(), credentials: 'include',
  })
  if (!res.ok) throw new Error("Failed to reset today's usage")
}

function fmtTok(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`
  return String(n)
}

// What a user's agents spent on LLMs (usage_spend.py), at list prices.
interface SpendTotals {
  usd: number
  tokens: number
  unpriced_tokens: number  // on models with no price, local ones usually
  own_usd: number  // on their own API keys: the deck doesn't pay these
  own_tokens: number
}
interface UserSpend { month: SpendTotals; all: SpendTotals; unpriced_models: string[] }
interface Spend { month: string; since: string; users: Record<string, UserSpend> }

const NO_SPEND: UserSpend = {
  month: { usd: 0, tokens: 0, unpriced_tokens: 0, own_usd: 0, own_tokens: 0 },
  all: { usd: 0, tokens: 0, unpriced_tokens: 0, own_usd: 0, own_tokens: 0 },
  unpriced_models: [],
}

async function fetchSpend(): Promise<Spend> {
  const res = await fetch('/fd/admin/usage/spend', { headers: adminHeaders(), credentials: 'include' })
  if (!res.ok) throw new Error('Failed to fetch spend')
  return res.json()
}

function fmtUsd(n: number): string {
  if (n > 0 && n < 0.01) return '<$0.01'
  return n.toLocaleString('en-US', { style: 'currency', currency: 'USD' })
}

// Dollars; tokens instead when nothing they used has a price.
function spentLabel(t: SpendTotals): string {
  return t.usd === 0 && t.unpriced_tokens > 0 ? `${fmtTok(t.unpriced_tokens)} tok` : fmtUsd(t.usd)
}

function fmtDay(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

const LOGIN_NOTE = 'Their last sign-in, or the last time a saved session let them back in. A tab left open counts.'

const perDay = (n: number | undefined, fmt: (n: number) => string = String) => (n ? fmt(n) : 'unlimited')

/** A user's limits as Flight Deck applies them: their plan's, with any
 *  per-user override from their metadata on top. */
function effectiveLimits(plan: PlanLimits | undefined, metadata: string): { limits?: PlanLimits; custom: Set<string> } {
  if (!plan) return { custom: new Set() }
  let meta: Record<string, unknown> = {}
  try { meta = JSON.parse(metadata || '{}') } catch { /* plan limits only */ }
  const limits = { ...plan }
  const custom = new Set<string>()
  for (const key of Object.keys(plan) as (keyof PlanLimits)[]) {
    if (typeof meta[key] === 'number') { limits[key] = meta[key] as number; custom.add(key) }
  }
  return { limits, custom }
}

// Today's messages or tokens reached the daily allowance (admins have none).
function limitReached(role: string, limits: PlanLimits | undefined, usage: DayUsage | undefined): boolean {
  return role !== 'admin' && !!limits && !!usage && (
    (limits.messages_per_day > 0 && usage.messages >= limits.messages_per_day)
    || (limits.tokens_per_day > 0 && usage.tokens >= limits.tokens_per_day))
}

function CreateUserForm({ onCreated, onClose, config }: {
  onCreated: () => void
  onClose: () => void
  config: SystemConfig
}) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [role, setRole] = useState('user')
  // Starts at the deck's "new users start in kiosk mode".
  const [kiosk, setKiosk] = useState(config.kiosk_new_users)
  // You know the password you type here: they pick their own at first sign-in.
  const [mustChange, setMustChange] = useState(true)
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState('')


  const valid = email.trim().includes('@') && password.length >= 6

  const handleCreate = async () => {
    if (!valid) return
    setSaving(true); setErr('')
    try {
      await createUser({
        email: email.trim(),
        password,
        display_name: displayName.trim() || undefined,
        role,
        must_change_password: mustChange,
        ...(role === 'admin' ? {} : { kiosk }),
      })
      onCreated()
      onClose()
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Failed to create user')
      setSaving(false)
    }
  }

  return (
    <div className="rounded-lg border border-violet-800/40 bg-zinc-900/50 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <UserPlus className="h-4 w-4 text-violet-400" />
        <h3 className="text-sm font-semibold text-zinc-200">New user</h3>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="col-span-2">
          <label className="block text-xs font-medium text-zinc-500 mb-1">Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="teammate@example.com"
            autoFocus
            autoComplete="off"
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-zinc-500 mb-1">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="At least 6 characters"
            autoComplete="new-password"
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
          />
          {password && password.length < 6 && (
            <p className="mt-1 text-[11px] text-amber-500/80">At least 6 characters.</p>
          )}
        </div>
        <div>
          <label className="block text-xs font-medium text-zinc-500 mb-1">Role</label>
          <select
            value={role}
            onChange={(e) => setRole(e.target.value)}
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
          >
            <option value="user">user</option>
            <option value="admin">admin</option>
          </select>
        </div>
        <div className="col-span-2">
          <label className="block text-xs font-medium text-zinc-500 mb-1">
            Display name <span className="text-zinc-600">(optional)</span>
          </label>
          <input
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder="Defaults to the email name"
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
          />
        </div>
        <div className="col-span-2">
          <MustChangeCheckbox
            checked={mustChange}
            onChange={setMustChange}
            label="They choose their own password at first sign-in"
            note="Turn off for a shared account that everyone signs in with (a kiosk)."
          />
        </div>
        {role !== 'admin' && (
          <div className="col-span-2">
            <KioskCheckbox checked={kiosk} onChange={setKiosk} note={kioskModeNote(config.kiosk_mode)} />
          </div>
        )}
      </div>
      {err && <p className="text-xs text-red-400">{err}</p>}
      <div className="flex items-center gap-2 pt-1">
        <button
          onClick={handleCreate}
          disabled={saving || !valid}
          className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
        >
          {saving ? 'Creating…' : 'Create user'}
        </button>
        <button
          onClick={onClose}
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}

function KioskCheckbox({ checked, onChange, note }: {
  checked: boolean
  onChange: (v: boolean) => void
  note: string | null
}) {
  return (
    <div>
      <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="h-3.5 w-3.5 rounded border-zinc-600 bg-zinc-950 accent-violet-600"
        />
        Kiosk mode <span className="text-xs text-zinc-500">— simple chat only</span>
      </label>
      {note && <p className="mt-1 ml-5 text-[11px] text-zinc-500">{note}</p>}
    </div>
  )
}

function MustChangeCheckbox({ checked, onChange, label, note }: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  note?: string
}) {
  return (
    <div>
      <label className="flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="h-3.5 w-3.5 rounded border-zinc-600 bg-zinc-950 accent-violet-600"
        />
        {label}
      </label>
      {note && <p className="mt-1 ml-5 text-[11px] text-zinc-500">{note}</p>}
    </div>
  )
}

function SpendLine({ label, totals }: { label: string; totals: SpendTotals }) {
  return (
    <div>
      <span className="text-zinc-500">{label}: </span>
      <span className="text-zinc-300">{fmtUsd(totals.usd)}</span>
      <span className="text-zinc-500"> · {fmtTok(totals.tokens)} tokens</span>
      {totals.unpriced_tokens > 0 && (
        <span className="text-zinc-500"> ({fmtTok(totals.unpriced_tokens)} with no price)</span>
      )}
      {totals.own_tokens > 0 && (
        <span className="text-zinc-500"> · on their own keys {fmtUsd(totals.own_usd)}, {fmtTok(totals.own_tokens)} tokens</span>
      )}
    </div>
  )
}

function UserRow({
  user, plans, usage, spend, spendSince, resets, resetCap, selected, onSelect, onUpdate, onDelete, currentUserId, kioskMode,
}: {
  user: UserInfo
  plans: Record<string, PlanLimits>
  usage?: DayUsage
  spend: UserSpend | null  // null: this deck can't tell yet
  spendSince: string
  resets: number
  resetCap: number
  selected: boolean
  onSelect: (on: boolean) => void
  onUpdate: () => void
  onDelete: (id: string) => void
  currentUserId: string
  kioskMode: KioskMode
}) {
  const [expanded, setExpanded] = useState(false)
  const [plan, setPlan] = useState(() => userPlan(user))
  const [role, setRole] = useState(user.role)
  const [kiosk, setKiosk] = useState(() => userKioskFlag(user))
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [agentsOpen, setAgentsOpen] = useState(false)
  const [resetting, setResetting] = useState(false)
  const [changingResets, setChangingResets] = useState(false)
  const [asking, setAsking] = useState(false)
  const [askedFrom, setAskedFrom] = useState('')
  const isSelf = user.id === currentUserId
  const { limits, custom } = effectiveLimits(plans[plan], user.metadata)
  const mark = (key: string) => (custom.has(key) ? ' (custom)' : '')
  const spent = limitReached(user.role, limits, usage)

  const handleResetUsage = async () => {
    setResetting(true); setSaveError('')
    try {
      await resetUsageToday(user.id)
      onUpdate()
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Failed to reset today's usage")
    }
    setResetting(false)
  }

  // One more free reset (never past the cap), or one taken back.
  const handleResets = async (delta: 1 | -1) => {
    setChangingResets(true); setSaveError('')
    try {
      if (delta > 0) await grantResets({ count: 1, user_ids: [user.id] })
      else await setResets(user.id, Math.max(0, resets - 1))
      onUpdate()
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : 'Failed to change resets')
    }
    setChangingResets(false)
  }

  const handleSave = async () => {
    setSaving(true); setSaveError('')
    try {
      await updateUser(user.id, { plan, role, kiosk })
      onUpdate()
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : 'Failed to update user')
    }
    setSaving(false)
  }

  return (
    <div className={`border rounded-lg overflow-hidden ${selected ? 'border-violet-500/50' : 'border-zinc-800'}`}>
      <div className="flex items-stretch hover:bg-zinc-800/50 transition-colors">
        {/* Admins have no daily limits, so nothing to give them. */}
        <label
          className={`flex items-center pl-4 ${user.role === 'admin' ? 'invisible' : 'cursor-pointer'}`}
          title="Select to give free resets"
        >
          <input
            type="checkbox"
            checked={selected}
            disabled={user.role === 'admin'}
            onChange={(e) => onSelect(e.target.checked)}
            aria-label={`Select ${user.display_name || user.email}`}
            className="h-3.5 w-3.5 rounded border-zinc-600 bg-zinc-950 accent-violet-600"
          />
        </label>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex-1 min-w-0 flex items-center gap-3 pl-3 pr-4 py-3 text-left"
        >
          {expanded ? <ChevronDown className="h-4 w-4 text-zinc-500 shrink-0" /> : <ChevronRight className="h-4 w-4 text-zinc-500 shrink-0" />}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-zinc-200 truncate">{user.display_name || user.email}</span>
              {user.role === 'admin' && <Crown className="h-3.5 w-3.5 text-amber-400" />}
              {isSelf && <span className="text-xs text-zinc-500">(you)</span>}
            </div>
            <div className="text-xs text-zinc-500">{user.email}</div>
            <div className="mt-0.5 flex flex-wrap gap-x-2 text-[11px] text-zinc-500">
              <span title={LOGIN_NOTE}>
                {user.last_login_at ? `Last login ${timeAgo(user.last_login_at)}` : 'No login on record'}
              </span>
              {spend && (
                <span title="What their agents spent on AI models, at list prices">
                  {spentLabel(spend.month)} this month · {spentLabel(spend.all)} all time
                </span>
              )}
            </div>
          </div>
          {userMustChangePassword(user) && (
            <span
              title="An admin set this password; they choose their own at their next sign-in"
              className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-700 dark:text-amber-300"
            >
              <KeyRound className="h-3 w-3" />
              temporary password
            </span>
          )}
          {kioskEffective(user.role, userKioskFlag(user), kioskMode) && (
            <span
              title="Locked to the simple chat layout (kiosk mode)"
              className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-300"
            >
              <MessageSquare className="h-3 w-3" />
              kiosk
            </span>
          )}
          {usage && (usage.messages > 0 || usage.tokens > 0) && (
            <span
              title={spent ? "Today's allowance is spent" : 'Used today'}
              className={`hidden sm:inline text-xs px-2 py-0.5 rounded-full ${spent ? 'bg-red-500/15 text-red-700 dark:text-red-300' : 'bg-zinc-800 text-zinc-500'}`}
            >
              {usage.messages} msg · {fmtTok(usage.tokens)} tok today
            </span>
          )}
          {resets > 0 && (
            <span
              title={`${nResets(resets)}: they can start a day over`}
              className="hidden sm:flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-700 dark:text-emerald-300"
            >
              <RotateCcw className="h-3 w-3" />
              {resets}
            </span>
          )}
          <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400 capitalize">{plan}</span>
        </button>
      </div>

      {expanded && (
        <div className="border-t border-zinc-800 px-4 py-3 bg-zinc-900/50 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-zinc-500 mb-1">Plan</label>
              <select
                value={plan}
                onChange={(e) => setPlan(e.target.value)}
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              >
                {Object.keys(plans).map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-zinc-500 mb-1">Role</label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value)}
                disabled={isSelf}
                className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none disabled:opacity-50"
              >
                <option value="user">user</option>
                <option value="admin">admin</option>
              </select>
            </div>
          </div>

          {role === 'admin' ? (
            <p className="text-[11px] text-zinc-500">Admins are never held to kiosk mode.</p>
          ) : (
            <KioskCheckbox checked={kiosk} onChange={setKiosk} note={kioskModeNote(kioskMode)} />
          )}

          {limits && (
            <div className="text-xs text-zinc-500 grid grid-cols-2 gap-1">
              <span>Max agents: {limits.max_agents}{mark('max_agents')}</span>
              <span>Storage: {limits.max_storage_mb} MB{mark('max_storage_mb')}</span>
              <span>Requests/min: {limits.requests_per_minute}{mark('requests_per_minute')}</span>
              <span>Spawns/hour: {limits.spawns_per_hour}{mark('spawns_per_hour')}</span>
              <span>Messages/day: {perDay(limits.messages_per_day)}{mark('messages_per_day')}</span>
              <span>Tokens/day: {perDay(limits.tokens_per_day, fmtTok)}{mark('tokens_per_day')}</span>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className={spent ? 'text-red-700 dark:text-red-300' : 'text-zinc-400'}>
              Today: {usage?.messages ?? 0} message{(usage?.messages ?? 0) === 1 ? '' : 's'} · {fmtTok(usage?.tokens ?? 0)} tokens
              {spent && ' — limit reached'}
              {user.role === 'admin' && ' (admins have no daily limits)'}
            </span>
            {usage && (usage.messages > 0 || usage.tokens > 0) && (
              <button
                onClick={handleResetUsage}
                disabled={resetting}
                title="Give this user today's messages and tokens back"
                className="rounded-md border border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-300 hover:border-violet-500/40 hover:text-zinc-100 disabled:opacity-50"
              >
                {resetting ? 'Resetting…' : 'Reset today'}
              </button>
            )}
          </div>

          {user.role !== 'admin' && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="text-zinc-400">
                <RotateCcw className="h-3 w-3 inline mr-1" />
                {resets > 0 ? nResets(resets) : 'No free resets'}
                <span className="text-zinc-600"> · holds at most {resetCap}</span>
              </span>
              <button
                onClick={() => handleResets(1)}
                disabled={changingResets || resets >= resetCap}
                title={resets >= resetCap ? `They hold the most a user can (${resetCap})` : 'Give them one more free reset'}
                className="rounded-md border border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-300 hover:border-violet-500/40 hover:text-zinc-100 disabled:opacity-50"
              >
                Give 1
              </button>
              {resets > 0 && (
                <button
                  onClick={() => handleResets(-1)}
                  disabled={changingResets}
                  title="Take one free reset back"
                  className="rounded-md border border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-300 hover:border-violet-500/40 hover:text-zinc-100 disabled:opacity-50"
                >
                  Take 1
                </button>
              )}
            </div>
          )}

          {spend && (
            <div className="space-y-0.5 text-xs">
              <SpendLine label="Spent this month" totals={spend.month} />
              <SpendLine label={spendSince ? `Spent since ${fmtDay(spendSince)}` : 'Spent all time'} totals={spend.all} />
              {spend.unpriced_models.length > 0 && (
                <div className="text-[11px] text-zinc-600">
                  No price for: {spend.unpriced_models.join(', ')}
                </div>
              )}
            </div>
          )}

          <div className="flex flex-wrap gap-x-4 text-xs text-zinc-600">
            <span>Joined: {new Date(user.created_at).toLocaleDateString()}</span>
            <span title={LOGIN_NOTE}>
              Last login: {user.last_login_at ? new Date(user.last_login_at).toLocaleString() : 'none on record'}
            </span>
            {user.feedback_asked_at && <span>Asked for feedback: {fmtDay(user.feedback_asked_at)}</span>}
          </div>
          {saveError && <p className="text-xs text-red-400">{saveError}</p>}

          <div className="flex items-center gap-2 pt-1">
            <button
              onClick={handleSave}
              disabled={saving}
              className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
            >
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
            <button
              onClick={() => setAgentsOpen(true)}
              title="Create, start, stop, configure and remove this user's agents"
              className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-violet-500/40 hover:text-zinc-100"
            >
              <Bot className="h-3 w-3 inline mr-1" />
              Agents
            </button>
            {!isSelf && (
              <button
                onClick={() => { setAsking(true); setAskedFrom('') }}
                disabled={asking}
                title="Email them from your own account and ask how the deck is working for them"
                className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-violet-500/40 hover:text-zinc-100 disabled:opacity-50"
              >
                <Mail className="h-3 w-3 inline mr-1" />
                Ask for feedback
              </button>
            )}
            {!isSelf && (
              <button
                onClick={() => { if (confirm(`Delete user ${user.email}? This cannot be undone.`)) onDelete(user.id) }}
                className="rounded-md border border-red-900/50 px-3 py-1.5 text-xs text-red-400 hover:bg-red-950/50"
              >
                <Trash2 className="h-3 w-3 inline mr-1" />
                Delete
              </button>
            )}
          </div>

          {askedFrom && <p className="text-xs text-emerald-400">Email sent from {askedFrom}. Their reply comes to that mailbox.</p>}
          {asking && (
            <FeedbackRequestPanel
              userId={user.id}
              onSent={(from) => { setAsking(false); setAskedFrom(from); onUpdate() }}
              onClose={() => setAsking(false)}
            />
          )}

          <SetPasswordSection userId={user.id} email={user.email} isSelf={isSelf} onSet={onUpdate} />
        </div>
      )}
      {agentsOpen && (
        <UserAgentsDialog
          userId={user.id}
          userLabel={user.display_name || user.email}
          onClose={() => setAgentsOpen(false)}
        />
      )}
    </div>
  )
}

// Admin sets someone's password (they forgot it, or it leaked) — no current
// password needed. Its own action, apart from Save Changes. Their own: Profile.
function SetPasswordSection({ userId, email, isSelf, onSet }: {
  userId: string
  email: string
  isSelf: boolean
  onSet: () => void
}) {
  const setView = useUIStore((s) => s.setView)
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [signOut, setSignOut] = useState(true)
  const [mustChange, setMustChange] = useState(true)
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState('')
  const [err, setErr] = useState('')

  if (isSelf) {
    return (
      <div className="border-t border-zinc-800 pt-3 text-xs text-zinc-500">
        <KeyRound className="h-3 w-3 inline mr-1" />
        {isAppShell()
          ? 'Change your own password in your profile: the user kiosk → Settings → Profile.'
          : <>Change your own password on your{' '}
              <button onClick={() => setView('profile')} className="text-violet-400 underline underline-offset-2 hover:no-underline">
                Profile
              </button>.</>}
      </div>
    )
  }

  const tooShort = password.length > 0 && password.length < MIN_PASSWORD_LENGTH
  const mismatch = confirm.length > 0 && confirm !== password
  const valid = password.length >= MIN_PASSWORD_LENGTH && confirm === password

  const handleSet = async () => {
    if (!valid) return
    setSaving(true); setErr(''); setNotice('')
    try {
      await updateUser(userId, { password, sign_out: signOut, must_change_password: mustChange })
      setPassword(''); setConfirm('')
      setNotice([
        'Password set.',
        signOut && 'They were signed out everywhere.',
        mustChange && 'They choose their own at their next sign-in.',
      ].filter(Boolean).join(' '))
      onSet()
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Failed to set password')
    }
    setSaving(false)
  }

  const input = 'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none'

  return (
    <div className="border-t border-zinc-800 pt-3 space-y-2">
      <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-400">
        <KeyRound className="h-3.5 w-3.5" />
        Set password
      </div>
      {/* For password managers: the account this password is for. */}
      <input type="text" autoComplete="username" value={email} readOnly hidden />
      <div className="grid grid-cols-2 gap-3">
        <div>
          <input
            type="password"
            value={password}
            onChange={(e) => { setPassword(e.target.value); setNotice('') }}
            placeholder="New password"
            autoComplete="new-password"
            aria-label="New password"
            className={input}
          />
          {tooShort && <p className="mt-1 text-[11px] text-amber-500/80">At least {MIN_PASSWORD_LENGTH} characters.</p>}
        </div>
        <div>
          <input
            type="password"
            value={confirm}
            onChange={(e) => { setConfirm(e.target.value); setNotice('') }}
            placeholder="Confirm new password"
            autoComplete="new-password"
            aria-label="Confirm new password"
            className={input}
          />
          {mismatch && <p className="mt-1 text-[11px] text-amber-500/80">Doesn't match.</p>}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={handleSet}
          disabled={saving || !valid}
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-200 hover:border-violet-500/50 hover:bg-zinc-800 disabled:opacity-50"
        >
          {saving ? 'Setting…' : 'Set password'}
        </button>
        <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
          <input
            type="checkbox"
            checked={signOut}
            onChange={(e) => setSignOut(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-zinc-600 bg-zinc-950 accent-violet-600"
          />
          Sign them out everywhere
        </label>
        <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
          <input
            type="checkbox"
            checked={mustChange}
            onChange={(e) => setMustChange(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-zinc-600 bg-zinc-950 accent-violet-600"
          />
          They choose their own at next sign-in
        </label>
      </div>
      {notice && <p className="text-xs text-emerald-400">{notice}</p>}
      {err && <p className="text-xs text-red-400">{err}</p>}
    </div>
  )
}

// Give free resets to the selected users or to every user on the deck. Nobody
// ends up holding more than the cap (set here, kept as the deck's), so repeated
// grants don't pile up; someone already over a lowered cap keeps theirs.
function GrantResetsPanel({ selected, everyone, cap, onDone, onClose }: {
  selected: string[]
  everyone: string
  cap: number
  onDone: (notice: string) => void
  onClose: () => void
}) {
  const [count, setCount] = useState(1)
  const [maxHeld, setMaxHeld] = useState(cap)
  const [target, setTarget] = useState<'selected' | 'all'>(selected.length ? 'selected' : 'all')
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState('')
  const valid = count >= 1 && count <= MAX_RESETS && maxHeld >= 1 && maxHeld <= MAX_RESETS
    && (target === 'all' || selected.length > 0)

  const handleGrant = async () => {
    if (!valid) return
    if (target === 'all' && !confirm(`Give ${nResets(count)} to ${everyone}?`)) return
    setSaving(true); setErr('')
    try {
      const r = await grantResets({
        count, cap: maxHeld,
        ...(target === 'all' ? { all: true } : { user_ids: selected }),
      })
      const full = r.users - r.given
      onDone([
        r.given
          ? `${r.given} user${r.given === 1 ? '' : 's'} got more free resets (each holds at most ${r.cap}).`
          : 'Nobody got a reset.',
        full > 0 && `${full} already held the most.`,
      ].filter(Boolean).join(' '))
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Failed to give resets')
      setSaving(false)
    }
  }

  const num = 'w-16 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none'
  const clamp = (v: string) => Math.max(0, Math.min(MAX_RESETS, Math.floor(Number(v) || 0)))

  return (
    <div className="rounded-lg border border-violet-800/40 bg-zinc-900/50 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <RotateCcw className="h-4 w-4 text-violet-400" />
        <h3 className="text-sm font-semibold text-zinc-200">Give free resets</h3>
      </div>
      <p className="text-xs text-zinc-500">
        A user spends a free reset to start the day's messages and AI tokens over, from the chat's usage bar or their Info dialog.
      </p>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-zinc-300">
        <label className="flex items-center gap-2">
          Give
          <input type="number" min={1} max={MAX_RESETS} value={count} aria-label="Resets to give"
            onChange={(e) => setCount(clamp(e.target.value))} className={num} />
          reset{count === 1 ? '' : 's'} to
        </label>
        <label className={`flex items-center gap-2 ${selected.length ? 'cursor-pointer' : 'opacity-50'}`}>
          <input type="radio" name="grant-target" checked={target === 'selected'} disabled={!selected.length}
            onChange={() => setTarget('selected')} className="accent-violet-600" />
          {selected.length ? `the ${selected.length} selected user${selected.length === 1 ? '' : 's'}` : 'selected users (none yet)'}
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="grant-target" checked={target === 'all'}
            onChange={() => setTarget('all')} className="accent-violet-600" />
          {everyone}
        </label>
      </div>
      <div>
        <label className="flex flex-wrap items-center gap-2 text-sm text-zinc-300">
          Most a user can hold
          <input type="number" min={1} max={MAX_RESETS} value={maxHeld} aria-label="Most resets a user can hold"
            onChange={(e) => setMaxHeld(clamp(e.target.value))} className={num} />
        </label>
        <p className="mt-1 text-[11px] text-zinc-500">
          Applies to every grant on this deck: resets stop adding up there. Admins have no daily limits and get none.
        </p>
      </div>
      {err && <p className="text-xs text-red-400">{err}</p>}
      <div className="flex items-center gap-2 pt-1">
        <button
          onClick={handleGrant}
          disabled={saving || !valid}
          className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
        >
          {saving ? 'Giving…' : 'Give resets'}
        </button>
        <button
          onClick={onClose}
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}

type RoleFilter = 'all' | 'admin' | 'user'
type StatusFilter = 'any' | 'kiosk' | 'not-kiosk' | 'temp-password' | 'used-today' | 'limit-reached' | 'has-resets'
type SortKey = 'newest' | 'oldest' | 'name' | 'usage' | 'login-recent' | 'login-longest' | 'spend-month' | 'spend-all'

const STATUS_LABEL: Record<StatusFilter, string> = {
  any: 'Any status',
  kiosk: 'Kiosk mode',
  'not-kiosk': 'Not in kiosk mode',
  'temp-password': 'Temporary password',
  'used-today': 'Active today',
  'limit-reached': 'Daily limit reached',
  'has-resets': 'Has free resets',
}

const SORT_LABEL: Record<SortKey, string> = {
  newest: 'Newest first',
  oldest: 'Oldest first',
  name: 'Name A–Z',
  usage: 'Most active today',
  'login-recent': 'Last login: most recent',
  'login-longest': 'Last login: longest ago',
  'spend-month': 'Most spent this month',
  'spend-all': 'Most spent all time',
}

export function UsersAdmin({ refreshTick, config, onTotal }: {
  refreshTick: number
  config: SystemConfig
  onTotal: (n: number) => void
}) {
  const currentUserId = useAuthStore((s) => s.user?.id ?? '')
  const [users, setUsers] = useState<UserInfo[] | null>(null)
  const [total, setTotal] = useState(0)
  const [plans, setPlans] = useState<Record<string, PlanLimits>>({})
  const [usageToday, setUsageToday] = useState<Record<string, DayUsage>>({})
  const [resets, setResetsMap] = useState<Record<string, number>>({})
  const [resetCap, setResetCap] = useState(NO_USAGE.reset_cap)
  const [spend, setSpend] = useState<Spend | null>(null)
  const [selected, setSelected] = useState<Set<string>>(() => new Set())
  const [granting, setGranting] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const [adding, setAdding] = useState(false)
  const [query, setQuery] = useState('')
  const [role, setRole] = useState<RoleFilter>('all')
  const [plan, setPlan] = useState('all')
  const [status, setStatus] = useState<StatusFilter>('any')
  const [sort, setSort] = useState<SortKey>('newest')

  const load = useCallback(async () => {
    try {
      const [data, plansData, today, spendData] = await Promise.all([
        fetchUsers(USERS_MAX), fetchPlans(), fetchUsageToday().catch(() => NO_USAGE), fetchSpend().catch(() => null),
      ])
      setUsers(data.users)
      setTotal(data.total)
      onTotal(data.total)
      setPlans(plansData)
      setUsageToday(today.usage)
      setResetsMap(today.resets)
      setResetCap(today.reset_cap)
      setSpend(spendData)
      // Selections of accounts that are gone (or became admins) drop out.
      const selectable = new Set(data.users.filter((u) => u.role !== 'admin').map((u) => u.id))
      setSelected((prev) => new Set([...prev].filter((id) => selectable.has(id))))
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load users')
    }
  }, [onTotal])

  useEffect(() => { load() }, [load, refreshTick])

  const handleDelete = async (userId: string) => {
    try {
      await deleteUser(userId)
      load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete user')
    }
  }

  // The deck's plans, and any plan a user is on that isn't one of them.
  const planNames = useMemo(
    () => [...new Set([...Object.keys(plans), ...(users ?? []).map(userPlan)])],
    [plans, users],
  )

  const shown = useMemo(() => {
    const list = (users ?? []).filter((u) => {
      if (!matchesQuery(query, u.display_name, u.email)) return false
      if (role !== 'all' && u.role !== role) return false
      if (plan !== 'all' && userPlan(u) !== plan) return false
      const usage = usageToday[u.id]
      switch (status) {
        case 'kiosk': return kioskEffective(u.role, userKioskFlag(u), config.kiosk_mode)
        case 'not-kiosk': return !kioskEffective(u.role, userKioskFlag(u), config.kiosk_mode)
        case 'temp-password': return userMustChangePassword(u)
        case 'used-today': return !!usage && (usage.messages > 0 || usage.tokens > 0)
        case 'limit-reached':
          return limitReached(u.role, effectiveLimits(plans[userPlan(u)], u.metadata).limits, usage)
        case 'has-resets': return (resets[u.id] ?? 0) > 0
        default: return true
      }
    })
    const name = (u: UserInfo) => u.display_name || u.email
    const used = (u: UserInfo) => usageToday[u.id] ?? { messages: 0, tokens: 0 }
    const spent = (u: UserInfo) => spend?.users[u.id] ?? NO_SPEND
    const login = (u: UserInfo) => u.last_login_at ?? ''
    return list.sort((a, b) => {
      // No login on record sorts as the longest ago.
      if (sort === 'login-recent') return login(b).localeCompare(login(a))
      if (sort === 'login-longest') return login(a).localeCompare(login(b))
      if (sort === 'spend-month') return spent(b).month.usd - spent(a).month.usd
      if (sort === 'spend-all') return spent(b).all.usd - spent(a).all.usd
      if (sort === 'oldest') return a.created_at.localeCompare(b.created_at)
      if (sort === 'name') return name(a).localeCompare(name(b), undefined, { sensitivity: 'base' })
      if (sort === 'usage') return used(b).tokens - used(a).tokens || used(b).messages - used(a).messages
      return b.created_at.localeCompare(a.created_at)
    })
  }, [users, query, role, plan, status, sort, usageToday, resets, plans, spend, config.kiosk_mode])

  const filtered = query.trim() !== '' || role !== 'all' || plan !== 'all' || status !== 'any'
  const clearFilters = () => { setQuery(''); setRole('all'); setPlan('all'); setStatus('any') }

  // Selecting is for free resets, so only users with daily limits (not admins).
  const shownSelectable = shown.filter((u) => u.role !== 'admin')
  const allShownSelected = shownSelectable.length > 0 && shownSelectable.every((u) => selected.has(u.id))
  const selectShown = (on: boolean) => setSelected((prev) => {
    const next = new Set(prev)
    for (const u of shownSelectable) { if (on) next.add(u.id); else next.delete(u.id) }
    return next
  })
  const selectOne = (id: string, on: boolean) => setSelected((prev) => {
    const next = new Set(prev)
    if (on) next.add(id); else next.delete(id)
    return next
  })
  const nonAdmins = (users ?? []).filter((u) => u.role !== 'admin').length
  const everyone = users && total <= users.length
    ? `every user (all ${nonAdmins})`
    : 'every user on the deck'

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <SearchBox value={query} onChange={setQuery} placeholder="Search by name or email" />
        <button
          onClick={() => setAdding(true)}
          disabled={adding}
          className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50"
        >
          <UserPlus className="h-3.5 w-3.5" />
          Add user
        </button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <FilterSelect label="Role" value={role} onChange={(v) => setRole(v as RoleFilter)} active={role !== 'all'}>
          <option value="all">All roles</option>
          <option value="admin">Admins</option>
          <option value="user">Users</option>
        </FilterSelect>
        <FilterSelect label="Plan" value={plan} onChange={setPlan} active={plan !== 'all'}>
          <option value="all">All plans</option>
          {planNames.map((p) => <option key={p} value={p}>{p}</option>)}
        </FilterSelect>
        <FilterSelect label="Status" value={status} onChange={(v) => setStatus(v as StatusFilter)} active={status !== 'any'}>
          {(Object.keys(STATUS_LABEL) as StatusFilter[]).map((s) => <option key={s} value={s}>{STATUS_LABEL[s]}</option>)}
        </FilterSelect>
        <span className="flex-1" />
        <FilterSelect label="Sort" value={sort} onChange={(v) => setSort(v as SortKey)} active={false}>
          {(Object.keys(SORT_LABEL) as SortKey[]).map((s) => <option key={s} value={s}>{SORT_LABEL[s]}</option>)}
        </FilterSelect>
      </div>

      {users && (shownSelectable.length > 0 || selected.size > 0) && (
        <div className="flex flex-wrap items-center gap-3 px-1 text-xs text-zinc-400">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={allShownSelected}
              disabled={shownSelectable.length === 0}
              onChange={(e) => selectShown(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-zinc-600 bg-zinc-950 accent-violet-600"
            />
            Select all shown
          </label>
          {selected.size > 0 && (
            <>
              <span className="text-zinc-300">{selected.size} selected</span>
              <button onClick={() => setSelected(new Set())} className="text-violet-400 underline underline-offset-2 hover:no-underline">
                Clear
              </button>
            </>
          )}
          <span className="flex-1" />
          <button
            onClick={() => { setGranting(true); setNotice('') }}
            disabled={granting}
            className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-200 hover:border-violet-500/50 hover:bg-zinc-800 disabled:opacity-50"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Give free resets…
          </button>
        </div>
      )}

      {granting && (
        <GrantResetsPanel
          selected={[...selected]}
          everyone={everyone}
          cap={resetCap}
          onDone={(msg) => { setGranting(false); setNotice(msg); setSelected(new Set()); load() }}
          onClose={() => setGranting(false)}
        />
      )}
      {notice && (
        <div className="flex items-center gap-2 rounded-md border border-emerald-700/40 bg-emerald-950/20 px-4 py-2 text-sm text-emerald-400">
          <span className="flex-1">{notice}</span>
          <button onClick={() => setNotice('')} className="text-xs text-zinc-500 hover:text-zinc-300">Dismiss</button>
        </div>
      )}

      {adding && <CreateUserForm config={config} onCreated={load} onClose={() => setAdding(false)} />}

      {error && (
        <div className="rounded-md border border-red-900/50 bg-red-950/20 px-4 py-2 text-sm text-red-400">{error}</div>
      )}

      {users === null && !error && (
        <div className="flex justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
        </div>
      )}

      {users && filtered && (
        <FilterSummary shown={shown.length} total={users.length} noun="users" onClear={clearFilters} />
      )}

      {shown.map((u) => (
        <UserRow
          key={u.id}
          user={u}
          plans={plans}
          usage={usageToday[u.id]}
          spend={spend ? spend.users[u.id] ?? NO_SPEND : null}
          spendSince={spend?.since ?? ''}
          resets={resets[u.id] ?? 0}
          resetCap={resetCap}
          selected={selected.has(u.id)}
          onSelect={(on) => selectOne(u.id, on)}
          onUpdate={load}
          onDelete={handleDelete}
          currentUserId={currentUserId}
          kioskMode={config.kiosk_mode}
        />
      ))}

      {users && shown.length === 0 && (
        <p className="text-center text-sm text-zinc-500 py-8">
          {filtered ? 'No users match these filters.' : 'No users found'}
        </p>
      )}

      {users && total > users.length && (
        <p className="text-center text-xs text-zinc-500">
          Showing the newest {users.length} of {total} accounts; search and filters cover these.
        </p>
      )}
    </div>
  )
}
