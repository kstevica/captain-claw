import { useEffect, useMemo, useState } from 'react'
import { ScrollText, ShieldCheck, ShieldAlert, History, Clock, Save, Loader2 } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { toolAuditApi, type AuditEvent, type AuditSettings, type VerifyResult } from '../../services/toolAuditApi'
import { ToolCallRegister } from '../audit/ToolCallRegister'
import { fetchAllUsers, type AdminUser } from './consoleApi'
import { Page, PageHeader, Card, Button, Notice, ChipPicker, Pill } from './parts'
import { timeAgo } from './format'

// Audit: the tool-call register for the whole deck — every tool call every
// user's agents made — with the register's own health: a check that no row was
// changed, added or removed, records agents made that never arrived, how long
// rows are kept, and what happened to the register itself.

const RETENTION_CHOICES = [30, 90, 180, 365, 730]

function unixAgo(sec: number): string {
  return timeAgo(new Date(sec * 1000).toISOString())
}

function eventText(e: AuditEvent): string {
  const d = e.detail || {}
  switch (e.kind) {
    case 'pruned': return `Removed ${Number(d.rows || 0).toLocaleString()} rows older than ${d.retention_days} days`
    case 'verified': return d.ok ? `Checked ${Number(d.checked || 0).toLocaleString()} rows — all intact`
      : `Checked ${Number(d.checked || 0).toLocaleString()} rows — ${d.problems} problem(s), ${d.gaps} gap(s)`
    case 'exported': return `Exported (${String(d.format || 'csv').toUpperCase()}${d.user ? ', one person' : ''})`
    case 'retention_changed': return `Keep rows for ${d.to} days (was ${d.from})`
    default: return e.kind
  }
}

export function AuditConsole() {
  const notify = useNotificationStore((s) => s.add)
  const accounts = useAuthStore((s) => s.authEnabled !== false)
  const [users, setUsers] = useState<AdminUser[]>([])
  const [verify, setVerify] = useState<VerifyResult | null>(null)
  const [checking, setChecking] = useState(false)
  const [settings, setSettings] = useState<AuditSettings | null>(null)
  const [days, setDays] = useState(90)
  const [saving, setSaving] = useState(false)
  const [events, setEvents] = useState<AuditEvent[]>([])

  const loadEvents = () => toolAuditApi.events().then((r) => setEvents(r.events)).catch(() => {})

  useEffect(() => {
    if (accounts) fetchAllUsers().then((r) => setUsers(r.users)).catch(() => {})
    toolAuditApi.settings().then((s) => { setSettings(s); setDays(s.retention_days) }).catch(() => {})
    loadEvents()
  }, [accounts])

  const people = useMemo(() => (accounts
    ? users.map((u) => ({ id: u.id, name: u.display_name || u.email || u.id.slice(0, 8) }))
      .sort((a, b) => a.name.localeCompare(b.name))
    : undefined), [users, accounts])

  const check = async () => {
    setChecking(true)
    try {
      setVerify(await toolAuditApi.verify())
      loadEvents()
    } catch (e) {
      notify('error', 'Could not check the register', e instanceof Error ? e.message : String(e))
    } finally {
      setChecking(false)
    }
  }

  const save = async () => {
    setSaving(true)
    try {
      const s = await toolAuditApi.saveSettings(days)
      setSettings(s)
      setDays(s.retention_days)
      loadEvents()
      notify('success', 'Saved', `Tool calls are kept for ${s.retention_days} days.`)
    } catch (e) {
      notify('error', 'Could not save', e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const lastCheck = events.find((e) => e.kind === 'verified')

  return (
    <Page>
      <PageHeader
        icon={ScrollText}
        title="Audit"
        description="Every tool call every agent on this deck made: which agent, when, which tool, with what, what started it and how it ended. Each person sees their own agents' calls under Activity; here you see everyone's."
      />

      <div className="grid gap-4 xl:grid-cols-2">
        <Card
          icon={verify && !verify.ok ? ShieldAlert : ShieldCheck}
          title="Integrity"
          hint="Each row is sealed to the one before it, and agents number their records. A check finds any row that was changed, added or removed, and records that never arrived."
          action={<Button icon={ShieldCheck} busy={checking} onClick={check}>Check now</Button>}
        >
          {!verify ? (
            <p className="text-sm text-zinc-500">
              {lastCheck
                ? <>Last checked {unixAgo(lastCheck.at)}{lastCheck.actor_name ? ` by ${lastCheck.actor_name}` : ''}: {eventText(lastCheck)}.</>
                : 'Not checked yet.'}
            </p>
          ) : verify.ok ? (
            <Notice tone="emerald">
              All {verify.checked.toLocaleString()} rows check out — nothing was changed, added or removed
              {verify.anchor_id ? ' since the oldest rows were pruned' : ''}, and no agent's records are missing.
            </Notice>
          ) : (
            <div className="space-y-3">
              {verify.problems.length > 0 && (
                <Notice tone="red">
                  <div className="font-medium">{verify.problems.length} row{verify.problems.length === 1 ? '' : 's'} failed the check</div>
                  <ul className="mt-1 space-y-0.5 text-xs">
                    {verify.problems.slice(0, 8).map((p) => <li key={`${p.id}-${p.problem}`}>Row {p.id}: {p.detail}</li>)}
                    {verify.problems.length > 8 && <li>…and {verify.problems.length - 8} more</li>}
                  </ul>
                </Notice>
              )}
              {verify.gaps.length > 0 && (
                <Notice tone="amber">
                  <div className="font-medium">Records that never arrived</div>
                  <ul className="mt-1 space-y-0.5 text-xs">
                    {verify.gaps.slice(0, 8).map((g) => (
                      <li key={`${g.agent_slug}-${g.boot}`}>
                        {g.agent_name}{g.owner_name ? ` (${g.owner_name})` : ''}: {g.missing} missing between #{g.first_seq} and #{g.last_seq}, {unixAgo(g.last_at)}
                      </li>
                    ))}
                  </ul>
                  <p className="mt-2 text-xs opacity-80">
                    An agent reports every call it makes. A gap means records were lost on the way, or the agent stopped reporting.
                  </p>
                </Notice>
              )}
              <p className="text-xs text-zinc-500">Checked {verify.checked.toLocaleString()} rows.</p>
            </div>
          )}
        </Card>

        <Card icon={Clock} title="Keep tool calls for" hint="Older rows are removed automatically. Each removal is logged below, and the check starts from the last row removed.">
          <div className="space-y-3">
            <ChipPicker value={days} options={RETENTION_CHOICES} onChange={setDays} format={(d) => (d >= 365 ? `${d / 365} year${d === 365 ? '' : 's'}` : `${d} days`)} />
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={settings?.min_days ?? 7}
                max={settings?.max_days ?? 3650}
                value={days}
                onChange={(e) => setDays(Number(e.target.value) || 0)}
                aria-label="Days"
                className="h-9 w-24 rounded-xl border border-zinc-700 bg-zinc-950 px-3 text-sm text-zinc-100 focus:border-violet-500/60 focus:outline-none"
              />
              <span className="text-sm text-zinc-500">days</span>
              {settings && (days < (settings.min_days ?? 7) || days > (settings.max_days ?? 3650)) && (
                <span className="text-xs text-amber-600 dark:text-amber-400">
                  {settings.min_days ?? 7} to {settings.max_days ?? 3650} days
                </span>
              )}
              <Button
                variant="primary"
                icon={Save}
                busy={saving}
                disabled={!settings || days === settings.retention_days
                  || days < (settings.min_days ?? 7) || days > (settings.max_days ?? 3650)}
                onClick={save}
                className="ml-auto"
              >
                Save
              </Button>
            </div>
          </div>
        </Card>
      </div>

      <Card title="Tool calls" bodyClassName="p-0">
        <ToolCallRegister scope="admin" embedded users={people} />
      </Card>

      <Card icon={History} title="Register history" hint="What happened to the register itself: checks, exports, retention changes and removals.">
        {!events.length ? (
          <p className="text-sm text-zinc-500">Nothing yet.</p>
        ) : (
          <ul className="divide-y divide-zinc-800/70">
            {events.slice(0, 20).map((e) => (
              <li key={e.id} className="flex items-center gap-3 py-2 text-sm">
                <Pill tone={e.kind === 'verified' && !e.detail?.ok ? 'red' : e.kind === 'pruned' ? 'amber' : 'zinc'}>{e.kind.replace('_', ' ')}</Pill>
                <span className="min-w-0 flex-1 truncate text-zinc-300">{eventText(e)}</span>
                <span className="shrink-0 text-xs text-zinc-500">
                  {e.actor_name && e.actor !== 'retention' ? `${e.actor_name} · ` : ''}{unixAgo(e.at)}
                </span>
              </li>
            ))}
          </ul>
        )}
        {checking && <Loader2 className="mt-2 h-4 w-4 animate-spin text-zinc-500" />}
      </Card>
    </Page>
  )
}
