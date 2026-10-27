import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  BarChart3, MessageSquare, Cpu, Users, DollarSign, Save, RefreshCw, Activity, SlidersHorizontal,
} from 'lucide-react'
import { fetchPlans, type PlanLimits } from '../admin/deckConfig'
import {
  fetchUsageToday, fetchAllUsers, fetchCosts, fetchUsageLogs, updatePlan,
  type UsageToday, type AdminUser, type CostRollup, type UsageLog,
} from './consoleApi'
import { Page, PageHeader, Card, StatTile, Segmented, Button, Notice, Pill } from './parts'
import { fmtNumber, timeAgo } from './format'
import { ByokSetting } from '../admin/ByokSetting'

// Usage & limits: what people used today against their plan, what the deck
// spent, the plans' limits, and the latest activity. The Admin page's Usage
// and Plans tabs, on one page.

const LIMITS: { key: keyof PlanLimits; label: string; hint: string }[] = [
  { key: 'messages_per_day', label: 'Messages a day', hint: '0 = unlimited' },
  { key: 'tokens_per_day', label: 'AI tokens a day', hint: '0 = unlimited' },
  { key: 'max_agents', label: 'Agents', hint: 'most at once' },
  { key: 'max_storage_mb', label: 'Storage (MB)', hint: '' },
  { key: 'requests_per_minute', label: 'Requests a minute', hint: '' },
  { key: 'spawns_per_hour', label: 'New agents an hour', hint: '' },
]

type Range = '7d' | '30d' | 'all'

function sinceFor(r: Range): string | undefined {
  if (r === 'all') return undefined
  const d = new Date(Date.now() - (r === '7d' ? 7 : 30) * 86400_000)
  return d.toISOString()
}

function userPlan(u?: AdminUser): string {
  try {
    const p = JSON.parse(u?.metadata || '{}').plan || 'free'
    return p === 'pro' ? 'premium' : p
  } catch { return 'free' }
}

function hoursUntil(iso: string): string {
  const ms = new Date(iso).getTime() - Date.now()
  if (!(ms > 0)) return ''
  const h = Math.floor(ms / 3600_000)
  return h >= 1 ? `in ${h} h` : `in ${Math.max(1, Math.round(ms / 60_000))} min`
}

export function UsageConsole() {
  const [today, setToday] = useState<UsageToday | null>(null)
  const [users, setUsers] = useState<AdminUser[]>([])
  const [range, setRange] = useState<Range>('30d')
  const [costs, setCosts] = useState<CostRollup | null>(null)
  const [logs, setLogs] = useState<UsageLog[]>([])
  const [plans, setPlans] = useState<Record<string, PlanLimits>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [tick, setTick] = useState(0)

  useEffect(() => {
    Promise.all([
      fetchUsageToday().then(setToday).catch(() => setToday(null)),
      fetchAllUsers().then((r) => setUsers(r.users)).catch(() => {}),
      fetchUsageLogs(30).then(setLogs).catch(() => {}),
      fetchPlans().then(setPlans).catch((e) => setError(e instanceof Error ? e.message : String(e))),
    ]).finally(() => setLoading(false))
  }, [tick])

  useEffect(() => {
    fetchCosts(sinceFor(range)).then(setCosts).catch(() => setCosts(null))
  }, [range, tick])

  const byId = useMemo(() => new Map(users.map((u) => [u.id, u])), [users])
  const rows = useMemo(() => Object.entries(today?.usage || {})
    .map(([id, u]) => ({ id, ...u, user: byId.get(id) }))
    .sort((a, b) => b.tokens - a.tokens || b.messages - a.messages), [today, byId])
  const totals = rows.reduce((t, r) => ({ messages: t.messages + r.messages, tokens: t.tokens + r.tokens }), { messages: 0, tokens: 0 })
  const name = (id: string, u?: AdminUser) => u?.display_name || u?.email || id.slice(0, 8)

  return (
    <Page>
      <PageHeader
        icon={BarChart3}
        title="Usage & limits"
        description="What people used today against their plan, what the deck spent on models, and the daily limits of each plan. A day is a UTC day; admins are never held to the limits."
        actions={<Button icon={RefreshCw} busy={loading} onClick={() => { setLoading(true); setTick((t) => t + 1) }}>Refresh</Button>}
      />

      {error && <Notice tone="red">{error}</Notice>}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile icon={MessageSquare} tone="violet" label="Messages today" value={fmtNumber(totals.messages)} loading={loading && !today} />
        <StatTile icon={Cpu} tone="cyan" label="AI tokens today" value={fmtNumber(totals.tokens)} loading={loading && !today} />
        <StatTile icon={Users} tone="emerald" label="People active today" value={rows.length} sub={today?.resets_at ? `New day ${hoursUntil(today.resets_at)}` : undefined} loading={loading && !today} />
        <StatTile icon={DollarSign} tone="amber" label={range === 'all' ? 'Spend, all time' : `Spend, last ${range === '7d' ? '7' : '30'} days`} value={costs ? `$${costs.total_usd.toFixed(2)}` : '—'} />
      </div>

      <Card icon={Users} title="Today, by person" hint="Messages and AI tokens since midnight UTC. Give someone their day back, or free resets, on the Users page." bodyClassName="">
        {rows.length === 0 ? (
          <p className="px-5 py-6 text-center text-sm text-zinc-500">{loading ? 'Loading…' : 'Nobody has used the deck today yet.'}</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-zinc-500">
                  <th className="px-5 py-2.5 font-medium">Person</th>
                  <th className="px-3 py-2.5 font-medium">Plan</th>
                  <th className="px-3 py-2.5 text-right font-medium">Messages</th>
                  <th className="px-5 py-2.5 text-right font-medium">Tokens</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 25).map((r) => {
                  const plan = userPlan(r.user)
                  const limit = plans[plan]
                  const over = (n: number, cap?: number) => !!cap && n >= cap
                  return (
                    <tr key={r.id} className="border-t border-zinc-800/70">
                      <td className="max-w-[16rem] truncate px-5 py-2.5 text-zinc-200">
                        {name(r.id, r.user)}
                        {r.user?.role === 'admin' && <span className="ml-1.5 text-[11px] text-zinc-500">admin</span>}
                      </td>
                      <td className="px-3 py-2.5"><Pill>{plan}</Pill></td>
                      <td className={`px-3 py-2.5 text-right tabular-nums ${over(r.messages, limit?.messages_per_day) ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-300'}`}>
                        {r.messages}{limit?.messages_per_day ? <span className="text-zinc-600"> / {limit.messages_per_day}</span> : null}
                      </td>
                      <td className={`px-5 py-2.5 text-right tabular-nums ${over(r.tokens, limit?.tokens_per_day) ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-300'}`}>
                        {fmtNumber(r.tokens)}{limit?.tokens_per_day ? <span className="text-zinc-600"> / {fmtNumber(limit.tokens_per_day)}</span> : null}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card
        icon={DollarSign}
        title="Model spend"
        hint="What runs cost, priced from each model's rates."
        action={<Segmented value={range} onChange={setRange} label="Period" options={[
          { value: '7d', label: '7 days' }, { value: '30d', label: '30 days' }, { value: 'all', label: 'All time' },
        ]} />}
      >
        {!costs ? (
          <p className="text-sm text-zinc-500">No spend recorded.</p>
        ) : (
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <SpendList title="By person" rows={costs.by_user.map((u) => ({ key: u.user_id, label: u.display_name || u.email || u.user_id.slice(0, 8), runs: u.runs, usd: u.usd }))} total={costs.total_usd} />
            <SpendList title="By kind of run" rows={costs.by_kind.map((k) => ({ key: k.run_kind || 'other', label: k.run_kind || 'other', runs: k.runs, usd: k.usd }))} total={costs.total_usd} />
          </div>
        )}
      </Card>

      <PlansCard plans={plans} onSaved={() => fetchPlans().then(setPlans).catch(() => {})} />

      <ByokSetting />

      <Card icon={Activity} title="Recent activity" hint="Sign-ins, new agents, resets and other events, newest first." bodyClassName="divide-y divide-zinc-800/70">
        {logs.length === 0 && <p className="px-5 py-6 text-center text-sm text-zinc-500">No activity yet.</p>}
        {logs.map((l) => (
          <div key={l.id} className="flex items-center gap-3 px-5 py-2.5 text-sm">
            <Pill>{l.event_type.replace(/_/g, ' ')}</Pill>
            <span className="min-w-0 flex-1 truncate text-xs text-zinc-500">{name(l.user_id, byId.get(l.user_id))} · {describe(l.detail)}</span>
            <span className="shrink-0 text-xs text-zinc-600" title={new Date(l.created_at).toLocaleString()}>{timeAgo(l.created_at)}</span>
          </div>
        ))}
      </Card>
    </Page>
  )
}

function describe(detail: string): string {
  try {
    const d = JSON.parse(detail)
    return Object.entries(d).map(([k, v]) => `${k.replace(/_/g, ' ')}: ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' · ')
  } catch { return detail }
}

function SpendList({ title, rows, total }: { title: string; rows: { key: string; label: string; runs: number; usd: number }[]; total: number }) {
  return (
    <div>
      <div className="mb-2 text-xs font-medium text-zinc-500">{title}</div>
      {rows.length === 0 && <p className="text-xs text-zinc-500">Nothing yet.</p>}
      <div className="space-y-2.5">
        {rows.slice(0, 8).map((r) => (
          <div key={r.key}>
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span className="min-w-0 truncate capitalize text-zinc-300">{r.label}</span>
              <span className="shrink-0 font-mono tabular-nums text-zinc-200">${(r.usd || 0).toFixed(2)}</span>
            </div>
            <div className="mt-1 flex items-center gap-2">
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800">
                <div className="h-full rounded-full bg-violet-500/70" style={{ width: `${total > 0 ? Math.max(2, (r.usd / total) * 100) : 0}%` }} />
              </div>
              <span className="w-14 shrink-0 text-right text-[11px] text-zinc-500">{r.runs} run{r.runs === 1 ? '' : 's'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function PlansCard({ plans, onSaved }: { plans: Record<string, PlanLimits>; onSaved: () => void }) {
  const names = Object.keys(plans)
  const [draft, setDraft] = useState<Record<string, PlanLimits>>(plans)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => { setDraft(plans) }, [plans])

  const changed = useCallback((p: string) => LIMITS.some((l) => draft[p]?.[l.key] !== plans[p]?.[l.key]), [draft, plans])
  const dirty = names.some(changed)

  const save = async () => {
    setSaving(true); setError('')
    try {
      for (const p of names.filter(changed)) await updatePlan(p, draft[p])
      onSaved()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card
      icon={SlidersHorizontal}
      title="Plan limits"
      hint="Each user is on a plan (set it on the Users page). Daily limits reset at midnight UTC; 0 is unlimited."
      action={dirty ? <>
        <Button variant="ghost" onClick={() => setDraft(plans)}>Discard</Button>
        <Button variant="primary" icon={Save} busy={saving} onClick={save}>Save limits</Button>
      </> : undefined}
      bodyClassName=""
    >
      {names.length === 0 ? (
        <p className="px-5 py-6 text-center text-sm text-zinc-500">No plans configured.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-zinc-500">
                <th className="px-5 py-2.5 font-medium">Limit</th>
                {names.map((p) => <th key={p} className="px-3 py-2.5 font-medium capitalize">{p}</th>)}
              </tr>
            </thead>
            <tbody>
              {LIMITS.map((l) => (
                <tr key={l.key} className="border-t border-zinc-800/70">
                  <td className="px-5 py-2">
                    <div className="text-zinc-200">{l.label}</div>
                    {l.hint && <div className="text-[11px] text-zinc-500">{l.hint}</div>}
                  </td>
                  {names.map((p) => (
                    <td key={p} className="px-3 py-2">
                      <input
                        type="number"
                        min={0}
                        value={draft[p]?.[l.key] ?? 0}
                        onChange={(e) => setDraft((d) => ({ ...d, [p]: { ...d[p], [l.key]: Math.max(0, parseInt(e.target.value, 10) || 0) } }))}
                        className={`w-28 rounded-lg border bg-zinc-950 px-2.5 py-1.5 text-sm tabular-nums text-zinc-100 focus:border-violet-500/60 focus:outline-none ${
                          draft[p]?.[l.key] !== plans[p]?.[l.key] ? 'border-violet-500/50' : 'border-zinc-700'
                        }`}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {error && <div className="px-5 pb-4"><Notice tone="red">{error}</Notice></div>}
    </Card>
  )
}
