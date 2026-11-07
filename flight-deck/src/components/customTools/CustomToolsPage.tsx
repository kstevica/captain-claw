import { useCallback, useEffect, useState } from 'react'
import {
  Wrench, Plus, RefreshCw, X, Users, Check, Ban, Inbox, Store, ShieldCheck,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { useCustomToolsStore, type ToolsTab } from '../../stores/customToolsStore'
import { customToolsApi, type Approval, type Offer, type ToolSummary } from '../../services/customToolsApi'
import { Page, PageHeader, Card, Button, Segmented, Notice, EmptyState, Pill } from '../console/parts'
import { DeclaresLine, statusPill } from './toolParts'
import { ToolSheet } from './ToolSheet'
import { NewToolDialog } from './NewToolDialog'

// Custom tools: the person's own (made by Crew, published once their tests and
// examples pass), the ones others offered them (Shared with me, Catalog), and
// for an admin the offers to everybody that wait for approval.

interface Selected { owner?: string; slug: string; mine: boolean }

export function CustomToolsPage({ phone = false, onClose }: { phone?: boolean; onClose?: () => void }) {
  const authEnabled = useAuthStore((s) => s.authEnabled !== false)
  const isAdmin = useAuthStore((s) => s.authEnabled === false || s.user?.role === 'admin')
  const status = useCustomToolsStore((s) => s.status)
  const refreshStatus = useCustomToolsStore((s) => s.refreshStatus)
  const [tab, setTab] = useState<ToolsTab>('mine')
  const [mine, setMine] = useState<ToolSummary[]>([])
  const [canBuild, setCanBuild] = useState(false)
  const [shared, setShared] = useState<Offer[]>([])
  const [catalog, setCatalog] = useState<Offer[]>([])
  const [approvals, setApprovals] = useState<Approval[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [selected, setSelected] = useState<Selected | null>(null)
  const [making, setMaking] = useState(false)
  const [notice, setNotice] = useState('')

  const load = useCallback(async () => {
    setError('')
    try {
      await refreshStatus()
      const own = await customToolsApi.mine()
      setMine(own.tools)
      setCanBuild(own.can_build)
      if (authEnabled) {
        const [s, c] = await Promise.all([customToolsApi.shared(), customToolsApi.catalog()])
        setShared(s.tools)
        setCatalog(c.tools)
        if (isAdmin) setApprovals((await customToolsApi.approvals()).approvals)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't load custom tools")
    } finally {
      setLoading(false)
    }
  }, [authEnabled, isAdmin, refreshStatus])

  useEffect(() => { load() }, [load])

  // A bell (or another page) can ask for a tab or a tool.
  useEffect(() => {
    const focus = useCustomToolsStore.getState().takeFocus()
    if (!focus) return
    if (focus.tab) setTab(focus.tab)
    if (focus.slug) setSelected({ owner: focus.owner, slug: focus.slug, mine: !focus.owner })
  }, [])

  // Something publishing: look again until it's done.
  useEffect(() => {
    if (!mine.some((t) => t.publish?.state === 'running')) return
    const timer = window.setInterval(load, 3000)
    return () => window.clearInterval(timer)
  }, [mine, load])

  const addOrRemove = async (o: Offer) => {
    setBusy(`${o.owner}/${o.slug}`)
    try {
      if (o.added) await customToolsApi.unadd(o.owner, o.slug)
      else await customToolsApi.add(o.owner, o.slug)
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setBusy('')
    }
  }

  const decide = async (a: Approval, approve: boolean) => {
    setBusy(`${a.owner}/${a.slug}`)
    try {
      await customToolsApi.decide(a.owner, a.slug, approve)
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally {
      setBusy('')
    }
  }

  const showApprovals = isAdmin && authEnabled && (approvals.length > 0 || !!status?.everybody_needs_approval)
  const tabs: { value: ToolsTab; label: string; count?: number }[] = [
    { value: 'mine', label: 'Mine', count: mine.length },
    ...(authEnabled ? [
      { value: 'shared' as const, label: 'Shared with me', count: shared.length },
      { value: 'catalog' as const, label: 'Everybody', count: catalog.length },
    ] : []),
    ...(showApprovals ? [{ value: 'approvals' as const, label: 'Approvals', count: approvals.length }] : []),
  ]

  return (
    <div className="h-full overflow-y-auto">
      <Page>
        <PageHeader
          icon={Wrench}
          title="Custom tools"
          description="Small programs Crew’s coding agents build for jobs code does better than a model. Flight Deck runs every call for your agents, with your secrets and files."
          actions={<>
            {canBuild && <Button variant="primary" icon={Plus} onClick={() => setMaking(true)}>New tool</Button>}
            <Button variant="ghost" icon={RefreshCw} onClick={() => { setLoading(true); load() }} aria-label="Refresh">{phone ? '' : 'Refresh'}</Button>
            {onClose && <Button variant="ghost" icon={X} onClick={onClose} aria-label="Close" />}
          </>}
        />

        {status && !status.enabled && (
          <Notice tone="amber">Custom tools are off on this deck. An admin turns them on under Settings.</Notice>
        )}
        {error && <Notice tone="red">{error}</Notice>}
        {notice && <Notice tone="emerald">{notice}</Notice>}

        {tabs.length > 1 && (
          <div className="overflow-x-auto">
            <Segmented label="Which tools" value={tab} options={tabs} onChange={setTab} />
          </div>
        )}

        {loading ? (
          <p className="text-sm text-zinc-500">Loading…</p>
        ) : tab === 'mine' ? (
          mine.length === 0 ? (
            <EmptyState icon={Wrench} title="No custom tools yet">
              {canBuild
                ? 'Ask an agent to build one (“make me a tool that totals a CSV column”), or start one here with New tool.'
                : 'Making tools needs Crew, the coding agents: an admin approves it on the Users page. Tools others offer you show under Shared with me.'}
            </EmptyState>
          ) : (
            <div className="grid gap-3 wide:grid-cols-2">
              {mine.map((t) => (
                <button key={t.slug} onClick={() => setSelected({ slug: t.slug, mine: true })}
                  className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition hover:border-zinc-700 hover:bg-zinc-900/70">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold text-zinc-100">{t.title}</div>
                      <div className="truncate font-mono text-[11px] text-zinc-500">{t.slug}</div>
                    </div>
                    {statusPill(t)}
                  </div>
                  {t.description && <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-zinc-400">{t.description}</p>}
                </button>
              ))}
            </div>
          )
        ) : tab === 'approvals' ? (
          approvals.length === 0 ? (
            <EmptyState icon={ShieldCheck} title="Nothing waits for approval">
              {status?.everybody_needs_approval
                ? 'When someone offers a tool to everybody, it waits here for an admin.'
                : 'Approval is off (Settings), so tools offered to everybody reach everyone at once.'}
            </EmptyState>
          ) : (
            <div className="space-y-3">
              {approvals.map((a) => (
                <Card key={`${a.owner}/${a.slug}`} title={a.title} hint={`${a.slug} · by ${a.owner_person.name}`} icon={Wrench}
                  action={<div className="flex gap-2">
                    <Button variant="primary" icon={Check} busy={busy === `${a.owner}/${a.slug}`} onClick={() => decide(a, true)}>Approve</Button>
                    <Button icon={Ban} onClick={() => { if (confirm(`Decline offering “${a.title}” to everybody?`)) decide(a, false) }}>Decline</Button>
                  </div>}>
                  <p className="text-sm text-zinc-400">{a.description}</p>
                  {a.manifest && <DeclaresLine d={{ version: a.live?.version || '', network: a.manifest.network,
                    secrets: a.manifest.secrets.map((s) => s.name), timeout_s: a.manifest.timeout_s, memory_mb: a.manifest.memory_mb }} />}
                  <p className="mt-3 text-xs text-zinc-500">Approving offers it to everyone on this deck. Each person still adds it before their agents use it, and it runs with their own secrets and files.</p>
                </Card>
              ))}
            </div>
          )
        ) : (
          (() => {
            const list = tab === 'shared' ? shared : catalog
            if (list.length === 0) {
              return (
                <EmptyState icon={tab === 'shared' ? Inbox : Store} title={tab === 'shared' ? 'Nobody has offered you a tool' : 'No tools are offered to everybody'}>
                  {tab === 'shared' ? 'When someone offers you one of their tools, it shows here for you to add.' : 'Tools someone offers to everyone on this deck show here.'}
                </EmptyState>
              )
            }
            return (
              <div className="grid gap-3 wide:grid-cols-2">
                {list.map((o) => (
                  <div key={`${o.owner}/${o.slug}`} className="flex flex-col rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <button className="min-w-0 text-left" onClick={() => setSelected({ owner: o.owner, slug: o.slug, mine: false })}>
                        <div className="truncate text-sm font-semibold text-zinc-100 hover:underline">{o.title}</div>
                        <div className="flex items-center gap-1 truncate text-[11px] text-zinc-500"><Users className="h-3 w-3" />by {o.owner_person.name}</div>
                      </button>
                      {o.added ? <Pill tone="emerald" icon={Check}>Added</Pill> : <Pill>Not added</Pill>}
                    </div>
                    {o.description && <p className="mt-2 line-clamp-3 text-xs leading-relaxed text-zinc-400">{o.description}</p>}
                    <DeclaresLine d={o.declares} />
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Button variant={o.added ? 'secondary' : 'primary'} icon={o.added ? X : Plus} busy={busy === `${o.owner}/${o.slug}`}
                        onClick={() => addOrRemove(o)}>{o.added ? 'Remove' : 'Add'}</Button>
                      <Button variant="ghost" onClick={() => setSelected({ owner: o.owner, slug: o.slug, mine: false })}>Details</Button>
                    </div>
                  </div>
                ))}
              </div>
            )
          })()
        )}
      </Page>

      {selected && (
        <ToolSheet
          slug={selected.slug}
          owner={selected.owner}
          mine={selected.mine}
          onClose={() => setSelected(null)}
          onChanged={load}
          onDeleted={() => { setSelected(null); load() }}
        />
      )}
      {making && (
        <NewToolDialog
          onClose={() => setMaking(false)}
          onStarted={(msg) => { setMaking(false); setNotice(msg); load() }}
        />
      )}
    </div>
  )
}
