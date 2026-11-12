import { Fragment, useEffect, useMemo, useState } from 'react'
import { Vault as VaultIcon, Search, Trash2, Save, History, Eye, EyeOff, ChevronDown, ChevronRight, RotateCcw } from 'lucide-react'
import { anonymizeApi, type Vault, type VaultEntry, type VaultEvent } from '../../services/anonymizeApi'
import { Card, Button, Dialog, EmptyState, Field, Notice, Segmented, Sheet, inputCls } from '../console/parts'
import { RESET_TEXT, kindLabel, kindTone, when } from './anonFormat'
import { TokenHistoryPanel } from './TokenHistoryPanel'

// The person's vaults: one per case. Opening one shows the real values behind
// its tokens (for this person only — every view is logged), and lets them
// rename it, give it an end date, or delete it for good.

const EXPIRY = [
  { value: 'never', label: 'Never', days: 0 },
  { value: '30', label: '30 days', days: 30 },
  { value: '90', label: '90 days', days: 90 },
  { value: '365', label: 'A year', days: 365 },
] as const
type ExpiryChoice = (typeof EXPIRY)[number]['value'] | 'keep'

const ACTIONS: Record<string, string> = {
  create: 'Created', anonymize: 'Anonymized into it', restore: 'Values put back', view: 'Opened',
  update: 'Changed', delete: 'Deleted', expired: 'Expired', reset: 'Reset (emptied)',
}

export function VaultsPanel({ vaults, reload, onSheet }: {
  vaults: Vault[]
  reload: () => Promise<void> | void
  onSheet: (open: boolean) => void
}) {
  const [open, setOpen] = useState<Vault | null>(null)
  useEffect(() => { onSheet(!!open) }, [open, onSheet])

  if (!vaults.length) {
    return (
      <EmptyState icon={VaultIcon} title="No vaults yet">
        A vault is made the first time you, or one of your agents, anonymize something “in a vault”.
      </EmptyState>
    )
  }
  return (
    <>
      <Card bodyClassName="divide-y divide-zinc-800/80">
        {vaults.map((v) => (
          <button
            key={v.id}
            type="button"
            onClick={() => setOpen(v)}
            className="flex w-full flex-col gap-2 px-5 py-4 text-left transition-colors hover:bg-zinc-900/60 wide:flex-row wide:items-center wide:justify-between"
          >
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-zinc-100">{v.name}</div>
              <div className="mt-0.5 text-xs text-zinc-500">
                {v.entity_count} value{v.entity_count === 1 ? '' : 's'} · last used {when(v.last_used_at)}
                {v.created_by.startsWith('agent:') ? ` · started by ${v.created_by.slice(6)}` : ''}
                {v.expires_at ? ` · ends ${when(v.expires_at)}` : ''}
              </div>
            </div>
            <div className="flex flex-wrap gap-1">
              {Object.entries(v.kinds).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([kind, n]) => (
                <span key={kind} className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${kindTone(kind)}`}>{n} {kindLabel(kind)}</span>
              ))}
            </div>
          </button>
        ))}
      </Card>
      {open && <VaultSheet vault={open} onClose={() => setOpen(null)} onChanged={reload} />}
    </>
  )
}

function VaultSheet({ vault, onClose, onChanged }: { vault: Vault; onClose: () => void; onChanged: () => Promise<void> | void }) {
  const [entries, setEntries] = useState<VaultEntry[] | null>(null)
  const [events, setEvents] = useState<VaultEvent[] | null>(null)
  const [query, setQuery] = useState('')
  const [shown, setShown] = useState(false)
  const [openToken, setOpenToken] = useState('')
  const [name, setName] = useState(vault.name)
  const [expiry, setExpiry] = useState<ExpiryChoice>(vault.expires_at ? 'keep' : 'never')
  const expiryOptions = [
    ...(vault.expires_at ? [{ value: 'keep' as ExpiryChoice, label: `As it is (ends ${when(vault.expires_at)})` }] : []),
    ...EXPIRY.map((x) => ({ value: x.value as ExpiryChoice, label: x.label })),
  ]
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [confirmReset, setConfirmReset] = useState(false)

  const reveal = async () => {
    setBusy('view'); setError('')
    try {
      setEntries((await anonymizeApi.vault(vault.id)).entries)
      setShown(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't open the vault")
    } finally {
      setBusy('')
    }
  }

  const loadEvents = async () => {
    try {
      setEvents((await anonymizeApi.events(vault.id)).events)
    } catch { setEvents([]) }
  }

  const save = async () => {
    setBusy('save'); setError('')
    try {
      const patch: { name?: string; expires_at?: number | null } = {}
      if (name.trim() && name.trim() !== vault.name) patch.name = name.trim()
      if (expiry !== 'keep') {
        const days = EXPIRY.find((x) => x.value === expiry)?.days || 0
        patch.expires_at = days ? Math.floor(Date.now() / 1000) + days * 86400 : null
      }
      await anonymizeApi.updateVault(vault.id, patch)
      await onChanged()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't save")
    } finally {
      setBusy('')
    }
  }

  const remove = async () => {
    setBusy('delete'); setError('')
    try {
      await anonymizeApi.deleteVault(vault.id)
      await onChanged()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't delete")
      setBusy('')
    }
  }

  // Token → what it puts back; a spelling's place too (⟦ORG_7:2⟧ → that spelling).
  const values = useMemo(() => new Map((entries || []).flatMap((e) => [
    [e.token, e.original] as [string, string],
    ...(e.spellings || []).map((sp, k) => [`${e.token.slice(0, -1)}:${k + 2}⟧`, sp] as [string, string]),
  ])), [entries])

  const reset = async () => {
    setBusy('reset'); setError('')
    try {
      await anonymizeApi.resetVault(vault.id)
      await onChanged()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't reset")
      setBusy('')
      setConfirmReset(false)
    }
  }

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!entries) return []
    if (!q) return entries
    return entries.filter((e) => e.token.toLowerCase().includes(q) || e.original.toLowerCase().includes(q)
      || kindLabel(e.kind).includes(q) || e.forms.some((f) => f.toLowerCase().includes(q))
      || (e.spellings || []).some((f) => f.toLowerCase().includes(q)))
  }, [entries, query])

  return (
    <Sheet
      title={vault.name}
      subtitle={`${vault.entity_count} values · made ${when(vault.created_at)}`}
      icon={VaultIcon}
      onClose={onClose}
      wide
      footer={<>
        <Button variant="danger" icon={Trash2} onClick={() => setConfirmDelete(true)}>Delete</Button>
        <Button icon={RotateCcw} onClick={() => setConfirmReset(true)} className="mr-auto">Reset</Button>
        <Button onClick={onClose}>Close</Button>
        <Button variant="primary" icon={Save} busy={busy === 'save'} onClick={save}>Save</Button>
      </>}
    >
      <div className="space-y-6">
        {error && <Notice tone="red">{error}</Notice>}
        <Field label="Name">
          <input value={name} onChange={(e) => setName(e.target.value)} className={inputCls} maxLength={80} />
        </Field>
        <Field label="Delete it by itself after" hint="Counted from today. When it ends, the real values are gone for good and tokens can't be put back.">
          <Segmented label="Delete after" value={expiry} onChange={setExpiry} options={expiryOptions} />
        </Field>

        <div>
          <div className="mb-2 flex items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-zinc-100">What it keeps</h3>
            {shown ? (
              <Button variant="ghost" icon={EyeOff} onClick={() => { setShown(false); setEntries(null) }}>Hide</Button>
            ) : (
              <Button icon={Eye} busy={busy === 'view'} onClick={reveal}>Show the real values</Button>
            )}
          </div>
          {!shown && <p className="text-xs leading-relaxed text-zinc-500">Only you can see them. Opening the vault is noted in its history.</p>}
          {shown && entries && (
            <div className="space-y-3">
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
                <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Find a token or a value"
                  className={`${inputCls} pl-9`} />
              </div>
              <div className="max-h-[45vh] overflow-y-auto rounded-xl border border-zinc-800">
                <table className="w-full text-sm">
                  <tbody className="divide-y divide-zinc-800/80">
                    {filtered.map((e) => (
                      <Fragment key={e.token}>
                        <tr className="align-top">
                          <td className="whitespace-nowrap px-3 py-2">
                            <span className={`rounded px-1 font-mono text-xs ${kindTone(e.kind)}`}>{e.token}</span>
                          </td>
                          <td className="px-3 py-2 text-zinc-200">
                            <div className="break-words">{e.original}</div>
                            {e.forms.length > 0 && <div className="mt-0.5 text-xs text-zinc-500">also: {e.forms.slice(0, 6).join(' · ')}</div>}
                            {(e.spellings || []).length > 0 && (
                              <div className="mt-0.5 text-xs text-zinc-500" title="Written like this, these come back exactly — a link, a legal name">
                                spelled: {(e.spellings || []).slice(0, 8).map((sp, k) => (
                                  <span key={k} className="mr-2"><span className="font-mono text-zinc-500">:{k + 2}</span> {sp}</span>
                                ))}
                              </div>
                            )}
                          </td>
                          <td className="whitespace-nowrap px-3 py-2 text-right">
                            <button type="button" onClick={() => setOpenToken(openToken === e.token ? '' : e.token)}
                              title="What added it, and every time it went out to a model and came back"
                              className="inline-flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200">
                              <span className="text-amber-600 dark:text-amber-400">↑{e.sent || 0}</span>
                              <span className="text-sky-600 dark:text-sky-400">↓{e.restored || 0}</span>
                              {openToken === e.token ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                            </button>
                          </td>
                        </tr>
                        {openToken === e.token && (
                          <tr>
                            <td colSpan={3} className="bg-zinc-950/40 px-3 pb-3 pt-1">
                              <TokenHistoryPanel vaultId={vault.id} token={e.token} values={values} />
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    ))}
                  </tbody>
                </table>
                {!filtered.length && <p className="px-3 py-4 text-sm text-zinc-500">Nothing matches.</p>}
              </div>
            </div>
          )}
        </div>

        <div>
          <button type="button" onClick={loadEvents} className="flex items-center gap-1.5 text-xs font-medium text-violet-600 hover:underline dark:text-violet-400">
            <History className="h-3.5 w-3.5" /> History
          </button>
          {events && (
            <ul className="mt-2 space-y-1 text-xs text-zinc-400">
              {events.map((ev, i) => (
                <li key={i}>
                  {new Date(ev.at * 1000).toLocaleString()} · {ACTIONS[ev.action] || ev.action}
                  {ev.actor && ev.actor !== 'person' ? ` · ${ev.actor.replace(/^agent:/, 'agent ')}` : ''}
                  {typeof ev.detail.new === 'number' && ev.detail.new ? ` · ${ev.detail.new} new` : ''}
                  {typeof ev.detail.restored === 'number' ? ` · ${ev.detail.restored} put back` : ''}
                </li>
              ))}
              {!events.length && <li>Nothing yet.</li>}
            </ul>
          )}
        </div>
      </div>

      {confirmReset && (
        <Dialog title={`Reset "${vault.name}"?`} icon={RotateCcw} onClose={() => setConfirmReset(false)} footer={<>
          <Button onClick={() => setConfirmReset(false)}>Keep it</Button>
          <Button variant="danger" icon={RotateCcw} busy={busy === 'reset'} onClick={reset}>Reset</Button>
        </>}>
          <p className="text-sm leading-relaxed text-zinc-300">{RESET_TEXT}</p>
        </Dialog>
      )}
      {confirmDelete && (
        <Dialog title={`Delete "${vault.name}"?`} icon={Trash2} onClose={() => setConfirmDelete(false)} footer={<>
          <Button onClick={() => setConfirmDelete(false)}>Keep it</Button>
          <Button variant="danger" icon={Trash2} busy={busy === 'delete'} onClick={remove}>Delete for good</Button>
        </>}>
          <p className="text-sm leading-relaxed text-zinc-300">
            Its {vault.entity_count} real value{vault.entity_count === 1 ? '' : 's'} and its key are deleted. Documents anonymized into it keep
            their tokens, but they can never be put back. This can't be undone.
          </p>
        </Dialog>
      )}
    </Sheet>
  )
}
