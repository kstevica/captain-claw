import { useEffect, useState } from 'react'
import { ShieldCheck } from 'lucide-react'
import { approvalsApi, MODE_HINTS, type ApprovalKind, type ApprovalMode, type ApprovalPolicy } from '../../services/approvalsApi'
import { Card, SettingRow, Segmented, Notice } from '../console/parts'

// Admin console → Settings: what agents on this deck must ask before they pay,
// send, post, share, delete or store a password (approvals.py). Every kind
// starts at "Don't ask"; the card answers go to the person the turn is for.

export function ApprovalsPolicyCard() {
  const [data, setData] = useState<ApprovalPolicy | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<ApprovalKind | ''>('')

  useEffect(() => {
    approvalsApi.policy().then(setData).catch(() => setError("Couldn't load what needs approval"))
  }, [])

  const set = async (kind: ApprovalKind, mode: ApprovalMode) => {
    setBusy(kind)
    setError('')
    try {
      setData(await approvalsApi.setPolicy({ [kind]: mode }))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setBusy('')
    }
  }

  return (
    <Card icon={ShieldCheck} title="Approvals"
      hint="What agents must ask a person before doing. The card opens in that person's Flight Deck and bell; a scheduled run with nobody there can only do what they chose to always allow."
      bodyClassName="divide-y divide-zinc-800/80 px-5 py-4">
      {error && <div className="pb-3"><Notice tone="red">{error}</Notice></div>}
      {!data ? (
        <p className="py-2 text-sm text-zinc-500">Loading…</p>
      ) : data.kinds.map(({ kind, label }) => (
        <SettingRow key={kind} title={label} hint={MODE_HINTS[data.policy[kind]]}>
          <div className={busy === kind ? 'pointer-events-none opacity-60' : ''}>
            <Segmented<ApprovalMode>
              label={label}
              value={data.policy[kind]}
              options={data.modes.map((m) => ({ value: m.mode, label: m.label }))}
              onChange={(mode) => set(kind, mode)}
            />
          </div>
        </SettingRow>
      ))}
    </Card>
  )
}
