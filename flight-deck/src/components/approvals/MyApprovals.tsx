import { useCallback, useEffect, useState } from 'react'
import { ShieldCheck, X } from 'lucide-react'
import { approvalsApi, type AlwaysAllowed, type ApprovalPolicy } from '../../services/approvalsApi'

// Profile → Personal: what this deck asks agents to check with you first, and
// what you chose to always allow an agent ("Always allow here" on a card).
// Hidden when the deck asks about nothing and you allowed nothing.

export function MyApprovals() {
  const [policy, setPolicy] = useState<ApprovalPolicy | null>(null)
  const [always, setAlways] = useState<AlwaysAllowed[]>([])
  const [busy, setBusy] = useState('')

  const load = useCallback(() => {
    approvalsApi.policy().then(setPolicy).catch(() => setPolicy(null))
    approvalsApi.mine().then((r) => setAlways(r.always)).catch(() => setAlways([]))
  }, [])
  useEffect(() => { load() }, [load])

  const asked = policy ? policy.kinds.filter((k) => ['ask', 'always_ask'].includes(policy.policy[k.kind])) : []
  const yours = policy ? policy.kinds.filter((k) => policy.policy[k.kind] === 'people_only') : []
  if (!asked.length && !yours.length && !always.length) return null
  const list = (ks: { label: string }[]) => ks.map((k) => k.label.toLowerCase()).join(', ')

  const forget = async (a: AlwaysAllowed) => {
    const key = `${a.agent}:${a.kind}`
    setBusy(key)
    try {
      await approvalsApi.forget(a.agent, a.kind)
      setAlways((now) => now.filter((x) => `${x.agent}:${x.kind}` !== key))
    } finally {
      setBusy('')
    }
  }

  return (
    <div className="space-y-3 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="flex items-start gap-2">
        <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
        <div>
          <h3 className="text-sm font-semibold text-zinc-200">Approvals</h3>
          <p className="mt-0.5 text-xs text-zinc-500">
            {asked.length ? <>Agents here ask you first before they: {list(asked)}. </> : null}
            {yours.length ? <>They never do these — you do them yourself: {list(yours)}.</> : null}
            {!asked.length && !yours.length ? 'This deck no longer asks before any of these.' : null}
          </p>
        </div>
      </div>
      {always.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-zinc-400">You always allow</p>
          <ul className="space-y-1">
            {always.map((a) => {
              const key = `${a.agent}:${a.kind}`
              return (
                <li key={key} className="flex items-center justify-between gap-3 rounded-lg border border-zinc-800 px-3 py-2 text-sm">
                  <span className="min-w-0 truncate text-zinc-300">
                    <span className="font-medium text-zinc-100">{a.agent_name || a.agent}</span> · {a.kind_label.toLowerCase()}
                  </span>
                  <button type="button" disabled={busy === key} onClick={() => forget(a)}
                    className="inline-flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
                    <X className="h-3.5 w-3.5" /> Ask me again
                  </button>
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </div>
  )
}
