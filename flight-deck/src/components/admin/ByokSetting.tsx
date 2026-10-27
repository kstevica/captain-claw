import { useEffect, useState } from 'react'
import { KeyRound } from 'lucide-react'
import { getAdminByok, setAdminByokAllowed, type AdminByok } from '../../services/byok'
import { Switch } from './Switch'

// "Own API keys" for the deck (byok.py): users may bring their own provider
// keys and, while they use them, aren't held to their plan's daily limits.
// Turning it off moves everyone who uses them back to the deck's models. The
// Admin page's Plans tab and the console's Usage page share it.

export function ByokSetting() {
  const [state, setState] = useState<AdminByok | null>(null)
  const [busy, setBusy] = useState(false)
  const [asking, setAsking] = useState(false)
  const [err, setErr] = useState('')

  useEffect(() => { getAdminByok().then(setState).catch(() => setState(null)) }, [])
  if (!state || !state.offered) return null
  const users = state.users_on.length

  const save = async (allowed: boolean) => {
    setBusy(true); setErr('')
    try {
      await setAdminByokAllowed(allowed)
      setState(await getAdminByok())
      setAsking(false)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
    setBusy(false)
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-sm font-medium text-zinc-200">
            <KeyRound className="h-4 w-4 text-violet-400" /> Let users bring their own API keys
          </div>
          <p className="mt-0.5 text-xs text-zinc-500">
            A user adds their own Anthropic, OpenAI, Gemini, OpenRouter or xAI key (Settings → Your own AI keys) and turns it on:
            their agents then run on it, and their plan's daily limits don't apply. Their provider bills them; the deck pays nothing for it.
            {users > 0 && ` ${users} user${users === 1 ? ' is' : 's are'} on their own keys now.`}
          </p>
        </div>
        <Switch
          on={state.allowed}
          disabled={busy}
          label="Let users bring their own API keys"
          onToggle={() => {
            if (state.allowed && users > 0) { setAsking(true); return }
            void save(!state.allowed)
          }}
        />
      </div>
      {asking && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-zinc-300">
          <span>
            This moves {users === 1 ? 'the 1 user' : `all ${users} users`} on their own keys back to the deck's models (their running agents restart),
            and their daily limits apply again.
          </span>
          <button onClick={() => save(false)} disabled={busy} className="rounded-md bg-red-600 px-2.5 py-1 font-medium text-white hover:bg-red-500 disabled:opacity-50">
            {busy ? 'Turning off…' : 'Turn off'}
          </button>
          <button onClick={() => setAsking(false)} disabled={busy} className="rounded-md border border-zinc-700 px-2.5 py-1 hover:bg-zinc-800">Cancel</button>
        </div>
      )}
      {err && <p className="mt-2 text-xs text-red-600 dark:text-red-400">{err}</p>}
    </div>
  )
}
