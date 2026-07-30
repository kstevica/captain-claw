import { useState } from 'react'
import { useAuth, usePack } from '../stores'

export default function Login() {
  const { login } = useAuth()
  const pack = usePack((s) => s.pack)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [register, setRegister] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true); setError('')
    try { await login(email, password, register) }
    catch (err) { setError(err instanceof Error ? err.message : 'sign-in failed') }
    finally { setBusy(false) }
  }

  return (
    <div className="h-full grid place-items-center px-4">
      <form onSubmit={submit}
            className="w-full max-w-sm rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] p-7 space-y-4">
        <div>
          <div className="text-2xl font-bold" style={{ color: 'var(--lp-accent)' }}>
            {pack?.name}
          </div>
          <div className="text-sm text-[var(--lp-text-dim)] mt-1">{pack?.tagline}</div>
        </div>
        <input
          type="email" required placeholder="email" value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2 text-sm outline-none focus:border-[var(--lp-accent)]"
        />
        <input
          type="password" required placeholder="password" value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2 text-sm outline-none focus:border-[var(--lp-accent)]"
        />
        {error && <div className="text-sm text-red-400">{error}</div>}
        <button
          type="submit" disabled={busy}
          className="w-full rounded-lg py-2 text-sm font-semibold text-black disabled:opacity-50"
          style={{ background: 'var(--lp-accent)' }}
        >
          {busy ? '…' : register ? 'Create account' : 'Sign in'}
        </button>
        <button
          type="button"
          onClick={() => setRegister(!register)}
          className="w-full text-xs text-[var(--lp-text-dim)] hover:text-[var(--lp-text)]"
        >
          {register ? 'Have an account? Sign in' : 'New here? Create an account'}
        </button>
      </form>
    </div>
  )
}
