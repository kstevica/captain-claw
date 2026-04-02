import { useState } from 'react'
import { loginUser, registerUser } from '../stores/authStore'
import { APP_VERSION, BUILD_DATE } from '../version'

export function LoginPage() {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (mode === 'login') {
        await loginUser(email, password)
      } else {
        await registerUser(email, password, displayName)
      }
    } catch (err: any) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#0a0a1a] flex items-center justify-center">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-white tracking-tight">Flight Deck</h1>
          <p className="text-zinc-500 text-sm mt-1">Agent Management Console</p>
          <p className="text-zinc-600 text-xs mt-1">v{APP_VERSION} &middot; {BUILD_DATE}</p>
        </div>

        <form onSubmit={handleSubmit} className="bg-[#12122a] border border-zinc-800 rounded-lg p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white">
            {mode === 'login' ? 'Sign In' : 'Create Account'}
          </h2>

          {error && (
            <div className="text-red-400 text-sm bg-red-950/30 border border-red-900/50 rounded px-3 py-2">
              {error}
            </div>
          )}

          {mode === 'register' && (
            <div>
              <label className="block text-sm text-zinc-300 mb-1">Display Name</label>
              <input
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                className="w-full bg-[#0a0a1a] border border-zinc-700 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-500 placeholder-zinc-600"
                placeholder="Your name"
              />
            </div>
          )}

          <div>
            <label className="block text-sm text-zinc-300 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-[#0a0a1a] border border-zinc-700 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-500 placeholder-zinc-600"
              placeholder="you@example.com"
              required
              autoFocus
            />
          </div>

          <div>
            <label className="block text-sm text-zinc-300 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-[#0a0a1a] border border-zinc-700 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-500 placeholder-zinc-600"
              placeholder="At least 6 characters"
              required
              minLength={6}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 disabled:cursor-not-allowed text-white font-medium py-2 rounded text-sm transition-colors"
          >
            {loading ? 'Please wait...' : mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>

          <div className="text-center text-sm text-zinc-400">
            {mode === 'login' ? (
              <>
                No account?{' '}
                <button type="button" onClick={() => { setMode('register'); setError('') }} className="text-indigo-400 hover:text-indigo-300">
                  Create one
                </button>
              </>
            ) : (
              <>
                Already have an account?{' '}
                <button type="button" onClick={() => { setMode('login'); setError('') }} className="text-indigo-400 hover:text-indigo-300">
                  Sign in
                </button>
              </>
            )}
          </div>
        </form>
      </div>
    </div>
  )
}
