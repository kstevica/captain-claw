import { useEffect, useState } from 'react'
import { Loader2, RefreshCw, Sparkles } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'

interface Status { installed: boolean; settings_safe: boolean; detail?: string }
interface Check { connected: boolean; quota: string; models: string[] }

async function request<T>(path: string, method = 'GET'): Promise<T> {
  const send = () => fetch(`/fd/antigravity/${path}`, {
    method,
    headers: useAuthStore.getState().token
      ? { Authorization: `Bearer ${useAuthStore.getState().token}` } : {},
  })
  let response = await send()
  if (response.status === 401 && await refreshAccessToken()) response = await send()
  const data = await response.json()
  if (!response.ok) throw new Error(data.detail || 'Connection check failed')
  return data as T
}

export default function AntigravityConnection() {
  const [status, setStatus] = useState<Status | null>(null)
  const [check, setCheck] = useState<Check | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const isAdmin = useAuthStore(s => s.authEnabled === false || s.user?.role === 'admin')

  useEffect(() => {
    if (!isAdmin) return
    let active = true
    request<Status>('status').then(s => { if (active) setStatus(s) })
      .catch(e => { if (active) setError(e instanceof Error ? e.message : 'Status unavailable') })
    return () => { active = false }
  }, [isAdmin])

  async function verify() {
    setLoading(true)
    setError('')
    setCheck(null)
    try {
      setStatus(await request<Status>('status'))
      setCheck(await request<Check>('check', 'POST'))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Connection check failed')
    } finally { setLoading(false) }
  }

  if (!isAdmin) return null
  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-900/60 overflow-hidden">
      <div className="flex items-center gap-3 px-5 py-4 border-b border-zinc-800">
        <Sparkles className="h-5 w-5 text-blue-400" />
        <div>
          <h3 className="text-sm font-semibold text-zinc-100">Google (Antigravity)</h3>
          <p className="text-xs text-zinc-500">Use your Google sign-in and subscription quota through the local CLI.</p>
        </div>
        <span className="ml-auto text-xs text-zinc-400">{check?.connected ? 'Connected' : status?.installed ? 'CLI installed' : 'Setup required'}</span>
      </div>
      <div className="px-5 py-4 space-y-3 text-xs text-zinc-400">
        <p>Install Antigravity CLI 1.2.1 or later and run <code className="text-zinc-200">agy</code> on this host to sign in with Google. Set AI Credit Overages to Never / Use G1 Credits to Off.</p>
        <p>Select <code className="text-zinc-200">antigravity-cli</code> as the model provider. API keys and extra credits are blocked. Text generation only; tools are unavailable. Use local processes under the same OS account; Docker needs its own CLI and sign-in.</p>
        <p>Subscription limits still apply. This check reads quota; it does not verify your Google One invoice.</p>
        {status?.detail && <p className="text-amber-400">{status.detail}</p>}
        {error && <p role="alert" className="text-red-400">{error}</p>}
        {check && <>
          <pre className="whitespace-pre-wrap break-words rounded-md bg-zinc-950 p-3">{check.quota}</pre>
          <div>Available Gemini model IDs: {check.models.map(model => <code key={model} className="block text-zinc-300">{model}</code>)}</div>
        </>}
        <button onClick={verify} disabled={loading} className="inline-flex items-center gap-2 rounded-md bg-violet-600 px-3 py-2 text-white disabled:opacity-50">
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />} Check connection and quota
        </button>
      </div>
    </section>
  )
}
