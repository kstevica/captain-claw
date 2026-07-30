import { useEffect, useState } from 'react'
import { Factory, LogOut } from 'lucide-react'
import { api } from './api'
import { useAuth, usePack } from './stores'
import Login from './pages/Login'
import Streams from './pages/Streams'
import StreamView from './pages/Stream'
import Studio from './pages/Studio'

export default function App() {
  const { pack, load: loadPack } = usePack()
  const { user, ready, boot, logout } = useAuth()
  const [streamId, setStreamId] = useState<string | null>(null)
  const [view, setView] = useState<'desk' | 'studio'>('desk')
  const [creator, setCreator] = useState(false)

  useEffect(() => { void loadPack(); void boot() }, [loadPack, boot])

  useEffect(() => {
    if (!user) { setCreator(false); return }
    void api<{ creator: boolean }>('/api/creators/me')
      .then((d) => setCreator(d.creator)).catch(() => setCreator(false))
  }, [user])

  if (!pack || !ready) {
    return <div className="h-full grid place-items-center text-[var(--lp-text-dim)]">…</div>
  }
  if (!user) return <Login />

  return (
    <div className="h-full flex flex-col">
      <header className="flex items-center gap-3 px-5 py-3 border-b border-[var(--lp-border)] bg-[var(--lp-surface)]">
        <button
          onClick={() => { setStreamId(null); setView('desk') }}
          className="text-lg font-bold tracking-tight hover:opacity-80"
          style={{ color: 'var(--lp-accent)' }}
        >
          {pack.name}
        </button>
        <span className="text-sm text-[var(--lp-text-dim)]">{pack.tagline}</span>
        <div className="flex-1" />
        {creator && (
          <button
            onClick={() => { setStreamId(null); setView(view === 'studio' ? 'desk' : 'studio') }}
            title="Pack Studio"
            className={`p-1.5 rounded hover:bg-[var(--lp-border)] flex items-center gap-1.5 text-sm ${
              view === 'studio' ? 'text-[var(--lp-accent)]' : 'text-[var(--lp-text-dim)]'}`}
          >
            <Factory size={16} /> Studio
          </button>
        )}
        <span className="text-sm text-[var(--lp-text-dim)]">{user.email ?? user.id}</span>
        <button onClick={() => void logout()} title="Sign out"
                className="p-1.5 rounded hover:bg-[var(--lp-border)] text-[var(--lp-text-dim)]">
          <LogOut size={16} />
        </button>
      </header>
      <main className="flex-1 overflow-y-auto">
        {view === 'studio'
          ? <Studio />
          : streamId
            ? <StreamView streamId={streamId} onBack={() => setStreamId(null)} />
            : <Streams onOpen={setStreamId} />}
      </main>
    </div>
  )
}
