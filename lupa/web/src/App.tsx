import { useEffect, useState } from 'react'
import { LogOut } from 'lucide-react'
import { useAuth, usePack } from './stores'
import Login from './pages/Login'
import Streams from './pages/Streams'
import StreamView from './pages/Stream'

export default function App() {
  const { pack, load: loadPack } = usePack()
  const { user, ready, boot, logout } = useAuth()
  const [streamId, setStreamId] = useState<string | null>(null)

  useEffect(() => { void loadPack(); void boot() }, [loadPack, boot])

  if (!pack || !ready) {
    return <div className="h-full grid place-items-center text-[var(--lp-text-dim)]">…</div>
  }
  if (!user) return <Login />

  return (
    <div className="h-full flex flex-col">
      <header className="flex items-center gap-3 px-5 py-3 border-b border-[var(--lp-border)] bg-[var(--lp-surface)]">
        <button
          onClick={() => setStreamId(null)}
          className="text-lg font-bold tracking-tight hover:opacity-80"
          style={{ color: 'var(--lp-accent)' }}
        >
          {pack.name}
        </button>
        <span className="text-sm text-[var(--lp-text-dim)]">{pack.tagline}</span>
        <div className="flex-1" />
        <span className="text-sm text-[var(--lp-text-dim)]">{user.email ?? user.id}</span>
        <button onClick={() => void logout()} title="Sign out"
                className="p-1.5 rounded hover:bg-[var(--lp-border)] text-[var(--lp-text-dim)]">
          <LogOut size={16} />
        </button>
      </header>
      <main className="flex-1 overflow-y-auto">
        {streamId
          ? <StreamView streamId={streamId} onBack={() => setStreamId(null)} />
          : <Streams onOpen={setStreamId} />}
      </main>
    </div>
  )
}
