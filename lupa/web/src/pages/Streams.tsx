import { useEffect, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Plus } from 'lucide-react'
import { usePack, useStreams, useVocab } from '../stores'

export default function Streams({ onOpen }: { onOpen: (id: string) => void }) {
  const { streams, load, create } = useStreams()
  const pack = usePack((s) => s.pack)
  const v = useVocab()
  const [title, setTitle] = useState('')
  const [creating, setCreating] = useState(false)

  useEffect(() => { void load() }, [load])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim()) return
    setCreating(true)
    try {
      const s = await create(title.trim())
      setTitle('')
      onOpen(s.id)
    } finally { setCreating(false) }
  }

  return (
    <div className="max-w-3xl mx-auto px-5 py-8 space-y-6">
      <form onSubmit={submit} className="flex gap-2">
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder={`${v('new_stream', 'New stream')}: title`}
          className="flex-1 rounded-lg bg-[var(--lp-surface)] border border-[var(--lp-border)] px-3 py-2 text-sm outline-none focus:border-[var(--lp-accent)]"
        />
        <button
          type="submit" disabled={creating || !title.trim()}
          className="rounded-lg px-4 py-2 text-sm font-semibold text-black disabled:opacity-40 flex items-center gap-1.5"
          style={{ background: 'var(--lp-accent)' }}
        >
          <Plus size={15} /> {v('new_stream', 'New stream')}
        </button>
      </form>

      {streams.length === 0 ? (
        <div className="space-y-6">
          <div className="text-[var(--lp-text-dim)] text-sm">{v('empty_streams')}</div>
          {pack?.onboarding_md && (
            <div className="lp-prose rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] px-6 py-4 text-sm">
              <Markdown remarkPlugins={[remarkGfm]}>{pack.onboarding_md}</Markdown>
            </div>
          )}
        </div>
      ) : (
        <ul className="space-y-2">
          {streams.map((s) => (
            <li key={s.id}>
              <button
                onClick={() => onOpen(s.id)}
                className="w-full text-left rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] px-5 py-4 hover:border-[var(--lp-accent)] transition-colors"
              >
                <div className="font-semibold">{s.title}</div>
                <div className="text-xs text-[var(--lp-text-dim)] mt-1">
                  {String(s.rounds ?? 0)} {v('round', 'Round').toLowerCase()}
                  {Number(s.rounds ?? 0) === 1 ? '' : 's'}
                  {' · '}{new Date(s.updated_at).toLocaleString()}
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
