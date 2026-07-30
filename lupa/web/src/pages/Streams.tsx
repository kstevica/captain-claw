import { useEffect, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Plus, RadioTower } from 'lucide-react'
import { api } from '../api'
import { usePack, useStreams, useVocab } from '../stores'
import HouseStyle from '../components/HouseStyle'

interface InboxRound {
  stream_id: string; stream_title: string; session_id: string
  round_no: number; created_at: string
}

interface DeskSummary {
  slug: string; status: string; name: string; tagline: string; accent: string
}

export default function Streams({ onOpen }: { onOpen: (id: string) => void }) {
  const { streams, load, create } = useStreams()
  const pack = usePack((s) => s.pack)
  const v = useVocab()
  const [title, setTitle] = useState('')
  const [creating, setCreating] = useState(false)
  const [inbox, setInbox] = useState<InboxRound[]>([])
  const [desks, setDesks] = useState<DeskSummary[]>([])

  const activeSlug = pack?.slug ?? ''

  useEffect(() => {
    void load(activeSlug || undefined)
    void api<{ rounds: InboxRound[] }>('/api/inbox').then((d) => setInbox(d.rounds)).catch(() => {})
    void api<{ packs: DeskSummary[] }>('/api/packs')
      .then((d) => setDesks(d.packs.filter((p) => p.status === 'published')))
      .catch(() => {})
  }, [load, activeSlug])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim()) return
    setCreating(true)
    try {
      const s = await create(title.trim(), activeSlug)
      setTitle('')
      onOpen(s.id)
    } finally { setCreating(false) }
  }

  return (
    <div className="max-w-3xl mx-auto px-5 py-8 space-y-6">
      {desks.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {desks.map((d) => {
            const active = d.slug === activeSlug
            return (
              <a key={d.slug}
                 href={d.slug === 'research-desk' ? '/' : `/desks/${d.slug}`}
                 className={`rounded-xl border px-4 py-2.5 transition-colors ${
                   active ? 'border-[var(--lp-accent)]'
                          : 'border-[var(--lp-border)] hover:border-[var(--lp-text-dim)]'}`}>
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full inline-block"
                        style={{ background: d.accent || 'var(--lp-accent)' }} />
                  <span className="text-sm font-semibold">{d.name}</span>
                </div>
                <div className="text-[11px] text-[var(--lp-text-dim)] mt-0.5">{d.tagline}</div>
              </a>
            )
          })}
        </div>
      )}

      {inbox.length > 0 && (
        <div className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] p-4">
          <div className="flex items-center gap-2 mb-2">
            <RadioTower size={14} style={{ color: 'var(--lp-accent)' }} />
            <span className="text-sm font-semibold">{v('inbox_title', 'Brief inbox')}</span>
          </div>
          <ul className="space-y-1">
            {inbox.slice(0, 5).map((r) => (
              <li key={r.session_id}>
                <button
                  onClick={() => onOpen(r.stream_id)}
                  className="w-full text-left text-sm flex items-baseline gap-2 rounded px-2 py-1 hover:bg-[var(--lp-border)]/40"
                >
                  <span className="font-medium">{r.stream_title}</span>
                  <span className="text-xs text-[var(--lp-text-dim)]">
                    {v('round', 'Round').toLowerCase()} {r.round_no} · {new Date(r.created_at).toLocaleString()}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

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

      <HouseStyle />

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
